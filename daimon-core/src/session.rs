//! Core types and trait for inter-session communication.
//!
//! A "session" is an independent, concurrently-running agent conversation.
//! Unlike the [`distributed`](crate::distributed) layer — which distributes
//! *work* to a pool of stateless workers — this module lets long-lived,
//! stateful sessions exchange messages with one another: directed
//! (session-to-session) or broadcast (to every subscriber).
//!
//! Provider crates implement [`SessionBus`] for their message service (Redis
//! pub/sub, NATS, etc.) to carry cross-session messages between processes. The
//! main `daimon` crate re-exports everything from here and ships an in-process
//! implementation.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Identifier for a session participating in inter-session communication.
///
/// A newtype over `String` so session addresses can't be confused with
/// arbitrary strings (task ids, message ids, etc.) at call sites. Cheap to
/// clone and freely convertible from string-like types.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SessionId(pub String);

impl SessionId {
    /// Creates a session id from anything string-like.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Generates a fresh, process-unique session id of the form
    /// `session-{nanos:x}-{counter:x}`.
    ///
    /// Mirrors [`AgentTask::generate_id`](crate::distributed::AgentTask): the
    /// nanosecond timestamp keeps ids distinct across processes and restarts,
    /// while a process-wide counter disambiguates ids minted within the same
    /// clock tick by concurrent callers.
    pub fn generate() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::{SystemTime, UNIX_EPOCH};

        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        Self(format!("session-{ts:x}-{seq:x}"))
    }

    /// Returns the id as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for SessionId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for SessionId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// A message exchanged between sessions over a [`SessionBus`].
///
/// A message names its `from` sender and, when directed, its `to` recipient.
/// A `None` recipient is a broadcast: every subscriber receives it. The
/// `body` is free-form text (typically an agent utterance or a serialized
/// payload) and `metadata` carries application-specific key/values, matching
/// [`AgentTask`](crate::distributed::AgentTask)'s convention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMessage {
    /// Unique identifier for this message (generated on creation).
    pub message_id: String,
    /// The session that sent the message.
    pub from: SessionId,
    /// The intended recipient, or `None` to broadcast to all subscribers.
    pub to: Option<SessionId>,
    /// The message payload.
    pub body: String,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl SessionMessage {
    /// Creates a directed message from `from` to `to`.
    pub fn direct(
        from: impl Into<SessionId>,
        to: impl Into<SessionId>,
        body: impl Into<String>,
    ) -> Self {
        Self {
            message_id: Self::generate_id(),
            from: from.into(),
            to: Some(to.into()),
            body: body.into(),
            metadata: HashMap::new(),
        }
    }

    /// Creates a broadcast message from `from` to every subscriber.
    pub fn broadcast(from: impl Into<SessionId>, body: impl Into<String>) -> Self {
        Self {
            message_id: Self::generate_id(),
            from: from.into(),
            to: None,
            body: body.into(),
            metadata: HashMap::new(),
        }
    }

    /// Adds a metadata key-value pair.
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Returns `true` if this message is addressed to every subscriber.
    pub fn is_broadcast(&self) -> bool {
        self.to.is_none()
    }

    /// Generates a message id of the form `msg-{nanos:x}-{counter:x}`.
    fn generate_id() -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::{SystemTime, UNIX_EPOCH};

        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("msg-{ts:x}-{seq:x}")
    }
}

/// Trait for exchanging messages between concurrent agent sessions.
///
/// Implement this for your transport (in-process channels, Redis pub/sub,
/// NATS, etc.) to route [`SessionMessage`]s. Two send shapes are supported:
///
/// - [`send`](Self::send) delivers a message to one named session.
/// - [`broadcast`](Self::broadcast) delivers a message to every subscribed
///   session.
///
/// A session receives its messages by [`subscribe`](Self::subscribe)-ing,
/// which registers the session (idempotently) and yields a
/// [`SessionReceiver`] that produces messages addressed to it plus any
/// broadcasts.
pub trait SessionBus: Send + Sync {
    /// Registers `session` (if new) and returns a receiver for messages
    /// directed to it and broadcasts, delivered in send order. A session's
    /// mailbox has a single consumer: subscribing a session that is already
    /// registered installs a fresh mailbox and retires any previous receiver.
    fn subscribe(&self, session: SessionId)
    -> impl Future<Output = Result<SessionReceiver>> + Send;

    /// Sends a directed message. The recipient is taken from
    /// [`SessionMessage::to`]; a message with no recipient is rejected — use
    /// [`broadcast`](Self::broadcast) instead.
    ///
    /// Delivery to a session that has never subscribed is a no-op rather than
    /// an error: sessions come and go, and a sender should not fail because a
    /// peer isn't currently listening.
    fn send(&self, message: SessionMessage) -> impl Future<Output = Result<()>> + Send;

    /// Broadcasts a message to every currently-subscribed session (including,
    /// potentially, the sender). Any [`SessionMessage::to`] is ignored.
    fn broadcast(&self, message: SessionMessage) -> impl Future<Output = Result<()>> + Send;

    /// Lists the ids of sessions currently registered with the bus.
    fn sessions(&self) -> impl Future<Output = Result<Vec<SessionId>>> + Send;
}

/// Receiver half of a session's inbox.
///
/// Yields [`SessionMessage`]s addressed to the owning session plus broadcasts,
/// in send order. Backed by whatever channel the [`SessionBus`] implementation
/// uses; the in-process bus uses a bounded `tokio::sync::mpsc` FIFO queue, so
/// messages queue behind the session's in-progress work and are drained one at
/// a time.
pub struct SessionReceiver {
    inner: Pin<Box<dyn ReceiverStream>>,
}

impl SessionReceiver {
    /// Wraps a transport-specific receiver.
    pub fn new(inner: impl ReceiverStream + 'static) -> Self {
        Self {
            inner: Box::pin(inner),
        }
    }

    /// Waits for the next message addressed to this session.
    ///
    /// Returns `Ok(None)` when the inbox is closed and no more messages will
    /// arrive (e.g. the bus was dropped).
    pub async fn recv(&mut self) -> Result<Option<SessionMessage>> {
        self.inner.as_mut().recv().await
    }
}

/// Transport-agnostic receiver backing a [`SessionReceiver`].
///
/// Implemented by each [`SessionBus`] for its channel type so the public
/// receiver stays object-safe and transport-independent.
pub trait ReceiverStream: Send {
    /// Waits for the next message; `Ok(None)` signals the inbox is closed.
    fn recv(
        self: Pin<&mut Self>,
    ) -> Pin<Box<dyn Future<Output = Result<Option<SessionMessage>>> + Send + '_>>;
}

/// Object-safe wrapper for [`SessionBus`], enabling `Arc<dyn ErasedSessionBus>`.
pub trait ErasedSessionBus: Send + Sync {
    fn subscribe_erased(
        &self,
        session: SessionId,
    ) -> Pin<Box<dyn Future<Output = Result<SessionReceiver>> + Send + '_>>;

    fn send_erased(
        &self,
        message: SessionMessage,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>>;

    fn broadcast_erased(
        &self,
        message: SessionMessage,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>>;

    fn sessions_erased(&self) -> Pin<Box<dyn Future<Output = Result<Vec<SessionId>>> + Send + '_>>;
}

impl<T: SessionBus> ErasedSessionBus for T {
    fn subscribe_erased(
        &self,
        session: SessionId,
    ) -> Pin<Box<dyn Future<Output = Result<SessionReceiver>> + Send + '_>> {
        Box::pin(self.subscribe(session))
    }

    fn send_erased(
        &self,
        message: SessionMessage,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        Box::pin(self.send(message))
    }

    fn broadcast_erased(
        &self,
        message: SessionMessage,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        Box::pin(self.broadcast(message))
    }

    fn sessions_erased(&self) -> Pin<Box<dyn Future<Output = Result<Vec<SessionId>>> + Send + '_>> {
        Box::pin(self.sessions())
    }
}

/// Shared ownership of a session bus via `Arc<dyn ErasedSessionBus>`.
pub type SharedSessionBus = std::sync::Arc<dyn ErasedSessionBus>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_id_generate_unique_across_concurrent_calls() {
        let handles: Vec<_> = (0..8)
            .map(|_| {
                std::thread::spawn(|| {
                    (0..100)
                        .map(|_| SessionId::generate().0)
                        .collect::<Vec<_>>()
                })
            })
            .collect();

        let mut ids = std::collections::HashSet::new();
        for handle in handles {
            for id in handle.join().expect("thread panicked") {
                assert!(
                    ids.insert(id.clone()),
                    "duplicate session id generated: {id}"
                );
            }
        }
        assert_eq!(ids.len(), 800);
    }

    #[test]
    fn test_direct_and_broadcast_construction() {
        let direct = SessionMessage::direct("a", "b", "hi");
        assert_eq!(direct.from, SessionId::new("a"));
        assert_eq!(direct.to, Some(SessionId::new("b")));
        assert!(!direct.is_broadcast());

        let bcast = SessionMessage::broadcast("a", "hello all");
        assert_eq!(bcast.to, None);
        assert!(bcast.is_broadcast());
    }

    #[test]
    fn test_message_ids_unique() {
        let a = SessionMessage::broadcast("s", "x");
        let b = SessionMessage::broadcast("s", "y");
        assert_ne!(a.message_id, b.message_id);
        assert!(a.message_id.starts_with("msg-"));
    }

    #[test]
    fn test_message_serialization_roundtrip() {
        let msg = SessionMessage::direct("s1", "s2", "payload")
            .with_metadata("priority", serde_json::json!(1));
        let json = serde_json::to_string(&msg).unwrap();
        let deser: SessionMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.from, SessionId::new("s1"));
        assert_eq!(deser.to, Some(SessionId::new("s2")));
        assert_eq!(deser.body, "payload");
        assert_eq!(deser.metadata["priority"], serde_json::json!(1));
    }

    #[test]
    fn test_session_id_display_and_conversions() {
        let id: SessionId = "abc".into();
        assert_eq!(id.to_string(), "abc");
        assert_eq!(id.as_str(), "abc");
        let id2: SessionId = String::from("xyz").into();
        assert_eq!(id2.0, "xyz");
    }
}
