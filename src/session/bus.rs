//! In-process session bus backed by per-session bounded `tokio::sync::mpsc`
//! FIFO queues — one ordered mailbox per session.
//!
//! Each subscribed session owns one bounded queue that serves as its inbox.
//! A directed [`send`](SessionBus::send) enqueues onto the recipient's queue;
//! a [`broadcast`](SessionBus::broadcast) enqueues onto every registered
//! session's queue. A session drains its mailbox at its own pace via a
//! [`SessionReceiver`], so **messages queue behind whatever the session is
//! currently doing** and are delivered in send order — a message sent while a
//! session is busy waits in the queue until that session finishes its current
//! work and calls `recv()` again.
//!
//! ## Ordering and backpressure
//!
//! Directed [`send`](SessionBus::send) is FIFO and lossless: unlike a
//! broadcast channel (which drops the oldest messages once a slow consumer
//! falls behind), a bounded mpsc queue preserves every message in order. When
//! a busy session's queue fills to `capacity`, `send` **waits** for room
//! rather than dropping the message — backpressure that paces fast producers
//! against a session still working through earlier messages. This mirrors
//! [`InProcessBroker`](crate::distributed::InProcessBroker)'s bounded work
//! queue.
//!
//! [`broadcast`](SessionBus::broadcast), by contrast, is **best-effort
//! fire-and-fan-out**: it never blocks and never lets one recipient stall the
//! others. Each recipient is offered the message with a non-blocking
//! `try_send`; a recipient whose mailbox is currently full is skipped (and the
//! drop logged). This deliberately trades broadcast losslessness for liveness,
//! so a single stuck session cannot hang a fan-out to every other session (and
//! the caller invoking it). Use directed `send` when a message must not be
//! dropped.
//!
//! ## One consumer per session
//!
//! A session's mailbox has a single consumer. Subscribing a session that is
//! already registered hands the caller a fresh receiver and **retires the
//! previous one** (its `recv()` then observes the closed queue), so exactly
//! one receiver drains a given session's messages at a time. This retirement
//! is the one case where a directed `send` can lose a message despite the FIFO
//! contract: a `send` that raced an in-flight resubscribe may target the
//! just-retired queue (see [`InProcessSessionBus::send`]).

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use tokio::sync::{Mutex, mpsc};

use crate::error::{DaimonError, Result};

pub use daimon_core::session::{
    ReceiverStream, SessionBus, SessionId, SessionMessage, SessionReceiver,
};

/// Default per-session mailbox capacity when using [`InProcessSessionBus::new`].
///
/// This many messages can queue behind a busy session before senders block
/// on backpressure.
pub const DEFAULT_INBOX_CAPACITY: usize = 256;

/// In-process implementation of [`SessionBus`] backed by bounded per-session
/// mpsc FIFO queues — one ordered mailbox per session.
///
/// Suitable for single-process, multi-session coordination and testing.
/// Clone-friendly: all clones share the same registry of session mailboxes,
/// so a message sent through one clone reaches the subscriber created through
/// another. For cross-process messaging, implement [`SessionBus`] over your
/// message broker.
///
/// Semantics:
/// - **Directed `send` — FIFO, lossless, backpressured:** directed messages
///   are delivered in send order and never dropped; they queue behind a
///   session's in-progress work, and sending to a full mailbox waits until the
///   session consumes a message. (The sole exception is a `send` racing a
///   concurrent resubscribe of the same session — see [`Self::send`].)
/// - **`broadcast` — best-effort, non-blocking:** a broadcast offers the
///   message to each session with a non-blocking `try_send` and skips (logs)
///   any recipient whose mailbox is full, so one stuck session never stalls
///   fan-out to the rest.
/// - **Unknown recipient:** a directed message to a session that has never
///   subscribed is a no-op, not an error (sessions come and go; a sender
///   should not fail because a peer isn't listening).
/// - **Single consumer:** re-subscribing a session retires its previous
///   receiver.
pub struct InProcessSessionBus {
    mailboxes: Arc<Mutex<HashMap<SessionId, mpsc::Sender<SessionMessage>>>>,
    capacity: usize,
}

impl InProcessSessionBus {
    /// Creates a bus with the [default mailbox capacity](DEFAULT_INBOX_CAPACITY).
    pub fn new() -> Self {
        // DEFAULT_INBOX_CAPACITY is a nonzero constant, so this cannot fail.
        Self::with_capacity(DEFAULT_INBOX_CAPACITY)
            .expect("DEFAULT_INBOX_CAPACITY must be greater than 0")
    }

    /// Creates a bus whose per-session mailboxes hold `capacity` queued
    /// messages before `send` blocks on backpressure.
    ///
    /// Returns [`DaimonError::Other`] if `capacity` is zero (a zero-capacity
    /// mpsc queue is invalid). Unlike [`tokio::sync::mpsc::channel`], which
    /// panics on a zero capacity, this library constructor surfaces the error
    /// so callers can handle it.
    pub fn with_capacity(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(DaimonError::Other(
                "session mailbox capacity must be greater than 0".into(),
            ));
        }
        Ok(Self {
            mailboxes: Arc::new(Mutex::new(HashMap::new())),
            capacity,
        })
    }

    /// Returns the current sender for `session`, if it has a live mailbox.
    async fn sender_for(&self, session: &SessionId) -> Option<mpsc::Sender<SessionMessage>> {
        self.mailboxes.lock().await.get(session).cloned()
    }
}

impl Default for InProcessSessionBus {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for InProcessSessionBus {
    fn clone(&self) -> Self {
        Self {
            mailboxes: Arc::clone(&self.mailboxes),
            capacity: self.capacity,
        }
    }
}

impl SessionBus for InProcessSessionBus {
    async fn subscribe(&self, session: SessionId) -> Result<SessionReceiver> {
        // Install a fresh queue for this session. Any prior sender is dropped
        // here, retiring the previous receiver so a single consumer drains the
        // mailbox at a time.
        let (tx, rx) = mpsc::channel(self.capacity);
        self.mailboxes.lock().await.insert(session, tx);
        Ok(SessionReceiver::new(MpscReceiverStream { rx }))
    }

    async fn send(&self, message: SessionMessage) -> Result<()> {
        let Some(to) = message.to.clone() else {
            return Err(DaimonError::InvalidSessionMessage(
                "SessionBus::send requires a recipient; use broadcast() for unaddressed messages"
                    .into(),
            ));
        };

        // Deliver only to a session that already has a mailbox. A directed
        // message to a session that never subscribed is a no-op, not an error.
        let Some(sender) = self.sender_for(&to).await else {
            return Ok(());
        };

        // Await capacity (backpressure) so directed delivery is lossless while
        // a busy session drains earlier messages. `mpsc::Sender::send` errors
        // only on a closed channel: the session retired its receiver, most
        // likely via a resubscribe that raced this send. That is the one
        // window where a directed message is dropped despite the FIFO
        // contract, so log it rather than swallowing it silently.
        if sender.send(message).await.is_err() {
            tracing::debug!(
                to = %to,
                "session mailbox closed before delivery (retired receiver); message dropped"
            );
        }
        Ok(())
    }

    async fn broadcast(&self, message: SessionMessage) -> Result<()> {
        // Snapshot the recipients and release the lock before delivering, so
        // no await happens under the registry mutex.
        let recipients: Vec<_> = {
            let mailboxes = self.mailboxes.lock().await;
            mailboxes
                .iter()
                .map(|(id, tx)| (id.clone(), tx.clone()))
                .collect()
        };

        // Best-effort fire-and-fan-out: a non-blocking `try_send` per
        // recipient. A recipient whose mailbox is full is skipped (logged)
        // rather than awaited, so a single stuck session can never stall the
        // broadcast to everyone else — broadcast trades losslessness for
        // liveness. Callers needing guaranteed delivery use `send`.
        for (to, sender) in recipients {
            match sender.try_send(message.clone()) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(_)) => {
                    tracing::debug!(
                        to = %to,
                        "session mailbox full; skipping broadcast delivery to this session"
                    );
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    tracing::debug!(
                        to = %to,
                        "session mailbox closed; skipping broadcast delivery to this session"
                    );
                }
            }
        }
        Ok(())
    }

    async fn sessions(&self) -> Result<Vec<SessionId>> {
        let mailboxes = self.mailboxes.lock().await;
        Ok(mailboxes.keys().cloned().collect())
    }
}

/// Adapts an `mpsc::Receiver` to the transport-agnostic [`ReceiverStream`]
/// backing a [`SessionReceiver`].
struct MpscReceiverStream {
    rx: mpsc::Receiver<SessionMessage>,
}

impl ReceiverStream for MpscReceiverStream {
    fn recv(
        self: Pin<&mut Self>,
    ) -> Pin<Box<dyn Future<Output = Result<Option<SessionMessage>>> + Send + '_>> {
        Box::pin(async move {
            let this = self.get_mut();
            // `None` means the queue is closed and drained — the bus (or this
            // session's sender) is gone. In-order delivery otherwise.
            Ok(this.rx.recv().await)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_directed_message_reaches_recipient_only() {
        let bus = InProcessSessionBus::new();
        let mut alice = bus.subscribe(SessionId::new("alice")).await.unwrap();
        let mut bob = bus.subscribe(SessionId::new("bob")).await.unwrap();

        bus.send(SessionMessage::direct("alice", "bob", "hi bob"))
            .await
            .unwrap();

        let got = bob.recv().await.unwrap().unwrap();
        assert_eq!(got.body, "hi bob");
        assert_eq!(got.from, SessionId::new("alice"));

        // Alice must not receive a message addressed to bob: her mailbox stays
        // empty, so a bounded wait times out.
        let idle = tokio::time::timeout(std::time::Duration::from_millis(50), alice.recv()).await;
        assert!(
            idle.is_err(),
            "alice must not receive bob's directed message"
        );
    }

    #[tokio::test]
    async fn test_broadcast_reaches_all_subscribers() {
        let bus = InProcessSessionBus::new();
        let mut a = bus.subscribe(SessionId::new("a")).await.unwrap();
        let mut b = bus.subscribe(SessionId::new("b")).await.unwrap();
        let mut c = bus.subscribe(SessionId::new("c")).await.unwrap();

        bus.broadcast(SessionMessage::broadcast("a", "hello all"))
            .await
            .unwrap();

        for rx in [&mut a, &mut b, &mut c] {
            let got = rx.recv().await.unwrap().unwrap();
            assert_eq!(got.body, "hello all");
            assert!(got.is_broadcast());
        }
    }

    #[tokio::test]
    async fn test_send_to_unknown_session_is_noop() {
        let bus = InProcessSessionBus::new();
        // Nobody subscribed as "ghost"; must not error.
        bus.send(SessionMessage::direct("a", "ghost", "anyone?"))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_send_without_recipient_errors() {
        let bus = InProcessSessionBus::new();
        let mut msg = SessionMessage::broadcast("a", "no target");
        msg.to = None;
        assert!(bus.send(msg).await.is_err());
    }

    #[tokio::test]
    async fn test_sessions_lists_registered() {
        let bus = InProcessSessionBus::new();
        let _a = bus.subscribe(SessionId::new("a")).await.unwrap();
        let _b = bus.subscribe(SessionId::new("b")).await.unwrap();

        let mut sessions = bus.sessions().await.unwrap();
        sessions.sort();
        assert_eq!(sessions, vec![SessionId::new("a"), SessionId::new("b")]);
    }

    #[tokio::test]
    async fn test_clone_shares_registry() {
        let bus = InProcessSessionBus::new();
        let clone = bus.clone();

        let mut bob = clone.subscribe(SessionId::new("bob")).await.unwrap();
        bus.send(SessionMessage::direct("alice", "bob", "via clone"))
            .await
            .unwrap();

        let got = bob.recv().await.unwrap().unwrap();
        assert_eq!(got.body, "via clone");
    }

    /// Messages sent while a session is "busy" (not calling `recv`) must queue
    /// in FIFO order and all be delivered once the session drains its mailbox.
    #[tokio::test]
    async fn test_messages_queue_in_fifo_order_behind_busy_session() {
        let bus = InProcessSessionBus::new();
        let mut worker = bus.subscribe(SessionId::new("worker")).await.unwrap();

        // The worker is "busy": it hasn't called recv yet. Enqueue several
        // messages; they must accumulate in order.
        for i in 0..5 {
            bus.send(SessionMessage::direct(
                "boss",
                "worker",
                format!("task-{i}"),
            ))
            .await
            .unwrap();
        }

        // Now the worker becomes free and drains — in send order.
        for i in 0..5 {
            let got = worker.recv().await.unwrap().unwrap();
            assert_eq!(got.body, format!("task-{i}"));
        }
    }

    /// A full mailbox exerts backpressure: a send does not complete until the
    /// busy session consumes a message, and no message is dropped.
    ///
    /// Determinism: rather than sleeping and hoping the send is still pending,
    /// this drives the runtime with bounded `yield_now` rounds and asserts the
    /// send stays unfinished while the mailbox is full, then completes only
    /// after a message is drained. The `yield` loop gives the spawned task
    /// ample opportunity to make progress if backpressure were (incorrectly)
    /// absent, without depending on wall-clock timing.
    #[tokio::test]
    async fn test_full_mailbox_applies_backpressure() {
        // Capacity 1: one queued message fills the mailbox.
        let bus = InProcessSessionBus::with_capacity(1).unwrap();
        let mut worker = bus.subscribe(SessionId::new("worker")).await.unwrap();

        // Fill the queue.
        bus.send(SessionMessage::direct("boss", "worker", "first"))
            .await
            .unwrap();

        // A second send must not complete while the mailbox is full.
        let bus2 = bus.clone();
        let send_task = tokio::spawn(async move {
            bus2.send(SessionMessage::direct("boss", "worker", "second"))
                .await
        });

        // Yield repeatedly so the spawned send is polled many times. If it
        // could complete without capacity (i.e. no backpressure), it would
        // finish within these rounds; a correct bus keeps it pending.
        for _ in 0..32 {
            tokio::task::yield_now().await;
        }
        assert!(
            !send_task.is_finished(),
            "send must block on a full mailbox"
        );

        // Draining one message frees capacity, unblocking the pending send.
        assert_eq!(worker.recv().await.unwrap().unwrap().body, "first");
        send_task.await.unwrap().unwrap();

        // Both messages arrive, in order, with none dropped.
        assert_eq!(worker.recv().await.unwrap().unwrap().body, "second");
    }

    #[tokio::test]
    async fn test_with_capacity_zero_errors() {
        assert!(InProcessSessionBus::with_capacity(0).is_err());
    }

    /// Broadcast is best-effort: a recipient whose mailbox is full is skipped,
    /// but every other recipient still receives the message. One stuck session
    /// must not block or drop the fan-out to the rest.
    #[tokio::test]
    async fn test_broadcast_skips_full_mailbox_and_reaches_others() {
        // Capacity 1 so a single un-drained message fills a mailbox.
        let bus = InProcessSessionBus::with_capacity(1).unwrap();
        let mut slow = bus.subscribe(SessionId::new("slow")).await.unwrap();
        let mut fast = bus.subscribe(SessionId::new("fast")).await.unwrap();

        // Fill "slow"'s mailbox and never drain it.
        bus.send(SessionMessage::direct("x", "slow", "occupied"))
            .await
            .unwrap();

        // Broadcast must complete promptly despite "slow" being full, and must
        // reach "fast".
        bus.broadcast(SessionMessage::broadcast("coordinator", "ping"))
            .await
            .unwrap();

        let got = fast.recv().await.unwrap().unwrap();
        assert_eq!(got.body, "ping");
        assert!(got.is_broadcast());

        // "slow" only ever holds its original message; the broadcast was
        // dropped for it (best-effort), not queued behind.
        let first = slow.recv().await.unwrap().unwrap();
        assert_eq!(first.body, "occupied");
        let no_more = tokio::time::timeout(std::time::Duration::from_millis(50), slow.recv()).await;
        assert!(
            no_more.is_err(),
            "broadcast to a full mailbox must be dropped, not queued"
        );
    }

    /// Re-subscribing a session retires the previous receiver (single
    /// consumer): the old receiver observes the closed mailbox.
    #[tokio::test]
    async fn test_resubscribe_retires_previous_receiver() {
        let bus = InProcessSessionBus::new();
        let mut first = bus.subscribe(SessionId::new("s")).await.unwrap();
        let mut second = bus.subscribe(SessionId::new("s")).await.unwrap();

        bus.send(SessionMessage::direct("x", "s", "for the live receiver"))
            .await
            .unwrap();

        // The retired receiver sees a closed queue (Ok(None)); the live one
        // receives the message.
        assert!(first.recv().await.unwrap().is_none());
        assert_eq!(
            second.recv().await.unwrap().unwrap().body,
            "for the live receiver"
        );
    }

    /// A directed send that targets the sender cloned *before* a resubscribe
    /// lands on the retired queue and is dropped (fail-open, logged), not
    /// delivered to the new receiver. This documents the one gap in the FIFO
    /// "lossless" contract; the send still reports success (no error to the
    /// caller).
    #[tokio::test]
    async fn test_send_to_retired_queue_is_dropped_not_delivered() {
        let bus = InProcessSessionBus::with_capacity(1).unwrap();
        let s = SessionId::new("s");

        // Subscribe, fill the mailbox so a pre-resubscribe sender is blocked,
        // then resubscribe (retiring that first queue) and confirm the new
        // receiver does not observe a message sent into the old queue.
        let _first = bus.subscribe(s.clone()).await.unwrap();
        let mut second_holder = None;

        // Send into the (soon-to-be-retired) first queue up to capacity, then
        // resubscribe.
        bus.send(SessionMessage::direct("x", "s", "into old queue"))
            .await
            .unwrap();
        let second = bus.subscribe(s.clone()).await.unwrap();
        second_holder.replace(second);
        let mut second = second_holder.unwrap();

        // A fresh send after resubscribe reaches the new receiver in order.
        bus.send(SessionMessage::direct("x", "s", "into new queue"))
            .await
            .unwrap();
        let got = second.recv().await.unwrap().unwrap();
        assert_eq!(
            got.body, "into new queue",
            "the new receiver must only see messages sent to the current queue"
        );

        // The message routed into the retired queue is not delivered to the
        // new receiver.
        let no_more =
            tokio::time::timeout(std::time::Duration::from_millis(50), second.recv()).await;
        assert!(
            no_more.is_err(),
            "messages sent into the retired queue must not surface on the new receiver"
        );
    }
}
