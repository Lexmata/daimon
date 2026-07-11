//! Episodic memory trait: a structured, timestamped event log.
//!
//! Unlike [`Memory`](crate::memory::Memory), which records what was *said*
//! (chat messages), [`EpisodicMemory`] records what *happened* — discrete,
//! typed events with metadata, queryable by type and time range. Built-in
//! implementations live in the `daimon` facade crate.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use crate::error::Result;

/// A single recorded event.
#[derive(Debug, Clone)]
pub struct EpisodicEvent {
    /// Backend-assigned unique identifier.
    pub id: String,
    /// Caller-defined event category (e.g. `"tool_call"`, `"user_login"`).
    pub event_type: String,
    /// Arbitrary structured payload.
    pub payload: Value,
    /// Unix epoch milliseconds at which the event was recorded.
    pub timestamp_ms: i64,
}

/// Filter for [`EpisodicMemory::query`]. All fields are optional; an
/// all-`None` query returns every event (subject to `limit`).
#[derive(Debug, Clone, Default)]
pub struct EpisodicQuery {
    /// Restrict to events with this exact `event_type`.
    pub event_type: Option<String>,
    /// Restrict to events at or after this Unix epoch millisecond timestamp.
    pub since_ms: Option<i64>,
    /// Restrict to events at or before this Unix epoch millisecond timestamp.
    pub until_ms: Option<i64>,
    /// Maximum number of events to return, most recent first.
    pub limit: Option<usize>,
}

impl EpisodicQuery {
    /// An unfiltered query with no limit.
    pub fn all() -> Self {
        Self::default()
    }

    /// Restricts the query to a single event type.
    pub fn of_type(mut self, event_type: impl Into<String>) -> Self {
        self.event_type = Some(event_type.into());
        self
    }

    /// Restricts the query to the given time range (inclusive).
    pub fn between(mut self, since_ms: i64, until_ms: i64) -> Self {
        self.since_ms = Some(since_ms);
        self.until_ms = Some(until_ms);
        self
    }

    /// Caps the number of returned events.
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }
}

/// Trait for structured, timestamped event log backends.
pub trait EpisodicMemory: Send + Sync {
    /// Records an event and returns its assigned id. Implementations stamp
    /// `timestamp_ms` with the current time.
    fn record(
        &self,
        event_type: &str,
        payload: Value,
    ) -> impl Future<Output = Result<String>> + Send;

    /// Returns events matching `query`, most recent first.
    fn query(
        &self,
        query: EpisodicQuery,
    ) -> impl Future<Output = Result<Vec<EpisodicEvent>>> + Send;

    /// Removes all recorded events.
    fn clear(&self) -> impl Future<Output = Result<()>> + Send;
}

/// Object-safe wrapper for the `EpisodicMemory` trait, enabling dynamic
/// dispatch via `Arc<dyn ErasedEpisodicMemory>`.
pub trait ErasedEpisodicMemory: Send + Sync {
    fn record_erased<'a>(
        &'a self,
        event_type: &'a str,
        payload: Value,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>>;

    fn query_erased(
        &self,
        query: EpisodicQuery,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<EpisodicEvent>>> + Send + '_>>;

    fn clear_erased(&self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>>;
}

impl<T: EpisodicMemory> ErasedEpisodicMemory for T {
    fn record_erased<'a>(
        &'a self,
        event_type: &'a str,
        payload: Value,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        Box::pin(self.record(event_type, payload))
    }

    fn query_erased(
        &self,
        query: EpisodicQuery,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<EpisodicEvent>>> + Send + '_>> {
        Box::pin(self.query(query))
    }

    fn clear_erased(&self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        Box::pin(self.clear())
    }
}

/// Shared ownership of episodic memory via `Arc<dyn ErasedEpisodicMemory>`.
pub type SharedEpisodicMemory = Arc<dyn ErasedEpisodicMemory>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct VecEpisodicMemory(Mutex<Vec<EpisodicEvent>>);

    impl EpisodicMemory for VecEpisodicMemory {
        async fn record(&self, event_type: &str, payload: Value) -> Result<String> {
            let mut events = self.0.lock().unwrap();
            let seq = events.len() as i64;
            let id = format!("evt-{seq}");
            events.push(EpisodicEvent {
                id: id.clone(),
                event_type: event_type.to_string(),
                payload,
                timestamp_ms: seq,
            });
            Ok(id)
        }

        async fn query(&self, query: EpisodicQuery) -> Result<Vec<EpisodicEvent>> {
            let events = self.0.lock().unwrap();
            let mut matched: Vec<EpisodicEvent> = events
                .iter()
                .filter(|e| {
                    query.event_type.as_ref().is_none_or(|t| &e.event_type == t)
                        && query.since_ms.is_none_or(|s| e.timestamp_ms >= s)
                        && query.until_ms.is_none_or(|u| e.timestamp_ms <= u)
                })
                .cloned()
                .collect();
            matched.sort_by_key(|e| std::cmp::Reverse(e.timestamp_ms));
            if let Some(limit) = query.limit {
                matched.truncate(limit);
            }
            Ok(matched)
        }

        async fn clear(&self) -> Result<()> {
            self.0.lock().unwrap().clear();
            Ok(())
        }
    }

    #[tokio::test]
    async fn episodic_memory_is_implementable_from_core_alone() {
        let mem = VecEpisodicMemory(Mutex::new(Vec::new()));
        mem.record("login", serde_json::json!({"user": "a"}))
            .await
            .unwrap();
        mem.record("logout", serde_json::json!({"user": "a"}))
            .await
            .unwrap();
        mem.record("login", serde_json::json!({"user": "b"}))
            .await
            .unwrap();

        let logins = mem
            .query(EpisodicQuery::all().of_type("login"))
            .await
            .unwrap();
        assert_eq!(logins.len(), 2);
        // Most recent first.
        assert_eq!(logins[0].payload["user"], "b");

        let ranged = mem.query(EpisodicQuery::all().between(0, 0)).await.unwrap();
        assert_eq!(ranged.len(), 1);

        let limited = mem.query(EpisodicQuery::all().limit(1)).await.unwrap();
        assert_eq!(limited.len(), 1);

        mem.clear().await.unwrap();
        assert!(mem.query(EpisodicQuery::all()).await.unwrap().is_empty());

        let shared: SharedEpisodicMemory = Arc::new(VecEpisodicMemory(Mutex::new(Vec::new())));
        shared
            .record_erased("test", serde_json::json!(null))
            .await
            .unwrap();
        assert_eq!(
            shared
                .query_erased(EpisodicQuery::all())
                .await
                .unwrap()
                .len(),
            1
        );
    }
}
