//! Episodic memory implementations.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;
use tokio::sync::RwLock;

use crate::error::Result;
use crate::memory::{EpisodicEvent, EpisodicMemory, EpisodicQuery};

/// In-process [`EpisodicMemory`] backed by a `Vec`. Data is lost when the
/// process exits; use [`SqliteEpisodicMemory`](super::SqliteEpisodicMemory)
/// (feature = "sqlite") for persistence.
#[derive(Default)]
pub struct InMemoryEpisodicMemory {
    events: RwLock<Vec<EpisodicEvent>>,
    next_id: AtomicU64,
}

impl InMemoryEpisodicMemory {
    /// Creates an empty episodic event log.
    pub fn new() -> Self {
        Self::default()
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

impl EpisodicMemory for InMemoryEpisodicMemory {
    async fn record(&self, event_type: &str, payload: Value) -> Result<String> {
        let id = format!("event-{}", self.next_id.fetch_add(1, Ordering::Relaxed));
        self.events.write().await.push(EpisodicEvent {
            id: id.clone(),
            event_type: event_type.to_string(),
            payload,
            timestamp_ms: now_ms(),
        });
        Ok(id)
    }

    async fn query(&self, query: EpisodicQuery) -> Result<Vec<EpisodicEvent>> {
        let events = self.events.read().await;
        // Millisecond timestamps can tie for events recorded in quick
        // succession, so break ties by insertion order (later insertions
        // are "more recent") rather than relying on an unstable sort key
        // alone.
        let mut matched: Vec<(usize, EpisodicEvent)> = events
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                query
                    .event_type
                    .as_deref()
                    .is_none_or(|t| e.event_type == t)
                    && query.since_ms.is_none_or(|s| e.timestamp_ms >= s)
                    && query.until_ms.is_none_or(|u| e.timestamp_ms <= u)
            })
            .map(|(i, e)| (i, e.clone()))
            .collect();

        matched.sort_by(|(ia, a), (ib, b)| b.timestamp_ms.cmp(&a.timestamp_ms).then(ib.cmp(ia)));
        let mut matched: Vec<EpisodicEvent> = matched.into_iter().map(|(_, e)| e).collect();
        if let Some(limit) = query.limit {
            matched.truncate(limit);
        }
        Ok(matched)
    }

    async fn clear(&self) -> Result<()> {
        self.events.write().await.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::ErasedEpisodicMemory;
    use std::sync::Arc;

    #[tokio::test]
    async fn record_and_query_all() {
        let mem = InMemoryEpisodicMemory::new();
        mem.record("login", serde_json::json!({"user": "a"}))
            .await
            .unwrap();
        mem.record("logout", serde_json::json!({"user": "a"}))
            .await
            .unwrap();

        let all = mem.query(EpisodicQuery::all()).await.unwrap();
        assert_eq!(all.len(), 2);
    }

    #[tokio::test]
    async fn query_filters_by_type() {
        let mem = InMemoryEpisodicMemory::new();
        mem.record("login", Value::Null).await.unwrap();
        mem.record("logout", Value::Null).await.unwrap();
        mem.record("login", Value::Null).await.unwrap();

        let logins = mem
            .query(EpisodicQuery::all().of_type("login"))
            .await
            .unwrap();
        assert_eq!(logins.len(), 2);
        assert!(logins.iter().all(|e| e.event_type == "login"));
    }

    #[tokio::test]
    async fn query_filters_by_time_range() {
        let mem = InMemoryEpisodicMemory::new();
        let id1 = mem.record("tick", Value::Null).await.unwrap();
        let events = mem.query(EpisodicQuery::all()).await.unwrap();
        let ts = events.iter().find(|e| e.id == id1).unwrap().timestamp_ms;

        let in_range = mem
            .query(EpisodicQuery::all().between(ts, ts))
            .await
            .unwrap();
        assert_eq!(in_range.len(), 1);

        let out_of_range = mem
            .query(EpisodicQuery::all().between(ts + 1, ts + 1000))
            .await
            .unwrap();
        assert!(out_of_range.is_empty());
    }

    #[tokio::test]
    async fn query_respects_limit_and_recency_order() {
        let mem = InMemoryEpisodicMemory::new();
        for i in 0..5 {
            mem.record("tick", serde_json::json!({"i": i}))
                .await
                .unwrap();
        }
        let recent = mem.query(EpisodicQuery::all().limit(2)).await.unwrap();
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].payload["i"], 4);
        assert_eq!(recent[1].payload["i"], 3);
    }

    #[tokio::test]
    async fn clear_removes_all_events() {
        let mem = InMemoryEpisodicMemory::new();
        mem.record("tick", Value::Null).await.unwrap();
        mem.clear().await.unwrap();
        assert!(mem.query(EpisodicQuery::all()).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn erased_wrapper_works() {
        let mem: Arc<dyn ErasedEpisodicMemory> = Arc::new(InMemoryEpisodicMemory::new());
        mem.record_erased("tick", Value::Null).await.unwrap();
        assert_eq!(
            mem.query_erased(EpisodicQuery::all()).await.unwrap().len(),
            1
        );
    }
}
