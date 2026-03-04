//! In-memory checkpoint backend (ephemeral, useful for testing).

use std::collections::HashMap;

use tokio::sync::Mutex;

use crate::error::Result;

use super::traits::Checkpoint;
use super::types::CheckpointState;

/// Stores checkpoints in memory. Lost on process exit.
///
/// Thread-safe via internal `Mutex`.
pub struct InMemoryCheckpoint {
    store: Mutex<HashMap<String, CheckpointState>>,
}

impl InMemoryCheckpoint {
    /// Creates an empty in-memory checkpoint store.
    pub fn new() -> Self {
        Self {
            store: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryCheckpoint {
    fn default() -> Self {
        Self::new()
    }
}

impl Checkpoint for InMemoryCheckpoint {
    async fn save(&self, state: &CheckpointState) -> Result<()> {
        let mut store = self.store.lock().await;
        store.insert(state.run_id.clone(), state.clone());
        Ok(())
    }

    async fn load(&self, run_id: &str) -> Result<Option<CheckpointState>> {
        let store = self.store.lock().await;
        Ok(store.get(run_id).cloned())
    }

    async fn list_runs(&self) -> Result<Vec<String>> {
        let store = self.store.lock().await;
        Ok(store.keys().cloned().collect())
    }

    async fn delete(&self, run_id: &str) -> Result<()> {
        let mut store = self.store.lock().await;
        store.remove(run_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::types::Message;

    #[tokio::test]
    async fn test_save_and_load() {
        let cp = InMemoryCheckpoint::new();
        let state = CheckpointState::new(
            "run-1",
            vec![Message::user("hello"), Message::assistant("hi")],
            1,
        );

        cp.save(&state).await.unwrap();
        let loaded = cp.load("run-1").await.unwrap().unwrap();
        assert_eq!(loaded.run_id, "run-1");
        assert_eq!(loaded.iteration, 1);
        assert_eq!(loaded.messages.len(), 2);
    }

    #[tokio::test]
    async fn test_load_nonexistent() {
        let cp = InMemoryCheckpoint::new();
        assert!(cp.load("nope").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_overwrite() {
        let cp = InMemoryCheckpoint::new();
        let s1 = CheckpointState::new("run-1", vec![Message::user("a")], 1);
        let s2 = CheckpointState::new("run-1", vec![Message::user("a"), Message::user("b")], 2);

        cp.save(&s1).await.unwrap();
        cp.save(&s2).await.unwrap();

        let loaded = cp.load("run-1").await.unwrap().unwrap();
        assert_eq!(loaded.iteration, 2);
        assert_eq!(loaded.messages.len(), 2);
    }

    #[tokio::test]
    async fn test_list_runs() {
        let cp = InMemoryCheckpoint::new();
        cp.save(&CheckpointState::new("a", vec![], 0))
            .await
            .unwrap();
        cp.save(&CheckpointState::new("b", vec![], 0))
            .await
            .unwrap();

        let mut runs = cp.list_runs().await.unwrap();
        runs.sort();
        assert_eq!(runs, vec!["a", "b"]);
    }

    #[tokio::test]
    async fn test_delete() {
        let cp = InMemoryCheckpoint::new();
        cp.save(&CheckpointState::new("run-1", vec![], 0))
            .await
            .unwrap();
        cp.delete("run-1").await.unwrap();
        assert!(cp.load("run-1").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_completed_metadata() {
        let cp = InMemoryCheckpoint::new();
        let state = CheckpointState::new("run-1", vec![], 3)
            .mark_completed()
            .with_metadata("final_answer", serde_json::json!("42"));

        cp.save(&state).await.unwrap();
        let loaded = cp.load("run-1").await.unwrap().unwrap();
        assert!(loaded.completed);
        assert_eq!(loaded.metadata["final_answer"], serde_json::json!("42"));
    }
}
