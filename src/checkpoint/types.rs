//! Checkpoint state types.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::model::types::Message;

/// Serializable snapshot of an agent run at a specific iteration.
///
/// Contains everything needed to resume the run: the message history,
/// the iteration count, and arbitrary metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointState {
    /// Unique identifier for the agent run this checkpoint belongs to.
    pub run_id: String,
    /// The complete message history at this point.
    pub messages: Vec<Message>,
    /// The iteration index when this checkpoint was taken (1-based).
    pub iteration: usize,
    /// Whether the run has completed.
    pub completed: bool,
    /// Arbitrary key-value metadata for user-specific state.
    pub metadata: HashMap<String, serde_json::Value>,
    /// Unix timestamp (seconds since epoch) when the checkpoint was created.
    pub created_at: u64,
}

impl CheckpointState {
    /// Creates a new checkpoint for the given run.
    pub fn new(run_id: impl Into<String>, messages: Vec<Message>, iteration: usize) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            run_id: run_id.into(),
            messages,
            iteration,
            completed: false,
            metadata: HashMap::new(),
            created_at: now,
        }
    }

    /// Marks this checkpoint as completed.
    pub fn mark_completed(mut self) -> Self {
        self.completed = true;
        self
    }

    /// Sets a metadata key.
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}
