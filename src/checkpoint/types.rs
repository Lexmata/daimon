//! Checkpoint state types.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::model::types::{Message, Usage};

/// Serde adapter for [`Usage`].
///
/// `Usage` lives in `daimon-core` and does not itself derive `Serialize` /
/// `Deserialize`, so it cannot be embedded directly in a `#[derive(Serialize,
/// Deserialize)]` struct. This module mirrors its public fields and produces
/// JSON identical to a derived implementation (a map of the three token
/// counts), keeping the checkpoint format stable and self-describing.
mod usage_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use super::Usage;

    #[derive(Serialize, Deserialize)]
    struct UsageRepr {
        #[serde(default)]
        input_tokens: u32,
        #[serde(default)]
        output_tokens: u32,
        #[serde(default)]
        cached_tokens: u32,
    }

    pub(super) fn serialize<S>(usage: &Usage, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        UsageRepr {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            cached_tokens: usage.cached_tokens,
        }
        .serialize(serializer)
    }

    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<Usage, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = UsageRepr::deserialize(deserializer)?;
        Ok(Usage {
            input_tokens: repr.input_tokens,
            output_tokens: repr.output_tokens,
            cached_tokens: repr.cached_tokens,
        })
    }
}

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
    /// Cumulative cost in USD spent so far in this run.
    ///
    /// Persisted so a cross-process resume can restore prior spend into the
    /// cost tracker; otherwise `max_budget` would only bound post-resume
    /// spend and the returned cost would under-report the true total.
    #[serde(default)]
    pub cumulative_cost: f64,
    /// Aggregated token usage accumulated so far in this run.
    #[serde(default, with = "usage_serde")]
    pub usage: Usage,
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
            cumulative_cost: 0.0,
            usage: Usage::default(),
        }
    }

    /// Records the cumulative cost and token usage spent up to this checkpoint.
    ///
    /// Set before saving so a later resume can reseed the cost tracker and
    /// usage accumulators with the pre-interruption totals.
    pub fn with_cost_usage(mut self, cost: f64, usage: Usage) -> Self {
        self.cumulative_cost = cost;
        self.usage = usage;
        self
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
