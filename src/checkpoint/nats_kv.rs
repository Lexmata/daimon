//! NATS JetStream KV checkpoint backend.
//!
//! Stores checkpoints in a NATS JetStream key-value bucket, giving you
//! distributed, replicated checkpoint storage with no external database.
//! Enable with `feature = "nats"`.
//!
//! ```ignore
//! use daimon::checkpoint::NatsKvCheckpoint;
//!
//! let cp = NatsKvCheckpoint::connect("nats://127.0.0.1:4222", "daimon-checkpoints").await?;
//! cp.save(&state).await?;
//! ```

use crate::error::{DaimonError, Result};

use super::traits::Checkpoint;
use super::types::CheckpointState;

/// Checkpoint backend using NATS JetStream key-value store.
///
/// Each run is stored as a key in the configured KV bucket, with the
/// `CheckpointState` serialized as JSON. Keys are prefixed with `cp.`.
pub struct NatsKvCheckpoint {
    kv: async_nats::jetstream::kv::Store,
}

impl NatsKvCheckpoint {
    /// Connects to NATS and opens (or creates) a KV bucket for checkpoints.
    ///
    /// * `url` — NATS server URL (e.g. `nats://127.0.0.1:4222`)
    /// * `bucket` — KV bucket name (e.g. `daimon-checkpoints`)
    pub async fn connect(url: &str, bucket: impl Into<String>) -> Result<Self> {
        let client = async_nats::connect(url)
            .await
            .map_err(|e| DaimonError::Other(format!("nats kv connect: {e}")))?;

        let jetstream = async_nats::jetstream::new(client);

        let kv = jetstream
            .create_key_value(async_nats::jetstream::kv::Config {
                bucket: bucket.into(),
                ..Default::default()
            })
            .await
            .map_err(|e| DaimonError::Other(format!("nats kv create bucket: {e}")))?;

        Ok(Self { kv })
    }

    /// Creates a checkpoint backend from an existing KV store handle.
    pub fn from_store(kv: async_nats::jetstream::kv::Store) -> Self {
        Self { kv }
    }

    fn key(run_id: &str) -> String {
        format!("cp.{run_id}")
    }
}

impl Checkpoint for NatsKvCheckpoint {
    async fn save(&self, state: &CheckpointState) -> Result<()> {
        let json = serde_json::to_string(state)?;
        self.kv
            .put(Self::key(&state.run_id), json.into())
            .await
            .map_err(|e| DaimonError::Other(format!("nats kv put: {e}")))?;
        Ok(())
    }

    async fn load(&self, run_id: &str) -> Result<Option<CheckpointState>> {
        match self.kv.get(Self::key(run_id)).await {
            Ok(Some(bytes)) => {
                let state: CheckpointState = serde_json::from_slice(&bytes)
                    .map_err(|e| DaimonError::Other(format!("nats kv deserialize: {e}")))?;
                Ok(Some(state))
            }
            Ok(None) => Ok(None),
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("not found") || msg.contains("no message") {
                    Ok(None)
                } else {
                    Err(DaimonError::Other(format!("nats kv get: {e}")))
                }
            }
        }
    }

    async fn list_runs(&self) -> Result<Vec<String>> {
        use futures::TryStreamExt;

        let keys: Vec<String> = self
            .kv
            .keys()
            .await
            .map_err(|e| DaimonError::Other(format!("nats kv keys: {e}")))?
            .try_collect()
            .await
            .map_err(|e| DaimonError::Other(format!("nats kv keys collect: {e}")))?;

        let prefix = "cp.";
        let runs = keys
            .into_iter()
            .filter_map(|k| k.strip_prefix(prefix).map(String::from))
            .collect();

        Ok(runs)
    }

    async fn delete(&self, run_id: &str) -> Result<()> {
        self.kv
            .purge(Self::key(run_id))
            .await
            .map_err(|e| DaimonError::Other(format!("nats kv delete: {e}")))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_format() {
        assert_eq!(NatsKvCheckpoint::key("run-1"), "cp.run-1");
        assert_eq!(NatsKvCheckpoint::key("abc"), "cp.abc");
    }

    #[test]
    fn test_state_serialization_roundtrip() {
        use crate::model::types::Message;

        let state = CheckpointState::new(
            "run-kv",
            vec![Message::user("hello"), Message::assistant("hi")],
            2,
        )
        .mark_completed()
        .with_metadata("key", serde_json::json!("val"));

        let json = serde_json::to_string(&state).unwrap();
        let deser: CheckpointState = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.run_id, "run-kv");
        assert_eq!(deser.iteration, 2);
        assert!(deser.completed);
        assert_eq!(deser.messages.len(), 2);
    }
}
