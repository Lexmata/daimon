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
        let bucket = bucket.into();
        let client = async_nats::connect(url)
            .await
            .map_err(|e| DaimonError::Other(format!("nats kv connect: {e}")))?;

        let jetstream = async_nats::jetstream::new(client);

        // Create-or-open the bucket. `create_key_value` fails if the
        // underlying KV stream already exists (e.g. a second process
        // connecting to the same checkpoint bucket), so on error we fall back
        // to opening the existing bucket — mirroring the NATS task broker's
        // status-bucket setup.
        let kv = match jetstream
            .create_key_value(async_nats::jetstream::kv::Config {
                bucket: bucket.clone(),
                ..Default::default()
            })
            .await
        {
            Ok(store) => store,
            Err(_) => jetstream
                .get_key_value(&bucket)
                .await
                .map_err(|e| DaimonError::Other(format!("nats kv open bucket: {e}")))?,
        };

        Ok(Self { kv })
    }

    /// Creates a checkpoint backend from an existing KV store handle.
    pub fn from_store(kv: async_nats::jetstream::kv::Store) -> Self {
        Self { kv }
    }

    /// Validates that `run_id` is safe to embed in a NATS KV key.
    ///
    /// KV keys are subject tokens: `.` is the token separator, so a run_id
    /// containing dots would silently create extra subject tokens under the
    /// `cp.` prefix (and `..` produces an outright invalid key), while
    /// characters outside the NATS key charset are rejected by the server
    /// with an opaque error. Following the allowlist approach of
    /// [`FileCheckpoint`](super::FileCheckpoint)'s run-id validation, we
    /// restrict ids to `[A-Za-z0-9_-]` and reject the empty string, failing
    /// fast with a clear error instead.
    fn validate_run_id(run_id: &str) -> Result<()> {
        if run_id.is_empty() {
            return Err(DaimonError::Other(
                "invalid run_id: must not be empty".to_string(),
            ));
        }
        if !run_id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
        {
            return Err(DaimonError::Other(format!(
                "invalid run_id '{run_id}': only [A-Za-z0-9_-] characters are allowed in NATS KV keys"
            )));
        }
        Ok(())
    }

    fn key(run_id: &str) -> Result<String> {
        Self::validate_run_id(run_id)?;
        Ok(format!("cp.{run_id}"))
    }
}

impl Checkpoint for NatsKvCheckpoint {
    async fn save(&self, state: &CheckpointState) -> Result<()> {
        let key = Self::key(&state.run_id)?;
        let json = serde_json::to_string(state)?;
        self.kv
            .put(key, json.into())
            .await
            .map_err(|e| DaimonError::Other(format!("nats kv put: {e}")))?;
        Ok(())
    }

    async fn load(&self, run_id: &str) -> Result<Option<CheckpointState>> {
        // `Store::get` already maps a missing key to `Ok(None)`; an `Err` is
        // always a real failure (invalid key, timeout, JetStream error), so it
        // is propagated with its typed kind rather than string-matched — the
        // previous `contains("not found")` check could swallow genuine errors
        // whose message happened to contain that text.
        match self.kv.get(Self::key(run_id)?).await {
            Ok(Some(bytes)) => {
                let state: CheckpointState = serde_json::from_slice(&bytes)
                    .map_err(|e| DaimonError::Other(format!("nats kv deserialize: {e}")))?;
                Ok(Some(state))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(DaimonError::Other(format!(
                "nats kv get ({kind:?}): {e}",
                kind = e.kind()
            ))),
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
            .purge(Self::key(run_id)?)
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
        assert_eq!(NatsKvCheckpoint::key("run-1").unwrap(), "cp.run-1");
        assert_eq!(NatsKvCheckpoint::key("abc").unwrap(), "cp.abc");
    }

    #[test]
    fn test_validate_run_id_accepts_safe_ids() {
        assert!(NatsKvCheckpoint::validate_run_id("run-1").is_ok());
        assert!(NatsKvCheckpoint::validate_run_id("Run_2").is_ok());
        assert!(NatsKvCheckpoint::validate_run_id("abc123").is_ok());
    }

    #[test]
    fn test_validate_run_id_rejects_unsafe_ids() {
        // `.` is the NATS subject token separator — a run_id containing dots
        // would create extra tokens under the `cp.` prefix.
        assert!(NatsKvCheckpoint::validate_run_id("a.b").is_err());
        assert!(NatsKvCheckpoint::validate_run_id("..").is_err());
        assert!(NatsKvCheckpoint::validate_run_id("").is_err());
        assert!(NatsKvCheckpoint::validate_run_id("run 1").is_err());
        assert!(NatsKvCheckpoint::validate_run_id("run/1").is_err());
        assert!(NatsKvCheckpoint::validate_run_id("run*").is_err());
        assert!(NatsKvCheckpoint::validate_run_id("run>").is_err());
    }

    #[test]
    fn test_key_rejects_invalid_run_id() {
        assert!(NatsKvCheckpoint::key("a.b").is_err());
        assert!(NatsKvCheckpoint::key("").is_err());
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
