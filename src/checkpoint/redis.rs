//! Redis checkpoint backend.
//!
//! Stores checkpoints in Redis hashes, providing fast, shared checkpoint
//! storage accessible from multiple processes. Enable with `feature = "redis"`.
//!
//! ```ignore
//! use daimon::checkpoint::RedisCheckpoint;
//!
//! let cp = RedisCheckpoint::new("redis://127.0.0.1/", "daimon:checkpoints").await?;
//! cp.save(&state).await?;
//! ```

use crate::error::{DaimonError, Result};

use super::traits::Checkpoint;
use super::types::CheckpointState;

/// Checkpoint backend using Redis.
///
/// Each checkpoint is stored as a field in a Redis hash (`{prefix}:data`),
/// keyed by run ID with the `CheckpointState` serialized as JSON.
/// A secondary hash (`{prefix}:runs`) tracks all known run IDs.
pub struct RedisCheckpoint {
    client: redis::Client,
    prefix: String,
}

impl RedisCheckpoint {
    /// Connects to Redis and creates a new checkpoint backend.
    ///
    /// * `url` — Redis connection URL (e.g. `redis://127.0.0.1/`)
    /// * `prefix` — key prefix for all Redis keys (e.g. `daimon:checkpoints`)
    pub async fn new(url: &str, prefix: impl Into<String>) -> Result<Self> {
        let client = redis::Client::open(url)
            .map_err(|e| DaimonError::Other(format!("redis checkpoint connection: {e}")))?;

        let mut conn = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| DaimonError::Other(format!("redis checkpoint connect: {e}")))?;

        redis::cmd("PING")
            .query_async::<String>(&mut conn)
            .await
            .map_err(|e| DaimonError::Other(format!("redis checkpoint ping: {e}")))?;

        Ok(Self {
            client,
            prefix: prefix.into(),
        })
    }

    fn data_key(&self) -> String {
        format!("{}:data", self.prefix)
    }

    async fn conn(&self) -> Result<redis::aio::MultiplexedConnection> {
        self.client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| DaimonError::Other(format!("redis checkpoint conn: {e}")))
    }
}

impl Checkpoint for RedisCheckpoint {
    async fn save(&self, state: &CheckpointState) -> Result<()> {
        use redis::AsyncCommands;

        let json = serde_json::to_string(state)?;
        let mut conn = self.conn().await?;

        conn.hset::<_, _, _, ()>(&self.data_key(), &state.run_id, &json)
            .await
            .map_err(|e| DaimonError::Other(format!("redis checkpoint save: {e}")))?;

        Ok(())
    }

    async fn load(&self, run_id: &str) -> Result<Option<CheckpointState>> {
        use redis::AsyncCommands;

        let mut conn = self.conn().await?;

        let json: Option<String> = conn
            .hget(&self.data_key(), run_id)
            .await
            .map_err(|e| DaimonError::Other(format!("redis checkpoint load: {e}")))?;

        match json {
            Some(j) => {
                let state: CheckpointState = serde_json::from_str(&j)
                    .map_err(|e| DaimonError::Other(format!("redis checkpoint deserialize: {e}")))?;
                Ok(Some(state))
            }
            None => Ok(None),
        }
    }

    async fn list_runs(&self) -> Result<Vec<String>> {
        use redis::AsyncCommands;

        let mut conn = self.conn().await?;

        let keys: Vec<String> = conn
            .hkeys(&self.data_key())
            .await
            .map_err(|e| DaimonError::Other(format!("redis checkpoint list: {e}")))?;

        Ok(keys)
    }

    async fn delete(&self, run_id: &str) -> Result<()> {
        use redis::AsyncCommands;

        let mut conn = self.conn().await?;

        conn.hdel::<_, _, ()>(&self.data_key(), run_id)
            .await
            .map_err(|e| DaimonError::Other(format!("redis checkpoint delete: {e}")))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_format() {
        let prefix = "daimon:cp";
        assert_eq!(format!("{prefix}:data"), "daimon:cp:data");
    }

    #[test]
    fn test_state_serialization_roundtrip() {
        use crate::model::types::Message;

        let state = CheckpointState::new(
            "run-redis",
            vec![Message::user("test")],
            3,
        )
        .with_metadata("model", serde_json::json!("gpt-4o"));

        let json = serde_json::to_string(&state).unwrap();
        let deser: CheckpointState = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.run_id, "run-redis");
        assert_eq!(deser.iteration, 3);
        assert_eq!(deser.metadata["model"], serde_json::json!("gpt-4o"));
    }
}
