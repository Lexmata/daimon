//! Redis-backed conversation memory.
//!
//! Persists message history to a Redis list, enabling shared or durable
//! conversation state across processes.
//!
//! Enable with `feature = "redis"`.
//!
//! ```ignore
//! use daimon::memory::RedisMemory;
//!
//! let memory = RedisMemory::new("redis://127.0.0.1/", "conversation:abc123").await?;
//! let agent = Agent::builder()
//!     .model(model)
//!     .memory(memory)
//!     .build()?;
//! ```

use tokio::sync::Mutex;

use crate::error::{DaimonError, Result};
use crate::memory::Memory;
use crate::model::types::Message;

/// Stores conversation messages in a Redis list.
///
/// Each message is JSON-serialized and appended to a Redis list at the
/// configured key. Messages are returned in insertion order.
pub struct RedisMemory {
    client: redis::Client,
    key: String,
    connection: Mutex<Option<redis::aio::MultiplexedConnection>>,
}

impl RedisMemory {
    /// Creates a new Redis memory backend.
    ///
    /// * `url` — Redis connection URL (e.g. `redis://127.0.0.1/`)
    /// * `key` — the Redis list key to store messages under
    pub async fn new(url: &str, key: impl Into<String>) -> Result<Self> {
        let client = redis::Client::open(url)
            .map_err(|e| DaimonError::Other(format!("redis connection: {e}")))?;

        let conn: redis::aio::MultiplexedConnection = client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| DaimonError::Other(format!("redis connect: {e}")))?;

        Ok(Self {
            client,
            key: key.into(),
            connection: Mutex::new(Some(conn)),
        })
    }

    /// Returns the Redis key being used.
    pub fn key(&self) -> &str {
        &self.key
    }

    async fn conn(&self) -> Result<redis::aio::MultiplexedConnection> {
        let mut guard = self.connection.lock().await;
        if let Some(conn) = guard.as_ref() {
            return Ok(conn.clone());
        }

        let conn: redis::aio::MultiplexedConnection = self
            .client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| DaimonError::Other(format!("redis reconnect: {e}")))?;

        *guard = Some(conn.clone());
        Ok(conn)
    }
}

impl Memory for RedisMemory {
    async fn add_message(&self, message: Message) -> Result<()> {
        use redis::AsyncCommands;

        let serialized = serde_json::to_string(&message)?;
        let mut conn = self.conn().await?;
        conn.rpush::<_, _, ()>(&self.key, &serialized)
            .await
            .map_err(|e| DaimonError::Other(format!("redis rpush: {e}")))?;
        Ok(())
    }

    async fn get_messages(&self) -> Result<Vec<Message>> {
        use redis::AsyncCommands;

        let mut conn = self.conn().await?;
        let items: Vec<String> = conn
            .lrange(&self.key, 0, -1)
            .await
            .map_err(|e| DaimonError::Other(format!("redis lrange: {e}")))?;

        let mut messages = Vec::with_capacity(items.len());
        for item in items {
            let msg: Message = serde_json::from_str(&item)?;
            messages.push(msg);
        }
        Ok(messages)
    }

    async fn clear(&self) -> Result<()> {
        use redis::AsyncCommands;

        let mut conn = self.conn().await?;
        conn.del::<_, ()>(&self.key)
            .await
            .map_err(|e| DaimonError::Other(format!("redis del: {e}")))?;
        Ok(())
    }
}
