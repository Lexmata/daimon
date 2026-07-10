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

use redis::aio::ConnectionManager;

use crate::error::{DaimonError, Result};
use crate::memory::Memory;
use crate::model::types::Message;

/// Stores conversation messages in a Redis list.
///
/// Each message is JSON-serialized and appended to a Redis list at the
/// configured key. Messages are returned in insertion order.
///
/// The connection is managed by [`redis::aio::ConnectionManager`], which
/// transparently reconnects (with retries) when the underlying connection
/// drops, so a transient network failure never permanently breaks the
/// memory backend.
pub struct RedisMemory {
    connection: ConnectionManager,
    key: String,
    /// Messages already fetched from the list, so `get_messages` only reads
    /// the tail that appeared since the last call instead of re-transferring
    /// and re-parsing the whole history every turn (which is O(n²) over a
    /// conversation). The list is append-only under normal operation; if
    /// `LLEN` ever reports fewer entries than cached (external `DEL`/`LTRIM`/
    /// expiry), the cache is dropped and the history refetched in full.
    cache: tokio::sync::Mutex<Vec<Message>>,
}

impl RedisMemory {
    /// Creates a new Redis memory backend.
    ///
    /// * `url` — Redis connection URL (e.g. `redis://127.0.0.1/`)
    /// * `key` — the Redis list key to store messages under
    pub async fn new(url: &str, key: impl Into<String>) -> Result<Self> {
        let client = redis::Client::open(url)
            .map_err(|e| DaimonError::Other(format!("redis connection: {e}")))?;

        let connection = ConnectionManager::new(client)
            .await
            .map_err(|e| DaimonError::Other(format!("redis connect: {e}")))?;

        Ok(Self {
            connection,
            key: key.into(),
            cache: tokio::sync::Mutex::new(Vec::new()),
        })
    }

    /// Returns the Redis key being used.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Returns a handle to the managed connection.
    ///
    /// `ConnectionManager` is a cheap `Arc`-backed clone; command methods
    /// take `&mut self`, so each operation clones a handle.
    fn conn(&self) -> ConnectionManager {
        self.connection.clone()
    }
}

impl Memory for RedisMemory {
    async fn add_message(&self, message: &Message) -> Result<()> {
        use redis::AsyncCommands;

        let serialized = serde_json::to_string(message)?;
        let mut conn = self.conn();
        conn.rpush::<_, _, ()>(&self.key, &serialized)
            .await
            .map_err(|e| DaimonError::Other(format!("redis rpush: {e}")))?;
        Ok(())
    }

    async fn get_messages(&self) -> Result<Vec<Message>> {
        use redis::AsyncCommands;

        let mut conn = self.conn();
        let mut cache = self.cache.lock().await;

        let len: usize = conn
            .llen(&self.key)
            .await
            .map_err(|e| DaimonError::Other(format!("redis llen: {e}")))?;

        if len < cache.len() {
            // The list shrank underneath us — the append-only assumption no
            // longer holds, so drop the cache and refetch from scratch.
            cache.clear();
        }

        if cache.len() < len {
            // Fetch only the entries that appeared since the last read. A
            // concurrent append between LLEN and LRANGE just means we read
            // slightly more than `len`, which is fine for an append-only list.
            let items: Vec<String> = conn
                .lrange(&self.key, cache.len() as isize, -1)
                .await
                .map_err(|e| DaimonError::Other(format!("redis lrange: {e}")))?;
            for item in items {
                let msg: Message = serde_json::from_str(&item)?;
                cache.push(msg);
            }
        }

        Ok(cache.clone())
    }

    async fn clear(&self) -> Result<()> {
        use redis::AsyncCommands;

        let mut conn = self.conn();
        let mut cache = self.cache.lock().await;
        conn.del::<_, ()>(&self.key)
            .await
            .map_err(|e| DaimonError::Other(format!("redis del: {e}")))?;
        cache.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_new_rejects_invalid_url() {
        let result = RedisMemory::new("not-a-redis-url", "conversation:test").await;
        let err = result.err().expect("invalid URL must produce an error");
        assert!(matches!(err, DaimonError::Other(_)));
        assert!(err.to_string().contains("redis connection"));
    }

    #[tokio::test]
    #[ignore = "requires a live Redis at redis://127.0.0.1/"]
    async fn test_live_round_trip() {
        let memory = RedisMemory::new("redis://127.0.0.1/", "daimon:test:round_trip")
            .await
            .unwrap();
        memory.clear().await.unwrap();

        memory.add_message(&Message::user("hello")).await.unwrap();
        memory.add_message(&Message::assistant("hi")).await.unwrap();

        let messages = memory.get_messages().await.unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content.as_deref(), Some("hello"));
        assert_eq!(messages[1].content.as_deref(), Some("hi"));
        assert_eq!(memory.key(), "daimon:test:round_trip");

        memory.clear().await.unwrap();
        assert!(memory.get_messages().await.unwrap().is_empty());
    }
}
