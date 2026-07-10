//! Redis-backed task broker for multi-process distributed execution.
//!
//! Uses Redis Lists for the task queue (LPUSH/BRPOP) and Redis Hashes
//! for status tracking. Enable with `feature = "redis"`.
//!
//! ```ignore
//! use daimon::distributed::RedisBroker;
//!
//! let broker = RedisBroker::new("redis://127.0.0.1/", "daimon:tasks").await?;
//! broker.submit(AgentTask::new("Summarize this")).await?;
//! ```

use redis::aio::ConnectionManager;

use crate::error::{DaimonError, Result};

use super::broker::TaskBroker;
use super::types::{AgentTask, TaskResult, TaskStatus};

/// Distributes agent tasks via Redis.
///
/// Tasks are pushed onto a Redis list (`{prefix}:queue`) and consumed
/// via blocking pop. Status is tracked in a Redis hash (`{prefix}:status`).
/// Results are stored in a separate hash (`{prefix}:results`).
///
/// Connections are established once at construction and reused (with
/// transparent reconnection via [`ConnectionManager`]) instead of paying a
/// TCP connect + handshake on every operation. `BRPOP` blocks the whole
/// multiplexed pipeline while it waits, so polling runs on its own dedicated
/// connection rather than the one shared by the non-blocking commands.
pub struct RedisBroker {
    /// Shared connection for non-blocking commands.
    connection: ConnectionManager,
    /// Dedicated connection for the blocking `BRPOP` poll.
    poll_connection: tokio::sync::Mutex<ConnectionManager>,
    prefix: String,
}

impl RedisBroker {
    /// Connects to Redis and creates a new broker.
    ///
    /// * `url` — Redis connection URL (e.g. `redis://127.0.0.1/`)
    /// * `prefix` — key prefix for all Redis keys (e.g. `daimon:tasks`)
    pub async fn new(url: &str, prefix: impl Into<String>) -> Result<Self> {
        let client = redis::Client::open(url)
            .map_err(|e| DaimonError::Other(format!("redis broker connection: {e}")))?;

        let mut connection = ConnectionManager::new(client.clone())
            .await
            .map_err(|e| DaimonError::Other(format!("redis broker connect: {e}")))?;

        redis::cmd("PING")
            .query_async::<String>(&mut connection)
            .await
            .map_err(|e| DaimonError::Other(format!("redis broker ping: {e}")))?;

        let poll_connection = ConnectionManager::new(client)
            .await
            .map_err(|e| DaimonError::Other(format!("redis broker connect: {e}")))?;

        Ok(Self {
            connection,
            poll_connection: tokio::sync::Mutex::new(poll_connection),
            prefix: prefix.into(),
        })
    }

    fn queue_key(&self) -> String {
        format!("{}:queue", self.prefix)
    }

    fn status_key(&self) -> String {
        format!("{}:status", self.prefix)
    }

    fn result_key(&self) -> String {
        format!("{}:results", self.prefix)
    }

    /// Returns a handle to the shared managed connection.
    ///
    /// `ConnectionManager` is a cheap `Arc`-backed clone; command methods
    /// take `&mut self`, so each operation clones a handle.
    fn conn(&self) -> ConnectionManager {
        self.connection.clone()
    }
}

impl TaskBroker for RedisBroker {
    async fn submit(&self, task: AgentTask) -> Result<String> {
        let id = task.task_id.clone();
        let json = serde_json::to_string(&task)
            .map_err(|e| DaimonError::Other(format!("serialize task: {e}")))?;

        let mut conn = self.conn();

        // One atomic roundtrip: the status write and the enqueue can't be
        // observed half-done, and the submit path pays a single RTT.
        redis::pipe()
            .atomic()
            .hset(self.status_key(), &id, "pending")
            .ignore()
            .lpush(self.queue_key(), &json)
            .ignore()
            .query_async::<()>(&mut conn)
            .await
            .map_err(|e| DaimonError::Other(format!("redis submit pipeline: {e}")))?;

        Ok(id)
    }

    async fn status(&self, task_id: &str) -> Result<TaskStatus> {
        use redis::AsyncCommands;

        let mut conn = self.conn();

        let status_str: Option<String> = conn
            .hget(self.status_key(), task_id)
            .await
            .map_err(|e| DaimonError::Other(format!("redis hget status: {e}")))?;

        match status_str.as_deref() {
            Some("pending") => Ok(TaskStatus::Pending),
            Some("running") => Ok(TaskStatus::Running),
            Some("completed") => {
                let result_json: Option<String> = conn
                    .hget(self.result_key(), task_id)
                    .await
                    .map_err(|e| DaimonError::Other(format!("redis hget result: {e}")))?;

                match result_json {
                    Some(json) => {
                        let result: TaskResult = serde_json::from_str(&json)
                            .map_err(|e| DaimonError::Other(format!("deserialize result: {e}")))?;
                        Ok(TaskStatus::Completed(result))
                    }
                    None => Ok(TaskStatus::Completed(TaskResult {
                        task_id: task_id.to_string(),
                        output: String::new(),
                        iterations: 0,
                        cost: 0.0,
                        error: None,
                    })),
                }
            }
            Some(s) if s.starts_with("failed:") => Ok(TaskStatus::Failed(s[7..].to_string())),
            _ => Ok(TaskStatus::Pending),
        }
    }

    async fn receive(&self) -> Result<Option<AgentTask>> {
        let result: Option<(String, String)> = {
            let mut poll_conn = self.poll_connection.lock().await;
            redis::cmd("BRPOP")
                .arg(self.queue_key())
                .arg(1)
                .query_async(&mut *poll_conn)
                .await
                .map_err(|e| DaimonError::Other(format!("redis brpop: {e}")))?
        };

        match result {
            Some((_key, json)) => {
                let task: AgentTask = serde_json::from_str(&json)
                    .map_err(|e| DaimonError::Other(format!("deserialize task: {e}")))?;

                use redis::AsyncCommands;
                let mut conn = self.conn();
                conn.hset::<_, _, _, ()>(&self.status_key(), &task.task_id, "running")
                    .await
                    .map_err(|e| DaimonError::Other(format!("redis hset running: {e}")))?;

                Ok(Some(task))
            }
            None => Ok(None),
        }
    }

    async fn complete(&self, task_id: &str, result: TaskResult) -> Result<()> {
        let json = serde_json::to_string(&result)
            .map_err(|e| DaimonError::Other(format!("serialize result: {e}")))?;

        let mut conn = self.conn();

        // Atomic pipeline: a task can never be observed "completed" without
        // its result already written, and both writes share one RTT.
        redis::pipe()
            .atomic()
            .hset(self.result_key(), task_id, &json)
            .ignore()
            .hset(self.status_key(), task_id, "completed")
            .ignore()
            .query_async::<()>(&mut conn)
            .await
            .map_err(|e| DaimonError::Other(format!("redis complete pipeline: {e}")))?;

        Ok(())
    }

    async fn fail(&self, task_id: &str, error: String) -> Result<()> {
        use redis::AsyncCommands;

        let mut conn = self.conn();

        conn.hset::<_, _, _, ()>(&self.status_key(), task_id, format!("failed:{error}"))
            .await
            .map_err(|e| DaimonError::Other(format!("redis hset fail: {e}")))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_generation() {
        let prefix = "daimon:test";
        assert_eq!(format!("{prefix}:queue"), "daimon:test:queue");
        assert_eq!(format!("{prefix}:status"), "daimon:test:status");
        assert_eq!(format!("{prefix}:results"), "daimon:test:results");
    }

    #[test]
    fn test_task_serialization_roundtrip() {
        let task = AgentTask::new("test input")
            .with_run_id("r1")
            .with_metadata("key", serde_json::json!("val"));

        let json = serde_json::to_string(&task).unwrap();
        let deser: AgentTask = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.input, "test input");
        assert_eq!(deser.run_id.as_deref(), Some("r1"));
    }

    #[test]
    fn test_result_serialization_roundtrip() {
        let result = TaskResult {
            task_id: "t-1".into(),
            output: "result text".into(),
            iterations: 2,
            cost: 0.005,
            error: None,
        };

        let json = serde_json::to_string(&result).unwrap();
        let deser: TaskResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.output, "result text");
        assert_eq!(deser.iterations, 2);
    }
}
