//! SQLite-backed conversation memory.
//!
//! Persists messages to a SQLite database. Requires the `sqlite` feature flag.
//!
//! # Example
//!
//! ```ignore
//! use daimon::memory::SqliteMemory;
//!
//! let memory = SqliteMemory::open("./conversations.db").await?;
//! // or in-memory:
//! let memory = SqliteMemory::in_memory().await?;
//! ```

use std::sync::Arc;

use rusqlite::{Connection, params};
use tokio::sync::Mutex;

use crate::error::{DaimonError, Result};
use crate::memory::Memory;
use crate::model::types::{Message, Role};
use crate::tool::ToolCall;

/// A [`Memory`] backend that stores messages in SQLite.
///
/// Thread-safe via internal `Mutex<Connection>`. All operations use
/// `tokio::task::spawn_blocking` to avoid blocking the async runtime.
pub struct SqliteMemory {
    conn: Arc<Mutex<Connection>>,
    session_id: String,
}

impl SqliteMemory {
    /// Opens (or creates) a SQLite database at the given path.
    pub async fn open(path: impl Into<String>) -> Result<Self> {
        let path = path.into();
        let conn = tokio::task::spawn_blocking(move || {
            Connection::open(&path).map_err(|e| DaimonError::Other(format!("sqlite open: {e}")))
        })
        .await
        .map_err(|e| DaimonError::Other(format!("spawn_blocking: {e}")))??;

        let mem = Self {
            conn: Arc::new(Mutex::new(conn)),
            session_id: uuid_v4(),
        };
        mem.create_tables().await?;
        Ok(mem)
    }

    /// Creates an in-memory SQLite database (data lost when dropped).
    pub async fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| DaimonError::Other(format!("sqlite open: {e}")))?;

        let mem = Self {
            conn: Arc::new(Mutex::new(conn)),
            session_id: uuid_v4(),
        };
        mem.create_tables().await?;
        Ok(mem)
    }

    /// Sets a custom session ID for partitioning conversations.
    pub fn with_session_id(mut self, id: impl Into<String>) -> Self {
        self.session_id = id.into();
        self
    }

    async fn create_tables(&self) -> Result<()> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.blocking_lock();
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT,
                    tool_calls TEXT,
                    tool_call_id TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id, id);",
            )
            .map_err(|e| DaimonError::Other(format!("sqlite create tables: {e}")))
        })
        .await
        .map_err(|e| DaimonError::Other(format!("spawn_blocking: {e}")))?
    }
}

impl Memory for SqliteMemory {
    async fn add_message(&self, message: Message) -> Result<()> {
        let conn = self.conn.clone();
        let session_id = self.session_id.clone();
        let role = role_to_str(&message.role);
        let content = message.content.clone();
        let tool_calls = if message.tool_calls.is_empty() {
            None
        } else {
            Some(
                serde_json::to_string(&message.tool_calls)
                    .map_err(|e| DaimonError::Other(format!("serialize tool_calls: {e}")))?,
            )
        };
        let tool_call_id = message.tool_call_id.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn.blocking_lock();
            conn.execute(
                "INSERT INTO messages (session_id, role, content, tool_calls, tool_call_id)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![session_id, role, content, tool_calls, tool_call_id],
            )
            .map_err(|e| DaimonError::Other(format!("sqlite insert: {e}")))?;
            Ok(())
        })
        .await
        .map_err(|e| DaimonError::Other(format!("spawn_blocking: {e}")))?
    }

    async fn get_messages(&self) -> Result<Vec<Message>> {
        let conn = self.conn.clone();
        let session_id = self.session_id.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn.blocking_lock();
            let mut stmt = conn
                .prepare(
                    "SELECT role, content, tool_calls, tool_call_id
                     FROM messages
                     WHERE session_id = ?1
                     ORDER BY id ASC",
                )
                .map_err(|e| DaimonError::Other(format!("sqlite prepare: {e}")))?;

            let rows = stmt
                .query_map(params![session_id], |row| {
                    let role: String = row.get(0)?;
                    let content: Option<String> = row.get(1)?;
                    let tool_calls_json: Option<String> = row.get(2)?;
                    let tool_call_id: Option<String> = row.get(3)?;
                    Ok((role, content, tool_calls_json, tool_call_id))
                })
                .map_err(|e| DaimonError::Other(format!("sqlite query: {e}")))?;

            let mut messages = Vec::new();
            for row in rows {
                let (role_str, content, tc_json, tc_id) =
                    row.map_err(|e| DaimonError::Other(format!("sqlite row: {e}")))?;

                let role = str_to_role(&role_str);
                let tool_calls: Vec<ToolCall> = tc_json
                    .as_deref()
                    .map(|s| serde_json::from_str(s).unwrap_or_default())
                    .unwrap_or_default();

                messages.push(Message {
                    role,
                    content,
                    tool_calls,
                    tool_call_id: tc_id,
                });
            }

            Ok(messages)
        })
        .await
        .map_err(|e| DaimonError::Other(format!("spawn_blocking: {e}")))?
    }

    async fn clear(&self) -> Result<()> {
        let conn = self.conn.clone();
        let session_id = self.session_id.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn.blocking_lock();
            conn.execute(
                "DELETE FROM messages WHERE session_id = ?1",
                params![session_id],
            )
            .map_err(|e| DaimonError::Other(format!("sqlite delete: {e}")))?;
            Ok(())
        })
        .await
        .map_err(|e| DaimonError::Other(format!("spawn_blocking: {e}")))?
    }
}

fn role_to_str(role: &Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

fn str_to_role(s: &str) -> Role {
    match s {
        "system" => Role::System,
        "user" => Role::User,
        "assistant" => Role::Assistant,
        "tool" => Role::Tool,
        _ => Role::User,
    }
}

fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{ts:032x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_in_memory_add_and_get() {
        let mem = SqliteMemory::in_memory().await.unwrap();
        mem.add_message(Message::user("hello")).await.unwrap();
        mem.add_message(Message::assistant("hi")).await.unwrap();

        let messages = mem.get_messages().await.unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content.as_deref(), Some("hello"));
        assert_eq!(messages[1].content.as_deref(), Some("hi"));
    }

    #[tokio::test]
    async fn test_clear() {
        let mem = SqliteMemory::in_memory().await.unwrap();
        mem.add_message(Message::user("hello")).await.unwrap();
        assert_eq!(mem.get_messages().await.unwrap().len(), 1);

        mem.clear().await.unwrap();
        assert_eq!(mem.get_messages().await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_roles_preserved() {
        let mem = SqliteMemory::in_memory().await.unwrap();
        mem.add_message(Message::system("sys")).await.unwrap();
        mem.add_message(Message::user("usr")).await.unwrap();
        mem.add_message(Message::assistant("ast")).await.unwrap();
        mem.add_message(Message::tool_result("id1", "result"))
            .await
            .unwrap();

        let messages = mem.get_messages().await.unwrap();
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[1].role, Role::User);
        assert_eq!(messages[2].role, Role::Assistant);
        assert_eq!(messages[3].role, Role::Tool);
        assert_eq!(messages[3].tool_call_id.as_deref(), Some("id1"));
    }

    #[tokio::test]
    async fn test_tool_calls_round_trip() {
        let mem = SqliteMemory::in_memory().await.unwrap();
        let msg = Message::assistant_with_tool_calls(vec![ToolCall {
            id: "tc_1".into(),
            name: "calc".into(),
            arguments: serde_json::json!({"expr": "1+1"}),
        }]);
        mem.add_message(msg).await.unwrap();

        let messages = mem.get_messages().await.unwrap();
        assert_eq!(messages[0].tool_calls.len(), 1);
        assert_eq!(messages[0].tool_calls[0].name, "calc");
        assert_eq!(messages[0].tool_calls[0].arguments["expr"], "1+1");
    }

    #[tokio::test]
    async fn test_session_isolation() {
        let mem1 = SqliteMemory::in_memory().await.unwrap();
        let mem2 = SqliteMemory {
            conn: mem1.conn.clone(),
            session_id: "other_session".into(),
        };

        mem1.add_message(Message::user("session1")).await.unwrap();
        mem2.add_message(Message::user("session2")).await.unwrap();

        assert_eq!(mem1.get_messages().await.unwrap().len(), 1);
        assert_eq!(mem2.get_messages().await.unwrap().len(), 1);
        assert_eq!(
            mem1.get_messages().await.unwrap()[0].content.as_deref(),
            Some("session1")
        );
    }

    #[test]
    fn test_role_round_trip() {
        for role in [Role::System, Role::User, Role::Assistant, Role::Tool] {
            let s = role_to_str(&role);
            let round_tripped = str_to_role(s);
            assert_eq!(role, round_tripped);
        }
    }
}
