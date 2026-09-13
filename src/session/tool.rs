//! A [`Tool`] that lets a running agent send messages to other sessions.
//!
//! Wraps a [`SharedSessionBus`] and the sending session's [`SessionId`] so a
//! model can, mid-conversation, address a peer session or broadcast to all —
//! the messaging analogue of the handoff crate's `transfer_to_*` tools, but
//! without transferring control.

use std::sync::Arc;

use crate::error::Result;
use crate::tool::{Tool, ToolOutput};

use daimon_core::session::{ErasedSessionBus, SessionId, SessionMessage, SharedSessionBus};

/// Tool name advertised to the model.
const TOOL_NAME: &str = "send_session_message";

/// A tool that publishes a [`SessionMessage`] on a [`SharedSessionBus`] on
/// behalf of the session that owns it.
///
/// The tool's `from` address is fixed at construction (the owning session);
/// the model chooses the recipient and body. Omitting the recipient — or
/// passing `broadcast: true` — sends to every subscribed session.
///
/// ```ignore
/// use std::sync::Arc;
/// use daimon::session::{InProcessSessionBus, SendMessageTool, SessionId};
///
/// let bus = Arc::new(InProcessSessionBus::new());
/// let tool = SendMessageTool::new(SessionId::new("planner"), bus.clone());
/// let agent = Agent::builder().model(model).tool(tool).build()?;
/// ```
pub struct SendMessageTool {
    from: SessionId,
    bus: SharedSessionBus,
}

impl SendMessageTool {
    /// Creates a tool that sends as `from` over `bus`.
    ///
    /// Accepts any bus implementation (it is erased into a
    /// [`SharedSessionBus`]); pass an `Arc<InProcessSessionBus>` for the
    /// in-process case.
    pub fn new<B>(from: impl Into<SessionId>, bus: Arc<B>) -> Self
    where
        B: ErasedSessionBus + 'static,
    {
        Self {
            from: from.into(),
            bus,
        }
    }

    /// Creates a tool from an already-erased [`SharedSessionBus`].
    pub fn from_shared(from: impl Into<SessionId>, bus: SharedSessionBus) -> Self {
        Self {
            from: from.into(),
            bus,
        }
    }
}

impl Tool for SendMessageTool {
    fn name(&self) -> &str {
        TOOL_NAME
    }

    fn description(&self) -> &str {
        "Send a message to another agent session. Provide the recipient session id in `to` \
         for a directed message, or set `broadcast` to true to send to every session. \
         Use this to coordinate with peer sessions running in parallel."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient session id. Omit when broadcasting."
                },
                "body": {
                    "type": "string",
                    "description": "The message content to deliver."
                },
                "broadcast": {
                    "type": "boolean",
                    "description": "If true, deliver to all sessions and ignore `to`.",
                    "default": false
                }
            },
            "required": ["body"]
        })
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<ToolOutput> {
        let body = match input.get("body").and_then(|v| v.as_str()) {
            Some(b) if !b.is_empty() => b,
            _ => {
                return Ok(ToolOutput::error(
                    "`body` is required and must be a non-empty string",
                ));
            }
        };

        let broadcast = input
            .get("broadcast")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if broadcast {
            let msg = SessionMessage::broadcast(self.from.clone(), body);
            self.bus.broadcast_erased(msg).await?;
            return Ok(ToolOutput::text("Broadcast delivered to all sessions."));
        }

        let to = match input.get("to").and_then(|v| v.as_str()) {
            Some(t) if !t.is_empty() => t,
            _ => {
                return Ok(ToolOutput::error(
                    "provide `to` for a directed message, or set `broadcast` to true",
                ));
            }
        };

        let msg = SessionMessage::direct(self.from.clone(), to, body);
        self.bus.send_erased(msg).await?;
        Ok(ToolOutput::text(format!("Message sent to '{to}'.")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{InProcessSessionBus, SessionBus};

    #[tokio::test]
    async fn test_tool_sends_directed_message() {
        let bus = Arc::new(InProcessSessionBus::new());
        let mut bob = bus.subscribe(SessionId::new("bob")).await.unwrap();

        let tool = SendMessageTool::new("alice", bus.clone());
        let out = tool
            .execute(&serde_json::json!({ "to": "bob", "body": "ping" }))
            .await
            .unwrap();
        assert!(!out.is_error);

        let got = bob.recv().await.unwrap().unwrap();
        assert_eq!(got.body, "ping");
        assert_eq!(got.from, SessionId::new("alice"));
    }

    #[tokio::test]
    async fn test_tool_broadcasts() {
        let bus = Arc::new(InProcessSessionBus::new());
        let mut a = bus.subscribe(SessionId::new("a")).await.unwrap();
        let mut b = bus.subscribe(SessionId::new("b")).await.unwrap();

        let tool = SendMessageTool::new("sender", bus.clone());
        let out = tool
            .execute(&serde_json::json!({ "body": "notice", "broadcast": true }))
            .await
            .unwrap();
        assert!(!out.is_error);

        assert_eq!(a.recv().await.unwrap().unwrap().body, "notice");
        assert_eq!(b.recv().await.unwrap().unwrap().body, "notice");
    }

    #[tokio::test]
    async fn test_missing_body_is_tool_error() {
        let bus = Arc::new(InProcessSessionBus::new());
        let tool = SendMessageTool::new("alice", bus);
        let out = tool
            .execute(&serde_json::json!({ "to": "bob" }))
            .await
            .unwrap();
        assert!(out.is_error);
    }

    #[tokio::test]
    async fn test_directed_without_to_is_tool_error() {
        let bus = Arc::new(InProcessSessionBus::new());
        let tool = SendMessageTool::new("alice", bus);
        let out = tool
            .execute(&serde_json::json!({ "body": "hi" }))
            .await
            .unwrap();
        assert!(out.is_error);
    }
}
