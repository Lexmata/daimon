//! A2A protocol types per the Google Agent-to-Agent specification (v0.2).
//!
//! These types map closely to the official JSON schema. All types derive
//! `Serialize`/`Deserialize` for direct JSON-RPC marshalling.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Agent Card (discovery)
// ---------------------------------------------------------------------------

/// An Agent Card advertises an agent's identity, capabilities, and endpoint.
///
/// Typically served at `/.well-known/agent.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentCard {
    /// Human-readable agent name.
    pub name: String,
    /// Description of what the agent does.
    pub description: String,
    /// Version string for this agent implementation.
    pub version: String,
    /// Base URL where this agent's A2A endpoint is hosted.
    pub url: String,
    /// Capabilities this agent supports.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<AgentCapability>,
    /// Authentication configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authentication: Option<AgentAuth>,
    /// Protocol versions supported.
    #[serde(default)]
    pub protocol_version: String,
}

/// A capability the agent supports.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentCapability {
    /// Capability type identifier (e.g. "streaming", "pushNotifications").
    #[serde(rename = "type")]
    pub capability_type: String,
    /// Human-readable description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Authentication method for the agent endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentAuth {
    /// Authentication type: "none", "api_key", "oauth", "jwt", "custom".
    #[serde(rename = "type")]
    pub auth_type: String,
    /// Human-readable instructions for authenticating.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

// ---------------------------------------------------------------------------
// Task
// ---------------------------------------------------------------------------

/// The lifecycle state of an A2A task.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum TaskState {
    /// Task was submitted and is waiting to be processed.
    Submitted,
    /// Agent is actively working on the task.
    Working,
    /// Task requires additional user input.
    InputRequired,
    /// Task completed successfully.
    Completed,
    /// Task failed.
    Failed,
    /// Task was cancelled.
    Canceled,
}

/// Status of a task including optional message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskStatus {
    /// Current state.
    pub state: TaskState,
    /// Optional human-readable status message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<A2aMessage>,
}

/// An A2A task — the primary unit of work.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct A2aTask {
    /// Unique task identifier (UUID).
    pub id: String,
    /// Context identifier grouping related tasks in a conversation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_id: Option<String>,
    /// Current task status.
    pub status: TaskStatus,
    /// Output artifacts produced by the agent.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<Artifact>,
    /// Message history for this task.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub history: Vec<A2aMessage>,
    /// Arbitrary metadata.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Message
// ---------------------------------------------------------------------------

/// Role of a message sender.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum A2aRole {
    User,
    Agent,
}

/// A message in the A2A protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct A2aMessage {
    /// Who sent this message.
    pub role: A2aRole,
    /// Content parts.
    pub parts: Vec<Part>,
    /// Optional unique message identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    /// Optional metadata.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// A content part within a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum Part {
    /// Plain text content.
    #[serde(rename = "text")]
    Text {
        /// The text content.
        text: String,
    },
    /// Structured data content.
    #[serde(rename = "data")]
    Data {
        /// The data payload.
        data: serde_json::Value,
    },
    /// File content (base64 or URI).
    #[serde(rename = "file")]
    File {
        /// File metadata.
        file: FileContent,
    },
}

/// File content in a message part.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileContent {
    /// File name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// MIME type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Base64-encoded content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes: Option<String>,
    /// URI to fetch the file from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
}

/// An output artifact produced by the agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Artifact {
    /// Unique artifact identifier.
    pub artifact_id: String,
    /// Human-readable name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Content parts.
    pub parts: Vec<Part>,
    /// Arbitrary metadata.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// JSON-RPC envelope
// ---------------------------------------------------------------------------

/// A JSON-RPC 2.0 request envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    /// Must be "2.0".
    pub jsonrpc: String,
    /// Request identifier.
    pub id: serde_json::Value,
    /// Method name.
    pub method: String,
    /// Parameters.
    #[serde(default)]
    pub params: serde_json::Value,
}

/// A JSON-RPC 2.0 response envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    /// Must be "2.0".
    pub jsonrpc: String,
    /// Must match the request id.
    pub id: serde_json::Value,
    /// The result (present on success).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    /// The error (present on failure).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// A JSON-RPC 2.0 error object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    /// Error code.
    pub code: i64,
    /// Human-readable message.
    pub message: String,
    /// Optional additional data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl JsonRpcResponse {
    /// Creates a success response.
    pub fn success(id: serde_json::Value, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    /// Creates an error response.
    pub fn error(id: serde_json::Value, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Task send/get params
// ---------------------------------------------------------------------------

/// Parameters for the `tasks/send` method.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskSendParams {
    /// Optional existing task ID to continue.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The message to send.
    pub message: A2aMessage,
    /// Optional context ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_id: Option<String>,
    /// Arbitrary metadata.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Parameters for the `tasks/get` method.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskGetParams {
    /// Task ID to retrieve.
    pub id: String,
}

/// Parameters for the `tasks/cancel` method.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TaskCancelParams {
    /// Task ID to cancel.
    pub id: String,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_card_roundtrip() {
        let card = AgentCard {
            name: "TestAgent".to_string(),
            description: "A test agent".to_string(),
            version: "1.0.0".to_string(),
            url: "https://example.com/a2a".to_string(),
            capabilities: vec![AgentCapability {
                capability_type: "streaming".to_string(),
                description: Some("Supports SSE streaming".to_string()),
            }],
            authentication: Some(AgentAuth {
                auth_type: "api_key".to_string(),
                instructions: Some("Set X-API-Key header".to_string()),
            }),
            protocol_version: "0.2".to_string(),
        };

        let json = serde_json::to_string(&card).unwrap();
        let parsed: AgentCard = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "TestAgent");
        assert_eq!(parsed.capabilities.len(), 1);
    }

    #[test]
    fn test_task_roundtrip() {
        let task = A2aTask {
            id: "task-123".to_string(),
            context_id: Some("ctx-1".to_string()),
            status: TaskStatus {
                state: TaskState::Completed,
                message: None,
            },
            artifacts: vec![Artifact {
                artifact_id: "art-1".to_string(),
                name: Some("result.txt".to_string()),
                parts: vec![Part::Text {
                    text: "Hello world".to_string(),
                }],
                metadata: HashMap::new(),
            }],
            history: vec![A2aMessage {
                role: A2aRole::User,
                parts: vec![Part::Text {
                    text: "Do something".to_string(),
                }],
                message_id: Some("msg-1".to_string()),
                metadata: HashMap::new(),
            }],
            metadata: HashMap::new(),
        };

        let json = serde_json::to_string_pretty(&task).unwrap();
        let parsed: A2aTask = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, "task-123");
        assert_eq!(parsed.status.state, TaskState::Completed);
        assert_eq!(parsed.artifacts.len(), 1);
    }

    #[test]
    fn test_json_rpc_response() {
        let resp =
            JsonRpcResponse::success(serde_json::json!(1), serde_json::json!({"status": "ok"}));
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"result\""));

        let err_resp = JsonRpcResponse::error(serde_json::json!(2), -32600, "Invalid request");
        let err_json = serde_json::to_string(&err_resp).unwrap();
        assert!(err_json.contains("\"error\""));
    }

    #[test]
    fn test_part_variants() {
        let text = Part::Text {
            text: "hello".to_string(),
        };
        let json = serde_json::to_string(&text).unwrap();
        assert!(json.contains("\"kind\":\"text\""));

        let data = Part::Data {
            data: serde_json::json!({"key": "value"}),
        };
        let json = serde_json::to_string(&data).unwrap();
        assert!(json.contains("\"kind\":\"data\""));
    }
}
