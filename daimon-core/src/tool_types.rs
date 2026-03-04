//! Tool call types shared between providers and the core framework.

use serde::{Deserialize, Serialize};

/// A tool invocation requested by a model, containing the tool name and arguments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-assigned identifier for this tool call.
    pub id: String,
    /// The name of the tool to invoke.
    pub name: String,
    /// JSON arguments to pass to the tool.
    pub arguments: serde_json::Value,
}
