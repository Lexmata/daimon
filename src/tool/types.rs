//! Types for tool calls, outputs, and configuration.

use serde::Serialize;

pub use daimon_core::ToolCall;

/// The result of executing a tool.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    /// The output content (text or serialized JSON).
    pub content: String,
    /// Whether this output represents an error.
    pub is_error: bool,
}

impl ToolOutput {
    /// Create a successful text output.
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }

    /// Create a successful output from a serializable value, encoded as JSON.
    pub fn json(value: &impl Serialize) -> crate::error::Result<Self> {
        Ok(Self {
            content: serde_json::to_string(value)?,
            is_error: false,
        })
    }

    /// Create an error output.
    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
        }
    }
}

/// Controls which tools the model is allowed to use.
#[derive(Debug, Clone, Default)]
pub enum ToolChoice {
    /// Let the model decide whether to call tools.
    #[default]
    Auto,
    /// Prevent the model from calling any tools.
    None,
    /// Force the model to call at least one tool.
    Required,
    /// Force the model to call a specific tool by name.
    Specific(String),
}
