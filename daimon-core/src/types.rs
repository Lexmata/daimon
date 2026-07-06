//! Core message and request types shared across all model providers.

use serde::{Deserialize, Serialize};

use crate::tool_types::ToolCall;

/// The role of a participant in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System-level instructions.
    System,
    /// The human user.
    User,
    /// The AI assistant.
    Assistant,
    /// A tool result message.
    Tool,
}

/// A single message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Who produced this message.
    pub role: Role,
    /// Text content of the message (may be `None` for tool-call-only messages).
    pub content: Option<String>,
    /// Tool calls requested by the assistant.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// The tool call ID this message is responding to (for `Role::Tool`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    /// Create a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    /// Create an assistant message with text content.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    /// Create an assistant message that contains tool calls (no text).
    pub fn assistant_with_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: None,
            tool_calls,
            tool_call_id: None,
        }
    }

    /// Create an assistant message that carries both text and tool calls.
    ///
    /// Models such as Anthropic and Gemini emit reasoning text alongside a
    /// tool call in the same turn; preserving both keeps replayed history
    /// faithful to what the model produced.
    pub fn assistant_with_text_and_tool_calls(
        content: impl Into<String>,
        tool_calls: Vec<ToolCall>,
    ) -> Self {
        Self {
            role: Role::Assistant,
            content: Some(content.into()),
            tool_calls,
            tool_call_id: None,
        }
    }

    /// Create a tool result message.
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: Some(content.into()),
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

/// Describes a tool that can be passed to a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Tool name.
    pub name: String,
    /// Human-readable description of what the tool does.
    pub description: String,
    /// JSON Schema describing the tool's input parameters.
    pub parameters: serde_json::Value,
}

/// A request to send to a model.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    /// The conversation messages.
    pub messages: Vec<Message>,
    /// Tool specifications available for this request.
    pub tools: Vec<ToolSpec>,
    /// Sampling temperature (0.0–2.0).
    pub temperature: Option<f32>,
    /// Maximum tokens to generate.
    pub max_tokens: Option<u32>,
}

impl ChatRequest {
    /// Create a new request with only messages.
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
        }
    }
}

/// Why the model stopped generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The model finished its response naturally.
    EndTurn,
    /// The model wants to call one or more tools.
    ToolUse,
    /// The response was truncated due to max token limits.
    MaxTokens,
}

/// A complete response from a model.
#[derive(Debug, Clone)]
pub struct ChatResponse {
    /// The assistant message produced by the model.
    pub message: Message,
    /// Why generation stopped.
    pub stop_reason: StopReason,
    /// Token usage for this request (if reported by the provider).
    pub usage: Option<Usage>,
}

impl ChatResponse {
    /// Get the text content of the response.
    pub fn text(&self) -> &str {
        self.message.content.as_deref().unwrap_or("")
    }

    /// Get the tool calls in this response.
    pub fn tool_calls(&self) -> &[ToolCall] {
        &self.message.tool_calls
    }

    /// Check if the response contains tool calls.
    pub fn has_tool_calls(&self) -> bool {
        !self.message.tool_calls.is_empty()
    }
}

/// Token usage statistics for a single model request.
#[derive(Debug, Clone, Default)]
pub struct Usage {
    /// Number of input (prompt) tokens.
    pub input_tokens: u32,
    /// Number of output (completion) tokens.
    pub output_tokens: u32,
    /// Number of input tokens served from the provider's cache.
    pub cached_tokens: u32,
}

impl Usage {
    /// Total tokens (input + output).
    pub fn total_tokens(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }

    /// Accumulate another usage into this one.
    pub fn accumulate(&mut self, other: &Usage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cached_tokens += other.cached_tokens;
    }
}
