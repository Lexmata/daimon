//! Error types for the Daimon agent framework.
//!
//! All fallible operations in Daimon return [`Result<T>`], which is an alias
//! for `std::result::Result<T, DaimonError>`.

use thiserror::Error;

/// The central error type for all Daimon operations.
#[derive(Error, Debug)]
pub enum DaimonError {
    /// An error originating from a model provider (API error, bad response, etc.).
    #[error("model error: {0}")]
    Model(String),

    /// A tool failed during execution.
    #[error("tool execution failed for '{tool}': {message}")]
    ToolExecution {
        /// Name of the tool that failed.
        tool: String,
        /// Description of the failure.
        message: String,
    },

    /// The requested tool was not found in the registry.
    #[error("tool '{0}' not found in registry")]
    ToolNotFound(String),

    /// Attempted to register a tool with a name that already exists.
    #[error("duplicate tool '{0}' in registry")]
    DuplicateTool(String),

    /// The agent builder failed validation (e.g. missing required model).
    #[error("agent builder validation failed: {0}")]
    Builder(String),

    /// The agent exceeded the configured maximum number of iterations.
    #[error("max iterations ({0}) exceeded")]
    MaxIterations(usize),

    /// A serialization or deserialization error.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Tool input failed JSON Schema validation.
    #[error("schema validation failed for tool '{tool}': {errors}")]
    SchemaValidation {
        /// Name of the tool whose input failed validation.
        tool: String,
        /// Human-readable description of validation errors.
        errors: String,
    },

    /// An HTTP request failed.
    #[error("HTTP request failed: {0}")]
    #[cfg(any(
        feature = "openai",
        feature = "anthropic",
        feature = "gemini",
        feature = "azure",
        feature = "ollama",
        feature = "mcp",
    ))]
    Http(#[from] reqwest::Error),

    /// A stream was closed before completing.
    #[error("stream closed unexpectedly")]
    StreamClosed,

    /// A request timed out.
    #[error("request timed out after {0:?}")]
    Timeout(std::time::Duration),

    /// The operation was cancelled via a cancellation token.
    #[error("operation cancelled")]
    Cancelled,

    /// An orchestration error (chain or graph execution failure).
    #[error("orchestration error: {0}")]
    Orchestration(String),

    /// An MCP protocol error.
    #[error("MCP error: {0}")]
    Mcp(String),

    /// A catch-all for other errors.
    #[error("{0}")]
    Other(String),
}

/// A type alias for `Result<T, DaimonError>`.
pub type Result<T> = std::result::Result<T, DaimonError>;
