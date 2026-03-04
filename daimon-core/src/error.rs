//! Error types for the Daimon agent framework.

use thiserror::Error;

/// The central error type for all Daimon operations.
///
/// Provider crates should map their transport-specific errors
/// (HTTP, gRPC, SDK) to [`DaimonError::Model`].
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

    /// The agent exceeded the configured spending budget.
    #[error("budget exceeded: ${spent:.6} spent, limit was ${limit:.6}")]
    BudgetExceeded {
        /// How much has been spent so far (USD).
        spent: f64,
        /// The configured limit (USD).
        limit: f64,
    },

    /// An input or output guardrail blocked the request.
    #[error("guardrail blocked: {0}")]
    GuardrailBlocked(String),

    /// A catch-all for other errors.
    #[error("{0}")]
    Other(String),
}

/// A type alias for `Result<T, DaimonError>`.
pub type Result<T> = std::result::Result<T, DaimonError>;
