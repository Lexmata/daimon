use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::error::Result;
use crate::tool::types::ToolOutput;

/// Trait for tools the agent can invoke. Tools must have unique names and declare a JSON Schema for parameters.
pub trait Tool: Send + Sync {
    /// Unique identifier for the tool. Used by the model when requesting a call.
    fn name(&self) -> &str;
    /// Human-readable description. The model uses this to decide when to call the tool.
    fn description(&self) -> &str;
    /// JSON Schema for the tool's parameters. Validates and guides the model's argument generation.
    fn parameters_schema(&self) -> serde_json::Value;

    /// Executes the tool with the given arguments. Arguments are validated by the model; implementors may still validate.
    fn execute(&self, input: &serde_json::Value)
    -> impl Future<Output = Result<ToolOutput>> + Send;
}

/// Object-safe wrapper for the `Tool` trait, enabling dynamic dispatch via `Arc<dyn ErasedTool>`.
pub trait ErasedTool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;

    fn execute_erased<'a>(
        &'a self,
        input: &'a serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput>> + Send + 'a>>;
}

impl<T: Tool> ErasedTool for T {
    fn name(&self) -> &str {
        Tool::name(self)
    }

    fn description(&self) -> &str {
        Tool::description(self)
    }

    fn parameters_schema(&self) -> serde_json::Value {
        Tool::parameters_schema(self)
    }

    fn execute_erased<'a>(
        &'a self,
        input: &'a serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput>> + Send + 'a>> {
        Box::pin(Tool::execute(self, input))
    }
}

/// Shared ownership of a tool via `Arc<dyn ErasedTool>`. Used by registry and agent.
pub type SharedTool = Arc<dyn ErasedTool>;
