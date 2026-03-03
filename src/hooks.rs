//! Lifecycle hooks for observing and controlling agent execution.
//!
//! Implement [`AgentHook`] to receive callbacks at key points in the ReAct loop:
//! iteration start/end, model responses, tool calls, and errors.

use std::future::Future;
use std::pin::Pin;

use crate::error::{DaimonError, Result};
use crate::model::types::ChatResponse;
use crate::tool::{ToolCall, ToolOutput};

/// Snapshot of the agent's state at a given point in the ReAct loop.
pub struct AgentState {
    /// Current iteration index (1-based). Increments each time the model is invoked.
    pub iteration: usize,
    /// Maximum iterations allowed before the loop aborts.
    pub max_iterations: usize,
}

/// Trait for receiving callbacks during agent execution.
///
/// All methods have default no-op implementations; override only what you need.
pub trait AgentHook: Send + Sync {
    /// Called at the start of each ReAct iteration, before the model is invoked.
    fn on_iteration_start(&self, _state: &AgentState) -> impl Future<Output = Result<()>> + Send {
        async { Ok(()) }
    }

    /// Called after the model returns a response, before tool execution or final output.
    fn on_model_response(
        &self,
        _response: &ChatResponse,
    ) -> impl Future<Output = Result<()>> + Send {
        async { Ok(()) }
    }

    /// Called when the model requests a tool call, before execution.
    fn on_tool_call(&self, _call: &ToolCall) -> impl Future<Output = Result<()>> + Send {
        async { Ok(()) }
    }

    /// Called after a tool completes, with the tool's output.
    fn on_tool_result(
        &self,
        _call: &ToolCall,
        _result: &ToolOutput,
    ) -> impl Future<Output = Result<()>> + Send {
        async { Ok(()) }
    }

    /// Called at the end of each iteration, after tools run (if any) or final response is produced.
    fn on_iteration_end(&self, _state: &AgentState) -> impl Future<Output = Result<()>> + Send {
        async { Ok(()) }
    }

    /// Called when a tool execution fails. The error is still propagated to the model.
    fn on_error(&self, _error: &DaimonError) -> impl Future<Output = Result<()>> + Send {
        async { Ok(()) }
    }
}

/// Object-safe wrapper for `AgentHook`.
pub trait ErasedAgentHook: Send + Sync {
    fn on_iteration_start_erased<'a>(
        &'a self,
        state: &'a AgentState,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    fn on_model_response_erased<'a>(
        &'a self,
        response: &'a ChatResponse,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    fn on_tool_call_erased<'a>(
        &'a self,
        call: &'a ToolCall,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    fn on_tool_result_erased<'a>(
        &'a self,
        call: &'a ToolCall,
        result: &'a ToolOutput,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    fn on_iteration_end_erased<'a>(
        &'a self,
        state: &'a AgentState,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    fn on_error_erased<'a>(
        &'a self,
        error: &'a DaimonError,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
}

impl<T: AgentHook> ErasedAgentHook for T {
    fn on_iteration_start_erased<'a>(
        &'a self,
        state: &'a AgentState,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(self.on_iteration_start(state))
    }

    fn on_model_response_erased<'a>(
        &'a self,
        response: &'a ChatResponse,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(self.on_model_response(response))
    }

    fn on_tool_call_erased<'a>(
        &'a self,
        call: &'a ToolCall,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(self.on_tool_call(call))
    }

    fn on_tool_result_erased<'a>(
        &'a self,
        call: &'a ToolCall,
        result: &'a ToolOutput,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(self.on_tool_result(call, result))
    }

    fn on_iteration_end_erased<'a>(
        &'a self,
        state: &'a AgentState,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(self.on_iteration_end(state))
    }

    fn on_error_erased<'a>(
        &'a self,
        error: &'a DaimonError,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(self.on_error(error))
    }
}

/// No-op hook implementation used when no custom hooks are configured.
/// All callbacks are no-ops; the agent uses this by default.
pub struct NoOpHook;

impl AgentHook for NoOpHook {}
