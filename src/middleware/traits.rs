use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::error::Result;
use crate::model::types::{ChatRequest, ChatResponse};
use crate::tool::ToolCall;

/// Outcome of a middleware hook. `Continue` proceeds to the next layer;
/// `ShortCircuit` stops the pipeline and returns the given response.
#[derive(Debug)]
pub enum MiddlewareAction {
    /// Proceed to the next middleware / model call.
    Continue,
    /// Stop processing and return this response directly.
    ShortCircuit(ChatResponse),
}

/// Composable layer that can inspect and mutate requests, responses, and tool
/// calls flowing through the agent's ReAct loop.
///
/// All methods default to no-op `Continue`. Override only the hooks you need.
pub trait Middleware: Send + Sync {
    /// Called before each model invocation. May mutate the request or short-circuit.
    fn on_request(
        &self,
        _request: &mut ChatRequest,
    ) -> impl Future<Output = Result<MiddlewareAction>> + Send {
        async { Ok(MiddlewareAction::Continue) }
    }

    /// Called after the model responds, before the agent processes tool calls or
    /// returns the final text. May mutate the response or short-circuit.
    fn on_response(
        &self,
        _response: &mut ChatResponse,
    ) -> impl Future<Output = Result<MiddlewareAction>> + Send {
        async { Ok(MiddlewareAction::Continue) }
    }

    /// Called before each tool is executed. May mutate the tool call arguments
    /// or short-circuit to skip execution.
    fn on_tool_call(
        &self,
        _call: &mut ToolCall,
    ) -> impl Future<Output = Result<MiddlewareAction>> + Send {
        async { Ok(MiddlewareAction::Continue) }
    }
}

/// Object-safe wrapper for [`Middleware`].
pub trait ErasedMiddleware: Send + Sync {
    fn on_request_erased<'a>(
        &'a self,
        request: &'a mut ChatRequest,
    ) -> Pin<Box<dyn Future<Output = Result<MiddlewareAction>> + Send + 'a>>;

    fn on_response_erased<'a>(
        &'a self,
        response: &'a mut ChatResponse,
    ) -> Pin<Box<dyn Future<Output = Result<MiddlewareAction>> + Send + 'a>>;

    fn on_tool_call_erased<'a>(
        &'a self,
        call: &'a mut ToolCall,
    ) -> Pin<Box<dyn Future<Output = Result<MiddlewareAction>> + Send + 'a>>;
}

impl<T: Middleware> ErasedMiddleware for T {
    fn on_request_erased<'a>(
        &'a self,
        request: &'a mut ChatRequest,
    ) -> Pin<Box<dyn Future<Output = Result<MiddlewareAction>> + Send + 'a>> {
        Box::pin(self.on_request(request))
    }

    fn on_response_erased<'a>(
        &'a self,
        response: &'a mut ChatResponse,
    ) -> Pin<Box<dyn Future<Output = Result<MiddlewareAction>> + Send + 'a>> {
        Box::pin(self.on_response(response))
    }

    fn on_tool_call_erased<'a>(
        &'a self,
        call: &'a mut ToolCall,
    ) -> Pin<Box<dyn Future<Output = Result<MiddlewareAction>> + Send + 'a>> {
        Box::pin(self.on_tool_call(call))
    }
}

/// Shared ownership of middleware.
pub type SharedMiddleware = Arc<dyn ErasedMiddleware>;
