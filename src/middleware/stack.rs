use std::sync::Arc;

use crate::error::Result;
use crate::middleware::traits::{ErasedMiddleware, Middleware, MiddlewareAction};
use crate::model::types::{ChatRequest, ChatResponse};
use crate::tool::ToolCall;

/// An ordered chain of middleware layers. Each layer sees the request/response
/// in registration order; the first non-`Continue` action short-circuits.
#[derive(Default, Clone)]
pub struct MiddlewareStack {
    layers: Vec<Arc<dyn ErasedMiddleware>>,
}

impl MiddlewareStack {
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a middleware to the stack.
    pub fn push<M: Middleware + 'static>(&mut self, mw: M) {
        self.layers.push(Arc::new(mw));
    }

    /// Appends a pre-boxed middleware to the stack.
    pub fn push_shared(&mut self, mw: Arc<dyn ErasedMiddleware>) {
        self.layers.push(mw);
    }

    /// Returns `true` if no middleware has been registered.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Runs `on_request` through every layer in order. Returns the first
    /// non-`Continue` action, or `Continue` if all layers pass.
    #[inline]
    pub async fn run_on_request(
        &self,
        request: &mut ChatRequest,
    ) -> Result<MiddlewareAction> {
        if self.layers.is_empty() {
            return Ok(MiddlewareAction::Continue);
        }
        for layer in &self.layers {
            match layer.on_request_erased(request).await? {
                MiddlewareAction::Continue => {}
                action => return Ok(action),
            }
        }
        Ok(MiddlewareAction::Continue)
    }

    /// Runs `on_response` through every layer in order.
    #[inline]
    pub async fn run_on_response(
        &self,
        response: &mut ChatResponse,
    ) -> Result<MiddlewareAction> {
        if self.layers.is_empty() {
            return Ok(MiddlewareAction::Continue);
        }
        for layer in &self.layers {
            match layer.on_response_erased(response).await? {
                MiddlewareAction::Continue => {}
                action => return Ok(action),
            }
        }
        Ok(MiddlewareAction::Continue)
    }

    /// Runs `on_tool_call` through every layer in order.
    #[inline]
    pub async fn run_on_tool_call(
        &self,
        call: &mut ToolCall,
    ) -> Result<MiddlewareAction> {
        if self.layers.is_empty() {
            return Ok(MiddlewareAction::Continue);
        }
        for layer in &self.layers {
            match layer.on_tool_call_erased(call).await? {
                MiddlewareAction::Continue => {}
                action => return Ok(action),
            }
        }
        Ok(MiddlewareAction::Continue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::types::{ChatRequest, ChatResponse, Message, StopReason, Usage};

    struct AppendSystemMiddleware;

    impl Middleware for AppendSystemMiddleware {
        async fn on_request(
            &self,
            request: &mut ChatRequest,
        ) -> Result<MiddlewareAction> {
            request.messages.push(Message::system("injected by middleware"));
            Ok(MiddlewareAction::Continue)
        }
    }

    struct ShortCircuitMiddleware;

    impl Middleware for ShortCircuitMiddleware {
        async fn on_response(
            &self,
            _response: &mut ChatResponse,
        ) -> Result<MiddlewareAction> {
            Ok(MiddlewareAction::ShortCircuit(ChatResponse {
                message: Message::assistant("short-circuited"),
                stop_reason: StopReason::EndTurn,
                usage: Some(Usage::default()),
            }))
        }
    }

    #[tokio::test]
    async fn test_empty_stack_continues() {
        let stack = MiddlewareStack::new();
        let mut req = ChatRequest::new(vec![Message::user("hi")]);
        let action = stack.run_on_request(&mut req).await.unwrap();
        assert!(matches!(action, MiddlewareAction::Continue));
    }

    #[tokio::test]
    async fn test_middleware_mutates_request() {
        let mut stack = MiddlewareStack::new();
        stack.push(AppendSystemMiddleware);

        let mut req = ChatRequest::new(vec![Message::user("hi")]);
        assert_eq!(req.messages.len(), 1);

        stack.run_on_request(&mut req).await.unwrap();
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[1].content.as_deref(), Some("injected by middleware"));
    }

    #[tokio::test]
    async fn test_short_circuit_stops_pipeline() {
        let mut stack = MiddlewareStack::new();
        stack.push(ShortCircuitMiddleware);

        let mut resp = ChatResponse {
            message: Message::assistant("original"),
            stop_reason: StopReason::EndTurn,
            usage: None,
        };

        let action = stack.run_on_response(&mut resp).await.unwrap();
        match action {
            MiddlewareAction::ShortCircuit(r) => {
                assert_eq!(r.message.content.as_deref(), Some("short-circuited"));
            }
            _ => panic!("expected short-circuit"),
        }
    }

    #[tokio::test]
    async fn test_multiple_middleware_ordered() {
        struct First;
        struct Second;

        impl Middleware for First {
            async fn on_request(
                &self,
                request: &mut ChatRequest,
            ) -> Result<MiddlewareAction> {
                request.messages.push(Message::system("first"));
                Ok(MiddlewareAction::Continue)
            }
        }

        impl Middleware for Second {
            async fn on_request(
                &self,
                request: &mut ChatRequest,
            ) -> Result<MiddlewareAction> {
                request.messages.push(Message::system("second"));
                Ok(MiddlewareAction::Continue)
            }
        }

        let mut stack = MiddlewareStack::new();
        stack.push(First);
        stack.push(Second);

        let mut req = ChatRequest::new(vec![]);
        stack.run_on_request(&mut req).await.unwrap();
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[0].content.as_deref(), Some("first"));
        assert_eq!(req.messages[1].content.as_deref(), Some("second"));
    }
}
