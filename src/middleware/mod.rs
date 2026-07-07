//! Composable middleware pipeline for mutating requests, responses, and tool calls.
//!
//! Unlike [`AgentHook`](crate::hooks::AgentHook) which is observer-only, middleware
//! can **mutate** the `ChatRequest` before it reaches the model, the `ChatResponse`
//! before the agent processes it, and individual `ToolCall`s before execution. A
//! middleware may also **short-circuit** the loop by returning a synthetic response.
//!
//! ```ignore
//! use daimon::middleware::{Middleware, MiddlewareAction};
//!
//! struct LoggingMiddleware;
//!
//! impl Middleware for LoggingMiddleware {
//!     async fn on_request(
//!         &self,
//!         request: &mut daimon::model::types::ChatRequest,
//!     ) -> daimon::Result<MiddlewareAction> {
//!         tracing::info!(messages = request.messages.len(), "model request");
//!         Ok(MiddlewareAction::Continue)
//!     }
//! }
//! ```

mod stack;
mod traits;

pub use stack::MiddlewareStack;
pub use traits::{ErasedMiddleware, Middleware, MiddlewareAction, SharedMiddleware};
