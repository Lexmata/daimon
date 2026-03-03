//! Convenience re-exports for common Daimon types.
//!
//! Use `use daimon::prelude::*` to bring in [`Agent`], [`AgentResponse`], [`Model`],
//! [`Tool`], [`Memory`], [`StreamEvent`], and related types without qualifying paths.

pub use crate::agent::{Agent, AgentResponse};
pub use crate::error::{DaimonError, Result};
pub use crate::hooks::AgentHook;
pub use crate::memory::{Memory, SlidingWindowMemory};
pub use crate::model::Model;
pub use crate::model::types::{ChatRequest, ChatResponse, Message, Role, Usage};
pub use crate::stream::{ResponseStream, StreamEvent};
pub use crate::tool::{Tool, ToolOutput, ToolRegistry};

pub use futures::StreamExt;
pub use serde_json::Value;
pub use serde_json::json;
pub use tokio_util::sync::CancellationToken;
