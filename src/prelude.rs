//! Convenience re-exports for common Daimon types.
//!
//! Use `use daimon::prelude::*` to bring in [`Agent`], [`AgentResponse`], [`Model`],
//! [`Tool`], [`Memory`], [`StreamEvent`], and related types without qualifying paths.

pub use crate::agent::{Agent, AgentResponse};
pub use crate::error::{DaimonError, Result};
pub use crate::hooks::AgentHook;
pub use crate::memory::{Memory, SlidingWindowMemory, SummaryMemory, TokenWindowMemory};
pub use crate::model::Model;
pub use crate::model::types::{ChatRequest, ChatResponse, Message, Role, Usage};
pub use crate::orchestration::{Chain, ChainContext, ChainStep, Graph, GraphContext, GraphNode, NodeOutcome};
pub use crate::stream::{ResponseStream, StreamEvent};
pub use crate::tool::{Tool, ToolOutput, ToolRegistry};

pub use futures::StreamExt;
pub use serde_json::Value;
pub use serde_json::json;
pub use tokio_util::sync::CancellationToken;

#[cfg(feature = "macros")]
pub use crate::tool_fn;

#[cfg(feature = "sqlite")]
pub use crate::memory::SqliteMemory;
