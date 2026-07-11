//! # daimon-core
//!
//! Core traits and types for the [Daimon](https://docs.rs/daimon) AI agent
//! framework. This crate is the **plugin interface** — implement [`Model`] to
//! add a new LLM provider.
//!
//! Provider crates depend on `daimon-core` for the shared types and trait.
//! The main `daimon` crate re-exports everything from here, so end users
//! typically never need to depend on `daimon-core` directly.
//!
//! ## Implementing a Provider
//!
//! ```ignore
//! use daimon_core::{Model, ChatRequest, ChatResponse, Result, ResponseStream};
//!
//! pub struct MyProvider { /* ... */ }
//!
//! impl Model for MyProvider {
//!     async fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
//!         // call your LLM API
//!         todo!()
//!     }
//!
//!     async fn generate_stream(&self, request: &ChatRequest) -> Result<ResponseStream> {
//!         todo!()
//!     }
//! }
//! ```

mod archival_memory;
mod core_memory;
pub mod distributed;
mod document;
mod embedding;
mod episodic_memory;
mod error;
mod memory;
mod model;
mod stream;
pub mod stream_util;
mod tool;
mod tool_types;
mod types;
pub mod vector_store;

pub use archival_memory::{
    ArchivalMemory, ArchivalRecord, ErasedArchivalMemory, SharedArchivalMemory,
};
pub use core_memory::{
    CoreMemory, CoreMemoryBlock, ErasedCoreMemory, SharedCoreMemory, render_blocks,
};
pub use distributed::{AgentTask, ErasedTaskBroker, TaskBroker, TaskResult, TaskStatus};
pub use document::{Document, ScoredDocument};
pub use embedding::{EmbeddingModel, ErasedEmbeddingModel, SharedEmbeddingModel};
pub use episodic_memory::{
    EpisodicEvent, EpisodicMemory, EpisodicQuery, ErasedEpisodicMemory, SharedEpisodicMemory,
};
pub use error::{DaimonError, Result};
pub use memory::{ErasedMemory, Memory, SharedMemory};
pub use model::{ErasedModel, Model, SharedModel};
pub use stream::{ResponseStream, StreamEvent};
pub use tool::{
    BackoffStrategy, ErasedTool, SharedTool, Tool, ToolChoice, ToolOutput, ToolRetryPolicy,
};
pub use tool_types::ToolCall;
pub use types::{ChatRequest, ChatResponse, Message, Role, StopReason, ToolSpec, Usage};
pub use vector_store::{ErasedVectorStore, SharedVectorStore, VectorStore};
