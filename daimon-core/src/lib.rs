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

mod embedding;
mod error;
mod model;
mod stream;
mod tool_types;
mod types;

pub use embedding::{EmbeddingModel, ErasedEmbeddingModel, SharedEmbeddingModel};
pub use error::{DaimonError, Result};
pub use model::{ErasedModel, Model, SharedModel};
pub use stream::{ResponseStream, StreamEvent};
pub use tool_types::ToolCall;
pub use types::{ChatRequest, ChatResponse, Message, Role, StopReason, ToolSpec, Usage};
