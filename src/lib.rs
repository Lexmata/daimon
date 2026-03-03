//! # Daimon
//!
//! A Rust-native AI agent framework for building LLM-powered agents with tool use,
//! memory, and streaming. Daimon implements the ReAct (Reason-Act-Observe) pattern:
//! the agent calls a model, optionally invokes tools, observes results, and repeats
//! until it produces a final response.
//!
//! ## Quick Start
//!
//! ```ignore
//! use daimon::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> daimon::Result<()> {
//!     let agent = Agent::builder()
//!         .model(daimon::model::openai::OpenAi::new("gpt-4o"))
//!         .system_prompt("You are a helpful assistant.")
//!         .build()?;
//!
//!     let response = agent.prompt("What is Rust?").await?;
//!     println!("{}", response.text());
//!     Ok(())
//! }
//! ```
//!
//! ## Feature Flags
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `openai` | OpenAI API provider (default) |
//! | `anthropic` | Anthropic Claude API provider (default) |
//! | `gemini` | Google Gemini / Vertex AI provider |
//! | `azure` | Azure OpenAI Service provider |
//! | `bedrock` | AWS Bedrock provider |
//! | `full` | All model providers |
//!
//! The core framework compiles with no features; enable providers as needed.
//!
//! ## Module Overview
//!
//! - [`agent`] - Agent builder and ReAct loop execution
//! - [`model`] - LLM provider trait and implementations
//! - [`tool`] - Tool trait, registry, and execution
//! - [`memory`] - Conversation memory implementations
//! - [`stream`] - Streaming response types
//! - [`hooks`] - Lifecycle hooks for observability and control

pub mod agent;
pub mod error;
pub mod hooks;
pub mod memory;
pub mod model;
pub mod prelude;
pub mod stream;
pub mod tool;

pub use error::{DaimonError, Result};
