//! LLM provider abstraction and implementations.
//!
//! Implement [`Model`] to add new providers. Built-in providers (each behind a feature flag):
//! `openai`, `anthropic`, `gemini`, `azure`, `bedrock`.

mod traits;
pub mod types;

pub use traits::{ErasedModel, Model, SharedModel};

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "gemini")]
pub mod gemini;

#[cfg(feature = "azure")]
pub mod azure;

#[cfg(feature = "bedrock")]
pub mod bedrock;
