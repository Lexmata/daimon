//! LLM provider abstraction and implementations.
//!
//! Implement [`Model`] (from [`daimon_core`]) to add new providers. Built-in providers
//! ship as separate crates (each behind a feature flag): `openai`, `anthropic`,
//! `gemini`, `azure`, `bedrock`, `ollama`.

mod traits;
pub mod types;

pub use traits::{EmbeddingModel, ErasedEmbeddingModel, ErasedModel, Model, SharedEmbeddingModel, SharedModel};

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "gemini")]
pub mod gemini {
    //! Google Gemini model provider (via [`daimon_provider_gemini`]).
    pub use daimon_provider_gemini::*;
}

#[cfg(feature = "azure")]
pub mod azure {
    //! Azure OpenAI model provider (via [`daimon_provider_azure`]).
    pub use daimon_provider_azure::*;
}

#[cfg(feature = "bedrock")]
pub mod bedrock {
    //! Amazon Bedrock model provider (via [`daimon_provider_bedrock`]).
    pub use daimon_provider_bedrock::*;
}

#[cfg(feature = "ollama")]
pub mod ollama;

#[cfg(feature = "openai")]
pub mod openai_embed;

#[cfg(feature = "ollama")]
pub mod ollama_embed;
