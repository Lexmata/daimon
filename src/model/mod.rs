//! LLM provider abstraction and implementations.
//!
//! Implement [`Model`] (from [`daimon_core`]) to add new providers. Built-in providers
//! ship as separate crates (each behind a feature flag): `openai`, `anthropic`,
//! `gemini`, `azure`, `bedrock`, `openrouter`. `ollama`, `llamacpp`, `llamars`, and
//! `local` are all aliases into one shared `daimon-provider-local` crate.

mod traits;
pub mod types;

pub use traits::{
    EmbeddingModel, ErasedEmbeddingModel, ErasedModel, Model, SharedEmbeddingModel, SharedModel,
};

#[cfg(feature = "openai")]
pub mod openai {
    //! OpenAI model provider (via [`daimon_provider_openai`]).
    pub use daimon_provider_openai::*;
}

#[cfg(feature = "anthropic")]
pub mod anthropic {
    //! Anthropic model provider (via [`daimon_provider_anthropic`]).
    pub use daimon_provider_anthropic::*;
}

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

#[cfg(feature = "openrouter")]
pub mod openrouter {
    //! OpenRouter model provider (via [`daimon_provider_openrouter`]).
    pub use daimon_provider_openrouter::*;
}

#[cfg(any(
    feature = "ollama",
    feature = "llamacpp",
    feature = "llamars",
    feature = "local"
))]
pub mod local {
    //! Locally-hosted model providers (via [`daimon_provider_local`]).
    pub use daimon_provider_local::*;
}

#[cfg(feature = "ollama")]
pub mod ollama {
    //! Ollama local model provider (via [`daimon_provider_local`]).
    pub use daimon_provider_local::ollama::*;
}

#[cfg(feature = "llamacpp")]
pub mod llamacpp {
    //! llama.cpp model provider (via [`daimon_provider_local`]).
    pub use daimon_provider_local::llamacpp::*;
}

#[cfg(feature = "llamars")]
pub mod llamars {
    //! llama-rs model provider (via [`daimon_provider_local`]).
    pub use daimon_provider_local::llamars::*;
}

#[cfg(feature = "openai")]
pub mod openai_embed {
    //! OpenAI embeddings provider (via [`daimon_provider_openai`]).
    pub use daimon_provider_openai::embedding::*;
}

#[cfg(feature = "ollama")]
pub mod ollama_embed {
    //! Ollama embeddings provider (via [`daimon_provider_local`]).
    pub use daimon_provider_local::ollama_embed::*;
}
