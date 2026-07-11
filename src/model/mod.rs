//! LLM provider abstraction and implementations.
//!
//! Implement [`Model`] (from [`daimon_core`]) to add new providers. Built-in providers
//! ship as separate crates (each behind a feature flag): `openai`, `anthropic`,
//! `gemini`, `azure`, `bedrock`, `ollama`, `llamacpp`.

mod traits;
pub mod types;

#[cfg(any(feature = "openai", feature = "anthropic"))]
pub(crate) mod retry;

pub use traits::{
    EmbeddingModel, ErasedEmbeddingModel, ErasedModel, Model, SharedEmbeddingModel, SharedModel,
};

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
#[allow(unused_imports)]
pub mod llamacpp {
    //! llama.cpp model provider (via [`daimon_provider_local`]).
    pub use daimon_provider_local::llamacpp::*;
}

#[cfg(feature = "openai")]
pub mod openai_embed;

#[cfg(feature = "ollama")]
pub mod ollama_embed {
    //! Ollama embeddings provider (via [`daimon_provider_local`]).
    pub use daimon_provider_local::ollama_embed::*;
}
