//! Locally-hosted model providers for the Daimon agent framework.
//!
//! Covers every runtime you can point at a machine under your own control:
//!
//! - [`ollama`] — [Ollama](https://ollama.com), via its native `/api/chat` API
//! - [`llamacpp`] — [llama.cpp](https://github.com/ggml-org/llama.cpp)'s
//!   `llama-server`, over its OpenAI-compatible API plus llama.cpp-native
//!   sampling extras
//! - [`llamars`] — [llama-rs](https://github.com/Lexmata/llama-rs)'s server,
//!   over its OpenAI-compatible API
//! - [`generic`] — any other OpenAI-compatible server (vLLM, LM Studio,
//!   llamafile, LocalAI, …)
//!
//! All four are HTTP clients; none embed inference in-process.

mod openai_compat;

pub mod llamacpp;
pub mod ollama;
pub mod ollama_embed;
// pub mod llamars;
// pub mod generic;

// pub use llamacpp::{LlamaCpp, LlamaCppEmbedding};
// pub use llamars::{LlamaRs, LlamaRsEmbedding};
// pub use generic::{OpenAiCompatible, OpenAiCompatibleEmbedding};
pub use ollama::Ollama;
pub use ollama_embed::OllamaEmbedding;
