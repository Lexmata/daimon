//! The [`Model`] trait — the plugin interface for LLM providers.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::error::Result;
use crate::stream::ResponseStream;
use crate::types::{ChatRequest, ChatResponse};

/// Trait for LLM providers. Supports both synchronous and streaming generation.
///
/// Implement this trait to add a new model provider to Daimon. The trait is
/// object-safe via [`ErasedModel`], so providers can be stored behind
/// `Arc<dyn ErasedModel>`.
pub trait Model: Send + Sync {
    /// Generates a complete response for the given request.
    fn generate(&self, request: &ChatRequest) -> impl Future<Output = Result<ChatResponse>> + Send;

    /// Returns a stream of response chunks for token-by-token output.
    fn generate_stream(
        &self,
        request: &ChatRequest,
    ) -> impl Future<Output = Result<ResponseStream>> + Send;

    /// The provider-side model identifier (e.g. `"claude-sonnet-5"`), used
    /// for cost attribution. Defaults to `"default"` so existing
    /// implementations keep compiling; providers should override it with the
    /// configured model name.
    fn model_id(&self) -> &str {
        "default"
    }
}

/// Shared ownership of a model via `Arc<dyn ErasedModel>`.
pub type SharedModel = Arc<dyn ErasedModel>;

/// Object-safe wrapper for [`Model`], enabling dynamic dispatch via `Arc<dyn ErasedModel>`.
pub trait ErasedModel: Send + Sync {
    /// Object-safe version of [`Model::generate`].
    fn generate_erased<'a>(
        &'a self,
        request: &'a ChatRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse>> + Send + 'a>>;

    /// Object-safe version of [`Model::generate_stream`].
    fn generate_stream_erased<'a>(
        &'a self,
        request: &'a ChatRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ResponseStream>> + Send + 'a>>;

    /// Object-safe version of [`Model::model_id`].
    fn model_id_erased(&self) -> &str;
}

impl<T: Model> ErasedModel for T {
    fn generate_erased<'a>(
        &'a self,
        request: &'a ChatRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse>> + Send + 'a>> {
        Box::pin(self.generate(request))
    }

    fn generate_stream_erased<'a>(
        &'a self,
        request: &'a ChatRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ResponseStream>> + Send + 'a>> {
        Box::pin(self.generate_stream(request))
    }

    fn model_id_erased(&self) -> &str {
        self.model_id()
    }
}
