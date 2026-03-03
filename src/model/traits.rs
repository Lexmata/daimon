use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::error::Result;
use crate::model::types::{ChatRequest, ChatResponse};
use crate::stream::ResponseStream;

/// Trait for LLM providers. Supports both synchronous and streaming generation.
pub trait Model: Send + Sync {
    /// Generates a complete response for the given request. Blocks until the model finishes.
    fn generate(&self, request: &ChatRequest) -> impl Future<Output = Result<ChatResponse>> + Send;

    /// Returns a stream of response chunks. Use for token-by-token or event-by-event streaming.
    fn generate_stream(
        &self,
        request: &ChatRequest,
    ) -> impl Future<Output = Result<ResponseStream>> + Send;
}

/// Shared ownership of a model via `Arc<dyn ErasedModel>`. Use for storing models in agents or sharing across threads.
pub type SharedModel = Arc<dyn ErasedModel>;

/// Object-safe wrapper for the `Model` trait, enabling dynamic dispatch via `Arc<dyn ErasedModel>`.
pub trait ErasedModel: Send + Sync {
    fn generate_erased<'a>(
        &'a self,
        request: &'a ChatRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse>> + Send + 'a>>;

    fn generate_stream_erased<'a>(
        &'a self,
        request: &'a ChatRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ResponseStream>> + Send + 'a>>;
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
}
