//! Embedding model trait for computing vector representations of text.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::error::Result;

/// Trait for models that produce vector embeddings from text.
///
/// Implement this for your embedding provider, then use it with vector stores
/// and RAG pipelines.
pub trait EmbeddingModel: Send + Sync {
    /// Computes embeddings for one or more texts. Returns one vector per input.
    fn embed(&self, texts: &[&str]) -> impl Future<Output = Result<Vec<Vec<f32>>>> + Send;

    /// Returns the dimensionality of the embedding vectors this model produces.
    fn dimensions(&self) -> usize;
}

/// Object-safe wrapper for [`EmbeddingModel`].
pub trait ErasedEmbeddingModel: Send + Sync {
    fn embed_erased<'a>(
        &'a self,
        texts: &'a [&'a str],
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Vec<f32>>>> + Send + 'a>>;

    fn dimensions(&self) -> usize;
}

impl<T: EmbeddingModel> ErasedEmbeddingModel for T {
    fn embed_erased<'a>(
        &'a self,
        texts: &'a [&'a str],
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Vec<f32>>>> + Send + 'a>> {
        Box::pin(self.embed(texts))
    }

    fn dimensions(&self) -> usize {
        EmbeddingModel::dimensions(self)
    }
}

/// Shared ownership of an embedding model.
pub type SharedEmbeddingModel = Arc<dyn ErasedEmbeddingModel>;
