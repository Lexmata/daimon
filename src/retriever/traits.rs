//! The Retriever trait and object-safe wrapper.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::error::Result;
use crate::retriever::types::Document;

/// Trait for document retrieval backends (vector stores, search engines, etc.).
///
/// Implement this for your storage backend, then wrap with
/// [`RetrieverTool`](super::RetrieverTool) to expose it to agents.
pub trait Retriever: Send + Sync {
    /// Retrieves up to `top_k` documents relevant to `query`.
    fn retrieve(
        &self,
        query: &str,
        top_k: usize,
    ) -> impl Future<Output = Result<Vec<Document>>> + Send;
}

/// Object-safe wrapper for [`Retriever`].
pub trait ErasedRetriever: Send + Sync {
    fn retrieve_erased<'a>(
        &'a self,
        query: &'a str,
        top_k: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Document>>> + Send + 'a>>;
}

impl<T: Retriever> ErasedRetriever for T {
    fn retrieve_erased<'a>(
        &'a self,
        query: &'a str,
        top_k: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Document>>> + Send + 'a>> {
        Box::pin(self.retrieve(query, top_k))
    }
}

/// Shared ownership of a retriever.
pub type SharedRetriever = Arc<dyn ErasedRetriever>;
