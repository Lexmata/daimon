//! Trait-based plugin system for vector stores.
//!
//! Implement [`VectorStore`] for your storage backend (pgvector, Qdrant,
//! Pinecone, Chroma, Weaviate, Milvus, etc.), then compose with a knowledge
//! base for a complete embedding-backed retrieval pipeline.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::document::{Document, ScoredDocument};
use crate::error::Result;

/// Low-level vector storage interface.
///
/// Implement this trait for your vector database. Operations deal in
/// pre-computed embeddings — combine with an `EmbeddingModel` and knowledge
/// base for automatic embedding computation.
pub trait VectorStore: Send + Sync {
    /// Inserts or updates a document with a pre-computed embedding vector.
    fn upsert(
        &self,
        id: &str,
        embedding: Vec<f32>,
        document: Document,
    ) -> impl Future<Output = Result<()>> + Send;

    /// Queries the store for the `top_k` most similar documents to the
    /// given embedding vector. Returns results sorted by descending similarity.
    fn query(
        &self,
        embedding: Vec<f32>,
        top_k: usize,
    ) -> impl Future<Output = Result<Vec<ScoredDocument>>> + Send;

    /// Deletes a document by ID. Returns `true` if the document was found
    /// and deleted.
    fn delete(&self, id: &str) -> impl Future<Output = Result<bool>> + Send;

    /// Returns the number of documents in the store.
    fn count(&self) -> impl Future<Output = Result<usize>> + Send;
}

/// Object-safe wrapper for [`VectorStore`].
pub trait ErasedVectorStore: Send + Sync {
    fn upsert_erased<'a>(
        &'a self,
        id: &'a str,
        embedding: Vec<f32>,
        document: Document,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    fn query_erased<'a>(
        &'a self,
        embedding: Vec<f32>,
        top_k: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<ScoredDocument>>> + Send + 'a>>;

    fn delete_erased<'a>(
        &'a self,
        id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<bool>> + Send + 'a>>;

    fn count_erased(&self) -> Pin<Box<dyn Future<Output = Result<usize>> + Send + '_>>;
}

impl<T: VectorStore> ErasedVectorStore for T {
    fn upsert_erased<'a>(
        &'a self,
        id: &'a str,
        embedding: Vec<f32>,
        document: Document,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(self.upsert(id, embedding, document))
    }

    fn query_erased<'a>(
        &'a self,
        embedding: Vec<f32>,
        top_k: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<ScoredDocument>>> + Send + 'a>> {
        Box::pin(self.query(embedding, top_k))
    }

    fn delete_erased<'a>(
        &'a self,
        id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<bool>> + Send + 'a>> {
        Box::pin(self.delete(id))
    }

    fn count_erased(&self) -> Pin<Box<dyn Future<Output = Result<usize>> + Send + '_>> {
        Box::pin(self.count())
    }
}

/// Shared ownership of a vector store.
pub type SharedVectorStore = Arc<dyn ErasedVectorStore>;
