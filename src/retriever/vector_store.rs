//! Trait-based plugin system for vector stores.
//!
//! Implement [`VectorStore`] for your storage backend (Qdrant, Pinecone, Chroma,
//! Weaviate, Milvus, etc.), then compose with [`SimpleKnowledgeBase`] for a
//! complete embedding-backed retrieval pipeline.
//!
//! ```ignore
//! use daimon::retriever::{VectorStore, ScoredDocument, Document};
//!
//! struct MyVectorDb { /* ... */ }
//!
//! impl VectorStore for MyVectorDb {
//!     async fn upsert(&self, id: &str, embedding: Vec<f32>, doc: Document) -> daimon::Result<()> { /* ... */ Ok(()) }
//!     async fn query(&self, embedding: Vec<f32>, top_k: usize) -> daimon::Result<Vec<ScoredDocument>> { /* ... */ Ok(vec![]) }
//!     async fn delete(&self, id: &str) -> daimon::Result<bool> { /* ... */ Ok(true) }
//!     async fn count(&self) -> daimon::Result<usize> { /* ... */ Ok(0) }
//! }
//! ```

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::error::Result;
use crate::retriever::types::Document;

/// A document paired with a similarity score.
#[derive(Debug, Clone)]
pub struct ScoredDocument {
    pub document: Document,
    pub score: f64,
}

impl ScoredDocument {
    pub fn new(document: Document, score: f64) -> Self {
        Self { document, score }
    }
}

/// Low-level vector storage interface.
///
/// Implement this trait for your vector database. Operations deal in
/// pre-computed embeddings — use [`SimpleKnowledgeBase`](super::SimpleKnowledgeBase)
/// to combine a `VectorStore` with an [`EmbeddingModel`](crate::model::EmbeddingModel)
/// for automatic embedding computation.
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
