//! High-level knowledge base abstraction.
//!
//! [`KnowledgeBase`] combines document ingestion with semantic search.
//! [`SimpleKnowledgeBase`] provides a concrete implementation that pairs
//! any [`VectorStore`] with an
//! [`EmbeddingModel`](crate::model::EmbeddingModel).
//!
//! ```ignore
//! use daimon::retriever::{SimpleKnowledgeBase, InMemoryVectorStoreBackend, Document};
//! use daimon::model::openai_embed::OpenAiEmbedding;
//!
//! let embed = Arc::new(OpenAiEmbedding::new("text-embedding-3-small"));
//! let store = InMemoryVectorStoreBackend::new();
//! let kb = SimpleKnowledgeBase::new(embed, store);
//!
//! kb.ingest(vec![Document::new("Rust is a systems language")]).await?;
//! let results = kb.search("what is Rust?", 3).await?;
//! ```

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::error::Result;
use crate::model::ErasedEmbeddingModel;
use crate::retriever::traits::Retriever;
use crate::retriever::types::Document;
use crate::retriever::vector_store::VectorStore;

/// High-level knowledge base interface.
///
/// Implement this trait for custom knowledge bases that manage document
/// ingestion and semantic search. Use [`SimpleKnowledgeBase`] for a
/// ready-made implementation.
pub trait KnowledgeBase: Send + Sync {
    /// Ingests documents, computing embeddings and storing them.
    /// Returns the IDs assigned to each document.
    fn ingest(&self, documents: Vec<Document>) -> impl Future<Output = Result<Vec<String>>> + Send;

    /// Searches for the `top_k` most relevant documents to the query.
    fn search(
        &self,
        query: &str,
        top_k: usize,
    ) -> impl Future<Output = Result<Vec<Document>>> + Send;

    /// Removes a document by its ID.
    fn remove(&self, id: &str) -> impl Future<Output = Result<bool>> + Send;

    /// Returns the number of documents in the knowledge base.
    fn count(&self) -> impl Future<Output = Result<usize>> + Send;
}

/// Object-safe wrapper for [`KnowledgeBase`].
pub trait ErasedKnowledgeBase: Send + Sync {
    fn ingest_erased<'a>(
        &'a self,
        documents: Vec<Document>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<String>>> + Send + 'a>>;

    fn search_erased<'a>(
        &'a self,
        query: &'a str,
        top_k: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Document>>> + Send + 'a>>;

    fn remove_erased<'a>(
        &'a self,
        id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<bool>> + Send + 'a>>;

    fn count_erased(&self) -> Pin<Box<dyn Future<Output = Result<usize>> + Send + '_>>;
}

impl<T: KnowledgeBase> ErasedKnowledgeBase for T {
    fn ingest_erased<'a>(
        &'a self,
        documents: Vec<Document>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<String>>> + Send + 'a>> {
        Box::pin(self.ingest(documents))
    }

    fn search_erased<'a>(
        &'a self,
        query: &'a str,
        top_k: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Document>>> + Send + 'a>> {
        Box::pin(self.search(query, top_k))
    }

    fn remove_erased<'a>(
        &'a self,
        id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<bool>> + Send + 'a>> {
        Box::pin(self.remove(id))
    }

    fn count_erased(&self) -> Pin<Box<dyn Future<Output = Result<usize>> + Send + '_>> {
        Box::pin(self.count())
    }
}

/// Shared ownership of a knowledge base.
pub type SharedKnowledgeBase = Arc<dyn ErasedKnowledgeBase>;

/// A knowledge base that combines an embedding model with a vector store.
///
/// Documents are embedded on ingestion and queries are embedded at search
/// time. IDs are auto-generated using a hash of the document content.
pub struct SimpleKnowledgeBase<V: VectorStore> {
    embedding_model: Arc<dyn ErasedEmbeddingModel>,
    store: V,
}

impl<V: VectorStore> SimpleKnowledgeBase<V> {
    /// Creates a new knowledge base from an embedding model and vector store.
    pub fn new(embedding_model: Arc<dyn ErasedEmbeddingModel>, store: V) -> Self {
        Self {
            embedding_model,
            store,
        }
    }

    /// Returns a reference to the underlying vector store.
    pub fn store(&self) -> &V {
        &self.store
    }
}

fn content_hash(content: &str) -> String {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    format!("doc_{:016x}", hasher.finish())
}

impl<V: VectorStore> KnowledgeBase for SimpleKnowledgeBase<V> {
    async fn ingest(&self, documents: Vec<Document>) -> Result<Vec<String>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let texts: Vec<&str> = documents.iter().map(|d| d.content.as_str()).collect();
        let embeddings = self.embedding_model.embed_erased(&texts).await?;
        let mut ids = Vec::with_capacity(documents.len());

        for (doc, embedding) in documents.into_iter().zip(embeddings) {
            let id = doc
                .metadata
                .get("id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| content_hash(&doc.content));
            self.store.upsert(&id, embedding, doc).await?;
            ids.push(id);
        }

        Ok(ids)
    }

    async fn search(&self, query: &str, top_k: usize) -> Result<Vec<Document>> {
        let texts = [query];
        let embeddings = self.embedding_model.embed_erased(&texts).await?;
        let query_vec = embeddings.into_iter().next().unwrap_or_default();

        let scored = self.store.query(query_vec, top_k).await?;
        Ok(scored
            .into_iter()
            .map(|sd| sd.document.with_score(sd.score))
            .collect())
    }

    async fn remove(&self, id: &str) -> Result<bool> {
        self.store.delete(id).await
    }

    async fn count(&self) -> Result<usize> {
        self.store.count().await
    }
}

impl<V: VectorStore> Retriever for SimpleKnowledgeBase<V> {
    async fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<Document>> {
        self.search(query, top_k).await
    }
}
