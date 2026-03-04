//! Retrieval-Augmented Generation (RAG) abstractions.
//!
//! The [`Retriever`] trait models any system that fetches relevant documents
//! for a query (vector stores, full-text search, hybrid). [`RetrieverTool`]
//! wraps a retriever as a [`Tool`](crate::tool::Tool) so agents can search
//! on demand.
//!
//! ## Plugin System
//!
//! For a pluggable storage backend, implement [`VectorStore`] for your
//! vector database, then compose with [`SimpleKnowledgeBase`] for a
//! complete embedding-backed retrieval pipeline.
//!
//! ```ignore
//! use daimon::retriever::{VectorStore, SimpleKnowledgeBase, InMemoryVectorStoreBackend, Document};
//!
//! let store = InMemoryVectorStoreBackend::new();
//! let kb = SimpleKnowledgeBase::new(embedding_model, store);
//!
//! kb.ingest(vec![Document::new("relevant text")]).await?;
//! let results = kb.search("query", 5).await?;
//! ```

pub mod in_memory;
pub mod in_memory_store;
pub mod knowledge_base;
mod traits;
mod types;
mod tool;
pub mod vector_store;

#[cfg(feature = "qdrant")]
pub mod qdrant;

#[cfg(feature = "pgvector")]
pub mod pgvector {
    //! pgvector-backed vector store — re-exported from `daimon-plugin-pgvector`.
    pub use daimon_plugin_pgvector::*;
}

pub use in_memory::InMemoryVectorStore;
pub use in_memory_store::InMemoryVectorStoreBackend;
pub use knowledge_base::{
    ErasedKnowledgeBase, KnowledgeBase, SharedKnowledgeBase, SimpleKnowledgeBase,
};
pub use traits::{ErasedRetriever, Retriever, SharedRetriever};
pub use tool::RetrieverTool;
pub use types::Document;
pub use vector_store::{ErasedVectorStore, ScoredDocument, SharedVectorStore, VectorStore};
