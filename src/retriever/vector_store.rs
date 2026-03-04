//! Trait-based plugin system for vector stores — re-exported from `daimon-core`.
//!
//! Implement [`VectorStore`] for your storage backend (pgvector, Qdrant,
//! Pinecone, Chroma, Weaviate, Milvus, etc.), then compose with
//! [`SimpleKnowledgeBase`] for a complete embedding-backed retrieval pipeline.
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

pub use daimon_core::vector_store::{ErasedVectorStore, SharedVectorStore, VectorStore};
pub use daimon_core::{Document, ScoredDocument};
