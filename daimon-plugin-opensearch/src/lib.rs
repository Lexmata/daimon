//! # daimon-plugin-opensearch
//!
//! An OpenSearch k-NN backed [`VectorStore`](daimon_core::VectorStore) plugin
//! for the [Daimon](https://docs.rs/daimon) AI agent framework.
//!
//! This crate provides [`OpenSearchVectorStore`], which stores document
//! embeddings in [OpenSearch](https://opensearch.org/) using its native
//! [k-NN plugin](https://opensearch.org/docs/latest/search-plugins/knn/index/).
//! It supports cosine similarity, L2 (euclidean), and inner product distance
//! metrics with HNSW indexing via nmslib, faiss, or lucene engines.
//!
//! ## Quick Start
//!
//! ```ignore
//! use daimon_plugin_opensearch::{OpenSearchVectorStoreBuilder, SpaceType, Engine};
//! use daimon::retriever::SimpleKnowledgeBase;
//! use std::sync::Arc;
//!
//! let store = OpenSearchVectorStoreBuilder::new("http://localhost:9200", 1536)
//!     .index("my_docs")
//!     .space_type(SpaceType::CosineSimilarity)
//!     .engine(Engine::Lucene)
//!     .build()
//!     .await?;
//!
//! // Compose with an embedding model for a full RAG pipeline:
//! let kb = SimpleKnowledgeBase::new(embedding_model, store);
//! ```
//!
//! ## Manual Index Setup
//!
//! If you prefer to manage index creation yourself, disable auto-creation
//! and use the JSON from [`index_settings`]:
//!
//! ```ignore
//! let store = OpenSearchVectorStoreBuilder::new(url, 1536)
//!     .auto_create_index(false)
//!     .build()
//!     .await?;
//! ```
//!
//! ## AWS OpenSearch Service
//!
//! Enable the `aws-auth` feature for SigV4 authentication:
//!
//! ```toml
//! daimon-plugin-opensearch = { version = "0.23", features = ["aws-auth"] }
//! ```

mod builder;
pub mod index_settings;
mod store;

pub use builder::OpenSearchVectorStoreBuilder;
pub use store::OpenSearchVectorStore;

/// Distance metric / space type for k-NN vector search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpaceType {
    /// Cosine similarity. Best for normalized embeddings.
    #[default]
    CosineSimilarity,
    /// Euclidean (L2) distance. Best for absolute spatial similarity.
    L2,
    /// Inner product. Best for maximum inner product search (MIPS).
    InnerProduct,
}

impl SpaceType {
    /// Returns the OpenSearch space type string used in index mappings.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CosineSimilarity => "cosinesimil",
            Self::L2 => "l2",
            Self::InnerProduct => "innerproduct",
        }
    }
}

/// k-NN engine used for approximate nearest neighbor search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Engine {
    /// Apache Lucene engine. Good default, supports all space types.
    #[default]
    Lucene,
    /// NMSLIB engine. High performance for cosine and L2.
    Nmslib,
    /// FAISS engine. Supports IVF and HNSW, GPU-accelerated options.
    Faiss,
}

impl Engine {
    /// Returns the OpenSearch engine string used in index mappings.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Lucene => "lucene",
            Self::Nmslib => "nmslib",
            Self::Faiss => "faiss",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_space_type_as_str() {
        assert_eq!(SpaceType::CosineSimilarity.as_str(), "cosinesimil");
        assert_eq!(SpaceType::L2.as_str(), "l2");
        assert_eq!(SpaceType::InnerProduct.as_str(), "innerproduct");
    }

    #[test]
    fn test_space_type_default() {
        assert_eq!(SpaceType::default(), SpaceType::CosineSimilarity);
    }

    #[test]
    fn test_engine_as_str() {
        assert_eq!(Engine::Lucene.as_str(), "lucene");
        assert_eq!(Engine::Nmslib.as_str(), "nmslib");
        assert_eq!(Engine::Faiss.as_str(), "faiss");
    }

    #[test]
    fn test_engine_default() {
        assert_eq!(Engine::default(), Engine::Lucene);
    }
}
