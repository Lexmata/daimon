//! # daimon-plugin-pgvector
//!
//! A pgvector-backed [`VectorStore`](daimon_core::VectorStore) plugin for
//! the [Daimon](https://docs.rs/daimon) AI agent framework.
//!
//! This crate provides [`PgVectorStore`], which stores document embeddings
//! in PostgreSQL using the [pgvector](https://github.com/pgvector/pgvector)
//! extension. It supports cosine, L2, and inner-product distance metrics
//! with HNSW indexing.
//!
//! ## Quick Start
//!
//! ```ignore
//! use daimon_plugin_pgvector::{PgVectorStoreBuilder, DistanceMetric};
//! use daimon::retriever::SimpleKnowledgeBase;
//! use std::sync::Arc;
//!
//! let store = PgVectorStoreBuilder::new("postgresql://user:pass@localhost/db", 1536)
//!     .table("my_docs")
//!     .distance_metric(DistanceMetric::Cosine)
//!     .build()
//!     .await?;
//!
//! // Compose with an embedding model for a full RAG pipeline:
//! let kb = SimpleKnowledgeBase::new(embedding_model, store);
//! ```
//!
//! ## Manual Schema Setup
//!
//! If you prefer to manage migrations yourself, disable auto-migration
//! and use the SQL from [`migrations`]:
//!
//! ```ignore
//! let store = PgVectorStoreBuilder::new(conn_str, 1536)
//!     .auto_migrate(false)
//!     .build()
//!     .await?;
//! ```
//!
//! Then run the SQL from [`migrations::CREATE_EXTENSION`],
//! [`migrations::create_table_sql`], and [`migrations::create_hnsw_index_sql`]
//! against your database.

mod builder;
pub mod migrations;
mod store;

pub use builder::PgVectorStoreBuilder;
pub use store::PgVectorStore;

/// Distance metric used for vector similarity search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Cosine similarity (1 - cosine distance). Best for normalized embeddings.
    Cosine,
    /// Euclidean (L2) distance. Best for absolute spatial similarity.
    L2,
    /// Inner product. Best for maximum inner product search (MIPS).
    InnerProduct,
}

impl DistanceMetric {
    /// Returns the pgvector operator class for HNSW index creation.
    pub fn ops_class(&self) -> &'static str {
        match self {
            Self::Cosine => "vector_cosine_ops",
            Self::L2 => "vector_l2_ops",
            Self::InnerProduct => "vector_ip_ops",
        }
    }
}

impl Default for DistanceMetric {
    fn default() -> Self {
        Self::Cosine
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_metric_ops_class() {
        assert_eq!(DistanceMetric::Cosine.ops_class(), "vector_cosine_ops");
        assert_eq!(DistanceMetric::L2.ops_class(), "vector_l2_ops");
        assert_eq!(DistanceMetric::InnerProduct.ops_class(), "vector_ip_ops");
    }

    #[test]
    fn test_distance_metric_default() {
        assert_eq!(DistanceMetric::default(), DistanceMetric::Cosine);
    }
}
