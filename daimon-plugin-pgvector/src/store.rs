//! [`PgVectorStore`] — a pgvector-backed [`VectorStore`] implementation.

use std::collections::HashMap;

use daimon_core::vector_store::VectorStore;
use daimon_core::{DaimonError, Document, Result, ScoredDocument};
use deadpool_postgres::Pool;
use pgvector::Vector;

use crate::DistanceMetric;

/// A [`VectorStore`] backed by PostgreSQL with the pgvector extension.
///
/// Use [`PgVectorStoreBuilder`](crate::PgVectorStoreBuilder) to construct.
pub struct PgVectorStore {
    pub(crate) pool: Pool,
    pub(crate) table: String,
    pub(crate) dimensions: usize,
    pub(crate) distance_metric: DistanceMetric,
}

impl PgVectorStore {
    /// Returns the distance operator used in SQL ORDER BY clauses.
    fn distance_operator(&self) -> &'static str {
        match self.distance_metric {
            DistanceMetric::Cosine => "<=>",
            DistanceMetric::L2 => "<->",
            DistanceMetric::InnerProduct => "<#>",
        }
    }

    /// Returns a reference to the underlying connection pool.
    pub fn pool(&self) -> &Pool {
        &self.pool
    }

    /// Returns the table name used by this store.
    pub fn table(&self) -> &str {
        &self.table
    }

    /// Returns the configured vector dimensions.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
}

impl VectorStore for PgVectorStore {
    async fn upsert(&self, id: &str, embedding: Vec<f32>, document: Document) -> Result<()> {
        if embedding.len() != self.dimensions {
            return Err(DaimonError::Other(format!(
                "embedding dimension mismatch: expected {}, got {}",
                self.dimensions,
                embedding.len()
            )));
        }

        let client = self
            .pool
            .get()
            .await
            .map_err(|e| DaimonError::Other(format!("pgvector pool error: {e}")))?;

        let vec = Vector::from(embedding);
        let metadata = serde_json::to_value(&document.metadata)
            .map_err(|e| DaimonError::Other(format!("metadata serialization error: {e}")))?;

        let sql = format!(
            "INSERT INTO {} (id, embedding, content, metadata) VALUES ($1, $2, $3, $4) \
             ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding, \
             content = EXCLUDED.content, metadata = EXCLUDED.metadata",
            self.table
        );

        client
            .execute(&sql as &str, &[&id, &vec, &document.content, &metadata])
            .await
            .map_err(|e| DaimonError::Other(format!("pgvector upsert error: {e}")))?;

        Ok(())
    }

    async fn query(&self, embedding: Vec<f32>, top_k: usize) -> Result<Vec<ScoredDocument>> {
        if embedding.len() != self.dimensions {
            return Err(DaimonError::Other(format!(
                "embedding dimension mismatch: expected {}, got {}",
                self.dimensions,
                embedding.len()
            )));
        }

        let client = self
            .pool
            .get()
            .await
            .map_err(|e| DaimonError::Other(format!("pgvector pool error: {e}")))?;

        let vec = Vector::from(embedding);
        let op = self.distance_operator();

        // For cosine and L2, lower distance = more similar, so score = 1 - distance.
        // For inner product, pgvector returns negative inner product, so score = -distance.
        let score_expr = match self.distance_metric {
            DistanceMetric::Cosine | DistanceMetric::L2 => {
                format!("1.0 - (embedding {op} $1)")
            }
            DistanceMetric::InnerProduct => {
                format!("-(embedding {op} $1)")
            }
        };

        let sql = format!(
            "SELECT id, content, metadata, {score_expr} AS score \
             FROM {} ORDER BY embedding {op} $1 LIMIT $2",
            self.table
        );

        let rows = client
            .query(&sql as &str, &[&vec, &(top_k as i64)])
            .await
            .map_err(|e| DaimonError::Other(format!("pgvector query error: {e}")))?;

        let mut results = Vec::with_capacity(rows.len());
        for row in rows {
            let content: String = row.get("content");
            let metadata_val: serde_json::Value = row.get("metadata");
            let score: f64 = row.get("score");

            let metadata: HashMap<String, serde_json::Value> =
                serde_json::from_value(metadata_val).unwrap_or_default();

            let doc = Document {
                content,
                metadata,
                score: Some(score),
            };
            results.push(ScoredDocument::new(doc, score));
        }

        Ok(results)
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        let client = self
            .pool
            .get()
            .await
            .map_err(|e| DaimonError::Other(format!("pgvector pool error: {e}")))?;

        let sql = format!("DELETE FROM {} WHERE id = $1", self.table);
        let deleted = client
            .execute(&sql as &str, &[&id])
            .await
            .map_err(|e| DaimonError::Other(format!("pgvector delete error: {e}")))?;

        Ok(deleted > 0)
    }

    async fn count(&self) -> Result<usize> {
        let client = self
            .pool
            .get()
            .await
            .map_err(|e| DaimonError::Other(format!("pgvector pool error: {e}")))?;

        let sql = format!("SELECT COUNT(*) AS cnt FROM {}", self.table);
        let row = client
            .query_one(&sql as &str, &[])
            .await
            .map_err(|e| DaimonError::Other(format!("pgvector count error: {e}")))?;

        let count: i64 = row.get("cnt");
        Ok(count as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_operator() {
        let store = PgVectorStore {
            pool: create_dummy_pool(),
            table: "t".into(),
            dimensions: 3,
            distance_metric: DistanceMetric::Cosine,
        };
        assert_eq!(store.distance_operator(), "<=>");

        let store = PgVectorStore {
            dimensions: 3,
            distance_metric: DistanceMetric::L2,
            ..store
        };
        assert_eq!(store.distance_operator(), "<->");

        let store = PgVectorStore {
            distance_metric: DistanceMetric::InnerProduct,
            ..store
        };
        assert_eq!(store.distance_operator(), "<#>");
    }

    fn create_dummy_pool() -> Pool {
        let cfg = deadpool_postgres::Config {
            host: Some("localhost".into()),
            port: Some(5432),
            dbname: Some("test".into()),
            ..Default::default()
        };
        cfg.create_pool(None, tokio_postgres::NoTls).unwrap()
    }
}
