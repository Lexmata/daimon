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
///
/// Statements are prepared through deadpool's per-connection statement cache
/// (`prepare_cached`), so the constant upsert/query/delete/count SQL is
/// parsed and planned once per pooled connection instead of per call.
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

    /// Builds the SQL expression that converts pgvector's distance operator
    /// (`$1` is the query vector) into a similarity score.
    ///
    /// See [`VectorStore::query`] for the rationale behind each transform.
    /// L2 uses `1 / (1 + distance)` so the score stays in `(0, 1]` and
    /// monotonic; a naive `1 - distance` would be unbounded and go negative.
    fn score_expr(&self) -> String {
        let op = self.distance_operator();
        match self.distance_metric {
            DistanceMetric::Cosine => format!("1.0 - (embedding {op} $1)"),
            DistanceMetric::L2 => format!("1.0 / (1.0 + (embedding {op} $1))"),
            DistanceMetric::InnerProduct => format!("-(embedding {op} $1)"),
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

/// Rows per multi-row `INSERT`: Postgres caps statements at 65535 bind
/// parameters and each row binds 4 (id, embedding, content, metadata).
const MAX_ROWS_PER_INSERT: usize = 65_535 / 4;

/// Builds the multi-row upsert statement for `rows` rows.
fn multi_upsert_sql(table: &str, rows: usize) -> String {
    let mut values = String::with_capacity(rows * 24);
    for i in 0..rows {
        if i > 0 {
            values.push_str(", ");
        }
        let base = i * 4;
        values.push_str(&format!(
            "(${}, ${}, ${}, ${})",
            base + 1,
            base + 2,
            base + 3,
            base + 4
        ));
    }
    format!(
        "INSERT INTO {table} (id, embedding, content, metadata) VALUES {values} \
         ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding, \
         content = EXCLUDED.content, metadata = EXCLUDED.metadata"
    )
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

        let stmt = client
            .prepare_cached(&sql)
            .await
            .map_err(|e| DaimonError::Other(format!("pgvector prepare error: {e}")))?;
        client
            .execute(&stmt, &[&id, &vec, &document.content, &metadata])
            .await
            .map_err(|e| DaimonError::Other(format!("pgvector upsert error: {e}")))?;

        Ok(())
    }

    async fn upsert_many(&self, items: Vec<(String, Vec<f32>, Document)>) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        for (_, embedding, _) in &items {
            if embedding.len() != self.dimensions {
                return Err(DaimonError::Other(format!(
                    "embedding dimension mismatch: expected {}, got {}",
                    self.dimensions,
                    embedding.len()
                )));
            }
        }

        // Postgres rejects duplicate ids within one ON CONFLICT statement
        // ("cannot affect row a second time"), so dedupe keeping the last
        // occurrence — the same outcome the sequential upsert loop produced.
        let mut index: HashMap<String, usize> = HashMap::with_capacity(items.len());
        let mut rows: Vec<(String, Vector, String, serde_json::Value)> =
            Vec::with_capacity(items.len());
        for (id, embedding, document) in items {
            let metadata = serde_json::to_value(&document.metadata)
                .map_err(|e| DaimonError::Other(format!("metadata serialization error: {e}")))?;
            let row = (id, Vector::from(embedding), document.content, metadata);
            match index.entry(row.0.clone()) {
                std::collections::hash_map::Entry::Occupied(e) => rows[*e.get()] = row,
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(rows.len());
                    rows.push(row);
                }
            }
        }

        let client = self
            .pool
            .get()
            .await
            .map_err(|e| DaimonError::Other(format!("pgvector pool error: {e}")))?;

        for chunk in rows.chunks(MAX_ROWS_PER_INSERT) {
            let sql = multi_upsert_sql(&self.table, chunk.len());
            let mut params: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> =
                Vec::with_capacity(chunk.len() * 4);
            for (id, vec, content, metadata) in chunk {
                params.push(id);
                params.push(vec);
                params.push(content);
                params.push(metadata);
            }
            let stmt = client
                .prepare_cached(&sql)
                .await
                .map_err(|e| DaimonError::Other(format!("pgvector prepare error: {e}")))?;
            client
                .execute(&stmt, &params)
                .await
                .map_err(|e| DaimonError::Other(format!("pgvector upsert error: {e}")))?;
        }

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
        let score_expr = self.score_expr();

        let sql = format!(
            "SELECT id, content, metadata, {score_expr} AS score \
             FROM {} ORDER BY embedding {op} $1 LIMIT $2",
            self.table
        );

        let stmt = client
            .prepare_cached(&sql)
            .await
            .map_err(|e| DaimonError::Other(format!("pgvector prepare error: {e}")))?;
        let rows = client
            .query(&stmt, &[&vec, &(top_k as i64)])
            .await
            .map_err(|e| DaimonError::Other(format!("pgvector query error: {e}")))?;

        let mut results = Vec::with_capacity(rows.len());
        for row in rows {
            let id: String = row.get("id");
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
            results.push(ScoredDocument::new(id, doc, score));
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
        let stmt = client
            .prepare_cached(&sql)
            .await
            .map_err(|e| DaimonError::Other(format!("pgvector prepare error: {e}")))?;
        let deleted = client
            .execute(&stmt, &[&id])
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
        let stmt = client
            .prepare_cached(&sql)
            .await
            .map_err(|e| DaimonError::Other(format!("pgvector prepare error: {e}")))?;
        let row = client
            .query_one(&stmt, &[])
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

    #[test]
    fn test_score_expr_per_metric() {
        let base = PgVectorStore {
            pool: create_dummy_pool(),
            table: "t".into(),
            dimensions: 3,
            distance_metric: DistanceMetric::Cosine,
        };
        assert_eq!(base.score_expr(), "1.0 - (embedding <=> $1)");

        let l2 = PgVectorStore {
            distance_metric: DistanceMetric::L2,
            ..base
        };
        // L2 must use the bounded transform, not `1 - distance`.
        assert_eq!(l2.score_expr(), "1.0 / (1.0 + (embedding <-> $1))");

        let ip = PgVectorStore {
            distance_metric: DistanceMetric::InnerProduct,
            ..l2
        };
        assert_eq!(ip.score_expr(), "-(embedding <#> $1)");
    }

    #[test]
    fn test_multi_upsert_sql_placeholders() {
        let one = multi_upsert_sql("t", 1);
        assert!(one.contains("VALUES ($1, $2, $3, $4) ON CONFLICT"));

        let two = multi_upsert_sql("t", 2);
        assert!(two.contains("VALUES ($1, $2, $3, $4), ($5, $6, $7, $8) ON CONFLICT"));

        // The largest chunk stays under Postgres's 65535-parameter cap.
        let max = multi_upsert_sql("t", MAX_ROWS_PER_INSERT);
        assert!(max.contains(&format!("${}", MAX_ROWS_PER_INSERT * 4)));
        assert!(!max.contains(&format!("${}", MAX_ROWS_PER_INSERT * 4 + 1)));
        const { assert!(MAX_ROWS_PER_INSERT * 4 <= 65_535) };
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
