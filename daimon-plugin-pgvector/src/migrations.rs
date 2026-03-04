//! SQL migration strings for manual schema setup.
//!
//! When [`PgVectorStoreBuilder::auto_migrate(false)`](crate::PgVectorStoreBuilder::auto_migrate)
//! is set, run these SQL statements against your database before using the store.

/// Creates the `vector` extension if it does not exist.
pub const CREATE_EXTENSION: &str = "CREATE EXTENSION IF NOT EXISTS vector";

/// Returns the `CREATE TABLE` statement for a given table name and dimension count.
///
/// The table stores document IDs, embedding vectors, text content, and
/// arbitrary JSONB metadata.
///
/// # Example
///
/// ```
/// let sql = daimon_plugin_pgvector::migrations::create_table_sql("documents", 1536);
/// assert!(sql.contains("vector(1536)"));
/// ```
pub fn create_table_sql(table: &str, dimensions: usize) -> String {
    format!(
        "CREATE TABLE IF NOT EXISTS {table} (\
         id TEXT PRIMARY KEY, \
         embedding vector({dimensions}), \
         content TEXT NOT NULL, \
         metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb\
         )"
    )
}

/// Returns the `CREATE INDEX` statement for an HNSW index.
///
/// The operator class is chosen based on the distance metric:
/// - Cosine → `vector_cosine_ops`
/// - L2 → `vector_l2_ops`
/// - InnerProduct → `vector_ip_ops`
///
/// # Parameters
///
/// - `table`: table name
/// - `ops_class`: one of `vector_cosine_ops`, `vector_l2_ops`, `vector_ip_ops`
/// - `m`: HNSW `m` parameter (max connections per layer). `None` uses the PG default (16).
/// - `ef_construction`: HNSW build-time search width. `None` uses the PG default (64).
pub fn create_hnsw_index_sql(
    table: &str,
    ops_class: &str,
    m: Option<usize>,
    ef_construction: Option<usize>,
) -> String {
    let mut with_parts = Vec::new();
    if let Some(m) = m {
        with_parts.push(format!("m = {m}"));
    }
    if let Some(ef) = ef_construction {
        with_parts.push(format!("ef_construction = {ef}"));
    }

    let with_clause = if with_parts.is_empty() {
        String::new()
    } else {
        format!(" WITH ({})", with_parts.join(", "))
    };

    format!(
        "CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw_idx \
         ON {table} USING hnsw (embedding {ops_class}){with_clause}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_table_sql() {
        let sql = create_table_sql("docs", 1536);
        assert!(sql.contains("docs"));
        assert!(sql.contains("vector(1536)"));
        assert!(sql.contains("id TEXT PRIMARY KEY"));
        assert!(sql.contains("content TEXT NOT NULL"));
        assert!(sql.contains("metadata JSONB"));
    }

    #[test]
    fn test_create_hnsw_index_defaults() {
        let sql = create_hnsw_index_sql("docs", "vector_cosine_ops", None, None);
        assert!(sql.contains("USING hnsw"));
        assert!(sql.contains("vector_cosine_ops"));
        assert!(!sql.contains("WITH"));
    }

    #[test]
    fn test_create_hnsw_index_custom_params() {
        let sql = create_hnsw_index_sql("docs", "vector_l2_ops", Some(32), Some(128));
        assert!(sql.contains("WITH (m = 32, ef_construction = 128)"));
    }
}
