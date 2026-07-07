//! Builder for [`PgVectorStore`].

use daimon_core::{DaimonError, Result};
use deadpool_postgres::{Config, Pool, Runtime};
use tokio_postgres::NoTls;

use crate::DistanceMetric;
use crate::migrations;
use crate::store::PgVectorStore;

/// Builds a [`PgVectorStore`] with connection pooling and optional auto-migration.
///
/// # Example
///
/// ```ignore
/// use daimon_plugin_pgvector::{PgVectorStoreBuilder, DistanceMetric};
///
/// let store = PgVectorStoreBuilder::new("host=localhost dbname=mydb", 1536)
///     .table("embeddings")
///     .distance_metric(DistanceMetric::Cosine)
///     .hnsw_m(16)
///     .hnsw_ef_construction(64)
///     .auto_migrate(true)
///     .build()
///     .await?;
/// ```
pub struct PgVectorStoreBuilder {
    connection_string: String,
    dimensions: usize,
    table: String,
    distance_metric: DistanceMetric,
    auto_migrate: bool,
    hnsw_m: Option<usize>,
    hnsw_ef_construction: Option<usize>,
    pool_size: usize,
}

impl PgVectorStoreBuilder {
    /// Creates a new builder.
    ///
    /// - `connection_string`: PostgreSQL connection string
    ///   (e.g. `"host=localhost dbname=mydb user=postgres"` or
    ///   `"postgresql://user:pass@host/db"`)
    /// - `dimensions`: the fixed vector dimension count (must match your embedding model)
    pub fn new(connection_string: impl Into<String>, dimensions: usize) -> Self {
        Self {
            connection_string: connection_string.into(),
            dimensions,
            table: "daimon_vectors".into(),
            distance_metric: DistanceMetric::Cosine,
            auto_migrate: true,
            hnsw_m: None,
            hnsw_ef_construction: None,
            pool_size: 16,
        }
    }

    /// Sets the table name. Default: `"daimon_vectors"`.
    pub fn table(mut self, table: impl Into<String>) -> Self {
        self.table = table.into();
        self
    }

    /// Sets the distance metric. Default: [`DistanceMetric::Cosine`].
    pub fn distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.distance_metric = metric;
        self
    }

    /// Enables or disables automatic schema creation on first connection.
    /// Default: `true`.
    ///
    /// When disabled, use the SQL from [`crate::migrations`] to set up
    /// the schema manually.
    pub fn auto_migrate(mut self, enabled: bool) -> Self {
        self.auto_migrate = enabled;
        self
    }

    /// Sets the HNSW `m` parameter (max connections per layer).
    /// `None` uses the PostgreSQL default (16).
    pub fn hnsw_m(mut self, m: usize) -> Self {
        self.hnsw_m = Some(m);
        self
    }

    /// Sets the HNSW `ef_construction` parameter (build-time search width).
    /// `None` uses the PostgreSQL default (64).
    pub fn hnsw_ef_construction(mut self, ef: usize) -> Self {
        self.hnsw_ef_construction = Some(ef);
        self
    }

    /// Sets the maximum number of connections in the pool. Default: `16`.
    pub fn pool_size(mut self, size: usize) -> Self {
        self.pool_size = size;
        self
    }

    /// Builds the [`PgVectorStore`], optionally running migrations.
    pub async fn build(self) -> Result<PgVectorStore> {
        // Validate the table name *before* it is ever interpolated into SQL.
        // PostgreSQL cannot bind identifiers as parameters, so the table name
        // is formatted directly into every statement in `store.rs` and
        // `migrations.rs`. An unvalidated name like `foo; DROP TABLE bar; --`
        // would be a straightforward SQL injection. We only accept plain or
        // schema-qualified identifiers matching `[A-Za-z_][A-Za-z0-9_]*`.
        validate_table_name(&self.table)?;

        let pool = self.create_pool()?;

        if self.auto_migrate {
            self.run_migrations(&pool).await?;
        }

        Ok(PgVectorStore {
            pool,
            table: self.table,
            dimensions: self.dimensions,
            distance_metric: self.distance_metric,
        })
    }

    // (validation lives in the free `validate_table_name` fn below)

    fn create_pool(&self) -> Result<Pool> {
        let mut cfg = Config::new();
        cfg.url = Some(self.connection_string.clone());
        cfg.pool = Some(deadpool_postgres::PoolConfig {
            max_size: self.pool_size,
            ..Default::default()
        });

        cfg.create_pool(Some(Runtime::Tokio1), NoTls)
            .map_err(|e| DaimonError::Other(format!("pgvector pool creation error: {e}")))
    }

    async fn run_migrations(&self, pool: &Pool) -> Result<()> {
        let client = pool
            .get()
            .await
            .map_err(|e| DaimonError::Other(format!("pgvector migration pool error: {e}")))?;

        tracing::info!("pgvector: creating extension and table '{}'", self.table);

        client
            .execute(migrations::CREATE_EXTENSION, &[])
            .await
            .map_err(|e| DaimonError::Other(format!("pgvector CREATE EXTENSION error: {e}")))?;

        let create_table = migrations::create_table_sql(&self.table, self.dimensions);
        client
            .execute(&create_table as &str, &[])
            .await
            .map_err(|e| DaimonError::Other(format!("pgvector CREATE TABLE error: {e}")))?;

        let ops_class = self.distance_metric.ops_class();
        let create_index = migrations::create_hnsw_index_sql(
            &self.table,
            ops_class,
            self.hnsw_m,
            self.hnsw_ef_construction,
        );
        client
            .execute(&create_index as &str, &[])
            .await
            .map_err(|e| DaimonError::Other(format!("pgvector CREATE INDEX error: {e}")))?;

        tracing::info!("pgvector: migration complete for '{}'", self.table);
        Ok(())
    }
}

/// Validates a (possibly schema-qualified) PostgreSQL table identifier.
///
/// Accepts a single identifier (`embeddings`) or a schema-qualified one
/// (`public.embeddings`) where every dot-separated part matches
/// `^[A-Za-z_][A-Za-z0-9_]*$`. Rejects anything containing quotes, whitespace,
/// semicolons, or other punctuation that could break out of the identifier
/// position in a formatted SQL statement.
fn validate_table_name(table: &str) -> Result<()> {
    fn is_valid_part(part: &str) -> bool {
        let mut chars = part.chars();
        match chars.next() {
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
            _ => return false,
        }
        chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
    }

    if table.is_empty() {
        return Err(DaimonError::Other(
            "pgvector: table name must not be empty".to_string(),
        ));
    }

    let parts: Vec<&str> = table.split('.').collect();
    if parts.len() > 2 || !parts.iter().all(|p| is_valid_part(p)) {
        return Err(DaimonError::Other(format!(
            "pgvector: invalid table name '{table}': expected an identifier matching \
             [A-Za-z_][A-Za-z0-9_]* (optionally schema-qualified as schema.table)"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_table_names() {
        assert!(validate_table_name("embeddings").is_ok());
        assert!(validate_table_name("daimon_vectors").is_ok());
        assert!(validate_table_name("_private").is_ok());
        assert!(validate_table_name("Table123").is_ok());
        assert!(validate_table_name("public.embeddings").is_ok());
    }

    #[test]
    fn test_invalid_table_names_rejected() {
        assert!(validate_table_name("").is_err());
        assert!(validate_table_name("foo; DROP TABLE bar").is_err());
        assert!(validate_table_name("foo; DROP TABLE bar; --").is_err());
        assert!(validate_table_name("\"foo\"").is_err());
        assert!(validate_table_name("foo bar").is_err());
        assert!(validate_table_name("1foo").is_err());
        assert!(validate_table_name("foo.bar.baz").is_err());
        assert!(validate_table_name("foo'").is_err());
        assert!(validate_table_name("foo)").is_err());
        assert!(validate_table_name("schema..table").is_err());
    }
}
