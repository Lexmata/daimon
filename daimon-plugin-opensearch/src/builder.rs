//! Builder for [`OpenSearchVectorStore`].

use daimon_core::{DaimonError, Result};
use opensearch::OpenSearch;
use opensearch::http::transport::Transport;

use crate::index_settings;
use crate::store::OpenSearchVectorStore;
use crate::{Engine, SpaceType};

/// Builds an [`OpenSearchVectorStore`] with optional auto-index-creation.
///
/// # Example
///
/// ```ignore
/// use daimon_plugin_opensearch::{OpenSearchVectorStoreBuilder, SpaceType, Engine};
///
/// let store = OpenSearchVectorStoreBuilder::new("http://localhost:9200", 1536)
///     .index("embeddings")
///     .space_type(SpaceType::CosineSimilarity)
///     .engine(Engine::Lucene)
///     .hnsw_m(16)
///     .hnsw_ef_construction(256)
///     .auto_create_index(true)
///     .build()
///     .await?;
/// ```
pub struct OpenSearchVectorStoreBuilder {
    url: String,
    dimensions: usize,
    index: String,
    space_type: SpaceType,
    engine: Engine,
    auto_create_index: bool,
    hnsw_m: Option<usize>,
    hnsw_ef_construction: Option<usize>,
}

impl OpenSearchVectorStoreBuilder {
    /// Creates a new builder.
    ///
    /// - `url`: OpenSearch cluster URL (e.g. `"http://localhost:9200"`)
    /// - `dimensions`: the fixed vector dimension count (must match your embedding model)
    pub fn new(url: impl Into<String>, dimensions: usize) -> Self {
        Self {
            url: url.into(),
            dimensions,
            index: "daimon_vectors".into(),
            space_type: SpaceType::default(),
            engine: Engine::default(),
            auto_create_index: true,
            hnsw_m: None,
            hnsw_ef_construction: None,
        }
    }

    /// Sets the index name. Default: `"daimon_vectors"`.
    pub fn index(mut self, index: impl Into<String>) -> Self {
        self.index = index.into();
        self
    }

    /// Sets the k-NN space type (distance metric). Default: [`SpaceType::CosineSimilarity`].
    pub fn space_type(mut self, space_type: SpaceType) -> Self {
        self.space_type = space_type;
        self
    }

    /// Sets the k-NN engine. Default: [`Engine::Lucene`].
    pub fn engine(mut self, engine: Engine) -> Self {
        self.engine = engine;
        self
    }

    /// Enables or disables automatic index creation on first use.
    /// Default: `true`.
    ///
    /// When disabled, use the JSON from [`crate::index_settings`] to create
    /// the index manually.
    pub fn auto_create_index(mut self, enabled: bool) -> Self {
        self.auto_create_index = enabled;
        self
    }

    /// Sets the HNSW `m` parameter (max connections per layer).
    /// `None` uses the engine default.
    pub fn hnsw_m(mut self, m: usize) -> Self {
        self.hnsw_m = Some(m);
        self
    }

    /// Sets the HNSW `ef_construction` parameter (build-time search width).
    /// `None` uses the engine default.
    pub fn hnsw_ef_construction(mut self, ef: usize) -> Self {
        self.hnsw_ef_construction = Some(ef);
        self
    }

    /// Builds an [`OpenSearchVectorStore`] from a pre-existing [`OpenSearch`] client.
    ///
    /// Use this when you need custom transport configuration (e.g. AWS SigV4,
    /// custom certificates, connection pool tuning).
    pub async fn build_with_client(self, client: OpenSearch) -> Result<OpenSearchVectorStore> {
        if self.auto_create_index {
            self.ensure_index(&client).await?;
        }

        Ok(OpenSearchVectorStore {
            client,
            index: self.index,
            dimensions: self.dimensions,
            space_type: self.space_type,
        })
    }

    /// Builds the [`OpenSearchVectorStore`], optionally creating the index.
    pub async fn build(self) -> Result<OpenSearchVectorStore> {
        let transport = Transport::single_node(&self.url)
            .map_err(|e| DaimonError::Other(format!("opensearch transport error: {e}")))?;
        let client = OpenSearch::new(transport);

        self.build_with_client(client).await
    }

    async fn ensure_index(&self, client: &OpenSearch) -> Result<()> {
        let exists = client
            .indices()
            .exists(opensearch::indices::IndicesExistsParts::Index(&[
                &self.index
            ]))
            .send()
            .await
            .map_err(|e| DaimonError::Other(format!("opensearch index check error: {e}")))?;

        if exists.status_code().is_success() {
            tracing::debug!("opensearch: index '{}' already exists", self.index);
            return Ok(());
        }

        tracing::info!("opensearch: creating k-NN index '{}'", self.index);

        let body = index_settings::create_index_body(
            self.dimensions,
            self.space_type,
            self.engine,
            self.hnsw_m,
            self.hnsw_ef_construction,
        );

        let response = client
            .indices()
            .create(opensearch::indices::IndicesCreateParts::Index(&self.index))
            .body(body)
            .send()
            .await
            .map_err(|e| DaimonError::Other(format!("opensearch index create error: {e}")))?;

        let status = response.status_code();
        if !status.is_success() {
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".into());
            return Err(DaimonError::Other(format!(
                "opensearch index creation failed ({status}): {text}"
            )));
        }

        tracing::info!("opensearch: index '{}' created", self.index);
        Ok(())
    }
}
