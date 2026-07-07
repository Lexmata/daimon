//! Qdrant vector store retriever.

use std::sync::Arc;

use crate::error::{DaimonError, Result};
use crate::model::ErasedEmbeddingModel;
use crate::retriever::traits::Retriever;
use crate::retriever::types::Document;

/// Retriever backed by a Qdrant vector database.
pub struct QdrantRetriever {
    client: qdrant_client::Qdrant,
    collection: String,
    embedding_model: Arc<dyn ErasedEmbeddingModel>,
    content_field: String,
}

impl QdrantRetriever {
    /// Creates a new Qdrant retriever.
    ///
    /// - `url`: Qdrant server URL (e.g. `http://localhost:6334`)
    /// - `collection`: Name of the Qdrant collection
    /// - `embedding_model`: Model used to embed queries
    pub async fn new(
        url: impl Into<String>,
        collection: impl Into<String>,
        embedding_model: Arc<dyn ErasedEmbeddingModel>,
    ) -> Result<Self> {
        let client = qdrant_client::Qdrant::from_url(&url.into())
            .build()
            .map_err(|e| DaimonError::Other(format!("Qdrant connection error: {e}")))?;

        Ok(Self {
            client,
            collection: collection.into(),
            embedding_model,
            content_field: "content".to_string(),
        })
    }

    /// Sets the payload field name that contains document text. Default: `"content"`.
    pub fn with_content_field(mut self, field: impl Into<String>) -> Self {
        self.content_field = field.into();
        self
    }
}

impl Retriever for QdrantRetriever {
    async fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<Document>> {
        let texts = [query];
        let embeddings = self.embedding_model.embed_erased(&texts).await?;
        let query_vec = embeddings.into_iter().next().unwrap_or_default();

        let results = self
            .client
            .search_points(
                qdrant_client::qdrant::SearchPointsBuilder::new(
                    &self.collection,
                    query_vec,
                    top_k as u64,
                )
                .with_payload(true),
            )
            .await
            .map_err(|e| DaimonError::Other(format!("Qdrant search error: {e}")))?;

        let mut docs = Vec::with_capacity(results.result.len());
        for point in results.result {
            let content = point
                .payload
                .get(&self.content_field)
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_default();

            let mut doc = Document::new(content).with_score(point.score as f64);

            for (key, val) in &point.payload {
                if key != &self.content_field
                    && let Some(s) = val.as_str()
                {
                    doc = doc.with_metadata(key, serde_json::json!(s));
                }
            }

            docs.push(doc);
        }

        Ok(docs)
    }
}
