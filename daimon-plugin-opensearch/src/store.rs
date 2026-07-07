//! [`OpenSearchVectorStore`] — an OpenSearch k-NN backed [`VectorStore`] implementation.

use std::collections::HashMap;

use daimon_core::vector_store::VectorStore;
use daimon_core::{DaimonError, Document, Result, ScoredDocument};
use opensearch::OpenSearch;
use serde_json::json;

use crate::SpaceType;

/// A [`VectorStore`] backed by OpenSearch with the k-NN plugin.
///
/// Use [`OpenSearchVectorStoreBuilder`](crate::OpenSearchVectorStoreBuilder) to construct.
pub struct OpenSearchVectorStore {
    pub(crate) client: OpenSearch,
    pub(crate) index: String,
    pub(crate) dimensions: usize,
    pub(crate) space_type: SpaceType,
}

impl OpenSearchVectorStore {
    /// Returns a reference to the underlying OpenSearch client.
    pub fn client(&self) -> &OpenSearch {
        &self.client
    }

    /// Returns the index name used by this store.
    pub fn index(&self) -> &str {
        &self.index
    }

    /// Returns the configured vector dimensions.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Returns the configured k-NN space type (distance metric).
    ///
    /// This determines how OpenSearch scores query hits; see [`query`](VectorStore::query)
    /// for why those scores are only comparable within a single space type.
    pub fn space_type(&self) -> SpaceType {
        self.space_type
    }

    fn map_os_error(resp: opensearch::Error) -> DaimonError {
        DaimonError::Other(format!("opensearch error: {resp}"))
    }
}

impl VectorStore for OpenSearchVectorStore {
    async fn upsert(&self, id: &str, embedding: Vec<f32>, document: Document) -> Result<()> {
        if embedding.len() != self.dimensions {
            return Err(DaimonError::Other(format!(
                "embedding dimension mismatch: expected {}, got {}",
                self.dimensions,
                embedding.len()
            )));
        }

        let body = json!({
            "embedding": embedding,
            "content": document.content,
            "metadata": document.metadata,
        });

        let response = self
            .client
            .index(opensearch::IndexParts::IndexId(&self.index, id))
            .body(body)
            .send()
            .await
            .map_err(Self::map_os_error)?;

        let status = response.status_code();
        if !status.is_success() {
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".into());
            return Err(DaimonError::Other(format!(
                "opensearch upsert failed ({status}): {text}"
            )));
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

        let body = json!({
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": embedding,
                        "k": top_k
                    }
                }
            },
            "_source": ["content", "metadata"]
        });

        let response = self
            .client
            .search(opensearch::SearchParts::Index(&[&self.index]))
            .body(body)
            .send()
            .await
            .map_err(Self::map_os_error)?;

        let status = response.status_code();
        if !status.is_success() {
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".into());
            return Err(DaimonError::Other(format!(
                "opensearch query failed ({status}): {text}"
            )));
        }

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| DaimonError::Other(format!("opensearch response parse error: {e}")))?;

        let hits = body["hits"]["hits"]
            .as_array()
            .unwrap_or(&Vec::new())
            .clone();

        let mut results = Vec::with_capacity(hits.len());
        for hit in &hits {
            let content = hit["_source"]["content"]
                .as_str()
                .unwrap_or_default()
                .to_string();

            let metadata: HashMap<String, serde_json::Value> = hit["_source"]
                .get("metadata")
                .and_then(|m| serde_json::from_value(m.clone()).ok())
                .unwrap_or_default();

            // This is the backend-raw `_score` returned by OpenSearch. Its scale
            // depends on the configured space type — OpenSearch applies a
            // different transform per metric (e.g. `1 / (1 + distance)` for l2,
            // and metric-specific formulas for cosinesimil / innerproduct), and
            // these are not on a common, cleanly comparable 0..1 scale. We
            // therefore surface the raw score unchanged: it is meaningful only
            // for *ranking within a single space type*, not for cross-metric
            // comparison or as a calibrated similarity probability.
            let score = hit["_score"].as_f64().unwrap_or(0.0);

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
        let response = self
            .client
            .delete(opensearch::DeleteParts::IndexId(&self.index, id))
            .send()
            .await
            .map_err(Self::map_os_error)?;

        let status = response.status_code();
        if status == opensearch::http::StatusCode::NOT_FOUND {
            return Ok(false);
        }
        if !status.is_success() {
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".into());
            return Err(DaimonError::Other(format!(
                "opensearch delete failed ({status}): {text}"
            )));
        }

        Ok(true)
    }

    async fn count(&self) -> Result<usize> {
        let response = self
            .client
            .count(opensearch::CountParts::Index(&[&self.index]))
            .send()
            .await
            .map_err(Self::map_os_error)?;

        let status = response.status_code();
        if !status.is_success() {
            let text = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".into());
            return Err(DaimonError::Other(format!(
                "opensearch count failed ({status}): {text}"
            )));
        }

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| DaimonError::Other(format!("opensearch response parse error: {e}")))?;

        let count = body["count"].as_u64().unwrap_or(0) as usize;
        Ok(count)
    }
}
