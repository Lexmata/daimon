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

        // One `_bulk` request instead of one `index` request per document.
        let mut body: Vec<opensearch::http::request::JsonBody<serde_json::Value>> =
            Vec::with_capacity(items.len() * 2);
        for (id, embedding, document) in items {
            body.push(json!({ "index": { "_id": id } }).into());
            body.push(
                json!({
                    "embedding": embedding,
                    "content": document.content,
                    "metadata": document.metadata,
                })
                .into(),
            );
        }

        let response = self
            .client
            .bulk(opensearch::BulkParts::Index(&self.index))
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
                "opensearch bulk upsert failed ({status}): {text}"
            )));
        }

        // A bulk request can return 200 with per-item failures.
        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| DaimonError::Other(format!("opensearch response parse error: {e}")))?;
        if body["errors"].as_bool().unwrap_or(false) {
            let first_error = body["items"]
                .as_array()
                .and_then(|items| {
                    items
                        .iter()
                        .find_map(|item| item["index"]["error"].as_object())
                })
                .map(|e| serde_json::Value::Object(e.clone()).to_string())
                .unwrap_or_else(|| "unknown item error".into());
            return Err(DaimonError::Other(format!(
                "opensearch bulk upsert had item failures: {first_error}"
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
            results.push(hit_to_scored_document(hit)?);
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

/// Converts a single OpenSearch search-response `hit` into a [`ScoredDocument`].
///
/// Extracted from [`OpenSearchVectorStore::query`] so the id-extraction path
/// (the same round-trip-to-delete id DAIM-29 introduced) is unit-testable
/// against a fake JSON hit without a live OpenSearch cluster. A missing or
/// non-string `_id` returns `Err` rather than silently degrading to an empty
/// string, since an empty id would produce a document that can never be
/// deleted again.
fn hit_to_scored_document(hit: &serde_json::Value) -> Result<ScoredDocument> {
    let id = hit["_id"]
        .as_str()
        .ok_or_else(|| DaimonError::Other(format!("opensearch hit missing _id: {hit}")))?
        .to_string();
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
    Ok(ScoredDocument::new(id, doc, score))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn hit_to_scored_document_extracts_id_content_and_score() {
        let hit = json!({
            "_id": "doc-1",
            "_score": 0.87,
            "_source": {
                "content": "hello world",
                "metadata": {"k": "v"}
            }
        });

        let scored = hit_to_scored_document(&hit).unwrap();
        assert_eq!(scored.id, "doc-1");
        assert_eq!(scored.document.content, "hello world");
        assert_eq!(scored.score, 0.87);
    }

    #[test]
    fn hit_to_scored_document_errors_on_missing_id() {
        let hit = json!({
            "_score": 0.5,
            "_source": {"content": "no id here"}
        });

        let err = hit_to_scored_document(&hit).unwrap_err();
        assert!(err.to_string().contains("_id"));
    }

    #[test]
    fn hit_to_scored_document_errors_on_non_string_id() {
        let hit = json!({
            "_id": 12345,
            "_score": 0.5,
            "_source": {"content": "numeric id"}
        });

        let err = hit_to_scored_document(&hit).unwrap_err();
        assert!(err.to_string().contains("_id"));
    }
}
