//! llama.cpp embedding model provider.
//!
//! Targets llama-server's OpenAI-compatible `/v1/embeddings` endpoint
//! (requires the server to run with `--embeddings`).

use std::time::Duration;

use serde::{Deserialize, Serialize};

use daimon_core::{DaimonError, EmbeddingModel, Result};

use crate::{Http, api_error};

/// llama.cpp embedding model, backed by a running `llama-server`.
///
/// ```ignore
/// use daimon_provider_llamacpp::LlamaCppEmbedding;
///
/// let embedding = LlamaCppEmbedding::new().with_dimensions(1024);
/// let vectors = embedding.embed(&["hello world"]).await?;
/// ```
#[derive(Debug)]
pub struct LlamaCppEmbedding {
    http: Http,
    model: Option<String>,
    dimensions: usize,
}

impl Default for LlamaCppEmbedding {
    fn default() -> Self {
        Self::new()
    }
}

impl LlamaCppEmbedding {
    /// Create a client targeting `http://localhost:8080`.
    pub fn new() -> Self {
        Self {
            http: Http::new(),
            model: None,
            dimensions: 768,
        }
    }

    /// Set the server base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.http.set_base_url(url);
        self
    }

    /// Set the model name; only meaningful for multi-model routers.
    pub fn with_model(mut self, name: impl Into<String>) -> Self {
        self.model = Some(name.into());
        self
    }

    /// Set the API key, for servers started with `--api-key`.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.http.set_api_key(key);
        self
    }

    /// Set a custom timeout for HTTP requests.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.http.set_timeout(timeout);
        self
    }

    /// Declare the dimensionality of the loaded model's embeddings.
    ///
    /// llama-server does not truncate or expand vectors; this must match the
    /// GGUF model actually loaded (default 768).
    pub fn with_dimensions(mut self, dims: usize) -> Self {
        self.dimensions = dims;
        self
    }
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    input: &'a [&'a str],
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedDatum>,
}

#[derive(Deserialize)]
struct EmbedDatum {
    embedding: Vec<f32>,
}

impl EmbeddingModel for LlamaCppEmbedding {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let body = EmbedRequest {
            model: self.model.as_deref(),
            input: texts,
        };

        let resp = self.http.post("/v1/embeddings", &body).await?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(api_error(status, &text));
        }

        let data: EmbedResponse = resp
            .json()
            .await
            .map_err(|e| DaimonError::Model(format!("llama.cpp embedding parse error: {e}")))?;

        Ok(data.data.into_iter().map(|d| d.embedding).collect())
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_defaults() {
        let embed = LlamaCppEmbedding::new();
        assert_eq!(embed.dimensions, 768);
        assert!(embed.model.is_none());
        assert_eq!(EmbeddingModel::dimensions(&embed), 768);
    }

    #[test]
    fn test_builder_chain() {
        let embed = LlamaCppEmbedding::new()
            .with_base_url("http://gpu-box:8080")
            .with_model("nomic-embed")
            .with_api_key("secret")
            .with_timeout(Duration::from_secs(10))
            .with_dimensions(1024);
        assert_eq!(embed.model.as_deref(), Some("nomic-embed"));
        assert_eq!(embed.dimensions, 1024);
    }

    #[test]
    fn test_request_serialization() {
        let body = EmbedRequest {
            model: None,
            input: &["hello", "world"],
        };
        let value = serde_json::to_value(&body).unwrap();
        assert!(!value.as_object().unwrap().contains_key("model"));
        assert_eq!(value["input"], serde_json::json!(["hello", "world"]));
    }

    #[test]
    fn test_response_parsing() {
        let resp: EmbedResponse = serde_json::from_str(
            r#"{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2]}],"model":"gguf"}"#,
        )
        .unwrap();
        let vectors: Vec<Vec<f32>> = resp.data.into_iter().map(|d| d.embedding).collect();
        assert_eq!(vectors, vec![vec![0.1, 0.2]]);
    }
}
