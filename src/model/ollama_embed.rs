//! Ollama embeddings provider.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::error::{DaimonError, Result};
use crate::model::EmbeddingModel;

/// Default total request timeout. Embedding calls are non-streaming and
/// bounded, so a hung endpoint now fails after a minute instead of stalling
/// RAG ingest or retrieval forever; override with `with_timeout`.
const DEFAULT_EMBED_TIMEOUT: Duration = Duration::from_secs(60);

/// Upper bound on establishing a TCP connection, so a dead or unreachable
/// endpoint fails fast.
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

fn build_client(timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(timeout)
        .connect_timeout(DEFAULT_CONNECT_TIMEOUT)
        .build()
        .expect("failed to build HTTP client")
}

/// Ollama embedding model client.
pub struct OllamaEmbedding {
    client: reqwest::Client,
    model_id: String,
    base_url: String,
    dimensions: usize,
}

impl OllamaEmbedding {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            client: build_client(DEFAULT_EMBED_TIMEOUT),
            model_id: model_id.into(),
            base_url: std::env::var("OLLAMA_HOST")
                .unwrap_or_else(|_| "http://localhost:11434".into()),
            dimensions: 768,
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Sets the total request timeout (default: 60 seconds).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.client = build_client(timeout);
        self
    }

    pub fn with_dimensions(mut self, dims: usize) -> Self {
        self.dimensions = dims;
        self
    }
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

impl EmbeddingModel for OllamaEmbedding {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let body = EmbedRequest {
            model: &self.model_id,
            input: texts,
        };

        let resp = self
            .client
            .post(format!("{}/api/embed", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| DaimonError::Model(format!("Ollama embedding HTTP error: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(DaimonError::Model(format!(
                "Ollama embedding error {status}: {text}"
            )));
        }

        let data: EmbedResponse = resp
            .json()
            .await
            .map_err(|e| DaimonError::Model(format!("Ollama embedding parse error: {e}")))?;

        Ok(data.embeddings)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}
