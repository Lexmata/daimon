//! Ollama embeddings provider.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use daimon_core::stream_util::{backoff_delay, parse_retry_after_secs};
use daimon_core::{DaimonError, EmbeddingModel, Result};

/// Default total request timeout. Embedding calls are non-streaming and
/// bounded, so a hung endpoint now fails after a minute instead of stalling
/// RAG ingest or retrieval forever; override with `with_timeout`.
const DEFAULT_EMBED_TIMEOUT: Duration = Duration::from_secs(60);

/// Upper bound on establishing a TCP connection, so a dead or unreachable
/// endpoint fails fast.
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

const DEFAULT_MAX_RETRIES: u32 = 3;

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
    max_retries: u32,
}

impl OllamaEmbedding {
    /// Creates a new Ollama embedding client.
    ///
    /// Reads the server URL from `OLLAMA_HOST` (default
    /// `http://localhost:11434`). Requests carry a 60-second default timeout;
    /// override with [`with_timeout`](Self::with_timeout).
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            client: build_client(DEFAULT_EMBED_TIMEOUT),
            model_id: model_id.into(),
            base_url: std::env::var("OLLAMA_HOST")
                .unwrap_or_else(|_| "http://localhost:11434".into()),
            dimensions: 768,
            max_retries: DEFAULT_MAX_RETRIES,
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

    /// Set the maximum number of retries for transient errors (default: 3).
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
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
        let url = format!("{}/api/embed", self.base_url);

        // Same transient-error retry policy as the chat providers.
        for attempt in 0..=self.max_retries {
            let resp = self
                .client
                .post(&url)
                .json(&body)
                .send()
                .await
                .map_err(|e| DaimonError::Model(format!("Ollama embedding HTTP error: {e}")))?;
            let status = resp.status();

            if status.is_success() {
                let data: EmbedResponse = resp.json().await.map_err(|e| {
                    DaimonError::Model(format!("Ollama embedding parse error: {e}"))
                })?;
                return Ok(data.embeddings);
            }

            let retry_after = resp
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .and_then(parse_retry_after_secs)
                .map(Duration::from_secs);
            let text = resp.text().await.unwrap_or_default();
            let is_retryable = status.as_u16() == 429 || status.is_server_error();

            if is_retryable && attempt < self.max_retries {
                let delay = backoff_delay(attempt, retry_after);
                tracing::debug!(status = %status, attempt, delay_ms = delay.as_millis(), "retryable embedding error, backing off");
                tokio::time::sleep(delay).await;
            } else {
                return Err(DaimonError::Model(format!(
                    "Ollama embedding error {status}: {text}"
                )));
            }
        }

        unreachable!("loop always returns or retries")
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_embedding_defaults() {
        let embed = OllamaEmbedding::new("nomic-embed-text");
        assert_eq!(embed.model_id, "nomic-embed-text");
        assert_eq!(embed.dimensions, 768);
        assert_eq!(embed.max_retries, DEFAULT_MAX_RETRIES);
    }

    #[test]
    fn test_builder_chain() {
        let embed = OllamaEmbedding::new("nomic-embed-text")
            .with_base_url("http://remote:11434")
            .with_dimensions(1024)
            .with_timeout(Duration::from_secs(5))
            .with_max_retries(1);
        assert_eq!(embed.base_url, "http://remote:11434");
        assert_eq!(embed.dimensions, 1024);
        assert_eq!(embed.max_retries, 1);
    }
}
