//! OpenAI embeddings provider (text-embedding-3-small, text-embedding-3-large, etc.).

use std::time::Duration;

use serde::{Deserialize, Serialize};

use daimon_core::{DaimonError, EmbeddingModel, Result};

use crate::retry;

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

/// OpenAI embedding model client.
pub struct OpenAiEmbedding {
    client: reqwest::Client,
    api_key: String,
    model_id: String,
    dimensions: usize,
    /// Whether the caller explicitly requested a dimensionality. Only then is
    /// the `dimensions` field sent on the wire — models that predate matryoshka
    /// truncation (e.g. text-embedding-ada-002) reject the parameter.
    dimensions_explicit: bool,
    base_url: String,
    max_retries: u32,
}

impl OpenAiEmbedding {
    /// Creates a new OpenAI embedding client.
    ///
    /// Reads `OPENAI_API_KEY` from the environment if `api_key` is not provided.
    /// Requests carry a 60-second default timeout; override with
    /// [`with_timeout`](Self::with_timeout). The constructor never fails; if
    /// the environment variable is unset or empty a warning is logged and
    /// requests will fail with an auth error.
    pub fn new(model_id: impl Into<String>) -> Self {
        let model_id = model_id.into();
        let dimensions = if model_id.contains("large") {
            3072
        } else {
            1536
        };
        let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        if api_key.is_empty() {
            tracing::warn!(
                "OPENAI_API_KEY is not set or empty; OpenAI embedding requests will fail authentication"
            );
        }
        Self {
            client: build_client(DEFAULT_EMBED_TIMEOUT),
            api_key,
            model_id,
            dimensions,
            dimensions_explicit: false,
            base_url: "https://api.openai.com/v1".to_string(),
            max_retries: DEFAULT_MAX_RETRIES,
        }
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = key.into();
        self
    }

    /// Sets the total request timeout (default: 60 seconds).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.client = build_client(timeout);
        self
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Request embeddings truncated to `dims` dimensions.
    ///
    /// The value is sent as the `dimensions` field on every request, so the
    /// returned vectors actually have this dimensionality (previously it was
    /// only reported by [`dimensions`](EmbeddingModel::dimensions) and never
    /// sent, so the API returned full-size vectors).
    pub fn with_dimensions(mut self, dims: usize) -> Self {
        self.dimensions = dims;
        self.dimensions_explicit = true;
        self
    }

    /// Set the maximum number of retries for transient errors (default: 3).
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    fn build_request_body<'a>(&'a self, texts: &'a [&'a str]) -> EmbedRequest<'a> {
        EmbedRequest {
            model: &self.model_id,
            input: texts,
            dimensions: self.dimensions_explicit.then_some(self.dimensions),
        }
    }
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
}

#[derive(Deserialize)]
struct EmbedData {
    embedding: Vec<f32>,
}

impl EmbeddingModel for OpenAiEmbedding {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let body = self.build_request_body(texts);
        let url = format!("{}/embeddings", self.base_url);

        // Same transient-error retry policy as the chat client.
        for attempt in 0..=self.max_retries {
            let resp = self
                .client
                .post(&url)
                .bearer_auth(&self.api_key)
                .json(&body)
                .send()
                .await
                .map_err(|e| DaimonError::Model(format!("OpenAI embedding HTTP error: {e}")))?;
            let status = resp.status();

            if status.is_success() {
                let data: EmbedResponse = resp.json().await.map_err(|e| {
                    DaimonError::Model(format!("OpenAI embedding parse error: {e}"))
                })?;
                return Ok(data.data.into_iter().map(|d| d.embedding).collect());
            }

            let retry_after = retry::parse_retry_after(resp.headers());
            let text = resp.text().await.unwrap_or_default();
            let is_retryable = status.as_u16() == 429 || status.is_server_error();

            if is_retryable && attempt < self.max_retries {
                let delay = retry::backoff_delay(attempt, retry_after);
                tracing::debug!(status = %status, attempt, delay_ms = delay.as_millis(), "retryable embedding error, backing off");
                tokio::time::sleep(delay).await;
            } else {
                return Err(DaimonError::Model(format!(
                    "OpenAI embedding error {status}: {text}"
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
    fn test_default_dimensions_by_model() {
        assert_eq!(
            OpenAiEmbedding::new("text-embedding-3-small").dimensions,
            1536
        );
        assert_eq!(
            OpenAiEmbedding::new("text-embedding-3-large").dimensions,
            3072
        );
    }

    #[test]
    fn test_builder_chain() {
        let embed = OpenAiEmbedding::new("text-embedding-3-small")
            .with_api_key("key")
            .with_base_url("https://custom.example.com")
            .with_dimensions(512)
            .with_timeout(Duration::from_secs(5))
            .with_max_retries(1);
        assert_eq!(embed.api_key, "key");
        assert_eq!(embed.base_url, "https://custom.example.com");
        assert_eq!(embed.dimensions, 512);
        assert_eq!(embed.max_retries, 1);
    }

    #[test]
    fn test_request_body_includes_dimensions_when_set() {
        // with_dimensions previously only changed the *reported* value; the
        // request body must carry it so the API actually truncates vectors.
        let embed = OpenAiEmbedding::new("text-embedding-3-small").with_dimensions(512);
        let texts = ["hi"];
        let body = embed.build_request_body(&texts);
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["dimensions"], 512);
    }

    #[test]
    fn test_request_body_omits_dimensions_when_not_set() {
        let embed = OpenAiEmbedding::new("text-embedding-3-small");
        let texts = ["hi"];
        let body = embed.build_request_body(&texts);
        let json = serde_json::to_value(&body).unwrap();
        assert!(
            json.get("dimensions").is_none(),
            "dimensions must be absent unless explicitly requested: {json}"
        );
    }
}
