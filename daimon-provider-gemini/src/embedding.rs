//! Google Gemini embedding model provider.
//!
//! Uses the `embedContent` / `batchEmbedContents` API endpoints.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use daimon_core::{DaimonError, EmbeddingModel, Result};

use crate::stream_util;

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

/// Google Gemini embedding model.
///
/// ```ignore
/// use daimon_provider_gemini::GeminiEmbedding;
///
/// let embedding = GeminiEmbedding::new("text-embedding-004");
/// let vectors = embedding.embed(&["hello world"]).await?;
/// ```
pub struct GeminiEmbedding {
    client: reqwest::Client,
    api_key: String,
    model_id: String,
    base_url: String,
    dimensions: usize,
    max_retries: u32,
    use_bearer_token: bool,
}

impl GeminiEmbedding {
    /// Creates a new Gemini embedding client, reading `GOOGLE_API_KEY` from env.
    ///
    /// Requests carry a 60-second default timeout; override with
    /// [`with_timeout`](Self::with_timeout). If the environment variable is
    /// unset or empty a warning is logged and requests will fail with an auth
    /// error.
    pub fn new(model_id: impl Into<String>) -> Self {
        let api_key = std::env::var("GOOGLE_API_KEY").unwrap_or_default();
        if api_key.is_empty() {
            tracing::warn!(
                "GOOGLE_API_KEY is not set or empty; Gemini embedding requests will fail authentication"
            );
        }
        Self {
            client: build_client(DEFAULT_EMBED_TIMEOUT),
            api_key,
            model_id: model_id.into(),
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            dimensions: 768,
            max_retries: DEFAULT_MAX_RETRIES,
            use_bearer_token: false,
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

    pub fn with_dimensions(mut self, dims: usize) -> Self {
        self.dimensions = dims;
        self
    }

    /// Set the maximum number of retries for transient errors (default: 3).
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Use `Authorization: Bearer <key>` instead of the `x-goog-api-key` header.
    pub fn with_bearer_token(mut self) -> Self {
        self.use_bearer_token = true;
        self
    }

    /// Attaches credentials to a request.
    ///
    /// The API key rides in the `x-goog-api-key` header rather than a `?key=`
    /// query parameter, which would leak into logs via reqwest error messages
    /// that include the full URL.
    fn apply_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if self.use_bearer_token {
            req.bearer_auth(&self.api_key)
        } else {
            req.header("x-goog-api-key", &self.api_key)
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BatchEmbedRequest {
    requests: Vec<EmbedContentRequest>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct EmbedContentRequest {
    model: String,
    content: EmbedContent,
    output_dimensionality: Option<usize>,
}

#[derive(Serialize)]
struct EmbedContent {
    parts: Vec<EmbedPart>,
}

#[derive(Serialize)]
struct EmbedPart {
    text: String,
}

#[derive(Deserialize)]
struct BatchEmbedResponse {
    embeddings: Vec<EmbedValues>,
}

#[derive(Deserialize)]
struct EmbedValues {
    values: Vec<f32>,
}

impl EmbeddingModel for GeminiEmbedding {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let requests: Vec<EmbedContentRequest> = texts
            .iter()
            .map(|text| EmbedContentRequest {
                model: format!("models/{}", self.model_id),
                content: EmbedContent {
                    parts: vec![EmbedPart {
                        text: text.to_string(),
                    }],
                },
                output_dimensionality: Some(self.dimensions),
            })
            .collect();

        let body = BatchEmbedRequest { requests };
        let url = format!(
            "{}/models/{}:batchEmbedContents",
            self.base_url, self.model_id
        );

        // Same transient-error retry policy as the chat client.
        for attempt in 0..=self.max_retries {
            let req = self.client.post(&url).json(&body);
            let req = self.apply_auth(req);

            let resp = req
                .send()
                .await
                .map_err(|e| DaimonError::Model(format!("Gemini embedding HTTP error: {e}")))?;
            let status = resp.status();

            if status.is_success() {
                let data: BatchEmbedResponse = resp.json().await.map_err(|e| {
                    DaimonError::Model(format!("Gemini embedding parse error: {e}"))
                })?;
                return Ok(data.embeddings.into_iter().map(|e| e.values).collect());
            }

            let retry_after = stream_util::parse_retry_after(resp.headers());
            let text = resp.text().await.unwrap_or_default();
            let is_retryable = status.as_u16() == 429 || status.is_server_error();

            if is_retryable && attempt < self.max_retries {
                let delay = stream_util::backoff_delay(attempt, retry_after);
                tracing::debug!(status = %status, attempt, delay_ms = delay.as_millis(), "retryable embedding error, backing off");
                tokio::time::sleep(delay).await;
            } else {
                return Err(DaimonError::Model(format!(
                    "Gemini embedding error {status}: {text}"
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
    fn test_gemini_embedding_new() {
        let embed = GeminiEmbedding::new("text-embedding-004");
        assert_eq!(embed.model_id, "text-embedding-004");
        assert_eq!(embed.dimensions, 768);
        assert_eq!(embed.max_retries, DEFAULT_MAX_RETRIES);
    }

    #[test]
    fn test_builder_chain() {
        let embed = GeminiEmbedding::new("text-embedding-004")
            .with_api_key("key")
            .with_base_url("https://custom.example.com")
            .with_dimensions(256)
            .with_timeout(Duration::from_secs(5))
            .with_max_retries(1)
            .with_bearer_token();
        assert_eq!(embed.api_key, "key");
        assert_eq!(embed.base_url, "https://custom.example.com");
        assert_eq!(embed.dimensions, 256);
        assert_eq!(embed.max_retries, 1);
        assert!(embed.use_bearer_token);
    }

    #[test]
    fn test_api_key_sent_as_header_not_query_param() {
        let embed = GeminiEmbedding::new("text-embedding-004").with_api_key("AIza-embed-secret");
        let req = embed
            .apply_auth(embed.client.post("https://example.com/v1"))
            .build()
            .unwrap();
        assert!(
            !req.url().as_str().contains("AIza-embed-secret"),
            "API key must not appear in the request URL: {}",
            req.url()
        );
        assert_eq!(
            req.headers().get("x-goog-api-key").unwrap(),
            "AIza-embed-secret"
        );
    }
}
