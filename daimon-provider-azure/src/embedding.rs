//! Azure OpenAI embedding model provider.
//!
//! Uses the Azure OpenAI Embeddings API, which follows the same wire format
//! as OpenAI but with Azure-specific URL structure and authentication.

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

/// Azure OpenAI embedding model.
///
/// ```ignore
/// use daimon_provider_azure::AzureOpenAiEmbedding;
///
/// let embedding = AzureOpenAiEmbedding::new(
///     "https://my-resource.openai.azure.com",
///     "text-embedding-3-large",
/// );
/// let vectors = embedding.embed(&["hello world"]).await?;
/// ```
pub struct AzureOpenAiEmbedding {
    client: reqwest::Client,
    api_key: String,
    resource_url: String,
    deployment_id: String,
    api_version: String,
    dimensions: usize,
    /// Whether the caller explicitly requested a dimensionality. Only then is
    /// the `dimensions` field sent on the wire — models that predate matryoshka
    /// truncation (e.g. text-embedding-ada-002) reject the parameter.
    dimensions_explicit: bool,
    max_retries: u32,
    use_bearer_token: bool,
}

impl std::fmt::Debug for AzureOpenAiEmbedding {
    /// Hand-written to avoid leaking the plaintext API key in logs or panic
    /// output; a derived `Debug` would print `api_key` verbatim.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AzureOpenAiEmbedding")
            .field("client", &self.client)
            .field("api_key", &"[redacted]")
            .field("resource_url", &self.resource_url)
            .field("deployment_id", &self.deployment_id)
            .field("api_version", &self.api_version)
            .field("dimensions", &self.dimensions)
            .field("dimensions_explicit", &self.dimensions_explicit)
            .field("max_retries", &self.max_retries)
            .field("use_bearer_token", &self.use_bearer_token)
            .finish()
    }
}

impl AzureOpenAiEmbedding {
    /// Creates a new Azure OpenAI embedding client, reading
    /// `AZURE_OPENAI_API_KEY` from the environment.
    ///
    /// Requests carry a 60-second default timeout; override with
    /// [`with_timeout`](Self::with_timeout). If the environment variable is
    /// unset or empty a warning is logged and requests will fail with an auth
    /// error.
    pub fn new(resource_url: impl Into<String>, deployment_id: impl Into<String>) -> Self {
        let api_key = std::env::var("AZURE_OPENAI_API_KEY").unwrap_or_default();
        if api_key.is_empty() {
            tracing::warn!(
                "AZURE_OPENAI_API_KEY is not set or empty; Azure embedding requests will fail authentication"
            );
        }
        let deployment = deployment_id.into();
        let dimensions = if deployment.contains("large") {
            3072
        } else {
            1536
        };
        Self {
            client: build_client(DEFAULT_EMBED_TIMEOUT),
            api_key,
            resource_url: resource_url.into().trim_end_matches('/').to_string(),
            deployment_id: deployment,
            api_version: "2024-10-21".to_string(),
            dimensions,
            dimensions_explicit: false,
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

    pub fn with_api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = version.into();
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

    /// Use `Authorization: Bearer <token>` instead of `api-key` header.
    pub fn with_bearer_token(mut self) -> Self {
        self.use_bearer_token = true;
        self
    }

    fn apply_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if self.use_bearer_token {
            req.bearer_auth(&self.api_key)
        } else {
            req.header("api-key", &self.api_key)
        }
    }

    fn build_request_body<'a>(&self, texts: &'a [&'a str]) -> EmbedRequest<'a> {
        EmbedRequest {
            input: texts,
            dimensions: self.dimensions_explicit.then_some(self.dimensions),
        }
    }
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
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

impl EmbeddingModel for AzureOpenAiEmbedding {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let body = self.build_request_body(texts);
        let url = format!(
            "{}/openai/deployments/{}/embeddings",
            self.resource_url, self.deployment_id
        );

        // Same transient-error retry policy as the chat client.
        for attempt in 0..=self.max_retries {
            let req = self
                .client
                .post(&url)
                .query(&[("api-version", &self.api_version)])
                .json(&body);
            let req = self.apply_auth(req);

            let resp = req
                .send()
                .await
                .map_err(|e| DaimonError::Model(format!("Azure embedding HTTP error: {e}")))?;
            let status = resp.status();

            if status.is_success() {
                let data: EmbedResponse = resp
                    .json()
                    .await
                    .map_err(|e| DaimonError::Model(format!("Azure embedding parse error: {e}")))?;
                return Ok(data.data.into_iter().map(|d| d.embedding).collect());
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
                    "Azure embedding error {status}: {text}"
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
    fn test_azure_embedding_new() {
        let embed = AzureOpenAiEmbedding::new(
            "https://my-resource.openai.azure.com",
            "text-embedding-3-small",
        );
        assert_eq!(embed.deployment_id, "text-embedding-3-small");
        assert_eq!(embed.dimensions, 1536);
        assert_eq!(embed.max_retries, DEFAULT_MAX_RETRIES);
    }

    #[test]
    fn test_large_model_dimensions() {
        let embed =
            AzureOpenAiEmbedding::new("https://x.openai.azure.com", "text-embedding-3-large");
        assert_eq!(embed.dimensions, 3072);
    }

    #[test]
    fn test_builder_chain() {
        let embed =
            AzureOpenAiEmbedding::new("https://x.openai.azure.com", "text-embedding-3-small")
                .with_api_key("key")
                .with_api_version("2025-01-01")
                .with_dimensions(512)
                .with_timeout(Duration::from_secs(5))
                .with_max_retries(1)
                .with_bearer_token();
        assert_eq!(embed.api_key, "key");
        assert_eq!(embed.api_version, "2025-01-01");
        assert_eq!(embed.dimensions, 512);
        assert_eq!(embed.max_retries, 1);
        assert!(embed.use_bearer_token);
    }

    #[test]
    fn test_request_body_includes_dimensions_when_set() {
        // with_dimensions previously only changed the *reported* value; the
        // request body must carry it so the API actually truncates vectors.
        let embed =
            AzureOpenAiEmbedding::new("https://x.openai.azure.com", "text-embedding-3-small")
                .with_dimensions(512);
        let body = embed.build_request_body(&["hi"]);
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["dimensions"], 512);
    }

    #[test]
    fn test_request_body_omits_dimensions_when_not_set() {
        let embed =
            AzureOpenAiEmbedding::new("https://x.openai.azure.com", "text-embedding-3-small");
        let body = embed.build_request_body(&["hi"]);
        let json = serde_json::to_value(&body).unwrap();
        assert!(
            json.get("dimensions").is_none(),
            "dimensions must be absent unless explicitly requested: {json}"
        );
    }

    #[test]
    fn test_debug_redacts_api_key() {
        let embed =
            AzureOpenAiEmbedding::new("https://x.openai.azure.com", "text-embedding-3-small")
                .with_api_key("azure-embed-supersecret-key");
        let dbg = format!("{embed:?}");
        assert!(
            !dbg.contains("azure-embed-supersecret-key"),
            "Debug output must not contain the plaintext API key: {dbg}"
        );
        assert!(dbg.contains("[redacted]"));
    }
}
