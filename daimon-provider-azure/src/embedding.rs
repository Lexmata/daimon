//! Azure OpenAI embedding model provider.
//!
//! Uses the Azure OpenAI Embeddings API, which follows the same wire format
//! as OpenAI but with Azure-specific URL structure and authentication.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use daimon_core::{DaimonError, EmbeddingModel, Result};

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
    use_bearer_token: bool,
}

impl AzureOpenAiEmbedding {
    /// Creates a new Azure OpenAI embedding client, reading
    /// `AZURE_OPENAI_API_KEY` from the environment.
    pub fn new(resource_url: impl Into<String>, deployment_id: impl Into<String>) -> Self {
        let api_key = std::env::var("AZURE_OPENAI_API_KEY").unwrap_or_default();
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

    pub fn with_dimensions(mut self, dims: usize) -> Self {
        self.dimensions = dims;
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
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    input: &'a [&'a str],
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
        let body = EmbedRequest { input: texts };
        let url = format!(
            "{}/openai/deployments/{}/embeddings",
            self.resource_url, self.deployment_id
        );

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

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(DaimonError::Model(format!(
                "Azure embedding error {status}: {text}"
            )));
        }

        let data: EmbedResponse = resp
            .json()
            .await
            .map_err(|e| DaimonError::Model(format!("Azure embedding parse error: {e}")))?;

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
    fn test_azure_embedding_new() {
        let embed = AzureOpenAiEmbedding::new(
            "https://my-resource.openai.azure.com",
            "text-embedding-3-small",
        );
        assert_eq!(embed.deployment_id, "text-embedding-3-small");
        assert_eq!(embed.dimensions, 1536);
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
                .with_bearer_token();
        assert_eq!(embed.api_key, "key");
        assert_eq!(embed.api_version, "2025-01-01");
        assert_eq!(embed.dimensions, 512);
        assert!(embed.use_bearer_token);
    }
}
