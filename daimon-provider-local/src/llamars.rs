//! [llama-rs](https://github.com/Lexmata/llama-rs) model provider.
//!
//! Talks to a running llama-rs server over its OpenAI-compatible
//! `/v1/chat/completions` and `/v1/embeddings` endpoints. llama-rs's request
//! shape is vanilla OpenAI (`model`, `messages`, `tools`, `tool_choice`,
//! `temperature`, `top_p`, `max_tokens`, `stream`, `stop`,
//! `frequency_penalty`, `presence_penalty`) — no provider-native sampling
//! extras beyond that today.
//!
//! ```ignore
//! use daimon_provider_local::llamars::LlamaRs;
//!
//! let model = LlamaRs::new().with_base_url("http://localhost:8080");
//! ```

use std::time::Duration;

use daimon_core::{
    ChatRequest, ChatResponse, DaimonError, EmbeddingModel, Model, ResponseStream, Result,
};

use crate::openai_compat::{
    EmbedRequest, Http, api_error, build_chat_request, parse_chat_response, parse_embed_response,
    stream_chat_response,
};

const DEFAULT_BASE_URL: &str = "http://localhost:8080";

/// llama-rs model provider, backed by a running llama-rs server.
///
/// `new()` targets `http://localhost:8080`, matching llama-rs's own default
/// bind address.
#[derive(Debug)]
pub struct LlamaRs {
    http: Http,
    model: Option<String>,
}

impl Default for LlamaRs {
    fn default() -> Self {
        Self::new()
    }
}

impl LlamaRs {
    /// Create a client targeting `http://localhost:8080`.
    pub fn new() -> Self {
        Self {
            http: Http::new(DEFAULT_BASE_URL),
            model: None,
        }
    }

    /// Set the server base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.http.set_base_url(url);
        self
    }

    /// Set the model name sent in the request body.
    pub fn with_model(mut self, name: impl Into<String>) -> Self {
        self.model = Some(name.into());
        self
    }

    /// Set the API key, if the server was started with one configured.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.http.set_api_key(key);
        self
    }

    /// Set a custom timeout for HTTP requests.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.http.set_timeout(timeout);
        self
    }
}

impl Model for LlamaRs {
    #[tracing::instrument(skip_all, fields(model = self.model.as_deref().unwrap_or("llama-rs")))]
    async fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let body = build_chat_request(
            &request.messages,
            &request.tools,
            self.model.as_deref(),
            request.temperature,
            request.max_tokens,
            false,
            Default::default(),
        );

        let response = self.http.post("/v1/chat/completions", &body).await?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(api_error(status, &text, "llama-rs"));
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| DaimonError::Model(format!("llama-rs response read error: {e}")))?;
        parse_chat_response(&bytes, "llama-rs")
    }

    #[tracing::instrument(skip_all, fields(model = self.model.as_deref().unwrap_or("llama-rs")))]
    async fn generate_stream(&self, request: &ChatRequest) -> Result<ResponseStream> {
        let body = build_chat_request(
            &request.messages,
            &request.tools,
            self.model.as_deref(),
            request.temperature,
            request.max_tokens,
            true,
            Default::default(),
        );

        let response = self.http.post("/v1/chat/completions", &body).await?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(api_error(status, &text, "llama-rs"));
        }

        Ok(stream_chat_response(response, "llama-rs"))
    }
}

/// llama-rs embedding model, backed by a running llama-rs server's
/// `/v1/embeddings` endpoint.
#[derive(Debug)]
pub struct LlamaRsEmbedding {
    http: Http,
    model: Option<String>,
    dimensions: usize,
}

impl Default for LlamaRsEmbedding {
    fn default() -> Self {
        Self::new()
    }
}

impl LlamaRsEmbedding {
    /// Create a client targeting `http://localhost:8080`.
    pub fn new() -> Self {
        Self {
            http: Http::new(DEFAULT_BASE_URL),
            model: None,
            dimensions: 768,
        }
    }

    /// Set the server base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.http.set_base_url(url);
        self
    }

    /// Set the model name sent in the request body.
    pub fn with_model(mut self, name: impl Into<String>) -> Self {
        self.model = Some(name.into());
        self
    }

    /// Set the API key, if the server was started with one configured.
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
    pub fn with_dimensions(mut self, dims: usize) -> Self {
        self.dimensions = dims;
        self
    }
}

impl EmbeddingModel for LlamaRsEmbedding {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let body = EmbedRequest {
            model: self.model.as_deref(),
            input: texts,
        };
        let resp = self.http.post("/v1/embeddings", &body).await?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(api_error(status, &text, "llama-rs"));
        }
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| DaimonError::Model(format!("llama-rs embedding read error: {e}")))?;
        parse_embed_response(&bytes, "llama-rs")
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
        let model = LlamaRs::new();
        assert_eq!(model.http.base_url(), DEFAULT_BASE_URL);
        assert!(model.model.is_none());
    }

    #[test]
    fn test_builder_chain() {
        let model = LlamaRs::new()
            .with_base_url("http://gpu-box:8080")
            .with_model("my-model")
            .with_api_key("secret")
            .with_timeout(Duration::from_secs(20));
        assert_eq!(model.model.as_deref(), Some("my-model"));
        assert_eq!(model.http.api_key(), Some("secret"));
    }

    #[test]
    fn test_embedding_builder_defaults() {
        let embed = LlamaRsEmbedding::new();
        assert_eq!(embed.dimensions, 768);
    }

    #[test]
    fn test_embedding_builder_chain() {
        let embed = LlamaRsEmbedding::new()
            .with_model("embed-model")
            .with_dimensions(4096);
        assert_eq!(embed.model.as_deref(), Some("embed-model"));
        assert_eq!(embed.dimensions, 4096);
    }

    #[test]
    fn test_debug_redacts_api_key() {
        let model = LlamaRs::new().with_api_key("sk-supersecret-key-value");
        let dbg = format!("{model:?}");
        assert!(
            !dbg.contains("sk-supersecret-key-value"),
            "Debug output must not contain the plaintext API key: {dbg}"
        );
        assert!(dbg.contains("[redacted]"));
    }

    #[test]
    fn test_embedding_debug_redacts_api_key() {
        let embed = LlamaRsEmbedding::new().with_api_key("sk-supersecret-embed-key");
        let dbg = format!("{embed:?}");
        assert!(
            !dbg.contains("sk-supersecret-embed-key"),
            "Debug output must not contain the plaintext API key: {dbg}"
        );
        assert!(dbg.contains("[redacted]"));
    }
}
