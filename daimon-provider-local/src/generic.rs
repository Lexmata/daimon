//! Generic OpenAI-compatible model provider.
//!
//! For any locally-hosted server that speaks the OpenAI chat/embeddings API
//! but isn't one of [`crate::llamacpp`], [`crate::llamars`], or
//! [`crate::ollama`] specifically — vLLM, LM Studio, llamafile, LocalAI, and
//! similar. Unlike the other local providers, [`OpenAiCompatible::new`]
//! requires an explicit base URL: there is no sensible default across such a
//! wide range of servers and ports.
//!
//! ```ignore
//! use daimon_core::{ChatRequest, Message, Model};
//! use daimon_provider_local::generic::OpenAiCompatible;
//!
//! let model = OpenAiCompatible::new("http://localhost:8000")
//!     .with_model("my-model")
//!     .with_extra_field("repetition_penalty", serde_json::json!(1.1));
//! let response = model
//!     .generate(&ChatRequest::new(vec![Message::user("hello")]))
//!     .await?;
//! ```

use std::time::Duration;

use daimon_core::{
    ChatRequest, ChatResponse, DaimonError, EmbeddingModel, Model, ResponseStream, Result,
};

use crate::openai_compat::{
    EmbedRequest, Http, api_error, build_chat_request, parse_chat_response, parse_embed_response,
    stream_chat_response,
};

/// Generic OpenAI-compatible model provider. No default base URL — always
/// construct with [`OpenAiCompatible::new`].
#[derive(Debug)]
pub struct OpenAiCompatible {
    http: Http,
    model: Option<String>,
    extra: serde_json::Map<String, serde_json::Value>,
}

impl OpenAiCompatible {
    /// Create a client targeting the given base URL (e.g. `http://localhost:8000`).
    pub fn new(base_url: impl Into<String>) -> Self {
        let base_url = base_url.into();
        Self {
            http: Http::new(&base_url),
            model: None,
            extra: serde_json::Map::new(),
        }
    }

    /// Override the base URL set at construction.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.http.set_base_url(url);
        self
    }

    /// Set the model name sent in the request body.
    pub fn with_model(mut self, name: impl Into<String>) -> Self {
        self.model = Some(name.into());
        self
    }

    /// Set the API key / bearer token, if the server requires one.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.http.set_api_key(key);
        self
    }

    /// Set a custom timeout for HTTP requests.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.http.set_timeout(timeout);
        self
    }

    /// Set the maximum number of retries for transient (429 / 5xx) errors
    /// on the initial request (default: 3).
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.http.set_max_retries(retries);
        self
    }

    /// Opts back into warn-and-send for an API key sent over a plaintext
    /// `http://` base URL (default: hard error). Only use this for a
    /// genuinely local, unauthenticated-but-keyed server.
    pub fn allow_plaintext_api_key(mut self) -> Self {
        self.http.set_allow_plaintext_api_key(true);
        self
    }

    /// Add an arbitrary top-level field to every chat request body, for
    /// server-specific sampling parameters this generic client doesn't know
    /// about by name (e.g. vLLM's `repetition_penalty`).
    pub fn with_extra_field(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra.insert(key.into(), value);
        self
    }
}

impl Model for OpenAiCompatible {
    #[tracing::instrument(skip_all, fields(model = self.model.as_deref().unwrap_or("openai-compatible")))]
    async fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let body = build_chat_request(
            &request.messages,
            &request.tools,
            self.model.as_deref(),
            request.temperature,
            request.max_tokens,
            false,
            self.extra.clone(),
        );

        let response = self.http.post("/v1/chat/completions", &body).await?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(api_error(status, &text, "OpenAI-compatible"));
        }

        let bytes = response.bytes().await.map_err(|e| {
            DaimonError::Model(format!("OpenAI-compatible response read error: {e}"))
        })?;
        parse_chat_response(&bytes, "OpenAI-compatible")
    }

    #[tracing::instrument(skip_all, fields(model = self.model.as_deref().unwrap_or("openai-compatible")))]
    async fn generate_stream(&self, request: &ChatRequest) -> Result<ResponseStream> {
        let body = build_chat_request(
            &request.messages,
            &request.tools,
            self.model.as_deref(),
            request.temperature,
            request.max_tokens,
            true,
            self.extra.clone(),
        );

        let response = self
            .http
            .post_streaming("/v1/chat/completions", &body)
            .await?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(api_error(status, &text, "OpenAI-compatible"));
        }

        Ok(stream_chat_response(response, "OpenAI-compatible"))
    }
}

/// Generic OpenAI-compatible embedding model. No default base URL — always
/// construct with [`OpenAiCompatibleEmbedding::new`].
#[derive(Debug)]
pub struct OpenAiCompatibleEmbedding {
    http: Http,
    model: Option<String>,
    dimensions: usize,
}

impl OpenAiCompatibleEmbedding {
    /// Create a client targeting the given base URL.
    pub fn new(base_url: impl Into<String>) -> Self {
        let base_url = base_url.into();
        Self {
            http: Http::new(&base_url),
            model: None,
            dimensions: 768,
        }
    }

    /// Override the base URL set at construction.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.http.set_base_url(url);
        self
    }

    /// Set the model name sent in the request body.
    pub fn with_model(mut self, name: impl Into<String>) -> Self {
        self.model = Some(name.into());
        self
    }

    /// Set the API key / bearer token, if the server requires one.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.http.set_api_key(key);
        self
    }

    /// Set a custom timeout for HTTP requests.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.http.set_timeout(timeout);
        self
    }

    /// Set the maximum number of retries for transient (429 / 5xx) errors
    /// on the initial request (default: 3).
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.http.set_max_retries(retries);
        self
    }

    /// Opts back into warn-and-send for an API key sent over a plaintext
    /// `http://` base URL (default: hard error). Only use this for a
    /// genuinely local, unauthenticated-but-keyed server.
    pub fn allow_plaintext_api_key(mut self) -> Self {
        self.http.set_allow_plaintext_api_key(true);
        self
    }

    /// Declare the dimensionality of the embeddings this server produces.
    pub fn with_dimensions(mut self, dims: usize) -> Self {
        self.dimensions = dims;
        self
    }
}

impl EmbeddingModel for OpenAiCompatibleEmbedding {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let body = EmbedRequest {
            model: self.model.as_deref(),
            input: texts,
        };
        let resp = self.http.post("/v1/embeddings", &body).await?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(api_error(status, &text, "OpenAI-compatible"));
        }
        let bytes = resp.bytes().await.map_err(|e| {
            DaimonError::Model(format!("OpenAI-compatible embedding read error: {e}"))
        })?;
        parse_embed_response(&bytes, "OpenAI-compatible")
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
        let model = OpenAiCompatible::new("http://localhost:8000");
        assert_eq!(model.http.base_url(), "http://localhost:8000");
        assert!(model.model.is_none());
        assert!(model.extra.is_empty());
    }

    #[test]
    fn test_extra_field() {
        let model = OpenAiCompatible::new("http://localhost:8000")
            .with_extra_field("repetition_penalty", serde_json::json!(1.1));
        assert_eq!(model.extra["repetition_penalty"], 1.1);
    }

    #[test]
    fn test_builder_chain() {
        let model = OpenAiCompatible::new("http://localhost:8000")
            .with_model("my-model")
            .with_api_key("secret")
            .with_timeout(Duration::from_secs(15))
            .with_max_retries(5);
        assert_eq!(model.model.as_deref(), Some("my-model"));
        assert_eq!(model.http.api_key(), Some("secret"));
        assert_eq!(model.http.max_retries(), 5);
    }

    #[test]
    fn test_embedding_builder_defaults() {
        let embed = OpenAiCompatibleEmbedding::new("http://localhost:8000");
        assert_eq!(embed.dimensions, 768);
    }

    #[test]
    fn test_debug_redacts_api_key() {
        let model =
            OpenAiCompatible::new("http://localhost:8000").with_api_key("sk-supersecret-key-value");
        let dbg = format!("{model:?}");
        assert!(
            !dbg.contains("sk-supersecret-key-value"),
            "Debug output must not contain the plaintext API key: {dbg}"
        );
        assert!(dbg.contains("[redacted]"));
    }

    #[test]
    fn test_embedding_debug_redacts_api_key() {
        let embed = OpenAiCompatibleEmbedding::new("http://localhost:8000")
            .with_api_key("sk-supersecret-embed-key");
        let dbg = format!("{embed:?}");
        assert!(
            !dbg.contains("sk-supersecret-embed-key"),
            "Debug output must not contain the plaintext API key: {dbg}"
        );
        assert!(dbg.contains("[redacted]"));
    }

    #[tokio::test]
    async fn test_plaintext_api_key_over_http_is_blocked_by_default() {
        let model = OpenAiCompatible::new("http://localhost:8000").with_api_key("secret");
        let request = ChatRequest::new(vec![daimon_core::Message::user("hi")]);
        let err = model.generate(&request).await.unwrap_err();
        assert!(matches!(err, DaimonError::Builder(_)));
    }

    #[tokio::test]
    async fn test_plaintext_api_key_allowed_when_opted_in_does_not_hard_error() {
        // No server is listening, so this still errors — but it must be a
        // transport/model error, not the plaintext-key Builder error, proving
        // the opt-in bypassed the hard block.
        let model = OpenAiCompatible::new("http://localhost:1")
            .with_api_key("secret")
            .with_max_retries(0)
            .allow_plaintext_api_key();
        let request = ChatRequest::new(vec![daimon_core::Message::user("hi")]);
        let err = model.generate(&request).await.unwrap_err();
        assert!(!matches!(err, DaimonError::Builder(_)));
    }
}
