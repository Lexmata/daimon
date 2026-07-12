//! llama.cpp model provider.
//!
//! Talks to a running [`llama-server`](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
//! over its OpenAI-compatible `/v1/chat/completions` endpoint. The server
//! applies the model's chat template, so no client-side prompt templating is
//! done here. llama.cpp-native sampling parameters (`grammar`, `json_schema`,
//! `min_p`, `top_k`, `repeat_penalty`, `cache_prompt`) are sent as extra
//! fields alongside the OpenAI-shaped body.
//!
//! ```ignore
//! use daimon_provider_local::llamacpp::LlamaCpp;
//!
//! let model = LlamaCpp::new()
//!     .with_base_url("http://localhost:8080")
//!     .with_grammar("root ::= \"yes\" | \"no\"");
//! ```
//!
//! Tool calling requires the server to run with `--jinja` and a chat template
//! that supports tools; otherwise llama-server rejects the request and the
//! error body is surfaced verbatim as a `DaimonError::Model`.

use std::time::Duration;

use daimon_core::{ChatRequest, ChatResponse, DaimonError, Model, ResponseStream, Result};

use crate::openai_compat::{
    Http, api_error, build_chat_request, parse_chat_response, stream_chat_response,
};

pub use crate::llamacpp_embed::LlamaCppEmbedding;

const DEFAULT_BASE_URL: &str = "http://localhost:8080";

/// llama.cpp model provider, backed by a running `llama-server`.
///
/// `new()` targets `http://localhost:8080`. All configuration is via builder
/// setters; llama.cpp-native sampling extras are sent alongside the
/// OpenAI-shaped request body.
#[derive(Debug)]
pub struct LlamaCpp {
    http: Http,
    model: Option<String>,
    grammar: Option<String>,
    json_schema: Option<serde_json::Value>,
    min_p: Option<f32>,
    top_k: Option<u32>,
    repeat_penalty: Option<f32>,
    cache_prompt: Option<bool>,
}

impl Default for LlamaCpp {
    fn default() -> Self {
        Self::new()
    }
}

impl LlamaCpp {
    /// Create a client targeting `http://localhost:8080`.
    pub fn new() -> Self {
        Self {
            http: Http::new(DEFAULT_BASE_URL),
            model: None,
            grammar: None,
            json_schema: None,
            min_p: None,
            top_k: None,
            repeat_penalty: None,
            cache_prompt: None,
        }
    }

    /// Set the server base URL (e.g. `http://gpu-box:8080`).
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.http.set_base_url(url);
        self
    }

    /// Set the model name sent in the request body.
    ///
    /// Only meaningful for multi-model routers; llama-server ignores it.
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

    /// Set the maximum number of retries for transient (429 / 5xx) errors
    /// on the initial request (default: 3).
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.http.set_max_retries(retries);
        self
    }

    /// Constrain sampling with a [GBNF grammar](https://github.com/ggml-org/llama.cpp/tree/master/grammars).
    pub fn with_grammar(mut self, gbnf: impl Into<String>) -> Self {
        self.grammar = Some(gbnf.into());
        self
    }

    /// Constrain output to match a JSON Schema (converted to a grammar server-side).
    pub fn with_json_schema(mut self, schema: serde_json::Value) -> Self {
        self.json_schema = Some(schema);
        self
    }

    /// Set min-p sampling (drop tokens below `p * max_prob`).
    pub fn with_min_p(mut self, min_p: f32) -> Self {
        self.min_p = Some(min_p);
        self
    }

    /// Set top-k sampling.
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Set the repetition penalty.
    pub fn with_repeat_penalty(mut self, penalty: f32) -> Self {
        self.repeat_penalty = Some(penalty);
        self
    }

    /// Enable or disable server-side prompt caching across requests.
    pub fn with_cache_prompt(mut self, enabled: bool) -> Self {
        self.cache_prompt = Some(enabled);
        self
    }

    fn extra_fields(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut extra = serde_json::Map::new();
        if let Some(g) = &self.grammar {
            extra.insert("grammar".to_string(), serde_json::Value::String(g.clone()));
        }
        if let Some(s) = &self.json_schema {
            extra.insert("json_schema".to_string(), s.clone());
        }
        if let Some(p) = self.min_p {
            extra.insert("min_p".to_string(), serde_json::json!(p));
        }
        if let Some(k) = self.top_k {
            extra.insert("top_k".to_string(), serde_json::json!(k));
        }
        if let Some(p) = self.repeat_penalty {
            extra.insert("repeat_penalty".to_string(), serde_json::json!(p));
        }
        if let Some(c) = self.cache_prompt {
            extra.insert("cache_prompt".to_string(), serde_json::json!(c));
        }
        extra
    }
}

impl Model for LlamaCpp {
    #[tracing::instrument(skip_all, fields(model = self.model.as_deref().unwrap_or("llama-server")))]
    async fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let body = build_chat_request(
            &request.messages,
            &request.tools,
            self.model.as_deref(),
            request.temperature,
            request.max_tokens,
            false,
            self.extra_fields(),
        );

        tracing::debug!("sending chat completion request");
        let response = self.http.post("/v1/chat/completions", &body).await?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(api_error(status, &text, "llama.cpp"));
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| DaimonError::Model(format!("llama.cpp response read error: {e}")))?;
        parse_chat_response(&bytes, "llama.cpp")
    }

    #[tracing::instrument(skip_all, fields(model = self.model.as_deref().unwrap_or("llama-server")))]
    async fn generate_stream(&self, request: &ChatRequest) -> Result<ResponseStream> {
        let body = build_chat_request(
            &request.messages,
            &request.tools,
            self.model.as_deref(),
            request.temperature,
            request.max_tokens,
            true,
            self.extra_fields(),
        );

        tracing::debug!("sending streaming chat completion request");
        let response = self
            .http
            .post_streaming("/v1/chat/completions", &body)
            .await?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(api_error(status, &text, "llama.cpp"));
        }

        tracing::debug!("stream established, processing chunks");
        Ok(stream_chat_response(response, "llama.cpp"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daimon_core::Message;

    #[test]
    fn test_builder_defaults() {
        let model = LlamaCpp::new();
        assert_eq!(model.http.base_url(), DEFAULT_BASE_URL);
        assert!(model.http.api_key().is_none());
        assert!(model.http.timeout().is_none());
        assert!(model.model.is_none());
        assert!(model.grammar.is_none());
    }

    #[test]
    fn test_builder_chain() {
        let model = LlamaCpp::new()
            .with_base_url("http://gpu-box:8080/")
            .with_model("qwen3")
            .with_api_key("secret")
            .with_timeout(Duration::from_secs(30))
            .with_max_retries(5)
            .with_grammar("root ::= \"yes\"")
            .with_min_p(0.05)
            .with_top_k(40)
            .with_repeat_penalty(1.1)
            .with_cache_prompt(true);
        assert_eq!(model.http.base_url(), "http://gpu-box:8080");
        assert_eq!(model.model.as_deref(), Some("qwen3"));
        assert_eq!(model.http.api_key(), Some("secret"));
        assert_eq!(model.http.timeout(), Some(Duration::from_secs(30)));
        assert_eq!(model.http.max_retries(), 5);
    }

    #[test]
    fn test_extra_fields_includes_all_extras() {
        let model = LlamaCpp::new()
            .with_grammar("root ::= \"yes\"")
            .with_min_p(0.05)
            .with_top_k(40)
            .with_repeat_penalty(1.1)
            .with_cache_prompt(true)
            .with_json_schema(serde_json::json!({"type": "object"}));
        let extra = model.extra_fields();
        assert_eq!(extra["grammar"], "root ::= \"yes\"");
        assert_eq!(extra["min_p"], 0.05f32 as f64);
        assert_eq!(extra["top_k"], 40);
        assert_eq!(extra["repeat_penalty"], 1.1f32 as f64);
        assert_eq!(extra["cache_prompt"], true);
        assert_eq!(extra["json_schema"], serde_json::json!({"type": "object"}));
    }

    #[test]
    fn test_extra_fields_empty_when_unset() {
        let model = LlamaCpp::new();
        assert!(model.extra_fields().is_empty());
    }

    #[test]
    fn test_request_body_includes_extras() {
        let model = LlamaCpp::new()
            .with_model("qwen3")
            .with_grammar("root ::= \"yes\"");
        let request = ChatRequest::new(vec![Message::user("hi")]);
        let body = build_chat_request(
            &request.messages,
            &request.tools,
            model.model.as_deref(),
            request.temperature,
            request.max_tokens,
            false,
            model.extra_fields(),
        );
        let value = serde_json::to_value(&body).unwrap();
        assert_eq!(value["model"], "qwen3");
        assert_eq!(value["grammar"], "root ::= \"yes\"");
        assert_eq!(value["messages"][0]["content"], "hi");
    }

    #[test]
    fn test_debug_redacts_api_key() {
        let model = LlamaCpp::new().with_api_key("sk-supersecret-key-value");
        let dbg = format!("{model:?}");
        assert!(!dbg.contains("sk-supersecret-key-value"));
        assert!(dbg.contains("[redacted]"));
    }

    #[test]
    fn test_llamacpp_embedding_reachable_from_llamacpp_module() {
        // Regression test: LlamaCppEmbedding must be reachable via
        // `crate::llamacpp::LlamaCppEmbedding` (not just the crate root), because
        // the facade's `daimon::model::llamacpp` feature module re-exports
        // `daimon_provider_local::llamacpp::*` specifically, not the crate root.
        // This path broke once already when llamacpp.rs and llamacpp_embed.rs were
        // split into separate files — this test pins it.
        fn _assert_reachable() -> super::LlamaCppEmbedding {
            super::LlamaCppEmbedding::new()
        }
    }
}
