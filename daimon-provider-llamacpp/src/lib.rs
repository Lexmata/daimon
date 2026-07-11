//! llama.cpp model provider for the Daimon agent framework.
//!
//! Talks to a running [`llama-server`](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
//! over its OpenAI-compatible `/v1/chat/completions` endpoint. The server
//! applies the model's chat template, so no client-side prompt templating is
//! done here. llama.cpp-native sampling parameters (`grammar`, `json_schema`,
//! `min_p`, `top_k`, `repeat_penalty`, `cache_prompt`) are sent as extra
//! fields alongside the OpenAI-shaped body.
//!
//! ```ignore
//! use daimon_provider_llamacpp::LlamaCpp;
//!
//! let model = LlamaCpp::new()
//!     .with_base_url("http://localhost:8080")
//!     .with_grammar("root ::= \"yes\" | \"no\"");
//! ```
//!
//! Tool calling requires the server to run with `--jinja` and a chat template
//! that supports tools; otherwise llama-server rejects the request and the
//! error body is surfaced verbatim as a [`DaimonError::Model`].

mod embedding;

pub use embedding::LlamaCppEmbedding;

use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};

use daimon_core::stream_util::LineBuffer;
use daimon_core::{
    ChatRequest, ChatResponse, DaimonError, Message, Model, ResponseStream, Result, Role,
    StopReason, StreamEvent, ToolCall, ToolSpec, Usage,
};

const DEFAULT_BASE_URL: &str = "http://localhost:8080";

/// Upper bound on establishing a TCP connection. Applied unconditionally so
/// a dead or unreachable upstream fails fast instead of blocking forever; it
/// does not bound the request itself, so long streaming generations are
/// unaffected.
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

fn build_client(timeout: Option<Duration>) -> Client {
    let mut builder = Client::builder().connect_timeout(DEFAULT_CONNECT_TIMEOUT);
    if let Some(t) = timeout {
        builder = builder.timeout(t);
    }
    builder.build().expect("failed to build HTTP client")
}

/// HTTP transport against a running llama-server.
///
/// Private on purpose: when in-process FFI inference (`llama-cpp-2`) is added
/// behind an `ffi` feature, it slots in behind the same public types without
/// touching the public API. No backend trait or enum until that day comes.
pub(crate) struct Http {
    client: Client,
    base_url: String,
    api_key: Option<String>,
    timeout: Option<Duration>,
}

impl Http {
    fn new() -> Self {
        Self {
            client: build_client(None),
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: None,
            timeout: None,
        }
    }

    fn set_base_url(&mut self, url: impl Into<String>) {
        self.base_url = url.into().trim_end_matches('/').to_string();
    }

    fn set_api_key(&mut self, key: impl Into<String>) {
        self.api_key = Some(key.into());
    }

    fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = Some(timeout);
        self.client = build_client(Some(timeout));
    }

    async fn post(&self, path: &str, body: &impl Serialize) -> Result<reqwest::Response> {
        let mut req = self
            .client
            .post(format!("{}{path}", self.base_url))
            .json(body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        req.send()
            .await
            .map_err(|e| DaimonError::Model(format!("llama.cpp HTTP error: {e}")))
    }
}

impl std::fmt::Debug for Http {
    /// Hand-written to avoid leaking the plaintext API key in logs or panic
    /// output; a derived `Debug` would print `api_key` verbatim.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Http")
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "[redacted]"))
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

/// Surfaces the llama-server error body verbatim so grammar and chat-template
/// failures are diagnosable from the error message alone.
pub(crate) fn api_error(status: reqwest::StatusCode, body: &str) -> DaimonError {
    DaimonError::Model(format!("llama.cpp API error ({status}): {body}"))
}

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
            http: Http::new(),
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

    fn build_request_body(&self, request: &ChatRequest) -> LlamaCppRequest {
        let messages: Vec<ApiMessage> = request.messages.iter().map(Into::into).collect();

        let tools: Option<Vec<ApiTool>> = if request.tools.is_empty() {
            None
        } else {
            Some(request.tools.iter().map(Into::into).collect())
        };

        LlamaCppRequest {
            model: self.model.clone(),
            messages,
            tools,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: false,
            grammar: self.grammar.clone(),
            json_schema: self.json_schema.clone(),
            min_p: self.min_p,
            top_k: self.top_k,
            repeat_penalty: self.repeat_penalty,
            cache_prompt: self.cache_prompt,
        }
    }
}

impl Model for LlamaCpp {
    #[tracing::instrument(skip_all, fields(model = self.model.as_deref().unwrap_or("llama-server")))]
    async fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let body = self.build_request_body(request);

        tracing::debug!("sending chat completion request");
        let response = self.http.post("/v1/chat/completions", &body).await?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(api_error(status, &text));
        }

        let parsed: LlamaCppResponse = response
            .json()
            .await
            .map_err(|e| DaimonError::Model(format!("llama.cpp response parse error: {e}")))?;
        parse_response(parsed)
    }

    #[tracing::instrument(skip_all, fields(model = self.model.as_deref().unwrap_or("llama-server")))]
    async fn generate_stream(&self, request: &ChatRequest) -> Result<ResponseStream> {
        let mut body = self.build_request_body(request);
        body.stream = true;

        tracing::debug!("sending streaming chat completion request");
        let response = self.http.post("/v1/chat/completions", &body).await?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(api_error(status, &text));
        }

        tracing::debug!("stream established, processing chunks");
        let byte_stream = response.bytes_stream();

        let stream = async_stream::try_stream! {
            use futures::StreamExt;

            let mut buffer = LineBuffer::new();
            let mut byte_stream = Box::pin(byte_stream);
            // Reused across the whole stream so the parse path allocates no
            // per-line Vec; drained after each line.
            let mut events: Vec<StreamEvent> = Vec::new();

            while let Some(chunk) = byte_stream.next().await {
                let chunk = chunk
                    .map_err(|e| DaimonError::Model(format!("llama.cpp stream error: {e}")))?;
                buffer.push(&chunk);

                while let Some(line) = buffer.next_line() {
                    sse_line_events_into(&line, &mut events);
                    for event in events.drain(..) {
                        yield event;
                    }
                }
            }

            // A stream may end without a trailing newline, leaving a final SSE
            // record buffered. Recover it through the identical parse path.
            if let Some(line) = buffer.take_remaining() {
                sse_line_events_into(&line, &mut events);
                for event in events.drain(..) {
                    yield event;
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

/// Test-facing wrapper around [`sse_line_events_into`] that returns the
/// produced events; the streaming loop reuses one buffer via the `_into`
/// variant instead, avoiding a fresh `Vec` allocation per SSE line.
#[cfg(test)]
fn sse_line_events(line: &str) -> Vec<StreamEvent> {
    let mut events = Vec::new();
    sse_line_events_into(line, &mut events);
    events
}

/// Parses one SSE line into the stream events it carries, appending them to
/// `events`.
///
/// Non-`data:` lines (comments, blank keep-alives) and unparseable payloads
/// yield no events, matching the other providers' tolerance for unknown
/// server chatter.
fn sse_line_events_into(line: &str, events: &mut Vec<StreamEvent>) {
    let line = line.trim();

    if line == "data: [DONE]" {
        events.push(StreamEvent::Done);
        return;
    }

    let Some(data) = line.strip_prefix("data: ") else {
        return;
    };
    let Ok(chunk) = serde_json::from_str::<StreamChunk>(data) else {
        return;
    };

    for choice in &chunk.choices {
        if let Some(content) = &choice.delta.content
            && !content.is_empty()
        {
            events.push(StreamEvent::TextDelta(content.clone()));
        }
        if let Some(tool_calls) = &choice.delta.tool_calls {
            for tc in tool_calls {
                if let Some(func) = &tc.function {
                    if let Some(name) = &func.name {
                        events.push(StreamEvent::ToolCallStart {
                            id: tc.index.to_string(),
                            name: name.clone(),
                        });
                    }
                    if let Some(args) = &func.arguments
                        && !args.is_empty()
                    {
                        events.push(StreamEvent::ToolCallDelta {
                            id: tc.index.to_string(),
                            arguments_delta: args.clone(),
                        });
                    }
                }
            }
        }
    }
}

fn parse_response(response: LlamaCppResponse) -> Result<ChatResponse> {
    let choice = response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| DaimonError::Model("no choices in llama.cpp response".into()))?;

    let tool_calls: Vec<ToolCall> = choice
        .message
        .tool_calls
        .unwrap_or_default()
        .into_iter()
        .map(|tc| ToolCall {
            id: tc.id,
            name: tc.function.name,
            arguments: serde_json::from_str(&tc.function.arguments).unwrap_or_default(),
        })
        .collect();

    let stop_reason = match choice.finish_reason.as_deref() {
        Some("tool_calls") => StopReason::ToolUse,
        Some("length") => StopReason::MaxTokens,
        _ => StopReason::EndTurn,
    };

    Ok(ChatResponse {
        message: Message {
            role: Role::Assistant,
            content: choice.message.content,
            tool_calls,
            tool_call_id: None,
        },
        stop_reason,
        usage: response.usage.map(|u| Usage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            cached_tokens: 0,
        }),
    })
}

// --- Wire types (OpenAI-shaped, plus llama.cpp-native extras) ---

#[derive(Serialize)]
struct LlamaCppRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    messages: Vec<ApiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ApiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    grammar: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    json_schema: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    repeat_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_prompt: Option<bool>,
}

#[derive(Serialize)]
struct ApiMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ApiToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

impl From<&Message> for ApiMessage {
    fn from(msg: &Message) -> Self {
        let role = match msg.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        };

        let tool_calls = if msg.tool_calls.is_empty() {
            None
        } else {
            Some(
                msg.tool_calls
                    .iter()
                    .map(|tc| ApiToolCall {
                        id: tc.id.clone(),
                        r#type: "function".to_string(),
                        function: ApiFunction {
                            name: tc.name.clone(),
                            arguments: serde_json::to_string(&tc.arguments).unwrap_or_default(),
                        },
                    })
                    .collect(),
            )
        };

        Self {
            role: role.to_string(),
            content: msg.content.clone(),
            tool_calls,
            tool_call_id: msg.tool_call_id.clone(),
        }
    }
}

#[derive(Serialize)]
struct ApiTool {
    r#type: String,
    function: ApiToolFunction,
}

impl From<&ToolSpec> for ApiTool {
    fn from(spec: &ToolSpec) -> Self {
        Self {
            r#type: "function".to_string(),
            function: ApiToolFunction {
                name: spec.name.clone(),
                description: spec.description.clone(),
                parameters: spec.parameters.clone(),
            },
        }
    }
}

#[derive(Serialize)]
struct ApiToolFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Deserialize)]
struct LlamaCppResponse {
    choices: Vec<ApiChoice>,
    usage: Option<ApiUsage>,
}

#[derive(Deserialize)]
struct ApiChoice {
    message: ApiChoiceMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ApiChoiceMessage {
    content: Option<String>,
    tool_calls: Option<Vec<ApiResponseToolCall>>,
}

#[derive(Serialize, Deserialize)]
struct ApiToolCall {
    id: String,
    r#type: String,
    function: ApiFunction,
}

#[derive(Deserialize)]
struct ApiResponseToolCall {
    #[serde(default)]
    id: String,
    #[serde(default)]
    function: ApiFunction,
}

#[derive(Serialize, Deserialize, Default)]
struct ApiFunction {
    #[serde(default)]
    name: String,
    #[serde(default)]
    arguments: String,
}

#[derive(Deserialize)]
struct ApiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Deserialize)]
struct StreamChunk {
    choices: Vec<StreamChoice>,
}

#[derive(Deserialize)]
struct StreamChoice {
    delta: StreamDelta,
}

#[derive(Deserialize)]
struct StreamDelta {
    content: Option<String>,
    tool_calls: Option<Vec<StreamToolCall>>,
}

#[derive(Deserialize)]
struct StreamToolCall {
    #[serde(default)]
    index: usize,
    function: Option<StreamFunction>,
}

#[derive(Deserialize)]
struct StreamFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_defaults() {
        let model = LlamaCpp::new();
        assert_eq!(model.http.base_url, DEFAULT_BASE_URL);
        assert!(model.http.api_key.is_none());
        assert!(model.http.timeout.is_none());
        assert!(model.model.is_none());
        assert!(model.grammar.is_none());
        assert!(model.json_schema.is_none());
        assert!(model.min_p.is_none());
        assert!(model.top_k.is_none());
        assert!(model.repeat_penalty.is_none());
        assert!(model.cache_prompt.is_none());
    }

    #[test]
    fn test_builder_chain() {
        let model = LlamaCpp::new()
            .with_base_url("http://gpu-box:8080/")
            .with_model("qwen3")
            .with_api_key("secret")
            .with_timeout(Duration::from_secs(30))
            .with_grammar("root ::= \"yes\"")
            .with_json_schema(serde_json::json!({"type": "object"}))
            .with_min_p(0.05)
            .with_top_k(40)
            .with_repeat_penalty(1.1)
            .with_cache_prompt(true);
        assert_eq!(model.http.base_url, "http://gpu-box:8080");
        assert_eq!(model.model.as_deref(), Some("qwen3"));
        assert_eq!(model.http.api_key.as_deref(), Some("secret"));
        assert_eq!(model.http.timeout, Some(Duration::from_secs(30)));
        assert_eq!(model.min_p, Some(0.05));
        assert_eq!(model.top_k, Some(40));
        assert_eq!(model.repeat_penalty, Some(1.1));
        assert_eq!(model.cache_prompt, Some(true));
    }

    #[test]
    fn test_request_body_includes_extras() {
        let model = LlamaCpp::new()
            .with_model("qwen3")
            .with_grammar("root ::= \"yes\"")
            .with_min_p(0.05)
            .with_top_k(40)
            .with_repeat_penalty(1.1)
            .with_cache_prompt(true);
        let request = ChatRequest::new(vec![Message::user("hi")]);
        let body = serde_json::to_value(model.build_request_body(&request)).unwrap();

        assert_eq!(body["model"], "qwen3");
        assert_eq!(body["grammar"], "root ::= \"yes\"");
        assert_eq!(body["min_p"], 0.05f32 as f64);
        assert_eq!(body["top_k"], 40);
        assert_eq!(body["repeat_penalty"], 1.1f32 as f64);
        assert_eq!(body["cache_prompt"], true);
        assert_eq!(body["stream"], false);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "hi");
    }

    #[test]
    fn test_request_body_omits_unset_extras() {
        let model = LlamaCpp::new();
        let request = ChatRequest::new(vec![Message::user("hi")]);
        let body = serde_json::to_value(model.build_request_body(&request)).unwrap();

        let obj = body.as_object().unwrap();
        for key in [
            "model",
            "tools",
            "temperature",
            "max_tokens",
            "grammar",
            "json_schema",
            "min_p",
            "top_k",
            "repeat_penalty",
            "cache_prompt",
        ] {
            assert!(!obj.contains_key(key), "unset '{key}' must be omitted");
        }
    }

    #[test]
    fn test_request_body_json_schema() {
        let schema = serde_json::json!({"type": "object", "properties": {"x": {"type": "number"}}});
        let model = LlamaCpp::new().with_json_schema(schema.clone());
        let request = ChatRequest::new(vec![Message::user("hi")]);
        let body = serde_json::to_value(model.build_request_body(&request)).unwrap();
        assert_eq!(body["json_schema"], schema);
    }

    #[test]
    fn test_tool_spec_conversion() {
        let spec = ToolSpec {
            name: "calc".into(),
            description: "Calculator".into(),
            parameters: serde_json::json!({"type": "object"}),
        };
        let tool: ApiTool = (&spec).into();
        assert_eq!(tool.r#type, "function");
        assert_eq!(tool.function.name, "calc");
    }

    #[test]
    fn test_message_conversion_tool_result() {
        let msg = Message::tool_result("tc_1", "42");
        let api: ApiMessage = (&msg).into();
        assert_eq!(api.role, "tool");
        assert_eq!(api.tool_call_id.as_deref(), Some("tc_1"));
        assert_eq!(api.content.as_deref(), Some("42"));
    }

    #[test]
    fn test_message_conversion_assistant_with_tools() {
        let msg = Message::assistant_with_tool_calls(vec![ToolCall {
            id: "tc_1".into(),
            name: "calc".into(),
            arguments: serde_json::json!({"x": 1}),
        }]);
        let api: ApiMessage = (&msg).into();
        assert_eq!(api.role, "assistant");
        assert_eq!(api.tool_calls.unwrap().len(), 1);
    }

    #[test]
    fn test_parse_response_end_turn() {
        let raw: LlamaCppResponse = serde_json::from_str(
            r#"{
                "choices": [{
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }"#,
        )
        .unwrap();
        let resp = parse_response(raw).unwrap();
        assert_eq!(resp.text(), "hello");
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        assert!(!resp.has_tool_calls());
        let usage = resp.usage.unwrap();
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
    }

    #[test]
    fn test_parse_response_tool_use() {
        let raw: LlamaCppResponse = serde_json::from_str(
            r#"{
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "calc", "arguments": "{\"x\":1}"}
                        }]
                    },
                    "finish_reason": "tool_calls"
                }]
            }"#,
        )
        .unwrap();
        let resp = parse_response(raw).unwrap();
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
        assert_eq!(resp.tool_calls()[0].name, "calc");
        assert_eq!(resp.tool_calls()[0].arguments, serde_json::json!({"x": 1}));
    }

    #[test]
    fn test_parse_response_max_tokens() {
        let raw: LlamaCppResponse = serde_json::from_str(
            r#"{"choices": [{"message": {"content": "trunc"}, "finish_reason": "length"}]}"#,
        )
        .unwrap();
        assert_eq!(
            parse_response(raw).unwrap().stop_reason,
            StopReason::MaxTokens
        );
    }

    #[test]
    fn test_parse_response_no_choices() {
        let raw: LlamaCppResponse = serde_json::from_str(r#"{"choices": []}"#).unwrap();
        assert!(parse_response(raw).is_err());
    }

    #[test]
    fn test_sse_text_delta() {
        let events =
            sse_line_events(r#"data: {"choices":[{"delta":{"content":"Hel"},"index":0}]}"#);
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], StreamEvent::TextDelta(t) if t == "Hel"));
    }

    #[test]
    fn test_sse_tool_call() {
        let events = sse_line_events(
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"calc","arguments":"{\"x\""}}]}}]}"#,
        );
        assert_eq!(events.len(), 2);
        assert!(matches!(
            &events[0],
            StreamEvent::ToolCallStart { id, name } if id == "0" && name == "calc"
        ));
        assert!(matches!(
            &events[1],
            StreamEvent::ToolCallDelta { id, arguments_delta } if id == "0" && arguments_delta == "{\"x\""
        ));
    }

    #[test]
    fn test_sse_done() {
        let events = sse_line_events("data: [DONE]");
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], StreamEvent::Done));
    }

    #[test]
    fn test_sse_ignores_noise() {
        assert!(sse_line_events("").is_empty());
        assert!(sse_line_events(": keep-alive").is_empty());
        assert!(sse_line_events("data: not json").is_empty());
    }

    #[test]
    fn test_api_error_surfaces_body_verbatim() {
        let err = api_error(
            reqwest::StatusCode::BAD_REQUEST,
            r#"{"error":{"message":"failed to parse grammar"}}"#,
        );
        let text = err.to_string();
        assert!(text.contains("400"), "status missing: {text}");
        assert!(
            text.contains("failed to parse grammar"),
            "body not verbatim: {text}"
        );
    }

    #[test]
    fn test_debug_redacts_api_key() {
        let model = LlamaCpp::new().with_api_key("sk-supersecret-key-value");
        let dbg = format!("{model:?}");
        assert!(
            !dbg.contains("sk-supersecret-key-value"),
            "Debug output must not contain the plaintext API key: {dbg}"
        );
        assert!(dbg.contains("[redacted]"));
    }
}
