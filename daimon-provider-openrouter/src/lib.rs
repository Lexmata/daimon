//! OpenRouter model provider for the [Daimon](https://docs.rs/daimon) agent framework.
//!
//! This crate provides an implementation of the [`Model`] trait that connects
//! to the [OpenRouter](https://openrouter.ai) Chat Completions API — an
//! OpenAI-compatible gateway that routes to hundreds of models (OpenAI,
//! Anthropic, Google, Meta, and more) behind a single API key. Model ids use
//! OpenRouter's `vendor/model` form, e.g. `openai/gpt-4o` or
//! `anthropic/claude-sonnet-4`.
//!
//! It supports configurable timeouts, retries with exponential backoff,
//! response format constraints, parallel tool calls, and the optional
//! `HTTP-Referer` / `X-Title` attribution headers OpenRouter uses for its
//! public rankings.
//!
//! # Example
//!
//! ```ignore
//! use daimon_provider_openrouter::OpenRouter;
//! use daimon_core::Model;
//!
//! // Reads OPENROUTER_API_KEY from the environment.
//! let model = OpenRouter::new("openai/gpt-4o")
//!     .with_app_name("my-app");
//! ```

use std::collections::HashMap;
use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};

mod retry;

use daimon_core::{
    ChatRequest, ChatResponse, DaimonError, Message, Model, ResponseStream, Result, Role,
    StopReason, StreamEvent, ToolCall, ToolSpec, Usage,
};

const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";

pub const DEFAULT_MAX_RETRIES: u32 = 3;

/// Default whole-request timeout applied to non-streaming `generate` calls.
///
/// Long completions can legitimately take minutes — and OpenRouter adds
/// routing latency on top of the upstream provider — so this is deliberately
/// generous. Override with [`OpenRouter::with_timeout`].
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// Default TCP/TLS connect timeout applied to the underlying HTTP client.
///
/// This bounds only connection establishment, so it is safe for the
/// long-lived SSE streams produced by `generate_stream` (a whole-request
/// timeout would kill a healthy stream mid-response).
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

fn build_client() -> Client {
    Client::builder()
        .connect_timeout(DEFAULT_CONNECT_TIMEOUT)
        .build()
        .expect("failed to build HTTP client")
}

/// OpenRouter model provider for the Chat Completions API.
///
/// Supports configurable timeouts, retries, response format, parallel tool
/// calls, and OpenRouter's attribution headers. Use `new()` or
/// `with_api_key()` to create, then chain builder methods as needed.
pub struct OpenRouter {
    client: Client,
    api_key: String,
    model_id: String,
    base_url: String,
    timeout: Option<Duration>,
    max_retries: u32,
    response_format: Option<String>,
    parallel_tool_calls: Option<bool>,
    site_url: Option<String>,
    app_name: Option<String>,
}

impl std::fmt::Debug for OpenRouter {
    /// Hand-written to avoid leaking the plaintext API key in logs or panic
    /// output; a derived `Debug` would print `api_key` verbatim.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenRouter")
            .field("client", &self.client)
            .field("api_key", &"[redacted]")
            .field("model_id", &self.model_id)
            .field("base_url", &self.base_url)
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .field("response_format", &self.response_format)
            .field("parallel_tool_calls", &self.parallel_tool_calls)
            .field("site_url", &self.site_url)
            .field("app_name", &self.app_name)
            .finish()
    }
}

impl OpenRouter {
    /// Create a new OpenRouter client for the given model id.
    ///
    /// Reads `OPENROUTER_API_KEY` from the environment. Use `with_api_key()`
    /// to provide the key explicitly. The constructor never fails; if the
    /// environment variable is unset or empty a warning is logged and
    /// requests will fail with an auth error.
    pub fn new(model_id: impl Into<String>) -> Self {
        let api_key = std::env::var("OPENROUTER_API_KEY").unwrap_or_default();
        if api_key.is_empty() {
            tracing::warn!(
                "OPENROUTER_API_KEY is not set or empty; OpenRouter requests will fail authentication"
            );
        }
        Self::with_api_key(model_id, api_key)
    }

    /// Create a new OpenRouter client with an explicit API key.
    pub fn with_api_key(model_id: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            client: build_client(),
            api_key: api_key.into(),
            model_id: model_id.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            timeout: None,
            max_retries: DEFAULT_MAX_RETRIES,
            response_format: None,
            parallel_tool_calls: None,
            site_url: None,
            app_name: None,
        }
    }

    /// Set a custom base URL (e.g. for proxies or test servers).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Set a custom timeout for non-streaming HTTP requests (default: 120s).
    ///
    /// The timeout applies per-request to `generate`; `generate_stream` is a
    /// long-lived SSE connection and is protected only by the client's
    /// connect timeout, since a whole-request deadline would abort healthy
    /// streams.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set the maximum number of retries for failed requests (429 and 5xx).
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Set the response format (e.g. `"json_object"` or `"text"`).
    ///
    /// Note that support depends on the routed model — not every upstream
    /// provider behind OpenRouter honors `response_format`.
    pub fn with_response_format(mut self, format: &str) -> Self {
        self.response_format = Some(format.to_string());
        self
    }

    /// Enable or disable parallel tool calls.
    pub fn with_parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = Some(enabled);
        self
    }

    /// Set the `HTTP-Referer` header OpenRouter uses to attribute requests to
    /// your site for its public rankings.
    pub fn with_site_url(mut self, site_url: impl Into<String>) -> Self {
        self.site_url = Some(site_url.into());
        self
    }

    /// Set the `X-Title` header OpenRouter uses as the display name for your
    /// app in its public rankings.
    pub fn with_app_name(mut self, app_name: impl Into<String>) -> Self {
        self.app_name = Some(app_name.into());
        self
    }

    /// Applies auth plus the optional OpenRouter attribution headers.
    fn apply_headers(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let builder = builder.header("Authorization", format!("Bearer {}", self.api_key));
        let builder = match &self.site_url {
            Some(url) => builder.header("HTTP-Referer", url),
            None => builder,
        };
        match &self.app_name {
            Some(name) => builder.header("X-Title", name),
            None => builder,
        }
    }

    fn build_request_body(&self, request: &ChatRequest) -> OpenRouterRequest {
        let messages: Vec<OpenRouterMessage> = request.messages.iter().map(Into::into).collect();

        let tools: Option<Vec<OpenRouterTool>> = if request.tools.is_empty() {
            None
        } else {
            Some(request.tools.iter().map(Into::into).collect())
        };

        OpenRouterRequest {
            model: self.model_id.clone(),
            messages,
            tools,
            temperature: request.temperature,
            // OpenRouter normalizes `max_tokens` across its upstream
            // providers (several of which, e.g. Anthropic, reject the
            // OpenAI-specific `max_completion_tokens`), so it is the portable
            // choice here.
            max_tokens: request.max_tokens,
            stream: false,
            response_format: self
                .response_format
                .as_ref()
                .map(|f| serde_json::json!({ "type": f })),
            parallel_tool_calls: self.parallel_tool_calls,
        }
    }
}

impl Model for OpenRouter {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    #[tracing::instrument(skip_all, fields(model = %self.model_id))]
    async fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let body = self.build_request_body(request);
        let url = format!("{}/chat/completions", self.base_url);
        let timeout = self.timeout.unwrap_or(DEFAULT_REQUEST_TIMEOUT);

        tracing::debug!("sending chat completion request");

        for attempt in 0..=self.max_retries {
            let response = self
                .apply_headers(self.client.post(&url))
                .timeout(timeout)
                .json(&body)
                .send()
                .await
                .map_err(|e| DaimonError::Model(format!("OpenRouter HTTP error: {e}")))?;

            let status = response.status();

            if status.is_success() {
                tracing::debug!("received successful response");
                let or_response: OpenRouterResponse = response.json().await.map_err(|e| {
                    DaimonError::Model(format!("OpenRouter response parse error: {e}"))
                })?;
                return parse_response(or_response);
            }

            let retry_after = retry::parse_retry_after(response.headers());
            let text = response.text().await.unwrap_or_default();
            let is_retryable = status.as_u16() == 429 || status.is_server_error();

            if is_retryable && attempt < self.max_retries {
                let delay = retry::backoff_delay(attempt, retry_after);
                tracing::debug!(
                    status = %status,
                    attempt = attempt + 1,
                    max_retries = self.max_retries,
                    delay_ms = delay.as_millis(),
                    "retryable error, backing off"
                );
                tokio::time::sleep(delay).await;
            } else {
                return Err(DaimonError::Model(format!(
                    "OpenRouter API error ({status}): {text}"
                )));
            }
        }

        unreachable!("loop always returns or retries")
    }

    #[tracing::instrument(skip_all, fields(model = %self.model_id))]
    async fn generate_stream(&self, request: &ChatRequest) -> Result<ResponseStream> {
        let mut body = self.build_request_body(request);
        body.stream = true;
        let url = format!("{}/chat/completions", self.base_url);

        // Retry only the initial POST/handshake — once the stream is
        // established, mid-stream failures must never be retried (the
        // consumer has already observed a partial response).
        let mut response = None;
        for attempt in 0..=self.max_retries {
            tracing::debug!(attempt, "sending streaming chat completion request");
            let resp = self
                .apply_headers(self.client.post(&url))
                .json(&body)
                .send()
                .await
                .map_err(|e| DaimonError::Model(format!("OpenRouter HTTP error: {e}")))?;
            let status = resp.status();

            if status.is_success() {
                response = Some(resp);
                break;
            }

            let retry_after = retry::parse_retry_after(resp.headers());
            let text = resp.text().await.unwrap_or_default();
            let is_retryable = status.as_u16() == 429 || status.is_server_error();

            if is_retryable && attempt < self.max_retries {
                let delay = retry::backoff_delay(attempt, retry_after);
                tracing::debug!(status = %status, attempt, delay_ms = delay.as_millis(), "retryable error on stream handshake, backing off");
                tokio::time::sleep(delay).await;
            } else {
                return Err(DaimonError::Model(format!(
                    "OpenRouter API error ({status}): {text}"
                )));
            }
        }
        let response = response.expect("loop breaks with a response or returns an error");

        tracing::debug!("stream established, processing chunks");
        let byte_stream = response.bytes_stream();

        let stream = async_stream::try_stream! {
            use futures::StreamExt;
            use daimon_core::stream_util::LineBuffer;

            let mut buffer = LineBuffer::new();
            let mut state = OpenRouterStreamState::default();
            let mut stream = Box::pin(byte_stream);

            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| DaimonError::Model(format!("OpenRouter stream error: {e}")))?;
                buffer.push(&chunk);

                while let Some(line) = buffer.next_line() {
                    for event in handle_openrouter_sse_line(&mut state, &line) {
                        yield event;
                    }
                }
            }

            // A stream may end without a trailing newline, leaving a final SSE
            // record buffered. Recover it through the identical parse path.
            if let Some(line) = buffer.take_remaining() {
                for event in handle_openrouter_sse_line(&mut state, &line) {
                    yield event;
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

/// Streaming state for the OpenRouter SSE parser.
///
/// OpenRouter announces a tool call's `id` only on its first chunk; subsequent
/// argument fragments carry just the array `index`. The map correlates the
/// index back to the announced id, and `open_calls` tracks announcement order
/// so `ToolCallEnd` can be emitted for every call when `finish_reason`
/// arrives.
#[derive(Default)]
struct OpenRouterStreamState {
    index_to_id: HashMap<usize, String>,
    open_calls: Vec<String>,
}

/// Parses one SSE line from an OpenRouter chat-completions stream into
/// [`StreamEvent`]s.
///
/// This is the single parse path shared by the main streaming loop and the
/// end-of-stream remainder recovery, extracted (following the Anthropic
/// provider's `handle_anthropic_stream_event` pattern) so it can be
/// unit-tested without a live HTTP stream.
///
/// Behavior notes:
/// - Lines that are not `data:` records are ignored. In particular,
///   OpenRouter periodically emits `: OPENROUTER PROCESSING` comments as
///   keep-alives; they are dropped here.
/// - Tool-call ids are the provider-assigned `id` when present; the array
///   index is only a fallback for upstreams that omit ids.
/// - `finish_reason` closes all open tool calls with `ToolCallEnd`, and
///   `content_filter` additionally surfaces an in-band [`StreamEvent::Error`]
///   ([`StreamEvent::Done`] carries no stop reason, so the error event is the
///   only channel that keeps a filtered termination from being silent).
/// - The `data: [DONE]` sentinel yields `Done`.
fn handle_openrouter_sse_line(state: &mut OpenRouterStreamState, line: &str) -> Vec<StreamEvent> {
    let line = line.trim();
    let mut events = Vec::new();

    if line.is_empty() {
        return events;
    }
    if line == "data: [DONE]" {
        // Defensive: close any calls the server never finished.
        for id in state.open_calls.drain(..) {
            events.push(StreamEvent::ToolCallEnd { id });
        }
        events.push(StreamEvent::Done);
        return events;
    }

    let Some(data) = line.strip_prefix("data: ") else {
        return events;
    };
    let chunk = match serde_json::from_str::<OpenRouterStreamChunk>(data) {
        Ok(chunk) => chunk,
        Err(e) => {
            // OpenRouter can deliver errors in-band as
            // `data: {"error": {"message": ..., "code": ...}}` — e.g. when an
            // upstream provider fails mid-stream. Surface them instead of
            // letting the stream end silently with a truncated response.
            if let Ok(err) = serde_json::from_str::<OpenRouterStreamError>(data) {
                events.push(StreamEvent::Error(err.error.message));
                return events;
            }
            tracing::debug!(error = %e, "dropping undeserializable OpenRouter SSE event");
            return events;
        }
    };

    for choice in &chunk.choices {
        if let Some(ref content) = choice.delta.content
            && !content.is_empty()
        {
            events.push(StreamEvent::TextDelta(content.clone()));
        }
        if let Some(ref tool_calls) = choice.delta.tool_calls {
            for tc in tool_calls {
                let Some(ref func) = tc.function else {
                    continue;
                };
                // Prefer the provider-assigned id; remember it so later
                // fragments (which omit the id) resolve through the index.
                let id = match tc.id.as_deref().filter(|id| !id.is_empty()) {
                    Some(id) => {
                        state.index_to_id.insert(tc.index, id.to_string());
                        id.to_string()
                    }
                    None => state
                        .index_to_id
                        .get(&tc.index)
                        .cloned()
                        .unwrap_or_else(|| tc.index.to_string()),
                };
                if let Some(ref name) = func.name {
                    state.open_calls.push(id.clone());
                    events.push(StreamEvent::ToolCallStart {
                        id: id.clone(),
                        name: name.clone(),
                    });
                }
                if let Some(ref args) = func.arguments
                    && !args.is_empty()
                {
                    events.push(StreamEvent::ToolCallDelta {
                        id,
                        arguments_delta: args.clone(),
                    });
                }
            }
        }
        if let Some(ref reason) = choice.finish_reason {
            // The turn is complete: every announced tool call is now final.
            for id in state.open_calls.drain(..) {
                events.push(StreamEvent::ToolCallEnd { id });
            }
            if reason == "content_filter" {
                events.push(StreamEvent::Error(
                    "OpenRouter blocked the response (finish_reason=content_filter)".to_string(),
                ));
            }
        }
    }

    events
}

fn parse_response(response: OpenRouterResponse) -> Result<ChatResponse> {
    let choice = response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| DaimonError::Model("no choices in OpenRouter response".into()))?;

    let mut tool_calls = Vec::new();
    for tc in choice.message.tool_calls.unwrap_or_default() {
        // Malformed arguments must surface as an error: silently coercing
        // them to null would run the tool with the corruption hidden.
        let arguments = if tc.function.arguments.trim().is_empty() {
            serde_json::Value::Object(serde_json::Map::new())
        } else {
            serde_json::from_str(&tc.function.arguments).map_err(|e| {
                DaimonError::Model(format!(
                    "OpenRouter returned malformed JSON arguments for tool call '{}' (id {}): {e}",
                    tc.function.name, tc.id
                ))
            })?
        };
        tool_calls.push(ToolCall {
            id: tc.id,
            name: tc.function.name,
            arguments,
        });
    }

    let stop_reason = match choice.finish_reason.as_deref() {
        Some("tool_calls") => StopReason::ToolUse,
        Some("length") => StopReason::MaxTokens,
        Some("content_filter") => StopReason::ContentFiltered,
        _ => StopReason::EndTurn,
    };

    let message = Message {
        role: Role::Assistant,
        content: choice.message.content,
        tool_calls,
        tool_call_id: None,
    };

    Ok(ChatResponse {
        message,
        stop_reason,
        usage: response.usage.map(|u| Usage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            cached_tokens: u
                .prompt_tokens_details
                .map(|d| d.cached_tokens)
                .unwrap_or(0),
        }),
    })
}

// --- OpenRouter API types (OpenAI-compatible wire format) ---

#[derive(Serialize)]
struct OpenRouterRequest {
    model: String,
    messages: Vec<OpenRouterMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenRouterTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
}

#[derive(Serialize, Deserialize)]
struct OpenRouterMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenRouterToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

impl From<&Message> for OpenRouterMessage {
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
                    .map(|tc| OpenRouterToolCall {
                        id: tc.id.clone(),
                        r#type: "function".to_string(),
                        function: OpenRouterFunction {
                            name: tc.name.clone(),
                            arguments: serde_json::to_string(&tc.arguments).unwrap_or_default(),
                        },
                        index: 0,
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
struct OpenRouterTool {
    r#type: String,
    function: OpenRouterToolFunction,
}

impl From<&ToolSpec> for OpenRouterTool {
    fn from(spec: &ToolSpec) -> Self {
        Self {
            r#type: "function".to_string(),
            function: OpenRouterToolFunction {
                name: spec.name.clone(),
                description: spec.description.clone(),
                parameters: spec.parameters.clone(),
            },
        }
    }
}

#[derive(Serialize)]
struct OpenRouterToolFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Deserialize)]
struct OpenRouterResponse {
    choices: Vec<OpenRouterChoice>,
    usage: Option<OpenRouterUsage>,
}

#[derive(Deserialize)]
struct OpenRouterChoice {
    message: OpenRouterChoiceMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenRouterChoiceMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenRouterToolCall>>,
}

#[derive(Serialize, Deserialize)]
struct OpenRouterToolCall {
    #[serde(default)]
    id: String,
    #[serde(default)]
    r#type: String,
    #[serde(default)]
    function: OpenRouterFunction,
    #[serde(default)]
    index: usize,
}

#[derive(Serialize, Deserialize, Default)]
struct OpenRouterFunction {
    #[serde(default)]
    name: String,
    #[serde(default)]
    arguments: String,
}

#[derive(Deserialize)]
struct OpenRouterUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    prompt_tokens_details: Option<OpenRouterPromptTokensDetails>,
}

#[derive(Deserialize)]
struct OpenRouterPromptTokensDetails {
    #[serde(default)]
    cached_tokens: u32,
}

#[derive(Deserialize)]
struct OpenRouterStreamChunk {
    choices: Vec<OpenRouterStreamChoice>,
}

/// In-band error payload OpenRouter emits inside the SSE stream (as opposed
/// to a non-200 response), typically when an upstream provider fails
/// mid-generation. Only `message` is needed; `code`/`metadata` are ignored.
#[derive(Deserialize)]
struct OpenRouterStreamError {
    error: OpenRouterStreamErrorBody,
}

#[derive(Deserialize)]
struct OpenRouterStreamErrorBody {
    message: String,
}

#[derive(Deserialize)]
struct OpenRouterStreamChoice {
    delta: OpenRouterStreamDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenRouterStreamDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenRouterStreamToolCall>>,
}

#[derive(Deserialize)]
struct OpenRouterStreamToolCall {
    index: usize,
    /// Provider-assigned id, present only on a call's first chunk.
    #[serde(default)]
    id: Option<String>,
    function: Option<OpenRouterStreamFunction>,
}

#[derive(Deserialize)]
struct OpenRouterStreamFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_conversion_user() {
        let msg = Message::user("hello");
        let or_msg: OpenRouterMessage = (&msg).into();
        assert_eq!(or_msg.role, "user");
        assert_eq!(or_msg.content.as_deref(), Some("hello"));
        assert!(or_msg.tool_calls.is_none());
    }

    #[test]
    fn test_message_conversion_assistant_with_tools() {
        let msg = Message::assistant_with_tool_calls(vec![ToolCall {
            id: "tc_1".into(),
            name: "calc".into(),
            arguments: serde_json::json!({"x": 1}),
        }]);
        let or_msg: OpenRouterMessage = (&msg).into();
        assert_eq!(or_msg.role, "assistant");
        assert!(or_msg.tool_calls.is_some());
        assert_eq!(or_msg.tool_calls.unwrap().len(), 1);
    }

    #[test]
    fn test_message_conversion_tool_result() {
        let msg = Message::tool_result("tc_1", "42");
        let or_msg: OpenRouterMessage = (&msg).into();
        assert_eq!(or_msg.role, "tool");
        assert_eq!(or_msg.tool_call_id.as_deref(), Some("tc_1"));
        assert_eq!(or_msg.content.as_deref(), Some("42"));
    }

    #[test]
    fn test_tool_spec_conversion() {
        let spec = ToolSpec {
            name: "calc".into(),
            description: "Calculator".into(),
            parameters: serde_json::json!({"type": "object"}),
        };
        let or_tool: OpenRouterTool = (&spec).into();
        assert_eq!(or_tool.r#type, "function");
        assert_eq!(or_tool.function.name, "calc");
    }

    #[test]
    fn test_request_body_uses_max_tokens() {
        // OpenRouter normalizes `max_tokens` across upstream providers; the
        // OpenAI-specific `max_completion_tokens` must not be sent.
        let model = OpenRouter::with_api_key("openai/gpt-4o", "key");
        let request = ChatRequest {
            messages: vec![Message::user("hi")],
            tools: vec![],
            temperature: None,
            max_tokens: Some(512),
        };
        let body = model.build_request_body(&request);
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["max_tokens"], 512);
        assert!(
            json.get("max_completion_tokens").is_none(),
            "OpenAI-specific max_completion_tokens must not be sent"
        );
    }

    #[test]
    fn test_parse_response_end_turn() {
        let raw = OpenRouterResponse {
            choices: vec![OpenRouterChoice {
                message: OpenRouterChoiceMessage {
                    content: Some("hello".into()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".into()),
            }],
            usage: Some(OpenRouterUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                prompt_tokens_details: None,
            }),
        };
        let resp = parse_response(raw).unwrap();
        assert_eq!(resp.text(), "hello");
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        assert!(!resp.has_tool_calls());
        assert_eq!(resp.usage.unwrap().input_tokens, 10);
    }

    #[test]
    fn test_parse_response_tool_use() {
        let raw = OpenRouterResponse {
            choices: vec![OpenRouterChoice {
                message: OpenRouterChoiceMessage {
                    content: None,
                    tool_calls: Some(vec![OpenRouterToolCall {
                        id: "tc_1".into(),
                        r#type: "function".into(),
                        function: OpenRouterFunction {
                            name: "calc".into(),
                            arguments: r#"{"x":1}"#.into(),
                        },
                        index: 0,
                    }]),
                },
                finish_reason: Some("tool_calls".into()),
            }],
            usage: None,
        };
        let resp = parse_response(raw).unwrap();
        assert!(resp.has_tool_calls());
        assert_eq!(resp.tool_calls()[0].name, "calc");
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
    }

    #[test]
    fn test_parse_response_malformed_tool_arguments_is_error() {
        let raw = OpenRouterResponse {
            choices: vec![OpenRouterChoice {
                message: OpenRouterChoiceMessage {
                    content: None,
                    tool_calls: Some(vec![OpenRouterToolCall {
                        id: "tc_1".into(),
                        r#type: "function".into(),
                        function: OpenRouterFunction {
                            name: "calc".into(),
                            arguments: r#"{"x": not-json"#.into(),
                        },
                        index: 0,
                    }]),
                },
                finish_reason: Some("tool_calls".into()),
            }],
            usage: None,
        };
        let err = parse_response(raw).unwrap_err();
        assert!(
            err.to_string().contains("malformed JSON arguments"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_parse_response_empty_tool_arguments_maps_to_empty_object() {
        let raw = OpenRouterResponse {
            choices: vec![OpenRouterChoice {
                message: OpenRouterChoiceMessage {
                    content: None,
                    tool_calls: Some(vec![OpenRouterToolCall {
                        id: "tc_1".into(),
                        r#type: "function".into(),
                        function: OpenRouterFunction {
                            name: "ping".into(),
                            arguments: String::new(),
                        },
                        index: 0,
                    }]),
                },
                finish_reason: Some("tool_calls".into()),
            }],
            usage: None,
        };
        let resp = parse_response(raw).unwrap();
        assert_eq!(resp.tool_calls()[0].arguments, serde_json::json!({}));
    }

    #[test]
    fn test_parse_response_content_filter_maps_to_content_filtered() {
        let raw = OpenRouterResponse {
            choices: vec![OpenRouterChoice {
                message: OpenRouterChoiceMessage {
                    content: None,
                    tool_calls: None,
                },
                finish_reason: Some("content_filter".into()),
            }],
            usage: None,
        };
        let resp = parse_response(raw).unwrap();
        assert_eq!(resp.stop_reason, StopReason::ContentFiltered);
    }

    #[test]
    fn test_parse_response_no_choices() {
        let raw = OpenRouterResponse {
            choices: vec![],
            usage: None,
        };
        let result = parse_response(raw);
        assert!(result.is_err());
    }

    #[test]
    fn test_openrouter_new_default() {
        let model = OpenRouter::new("openai/gpt-4o");
        assert_eq!(model.model_id, "openai/gpt-4o");
        assert_eq!(model.base_url, DEFAULT_BASE_URL);
        assert_eq!(model.max_retries, DEFAULT_MAX_RETRIES);
    }

    #[test]
    fn test_openrouter_with_base_url() {
        let model = OpenRouter::new("openai/gpt-4o").with_base_url("http://localhost:8080");
        assert_eq!(model.base_url, "http://localhost:8080");
    }

    #[test]
    fn test_with_timeout() {
        let model =
            OpenRouter::new("openai/gpt-4o").with_timeout(std::time::Duration::from_secs(60));
        assert_eq!(model.timeout, Some(std::time::Duration::from_secs(60)));
    }

    #[test]
    fn test_with_max_retries() {
        let model = OpenRouter::new("openai/gpt-4o").with_max_retries(5);
        assert_eq!(model.max_retries, 5);
    }

    #[test]
    fn test_with_response_format() {
        let model = OpenRouter::new("openai/gpt-4o").with_response_format("json_object");
        assert_eq!(model.response_format.as_deref(), Some("json_object"));
    }

    #[test]
    fn test_with_parallel_tool_calls() {
        let model = OpenRouter::new("openai/gpt-4o").with_parallel_tool_calls(true);
        assert_eq!(model.parallel_tool_calls, Some(true));
    }

    #[test]
    fn test_with_site_url_and_app_name() {
        let model = OpenRouter::new("openai/gpt-4o")
            .with_site_url("https://example.com")
            .with_app_name("example-app");
        assert_eq!(model.site_url.as_deref(), Some("https://example.com"));
        assert_eq!(model.app_name.as_deref(), Some("example-app"));
    }

    #[test]
    fn test_debug_redacts_api_key() {
        let model = OpenRouter::with_api_key("openai/gpt-4o", "sk-or-supersecret-key-value");
        let dbg = format!("{model:?}");
        assert!(
            !dbg.contains("sk-or-supersecret-key-value"),
            "Debug output must not contain the plaintext API key: {dbg}"
        );
        assert!(dbg.contains("[redacted]"));
    }

    // --- Streaming SSE-parser tests ---
    //
    // These exercise `handle_openrouter_sse_line` directly, feeding lines in
    // the exact shapes OpenRouter emits over SSE.

    fn feed(state: &mut OpenRouterStreamState, lines: &[&str]) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        for line in lines {
            events.extend(handle_openrouter_sse_line(state, line));
        }
        events
    }

    #[test]
    fn test_stream_text_delta_and_done() {
        let mut state = OpenRouterStreamState::default();
        let events = feed(
            &mut state,
            &[
                r#"data: {"choices":[{"delta":{"content":"Hel"}}]}"#,
                r#"data: {"choices":[{"delta":{"content":"lo"}}]}"#,
                r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
                "data: [DONE]",
            ],
        );
        assert_eq!(events.len(), 3, "got {events:?}");
        assert!(matches!(&events[0], StreamEvent::TextDelta(t) if t == "Hel"));
        assert!(matches!(&events[1], StreamEvent::TextDelta(t) if t == "lo"));
        assert!(matches!(&events[2], StreamEvent::Done));
    }

    #[test]
    fn test_stream_tool_call_uses_real_id_and_emits_end() {
        let mut state = OpenRouterStreamState::default();
        let events = feed(
            &mut state,
            &[
                r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","function":{"name":"calc","arguments":""}}]}}]}"#,
                r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"x\":"}}]}}]}"#,
                r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}"#,
                r#"data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
                "data: [DONE]",
            ],
        );
        assert_eq!(events.len(), 5, "got {events:?}");
        assert!(matches!(&events[0], StreamEvent::ToolCallStart { id, name }
            if id == "call_abc" && name == "calc"));
        assert!(matches!(&events[1], StreamEvent::ToolCallDelta { id, .. } if id == "call_abc"));
        assert!(matches!(&events[2], StreamEvent::ToolCallDelta { id, .. } if id == "call_abc"));
        assert!(
            matches!(&events[3], StreamEvent::ToolCallEnd { id } if id == "call_abc"),
            "ToolCallEnd must be emitted when finish_reason arrives: {events:?}"
        );
        assert!(matches!(&events[4], StreamEvent::Done));
    }

    #[test]
    fn test_stream_parallel_tool_calls_route_by_index() {
        let mut state = OpenRouterStreamState::default();
        let events = feed(
            &mut state,
            &[
                r#"data: {"choices":[{"delta":{"tool_calls":[
                    {"index":0,"id":"call_a","function":{"name":"f","arguments":""}},
                    {"index":1,"id":"call_b","function":{"name":"g","arguments":""}}]}}]}"#,
                r#"data: {"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"b"}}]}}]}"#,
                r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"a"}}]}}]}"#,
                r#"data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
            ],
        );
        assert!(matches!(&events[0], StreamEvent::ToolCallStart { id, .. } if id == "call_a"));
        assert!(matches!(&events[1], StreamEvent::ToolCallStart { id, .. } if id == "call_b"));
        assert!(matches!(&events[2],
            StreamEvent::ToolCallDelta { id, arguments_delta } if id == "call_b" && arguments_delta == "b"));
        assert!(matches!(&events[3],
            StreamEvent::ToolCallDelta { id, arguments_delta } if id == "call_a" && arguments_delta == "a"));
        let ends: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolCallEnd { id } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(ends, vec!["call_a", "call_b"]);
    }

    #[test]
    fn test_stream_tool_call_without_id_falls_back_to_index() {
        // Some OpenRouter upstreams omit the id entirely; the index is the
        // documented fallback so deltas still correlate.
        let mut state = OpenRouterStreamState::default();
        let events = feed(
            &mut state,
            &[
                r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"calc","arguments":""}}]}}]}"#,
                r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}}]}"#,
            ],
        );
        assert!(matches!(&events[0], StreamEvent::ToolCallStart { id, .. } if id == "0"));
        assert!(matches!(&events[1], StreamEvent::ToolCallDelta { id, .. } if id == "0"));
    }

    #[test]
    fn test_stream_content_filter_emits_error() {
        let mut state = OpenRouterStreamState::default();
        let events = feed(
            &mut state,
            &[
                r#"data: {"choices":[{"delta":{},"finish_reason":"content_filter"}]}"#,
                "data: [DONE]",
            ],
        );
        assert_eq!(events.len(), 2, "got {events:?}");
        assert!(matches!(&events[0], StreamEvent::Error(msg) if msg.contains("content_filter")));
        assert!(matches!(&events[1], StreamEvent::Done));
    }

    #[test]
    fn test_stream_keepalive_comment_is_dropped() {
        // OpenRouter periodically emits `: OPENROUTER PROCESSING` comment
        // lines as keep-alives; they must not produce events.
        let mut state = OpenRouterStreamState::default();
        let events = feed(
            &mut state,
            &[": OPENROUTER PROCESSING", ": keep-alive comment", ""],
        );
        assert!(events.is_empty());
    }

    #[test]
    fn test_stream_undeserializable_line_is_dropped() {
        let mut state = OpenRouterStreamState::default();
        let events = feed(&mut state, &["data: {not json"]);
        assert!(events.is_empty());
    }

    #[test]
    fn test_stream_in_band_error_payload_is_surfaced() {
        // OpenRouter delivers mid-stream errors as
        // `data: {"error": {"message": ..., "code": ...}}`; they must reach
        // the consumer as an error event, not vanish into the drop path.
        let mut state = OpenRouterStreamState::default();
        let events = feed(
            &mut state,
            &[
                r#"data: {"choices":[{"delta":{"content":"partial"}}]}"#,
                r#"data: {"error":{"message":"upstream provider error","code":502}}"#,
            ],
        );
        assert_eq!(events.len(), 2, "got {events:?}");
        assert!(matches!(&events[0], StreamEvent::TextDelta(t) if t == "partial"));
        assert!(
            matches!(&events[1], StreamEvent::Error(msg) if msg == "upstream provider error"),
            "in-band error must surface as StreamEvent::Error: {events:?}"
        );
    }
}
