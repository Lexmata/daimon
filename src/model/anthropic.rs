//! Anthropic Claude model provider.
//!
//! This module provides integration with the Anthropic API for chat completions,
//! streaming, and tool use. Configure via builder methods for timeout, retries,
//! and prompt caching.

use std::collections::HashMap;
use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::error::{DaimonError, Result};
use crate::model::Model;
use crate::model::types::{ChatRequest, ChatResponse, Message, Role, StopReason, ToolSpec, Usage};
use crate::stream::{ResponseStream, StreamEvent};
use crate::tool::ToolCall;

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const API_VERSION: &str = "2023-06-01";

/// Default `max_tokens` sent when the request does not specify one.
///
/// The Anthropic API requires `max_tokens` on every request; 4096 is a
/// conservative ceiling that every Claude model supports.
const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Default whole-request timeout applied to non-streaming `generate` calls.
///
/// Long completions can legitimately take minutes, so this is deliberately
/// generous. Override with [`Anthropic::with_timeout`].
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

/// Anthropic Claude model provider for the Daimon agent framework.
///
/// Supports chat completions, streaming, tool use, and configurable timeout,
/// retries, and prompt caching.
pub struct Anthropic {
    client: Client,
    api_key: String,
    model_id: String,
    base_url: String,
    timeout: Option<Duration>,
    max_retries: u32,
    use_prompt_caching: bool,
}

impl std::fmt::Debug for Anthropic {
    /// Hand-written to avoid leaking the plaintext API key in logs or panic
    /// output; a derived `Debug` would print `api_key` verbatim.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Anthropic")
            .field("client", &self.client)
            .field("api_key", &"[redacted]")
            .field("model_id", &self.model_id)
            .field("base_url", &self.base_url)
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .field("use_prompt_caching", &self.use_prompt_caching)
            .finish()
    }
}

impl Anthropic {
    /// Creates a new Anthropic client using `ANTHROPIC_API_KEY` from the environment.
    ///
    /// The constructor never fails; if the environment variable is unset or
    /// empty a warning is logged and requests will fail with an auth error.
    pub fn new(model_id: impl Into<String>) -> Self {
        let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
        if api_key.is_empty() {
            tracing::warn!(
                "ANTHROPIC_API_KEY is not set or empty; Anthropic requests will fail authentication"
            );
        }
        Self::with_api_key(model_id, api_key)
    }

    /// Creates a new Anthropic client with an explicit API key.
    pub fn with_api_key(model_id: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            client: build_client(),
            api_key: api_key.into(),
            model_id: model_id.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            timeout: None,
            max_retries: 3,
            use_prompt_caching: false,
        }
    }

    /// Sets a custom base URL for the API (e.g. for proxies or testing).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Sets a timeout for non-streaming HTTP requests (default: 120s).
    ///
    /// The timeout applies per-request to `generate`; `generate_stream` is a
    /// long-lived SSE connection and is protected only by the client's
    /// connect timeout, since a whole-request deadline would abort healthy
    /// streams.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Sets the maximum number of retries for retriable errors (429, 529, 5xx).
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Enables prompt caching by attaching `cache_control` breakpoints.
    ///
    /// Prompt caching is generally available; the request needs no
    /// `anthropic-beta` header, only the `cache_control` blocks this enables.
    pub fn with_prompt_caching(mut self) -> Self {
        self.use_prompt_caching = true;
        self
    }

    fn build_request_body(&self, request: &ChatRequest) -> AnthropicRequest {
        let mut system: Option<serde_json::Value> = None;
        let mut messages = Vec::new();

        for msg in &request.messages {
            match msg.role {
                Role::System => {
                    if let Some(ref text) = msg.content {
                        if self.use_prompt_caching {
                            system = Some(serde_json::json!([{
                                "type": "text",
                                "text": text,
                                "cache_control": {"type": "ephemeral"}
                            }]));
                        } else {
                            system = Some(serde_json::Value::String(text.clone()));
                        }
                    }
                }
                Role::User => {
                    messages.push(AnthropicMessage {
                        role: "user".to_string(),
                        content: AnthropicContent::Text(msg.content.clone().unwrap_or_default()),
                    });
                }
                Role::Assistant => {
                    if !msg.tool_calls.is_empty() {
                        // Preserve any assistant text that accompanied the tool
                        // calls. Anthropic requires the text block to precede
                        // the tool_use blocks; dropping it (the previous
                        // behavior) loses the model's reasoning and can break
                        // multi-turn continuity.
                        let mut blocks: Vec<AnthropicContentBlock> = Vec::new();
                        if let Some(ref text) = msg.content
                            && !text.is_empty()
                        {
                            blocks.push(AnthropicContentBlock::Text { text: text.clone() });
                        }
                        blocks.extend(msg.tool_calls.iter().map(|tc| {
                            AnthropicContentBlock::ToolUse {
                                id: tc.id.clone(),
                                name: tc.name.clone(),
                                input: tc.arguments.clone(),
                            }
                        }));
                        messages.push(AnthropicMessage {
                            role: "assistant".to_string(),
                            content: AnthropicContent::Blocks(blocks),
                        });
                    } else {
                        messages.push(AnthropicMessage {
                            role: "assistant".to_string(),
                            content: AnthropicContent::Text(
                                msg.content.clone().unwrap_or_default(),
                            ),
                        });
                    }
                }
                Role::Tool => {
                    messages.push(AnthropicMessage {
                        role: "user".to_string(),
                        content: AnthropicContent::Blocks(vec![
                            AnthropicContentBlock::ToolResult {
                                tool_use_id: msg.tool_call_id.clone().unwrap_or_default(),
                                content: msg.content.clone().unwrap_or_default(),
                            },
                        ]),
                    });
                }
            }
        }

        let tools: Option<Vec<AnthropicTool>> = if request.tools.is_empty() {
            None
        } else {
            let mut tool_list: Vec<AnthropicTool> = request.tools.iter().map(Into::into).collect();
            if self.use_prompt_caching
                && let Some(last) = tool_list.last_mut()
            {
                last.cache_control = Some(serde_json::json!({"type": "ephemeral"}));
            }
            Some(tool_list)
        };

        AnthropicRequest {
            model: self.model_id.clone(),
            system,
            messages,
            tools,
            max_tokens: request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            temperature: request.temperature,
            stream: false,
        }
    }
}

impl Model for Anthropic {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    #[tracing::instrument(skip_all, fields(model = %self.model_id))]
    async fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let body = self.build_request_body(request);
        let url = format!("{}/v1/messages", self.base_url);

        tracing::debug!("building request for non-streaming generate");
        let timeout = self.timeout.unwrap_or(DEFAULT_REQUEST_TIMEOUT);
        let mut attempt = 0u32;

        loop {
            let req_builder = self
                .client
                .post(&url)
                .timeout(timeout)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", API_VERSION)
                .header("content-type", "application/json")
                .json(&body);

            tracing::debug!(attempt = attempt, "sending request to Anthropic API");
            let response = req_builder
                .send()
                .await
                .map_err(|e| DaimonError::Model(format!("Anthropic HTTP error: {e}")))?;
            let status = response.status();
            let retry_after = crate::model::retry::parse_retry_after(response.headers());
            let text = response.text().await.unwrap_or_default();

            if status.is_success() {
                tracing::debug!("request succeeded, parsing response");
                let api_response: AnthropicResponse =
                    serde_json::from_str(&text).map_err(DaimonError::Serialization)?;
                return parse_response(api_response);
            }

            let code = status.as_u16();
            let is_retriable = code == 429 || code == 529 || (500..600).contains(&code);

            if is_retriable && attempt < self.max_retries {
                let delay = crate::model::retry::backoff_delay(attempt, retry_after);
                tracing::debug!(
                    status = %status,
                    attempt = attempt,
                    delay_ms = delay.as_millis(),
                    "retriable error, backing off"
                );
                tokio::time::sleep(delay).await;
                attempt += 1;
            } else {
                return Err(DaimonError::Model(format!(
                    "Anthropic API error ({status}): {text}"
                )));
            }
        }
    }

    #[tracing::instrument(skip_all, fields(model = %self.model_id))]
    async fn generate_stream(&self, request: &ChatRequest) -> Result<ResponseStream> {
        let mut body = self.build_request_body(request);
        body.stream = true;
        let url = format!("{}/v1/messages", self.base_url);

        tracing::debug!("building request for streaming generate");
        // Retry only the initial POST/handshake — once the stream is
        // established, mid-stream failures must never be retried (the
        // consumer has already observed a partial response).
        let mut response = None;
        for attempt in 0..=self.max_retries {
            let req_builder = self
                .client
                .post(&url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", API_VERSION)
                .header("content-type", "application/json")
                .json(&body);

            tracing::debug!(attempt, "sending streaming request to Anthropic API");
            let resp = req_builder
                .send()
                .await
                .map_err(|e| DaimonError::Model(format!("Anthropic HTTP error: {e}")))?;
            let status = resp.status();

            if status.is_success() {
                response = Some(resp);
                break;
            }

            let retry_after = crate::model::retry::parse_retry_after(resp.headers());
            let text = resp.text().await.unwrap_or_default();
            let code = status.as_u16();
            let is_retriable = code == 429 || code == 529 || (500..600).contains(&code);

            if is_retriable && attempt < self.max_retries {
                let delay = crate::model::retry::backoff_delay(attempt, retry_after);
                tracing::debug!(status = %status, attempt, delay_ms = delay.as_millis(), "retriable error on stream handshake, backing off");
                tokio::time::sleep(delay).await;
            } else {
                return Err(DaimonError::Model(format!(
                    "Anthropic API error ({status}): {text}"
                )));
            }
        }
        let response = response.expect("loop breaks with a response or returns an error");

        tracing::debug!("stream established, processing events");
        let byte_stream = response.bytes_stream();

        let stream = async_stream::try_stream! {
            use futures::StreamExt;
            use crate::model::line_buffer::LineBuffer;

            // Byte-accurate line buffer: a multi-byte UTF-8 character can be
            // split across two network chunks; only complete lines are decoded.
            let mut buffer = LineBuffer::new();
            // Maps an Anthropic content-block `index` to the tool_use id opened
            // at that index. The streaming spec correlates `content_block_delta`
            // (partial_json) and `content_block_stop` events to their block only
            // via this index — the id is present just once, at
            // `content_block_start`.
            let mut index_to_id: HashMap<u64, String> = HashMap::new();
            let mut stream = Box::pin(byte_stream);
            // Reused across the whole stream so the parse path allocates no
            // per-event Vec; drained after each event.
            let mut events: Vec<StreamEvent> = Vec::new();

            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| DaimonError::Model(format!("Anthropic stream error: {e}")))?;
                buffer.push(&chunk);

                while let Some(line) = buffer.next_line() {
                    let line = line.trim();

                    if line.is_empty() {
                        continue;
                    }

                    if let Some(data) = line.strip_prefix("data: ") {
                        match serde_json::from_str::<AnthropicStreamEvent>(data) {
                            Ok(event) => {
                                // Fill the reusable buffer (no per-event Vec
                                // allocation), then drain it into the stream.
                                handle_anthropic_stream_event_into(&mut index_to_id, event, &mut events);
                                for stream_event in events.drain(..) {
                                    yield stream_event;
                                }
                            }
                            Err(e) => {
                                tracing::debug!(error = %e, "dropping undeserializable Anthropic SSE event");
                            }
                        }
                    }
                }
            }

            // A stream can terminate without a trailing newline on its final
            // `data:` line. `next_line` only yields newline-terminated lines, so
            // that last event would otherwise be silently dropped. Drain the
            // buffered remainder and run it through the identical SSE parse path.
            if let Some(line) = buffer.take_remaining()
                && let Some(data) = line.strip_prefix("data: ") {
                    match serde_json::from_str::<AnthropicStreamEvent>(data) {
                        Ok(event) => {
                            handle_anthropic_stream_event_into(&mut index_to_id, event, &mut events);
                            for stream_event in events.drain(..) {
                                yield stream_event;
                            }
                        }
                        Err(e) => {
                            tracing::debug!(error = %e, "dropping undeserializable Anthropic SSE event");
                        }
                    }
                }
        };

        Ok(Box::pin(stream))
    }
}

/// Test-facing wrapper around [`handle_anthropic_stream_event_into`] that
/// returns the produced events; the streaming loop reuses one buffer via the
/// `_into` variant instead, avoiding a fresh `Vec` allocation per SSE event.
#[cfg(test)]
fn handle_anthropic_stream_event(
    index_to_id: &mut HashMap<u64, String>,
    event: AnthropicStreamEvent,
) -> Vec<StreamEvent> {
    let mut events = Vec::new();
    handle_anthropic_stream_event_into(index_to_id, event, &mut events);
    events
}

/// Advances the Anthropic streaming state machine by one SSE event, appending
/// the [`StreamEvent`]s it produces (in order) to `events`.
///
/// This is the correctness-critical core of `generate_stream`: it maintains the
/// `index -> tool_use id` correlation map so that `content_block_delta`
/// (partial JSON arguments) and `content_block_stop` events — which carry only
/// a block `index`, never the id — are attributed to the tool call opened at
/// that index by the earlier `content_block_start`. Extracted from the
/// `async_stream::try_stream!` block so it can be unit-tested without a live
/// HTTP stream; the streaming loop drains whatever this appends.
///
/// Terminal handling: `message_delta` carries the final `stop_reason` and
/// usage. [`StreamEvent::Done`] cannot carry either, so usage is logged at
/// debug level and abnormal stop reasons (`refusal`, `pause_turn`) surface as
/// an in-band [`StreamEvent::Error`] — the only channel that keeps them from
/// ending the stream indistinguishably from a normal turn. Server-sent
/// `{"type":"error"}` events (e.g. `overloaded_error` mid-stream) also become
/// [`StreamEvent::Error`] items instead of being silently dropped.
fn handle_anthropic_stream_event_into(
    index_to_id: &mut HashMap<u64, String>,
    event: AnthropicStreamEvent,
    events: &mut Vec<StreamEvent>,
) {
    match event.r#type.as_str() {
        "content_block_start" => {
            if let Some(block) = event.content_block
                && block.r#type == "tool_use"
            {
                let id = block.id.unwrap_or_default();
                if let Some(idx) = event.index {
                    index_to_id.insert(idx, id.clone());
                }
                events.push(StreamEvent::ToolCallStart {
                    id,
                    name: block.name.unwrap_or_default(),
                });
            }
        }
        "content_block_delta" => {
            if let Some(delta) = event.delta {
                if let Some(text) = delta.text {
                    events.push(StreamEvent::TextDelta(text));
                }
                if let Some(json) = delta.partial_json {
                    // Resolve the id from the block index so the consumer keys
                    // accumulation to the right tool call.
                    let id = event
                        .index
                        .and_then(|idx| index_to_id.get(&idx).cloned())
                        .unwrap_or_default();
                    events.push(StreamEvent::ToolCallDelta {
                        id,
                        arguments_delta: json,
                    });
                }
            }
        }
        "content_block_stop" => {
            if let Some(idx) = event.index
                && let Some(id) = index_to_id.remove(&idx)
            {
                events.push(StreamEvent::ToolCallEnd { id });
            }
        }
        "message_delta" => {
            if let Some(usage) = event.usage {
                tracing::debug!(
                    output_tokens = usage.output_tokens,
                    "Anthropic stream usage (message_delta)"
                );
            }
            match event.delta.as_ref().and_then(|d| d.stop_reason.as_deref()) {
                Some("refusal") => {
                    events.push(StreamEvent::Error(
                        "Anthropic ended the stream with stop_reason=refusal (the model declined to answer)"
                            .to_string(),
                    ));
                }
                Some("pause_turn") => {
                    events.push(StreamEvent::Error(
                        "Anthropic paused the turn (stop_reason=pause_turn); continue it with a follow-up request"
                            .to_string(),
                    ));
                }
                _ => {}
            }
        }
        "error" => {
            // Mid-stream server errors (e.g. overloaded_error) arrive as SSE
            // events with HTTP 200; dropping them ended streams silently.
            let detail = event
                .error
                .map(|e| e.to_string())
                .unwrap_or_else(|| "unknown error".to_string());
            events.push(StreamEvent::Error(format!(
                "Anthropic stream error event: {detail}"
            )));
        }
        "message_stop" => {
            events.push(StreamEvent::Done);
        }
        other => {
            tracing::debug!(
                event_type = other,
                "ignoring unhandled Anthropic SSE event type"
            );
        }
    }
}

fn parse_response(response: AnthropicResponse) -> Result<ChatResponse> {
    let mut text_content = String::new();
    let mut tool_calls = Vec::new();

    for block in &response.content {
        match block {
            AnthropicResponseBlock::Text { text } => {
                text_content.push_str(text);
            }
            AnthropicResponseBlock::ToolUse { id, name, input } => {
                tool_calls.push(ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: input.clone(),
                });
            }
        }
    }

    let stop_reason = match response.stop_reason.as_deref() {
        Some("tool_use") => StopReason::ToolUse,
        Some("max_tokens") => StopReason::MaxTokens,
        Some("refusal") => StopReason::Refusal,
        Some("pause_turn") => StopReason::PauseTurn,
        _ => StopReason::EndTurn,
    };

    let message = if tool_calls.is_empty() {
        Message::assistant(text_content)
    } else {
        Message {
            role: Role::Assistant,
            content: if text_content.is_empty() {
                None
            } else {
                Some(text_content)
            },
            tool_calls,
            tool_call_id: None,
        }
    };

    Ok(ChatResponse {
        message,
        stop_reason,
        usage: response.usage.map(|u| {
            if u.cache_creation_input_tokens > 0 || u.cache_read_input_tokens > 0 {
                tracing::debug!(
                    cache_write = u.cache_creation_input_tokens,
                    cache_read = u.cache_read_input_tokens,
                    "prompt caching stats"
                );
            }
            Usage {
                input_tokens: u.input_tokens,
                output_tokens: u.output_tokens,
                cached_tokens: u.cache_read_input_tokens,
            }
        }),
    })
}

// --- Anthropic API types ---

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<serde_json::Value>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    stream: bool,
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

#[derive(Serialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

#[derive(Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<serde_json::Value>,
}

impl From<&ToolSpec> for AnthropicTool {
    fn from(spec: &ToolSpec) -> Self {
        Self {
            name: spec.name.clone(),
            description: spec.description.clone(),
            input_schema: spec.parameters.clone(),
            cache_control: None,
        }
    }
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicResponseBlock>,
    stop_reason: Option<String>,
    usage: Option<AnthropicUsage>,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum AnthropicResponseBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
    #[serde(default)]
    cache_creation_input_tokens: u32,
    #[serde(default)]
    cache_read_input_tokens: u32,
}

#[derive(Deserialize)]
struct AnthropicStreamEvent {
    r#type: String,
    /// Content-block index. Present on `content_block_start`,
    /// `content_block_delta`, and `content_block_stop`; correlates deltas to
    /// the block (and thus the tool_use id) they belong to.
    #[serde(default)]
    index: Option<u64>,
    content_block: Option<AnthropicStreamBlock>,
    delta: Option<AnthropicStreamDelta>,
    /// Cumulative usage, present on `message_delta`.
    #[serde(default)]
    usage: Option<AnthropicStreamUsage>,
    /// Error payload, present on `{"type":"error"}` events.
    #[serde(default)]
    error: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct AnthropicStreamBlock {
    r#type: String,
    id: Option<String>,
    name: Option<String>,
}

#[derive(Deserialize)]
struct AnthropicStreamDelta {
    text: Option<String>,
    partial_json: Option<String>,
    /// Final stop reason, present on `message_delta`.
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
struct AnthropicStreamUsage {
    #[serde(default)]
    output_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_response_text() {
        let raw = AnthropicResponse {
            content: vec![AnthropicResponseBlock::Text {
                text: "hello world".into(),
            }],
            stop_reason: Some("end_turn".into()),
            usage: Some(AnthropicUsage {
                input_tokens: 10,
                output_tokens: 5,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            }),
        };
        let resp = parse_response(raw).unwrap();
        assert_eq!(resp.text(), "hello world");
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        assert!(!resp.has_tool_calls());
    }

    #[test]
    fn test_parse_response_tool_use() {
        let raw = AnthropicResponse {
            content: vec![AnthropicResponseBlock::ToolUse {
                id: "tu_1".into(),
                name: "calc".into(),
                input: serde_json::json!({"expr": "2+2"}),
            }],
            stop_reason: Some("tool_use".into()),
            usage: None,
        };
        let resp = parse_response(raw).unwrap();
        assert!(resp.has_tool_calls());
        assert_eq!(resp.tool_calls()[0].name, "calc");
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
    }

    #[test]
    fn test_parse_response_mixed_content() {
        let raw = AnthropicResponse {
            content: vec![
                AnthropicResponseBlock::Text {
                    text: "Let me calculate".into(),
                },
                AnthropicResponseBlock::ToolUse {
                    id: "tu_1".into(),
                    name: "calc".into(),
                    input: serde_json::json!({}),
                },
            ],
            stop_reason: Some("tool_use".into()),
            usage: None,
        };
        let resp = parse_response(raw).unwrap();
        assert!(resp.has_tool_calls());
        assert_eq!(resp.message.content.as_deref(), Some("Let me calculate"));
    }

    #[test]
    fn test_anthropic_new_default() {
        let model = Anthropic::new("claude-sonnet-5");
        assert_eq!(model.model_id, "claude-sonnet-5");
        assert_eq!(model.base_url, DEFAULT_BASE_URL);
    }

    #[test]
    fn test_with_timeout_sets_timeout() {
        let timeout = std::time::Duration::from_secs(60);
        let model = Anthropic::with_api_key("test", "key").with_timeout(timeout);
        assert_eq!(model.timeout, Some(timeout));
    }

    #[test]
    fn test_with_max_retries_sets_retries() {
        let model = Anthropic::with_api_key("test", "key").with_max_retries(5);
        assert_eq!(model.max_retries, 5);
    }

    #[test]
    fn test_with_prompt_caching_enables_caching() {
        let model = Anthropic::with_api_key("test", "key").with_prompt_caching();
        assert!(model.use_prompt_caching);
    }

    #[test]
    fn test_builder_chain_preserves_all_options() {
        let timeout = std::time::Duration::from_secs(30);
        let model = Anthropic::with_api_key("test", "key")
            .with_base_url("https://custom.example.com")
            .with_timeout(timeout)
            .with_max_retries(2)
            .with_prompt_caching();
        assert_eq!(model.model_id, "test");
        assert_eq!(model.base_url, "https://custom.example.com");
        assert_eq!(model.timeout, Some(timeout));
        assert_eq!(model.max_retries, 2);
        assert!(model.use_prompt_caching);
    }

    #[test]
    fn test_tool_spec_conversion() {
        let spec = ToolSpec {
            name: "search".into(),
            description: "Web search".into(),
            parameters: serde_json::json!({"type": "object"}),
        };
        let tool: AnthropicTool = (&spec).into();
        assert_eq!(tool.name, "search");
        assert_eq!(tool.description, "Web search");
    }

    #[test]
    fn test_message_conversion_preserves_system_prompt() {
        let model = Anthropic::new("test");
        let request = ChatRequest {
            messages: vec![Message::system("Be helpful"), Message::user("hi")],
            tools: vec![],
            temperature: None,
            max_tokens: None,
        };
        let body = model.build_request_body(&request);
        assert_eq!(body.system.as_ref().unwrap().as_str(), Some("Be helpful"));
        assert_eq!(body.messages.len(), 1);
    }

    #[test]
    fn test_prompt_caching_system_block() {
        let model = Anthropic::with_api_key("test", "key").with_prompt_caching();
        let request = ChatRequest {
            messages: vec![Message::system("Be helpful"), Message::user("hi")],
            tools: vec![],
            temperature: None,
            max_tokens: None,
        };
        let body = model.build_request_body(&request);
        let sys = body.system.unwrap();
        assert!(
            sys.is_array(),
            "system should be array when caching enabled"
        );
        let blocks = sys.as_array().unwrap();
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[0]["text"], "Be helpful");
        assert!(blocks[0]["cache_control"].is_object());
    }

    #[test]
    fn test_assistant_text_preserved_with_tool_call() {
        // An assistant history message carrying both text and a tool call must
        // serialize a leading text block followed by the tool_use block.
        let model = Anthropic::new("test");
        let assistant = Message {
            role: Role::Assistant,
            content: Some("Let me look that up.".to_string()),
            tool_calls: vec![ToolCall {
                id: "toolu_1".into(),
                name: "search".into(),
                arguments: serde_json::json!({"q": "rust"}),
            }],
            tool_call_id: None,
        };
        let request = ChatRequest {
            messages: vec![Message::user("hi"), assistant],
            tools: vec![],
            temperature: None,
            max_tokens: None,
        };
        let body = model.build_request_body(&request);
        // Serialize and inspect the assistant message content blocks.
        let json = serde_json::to_value(&body).unwrap();
        let blocks = json["messages"][1]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 2, "text block + tool_use block");
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[0]["text"], "Let me look that up.");
        assert_eq!(blocks[1]["type"], "tool_use");
        assert_eq!(blocks[1]["id"], "toolu_1");
        assert_eq!(blocks[1]["name"], "search");
    }

    #[test]
    fn test_assistant_tool_call_without_text_has_no_text_block() {
        let model = Anthropic::new("test");
        let assistant = Message::assistant_with_tool_calls(vec![ToolCall {
            id: "toolu_1".into(),
            name: "search".into(),
            arguments: serde_json::json!({}),
        }]);
        let request = ChatRequest {
            messages: vec![assistant],
            tools: vec![],
            temperature: None,
            max_tokens: None,
        };
        let body = model.build_request_body(&request);
        let json = serde_json::to_value(&body).unwrap();
        let blocks = json["messages"][0]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 1, "only the tool_use block");
        assert_eq!(blocks[0]["type"], "tool_use");
    }

    #[test]
    fn test_prompt_caching_tool_cache_control() {
        let model = Anthropic::with_api_key("test", "key").with_prompt_caching();
        let request = ChatRequest {
            messages: vec![Message::user("hi")],
            tools: vec![
                ToolSpec {
                    name: "a".into(),
                    description: "first".into(),
                    parameters: serde_json::json!({"type": "object"}),
                },
                ToolSpec {
                    name: "b".into(),
                    description: "second".into(),
                    parameters: serde_json::json!({"type": "object"}),
                },
            ],
            temperature: None,
            max_tokens: None,
        };
        let body = model.build_request_body(&request);
        let tools = body.tools.unwrap();
        assert!(
            tools[0].cache_control.is_none(),
            "only last tool gets cache_control"
        );
        assert!(
            tools[1].cache_control.is_some(),
            "last tool should have cache_control"
        );
    }

    // --- Streaming state-machine tests ---
    //
    // These exercise `handle_anthropic_stream_event` directly, feeding events
    // deserialized from the exact JSON shapes Anthropic emits over SSE. This is
    // the logic that previously lived (untested) inside the `try_stream!` block
    // and that emitted tool-call argument deltas with an empty id before the
    // `index -> id` correlation map was added.

    /// Deserializes one SSE event payload the way the streaming loop does.
    fn event(json: &str) -> AnthropicStreamEvent {
        serde_json::from_str(json).expect("valid AnthropicStreamEvent JSON")
    }

    #[test]
    fn test_stream_event_tool_use_start_maps_index_to_id() {
        let mut idx = HashMap::new();
        let out = handle_anthropic_stream_event(
            &mut idx,
            event(
                r#"{"type":"content_block_start","index":0,
                    "content_block":{"type":"tool_use","id":"toolu_A","name":"f","input":{}}}"#,
            ),
        );
        assert_eq!(out.len(), 1);
        match &out[0] {
            StreamEvent::ToolCallStart { id, name } => {
                assert_eq!(id, "toolu_A");
                assert_eq!(name, "f");
            }
            other => panic!("expected ToolCallStart, got {other:?}"),
        }
        // The start must record the index so later deltas resolve to it.
        assert_eq!(idx.get(&0).map(String::as_str), Some("toolu_A"));
    }

    #[test]
    fn test_stream_event_deltas_carry_correlated_id_and_concatenate() {
        let mut idx = HashMap::new();
        handle_anthropic_stream_event(
            &mut idx,
            event(
                r#"{"type":"content_block_start","index":0,
                    "content_block":{"type":"tool_use","id":"toolu_A","name":"f","input":{}}}"#,
            ),
        );

        let mut assembled = String::new();
        for fragment in [r#"{\"x\":"#, "1}"] {
            let json = format!(
                r#"{{"type":"content_block_delta","index":0,
                     "delta":{{"type":"input_json_delta","partial_json":"{fragment}"}}}}"#
            );
            let out = handle_anthropic_stream_event(&mut idx, event(&json));
            assert_eq!(out.len(), 1);
            match &out[0] {
                StreamEvent::ToolCallDelta {
                    id,
                    arguments_delta,
                } => {
                    // Every delta must carry the correlated id, not "".
                    assert_eq!(
                        id, "toolu_A",
                        "delta must resolve to the opening block's id"
                    );
                    assembled.push_str(arguments_delta);
                }
                other => panic!("expected ToolCallDelta, got {other:?}"),
            }
        }
        assert_eq!(assembled, r#"{"x":1}"#);
    }

    #[test]
    fn test_stream_event_stop_emits_end_and_clears_mapping() {
        let mut idx = HashMap::new();
        handle_anthropic_stream_event(
            &mut idx,
            event(
                r#"{"type":"content_block_start","index":0,
                    "content_block":{"type":"tool_use","id":"toolu_A","name":"f","input":{}}}"#,
            ),
        );
        let out = handle_anthropic_stream_event(
            &mut idx,
            event(r#"{"type":"content_block_stop","index":0}"#),
        );
        assert_eq!(out.len(), 1);
        match &out[0] {
            StreamEvent::ToolCallEnd { id } => assert_eq!(id, "toolu_A"),
            other => panic!("expected ToolCallEnd, got {other:?}"),
        }
        // The mapping is consumed on stop so a reused index can't leak the id.
        assert!(idx.is_empty(), "index mapping must be cleared on stop");
    }

    #[test]
    fn test_stream_event_full_tool_call_lifecycle() {
        let mut idx = HashMap::new();
        let mut events = Vec::new();
        for json in [
            r#"{"type":"message_start"}"#,
            r#"{"type":"content_block_start","index":0,
                "content_block":{"type":"tool_use","id":"toolu_A","name":"f","input":{}}}"#,
            r#"{"type":"content_block_delta","index":0,
                "delta":{"type":"input_json_delta","partial_json":"{\"x\":"}}"#,
            r#"{"type":"content_block_delta","index":0,
                "delta":{"type":"input_json_delta","partial_json":"1}"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"message_delta"}"#,
            r#"{"type":"message_stop"}"#,
        ] {
            events.extend(handle_anthropic_stream_event(&mut idx, event(json)));
        }
        // message_start / message_delta produce nothing; the rest form the
        // ordered lifecycle: Start, Delta, Delta, End, Done.
        assert_eq!(events.len(), 5, "got {events:?}");
        assert!(matches!(&events[0], StreamEvent::ToolCallStart { id, name }
            if id == "toolu_A" && name == "f"));
        assert!(matches!(&events[1], StreamEvent::ToolCallDelta { id, .. } if id == "toolu_A"));
        assert!(matches!(&events[2], StreamEvent::ToolCallDelta { id, .. } if id == "toolu_A"));
        assert!(matches!(&events[3], StreamEvent::ToolCallEnd { id } if id == "toolu_A"));
        assert!(matches!(&events[4], StreamEvent::Done));
    }

    #[test]
    fn test_stream_event_text_block_index_not_confused_with_tool() {
        // A text block occupies index 0; a tool_use opens at index 1. The text
        // index must never be mapped to a tool id, and the tool's deltas at
        // index 1 must resolve to that tool's id.
        let mut idx = HashMap::new();

        // Text block at index 0 — start is a no-op for the tool map.
        let start_text = handle_anthropic_stream_event(
            &mut idx,
            event(
                r#"{"type":"content_block_start","index":0,
                     "content_block":{"type":"text","text":""}}"#,
            ),
        );
        assert!(
            start_text.is_empty(),
            "text block start yields no StreamEvent"
        );
        assert!(
            idx.is_empty(),
            "text block index must not be mapped to a tool id"
        );

        // Text delta at index 0.
        let text_delta = handle_anthropic_stream_event(
            &mut idx,
            event(
                r#"{"type":"content_block_delta","index":0,
                     "delta":{"type":"text_delta","text":"hi"}}"#,
            ),
        );
        assert!(matches!(text_delta.as_slice(), [StreamEvent::TextDelta(t)] if t == "hi"));

        // tool_use opens at index 1.
        handle_anthropic_stream_event(
            &mut idx,
            event(
                r#"{"type":"content_block_start","index":1,
                     "content_block":{"type":"tool_use","id":"toolu_B","name":"g","input":{}}}"#,
            ),
        );
        let tool_delta = handle_anthropic_stream_event(
            &mut idx,
            event(
                r#"{"type":"content_block_delta","index":1,
                     "delta":{"type":"input_json_delta","partial_json":"{}"}}"#,
            ),
        );
        assert!(
            matches!(tool_delta.as_slice(), [StreamEvent::ToolCallDelta { id, .. }] if id == "toolu_B"),
            "tool delta at index 1 must resolve to toolu_B, got {tool_delta:?}"
        );
    }

    #[test]
    fn test_parse_response_refusal_maps_to_refusal() {
        let raw = AnthropicResponse {
            content: vec![],
            stop_reason: Some("refusal".into()),
            usage: None,
        };
        let resp = parse_response(raw).unwrap();
        assert_eq!(resp.stop_reason, StopReason::Refusal);
    }

    #[test]
    fn test_parse_response_pause_turn_maps_to_pause_turn() {
        let raw = AnthropicResponse {
            content: vec![],
            stop_reason: Some("pause_turn".into()),
            usage: None,
        };
        let resp = parse_response(raw).unwrap();
        assert_eq!(resp.stop_reason, StopReason::PauseTurn);
    }

    #[test]
    fn test_debug_redacts_api_key() {
        let model = Anthropic::with_api_key("claude-sonnet-5", "sk-ant-supersecret-key");
        let dbg = format!("{model:?}");
        assert!(
            !dbg.contains("sk-ant-supersecret-key"),
            "Debug output must not contain the plaintext API key: {dbg}"
        );
        assert!(dbg.contains("[redacted]"));
    }

    #[test]
    fn test_stream_event_error_is_surfaced() {
        // {"type":"error"} events (e.g. overloaded_error mid-stream) were
        // previously dropped by the `_ => {}` arm, ending streams silently.
        let mut idx = HashMap::new();
        let out = handle_anthropic_stream_event(
            &mut idx,
            event(r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#),
        );
        assert_eq!(out.len(), 1, "got {out:?}");
        assert!(
            matches!(&out[0], StreamEvent::Error(msg) if msg.contains("overloaded_error")),
            "expected an in-band Error event, got {out:?}"
        );
    }

    #[test]
    fn test_stream_message_delta_refusal_emits_error() {
        let mut idx = HashMap::new();
        let out = handle_anthropic_stream_event(
            &mut idx,
            event(
                r#"{"type":"message_delta","delta":{"stop_reason":"refusal"},"usage":{"output_tokens":3}}"#,
            ),
        );
        assert_eq!(out.len(), 1, "got {out:?}");
        assert!(matches!(&out[0], StreamEvent::Error(msg) if msg.contains("refusal")));
    }

    #[test]
    fn test_stream_message_delta_pause_turn_emits_error() {
        let mut idx = HashMap::new();
        let out = handle_anthropic_stream_event(
            &mut idx,
            event(r#"{"type":"message_delta","delta":{"stop_reason":"pause_turn"}}"#),
        );
        assert_eq!(out.len(), 1, "got {out:?}");
        assert!(matches!(&out[0], StreamEvent::Error(msg) if msg.contains("pause_turn")));
    }

    #[test]
    fn test_stream_message_delta_normal_stop_is_silent() {
        // end_turn / tool_use terminations are the normal path; message_stop
        // provides the Done, so message_delta must not add events.
        let mut idx = HashMap::new();
        let out = handle_anthropic_stream_event(
            &mut idx,
            event(
                r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":12}}"#,
            ),
        );
        assert!(out.is_empty(), "got {out:?}");
    }

    #[test]
    fn test_stream_event_interleaved_tool_blocks_route_by_index() {
        // Two tool_use blocks open at indices 0 and 1; deltas addressed to each
        // index must route to the correct id even when interleaved.
        let mut idx = HashMap::new();
        for json in [
            r#"{"type":"content_block_start","index":0,
                "content_block":{"type":"tool_use","id":"toolu_A","name":"f","input":{}}}"#,
            r#"{"type":"content_block_start","index":1,
                "content_block":{"type":"tool_use","id":"toolu_B","name":"g","input":{}}}"#,
        ] {
            handle_anthropic_stream_event(&mut idx, event(json));
        }

        let d0 = handle_anthropic_stream_event(
            &mut idx,
            event(
                r#"{"type":"content_block_delta","index":0,
                     "delta":{"type":"input_json_delta","partial_json":"a"}}"#,
            ),
        );
        let d1 = handle_anthropic_stream_event(
            &mut idx,
            event(
                r#"{"type":"content_block_delta","index":1,
                     "delta":{"type":"input_json_delta","partial_json":"b"}}"#,
            ),
        );
        let d0b = handle_anthropic_stream_event(
            &mut idx,
            event(
                r#"{"type":"content_block_delta","index":0,
                     "delta":{"type":"input_json_delta","partial_json":"c"}}"#,
            ),
        );

        assert!(matches!(d0.as_slice(),
            [StreamEvent::ToolCallDelta { id, arguments_delta }] if id == "toolu_A" && arguments_delta == "a"));
        assert!(matches!(d1.as_slice(),
            [StreamEvent::ToolCallDelta { id, arguments_delta }] if id == "toolu_B" && arguments_delta == "b"));
        assert!(matches!(d0b.as_slice(),
            [StreamEvent::ToolCallDelta { id, arguments_delta }] if id == "toolu_A" && arguments_delta == "c"));
    }
}
