//! Shared HTTP client core for OpenAI-compatible chat/embedding endpoints.
//!
//! Used by [`crate::llamacpp`], [`crate::llamars`], and [`crate::generic`] —
//! each owns its own request-shaping and defaults, and calls into this module
//! for the wire format, SSE parsing, retry, and error surfacing they share.

use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};

use daimon_core::stream_util::LineBuffer;
use daimon_core::{
    ChatResponse, DaimonError, Message, ResponseStream, Result, Role, StopReason, StreamEvent,
    ToolCall, ToolSpec, Usage,
};

/// Upper bound on establishing a TCP connection. Applied unconditionally so a
/// dead or unreachable upstream fails fast instead of blocking forever; it
/// does not bound the request itself, so long streaming generations are
/// unaffected.
pub(crate) const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

pub(crate) fn build_client(timeout: Option<Duration>) -> Client {
    let mut builder = Client::builder().connect_timeout(DEFAULT_CONNECT_TIMEOUT);
    if let Some(t) = timeout {
        builder = builder.timeout(t);
    }
    builder.build().expect("failed to build HTTP client")
}

/// HTTP transport shared by every OpenAI-compatible local provider.
pub(crate) struct Http {
    client: Client,
    base_url: String,
    api_key: Option<String>,
    timeout: Option<Duration>,
}

impl Http {
    pub(crate) fn new(default_base_url: &str) -> Self {
        Self {
            client: build_client(None),
            base_url: default_base_url.to_string(),
            api_key: None,
            timeout: None,
        }
    }

    pub(crate) fn set_base_url(&mut self, url: impl Into<String>) {
        self.base_url = url.into().trim_end_matches('/').to_string();
    }

    pub(crate) fn set_api_key(&mut self, key: impl Into<String>) {
        self.api_key = Some(key.into());
    }

    pub(crate) fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = Some(timeout);
        self.client = build_client(Some(timeout));
    }

    #[cfg(test)]
    pub(crate) fn base_url(&self) -> &str {
        &self.base_url
    }

    #[cfg(test)]
    pub(crate) fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }

    #[cfg(test)]
    pub(crate) fn timeout(&self) -> Option<Duration> {
        self.timeout
    }

    pub(crate) async fn post(
        &self,
        path: &str,
        body: &impl Serialize,
    ) -> Result<reqwest::Response> {
        let mut req = self
            .client
            .post(format!("{}{path}", self.base_url))
            .json(body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        req.send()
            .await
            .map_err(|e| DaimonError::Model(format!("HTTP error: {e}")))
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

/// Surfaces the server's error body verbatim so failures (grammar errors,
/// bad chat templates, auth failures) are diagnosable from the error message
/// alone.
pub(crate) fn api_error(status: reqwest::StatusCode, body: &str, provider: &str) -> DaimonError {
    DaimonError::Model(format!("{provider} API error ({status}): {body}"))
}

#[derive(Serialize)]
pub(crate) struct ChatCompletionRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) model: Option<String>,
    pub(crate) messages: Vec<ApiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tools: Option<Vec<ApiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_tokens: Option<u32>,
    pub(crate) stream: bool,
    #[serde(flatten)]
    pub(crate) extra: serde_json::Map<String, serde_json::Value>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_chat_request(
    messages: &[Message],
    tools: &[ToolSpec],
    model: Option<&str>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    stream: bool,
    extra: serde_json::Map<String, serde_json::Value>,
) -> ChatCompletionRequest {
    let api_messages: Vec<ApiMessage> = messages.iter().map(Into::into).collect();
    let api_tools: Option<Vec<ApiTool>> = if tools.is_empty() {
        None
    } else {
        Some(tools.iter().map(Into::into).collect())
    };

    ChatCompletionRequest {
        model: model.map(str::to_string),
        messages: api_messages,
        tools: api_tools,
        temperature,
        max_tokens,
        stream,
        extra,
    }
}

#[derive(Serialize)]
pub(crate) struct ApiMessage {
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
pub(crate) struct ApiTool {
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

#[derive(Serialize, Deserialize)]
struct ApiToolCall {
    id: String,
    r#type: String,
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
struct ChatCompletionResponse {
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

#[derive(Deserialize)]
struct ApiResponseToolCall {
    #[serde(default)]
    id: String,
    #[serde(default)]
    function: ApiFunction,
}

#[derive(Deserialize)]
struct ApiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

/// Maps an OpenAI-shaped `finish_reason` to a [`StopReason`].
///
/// `"content_filter"` maps to [`StopReason::ContentFiltered`] so safety stops
/// from OpenAI-compatible local servers stop masquerading as normal
/// completions (matching the facade's OpenAI provider, DAIM-9).
pub(crate) fn finish_reason_to_stop(reason: Option<&str>) -> StopReason {
    match reason {
        Some("tool_calls") => StopReason::ToolUse,
        Some("length") => StopReason::MaxTokens,
        Some("content_filter") => StopReason::ContentFiltered,
        _ => StopReason::EndTurn,
    }
}

pub(crate) fn parse_chat_response(body: &[u8], provider: &str) -> Result<ChatResponse> {
    let parsed: ChatCompletionResponse = serde_json::from_slice(body)
        .map_err(|e| DaimonError::Model(format!("{provider} response parse error: {e}")))?;

    let choice = parsed
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| DaimonError::Model(format!("no choices in {provider} response")))?;

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

    let stop_reason = finish_reason_to_stop(choice.finish_reason.as_deref());

    Ok(ChatResponse {
        message: Message {
            role: Role::Assistant,
            content: choice.message.content,
            tool_calls,
            tool_call_id: None,
        },
        stop_reason,
        usage: parsed.usage.map(|u| Usage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            cached_tokens: 0,
        }),
    })
}

#[derive(Deserialize)]
struct StreamChunk {
    choices: Vec<StreamChoice>,
}

#[derive(Deserialize)]
struct StreamChoice {
    delta: StreamDelta,
    #[serde(default)]
    finish_reason: Option<String>,
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

/// Parses one SSE line into the stream events it carries, appending them to
/// `events`. Reused across a whole stream by callers so no per-line `Vec` is
/// allocated (DAIM-18 convention).
///
/// Non-`data:` lines (comments, blank keep-alives) and unparseable payloads
/// yield no events, matching the other providers' tolerance for unknown
/// server chatter. `finish_reason: "content_filter"` additionally emits a
/// [`StreamEvent::Error`] since [`StreamEvent::Done`] carries no stop reason.
pub(crate) fn sse_line_events_into(line: &str, events: &mut Vec<StreamEvent>) {
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
        if choice.finish_reason.as_deref() == Some("content_filter") {
            events.push(StreamEvent::Error(
                "response blocked (finish_reason=content_filter)".to_string(),
            ));
        }
    }
}

/// Consumes a successful chat-completion HTTP response's SSE body into a
/// [`ResponseStream`] of [`StreamEvent`]s.
///
/// Shared by every `openai_compat`-based provider's `generate_stream` so the
/// buffer/parse/drain loop is written once instead of once per provider.
/// Takes the whole [`reqwest::Response`] (rather than a bare byte stream) so
/// this crate never has to name `bytes::Bytes` as its own dependency — the
/// type stays internal to `reqwest`. `provider` only appears in the
/// transport-error message. Caller has already checked `response.status()`.
pub(crate) fn stream_chat_response(
    response: reqwest::Response,
    provider: &'static str,
) -> ResponseStream {
    let stream = async_stream::try_stream! {
        use futures::StreamExt;

        let mut buffer = LineBuffer::new();
        let mut byte_stream = Box::pin(response.bytes_stream());
        // Reused across the whole stream so the parse path allocates no
        // per-line Vec; drained after each line.
        let mut events: Vec<StreamEvent> = Vec::new();

        while let Some(chunk) = byte_stream.next().await {
            let chunk = chunk
                .map_err(|e| DaimonError::Model(format!("{provider} stream error: {e}")))?;
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

    Box::pin(stream)
}

#[derive(Serialize)]
pub(crate) struct EmbedRequest<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) model: Option<&'a str>,
    pub(crate) input: &'a [&'a str],
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedDatum>,
}

#[derive(Deserialize)]
struct EmbedDatum {
    embedding: Vec<f32>,
}

pub(crate) fn parse_embed_response(body: &[u8], provider: &str) -> Result<Vec<Vec<f32>>> {
    let parsed: EmbedResponse = serde_json::from_slice(body)
        .map_err(|e| DaimonError::Model(format!("{provider} embedding parse error: {e}")))?;
    Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
}

/// Test-facing wrapper that returns the events a line produces, rather than
/// appending to a caller-owned buffer.
#[cfg(test)]
pub(crate) fn sse_line_events(line: &str) -> Vec<StreamEvent> {
    let mut events = Vec::new();
    sse_line_events_into(line, &mut events);
    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use daimon_core::Message;

    #[test]
    fn test_finish_reason_mapping() {
        assert_eq!(
            finish_reason_to_stop(Some("tool_calls")),
            StopReason::ToolUse
        );
        assert_eq!(finish_reason_to_stop(Some("length")), StopReason::MaxTokens);
        assert_eq!(
            finish_reason_to_stop(Some("content_filter")),
            StopReason::ContentFiltered
        );
        assert_eq!(finish_reason_to_stop(Some("stop")), StopReason::EndTurn);
        assert_eq!(finish_reason_to_stop(None), StopReason::EndTurn);
    }

    #[test]
    fn test_build_chat_request_basic() {
        let messages = vec![Message::user("hi")];
        let req = build_chat_request(&messages, &[], None, None, None, false, Default::default());
        let value = serde_json::to_value(&req).unwrap();
        assert_eq!(value["messages"][0]["role"], "user");
        assert_eq!(value["messages"][0]["content"], "hi");
        assert_eq!(value["stream"], false);
        assert!(value.as_object().unwrap().get("model").is_none());
    }

    #[test]
    fn test_build_chat_request_with_extra_fields() {
        let mut extra = serde_json::Map::new();
        extra.insert("grammar".to_string(), serde_json::json!("root ::= \"yes\""));
        let messages = vec![Message::user("hi")];
        let req = build_chat_request(&messages, &[], Some("qwen3"), None, None, false, extra);
        let value = serde_json::to_value(&req).unwrap();
        assert_eq!(value["model"], "qwen3");
        assert_eq!(value["grammar"], "root ::= \"yes\"");
    }

    #[test]
    fn test_parse_chat_response_end_turn() {
        let body = br#"{
            "choices": [{
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }"#;
        let resp = parse_chat_response(body, "test").unwrap();
        assert_eq!(resp.message.content.as_deref(), Some("hello"));
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        let usage = resp.usage.unwrap();
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
    }

    #[test]
    fn test_parse_chat_response_content_filter() {
        let body =
            br#"{"choices": [{"message": {"content": null}, "finish_reason": "content_filter"}]}"#;
        let resp = parse_chat_response(body, "test").unwrap();
        assert_eq!(resp.stop_reason, StopReason::ContentFiltered);
    }

    #[test]
    fn test_parse_chat_response_no_choices() {
        let body = br#"{"choices": []}"#;
        assert!(parse_chat_response(body, "test").is_err());
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
    }

    #[test]
    fn test_sse_content_filter_emits_error() {
        let events =
            sse_line_events(r#"data: {"choices":[{"delta":{},"finish_reason":"content_filter"}]}"#);
        assert!(events.iter().any(|e| matches!(e, StreamEvent::Error(_))));
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
            "test",
        );
        let text = err.to_string();
        assert!(text.contains("400"));
        assert!(text.contains("failed to parse grammar"));
    }

    #[test]
    fn test_debug_redacts_api_key() {
        let mut http = Http::new("http://localhost:1234");
        http.set_api_key("sk-supersecret-key-value");
        let dbg = format!("{http:?}");
        assert!(!dbg.contains("sk-supersecret-key-value"));
        assert!(dbg.contains("[redacted]"));
    }

    #[test]
    fn test_parse_embed_response() {
        let body = br#"{"data":[{"embedding":[0.1,0.2]}]}"#;
        let vecs = parse_embed_response(body, "test").unwrap();
        assert_eq!(vecs, vec![vec![0.1, 0.2]]);
    }
}
