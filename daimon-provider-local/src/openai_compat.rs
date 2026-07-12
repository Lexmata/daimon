//! Shared HTTP client core for OpenAI-compatible chat/embedding endpoints.
//!
//! Used by [`crate::llamacpp`], [`crate::llamars`], and [`crate::generic`] —
//! each owns its own request-shaping and defaults, and calls into this module
//! for the wire format, SSE parsing, retry, and error surfacing they share.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};

use daimon_core::stream_util::{LineBuffer, backoff_delay, parse_retry_after_secs};
use daimon_core::{
    ChatResponse, DaimonError, Message, ResponseStream, Result, Role, StopReason, StreamEvent,
    ToolCall, ToolSpec, Usage,
};

/// Upper bound on establishing a TCP connection. Applied unconditionally so a
/// dead or unreachable upstream fails fast instead of blocking forever; it
/// does not bound the request itself, so long streaming generations are
/// unaffected.
pub(crate) const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Default whole-request timeout applied to non-streaming requests
/// (`generate`, `embed`) when the caller hasn't set one explicitly.
///
/// Without this, a server that accepts the TCP connection but stalls
/// mid-response hangs forever — `connect_timeout` only bounds connection
/// establishment. Matches the facade's `daimon-provider-openai` convention.
/// Never applied to `generate_stream`: a whole-request deadline would abort a
/// healthy, long-lived SSE stream.
pub(crate) const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// Upper bound on the streaming handshake — the `send()` call that completes
/// once response headers arrive, before any body bytes are consumed.
///
/// `post_streaming` deliberately passes no whole-request timeout so a
/// healthy long-lived SSE stream is never aborted mid-body. But that leaves
/// nothing bounding the time between a successful TCP/TLS connect and the
/// server actually sending headers — a server that accepts the connection
/// and then never responds (e.g. still loading a model) would hang forever,
/// before the retry loop ever got a chance to engage. This bounds only that
/// handshake window; once bytes start flowing, consumption is unbounded.
pub(crate) const DEFAULT_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);

/// Default number of retries for transient (429 / 5xx) errors on the initial
/// request. Matches [`crate::ollama_embed`]'s `DEFAULT_MAX_RETRIES`.
pub(crate) const DEFAULT_MAX_RETRIES: u32 = 3;

pub(crate) fn build_client() -> Client {
    // No client-wide request timeout: the per-request timeout is applied at
    // send time in `Http::post`/`Http::post_streaming` instead, so streaming
    // and non-streaming requests can have independent policies from the same
    // client.
    Client::builder()
        .connect_timeout(DEFAULT_CONNECT_TIMEOUT)
        .build()
        .expect("failed to build HTTP client")
}

/// HTTP transport shared by every OpenAI-compatible local provider.
pub(crate) struct Http {
    client: Client,
    base_url: String,
    api_key: Option<String>,
    timeout: Option<Duration>,
    max_retries: u32,
    /// When `false` (the default), sending an API key over a plaintext
    /// `http://` base URL is a hard configuration error. Set via
    /// [`Http::set_allow_plaintext_api_key`] to opt back into warn-and-send
    /// for genuinely local, unauthenticated-but-keyed servers.
    allow_plaintext_api_key: bool,
    /// Set once an `http://` + API key combination has been warned about (in
    /// the opt-in-allowed case), so repeated requests don't spam logs.
    warned_plaintext_key: AtomicBool,
}

impl Http {
    pub(crate) fn new(default_base_url: &str) -> Self {
        Self {
            client: build_client(),
            base_url: default_base_url.to_string(),
            api_key: None,
            timeout: None,
            max_retries: DEFAULT_MAX_RETRIES,
            allow_plaintext_api_key: false,
            warned_plaintext_key: AtomicBool::new(false),
        }
    }

    pub(crate) fn set_base_url(&mut self, url: impl Into<String>) {
        self.base_url = url.into().trim_end_matches('/').to_string();
    }

    pub(crate) fn set_api_key(&mut self, key: impl Into<String>) {
        self.api_key = Some(key.into());
    }

    /// Sets the whole-request timeout applied to non-streaming requests
    /// (default: 120s if never called). Never applied to `generate_stream`.
    pub(crate) fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = Some(timeout);
    }

    /// Sets the maximum number of retries for transient errors (default: 3).
    pub(crate) fn set_max_retries(&mut self, retries: u32) {
        self.max_retries = retries;
    }

    /// Opts back into warn-and-send-anyway for an API key over a plaintext
    /// `http://` base URL (default: hard error). Intended only for genuinely
    /// local, unauthenticated-but-keyed servers where the caller has made a
    /// deliberate choice to accept the cleartext exposure.
    pub(crate) fn set_allow_plaintext_api_key(&mut self, allow: bool) {
        self.allow_plaintext_api_key = allow;
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

    #[cfg(test)]
    pub(crate) fn max_retries(&self) -> u32 {
        self.max_retries
    }

    /// Refuses (by default) to send an API key over a plaintext `http://`
    /// base URL — a bearer token sent in cleartext is trivially sniffable by
    /// anything on the network path. When [`Http::set_allow_plaintext_api_key`]
    /// has opted back in, this instead logs a one-time warning and allows the
    /// request through, for genuinely local unauthenticated-but-keyed
    /// servers.
    ///
    /// The scheme check is case-insensitive (`HTTP://` is just as plaintext
    /// as `http://`).
    fn check_plaintext_key(&self) -> Result<()> {
        if self.api_key.is_none() || !self.base_url.to_ascii_lowercase().starts_with("http://") {
            return Ok(());
        }

        if !self.allow_plaintext_api_key {
            return Err(DaimonError::Builder(format!(
                "refusing to send API key over plaintext HTTP to {} — use https://, or call \
                 .allow_plaintext_api_key() to opt into cleartext for a genuinely local, \
                 unauthenticated-but-keyed server",
                self.base_url
            )));
        }

        if !self.warned_plaintext_key.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                base_url = %self.base_url,
                "sending API key over plaintext HTTP to {} (allowed via allow_plaintext_api_key)",
                self.base_url
            );
        }
        Ok(())
    }

    /// Sends a non-streaming request, retrying transient (429 / 5xx) errors
    /// with jittered backoff, and applying the default (or overridden)
    /// whole-request timeout.
    ///
    /// Always resolves to `Ok` once a response (successful or not) is
    /// obtained — callers inspect `response.status()` themselves to produce
    /// provider-specific error messages, so this never surfaces an HTTP error
    /// status as an `Err` on its own.
    pub(crate) async fn post(
        &self,
        path: &str,
        body: &impl Serialize,
    ) -> Result<reqwest::Response> {
        self.post_with_retry(path, body, self.effective_timeout(false))
            .await
    }

    /// Sends the initial request/response-status-check for a streaming call,
    /// retrying transient errors exactly like [`Http::post`] but with no
    /// whole-request timeout — once bytes start flowing, `generate_stream`
    /// consumes the body itself and this function is never involved again.
    pub(crate) async fn post_streaming(
        &self,
        path: &str,
        body: &impl Serialize,
    ) -> Result<reqwest::Response> {
        self.post_with_retry(path, body, self.effective_timeout(true))
            .await
    }

    /// Resolves the whole-request timeout to apply for a given call shape.
    ///
    /// Non-streaming calls get the default (or overridden) whole-request
    /// timeout; streaming calls get `None` so a healthy long-lived SSE
    /// stream is never aborted mid-body (see [`DEFAULT_HANDSHAKE_TIMEOUT`]
    /// for the narrower bound that still applies to the streaming
    /// handshake itself).
    fn effective_timeout(&self, streaming: bool) -> Option<Duration> {
        if streaming {
            None
        } else {
            Some(self.timeout.unwrap_or(DEFAULT_REQUEST_TIMEOUT))
        }
    }

    async fn post_with_retry(
        &self,
        path: &str,
        body: &impl Serialize,
        timeout: Option<Duration>,
    ) -> Result<reqwest::Response> {
        self.check_plaintext_key()?;

        for attempt in 0..=self.max_retries {
            let mut req = self
                .client
                .post(format!("{}{path}", self.base_url))
                .json(body);
            if let Some(t) = timeout {
                req = req.timeout(t);
            }
            if let Some(key) = &self.api_key {
                req = req.bearer_auth(key);
            }

            let response = match send_with_handshake_bound(req, timeout).await {
                Ok(response) => response,
                Err(e) => {
                    if e.is_retryable() && attempt < self.max_retries {
                        let delay = backoff_delay(attempt, None);
                        tracing::debug!(
                            error = %e,
                            attempt,
                            delay_ms = delay.as_millis(),
                            "retryable transport error, backing off"
                        );
                        tokio::time::sleep(delay).await;
                        continue;
                    }
                    return Err(DaimonError::Model(format!("HTTP error: {e}")));
                }
            };

            let status = response.status();
            let is_retryable = is_retryable_status(status);

            if !is_retryable || attempt == self.max_retries {
                return Ok(response);
            }

            let retry_after = response
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .and_then(parse_retry_after_secs)
                .map(Duration::from_secs);
            let delay = backoff_delay(attempt, retry_after);
            tracing::debug!(
                status = %status,
                attempt,
                delay_ms = delay.as_millis(),
                "retryable HTTP error, backing off"
            );
            tokio::time::sleep(delay).await;
        }

        unreachable!("loop always returns")
    }
}

/// Whether an HTTP response status should be retried: 429 (rate limited) or
/// any 5xx (server error). Extracted as a pure function so the
/// retry-decision logic is directly unit-testable without standing up a
/// server.
pub(crate) fn is_retryable_status(status: reqwest::StatusCode) -> bool {
    status.as_u16() == 429 || status.is_server_error()
}

/// Whether a transport-level (pre-response) `reqwest::Error` should be
/// retried: connection refused/reset, DNS/connect failure, or a client-side
/// timeout. These are the dominant failure modes against local model
/// servers (still loading, not yet listening, or slow to accept) and
/// previously bypassed retry entirely because only HTTP-status failures
/// were classified.
pub(crate) fn is_retryable_transport_error(e: &reqwest::Error) -> bool {
    e.is_timeout() || e.is_connect() || e.is_request()
}

/// A failure from [`send_with_handshake_bound`]: either a real
/// `reqwest::Error` from `send()`, or the handshake exceeding
/// [`DEFAULT_HANDSHAKE_TIMEOUT`] with no `reqwest::Error` to classify (the
/// future was cancelled, not failed).
enum TransportFailure {
    Reqwest(reqwest::Error),
    HandshakeTimedOut,
}

impl TransportFailure {
    fn is_retryable(&self) -> bool {
        match self {
            Self::Reqwest(e) => is_retryable_transport_error(e),
            // A handshake that never got headers is exactly the "server
            // still loading" case retry exists for.
            Self::HandshakeTimedOut => true,
        }
    }
}

impl std::fmt::Display for TransportFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reqwest(e) => write!(f, "{e}"),
            Self::HandshakeTimedOut => {
                write!(
                    f,
                    "streaming handshake exceeded {DEFAULT_HANDSHAKE_TIMEOUT:?} without a response"
                )
            }
        }
    }
}

/// Sends `req`, applying [`DEFAULT_HANDSHAKE_TIMEOUT`] to the `send()` call
/// itself when `timeout` is `None` (the streaming case) — nothing else
/// bounds the time between a successful connect and the server actually
/// sending headers. When `timeout` is `Some`, the whole-request timeout
/// already covers this window, so `send()` is awaited directly.
async fn send_with_handshake_bound(
    req: reqwest::RequestBuilder,
    timeout: Option<Duration>,
) -> std::result::Result<reqwest::Response, TransportFailure> {
    if timeout.is_none() {
        match tokio::time::timeout(DEFAULT_HANDSHAKE_TIMEOUT, req.send()).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(e)) => Err(TransportFailure::Reqwest(e)),
            Err(_) => Err(TransportFailure::HandshakeTimedOut),
        }
    } else {
        req.send().await.map_err(TransportFailure::Reqwest)
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
            .field("max_retries", &self.max_retries)
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
        .map(|tc| {
            let arguments = if tc.function.arguments.trim().is_empty() {
                serde_json::Value::Null
            } else {
                serde_json::from_str(&tc.function.arguments).map_err(|e| {
                    DaimonError::Model(format!(
                        "{provider} returned malformed tool-call arguments for {:?} (id {:?}): {e}; raw: {:?}",
                        tc.function.name, tc.id, tc.function.arguments
                    ))
                })?
            };
            Ok(ToolCall {
                id: tc.id,
                name: tc.function.name,
                arguments,
            })
        })
        .collect::<Result<Vec<ToolCall>>>()?;

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
    fn test_parse_chat_response_valid_tool_call_args() {
        let body = br#"{
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{"id": "call_1", "function": {"name": "calc", "arguments": "{\"x\": 1}"}}]
                },
                "finish_reason": "tool_calls"
            }]
        }"#;
        let resp = parse_chat_response(body, "test").unwrap();
        assert_eq!(resp.message.tool_calls.len(), 1);
        assert_eq!(
            resp.message.tool_calls[0].arguments,
            serde_json::json!({"x": 1})
        );
    }

    #[test]
    fn test_parse_chat_response_malformed_tool_call_args_errors() {
        // A truncated/invalid arguments string must surface as an error, not
        // silently become `null` args (a local model emitting broken JSON
        // should not look identical to a legitimate empty-args call).
        let body = br#"{
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{"id": "call_1", "function": {"name": "calc", "arguments": "{\"x\": "}}]
                },
                "finish_reason": "tool_calls"
            }]
        }"#;
        let err = parse_chat_response(body, "test").unwrap_err();
        let text = err.to_string();
        assert!(text.contains("malformed tool-call arguments"));
        assert!(text.contains("calc"));
    }

    #[test]
    fn test_parse_chat_response_empty_tool_call_args_is_not_an_error() {
        // A tool with zero parameters (e.g. `get_current_time()`) legitimately
        // has an empty `arguments` string on the wire. That must not be
        // treated as malformed JSON; it should parse to `Value::Null`.
        let body = br#"{
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{"id": "call_1", "function": {"name": "get_current_time", "arguments": ""}}]
                },
                "finish_reason": "tool_calls"
            }]
        }"#;
        let resp = parse_chat_response(body, "test").unwrap();
        assert_eq!(resp.message.tool_calls.len(), 1);
        assert_eq!(
            resp.message.tool_calls[0].arguments,
            serde_json::Value::Null
        );
    }

    #[test]
    fn test_parse_chat_response_whitespace_only_tool_call_args_is_not_an_error() {
        // Some servers pad the empty-args case with whitespace instead of a
        // true empty string; treat it the same way.
        let body = br#"{
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{"id": "call_1", "function": {"name": "get_current_time", "arguments": "   "}}]
                },
                "finish_reason": "tool_calls"
            }]
        }"#;
        let resp = parse_chat_response(body, "test").unwrap();
        assert_eq!(resp.message.tool_calls.len(), 1);
        assert_eq!(
            resp.message.tool_calls[0].arguments,
            serde_json::Value::Null
        );
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
    fn test_http_defaults() {
        let http = Http::new("http://localhost:1234");
        assert_eq!(http.max_retries(), DEFAULT_MAX_RETRIES);
        assert_eq!(http.timeout(), None);
    }

    #[test]
    fn test_http_set_max_retries() {
        let mut http = Http::new("http://localhost:1234");
        http.set_max_retries(7);
        assert_eq!(http.max_retries(), 7);
    }

    #[test]
    fn test_http_set_timeout() {
        let mut http = Http::new("http://localhost:1234");
        http.set_timeout(Duration::from_secs(5));
        assert_eq!(http.timeout(), Some(Duration::from_secs(5)));
    }

    // --- is_retryable_status / is_retryable_transport_error (Finding 2) ---

    #[test]
    fn test_is_retryable_status_429() {
        assert!(is_retryable_status(
            reqwest::StatusCode::from_u16(429).unwrap()
        ));
    }

    #[test]
    fn test_is_retryable_status_5xx() {
        for code in 500..600 {
            assert!(
                is_retryable_status(reqwest::StatusCode::from_u16(code).unwrap()),
                "status {code} should be retryable"
            );
        }
    }

    #[test]
    fn test_is_retryable_status_4xx_non_429_not_retryable() {
        for code in [400, 401, 404] {
            assert!(
                !is_retryable_status(reqwest::StatusCode::from_u16(code).unwrap()),
                "status {code} should not be retryable"
            );
        }
    }

    #[test]
    fn test_is_retryable_status_2xx_not_retryable() {
        for code in [200, 201] {
            assert!(
                !is_retryable_status(reqwest::StatusCode::from_u16(code).unwrap()),
                "status {code} should not be retryable"
            );
        }
    }

    #[tokio::test]
    async fn test_is_retryable_transport_error_connect_refused() {
        // Bind then immediately drop: the OS frees the port but guarantees
        // nothing listens on it, so the connection attempt fails
        // synchronously with ECONNREFUSED — a real transport-level error,
        // not a mock.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let client = reqwest::Client::new();
        let err = client
            .get(format!("http://{addr}"))
            .send()
            .await
            .unwrap_err();
        assert!(err.is_connect());
        assert!(is_retryable_transport_error(&err));
    }

    #[tokio::test]
    async fn test_is_retryable_transport_error_timeout() {
        // A listener that accepts the TCP connection but never writes a
        // response forces a real client-side timeout, distinct from a
        // connection refusal.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let _ = listener.accept();
            std::thread::sleep(Duration::from_secs(5));
        });

        let client = reqwest::Client::new();
        let err = client
            .get(format!("http://{addr}"))
            .timeout(Duration::from_millis(100))
            .send()
            .await
            .unwrap_err();
        // Older/newer reqwest versions classify a request-phase timeout
        // under slightly different `Error::is_*` flags (`is_timeout` vs
        // `is_request`); what matters for retry purposes is that our
        // classifier treats it as retryable either way.
        assert!(is_retryable_transport_error(&err), "err was: {err:?}");
    }

    // --- effective_timeout (Finding 3) ---

    #[test]
    fn test_effective_timeout_non_streaming_default() {
        let http = Http::new("http://localhost:1234");
        assert_eq!(http.effective_timeout(false), Some(DEFAULT_REQUEST_TIMEOUT));
    }

    #[test]
    fn test_effective_timeout_non_streaming_overridden() {
        let mut http = Http::new("http://localhost:1234");
        http.set_timeout(Duration::from_secs(5));
        assert_eq!(http.effective_timeout(false), Some(Duration::from_secs(5)));
    }

    #[test]
    fn test_effective_timeout_streaming_is_unbounded() {
        let http = Http::new("http://localhost:1234");
        assert_eq!(http.effective_timeout(true), None);
    }

    // --- connection-level retry, end-to-end (Finding 1) ---

    #[tokio::test]
    async fn test_post_retries_transport_errors_then_succeeds() {
        use std::io::Write;
        use std::net::TcpListener;
        use std::sync::Arc;
        use std::sync::atomic::AtomicUsize;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let accept_count = Arc::new(AtomicUsize::new(0));
        let accept_count_thread = Arc::clone(&accept_count);

        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let n = accept_count_thread.fetch_add(1, Ordering::SeqCst);
                let Ok(mut stream) = stream else { continue };
                if n < 2 {
                    // First two connections: drop immediately without ever
                    // writing a response, simulating a server that resets
                    // the connection (still loading, out of worker slots).
                    drop(stream);
                } else {
                    // Read (and discard) the client's request headers before
                    // responding — writing a response before the client has
                    // finished sending its request confuses hyper's
                    // client-side parser (observed as a spurious
                    // `Canceled`/`UnexpectedMessage` error unrelated to what
                    // this test exercises).
                    let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());
                    loop {
                        use std::io::BufRead;
                        let mut line = String::new();
                        if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                            break;
                        }
                    }

                    let body =
                        r#"{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}"#;
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = stream.write_all(response.as_bytes());
                    break;
                }
            }
        });

        let mut http = Http::new(&format!("http://{addr}"));
        http.set_max_retries(3);
        let response = http
            .post("/v1/chat/completions", &serde_json::json!({}))
            .await
            .expect("should succeed after retrying connection-level failures");
        assert!(response.status().is_success());
        assert!(accept_count.load(Ordering::SeqCst) >= 3);
    }

    #[tokio::test]
    async fn test_post_gives_up_after_max_retries_on_persistent_connection_refused() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let mut http = Http::new(&format!("http://{addr}"));
        http.set_max_retries(1);
        let err = http
            .post("/v1/chat/completions", &serde_json::json!({}))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("HTTP error"));
    }

    // --- plaintext API key hard-block (Finding 5) ---

    #[test]
    fn test_plaintext_api_key_blocked_by_default() {
        let mut http = Http::new("http://localhost:1234");
        http.set_api_key("secret");
        let err = http.check_plaintext_key().unwrap_err();
        assert!(matches!(err, DaimonError::Builder(_)));
    }

    #[test]
    fn test_plaintext_api_key_blocked_case_insensitive_scheme() {
        let mut http = Http::new("HTTP://localhost:1234");
        http.set_api_key("secret");
        assert!(http.check_plaintext_key().is_err());
    }

    #[test]
    fn test_plaintext_api_key_allowed_when_opted_in() {
        let mut http = Http::new("http://localhost:1234");
        http.set_api_key("secret");
        http.set_allow_plaintext_api_key(true);
        assert!(http.check_plaintext_key().is_ok());
    }

    #[test]
    fn test_plaintext_api_key_not_blocked_over_https() {
        let mut http = Http::new("https://localhost:1234");
        http.set_api_key("secret");
        assert!(http.check_plaintext_key().is_ok());
    }

    #[test]
    fn test_plaintext_key_not_blocked_without_api_key() {
        let http = Http::new("http://localhost:1234");
        assert!(http.check_plaintext_key().is_ok());
    }

    #[test]
    fn test_parse_embed_response() {
        let body = br#"{"data":[{"embedding":[0.1,0.2]}]}"#;
        let vecs = parse_embed_response(body, "test").unwrap();
        assert_eq!(vecs, vec![vec![0.1, 0.2]]);
    }

    // --- Property tests ---
    //
    // `parse_chat_response` and `sse_line_events_into` both parse bytes that
    // came straight from a remote server's HTTP response body — untrusted
    // input. These don't assert anything about the parsed content; they
    // assert that no arbitrary input can panic the parser (only a
    // `Result::Err`, or well-formed output, is acceptable).

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn parse_chat_response_never_panics(body in prop::collection::vec(any::<u8>(), 0..256)) {
            let _ = parse_chat_response(&body, "test");
        }

        #[test]
        fn parse_chat_response_never_panics_on_arbitrary_json(body in "\\PC{0,256}") {
            let _ = parse_chat_response(body.as_bytes(), "test");
        }

        #[test]
        fn sse_line_events_into_never_panics(line in "\\PC{0,256}") {
            let mut events = Vec::new();
            sse_line_events_into(&line, &mut events);
            // No panic occurred: that is the property under test.
        }
    }
}
