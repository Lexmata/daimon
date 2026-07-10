//! Google Gemini model provider for the [Daimon](https://docs.rs/daimon) agent framework.
//!
//! Supports the Generative AI endpoint and Vertex AI, tool use, SSE streaming,
//! configurable timeouts, retries with exponential backoff, and cached content.
//!
//! # Example
//!
//! ```ignore
//! use daimon_provider_gemini::Gemini;
//! use daimon_core::Model;
//!
//! let model = Gemini::new("gemini-2.0-flash");
//! ```

use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};

mod embedding;
mod stream_util;

#[cfg(feature = "pubsub")]
pub mod pubsub;

pub use embedding::GeminiEmbedding;

#[cfg(feature = "pubsub")]
pub use pubsub::PubSubBroker;

use daimon_core::{
    ChatRequest, ChatResponse, DaimonError, Message, Model, ResponseStream, Result, Role,
    StopReason, StreamEvent, ToolCall, ToolSpec, Usage,
};

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
const DEFAULT_MAX_RETRIES: u32 = 3;

/// Default whole-request timeout applied to non-streaming `generate` calls.
///
/// Chat completions can legitimately run for minutes on long outputs, so this
/// is deliberately generous. Override with [`Gemini::with_timeout`].
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// Default TCP/TLS connect timeout applied to the underlying HTTP client.
///
/// This bounds only connection establishment, so it is safe for the
/// long-lived SSE streams produced by `generate_stream` (a whole-request
/// timeout would kill a healthy stream mid-response).
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Prefix used for synthesized tool-call ids.
///
/// Gemini does not assign tool-call ids, so this provider synthesizes them as
/// `gemini_{seq}_{name}` — the sequence number comes first so the function
/// name (which may itself contain digits and underscores) is unambiguously
/// recoverable when the id is echoed back as a `tool_call_id`.
const TOOL_CALL_ID_PREFIX: &str = "gemini_";

fn build_client() -> Client {
    Client::builder()
        .connect_timeout(DEFAULT_CONNECT_TIMEOUT)
        .build()
        .expect("failed to build HTTP client")
}

/// Synthesizes a tool-call id in the `gemini_{seq}_{name}` format.
fn make_tool_call_id(seq: u64, name: &str) -> String {
    format!("{TOOL_CALL_ID_PREFIX}{seq}_{name}")
}

/// Recovers the function name from a synthetic `gemini_{seq}_{name}` id.
///
/// Returns `None` when the id does not follow the synthetic format, in which
/// case the caller should fall back to treating the id as the name itself
/// (callers may construct tool results with the plain function name).
fn function_name_from_tool_call_id(id: &str) -> Option<&str> {
    let rest = id.strip_prefix(TOOL_CALL_ID_PREFIX)?;
    let (seq, name) = rest.split_once('_')?;
    if seq.is_empty() || !seq.bytes().all(|b| b.is_ascii_digit()) || name.is_empty() {
        return None;
    }
    Some(name)
}

/// Google Gemini model provider.
///
/// Connects to the Gemini REST API. Supports both the public Generative AI
/// endpoint (default) and Vertex AI via `with_base_url()`. Authentication is
/// via API key (sent as the `x-goog-api-key` header, never in the URL, so the
/// key cannot leak into logs through request URLs) or bearer token for Vertex
/// AI.
pub struct Gemini {
    client: Client,
    api_key: String,
    model_id: String,
    base_url: String,
    timeout: Option<Duration>,
    max_retries: u32,
    use_bearer_token: bool,
    cached_content: Option<String>,
}

impl std::fmt::Debug for Gemini {
    /// Hand-written to avoid leaking the plaintext API key (or Vertex bearer
    /// token) in logs or panic output; a derived `Debug` would print it verbatim.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gemini")
            .field("client", &self.client)
            .field("api_key", &"[redacted]")
            .field("model_id", &self.model_id)
            .field("base_url", &self.base_url)
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .field("use_bearer_token", &self.use_bearer_token)
            .field("cached_content", &self.cached_content)
            .finish()
    }
}

impl Gemini {
    /// Create a new Gemini client, reading `GOOGLE_API_KEY` from the environment.
    ///
    /// The constructor never fails; if the environment variable is unset or
    /// empty a warning is logged and requests will fail with an auth error.
    pub fn new(model_id: impl Into<String>) -> Self {
        let api_key = std::env::var("GOOGLE_API_KEY").unwrap_or_default();
        if api_key.is_empty() {
            tracing::warn!(
                "GOOGLE_API_KEY is not set or empty; Gemini requests will fail authentication"
            );
        }
        Self::with_api_key(model_id, api_key)
    }

    /// Create a new Gemini client with an explicit API key.
    pub fn with_api_key(model_id: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            client: build_client(),
            api_key: api_key.into(),
            model_id: model_id.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            timeout: None,
            max_retries: DEFAULT_MAX_RETRIES,
            use_bearer_token: false,
            cached_content: None,
        }
    }

    /// Set a custom base URL (e.g. for Vertex AI endpoints).
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set an HTTP timeout for non-streaming requests (default: 120s).
    ///
    /// The timeout applies per-request to `generate`; `generate_stream` is a
    /// long-lived SSE connection and is protected only by the client's connect
    /// timeout, since a whole-request deadline would abort healthy streams.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set the maximum number of retries for transient errors.
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Use `Authorization: Bearer <key>` instead of the `x-goog-api-key` header.
    ///
    /// Required for Vertex AI endpoints where the key is an OAuth2 access token.
    pub fn with_bearer_token(mut self) -> Self {
        self.use_bearer_token = true;
        self
    }

    /// Reference a previously-created cached content resource.
    ///
    /// The name should be in the format `cachedContents/<id>`, as returned
    /// by the Gemini Caching API.
    pub fn with_cached_content(mut self, name: impl Into<String>) -> Self {
        self.cached_content = Some(name.into());
        self
    }

    fn endpoint_url(&self, method: &str) -> String {
        format!("{}/models/{}:{}", self.base_url, self.model_id, method)
    }

    /// Attaches credentials to a request.
    ///
    /// API keys go in the `x-goog-api-key` header rather than a `?key=` query
    /// parameter: reqwest error messages include the full request URL, so a
    /// query-parameter key would leak into logs on any transport error.
    fn apply_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if self.use_bearer_token {
            req.bearer_auth(&self.api_key)
        } else {
            req.header("x-goog-api-key", &self.api_key)
        }
    }

    fn build_request_body(&self, request: &ChatRequest) -> GeminiRequest {
        let mut system_instruction = None;
        let mut contents = Vec::new();
        // Maps tool-call ids from earlier assistant turns to their function
        // names. Gemini requires `functionResponse.name` to be the declared
        // function name, but a `Role::Tool` message only carries the
        // `tool_call_id` — resolving through this map restores the name the
        // model actually called.
        let mut id_to_name: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();

        for msg in &request.messages {
            match msg.role {
                Role::System => {
                    if let Some(text) = &msg.content {
                        system_instruction = Some(GeminiContent {
                            role: "user".to_string(),
                            parts: vec![GeminiPart::Text { text: text.clone() }],
                        });
                    }
                }
                Role::User => {
                    if let Some(text) = &msg.content {
                        contents.push(GeminiContent {
                            role: "user".to_string(),
                            parts: vec![GeminiPart::Text { text: text.clone() }],
                        });
                    }
                }
                Role::Assistant => {
                    if !msg.tool_calls.is_empty() {
                        // Preserve any assistant text emitted alongside the tool
                        // calls as a leading text part; dropping it (the prior
                        // behavior) discards the model's reasoning turn.
                        let mut parts: Vec<GeminiPart> = Vec::new();
                        if let Some(text) = &msg.content
                            && !text.is_empty()
                        {
                            parts.push(GeminiPart::Text { text: text.clone() });
                        }
                        for tc in &msg.tool_calls {
                            id_to_name.insert(tc.id.clone(), tc.name.clone());
                            parts.push(GeminiPart::FunctionCall {
                                function_call: GeminiFunctionCall {
                                    name: tc.name.clone(),
                                    args: tc.arguments.clone(),
                                },
                            });
                        }
                        contents.push(GeminiContent {
                            role: "model".to_string(),
                            parts,
                        });
                    } else if let Some(text) = &msg.content {
                        contents.push(GeminiContent {
                            role: "model".to_string(),
                            parts: vec![GeminiPart::Text { text: text.clone() }],
                        });
                    }
                }
                Role::Tool => {
                    // `functionResponse.name` must be the real function name.
                    // Resolve it from the preceding assistant turn's tool
                    // calls; fall back to parsing the synthetic id format,
                    // then to the raw id (callers may use the plain function
                    // name as the tool_call_id).
                    let call_id = msg.tool_call_id.clone().unwrap_or_default();
                    let name = id_to_name
                        .get(&call_id)
                        .cloned()
                        .or_else(|| function_name_from_tool_call_id(&call_id).map(String::from))
                        .unwrap_or(call_id);
                    let content = msg.content.clone().unwrap_or_default();
                    let response_value: serde_json::Value = serde_json::from_str(&content)
                        .unwrap_or_else(|_| serde_json::json!({ "result": content }));
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![GeminiPart::FunctionResponse {
                            function_response: GeminiFunctionResponse {
                                name,
                                response: response_value,
                            },
                        }],
                    });
                }
            }
        }

        let tools = if request.tools.is_empty() {
            None
        } else {
            let declarations: Vec<GeminiFunctionDeclaration> =
                request.tools.iter().map(Into::into).collect();
            Some(vec![GeminiToolConfig {
                function_declarations: declarations,
            }])
        };

        let generation_config = Some(GeminiGenerationConfig {
            temperature: request.temperature,
            max_output_tokens: request.max_tokens,
        });

        GeminiRequest {
            cached_content: self.cached_content.clone(),
            system_instruction,
            contents,
            tools,
            generation_config,
        }
    }
}

impl Model for Gemini {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    #[tracing::instrument(skip_all, fields(model = %self.model_id))]
    async fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let body = self.build_request_body(request);
        let url = self.endpoint_url("generateContent");
        let timeout = self.timeout.unwrap_or(DEFAULT_REQUEST_TIMEOUT);

        for attempt in 0..=self.max_retries {
            let req = self.client.post(&url).timeout(timeout).json(&body);
            let req = self.apply_auth(req);

            tracing::debug!(attempt, "sending Gemini generateContent request");
            let response = req
                .send()
                .await
                .map_err(|e| DaimonError::Model(format!("Gemini HTTP error: {e}")))?;
            let status = response.status();

            if status.is_success() {
                let api_resp: GeminiResponse = response
                    .json()
                    .await
                    .map_err(|e| DaimonError::Model(format!("Gemini response parse error: {e}")))?;
                tracing::debug!("received successful Gemini response");
                return parse_response(api_resp);
            }

            let retry_after = stream_util::parse_retry_after(response.headers());
            let text = response.text().await.unwrap_or_default();
            let is_retryable = status.as_u16() == 429 || status.is_server_error();

            if is_retryable && attempt < self.max_retries {
                let delay = stream_util::backoff_delay(attempt, retry_after);
                tracing::debug!(status = %status, attempt, delay_ms = delay.as_millis(), "retryable error, backing off");
                tokio::time::sleep(delay).await;
            } else {
                return Err(DaimonError::Model(format!(
                    "Gemini API error ({status}): {text}"
                )));
            }
        }

        unreachable!("loop always returns or retries")
    }

    #[tracing::instrument(skip_all, fields(model = %self.model_id))]
    async fn generate_stream(&self, request: &ChatRequest) -> Result<ResponseStream> {
        let body = self.build_request_body(request);
        let url = self.endpoint_url("streamGenerateContent");

        // Retry only the initial POST/handshake — once the stream is
        // established, mid-stream failures must never be retried (the
        // consumer has already observed a partial response).
        let mut response = None;
        for attempt in 0..=self.max_retries {
            let req = self.client.post(&url).query(&[("alt", "sse")]).json(&body);
            let req = self.apply_auth(req);

            tracing::debug!(attempt, "sending Gemini streaming request");
            let resp = req
                .send()
                .await
                .map_err(|e| DaimonError::Model(format!("Gemini HTTP error: {e}")))?;
            let status = resp.status();

            if status.is_success() {
                response = Some(resp);
                break;
            }

            let retry_after = stream_util::parse_retry_after(resp.headers());
            let text = resp.text().await.unwrap_or_default();
            let is_retryable = status.as_u16() == 429 || status.is_server_error();

            if is_retryable && attempt < self.max_retries {
                let delay = stream_util::backoff_delay(attempt, retry_after);
                tracing::debug!(status = %status, attempt, delay_ms = delay.as_millis(), "retryable error on stream handshake, backing off");
                tokio::time::sleep(delay).await;
            } else {
                return Err(DaimonError::Model(format!(
                    "Gemini API error ({status}): {text}"
                )));
            }
        }
        let response = response.expect("loop breaks with a response or returns an error");

        tracing::debug!("Gemini stream established");
        let byte_stream = response.bytes_stream();

        let stream = async_stream::try_stream! {
            use futures::StreamExt;
            use crate::stream_util::LineBuffer;

            let mut buffer = LineBuffer::new();
            let mut stream = Box::pin(byte_stream);
            // Monotonic across the whole stream so parallel calls to the same
            // function get distinct ids; the function name alone collides.
            let mut tool_call_seq: u64 = 0;

            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| DaimonError::Model(format!("Gemini stream error: {e}")))?;
                buffer.push(&chunk);

                while let Some(line) = buffer.next_line() {
                    let line = line.trim();

                    if line.is_empty() {
                        continue;
                    }

                    if let Some(data) = line.strip_prefix("data: ") {
                        match serde_json::from_str::<GeminiResponse>(data) {
                            Ok(chunk_resp) => {
                                for event in handle_gemini_stream_chunk(&mut tool_call_seq, chunk_resp) {
                                    yield event;
                                }
                            }
                            Err(e) => {
                                tracing::debug!(error = %e, "dropping undeserializable Gemini SSE event");
                            }
                        }
                    }
                }
            }

            // A stream may end without a trailing newline, leaving a final SSE
            // record buffered. Recover it through the identical parse path.
            if let Some(line) = buffer.take_remaining() {
                let line = line.trim();
                if let Some(data) = line.strip_prefix("data: ") {
                    match serde_json::from_str::<GeminiResponse>(data) {
                        Ok(chunk_resp) => {
                            for event in handle_gemini_stream_chunk(&mut tool_call_seq, chunk_resp) {
                                yield event;
                            }
                        }
                        Err(e) => {
                            tracing::debug!(error = %e, "dropping undeserializable Gemini SSE event");
                        }
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

/// Finish reasons that indicate a provider-side content filter blocked output.
fn is_filtered_finish_reason(reason: &str) -> bool {
    matches!(
        reason,
        "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" | "IMAGE_SAFETY"
    )
}

/// Converts one streamed `GeminiResponse` chunk into [`StreamEvent`]s.
///
/// This is the single parse path shared by the main streaming loop and the
/// end-of-stream remainder recovery, extracted so it can be unit-tested
/// without a live HTTP stream.
///
/// Terminal handling: `STOP` / `MAX_TOKENS` end the stream with `Done`.
/// Content-filter finish reasons (`SAFETY`, `RECITATION`, prompt blocking via
/// `promptFeedback.blockReason`, etc.) and `MALFORMED_FUNCTION_CALL` also end
/// the stream, but first surface an in-band [`StreamEvent::Error`] describing
/// why — [`StreamEvent::Done`] carries no stop reason, so the error event is
/// the only channel that keeps these terminations from being silent.
fn handle_gemini_stream_chunk(tool_call_seq: &mut u64, chunk: GeminiResponse) -> Vec<StreamEvent> {
    let mut events = Vec::new();

    // A prompt blocked by safety filters yields no candidates at all; the
    // reason is reported via promptFeedback.
    if chunk.candidates.is_empty() {
        if let Some(reason) = chunk
            .prompt_feedback
            .as_ref()
            .and_then(|f| f.block_reason.as_deref())
        {
            events.push(StreamEvent::Error(format!(
                "Gemini blocked the prompt (blockReason={reason})"
            )));
            events.push(StreamEvent::Done);
        }
        return events;
    }

    for candidate in &chunk.candidates {
        for part in &candidate.content.parts {
            match part {
                GeminiResponsePart::Text { text } => {
                    if !text.is_empty() {
                        events.push(StreamEvent::TextDelta(text.clone()));
                    }
                }
                GeminiResponsePart::FunctionCall { function_call } => {
                    let id = make_tool_call_id(*tool_call_seq, &function_call.name);
                    *tool_call_seq += 1;
                    events.push(StreamEvent::ToolCallStart {
                        id: id.clone(),
                        name: function_call.name.clone(),
                    });
                    let args = serde_json::to_string(&function_call.args).unwrap_or_default();
                    events.push(StreamEvent::ToolCallDelta {
                        id: id.clone(),
                        arguments_delta: args,
                    });
                    events.push(StreamEvent::ToolCallEnd { id });
                }
            }
        }
    }

    let mut done = false;
    for candidate in &chunk.candidates {
        match candidate.finish_reason.as_deref() {
            Some("STOP") | Some("MAX_TOKENS") => done = true,
            Some("MALFORMED_FUNCTION_CALL") => {
                events.push(StreamEvent::Error(
                    "Gemini emitted a malformed function call (finishReason=MALFORMED_FUNCTION_CALL)"
                        .to_string(),
                ));
                done = true;
            }
            Some(reason) if is_filtered_finish_reason(reason) => {
                events.push(StreamEvent::Error(format!(
                    "Gemini blocked the response (finishReason={reason})"
                )));
                done = true;
            }
            _ => {}
        }
    }
    if done {
        events.push(StreamEvent::Done);
    }

    events
}

fn parse_response(response: GeminiResponse) -> Result<ChatResponse> {
    let usage = response.usage_metadata.map(|u| Usage {
        input_tokens: u.prompt_token_count,
        output_tokens: u.candidates_token_count,
        cached_tokens: u.cached_content_token_count,
    });

    let Some(candidate) = response.candidates.into_iter().next() else {
        // A safety-blocked prompt returns no candidates; surface it as a
        // content-filtered response rather than a generic parse error so the
        // caller can distinguish it from a malformed reply.
        if response
            .prompt_feedback
            .as_ref()
            .is_some_and(|f| f.block_reason.is_some())
        {
            return Ok(ChatResponse {
                message: Message::assistant(String::new()),
                stop_reason: StopReason::ContentFiltered,
                usage,
            });
        }
        return Err(DaimonError::Model(
            "no candidates in Gemini response".into(),
        ));
    };

    if let Some(reason) = candidate.finish_reason.as_deref()
        && reason == "MALFORMED_FUNCTION_CALL"
    {
        return Err(DaimonError::Model(
            "Gemini emitted a malformed function call (finishReason=MALFORMED_FUNCTION_CALL)"
                .into(),
        ));
    }

    let mut text_content = String::new();
    let mut tool_calls = Vec::new();

    for (i, part) in candidate.content.parts.into_iter().enumerate() {
        match part {
            GeminiResponsePart::Text { text } => {
                text_content.push_str(&text);
            }
            GeminiResponsePart::FunctionCall { function_call } => {
                // The per-part index disambiguates parallel calls to the same
                // function within one response (a name-only id collides).
                tool_calls.push(ToolCall {
                    id: make_tool_call_id(i as u64, &function_call.name),
                    name: function_call.name,
                    arguments: function_call.args,
                });
            }
        }
    }

    let stop_reason = if !tool_calls.is_empty() {
        StopReason::ToolUse
    } else {
        match candidate.finish_reason.as_deref() {
            Some("MAX_TOKENS") => StopReason::MaxTokens,
            Some(reason) if is_filtered_finish_reason(reason) => StopReason::ContentFiltered,
            _ => StopReason::EndTurn,
        }
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
        usage,
    })
}

// --- Gemini API types (request) ---

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    cached_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiContent>,
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GeminiToolConfig>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
}

#[derive(Serialize)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum GeminiPart {
    Text {
        text: String,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: GeminiFunctionCall,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: GeminiFunctionResponse,
    },
}

#[derive(Serialize)]
struct GeminiFunctionCall {
    name: String,
    args: serde_json::Value,
}

#[derive(Serialize)]
struct GeminiFunctionResponse {
    name: String,
    response: serde_json::Value,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiToolConfig {
    function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Serialize)]
struct GeminiFunctionDeclaration {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

impl From<&ToolSpec> for GeminiFunctionDeclaration {
    fn from(spec: &ToolSpec) -> Self {
        Self {
            name: spec.name.clone(),
            description: spec.description.clone(),
            parameters: spec.parameters.clone(),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
}

// --- Gemini API types (response) ---

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponse {
    #[serde(default)]
    candidates: Vec<GeminiCandidate>,
    usage_metadata: Option<GeminiUsageMetadata>,
    prompt_feedback: Option<GeminiPromptFeedback>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiPromptFeedback {
    block_reason: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiCandidate {
    #[serde(default)]
    content: GeminiResponseContent,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Default)]
struct GeminiResponseContent {
    #[serde(default)]
    parts: Vec<GeminiResponsePart>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum GeminiResponsePart {
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: GeminiResponseFunctionCall,
    },
    Text {
        text: String,
    },
}

#[derive(Deserialize)]
struct GeminiResponseFunctionCall {
    name: String,
    args: serde_json::Value,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiUsageMetadata {
    #[serde(default)]
    prompt_token_count: u32,
    #[serde(default)]
    candidates_token_count: u32,
    #[serde(default)]
    cached_content_token_count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemini_new_default() {
        let model = Gemini::new("gemini-2.0-flash");
        assert_eq!(model.model_id, "gemini-2.0-flash");
        assert_eq!(model.base_url, DEFAULT_BASE_URL);
        assert_eq!(model.max_retries, DEFAULT_MAX_RETRIES);
        assert!(!model.use_bearer_token);
        assert!(
            model.timeout.is_none(),
            "default request timeout is applied per-request"
        );
    }

    #[test]
    fn test_with_base_url() {
        let model = Gemini::new("gemini-pro").with_base_url("https://vertex.example.com");
        assert_eq!(model.base_url, "https://vertex.example.com");
    }

    #[test]
    fn test_with_timeout() {
        let model = Gemini::new("gemini-pro").with_timeout(Duration::from_secs(30));
        assert_eq!(model.timeout, Some(Duration::from_secs(30)));
    }

    #[test]
    fn test_with_max_retries() {
        let model = Gemini::new("gemini-pro").with_max_retries(5);
        assert_eq!(model.max_retries, 5);
    }

    #[test]
    fn test_with_bearer_token() {
        let model = Gemini::new("gemini-pro").with_bearer_token();
        assert!(model.use_bearer_token);
    }

    #[test]
    fn test_endpoint_url() {
        let model = Gemini::new("gemini-2.0-flash");
        assert_eq!(
            model.endpoint_url("generateContent"),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        );
    }

    #[test]
    fn test_api_key_sent_as_header_not_query_param() {
        // A `?key=` query parameter leaks the key into logs whenever reqwest
        // includes the URL in an error message; the key must ride in the
        // `x-goog-api-key` header instead.
        let model = Gemini::with_api_key("gemini-pro", "AIza-secret-key");
        let req = model
            .apply_auth(model.client.post("https://example.com/v1"))
            .build()
            .unwrap();
        assert!(
            !req.url().as_str().contains("AIza-secret-key"),
            "API key must not appear in the request URL: {}",
            req.url()
        );
        assert_eq!(
            req.headers().get("x-goog-api-key").unwrap(),
            "AIza-secret-key"
        );
    }

    #[test]
    fn test_bearer_token_used_when_configured() {
        let model = Gemini::with_api_key("gemini-pro", "oauth-token").with_bearer_token();
        let req = model
            .apply_auth(model.client.post("https://example.com/v1"))
            .build()
            .unwrap();
        assert!(!req.url().as_str().contains("oauth-token"));
        assert_eq!(
            req.headers().get("authorization").unwrap(),
            "Bearer oauth-token"
        );
    }

    #[test]
    fn test_tool_spec_conversion() {
        let spec = ToolSpec {
            name: "search".into(),
            description: "Web search".into(),
            parameters: serde_json::json!({"type": "object"}),
        };
        let decl: GeminiFunctionDeclaration = (&spec).into();
        assert_eq!(decl.name, "search");
        assert_eq!(decl.description, "Web search");
    }

    #[test]
    fn test_function_name_from_tool_call_id() {
        assert_eq!(
            function_name_from_tool_call_id("gemini_0_search"),
            Some("search")
        );
        // Function names may contain underscores and digits.
        assert_eq!(
            function_name_from_tool_call_id("gemini_12_web_search_v2"),
            Some("web_search_v2")
        );
        // Not the synthetic format: no prefix, no seq, or empty name.
        assert_eq!(function_name_from_tool_call_id("search"), None);
        assert_eq!(function_name_from_tool_call_id("gemini_search"), None);
        assert_eq!(function_name_from_tool_call_id("gemini_3_"), None);
        assert_eq!(function_name_from_tool_call_id("gemini__x"), None);
    }

    #[test]
    fn test_parse_response_text() {
        let raw = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: GeminiResponseContent {
                    parts: vec![GeminiResponsePart::Text {
                        text: "Hello world".into(),
                    }],
                },
                finish_reason: Some("STOP".into()),
            }],
            usage_metadata: Some(GeminiUsageMetadata {
                prompt_token_count: 10,
                candidates_token_count: 5,
                cached_content_token_count: 0,
            }),
            prompt_feedback: None,
        };
        let resp = parse_response(raw).unwrap();
        assert_eq!(resp.text(), "Hello world");
        assert_eq!(resp.stop_reason, StopReason::EndTurn);
        assert!(!resp.has_tool_calls());
        assert_eq!(resp.usage.unwrap().input_tokens, 10);
    }

    #[test]
    fn test_parse_response_function_call() {
        let raw = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: GeminiResponseContent {
                    parts: vec![GeminiResponsePart::FunctionCall {
                        function_call: GeminiResponseFunctionCall {
                            name: "calculator".into(),
                            args: serde_json::json!({"expr": "2+2"}),
                        },
                    }],
                },
                finish_reason: Some("STOP".into()),
            }],
            usage_metadata: None,
            prompt_feedback: None,
        };
        let resp = parse_response(raw).unwrap();
        assert!(resp.has_tool_calls());
        assert_eq!(resp.tool_calls()[0].name, "calculator");
        assert_eq!(resp.tool_calls()[0].id, "gemini_0_calculator");
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
    }

    #[test]
    fn test_parse_response_parallel_same_function_gets_distinct_ids() {
        // Two parallel calls to the same function previously produced the
        // identical id `gemini_calc`, so tool results could not be attributed.
        let call = |expr: &str| GeminiResponsePart::FunctionCall {
            function_call: GeminiResponseFunctionCall {
                name: "calc".into(),
                args: serde_json::json!({"expr": expr}),
            },
        };
        let raw = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: GeminiResponseContent {
                    parts: vec![call("1+1"), call("2+2")],
                },
                finish_reason: Some("STOP".into()),
            }],
            usage_metadata: None,
            prompt_feedback: None,
        };
        let resp = parse_response(raw).unwrap();
        let calls = resp.tool_calls();
        assert_eq!(calls.len(), 2);
        assert_ne!(
            calls[0].id, calls[1].id,
            "parallel calls must get distinct ids"
        );
        // Both ids must round-trip to the real function name.
        assert_eq!(function_name_from_tool_call_id(&calls[0].id), Some("calc"));
        assert_eq!(function_name_from_tool_call_id(&calls[1].id), Some("calc"));
    }

    #[test]
    fn test_parse_response_no_candidates() {
        let raw = GeminiResponse {
            candidates: vec![],
            usage_metadata: None,
            prompt_feedback: None,
        };
        assert!(parse_response(raw).is_err());
    }

    #[test]
    fn test_parse_response_prompt_blocked_maps_to_content_filtered() {
        let raw = GeminiResponse {
            candidates: vec![],
            usage_metadata: None,
            prompt_feedback: Some(GeminiPromptFeedback {
                block_reason: Some("SAFETY".into()),
            }),
        };
        let resp = parse_response(raw).unwrap();
        assert_eq!(resp.stop_reason, StopReason::ContentFiltered);
        assert_eq!(resp.text(), "");
    }

    #[test]
    fn test_parse_response_safety_finish_maps_to_content_filtered() {
        let raw = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: GeminiResponseContent { parts: vec![] },
                finish_reason: Some("SAFETY".into()),
            }],
            usage_metadata: None,
            prompt_feedback: None,
        };
        let resp = parse_response(raw).unwrap();
        assert_eq!(resp.stop_reason, StopReason::ContentFiltered);
    }

    #[test]
    fn test_parse_response_malformed_function_call_is_error() {
        let raw = GeminiResponse {
            candidates: vec![GeminiCandidate {
                content: GeminiResponseContent { parts: vec![] },
                finish_reason: Some("MALFORMED_FUNCTION_CALL".into()),
            }],
            usage_metadata: None,
            prompt_feedback: None,
        };
        assert!(parse_response(raw).is_err());
    }

    #[test]
    fn test_build_request_with_system_prompt() {
        let model = Gemini::with_api_key("gemini-pro", "key");
        let request = ChatRequest {
            messages: vec![Message::system("Be helpful"), Message::user("Hello")],
            tools: vec![],
            temperature: Some(0.7),
            max_tokens: Some(1024),
        };
        let body = model.build_request_body(&request);
        assert!(body.system_instruction.is_some());
        assert_eq!(body.contents.len(), 1);
        assert_eq!(
            body.generation_config.as_ref().unwrap().temperature,
            Some(0.7)
        );
    }

    #[test]
    fn test_build_request_with_tools() {
        let model = Gemini::with_api_key("gemini-pro", "key");
        let request = ChatRequest {
            messages: vec![Message::user("hi")],
            tools: vec![ToolSpec {
                name: "calc".into(),
                description: "Calculator".into(),
                parameters: serde_json::json!({"type": "object"}),
            }],
            temperature: None,
            max_tokens: None,
        };
        let body = model.build_request_body(&request);
        assert!(body.tools.is_some());
        assert_eq!(body.tools.unwrap()[0].function_declarations.len(), 1);
    }

    #[test]
    fn test_build_request_with_tool_results() {
        let model = Gemini::with_api_key("gemini-pro", "key");
        let request = ChatRequest {
            messages: vec![
                Message::user("calc 2+2"),
                Message::assistant_with_tool_calls(vec![ToolCall {
                    id: "gemini_0_calc".into(),
                    name: "calc".into(),
                    arguments: serde_json::json!({"expr": "2+2"}),
                }]),
                Message::tool_result("gemini_0_calc", "4"),
            ],
            tools: vec![],
            temperature: None,
            max_tokens: None,
        };
        let body = model.build_request_body(&request);
        assert_eq!(body.contents.len(), 3);
    }

    #[test]
    fn test_tool_result_round_trip_uses_function_name() {
        // The id round-trip bug: parse_response synthesizes an id, the agent
        // echoes it as tool_call_id, and build_request_body must send the real
        // function name — not the synthetic id — as functionResponse.name.
        let model = Gemini::with_api_key("gemini-pro", "key");
        let request = ChatRequest {
            messages: vec![
                Message::user("search rust"),
                Message::assistant_with_tool_calls(vec![ToolCall {
                    id: "gemini_0_search".into(),
                    name: "search".into(),
                    arguments: serde_json::json!({"q": "rust"}),
                }]),
                Message::tool_result("gemini_0_search", r#"{"hits": 3}"#),
            ],
            tools: vec![],
            temperature: None,
            max_tokens: None,
        };
        let body = model.build_request_body(&request);
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(
            json["contents"][2]["parts"][0]["functionResponse"]["name"], "search",
            "functionResponse.name must be the declared function name"
        );
        // The declared call and the response must agree so Gemini can match them.
        assert_eq!(
            json["contents"][1]["parts"][0]["functionCall"]["name"],
            "search"
        );
    }

    #[test]
    fn test_tool_result_falls_back_to_synthetic_id_parse() {
        // If the assistant turn that produced the call is absent from the
        // conversation, the name is recovered from the synthetic id format.
        let model = Gemini::with_api_key("gemini-pro", "key");
        let request = ChatRequest {
            messages: vec![Message::tool_result("gemini_7_web_search", "ok")],
            tools: vec![],
            temperature: None,
            max_tokens: None,
        };
        let body = model.build_request_body(&request);
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(
            json["contents"][0]["parts"][0]["functionResponse"]["name"],
            "web_search"
        );
    }

    #[test]
    fn test_tool_result_with_plain_name_id_passes_through() {
        // Callers may hand-construct tool results keyed by the function name.
        let model = Gemini::with_api_key("gemini-pro", "key");
        let request = ChatRequest {
            messages: vec![Message::tool_result("calc", "4")],
            tools: vec![],
            temperature: None,
            max_tokens: None,
        };
        let body = model.build_request_body(&request);
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(
            json["contents"][0]["parts"][0]["functionResponse"]["name"],
            "calc"
        );
    }

    #[test]
    fn test_builder_chain() {
        let model = Gemini::with_api_key("gemini-2.0-flash", "key")
            .with_base_url("https://custom.example.com")
            .with_timeout(Duration::from_secs(60))
            .with_max_retries(5)
            .with_bearer_token();

        assert_eq!(model.model_id, "gemini-2.0-flash");
        assert_eq!(model.base_url, "https://custom.example.com");
        assert_eq!(model.timeout, Some(Duration::from_secs(60)));
        assert_eq!(model.max_retries, 5);
        assert!(model.use_bearer_token);
    }

    #[test]
    fn test_debug_redacts_api_key() {
        let model = Gemini::with_api_key("gemini-pro", "AIza-supersecret-token");
        let dbg = format!("{model:?}");
        assert!(
            !dbg.contains("AIza-supersecret-token"),
            "Debug output must not contain the plaintext API key: {dbg}"
        );
        assert!(dbg.contains("[redacted]"));
    }

    #[test]
    fn test_assistant_text_preserved_with_tool_call() {
        let model = Gemini::with_api_key("gemini-pro", "key");
        let assistant = Message {
            role: Role::Assistant,
            content: Some("Calling the tool now.".to_string()),
            tool_calls: vec![ToolCall {
                id: "gemini_0_calc".into(),
                name: "calc".into(),
                arguments: serde_json::json!({"expr": "2+2"}),
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
        let json = serde_json::to_value(&body).unwrap();
        let parts = json["contents"][1]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 2, "text part + functionCall part");
        assert_eq!(parts[0]["text"], "Calling the tool now.");
        assert_eq!(parts[1]["functionCall"]["name"], "calc");
    }

    // --- Streaming chunk-handler tests ---

    fn chunk(json: &str) -> GeminiResponse {
        serde_json::from_str(json).expect("valid GeminiResponse JSON")
    }

    #[test]
    fn test_stream_chunk_text_delta() {
        let mut seq = 0;
        let events = handle_gemini_stream_chunk(
            &mut seq,
            chunk(r#"{"candidates":[{"content":{"parts":[{"text":"hi"}]}}]}"#),
        );
        assert!(matches!(events.as_slice(), [StreamEvent::TextDelta(t)] if t == "hi"));
    }

    #[test]
    fn test_stream_chunk_function_call_lifecycle_and_distinct_ids() {
        let mut seq = 0;
        let events = handle_gemini_stream_chunk(
            &mut seq,
            chunk(
                r#"{"candidates":[{"content":{"parts":[
                    {"functionCall":{"name":"calc","args":{"x":1}}},
                    {"functionCall":{"name":"calc","args":{"x":2}}}
                ]}}]}"#,
            ),
        );
        assert_eq!(events.len(), 6, "two Start/Delta/End triples: {events:?}");
        let (id_a, id_b) = match (&events[0], &events[3]) {
            (
                StreamEvent::ToolCallStart { id: a, name: na },
                StreamEvent::ToolCallStart { id: b, name: nb },
            ) => {
                assert_eq!(na, "calc");
                assert_eq!(nb, "calc");
                (a.clone(), b.clone())
            }
            other => panic!("expected two ToolCallStart events, got {other:?}"),
        };
        assert_ne!(id_a, id_b, "parallel same-function calls need distinct ids");
        assert_eq!(function_name_from_tool_call_id(&id_a), Some("calc"));
        assert_eq!(function_name_from_tool_call_id(&id_b), Some("calc"));
        assert!(matches!(&events[2], StreamEvent::ToolCallEnd { id } if *id == id_a));
        assert!(matches!(&events[5], StreamEvent::ToolCallEnd { id } if *id == id_b));
        assert_eq!(seq, 2, "sequence advances across the stream");
    }

    #[test]
    fn test_stream_chunk_stop_emits_done() {
        let mut seq = 0;
        let events = handle_gemini_stream_chunk(
            &mut seq,
            chunk(r#"{"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}]}"#),
        );
        assert!(matches!(events.as_slice(), [StreamEvent::Done]));
    }

    #[test]
    fn test_stream_chunk_safety_emits_error_then_done() {
        let mut seq = 0;
        let events = handle_gemini_stream_chunk(
            &mut seq,
            chunk(r#"{"candidates":[{"content":{"parts":[]},"finishReason":"SAFETY"}]}"#),
        );
        assert_eq!(events.len(), 2, "got {events:?}");
        assert!(matches!(&events[0], StreamEvent::Error(msg) if msg.contains("SAFETY")));
        assert!(matches!(&events[1], StreamEvent::Done));
    }

    #[test]
    fn test_stream_chunk_recitation_emits_error_then_done() {
        let mut seq = 0;
        let events = handle_gemini_stream_chunk(
            &mut seq,
            chunk(r#"{"candidates":[{"content":{"parts":[]},"finishReason":"RECITATION"}]}"#),
        );
        assert!(matches!(&events[0], StreamEvent::Error(msg) if msg.contains("RECITATION")));
        assert!(matches!(&events[1], StreamEvent::Done));
    }

    #[test]
    fn test_stream_chunk_malformed_function_call_emits_error_then_done() {
        let mut seq = 0;
        let events = handle_gemini_stream_chunk(
            &mut seq,
            chunk(
                r#"{"candidates":[{"content":{"parts":[]},"finishReason":"MALFORMED_FUNCTION_CALL"}]}"#,
            ),
        );
        assert!(
            matches!(&events[0], StreamEvent::Error(msg) if msg.contains("MALFORMED_FUNCTION_CALL"))
        );
        assert!(matches!(&events[1], StreamEvent::Done));
    }

    #[test]
    fn test_stream_chunk_prompt_blocked_emits_error_then_done() {
        let mut seq = 0;
        let events = handle_gemini_stream_chunk(
            &mut seq,
            chunk(r#"{"candidates":[],"promptFeedback":{"blockReason":"SAFETY"}}"#),
        );
        assert_eq!(events.len(), 2, "got {events:?}");
        assert!(matches!(&events[0], StreamEvent::Error(msg) if msg.contains("SAFETY")));
        assert!(matches!(&events[1], StreamEvent::Done));
    }

    #[test]
    fn test_stream_chunk_empty_candidates_without_feedback_is_silent() {
        // A keep-alive-ish empty chunk must not end the stream.
        let mut seq = 0;
        let events = handle_gemini_stream_chunk(&mut seq, chunk(r#"{"candidates":[]}"#));
        assert!(events.is_empty());
    }
}
