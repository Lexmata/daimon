//! Azure OpenAI model provider for the [Daimon](https://docs.rs/daimon) agent framework.
//!
//! The API wire format is identical to OpenAI but uses a different URL
//! structure and supports both API key and Microsoft Entra ID (Azure AD)
//! bearer token authentication.
//!
//! # Example
//!
//! ```ignore
//! use daimon_provider_azure::AzureOpenAi;
//! use daimon_core::Model;
//!
//! let model = AzureOpenAi::new(
//!     "https://my-resource.openai.azure.com",
//!     "gpt-4o",
//! );
//! ```

use std::collections::HashMap;
use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};

mod embedding;
mod stream_util;

#[cfg(feature = "servicebus")]
pub mod servicebus;

pub use embedding::AzureOpenAiEmbedding;

#[cfg(feature = "servicebus")]
pub use servicebus::ServiceBusBroker;

use daimon_core::{
    ChatRequest, ChatResponse, DaimonError, Message, Model, ResponseStream, Result, Role,
    StopReason, StreamEvent, ToolCall, ToolSpec, Usage,
};

const DEFAULT_API_VERSION: &str = "2024-10-21";
const DEFAULT_MAX_RETRIES: u32 = 3;

/// Default whole-request timeout applied to non-streaming `generate` calls.
///
/// Long completions can legitimately take minutes, so this is deliberately
/// generous. Override with [`AzureOpenAi::with_timeout`].
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

/// Azure OpenAI model provider.
///
/// Connects to an Azure OpenAI deployment. Authentication is via API key
/// (default, using the `api-key` header) or Microsoft Entra ID bearer token.
pub struct AzureOpenAi {
    client: Client,
    api_key: String,
    resource_url: String,
    deployment_id: String,
    api_version: String,
    timeout: Option<Duration>,
    max_retries: u32,
    use_bearer_token: bool,
}

impl std::fmt::Debug for AzureOpenAi {
    /// Hand-written to avoid leaking the plaintext API key (or Entra ID bearer
    /// token) in logs or panic output; a derived `Debug` would print it verbatim.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AzureOpenAi")
            .field("client", &self.client)
            .field("api_key", &"[redacted]")
            .field("resource_url", &self.resource_url)
            .field("deployment_id", &self.deployment_id)
            .field("api_version", &self.api_version)
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .field("use_bearer_token", &self.use_bearer_token)
            .finish()
    }
}

impl AzureOpenAi {
    /// Create a new Azure OpenAI client, reading `AZURE_OPENAI_API_KEY` from the environment.
    ///
    /// The constructor never fails; if the environment variable is unset or
    /// empty a warning is logged and requests will fail with an auth error.
    pub fn new(resource_url: impl Into<String>, deployment_id: impl Into<String>) -> Self {
        let api_key = std::env::var("AZURE_OPENAI_API_KEY").unwrap_or_default();
        if api_key.is_empty() {
            tracing::warn!(
                "AZURE_OPENAI_API_KEY is not set or empty; Azure OpenAI requests will fail authentication"
            );
        }
        Self::with_api_key(resource_url, deployment_id, api_key)
    }

    /// Create a new Azure OpenAI client with an explicit API key.
    pub fn with_api_key(
        resource_url: impl Into<String>,
        deployment_id: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self {
            client: build_client(),
            api_key: api_key.into(),
            resource_url: resource_url.into().trim_end_matches('/').to_string(),
            deployment_id: deployment_id.into(),
            api_version: DEFAULT_API_VERSION.to_string(),
            timeout: None,
            max_retries: DEFAULT_MAX_RETRIES,
            use_bearer_token: false,
        }
    }

    /// Set the Azure OpenAI API version (default: `2024-10-21`).
    pub fn with_api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = version.into();
        self
    }

    /// Set an HTTP timeout for non-streaming requests (default: 120s).
    ///
    /// The timeout applies per-request to `generate`; `generate_stream` is a
    /// long-lived SSE connection and is protected only by the client's
    /// connect timeout, since a whole-request deadline would abort healthy
    /// streams.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set the maximum number of retries for transient errors.
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Use `Authorization: Bearer <token>` instead of `api-key` header.
    ///
    /// Required for Microsoft Entra ID (Azure AD) authentication.
    pub fn with_bearer_token(mut self) -> Self {
        self.use_bearer_token = true;
        self
    }

    fn endpoint_url(&self) -> String {
        format!(
            "{}/openai/deployments/{}/chat/completions",
            self.resource_url, self.deployment_id
        )
    }

    fn apply_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if self.use_bearer_token {
            req.bearer_auth(&self.api_key)
        } else {
            req.header("api-key", &self.api_key)
        }
    }

    fn build_request_body(&self, request: &ChatRequest, stream: bool) -> AzureRequest {
        let messages: Vec<AzureMessage> = request.messages.iter().map(Into::into).collect();

        let tools: Option<Vec<AzureTool>> = if request.tools.is_empty() {
            None
        } else {
            Some(request.tools.iter().map(Into::into).collect())
        };

        AzureRequest {
            messages,
            tools,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream,
        }
    }
}

impl Model for AzureOpenAi {
    fn model_id(&self) -> &str {
        &self.deployment_id
    }

    #[tracing::instrument(skip_all, fields(deployment = %self.deployment_id))]
    async fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let body = self.build_request_body(request, false);
        let url = self.endpoint_url();
        let timeout = self.timeout.unwrap_or(DEFAULT_REQUEST_TIMEOUT);

        for attempt in 0..=self.max_retries {
            let req = self
                .client
                .post(&url)
                .timeout(timeout)
                .query(&[("api-version", &self.api_version)])
                .json(&body);
            let req = self.apply_auth(req);

            tracing::debug!(attempt, "sending Azure OpenAI request");
            let response = req
                .send()
                .await
                .map_err(|e| DaimonError::Model(format!("Azure OpenAI HTTP error: {e}")))?;
            let status = response.status();

            if status.is_success() {
                let api_resp: AzureResponse = response.json().await.map_err(|e| {
                    DaimonError::Model(format!("Azure OpenAI response parse error: {e}"))
                })?;
                tracing::debug!("received successful Azure OpenAI response");
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
                    "Azure OpenAI API error ({status}): {text}"
                )));
            }
        }

        unreachable!("loop always returns or retries")
    }

    #[tracing::instrument(skip_all, fields(deployment = %self.deployment_id))]
    async fn generate_stream(&self, request: &ChatRequest) -> Result<ResponseStream> {
        let body = self.build_request_body(request, true);
        let url = self.endpoint_url();

        // Retry only the initial POST/handshake — once the stream is
        // established, mid-stream failures must never be retried (the
        // consumer has already observed a partial response).
        let mut response = None;
        for attempt in 0..=self.max_retries {
            let req = self
                .client
                .post(&url)
                .query(&[("api-version", &self.api_version)])
                .json(&body);
            let req = self.apply_auth(req);

            tracing::debug!(attempt, "sending Azure OpenAI streaming request");
            let resp = req
                .send()
                .await
                .map_err(|e| DaimonError::Model(format!("Azure OpenAI HTTP error: {e}")))?;
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
                    "Azure OpenAI API error ({status}): {text}"
                )));
            }
        }
        let response = response.expect("loop breaks with a response or returns an error");

        tracing::debug!("Azure OpenAI stream established");
        let byte_stream = response.bytes_stream();

        let stream = async_stream::try_stream! {
            use futures::StreamExt;
            use crate::stream_util::LineBuffer;

            let mut buffer = LineBuffer::new();
            let mut state = AzureStreamState::default();
            let mut stream = Box::pin(byte_stream);

            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| DaimonError::Model(format!("Azure OpenAI stream error: {e}")))?;
                buffer.push(&chunk);

                while let Some(line) = buffer.next_line() {
                    for event in handle_azure_sse_line(&mut state, &line) {
                        yield event;
                    }
                }
            }

            // A stream may end without a trailing newline, leaving a final SSE
            // record buffered. Recover it through the identical parse path.
            if let Some(line) = buffer.take_remaining() {
                for event in handle_azure_sse_line(&mut state, &line) {
                    yield event;
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

/// Streaming state for the Azure OpenAI SSE parser.
///
/// The wire format is OpenAI's: a tool call's `id` is announced only on its
/// first chunk; subsequent argument fragments carry just the array `index`.
/// The map correlates the index back to the announced id, and `open_calls`
/// tracks announcement order so `ToolCallEnd` fires for every call when
/// `finish_reason` arrives.
#[derive(Default)]
struct AzureStreamState {
    index_to_id: HashMap<usize, String>,
    open_calls: Vec<String>,
}

/// Parses one SSE line from an Azure OpenAI chat-completions stream into
/// [`StreamEvent`]s.
///
/// This mirrors the extracted, unit-tested parser in the built-in OpenAI
/// provider (the wire format is identical); azure is a separate crate that
/// cannot depend on daimon's internals, so the logic is single-sourced per
/// crate rather than shared. Behavior:
///
/// - Tool-call ids are the provider-assigned `id` when present; the array
///   index is only a fallback for servers that omit ids.
/// - `finish_reason` closes all open tool calls with `ToolCallEnd`, and
///   `content_filter` additionally surfaces an in-band [`StreamEvent::Error`]
///   ([`StreamEvent::Done`] carries no stop reason, so the error event is the
///   only channel that keeps a filtered termination from being silent).
/// - The `data: [DONE]` sentinel yields `Done`.
fn handle_azure_sse_line(state: &mut AzureStreamState, line: &str) -> Vec<StreamEvent> {
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
    let chunk = match serde_json::from_str::<AzureStreamChunk>(data) {
        Ok(chunk) => chunk,
        Err(e) => {
            tracing::debug!(error = %e, "dropping undeserializable Azure OpenAI SSE event");
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
                    "Azure OpenAI blocked the response (finish_reason=content_filter)".to_string(),
                ));
            }
        }
    }

    events
}

fn parse_response(response: AzureResponse) -> Result<ChatResponse> {
    let choice = response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| DaimonError::Model("no choices in Azure OpenAI response".into()))?;

    let mut tool_calls = Vec::new();
    for tc in choice.message.tool_calls.unwrap_or_default() {
        // Malformed arguments must surface as an error: silently coercing
        // them to null would run the tool with the corruption hidden.
        let arguments = if tc.function.arguments.trim().is_empty() {
            serde_json::Value::Object(serde_json::Map::new())
        } else {
            serde_json::from_str(&tc.function.arguments).map_err(|e| {
                DaimonError::Model(format!(
                    "Azure OpenAI returned malformed JSON arguments for tool call '{}' (id {}): {e}",
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

// --- Azure OpenAI API types ---

#[derive(Serialize)]
struct AzureRequest {
    messages: Vec<AzureMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AzureTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    stream: bool,
}

#[derive(Serialize, Deserialize)]
struct AzureMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<AzureToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

impl From<&Message> for AzureMessage {
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
                    .map(|tc| AzureToolCall {
                        id: tc.id.clone(),
                        r#type: "function".to_string(),
                        function: AzureFunction {
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
struct AzureTool {
    r#type: String,
    function: AzureToolFunction,
}

impl From<&ToolSpec> for AzureTool {
    fn from(spec: &ToolSpec) -> Self {
        Self {
            r#type: "function".to_string(),
            function: AzureToolFunction {
                name: spec.name.clone(),
                description: spec.description.clone(),
                parameters: spec.parameters.clone(),
            },
        }
    }
}

#[derive(Serialize)]
struct AzureToolFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Deserialize)]
struct AzureResponse {
    choices: Vec<AzureChoice>,
    usage: Option<AzureUsage>,
}

#[derive(Deserialize)]
struct AzureChoice {
    message: AzureChoiceMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct AzureChoiceMessage {
    content: Option<String>,
    tool_calls: Option<Vec<AzureToolCall>>,
}

#[derive(Serialize, Deserialize)]
struct AzureToolCall {
    #[serde(default)]
    id: String,
    #[serde(default)]
    r#type: String,
    #[serde(default)]
    function: AzureFunction,
    #[serde(default)]
    index: usize,
}

#[derive(Serialize, Deserialize, Default)]
struct AzureFunction {
    #[serde(default)]
    name: String,
    #[serde(default)]
    arguments: String,
}

#[derive(Deserialize)]
struct AzureUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    prompt_tokens_details: Option<AzurePromptTokensDetails>,
}

#[derive(Deserialize)]
struct AzurePromptTokensDetails {
    #[serde(default)]
    cached_tokens: u32,
}

#[derive(Deserialize)]
struct AzureStreamChunk {
    choices: Vec<AzureStreamChoice>,
}

#[derive(Deserialize)]
struct AzureStreamChoice {
    delta: AzureStreamDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct AzureStreamDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<AzureStreamToolCall>>,
}

#[derive(Deserialize)]
struct AzureStreamToolCall {
    index: usize,
    /// Provider-assigned id, present only on a call's first chunk.
    #[serde(default)]
    id: Option<String>,
    function: Option<AzureStreamFunction>,
}

#[derive(Deserialize)]
struct AzureStreamFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_azure_new_default() {
        let model = AzureOpenAi::new("https://my-resource.openai.azure.com", "gpt-4o");
        assert_eq!(model.deployment_id, "gpt-4o");
        assert_eq!(model.resource_url, "https://my-resource.openai.azure.com");
        assert_eq!(model.api_version, DEFAULT_API_VERSION);
        assert_eq!(model.max_retries, DEFAULT_MAX_RETRIES);
        assert!(!model.use_bearer_token);
    }

    #[test]
    fn test_resource_url_trailing_slash_stripped() {
        let model = AzureOpenAi::new("https://my-resource.openai.azure.com/", "gpt-4o");
        assert_eq!(model.resource_url, "https://my-resource.openai.azure.com");
    }

    #[test]
    fn test_endpoint_url() {
        let model =
            AzureOpenAi::with_api_key("https://my-resource.openai.azure.com", "gpt-4o", "key");
        assert_eq!(
            model.endpoint_url(),
            "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions"
        );
    }

    #[test]
    fn test_with_api_version() {
        let model =
            AzureOpenAi::new("https://x.openai.azure.com", "gpt-4o").with_api_version("2025-01-01");
        assert_eq!(model.api_version, "2025-01-01");
    }

    #[test]
    fn test_with_timeout() {
        let model = AzureOpenAi::new("https://x.openai.azure.com", "gpt-4o")
            .with_timeout(Duration::from_secs(60));
        assert_eq!(model.timeout, Some(Duration::from_secs(60)));
    }

    #[test]
    fn test_with_max_retries() {
        let model = AzureOpenAi::new("https://x.openai.azure.com", "gpt-4o").with_max_retries(10);
        assert_eq!(model.max_retries, 10);
    }

    #[test]
    fn test_with_bearer_token() {
        let model = AzureOpenAi::new("https://x.openai.azure.com", "gpt-4o").with_bearer_token();
        assert!(model.use_bearer_token);
    }

    #[test]
    fn test_message_conversion_user() {
        let msg = Message::user("hello");
        let azure: AzureMessage = (&msg).into();
        assert_eq!(azure.role, "user");
        assert_eq!(azure.content.as_deref(), Some("hello"));
        assert!(azure.tool_calls.is_none());
    }

    #[test]
    fn test_message_conversion_tool_result() {
        let msg = Message::tool_result("tc_1", "42");
        let azure: AzureMessage = (&msg).into();
        assert_eq!(azure.role, "tool");
        assert_eq!(azure.tool_call_id.as_deref(), Some("tc_1"));
    }

    #[test]
    fn test_message_conversion_assistant_with_tools() {
        let msg = Message::assistant_with_tool_calls(vec![ToolCall {
            id: "tc_1".into(),
            name: "calc".into(),
            arguments: serde_json::json!({"x": 1}),
        }]);
        let azure: AzureMessage = (&msg).into();
        assert_eq!(azure.role, "assistant");
        assert!(azure.tool_calls.is_some());
        assert_eq!(azure.tool_calls.unwrap().len(), 1);
    }

    #[test]
    fn test_tool_spec_conversion() {
        let spec = ToolSpec {
            name: "search".into(),
            description: "Web search".into(),
            parameters: serde_json::json!({"type": "object"}),
        };
        let tool: AzureTool = (&spec).into();
        assert_eq!(tool.r#type, "function");
        assert_eq!(tool.function.name, "search");
    }

    #[test]
    fn test_parse_response_text() {
        let raw = AzureResponse {
            choices: vec![AzureChoice {
                message: AzureChoiceMessage {
                    content: Some("hello".into()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".into()),
            }],
            usage: Some(AzureUsage {
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
    fn test_parse_response_tool_calls() {
        let raw = AzureResponse {
            choices: vec![AzureChoice {
                message: AzureChoiceMessage {
                    content: None,
                    tool_calls: Some(vec![AzureToolCall {
                        id: "tc_1".into(),
                        r#type: "function".into(),
                        function: AzureFunction {
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
        // Previously malformed args were silently coerced to null via
        // `unwrap_or_default`, hiding the corruption from the caller.
        let raw = AzureResponse {
            choices: vec![AzureChoice {
                message: AzureChoiceMessage {
                    content: None,
                    tool_calls: Some(vec![AzureToolCall {
                        id: "tc_1".into(),
                        r#type: "function".into(),
                        function: AzureFunction {
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
        let raw = AzureResponse {
            choices: vec![AzureChoice {
                message: AzureChoiceMessage {
                    content: None,
                    tool_calls: Some(vec![AzureToolCall {
                        id: "tc_1".into(),
                        r#type: "function".into(),
                        function: AzureFunction {
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
        let raw = AzureResponse {
            choices: vec![AzureChoice {
                message: AzureChoiceMessage {
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
        let raw = AzureResponse {
            choices: vec![],
            usage: None,
        };
        assert!(parse_response(raw).is_err());
    }

    #[test]
    fn test_builder_chain() {
        let model = AzureOpenAi::with_api_key("https://x.openai.azure.com", "gpt-4o", "key")
            .with_api_version("2025-01-01")
            .with_timeout(Duration::from_secs(30))
            .with_max_retries(5)
            .with_bearer_token();

        assert_eq!(model.deployment_id, "gpt-4o");
        assert_eq!(model.api_version, "2025-01-01");
        assert_eq!(model.timeout, Some(Duration::from_secs(30)));
        assert_eq!(model.max_retries, 5);
        assert!(model.use_bearer_token);
    }

    #[test]
    fn test_debug_redacts_api_key() {
        let model = AzureOpenAi::with_api_key(
            "https://x.openai.azure.com",
            "gpt-4o",
            "azure-supersecret-key",
        );
        let dbg = format!("{model:?}");
        assert!(
            !dbg.contains("azure-supersecret-key"),
            "Debug output must not contain the plaintext API key: {dbg}"
        );
        assert!(dbg.contains("[redacted]"));
    }

    // --- Streaming SSE-parser tests ---
    //
    // These exercise `handle_azure_sse_line` directly, feeding lines in the
    // exact shapes the (OpenAI-compatible) Azure endpoint emits over SSE.
    // This is the logic that previously lived (untested, duplicated) inside
    // the `try_stream!` block and used the chunk index ("0", "1") as the
    // tool-call id.

    fn feed(state: &mut AzureStreamState, lines: &[&str]) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        for line in lines {
            events.extend(handle_azure_sse_line(state, line));
        }
        events
    }

    #[test]
    fn test_stream_text_delta_and_done() {
        let mut state = AzureStreamState::default();
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
        let mut state = AzureStreamState::default();
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
        let mut state = AzureStreamState::default();
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
        let mut state = AzureStreamState::default();
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
        let mut state = AzureStreamState::default();
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
    fn test_stream_undeserializable_line_is_dropped() {
        let mut state = AzureStreamState::default();
        let events = feed(&mut state, &["data: {not json", ": keep-alive comment", ""]);
        assert!(events.is_empty());
    }
}
