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
    pub fn new(resource_url: impl Into<String>, deployment_id: impl Into<String>) -> Self {
        let api_key = std::env::var("AZURE_OPENAI_API_KEY").unwrap_or_default();
        Self::with_api_key(resource_url, deployment_id, api_key)
    }

    /// Create a new Azure OpenAI client with an explicit API key.
    pub fn with_api_key(
        resource_url: impl Into<String>,
        deployment_id: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self {
            client: build_client(None),
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

    /// Set an HTTP timeout for requests.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self.client = build_client(Some(timeout));
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

        for attempt in 0..=self.max_retries {
            let req = self
                .client
                .post(&url)
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

        let req = self
            .client
            .post(&url)
            .query(&[("api-version", &self.api_version)])
            .json(&body);
        let req = self.apply_auth(req);

        tracing::debug!("sending Azure OpenAI streaming request");
        let response = req
            .send()
            .await
            .map_err(|e| DaimonError::Model(format!("Azure OpenAI HTTP error: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(DaimonError::Model(format!(
                "Azure OpenAI API error ({status}): {text}"
            )));
        }

        tracing::debug!("Azure OpenAI stream established");
        let byte_stream = response.bytes_stream();

        let stream = async_stream::try_stream! {
            use futures::StreamExt;
            use crate::stream_util::LineBuffer;

            let mut buffer = LineBuffer::new();
            let mut stream = Box::pin(byte_stream);

            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| DaimonError::Model(format!("Azure OpenAI stream error: {e}")))?;
                buffer.push(&chunk);

                while let Some(line) = buffer.next_line() {
                    let line = line.trim();

                    if line.is_empty() || line == "data: [DONE]" {
                        if line == "data: [DONE]" {
                            yield StreamEvent::Done;
                        }
                        continue;
                    }

                    if let Some(data) = line.strip_prefix("data: ")
                        && let Ok(chunk) = serde_json::from_str::<AzureStreamChunk>(data) {
                            for choice in &chunk.choices {
                                if let Some(ref content) = choice.delta.content
                                    && !content.is_empty() {
                                        yield StreamEvent::TextDelta(content.clone());
                                    }
                                if let Some(ref tool_calls) = choice.delta.tool_calls {
                                    for tc in tool_calls {
                                        if let Some(ref func) = tc.function {
                                            if let Some(ref name) = func.name {
                                                yield StreamEvent::ToolCallStart {
                                                    id: tc.index.to_string(),
                                                    name: name.clone(),
                                                };
                                            }
                                            if let Some(ref args) = func.arguments
                                                && !args.is_empty() {
                                                    yield StreamEvent::ToolCallDelta {
                                                        id: tc.index.to_string(),
                                                        arguments_delta: args.clone(),
                                                    };
                                                }
                                        }
                                    }
                                }
                            }
                        }
                }
            }

            // A stream may end without a trailing newline, leaving a final SSE
            // record buffered. Recover it through the identical parse path.
            if let Some(line) = buffer.take_remaining() {
                let line = line.trim();
                if line == "data: [DONE]" {
                    yield StreamEvent::Done;
                } else if let Some(data) = line.strip_prefix("data: ")
                    && let Ok(chunk) = serde_json::from_str::<AzureStreamChunk>(data) {
                        for choice in &chunk.choices {
                            if let Some(ref content) = choice.delta.content
                                && !content.is_empty() {
                                    yield StreamEvent::TextDelta(content.clone());
                                }
                            if let Some(ref tool_calls) = choice.delta.tool_calls {
                                for tc in tool_calls {
                                    if let Some(ref func) = tc.function {
                                        if let Some(ref name) = func.name {
                                            yield StreamEvent::ToolCallStart {
                                                id: tc.index.to_string(),
                                                name: name.clone(),
                                            };
                                        }
                                        if let Some(ref args) = func.arguments
                                            && !args.is_empty() {
                                                yield StreamEvent::ToolCallDelta {
                                                    id: tc.index.to_string(),
                                                    arguments_delta: args.clone(),
                                                };
                                            }
                                    }
                                }
                            }
                        }
                    }
            }
        };

        Ok(Box::pin(stream))
    }
}

fn parse_response(response: AzureResponse) -> Result<ChatResponse> {
    let choice = response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| DaimonError::Model("no choices in Azure OpenAI response".into()))?;

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
}

#[derive(Deserialize)]
struct AzureStreamDelta {
    content: Option<String>,
    tool_calls: Option<Vec<AzureStreamToolCall>>,
}

#[derive(Deserialize)]
struct AzureStreamToolCall {
    index: usize,
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
}
