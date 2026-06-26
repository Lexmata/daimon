//! OpenAI model provider for the Daimon agent framework.
//!
//! This module provides an implementation of the [`Model`] trait that connects
//! to the OpenAI Chat Completions API. It supports configurable timeouts,
//! retries with exponential backoff, response format constraints, and
//! parallel tool calls.

use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::error::{DaimonError, Result};
use crate::model::Model;
use crate::model::types::{ChatRequest, ChatResponse, Message, Role, StopReason, ToolSpec, Usage};
use crate::stream::{ResponseStream, StreamEvent};
use crate::tool::ToolCall;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

pub const DEFAULT_MAX_RETRIES: u32 = 3;

fn build_client(timeout: Option<Duration>) -> Client {
    let mut builder = Client::builder();
    if let Some(t) = timeout {
        builder = builder.timeout(t);
    }
    builder.build().expect("failed to build HTTP client")
}

/// OpenAI model provider for the Chat Completions API.
///
/// Supports configurable timeouts, retries, response format, and parallel tool calls.
/// Use `new()` or `with_api_key()` to create, then chain builder methods as needed.
#[derive(Debug)]
pub struct OpenAi {
    client: Client,
    api_key: String,
    model_id: String,
    base_url: String,
    timeout: Option<Duration>,
    max_retries: u32,
    response_format: Option<String>,
    parallel_tool_calls: Option<bool>,
}

impl OpenAi {
    /// Create a new OpenAI client using the default model ID.
    ///
    /// Reads `OPENAI_API_KEY` from the environment. Use `with_api_key()` to
    /// provide the key explicitly.
    pub fn new(model_id: impl Into<String>) -> Self {
        let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        Self::with_api_key(model_id, api_key)
    }

    /// Create a new OpenAI client with an explicit API key.
    pub fn with_api_key(model_id: impl Into<String>, api_key: impl Into<String>) -> Self {
        let timeout = None;
        Self {
            client: build_client(timeout),
            api_key: api_key.into(),
            model_id: model_id.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            timeout,
            max_retries: DEFAULT_MAX_RETRIES,
            response_format: None,
            parallel_tool_calls: None,
        }
    }

    /// Set a custom base URL (e.g. for proxies or local endpoints).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Set a custom timeout for HTTP requests.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self.client = build_client(Some(timeout));
        self
    }

    /// Set the maximum number of retries for failed requests (429 and 5xx).
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Set the response format (e.g. `"json_object"` or `"text"`).
    pub fn with_response_format(mut self, format: &str) -> Self {
        self.response_format = Some(format.to_string());
        self
    }

    /// Enable or disable parallel tool calls.
    pub fn with_parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = Some(enabled);
        self
    }

    fn build_request_body(&self, request: &ChatRequest) -> OpenAiRequest {
        let messages: Vec<OpenAiMessage> = request.messages.iter().map(Into::into).collect();

        let tools: Option<Vec<OpenAiTool>> = if request.tools.is_empty() {
            None
        } else {
            Some(request.tools.iter().map(Into::into).collect())
        };

        OpenAiRequest {
            model: self.model_id.clone(),
            messages,
            tools,
            temperature: request.temperature,
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

impl Model for OpenAi {
    #[tracing::instrument(skip_all, fields(model = %self.model_id))]
    async fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let body = self.build_request_body(request);
        let url = format!("{}/chat/completions", self.base_url);

        tracing::debug!("sending chat completion request");

        for attempt in 0..=self.max_retries {
            let response = self
                .client
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .json(&body)
                .send()
                .await
                .map_err(|e| DaimonError::Model(format!("OpenAI HTTP error: {e}")))?;

            let status = response.status();

            if status.is_success() {
                tracing::debug!("received successful response");
                let oai_response: OpenAiResponse = response
                    .json()
                    .await
                    .map_err(|e| DaimonError::Model(format!("OpenAI response parse error: {e}")))?;
                return parse_response(oai_response);
            }

            let text = response.text().await.unwrap_or_default();
            let is_retryable = status.as_u16() == 429 || status.is_server_error();

            if is_retryable && attempt < self.max_retries {
                let delay_ms = 100 * 2u64.pow(attempt);
                tracing::debug!(
                    status = %status,
                    attempt = attempt + 1,
                    max_retries = self.max_retries,
                    delay_ms,
                    "retryable error, backing off"
                );
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            } else {
                return Err(DaimonError::Model(format!(
                    "OpenAI API error ({status}): {text}"
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

        tracing::debug!("sending streaming chat completion request");
        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| DaimonError::Model(format!("OpenAI HTTP error: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(DaimonError::Model(format!(
                "OpenAI API error ({status}): {text}"
            )));
        }

        tracing::debug!("stream established, processing chunks");
        let byte_stream = response.bytes_stream();

        let stream = async_stream::try_stream! {
            use futures::StreamExt;

            let mut buffer = String::new();
            let mut stream = Box::pin(byte_stream);

            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| DaimonError::Model(format!("OpenAI stream error: {e}")))?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(line_end) = buffer.find('\n') {
                    let line = buffer[..line_end].trim().to_string();
                    buffer = buffer[line_end + 1..].to_string();

                    if line.is_empty() || line == "data: [DONE]" {
                        if line == "data: [DONE]" {
                            yield StreamEvent::Done;
                        }
                        continue;
                    }

                    if let Some(data) = line.strip_prefix("data: ")
                        && let Ok(chunk) = serde_json::from_str::<OpenAiStreamChunk>(data) {
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
        };

        Ok(Box::pin(stream))
    }
}

fn parse_response(response: OpenAiResponse) -> Result<ChatResponse> {
    let choice = response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| DaimonError::Model("no choices in OpenAI response".into()))?;

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

    let message = if tool_calls.is_empty() {
        Message {
            role: Role::Assistant,
            content: choice.message.content,
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    } else {
        Message {
            role: Role::Assistant,
            content: choice.message.content,
            tool_calls,
            tool_call_id: None,
        }
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

// --- OpenAI API types ---

#[derive(Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAiTool>>,
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
struct OpenAiMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAiToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

impl From<&Message> for OpenAiMessage {
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
                    .map(|tc| OpenAiToolCall {
                        id: tc.id.clone(),
                        r#type: "function".to_string(),
                        function: OpenAiFunction {
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
struct OpenAiTool {
    r#type: String,
    function: OpenAiToolFunction,
}

impl From<&ToolSpec> for OpenAiTool {
    fn from(spec: &ToolSpec) -> Self {
        Self {
            r#type: "function".to_string(),
            function: OpenAiToolFunction {
                name: spec.name.clone(),
                description: spec.description.clone(),
                parameters: spec.parameters.clone(),
            },
        }
    }
}

#[derive(Serialize)]
struct OpenAiToolFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
    usage: Option<OpenAiUsage>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    message: OpenAiChoiceMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiChoiceMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Serialize, Deserialize)]
struct OpenAiToolCall {
    #[serde(default)]
    id: String,
    #[serde(default)]
    r#type: String,
    #[serde(default)]
    function: OpenAiFunction,
    #[serde(default)]
    index: usize,
}

#[derive(Serialize, Deserialize, Default)]
struct OpenAiFunction {
    #[serde(default)]
    name: String,
    #[serde(default)]
    arguments: String,
}

#[derive(Deserialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    prompt_tokens_details: Option<OpenAiPromptTokensDetails>,
}

#[derive(Deserialize)]
struct OpenAiPromptTokensDetails {
    #[serde(default)]
    cached_tokens: u32,
}

#[derive(Deserialize)]
struct OpenAiStreamChunk {
    choices: Vec<OpenAiStreamChoice>,
}

#[derive(Deserialize)]
struct OpenAiStreamChoice {
    delta: OpenAiStreamDelta,
}

#[derive(Deserialize)]
struct OpenAiStreamDelta {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAiStreamToolCall>>,
}

#[derive(Deserialize)]
struct OpenAiStreamToolCall {
    index: usize,
    function: Option<OpenAiStreamFunction>,
}

#[derive(Deserialize)]
struct OpenAiStreamFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_conversion_user() {
        let msg = Message::user("hello");
        let oai: OpenAiMessage = (&msg).into();
        assert_eq!(oai.role, "user");
        assert_eq!(oai.content.as_deref(), Some("hello"));
        assert!(oai.tool_calls.is_none());
    }

    #[test]
    fn test_message_conversion_assistant_with_tools() {
        let msg = Message::assistant_with_tool_calls(vec![ToolCall {
            id: "tc_1".into(),
            name: "calc".into(),
            arguments: serde_json::json!({"x": 1}),
        }]);
        let oai: OpenAiMessage = (&msg).into();
        assert_eq!(oai.role, "assistant");
        assert!(oai.tool_calls.is_some());
        assert_eq!(oai.tool_calls.unwrap().len(), 1);
    }

    #[test]
    fn test_message_conversion_tool_result() {
        let msg = Message::tool_result("tc_1", "42");
        let oai: OpenAiMessage = (&msg).into();
        assert_eq!(oai.role, "tool");
        assert_eq!(oai.tool_call_id.as_deref(), Some("tc_1"));
        assert_eq!(oai.content.as_deref(), Some("42"));
    }

    #[test]
    fn test_tool_spec_conversion() {
        let spec = ToolSpec {
            name: "calc".into(),
            description: "Calculator".into(),
            parameters: serde_json::json!({"type": "object"}),
        };
        let oai: OpenAiTool = (&spec).into();
        assert_eq!(oai.r#type, "function");
        assert_eq!(oai.function.name, "calc");
    }

    #[test]
    fn test_parse_response_end_turn() {
        let raw = OpenAiResponse {
            choices: vec![OpenAiChoice {
                message: OpenAiChoiceMessage {
                    content: Some("hello".into()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".into()),
            }],
            usage: Some(OpenAiUsage {
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
        let raw = OpenAiResponse {
            choices: vec![OpenAiChoice {
                message: OpenAiChoiceMessage {
                    content: None,
                    tool_calls: Some(vec![OpenAiToolCall {
                        id: "tc_1".into(),
                        r#type: "function".into(),
                        function: OpenAiFunction {
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
        let raw = OpenAiResponse {
            choices: vec![],
            usage: None,
        };
        let result = parse_response(raw);
        assert!(result.is_err());
    }

    #[test]
    fn test_openai_new_default() {
        let model = OpenAi::new("gpt-4o");
        assert_eq!(model.model_id, "gpt-4o");
        assert_eq!(model.base_url, DEFAULT_BASE_URL);
        assert_eq!(model.max_retries, DEFAULT_MAX_RETRIES);
    }

    #[test]
    fn test_openai_with_base_url() {
        let model = OpenAi::new("gpt-4o").with_base_url("http://localhost:8080");
        assert_eq!(model.base_url, "http://localhost:8080");
    }

    #[test]
    fn test_with_timeout() {
        let model = OpenAi::new("gpt-4o").with_timeout(std::time::Duration::from_secs(60));
        assert_eq!(model.timeout, Some(std::time::Duration::from_secs(60)));
    }

    #[test]
    fn test_with_max_retries() {
        let model = OpenAi::new("gpt-4o").with_max_retries(5);
        assert_eq!(model.max_retries, 5);
    }

    #[test]
    fn test_with_response_format() {
        let model = OpenAi::new("gpt-4o").with_response_format("json_object");
        assert_eq!(model.response_format.as_deref(), Some("json_object"));
    }

    #[test]
    fn test_with_parallel_tool_calls() {
        let model = OpenAi::new("gpt-4o").with_parallel_tool_calls(true);
        assert_eq!(model.parallel_tool_calls, Some(true));
    }
}
