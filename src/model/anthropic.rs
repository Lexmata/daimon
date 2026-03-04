//! Anthropic Claude model provider.
//!
//! This module provides integration with the Anthropic API for chat completions,
//! streaming, and tool use. Configure via builder methods for timeout, retries,
//! and prompt caching.

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
const PROMPT_CACHING_BETA: &str = "prompt-caching-2024-07-31";

fn build_client(timeout: Option<Duration>) -> Client {
    let mut builder = Client::builder();
    if let Some(t) = timeout {
        builder = builder.timeout(t);
    }
    builder.build().expect("failed to build HTTP client")
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

impl Anthropic {
    /// Creates a new Anthropic client using `ANTHROPIC_API_KEY` from the environment.
    pub fn new(model_id: impl Into<String>) -> Self {
        let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
        Self::with_api_key(model_id, api_key)
    }

    /// Creates a new Anthropic client with an explicit API key.
    pub fn with_api_key(model_id: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            client: build_client(None),
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

    /// Sets a timeout for HTTP requests.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self.client = build_client(Some(timeout));
        self
    }

    /// Sets the maximum number of retries for retriable errors (429, 529, 5xx).
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Enables prompt caching via the `anthropic-beta` header.
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
                        let blocks: Vec<AnthropicContentBlock> = msg
                            .tool_calls
                            .iter()
                            .map(|tc| AnthropicContentBlock::ToolUse {
                                id: tc.id.clone(),
                                name: tc.name.clone(),
                                input: tc.arguments.clone(),
                            })
                            .collect();
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
            let mut tool_list: Vec<AnthropicTool> =
                request.tools.iter().map(Into::into).collect();
            if self.use_prompt_caching {
                if let Some(last) = tool_list.last_mut() {
                    last.cache_control =
                        Some(serde_json::json!({"type": "ephemeral"}));
                }
            }
            Some(tool_list)
        };

        AnthropicRequest {
            model: self.model_id.clone(),
            system,
            messages,
            tools,
            max_tokens: request.max_tokens.unwrap_or(4096),
            temperature: request.temperature,
            stream: false,
        }
    }
}

impl Model for Anthropic {
    #[tracing::instrument(skip_all, fields(model = %self.model_id))]
    async fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let body = self.build_request_body(request);
        let url = format!("{}/v1/messages", self.base_url);

        tracing::debug!("building request for non-streaming generate");
        let mut attempt = 0u32;

        loop {
            let mut req_builder = self
                .client
                .post(&url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", API_VERSION)
                .header("content-type", "application/json")
                .json(&body);

            if self.use_prompt_caching {
                req_builder = req_builder.header("anthropic-beta", PROMPT_CACHING_BETA);
            }

            tracing::debug!(attempt = attempt, "sending request to Anthropic API");
            let response = req_builder.send().await
                .map_err(|e| DaimonError::Model(format!("Anthropic HTTP error: {e}")))?;
            let status = response.status();
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
                let delay_ms = 100 * 2_u64.pow(attempt);
                let delay = Duration::from_millis(delay_ms);
                tracing::debug!(
                    status = %status,
                    attempt = attempt,
                    delay_ms = delay_ms,
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
        let mut req_builder = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
            .json(&body);

        if self.use_prompt_caching {
            req_builder = req_builder.header("anthropic-beta", PROMPT_CACHING_BETA);
        }

        tracing::debug!("sending streaming request to Anthropic API");
        let response = req_builder.send().await
            .map_err(|e| DaimonError::Model(format!("Anthropic HTTP error: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(DaimonError::Model(format!(
                "Anthropic API error ({status}): {text}"
            )));
        }

        tracing::debug!("stream established, processing events");
        let byte_stream = response.bytes_stream();

        let stream = async_stream::try_stream! {
            use futures::StreamExt;

            let mut buffer = String::new();
            let mut stream = Box::pin(byte_stream);

            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| DaimonError::Model(format!("Anthropic stream error: {e}")))?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(line_end) = buffer.find('\n') {
                    let line = buffer[..line_end].trim().to_string();
                    buffer = buffer[line_end + 1..].to_string();

                    if line.is_empty() {
                        continue;
                    }

                    if let Some(data) = line.strip_prefix("data: ") {
                        if let Ok(event) = serde_json::from_str::<AnthropicStreamEvent>(data) {
                            match event.r#type.as_str() {
                                "content_block_start" => {
                                    if let Some(block) = event.content_block {
                                        if block.r#type == "tool_use" {
                                            yield StreamEvent::ToolCallStart {
                                                id: block.id.unwrap_or_default(),
                                                name: block.name.unwrap_or_default(),
                                            };
                                        }
                                    }
                                }
                                "content_block_delta" => {
                                    if let Some(delta) = event.delta {
                                        if let Some(text) = delta.text {
                                            yield StreamEvent::TextDelta(text);
                                        }
                                        if let Some(json) = delta.partial_json {
                                            yield StreamEvent::ToolCallDelta {
                                                id: String::new(),
                                                arguments_delta: json,
                                            };
                                        }
                                    }
                                }
                                "message_stop" => {
                                    yield StreamEvent::Done;
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        };

        Ok(Box::pin(stream))
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
    content_block: Option<AnthropicStreamBlock>,
    delta: Option<AnthropicStreamDelta>,
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
        let model = Anthropic::new("claude-sonnet-4-20250514");
        assert_eq!(model.model_id, "claude-sonnet-4-20250514");
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
        assert!(sys.is_array(), "system should be array when caching enabled");
        let blocks = sys.as_array().unwrap();
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[0]["text"], "Be helpful");
        assert!(blocks[0]["cache_control"].is_object());
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
        assert!(tools[0].cache_control.is_none(), "only last tool gets cache_control");
        assert!(tools[1].cache_control.is_some(), "last tool should have cache_control");
    }
}
