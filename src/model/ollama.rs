//! Ollama local model provider.
//!
//! Connects to an [Ollama](https://ollama.com) instance running locally (or remotely).
//! Uses the `/api/chat` endpoint with streaming support.
//!
//! # Example
//!
//! ```ignore
//! use daimon::model::ollama::Ollama;
//!
//! let model = Ollama::new("llama3.1");
//! ```

use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::error::{DaimonError, Result};
use crate::model::Model;
use crate::model::types::{ChatRequest, ChatResponse, Message, Role, StopReason, ToolSpec, Usage};
use crate::stream::{ResponseStream, StreamEvent};
use crate::tool::ToolCall;

/// Ollama model provider.
///
/// Communicates with a running Ollama server via its REST API. Defaults to
/// `http://localhost:11434` but can be configured with [`with_base_url`](Ollama::with_base_url).
pub struct Ollama {
    model: String,
    base_url: String,
    client: Client,
    timeout: Duration,
    keep_alive: Option<String>,
}

impl Ollama {
    /// Creates a new Ollama provider for the given model name (e.g. `"llama3.1"`).
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            base_url: "http://localhost:11434".to_string(),
            // A dead or unreachable server fails fast at connect time instead
            // of blocking; the request itself is bounded by `timeout` below.
            client: Client::builder()
                .connect_timeout(Duration::from_secs(10))
                .build()
                .expect("failed to build HTTP client"),
            timeout: Duration::from_secs(300),
            keep_alive: None,
        }
    }

    /// Overrides the Ollama server URL (default: `http://localhost:11434`).
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into().trim_end_matches('/').to_string();
        self
    }

    /// Sets the request timeout (default: 300 seconds).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Controls how long the model stays loaded in GPU memory after a request.
    ///
    /// Ollama keeps the model loaded so subsequent requests reuse the KV cache.
    /// Pass a duration string like `"5m"`, `"1h"`, or `"0"` to unload immediately.
    /// The default Ollama behaviour (when unset) is `"5m"`.
    pub fn with_keep_alive(mut self, keep_alive: impl Into<String>) -> Self {
        self.keep_alive = Some(keep_alive.into());
        self
    }

    fn build_request_body(&self, request: &ChatRequest, stream: bool) -> serde_json::Value {
        let messages: Vec<serde_json::Value> =
            request.messages.iter().map(convert_message).collect();

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "stream": stream,
        });

        if !request.tools.is_empty() {
            let tools: Vec<serde_json::Value> =
                request.tools.iter().map(convert_tool_spec).collect();
            body["tools"] = serde_json::Value::Array(tools);
        }

        if let Some(temp) = request.temperature {
            body["options"]["temperature"] = serde_json::json!(temp);
        }

        // Ollama expresses the output-token limit as `options.num_predict`.
        // Previously `request.max_tokens` was silently dropped, so callers
        // could not bound generation length at all.
        if let Some(mt) = request.max_tokens {
            body["options"]["num_predict"] = serde_json::json!(mt);
        }

        if let Some(ref ka) = self.keep_alive {
            body["keep_alive"] = serde_json::Value::String(ka.clone());
        }

        body
    }
}

impl Model for Ollama {
    fn model_id(&self) -> &str {
        &self.model
    }

    #[tracing::instrument(skip_all, fields(model = %self.model))]
    async fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let body = self.build_request_body(request, false);
        let url = format!("{}/api/chat", self.base_url);

        let resp = self
            .client
            .post(&url)
            .timeout(self.timeout)
            .json(&body)
            .send()
            .await
            .map_err(|e| DaimonError::Model(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(DaimonError::Model(format!("Ollama {status}: {text}")));
        }

        let response: OllamaResponse = resp
            .json()
            .await
            .map_err(|e| DaimonError::Model(e.to_string()))?;

        parse_response(response)
    }

    #[tracing::instrument(skip_all, fields(model = %self.model))]
    async fn generate_stream(&self, request: &ChatRequest) -> Result<ResponseStream> {
        let body = self.build_request_body(request, true);
        let url = format!("{}/api/chat", self.base_url);

        let resp = self
            .client
            .post(&url)
            .timeout(self.timeout)
            .json(&body)
            .send()
            .await
            .map_err(|e| DaimonError::Model(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(DaimonError::Model(format!("Ollama {status}: {text}")));
        }

        let stream = async_stream::try_stream! {
            use futures::StreamExt;
            use crate::model::line_buffer::LineBuffer;

            let mut byte_stream = resp.bytes_stream();
            let mut buffer = LineBuffer::new();
            // Monotonic across the whole stream: Ollama emits one tool call per
            // NDJSON message, so a per-message index resets every iteration and
            // collides across messages. A stream-wide counter keeps ids unique.
            let mut tool_call_seq: u64 = 0;

            while let Some(chunk) = byte_stream.next().await {
                let chunk = chunk.map_err(|e| DaimonError::Model(e.to_string()))?;
                buffer.push(&chunk);

                while let Some(line) = buffer.next_line() {
                    let line = line.trim();

                    if line.is_empty() {
                        continue;
                    }

                    let parsed: OllamaResponse = serde_json::from_str(line)
                        .map_err(|e| DaimonError::Model(format!("invalid JSON: {e}")))?;

                    if let Some(ref msg) = parsed.message {
                        if !msg.tool_calls.is_empty() {
                            for tc in &msg.tool_calls {
                                let id = format!("ollama_tc_{tool_call_seq}");
                                tool_call_seq += 1;
                                yield StreamEvent::ToolCallStart {
                                    id: id.clone(),
                                    name: tc.function.name.clone(),
                                };
                                let args_str = serde_json::to_string(&tc.function.arguments)
                                    .unwrap_or_default();
                                yield StreamEvent::ToolCallDelta {
                                    id: id.clone(),
                                    arguments_delta: args_str,
                                };
                                yield StreamEvent::ToolCallEnd { id };
                            }
                        }

                        if let Some(ref content) = msg.content
                            && !content.is_empty() {
                                yield StreamEvent::TextDelta(content.clone());
                            }
                    }

                    if parsed.done {
                        yield StreamEvent::Done;
                    }
                }
            }

            // Recover a final NDJSON record the server sent without a trailing
            // newline through the identical parse path used for normal lines.
            if let Some(line) = buffer.take_remaining() {
                let line = line.trim();
                if !line.is_empty() {
                    let parsed: OllamaResponse = serde_json::from_str(line)
                        .map_err(|e| DaimonError::Model(format!("invalid JSON: {e}")))?;

                    if let Some(ref msg) = parsed.message {
                        if !msg.tool_calls.is_empty() {
                            for tc in &msg.tool_calls {
                                let id = format!("ollama_tc_{tool_call_seq}");
                                tool_call_seq += 1;
                                yield StreamEvent::ToolCallStart {
                                    id: id.clone(),
                                    name: tc.function.name.clone(),
                                };
                                let args_str = serde_json::to_string(&tc.function.arguments)
                                    .unwrap_or_default();
                                yield StreamEvent::ToolCallDelta {
                                    id: id.clone(),
                                    arguments_delta: args_str,
                                };
                                yield StreamEvent::ToolCallEnd { id };
                            }
                        }

                        if let Some(ref content) = msg.content
                            && !content.is_empty() {
                                yield StreamEvent::TextDelta(content.clone());
                            }
                    }

                    if parsed.done {
                        yield StreamEvent::Done;
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

fn convert_message(msg: &Message) -> serde_json::Value {
    let role = match msg.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };

    let mut obj = serde_json::json!({"role": role});

    if let Some(ref content) = msg.content {
        obj["content"] = serde_json::Value::String(content.clone());
    }

    if !msg.tool_calls.is_empty() {
        let calls: Vec<serde_json::Value> = msg
            .tool_calls
            .iter()
            .map(|tc| {
                serde_json::json!({
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                })
            })
            .collect();
        obj["tool_calls"] = serde_json::Value::Array(calls);
    }

    obj
}

fn convert_tool_spec(spec: &ToolSpec) -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        }
    })
}

fn parse_response(resp: OllamaResponse) -> Result<ChatResponse> {
    let msg = resp
        .message
        .ok_or_else(|| DaimonError::Model("missing message in Ollama response".into()))?;

    let has_tool_calls = !msg.tool_calls.is_empty();

    let tool_calls: Vec<ToolCall> = msg
        .tool_calls
        .into_iter()
        .enumerate()
        .map(|(i, tc)| ToolCall {
            id: format!("ollama_tc_{i}"),
            name: tc.function.name,
            arguments: tc.function.arguments,
        })
        .collect();

    let stop_reason = if has_tool_calls {
        StopReason::ToolUse
    } else {
        StopReason::EndTurn
    };

    let message = if tool_calls.is_empty() {
        Message::assistant(msg.content.unwrap_or_default())
    } else {
        let mut m = Message::assistant_with_tool_calls(tool_calls);
        m.content = msg.content;
        m
    };

    let usage = resp.prompt_eval_count.map(|input| Usage {
        input_tokens: input,
        output_tokens: resp.eval_count.unwrap_or(0),
        cached_tokens: 0,
    });

    Ok(ChatResponse {
        message,
        stop_reason,
        usage,
    })
}

#[derive(Deserialize)]
struct OllamaResponse {
    #[serde(default)]
    message: Option<OllamaMessage>,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    eval_count: Option<u32>,
}

#[derive(Deserialize)]
struct OllamaMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<OllamaToolCall>,
}

#[derive(Deserialize)]
struct OllamaToolCall {
    function: OllamaFunction,
}

#[derive(Deserialize)]
struct OllamaFunction {
    name: String,
    #[serde(default)]
    arguments: serde_json::Value,
}

#[allow(dead_code)]
#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    messages: Vec<serde_json::Value>,
    stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_new() {
        let model = Ollama::new("llama3.1");
        assert_eq!(model.model, "llama3.1");
        assert_eq!(model.base_url, "http://localhost:11434");
    }

    #[test]
    fn test_with_base_url() {
        let model = Ollama::new("llama3.1").with_base_url("http://remote:11434/");
        assert_eq!(model.base_url, "http://remote:11434");
    }

    #[test]
    fn test_convert_message_user() {
        let msg = Message::user("hello");
        let json = convert_message(&msg);
        assert_eq!(json["role"], "user");
        assert_eq!(json["content"], "hello");
    }

    #[test]
    fn test_convert_message_assistant_with_tool_calls() {
        let msg = Message::assistant_with_tool_calls(vec![ToolCall {
            id: "1".into(),
            name: "test".into(),
            arguments: serde_json::json!({"a": 1}),
        }]);
        let json = convert_message(&msg);
        assert_eq!(json["role"], "assistant");
        assert!(json["tool_calls"].is_array());
    }

    #[test]
    fn test_convert_tool_spec() {
        let spec = ToolSpec {
            name: "calc".into(),
            description: "Calculator".into(),
            parameters: serde_json::json!({"type": "object"}),
        };
        let json = convert_tool_spec(&spec);
        assert_eq!(json["type"], "function");
        assert_eq!(json["function"]["name"], "calc");
    }

    #[test]
    fn test_parse_response_text() {
        let resp = OllamaResponse {
            message: Some(OllamaMessage {
                content: Some("Hello!".into()),
                tool_calls: vec![],
            }),
            done: true,
            prompt_eval_count: Some(10),
            eval_count: Some(5),
        };
        let result = parse_response(resp).unwrap();
        assert_eq!(result.message.content.as_deref(), Some("Hello!"));
        assert_eq!(result.stop_reason, StopReason::EndTurn);
        assert_eq!(result.usage.as_ref().unwrap().input_tokens, 10);
    }

    #[test]
    fn test_parse_response_tool_call() {
        let resp = OllamaResponse {
            message: Some(OllamaMessage {
                content: None,
                tool_calls: vec![OllamaToolCall {
                    function: OllamaFunction {
                        name: "calc".into(),
                        arguments: serde_json::json!({"expr": "1+1"}),
                    },
                }],
            }),
            done: true,
            prompt_eval_count: None,
            eval_count: None,
        };
        let result = parse_response(resp).unwrap();
        assert_eq!(result.stop_reason, StopReason::ToolUse);
        assert_eq!(result.message.tool_calls.len(), 1);
        assert_eq!(result.message.tool_calls[0].name, "calc");
    }

    #[test]
    fn test_build_request_body() {
        let model = Ollama::new("llama3.1");
        let request = ChatRequest::new(vec![Message::user("hi")]);
        let body = model.build_request_body(&request, false);
        assert_eq!(body["model"], "llama3.1");
        assert_eq!(body["stream"], false);
    }

    #[test]
    fn test_build_request_body_with_tools() {
        let model = Ollama::new("llama3.1");
        let request = ChatRequest {
            messages: vec![Message::user("hi")],
            tools: vec![ToolSpec {
                name: "test".into(),
                description: "test".into(),
                parameters: serde_json::json!({"type": "object"}),
            }],
            temperature: Some(0.5),
            max_tokens: None,
        };
        let body = model.build_request_body(&request, true);
        assert!(body["tools"].is_array());
        assert_eq!(body["options"]["temperature"], 0.5);
    }

    #[test]
    fn test_build_request_body_maps_max_tokens() {
        let model = Ollama::new("llama3.1");
        let request = ChatRequest {
            messages: vec![Message::user("hi")],
            tools: vec![],
            temperature: Some(0.5),
            max_tokens: Some(256),
        };
        let body = model.build_request_body(&request, false);
        assert_eq!(body["options"]["num_predict"], 256);
        // temperature and num_predict must coexist under options.
        assert_eq!(body["options"]["temperature"], 0.5);
    }

    #[test]
    fn test_build_request_body_no_max_tokens() {
        let model = Ollama::new("llama3.1");
        let request = ChatRequest::new(vec![Message::user("hi")]);
        let body = model.build_request_body(&request, false);
        assert!(body["options"]["num_predict"].is_null());
    }
}
