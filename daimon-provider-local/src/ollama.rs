//! Ollama local model provider.
//!
//! Connects to an [Ollama](https://ollama.com) instance running locally (or remotely).
//! Uses the `/api/chat` endpoint with streaming support.
//!
//! # Example
//!
//! ```ignore
//! use daimon::model::ollama::Ollama;
//! use daimon_core::{ChatRequest, Message, Model};
//!
//! let model = Ollama::new("llama3.1");
//! let response = model
//!     .generate(&ChatRequest::new(vec![Message::user("hello")]))
//!     .await?;
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};

use daimon_core::stream_util::{backoff_delay, parse_retry_after_secs};
use daimon_core::{
    ChatRequest, ChatResponse, DaimonError, Message, Model, ResponseStream, Result, Role,
    StopReason, StreamEvent, ToolCall, ToolSpec, Usage,
};

/// Default number of retries for transient (429 / 5xx) errors on the initial
/// request. Matches [`crate::ollama_embed`]'s `DEFAULT_MAX_RETRIES`.
const DEFAULT_MAX_RETRIES: u32 = 3;

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
    max_retries: u32,
    /// Client-wide monotonic counter for synthesized tool-call ids.
    ///
    /// Ollama does not assign tool-call ids, so this provider synthesizes
    /// them as `ollama_tc_{seq}_{name}`. The counter is shared by the
    /// streaming and non-streaming paths so ids never collide across turns
    /// of a conversation (a per-response index restarts at 0 every reply).
    /// The sequence number precedes the name so the function name — which may
    /// itself contain digits and underscores — is unambiguously recoverable
    /// from an echoed `tool_call_id`.
    tool_call_seq: Arc<AtomicU64>,
}

/// Prefix used for synthesized tool-call ids (`ollama_tc_{seq}_{name}`).
const TOOL_CALL_ID_PREFIX: &str = "ollama_tc_";

/// Synthesizes a tool-call id in the `ollama_tc_{seq}_{name}` format.
fn make_tool_call_id(seq: u64, name: &str) -> String {
    format!("{TOOL_CALL_ID_PREFIX}{seq}_{name}")
}

/// Recovers the function name from a synthetic `ollama_tc_{seq}_{name}` id.
///
/// Returns `None` when the id does not follow the synthetic format; callers
/// then omit the `tool_name` field gracefully.
fn tool_name_from_call_id(id: &str) -> Option<&str> {
    let rest = id.strip_prefix(TOOL_CALL_ID_PREFIX)?;
    let (seq, name) = rest.split_once('_')?;
    if seq.is_empty() || !seq.bytes().all(|b| b.is_ascii_digit()) || name.is_empty() {
        return None;
    }
    Some(name)
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
            max_retries: DEFAULT_MAX_RETRIES,
            tool_call_seq: Arc::new(AtomicU64::new(0)),
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

    /// Set the maximum number of retries for transient (429 / 5xx) errors
    /// on the initial request (default: 3).
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
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

        for attempt in 0..=self.max_retries {
            let resp = self
                .client
                .post(&url)
                .timeout(self.timeout)
                .json(&body)
                .send()
                .await
                .map_err(|e| DaimonError::Model(e.to_string()))?;

            let status = resp.status();
            if status.is_success() {
                let response: OllamaResponse = resp
                    .json()
                    .await
                    .map_err(|e| DaimonError::Model(e.to_string()))?;
                return parse_response(response, &self.tool_call_seq);
            }

            let retry_after = parse_retry_after(&resp);
            let text = resp.text().await.unwrap_or_default();
            let is_retryable = status.as_u16() == 429 || status.is_server_error();

            if is_retryable && attempt < self.max_retries {
                let delay = backoff_delay(attempt, retry_after);
                tracing::debug!(status = %status, attempt, delay_ms = delay.as_millis(), "retryable Ollama error, backing off");
                tokio::time::sleep(delay).await;
            } else {
                return Err(DaimonError::Model(format!("Ollama {status}: {text}")));
            }
        }

        unreachable!("loop always returns or retries")
    }

    #[tracing::instrument(skip_all, fields(model = %self.model))]
    async fn generate_stream(&self, request: &ChatRequest) -> Result<ResponseStream> {
        let body = self.build_request_body(request, true);
        let url = format!("{}/api/chat", self.base_url);

        // Retry only the initial POST/handshake — once the stream is
        // established, mid-stream failures must never be retried (the
        // consumer has already observed a partial response).
        let mut response = None;
        for attempt in 0..=self.max_retries {
            let resp = self
                .client
                .post(&url)
                .timeout(self.timeout)
                .json(&body)
                .send()
                .await
                .map_err(|e| DaimonError::Model(e.to_string()))?;

            let status = resp.status();
            if status.is_success() {
                response = Some(resp);
                break;
            }

            let retry_after = parse_retry_after(&resp);
            let text = resp.text().await.unwrap_or_default();
            let is_retryable = status.as_u16() == 429 || status.is_server_error();

            if is_retryable && attempt < self.max_retries {
                let delay = backoff_delay(attempt, retry_after);
                tracing::debug!(status = %status, attempt, delay_ms = delay.as_millis(), "retryable Ollama stream-handshake error, backing off");
                tokio::time::sleep(delay).await;
            } else {
                return Err(DaimonError::Model(format!("Ollama {status}: {text}")));
            }
        }
        let resp = response.expect("loop breaks with a response or returns an error");

        let tool_call_seq = Arc::clone(&self.tool_call_seq);
        let stream = async_stream::try_stream! {
            use futures::StreamExt;
            use daimon_core::stream_util::LineBuffer;

            let mut byte_stream = resp.bytes_stream();
            let mut buffer = LineBuffer::new();

            while let Some(chunk) = byte_stream.next().await {
                let chunk = chunk.map_err(|e| DaimonError::Model(e.to_string()))?;
                buffer.push(&chunk);

                while let Some(line) = buffer.next_line() {
                    for event in ndjson_line_events(&line, &tool_call_seq)? {
                        yield event;
                    }
                }
            }

            // Recover a final NDJSON record the server sent without a trailing
            // newline through the identical parse path used for normal lines.
            if let Some(line) = buffer.take_remaining() {
                for event in ndjson_line_events(&line, &tool_call_seq)? {
                    yield event;
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

/// Extracts the `Retry-After` header (integer seconds) from an Ollama
/// response, if present.
fn parse_retry_after(resp: &reqwest::Response) -> Option<Duration> {
    resp.headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(parse_retry_after_secs)
        .map(Duration::from_secs)
}

/// Parses one NDJSON line from Ollama's `/api/chat` stream into the
/// [`StreamEvent`]s it carries.
///
/// Pure and independent of the stream so it's unit-testable; shared by both
/// the main loop and the no-trailing-newline recovery path in
/// [`Ollama::generate_stream`], which previously hand-copied this logic
/// twice.
fn ndjson_line_events(line: &str, tool_call_seq: &AtomicU64) -> Result<Vec<StreamEvent>> {
    let line = line.trim();
    let mut events = Vec::new();

    if line.is_empty() {
        return Ok(events);
    }

    let parsed: OllamaResponse =
        serde_json::from_str(line).map_err(|e| DaimonError::Model(format!("invalid JSON: {e}")))?;

    if let Some(ref msg) = parsed.message {
        if !msg.tool_calls.is_empty() {
            for tc in &msg.tool_calls {
                let seq = tool_call_seq.fetch_add(1, Ordering::Relaxed);
                let id = make_tool_call_id(seq, &tc.function.name);
                events.push(StreamEvent::ToolCallStart {
                    id: id.clone(),
                    name: tc.function.name.clone(),
                });
                let args_str = serde_json::to_string(&tc.function.arguments).unwrap_or_default();
                events.push(StreamEvent::ToolCallDelta {
                    id: id.clone(),
                    arguments_delta: args_str,
                });
                events.push(StreamEvent::ToolCallEnd { id });
            }
        }

        if let Some(ref content) = msg.content
            && !content.is_empty()
        {
            events.push(StreamEvent::TextDelta(content.clone()));
        }
    }

    if parsed.done {
        events.push(StreamEvent::Done);
    }

    Ok(events)
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

    // Newer Ollama versions use `tool_name` to attribute a tool result to the
    // function that produced it. The name is recoverable from this provider's
    // synthetic id format; for foreign ids the field is omitted gracefully.
    if msg.role == Role::Tool
        && let Some(name) = msg.tool_call_id.as_deref().and_then(tool_name_from_call_id)
    {
        obj["tool_name"] = serde_json::Value::String(name.to_string());
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

fn parse_response(resp: OllamaResponse, tool_call_seq: &AtomicU64) -> Result<ChatResponse> {
    let msg = resp
        .message
        .ok_or_else(|| DaimonError::Model("missing message in Ollama response".into()))?;

    let has_tool_calls = !msg.tool_calls.is_empty();

    // Ids draw from the client-wide counter shared with the streaming path,
    // so parallel calls and repeated turns never collide (a per-response
    // index restarted at 0 on every reply).
    let tool_calls: Vec<ToolCall> = msg
        .tool_calls
        .into_iter()
        .map(|tc| {
            let seq = tool_call_seq.fetch_add(1, Ordering::Relaxed);
            ToolCall {
                id: make_tool_call_id(seq, &tc.function.name),
                name: tc.function.name,
                arguments: tc.function.arguments,
            }
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
        assert_eq!(model.max_retries, DEFAULT_MAX_RETRIES);
    }

    #[test]
    fn test_with_base_url() {
        let model = Ollama::new("llama3.1").with_base_url("http://remote:11434/");
        assert_eq!(model.base_url, "http://remote:11434");
    }

    #[test]
    fn test_with_max_retries() {
        let model = Ollama::new("llama3.1").with_max_retries(5);
        assert_eq!(model.max_retries, 5);
    }

    #[test]
    fn test_ndjson_line_events_text_delta() {
        let seq = AtomicU64::new(0);
        let events = ndjson_line_events(
            r#"{"message":{"role":"assistant","content":"Hel"},"done":false}"#,
            &seq,
        )
        .unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], StreamEvent::TextDelta(t) if t == "Hel"));
    }

    #[test]
    fn test_ndjson_line_events_tool_call() {
        let seq = AtomicU64::new(0);
        let events = ndjson_line_events(
            r#"{"message":{"role":"assistant","tool_calls":[{"function":{"name":"calc","arguments":{"x":1}}}]},"done":false}"#,
            &seq,
        )
        .unwrap();
        assert_eq!(events.len(), 3);
        assert!(matches!(
            &events[0],
            StreamEvent::ToolCallStart { id, name } if id == "ollama_tc_0_calc" && name == "calc"
        ));
        assert!(matches!(&events[1], StreamEvent::ToolCallDelta { .. }));
        assert!(matches!(
            &events[2],
            StreamEvent::ToolCallEnd { id } if id == "ollama_tc_0_calc"
        ));
        assert_eq!(seq.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_ndjson_line_events_done() {
        let seq = AtomicU64::new(0);
        let events = ndjson_line_events(r#"{"done":true}"#, &seq).unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], StreamEvent::Done));
    }

    #[test]
    fn test_ndjson_line_events_empty_line_ignored() {
        let seq = AtomicU64::new(0);
        assert!(ndjson_line_events("", &seq).unwrap().is_empty());
        assert!(ndjson_line_events("   ", &seq).unwrap().is_empty());
    }

    #[test]
    fn test_ndjson_line_events_invalid_json_errors() {
        let seq = AtomicU64::new(0);
        assert!(ndjson_line_events("not json", &seq).is_err());
    }

    #[test]
    fn test_ndjson_line_events_ids_do_not_collide_across_calls() {
        // The counter is shared across the whole stream (and with the
        // non-streaming path): two sequential lines through the same
        // counter must not reuse tool-call ids.
        let seq = AtomicU64::new(0);
        let line = r#"{"message":{"role":"assistant","tool_calls":[{"function":{"name":"calc","arguments":{}}}]},"done":false}"#;
        let first = ndjson_line_events(line, &seq).unwrap();
        let second = ndjson_line_events(line, &seq).unwrap();
        let StreamEvent::ToolCallStart { id: first_id, .. } = &first[0] else {
            panic!("expected ToolCallStart");
        };
        let StreamEvent::ToolCallStart { id: second_id, .. } = &second[0] else {
            panic!("expected ToolCallStart");
        };
        assert_ne!(
            first_id, second_id,
            "tool-call ids must be unique across calls"
        );
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
        let result = parse_response(resp, &AtomicU64::new(0)).unwrap();
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
        let result = parse_response(resp, &AtomicU64::new(0)).unwrap();
        assert_eq!(result.stop_reason, StopReason::ToolUse);
        assert_eq!(result.message.tool_calls.len(), 1);
        assert_eq!(result.message.tool_calls[0].name, "calc");
        assert_eq!(result.message.tool_calls[0].id, "ollama_tc_0_calc");
    }

    #[test]
    fn test_parse_response_ids_do_not_collide_across_turns() {
        // The counter is client-wide: two responses parsed through the same
        // model must not reuse ids (a per-response index restarted at 0).
        let make_resp = || OllamaResponse {
            message: Some(OllamaMessage {
                content: None,
                tool_calls: vec![OllamaToolCall {
                    function: OllamaFunction {
                        name: "calc".into(),
                        arguments: serde_json::json!({}),
                    },
                }],
            }),
            done: true,
            prompt_eval_count: None,
            eval_count: None,
        };
        let seq = AtomicU64::new(0);
        let first = parse_response(make_resp(), &seq).unwrap();
        let second = parse_response(make_resp(), &seq).unwrap();
        assert_ne!(
            first.message.tool_calls[0].id, second.message.tool_calls[0].id,
            "tool-call ids must be unique across turns"
        );
    }

    #[test]
    fn test_tool_name_from_call_id() {
        assert_eq!(tool_name_from_call_id("ollama_tc_0_calc"), Some("calc"));
        // Function names may contain underscores and digits.
        assert_eq!(
            tool_name_from_call_id("ollama_tc_12_web_search_v2"),
            Some("web_search_v2")
        );
        assert_eq!(tool_name_from_call_id("ollama_tc_3"), None);
        assert_eq!(tool_name_from_call_id("ollama_tc_x_calc"), None);
        assert_eq!(tool_name_from_call_id("foreign-id"), None);
    }

    #[test]
    fn test_convert_message_tool_result_includes_tool_name() {
        // Newer Ollama uses `tool_name` for attribution; it is derived from
        // the synthetic tool_call_id format.
        let msg = Message::tool_result("ollama_tc_4_calc", "42");
        let json = convert_message(&msg);
        assert_eq!(json["role"], "tool");
        assert_eq!(json["content"], "42");
        assert_eq!(json["tool_name"], "calc");
    }

    #[test]
    fn test_convert_message_tool_result_omits_tool_name_for_foreign_id() {
        let msg = Message::tool_result("some-other-id", "42");
        let json = convert_message(&msg);
        assert_eq!(json["role"], "tool");
        assert!(
            json.get("tool_name").is_none(),
            "tool_name must be omitted when the name cannot be derived: {json}"
        );
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
