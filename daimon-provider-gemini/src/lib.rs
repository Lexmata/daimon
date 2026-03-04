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

fn build_client(timeout: Option<Duration>) -> Client {
    let mut builder = Client::builder();
    if let Some(t) = timeout {
        builder = builder.timeout(t);
    }
    builder.build().expect("failed to build HTTP client")
}

/// Google Gemini model provider.
///
/// Connects to the Gemini REST API. Supports both the public Generative AI
/// endpoint (default) and Vertex AI via `with_base_url()`. Authentication is
/// via API key (passed as `?key=` query parameter) or bearer token for Vertex AI.
#[derive(Debug)]
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

impl Gemini {
    /// Create a new Gemini client, reading `GOOGLE_API_KEY` from the environment.
    pub fn new(model_id: impl Into<String>) -> Self {
        let api_key = std::env::var("GOOGLE_API_KEY").unwrap_or_default();
        Self::with_api_key(model_id, api_key)
    }

    /// Create a new Gemini client with an explicit API key.
    pub fn with_api_key(model_id: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            client: build_client(None),
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

    /// Use `Authorization: Bearer <key>` instead of `?key=` query parameter.
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
        format!(
            "{}/models/{}:{}",
            self.base_url, self.model_id, method
        )
    }

    fn apply_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if self.use_bearer_token {
            req.bearer_auth(&self.api_key)
        } else {
            req.query(&[("key", &self.api_key)])
        }
    }

    fn build_request_body(&self, request: &ChatRequest) -> GeminiRequest {
        let mut system_instruction = None;
        let mut contents = Vec::new();

        for msg in &request.messages {
            match msg.role {
                Role::System => {
                    if let Some(text) = &msg.content {
                        system_instruction = Some(GeminiContent {
                            role: "user".to_string(),
                            parts: vec![GeminiPart::Text {
                                text: text.clone(),
                            }],
                        });
                    }
                }
                Role::User => {
                    if let Some(text) = &msg.content {
                        contents.push(GeminiContent {
                            role: "user".to_string(),
                            parts: vec![GeminiPart::Text {
                                text: text.clone(),
                            }],
                        });
                    }
                }
                Role::Assistant => {
                    if !msg.tool_calls.is_empty() {
                        let parts = msg
                            .tool_calls
                            .iter()
                            .map(|tc| GeminiPart::FunctionCall {
                                function_call: GeminiFunctionCall {
                                    name: tc.name.clone(),
                                    args: tc.arguments.clone(),
                                },
                            })
                            .collect();
                        contents.push(GeminiContent {
                            role: "model".to_string(),
                            parts,
                        });
                    } else if let Some(text) = &msg.content {
                        contents.push(GeminiContent {
                            role: "model".to_string(),
                            parts: vec![GeminiPart::Text {
                                text: text.clone(),
                            }],
                        });
                    }
                }
                Role::Tool => {
                    let name = msg.tool_call_id.clone().unwrap_or_default();
                    let content = msg.content.clone().unwrap_or_default();
                    let response_value: serde_json::Value =
                        serde_json::from_str(&content).unwrap_or_else(|_| {
                            serde_json::json!({ "result": content })
                        });
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
    #[tracing::instrument(skip_all, fields(model = %self.model_id))]
    async fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let body = self.build_request_body(request);
        let url = self.endpoint_url("generateContent");

        for attempt in 0..=self.max_retries {
            let req = self.client.post(&url).json(&body);
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

            let text = response.text().await.unwrap_or_default();
            let is_retryable = status.as_u16() == 429 || status.is_server_error();

            if is_retryable && attempt < self.max_retries {
                let delay_ms = 100 * 2u64.pow(attempt);
                tracing::debug!(status = %status, attempt, delay_ms, "retryable error, backing off");
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
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

        let req = self
            .client
            .post(&url)
            .query(&[("alt", "sse")])
            .json(&body);
        let req = self.apply_auth(req);

        tracing::debug!("sending Gemini streaming request");
        let response = req
            .send()
            .await
            .map_err(|e| DaimonError::Model(format!("Gemini HTTP error: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(DaimonError::Model(format!(
                "Gemini API error ({status}): {text}"
            )));
        }

        tracing::debug!("Gemini stream established");
        let byte_stream = response.bytes_stream();

        let stream = async_stream::try_stream! {
            use futures::StreamExt;

            let mut buffer = String::new();
            let mut stream = Box::pin(byte_stream);

            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| DaimonError::Model(format!("Gemini stream error: {e}")))?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(line_end) = buffer.find('\n') {
                    let line = buffer[..line_end].trim().to_string();
                    buffer = buffer[line_end + 1..].to_string();

                    if line.is_empty() {
                        continue;
                    }

                    if let Some(data) = line.strip_prefix("data: ") {
                        if let Ok(chunk_resp) = serde_json::from_str::<GeminiResponse>(data) {
                            for candidate in &chunk_resp.candidates {
                                for part in &candidate.content.parts {
                                    match part {
                                        GeminiResponsePart::Text { text } => {
                                            if !text.is_empty() {
                                                yield StreamEvent::TextDelta(text.clone());
                                            }
                                        }
                                        GeminiResponsePart::FunctionCall { function_call } => {
                                            let id = format!("gemini_{}", function_call.name);
                                            yield StreamEvent::ToolCallStart {
                                                id: id.clone(),
                                                name: function_call.name.clone(),
                                            };
                                            let args = serde_json::to_string(&function_call.args)
                                                .unwrap_or_default();
                                            yield StreamEvent::ToolCallDelta {
                                                id: id.clone(),
                                                arguments_delta: args,
                                            };
                                            yield StreamEvent::ToolCallEnd { id };
                                        }
                                    }
                                }
                            }

                            let is_done = chunk_resp.candidates.iter().any(|c| {
                                c.finish_reason.as_deref() == Some("STOP")
                                    || c.finish_reason.as_deref() == Some("MAX_TOKENS")
                            });
                            if is_done {
                                yield StreamEvent::Done;
                            }
                        }
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

fn parse_response(response: GeminiResponse) -> Result<ChatResponse> {
    let candidate = response
        .candidates
        .into_iter()
        .next()
        .ok_or_else(|| DaimonError::Model("no candidates in Gemini response".into()))?;

    let mut text_content = String::new();
    let mut tool_calls = Vec::new();

    for part in candidate.content.parts {
        match part {
            GeminiResponsePart::Text { text } => {
                text_content.push_str(&text);
            }
            GeminiResponsePart::FunctionCall { function_call } => {
                tool_calls.push(ToolCall {
                    id: format!("gemini_{}", function_call.name),
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
        usage: response.usage_metadata.map(|u| Usage {
            input_tokens: u.prompt_token_count,
            output_tokens: u.candidates_token_count,
            cached_tokens: u.cached_content_token_count,
        }),
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
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiCandidate {
    content: GeminiResponseContent,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
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
        };
        let resp = parse_response(raw).unwrap();
        assert!(resp.has_tool_calls());
        assert_eq!(resp.tool_calls()[0].name, "calculator");
        assert_eq!(resp.stop_reason, StopReason::ToolUse);
    }

    #[test]
    fn test_parse_response_no_candidates() {
        let raw = GeminiResponse {
            candidates: vec![],
            usage_metadata: None,
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
                    id: "gemini_calc".into(),
                    name: "calc".into(),
                    arguments: serde_json::json!({"expr": "2+2"}),
                }]),
                Message::tool_result("calc", "4"),
            ],
            tools: vec![],
            temperature: None,
            max_tokens: None,
        };
        let body = model.build_request_body(&request);
        assert_eq!(body.contents.len(), 3);
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
}
