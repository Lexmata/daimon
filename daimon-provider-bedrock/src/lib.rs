//! Amazon Bedrock model provider for the [Daimon](https://docs.rs/daimon) agent framework.
//!
//! Supports the Bedrock Converse API for non-streaming and streaming inference,
//! with optional guardrails, prompt caching, and configurable retries.
//!
//! # Example
//!
//! ```ignore
//! use daimon_provider_bedrock::Bedrock;
//! use daimon_core::Model;
//!
//! let model = Bedrock::new("us.anthropic.claude-sonnet-4-20250514")
//!     .with_region("us-east-1")
//!     .with_prompt_caching();
//! ```

use std::time::Duration;

use aws_sdk_bedrockruntime::Client as BedrockClient;
use aws_sdk_bedrockruntime::types::{
    CachePointBlock, CachePointType, ContentBlock, ConversationRole, GuardrailConfiguration,
    GuardrailStreamConfiguration, InferenceConfiguration, Message as BedrockMessage,
    SystemContentBlock, ToolConfiguration, ToolInputSchema, ToolResultBlock,
    ToolResultContentBlock, ToolResultStatus, ToolSpecification, ToolUseBlock,
};

mod embedding;

#[cfg(feature = "sqs")]
pub mod sqs;

pub use embedding::BedrockEmbedding;

#[cfg(feature = "sqs")]
pub use sqs::SqsBroker;

use daimon_core::{
    ChatRequest, ChatResponse, DaimonError, Message, Model, ResponseStream, Result, Role,
    StopReason, StreamEvent, ToolCall, Usage,
};

/// Amazon Bedrock model provider using the Converse API.
///
/// Supports both non-streaming and streaming inference, with optional
/// guardrails for content filtering and configurable retry behavior.
#[derive(Debug)]
pub struct Bedrock {
    model_id: String,
    client: Option<BedrockClient>,
    region: Option<String>,
    max_retries: u32,
    guardrail_id: Option<String>,
    guardrail_version: Option<String>,
    use_prompt_caching: bool,
}

impl Bedrock {
    /// Creates a new Bedrock provider for the given model ID.
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            client: None,
            region: None,
            max_retries: 3,
            guardrail_id: None,
            guardrail_version: None,
            use_prompt_caching: false,
        }
    }

    /// Sets the Bedrock client to use (otherwise created from env config).
    pub fn with_client(mut self, client: BedrockClient) -> Self {
        self.client = Some(client);
        self
    }

    /// Sets the AWS region for the Bedrock client.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Sets the maximum number of retries for throttling/server errors (default: 3).
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Configures a guardrail for content filtering.
    pub fn with_guardrail(mut self, id: impl Into<String>, version: impl Into<String>) -> Self {
        self.guardrail_id = Some(id.into());
        self.guardrail_version = Some(version.into());
        self
    }

    /// Enables prompt caching for system messages and tool definitions.
    ///
    /// When enabled, a `CachePoint` content block is appended after the
    /// system prompt and after the tool configuration in each request.
    pub fn with_prompt_caching(mut self) -> Self {
        self.use_prompt_caching = true;
        self
    }

    async fn get_client(&self) -> Result<BedrockClient> {
        if let Some(ref client) = self.client {
            return Ok(client.clone());
        }

        let mut config_loader = aws_config::from_env();
        if let Some(ref region) = self.region {
            config_loader = config_loader.region(aws_config::Region::new(region.clone()));
        }
        let config = config_loader.load().await;
        Ok(BedrockClient::new(&config))
    }

    fn build_messages(
        request: &ChatRequest,
        use_prompt_caching: bool,
    ) -> (Vec<SystemContentBlock>, Vec<BedrockMessage>) {
        let mut system_blocks = Vec::new();
        let mut messages = Vec::new();

        for msg in &request.messages {
            match msg.role {
                Role::System => {
                    if let Some(ref text) = msg.content {
                        system_blocks.push(SystemContentBlock::Text(text.clone()));
                    }
                }
                Role::User => {
                    if let Some(ref text) = msg.content {
                        messages.push(
                            BedrockMessage::builder()
                                .role(ConversationRole::User)
                                .content(ContentBlock::Text(text.clone()))
                                .build()
                                .expect("valid bedrock message"),
                        );
                    }
                }
                Role::Assistant => {
                    let mut content_blocks = Vec::new();
                    if let Some(ref text) = msg.content {
                        content_blocks.push(ContentBlock::Text(text.clone()));
                    }
                    for tc in &msg.tool_calls {
                        let input_doc = json_to_document(&tc.arguments);
                        content_blocks.push(ContentBlock::ToolUse(
                            ToolUseBlock::builder()
                                .tool_use_id(&tc.id)
                                .name(&tc.name)
                                .input(input_doc)
                                .build()
                                .expect("valid tool use block"),
                        ));
                    }
                    if !content_blocks.is_empty() {
                        let mut builder =
                            BedrockMessage::builder().role(ConversationRole::Assistant);
                        for block in content_blocks {
                            builder = builder.content(block);
                        }
                        messages.push(builder.build().expect("valid bedrock message"));
                    }
                }
                Role::Tool => {
                    let tool_call_id = msg.tool_call_id.clone().unwrap_or_default();
                    let content = msg.content.clone().unwrap_or_default();
                    let tool_result = ContentBlock::ToolResult(
                        ToolResultBlock::builder()
                            .tool_use_id(tool_call_id)
                            .status(ToolResultStatus::Success)
                            .content(ToolResultContentBlock::Text(content))
                            .build()
                            .expect("valid tool result block"),
                    );
                    messages.push(
                        BedrockMessage::builder()
                            .role(ConversationRole::User)
                            .content(tool_result)
                            .build()
                            .expect("valid bedrock message"),
                    );
                }
            }
        }

        if use_prompt_caching && !system_blocks.is_empty() {
            system_blocks.push(SystemContentBlock::CachePoint(
                CachePointBlock::builder()
                    .r#type(CachePointType::Default)
                    .build()
                    .expect("valid cache point block"),
            ));
        }

        (system_blocks, messages)
    }

    fn build_tool_config(
        request: &ChatRequest,
        use_prompt_caching: bool,
    ) -> Option<ToolConfiguration> {
        if request.tools.is_empty() {
            return None;
        }

        let tools: Vec<aws_sdk_bedrockruntime::types::Tool> = request
            .tools
            .iter()
            .map(|spec| {
                let schema_doc = json_to_document(&spec.parameters);
                aws_sdk_bedrockruntime::types::Tool::ToolSpec(
                    ToolSpecification::builder()
                        .name(&spec.name)
                        .description(&spec.description)
                        .input_schema(ToolInputSchema::Json(schema_doc))
                        .build()
                        .expect("valid tool spec"),
                )
            })
            .collect();

        let mut builder = ToolConfiguration::builder();
        for tool in tools {
            builder = builder.tools(tool);
        }
        if use_prompt_caching {
            builder = builder.tools(aws_sdk_bedrockruntime::types::Tool::CachePoint(
                CachePointBlock::builder()
                    .r#type(CachePointType::Default)
                    .build()
                    .expect("valid cache point block"),
            ));
        }
        Some(builder.build().expect("valid tool config"))
    }

    fn parse_converse_output(
        &self,
        output: aws_sdk_bedrockruntime::operation::converse::ConverseOutput,
    ) -> Result<ChatResponse> {
        let stop_reason = match *output.stop_reason() {
            aws_sdk_bedrockruntime::types::StopReason::ToolUse => StopReason::ToolUse,
            aws_sdk_bedrockruntime::types::StopReason::MaxTokens => StopReason::MaxTokens,
            _ => StopReason::EndTurn,
        };

        let mut text_content = String::new();
        let mut tool_calls = Vec::new();

        if let Some(aws_sdk_bedrockruntime::types::ConverseOutput::Message(msg)) = output.output() {
            for block in msg.content() {
                match block {
                    ContentBlock::Text(t) => text_content.push_str(t),
                    ContentBlock::ToolUse(tu) => {
                        let args = document_to_json(tu.input());
                        tool_calls.push(ToolCall {
                            id: tu.tool_use_id().to_string(),
                            name: tu.name().to_string(),
                            arguments: args,
                        });
                    }
                    _ => {}
                }
            }
        }

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

        let usage = output.usage().map(|u| Usage {
            input_tokens: u.input_tokens() as u32,
            output_tokens: u.output_tokens() as u32,
            cached_tokens: u.cache_read_input_tokens().unwrap_or(0) as u32,
        });

        Ok(ChatResponse {
            message,
            stop_reason,
            usage,
        })
    }
}

fn is_retryable_error(err: impl std::fmt::Display) -> bool {
    let s = err.to_string();
    let s_lower = s.to_lowercase();
    s_lower.contains("throttl")
        || s_lower.contains("service unavailable")
        || s_lower.contains("internal server")
        || s.contains("503")
        || s.contains("429")
}

impl Model for Bedrock {
    #[tracing::instrument(skip_all, fields(model = %self.model_id))]
    async fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let client = self.get_client().await?;
        tracing::debug!("obtained Bedrock client");

        let (system_blocks, messages) = Self::build_messages(request, self.use_prompt_caching);
        let tool_config = Self::build_tool_config(request, self.use_prompt_caching);
        tracing::debug!(
            system_blocks = system_blocks.len(),
            message_count = messages.len(),
            has_tools = tool_config.is_some(),
            prompt_caching = self.use_prompt_caching,
            "built request messages"
        );

        let mut last_error = None;
        for attempt in 0..=self.max_retries {
            let mut req_builder = client.converse().model_id(&self.model_id);

            for block in system_blocks.clone() {
                req_builder = req_builder.system(block);
            }
            for msg in messages.clone() {
                req_builder = req_builder.messages(msg);
            }
            if let Some(ref tc) = tool_config {
                req_builder = req_builder.tool_config(tc.clone());
            }

            let mut inference_config = InferenceConfiguration::builder();
            if let Some(temp) = request.temperature {
                inference_config = inference_config.temperature(temp);
            }
            if let Some(max_tok) = request.max_tokens {
                inference_config = inference_config.max_tokens(max_tok as i32);
            }
            req_builder = req_builder.inference_config(inference_config.build());

            if let (Some(id), Some(version)) = (&self.guardrail_id, &self.guardrail_version) {
                let guardrail_config = GuardrailConfiguration::builder()
                    .guardrail_identifier(id)
                    .guardrail_version(version)
                    .build()
                    .expect("valid guardrail config");
                req_builder = req_builder.guardrail_config(guardrail_config);
                tracing::debug!(guardrail_id = %id, "applied guardrail config");
            }

            match req_builder.send().await {
                Ok(output) => {
                    tracing::debug!("received successful Converse response");
                    return self.parse_converse_output(output);
                }
                Err(e) => {
                    last_error = Some(e.to_string());
                    if is_retryable_error(e.to_string()) && attempt < self.max_retries {
                        let delay_ms = 100u64
                            .saturating_mul(2u64.saturating_pow(attempt))
                            .min(30_000);
                        tracing::debug!(
                            attempt = attempt + 1,
                            max_retries = self.max_retries,
                            delay_ms,
                            "retryable error, backing off"
                        );
                        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    } else {
                        return Err(DaimonError::Model(format!(
                            "Bedrock Converse error: {}",
                            last_error.unwrap_or_default()
                        )));
                    }
                }
            }
        }

        Err(DaimonError::Model(format!(
            "Bedrock Converse error: {}",
            last_error.unwrap_or_else(|| "unknown".into())
        )))
    }

    #[tracing::instrument(skip_all, fields(model = %self.model_id))]
    async fn generate_stream(&self, request: &ChatRequest) -> Result<ResponseStream> {
        let client = self.get_client().await?;
        tracing::debug!("obtained Bedrock client for streaming");

        let (system_blocks, messages) = Self::build_messages(request, self.use_prompt_caching);
        let tool_config = Self::build_tool_config(request, self.use_prompt_caching);
        tracing::debug!(
            system_blocks = system_blocks.len(),
            message_count = messages.len(),
            has_tools = tool_config.is_some(),
            prompt_caching = self.use_prompt_caching,
            "built request messages for stream"
        );

        let mut req_builder = client.converse_stream().model_id(&self.model_id);

        for block in system_blocks {
            req_builder = req_builder.system(block);
        }
        for msg in messages {
            req_builder = req_builder.messages(msg);
        }
        if let Some(tc) = tool_config {
            req_builder = req_builder.tool_config(tc);
        }

        let mut inference_config = InferenceConfiguration::builder();
        if let Some(temp) = request.temperature {
            inference_config = inference_config.temperature(temp);
        }
        if let Some(max_tok) = request.max_tokens {
            inference_config = inference_config.max_tokens(max_tok as i32);
        }
        req_builder = req_builder.inference_config(inference_config.build());

        if let (Some(id), Some(version)) = (&self.guardrail_id, &self.guardrail_version) {
            let guardrail_config = GuardrailStreamConfiguration::builder()
                .guardrail_identifier(id)
                .guardrail_version(version)
                .build()
                .expect("valid guardrail stream config");
            req_builder = req_builder.guardrail_config(guardrail_config);
            tracing::debug!(guardrail_id = %id, "applied guardrail config for stream");
        }

        let mut event_stream = req_builder
            .send()
            .await
            .map_err(|e| DaimonError::Model(format!("Bedrock ConverseStream error: {e}")))?;

        tracing::debug!("stream established, processing events");

        let stream = async_stream::try_stream! {
            use std::collections::HashMap;

            let stream_output = &mut event_stream.stream;
            // Maps a Converse `contentBlockIndex` to the tool_use id opened at
            // that block. The Converse stream only carries the tool_use id once,
            // on `ContentBlockStart`; subsequent `ContentBlockDelta` (argument
            // fragments) and `ContentBlockStop` events correlate to their block
            // solely via the index. Emitting an empty id here dropped every
            // argument fragment, so tools ran with `{}`.
            let mut index_to_id: HashMap<i32, String> = HashMap::new();

            while let Some(event) = stream_output.recv().await.map_err(|e| {
                DaimonError::Model(format!("Bedrock stream error: {e}"))
            })? {
                use aws_sdk_bedrockruntime::types::ConverseStreamOutput;
                match event {
                    ConverseStreamOutput::ContentBlockStart(start) => {
                        if let Some(s) = start.start() {
                            use aws_sdk_bedrockruntime::types::ContentBlockStart as CBS;
                            if let CBS::ToolUse(tu) = s {
                                let id = tu.tool_use_id().to_string();
                                index_to_id.insert(start.content_block_index(), id.clone());
                                yield StreamEvent::ToolCallStart {
                                    id,
                                    name: tu.name().to_string(),
                                };
                            }
                        }
                    }
                    ConverseStreamOutput::ContentBlockDelta(delta) => {
                        let index = delta.content_block_index();
                        if let Some(d) = delta.delta() {
                            use aws_sdk_bedrockruntime::types::ContentBlockDelta as CBD;
                            match d {
                                CBD::Text(t) => {
                                    yield StreamEvent::TextDelta(t.to_string());
                                }
                                CBD::ToolUse(tu) => {
                                    let id = index_to_id
                                        .get(&index)
                                        .cloned()
                                        .unwrap_or_default();
                                    yield StreamEvent::ToolCallDelta {
                                        id,
                                        arguments_delta: tu.input().to_string(),
                                    };
                                }
                                _ => {}
                            }
                        }
                    }
                    ConverseStreamOutput::ContentBlockStop(stop) => {
                        if let Some(id) = index_to_id.remove(&stop.content_block_index()) {
                            yield StreamEvent::ToolCallEnd { id };
                        }
                    }
                    ConverseStreamOutput::MessageStop(_) => {
                        yield StreamEvent::Done;
                    }
                    _ => {}
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

fn json_to_document(value: &serde_json::Value) -> aws_smithy_types::Document {
    match value {
        serde_json::Value::Null => aws_smithy_types::Document::Null,
        serde_json::Value::Bool(b) => aws_smithy_types::Document::Bool(*b),
        serde_json::Value::Number(n) => {
            // Order matters: try unsigned first (covers all non-negative
            // values, including those above i64::MAX), then signed for
            // negatives, then float. The previous code funneled every integer
            // through `PosInt(i as u64)`, which reinterprets a negative i64 as a
            // huge unsigned value — e.g. -1 became 18446744073709551615.
            if let Some(u) = n.as_u64() {
                aws_smithy_types::Document::Number(aws_smithy_types::Number::PosInt(u))
            } else if let Some(i) = n.as_i64() {
                aws_smithy_types::Document::Number(aws_smithy_types::Number::NegInt(i))
            } else if let Some(f) = n.as_f64() {
                aws_smithy_types::Document::Number(aws_smithy_types::Number::Float(f))
            } else {
                aws_smithy_types::Document::Null
            }
        }
        serde_json::Value::String(s) => aws_smithy_types::Document::String(s.clone()),
        serde_json::Value::Array(arr) => {
            aws_smithy_types::Document::Array(arr.iter().map(json_to_document).collect())
        }
        serde_json::Value::Object(obj) => {
            let map: std::collections::HashMap<String, aws_smithy_types::Document> = obj
                .iter()
                .map(|(k, v)| (k.clone(), json_to_document(v)))
                .collect();
            aws_smithy_types::Document::Object(map)
        }
    }
}

fn document_to_json(doc: &aws_smithy_types::Document) -> serde_json::Value {
    match doc {
        aws_smithy_types::Document::Object(map) => {
            let obj: serde_json::Map<String, serde_json::Value> = map
                .iter()
                .map(|(k, v)| (k.clone(), document_to_json(v)))
                .collect();
            serde_json::Value::Object(obj)
        }
        aws_smithy_types::Document::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(document_to_json).collect())
        }
        aws_smithy_types::Document::Number(n) => match n {
            aws_smithy_types::Number::PosInt(i) => serde_json::Value::Number((*i).into()),
            aws_smithy_types::Number::NegInt(i) => serde_json::Value::Number((*i).into()),
            aws_smithy_types::Number::Float(f) => serde_json::Value::Number(
                serde_json::Number::from_f64(*f).unwrap_or(serde_json::Number::from(0)),
            ),
        },
        aws_smithy_types::Document::String(s) => serde_json::Value::String(s.clone()),
        aws_smithy_types::Document::Bool(b) => serde_json::Value::Bool(*b),
        aws_smithy_types::Document::Null => serde_json::Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daimon_core::ToolSpec;

    #[test]
    fn test_bedrock_new() {
        let model = Bedrock::new("us.anthropic.claude-sonnet-4-20250514");
        assert_eq!(model.model_id, "us.anthropic.claude-sonnet-4-20250514");
        assert!(model.client.is_none());
    }

    #[test]
    fn test_bedrock_with_region() {
        let model = Bedrock::new("test").with_region("us-east-1");
        assert_eq!(model.region.as_deref(), Some("us-east-1"));
    }

    #[test]
    fn test_bedrock_with_max_retries() {
        let model = Bedrock::new("test").with_max_retries(5);
        assert_eq!(model.max_retries, 5);
    }

    #[test]
    fn test_bedrock_with_max_retries_default() {
        let model = Bedrock::new("test");
        assert_eq!(model.max_retries, 3);
    }

    #[test]
    fn test_bedrock_with_guardrail() {
        let model = Bedrock::new("test").with_guardrail("guardrail-123", "DRAFT");
        assert_eq!(model.guardrail_id.as_deref(), Some("guardrail-123"));
        assert_eq!(model.guardrail_version.as_deref(), Some("DRAFT"));
    }

    #[test]
    fn test_bedrock_with_guardrail_default_none() {
        let model = Bedrock::new("test");
        assert!(model.guardrail_id.is_none());
        assert!(model.guardrail_version.is_none());
    }

    #[test]
    fn test_build_messages_basic() {
        let request = ChatRequest {
            messages: vec![Message::system("Be helpful"), Message::user("hello")],
            tools: vec![],
            temperature: None,
            max_tokens: None,
        };
        let (system, messages) = Bedrock::build_messages(&request, false);
        assert_eq!(system.len(), 1);
        assert_eq!(messages.len(), 1);
    }

    #[test]
    fn test_build_messages_with_tool_results() {
        let request = ChatRequest {
            messages: vec![
                Message::user("calc"),
                Message::assistant_with_tool_calls(vec![ToolCall {
                    id: "tc_1".into(),
                    name: "calc".into(),
                    arguments: serde_json::json!({}),
                }]),
                Message::tool_result("tc_1", "42"),
            ],
            tools: vec![],
            temperature: None,
            max_tokens: None,
        };
        let (_, messages) = Bedrock::build_messages(&request, false);
        assert_eq!(messages.len(), 3);
    }

    #[test]
    fn test_build_messages_with_caching() {
        let request = ChatRequest {
            messages: vec![Message::system("Be helpful"), Message::user("hello")],
            tools: vec![],
            temperature: None,
            max_tokens: None,
        };
        let (system, _) = Bedrock::build_messages(&request, true);
        assert_eq!(system.len(), 2, "should have text + cache point");
    }

    #[test]
    fn test_build_messages_caching_no_system() {
        let request = ChatRequest {
            messages: vec![Message::user("hello")],
            tools: vec![],
            temperature: None,
            max_tokens: None,
        };
        let (system, _) = Bedrock::build_messages(&request, true);
        assert!(system.is_empty(), "no cache point when no system prompt");
    }

    #[test]
    fn test_json_to_document_string() {
        let json = serde_json::json!("hello");
        let doc = json_to_document(&json);
        assert!(matches!(doc, aws_smithy_types::Document::String(s) if s == "hello"));
    }

    #[test]
    fn test_json_to_document_object() {
        let json = serde_json::json!({"key": "value"});
        let doc = json_to_document(&json);
        if let aws_smithy_types::Document::Object(map) = doc {
            assert!(map.contains_key("key"));
        } else {
            panic!("expected Document::Object");
        }
    }

    #[test]
    fn test_json_to_document_null() {
        let json = serde_json::Value::Null;
        let doc = json_to_document(&json);
        assert!(matches!(doc, aws_smithy_types::Document::Null));
    }

    #[test]
    fn test_document_to_json_object() {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "key".to_string(),
            aws_smithy_types::Document::String("value".into()),
        );
        let doc = aws_smithy_types::Document::Object(map);
        let json = document_to_json(&doc);
        assert_eq!(json["key"], "value");
    }

    #[test]
    fn test_document_to_json_null() {
        let json = document_to_json(&aws_smithy_types::Document::Null);
        assert!(json.is_null());
    }

    #[test]
    fn test_document_to_json_bool() {
        let json = document_to_json(&aws_smithy_types::Document::Bool(true));
        assert_eq!(json, serde_json::Value::Bool(true));
    }

    #[test]
    fn test_document_to_json_array() {
        let doc = aws_smithy_types::Document::Array(vec![
            aws_smithy_types::Document::String("a".into()),
            aws_smithy_types::Document::String("b".into()),
        ]);
        let json = document_to_json(&doc);
        assert!(json.is_array());
        assert_eq!(json.as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_roundtrip_json_document() {
        let original = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": 42,
                "active": true
            }
        });
        let doc = json_to_document(&original);
        let back = document_to_json(&doc);
        assert_eq!(original, back);
    }

    #[test]
    fn test_roundtrip_negative_and_zero_integers() {
        // Regression: negative integers were corrupted by `PosInt(i as u64)`.
        let original = serde_json::json!({
            "neg": -1,
            "zero": 0,
            "big_neg": -9007199254740991i64,
            "pos": 7,
            "temp": -0.5,
            "nested": [-3, 0, 5]
        });
        let doc = json_to_document(&original);
        let back = document_to_json(&doc);
        assert_eq!(original, back);
    }

    #[test]
    fn test_negative_integer_maps_to_negint() {
        let doc = json_to_document(&serde_json::json!(-42));
        match doc {
            aws_smithy_types::Document::Number(aws_smithy_types::Number::NegInt(i)) => {
                assert_eq!(i, -42);
            }
            other => panic!("expected NegInt(-42), got {other:?}"),
        }
    }

    #[test]
    fn test_zero_maps_to_posint() {
        let doc = json_to_document(&serde_json::json!(0));
        match doc {
            aws_smithy_types::Document::Number(aws_smithy_types::Number::PosInt(u)) => {
                assert_eq!(u, 0);
            }
            other => panic!("expected PosInt(0), got {other:?}"),
        }
    }

    #[test]
    fn test_build_tool_config_empty() {
        let request = ChatRequest {
            messages: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
        };
        assert!(Bedrock::build_tool_config(&request, false).is_none());
    }

    #[test]
    fn test_build_tool_config_with_tools() {
        let request = ChatRequest {
            messages: vec![],
            tools: vec![ToolSpec {
                name: "calc".into(),
                description: "Calculator".into(),
                parameters: serde_json::json!({"type": "object"}),
            }],
            temperature: None,
            max_tokens: None,
        };
        assert!(Bedrock::build_tool_config(&request, false).is_some());
    }

    #[test]
    fn test_with_prompt_caching() {
        let model = Bedrock::new("test").with_prompt_caching();
        assert!(model.use_prompt_caching);
    }
}
