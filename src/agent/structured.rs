//! Structured output: extract typed data from LLM responses.
//!
//! [`Agent::prompt_structured`] instructs the model to return JSON matching
//! the schema of a `serde::Deserialize + JsonSchema` type, then parses the
//! response into that type.
//!
//! ```ignore
//! use daimon::agent::structured::StructuredOutput;
//! use serde::Deserialize;
//!
//! #[derive(Deserialize)]
//! struct Sentiment {
//!     label: String,
//!     confidence: f64,
//! }
//!
//! let result: StructuredOutput<Sentiment> = agent
//!     .prompt_structured::<Sentiment>(
//!         "Analyze sentiment: 'Rust is amazing!'",
//!         "Sentiment",
//!     )
//!     .await?;
//!
//! assert_eq!(result.data.label, "positive");
//! ```

use serde::de::DeserializeOwned;

use crate::agent::Agent;
use crate::error::{DaimonError, Result};
use crate::model::types::{ChatRequest, Message, Usage};

/// The result of a structured extraction, containing the parsed data
/// and the raw text from the model.
#[derive(Debug, Clone)]
pub struct StructuredOutput<T> {
    /// The deserialized data.
    pub data: T,
    /// The raw text response from the model.
    pub raw_text: String,
    /// Token usage for this extraction.
    pub usage: Usage,
}

impl Agent {
    /// Prompts the model and parses the response as the given type `T`.
    ///
    /// Adds an instruction to return JSON matching the target schema. The
    /// `type_name` is used in the instruction to help the model understand
    /// what it should produce.
    ///
    /// If the model's response is not valid JSON or cannot be deserialized
    /// into `T`, a retry is attempted with the error message to let the
    /// model correct itself (up to 3 total attempts).
    #[tracing::instrument(skip_all, fields(type_name = %type_name))]
    pub async fn prompt_structured<T: DeserializeOwned>(
        &self,
        input: &str,
        type_name: &str,
    ) -> Result<StructuredOutput<T>> {
        let mut messages = Vec::new();

        if let Some(system) = &self.system_prompt {
            messages.push(Message::system(system));
        }

        let extraction_instruction = format!(
            "You MUST respond with ONLY valid JSON (no markdown, no code fences, no explanation). \
             The JSON must be a single {type_name} object. Do not include any text before or after the JSON."
        );
        messages.push(Message::system(extraction_instruction));
        messages.push(Message::user(input));

        let mut total_usage = Usage::default();
        let max_attempts = 3;

        for attempt in 0..max_attempts {
            let request = ChatRequest {
                messages: messages.clone(),
                tools: Vec::new(),
                temperature: self.temperature,
                max_tokens: self.max_tokens,
            };

            let response = self.model.generate_erased(&request).await?;

            if let Some(ref usage) = response.usage {
                total_usage.accumulate(usage);
            }

            let raw_text = response.text().to_string();
            let cleaned = extract_json(&raw_text);

            match serde_json::from_str::<T>(cleaned) {
                Ok(data) => {
                    return Ok(StructuredOutput {
                        data,
                        raw_text,
                        usage: total_usage,
                    });
                }
                Err(e) if attempt < max_attempts - 1 => {
                    tracing::debug!(
                        attempt,
                        error = %e,
                        "structured output parse failed, retrying"
                    );
                    messages.push(Message::assistant(&raw_text));
                    messages.push(Message::user(format!(
                        "Your response was not valid JSON for {type_name}. Error: {e}. \
                         Please respond with ONLY the corrected JSON."
                    )));
                }
                Err(e) => {
                    return Err(DaimonError::Other(format!(
                        "failed to parse {type_name} after {max_attempts} attempts: {e}\nRaw: {raw_text}"
                    )));
                }
            }
        }

        unreachable!()
    }
}

/// Extracts JSON from a model response that may include markdown code fences.
fn extract_json(text: &str) -> &str {
    let trimmed = text.trim();

    if let Some(start) = trimmed.find("```json") {
        let after = &trimmed[start + 7..];
        if let Some(end) = after.find("```") {
            return after[..end].trim();
        }
    }

    if let Some(start) = trimmed.find("```") {
        let after = &trimmed[start + 3..];
        if let Some(newline) = after.find('\n') {
            let rest = &after[newline + 1..];
            if let Some(end) = rest.find("```") {
                return rest[..end].trim();
            }
        }
    }

    trimmed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Model;
    use crate::model::types::*;
    use crate::stream::ResponseStream;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, PartialEq)]
    struct Sentiment {
        label: String,
        confidence: f64,
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct Person {
        name: String,
        age: u32,
    }

    struct JsonModel {
        response: String,
    }

    impl JsonModel {
        fn new(response: &str) -> Self {
            Self {
                response: response.to_string(),
            }
        }
    }

    impl Model for JsonModel {
        async fn generate(&self, _request: &ChatRequest) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message::assistant(&self.response),
                stop_reason: StopReason::EndTurn,
                usage: Some(Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    cached_tokens: 0,
                }),
            })
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    #[tokio::test]
    async fn test_structured_output_basic() {
        let agent = Agent::builder()
            .model(JsonModel::new(
                r#"{"label": "positive", "confidence": 0.95}"#,
            ))
            .build()
            .unwrap();

        let result: StructuredOutput<Sentiment> =
            agent.prompt_structured("test", "Sentiment").await.unwrap();

        assert_eq!(result.data.label, "positive");
        assert_eq!(result.data.confidence, 0.95);
    }

    #[tokio::test]
    async fn test_structured_output_with_code_fences() {
        let response = "```json\n{\"name\": \"Alice\", \"age\": 30}\n```";
        let agent = Agent::builder()
            .model(JsonModel::new(response))
            .build()
            .unwrap();

        let result: StructuredOutput<Person> =
            agent.prompt_structured("test", "Person").await.unwrap();

        assert_eq!(result.data.name, "Alice");
        assert_eq!(result.data.age, 30);
    }

    #[tokio::test]
    async fn test_structured_output_invalid_json() {
        let agent = Agent::builder()
            .model(JsonModel::new("this is not json at all"))
            .build()
            .unwrap();

        let result = agent
            .prompt_structured::<Sentiment>("test", "Sentiment")
            .await;

        assert!(result.is_err());
    }

    #[test]
    fn test_extract_json_plain() {
        assert_eq!(extract_json(r#"{"a": 1}"#), r#"{"a": 1}"#);
    }

    #[test]
    fn test_extract_json_code_fence() {
        assert_eq!(extract_json("```json\n{\"a\": 1}\n```"), "{\"a\": 1}");
    }

    #[test]
    fn test_extract_json_generic_fence() {
        assert_eq!(extract_json("```\n{\"a\": 1}\n```"), "{\"a\": 1}");
    }

    #[test]
    fn test_extract_json_whitespace() {
        assert_eq!(extract_json("  \n  {\"a\": 1}  \n  "), "{\"a\": 1}");
    }
}
