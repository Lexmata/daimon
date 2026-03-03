//! Human-in-the-loop support.
//!
//! Provides [`HumanInputHandler`] trait and [`AskHumanTool`] that allows an agent
//! to pause its execution and request input from a human operator.
//!
//! # Example
//!
//! ```ignore
//! use daimon::prelude::*;
//! use daimon::agent::hitl::{HumanInputHandler, HumanInputRequest};
//!
//! struct ConsoleHandler;
//!
//! impl HumanInputHandler for ConsoleHandler {
//!     async fn request_input(&self, request: &HumanInputRequest) -> daimon::Result<String> {
//!         println!("Agent asks: {}", request.prompt);
//!         // read from stdin...
//!         Ok("user response".into())
//!     }
//! }
//!
//! let agent = Agent::builder()
//!     .model(model)
//!     .human_input(ConsoleHandler)
//!     .build()?;
//! ```

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::error::Result;
use crate::tool::{Tool, ToolOutput};

/// A request for human input, passed to a [`HumanInputHandler`].
#[derive(Debug, Clone)]
pub struct HumanInputRequest {
    /// The question or prompt for the human.
    pub prompt: String,
    /// Optional choices to present. If empty, free-form input is expected.
    pub choices: Vec<String>,
    /// Optional context about what the agent is doing.
    pub context: Option<String>,
}

/// Trait for receiving human input during agent execution.
///
/// Implement this trait to integrate with your application's UI (console,
/// web, Slack, etc.). The agent calls this via the `ask_human` tool.
pub trait HumanInputHandler: Send + Sync {
    /// Called when the agent needs human input. Must return the human's response text.
    fn request_input(
        &self,
        request: &HumanInputRequest,
    ) -> impl Future<Output = Result<String>> + Send;
}

/// Object-safe wrapper for [`HumanInputHandler`].
pub trait ErasedHumanInputHandler: Send + Sync {
    fn request_input_erased<'a>(
        &'a self,
        request: &'a HumanInputRequest,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>>;
}

impl<T: HumanInputHandler> ErasedHumanInputHandler for T {
    fn request_input_erased<'a>(
        &'a self,
        request: &'a HumanInputRequest,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        Box::pin(self.request_input(request))
    }
}

/// A tool that the agent can call to ask the human a question.
///
/// Register this tool via [`AgentBuilder::human_input`](crate::agent::AgentBuilder::human_input)
/// or manually with [`AgentBuilder::tool`](crate::agent::AgentBuilder::tool).
pub struct AskHumanTool {
    handler: Arc<dyn ErasedHumanInputHandler>,
}

impl AskHumanTool {
    /// Creates an `ask_human` tool backed by the given handler.
    pub fn new<H: HumanInputHandler + 'static>(handler: H) -> Self {
        Self {
            handler: Arc::new(handler),
        }
    }
}

impl Tool for AskHumanTool {
    fn name(&self) -> &str {
        "ask_human"
    }

    fn description(&self) -> &str {
        "Ask the human user a question or request their input. Use this when you need \
         clarification, approval, or any information that only the user can provide."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question or prompt for the human user"
                },
                "choices": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional list of choices to present to the user"
                },
                "context": {
                    "type": "string",
                    "description": "Optional context about what you are doing and why you need input"
                }
            },
            "required": ["question"]
        })
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<ToolOutput> {
        let question = input["question"]
            .as_str()
            .unwrap_or("(no question provided)")
            .to_string();

        let choices: Vec<String> = input
            .get("choices")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let context = input.get("context").and_then(|v| v.as_str()).map(String::from);

        let request = HumanInputRequest {
            prompt: question,
            choices,
            context,
        };

        let response = self.handler.request_input_erased(&request).await?;
        Ok(ToolOutput::text(response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockHandler {
        response: String,
    }

    impl HumanInputHandler for MockHandler {
        async fn request_input(&self, _request: &HumanInputRequest) -> Result<String> {
            Ok(self.response.clone())
        }
    }

    #[tokio::test]
    async fn test_ask_human_tool_basic() {
        let tool = AskHumanTool::new(MockHandler {
            response: "42".into(),
        });

        assert_eq!(tool.name(), "ask_human");

        let input = serde_json::json!({"question": "What is the answer?"});
        let output = tool.execute(&input).await.unwrap();
        assert_eq!(output.content, "42");
        assert!(!output.is_error);
    }

    #[tokio::test]
    async fn test_ask_human_tool_with_choices() {
        let tool = AskHumanTool::new(MockHandler {
            response: "yes".into(),
        });

        let input = serde_json::json!({
            "question": "Approve?",
            "choices": ["yes", "no"],
            "context": "Deploying to production"
        });
        let output = tool.execute(&input).await.unwrap();
        assert_eq!(output.content, "yes");
    }

    #[tokio::test]
    async fn test_ask_human_tool_schema() {
        let tool = AskHumanTool::new(MockHandler {
            response: "".into(),
        });
        let schema = tool.parameters_schema();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["question"].is_object());
        assert!(schema["required"].as_array().unwrap().contains(&serde_json::json!("question")));
    }

    struct ContextCapturingHandler {
        captured: tokio::sync::Mutex<Option<HumanInputRequest>>,
    }

    impl HumanInputHandler for ContextCapturingHandler {
        async fn request_input(&self, request: &HumanInputRequest) -> Result<String> {
            *self.captured.lock().await = Some(request.clone());
            Ok("ok".into())
        }
    }

    #[tokio::test]
    async fn test_handler_receives_full_request() {
        let handler = Arc::new(ContextCapturingHandler {
            captured: tokio::sync::Mutex::new(None),
        });

        let tool = AskHumanTool {
            handler: handler.clone(),
        };

        let input = serde_json::json!({
            "question": "Q?",
            "choices": ["a", "b"],
            "context": "testing"
        });
        tool.execute(&input).await.unwrap();

        let captured = handler.captured.lock().await;
        let req = captured.as_ref().unwrap();
        assert_eq!(req.prompt, "Q?");
        assert_eq!(req.choices, vec!["a", "b"]);
        assert_eq!(req.context.as_deref(), Some("testing"));
    }
}
