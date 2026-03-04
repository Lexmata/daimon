//! Wrap an [`Agent`] as a [`Tool`] so one agent can delegate to another.
//!
//! This enables multi-agent patterns where a "coordinator" agent treats
//! specialized agents as callable tools.
//!
//! ```ignore
//! use daimon::agent::as_tool::AgentTool;
//!
//! let research_agent = Arc::new(
//!     Agent::builder()
//!         .model(model.clone())
//!         .system_prompt("You are a research specialist.")
//!         .build()?
//! );
//!
//! let coordinator = Agent::builder()
//!     .model(model)
//!     .tool(AgentTool::new(research_agent, "research", "Perform deep research on a topic"))
//!     .build()?;
//! ```

use std::sync::Arc;

use crate::agent::Agent;
use crate::error::Result;
use crate::tool::{Tool, ToolOutput};

/// Wraps an [`Agent`] as a [`Tool`], letting another agent invoke it by name.
///
/// The wrapped agent receives the `"input"` field from the tool arguments as
/// its prompt and returns its `final_text` as the tool output.
pub struct AgentTool {
    agent: Arc<Agent>,
    name: String,
    description: String,
}

impl AgentTool {
    /// Creates a new agent-as-tool wrapper.
    ///
    /// * `agent` — the agent to wrap (shared ownership via `Arc`)
    /// * `name` — tool name the calling agent will use
    /// * `description` — description the model sees when deciding to call this tool
    pub fn new(
        agent: Arc<Agent>,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            agent,
            name: name.into(),
            description: description.into(),
        }
    }
}

impl Tool for AgentTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The task or question to send to the agent"
                }
            },
            "required": ["input"]
        })
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<ToolOutput> {
        let prompt = input
            .get("input")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        match self.agent.prompt(prompt).await {
            Ok(response) => Ok(ToolOutput::text(response.final_text)),
            Err(e) => Ok(ToolOutput::error(format!(
                "Agent '{}' failed: {e}",
                self.name
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Model;
    use crate::model::types::*;
    use crate::stream::ResponseStream;

    struct EchoModel;

    impl Model for EchoModel {
        async fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
            let last = request
                .messages
                .last()
                .and_then(|m| m.content.as_deref())
                .unwrap_or("nothing");
            Ok(ChatResponse {
                message: Message::assistant(format!("sub-agent says: {last}")),
                stop_reason: StopReason::EndTurn,
                usage: Some(Usage::default()),
            })
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    #[tokio::test]
    async fn test_agent_tool_name_and_description() {
        let agent = Arc::new(Agent::builder().model(EchoModel).build().unwrap());
        let tool = AgentTool::new(agent, "researcher", "Does research");

        assert_eq!(tool.name(), "researcher");
        assert_eq!(tool.description(), "Does research");
    }

    #[tokio::test]
    async fn test_agent_tool_schema() {
        let agent = Arc::new(Agent::builder().model(EchoModel).build().unwrap());
        let tool = AgentTool::new(agent, "test", "test");
        let schema = tool.parameters_schema();

        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["input"].is_object());
        assert_eq!(schema["required"][0], "input");
    }

    #[tokio::test]
    async fn test_agent_tool_execute() {
        let agent = Arc::new(Agent::builder().model(EchoModel).build().unwrap());
        let tool = AgentTool::new(agent, "sub", "sub-agent");

        let input = serde_json::json!({"input": "hello world"});
        let output = tool.execute(&input).await.unwrap();

        assert!(!output.is_error);
        assert!(output.content.contains("sub-agent says: hello world"));
    }

    #[tokio::test]
    async fn test_agent_tool_missing_input() {
        let agent = Arc::new(Agent::builder().model(EchoModel).build().unwrap());
        let tool = AgentTool::new(agent, "sub", "sub-agent");

        let input = serde_json::json!({});
        let output = tool.execute(&input).await.unwrap();
        assert!(!output.is_error);
    }
}
