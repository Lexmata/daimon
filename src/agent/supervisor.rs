//! Supervisor pattern: one coordinating agent delegates tasks to specialized sub-agents.
//!
//! The supervisor wraps each sub-agent as an [`AgentTool`](super::as_tool::AgentTool),
//! then uses a coordinator agent whose tools *are* the sub-agents. The LLM decides
//! which sub-agent to invoke based on the task.
//!
//! ```ignore
//! use daimon::agent::supervisor::Supervisor;
//!
//! let supervisor = Supervisor::builder()
//!     .model(model.clone())
//!     .system_prompt("You coordinate research and writing tasks.")
//!     .agent("researcher", research_agent, "Performs deep research")
//!     .agent("writer", writing_agent, "Writes polished prose")
//!     .build()?;
//!
//! let response = supervisor.run("Write a blog post about Rust async").await?;
//! ```

use std::sync::Arc;

use crate::agent::as_tool::AgentTool;
use crate::agent::runner::AgentResponse;
use crate::agent::Agent;
use crate::error::{DaimonError, Result};
use crate::model::{Model, SharedModel};
use crate::tool::{Tool, ToolRegistry};

/// A supervisor that coordinates specialized sub-agents via tool calling.
pub struct Supervisor {
    coordinator: Agent,
}

impl Supervisor {
    /// Returns a new supervisor builder.
    pub fn builder() -> SupervisorBuilder {
        SupervisorBuilder::new()
    }

    /// Sends a task to the supervisor, which delegates to sub-agents as needed.
    pub async fn run(&self, input: &str) -> Result<AgentResponse> {
        self.coordinator.prompt(input).await
    }
}

/// Builder for constructing a [`Supervisor`].
pub struct SupervisorBuilder {
    model: Option<SharedModel>,
    system_prompt: Option<String>,
    agents: Vec<(String, Arc<Agent>, String)>,
    extra_tools: Vec<Box<dyn ErasedToolBoxed>>,
    max_iterations: usize,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
}

trait ErasedToolBoxed: Send + Sync {
    fn into_shared(self: Box<Self>) -> crate::tool::SharedTool;
}

impl<T: Tool + 'static> ErasedToolBoxed for T {
    fn into_shared(self: Box<Self>) -> crate::tool::SharedTool {
        Arc::new(*self)
    }
}

impl SupervisorBuilder {
    fn new() -> Self {
        Self {
            model: None,
            system_prompt: None,
            agents: Vec::new(),
            extra_tools: Vec::new(),
            max_iterations: 25,
            temperature: None,
            max_tokens: None,
        }
    }

    /// Sets the LLM for the coordinator agent.
    pub fn model<M: Model + 'static>(mut self, model: M) -> Self {
        self.model = Some(Arc::new(model));
        self
    }

    /// Sets a pre-boxed shared model.
    pub fn shared_model(mut self, model: SharedModel) -> Self {
        self.model = Some(model);
        self
    }

    /// Sets the coordinator's system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Registers a sub-agent. The agent is exposed as a tool with the given
    /// name and description.
    pub fn agent(
        mut self,
        name: impl Into<String>,
        agent: Arc<Agent>,
        description: impl Into<String>,
    ) -> Self {
        self.agents
            .push((name.into(), agent, description.into()));
        self
    }

    /// Registers an additional tool available to the coordinator alongside
    /// the sub-agent tools.
    pub fn tool<T: Tool + 'static>(mut self, tool: T) -> Self {
        self.extra_tools.push(Box::new(tool));
        self
    }

    /// Sets the maximum ReAct loop iterations. Default: 25.
    pub fn max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Sets model temperature.
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Sets max output tokens.
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Builds the supervisor.
    pub fn build(self) -> Result<Supervisor> {
        let model = self
            .model
            .ok_or_else(|| DaimonError::Builder("supervisor requires a model".into()))?;

        if self.agents.is_empty() {
            return Err(DaimonError::Builder(
                "supervisor requires at least one sub-agent".into(),
            ));
        }

        let mut tools = ToolRegistry::new();

        for (name, agent, description) in &self.agents {
            tools.register(AgentTool::new(agent.clone(), name, description))?;
        }

        for extra in self.extra_tools {
            tools.register_shared(extra.into_shared())?;
        }

        let mut builder = Agent::builder();
        builder = builder.shared_model(model).max_iterations(self.max_iterations);

        if let Some(prompt) = self.system_prompt {
            builder = builder.system_prompt(prompt);
        }
        if let Some(temp) = self.temperature {
            builder = builder.temperature(temp);
        }
        if let Some(max) = self.max_tokens {
            builder = builder.max_tokens(max);
        }

        let mut coordinator = builder.build()?;
        coordinator.tools = tools;

        Ok(Supervisor { coordinator })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::types::*;
    use crate::stream::ResponseStream;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct EchoModel;

    impl Model for EchoModel {
        async fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
            let last = request
                .messages
                .last()
                .and_then(|m| m.content.as_deref())
                .unwrap_or("nothing");
            Ok(ChatResponse {
                message: Message::assistant(format!("Echo: {last}")),
                stop_reason: StopReason::EndTurn,
                usage: Some(Usage::default()),
            })
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    struct DelegatingModel {
        call_count: AtomicUsize,
    }

    impl DelegatingModel {
        fn new() -> Self {
            Self {
                call_count: AtomicUsize::new(0),
            }
        }
    }

    impl Model for DelegatingModel {
        async fn generate(&self, _request: &ChatRequest) -> Result<ChatResponse> {
            let count = self.call_count.fetch_add(1, Ordering::SeqCst);
            if count == 0 {
                Ok(ChatResponse {
                    message: Message::assistant_with_tool_calls(vec![crate::tool::ToolCall {
                        id: "call_1".into(),
                        name: "researcher".into(),
                        arguments: serde_json::json!({"input": "what is Rust?"}),
                    }]),
                    stop_reason: StopReason::ToolUse,
                    usage: Some(Usage::default()),
                })
            } else {
                Ok(ChatResponse {
                    message: Message::assistant("Based on the research, Rust is great."),
                    stop_reason: StopReason::EndTurn,
                    usage: Some(Usage::default()),
                })
            }
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    #[test]
    fn test_supervisor_requires_model() {
        let agent = Arc::new(Agent::builder().model(EchoModel).build().unwrap());
        let result = Supervisor::builder()
            .agent("sub", agent, "sub-agent")
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_supervisor_requires_agents() {
        let result = Supervisor::builder().model(EchoModel).build();
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_supervisor_delegates_to_sub_agent() {
        let sub = Arc::new(Agent::builder().model(EchoModel).build().unwrap());

        let supervisor = Supervisor::builder()
            .model(DelegatingModel::new())
            .agent("researcher", sub, "Researches topics")
            .build()
            .unwrap();

        let response = supervisor.run("research Rust").await.unwrap();
        assert_eq!(response.text(), "Based on the research, Rust is great.");
        assert_eq!(response.iterations, 2);
    }
}
