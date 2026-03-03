use std::sync::Arc;

use crate::agent::Agent;
use crate::error::{DaimonError, Result};
use crate::hooks::{AgentHook, ErasedAgentHook, NoOpHook};
use crate::memory::{Memory, SharedMemory, SlidingWindowMemory};
use crate::model::{Model, SharedModel};
use crate::tool::{Tool, ToolRegistry};

/// Fluent builder for constructing an [`Agent`].
///
/// Model is required; all other fields have defaults. Call [`build`](AgentBuilder::build) to produce the agent.
pub struct AgentBuilder {
    model: Option<SharedModel>,
    system_prompt: Option<String>,
    tools: ToolRegistry,
    memory: Option<SharedMemory>,
    hooks: Option<Arc<dyn ErasedAgentHook>>,
    max_iterations: usize,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
}

impl AgentBuilder {
    /// Creates a new builder with default settings (max_iterations: 25, no model/tools/memory).
    pub fn new() -> Self {
        Self {
            model: None,
            system_prompt: None,
            tools: ToolRegistry::new(),
            memory: None,
            hooks: None,
            max_iterations: 25,
            temperature: None,
            max_tokens: None,
        }
    }

    /// Sets the LLM provider. Required for [`build`](AgentBuilder::build) to succeed.
    pub fn model<M: Model + 'static>(mut self, model: M) -> Self {
        self.model = Some(Arc::new(model));
        self
    }

    /// Sets a pre-boxed shared model. Use when you need to share a model across multiple agents.
    pub fn shared_model(mut self, model: SharedModel) -> Self {
        self.model = Some(model);
        self
    }

    /// Sets the system prompt injected at the start of each conversation.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Registers a tool the agent can invoke. Tools must have unique names.
    pub fn tool<T: Tool + 'static>(mut self, tool: T) -> Self {
        let _ = self.tools.register(tool);
        self
    }

    /// Sets conversation memory. Defaults to [`SlidingWindowMemory`] with 50 messages if not set.
    pub fn memory<M: Memory + 'static>(mut self, memory: M) -> Self {
        self.memory = Some(Arc::new(memory));
        self
    }

    /// Sets lifecycle hooks for observability or control. Defaults to [`NoOpHook`] if not set.
    pub fn hooks<H: AgentHook + 'static>(mut self, hooks: H) -> Self {
        self.hooks = Some(Arc::new(hooks));
        self
    }

    /// Sets the maximum ReAct loop iterations before aborting. Default: 25.
    pub fn max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Sets model temperature (0.0–2.0). Passed through to the model if supported.
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Sets max tokens for model output. Passed through to the model if supported.
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Builds the agent. Fails if model is not set.
    pub fn build(self) -> Result<Agent> {
        let model = self
            .model
            .ok_or_else(|| DaimonError::Builder("model is required".into()))?;

        let memory = self
            .memory
            .unwrap_or_else(|| Arc::new(SlidingWindowMemory::default()));

        let hooks = self.hooks.unwrap_or_else(|| Arc::new(NoOpHook));

        Ok(Agent {
            model,
            system_prompt: self.system_prompt,
            tools: self.tools,
            memory,
            hooks,
            max_iterations: self.max_iterations,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
        })
    }
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::types::{ChatRequest, ChatResponse, Message, StopReason, Usage};
    use crate::stream::ResponseStream;
    use crate::tool::ToolOutput;

    struct FakeModel;

    impl Model for FakeModel {
        async fn generate(&self, _request: &ChatRequest) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message::assistant("hello"),
                stop_reason: StopReason::EndTurn,
                usage: Some(Usage::default()),
            })
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    struct FakeTool;

    impl Tool for FakeTool {
        fn name(&self) -> &str {
            "fake"
        }
        fn description(&self) -> &str {
            "fake tool"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(&self, _input: &serde_json::Value) -> Result<ToolOutput> {
            Ok(ToolOutput::text("done"))
        }
    }

    #[test]
    fn test_build_without_model_fails() {
        let result = AgentBuilder::new().build();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), DaimonError::Builder(_)));
    }

    #[test]
    fn test_build_with_model_succeeds() {
        let agent = AgentBuilder::new().model(FakeModel).build();
        assert!(agent.is_ok());
    }

    #[test]
    fn test_build_with_all_options() {
        let agent = AgentBuilder::new()
            .model(FakeModel)
            .system_prompt("You are helpful.")
            .tool(FakeTool)
            .memory(SlidingWindowMemory::new(10))
            .max_iterations(5)
            .temperature(0.7)
            .max_tokens(1000)
            .build();
        assert!(agent.is_ok());

        let agent = agent.unwrap();
        assert_eq!(agent.max_iterations, 5);
        assert_eq!(agent.system_prompt.as_deref(), Some("You are helpful."));
    }

    #[test]
    fn test_default_max_iterations() {
        let agent = AgentBuilder::new().model(FakeModel).build().unwrap();
        assert_eq!(agent.max_iterations, 25);
    }
}
