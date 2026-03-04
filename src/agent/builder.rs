use std::sync::Arc;

use crate::agent::Agent;
use crate::agent::hitl::{AskHumanTool, HumanInputHandler};
use crate::cost::{CostModel, CostTracker};
use crate::error::{DaimonError, Result};
use crate::guardrails::{InputGuardrail, OutputGuardrail};
use crate::hooks::{AgentHook, ErasedAgentHook, NoOpHook};
use crate::memory::{Memory, SharedMemory, SlidingWindowMemory};
use crate::middleware::{Middleware, MiddlewareStack};
use crate::model::{Model, SharedModel};
use crate::prompt::PromptTemplate;
use crate::tool::{Tool, ToolRegistry, ToolRetryPolicy};

/// Fluent builder for constructing an [`Agent`].
///
/// Model is required; all other fields have defaults. Call [`build`](AgentBuilder::build) to produce the agent.
pub struct AgentBuilder {
    model: Option<SharedModel>,
    system_prompt: Option<String>,
    prompt_template: Option<PromptTemplate>,
    tools: ToolRegistry,
    memory: Option<SharedMemory>,
    hooks: Option<Arc<dyn ErasedAgentHook>>,
    middleware: MiddlewareStack,
    input_guardrails: Vec<Arc<dyn crate::guardrails::ErasedInputGuardrail>>,
    output_guardrails: Vec<Arc<dyn crate::guardrails::ErasedOutputGuardrail>>,
    max_iterations: usize,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    validate_tool_inputs: bool,
    cost_model: Option<Arc<dyn CostModel>>,
    max_budget: Option<f64>,
    tool_retry_policy: Option<ToolRetryPolicy>,
}

impl AgentBuilder {
    /// Creates a new builder with default settings (max_iterations: 25, no model/tools/memory).
    pub fn new() -> Self {
        Self {
            model: None,
            system_prompt: None,
            prompt_template: None,
            tools: ToolRegistry::new(),
            memory: None,
            hooks: None,
            middleware: MiddlewareStack::new(),
            input_guardrails: Vec::new(),
            output_guardrails: Vec::new(),
            max_iterations: 25,
            temperature: None,
            max_tokens: None,
            validate_tool_inputs: true,
            cost_model: None,
            max_budget: None,
            tool_retry_policy: None,
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

    /// Enables or disables JSON Schema validation of tool inputs before execution.
    ///
    /// When enabled (the default), the agent validates each tool call's arguments
    /// against the tool's declared `parameters_schema()` before executing it.
    /// Invalid inputs are returned as error messages to the model so it can
    /// correct itself on the next iteration.
    pub fn validate_tool_inputs(mut self, enabled: bool) -> Self {
        self.validate_tool_inputs = enabled;
        self
    }

    /// Adds human-in-the-loop support. Registers an `ask_human` tool that the
    /// agent can call to request input from the user. The handler receives the
    /// request and must return the human's response.
    pub fn human_input<H: HumanInputHandler + 'static>(mut self, handler: H) -> Self {
        let _ = self.tools.register(AskHumanTool::new(handler));
        self
    }

    /// Adds a middleware layer to the agent's pipeline.
    pub fn middleware<M: Middleware + 'static>(mut self, mw: M) -> Self {
        self.middleware.push(mw);
        self
    }

    /// Adds an input guardrail that validates user input before processing.
    pub fn input_guardrail<G: InputGuardrail + 'static>(mut self, guard: G) -> Self {
        self.input_guardrails.push(Arc::new(guard));
        self
    }

    /// Adds an output guardrail that validates model output before returning.
    pub fn output_guardrail<G: OutputGuardrail + 'static>(mut self, guard: G) -> Self {
        self.output_guardrails.push(Arc::new(guard));
        self
    }

    /// Sets a prompt template with variable interpolation instead of a static system prompt.
    pub fn prompt_template(mut self, template: PromptTemplate) -> Self {
        self.prompt_template = Some(template);
        self
    }

    /// Sets a cost model for tracking token spend. Combine with [`max_budget`](Self::max_budget)
    /// to enforce a spending limit.
    pub fn cost_model<C: CostModel + 'static>(mut self, model: C) -> Self {
        self.cost_model = Some(Arc::new(model));
        self
    }

    /// Sets the maximum dollar budget for a single prompt call. The agent aborts
    /// with [`DaimonError::BudgetExceeded`] once cumulative cost crosses this.
    pub fn max_budget(mut self, dollars: f64) -> Self {
        self.max_budget = Some(dollars);
        self
    }

    /// Sets a retry policy for transient tool execution failures.
    pub fn tool_retry_policy(mut self, policy: ToolRetryPolicy) -> Self {
        self.tool_retry_policy = Some(policy);
        self
    }

    /// Builds the agent. Fails if model is not set.
    ///
    /// Pre-compiles JSON Schema validators and caches tool specs so the
    /// ReAct loop avoids per-iteration allocation.
    pub fn build(mut self) -> Result<Agent> {
        let model = self
            .model
            .ok_or_else(|| DaimonError::Builder("model is required".into()))?;

        let memory = self
            .memory
            .unwrap_or_else(|| Arc::new(SlidingWindowMemory::default()));

        let hooks = self.hooks.unwrap_or_else(|| Arc::new(NoOpHook));

        self.tools.warm_cache();

        let system_prompt = if let Some(ref tpl) = self.prompt_template {
            Some(tpl.render_static())
        } else {
            self.system_prompt
        };

        let cost_tracker = self.cost_model.map(CostTracker::new);

        Ok(Agent {
            model,
            system_prompt,
            tools: self.tools,
            memory,
            hooks,
            middleware: self.middleware,
            input_guardrails: self.input_guardrails,
            output_guardrails: self.output_guardrails,
            max_iterations: self.max_iterations,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            validate_tool_inputs: self.validate_tool_inputs,
            cost_tracker,
            max_budget: self.max_budget,
            tool_retry_policy: self.tool_retry_policy,
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
