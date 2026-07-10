//! Agent cloning and forking from checkpoints.
//!
//! Use [`Agent::fork`] to create a duplicate of an agent with independent
//! (fresh) memory, sharing the same model, tools, hooks, and middleware.
//! Use [`Agent::fork_from_checkpoint`] to pre-populate the forked agent's
//! memory with messages from a saved checkpoint.
//! Use [`Agent::fork_builder`] for fine-grained mutation (change system prompt,
//! add/remove tools, swap model) before finalizing the fork.

use std::sync::Arc;

use crate::agent::Agent;
use crate::checkpoint::ErasedCheckpoint;
use crate::cost::CostTracker;
use crate::error::{DaimonError, Result};
use crate::guardrails::{InputGuardrail, OutputGuardrail};
use crate::hooks::{AgentHook, ErasedAgentHook};
use crate::memory::{Memory, SharedMemory, SlidingWindowMemory};
use crate::middleware::{Middleware, MiddlewareStack};
use crate::model::{Model, SharedModel};
use crate::tool::{Tool, ToolRegistry, ToolRetryPolicy};

impl Agent {
    /// Creates a new agent that shares the current agent's model, tools,
    /// hooks, middleware, and guardrails but has independent (empty) memory.
    ///
    /// The forked agent can run concurrently without affecting the original
    /// agent's conversation history.
    pub fn fork(&self) -> Agent {
        let cost_tracker = self
            .cost_tracker
            .as_ref()
            .map(|t| CostTracker::new(Arc::clone(&t.cost_model)));

        Agent {
            model: self.model.clone(),
            system_prompt: self.system_prompt.clone(),
            tools: self.tools.clone(),
            memory: Arc::new(SlidingWindowMemory::default()),
            hooks: self.hooks.clone(),
            middleware: self.middleware.clone(),
            input_guardrails: self.input_guardrails.clone(),
            output_guardrails: self.output_guardrails.clone(),
            max_iterations: self.max_iterations,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            validate_tool_inputs: self.validate_tool_inputs,
            cost_tracker,
            max_budget: self.max_budget,
            tool_retry_policy: self.tool_retry_policy.clone(),
        }
    }

    /// Creates a new agent from a checkpoint, pre-loading the checkpoint's
    /// message history into fresh memory.
    ///
    /// The forked agent shares the current agent's model, tools, hooks, and
    /// configuration but starts with independent memory seeded from the
    /// checkpoint. This enables "what-if" branching: modify the forked
    /// agent's tools or system prompt and see how the run diverges.
    ///
    /// Returns an error if the checkpoint does not exist for the given `run_id`.
    pub async fn fork_from_checkpoint(
        &self,
        run_id: &str,
        checkpoint: &Arc<dyn ErasedCheckpoint>,
    ) -> Result<Agent> {
        let state = checkpoint
            .load_erased(run_id)
            .await?
            .ok_or_else(|| DaimonError::Other(format!("no checkpoint for run '{run_id}'")))?;

        // The window must leave real headroom beyond the seeded history: a
        // fixed +50 margin evicts the seeded context after a short post-fork
        // conversation, so scale the margin with the history size instead.
        let capacity = (state.messages.len() * 2).max(state.messages.len() + 50);
        let memory = SlidingWindowMemory::new(capacity);
        for msg in &state.messages {
            memory.add_message(msg.clone()).await?;
        }

        let cost_tracker = self
            .cost_tracker
            .as_ref()
            .map(|t| CostTracker::new(Arc::clone(&t.cost_model)));

        Ok(Agent {
            model: self.model.clone(),
            system_prompt: self.system_prompt.clone(),
            tools: self.tools.clone(),
            memory: Arc::new(memory),
            hooks: self.hooks.clone(),
            middleware: self.middleware.clone(),
            input_guardrails: self.input_guardrails.clone(),
            output_guardrails: self.output_guardrails.clone(),
            max_iterations: self.max_iterations,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            validate_tool_inputs: self.validate_tool_inputs,
            cost_tracker,
            max_budget: self.max_budget,
            tool_retry_policy: self.tool_retry_policy.clone(),
        })
    }

    /// Creates a new agent with the same configuration but a different
    /// memory backend.
    ///
    /// Useful for switching from in-memory to persistent storage, or for
    /// running the same agent configuration against a fresh conversation.
    pub fn fork_with_memory<M: Memory + 'static>(&self, memory: M) -> Agent {
        let cost_tracker = self
            .cost_tracker
            .as_ref()
            .map(|t| CostTracker::new(Arc::clone(&t.cost_model)));

        Agent {
            model: self.model.clone(),
            system_prompt: self.system_prompt.clone(),
            tools: self.tools.clone(),
            memory: Arc::new(memory),
            hooks: self.hooks.clone(),
            middleware: self.middleware.clone(),
            input_guardrails: self.input_guardrails.clone(),
            output_guardrails: self.output_guardrails.clone(),
            max_iterations: self.max_iterations,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            validate_tool_inputs: self.validate_tool_inputs,
            cost_tracker,
            max_budget: self.max_budget,
            tool_retry_policy: self.tool_retry_policy.clone(),
        }
    }

    /// Returns a [`ForkBuilder`] pre-populated with this agent's configuration.
    ///
    /// The builder lets you selectively mutate the system prompt, tools,
    /// model, memory, hooks, or any other setting before constructing
    /// the forked agent.
    ///
    /// ```ignore
    /// let variant = agent.fork_builder()
    ///     .system_prompt("You are a code reviewer.")
    ///     .tool(ReviewTool)
    ///     .remove_tool("search")
    ///     .build()?;
    /// ```
    pub fn fork_builder(&self) -> ForkBuilder {
        ForkBuilder {
            tool_error: None,
            model: self.model.clone(),
            system_prompt: self.system_prompt.clone(),
            tools: self.tools.clone(),
            memory: None,
            hooks: Some(self.hooks.clone()),
            middleware: self.middleware.clone(),
            input_guardrails: self.input_guardrails.clone(),
            output_guardrails: self.output_guardrails.clone(),
            max_iterations: self.max_iterations,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            validate_tool_inputs: self.validate_tool_inputs,
            cost_model: self
                .cost_tracker
                .as_ref()
                .map(|t| Arc::clone(&t.cost_model)),
            max_budget: self.max_budget,
            tool_retry_policy: self.tool_retry_policy.clone(),
        }
    }
}

/// Builder for creating a mutated fork of an existing agent.
///
/// Obtained via [`Agent::fork_builder()`]. All fields start with the
/// parent agent's values; call setters to override specific fields,
/// then [`build()`](ForkBuilder::build) to produce the new agent.
pub struct ForkBuilder {
    model: SharedModel,
    system_prompt: Option<String>,
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
    cost_model: Option<Arc<dyn crate::cost::CostModel>>,
    max_budget: Option<f64>,
    tool_retry_policy: Option<ToolRetryPolicy>,
    /// First tool-registration error, surfaced by [`build`](ForkBuilder::build)
    /// so a duplicate tool name cannot be silently dropped.
    tool_error: Option<DaimonError>,
}

impl ForkBuilder {
    /// Replaces the LLM model.
    pub fn model<M: Model + 'static>(mut self, model: M) -> Self {
        self.model = Arc::new(model);
        self
    }

    /// Replaces the system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Clears the system prompt.
    pub fn no_system_prompt(mut self) -> Self {
        self.system_prompt = None;
        self
    }

    /// Adds a tool to the forked agent's registry.
    ///
    /// A name collision with an inherited or previously added tool is an
    /// error surfaced by [`build`](ForkBuilder::build) — the duplicate is
    /// never silently dropped. Use [`remove_tool`](ForkBuilder::remove_tool)
    /// first to replace an inherited tool.
    pub fn tool<T: Tool + 'static>(mut self, tool: T) -> Self {
        if let Err(e) = self.tools.register(tool) {
            self.tool_error.get_or_insert(e);
        }
        self
    }

    /// Removes a tool by name from the forked agent's registry.
    pub fn remove_tool(mut self, name: &str) -> Self {
        self.tools.unregister(name);
        self
    }

    /// Replaces the memory backend. Defaults to fresh `SlidingWindowMemory`.
    pub fn memory<M: Memory + 'static>(mut self, memory: M) -> Self {
        self.memory = Some(Arc::new(memory));
        self
    }

    /// Replaces the lifecycle hooks.
    pub fn hooks<H: AgentHook + 'static>(mut self, hooks: H) -> Self {
        self.hooks = Some(Arc::new(hooks));
        self
    }

    /// Adds a middleware layer.
    pub fn middleware<M: Middleware + 'static>(mut self, mw: M) -> Self {
        self.middleware.push(mw);
        self
    }

    /// Adds an input guardrail.
    pub fn input_guardrail<G: InputGuardrail + 'static>(mut self, guard: G) -> Self {
        self.input_guardrails.push(Arc::new(guard));
        self
    }

    /// Adds an output guardrail.
    pub fn output_guardrail<G: OutputGuardrail + 'static>(mut self, guard: G) -> Self {
        self.output_guardrails.push(Arc::new(guard));
        self
    }

    /// Overrides the maximum number of ReAct iterations.
    pub fn max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Overrides the sampling temperature.
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Overrides the max output tokens.
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Overrides tool input validation.
    pub fn validate_tool_inputs(mut self, enabled: bool) -> Self {
        self.validate_tool_inputs = enabled;
        self
    }

    /// Overrides the tool retry policy.
    pub fn tool_retry_policy(mut self, policy: ToolRetryPolicy) -> Self {
        self.tool_retry_policy = Some(policy);
        self
    }

    /// Builds the forked agent with all applied mutations.
    ///
    /// Fails if a tool registration failed (e.g. [`tool`](ForkBuilder::tool)
    /// collided with an inherited tool's name).
    pub fn build(mut self) -> Result<Agent> {
        if let Some(e) = self.tool_error {
            return Err(e);
        }

        let memory = self
            .memory
            .unwrap_or_else(|| Arc::new(SlidingWindowMemory::default()));

        let hooks = self
            .hooks
            .unwrap_or_else(|| Arc::new(crate::hooks::NoOpHook));

        self.tools.warm_cache();

        let cost_tracker = self.cost_model.map(CostTracker::new);

        Ok(Agent {
            model: self.model,
            system_prompt: self.system_prompt,
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::agent::Agent;
    use crate::checkpoint::{Checkpoint, CheckpointState, InMemoryCheckpoint};
    use crate::error::Result;
    use crate::memory::SlidingWindowMemory;
    use crate::model::Model;
    use crate::model::types::*;
    use crate::stream::ResponseStream;
    use crate::tool::{Tool, ToolOutput};

    struct EchoModel;

    impl Model for EchoModel {
        async fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
            let last = request
                .messages
                .last()
                .and_then(|m| m.content.as_deref())
                .unwrap_or("none");
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

    struct DummyTool {
        tool_name: &'static str,
    }

    impl Tool for DummyTool {
        fn name(&self) -> &str {
            self.tool_name
        }
        fn description(&self) -> &str {
            "A dummy tool"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(&self, _input: &serde_json::Value) -> Result<ToolOutput> {
            Ok(ToolOutput::text("ok"))
        }
    }

    #[tokio::test]
    async fn test_fork_has_independent_memory() {
        let agent = Agent::builder().model(EchoModel).build().unwrap();
        agent.prompt("hello").await.unwrap();

        let forked = agent.fork();

        let original_msgs = agent.memory.get_messages_erased().await.unwrap();
        let forked_msgs = forked.memory.get_messages_erased().await.unwrap();

        assert_eq!(original_msgs.len(), 2);
        assert_eq!(forked_msgs.len(), 0);
    }

    #[tokio::test]
    async fn test_fork_preserves_config() {
        let agent = Agent::builder()
            .model(EchoModel)
            .system_prompt("Be helpful")
            .max_iterations(10)
            .temperature(0.5)
            .build()
            .unwrap();

        let forked = agent.fork();
        assert_eq!(forked.system_prompt.as_deref(), Some("Be helpful"));
        assert_eq!(forked.max_iterations, 10);
        assert_eq!(forked.temperature, Some(0.5));
    }

    #[tokio::test]
    async fn test_fork_from_checkpoint() {
        let agent = Agent::builder().model(EchoModel).build().unwrap();
        let cp = Arc::new(InMemoryCheckpoint::new());

        let state = CheckpointState::new(
            "run-1",
            vec![Message::user("hi"), Message::assistant("hello")],
            1,
        );
        cp.save(&state).await.unwrap();

        let forked = agent
            .fork_from_checkpoint("run-1", &(cp as Arc<_>))
            .await
            .unwrap();
        let msgs = forked.memory.get_messages_erased().await.unwrap();
        assert_eq!(msgs.len(), 2);
    }

    #[tokio::test]
    async fn test_fork_from_checkpoint_missing_run() {
        let agent = Agent::builder().model(EchoModel).build().unwrap();
        let cp: Arc<dyn crate::checkpoint::ErasedCheckpoint> = Arc::new(InMemoryCheckpoint::new());

        let result = agent.fork_from_checkpoint("missing", &cp).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_fork_with_memory() {
        let agent = Agent::builder().model(EchoModel).build().unwrap();
        agent.prompt("original").await.unwrap();

        let custom_mem = SlidingWindowMemory::new(5);
        let forked = agent.fork_with_memory(custom_mem);

        let original_msgs = agent.memory.get_messages_erased().await.unwrap();
        let forked_msgs = forked.memory.get_messages_erased().await.unwrap();
        assert_eq!(original_msgs.len(), 2);
        assert_eq!(forked_msgs.len(), 0);
    }

    #[tokio::test]
    async fn test_forked_agents_run_independently() {
        let agent = Agent::builder().model(EchoModel).build().unwrap();
        let forked = agent.fork();

        agent.prompt("msg1").await.unwrap();
        forked.prompt("msg2").await.unwrap();

        let a_msgs = agent.memory.get_messages_erased().await.unwrap();
        let f_msgs = forked.memory.get_messages_erased().await.unwrap();

        assert_eq!(a_msgs.len(), 2);
        assert_eq!(f_msgs.len(), 2);
        assert!(a_msgs[0].content.as_deref().unwrap().contains("msg1"));
        assert!(f_msgs[0].content.as_deref().unwrap().contains("msg2"));
    }

    #[tokio::test]
    async fn test_fork_builder_changes_system_prompt() {
        let agent = Agent::builder()
            .model(EchoModel)
            .system_prompt("Original prompt")
            .build()
            .unwrap();

        let forked = agent
            .fork_builder()
            .system_prompt("New prompt")
            .build()
            .unwrap();

        assert_eq!(forked.system_prompt.as_deref(), Some("New prompt"));
    }

    #[tokio::test]
    async fn test_fork_builder_clears_system_prompt() {
        let agent = Agent::builder()
            .model(EchoModel)
            .system_prompt("Original")
            .build()
            .unwrap();

        let forked = agent.fork_builder().no_system_prompt().build().unwrap();
        assert!(forked.system_prompt.is_none());
    }

    #[tokio::test]
    async fn test_fork_builder_adds_and_removes_tools() {
        let agent = Agent::builder()
            .model(EchoModel)
            .tool(DummyTool { tool_name: "alpha" })
            .tool(DummyTool { tool_name: "beta" })
            .build()
            .unwrap();

        assert_eq!(agent.tools.len(), 2);

        let forked = agent
            .fork_builder()
            .remove_tool("alpha")
            .tool(DummyTool { tool_name: "gamma" })
            .build()
            .unwrap();

        assert!(forked.tools.get("alpha").is_none());
        assert!(forked.tools.get("beta").is_some());
        assert!(forked.tools.get("gamma").is_some());
        assert_eq!(forked.tools.len(), 2);
    }

    #[tokio::test]
    async fn test_fork_builder_overrides_iterations_and_temp() {
        let agent = Agent::builder()
            .model(EchoModel)
            .max_iterations(5)
            .temperature(0.7)
            .build()
            .unwrap();

        let forked = agent
            .fork_builder()
            .max_iterations(20)
            .temperature(0.1)
            .build()
            .unwrap();

        assert_eq!(forked.max_iterations, 20);
        assert_eq!(forked.temperature, Some(0.1));
    }

    #[tokio::test]
    async fn test_fork_builder_preserves_unchanged() {
        let agent = Agent::builder()
            .model(EchoModel)
            .system_prompt("Keep me")
            .max_iterations(8)
            .build()
            .unwrap();

        let forked = agent.fork_builder().build().unwrap();

        assert_eq!(forked.system_prompt.as_deref(), Some("Keep me"));
        assert_eq!(forked.max_iterations, 8);
    }

    #[tokio::test]
    async fn test_fork_builder_independent_memory() {
        let agent = Agent::builder().model(EchoModel).build().unwrap();
        agent.prompt("hello").await.unwrap();

        let forked = agent.fork_builder().build().unwrap();

        let original_msgs = agent.memory.get_messages_erased().await.unwrap();
        let forked_msgs = forked.memory.get_messages_erased().await.unwrap();
        assert_eq!(original_msgs.len(), 2);
        assert_eq!(forked_msgs.len(), 0);
    }

    #[tokio::test]
    async fn test_fork_builder_custom_memory() {
        let agent = Agent::builder().model(EchoModel).build().unwrap();

        let mem = SlidingWindowMemory::new(3);
        let forked = agent.fork_builder().memory(mem).build().unwrap();

        forked.prompt("a").await.unwrap();
        let msgs = forked.memory.get_messages_erased().await.unwrap();
        assert_eq!(msgs.len(), 2);
    }

    #[tokio::test]
    async fn test_fork_builder_replace_model() {
        struct AltModel;
        impl Model for AltModel {
            async fn generate(&self, _req: &ChatRequest) -> Result<ChatResponse> {
                Ok(ChatResponse {
                    message: Message::assistant("alt".to_string()),
                    stop_reason: StopReason::EndTurn,
                    usage: Some(Usage::default()),
                })
            }
            async fn generate_stream(&self, _req: &ChatRequest) -> Result<ResponseStream> {
                Ok(Box::pin(futures::stream::empty()))
            }
        }

        let agent = Agent::builder().model(EchoModel).build().unwrap();
        let forked = agent.fork_builder().model(AltModel).build().unwrap();

        let resp = forked.prompt("test").await.unwrap();
        assert_eq!(resp.text(), "alt");
    }

    #[tokio::test]
    async fn test_fork_builder_duplicate_tool_fails_build() {
        // Adding a tool whose name collides with an inherited tool must
        // surface as an error from build(), never be silently dropped.
        let agent = Agent::builder()
            .model(EchoModel)
            .tool(DummyTool { tool_name: "alpha" })
            .build()
            .unwrap();

        let result = agent
            .fork_builder()
            .tool(DummyTool { tool_name: "alpha" })
            .build();

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            crate::error::DaimonError::DuplicateTool(name) if name == "alpha"
        ));
    }

    #[tokio::test]
    async fn test_fork_from_checkpoint_scales_memory_with_history() {
        // 60 seeded + 55 appended = 115 messages. The old fixed +50 margin
        // capped the window at 110 and evicted the seeded context; the
        // proportional margin (len * 2 = 120) retains it.
        let agent = Agent::builder().model(EchoModel).build().unwrap();
        let cp = Arc::new(InMemoryCheckpoint::new());

        let seeded: Vec<Message> = (0..60)
            .map(|i| Message::user(format!("seed-{i}")))
            .collect();
        let state = CheckpointState::new("run-k", seeded, 1);
        cp.save(&state).await.unwrap();

        let forked = agent
            .fork_from_checkpoint("run-k", &(cp as Arc<_>))
            .await
            .unwrap();
        for i in 0..55 {
            forked
                .memory
                .add_message_erased(Message::user(format!("post-{i}")))
                .await
                .unwrap();
        }

        let msgs = forked.memory.get_messages_erased().await.unwrap();
        assert_eq!(msgs.len(), 115, "no message may be evicted at this depth");
        assert_eq!(
            msgs[0].content.as_deref(),
            Some("seed-0"),
            "the earliest seeded context must survive the post-fork conversation"
        );
    }
}
