//! Hot-reloadable agent wrapper.
//!
//! [`HotSwapAgent`] wraps an [`Agent`] behind an `RwLock<Arc<Agent>>`,
//! allowing you to swap the model, tools, system prompt, or other
//! configuration at runtime without restarting or rebuilding the agent.
//! Prompts run on an `Arc` snapshot taken at call time, so in-flight
//! prompts never block a swap (they simply finish on the old
//! configuration).
//!
//! ```ignore
//! use daimon::prelude::*;
//! use daimon::agent::hot_swap::HotSwapAgent;
//!
//! let agent = Agent::builder().model(my_model).build()?;
//! let hot = HotSwapAgent::new(agent);
//!
//! // Use normally
//! let response = hot.prompt("Hello").await?;
//!
//! // Swap the model at runtime
//! hot.swap_model(new_model).await;
//!
//! // Next prompt uses the new model
//! let response = hot.prompt("Hello again").await?;
//! ```

use std::sync::Arc;

use tokio::sync::RwLock;

use crate::agent::{Agent, AgentResponse};
use crate::cost::CostTracker;
use crate::error::Result;
use crate::guardrails::{InputGuardrail, OutputGuardrail};
use crate::hooks::AgentHook;
use crate::memory::Memory;
use crate::middleware::Middleware;
use crate::model::{Model, SharedModel};
use crate::stream::ResponseStream;
use crate::tool::{Tool, ToolRetryPolicy};

/// An agent wrapper that supports hot-reloading configuration at runtime.
///
/// The current agent lives behind an `RwLock<Arc<Agent>>`. Prompt operations
/// clone the `Arc` under a briefly-held read guard and run on that snapshot
/// **without holding the lock**, so a long multi-iteration ReAct loop never
/// blocks a swap, and a pending swap never stalls new prompts.
///
/// Swap operations build an updated agent and replace the `Arc` under a
/// write lock, blocking only for the duration of that pointer swap.
/// In-flight prompts complete on the configuration that was current when
/// they started; prompts issued after the swap see the new configuration.
/// Conversation memory is shared across swaps (unless explicitly replaced
/// via [`swap_memory`](HotSwapAgent::swap_memory) or
/// [`replace`](HotSwapAgent::replace)).
pub struct HotSwapAgent {
    inner: Arc<RwLock<Arc<Agent>>>,
}

/// Builds an updated copy of `agent` sharing all of its `Arc`-backed parts
/// (model, tools, memory, hooks, middleware, guardrails).
///
/// The cost tracker cannot be shared (it holds an atomic accumulator), so a
/// fresh one is seeded from the same cost model. This preserves semantics:
/// budget enforcement is per-run (see [`Agent::new_run_tracker`]), and the
/// agent-level tracker only serves as the cost-model holder.
fn clone_agent(agent: &Agent) -> Agent {
    Agent {
        model: agent.model.clone(),
        system_prompt: agent.system_prompt.clone(),
        tools: agent.tools.clone(),
        memory: agent.memory.clone(),
        hooks: agent.hooks.clone(),
        middleware: agent.middleware.clone(),
        input_guardrails: agent.input_guardrails.clone(),
        output_guardrails: agent.output_guardrails.clone(),
        max_iterations: agent.max_iterations,
        temperature: agent.temperature,
        max_tokens: agent.max_tokens,
        validate_tool_inputs: agent.validate_tool_inputs,
        cost_tracker: agent
            .cost_tracker
            .as_ref()
            .map(|t| CostTracker::new(Arc::clone(&t.cost_model))),
        max_budget: agent.max_budget,
        tool_retry_policy: agent.tool_retry_policy.clone(),
    }
}

impl HotSwapAgent {
    /// Wraps an existing agent for hot-reload support.
    pub fn new(agent: Agent) -> Self {
        Self {
            inner: Arc::new(RwLock::new(Arc::new(agent))),
        }
    }

    /// Returns the current agent snapshot, holding the read lock only for
    /// the duration of the `Arc` clone.
    async fn snapshot(&self) -> Arc<Agent> {
        Arc::clone(&*self.inner.read().await)
    }

    /// Applies a configuration update by cloning the current agent, mutating
    /// the clone, and swapping it in. The write lock is held only for the
    /// clone-and-swap, never across a model call.
    async fn update(&self, apply: impl FnOnce(&mut Agent)) {
        let mut guard = self.inner.write().await;
        let mut next = clone_agent(&guard);
        apply(&mut next);
        *guard = Arc::new(next);
    }

    /// Runs a prompt through the agent.
    ///
    /// The prompt executes on a snapshot of the configuration taken at call
    /// time; swaps performed while it is running do not affect it.
    pub async fn prompt(&self, input: &str) -> Result<AgentResponse> {
        let agent = self.snapshot().await;
        agent.prompt(input).await
    }

    /// Runs a streaming prompt through the agent.
    ///
    /// Like [`prompt`](HotSwapAgent::prompt), the stream is bound to the
    /// configuration current at call time.
    pub async fn prompt_stream(&self, input: &str) -> Result<ResponseStream> {
        let agent = self.snapshot().await;
        agent.prompt_stream(input).await
    }

    /// Replaces the LLM model at runtime.
    pub async fn swap_model<M: Model + 'static>(&self, model: M) {
        let model: SharedModel = Arc::new(model);
        self.update(move |agent| agent.model = model).await;
    }

    /// Replaces the model with a pre-boxed shared model.
    pub async fn swap_shared_model(&self, model: SharedModel) {
        self.update(move |agent| agent.model = model).await;
    }

    /// Replaces the system prompt.
    pub async fn swap_system_prompt(&self, prompt: Option<String>) {
        self.update(move |agent| agent.system_prompt = prompt).await;
    }

    /// Adds a tool to the agent's registry.
    /// Returns `true` if the tool was added (no duplicate name).
    pub async fn add_tool<T: Tool + 'static>(&self, tool: T) -> bool {
        let mut guard = self.inner.write().await;
        let mut next = clone_agent(&guard);
        let added = next.tools.register(tool).is_ok();
        if added {
            // Re-warm the spec/validator caches so post-swap prompts don't
            // pay a rebuild on their first iteration.
            next.tools.warm_cache();
            *guard = Arc::new(next);
        }
        added
    }

    /// Removes a tool by name. Returns `true` if it was present.
    pub async fn remove_tool(&self, name: &str) -> bool {
        let mut guard = self.inner.write().await;
        let mut next = clone_agent(&guard);
        let removed = next.tools.unregister(name);
        if removed {
            next.tools.warm_cache();
            *guard = Arc::new(next);
        }
        removed
    }

    /// Replaces the conversation memory backend.
    pub async fn swap_memory<M: Memory + 'static>(&self, memory: M) {
        let memory = Arc::new(memory);
        self.update(move |agent| agent.memory = memory).await;
    }

    /// Replaces the lifecycle hooks.
    pub async fn swap_hooks<H: AgentHook + 'static>(&self, hooks: H) {
        let hooks = Arc::new(hooks);
        self.update(move |agent| agent.hooks = hooks).await;
    }

    /// Replaces the entire middleware stack.
    pub async fn swap_middleware(&self, stack: crate::middleware::MiddlewareStack) {
        self.update(move |agent| agent.middleware = stack).await;
    }

    /// Adds a middleware layer to the existing stack.
    pub async fn add_middleware<M: Middleware + 'static>(&self, mw: M) {
        self.update(move |agent| agent.middleware.push(mw)).await;
    }

    /// Adds an input guardrail.
    pub async fn add_input_guardrail<G: InputGuardrail + 'static>(&self, guard: G) {
        let guard = Arc::new(guard);
        self.update(move |agent| agent.input_guardrails.push(guard))
            .await;
    }

    /// Adds an output guardrail.
    pub async fn add_output_guardrail<G: OutputGuardrail + 'static>(&self, guard: G) {
        let guard = Arc::new(guard);
        self.update(move |agent| agent.output_guardrails.push(guard))
            .await;
    }

    /// Clears all input guardrails.
    pub async fn clear_input_guardrails(&self) {
        self.update(|agent| agent.input_guardrails.clear()).await;
    }

    /// Clears all output guardrails.
    pub async fn clear_output_guardrails(&self) {
        self.update(|agent| agent.output_guardrails.clear()).await;
    }

    /// Updates the maximum number of ReAct iterations.
    pub async fn set_max_iterations(&self, max: usize) {
        self.update(move |agent| agent.max_iterations = max).await;
    }

    /// Updates the model temperature.
    pub async fn set_temperature(&self, temp: Option<f32>) {
        self.update(move |agent| agent.temperature = temp).await;
    }

    /// Updates the max tokens setting.
    pub async fn set_max_tokens(&self, tokens: Option<u32>) {
        self.update(move |agent| agent.max_tokens = tokens).await;
    }

    /// Enables or disables tool input validation.
    pub async fn set_validate_tool_inputs(&self, enabled: bool) {
        self.update(move |agent| agent.validate_tool_inputs = enabled)
            .await;
    }

    /// Updates the tool retry policy.
    pub async fn set_tool_retry_policy(&self, policy: Option<ToolRetryPolicy>) {
        self.update(move |agent| agent.tool_retry_policy = policy)
            .await;
    }

    /// Replaces the entire agent atomically.
    pub async fn replace(&self, agent: Agent) {
        let mut inner = self.inner.write().await;
        *inner = Arc::new(agent);
    }

    /// Returns a snapshot of the current system prompt.
    pub async fn system_prompt(&self) -> Option<String> {
        self.snapshot().await.system_prompt.clone()
    }

    /// Returns the number of registered tools.
    pub async fn tool_count(&self) -> usize {
        self.snapshot().await.tools.len()
    }

    /// Returns the names of all registered tools.
    pub async fn tool_names(&self) -> Vec<String> {
        self.snapshot()
            .await
            .tools
            .list()
            .into_iter()
            .map(String::from)
            .collect()
    }
}

impl Clone for HotSwapAgent {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result as DResult;
    use crate::model::Model;
    use crate::model::types::*;
    use crate::stream::ResponseStream;
    use crate::tool::ToolOutput;

    struct ModelA;
    struct ModelB;

    impl Model for ModelA {
        async fn generate(&self, _request: &ChatRequest) -> DResult<ChatResponse> {
            Ok(ChatResponse {
                message: Message::assistant("from-A"),
                stop_reason: StopReason::EndTurn,
                usage: Some(Usage::default()),
            })
        }
        async fn generate_stream(&self, _request: &ChatRequest) -> DResult<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    impl Model for ModelB {
        async fn generate(&self, _request: &ChatRequest) -> DResult<ChatResponse> {
            Ok(ChatResponse {
                message: Message::assistant("from-B"),
                stop_reason: StopReason::EndTurn,
                usage: Some(Usage::default()),
            })
        }
        async fn generate_stream(&self, _request: &ChatRequest) -> DResult<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    struct TestTool {
        tool_name: String,
    }

    impl crate::tool::Tool for TestTool {
        fn name(&self) -> &str {
            &self.tool_name
        }
        fn description(&self) -> &str {
            "test"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(&self, _input: &serde_json::Value) -> DResult<ToolOutput> {
            Ok(ToolOutput::text("ok"))
        }
    }

    #[tokio::test]
    async fn test_swap_model() {
        let agent = Agent::builder().model(ModelA).build().unwrap();
        let hot = HotSwapAgent::new(agent);

        let r1 = hot.prompt("hi").await.unwrap();
        assert_eq!(r1.final_text, "from-A");

        hot.swap_model(ModelB).await;

        let r2 = hot.prompt("hi").await.unwrap();
        assert_eq!(r2.final_text, "from-B");
    }

    #[tokio::test]
    async fn test_swap_system_prompt() {
        let agent = Agent::builder()
            .model(ModelA)
            .system_prompt("original")
            .build()
            .unwrap();
        let hot = HotSwapAgent::new(agent);

        assert_eq!(hot.system_prompt().await.as_deref(), Some("original"));

        hot.swap_system_prompt(Some("updated".into())).await;
        assert_eq!(hot.system_prompt().await.as_deref(), Some("updated"));

        hot.swap_system_prompt(None).await;
        assert!(hot.system_prompt().await.is_none());
    }

    #[tokio::test]
    async fn test_add_remove_tools() {
        let agent = Agent::builder().model(ModelA).build().unwrap();
        let hot = HotSwapAgent::new(agent);

        assert_eq!(hot.tool_count().await, 0);

        hot.add_tool(TestTool {
            tool_name: "alpha".into(),
        })
        .await;
        assert_eq!(hot.tool_count().await, 1);

        hot.add_tool(TestTool {
            tool_name: "beta".into(),
        })
        .await;
        assert_eq!(hot.tool_count().await, 2);

        assert!(hot.remove_tool("alpha").await);
        assert_eq!(hot.tool_count().await, 1);

        assert!(!hot.remove_tool("nonexistent").await);
    }

    #[tokio::test]
    async fn test_set_iterations_and_temp() {
        let agent = Agent::builder().model(ModelA).build().unwrap();
        let hot = HotSwapAgent::new(agent);

        hot.set_max_iterations(10).await;
        hot.set_temperature(Some(0.5)).await;
        hot.set_max_tokens(Some(100)).await;

        let inner = hot.inner.read().await;
        assert_eq!(inner.max_iterations, 10);
        assert_eq!(inner.temperature, Some(0.5));
        assert_eq!(inner.max_tokens, Some(100));
    }

    #[tokio::test]
    async fn test_clone_shares_state() {
        let agent = Agent::builder()
            .model(ModelA)
            .system_prompt("shared")
            .build()
            .unwrap();
        let hot = HotSwapAgent::new(agent);
        let clone = hot.clone();

        clone.swap_system_prompt(Some("from-clone".into())).await;
        assert_eq!(hot.system_prompt().await.as_deref(), Some("from-clone"));
    }

    #[tokio::test]
    async fn test_replace_agent() {
        let agent_a = Agent::builder()
            .model(ModelA)
            .system_prompt("A")
            .build()
            .unwrap();
        let hot = HotSwapAgent::new(agent_a);

        assert_eq!(hot.system_prompt().await.as_deref(), Some("A"));

        let agent_b = Agent::builder()
            .model(ModelB)
            .system_prompt("B")
            .build()
            .unwrap();
        hot.replace(agent_b).await;

        assert_eq!(hot.system_prompt().await.as_deref(), Some("B"));
        let r = hot.prompt("test").await.unwrap();
        assert_eq!(r.final_text, "from-B");
    }

    /// A model that signals when `generate` has started and then blocks until
    /// released, so tests can deterministically hold a prompt "in flight".
    struct GatedModel {
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }

    impl Model for GatedModel {
        async fn generate(&self, _request: &ChatRequest) -> DResult<ChatResponse> {
            self.started.notify_one();
            self.release.notified().await;
            Ok(ChatResponse {
                message: Message::assistant("from-gated"),
                stop_reason: StopReason::EndTurn,
                usage: Some(Usage::default()),
            })
        }
        async fn generate_stream(&self, _request: &ChatRequest) -> DResult<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    #[tokio::test]
    async fn test_swap_does_not_block_on_inflight_prompt() {
        let started = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());

        let agent = Agent::builder()
            .model(GatedModel {
                started: Arc::clone(&started),
                release: Arc::clone(&release),
            })
            .build()
            .unwrap();
        let hot = HotSwapAgent::new(agent);

        // Start a prompt and wait until it is inside the model call.
        let hot_for_prompt = hot.clone();
        let inflight = tokio::spawn(async move { hot_for_prompt.prompt("hi").await });
        started.notified().await;

        // The swap must complete promptly even though a prompt is mid-flight.
        tokio::time::timeout(std::time::Duration::from_secs(1), hot.swap_model(ModelB))
            .await
            .expect("swap must not block on an in-flight prompt");

        // A new prompt sees the new model while the old one is still pending.
        let fresh = tokio::time::timeout(std::time::Duration::from_secs(1), hot.prompt("hi"))
            .await
            .expect("new prompt must not block behind the in-flight one")
            .unwrap();
        assert_eq!(fresh.final_text, "from-B");

        // The in-flight prompt completes on the OLD configuration.
        release.notify_one();
        let old = inflight.await.unwrap().unwrap();
        assert_eq!(old.final_text, "from-gated");
    }

    #[tokio::test]
    async fn test_add_tool_after_swap_serves_warm_specs() {
        let agent = Agent::builder().model(ModelA).build().unwrap();
        let hot = HotSwapAgent::new(agent);

        assert!(
            hot.add_tool(TestTool {
                tool_name: "alpha".into(),
            })
            .await
        );

        // The swapped-in agent must have a pre-warmed spec cache: repeated
        // calls return the same allocation instead of recomputing per call.
        let agent = hot.snapshot().await;
        let first = agent.tools.tool_specs();
        let second = agent.tools.tool_specs();
        assert!(std::sync::Arc::ptr_eq(&first, &second));
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].name, "alpha");
    }

    #[tokio::test]
    async fn test_tool_names() {
        let agent = Agent::builder().model(ModelA).build().unwrap();
        let hot = HotSwapAgent::new(agent);

        hot.add_tool(TestTool {
            tool_name: "foo".into(),
        })
        .await;
        hot.add_tool(TestTool {
            tool_name: "bar".into(),
        })
        .await;

        let mut names = hot.tool_names().await;
        names.sort();
        assert_eq!(names, vec!["bar", "foo"]);
    }
}
