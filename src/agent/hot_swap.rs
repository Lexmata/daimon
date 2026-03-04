//! Hot-reloadable agent wrapper.
//!
//! [`HotSwapAgent`] wraps an [`Agent`] behind a `RwLock`, allowing you to
//! swap the model, tools, system prompt, or other configuration at runtime
//! without restarting or rebuilding the agent.
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
/// All prompt operations acquire a read lock, so they can run concurrently.
/// Swap operations acquire a write lock, blocking only during the brief
/// pointer swap. Ongoing prompts complete with their original configuration.
pub struct HotSwapAgent {
    inner: Arc<RwLock<Agent>>,
}

impl HotSwapAgent {
    /// Wraps an existing agent for hot-reload support.
    pub fn new(agent: Agent) -> Self {
        Self {
            inner: Arc::new(RwLock::new(agent)),
        }
    }

    /// Runs a prompt through the agent.
    pub async fn prompt(&self, input: &str) -> Result<AgentResponse> {
        let agent = self.inner.read().await;
        agent.prompt(input).await
    }

    /// Runs a streaming prompt through the agent.
    pub async fn prompt_stream(&self, input: &str) -> Result<ResponseStream> {
        let agent = self.inner.read().await;
        agent.prompt_stream(input).await
    }

    /// Replaces the LLM model at runtime.
    pub async fn swap_model<M: Model + 'static>(&self, model: M) {
        let mut agent = self.inner.write().await;
        agent.model = Arc::new(model);
    }

    /// Replaces the model with a pre-boxed shared model.
    pub async fn swap_shared_model(&self, model: SharedModel) {
        let mut agent = self.inner.write().await;
        agent.model = model;
    }

    /// Replaces the system prompt.
    pub async fn swap_system_prompt(&self, prompt: Option<String>) {
        let mut agent = self.inner.write().await;
        agent.system_prompt = prompt;
    }

    /// Adds a tool to the agent's registry.
    /// Returns `true` if the tool was added (no duplicate name).
    pub async fn add_tool<T: Tool + 'static>(&self, tool: T) -> bool {
        let mut agent = self.inner.write().await;
        agent.tools.register(tool).is_ok()
    }

    /// Removes a tool by name. Returns `true` if it was present.
    pub async fn remove_tool(&self, name: &str) -> bool {
        let mut agent = self.inner.write().await;
        agent.tools.unregister(name)
    }

    /// Replaces the conversation memory backend.
    pub async fn swap_memory<M: Memory + 'static>(&self, memory: M) {
        let mut agent = self.inner.write().await;
        agent.memory = Arc::new(memory);
    }

    /// Replaces the lifecycle hooks.
    pub async fn swap_hooks<H: AgentHook + 'static>(&self, hooks: H) {
        let mut agent = self.inner.write().await;
        agent.hooks = Arc::new(hooks);
    }

    /// Replaces the entire middleware stack.
    pub async fn swap_middleware(&self, stack: crate::middleware::MiddlewareStack) {
        let mut agent = self.inner.write().await;
        agent.middleware = stack;
    }

    /// Adds a middleware layer to the existing stack.
    pub async fn add_middleware<M: Middleware + 'static>(&self, mw: M) {
        let mut agent = self.inner.write().await;
        agent.middleware.push(mw);
    }

    /// Adds an input guardrail.
    pub async fn add_input_guardrail<G: InputGuardrail + 'static>(&self, guard: G) {
        let mut agent = self.inner.write().await;
        agent.input_guardrails.push(Arc::new(guard));
    }

    /// Adds an output guardrail.
    pub async fn add_output_guardrail<G: OutputGuardrail + 'static>(&self, guard: G) {
        let mut agent = self.inner.write().await;
        agent.output_guardrails.push(Arc::new(guard));
    }

    /// Clears all input guardrails.
    pub async fn clear_input_guardrails(&self) {
        let mut agent = self.inner.write().await;
        agent.input_guardrails.clear();
    }

    /// Clears all output guardrails.
    pub async fn clear_output_guardrails(&self) {
        let mut agent = self.inner.write().await;
        agent.output_guardrails.clear();
    }

    /// Updates the maximum number of ReAct iterations.
    pub async fn set_max_iterations(&self, max: usize) {
        let mut agent = self.inner.write().await;
        agent.max_iterations = max;
    }

    /// Updates the model temperature.
    pub async fn set_temperature(&self, temp: Option<f32>) {
        let mut agent = self.inner.write().await;
        agent.temperature = temp;
    }

    /// Updates the max tokens setting.
    pub async fn set_max_tokens(&self, tokens: Option<u32>) {
        let mut agent = self.inner.write().await;
        agent.max_tokens = tokens;
    }

    /// Enables or disables tool input validation.
    pub async fn set_validate_tool_inputs(&self, enabled: bool) {
        let mut agent = self.inner.write().await;
        agent.validate_tool_inputs = enabled;
    }

    /// Updates the tool retry policy.
    pub async fn set_tool_retry_policy(&self, policy: Option<ToolRetryPolicy>) {
        let mut agent = self.inner.write().await;
        agent.tool_retry_policy = policy;
    }

    /// Replaces the entire agent atomically.
    pub async fn replace(&self, agent: Agent) {
        let mut inner = self.inner.write().await;
        *inner = agent;
    }

    /// Returns a snapshot of the current system prompt.
    pub async fn system_prompt(&self) -> Option<String> {
        let agent = self.inner.read().await;
        agent.system_prompt.clone()
    }

    /// Returns the number of registered tools.
    pub async fn tool_count(&self) -> usize {
        let agent = self.inner.read().await;
        agent.tools.len()
    }

    /// Returns the names of all registered tools.
    pub async fn tool_names(&self) -> Vec<String> {
        let agent = self.inner.read().await;
        agent.tools.list().into_iter().map(String::from).collect()
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

        clone
            .swap_system_prompt(Some("from-clone".into()))
            .await;
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
