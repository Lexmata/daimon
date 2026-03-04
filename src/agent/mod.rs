//! Agent construction and ReAct loop execution.
//!
//! Build an agent with [`Agent::builder()`], configure model, tools, memory, and hooks,
//! then call [`Agent::prompt`] or [`Agent::prompt_stream`] to run the ReAct loop.
//!
//! ## Multi-Agent Patterns
//!
//! - [`as_tool::AgentTool`] — wrap an agent as a tool for another agent
//! - [`supervisor::Supervisor`] — one agent delegates to specialized sub-agents
//! - [`handoff::HandoffNetwork`] — agents transfer control to each other
//! - [`structured::StructuredOutput`] — extract typed data from LLM responses
//! - [`resumable`] — checkpoint-based resumable agent runs

pub mod as_tool;
mod builder;
pub mod fork;
pub mod handoff;
pub mod hitl;
pub mod resumable;
mod runner;
pub mod structured;
pub mod supervisor;

pub use builder::AgentBuilder;
pub use runner::AgentResponse;

use std::sync::Arc;

use crate::cost::CostTracker;
use crate::guardrails::{ErasedInputGuardrail, ErasedOutputGuardrail};
use crate::hooks::ErasedAgentHook;
use crate::memory::SharedMemory;
use crate::middleware::MiddlewareStack;
use crate::model::SharedModel;
use crate::tool::{ToolRegistry, ToolRetryPolicy};

/// An AI agent that runs the ReAct loop: model → tool calls (optional) → model → … → final response.
///
/// Construct via [`Agent::builder()`]. Requires a [`Model`](crate::model::Model); tools, memory,
/// and hooks are optional. Memory defaults to [`SlidingWindowMemory`](crate::memory::SlidingWindowMemory) with 50 messages.
pub struct Agent {
    pub(crate) model: SharedModel,
    pub(crate) system_prompt: Option<String>,
    pub(crate) tools: ToolRegistry,
    pub(crate) memory: SharedMemory,
    pub(crate) hooks: Arc<dyn ErasedAgentHook>,
    pub(crate) middleware: MiddlewareStack,
    pub(crate) input_guardrails: Vec<Arc<dyn ErasedInputGuardrail>>,
    pub(crate) output_guardrails: Vec<Arc<dyn ErasedOutputGuardrail>>,
    pub(crate) max_iterations: usize,
    pub(crate) temperature: Option<f32>,
    pub(crate) max_tokens: Option<u32>,
    pub(crate) validate_tool_inputs: bool,
    pub(crate) cost_tracker: Option<CostTracker>,
    pub(crate) max_budget: Option<f64>,
    pub(crate) tool_retry_policy: Option<ToolRetryPolicy>,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("system_prompt", &self.system_prompt)
            .field("max_iterations", &self.max_iterations)
            .field("temperature", &self.temperature)
            .field("max_tokens", &self.max_tokens)
            .field("tools_count", &self.tools.len())
            .field("validate_tool_inputs", &self.validate_tool_inputs)
            .finish_non_exhaustive()
    }
}

impl Agent {
    /// Returns a new builder for configuring and constructing an agent.
    pub fn builder() -> AgentBuilder {
        AgentBuilder::new()
    }

    /// Returns the agent's conversation memory. Use this to inspect or export message history.
    pub fn memory(&self) -> &SharedMemory {
        &self.memory
    }
}
