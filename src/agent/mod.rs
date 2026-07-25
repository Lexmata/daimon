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
pub mod hot_swap;
pub mod resumable;
mod runner;
pub mod structured;
pub mod supervisor;

pub use builder::AgentBuilder;
pub use runner::AgentResponse;

use std::sync::Arc;

use crate::cost::CostTracker;
use crate::error::{DaimonError, Result};
use crate::guardrails::{ErasedInputGuardrail, ErasedOutputGuardrail};
use crate::hooks::ErasedAgentHook;
use crate::memory::SharedMemory;
use crate::middleware::MiddlewareStack;
use crate::model::SharedModel;
use crate::model::types::{ChatRequest, ChatResponse};
use crate::routing::{ModelRouter, RouteDecision};
use crate::tool::{ToolRegistry, ToolRetryPolicy};

/// What serves model calls for this agent: one fixed model, or a router
/// choosing per call among registered models.
#[derive(Clone)]
pub(crate) enum ModelSource {
    Single(SharedModel),
    Routed(ModelRouter),
}

/// An AI agent that runs the ReAct loop: model → tool calls (optional) → model → … → final response.
///
/// Construct via [`Agent::builder()`]. Requires a [`Model`](crate::model::Model); tools, memory,
/// and hooks are optional. Memory defaults to [`SlidingWindowMemory`](crate::memory::SlidingWindowMemory) with 50 messages.
pub struct Agent {
    pub(crate) model: ModelSource,
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

/// Outcome of one model call: the response, the id of the model that
/// actually served it, and the routing decisions made along the way
/// (empty for single-model agents).
pub(crate) struct GenerationOutcome {
    pub response: ChatResponse,
    pub serving_model_id: String,
    pub decisions: Vec<RouteDecision>,
}

impl Agent {
    /// Selects the model for one call and generates, applying routing and
    /// tier escalation when the agent has a router. Single-model agents call
    /// their model directly, exactly as before.
    pub(crate) async fn generate_routed(
        &self,
        iteration: usize,
        request: &ChatRequest,
    ) -> Result<GenerationOutcome> {
        match &self.model {
            ModelSource::Single(model) => {
                let response = model.generate_erased(request).await?;
                Ok(GenerationOutcome {
                    response,
                    serving_model_id: model.model_id_erased().to_string(),
                    decisions: Vec::new(),
                })
            }
            ModelSource::Routed(router) => {
                let mut routed = router.route(iteration, request).await?;
                let mut decisions = Vec::new();
                loop {
                    match routed.handle.generate_erased(request).await {
                        Ok(response) => {
                            let serving_model_id = routed.decision.selected_model_id.clone();
                            decisions.push(routed.decision);
                            return Ok(GenerationOutcome {
                                response,
                                serving_model_id,
                                decisions,
                            });
                        }
                        Err(DaimonError::Cancelled) => return Err(DaimonError::Cancelled),
                        Err(e) => {
                            decisions.push(routed.decision);
                            match router.escalate(decisions.last().expect("decision just pushed")) {
                                Some(next) => {
                                    tracing::warn!(
                                        error = %e,
                                        to_tier = ?next.decision.selected_tier,
                                        to_model = %next.decision.selected_model_id,
                                        "model call failed; escalating tier"
                                    );
                                    routed = next;
                                }
                                None => return Err(e),
                            }
                        }
                    }
                }
            }
        }
    }
}
