//! Handoff pattern: agents transfer control to each other mid-conversation.
//!
//! Inspired by OpenAI Swarm, a [`HandoffNetwork`] manages multiple agents that
//! can hand off the conversation to one another via synthetic `transfer_to_<name>`
//! tools. The network runs its own agent loop, detecting handoff signals and
//! switching the active agent while preserving conversation context.
//!
//! ```ignore
//! use daimon::agent::handoff::HandoffNetwork;
//!
//! let network = HandoffNetwork::builder()
//!     .entry("triage")
//!     .agent("triage", triage_agent)
//!     .agent("billing", billing_agent)
//!     .agent("support", support_agent)
//!     .build()?;
//!
//! let response = network.run("I need help with my bill").await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use crate::agent::Agent;
use crate::agent::runner::{GuardrailDecision, budget_exceeded};
use crate::cost::CostTracker;
use crate::error::{DaimonError, Result};
use crate::model::types::{ChatRequest, Message, Role, ToolSpec, Usage};
use crate::tool::ToolCall;

const HANDOFF_PREFIX: &str = "transfer_to_";

/// A network of agents that can hand off conversations to each other.
pub struct HandoffNetwork {
    agents: HashMap<String, Arc<Agent>>,
    entry: String,
    max_handoffs: usize,
    max_iterations_per_agent: usize,
}

impl HandoffNetwork {
    /// Returns a new builder.
    pub fn builder() -> HandoffBuilder {
        HandoffBuilder::new()
    }

    /// Runs the handoff network with the given user input.
    ///
    /// Starts with the entry agent, processes the conversation, and follows
    /// handoffs until an agent produces a final text response or limits are
    /// reached.
    ///
    /// Each agent turn goes through that agent's own execution pipeline as far
    /// as the shared conversation allows:
    ///
    /// - the entry agent's input guardrails run on the user input (a blocked
    ///   input aborts the run);
    /// - the active agent's output guardrails run on any final text before the
    ///   network returns it (`Transform` applies, `Block` aborts);
    /// - each agent's `max_budget` is enforced across its own turns via its
    ///   cost model;
    /// - genuine tool calls run through the same middleware/validation/retry
    ///   path as [`Agent::prompt`].
    ///
    /// A registered tool whose name starts with `transfer_to_` always shadows
    /// the synthetic transfer tool of the same name. A transfer to an unknown
    /// agent surfaces as a tool error the model can react to. When one response
    /// contains several transfer calls, the first is honored and the rest are
    /// answered with an error.
    #[tracing::instrument(skip_all, fields(entry = %self.entry))]
    pub async fn run(&self, input: &str) -> Result<HandoffResponse> {
        let mut current_agent_name = self.entry.clone();
        let mut handoffs = 0usize;
        let mut total_usage = Usage::default();
        let mut total_iterations = 0usize;

        let mut messages: Vec<Message> = Vec::new();

        let entry_agent = self.agents.get(&current_agent_name).ok_or_else(|| {
            DaimonError::Orchestration(format!("entry agent '{}' not found", self.entry))
        })?;

        // The entry agent's input guardrails vet the user input exactly as a
        // direct `prompt()` on that agent would; a Block aborts the run.
        let checked_input = entry_agent.run_input_guardrails(input).await?;

        if let Some(system) = &entry_agent.system_prompt {
            messages.push(Message::system(system));
        }
        messages.push(Message::user(&checked_input));

        // Per-agent cost trackers: each agent's max_budget bounds that agent's
        // own spend across all of its turns in this run.
        let mut trackers: HashMap<String, Option<CostTracker>> = HashMap::new();

        'network: loop {
            let agent = self.agents.get(&current_agent_name).ok_or_else(|| {
                DaimonError::Orchestration(format!("agent '{current_agent_name}' not found"))
            })?;

            let tracker = trackers
                .entry(current_agent_name.clone())
                .or_insert_with(|| agent.new_run_tracker())
                .as_ref();

            let transfer_tools = self.build_transfer_tools(&current_agent_name, agent);
            let mut agent_tool_specs: Vec<ToolSpec> = agent.tools.tool_specs().to_vec();
            agent_tool_specs.extend(transfer_tools);

            let mut agent_iterations = 0usize;

            loop {
                agent_iterations += 1;
                total_iterations += 1;

                if agent_iterations > self.max_iterations_per_agent {
                    return Err(DaimonError::MaxIterations(self.max_iterations_per_agent));
                }

                if let Some((spent, limit)) = budget_exceeded(tracker, agent.max_budget) {
                    return Err(DaimonError::BudgetExceeded { spent, limit });
                }

                let request = ChatRequest {
                    messages: messages.clone(),
                    tools: agent_tool_specs.clone(),
                    temperature: agent.temperature,
                    max_tokens: agent.max_tokens,
                };

                let mut response = agent.model.generate_erased(&request).await?;

                if let Some(ref usage) = response.usage {
                    total_usage.accumulate(usage);
                    if let Some(tracker) = tracker {
                        tracker.record("default", usage);
                    }
                }

                if !response.has_tool_calls() {
                    // The active agent's output guardrails vet the final text
                    // before the network returns it, mirroring `prompt()`.
                    let mut final_text = response.text().to_string();
                    match agent
                        .evaluate_output_guardrails(&final_text, &total_usage)
                        .await?
                    {
                        GuardrailDecision::Pass => {}
                        GuardrailDecision::Block(msg) => {
                            return Err(DaimonError::GuardrailBlocked(msg));
                        }
                        GuardrailDecision::Transform(new_text) => {
                            response.message.content = Some(new_text.clone());
                            final_text = new_text;
                        }
                    }
                    messages.push(response.message);

                    return Ok(HandoffResponse {
                        messages,
                        final_text,
                        final_agent: current_agent_name,
                        handoff_count: handoffs,
                        iterations: total_iterations,
                        usage: total_usage,
                    });
                }

                let tool_calls = response.tool_calls().to_vec();
                // Preserve any assistant text emitted alongside the tool calls.
                let assistant_msg = match response.message.content.take() {
                    Some(text) if !text.is_empty() => {
                        Message::assistant_with_text_and_tool_calls(text, tool_calls.clone())
                    }
                    _ => Message::assistant_with_tool_calls(tool_calls.clone()),
                };
                messages.push(assistant_msg);

                let (handoff_target, result_contents) =
                    self.dispatch_tool_calls(agent, &tool_calls).await;

                for (call, content) in tool_calls.iter().zip(result_contents) {
                    messages.push(Message::tool_result(&call.id, content));
                }

                if let Some(target) = handoff_target {
                    handoffs += 1;
                    if handoffs > self.max_handoffs {
                        return Err(DaimonError::Orchestration(format!(
                            "max handoffs ({}) exceeded",
                            self.max_handoffs
                        )));
                    }

                    // The previous agent's system message never carries over:
                    // it is removed even when the new agent has no system
                    // prompt of its own.
                    messages.retain(|m| m.role != Role::System);
                    if let Some(new_agent) = self.agents.get(&target)
                        && let Some(system) = &new_agent.system_prompt
                    {
                        messages.insert(0, Message::system(system));
                    }

                    tracing::info!(
                        from = %current_agent_name,
                        to = %target,
                        "agent handoff"
                    );
                    current_agent_name = target;
                    continue 'network;
                }
            }
        }
    }

    /// Executes one response's tool calls, returning the accepted handoff
    /// target (if any) and one result string per call, in call order.
    ///
    /// Transfer calls are resolved here; genuine tool calls — including any
    /// call shadowing a `transfer_to_` name with a registered tool — run
    /// through [`Agent::execute_tools_parallel`], the same
    /// middleware/validation/retry path the ReAct runner uses.
    async fn dispatch_tool_calls(
        &self,
        agent: &Agent,
        tool_calls: &[ToolCall],
    ) -> (Option<String>, Vec<String>) {
        let mut results: Vec<Option<String>> = vec![None; tool_calls.len()];
        let mut regular: Vec<(usize, ToolCall)> = Vec::new();
        let mut handoff_target: Option<String> = None;
        let mut transfer_seen = false;

        for (idx, call) in tool_calls.iter().enumerate() {
            let target = call.name.strip_prefix(HANDOFF_PREFIX);
            // A registered tool always wins over the synthetic transfer tool.
            let is_transfer = target.is_some() && agent.tools.get(&call.name).is_none();
            if !is_transfer {
                regular.push((idx, call.clone()));
                continue;
            }

            // strip_prefix returned Some to classify this call as a transfer.
            let Some(target) = target else { continue };

            if transfer_seen {
                // Only the first transfer in a response is honored.
                results[idx] = Some(format!(
                    "Error: a transfer has already been requested in this turn; \
                     ignoring transfer to '{target}'"
                ));
                continue;
            }
            transfer_seen = true;

            if self.agents.contains_key(target) {
                let reason = call
                    .arguments
                    .get("reason")
                    .and_then(|v| v.as_str())
                    .unwrap_or("transferring");
                results[idx] = Some(format!("Transferring to {target}. Reason: {reason}"));
                handoff_target = Some(target.to_string());
            } else {
                // Hallucinated target: report it to the model instead of
                // killing the run, so it can pick a real agent or answer.
                results[idx] = Some(format!(
                    "Error: cannot transfer: agent '{target}' does not exist"
                ));
            }
        }

        if !regular.is_empty() {
            let calls: Vec<ToolCall> = regular.iter().map(|(_, c)| c.clone()).collect();
            let outputs = agent.execute_tools_parallel(&calls).await;
            for ((idx, _), output) in regular.iter().zip(outputs) {
                results[*idx] = Some(output.content);
            }
        }

        let contents = results
            .into_iter()
            .map(|r| r.unwrap_or_else(|| "tool call produced no result".to_string()))
            .collect();

        (handoff_target, contents)
    }

    fn build_transfer_tools(&self, current: &str, agent: &Agent) -> Vec<ToolSpec> {
        self.agents
            .keys()
            .filter(|name| *name != current)
            // Don't advertise a synthetic transfer tool whose name collides
            // with one of the agent's real tools — the real tool wins.
            .filter(|name| {
                agent
                    .tools
                    .get(&format!("{HANDOFF_PREFIX}{name}"))
                    .is_none()
            })
            .map(|name| ToolSpec {
                name: format!("{HANDOFF_PREFIX}{name}"),
                description: format!("Transfer the conversation to the '{name}' agent"),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Why the conversation is being transferred"
                        }
                    },
                    "required": ["reason"]
                }),
            })
            .collect()
    }
}

/// Response from a handoff network run.
#[derive(Debug, Clone)]
pub struct HandoffResponse {
    /// Full message history.
    pub messages: Vec<Message>,
    /// Final text output.
    pub final_text: String,
    /// Name of the agent that produced the final response.
    pub final_agent: String,
    /// How many handoffs occurred.
    pub handoff_count: usize,
    /// Total model iterations across all agents.
    pub iterations: usize,
    /// Aggregated token usage.
    pub usage: Usage,
}

impl HandoffResponse {
    /// Get the final response text.
    pub fn text(&self) -> &str {
        &self.final_text
    }
}

/// Builder for constructing a [`HandoffNetwork`].
pub struct HandoffBuilder {
    agents: HashMap<String, Arc<Agent>>,
    entry: Option<String>,
    max_handoffs: usize,
    max_iterations_per_agent: usize,
}

impl HandoffBuilder {
    fn new() -> Self {
        Self {
            agents: HashMap::new(),
            entry: None,
            max_handoffs: 10,
            max_iterations_per_agent: 25,
        }
    }

    /// Registers an agent in the network.
    pub fn agent(mut self, name: impl Into<String>, agent: Arc<Agent>) -> Self {
        self.agents.insert(name.into(), agent);
        self
    }

    /// Sets the entry agent (the first to receive the user's input).
    pub fn entry(mut self, name: impl Into<String>) -> Self {
        self.entry = Some(name.into());
        self
    }

    /// Sets the maximum number of handoffs allowed. Default: 10.
    pub fn max_handoffs(mut self, max: usize) -> Self {
        self.max_handoffs = max;
        self
    }

    /// Sets the max iterations each agent gets before the loop aborts. Default: 25.
    pub fn max_iterations_per_agent(mut self, max: usize) -> Self {
        self.max_iterations_per_agent = max;
        self
    }

    /// Builds the handoff network.
    pub fn build(self) -> Result<HandoffNetwork> {
        let entry = self.entry.ok_or_else(|| {
            DaimonError::Builder("handoff network requires an entry agent".into())
        })?;

        if !self.agents.contains_key(&entry) {
            return Err(DaimonError::Builder(format!(
                "entry agent '{entry}' not found in registered agents"
            )));
        }

        if self.agents.len() < 2 {
            return Err(DaimonError::Builder(
                "handoff network requires at least two agents".into(),
            ));
        }

        Ok(HandoffNetwork {
            agents: self.agents,
            entry,
            max_handoffs: self.max_handoffs,
            max_iterations_per_agent: self.max_iterations_per_agent,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Model;
    use crate::model::types::*;
    use crate::stream::ResponseStream;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct DirectResponseModel {
        response: String,
    }

    impl DirectResponseModel {
        fn new(response: &str) -> Self {
            Self {
                response: response.to_string(),
            }
        }
    }

    impl Model for DirectResponseModel {
        async fn generate(&self, _request: &ChatRequest) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message::assistant(&self.response),
                stop_reason: StopReason::EndTurn,
                usage: Some(Usage::default()),
            })
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    struct HandoffModel {
        target: String,
        call_count: AtomicUsize,
    }

    impl HandoffModel {
        fn new(target: &str) -> Self {
            Self {
                target: target.to_string(),
                call_count: AtomicUsize::new(0),
            }
        }
    }

    impl Model for HandoffModel {
        async fn generate(&self, _request: &ChatRequest) -> Result<ChatResponse> {
            let count = self.call_count.fetch_add(1, Ordering::SeqCst);
            if count == 0 {
                Ok(ChatResponse {
                    message: Message::assistant_with_tool_calls(vec![crate::tool::ToolCall {
                        id: "transfer_1".into(),
                        name: format!("{HANDOFF_PREFIX}{}", self.target),
                        arguments: serde_json::json!({"reason": "user needs billing help"}),
                    }]),
                    stop_reason: StopReason::ToolUse,
                    usage: Some(Usage::default()),
                })
            } else {
                Ok(ChatResponse {
                    message: Message::assistant("Triage done but model called again"),
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
    fn test_builder_requires_entry() {
        let a = Arc::new(
            Agent::builder()
                .model(DirectResponseModel::new("a"))
                .build()
                .unwrap(),
        );
        let b = Arc::new(
            Agent::builder()
                .model(DirectResponseModel::new("b"))
                .build()
                .unwrap(),
        );
        let result = HandoffNetwork::builder()
            .agent("a", a)
            .agent("b", b)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_requires_two_agents() {
        let a = Arc::new(
            Agent::builder()
                .model(DirectResponseModel::new("a"))
                .build()
                .unwrap(),
        );
        let result = HandoffNetwork::builder().entry("a").agent("a", a).build();
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_direct_response_no_handoff() {
        let a = Arc::new(
            Agent::builder()
                .model(DirectResponseModel::new("I can help!"))
                .build()
                .unwrap(),
        );
        let b = Arc::new(
            Agent::builder()
                .model(DirectResponseModel::new("billing help"))
                .build()
                .unwrap(),
        );

        let network = HandoffNetwork::builder()
            .entry("triage")
            .agent("triage", a)
            .agent("billing", b)
            .build()
            .unwrap();

        let response = network.run("hello").await.unwrap();
        assert_eq!(response.text(), "I can help!");
        assert_eq!(response.final_agent, "triage");
        assert_eq!(response.handoff_count, 0);
    }

    #[tokio::test]
    async fn test_handoff_transfers_to_target() {
        let triage = Arc::new(
            Agent::builder()
                .model(HandoffModel::new("billing"))
                .build()
                .unwrap(),
        );
        let billing = Arc::new(
            Agent::builder()
                .model(DirectResponseModel::new("Here is your bill info"))
                .build()
                .unwrap(),
        );

        let network = HandoffNetwork::builder()
            .entry("triage")
            .agent("triage", triage)
            .agent("billing", billing)
            .build()
            .unwrap();

        let response = network.run("billing question").await.unwrap();
        assert_eq!(response.text(), "Here is your bill info");
        assert_eq!(response.final_agent, "billing");
        assert_eq!(response.handoff_count, 1);
    }

    struct AlwaysHandoffModel {
        target: String,
    }

    impl AlwaysHandoffModel {
        fn new(target: &str) -> Self {
            Self {
                target: target.to_string(),
            }
        }
    }

    impl Model for AlwaysHandoffModel {
        async fn generate(&self, _request: &ChatRequest) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message::assistant_with_tool_calls(vec![crate::tool::ToolCall {
                    id: "t".into(),
                    name: format!("{HANDOFF_PREFIX}{}", self.target),
                    arguments: serde_json::json!({"reason": "bounce"}),
                }]),
                stop_reason: StopReason::ToolUse,
                usage: Some(Usage::default()),
            })
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    #[tokio::test]
    async fn test_max_handoffs_exceeded() {
        let a = Arc::new(
            Agent::builder()
                .model(AlwaysHandoffModel::new("b"))
                .build()
                .unwrap(),
        );
        let b = Arc::new(
            Agent::builder()
                .model(AlwaysHandoffModel::new("a"))
                .build()
                .unwrap(),
        );

        let network = HandoffNetwork::builder()
            .entry("a")
            .agent("a", a)
            .agent("b", b)
            .max_handoffs(3)
            .build()
            .unwrap();

        let result = network.run("ping pong").await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("max handoffs"), "got: {err}");
    }

    // ---- Pipeline-parity fixes (DAIM-8) ----

    struct BlockingInputGuardrail;

    impl crate::guardrails::InputGuardrail for BlockingInputGuardrail {
        async fn check(
            &self,
            _input: &str,
            _messages: &[Message],
        ) -> Result<crate::guardrails::GuardrailResult> {
            Ok(crate::guardrails::GuardrailResult::Block("nope".into()))
        }
    }

    struct TransformOutputGuardrail;

    impl crate::guardrails::OutputGuardrail for TransformOutputGuardrail {
        async fn check(
            &self,
            _response: &ChatResponse,
        ) -> Result<crate::guardrails::GuardrailResult> {
            Ok(crate::guardrails::GuardrailResult::Transform(
                "REPLACED".into(),
            ))
        }
    }

    #[tokio::test]
    async fn test_entry_input_guardrail_blocks_run() {
        let a = Arc::new(
            Agent::builder()
                .model(DirectResponseModel::new("hi"))
                .input_guardrail(BlockingInputGuardrail)
                .build()
                .unwrap(),
        );
        let b = Arc::new(
            Agent::builder()
                .model(DirectResponseModel::new("b"))
                .build()
                .unwrap(),
        );

        let network = HandoffNetwork::builder()
            .entry("a")
            .agent("a", a)
            .agent("b", b)
            .build()
            .unwrap();

        let result = network.run("blocked input").await;
        assert!(matches!(
            result.err(),
            Some(DaimonError::GuardrailBlocked(_))
        ));
    }

    #[tokio::test]
    async fn test_output_guardrail_transforms_final_text() {
        let a = Arc::new(
            Agent::builder()
                .model(DirectResponseModel::new("original answer"))
                .output_guardrail(TransformOutputGuardrail)
                .build()
                .unwrap(),
        );
        let b = Arc::new(
            Agent::builder()
                .model(DirectResponseModel::new("b"))
                .build()
                .unwrap(),
        );

        let network = HandoffNetwork::builder()
            .entry("a")
            .agent("a", a)
            .agent("b", b)
            .build()
            .unwrap();

        let response = network.run("hello").await.unwrap();
        assert_eq!(response.text(), "REPLACED");
        assert_eq!(
            response.messages.last().and_then(|m| m.content.as_deref()),
            Some("REPLACED"),
            "the message log must carry the transformed final text"
        );
    }

    #[tokio::test]
    async fn test_handoff_to_agent_without_system_prompt_removes_system_message() {
        // Handing off from an agent WITH a system prompt to one WITHOUT must
        // remove the previous agent's system message rather than letting the
        // new agent inherit instructions meant for its predecessor.
        let triage = Arc::new(
            Agent::builder()
                .model(HandoffModel::new("billing"))
                .system_prompt("You are triage")
                .build()
                .unwrap(),
        );
        let billing = Arc::new(
            Agent::builder()
                .model(DirectResponseModel::new("bill info"))
                .build()
                .unwrap(),
        );

        let network = HandoffNetwork::builder()
            .entry("triage")
            .agent("triage", triage)
            .agent("billing", billing)
            .build()
            .unwrap();

        let response = network.run("billing question").await.unwrap();
        assert_eq!(response.final_agent, "billing");
        assert!(
            !response.messages.iter().any(|m| m.role == Role::System),
            "the triage system prompt must not leak into the billing turn"
        );
    }

    /// Calls a transfer tool for an agent that doesn't exist, then answers
    /// with final text once it sees the error result.
    struct UnknownTransferModel {
        call_count: AtomicUsize,
    }

    impl Model for UnknownTransferModel {
        async fn generate(&self, _request: &ChatRequest) -> Result<ChatResponse> {
            let count = self.call_count.fetch_add(1, Ordering::SeqCst);
            if count == 0 {
                Ok(ChatResponse {
                    message: Message::assistant_with_tool_calls(vec![crate::tool::ToolCall {
                        id: "t1".into(),
                        name: format!("{HANDOFF_PREFIX}ghost"),
                        arguments: serde_json::json!({"reason": "hallucinated"}),
                    }]),
                    stop_reason: StopReason::ToolUse,
                    usage: Some(Usage::default()),
                })
            } else {
                Ok(ChatResponse {
                    message: Message::assistant("recovered"),
                    stop_reason: StopReason::EndTurn,
                    usage: Some(Usage::default()),
                })
            }
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    #[tokio::test]
    async fn test_unknown_transfer_target_surfaces_tool_error() {
        // A hallucinated transfer_to_<unknown> must not kill the run: the
        // model gets a tool error and can recover.
        let a = Arc::new(
            Agent::builder()
                .model(UnknownTransferModel {
                    call_count: AtomicUsize::new(0),
                })
                .build()
                .unwrap(),
        );
        let b = Arc::new(
            Agent::builder()
                .model(DirectResponseModel::new("b"))
                .build()
                .unwrap(),
        );

        let network = HandoffNetwork::builder()
            .entry("a")
            .agent("a", a)
            .agent("b", b)
            .build()
            .unwrap();

        let response = network.run("go").await.unwrap();
        assert_eq!(response.text(), "recovered");
        assert_eq!(response.handoff_count, 0);
        assert_eq!(response.final_agent, "a");
        assert!(
            response.messages.iter().any(|m| m.role == Role::Tool
                && m.content
                    .as_deref()
                    .unwrap_or("")
                    .contains("does not exist")),
            "the model must see a tool error for the unknown target"
        );
    }

    /// A genuine registered tool whose name collides with the transfer prefix.
    struct RealTransferTool;

    impl crate::tool::Tool for RealTransferTool {
        fn name(&self) -> &str {
            "transfer_to_billing"
        }
        fn description(&self) -> &str {
            "A real tool that happens to share the transfer prefix"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(&self, _input: &serde_json::Value) -> Result<crate::tool::ToolOutput> {
            Ok(crate::tool::ToolOutput::text("real tool output"))
        }
    }

    /// Calls the genuine transfer_to_billing tool, then finishes.
    struct CallsRealToolModel {
        call_count: AtomicUsize,
    }

    impl Model for CallsRealToolModel {
        async fn generate(&self, _request: &ChatRequest) -> Result<ChatResponse> {
            let count = self.call_count.fetch_add(1, Ordering::SeqCst);
            if count == 0 {
                Ok(ChatResponse {
                    message: Message::assistant_with_tool_calls(vec![crate::tool::ToolCall {
                        id: "t1".into(),
                        name: "transfer_to_billing".into(),
                        arguments: serde_json::json!({}),
                    }]),
                    stop_reason: StopReason::ToolUse,
                    usage: Some(Usage::default()),
                })
            } else {
                Ok(ChatResponse {
                    message: Message::assistant("done"),
                    stop_reason: StopReason::EndTurn,
                    usage: Some(Usage::default()),
                })
            }
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    #[tokio::test]
    async fn test_registered_tool_shadows_transfer_name() {
        // An agent's real tool named transfer_to_billing must execute as a
        // tool — even though "billing" is a registered agent — instead of
        // being hijacked into a handoff.
        let a = Arc::new(
            Agent::builder()
                .model(CallsRealToolModel {
                    call_count: AtomicUsize::new(0),
                })
                .tool(RealTransferTool)
                .build()
                .unwrap(),
        );
        let billing = Arc::new(
            Agent::builder()
                .model(DirectResponseModel::new("bill info"))
                .build()
                .unwrap(),
        );

        let network = HandoffNetwork::builder()
            .entry("a")
            .agent("a", a)
            .agent("billing", billing)
            .build()
            .unwrap();

        let response = network.run("go").await.unwrap();
        assert_eq!(response.handoff_count, 0, "no handoff may occur");
        assert_eq!(response.final_agent, "a");
        assert!(
            response
                .messages
                .iter()
                .any(|m| m.role == Role::Tool && m.content.as_deref() == Some("real tool output")),
            "the genuine tool must have executed"
        );
    }
}
