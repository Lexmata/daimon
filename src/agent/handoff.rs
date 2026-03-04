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
use crate::error::{DaimonError, Result};
use crate::model::types::{ChatRequest, Message, ToolSpec, Usage};
use crate::tool::ToolOutput;

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
    #[tracing::instrument(skip_all, fields(entry = %self.entry))]
    pub async fn run(&self, input: &str) -> Result<HandoffResponse> {
        let mut current_agent_name = self.entry.clone();
        let mut handoffs = 0usize;
        let mut total_usage = Usage::default();
        let mut total_iterations = 0usize;

        let mut messages: Vec<Message> = Vec::new();

        let current_agent = self.agents.get(&current_agent_name).ok_or_else(|| {
            DaimonError::Orchestration(format!("entry agent '{}' not found", self.entry))
        })?;

        if let Some(system) = &current_agent.system_prompt {
            messages.push(Message::system(system));
        }
        messages.push(Message::user(input));

        loop {
            let agent = self.agents.get(&current_agent_name).ok_or_else(|| {
                DaimonError::Orchestration(format!("agent '{current_agent_name}' not found"))
            })?;

            let transfer_tools = self.build_transfer_tools(&current_agent_name);
            let mut agent_tool_specs: Vec<ToolSpec> = agent.tools.tool_specs().to_vec();
            agent_tool_specs.extend(transfer_tools);

            let mut agent_iterations = 0usize;

            loop {
                agent_iterations += 1;
                total_iterations += 1;

                if agent_iterations > self.max_iterations_per_agent {
                    return Err(DaimonError::MaxIterations(self.max_iterations_per_agent));
                }

                let request = ChatRequest {
                    messages: messages.clone(),
                    tools: agent_tool_specs.clone(),
                    temperature: agent.temperature,
                    max_tokens: agent.max_tokens,
                };

                let response = agent.model.generate_erased(&request).await?;

                if let Some(ref usage) = response.usage {
                    total_usage.accumulate(usage);
                }

                if !response.has_tool_calls() {
                    let final_text = response.text().to_string();
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
                messages.push(Message::assistant_with_tool_calls(tool_calls.clone()));

                let mut handoff_target: Option<String> = None;

                for call in &tool_calls {
                    if let Some(target) = call.name.strip_prefix(HANDOFF_PREFIX) {
                        let reason = call
                            .arguments
                            .get("reason")
                            .and_then(|v| v.as_str())
                            .unwrap_or("transferring");

                        messages.push(Message::tool_result(
                            &call.id,
                            format!("Transferring to {target}. Reason: {reason}"),
                        ));
                        handoff_target = Some(target.to_string());
                    } else {
                        let output = match agent.tools.get(&call.name) {
                            Some(tool) => match tool.execute_erased(&call.arguments).await {
                                Ok(out) => out,
                                Err(e) => ToolOutput::error(e.to_string()),
                            },
                            None => {
                                ToolOutput::error(format!("tool '{}' not found", call.name))
                            }
                        };
                        messages.push(Message::tool_result(&call.id, &output.content));
                    }
                }

                if let Some(target) = handoff_target {
                    handoffs += 1;
                    if handoffs > self.max_handoffs {
                        return Err(DaimonError::Orchestration(format!(
                            "max handoffs ({}) exceeded",
                            self.max_handoffs
                        )));
                    }

                    if let Some(new_agent) = self.agents.get(&target) {
                        if let Some(system) = &new_agent.system_prompt {
                            messages.retain(|m| m.role != crate::model::types::Role::System);
                            messages.insert(0, Message::system(system));
                        }
                    }

                    tracing::info!(
                        from = %current_agent_name,
                        to = %target,
                        "agent handoff"
                    );
                    current_agent_name = target;
                    break;
                }
            }
        }
    }

    fn build_transfer_tools(&self, current: &str) -> Vec<ToolSpec> {
        self.agents
            .keys()
            .filter(|name| *name != current)
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
        let entry = self
            .entry
            .ok_or_else(|| DaimonError::Builder("handoff network requires an entry agent".into()))?;

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
        let result = HandoffNetwork::builder()
            .entry("a")
            .agent("a", a)
            .build();
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
}
