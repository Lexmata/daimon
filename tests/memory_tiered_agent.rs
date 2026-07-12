//! Integration coverage for [`TieredMemory`] wired into a real
//! [`AgentBuilder`]/ReAct loop.
//!
//! `TieredMemory` itself is well covered by unit tests in
//! `src/memory/tiered.rs` (composition, tier independence, the
//! `system_prompt_block` helper). What isn't covered anywhere is that it
//! actually drops into `AgentBuilder::memory()` and survives a real turn of
//! the agent loop — i.e. the `Memory` trait impl the agent runner calls
//! against (`add_message`/`get_messages`) behaves correctly when reached
//! through the full `Agent::prompt` path rather than called directly.

use daimon::agent::Agent;
use daimon::error::Result;
use daimon::memory::{
    ArchivalMemory, CoreMemory, CoreMemoryBlock, InMemoryArchivalMemory, InMemoryCoreMemory,
    InMemoryEpisodicMemory, SlidingWindowMemory, TieredMemory,
};
use daimon::model::Model;
use daimon::model::types::*;
use daimon::stream::ResponseStream;
use daimon::tool::{Tool, ToolCall, ToolOutput};

struct EchoModel;

impl Model for EchoModel {
    async fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let turns = request
            .messages
            .iter()
            .filter(|m| m.role == Role::User)
            .count();
        Ok(ChatResponse {
            message: Message::assistant(format!("turn {turns}")),
            stop_reason: StopReason::EndTurn,
            usage: Some(Usage::default()),
        })
    }

    async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

struct Doubler;

impl Tool for Doubler {
    fn name(&self) -> &str {
        "doubler"
    }
    fn description(&self) -> &str {
        "Doubles a number"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {"n": {"type": "number"}},
            "required": ["n"]
        })
    }
    async fn execute(&self, input: &serde_json::Value) -> Result<ToolOutput> {
        let n = input["n"].as_f64().unwrap_or(0.0);
        Ok(ToolOutput::text(format!("{}", n * 2.0)))
    }
}

#[tokio::test]
async fn tiered_memory_drives_a_real_agent_loop() {
    let archival = InMemoryArchivalMemory::new();
    archival
        .insert("the user prefers concise answers", Default::default())
        .await
        .unwrap();

    let memory = TieredMemory::new(SlidingWindowMemory::new(10)).with_archival(archival);

    let agent = Agent::builder()
        .model(EchoModel)
        .memory(memory)
        .build()
        .unwrap();

    let first = agent.prompt("hello").await.unwrap();
    assert_eq!(first.text(), "turn 1");

    let second = agent.prompt("again").await.unwrap();
    // The conversation tier must have retained the first turn's user + reply
    // for the second `generate` call to see two user messages.
    assert_eq!(second.text(), "turn 2");

    let messages = agent.memory().get_messages_erased().await.unwrap();
    // 2 user prompts + 2 assistant replies.
    assert_eq!(messages.len(), 4);
}

struct ToolCallThenAnswer {
    call_count: std::sync::atomic::AtomicUsize,
}

impl Model for ToolCallThenAnswer {
    async fn generate(&self, _request: &ChatRequest) -> Result<ChatResponse> {
        use std::sync::atomic::Ordering;
        let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
        if idx == 0 {
            Ok(ChatResponse {
                message: Message::assistant_with_tool_calls(vec![ToolCall {
                    id: "tc_1".into(),
                    name: "doubler".into(),
                    arguments: serde_json::json!({"n": 21}),
                }]),
                stop_reason: StopReason::ToolUse,
                usage: None,
            })
        } else {
            Ok(ChatResponse {
                message: Message::assistant("done"),
                stop_reason: StopReason::EndTurn,
                usage: None,
            })
        }
    }

    async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

/// Builds a `TieredMemory` with all three optional tiers attached (core,
/// archival, episodic) and drives a full tool-calling ReAct loop through it,
/// confirming the composed memory works end-to-end as the agent's `Memory`
/// — not just that each tier behaves correctly in isolation (already
/// covered by unit tests in `src/memory/tiered.rs` and
/// `src/memory/archival_memory.rs`).
#[tokio::test]
async fn tiered_memory_with_all_tiers_supports_tool_calls_in_the_react_loop() {
    let core = InMemoryCoreMemory::new();
    core.put_block(CoreMemoryBlock::new("persona", "a math assistant"))
        .await
        .unwrap();
    let archival = InMemoryArchivalMemory::new();
    archival
        .insert("the user prefers concise answers", Default::default())
        .await
        .unwrap();
    let episodic = InMemoryEpisodicMemory::new();

    let memory = TieredMemory::new(SlidingWindowMemory::new(10))
        .with_core(core)
        .with_archival(archival)
        .with_episodic(episodic);

    let model = ToolCallThenAnswer {
        call_count: std::sync::atomic::AtomicUsize::new(0),
    };

    let agent = Agent::builder()
        .model(model)
        .memory(memory)
        .tool(Doubler)
        .build()
        .unwrap();

    let response = agent.prompt("double 21").await.unwrap();
    assert_eq!(response.text(), "done");
    assert_eq!(response.iterations, 2);

    // The conversation tier recorded the user prompt, the tool-call
    // assistant turn, the tool result, and the final assistant reply.
    let messages = agent.memory().get_messages_erased().await.unwrap();
    assert_eq!(messages.len(), 4);
}
