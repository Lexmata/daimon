# Daimon Multi-Agent Guide

This document covers all multi-agent patterns in the Daimon Rust AI agent framework: Agent-as-Tool, Supervisor, Handoff Network, Fork Builder, and Hot-Swap Agent. Each pattern serves a distinct purpose; choose based on your delegation model, flow control, and runtime requirements.

---

## Table of Contents

1. [Agent-as-Tool](#agent-as-tool)
2. [Supervisor](#supervisor)
3. [Handoff Network](#handoff-network)
4. [Fork Builder](#fork-builder)
5. [Hot-Swap Agent](#hot-swap-agent)
6. [Patterns & Best Practices](#patterns--best-practices)

---

## Agent-as-Tool

`AgentTool` wraps an agent as a `Tool`, so another agent can invoke it by name. The caller sends an `input` string; the inner agent runs its ReAct loop; the caller receives the response text as the tool result.

### How It Works

- The wrapped agent receives the `"input"` field from the tool arguments as its prompt.
- The agent runs `agent.prompt(input)` and returns `response.final_text` as the tool output.
- If the inner agent fails, the tool returns an error message (not a panic) so the caller can react.

### API

```rust
AgentTool::new(agent: Arc<Agent>, name: impl Into<String>, description: impl Into<String>)
```

- **agent** — The agent to wrap (shared via `Arc`)
- **name** — Tool name the calling agent uses when invoking
- **description** — Description the model sees when deciding to call this tool

### Example: Research Agent + Writing Agent

A writer agent delegates research to a specialist. The writer calls the `research` tool with a topic; the researcher returns findings; the writer composes the final output.

```rust
use daimon::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> daimon::Result<()> {
    let model = Arc::new(daimon::model::openai::OpenAi::new("gpt-4o"));

    let researcher = Arc::new(
        Agent::builder()
            .shared_model(Arc::clone(&model))
            .system_prompt(
                "You are a research specialist. Search for facts and summarize findings concisely.",
            )
            .build()?,
    );

    let writer = Agent::builder()
        .shared_model(model)
        .system_prompt(
            "You are a content writer. Use the research tool to gather facts, \
             then write polished prose. Always research before writing.",
        )
        .tool(AgentTool::new(
            researcher,
            "research",
            "Perform research on a topic. Input: the research question or topic.",
        ))
        .build()?;

    let response = writer
        .prompt("Write a short paragraph about Rust's ownership model.")
        .await?;

    println!("{}", response.text());
    Ok(())
}
```

### Tool Schema

`AgentTool` exposes a JSON Schema with a single required `input` string:

```json
{
  "type": "object",
  "properties": {
    "input": {
      "type": "string",
      "description": "The task or question to send to the agent"
    }
  },
  "required": ["input"]
}
```

The calling agent passes `{"input": "research question"}` when invoking the tool.

---

## Supervisor

A **Supervisor** is a coordinator agent whose tools *are* sub-agents. The LLM decides which sub-agent to invoke based on the task. Use for request-response delegation: one input, one routed call, one aggregated result.

### How It Works

- Each sub-agent is wrapped as an `AgentTool` and registered with the coordinator.
- The coordinator runs the ReAct loop; when it needs help, it calls a sub-agent tool.
- Sub-agents run independently; their responses are fed back as tool results.
- The coordinator can call multiple sub-agents in sequence or combine their outputs.

### API

```rust
Supervisor::builder()
    .model(M)
    .system_prompt("...")
    .agent(name, agent, description)
    .tool(extra_tool)           // optional
    .max_iterations(25)
    .build()?
```

### Example: Customer Service Supervisor

A supervisor routes to billing, technical support, or general inquiry agents based on the user's message.

```rust
use daimon::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> daimon::Result<()> {
    let model = Arc::new(daimon::model::openai::OpenAi::new("gpt-4o"));

    let billing = Arc::new(
        Agent::builder()
            .shared_model(Arc::clone(&model))
            .system_prompt(
                "You handle billing questions: invoices, payments, refunds. \
                 Be precise and reference account details when available.",
            )
            .build()?,
    );

    let technical = Arc::new(
        Agent::builder()
            .shared_model(Arc::clone(&model))
            .system_prompt(
                "You handle technical support: bugs, integrations, API usage. \
                 Provide step-by-step guidance.",
            )
            .build()?,
    );

    let general = Arc::new(
        Agent::builder()
            .shared_model(Arc::clone(&model))
            .system_prompt(
                "You handle general inquiries: product info, policies, FAQs. \
                 Be friendly and helpful.",
            )
            .build()?,
    );

    let supervisor = Supervisor::builder()
        .shared_model(model)
        .system_prompt(
            "You are a customer service coordinator. Route the user's request \
             to the appropriate specialist: billing, technical, or general. \
             Call the right agent tool based on the query.",
        )
        .agent("billing", billing, "Handles billing, invoices, payments, refunds")
        .agent("technical", technical, "Handles technical support and integrations")
        .agent("general", general, "Handles general inquiries and FAQs")
        .build()?;

    let response = supervisor
        .run("I was charged twice for my subscription last month")
        .await?;

    println!("{}", response.text());
    Ok(())
}
```

---

## Handoff Network

A **HandoffNetwork** lets agents transfer control to each other mid-conversation. Each agent has synthetic `transfer_to_<name>` tools; when invoked, the conversation switches to that agent while preserving message history. Use for conversational flows where control passes between specialists.

### How It Works

- The network runs its own loop, starting with the entry agent.
- Each agent receives its normal tools plus `transfer_to_<other_agent>` for every other agent.
- When the model calls `transfer_to_billing`, the network switches the active agent, updates the system prompt, and continues with the same message history.
- Context is preserved across handoffs; the new agent sees the full conversation.

### API

```rust
HandoffNetwork::builder()
    .entry(name)                    // first agent to receive input
    .agent(name, agent)
    .max_handoffs(10)
    .max_iterations_per_agent(25)
    .build()?
```

### Example: Triage → Specialist → Escalation

A triage agent classifies the request; if needed, it hands off to a specialist; the specialist can escalate to a senior agent.

```rust
use daimon::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> daimon::Result<()> {
    let model = Arc::new(daimon::model::openai::OpenAi::new("gpt-4o"));

    let triage = Arc::new(
        Agent::builder()
            .shared_model(Arc::clone(&model))
            .system_prompt(
                "You are a triage agent. Classify the request: billing, technical, \
                 or escalation. Use transfer_to_billing, transfer_to_technical, \
                 or transfer_to_escalation when the user needs a specialist.",
            )
            .build()?,
    );

    let billing = Arc::new(
        Agent::builder()
            .shared_model(Arc::clone(&model))
            .system_prompt(
                "You handle billing. Resolve invoices, payments, refunds. \
                 Use transfer_to_escalation if the issue requires a manager.",
            )
            .build()?,
    );

    let technical = Arc::new(
        Agent::builder()
            .shared_model(Arc::clone(&model))
            .system_prompt(
                "You handle technical support. Use transfer_to_escalation \
                 for complex or unresolved issues.",
            )
            .build()?,
    );

    let escalation = Arc::new(
        Agent::builder()
            .shared_model(Arc::clone(&model))
            .system_prompt(
                "You are a senior agent. Handle escalated cases. \
                 Provide final resolution or next steps.",
            )
            .build()?,
    );

    let network = HandoffNetwork::builder()
        .entry("triage")
        .agent("triage", triage)
        .agent("billing", billing)
        .agent("technical", technical)
        .agent("escalation", escalation)
        .max_handoffs(10)
        .build()?;

    let response = network
        .run("I need help with my bill and a bug in the API")
        .await?;

    println!("Final agent: {}", response.final_agent);
    println!("Handoffs: {}", response.handoff_count);
    println!("{}", response.text());
    Ok(())
}
```

### HandoffResponse

```rust
pub struct HandoffResponse {
    pub messages: Vec<Message>,
    pub final_text: String,
    pub final_agent: String,
    pub handoff_count: usize,
    pub iterations: usize,
    pub usage: Usage,
}
```

---

## Fork Builder

Use `agent.fork_builder()` to create a mutated copy of an agent. The builder starts with the parent's config; you override specific fields, then `build()` to produce an independent agent. Use for A/B testing, specialized variants, or branching configurations.

### How It Works

- `fork_builder()` returns a `ForkBuilder` pre-populated with the agent's model, tools, hooks, middleware, guardrails, limits, cost config, etc. — but **not** its memory.
- Mutate with `.system_prompt()`, `.tool()`, `.remove_tool()`, `.model()`, `.memory()`, etc.
- `build()` creates a new `Agent` with fresh memory (unless you set `.memory()` explicitly).
- The forked agent is independent; changes do not affect the original.

### API

```rust
agent.fork_builder()
    .system_prompt("...")
    .tool(my_tool)
    .remove_tool("old_tool")
    .model(new_model)
    .memory(custom_memory)
    .hooks(my_hooks)
    .max_iterations(10)
    .build()
```

### Example: A/B Testing Prompts

```rust
use daimon::prelude::*;

let base = Agent::builder()
    .model(my_model)
    .tool(search_tool)
    .build()?;

let variant_a = base
    .fork_builder()
    .system_prompt("You are concise. Answer in 1-2 sentences.")
    .build()?;

let variant_b = base
    .fork_builder()
    .system_prompt("You are thorough. Provide detailed explanations.")
    .build()?;

let (resp_a, resp_b) = tokio::join!(
    variant_a.prompt("What is async in Rust?"),
    variant_b.prompt("What is async in Rust?"),
);
```

### Simple Fork Variants

| Method | Description |
|-------|-------------|
| `agent.fork()` | Clone with fresh memory; same config |
| `agent.fork_from_checkpoint(run_id, checkpoint)` | Resume from checkpoint; same config |
| `agent.fork_with_memory(mem)` | Same config, different memory backend |

```rust
// Simple clone: independent memory, same everything else
let copy = agent.fork();

// Resume from saved state
let resumed = agent.fork_from_checkpoint("run-123", &checkpoint).await?;

// Switch to persistent memory
let persistent = agent.fork_with_memory(SqliteMemory::open("./db").await?);
```

---

## Hot-Swap Agent

`HotSwapAgent` wraps an agent behind a `RwLock`, allowing runtime reconfiguration without rebuilding. Concurrent prompts hold a read lock; swap operations hold a write lock. Use for feature flags, gradual rollout, or live experimentation.

### How It Works

- All prompt operations acquire a read lock; they can run concurrently.
- Swap operations acquire a write lock; they block only during the brief update.
- Ongoing prompts complete with their original configuration; new prompts use the updated config.

### API

```rust
let hot = HotSwapAgent::new(agent);

// Swaps (write lock)
hot.swap_model(new_model).await;
hot.swap_system_prompt(Some("...".into())).await;
hot.add_tool(my_tool).await;
hot.remove_tool("name").await;
hot.swap_memory(new_memory).await;
hot.swap_hooks(my_hooks).await;
hot.swap_middleware(stack).await;
hot.add_input_guardrail(guard).await;
hot.add_output_guardrail(guard).await;
hot.clear_input_guardrails().await;
hot.clear_output_guardrails().await;
hot.set_max_iterations(20).await;
hot.set_temperature(Some(0.5)).await;
hot.replace(new_agent).await;

// Reads (no lock conflict with prompts)
let prompt = hot.system_prompt().await;
let count = hot.tool_count().await;
let names = hot.tool_names().await;
```

### Example: Feature Flag Model Swap

```rust
use daimon::prelude::*;
use daimon::model::openai::OpenAi;

let agent = Agent::builder()
    .model(OpenAi::new("gpt-4o"))
    .system_prompt("You are helpful.")
    .build()?;

let hot = HotSwapAgent::new(agent);

// 90% of traffic uses gpt-4o; 10% uses gpt-4o-mini for cost testing
if rand::random::<f32>() < 0.1 {
    hot.swap_model(OpenAi::new("gpt-4o-mini")).await;
}

let response = hot.prompt("Explain recursion").await?;
```

### Example: Gradual Prompt Rollout

```rust
let hot = HotSwapAgent::new(agent);

// Admin endpoint: update system prompt without restart
async fn update_prompt(hot: &HotSwapAgent, new_prompt: String) {
    hot.swap_system_prompt(Some(new_prompt)).await;
}
```

---

## Patterns & Best Practices

### Supervisor vs Handoff: When to Use Which

| Pattern | Use When | Flow |
|---------|----------|-----|
| **Supervisor** | Request-response delegation; coordinator decides and aggregates | User → Supervisor → sub-agent tool call → Supervisor → final response |
| **Handoff** | Conversational flow; control passes between agents mid-dialog | User → Agent A → transfer_to_B → Agent B → transfer_to_C → Agent C → final response |

- **Supervisor**: The coordinator stays in control. It calls sub-agents as tools, gets results, and produces the final answer. Good for: routing, parallel sub-tasks, aggregation.
- **Handoff**: The active agent changes. Each agent "owns" the conversation until it hands off. Good for: triage → specialist → escalation, multi-step support flows, collaborative dialogue.

### Fork vs Hot-Swap: When to Use Which

| Pattern | Use When | Mutability |
|---------|----------|-------------|
| **Fork** | Parallel variants, A/B tests, branching configs | Creates new agent; original unchanged |
| **Hot-Swap** | Live updates, feature flags, runtime experimentation | Mutates wrapped agent in place |

- **Fork**: Use when you need multiple independent copies (e.g., run variant A and B in parallel, or create specialized agents from a template). No shared state.
- **Hot-Swap**: Use when one agent must be reconfigured at runtime (e.g., swap model for cost testing, update prompt via admin UI). All callers see the same agent; updates apply to future calls.

### Composing Multi-Agent with Orchestration

You can use agents, supervisors, and handoff networks as nodes in Chains, Graphs, and DAGs.

```rust
use daimon::orchestration::{Chain, FnNode, NodeOutcome};
use std::sync::Arc;

// Supervisor as a Graph node
let supervisor = Arc::new(Supervisor::builder()
    .model(model)
    .agent("researcher", research_agent, "Research")
    .agent("writer", writer_agent, "Writing")
    .build()?);

let supervisor_node = FnNode::new(move |ctx| {
    let sup = Arc::clone(&supervisor);
    Box::pin(async move {
        let input = ctx.get_str("input").unwrap_or("").to_string();
        let response = sup.run(&input).await?;
        ctx.set("output", serde_json::json!(response.final_text));
        Ok(NodeOutcome::Continue)
    })
});

// Handoff network as a Chain step (HandoffNetwork is not Clone; share via Arc)
let handoff = Arc::new(
    HandoffNetwork::builder()
        .entry("triage")
        .agent("triage", triage_agent)
        .agent("specialist", specialist_agent)
        .build()?,
);

let chain = Chain::builder()
    .transform(|mut ctx| {
        let h = Arc::clone(&handoff);
        Box::pin(async move {
            let resp = h.run(&ctx.text).await?;
            ctx.text = resp.final_text;
            Ok(ctx)
        })
    })
    .build()?;
```

### Memory Isolation Between Agents

- **AgentTool**: The inner agent uses its own memory. Each call is a fresh prompt from the caller's perspective; the inner agent's memory accumulates across calls within the same coordinator run.
- **Supervisor**: Sub-agents have independent memory. The coordinator has its own memory (conversation with the user + tool results).
- **Handoff**: All agents share the same message list. The network maintains one `messages` vec; only the system prompt and active agent change on handoff. No per-agent memory isolation.
- **Fork**: Each fork has independent memory by default. Use `fork_with_memory` to share or customize.

### Cost Tracking Across Multi-Agent Systems

- **AgentTool**: Cost is incurred by the inner agent. The coordinator's cost tracker does not automatically include sub-agent spend. Attach a `CostModel` to each agent and aggregate manually if needed.
- **Supervisor**: The coordinator's `AgentResponse` includes its own cost. Sub-agent costs are not rolled up. Use hooks or middleware to aggregate.
- **Handoff**: `HandoffResponse.usage` aggregates token usage across all agents. Cost is not computed unless each agent has a `CostModel`; you must sum manually.
- **Best practice**: Use a shared `CostTracker` or a hook that records cost per agent, then aggregate in your application layer.

### Quick Reference

| Need | Use |
|------|-----|
| One agent calls another as a tool | `AgentTool::new(agent, name, description)` |
| Coordinator routes to specialists | `Supervisor::builder().agent(...).build()` |
| Agents pass conversation to each other | `HandoffNetwork::builder().entry(...).agent(...).build()` |
| Create mutated copy for A/B testing | `agent.fork_builder().system_prompt(...).build()` |
| Simple clone with fresh memory | `agent.fork()` |
| Resume from checkpoint | `agent.fork_from_checkpoint(run_id, checkpoint)` |
| Different memory, same config | `agent.fork_with_memory(mem)` |
| Runtime model/prompt/tool swap | `HotSwapAgent::new(agent)` + `swap_*` / `add_*` |
