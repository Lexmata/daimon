# Daimon Agent Reference

This document is the definitive reference for the **Agent** — the central type in the Daimon Rust AI agent framework. It covers the builder API, the ReAct execution loop, memory backends, hooks, middleware, guardrails, cost tracking, prompt templates, streaming, and the response type.

---

## Table of Contents

1. [The Agent Builder](#the-agent-builder)
2. [The ReAct Loop in Detail](#the-react-loop-in-detail)
3. [Memory Deep-Dive](#memory-deep-dive)
4. [Hooks](#hooks)
5. [Middleware](#middleware)
6. [Guardrails](#guardrails)
7. [Cost Tracking](#cost-tracking)
8. [Prompt Templates](#prompt-templates)
9. [Streaming](#streaming)
10. [AgentResponse](#agentresponse)

---

## The Agent Builder

Every agent is constructed via the fluent builder pattern. Model is **required**; all other fields have sensible defaults.

```rust
use daimon::prelude::*;

let agent = Agent::builder()
    .model(my_model)
    .system_prompt("You are a helpful assistant.")
    .tool(calculator_tool)
    .memory(SlidingWindowMemory::new(50))
    .max_iterations(25)
    .build()?;
```

### Builder Methods Reference

| Method | Type | Default | Description |
|--------|------|---------|-------------|
| `model` | `M: Model + 'static` | *required* | Sets the LLM provider. |
| `shared_model` | `SharedModel` | *required* | Sets a pre-boxed shared model. |
| `system_prompt` | `impl Into<String>` | `None` | Static system prompt injected at conversation start. |
| `prompt_template` | `PromptTemplate` | `None` | Dynamic prompt with `{variable}` interpolation. |
| `tool` | `T: Tool + 'static` | `ToolRegistry::new()` | Registers a tool. Tools must have unique names. |
| `memory` | `M: Memory + 'static` | `SlidingWindowMemory::default()` (50 msgs) | Conversation memory backend. |
| `hooks` | `H: AgentHook + 'static` | `NoOpHook` | Lifecycle observation callbacks. |
| `max_iterations` | `usize` | `25` | Maximum ReAct loop iterations before abort. |
| `temperature` | `f32` | `None` | Model sampling temperature (0.0–2.0). |
| `max_tokens` | `u32` | `None` | Max tokens for model output. |
| `validate_tool_inputs` | `bool` | `true` | JSON Schema validation of tool arguments. |
| `middleware` | `M: Middleware + 'static` | `MiddlewareStack::new()` | Request/response/tool_call interception. |
| `input_guardrail` | `G: InputGuardrail + 'static` | `[]` | Input validation before processing. |
| `output_guardrail` | `G: OutputGuardrail + 'static` | `[]` | Output validation before returning. |
| `cost_model` | `C: CostModel + 'static` | `None` | Token cost tracking. |
| `max_budget` | `f64` | `None` | Dollar budget limit per prompt. |
| `tool_retry_policy` | `ToolRetryPolicy` | `None` | Automatic tool retry on transient failures. |
| `human_input` | `H: HumanInputHandler + 'static` | *none* | Registers `ask_human` tool for human-in-the-loop. |

### model vs shared_model

- **`model(M)`** — Takes any `impl Model`. The builder wraps it in `Arc<dyn ErasedModel>`. Use when you have a concrete model instance and don't need to share it across agents.
- **`shared_model(SharedModel)`** — Takes a pre-boxed `Arc<dyn ErasedModel>`. Use when you need to **share the same model** across multiple agents (e.g., a pool of agents sharing one API client). Avoids duplicate allocations.

```rust
// Static dispatch: one agent, one model
let agent = Agent::builder()
    .model(OpenAi::new("gpt-4o"))
    .build()?;

// Dynamic dispatch: share model across agents
let shared = Arc::new(OpenAi::new("gpt-4o"));
let agent1 = Agent::builder().shared_model(Arc::clone(&shared)).build()?;
let agent2 = Agent::builder().shared_model(Arc::clone(&shared)).build()?;
```

### system_prompt vs prompt_template

- **`system_prompt(s)`** — A static string. Same prompt for every conversation. Use for fixed personas or instructions.
- **`prompt_template(tpl)`** — A `PromptTemplate` with `{variable}` placeholders. Rendered once at `build()` via `render_static()`. Use when the system prompt depends on configuration (e.g., `"You are {role}. Today is {date}."`). At build time, all variables must be set; the result is stored as the system prompt. If both `system_prompt` and `prompt_template` are set, `prompt_template` takes precedence.

```rust
// Static
.system_prompt("You are a helpful assistant.")

// Dynamic (variables set at build time)
.prompt_template(
    PromptTemplate::new("You are {role}. Today is {date}.")
        .var("role", "a research assistant")
        .var("date", "2026-03-04")
)
```

### tool and warm_cache

Tools are registered with `tool(T)`. Each tool must have a unique `name()`. The builder calls `tools.warm_cache()` at `build()` time, which:

1. Pre-compiles JSON Schema validators for all tools (avoids per-call compilation in the ReAct loop).
2. Caches `tool_specs()` so the model receives the same `Arc<[ToolSpec]>` on every iteration.

This reduces allocation and CPU in the hot path. You don't need to call `warm_cache` manually — the builder does it.

### human_input

Adds an `ask_human` tool that the agent can call to request input from a human. The handler receives `HumanInputRequest` (prompt, choices, context) and returns the human's response. Use for approval flows, clarification, or any step requiring human judgment.

```rust
use daimon::agent::hitl::{HumanInputHandler, HumanInputRequest};

struct ConsoleHandler;

impl HumanInputHandler for ConsoleHandler {
    async fn request_input(&self, request: &HumanInputRequest) -> Result<String> {
        println!("Agent asks: {}", request.prompt);
        // Read from stdin, show choices, etc.
        Ok("user response".into())
    }
}

let agent = Agent::builder()
    .model(model)
    .human_input(ConsoleHandler)
    .build()?;
```

---

## The ReAct Loop in Detail

When you call `agent.prompt("...")`, the following steps occur in order.

### 1. Input Guardrails Check

Each input guardrail runs on the raw user input. Results:

- **`Pass`** — Continue to the next guardrail.
- **`Block(msg)`** — Return `Err(DaimonError::GuardrailBlocked(msg))` immediately.
- **`Transform(new_input)`** — Replace the input with `new_input` and continue.

The (possibly transformed) input is used for the rest of the flow.

### 2. Memory Retrieval

`memory.get_messages()` returns the conversation history. This is combined with the system prompt and the new user message to form the initial message list.

### 3. Message Assembly

```
[system message (if any)] + [history from memory] + [user message]
```

### 4. Memory Update

The user message is appended to memory via `memory.add_message(...)`.

### 5. Iteration Loop

The loop runs until the model produces a final text response (no tool calls) or an exit condition is hit.

#### a. Cancellation Check

Before each iteration, `cancel.is_cancelled()` is checked. If true, return `Err(DaimonError::Cancelled)`.

#### b. Budget Check

If `cost_tracker` and `max_budget` are set, and `tracker.cumulative_cost() >= max_budget`, return `Err(DaimonError::BudgetExceeded { spent, limit })`.

#### c. Hooks: on_iteration_start

`hooks.on_iteration_start_erased(&state)` is called with `AgentState { iteration, max_iterations }`.

#### d. ChatRequest Construction

A `ChatRequest` is built from:

- `messages` (moved via `std::mem::take`)
- `tools` (tool specs)
- `temperature`, `max_tokens` from the agent

#### e. Middleware: on_request

`middleware.run_on_request(&mut request)` runs each layer in order. Any layer may return `ShortCircuit(ChatResponse)` to skip the model call and return that response immediately.

#### f. Model Call

`model.generate_erased(&request)` is invoked.

#### g. Cost Tracking

If the response includes `usage`, it is accumulated into `total_usage` and `tracker.record(model_id, usage)` is called to update the cost tracker.

#### h. Middleware: on_response

`middleware.run_on_response(&mut response)` runs. Any layer may return `ShortCircuit(ChatResponse)` to replace the model's response and return immediately.

#### i. Hooks: on_model_response

`hooks.on_model_response_erased(&response)` is called.

#### j. If Tool Calls

- Add the assistant message (with `tool_calls`) to memory and to the message list.
- Run `execute_tools_parallel(tool_calls)`:
  - For each call: middleware `on_tool_call`, hooks `on_tool_call`, schema validation (if enabled), execute with retry.
  - Tool results are collected in order.
- Add each tool result message to memory and to the message list.
- Call `hooks.on_iteration_end_erased(&state)`.
- If `iteration >= max_iterations`, return `Err(DaimonError::MaxIterations(n))`.
- Otherwise, `continue` to the next iteration.

#### k. If No Tool Calls

- Run output guardrails on the final text: `Block` returns
  `Err(DaimonError::GuardrailBlocked)` without persisting the response;
  `Transform` rewrites it before persistence.
- Call `hooks.on_iteration_end_erased(&state)`.
- Add the (possibly transformed) assistant message to memory and to the message list.
- Return `AgentResponse { messages, final_text, iterations, usage, cost }`.

### Prompt Methods Comparison

| Method | Input Guardrails | Output Guardrails | Memory | System Prompt |
|--------|------------------|-------------------|--------|---------------|
| `prompt(input)` | Yes | Yes | Load + update | Yes |
| `prompt_with_cancellation(input, cancel)` | Yes | Yes | Load + update | Yes |
| `prompt_with_messages(messages)` | Yes (last user message) | Yes | Update only (caller-provided messages are persisted, no load) | Bypassed (caller supplies) |

Use `prompt_with_cancellation` when you need cooperative cancellation (e.g., user timeout). Use `prompt_with_messages` for replay, custom context injection, or when you manage the full message history yourself.

### Flow Diagram (Simplified)

```
User Input
    → Input Guardrails (Pass/Block/Transform)
    → Memory Retrieval
    → Message Assembly (system + history + user)
    → Add User Message to Memory
    → Loop:
        → Cancel? → Err
        → Budget? → Err
        → on_iteration_start
        → on_request (middleware, may short-circuit)
        → model.generate
        → Cost tracking
        → on_response (middleware, may short-circuit)
        → on_model_response
        → Tool calls? → Execute tools, add results, continue
        → No tools? → Output guardrails → Return
```

---

## Memory Deep-Dive

### SlidingWindowMemory

Keeps only the most recent N messages in a `VecDeque`. When the window is exceeded, the oldest message is evicted. Default window: 50.

```rust
use daimon::memory::SlidingWindowMemory;

// Default: 50 messages
let memory = SlidingWindowMemory::default();

// Custom window
let memory = SlidingWindowMemory::new(20);

let agent = Agent::builder()
    .model(model)
    .memory(memory)
    .build()?;
```

**When to use:** Single-session conversations, stateless agents, development. No persistence; data is lost when the agent is dropped.

---

### TokenWindowMemory

Keeps messages within a **token budget** instead of a message count. Uses a heuristic (~4 chars per token) by default; you can plug in a precise tokenizer via `with_token_counter`.

```rust
use daimon::memory::TokenWindowMemory;

// 4096 token budget, default estimator
let memory = TokenWindowMemory::new(4096);

// Custom tokenizer (e.g. tiktoken-rs)
let memory = TokenWindowMemory::new(4096)
    .with_token_counter(|msg| {
        msg.content.as_ref().map_or(0, |c| my_tokenizer.count(c))
    });

// Check current usage
let tokens = memory.current_tokens().await;
```

**When to use:** When you need to stay within model context limits (e.g., 128K tokens). Evicts oldest messages when the budget is exceeded. Single message exceeding the budget is kept (cannot evict the only message).

---

### SummaryMemory

Uses an LLM to **summarize** old messages instead of dropping them. When the message count exceeds `max_messages`, the oldest messages (all except the last `retain_recent`) are summarized into a single system message. The summary is prepended to future context.

```rust
use daimon::memory::SummaryMemory;
use std::sync::Arc;

let model: SharedModel = Arc::new(OpenAi::new("gpt-4o-mini"));
let memory = SummaryMemory::new(model)
    .with_max_messages(20)   // Summarize when > 20 messages
    .with_retain_recent(10) // Keep 10 most recent unsummarized
    .with_summary_prompt("Custom summarization instructions...");

// Inspect current summary
let summary = memory.current_summary().await;
```

**Defaults:** `max_messages = 20`, `retain_recent = 10`. The built-in summary prompt instructs the model to preserve facts, decisions, tool results, and context.

**When to use:** Long conversations where you want to preserve context without consuming the full token budget. Adds LLM cost for summarization.

---

### SqliteMemory

Persists messages to SQLite. Survives process restarts. Requires the `sqlite` feature.

```rust
#[cfg(feature = "sqlite")]
use daimon::memory::SqliteMemory;

// Persistent file
let memory = SqliteMemory::open("./conversations.db").await?;

// In-memory (data lost when dropped)
let memory = SqliteMemory::in_memory().await?;

// Multiple sessions via session_id
let memory = SqliteMemory::open("./db").await?
    .with_session_id("user-123");
```

**When to use:** Development, low-concurrency production, single-process deployments. Uses `spawn_blocking` for DB operations to avoid blocking the async runtime.

---

### RedisMemory

Stores messages in a Redis list. Supports distributed or multi-instance deployments. Requires the `redis` feature.

```rust
#[cfg(feature = "redis")]
use daimon::memory::RedisMemory;

let memory = RedisMemory::new("redis://127.0.0.1/", "conversation:abc123").await?;

let agent = Agent::builder()
    .model(model)
    .memory(memory)
    .build()?;
```

**When to use:** Multi-process or multi-instance setups where conversation state must be shared. Each `RedisMemory` instance uses a key; use different keys per session.

---

## Hooks

The `AgentHook` trait provides lifecycle callbacks. All methods have default no-op implementations; override only what you need.

### AgentHook Methods

| Method | When Called |
|--------|-------------|
| `on_iteration_start(&self, state: &AgentState)` | Start of each iteration, before model call |
| `on_model_response(&self, response: &ChatResponse)` | After model returns, before tool execution or final output |
| `on_tool_call(&self, call: &ToolCall)` | Before each tool is executed |
| `on_tool_result(&self, call: &ToolCall, result: &ToolOutput)` | After each tool completes |
| `on_iteration_end(&self, state: &AgentState)` | End of iteration, after tools run or final response |
| `on_error(&self, error: &DaimonError)` | When a tool execution fails (error still propagated to model) |

### AgentState

```rust
pub struct AgentState {
    pub iteration: usize,      // 1-based, increments each model call
    pub max_iterations: usize,
}
```

### Example: Logging Hook

```rust
use daimon::hooks::{AgentHook, AgentState};

struct LoggingHook;

impl AgentHook for LoggingHook {
    async fn on_iteration_start(&self, state: &AgentState) -> Result<()> {
        tracing::info!(iteration = state.iteration, "ReAct iteration started");
        Ok(())
    }

    async fn on_model_response(&self, response: &ChatResponse) -> Result<()> {
        if let Some(ref u) = response.usage {
            tracing::info!(
                input_tokens = u.input_tokens,
                output_tokens = u.output_tokens,
                "model response"
            );
        }
        Ok(())
    }

    async fn on_tool_call(&self, call: &ToolCall) -> Result<()> {
        tracing::info!(tool = %call.name, id = %call.id, "tool call");
        Ok(())
    }
}

let agent = Agent::builder()
    .model(model)
    .hooks(LoggingHook)
    .build()?;
```

### Example: Metrics Hook

```rust
use std::sync::atomic::{AtomicUsize, Ordering};

struct MetricsHook {
    tool_calls: AtomicUsize,
}

impl AgentHook for MetricsHook {
    async fn on_tool_call(&self, _call: &ToolCall) -> Result<()> {
        self.tool_calls.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}
```

### Example: Checkpoint Hook

```rust
impl AgentHook for CheckpointHook {
    async fn on_iteration_end(&self, state: &AgentState) -> Result<()> {
        if state.iteration % 5 == 0 {
            // Persist state every 5 iterations
            self.checkpoint.save(&self.current_state()).await?;
        }
        Ok(())
    }
}
```

---

## Middleware

Unlike hooks, middleware can **mutate** requests and responses and **short-circuit** the pipeline. Middleware runs in registration order; the first non-`Continue` action stops the pipeline.

### Middleware Trait

```rust
pub trait Middleware: Send + Sync {
    fn on_request(&self, request: &mut ChatRequest) -> impl Future<Output = Result<MiddlewareAction>> + Send;
    fn on_response(&self, response: &mut ChatResponse) -> impl Future<Output = Result<MiddlewareAction>> + Send;
    fn on_tool_call(&self, call: &mut ToolCall) -> impl Future<Output = Result<MiddlewareAction>> + Send;
}

pub enum MiddlewareAction {
    Continue,
    ShortCircuit(ChatResponse),
}
```

### Example: Logging Middleware

```rust
use daimon::middleware::{Middleware, MiddlewareAction};

struct LoggingMiddleware;

impl Middleware for LoggingMiddleware {
    async fn on_request(&self, request: &mut ChatRequest) -> Result<MiddlewareAction> {
        tracing::info!(messages = request.messages.len(), "model request");
        Ok(MiddlewareAction::Continue)
    }

    async fn on_response(&self, response: &mut ChatResponse) -> Result<MiddlewareAction> {
        tracing::info!(text_len = response.text().len(), "model response");
        Ok(MiddlewareAction::Continue)
    }
}
```

### Example: Rate Limiting (Short-Circuit)

```rust
impl Middleware for RateLimitMiddleware {
    async fn on_request(&self, request: &mut ChatRequest) -> Result<MiddlewareAction> {
        if !self.limiter.try_acquire().await {
            return Ok(MiddlewareAction::ShortCircuit(ChatResponse {
                message: Message::assistant("Rate limit exceeded. Please try again later."),
                stop_reason: StopReason::EndTurn,
                usage: None,
            }));
        }
        Ok(MiddlewareAction::Continue)
    }
}
```

### Example: Content Injection

```rust
impl Middleware for InjectContextMiddleware {
    async fn on_request(&self, request: &mut ChatRequest) -> Result<MiddlewareAction> {
        request.messages.insert(
            1,
            Message::system("Additional context: the user is a premium subscriber."),
        );
        Ok(MiddlewareAction::Continue)
    }
}
```

### MiddlewareStack Execution Order

Layers run in the order they were pushed. For `on_request`: first layer runs, then second, etc. The first `ShortCircuit` returned stops the rest of the stack and the model is not called. Same for `on_response` and `on_tool_call`.

---

## Guardrails

Guardrails validate input (before the model sees it) and output (before returning to the caller). They can **Pass**, **Block** (with error message), or **Transform** (replace content).

### GuardrailResult

```rust
pub enum GuardrailResult {
    Pass,
    Block(String),
    Transform(String),
}
```

- **Input guardrails:** `Transform(String)` replaces the user input for the rest of the pipeline.
- **Output guardrails:** `Transform(String)` replaces `response.final_text` in the `AgentResponse`.

### Built-in Guardrails

#### MaxTokenGuardrail

Rejects input whose estimated token count exceeds a limit. Uses ~4 chars per token.

```rust
use daimon::guardrails::MaxTokenGuardrail;

let guard = MaxTokenGuardrail::new(4096);
agent.input_guardrail(guard);
```

#### RegexFilterGuardrail

Blocks or redacts input matching regex patterns. `block` and `redact` are
fallible: an invalid regex returns an error instead of being silently
dropped, so a typo'd filter can never fail open.

```rust
use daimon::guardrails::RegexFilterGuardrail;

// Block when pattern matches
let guard = RegexFilterGuardrail::new()
    .block(r"(?i)password\s*[:=]", "potential credential leak")?;

// Redact matched text
let guard = RegexFilterGuardrail::new()
    .redact(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN REDACTED]")?;
```

#### ContentPolicyGuardrail

Uses an LLM to evaluate content against a policy. The model must respond with `PASS` or `BLOCK: <reason>`.

```rust
use daimon::guardrails::ContentPolicyGuardrail;

let guard = ContentPolicyGuardrail::new(
    model.clone(),
    "No hate speech, threats, or illegal content.",
);

agent.input_guardrail(guard.clone());
agent.output_guardrail(guard);
```

---

## Cost Tracking

Attach a `CostModel` to track token spend. Set `max_budget` to abort when a dollar limit is reached.

### CostModel Trait

```rust
pub trait CostModel: Send + Sync {
    fn cost_per_token(&self, model_id: &str, direction: TokenDirection) -> f64;
}

pub enum TokenDirection {
    Input,
    Output,
}
```

### Built-in Models

- **OpenAiCostModel** — GPT-4o, GPT-4, GPT-3.5, o1/o3 pricing (approximate, early 2026).
- **AnthropicCostModel** — Claude Opus, Sonnet, Haiku pricing.

### CostTracker

Created automatically when you set `cost_model`. Tracks cumulative cost across iterations. `record(model_id, usage)` adds to the total; `cumulative_cost()` returns USD spent. Reset at the start of each `prompt` call.

### Example

```rust
use daimon::cost::{OpenAiCostModel, AnthropicCostModel};

let agent = Agent::builder()
    .model(model)
    .cost_model(OpenAiCostModel)
    .max_budget(0.50)  // $0.50 per prompt
    .build()?;

let response = agent.prompt("Explain quantum computing").await?;
println!("Cost: ${:.6}", response.cost);
```

### Streaming Cost

In `prompt_stream`, token counts are **estimated** from character length (~4 chars/token). Each iteration emits `StreamEvent::Usage { iteration, input_tokens, output_tokens, estimated_cost }`. The `estimated_cost` is computed via the agent's `CostTracker` if configured.

---

## Prompt Templates

### PromptTemplate

String templates with `{variable}` interpolation. Variables are replaced at render time; unknown variables remain literal.

```rust
use daimon::prompt::PromptTemplate;

let tpl = PromptTemplate::new("You are {role}. Today is {date}.")
    .var("role", "a helpful assistant")
    .var("date", "2026-03-04");

let rendered = tpl.render_static();
// "You are a helpful assistant. Today is 2026-03-04."

// Override at render time
let overrides = [("date".into(), "2026-03-05".into())].into_iter().collect();
let rendered = tpl.render_with(&overrides);
```

### PromptBuilder

Fluent API for composing prompts from sections: persona, instructions, constraints, examples.

```rust
use daimon::prompt::PromptBuilder;

let tpl = PromptBuilder::new()
    .persona("You are an expert Rust developer.")
    .instruction("Answer concisely.")
    .constraint("Never reveal internal implementation details.")
    .example("Q: What is ownership?\nA: Ownership is Rust's memory management model.")
    .build();
```

### DynamicContext

For async resolution of variables (e.g., current date, user profile, DB lookup). Implement `DynamicContext` and pass to `render_dynamic`.

```rust
use daimon::prompt::{DynamicContext, PromptTemplate};

struct CurrentDate;

impl DynamicContext for CurrentDate {
    fn key(&self) -> &str { "date" }
    async fn resolve(&self) -> String {
        chrono::Local::now().format("%Y-%m-%d").to_string()
    }
}

let tpl = PromptTemplate::new("Today is {date}.");
let rendered = tpl.render_dynamic(&[&CurrentDate]).await;
```

### FewShotTemplate

Injects example input/output pairs for in-context learning.

```rust
use daimon::prompt::FewShotTemplate;

let tpl = FewShotTemplate::new()
    .example("What is 2+2?", "4")
    .example("Capital of France?", "Paris")
    .with_prefix("Here are some examples:");

let rendered = tpl.render();
```

---

## Streaming

`prompt_stream` returns a `ResponseStream` that emits `StreamEvent`s as the model generates. The stream runs the full ReAct loop: tool call deltas are accumulated, tools execute when complete, and the model is re-invoked — all within the same stream.

### StreamEvent Variants

| Variant | Description |
|---------|-------------|
| `TextDelta(String)` | Chunk of generated text |
| `ToolCallStart { id, name }` | Tool call starting; arguments pending |
| `ToolCallDelta { id, arguments_delta }` | Chunk of tool call arguments (JSON fragment) |
| `ToolCallEnd { id }` | Tool call arguments complete; tool will execute |
| `ToolResult { id, content, is_error }` | Tool execution result |
| `Usage { iteration, input_tokens, output_tokens, estimated_cost }` | Token usage for the iteration (estimated in streaming) |
| `Error(String)` | Non-fatal error; stream may continue |
| `Done` | Stream complete |

### Consuming the Stream

```rust
use futures::StreamExt;

let mut stream = agent.prompt_stream("What is 2+2?").await?;

while let Some(event) = stream.next().await {
    let event = event?;
    match &event {
        StreamEvent::TextDelta(text) => print!("{text}"),
        StreamEvent::ToolCallStart { name, .. } => eprintln!("\n[calling: {name}]"),
        StreamEvent::ToolCallDelta { .. } => {}
        StreamEvent::ToolCallEnd { .. } => {}
        StreamEvent::ToolResult { content, .. } => eprintln!("\n[result: {content}]"),
        StreamEvent::Usage { iteration, estimated_cost, .. } => {
            eprintln!("\n[iteration {iteration}, cost ${estimated_cost:.6}]");
        }
        StreamEvent::Error(msg) => eprintln!("\n[error: {msg}]"),
        StreamEvent::Done => { println!(); break; }
    }
}
```

### Usage Estimation in Streaming

During streaming, token counts are **estimated** from character length (~4 chars/token). The `Usage` event's `input_tokens` and `output_tokens` are estimates. `estimated_cost` uses the agent's `CostModel` if configured; otherwise it is 0.

---

## AgentResponse

The result of `agent.prompt(...)` (and `prompt_with_messages`).

```rust
pub struct AgentResponse {
    /// Full message log (system + history + user + all iterations)
    pub messages: Vec<Message>,
    /// Final text from the model
    pub final_text: String,
    /// Number of model invocations
    pub iterations: usize,
    /// Aggregated token usage (if providers reported it)
    pub usage: Usage,
    /// Estimated cost in USD (requires CostModel on agent)
    pub cost: f64,
}

impl AgentResponse {
    pub fn text(&self) -> &str {
        &self.final_text
    }
}
```

### Fields

- **`messages`** — Complete conversation including system prompt, history, user message, assistant messages (with tool calls), tool results, and final response.
- **`final_text`** — The model's last text response (no tool calls). Use `text()` for convenient access.
- **`iterations`** — How many times the model was invoked. 1 = no tool calls; 2+ = one or more tool rounds.
- **`usage`** — `Usage { input_tokens, output_tokens, cached_tokens }` aggregated across all iterations.
- **`cost`** — USD cost from the `CostTracker` if a `CostModel` was configured; otherwise 0.

---

## Quick Reference

| Need | Use |
|------|-----|
| Static system prompt | `system_prompt("...")` |
| Dynamic system prompt | `prompt_template(PromptTemplate::new("...").var(...))` |
| Share model across agents | `shared_model(Arc::clone(&model))` |
| Pre-compile tool validators | Automatic via `warm_cache` at build |
| In-memory, N messages | `memory(SlidingWindowMemory::new(N))` |
| Token budget | `memory(TokenWindowMemory::new(budget))` |
| Long conversations | `memory(SummaryMemory::new(model))` |
| Persistent sessions | `memory(SqliteMemory::open(path).await?)` |
| Distributed sessions | `memory(RedisMemory::new(url, key).await?)` |
| Observe execution | `hooks(MyHook)` |
| Mutate/short-circuit | `middleware(MyMiddleware)` |
| Block long input | `input_guardrail(MaxTokenGuardrail::new(4096))` |
| Redact PII | `input_guardrail(RegexFilterGuardrail::new().redact(...)?)` |
| LLM content policy | `input_guardrail(ContentPolicyGuardrail::new(model, policy))` |
| Track spend | `cost_model(OpenAiCostModel).max_budget(0.50)` |
| Retry transient tool failures | `tool_retry_policy(ToolRetryPolicy::exponential(3))` |
| Human approval | `human_input(MyHandler)` |
| Streaming | `agent.prompt_stream("...").await?` |
| Cancellation | `agent.prompt_with_cancellation("...", &cancel).await?` |
