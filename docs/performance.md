# Performance and Optimization Guide

This guide helps experienced developers squeeze maximum performance from the Daimon Rust AI agent framework. It covers feature flags, memory strategies, tool registry optimization, middleware hot paths, streaming trade-offs, parallel execution, connection pooling, cost control, token reduction, benchmarks, and a production deployment checklist.

---

## Feature Flags: Compile Only What You Need

**The single biggest optimization.** Default features pull in `reqwest` for OpenAI, Anthropic, and Ollama. If you only need Bedrock, disable defaults and enable only what you use:

```toml
[dependencies]
daimon = { version = "0.23", default-features = false, features = ["bedrock", "macros"] }
```

### Default vs Minimal

| Configuration | Features | Typical Impact |
|---------------|----------|----------------|
| Default | `openai`, `anthropic`, `ollama`, `macros` | Full reqwest, 3 provider crates |
| Bedrock-only | `bedrock`, `macros` | No reqwest in main crate; Bedrock uses aws-sdk |
| Minimal core | `macros` only | Traits + macros; bring your own model |

### Feature Flag Reference

| Feature | Pulls In | Use When |
|---------|---------|----------|
| `openai` | reqwest | OpenAI Chat/Embeddings |
| `anthropic` | reqwest | Anthropic Claude |
| `ollama` | reqwest | Local Ollama |
| `gemini` | daimon-provider-gemini | Google Gemini |
| `azure` | daimon-provider-azure | Azure OpenAI |
| `bedrock` | daimon-provider-bedrock | AWS Bedrock |
| `mcp` | reqwest, tokio-tungstenite | MCP client/server |
| `sqlite` | rusqlite | SqliteMemory |
| `redis` | redis | RedisMemory |
| `otel` | opentelemetry, tracing-opentelemetry | Observability |
| `pgvector` | daimon-plugin-pgvector | RAG vector store |
| `opensearch` | daimon-plugin-opensearch | OpenSearch vector store |

### Binary Size and Compile Time

- **Fewer features** → smaller binary, faster incremental builds.
- **`full` feature** → enables everything; use only for integration tests or all-in-one binaries.

**Measure impact:**

```bash
# Default (openai, anthropic, ollama, macros)
cargo build --release
ls -la target/release/libdaimon.rlib  # or your binary

# Bedrock-only
cargo build --release --no-default-features --features "bedrock,macros"

# Clean compile time comparison
cargo clean && time cargo build --release
cargo clean && time cargo build --release --no-default-features --features "bedrock,macros"
```

Expect fewer dependencies and faster compile times with minimal features. The `reqwest` crate alone adds ~30+ transitive dependencies; removing it when unused reduces both compile time and binary size.

---

## Memory Strategy

### Choosing the Right Memory

| Implementation | Best For | Overhead | Persistence |
|----------------|----------|----------|-------------|
| `SlidingWindowMemory(50)` | Most cases | Minimal | None |
| `TokenWindowMemory(4000)` | Large context windows | Low | None |
| `SummaryMemory` | Long conversations | 1 LLM call per summarization | None |
| `SqliteMemory` | Persistence | I/O per access | SQLite |
| `RedisMemory` | Distributed persistence | Network + I/O | Redis |

### SlidingWindowMemory

- **Fast, fixed-size, no external deps.** Keeps the most recent N messages in a `VecDeque`.
- Default: 50 messages. Evicts oldest when exceeded.
- **Benchmark:** ~7.3 µs for add 100 + get 50 messages.

```rust
use daimon::memory::SlidingWindowMemory;

let memory = SlidingWindowMemory::new(50);
```

### TokenWindowMemory

- Better for large context windows. Evicts by token budget, not message count.
- Default token estimator: ~4 chars/token. Use `with_token_counter()` for accuracy (e.g., tiktoken-rs).
- **Benchmark:** ~12 µs for 100 messages within 1000-token budget.

```rust
use daimon::memory::TokenWindowMemory;

let memory = TokenWindowMemory::new(4000);

// Custom tokenizer for accuracy (Message.content is Option<String>)
let memory = TokenWindowMemory::new(4000)
    .with_token_counter(|msg| my_tokenizer.count(msg.content.as_deref().unwrap_or("")));
```

### SummaryMemory

- Best for long conversations. Summarizes old messages via LLM instead of dropping them.
- **Cost:** One LLM call per summarization. Configure `retain_recent` to keep the last N messages verbatim.
- Use `with_max_messages()` and `with_retain_recent()` to tune when summarization triggers.

```rust
use daimon::memory::SummaryMemory;

let memory = SummaryMemory::new(model)
    .with_max_messages(20)
    .with_retain_recent(10);
```

### SqliteMemory / RedisMemory

- Only when you need persistence across restarts or processes.
- Adds I/O overhead on every `add_message` and `get_messages`.
- Prefer in-memory strategies for single-session or stateless agents.

### Memory + Performance

- Memory is accessed **every iteration** of the ReAct loop. Keep it fast.
- `SlidingWindowMemory` uses `VecDeque::make_contiguous().to_vec()` for O(n) retrieval; acceptable for typical window sizes (50–200 messages).
- **Don't put 100k messages in memory.** Use summarization or token windowing.
- For RAG pipelines, keep retrieved context in the request, not in long-term memory.

### Memory Selection Decision Tree

```
Need persistence across restarts? → SqliteMemory or RedisMemory
Long conversation (100+ messages)? → SummaryMemory (with retain_recent)
Strict token budget (e.g., 4K context)? → TokenWindowMemory
Otherwise → SlidingWindowMemory(50)
```

---

## Tool Registry Optimization

### warm_cache()

- Called at build time (when constructing the agent). Pre-compiles JSON Schema validators and caches tool specs.
- **Generation-based invalidation:** Specs are recomputed only when tools are registered or unregistered.

### Benchmark: Tool Registry (50 tools)

| Operation | Time | Notes |
|-----------|------|-------|
| `tool_specs()` | ~15 ns | After first call — the cache is lazy (interior mutability), so even an "uncached" registry only pays spec generation once |
| `validate_input()` | ~13 ns | With pre-compiled validators (`warm_cache()`) |
| `get("tool_25")` | ~10 ns | HashMap lookup |

Spec caching no longer needs `warm_cache()`: `tool_specs()` populates the cache on first call even through `&self`, and the ReAct loop calls it once per `prompt()` run (outside the iteration loop), so only the first call pays spec generation. What genuinely requires `warm_cache()` (or `compile_validators()`) is JSON Schema **validator** precompilation: without it, `validate_input()` compiles the schema on every call. `Agent::builder().build()` calls `warm_cache()` for you, so the uncached path only matters for hand-rolled `ToolRegistry` usage.

### Usage

When using `Agent::builder()`, `warm_cache()` is called automatically at `build()` time:

```rust
let agent = Agent::builder()
    .model(model)
    .tool(tool1)
    .tool(tool2)
    .build()?;  // warm_cache() runs here
```

If you construct a `ToolRegistry` outside the builder (e.g., for MCP server or shared tool sets), call `warm_cache()` before using it in hot paths:

```rust
let mut registry = ToolRegistry::new();
registry.register(tool1)?;
registry.register(tool2)?;
registry.warm_cache();
```

---

## Middleware Hot Path

- **Empty middleware stack:** Zero overhead. An early-return check (`layers.is_empty()`) skips all layers.
- Middleware runs on **every iteration.** Keep `on_request`, `on_response`, and `on_tool_call` lightweight.
- **Avoid allocations** in middleware. Use `&mut ChatRequest` to modify in place.
- Short-circuit via `MiddlewareAction::ShortCircuit(ChatResponse)` to skip the model call when appropriate.

```rust
impl Middleware for LoggingMiddleware {
    async fn on_request(&self, request: &mut ChatRequest) -> Result<MiddlewareAction> {
        tracing::info!(messages = request.messages.len(), "request");
        Ok(MiddlewareAction::Continue)
    }
}
```

---

## Streaming vs Non-Streaming

| Method | Overhead | Use Case |
|--------|----------|----------|
| `prompt()` | Lower | Batch, background, serverless |
| `prompt_stream()` | async-stream + channel | Interactive UIs, real-time feedback |

- `prompt()` is simpler and slightly more efficient (no stream machinery).
- `prompt_stream()` adds async-stream and channel overhead but gives real-time tokens.
- **Use `prompt()`** for batch/background processing.
- **Use `prompt_stream()`** for interactive UIs where latency-to-first-token matters.

---

## Parallel Tool Execution

- Tools within a **single iteration** run in parallel via `tokio::task::JoinSet` (non-streaming `prompt()`; the streaming loop in `prompt_stream()` executes them sequentially).
- If your tools are **I/O bound** (API calls, DB queries), this is a significant win. Multiple tool calls in one turn complete in roughly the time of the slowest call, not the sum.
- If tools are **CPU-bound**, consider `tokio::task::spawn_blocking` inside the tool to avoid starving the async runtime:

```rust
async fn execute(&self, input: &Value) -> Result<ToolOutput> {
    let data = input.clone();
    let result = tokio::task::spawn_blocking(move || {
        heavy_computation(&data)
    }).await?;
    Ok(ToolOutput::json(&result)?)
}
```

- For CPU-heavy workloads, a dedicated `tokio::runtime::Builder` with a separate thread pool can isolate blocking work from the main agent loop.

---

## Connection Pooling

| Component | Pool Implementation | Default | Tuning |
|-----------|---------------------|---------|--------|
| pgvector | `deadpool-postgres` | 16 | `pool_size(n)` |
| OpenSearch | `opensearch` crate (reqwest) | Built-in | Transport config |
| Redis | `redis` ConnectionManager | Multiplexed, auto-reconnect | Shared per `RedisMemory` |
| HTTP providers | reqwest | Built-in | Per-client, shared across requests |

### pgvector

```rust
let store = PgVectorStoreBuilder::new("postgresql://...", 1536)
    .pool_size(16)  // Tune for concurrency
    .build()
    .await?;
```

### HTTP Providers

OpenAI, Anthropic, Gemini, Azure, and the local providers all use `reqwest`, which maintains connection pooling per client (Bedrock uses the AWS SDK instead, with its own connection management). **Reuse the same model instance** across requests—do not create a new `OpenAi` or `Anthropic` for every prompt.

```rust
// Good: single model, shared across requests
let model = Arc::new(OpenAi::new("gpt-4o").with_timeout(Duration::from_secs(60)));
let agent = Agent::builder().shared_model(Arc::clone(&model)).build()?;
// Reuse agent for many prompts
```

### Redis

The `redis` crate is enabled with `tokio-comp`, `aio`, and `connection-manager` features. `RedisMemory` (and the Redis broker/checkpoint stores) use `redis::aio::ConnectionManager` — a single multiplexed, auto-reconnecting connection — so connection management is built in; no external pool is needed.

---

## Cost Control

- **`max_budget`** — Set a dollar limit per prompt. Checked every iteration; aborts with `DaimonError::BudgetExceeded`.
- **`CostTracker`** — A fresh tracker is created for each `prompt()` run, so cost is per-prompt. Sum `AgentResponse.cost` for session-level tracking.
- **Streaming:** `StreamEvent::Usage { iteration, input_tokens, output_tokens, estimated_cost }` gives per-iteration estimates.
- **Streaming cost** is estimated (input_chars/4 + output_chars/4). Non-streaming cost is exact from the API `usage` field.

```rust
let agent = Agent::builder()
    .model(OpenAi::new("gpt-4o"))
    .cost_model(OpenAiCostModel)
    .max_budget(0.50)
    .build()?;
```

### Cost Tracking Across Multiple Prompts

Each `prompt()` call gets a fresh `CostTracker` seeded from the shared cost model, so cost is tracked per prompt (and concurrent prompts account spend independently). For session-level or batch-level tracking, sum `AgentResponse.cost` from each call:

```rust
let agent = Agent::builder()
    .model(model)
    .cost_model(OpenAiCostModel)
    .max_budget(0.50)  // Per-prompt limit
    .build()?;

let mut session_cost = 0.0;
for task in tasks {
    let response = agent.prompt(&task).await?;
    session_cost += response.cost;
    println!("Prompt cost: ${:.4}, session total: ${:.4}", response.cost, session_cost);
}
```

### Streaming Cost Estimates

In `prompt_stream()`, `StreamEvent::Usage` is emitted after each ReAct iteration. The `estimated_cost` uses character-based token estimation (~4 chars/token) when the API does not return exact counts. Note the streaming path yields no `AgentResponse` — for precise totals, sum the `estimated_cost` across `Usage` events, or use `prompt()` and read `AgentResponse.cost`.

---

## Reducing Token Usage

| Strategy | Impact |
|----------|--------|
| Concise system prompts | High — sent every iteration |
| `PromptTemplate` with variables | Medium — avoid string concatenation |
| Few-shot examples | High — each example costs tokens every iteration |
| Tool count | High — tool descriptions are sent every request |
| `SummaryMemory` | High — compresses old history |

### Practical Tips

1. **System prompts:** Keep them short. Every token is billed every iteration.

2. **`PromptTemplate`:** Use `{variable}` placeholders instead of string concatenation:

```rust
let tpl = PromptTemplate::new("You are {role}. Today is {date}.")
    .var("role", "a helpful assistant")
    .var("date", "2025-03-04");
let agent = Agent::builder()
    .model(model)
    .prompt_template(tpl)
    .build()?;
```

3. **`FewShotTemplate`:** Use judiciously. Each example is sent on every request. Prefer 1–3 high-quality examples over many mediocre ones.

4. **Tools:** Only register tools the agent actually needs. Each tool's `name`, `description`, and `parameters_schema` are sent every request. For a 20-tool agent, tool metadata can consume thousands of tokens per call.

5. **`SummaryMemory`:** For long conversations, summarization compresses old history into a single system message. Tune `retain_recent` to balance context freshness vs. token savings.

### Token Budget Rule of Thumb

For a 128K context model with a 4K system prompt and 10 tools (~2K tokens): you have ~122K tokens for conversation. At ~4 chars/token, that's ~488K characters of history. `TokenWindowMemory::new(120_000)` keeps you within budget.

---

## Benchmarks

The framework includes a `criterion` benchmark suite. Run with:

```bash
cargo bench
```

### Key Results (v0.22.3)

| Benchmark | Time | Notes |
|-----------|------|-------|
| `agent_prompt_simple` | ~2.0 µs | Mock model, no network |
| `agent_prompt_with_tools` | ~2.0 µs | One tool, mock model |
| `memory_sliding_window_50` | ~7.8 µs | Add 100, get 50 |
| `memory_token_window_1000` | ~11 µs | Add 100, get within budget |
| `tool_registry_specs_50_uncached` | ~15 ns | Lazy cache: only the first call pays generation |
| `tool_registry_specs_50_cached` | ~15 ns | Cached read |
| `tool_registry_validate_cached` | ~13 ns | Pre-compiled validators |
| `tool_registry_lookup_50` | ~10 ns | HashMap lookup |
| `chain_3_transforms` | ~212 ns | 3-step chain |
| `dag_fan_out_3_merge` | ~8.8 µs | 3-way fan-out + merge |
| `hot_swap_prompt_simple` | ~1.9 µs | HotSwapAgent overhead |
| `hot_swap_swap_model` | ~182 ns | Model swap |
| `broker_submit_receive_complete` | ~633 ns | InProcessBroker round-trip |
| `event_bus_publish_receive` | ~136 ns | Streaming event bus |
| `checkpoint_save_load_memory` | ~291 ns | In-memory checkpoint baseline |
| `stream_event_serialize` / `stream_event_deserialize` | ~56 ns / ~70 ns | `SerializableStreamEvent` JSON round-trip |
| `token_estimate_4500_chars` | ~183 ps | ~4 chars/token estimator |

*Note: Agent benchmarks use a mock model with no network I/O. Real latency is dominated by LLM API calls. Numbers are from a single development machine — re-run `cargo bench` for figures on your hardware.*

---

## Production Deployment Checklist

### Feature Flags

- [ ] Enable only what you use; disable defaults if using a single provider
- [ ] Avoid `full` feature in production unless you need every integration

### Memory

- [ ] Choose the right strategy: `SlidingWindowMemory` for short chats, `TokenWindowMemory` for context limits, `SummaryMemory` for long conversations
- [ ] Don't use `SqliteMemory` or `RedisMemory` unless you need persistence

### Budget and Cost

- [ ] Set `max_budget` to prevent cost overruns
- [ ] Attach a `CostModel` when using paid APIs
- [ ] For batch jobs, sum `AgentResponse.cost` across prompts for session-level tracking

### Timeouts and Retries

- [ ] Configure per-provider timeouts: `model.with_timeout(Duration::from_secs(60))`
- [ ] Set `tool_retry_policy` for flaky external tools:

```rust
.tool_retry_policy(
    ToolRetryPolicy::exponential(3)
        .retryable_on(vec!["timeout".into(), "503".into(), "rate limit".into()])
)
```

### Guardrails

- [ ] Add `input_guardrail` for input validation (e.g., `MaxTokenGuardrail`, `RegexFilterGuardrail`)
- [ ] Add `output_guardrail` for output filtering (e.g., PII redaction, content policy)
- [ ] For Bedrock, consider `with_guardrail(id, version)` for native content filtering

### Observability

- [ ] Enable `otel` feature for OpenTelemetry export
- [ ] Use `tracing` spans—the framework instruments agent iterations, tool calls, and model requests
- [ ] Log `DaimonError` variants for debugging

### Checkpointing

- [ ] Enable checkpointing for long-running agents (resumable runs, crash recovery)
- [ ] Use `InMemoryCheckpoint` for single-process; `FileCheckpoint` or custom backend for persistence

### Error Handling

- [ ] Handle all `DaimonError` variants:
  - `BudgetExceeded { spent, limit }` — cost limit hit
  - `MaxIterations(n)` — agent loop exceeded iteration cap
  - `Cancelled` — user or timeout cancelled the run
  - `ToolExecution { tool, message }` — tool returned an error
  - `GuardrailBlocked(msg)` — input or output guardrail rejected
  - `Model(msg)`, `Serialization(_)`, `Storage { message, transient }`, etc. (the enum is `#[non_exhaustive]` — include a wildcard arm)

```rust
match agent.prompt(input).await {
    Ok(r) => println!("{}", r.final_text),
    Err(DaimonError::BudgetExceeded { spent, limit }) => {
        tracing::warn!(spent, limit, "budget exceeded");
    }
    Err(DaimonError::MaxIterations(n)) => {
        tracing::warn!(n, "max iterations reached");
    }
    Err(e) => return Err(e.into()),
}
```

---

## Quick Reference

| Concern | Recommendation |
|---------|----------------|
| Binary size | `default-features = false`, enable only needed features |
| Memory speed | `SlidingWindowMemory` or `TokenWindowMemory` for most cases |
| Tool registry | `warm_cache()` after registration (automatic in `Agent::builder`) |
| Middleware | Keep hooks lightweight; empty stack has zero overhead |
| Batch jobs | Use `prompt()` instead of `prompt_stream()` |
| I/O-bound tools | Parallel execution is automatic via `JoinSet` |
| Cost | `max_budget` + `CostTracker` |
| Tokens | Short system prompts, minimal tools, `SummaryMemory` for long chats |

---

## Common Pitfalls

1. **Creating a new model per request** — Reuse the same `OpenAi`/`Anthropic`/etc. instance. Each instance has its own reqwest client and connection pool.

2. **Registering every possible tool** — Tool descriptions and schemas are sent on every request. Register only what the agent needs for the current task. Use `ForkBuilder` to create task-specific agents with different tool sets.

3. **Ignoring `warm_cache()` when building registries manually** — If you construct a `ToolRegistry` outside of `Agent::builder`, call `warm_cache()` before passing it in. The builder does this automatically when you use `.tool()`.

4. **Heavy middleware** — Middleware runs every iteration. Avoid allocations, network calls, or expensive computations. Use hooks for observability; use middleware only when you need to mutate or short-circuit.

5. **SummaryMemory without tuning** — Default `max_messages` (20) and `retain_recent` (10) may not fit your use case. For very long conversations, increase `max_messages` to reduce summarization frequency; for more context, increase `retain_recent`.

6. **Skipping `max_budget` in production** — Without a budget, a misbehaving agent or prompt can incur unbounded cost. Always set `max_budget` when using paid APIs.
