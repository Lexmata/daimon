# Daimon TODO

## v0.1.0 -- Gaps in Current Implementation

### Observability
- [x] Add `tracing::instrument` spans to `Agent::prompt()` and `Agent::prompt_stream()`
- [x] Add `tracing` spans around each model provider's `generate()` / `generate_stream()` (OpenAI, Anthropic, Bedrock)
- [x] Add `tracing` spans around tool execution in the ReAct loop
- [x] Emit structured span fields: model_id, tool_name, iteration, input/output token counts

### Streaming
- [x] Implement full streaming ReAct loop (currently `prompt_stream()` only forwards the first model call — it does not handle tool calls within the stream and re-invoke the model)
- [x] Accumulate streamed tool call deltas into complete `ToolCall` objects before executing
- [x] Emit `StreamEvent::ToolCallEnd` after tool execution completes
- [x] Emit error events through the stream instead of only via `Result`

### Agent
- [x] Support cancellation (accept a `CancellationToken` or `tokio::select!` pattern)
- [x] Expose `AgentResponse.usage` aggregated across all iterations
- [x] Save the assistant's tool-call messages to memory (currently only the final assistant message is saved)
- [x] Add `Agent::prompt_with_messages()` to accept pre-built `Vec<Message>` instead of only `&str`

### Error Handling
- [x] Add `DaimonError::Timeout` variant for request timeouts
- [x] Add `DaimonError::Cancelled` variant for cancellation
- [x] Add retry logic with configurable backoff for transient model errors (429, 5xx)

### Model Providers
- [x] OpenAI: support `response_format` (JSON mode / structured output)
- [x] OpenAI: support `parallel_tool_calls` option
- [x] Anthropic: support `cache_control` (prompt caching)
- [x] Anthropic: handle `overloaded` error with automatic retry
- [x] Bedrock: support guardrails configuration
- [x] All providers: configurable HTTP timeout
- [x] All providers: configurable max retries

### Tool System
- [ ] JSON Schema validation of tool input before execution
- [x] `ToolOutput::json()` convenience constructor for structured responses
- [x] Parallel tool execution (currently tools within one iteration run sequentially)

### Memory
- [ ] Token-based window (count tokens, not messages) for `SlidingWindowMemory`
- [ ] `SummaryMemory` — summarize old messages instead of dropping them

### Documentation
- [x] Rustdoc on all public types, traits, and methods
- [x] Module-level documentation for each module
- [x] Top-level crate documentation with usage examples in `lib.rs`

### CI / Quality
- [x] Set up GitHub Actions workflow (check, clippy, test, coverage gate)
- [x] Add `deny.toml` for `cargo-deny` (license + advisory audit)
- [x] Pre-commit hook that runs `cargo fmt --check` and `cargo clippy`
- [x] Ensure all examples compile-check in CI (feature-gated)

---

## v0.2.0 -- Orchestration & MCP

- [ ] Chain orchestration (linear pipelines of agents/transforms)
- [ ] Graph orchestration (conditional routing, cycles, fan-out/fan-in)
- [ ] MCP client (connect to external tool servers via stdio/SSE/streamable HTTP)
- [ ] Human-in-the-loop (interrupt agent loop, present choices, resume)
- [ ] `daimon-macros` crate with `#[tool]` proc macro to auto-derive `Tool` from a function
- [ ] SQLite memory backend (`feature = "sqlite"`)
- [ ] Ollama provider (`feature = "ollama"`)
- [ ] Transition to workspace: split into `daimon-core`, `daimon-macros`, provider crates
- [ ] JSON Schema validation of tool input before execution
- [ ] Token-based window (count tokens, not messages) for `SlidingWindowMemory`
- [ ] `SummaryMemory` — summarize old messages instead of dropping them

## v0.3.0 -- Multi-Agent

- [ ] Agent-as-Tool pattern (wrap an `Agent` as a `Tool` for another agent)
- [ ] Supervisor pattern (one agent delegates to specialized sub-agents)
- [ ] Handoff pattern (agents transfer control to each other)
- [ ] MCP server (expose Daimon tools as an MCP server)
- [ ] `Retriever` trait + vector store integration (RAG)
- [ ] Redis memory backend (`feature = "redis"`)
- [ ] Structured output / extraction (typed responses via serde)

## v0.4.0+ -- Production Hardening

- [ ] Workflow orchestration (Eino-style DAG with field mapping)
- [ ] Checkpointing and state persistence (resume interrupted agent runs)
- [ ] A2A protocol support
- [ ] OpenTelemetry export (bridge `tracing` spans to OTLP)
- [ ] Benchmarking suite (latency, throughput, token efficiency)
- [ ] Publish to crates.io as open source
