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
- [x] JSON Schema validation of tool input before execution
- [x] `ToolOutput::json()` convenience constructor for structured responses
- [x] Parallel tool execution (currently tools within one iteration run sequentially)

### Memory
- [x] Token-based window (count tokens, not messages) for `SlidingWindowMemory`
- [x] `SummaryMemory` — summarize old messages instead of dropping them

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

- [x] Chain orchestration (linear pipelines of agents/transforms)
- [x] Graph orchestration (conditional routing, cycles, fan-out/fan-in)
- [x] MCP client (connect to external tool servers via stdio/SSE/streamable HTTP)
- [x] Human-in-the-loop (interrupt agent loop, present choices, resume)
- [x] `daimon-macros` crate with `#[tool_fn]` proc macro to auto-derive `Tool` from a function
- [x] SQLite memory backend (`feature = "sqlite"`)
- [x] Ollama provider (`feature = "ollama"`)
- [x] Transition to workspace: `daimon-macros` proc-macro crate split out
- [x] JSON Schema validation of tool input before execution
- [x] Token-based window (count tokens, not messages) for `SlidingWindowMemory`
- [x] `SummaryMemory` — summarize old messages instead of dropping them

## v0.3.0 -- Multi-Agent

- [x] Agent-as-Tool pattern (wrap an `Agent` as a `Tool` for another agent)
- [x] Supervisor pattern (one agent delegates to specialized sub-agents)
- [x] Handoff pattern (agents transfer control to each other)
- [x] MCP server (expose Daimon tools as an MCP server)
- [x] `Retriever` trait + vector store integration (RAG)
- [x] Redis memory backend (`feature = "redis"`)
- [x] Structured output / extraction (typed responses via serde)

## v0.4.0+ -- Production Hardening

- [x] Workflow orchestration (Eino-style DAG with field mapping)
- [x] Checkpointing and state persistence (resume interrupted agent runs)
- [x] A2A protocol support
- [x] OpenTelemetry export (bridge `tracing` spans to OTLP)
- [x] Benchmarking suite (latency, throughput, token efficiency)
- [x] Publish to crates.io as open source
- [x] Performance optimizations: zero-copy ReAct loop, cached ToolRegistry, VecDeque-based memory, allocation-free token estimation
- [x] Provider plugin interface: split AWS Bedrock, Google Gemini, Azure OpenAI into `daimon-provider-*` sub-crates with trait-based `Model` plugin system via `daimon-core`

## v0.5.0 -- Core Primitives

- [x] Middleware pipeline (`src/middleware/`): `Middleware` trait with `on_request`, `on_response`, `on_tool_call` hooks; `MiddlewareStack` with short-circuit support; object-safe `ErasedMiddleware`; wired into ReAct loop and tool execution
- [x] Guardrails (`src/guardrails/`): `InputGuardrail` and `OutputGuardrail` traits with `Pass`/`Block`/`Transform` results; built-in `MaxTokenGuardrail` and `RegexFilterGuardrail`; integrated into `Agent::prompt()`
- [x] Prompt templates (`src/prompt/`): `PromptTemplate` with `{variable}` interpolation; `PromptBuilder` for composing persona, instructions, constraints, examples; `AgentBuilder::prompt_template()`
- [x] Cost tracking (`src/cost/`): `CostModel` trait; `CostTracker` with lock-free atomic accumulation; built-in `OpenAiCostModel` and `AnthropicCostModel`; `AgentBuilder::max_budget()` with `BudgetExceeded` error; `AgentResponse.cost` field
- [x] Embeddings API (`daimon-core/src/embedding.rs`): `EmbeddingModel` trait with `embed()` and `dimensions()`; `ErasedEmbeddingModel` wrapper; `OpenAiEmbedding` and `OllamaEmbedding` provider implementations

## v0.6.0 -- Ecosystem Integrations

- [x] Vector store integrations: `InMemoryVectorStore` (brute-force cosine similarity); `QdrantRetriever` (`feature = "qdrant"`); all implement existing `Retriever` trait
- [x] Self-healing tool retry (`src/tool/retry.rs`): `ToolRetryPolicy` with `BackoffStrategy` (Fixed, Exponential); retryable error patterns; `AgentBuilder::tool_retry_policy()`; integrated into parallel tool execution
- [x] Deployment helpers (`src/server/`, `feature = "http-server"`): `AgentServer` wrapping agent behind axum; `POST /prompt`, `POST /prompt/stream` (SSE), `GET /health` endpoints

## v0.7.0 -- Production Hardening

- [x] Evaluation harness (`src/eval/`, `feature = "eval"`): `EvalScenario` with `Scorer` strategies (ExactMatch, Contains, Regex, Custom); `EvalRunner` with concurrency; `EvalResult` with pass/fail, latency, cost
- [x] Time-travel debugging (`src/checkpoint/replay.rs`): `inspect_run()` to reconstruct `ExecutionTrace` from checkpoints; `list_runs()` for `RunSummary`; `TraceStep` captures per-iteration state

## v0.8.0 -- Framework Polish

- [x] `ContentPolicyGuardrail` — LLM-as-judge guardrail for content safety (input + output)
- [x] `DynamicContext` trait for runtime prompt variable resolution with `render_dynamic()`
- [x] `FewShotTemplate` for injecting example input/output pairs into prompts
- [x] `SemanticSimilarity` scorer using `EmbeddingModel` for evaluation harness
- [x] `LlmJudge` scorer — LLM-as-judge for evaluation harness
- [x] `Agent::replay()` — re-run from a checkpoint with modified context (time-travel debugging)
- [x] Per-tool retry policy override via `Tool::retry_policy()` (takes precedence over agent-level policy)
- [x] API key authentication middleware for `AgentServer` (`Authorization: Bearer` / `X-API-Key`)
- [x] Additional embedding providers: `GeminiEmbedding`, `AzureOpenAiEmbedding`, `BedrockEmbedding`
- [x] Trait-based `VectorStore` plugin system with `upsert`/`query`/`delete`/`count` operations
- [x] `KnowledgeBase` trait + `SimpleKnowledgeBase` composing `EmbeddingModel` + `VectorStore`
- [x] `InMemoryVectorStoreBackend` implementing `VectorStore` for development/testing
- [x] `SimpleKnowledgeBase` auto-implements `Retriever` for seamless agent integration

## v0.9.0 -- Distribution & Runtime

- [x] Streaming cost tracking: `StreamEvent::Usage { iteration, input_tokens, output_tokens, estimated_cost }` emitted after each ReAct iteration
- [x] Agent cloning / forking: `Agent::fork()`, `Agent::fork_from_checkpoint()`, `Agent::fork_with_memory()` for independent branched agents sharing config
- [x] Distributed execution: `TaskBroker` trait, `InProcessBroker` (tokio channels), `TaskWorker` with sequential and parallel execution, serializable `AgentTask`/`TaskResult`/`TaskStatus`
- [x] WebSocket MCP transport: `WebSocketTransport` implementing `McpTransport` via `tokio-tungstenite` with `ws://`/`wss://` support

## v0.10.0 -- Distributed & Ecosystem Polish

- [x] Redis task broker (`RedisBroker`) for multi-process distributed execution (`feature = "redis"`)
- [x] Builder-style fork (`ForkBuilder`): `Agent::fork_builder()` with system prompt, tool, model, memory, hooks, and guardrail mutation before forking
- [x] WebSocket MCP server (`McpWsServer`): serve tools over WebSocket connections (`feature = "mcp"`)
- [x] gRPC transport for distributed execution: `GrpcBrokerServer` / `GrpcBrokerClient` via `tonic` (`feature = "grpc"`)
- [x] `ToolRegistry::unregister()` for removing tools by name

## v0.11.0 -- Full Distribution Stack

- [x] NATS JetStream task broker (`NatsBroker`) with durable pull consumers (`feature = "nats"`)
- [x] RabbitMQ task broker (`AmqpBroker`) via AMQP 0-9-1 with manual ack (`feature = "amqp"`)
- [x] gRPC MCP transport: `McpGrpcServer` and `McpGrpcTransport` for serving/consuming MCP tools over gRPC (`feature = "grpc"` + `feature = "mcp"`)
- [x] Distributed checkpoint sync: `CheckpointSync` (write-through local+remote), `CheckpointReplicator` (background pull loop), `pull_all()` / `push_all()` for bulk sync

## v0.12.0 -- Runtime & Persistence

- [x] NATS KV-based checkpoint backend (`NatsKvCheckpoint`) using JetStream key-value store (`feature = "nats"`)
- [x] Redis checkpoint backend (`RedisCheckpoint`) using Redis hashes (`feature = "redis"`)
- [x] Agent hot-reload (`HotSwapAgent`): swap model, tools, system prompt, memory, hooks, middleware, guardrails at runtime without restarting
- [x] Streaming distributed execution: `TaskEventBus` trait, `InProcessEventBus`, `StreamingTaskWorker`, serializable `TaskStreamEvent`/`SerializableStreamEvent`

## v0.13.0 -- Cloud-Native Brokers

- [x] Moved `TaskBroker` trait, `ErasedTaskBroker`, `AgentTask`, `TaskResult`, `TaskStatus` into `daimon-core` so provider crates can implement the trait
- [x] AWS SQS task broker (`SqsBroker`) in `daimon-provider-bedrock` (`feature = "sqs"`)
- [x] Google Cloud Pub/Sub task broker (`PubSubBroker`) in `daimon-provider-gemini` (`feature = "pubsub"`)
- [x] Azure Service Bus task broker (`ServiceBusBroker`) in `daimon-provider-azure` (`feature = "servicebus"`)

## v0.14.0 -- Performance & Benchmarking

- [x] ToolRegistry generation-based cache invalidation with `tool_specs_mut()` (uncached spec gen **-33%**)
- [x] Memory `get_messages()` contiguous slice clone via `make_contiguous().to_vec()` (single memcpy)
- [x] ReAct loop: move tool_calls instead of clone, move messages instead of clone in short-circuit paths
- [x] MiddlewareStack: early return when empty (avoid async iteration on hot path)
- [x] SlidingWindowMemory: single-pop eviction (if/pop_front instead of while loop)
- [x] Fixed unused assignment warning in runner.rs
- [x] New benchmarks: HotSwapAgent, InProcessBroker, InProcessEventBus, InMemoryCheckpoint, SerializableStreamEvent serde

## v0.15.0 -- Vector Store Plugins

- [x] Moved `Document`, `ScoredDocument`, `VectorStore`, `ErasedVectorStore`, `SharedVectorStore` into `daimon-core` so plugin crates can implement the trait
- [x] `daimon-plugin-pgvector` crate: pgvector-backed `VectorStore` via `tokio-postgres` + `deadpool-postgres` (`feature = "pgvector"`)
  - `PgVectorStore` with UPSERT, cosine/L2/inner-product queries, HNSW indexing
  - `PgVectorStoreBuilder` with connection pooling, auto-migration, configurable HNSW params
  - `DistanceMetric` enum (Cosine, L2, InnerProduct) with operator class mapping
  - `migrations` module exporting raw SQL for manual schema setup
  - Composes with `SimpleKnowledgeBase` for full RAG pipeline

## Future

- [ ] ChromaDB vector store retriever (`feature = "chromadb"`)
- [ ] Pinecone vector store plugin (`feature = "pinecone"`)
- [ ] Weaviate vector store plugin (`feature = "weaviate"`)