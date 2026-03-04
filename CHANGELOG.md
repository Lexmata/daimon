# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`daimon-plugin-opensearch` crate** — OpenSearch k-NN backed `VectorStore` implementation using the official `opensearch-rs` client. Supports cosine similarity, L2, and inner-product distance metrics with HNSW indexing via Lucene, NMSLIB, or FAISS engines. Auto-creates k-NN index by default; JSON exported in `index_settings` module for manual setups.
- **`OpenSearchVectorStoreBuilder`** — builder pattern for configuring index name, space type, engine, HNSW `m` and `ef_construction` parameters, with `build_with_client()` for custom transport (AWS SigV4, etc.).
- **`opensearch` feature flag** in the main `daimon` crate for opt-in OpenSearch support (included in `full`).
- **`aws-auth` feature** on `daimon-plugin-opensearch` for Amazon OpenSearch Service SigV4 authentication.

## [0.15.0] - 2026-03-04

### Added

- **`daimon-plugin-pgvector` crate** — pgvector-backed `VectorStore` implementation using `tokio-postgres` and `deadpool-postgres` for connection pooling. Supports cosine, L2, and inner-product distance metrics with HNSW indexing. Auto-migrates schema by default; raw SQL exported in `migrations` module for manual setups.
- **`PgVectorStoreBuilder`** — builder pattern for configuring table name, distance metric, pool size, HNSW `m` and `ef_construction` parameters.
- **`pgvector` feature flag** in the main `daimon` crate for opt-in pgvector support (included in `full`).

### Changed

- **Moved `Document`, `ScoredDocument`, `VectorStore`, `ErasedVectorStore`, `SharedVectorStore` to `daimon-core`** — plugin crates can now implement `VectorStore` by depending only on `daimon-core`. The main `daimon` crate re-exports everything; existing code is unaffected.

## [0.14.0] - 2026-03-04

### Changed

- **ToolRegistry: generation-based cache invalidation** — `tool_specs()` now uses a generation counter to detect stale caches. Added `tool_specs_mut()` for callers with `&mut self` that need to persist the computed specs into the cache. `warm_cache()` now delegates to `tool_specs_mut()`, avoiding duplicate logic. Uncached spec generation **-33%** (10.4us to 6.9us).
- **SlidingWindowMemory & TokenWindowMemory: contiguous slice clone** — `get_messages()` now calls `make_contiguous().to_vec()` instead of `iter().cloned().collect()`, producing a single memcpy when the deque is already contiguous and avoiding per-element overhead.
- **SlidingWindowMemory: single-pop eviction** — replaced `while messages.len() > max` loop with a single `if len >= max { pop_front() }` since at most one message is added at a time.
- **ReAct loop: reduced cloning** — tool calls are now moved out with `std::mem::take` instead of `.to_vec()` when the response message is consumed. Middleware short-circuit paths move messages instead of cloning them.
- **MiddlewareStack: early return when empty** — all three middleware pipeline methods (`run_on_request`, `run_on_response`, `run_on_tool_call`) now return `Continue` immediately when no middleware is registered, avoiding async iteration overhead on the hot path. Chain transforms benchmark **-30%** (287ns to 210ns).
- **Fixed unused assignment warning** in `runner.rs` short-circuit branch.

### Added

- **New benchmarks** for `HotSwapAgent` (prompt, swap_model), `InProcessBroker` (submit/receive/complete roundtrip), `InProcessEventBus` (publish/receive), `InMemoryCheckpoint` (save/load), and `SerializableStreamEvent` (serialize/deserialize).

## [0.13.0] - 2026-03-04

### Added

- **TaskBroker trait moved to `daimon-core`**: `TaskBroker`, `ErasedTaskBroker`, `AgentTask`, `TaskResult`, and `TaskStatus` now live in `daimon-core::distributed`, enabling provider crates to implement cloud-native brokers. The main `daimon` crate re-exports everything — existing code is unaffected.

- **AWS SQS task broker** (`daimon-provider-bedrock`, `feature = "sqs"`):
  - `SqsBroker` implementing `TaskBroker` via `aws-sdk-sqs`.
  - Uses SQS visibility timeout for in-flight task tracking, long polling for `receive()`.
  - `new()` with default credentials, `with_region()`, `from_client()` constructors.
  - Configurable visibility timeout via `with_visibility_timeout()`.
  - Deletes messages on `complete()`, releases visibility on `fail()`.

- **Google Cloud Pub/Sub task broker** (`daimon-provider-gemini`, `feature = "pubsub"`):
  - `PubSubBroker` implementing `TaskBroker` via Pub/Sub REST API.
  - Base64-encoded JSON message bodies for cross-platform compatibility.
  - `with_api_key()` and `with_bearer_token()` authentication modes.
  - Refreshable bearer token via `set_bearer_token()`.
  - Acknowledges messages on `complete()`, drops ack on `fail()` for automatic retry.

- **Azure Service Bus task broker** (`daimon-provider-azure`, `feature = "servicebus"`):
  - `ServiceBusBroker` implementing `TaskBroker` via Service Bus REST API.
  - Peek-lock receive with configurable lock duration.
  - SAS token authentication.
  - Deletes messages on `complete()`, releases lock on `fail()` for redelivery.

- **New feature flags** in the main `daimon` crate: `sqs`, `pubsub`, `servicebus` (all included in `full`).

## [0.12.0] - 2026-03-04

### Added

- **NATS KV checkpoint backend** (`src/checkpoint/nats_kv.rs`, `feature = "nats"`):
  - `NatsKvCheckpoint` implementing `Checkpoint` using NATS JetStream key-value store.
  - Distributed, replicated checkpoint storage with no external database.
  - `connect()` creates or opens a KV bucket; `from_store()` wraps an existing handle.
  - Keys prefixed with `cp.` for namespace isolation.

- **Redis checkpoint backend** (`src/checkpoint/redis.rs`, `feature = "redis"`):
  - `RedisCheckpoint` implementing `Checkpoint` using Redis hashes.
  - Fast, shared checkpoint storage accessible from multiple processes.
  - Uses `{prefix}:data` hash key for all checkpoint data.

- **Agent hot-reload** (`src/agent/hot_swap.rs`):
  - `HotSwapAgent` wraps an `Agent` behind a `RwLock` for runtime reconfiguration.
  - `swap_model()`, `swap_system_prompt()`, `swap_memory()`, `swap_hooks()` for swapping core components.
  - `add_tool()`, `remove_tool()` for dynamic tool management.
  - `swap_middleware()`, `add_middleware()` for middleware stack changes.
  - `add_input_guardrail()`, `add_output_guardrail()`, `clear_input_guardrails()`, `clear_output_guardrails()`.
  - `set_max_iterations()`, `set_temperature()`, `set_max_tokens()`, `set_validate_tool_inputs()`, `set_tool_retry_policy()`.
  - `replace()` for atomic full-agent swap.
  - Clone-friendly: all clones share the same underlying agent.

- **Streaming distributed execution** (`src/distributed/streaming.rs`):
  - `TaskEventBus` trait for publishing stream events across process boundaries.
  - `InProcessEventBus` backed by `tokio::sync::broadcast` for single-process use.
  - `StreamingTaskWorker` uses `Agent::prompt_stream()` and publishes each `StreamEvent` through the event bus.
  - `TaskStreamEvent` and `SerializableStreamEvent` for cross-process serializable stream events.
  - All `StreamEvent` variants mapped to serializable equivalents with full round-trip support.

## [0.11.0] - 2026-03-03

### Added

- **NATS JetStream task broker** (`src/distributed/nats_broker.rs`, `feature = "nats"`):
  - `NatsBroker` implementing `TaskBroker` for durable, at-least-once task delivery via NATS JetStream.
  - Uses a JetStream stream with `WorkQueue` retention and pull-based consumers with explicit ack.
  - `NatsBroker::connect(url, stream_name)` auto-creates the stream and configures subjects.
  - New `nats` feature flag adding `async-nats` dependency.
  - Re-exported from `daimon::distributed::NatsBroker` and prelude.

- **RabbitMQ task broker** (`src/distributed/amqp_broker.rs`, `feature = "amqp"`):
  - `AmqpBroker` implementing `TaskBroker` via AMQP 0-9-1 (RabbitMQ).
  - Durable queue with `delivery_mode = 2` (persistent messages) and manual acknowledgement.
  - `AmqpBroker::connect(url, queue_name)` declares the queue and creates a channel.
  - New `amqp` feature flag adding `lapin` dependency.
  - Re-exported from `daimon::distributed::AmqpBroker` and prelude.

- **gRPC MCP transport** (`src/mcp/grpc_transport.rs`, `feature = "grpc"` + `feature = "mcp"`):
  - `McpGrpcServer` wraps an `McpServer` and serves MCP tools over gRPC with typed RPCs for Initialize, ToolsList, ToolsCall, plus a raw HandleRaw passthrough.
  - `McpGrpcTransport` implements `McpTransport` by connecting to a remote gRPC MCP server, enabling transparent gRPC-backed MCP tool discovery and execution.
  - Proto definition in `proto/daimon_mcp.proto`.
  - Conditionally compiled when both `grpc` and `mcp` features are enabled.
  - Re-exported from `daimon::mcp::{McpGrpcServer, McpGrpcTransport}` and prelude.

- **Distributed checkpoint sync** (`src/checkpoint/sync.rs`):
  - `CheckpointSync` — write-through checkpoint combining a local (fast) and remote (shared) backend. Saves write to both; loads prefer local with remote fallback and automatic backfill. `list_runs()` returns the union.
  - `CheckpointSync::pull_all()` / `push_all()` for bulk bidirectional synchronization.
  - `CheckpointReplicator` — background task that periodically pulls new checkpoints from remote to local at a configurable interval.
  - Both implement/compose the existing `Checkpoint` trait; no new feature flag required.
  - Re-exported from `daimon::checkpoint::{CheckpointSync, CheckpointReplicator}` and prelude.

- **Redis task broker** (`src/distributed/redis_broker.rs`, `feature = "redis"`):
  - `RedisBroker` implementing `TaskBroker` for multi-process distributed execution.
  - Uses Redis Lists (`LPUSH`/`BRPOP`) for the task queue with 1-second blocking pop.
  - Status tracked in a Redis Hash (`{prefix}:status`); results stored in `{prefix}:results`.
  - Supports configurable key prefix for namespace isolation.
  - Serializes `AgentTask` and `TaskResult` as JSON for cross-language interoperability.

- **Builder-style agent fork** (`src/agent/fork.rs`):
  - `Agent::fork_builder()` returns a `ForkBuilder` pre-populated with the parent agent's config.
  - Mutate any field before building: `system_prompt()`, `no_system_prompt()`, `model()`, `tool()`, `remove_tool()`, `memory()`, `hooks()`, `middleware()`, `input_guardrail()`, `output_guardrail()`, `max_iterations()`, `temperature()`, `max_tokens()`, `validate_tool_inputs()`, `tool_retry_policy()`.
  - `build()` produces an independent forked agent with the specified mutations applied.
  - Enables patterns like A/B testing agent configurations, role specialization from a base agent, and checkpoint-based branching with modified tools.

- **`ToolRegistry::unregister()`** (`src/tool/registry.rs`):
  - Removes a tool by name. Returns `true` if the tool was present.
  - Invalidates spec and validator caches on removal.

- **WebSocket MCP server** (`src/mcp/ws_server.rs`, `feature = "mcp"`):
  - `McpWsServer` listens on a TCP port and serves MCP tools over WebSocket connections.
  - Each connection handled in a separate task for concurrent multi-client support.
  - Reuses `McpServer::handle_request_raw()` for JSON-RPC dispatch (initialize, tools/list, tools/call).
  - `McpWsServer::new(registry)` or `McpWsServer::from_server(server)` constructors.
  - `serve(addr)` for production (runs indefinitely) and `serve_one(addr)` for testing.
  - Re-exported from `daimon::mcp::McpWsServer` and `daimon::prelude::McpWsServer`.

- **gRPC transport for distributed execution** (`src/distributed/grpc.rs`, `feature = "grpc"`):
  - `GrpcBrokerServer` wraps any `TaskBroker` (or `ErasedTaskBroker`) and serves it as a gRPC service.
  - `GrpcBrokerClient` connects to a remote gRPC broker and implements `TaskBroker` transparently.
  - Proto service definition in `proto/daimon_distributed.proto` with Submit, GetStatus, Complete, and Fail RPCs.
  - JSON encoding for task/result payloads ensures cross-language compatibility.
  - New `grpc` feature flag adding `tonic` and `prost` dependencies.
  - `tonic-build` compiles proto at build time (conditional on `grpc` feature).
  - Re-exported from `daimon::distributed::{GrpcBrokerServer, GrpcBrokerClient}` and prelude.

- **Streaming cost tracking** (`daimon-core/src/stream.rs`, `src/agent/runner.rs`):
  - New `StreamEvent::Usage { iteration, input_tokens, output_tokens, estimated_cost }` variant emitted after each ReAct iteration in `prompt_stream()`.
  - Token counts are estimated from character length (~4 chars/token) since streaming providers typically don't report usage inline.
  - When a `CostModel` is configured on the agent, `estimated_cost` accumulates across iterations.
  - Non-breaking addition to the existing streaming API (new enum variant).

- **Agent cloning / forking** (`src/agent/fork.rs`):
  - `Agent::fork()` — creates a new agent sharing the same model, tools, hooks, middleware, and guardrails but with independent (empty) memory.
  - `Agent::fork_from_checkpoint(run_id, checkpoint)` — forks with memory pre-loaded from a saved checkpoint. Enables "what-if" branching: modify tools or system prompt on the fork and diverge from a historical run.
  - `Agent::fork_with_memory(memory)` — forks with a custom memory backend (e.g. switch from in-memory to SQLite).
  - All forked agents share model/tools via `Arc` — lightweight and memory-efficient.

- **Distributed agent execution** (`src/distributed/`):
  - `TaskBroker` trait with `submit`, `status`, `receive`, `complete`, and `fail` methods for distributing agent tasks across workers.
  - `ErasedTaskBroker` object-safe wrapper for dynamic dispatch.
  - `InProcessBroker` backed by tokio MPSC channels for single-process parallelism and testing.
  - `AgentTask` — serializable unit of work with input, optional run ID, and metadata.
  - `TaskResult` — serializable output with text, iterations, cost, and error fields.
  - `TaskStatus` enum: `Pending`, `Running`, `Completed(TaskResult)`, `Failed(String)`.
  - `TaskWorker` with `run_once()`, `run()` (continuous loop), and `run_parallel(concurrency)` for concurrent task processing.
  - Agent factory pattern ensures each task gets independent memory.
  - Implement `TaskBroker` for Redis, NATS, or RabbitMQ to enable multi-process/multi-machine execution.

- **WebSocket transport for MCP** (`src/mcp/websocket.rs`):
  - `WebSocketTransport` implementing `McpTransport` for persistent WebSocket connections.
  - Supports `ws://` and `wss://` (TLS via rustls) connections.
  - JSON-RPC messages sent as text frames; automatic ping/pong handling.
  - Thread-safe via internal mutex on the WebSocket stream.
  - New `tokio-tungstenite` dependency (optional, behind `mcp` feature).
  - Re-exported from `daimon::mcp::WebSocketTransport`.

- **Middleware pipeline** (`src/middleware/`):
  - `Middleware` trait with `on_request(&mut ChatRequest)`, `on_response(&mut ChatResponse)`, and `on_tool_call(&mut ToolCall)` hooks.
  - `MiddlewareAction` enum: `Continue` or `ShortCircuit(ChatResponse)` for early exit.
  - `MiddlewareStack` chains layers in registration order; first non-`Continue` action short-circuits.
  - Object-safe `ErasedMiddleware` wrapper for dynamic dispatch.
  - `AgentBuilder::middleware()` to add layers.
  - Wired into the ReAct loop (request/response mutation) and tool execution path.

- **Guardrails** (`src/guardrails/`):
  - `InputGuardrail` trait: validates user input before the model sees it.
  - `OutputGuardrail` trait: validates model output before returning to caller.
  - `GuardrailResult` enum: `Pass`, `Block(String)`, `Transform(String)`.
  - Built-in `MaxTokenGuardrail` — rejects inputs exceeding an estimated token limit.
  - Built-in `RegexFilterGuardrail` — block or redact text matching regex patterns (PII, profanity).
  - `AgentBuilder::input_guardrail()` and `AgentBuilder::output_guardrail()` methods.
  - `DaimonError::GuardrailBlocked` error variant.

- **Prompt templates** (`src/prompt/`):
  - `PromptTemplate` with `{variable}` placeholder interpolation via `render_static()` and `render_with()`.
  - `PromptBuilder` for composing sections: persona, instructions, constraints, examples, and custom sections.
  - `AgentBuilder::prompt_template()` as an alternative to `system_prompt()`.

- **Cost tracking and budget limits** (`src/cost/`):
  - `CostModel` trait mapping `(model_id, TokenDirection)` to per-token USD cost.
  - Built-in `OpenAiCostModel` and `AnthropicCostModel` with approximate pricing.
  - `CostTracker` with lock-free atomic accumulation (micro-dollar precision).
  - `AgentBuilder::cost_model()` and `AgentBuilder::max_budget()` — aborts with `DaimonError::BudgetExceeded` when limit is crossed.
  - `AgentResponse.cost` field for per-prompt cost in USD.

- **Embeddings API** (`daimon-core/src/embedding.rs`):
  - `EmbeddingModel` trait with `embed(&[&str]) -> Vec<Vec<f32>>` and `dimensions()`.
  - `ErasedEmbeddingModel` object-safe wrapper and `SharedEmbeddingModel` type alias.
  - `OpenAiEmbedding` provider (text-embedding-3-small/large) behind `openai` feature.
  - `OllamaEmbedding` provider behind `ollama` feature.

- **Vector store integrations** (`src/retriever/`):
  - `InMemoryVectorStore` — brute-force cosine similarity for development and testing, no feature gate required.
  - `QdrantRetriever` (`feature = "qdrant"`) — retriever backed by Qdrant vector database via `qdrant-client`.
  - Both accept an `EmbeddingModel` and implement the existing `Retriever` trait.

- **Self-healing tool retry** (`src/tool/retry.rs`):
  - `ToolRetryPolicy` with `BackoffStrategy::Fixed` and `BackoffStrategy::Exponential`.
  - Configurable retryable error patterns — only retry errors matching specified substrings.
  - `AgentBuilder::tool_retry_policy()` applies to all tool calls.
  - Integrated into `execute_tools_parallel()` with backoff delays between attempts.

- **Deployment helpers** (`src/server/`, `feature = "http-server"`):
  - `AgentServer` wrapping an `Agent` behind an `axum` router.
  - `POST /prompt` — JSON request/response endpoint.
  - `POST /prompt/stream` — Server-Sent Events streaming endpoint.
  - `GET /health` — health check returning `"ok"`.
  - `AgentServer::new(agent).bind("0.0.0.0:8080").serve().await`.

- **Evaluation harness** (`src/eval/`, `feature = "eval"`):
  - `EvalScenario` with configurable input, scorers, max iterations, and max cost.
  - `Scorer` strategies: `ExactMatch`, `Contains`, `Regex`, and `Custom(Box<dyn Fn>)`.
  - `EvalRunner` executes scenarios with configurable concurrency.
  - `EvalResult` with pass/fail, output text, latency, cost, iteration count, and error details.

- **Time-travel debugging** (`src/checkpoint/replay.rs`):
  - `inspect_run()` reconstructs an `ExecutionTrace` from checkpoint message history.
  - `list_runs()` returns `RunSummary` for all checkpointed runs.
  - `TraceStep` captures per-iteration messages, tool calls, response text, and usage.
  - `ExecutionTrace` with `final_text()` and `total_tool_calls()` helpers.

- **`ContentPolicyGuardrail`** (`src/guardrails/content_policy.rs`):
  - LLM-as-judge guardrail that evaluates content against a policy description.
  - Implements both `InputGuardrail` and `OutputGuardrail` — usable on either side.
  - Configurable with any `SharedModel` and custom policy string.

- **`DynamicContext` trait** (`src/prompt/dynamic.rs`):
  - `DynamicContext` trait with `key()` and async `resolve()` for runtime prompt variable injection.
  - `ErasedDynamicContext` object-safe wrapper for dynamic dispatch.
  - `PromptTemplate::render_dynamic(&[&dyn ErasedDynamicContext])` resolves async contexts at render time.

- **`FewShotTemplate`** (`src/prompt/few_shot.rs`):
  - Builder for composing example input/output pairs with configurable labels and prefix.
  - `render()` produces a formatted string suitable for injection into prompts.

- **`SemanticSimilarity` scorer** (`src/eval/scoring.rs`):
  - New `Scorer::SemanticSimilarity` variant using `EmbeddingModel` for cosine similarity.
  - `Scorer::semantic(expected, embedding_model, threshold)` constructor.
  - `EvalScenario::expect_semantic()` convenience method.

- **`LlmJudge` scorer** (`src/eval/scoring.rs`):
  - New `Scorer::LlmJudge` variant using a `Model` to grade output against a rubric.
  - `Scorer::llm_judge(rubric, model)` constructor.
  - `EvalScenario::expect_llm_judge()` convenience method.
  - All `Scorer::evaluate()` methods are now async to support these advanced scorers.

- **`Agent::replay()`** (`src/agent/resumable.rs`):
  - Re-run a previous agent execution from a checkpoint with optionally modified context.
  - `from_iteration` parameter truncates to a specific iteration for "what-if" debugging.
  - Useful for testing changes to tools, system prompts, or models against prior runs.

- **Per-tool retry policy** (`src/tool/traits.rs`):
  - `Tool::retry_policy() -> Option<ToolRetryPolicy>` with default `None`.
  - Per-tool policy takes precedence over agent-level `tool_retry_policy` in the ReAct loop.
  - `ErasedTool` object-safe wrapper updated to forward `retry_policy()`.

- **API key authentication** (`src/server/mod.rs`):
  - `AgentServer::api_key()` builder method enables request authentication.
  - Supports `Authorization: Bearer <key>` and `X-API-Key: <key>` headers.
  - Returns 401 Unauthorized for missing or invalid keys.

- **Additional embedding providers**:
  - `GeminiEmbedding` (`daimon-provider-gemini`) — Google batchEmbedContents API.
  - `AzureOpenAiEmbedding` (`daimon-provider-azure`) — Azure OpenAI Embeddings API.
  - `BedrockEmbedding` (`daimon-provider-bedrock`) — Amazon Titan Embeddings via InvokeModel.

- **Vector Store plugin system** (`src/retriever/vector_store.rs`, `src/retriever/knowledge_base.rs`):
  - `VectorStore` trait with `upsert()`, `query()`, `delete()`, `count()` operations.
  - `ErasedVectorStore` object-safe wrapper and `SharedVectorStore` type alias.
  - `ScoredDocument` struct pairing documents with similarity scores.
  - `KnowledgeBase` trait for high-level ingest/search/remove/count operations.
  - `SimpleKnowledgeBase<V: VectorStore>` composing `EmbeddingModel` + `VectorStore`.
  - `SimpleKnowledgeBase` auto-implements `Retriever` for seamless agent integration.
  - `InMemoryVectorStoreBackend` implementing `VectorStore` for development/testing.
  - `ErasedKnowledgeBase` object-safe wrapper and `SharedKnowledgeBase` type alias.

- New feature flags: `http-server`, `qdrant`, `eval`.
- New dependencies: `regex-lite`, `axum` (optional), `tower-http` (optional), `qdrant-client` (optional).
- Updated `prelude` module with all new types.

## [0.2.0] - 2026-03-03

### Added

- **Provider prompt caching** across all backends:
  - **Anthropic**: `with_prompt_caching()` now correctly injects `cache_control` blocks on the system message and the last tool definition, enabling actual cache hits. Parses `cache_creation_input_tokens` and `cache_read_input_tokens` from usage with tracing.
  - **OpenAI**: Parses `prompt_tokens_details.cached_tokens` from the usage response (automatic prompt caching).
  - **Azure OpenAI**: Same as OpenAI — parses `prompt_tokens_details.cached_tokens`.
  - **Google Gemini**: New `with_cached_content(name)` builder method to reference a `cachedContents/<id>` resource. Parses `cachedContentTokenCount` from usage metadata.
  - **AWS Bedrock**: New `with_prompt_caching()` builder inserts `CachePoint` blocks after system messages and tool definitions. Parses `cache_read_input_tokens` from the Converse API response.
  - **Ollama**: New `with_keep_alive(duration)` builder to control KV cache retention (e.g. `"5m"`, `"1h"`, `"0"`).
- `Usage::cached_tokens` field — number of input tokens served from the provider's cache (subset of `input_tokens`, defaults to 0).

### Fixed

- **Anthropic caching was non-functional**: the `with_prompt_caching()` flag sent the beta header but never added `cache_control` content blocks, so no caching actually occurred. Now correctly marks the system prompt and last tool definition as cache breakpoints.

## [0.1.0] - 2026-03-03

### Added

- **Core framework** with ReAct (Reason-Act-Observe) agent loop.
- `Agent` struct with builder pattern for fluent configuration.
- `Agent::prompt()` for synchronous (non-streaming) agent responses.
- `Agent::prompt_stream()` with full streaming ReAct loop — accumulates tool call deltas, executes tools, re-invokes the model, all within a single `ResponseStream`.
- `Agent::prompt_with_messages()` for pre-built conversation histories.
- `Agent::prompt_with_cancellation()` with `tokio_util::CancellationToken` support.
- `AgentResponse` with aggregated `Usage` across all iterations.
- **Model trait** with `generate()` and `generate_stream()` async methods, plus object-safe `ErasedModel` wrapper for dynamic dispatch.
- **OpenAI provider** (`feature = "openai"`, default) — Chat Completions API with SSE streaming, tool calling, `response_format`, and `parallel_tool_calls` support.
- **Anthropic provider** (`feature = "anthropic"`, default) — Messages API with streaming, tool use blocks, prompt caching (`cache_control` beta header), and 529 overloaded retry.
- **Google Gemini provider** (`feature = "gemini"`) — Generative Language REST API with function calling, SSE streaming via `streamGenerateContent`, system instruction support. Configurable for Vertex AI via `with_base_url()` and `with_bearer_token()` for OAuth2.
- **Azure OpenAI provider** (`feature = "azure"`) — Azure OpenAI Service deployments with the same wire format as OpenAI. Supports both `api-key` header and Microsoft Entra ID bearer token authentication, configurable `api-version`.
- **AWS Bedrock provider** (`feature = "bedrock"`) — Converse/ConverseStream API via `aws-sdk-bedrockruntime`, with guardrails configuration.
- All providers: configurable HTTP timeout, max retries with exponential backoff for 429/5xx errors.
- **Tool trait** with `name()`, `description()`, `parameters_schema()`, and `execute()`, plus object-safe `ErasedTool` wrapper.
- `ToolRegistry` for named tool management with duplicate detection.
- `ToolOutput::text()`, `ToolOutput::json()`, and `ToolOutput::error()` constructors.
- Parallel tool execution within iterations via `tokio::task::JoinSet`.
- **Memory trait** with `add_message()`, `get_messages()`, and `clear()`, plus object-safe `ErasedMemory` wrapper.
- `SlidingWindowMemory` with configurable message window size.
- Tool-call messages (assistant + tool results) now persisted to memory alongside user/assistant messages.
- **AgentHook trait** for lifecycle events: `on_iteration_start`, `on_model_response`, `on_tool_call`, `on_tool_result`, `on_iteration_end`, `on_error`.
- **Streaming types**: `StreamEvent` enum with `TextDelta`, `ToolCallStart`, `ToolCallDelta`, `ToolCallEnd`, `ToolResult`, `Error`, and `Done` variants.
- **Error handling**: `DaimonError` with `Timeout` and `Cancelled` variants; retry logic in all providers.
- **Observability**: `tracing::instrument` spans on all agent methods and provider calls with structured fields (model_id, tool name/id, iteration, token counts).
- `prelude` module re-exporting common types including `CancellationToken`.
- Rustdoc on all public types, traits, methods, and modules.
- Six runnable examples: `simple_agent`, `with_tools`, `streaming`, `bedrock_agent`, `gemini_agent`, `azure_agent`.
- `cargo-husky` dev-dependency with `user-hooks` for automatic Git hook installation on `cargo test`.
- `pre-commit` hook: `cargo fmt --check` + `cargo clippy --features full -- -D warnings`.
- `commit-msg` hook: Conventional Commits validation via `cargo-commitlint`.
- `pre-push` hook: full test suite + documentation build check.
- GitHub Actions CI workflow (check, fmt, clippy, test, coverage gate at 90%, example compilation).
- `deny.toml` for `cargo-deny` license and advisory auditing.
- `commitlint.toml` for Conventional Commits enforcement.
- `rustfmt.toml` and `clippy.toml` for consistent code style.

[Unreleased]: https://github.com/Lexmata/daimon/compare/v0.15.0...HEAD
[0.15.0]: https://github.com/Lexmata/daimon/compare/v0.14.0...v0.15.0
[0.14.0]: https://github.com/Lexmata/daimon/compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com/Lexmata/daimon/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/Lexmata/daimon/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/Lexmata/daimon/compare/v0.2.0...v0.11.0
[0.2.0]: https://github.com/Lexmata/daimon/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Lexmata/daimon/releases/tag/v0.1.0
