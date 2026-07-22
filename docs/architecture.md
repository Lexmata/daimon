# Daimon Architecture

This document describes the deep architectural design of the Daimon Rust AI agent framework. It is written for senior Rust developers who want to understand the design before contributing.

---

## 1. Design Philosophy

### Why Rust for Agents

Daimon is built in Rust to deliver:

- **Zero-cost abstractions** — Trait-based polymorphism compiles to static dispatch when the concrete type is known. The `Erased*` pattern adds dynamic dispatch only when needed; users choose per call site.
- **Trait-based plugin system** — New LLM providers, vector stores, and task brokers are added by implementing traits. No reflection, no runtime registration. Compile-time composition.
- **Async-first** — All I/O is async. The ReAct loop, tool execution, and streaming are built on `async fn` and `Future`. No blocking calls in the hot path.
- **No runtime overhead from unused features** — Every provider, storage backend, and protocol is behind a Cargo feature flag. The core framework compiles with zero optional dependencies. Users pay only for what they enable.
- **Memory safety without GC** — Long-running agents with tool loops and streaming avoid GC pauses. Ownership and borrowing prevent data races and leaks.
- **Single binary deployment** — Compile one binary with the features you need. No Python runtime, no Node.js, no interpreter startup.

### Core Principles

1. **daimon-core is the plugin boundary** — Provider and plugin crates depend only on `daimon-core`. The main `daimon` crate re-exports everything. This keeps the interface stable and avoids circular dependencies.
2. **Traits over concrete types** — The agent accepts `SharedModel`, `SharedMemory`, `Arc<dyn ErasedTool>`. Implementors are decoupled from the orchestration layer.
3. **Builder pattern for complex types** — `Agent::builder()`, `ChainBuilder`, `PgVectorStoreBuilder`. Validation at `.build()` time; invalid config fails to compile or returns `Result`.
4. **Structured errors** — Single `DaimonError` enum. Provider crates map transport errors to `DaimonError::Model(String)` or `DaimonError::Other(String)`.

---

## 2. Workspace Layout

Daimon is a Cargo workspace (mono-repo). The root `Cargo.toml` defines the main `daimon` package and workspace members.

```
daimon/
├── Cargo.toml              # Workspace root + daimon package
├── daimon-core/            # Plugin interface traits + shared types
├── daimon-macros/          # #[tool_fn] proc macro
├── daimon-provider-openai/
├── daimon-provider-anthropic/
├── daimon-provider-bedrock/
├── daimon-provider-gemini/
├── daimon-provider-azure/
├── daimon-provider-local/  # Ollama, llama.cpp, llama-rs, OpenAI-compatible
├── daimon-provider-llamacpp/
├── daimon-plugin-pgvector/
├── daimon-plugin-opensearch/
├── src/                    # Main daimon crate source
│   ├── agent/              # Agent builder, ReAct loop, multi-agent patterns
│   ├── model/              # Model trait re-exports from the provider crates
│   ├── tool/               # Tool trait, registry, retry
│   ├── memory/             # Memory trait + SQLite, Redis, sliding window
│   ├── stream/             # ResponseStream, StreamEvent
│   ├── hooks.rs            # AgentHook lifecycle callbacks
│   ├── middleware/         # Request/response/tool_call interception
│   ├── guardrails/         # InputGuardrail, OutputGuardrail
│   ├── retriever/          # Retriever, KnowledgeBase, RAG tool
│   ├── checkpoint/         # Checkpoint trait + file, Redis, NATS KV
│   ├── cost/               # CostModel, CostTracker, budget enforcement
│   ├── orchestration/       # Chain, Graph, DAG, Workflow
│   ├── distributed/        # TaskBroker implementations (Redis, NATS, AMQP, gRPC)
│   ├── mcp/                # Model Context Protocol client & server
│   ├── a2a/                # Google Agent-to-Agent protocol
│   ├── server/             # HTTP server (axum) for REST API
│   ├── eval/               # EvalScenario, EvalRunner, Scorer
│   └── ...
├── examples/
├── tests/
└── benches/
```

### Crate Responsibilities

| Crate | Purpose |
|-------|---------|
| **daimon** | Main crate. Agent runtime, orchestration, RAG, memory, hooks, tools, MCP, A2A, eval, server. Re-exports from daimon-core and provider/plugin crates. Users depend on `daimon` with feature flags. |
| **daimon-core** | Plugin interface. Defines `Model`, `EmbeddingModel`, `VectorStore`, `TaskBroker` traits and shared types: `Message`, `ChatRequest`, `ChatResponse`, `Document`, `ToolCall`, `StreamEvent`, `DaimonError`. Zero optional deps. |
| **daimon-macros** | `#[tool_fn]` proc macro. Derives `Tool` from async functions with JSON Schema from parameter types. |
| **daimon-provider-*** | LLM provider crates. Each implements `Model` from daimon-core; most also implement `EmbeddingModel` (Anthropic does not — no embeddings API). Depend only on daimon-core + HTTP/SDK. |
| **daimon-plugin-*** | Storage plugins. Each implements `VectorStore` from daimon-core. Depend only on daimon-core + DB client. |

---

## 3. The Plugin Boundary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         daimon (main crate)                               │
│  Agent, Chain, Graph, Memory, Hooks, MCP, Server, Eval, ...              │
│  Re-exports: daimon_core::*, daimon_provider_*::*, daimon_plugin_*::*     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ depends on
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         daimon-core                                       │
│  Traits: Model, EmbeddingModel, VectorStore, TaskBroker                  │
│  Types: Message, ChatRequest, ChatResponse, Document, ToolCall, ...      │
│  Error: DaimonError, Result                                              │
└─────────────────────────────────────────────────────────────────────────┘
         ▲                    ▲                    ▲
         │                    │                    │
         │ implements        │ implements        │ implements
         │                    │                    │
┌────────┴────────┐  ┌────────┴────────┐  ┌───────┴────────┐
│ daimon-provider-│  │ daimon-provider-│  │ daimon-plugin- │
│ bedrock         │  │ gemini          │  │ pgvector       │
│ azure           │  │                 │  │ opensearch     │
└─────────────────┘  └─────────────────┘  └────────────────┘
```

### Dependency Rules

1. **Provider crates** (`daimon-provider-*`) depend only on `daimon-core`. They implement `Model` and `EmbeddingModel`. They do not depend on `daimon`.
2. **Plugin crates** (`daimon-plugin-*`) depend only on `daimon-core`. They implement `VectorStore`. They do not depend on `daimon`.
3. **Main crate** (`daimon`) depends on `daimon-core` and optionally on provider/plugin crates via feature flags. It re-exports their types.
4. **Users** depend on `daimon` with features like `gemini`, `pgvector`. They never depend on `daimon-core` directly unless implementing a new provider.

### Adding New Components

| Component | Action |
|-----------|--------|
| New LLM provider | Create `daimon-provider-xyz`, implement `Model` (+ `EmbeddingModel` if supported), add `xyz` feature to daimon. |
| New vector store | Create `daimon-plugin-xyz`, implement `VectorStore`, add `xyz` feature to daimon. |
| New task broker | Implement `TaskBroker` in daimon (or a provider crate if cloud-specific), add feature flag. |
| New memory backend | Implement `Memory` in daimon, add feature flag. |

No circular dependencies. Clean separation.

---

## 4. The ReAct Loop

The central execution model is the ReAct (Reason-Act-Observe) loop. The agent receives a prompt, builds messages, calls the model, and either returns a final response or executes tools and loops.

### Flow Diagram

```
                    ┌──────────────────────────────────────────────────┐
                    │                  User Prompt                      │
                    └──────────────────────────────────────────────────┘
                                              │
                                              ▼
                    ┌──────────────────────────────────────────────────┐
                    │  Input Guardrails (optional transform/block)        │
                    └──────────────────────────────────────────────────┘
                                              │
                                              ▼
                    ┌──────────────────────────────────────────────────┐
                    │  Build messages: system + history + user           │
                    │  Add user message to memory                       │
                    └──────────────────────────────────────────────────┘
                                              │
                                              ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         ReAct Loop (max_iterations)                   │
    │                                                                      │
    │   ┌─────────────────────────────────────────────────────────────┐   │
    │   │  Check: CancellationToken cancelled? → Err(Cancelled)         │   │
    │   │  Check: Budget exceeded? → Err(BudgetExceeded)               │   │
    │   └─────────────────────────────────────────────────────────────┘   │
    │                              │                                       │
    │                              ▼                                       │
    │   ┌─────────────────────────────────────────────────────────────┐   │
    │   │  Hook: on_iteration_start                                    │   │
    │   └─────────────────────────────────────────────────────────────┘   │
    │                              │                                       │
    │                              ▼                                       │
    │   ┌─────────────────────────────────────────────────────────────┐   │
    │   │  Middleware: on_request (may short-circuit)                  │   │
    │   └─────────────────────────────────────────────────────────────┘   │
    │                              │                                       │
    │                              ▼                                       │
    │   ┌─────────────────────────────────────────────────────────────┐   │
    │   │  model.generate_erased(&ChatRequest)                           │   │
    │   └─────────────────────────────────────────────────────────────┘   │
    │                              │                                       │
    │                              ▼                                       │
    │   ┌─────────────────────────────────────────────────────────────┐   │
    │   │  Middleware: on_response (may short-circuit)                 │   │
    │   └─────────────────────────────────────────────────────────────┘   │
    │                              │                                       │
    │                              ▼                                       │
    │   ┌─────────────────────────────────────────────────────────────┐   │
    │   │  Hook: on_model_response                                     │   │
    │   └─────────────────────────────────────────────────────────────┘   │
    │                              │                                       │
     │              ┌────────────────┴────────────────┐                       │
     │              │                               │                       │
     │              ▼                               ▼                       │
     │   ┌─────────────────────┐         ┌─────────────────────────────────┐│
     │   │ has_tool_calls?     │   NO   │  Output guardrails (optional)    ││
     │   │                    │────────▶│  Hook: on_iteration_end          ││
     │   │                    │        │  Block → Err(GuardrailBlocked)   ││
     │   │                    │        │  Add (transformed) msg to memory ││
     │   │                    │        │  Return AgentResponse             ││
     │   └─────────────────────┘        └─────────────────────────────────┘│
     │              │ YES                                                      │
     │              ▼                                                         │
     │   ┌─────────────────────────────────────────────────────────────┐   │
     │   │  Add assistant message (with tool_calls) to memory            │   │
     │   │  execute_tools_parallel(tool_calls) — middleware on_tool_call   │   │
     │   │  Add tool result messages to memory                           │   │
     │   │  Hook: on_iteration_end                                        │   │
     │   │  Check: iteration >= max_iterations? → Err(MaxIterations)     │   │
     │   │  iteration += 1, continue loop                                │   │
     │   └─────────────────────────────────────────────────────────────┘   │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘
```

### Exit Conditions

- **No tool calls** — Model returns text; agent runs output guardrails first (a `Block` returns `Err(DaimonError::GuardrailBlocked)` and never reaches memory; a `Transform` rewrites the text), then persists the final assistant message to memory and returns `AgentResponse`.
- **Tool calls** — Agent executes tools (with optional schema validation), appends results to messages, loops. Tools run in parallel via `tokio::task::JoinSet`.
- **Max iterations** — After `max_iterations` model invocations, returns `Err(DaimonError::MaxIterations(n))`.
- **Cancellation** — Before each iteration, checks `CancellationToken`. If cancelled, returns `Err(DaimonError::Cancelled)`.
- **Budget exceeded** — If `max_budget` is set and `CostTracker` cumulative cost ≥ limit, returns `Err(DaimonError::BudgetExceeded { spent, limit })`.
- **Middleware short-circuit** — Middleware can return `ShortCircuit(ChatResponse)` to skip the model call or inject a response.

### Key Implementation Details

- **Zero-copy message handling** — `std::mem::take` moves `messages` and `tool_specs` in/out of `ChatRequest` to avoid cloning.
- **Tool execution** — Parallel via `JoinSet`. Each tool call is validated (if `validate_tool_inputs`), executed with optional retry, and results are collected in order.
- **Streaming** — `prompt_stream` runs the same loop but uses `generate_stream_erased`. Tool call deltas are accumulated; when complete, tools run and the model is re-invoked. Events are yielded via `async_stream::try_stream!`.

---

## 5. Trait Hierarchy

### Core Traits (daimon-core)

```rust
/// LLM provider. Implement to add a new model backend.
pub trait Model: Send + Sync {
    fn generate(&self, request: &ChatRequest) -> impl Future<Output = Result<ChatResponse>> + Send;
    fn generate_stream(&self, request: &ChatRequest) -> impl Future<Output = Result<ResponseStream>> + Send;
    /// Provider-side model id used for cost attribution. Defaults to `"default"`.
    fn model_id(&self) -> &str { "default" }
}

/// Embedding model for RAG and vector search.
pub trait EmbeddingModel: Send + Sync {
    fn embed(&self, texts: &[&str]) -> impl Future<Output = Result<Vec<Vec<f32>>>> + Send;
    fn dimensions(&self) -> usize;
}

/// Low-level vector storage. Pre-computed embeddings only.
pub trait VectorStore: Send + Sync {
    fn upsert(&self, id: &str, embedding: Vec<f32>, document: Document) -> impl Future<Output = Result<()>> + Send;
    /// Bulk upsert; default calls `upsert` per item. Backends with a bulk API override it.
    fn upsert_many(&self, items: Vec<(String, Vec<f32>, Document)>) -> impl Future<Output = Result<()>> + Send { /* ... */ }
    fn query(&self, embedding: Vec<f32>, top_k: usize) -> impl Future<Output = Result<Vec<ScoredDocument>>> + Send;
    fn delete(&self, id: &str) -> impl Future<Output = Result<bool>> + Send;
    fn count(&self) -> impl Future<Output = Result<usize>> + Send;
}

/// Distributed task broker for multi-process agent execution.
pub trait TaskBroker: Send + Sync {
    fn submit(&self, task: AgentTask) -> impl Future<Output = Result<String>> + Send;
    fn status(&self, task_id: &str) -> impl Future<Output = Result<TaskStatus>> + Send;
    fn receive(&self) -> impl Future<Output = Result<Option<AgentTask>>> + Send;
    /// Whether `receive()` returning `None` means permanently closed (true) or just idle (default false).
    fn none_means_closed(&self) -> bool { false }
    fn complete(&self, task_id: &str, result: TaskResult) -> impl Future<Output = Result<()>> + Send;
    fn fail(&self, task_id: &str, error: String) -> impl Future<Output = Result<()>> + Send;
}
```

### Main Crate Traits

`Tool` and `Memory` also live in **daimon-core** (`daimon_core::Tool`, `daimon_core::Memory`) — the main crate re-exports them so providers/plugins can implement them without depending on `daimon` itself. The remaining traits below are defined by the main crate.

```rust
/// Tool the agent can invoke. JSON Schema for parameters. (daimon-core)
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;
    fn execute(&self, input: &serde_json::Value) -> impl Future<Output = Result<ToolOutput>> + Send;
    fn retry_policy(&self) -> Option<ToolRetryPolicy> { None }
}

/// Conversation memory. Stores and retrieves messages. (daimon-core)
pub trait Memory: Send + Sync {
    fn add_message(&self, message: &Message) -> impl Future<Output = Result<()>> + Send;
    fn get_messages(&self) -> impl Future<Output = Result<Vec<Message>>> + Send;
    /// Borrowing read over the stored messages (default: clones via `get_messages`).
    fn with_messages<R, F>(&self, f: F) -> impl Future<Output = Result<R>> + Send
    where
        F: FnOnce(&[Message]) -> R + Send,
        R: Send,
    { /* ... */ }
    fn clear(&self) -> impl Future<Output = Result<()>> + Send;
}

/// Lifecycle callbacks during agent execution.
pub trait AgentHook: Send + Sync {
    fn on_iteration_start(&self, state: &AgentState) -> impl Future<Output = Result<()>> + Send;
    fn on_model_response(&self, response: &ChatResponse) -> impl Future<Output = Result<()>> + Send;
    fn on_tool_call(&self, call: &ToolCall) -> impl Future<Output = Result<()>> + Send;
    fn on_tool_result(&self, call: &ToolCall, result: &ToolOutput) -> impl Future<Output = Result<()>> + Send;
    fn on_iteration_end(&self, state: &AgentState) -> impl Future<Output = Result<()>> + Send;
    fn on_error(&self, error: &DaimonError) -> impl Future<Output = Result<()>> + Send;
}

/// Request/response/tool_call interception. Can short-circuit.
pub trait Middleware: Send + Sync {
    fn on_request(&self, request: &mut ChatRequest) -> impl Future<Output = Result<MiddlewareAction>> + Send;
    fn on_response(&self, response: &mut ChatResponse) -> impl Future<Output = Result<MiddlewareAction>> + Send;
    fn on_tool_call(&self, call: &mut ToolCall) -> impl Future<Output = Result<MiddlewareAction>> + Send;
}

/// Input validation before the model sees it.
pub trait InputGuardrail: Send + Sync {
    fn check(&self, input: &str, messages: &[Message]) -> impl Future<Output = Result<GuardrailResult>> + Send;
}

/// Output validation before returning to the caller.
pub trait OutputGuardrail: Send + Sync {
    fn check(&self, response: &ChatResponse) -> impl Future<Output = Result<GuardrailResult>> + Send;
}

/// Document retrieval (vector store, search engine, etc.).
pub trait Retriever: Send + Sync {
    fn retrieve(&self, query: &str, top_k: usize) -> impl Future<Output = Result<Vec<Document>>> + Send;
}

/// High-level RAG: ingest, search, remove, count.
pub trait KnowledgeBase: Send + Sync {
    fn ingest(&self, documents: Vec<Document>) -> impl Future<Output = Result<Vec<String>>> + Send;
    fn search(&self, query: &str, top_k: usize) -> impl Future<Output = Result<Vec<Document>>> + Send;
    fn remove(&self, id: &str) -> impl Future<Output = Result<bool>> + Send;
    fn count(&self) -> impl Future<Output = Result<usize>> + Send;
}

/// Checkpoint persistence for resumable runs.
pub trait Checkpoint: Send + Sync {
    fn save(&self, state: &CheckpointState) -> impl Future<Output = Result<()>> + Send;
    fn load(&self, run_id: &str) -> impl Future<Output = Result<Option<CheckpointState>>> + Send;
    fn list_runs(&self) -> impl Future<Output = Result<Vec<String>>> + Send;
    fn delete(&self, run_id: &str) -> impl Future<Output = Result<()>> + Send;
}

/// Token cost model for budget tracking.
pub trait CostModel: Send + Sync {
    fn cost_per_token(&self, model_id: &str, direction: TokenDirection) -> f64;
}
```

### Trait Relationships

```
Model ──────────────────────────────────────────────────────────────┐
  │                                                                   │
  └──► ChatRequest, ChatResponse, Message, ToolSpec, Usage             │
                                                                      │
EmbeddingModel ───────────────────────────────────────────────────────┤
  │                                                                   │
  └──► embed(), dimensions()                                          │
                                                                      │
VectorStore ◄─── SimpleKnowledgeBase ◄─── Retriever ◄─── RetrieverTool │
  │                     │                    │                         │
  └──► Document, ScoredDocument               └──► Tool                 │
                                                                      │
TaskBroker ───► AgentTask, TaskResult, TaskStatus                     │
                                                                      │
Tool ───► ToolRegistry ───► Agent                                      │
Memory ───► Agent                                                       │
AgentHook ───► Agent                                                   │
Middleware ───► MiddlewareStack ───► Agent                              │
InputGuardrail, OutputGuardrail ───► Agent                              │
CostModel ───► CostTracker ───► Agent                                  │
Checkpoint ───► Agent::prompt_resumable / replay / fork_from_checkpoint                                         │
```

---

## 6. Object-Safety Pattern

Rust traits with `impl Future` return types are not object-safe (the compiler cannot know the future size). Daimon uses the **Erased* + Shared*** pattern for every async trait.

### Pattern

1. **Static trait** — Uses `impl Future<Output = T> + Send` for zero-cost static dispatch.
2. **Erased trait** — Object-safe wrapper with `Pin<Box<dyn Future<Output = T> + Send + 'a>>`.
3. **Blanket impl** — `impl<T: Trait> ErasedTrait for T` forwards to the static trait.
4. **Shared type alias** — `type SharedTrait = Arc<dyn ErasedTrait>`.

### Example: Model

```rust
pub trait Model: Send + Sync {
    fn generate(&self, request: &ChatRequest) -> impl Future<Output = Result<ChatResponse>> + Send;
    fn generate_stream(&self, request: &ChatRequest) -> impl Future<Output = Result<ResponseStream>> + Send;
}

pub trait ErasedModel: Send + Sync {
    fn generate_erased<'a>(&'a self, request: &'a ChatRequest)
        -> Pin<Box<dyn Future<Output = Result<ChatResponse>> + Send + 'a>>;
    fn generate_stream_erased<'a>(&'a self, request: &'a ChatRequest)
        -> Pin<Box<dyn Future<Output = Result<ResponseStream>> + Send + 'a>>;
    fn model_id_erased(&self) -> &str;
}

impl<T: Model> ErasedModel for T {
    fn generate_erased<'a>(&'a self, request: &'a ChatRequest)
        -> Pin<Box<dyn Future<Output = Result<ChatResponse>> + Send + 'a>>
    {
        Box::pin(self.generate(request))
    }
    fn generate_stream_erased<'a>(&'a self, request: &'a ChatRequest)
        -> Pin<Box<dyn Future<Output = Result<ResponseStream>> + Send + 'a>>
    {
        Box::pin(self.generate_stream(request))
    }
}

pub type SharedModel = Arc<dyn ErasedModel>;
```

### Traits Using This Pattern

| Trait | Erased* | Shared* |
|-------|---------|---------|
| Model | ErasedModel | SharedModel |
| EmbeddingModel | ErasedEmbeddingModel | SharedEmbeddingModel |
| VectorStore | ErasedVectorStore | SharedVectorStore |
| TaskBroker | ErasedTaskBroker | (no alias; used directly) |
| Tool | ErasedTool | SharedTool |
| Memory | ErasedMemory | SharedMemory |
| AgentHook | ErasedAgentHook | (Arc<dyn ErasedAgentHook>) |
| Middleware | ErasedMiddleware | SharedMiddleware |
| InputGuardrail | ErasedInputGuardrail | (Vec<Arc<dyn ErasedInputGuardrail>>) |
| OutputGuardrail | ErasedOutputGuardrail | (Vec<Arc<dyn ErasedOutputGuardrail>>) |
| Retriever | ErasedRetriever | SharedRetriever |
| KnowledgeBase | ErasedKnowledgeBase | SharedKnowledgeBase |
| Checkpoint | ErasedCheckpoint | (Arc<dyn ErasedCheckpoint>) |

### Usage

- **Static dispatch** — When the concrete type is known at compile time, use `impl Model` or `M: Model`. No heap allocation, no vtable.
- **Dynamic dispatch** — When you need heterogeneous collections or runtime plugin loading, use `SharedModel` (`Arc<dyn ErasedModel>`). The agent stores `SharedModel` so users can pass any provider.

---

## 7. Feature Flags

Every provider, storage backend, and protocol is behind a feature flag. The core framework compiles with no features (or with `default`).

### Complete Feature Matrix

| Feature | Pulls In | Description |
|---------|----------|-------------|
| **default** | openai, anthropic, macros, ollama | Common providers + tool macro + local Ollama |
| **macros** | daimon-macros | `#[tool_fn]` proc macro |
| **openai** | daimon-provider-openai | OpenAI API provider (GPT-4, etc.) |
| **anthropic** | daimon-provider-anthropic | Anthropic Claude API |
| **gemini** | daimon-provider-gemini | Google Gemini / Vertex AI |
| **azure** | daimon-provider-azure | Azure OpenAI Service |
| **bedrock** | daimon-provider-bedrock | AWS Bedrock |
| **ollama** | reqwest, daimon-provider-local | Ollama local models |
| **llamacpp** | daimon-provider-local | llama.cpp (llama-server) provider |
| **llamars** | daimon-provider-local | llama-rs provider |
| **local** | daimon-provider-local | All local providers (Ollama, llama.cpp, llama-rs, OpenAI-compatible) |
| **a2a** | reqwest | Agent-to-Agent (A2A) protocol client |
| **sqs** | bedrock, daimon-provider-bedrock/sqs | AWS SQS task broker |
| **pubsub** | gemini, daimon-provider-gemini/pubsub | Google Cloud Pub/Sub task broker |
| **servicebus** | azure, daimon-provider-azure/servicebus | Azure Service Bus task broker |
| **sqlite** | rusqlite | SQLite memory backend |
| **redis** | redis | Redis memory + task broker |
| **nats** | async-nats | NATS JetStream task broker |
| **amqp** | lapin | RabbitMQ (AMQP) task broker |
| **mcp** | reqwest, tokio-tungstenite | Model Context Protocol client & server |
| **otel** | opentelemetry, opentelemetry_sdk, opentelemetry-otlp, tracing-opentelemetry, tracing-subscriber | OpenTelemetry OTLP span export |
| **http-server** | axum, tower-http | REST API server (POST /prompt, SSE) |
| **qdrant** | qdrant-client | Qdrant vector store retriever |
| **pgvector** | daimon-plugin-pgvector | pgvector-backed VectorStore |
| **opensearch** | daimon-plugin-opensearch | OpenSearch k-NN VectorStore |
| **grpc** | tonic, prost, tonic-prost | gRPC transport for distributed execution |
| **eval** | (none) | EvalScenario, EvalRunner, Scorer |
| **full** | All of the above | Everything enabled |

### Dependency Graph (Simplified)

```
daimon
├── daimon-core (always)
├── daimon-macros (macros)
├── daimon-provider-openai (openai)
├── daimon-provider-anthropic (anthropic)
├── daimon-provider-bedrock (bedrock)
├── daimon-provider-gemini (gemini)
├── daimon-provider-azure (azure)
├── daimon-provider-local (ollama | llamacpp | llamars | local)
├── daimon-plugin-pgvector (pgvector)
├── daimon-plugin-opensearch (opensearch)
├── reqwest (ollama | mcp | a2a)
├── rusqlite (sqlite)
├── redis (redis)
├── async-nats (nats)
├── lapin (amqp)
├── tokio-tungstenite (mcp)
├── opentelemetry* (otel)
├── axum, tower-http (http-server)
├── qdrant-client (qdrant)
├── tonic, prost (grpc)
└── ...
```

### Compiling Without Optional Deps

```bash
cargo build --no-default-features   # Core only; no providers
cargo build --no-default-features -F gemini   # Core + Gemini only
cargo build -F full   # Everything
```

---

## 8. Error Model

All fallible operations return `Result<T>`, an alias for `std::result::Result<T, DaimonError>`.

### DaimonError Variants

```rust
#[derive(Error, Debug)]
#[non_exhaustive]  // match arms need a wildcard arm
pub enum DaimonError {
    #[error("model error: {0}")]
    Model(String),

    #[error("tool execution failed for '{tool}': {message}")]
    ToolExecution { tool: String, message: String },

    #[error("tool '{0}' not found in registry")]
    ToolNotFound(String),

    #[error("duplicate tool '{0}' in registry")]
    DuplicateTool(String),

    #[error("agent builder validation failed: {0}")]
    Builder(String),

    #[error("max iterations ({0}) exceeded")]
    MaxIterations(usize),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("schema validation failed for tool '{tool}': {errors}")]
    SchemaValidation { tool: String, errors: String },

    #[error("stream closed unexpectedly")]
    StreamClosed,

    #[error("request timed out after {0:?}")]
    Timeout(std::time::Duration),

    #[error("operation cancelled")]
    Cancelled,

    #[error("orchestration error: {0}")]
    Orchestration(String),

    #[error("MCP error: {0}")]
    Mcp(String),

    #[error("budget exceeded: ${spent:.6} spent, limit was ${limit:.6}")]
    BudgetExceeded { spent: f64, limit: f64 },

    #[error("guardrail blocked: {0}")]
    GuardrailBlocked(String),

    /// Storage backend failure (checkpoint, memory, broker state).
    /// `transient` = retry may succeed.
    #[error("storage error: {message}")]
    Storage { message: String, transient: bool },

    #[error("{0}")]
    Other(String),
}
```

### Provider Error Mapping

Provider crates (HTTP, gRPC, SDK) map their errors to `DaimonError`:

```rust
// Example: in daimon-provider-gemini
.map_err(|e| DaimonError::Model(e.to_string()))
```

Use `DaimonError::Model(String)` for API errors, rate limits, bad responses. Use `DaimonError::Other(String)` for unexpected or uncategorized failures.

---

## 9. Async Runtime

### Tokio

Daimon uses **tokio** as the async runtime. All traits assume a tokio executor. The main crate depends on `tokio` with `features = ["full"]`; daimon-core uses `features = ["rt"]` for minimal footprint.

### Send + Sync

Every trait is `Send + Sync`. Types used across async boundaries (agent, model, tools, memory) must be `Send` so they can be moved between tasks. `Sync` allows shared references across threads (e.g., `Arc<dyn ErasedModel>`).

### Cancellation

Cooperative cancellation via `tokio_util::sync::CancellationToken`:

```rust
pub async fn prompt_with_cancellation(&self, input: &str, cancel: &CancellationToken) -> Result<AgentResponse>
```

The ReAct loop checks `cancel.is_cancelled()` before each iteration. If cancelled, returns `Err(DaimonError::Cancelled)`.

### Streaming

Streaming uses `async-stream` and `futures::Stream`:

```rust
pub type ResponseStream = Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>;
```

Events: `TextDelta`, `ToolCallStart`, `ToolCallDelta`, `ToolCallEnd`, `ToolResult`, `Usage`, `Error`, `Done`. The stream runs the full ReAct loop internally; tool calls are accumulated and executed between model invocations.

---

## 10. Orchestration Layer

Beyond the single-agent ReAct loop, Daimon provides multi-step orchestration:

### Chain

Linear pipeline of steps. Each step transforms input to output; context flows sequentially.

```rust
pub trait ChainStep: Send + Sync {
    fn process<'a>(&'a self, ctx: ChainContext)
        -> Pin<Box<dyn Future<Output = Result<ChainContext>> + Send + 'a>>;
}
```

`Chain` runs steps in order. `AgentStep` wraps an agent; `TransformStep` wraps a closure. Use for simple pipelines: prompt → agent → post-process.

### Graph

Directed graph with conditional nodes. Supports cycles, fan-out, fan-in. Pregel-style sequential walker.

```rust
pub trait GraphNode: Send + Sync {
    fn process<'a>(&'a self, ctx: GraphContext)
        -> Pin<Box<dyn Future<Output = Result<NodeOutcome>> + Send + 'a>>;
}
```

`NodeOutcome` can route to another node by name. Use for branching workflows: router → specialist A or B → aggregator.

### DAG

DagBuilder builds a DAG at construction time. Independent nodes run in parallel; dependencies are enforced. AllPredecessor model: a node runs when all predecessors complete.

```rust
pub struct Dag { /* nodes, edges */ }
pub struct DagBuilder { /* ... */ }
```

Use for parallelizable workflows: fetch data from N sources → merge → agent.

### Workflow

Extends DAG with typed field-level data mapping. Nodes read/write named fields; the engine wires outputs to inputs by field name. Inspired by Eino's typed-field workflow engine.

```rust
pub trait WorkflowNode: Send + Sync {
    // Nodes are JSON-in / JSON-out; typed field mapping happens at the edges.
    fn process<'a>(&'a self, input: serde_json::Value)
        -> Pin<Box<dyn Future<Output = Result<serde_json::Value>> + Send + 'a>>;
}
```

---

## 11. Distributed Execution

For multi-process agent execution, Daimon uses the `TaskBroker` trait from daimon-core.

### Flow

1. **Submit** — Client submits `AgentTask` (input, optional run_id, metadata). Broker returns task_id.
2. **Receive** — Worker blocks on `receive()` until a task is available.
3. **Execute** — Worker runs the agent loop with the task input.
4. **Complete/Fail** — Worker calls `complete(task_id, result)` or `fail(task_id, error)`.
5. **Status** — Client polls `status(task_id)` for Pending/Running/Completed/Failed.

### Broker Implementations

| Feature | Broker | Backend |
|---------|--------|---------|
| sqs | SqsBroker | AWS SQS (via daimon-provider-bedrock) |
| pubsub | PubSubBroker | Google Cloud Pub/Sub (via daimon-provider-gemini) |
| servicebus | ServiceBusBroker | Azure Service Bus (via daimon-provider-azure) |
| redis | RedisBroker | Redis |
| nats | NatsBroker | NATS JetStream |
| amqp | AmqpBroker | RabbitMQ (AMQP) |
| grpc | gRPC transport | Tonic + Prost |

### Worker Pattern

```rust
let broker = RedisBroker::new(redis_url, "daimon").await?;
loop {
    let task = broker.receive().await?.ok_or(...)?;
    let result = agent.prompt(&task.input).await?;
    broker.complete(&task.task_id, TaskResult { ... }).await?;
}
```

---

## 12. MCP (Model Context Protocol)

The `mcp` feature enables the Model Context Protocol for tool discovery and execution across process boundaries.

### Client

Connects to MCP servers (stdio, HTTP, WebSocket). Discovers tools from the server and registers them with the agent. Tool calls are forwarded to the remote server.

### Server

Exposes Daimon tools to external MCP clients. Tools registered in the agent are advertised; clients invoke them via the protocol.

### Transports

- **stdio** — For local subprocess servers.
- **HTTP/SSE** — For remote HTTP-based servers.
- **WebSocket** — For bidirectional streaming.

---

## 13. A2A (Agent-to-Agent)

The `a2a` module provides Google's Agent-to-Agent protocol support for inter-agent communication. Agents can send structured messages to other agents.

---

## 14. Checkpointing and Resumable Runs

The `Checkpoint` trait enables resumable agent execution. When a run is interrupted, the agent can load the last checkpoint and continue.

### CheckpointState

Stores messages, iteration count, and any custom state. Implementations persist to file, Redis, or NATS KV.

### Implementations

| Backend | Feature | Use Case |
|---------|---------|----------|
| FileCheckpoint | (built-in) | Local development |
| RedisCheckpoint | redis | Production |
| NatsKvCheckpoint | nats | Distributed |

### Resumable Runs

There is no separate wrapper type: `Agent` itself offers `prompt_resumable(input, run_id, &Arc<dyn ErasedCheckpoint>)`. When called with a `run_id`, it loads the checkpoint (if any), restores the message history, and continues from the last state — saving state at each iteration so an interrupted run can resume. Related: `agent.replay(run_id, checkpoint, up_to)` for time-travel debugging and `agent.fork_from_checkpoint(run_id, &checkpoint)` to branch a run.

---

## 15. Cost Tracking and Budget

Attach a `CostModel` to an agent to track token spend. Set `max_budget` to abort when a dollar limit is reached.

### CostModel

```rust
pub trait CostModel: Send + Sync {
    fn cost_per_token(&self, model_id: &str, direction: TokenDirection) -> f64;
}
```

`TokenDirection::Input` or `Output`. Built-in: `OpenAiCostModel`, `AnthropicCostModel`.

### CostTracker

Wraps a `CostModel`. On each model response, `record(model_id, usage)` computes cost and accumulates. `cumulative_cost()` returns total spent. The ReAct loop checks `cumulative_cost() >= max_budget` before each iteration.

---

## 16. Prompt System

The `prompt` module provides:

- **PromptTemplate** — String templates with `{{variable}}` placeholders.
- **Builder** — Fluent API for building prompts with variables.
- **Few-shot** — Inject example messages for in-context learning.
- **Dynamic** — Runtime prompt construction from context.

---

## 17. Eval (Evaluation)

The `eval` feature provides a testing harness for agent behavior:

- **EvalScenario** — Input + expected outcome (e.g., `expect_contains("4")`).
- **EvalRunner** — Runs scenarios through an agent, collects results.
- **EvalResult** — Pass/fail, output, iterations, latency, cost, per-scorer results.
- **Scorer** — Custom scoring logic.

---

## 18. Shared Types (daimon-core)

Types used across the plugin boundary. All defined in `daimon-core` and re-exported by `daimon`:

| Type | Description |
|------|-------------|
| `Message` | Conversation message: role (System/User/Assistant/Tool), content, tool_calls, tool_call_id |
| `ChatRequest` | messages, tools (ToolSpec), temperature, max_tokens |
| `ChatResponse` | message, stop_reason (`StopReason`: EndTurn/ToolUse/MaxTokens/Refusal/ContentFiltered/PauseTurn — `#[non_exhaustive]`), usage |
| `ToolSpec` | name, description, parameters (JSON Schema) |
| `ToolCall` | id, name, arguments (JSON) |
| `Usage` | input_tokens, output_tokens, cached_tokens |
| `Document` | content, metadata, score |
| `ScoredDocument` | id, document, score |
| `StreamEvent` | TextDelta, ToolCallStart, ToolCallDelta, ToolCallEnd, ToolResult, Usage, Error, Done |
| `AgentTask` | task_id, input, run_id, metadata |
| `TaskResult` | task_id, output, iterations, cost, error |
| `TaskStatus` | Pending, Running, Completed(TaskResult), Failed(String) |

---

## 19. Memory Implementations

### SlidingWindowMemory (default)

In-memory store that keeps only the last N messages. Configurable window size (default 50). No persistence; suitable for single-session conversations.

### SqliteMemory (sqlite feature)

Persists messages to SQLite. Survives process restarts. Use for development or low-concurrency production.

### RedisMemory (redis feature)

Stores messages in Redis. Supports multi-process or multi-instance deployments. Keyed by session ID.

### TokenWindowMemory

Standalone in-memory store (own message log + token counter). Truncates history to fit within a token budget (estimated by character count, ~4 chars/token, overridable via `with_token_counter`). Ensures the context never exceeds model limits.

---

## 20. Tool Registry and Validation

### ToolRegistry

Stores tools as `HashMap<String, Arc<dyn ErasedTool>>`. `register(tool)` inserts by `tool.name()`; duplicate names return `Err(DuplicateTool)`. `tool_specs()` builds `Vec<ToolSpec>` for the model (name, description, parameters_schema).

### Schema Validation

When `validate_tool_inputs` is true, each tool call's `arguments` are validated against the tool's `parameters_schema` via `jsonschema`. Invalid inputs produce an error message returned to the model (so it can correct on the next iteration) rather than executing the tool.

### Retry Policy

Tools can declare `retry_policy()` returning `Option<ToolRetryPolicy>`. The policy specifies max_retries, backoff (exponential), and `is_retryable(&str)` to decide if an error message warrants a retry. Agent-level `tool_retry_policy` applies when the tool does not override.

---

## 21. Multi-Agent Patterns

### Agent-as-Tool (AgentTool)

Wraps an agent as a `Tool`. The tool's `execute` runs `agent.prompt(input)`. Use when one agent needs to delegate to another (e.g., a router agent calling specialist agents).

### Supervisor

One agent delegates to specialized sub-agents. Sub-agents are registered as `(name, agent, description)` triples and exposed to a coordinator agent as `AgentTool`s; the coordinator LLM picks which sub-agent to call and returns the result.

### HandoffNetwork

Agents transfer control to each other mid-conversation. A handoff is a synthetic `transfer_to_<name>` tool call carrying a reason string; the conversation history is shared, so the target agent picks up with full context. Use for multi-agent conversations where control passes between participants.

### StructuredOutput

Extracts typed data from LLM responses via `agent.prompt_structured::<T>(input, type_name)`. It instructs the model (textually) to respond with only valid JSON for the named type and deserializes with serde — no JSON Schema is generated. Retries up to 3 total attempts on parse failure, then errors.

---

## 22. Middleware Stack

`MiddlewareStack` holds a list of `Arc<dyn ErasedMiddleware>`. On each request, response, and tool call, the stack runs middleware in order. Any middleware can return `ShortCircuit(ChatResponse)` to skip the rest of the pipeline and return immediately.

Use cases: caching, logging, rate limiting, request/response transformation, A/B testing (inject different prompts).

---

## 23. Proc Macro: tool_fn

The `#[tool_fn]` macro derives a `Tool` implementation from an async function:

```rust
/// Adds two numbers together.
#[tool_fn]
async fn add(
    /// The first number.
    a: f64,
    /// The second number.
    b: f64,
) -> daimon::Result<ToolOutput> {
    Ok(ToolOutput::text(format!("{}", a + b)))
}
```

- Function name → tool name (or override via `name = "..."`).
- Doc comments → tool description (or override via `description = "..."`).
- Parameters → JSON Schema properties. `Option<T>` marks optional.
- Supported types: `String`, `i8`–`i128`, `u8`–`u128`, `f32`, `f64`, `bool`, `Option<T>`, `Vec<T>`, `serde_json::Value`.

---

## 24. Summary

Daimon is a Rust-native AI agent framework with:

- **daimon-core** as the stable plugin boundary (Model, EmbeddingModel, VectorStore, TaskBroker + shared types).
- **Provider/plugin crates** that depend only on daimon-core.
- **ReAct loop** as the central execution model: prompt → model → tool calls (optional) → loop until done.
- **Erased* + Shared*** pattern for object-safe async traits.
- **Feature flags** for every optional component; zero unused deps.
- **Single DaimonError** enum (`#[non_exhaustive]`); providers map to `Model`, storage backends to `Storage`, anything else to `Other`.
- **Tokio** async runtime; `CancellationToken` for cancellation; `async-stream` for streaming.

To contribute: implement a trait from daimon-core, add a feature flag, and wire it in the main crate.
