# Plugin Development Guide

This guide teaches you how to extend the Daimon AI agent framework by writing your own plugins. Daimon uses a trait-based plugin architecture: implement the appropriate trait from `daimon-core`, and your implementation can be composed with the rest of the framework.

---

## Plugin Architecture

The **`daimon-core`** crate is the plugin interface. It defines the traits that plugins implement:

| Trait | Purpose |
|-------|---------|
| `Model` | LLM providers (chat completion, streaming) |
| `EmbeddingModel` | Text embedding providers for RAG |
| `VectorStore` | Vector databases for semantic search |
| `TaskBroker` | Distributed task queues for multi-process execution |

**Critical rule:** Plugin crates depend **only** on `daimon-core`. They never depend on the main `daimon` crate. The main crate optionally depends on plugins via feature flags and re-exports them.

```
┌─────────────────────────────────────────────────────────────────┐
│ daimon (main crate)                                              │
│   - Agent, tools, memory, orchestration                          │
│   - Re-exports plugins when features enabled                     │
├─────────────────────────────────────────────────────────────────┤
│ daimon-core (plugin interface)                                  │
│   - Model, EmbeddingModel, VectorStore, TaskBroker traits       │
│   - ChatRequest, ChatResponse, Document, AgentTask, etc.         │
├─────────────────────────────────────────────────────────────────┤
│ daimon-provider-* / daimon-plugin-* (plugin crates)              │
│   - Depend ONLY on daimon-core                                  │
│   - Implement traits, never import daimon                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Writing a Model Provider

A model provider implements the `Model` trait to add a new LLM backend. Follow these steps.

### 1. Create a New Crate

```bash
mkdir daimon-provider-myllm
cd daimon-provider-myllm
```

**Cargo.toml:**

```toml
[package]
name = "daimon-provider-myllm"
version = "0.16.0"
edition = "2024"
rust-version = "1.85"
description = "MyLLM provider for the Daimon AI agent framework"
license = "MIT OR Apache-2.0"
repository = "https://github.com/Lexmata/daimon"
keywords = ["ai", "agent", "llm", "myllm"]
categories = ["asynchronous", "api-bindings"]

[dependencies]
daimon-core = { version = "0.16.0", path = "../daimon-core" }
tokio = { version = "1", features = ["rt", "time", "sync"] }
reqwest = { version = "0.12", features = ["json", "rustls-tls", "stream"], default-features = false }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tracing = "0.1"
async-stream = "0.3"
futures = "0.3"
```

### 2. Implement the Model Trait

The `Model` trait has two methods:

- `generate(&self, request: &ChatRequest) -> Result<ChatResponse>` — non-streaming
- `generate_stream(&self, request: &ChatRequest) -> Result<ResponseStream>` — streaming

```rust
// src/lib.rs

use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};

use daimon_core::{
    ChatRequest, ChatResponse, DaimonError, Message, Model, ResponseStream, Result, Role,
    StopReason, StreamEvent, ToolCall, ToolSpec, Usage,
};

/// MyLLM model provider.
#[derive(Debug)]
pub struct MyLLM {
    client: Client,
    api_key: String,
    model_id: String,
    base_url: String,
}

impl MyLLM {
    pub fn new(model_id: impl Into<String>) -> Self {
        let api_key = std::env::var("MYLLM_API_KEY").unwrap_or_default();
        Self::with_api_key(model_id, api_key)
    }

    pub fn with_api_key(model_id: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .expect("HTTP client"),
            api_key: api_key.into(),
            model_id: model_id.into(),
            base_url: "https://api.myllm.com/v1".to_string(),
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    fn build_request_body(&self, request: &ChatRequest) -> MyLLMRequest {
        // Convert ChatRequest messages to your API's format
        let messages: Vec<MyLLMMessage> = request.messages.iter().map(|m| {
            let role = match m.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::Tool => "tool",
            };
            MyLLMMessage {
                role: role.to_string(),
                content: m.content.clone().unwrap_or_default(),
                tool_calls: m.tool_calls.iter().map(|tc| MyLLMToolCall {
                    id: tc.id.clone(),
                    name: tc.name.clone(),
                    arguments: tc.arguments.clone(),
                }).collect(),
                tool_call_id: m.tool_call_id.clone(),
            }
        }).collect();

        let tools = if request.tools.is_empty() {
            None
        } else {
            Some(request.tools.iter().map(|t| MyLLMToolSpec {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.parameters.clone(),
            }).collect())
        };

        MyLLMRequest {
            model: self.model_id.clone(),
            messages,
            tools,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
        }
    }
}

impl Model for MyLLM {
    async fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let body = self.build_request_body(request);
        let url = format!("{}/chat/completions", self.base_url);

        let response = self.client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| DaimonError::Model(format!("MyLLM HTTP error: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(DaimonError::Model(format!("MyLLM API error ({status}): {text}")));
        }

        let api_resp: MyLLMResponse = response
            .json()
            .await
            .map_err(|e| DaimonError::Model(format!("MyLLM parse error: {e}")))?;

        parse_response(api_resp)
    }

    async fn generate_stream(&self, request: &ChatRequest) -> Result<ResponseStream> {
        let body = self.build_request_body(request);
        let url = format!("{}/chat/completions", self.base_url);

        let response = self.client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| DaimonError::Model(format!("MyLLM HTTP error: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(DaimonError::Model(format!("MyLLM API error ({status}): {text}")));
        }

        let byte_stream = response.bytes_stream();

        let stream = async_stream::try_stream! {
            use futures::StreamExt;

            let mut buffer = String::new();
            let mut stream = Box::pin(byte_stream);

            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| DaimonError::Model(format!("stream error: {e}")))?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                // Parse SSE or NDJSON lines, emit StreamEvent::TextDelta, ToolCallStart, etc.
                // When done: yield StreamEvent::Done;
            }
        };

        Ok(Box::pin(stream))
    }
}

fn parse_response(response: MyLLMResponse) -> Result<ChatResponse> {
    let choice = response.choices.into_iter().next()
        .ok_or_else(|| DaimonError::Model("no choices in response".into()))?;

    let message = Message::assistant(choice.message.content.unwrap_or_default());
    let stop_reason = match choice.finish_reason.as_deref() {
        Some("tool_calls") => StopReason::ToolUse,
        Some("length") => StopReason::MaxTokens,
        _ => StopReason::EndTurn,
    };

    Ok(ChatResponse {
        message,
        stop_reason,
        usage: response.usage.map(|u| Usage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            cached_tokens: 0,
        }),
    })
}

// Define API-specific request/response types (Serialize/Deserialize)
// for your provider's schema.
```

### Key Implementation Details

1. **Messages → API format**: Map `Role`, `Message`, `ToolCall` to your provider's schema. System messages may go in a separate field.
2. **Tool calls**: Convert `ToolSpec` to function declarations; convert `Message::tool_result` to function responses.
3. **Streaming deltas**: Emit `StreamEvent::TextDelta`, `ToolCallStart`, `ToolCallDelta`, `ToolCallEnd`, then `StreamEvent::Done`.
4. **Usage tracking**: Map `input_tokens`, `output_tokens` to `Usage`; set `cached_tokens` if supported.
5. **Error mapping**: Always map transport/API errors to `DaimonError::Model(String)`.

### Optional: EmbeddingModel

```rust
use daimon_core::{DaimonError, EmbeddingModel, Result};

pub struct MyLLMEmbedding { /* ... */ }

impl EmbeddingModel for MyLLMEmbedding {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // POST to embeddings endpoint, return Vec<Vec<f32>>
        todo!()
    }
    fn dimensions(&self) -> usize { 1536 }
}
```

---

## Writing a Vector Store Plugin

A vector store plugin implements `VectorStore` for a vector database. Combine it with `SimpleKnowledgeBase` for a full RAG pipeline.

### 1. Create a New Crate

```bash
mkdir daimon-plugin-myvectordb
cd daimon-plugin-myvectordb
```

**Cargo.toml:**

```toml
[package]
name = "daimon-plugin-myvectordb"
version = "0.16.0"
edition = "2024"
description = "MyVectorDB VectorStore plugin for Daimon"
license = "MIT OR Apache-2.0"
repository = "https://github.com/Lexmata/daimon"
keywords = ["ai", "agent", "rag", "vector-store"]

[dependencies]
daimon-core = { version = "0.16.0", path = "../daimon-core" }
tokio = { version = "1", features = ["rt"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tracing = "0.1"
# Add your DB client (e.g. deadpool-postgres, opensearch, etc.)
```

### 2. Implement VectorStore

```rust
// src/lib.rs

use std::collections::HashMap;

use daimon_core::{DaimonError, Document, Result, ScoredDocument, VectorStore};

mod builder;
mod store;

pub use builder::MyVectorStoreBuilder;
pub use store::MyVectorStore;

// src/store.rs

impl VectorStore for MyVectorStore {
    async fn upsert(&self, id: &str, embedding: Vec<f32>, document: Document) -> Result<()> {
        if embedding.len() != self.dimensions {
            return Err(DaimonError::Other(format!(
                "embedding dimension mismatch: expected {}, got {}",
                self.dimensions, embedding.len()
            )));
        }

        let client = self.pool.get().await
            .map_err(|e| DaimonError::Other(format!("pool error: {e}")))?;

        // INSERT or UPSERT into your DB
        // Serialize document.metadata as JSON
        client.execute("INSERT INTO vectors (id, embedding, content, metadata) VALUES ($1, $2, $3, $4) ON CONFLICT (id) DO UPDATE SET ...", &[&id, &embedding, &document.content, &serde_json::to_value(&document.metadata)?])
            .await
            .map_err(|e| DaimonError::Other(format!("upsert error: {e}")))?;

        Ok(())
    }

    async fn query(&self, embedding: Vec<f32>, top_k: usize) -> Result<Vec<ScoredDocument>> {
        let client = self.pool.get().await
            .map_err(|e| DaimonError::Other(format!("pool error: {e}")))?;

        // Run similarity search, return (id, content, metadata, score) rows
        let rows = client.query("SELECT id, content, metadata, similarity(embedding, $1) AS score FROM vectors ORDER BY embedding <-> $1 LIMIT $2", &[&embedding, &(top_k as i64)])
            .await
            .map_err(|e| DaimonError::Other(format!("query error: {e}")))?;

        let results: Vec<ScoredDocument> = rows.into_iter().map(|row| {
            let id: String = row.get("id");
            let content: String = row.get("content");
            let metadata: HashMap<String, serde_json::Value> = serde_json::from_value(row.get("metadata")).unwrap_or_default();
            let score: f64 = row.get("score");
            // `id` must be the same stable id the document was upserted
            // with, so callers can pass a search result straight to `delete`.
            ScoredDocument::new(id, Document { content, metadata, score: Some(score) }, score)
        }).collect();

        Ok(results)
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        let client = self.pool.get().await
            .map_err(|e| DaimonError::Other(format!("pool error: {e}")))?;
        let deleted = client.execute("DELETE FROM vectors WHERE id = $1", &[&id]).await
            .map_err(|e| DaimonError::Other(format!("delete error: {e}")))?;
        Ok(deleted > 0)
    }

    async fn count(&self) -> Result<usize> {
        let client = self.pool.get().await
            .map_err(|e| DaimonError::Other(format!("pool error: {e}")))?;
        let row = client.query_one("SELECT COUNT(*) AS cnt FROM vectors", &[]).await
            .map_err(|e| DaimonError::Other(format!("count error: {e}")))?;
        Ok(row.get::<_, i64>("cnt") as usize)
    }
}
```

### 3. Builder Pattern

```rust
// src/builder.rs

pub struct MyVectorStoreBuilder {
    connection_string: String,
    dimensions: usize,
    table: String,
    pool_size: usize,
}

impl MyVectorStoreBuilder {
    pub fn new(connection_string: impl Into<String>, dimensions: usize) -> Self {
        Self {
            connection_string: connection_string.into(),
            dimensions,
            table: "daimon_vectors".into(),
            pool_size: 16,
        }
    }

    pub fn table(mut self, table: impl Into<String>) -> Self {
        self.table = table.into();
        self
    }

    pub fn pool_size(mut self, size: usize) -> Self {
        self.pool_size = size;
        self
    }

    pub async fn build(self) -> Result<MyVectorStore> {
        let pool = self.create_pool()?;
        // Optionally run migrations
        Ok(MyVectorStore { pool, table: self.table, dimensions: self.dimensions })
    }
}
```

---

## Writing a Task Broker Plugin

A task broker distributes `AgentTask`s across workers. Implement `TaskBroker` for your message queue.

### 1. Create or Extend a Crate

Task brokers often live in provider crates (e.g. `daimon-provider-bedrock` has `SqsBroker`). You can add a new crate `daimon-plugin-myqueue` or extend an existing provider.

### 2. Implement TaskBroker

```rust
use daimon_core::distributed::{AgentTask, TaskBroker, TaskResult, TaskStatus};
use daimon_core::{DaimonError, Result};

pub struct MyQueueBroker {
    client: QueueClient,
    queue_url: String,
    status_store: Arc<Mutex<HashMap<String, TaskStatus>>>,
}

impl TaskBroker for MyQueueBroker {
    async fn submit(&self, task: AgentTask) -> Result<String> {
        let id = task.task_id.clone();
        let json = serde_json::to_string(&task)
            .map_err(|e| DaimonError::Other(format!("serialize: {e}")))?;

        self.status_store.lock().await.insert(id.clone(), TaskStatus::Pending);
        self.client.send_message(&self.queue_url, &json).await
            .map_err(|e| DaimonError::Other(format!("send: {e}")))?;

        Ok(id)
    }

    async fn status(&self, task_id: &str) -> Result<TaskStatus> {
        Ok(self.status_store.lock().await.get(task_id).cloned().unwrap_or(TaskStatus::Pending))
    }

    async fn receive(&self) -> Result<Option<AgentTask>> {
        let msg = self.client.receive_message(&self.queue_url).await
            .map_err(|e| DaimonError::Other(format!("receive: {e}")))?;

        let Some(msg) = msg else { return Ok(None) };

        let task: AgentTask = serde_json::from_str(&msg.body)
            .map_err(|e| DaimonError::Other(format!("deserialize: {e}")))?;

        self.status_store.lock().await.insert(task.task_id.clone(), TaskStatus::Running);
        // Store receipt_handle for ack on complete/fail
        Ok(Some(task))
    }

    async fn complete(&self, task_id: &str, result: TaskResult) -> Result<()> {
        // Acknowledge message, update status
        self.status_store.lock().await.insert(task_id.into(), TaskStatus::Completed(result));
        Ok(())
    }

    async fn fail(&self, task_id: &str, error: String) -> Result<()> {
        // Nack or move to DLQ
        self.status_store.lock().await.insert(task_id.into(), TaskStatus::Failed(error));
        Ok(())
    }
}
```

### Key Points

- `AgentTask` and `TaskResult` are `Serialize`/`Deserialize` — use JSON (or base64+JSON for binary transports).
- `status()` often uses an in-memory map; for production, use Redis, DB, or the broker's native status API.
- `receive()` blocks until a message is available; return `Ok(None)` if the broker is closed.

---

## Integrating with the Main Crate

To wire your plugin into `daimon`:

### 1. Add Workspace Member

In the root `Cargo.toml`:

```toml
[workspace]
members = [
    "daimon-core",
    "daimon-provider-myllm",   # or daimon-plugin-myvectordb
]
```

### 2. Add Optional Dependency and Feature

```toml
[features]
myllm = ["dep:daimon-provider-myllm"]
full = [..., "myllm"]

[dependencies]
daimon-provider-myllm = { path = "daimon-provider-myllm", version = "0.16.0", optional = true }
```

### 3. Add Re-export Module

**For a model provider** — in `src/model/mod.rs`:

```rust
#[cfg(feature = "myllm")]
pub mod myllm {
    pub use daimon_provider_myllm::*;
}
```

**For a vector store** — in `src/retriever/mod.rs`:

```rust
#[cfg(feature = "myvectordb")]
pub mod myvectordb {
    pub use daimon_plugin_myvectordb::*;
}
```

**For a task broker** — in `src/prelude.rs` or `src/distributed/mod.rs`:

```rust
#[cfg(feature = "myqueue")]
pub use daimon_plugin_myqueue::MyQueueBroker;
```

### 4. Add to Prelude (Optional)

In `src/prelude.rs`:

```rust
#[cfg(feature = "myllm")]
pub use daimon_provider_myllm::MyLLM;

#[cfg(feature = "myvectordb")]
pub use daimon_plugin_myvectordb::{MyVectorStore, MyVectorStoreBuilder};
```

---

## Publishing

### Version Alignment

Keep plugin versions aligned with the workspace. When releasing `daimon 0.17.0`, bump `daimon-core`, all providers, and all plugins to `0.17.0`.

### Cargo.toml Metadata

```toml
[package]
name = "daimon-provider-myllm"
version = "0.16.0"
edition = "2024"
description = "MyLLM provider for the Daimon AI agent framework"
license = "MIT OR Apache-2.0"
repository = "https://github.com/Lexmata/daimon"
homepage = "https://github.com/Lexmata/daimon"
documentation = "https://docs.rs/daimon-provider-myllm"
keywords = ["ai", "agent", "llm", "myllm"]
categories = ["asynchronous", "api-bindings"]
```

### Publishing Order

1. `daimon-core` (plugins depend on it)
2. `daimon-macros` (if used)
3. Plugin crates (order doesn't matter once core is published)
4. `daimon` (main crate last)

```bash
cargo publish -p daimon-core
cargo publish -p daimon-provider-myllm
cargo publish -p daimon
```

---

## Testing Plugins

### Unit Tests with Mocks

Use mock implementations for fast, isolated tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use daimon_core::{ChatRequest, Message, Model};

    #[test]
    fn test_builder_chain() {
        let model = MyLLM::with_api_key("model-1", "key")
            .with_base_url("https://custom.example.com");
        assert_eq!(model.model_id, "model-1");
    }

    #[test]
    fn test_request_body_build() {
        let model = MyLLM::with_api_key("m", "k");
        let req = ChatRequest {
            messages: vec![Message::user("hi")],
            tools: vec![],
            temperature: Some(0.7),
            max_tokens: Some(100),
        };
        let body = model.build_request_body(&req);
        assert_eq!(body.temperature, Some(0.7));
    }
}
```

### Integration Tests

When infrastructure is available (e.g. local PostgreSQL for pgvector), add integration tests in `tests/`:

```rust
// tests/integration.rs
#[tokio::test]
#[ignore] // Run with: cargo test --ignored
async fn test_myvectordb_upsert_query() {
    let store = MyVectorStoreBuilder::new("postgresql://...", 1536).build().await.unwrap();
    let doc = Document::new("test content");
    store.upsert("id1", vec![0.1; 1536], doc).await.unwrap();
    let results = store.query(vec![0.1; 1536], 5).await.unwrap();
    assert!(!results.is_empty());
}
```

### Erased* Pattern for Dynamic Dispatch

The `ErasedModel`, `ErasedVectorStore`, `ErasedTaskBroker` traits provide object-safe wrappers. Use them when testing with `Arc<dyn ErasedModel>`:

```rust
let model: Arc<dyn daimon_core::ErasedModel> = Arc::new(MyLLM::new("model-1"));
let response = model.generate_erased(&request).await?;
```

---

## Design Guidelines

| Guideline | Description |
|-----------|-------------|
| **Builder pattern** | Use builders for configuration (connection strings, timeouts, pool size). |
| **Error mapping** | Map all errors to `DaimonError::Model` or `DaimonError::Other`. Never expose provider-specific error types in the public API. |
| **Send + Sync** | All traits require `Send + Sync`; use `Arc` for shared ownership across async tasks. |
| **Connection pooling** | Use connection pools (e.g. `deadpool-postgres`) for network resources. |
| **Feature-gate heavy deps** | Put optional integrations behind features (e.g. `aws-auth` in opensearch). |
| **Rustdoc examples** | Document public types with `/// # Example` blocks; use `ignore` for examples that need env vars. |

### Example Doc Comment

```rust
/// MyLLM model provider.
///
/// # Example
///
/// ```ignore
/// use daimon_provider_myllm::MyLLM;
/// use daimon_core::Model;
///
/// let model = MyLLM::new("model-1");
/// let response = model.generate(&request).await?;
/// ```
pub struct MyLLM { /* ... */ }
```

---

## Quick Reference

| Plugin Type | Trait | Key Methods |
|-------------|-------|-------------|
| Model | `Model` | `generate`, `generate_stream` |
| Embedding | `EmbeddingModel` | `embed`, `dimensions` |
| Vector Store | `VectorStore` | `upsert`, `query`, `delete`, `count` |
| Task Broker | `TaskBroker` | `submit`, `status`, `receive`, `complete`, `fail` |

All types live in `daimon-core`. The main `daimon` crate re-exports them and composes plugins into agents, knowledge bases, and distributed workers.
