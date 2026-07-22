# RAG (Retrieval-Augmented Generation)

Daimon provides a complete RAG pipeline for embedding-backed document retrieval. This guide covers the three-layer architecture, embedding models, vector store implementations, and how to build end-to-end RAG pipelines with agent integration.

---

## RAG Architecture in Daimon

The RAG stack is organized in three layers:

| Layer | Trait / Type | Responsibility |
|-------|--------------|-----------------|
| **VectorStore** (low-level) | `VectorStore` | Pre-computed embeddings only. Upsert, query, delete, count. No embedding logic. |
| **KnowledgeBase** (mid-level) | `KnowledgeBase` | Embedding model + vector store. Auto-embeds on ingest and search. |
| **Retriever** (high-level) | `Retriever` | Unified `retrieve(query, top_k)` interface. Query-agnostic. |

`SimpleKnowledgeBase` implements both `KnowledgeBase` and `Retriever`, bridging all three layers. You can use it as a drop-in retriever or as a full knowledge base with ingest and search.

```
┌─────────────────────────────────────────────────────────────────┐
│ Application Layer (Agent, RetrieverTool)                         │
├─────────────────────────────────────────────────────────────────┤
│ Retriever Layer — retrieve(query, top_k) → Vec<Document>        │
│   SimpleKnowledgeBase, QdrantRetriever, custom impls            │
├─────────────────────────────────────────────────────────────────┤
│ KnowledgeBase Layer — ingest + search + remove + count           │
│   SimpleKnowledgeBase (embedding + embedding on query)          │
├─────────────────────────────────────────────────────────────────┤
│ VectorStore Layer — upsert, query, delete, count (embeddings)   │
│   InMemoryVectorStoreBackend, PgVectorStore, OpenSearchVectorStore│
└─────────────────────────────────────────────────────────────────┘
```

---

## Embedding Models

The `EmbeddingModel` trait defines the embedding interface:

```rust
pub trait EmbeddingModel: Send + Sync {
    fn embed(&self, texts: &[&str]) -> impl Future<Output = Result<Vec<Vec<f32>>>> + Send;
    fn dimensions(&self) -> usize;
}
```

Implementations are available from various providers. Use `Arc<dyn ErasedEmbeddingModel>` when composing with `SimpleKnowledgeBase` or `QdrantRetriever`.

### OpenAI Embeddings

```rust
use daimon::model::openai_embed::OpenAiEmbedding;
use std::sync::Arc;

// text-embedding-3-small (1536 dims) or text-embedding-3-large (3072 dims)
let embed = Arc::new(OpenAiEmbedding::new("text-embedding-3-small"));

// Optional: override API key, base URL, or dimensions
let embed = Arc::new(
    OpenAiEmbedding::new("text-embedding-3-small")
        .with_api_key("sk-...")
        .with_base_url("https://api.openai.com/v1")
        .with_dimensions(1536)
);
```

Requires `OPENAI_API_KEY` in the environment unless set via `with_api_key`. Feature: `openai`.

### Ollama Embeddings

```rust
use daimon::model::ollama_embed::OllamaEmbedding;
use std::sync::Arc;

// nomic-embed-text, mxbai-embed-large, etc.
let embed = Arc::new(OllamaEmbedding::new("nomic-embed-text"));

// Optional: override host or dimensions
let embed = Arc::new(
    OllamaEmbedding::new("nomic-embed-text")
        .with_base_url("http://localhost:11434")
        .with_dimensions(768)
);
```

Uses `OLLAMA_HOST` (default `http://localhost:11434`) if not overridden. Feature: `ollama`.

### Amazon Bedrock Embeddings

```rust
use daimon::model::bedrock::BedrockEmbedding;
use std::sync::Arc;

// Titan Embeddings only (e.g. amazon.titan-embed-text-v2:0) — the request/
// response wire format is Titan-specific; Cohere Embed is not supported.
let embed = Arc::new(
    BedrockEmbedding::new("amazon.titan-embed-text-v2:0")
        .with_region("us-east-1")
        .with_dimensions(1024)
        .with_normalize(true)
);
```

Uses AWS SDK default credential chain. Feature: `bedrock`.

### Google Gemini Embeddings

```rust
use daimon::model::gemini::GeminiEmbedding;
use std::sync::Arc;

let embed = Arc::new(
    GeminiEmbedding::new("text-embedding-004")
        .with_api_key("...")  // or GOOGLE_API_KEY env
        .with_dimensions(768)
);
```

Feature: `gemini`.

### Azure OpenAI Embeddings

```rust
use daimon::model::azure::AzureOpenAiEmbedding;
use std::sync::Arc;

let embed = Arc::new(
    AzureOpenAiEmbedding::new(
        "https://my-resource.openai.azure.com",
        "text-embedding-3-small"
    )
    .with_api_key("...")  // or AZURE_OPENAI_API_KEY env
    .with_api_version("2024-10-21")
    .with_dimensions(1536)
);
```

Feature: `azure`.

---

## Vector Store Implementations

### In-Memory (Built-in)

`InMemoryVectorStoreBackend` uses brute-force cosine similarity. Ideal for development and testing.

```rust
use daimon::retriever::InMemoryVectorStoreBackend;

let store = InMemoryVectorStoreBackend::new();
```

No feature flag required. Data is lost when the process exits.

### Qdrant (feature = "qdrant")

`QdrantRetriever` implements `Retriever` directly (not `VectorStore`). It embeds queries and searches a Qdrant collection. You must ingest documents into Qdrant separately (e.g. via Qdrant SDK or another pipeline).

```rust
use daimon::retriever::qdrant::QdrantRetriever;  // also re-exported via daimon::prelude
use std::sync::Arc;

let retriever = QdrantRetriever::new(
    "http://localhost:6334",
    "my_collection",
    Arc::clone(&embedding_model),
)
.await?;

// Optional: custom payload field for document content
let retriever = retriever.with_content_field("text");
```

Requires a running Qdrant instance. Feature: `qdrant`.

### pgvector (feature = "pgvector")

PostgreSQL with the `pgvector` extension. Implements `VectorStore`. Use `PgVectorStoreBuilder` to configure and build.

```rust
use daimon_plugin_pgvector::{PgVectorStoreBuilder, DistanceMetric};
// or: use daimon::prelude::*;  (includes PgVectorStoreBuilder when pgvector enabled)

let store = PgVectorStoreBuilder::new("postgresql://user:pass@localhost/db", 1536)
    .table("my_docs")
    .distance_metric(DistanceMetric::Cosine)
    .hnsw_m(16)
    .hnsw_ef_construction(64)
    .pool_size(16)
    .auto_migrate(true)
    .build()
    .await?;
```

**Builder options:**

| Method | Default | Description |
|--------|---------|-------------|
| `table(name)` | `"daimon_vectors"` | Table name |
| `distance_metric(metric)` | `Cosine` | `Cosine`, `L2`, or `InnerProduct` |
| `hnsw_m(m)` | 16 | HNSW max connections per layer |
| `hnsw_ef_construction(ef)` | 64 | HNSW build-time search width |
| `pool_size(n)` | 16 | Connection pool size |
| `auto_migrate(enabled)` | `true` | Create extension and table on first use |

Disable `auto_migrate` and use `daimon_plugin_pgvector::migrations` for manual schema setup.

### OpenSearch (feature = "opensearch")

OpenSearch k-NN plugin. Implements `VectorStore`. Use `OpenSearchVectorStoreBuilder` to configure and build.

```rust
use daimon_plugin_opensearch::{OpenSearchVectorStoreBuilder, SpaceType, Engine};
// Note: the daimon prelude re-exports these types under renamed aliases to
// avoid collisions — `OpenSearchVectorStoreBuilder`, `OpenSearchSpaceType`,
// `OpenSearchEngine`. With `use daimon::prelude::*;` write
// `OpenSearchSpaceType::CosineSimilarity` / `OpenSearchEngine::Lucene` below.

let store = OpenSearchVectorStoreBuilder::new("http://localhost:9200", 1536)
    .index("my_docs")
    .space_type(SpaceType::CosineSimilarity)
    .engine(Engine::Lucene)
    .hnsw_m(16)
    .hnsw_ef_construction(256)
    .auto_create_index(true)
    .build()
    .await?;
```

**Builder options:**

| Method | Default | Description |
|--------|---------|-------------|
| `index(name)` | `"daimon_vectors"` | Index name |
| `space_type(t)` | `CosineSimilarity` | `CosineSimilarity`, `L2`, or `InnerProduct` |
| `engine(e)` | `Lucene` | `Lucene`, `Nmslib`, or `Faiss` |
| `hnsw_m(m)` | engine default | HNSW max connections per layer |
| `hnsw_ef_construction(ef)` | engine default | HNSW build-time search width |
| `auto_create_index(enabled)` | `true` | Create index on first use |

**AWS OpenSearch Service:** Use `build_with_client()` with a pre-configured OpenSearch client (e.g. with SigV4 auth):

```toml
# Cargo.toml
daimon-plugin-opensearch = { version = "0.22", features = ["aws-auth"] }
```

```rust
use opensearch::OpenSearch;
use opensearch::http::transport::Transport;

// Transport::single_node returns Result<Transport> directly (no .build()).
let transport = Transport::single_node("https://my-domain.us-east-1.es.amazonaws.com")?;
let client = OpenSearch::new(transport);  // Configure AWS auth per opensearch-rs docs

let store = OpenSearchVectorStoreBuilder::new("https://my-domain.us-east-1.es.amazonaws.com", 1536)
    .index("my_docs")
    .build_with_client(client)
    .await?;
```

---

## Building a RAG Pipeline

Full example: create embedding model, vector store, compose into `SimpleKnowledgeBase`, ingest documents, and search.

```rust
use daimon::prelude::*;
use serde_json::json;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Create embedding model
    let embed = Arc::new(
        daimon::model::openai_embed::OpenAiEmbedding::new("text-embedding-3-small")
    );

    // 2. Create vector store

    // Option A: In-memory (dev/testing)
    let store = InMemoryVectorStoreBackend::new();

    // Option B: pgvector (production)
    // let store = PgVectorStoreBuilder::new("postgresql://localhost/db", 1536)
    //     .table("docs")
    //     .build()
    //     .await?;

    // 3. Compose into SimpleKnowledgeBase
    let kb = SimpleKnowledgeBase::new(embed, store);

    // 4. Ingest documents
    let docs = vec![
        Document::new("Rust is a systems programming language focused on safety and performance.")
            .with_metadata("source", json!("rust-lang.org")),
        Document::new("Daimon is a Rust-native AI agent framework.")
            .with_metadata("source", json!("github.com/Lexmata/daimon")),
        Document::new("Embeddings are dense vector representations of text.")
            .with_metadata("topic", json!("ml")),
    ];

    let ids = kb.ingest(docs).await?;
    println!("Ingested {} documents", ids.len());

    // 5. Search documents
    let results = kb.search("What is Daimon?", 3).await?;
    for (i, doc) in results.iter().enumerate() {
        println!("--- Result {} (score: {:?}) ---", i + 1, doc.score);
        println!("{}", doc.content);
    }

    // 6. Use as agent tool via RetrieverTool
    let tool = RetrieverTool::new(
        kb,
        "search_docs",
        "Search the knowledge base for relevant documents. Use when you need to look up information.",
    )
    .with_default_top_k(5);

    let agent = Agent::builder()
        .model(/* ... */)
        .tool(tool)
        .build()?;

    let response = agent.prompt("What can you tell me about Daimon?").await?;
    println!("{}", response.text());

    Ok(())
}
```

---

## RetrieverTool

`RetrieverTool` wraps any `Retriever` as a `Tool`. The agent can search the knowledge base on demand.

```rust
use daimon::retriever::RetrieverTool;

let tool = RetrieverTool::new(
    retriever,
    "search_knowledge_base",
    "Search the knowledge base for relevant information. Use when you need to look up facts.",
);

// Optional: set default top_k when the agent omits it
let tool = tool.with_default_top_k(8);
```

**Parameters (JSON Schema):**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | — | The search query |
| `top_k` | integer | No | 5 (or `with_default_top_k`) | Maximum number of results |

The tool returns formatted text: each document with its score, metadata, and content.

---

## Document Type

`Document` represents a retrieved document fragment with optional metadata and relevance score.

```rust
use daimon::retriever::Document;
use serde_json::json;

// Minimal: content only
let doc = Document::new("Hello world");

// With metadata
let doc = Document::new("Rust is fast.")
    .with_metadata("source", json!("rust-lang.org"))
    .with_metadata("page", json!(42));

// With custom ID (used by SimpleKnowledgeBase.ingest if present)
let doc = Document::new("Content")
    .with_metadata("id", json!("custom-doc-id"));

// Score is set by retrieval backends (not typically set by user)
let doc = doc.with_score(0.92);
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `content` | `String` | The text content |
| `metadata` | `HashMap<String, serde_json::Value>` | Arbitrary key-value metadata (source, page, etc.) |
| `score` | `Option<f64>` | Relevance score from retrieval. `None` if backend does not provide scores |

`ScoredDocument` is the public query-result type: `{ id: String, document: Document, score: f64 }`. `id` is the same stable id passed to `VectorStore::upsert` when the document was stored — implementations must populate it with the real id, not a synthetic/rank-derived value, so callers can round-trip a search result into `delete`. Vector stores return `Vec<ScoredDocument>`; `SimpleKnowledgeBase` converts to `Document` with `with_score` applied.

---

## Choosing a Vector Store

| Use Case | Recommendation | Notes |
|----------|----------------|-------|
| **Development / testing** | `InMemoryVectorStoreBackend` | No setup, data lost on restart |
| **PostgreSQL already in stack** | pgvector | Reuse existing DB, connection pooling |
| **Search infrastructure** | OpenSearch | k-NN + full-text, good for hybrid search |
| **Dedicated vector DB** | Qdrant | High-performance, built for vectors |
| **Custom backend** | Implement `VectorStore` | See `daimon-core::VectorStore` trait |

**Implementing a custom vector store:**

```rust
use daimon_core::{Document, ScoredDocument, VectorStore};

struct MyVectorStore { /* ... */ }

impl VectorStore for MyVectorStore {
    async fn upsert(&self, id: &str, embedding: Vec<f32>, document: Document) -> Result<()> {
        // Store in your backend
        Ok(())
    }

    async fn query(&self, embedding: Vec<f32>, top_k: usize) -> Result<Vec<ScoredDocument>> {
        // Search and return scored documents, each carrying the stable id
        // it was upserted with (required for delete to round-trip).
        Ok(vec![])
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        Ok(true)
    }

    async fn count(&self) -> Result<usize> {
        Ok(0)
    }
}
```

Then compose with `SimpleKnowledgeBase::new(embedding_model, my_store)`.

---

## Performance Tips

### Batch ingest

`SimpleKnowledgeBase::ingest` accepts `Vec<Document>` and embeds them in a single batch. Prefer batching over multiple single-document calls:

```rust
// Good: one batch
let ids = kb.ingest(docs).await?;

// Avoid: many small batches
for doc in docs {
    kb.ingest(vec![doc]).await?;
}
```

### HNSW parameters

- **`m`** (max connections per layer): Higher = better recall, slower writes. Typical: 16–32.
- **`ef_construction`**: Build-time search width. Higher = better index quality, slower build. Typical: 64–256.

### Connection pooling

- **pgvector:** Uses `deadpool`; tune `pool_size` to match concurrency.
- **OpenSearch:** Uses `opensearch` crate's internal transport; configure via `build_with_client` if needed.

### Embedding dimensions

- Smaller dimensions (e.g. 256) = faster search, lower quality.
- Larger dimensions (e.g. 3072) = better quality, slower and more storage.
- Match model dimensions to your vector store configuration.

### Qdrant vs VectorStore + KnowledgeBase

`QdrantRetriever` implements `Retriever` only. You must populate the Qdrant collection separately (e.g. with the same embedding model). For a unified ingest pipeline, use `SimpleKnowledgeBase` with a `VectorStore` backend (pgvector, OpenSearch, or in-memory).

---

## Document Chunking and Ingestion

Daimon does not include built-in chunking. For long documents, split text into chunks before passing to `SimpleKnowledgeBase::ingest`:

```rust
fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < chars.len() {
        let end = (start + chunk_size).min(chars.len());
        chunks.push(chars[start..end].iter().collect());
        start = end.saturating_sub(overlap);
    }

    chunks
}

// Usage
let long_doc = "...";  // e.g. from a file or API
let chunks = chunk_text(long_doc, 512, 50);
let docs: Vec<Document> = chunks
    .into_iter()
    .enumerate()
    .map(|(i, content)| {
        Document::new(content)
            .with_metadata("source", json!("manual.pdf"))
            .with_metadata("chunk", json!(i))
    })
    .collect();

let ids = kb.ingest(docs).await?;
```

**Chunking tips:**

- **Chunk size:** 256–1024 tokens (or ~100–400 words) is typical. Match to your embedding model's context window.
- **Overlap:** 10–20% overlap reduces boundary effects and improves context continuity.
- **Metadata:** Store `source`, `page`, `chunk` index so the agent can cite sources.

---

## Feature Flags

| Feature | Enables |
|---------|---------|
| `openai` | `OpenAiEmbedding` |
| `ollama` | `OllamaEmbedding` |
| `bedrock` | `BedrockEmbedding` |
| `gemini` | `GeminiEmbedding` |
| `azure` | `AzureOpenAiEmbedding` |
| `qdrant` | `QdrantRetriever` |
| `pgvector` | `PgVectorStore`, `PgVectorStoreBuilder` |
| `opensearch` | `OpenSearchVectorStore`, `OpenSearchVectorStoreBuilder` |

Use `full` to enable all providers and vector stores.
