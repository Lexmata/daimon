# Daimon Provider Guide

This document covers every LLM and embedding provider in the Daimon framework, with configuration details, feature flags, and practical usage patterns.

---

## Provider Overview

| Provider | Feature Flag | Chat | Streaming | Embedding | Caching | Cost Model | Cloud Broker |
|----------|--------------|------|-----------|-----------|---------|------------|--------------|
| OpenAI | `openai` (default) | ✓ | ✓ | ✓ | ✓ (system) | `OpenAiCostModel` | — |
| Anthropic | `anthropic` (default) | ✓ | ✓ | — | ✓ (cache_control) | `AnthropicCostModel` | — |
| Ollama | `ollama` (default) | ✓ | ✓ | ✓ | — | — | — |
| Google Gemini | `gemini` | ✓ | ✓ | ✓ | ✓ (cached_content) | — | `PubSubBroker` (pubsub) |
| Azure OpenAI | `azure` | ✓ | ✓ | ✓ | ✓ (system) | — | `ServiceBusBroker` (servicebus) |
| AWS Bedrock | `bedrock` | ✓ | ✓ | ✓ | ✓ (Claude) | — | `SqsBroker` (sqs) |
| OpenRouter | `openrouter` | ✓ | ✓ | — | — | — | — |

---

## OpenAI (default)

**Feature:** `openai` (included in default features)

### Chat Model

```rust
use daimon::model::openai::OpenAi;
use daimon::prelude::*;
use std::time::Duration;

// From environment (OPENAI_API_KEY)
let model = OpenAi::new("gpt-4o");

// With explicit API key
let model = OpenAi::with_api_key("gpt-4o", std::env::var("OPENAI_API_KEY")?)
    .with_base_url("https://api.openai.com/v1")  // or proxy/local endpoint
    .with_timeout(Duration::from_secs(60))
    .with_max_retries(5)
    .with_response_format("json_object")         // JSON mode
    .with_parallel_tool_calls(true);
```

**Configuration methods:**

| Method | Description |
|--------|-------------|
| `.with_api_key(model_id, key)` | Explicit API key (otherwise reads `OPENAI_API_KEY`) |
| `.with_base_url(url)` | Custom base URL (proxies, local endpoints) |
| `.with_timeout(duration)` | Timeout for non-streaming requests (default: 120s; streams use a connect timeout only) |
| `.with_max_retries(n)` | Retries for 429 and 5xx (default: 3) |
| `.with_response_format(format)` | `"json_object"` or `"text"` |
| `.with_parallel_tool_calls(enabled)` | Allow multiple tool calls per turn |

**Models:** `gpt-5`, `gpt-5-mini`, `gpt-4.1`, `gpt-4o`, `gpt-4o-mini`, `o3`, `o4-mini`

**Capabilities:** Tool calls, streaming, JSON mode, parallel tool calls, response format. Caching: system message caching via API (reported in `usage.cached_tokens`).

### Embedding

```rust
use daimon::model::openai_embed::OpenAiEmbedding;

let embedding = OpenAiEmbedding::new("text-embedding-3-small")
    .with_api_key(api_key)
    .with_base_url("https://api.openai.com/v1")
    .with_dimensions(1536);  // 1536 for small, 3072 for large
```

**Models:** `text-embedding-3-small` (1536 dims), `text-embedding-3-large` (3072 dims)

### Cost Model

```rust
use daimon::cost::OpenAiCostModel;

let agent = Agent::builder()
    .model(OpenAi::new("gpt-4o"))
    .cost_model(OpenAiCostModel)
    .max_budget(0.50)
    .build()?;
```

---

## Anthropic (default)

**Feature:** `anthropic` (included in default features)

### Chat Model

```rust
use daimon::model::anthropic::Anthropic;
use std::time::Duration;

let model = Anthropic::new("claude-sonnet-5");

let model = Anthropic::with_api_key("claude-sonnet-5", std::env::var("ANTHROPIC_API_KEY")?)
    .with_base_url("https://api.anthropic.com")
    .with_timeout(Duration::from_secs(60))
    .with_max_retries(5)
    .with_prompt_caching();  // cache_control breakpoints for system + tools
```

**Configuration methods:**

| Method | Description |
|--------|-------------|
| `.with_api_key(model_id, key)` | Explicit API key (otherwise `ANTHROPIC_API_KEY`) |
| `.with_base_url(url)` | Custom base URL |
| `.with_timeout(duration)` | Timeout for non-streaming requests (default: 120s; streams use a connect timeout only) |
| `.with_max_retries(n)` | Retries for 429, 529, 5xx (default: 3) |
| `.with_prompt_caching()` | Enables `cache_control` breakpoints (system, tools) |

**Models:** `claude-sonnet-5`, `claude-opus-4-8`, `claude-haiku-4-5`, etc.

**Capabilities:** Tool calls, streaming, overloaded retry (429/529/5xx). Caching: native `cache_control` breakpoints for system and tool definitions; `usage.cached_tokens` reports cache reads.

### Cost Model

```rust
use daimon::cost::AnthropicCostModel;

let agent = Agent::builder()
    .model(Anthropic::new("claude-sonnet-5"))
    .cost_model(AnthropicCostModel)
    .build()?;
```

---

## Ollama (default)

**Feature:** `ollama` (included in default features)

### Chat Model

```rust
use daimon::model::ollama::Ollama;
use std::time::Duration;

let model = Ollama::new("llama3.2");

let model = Ollama::new("llama3.2")
    .with_base_url("http://localhost:11434")  // default
    .with_timeout(Duration::from_secs(300))
    .with_keep_alive("5m");  // keep model loaded; "0" to unload immediately
```

**Configuration methods:**

| Method | Description |
|--------|-------------|
| `.with_base_url(url)` | Ollama server URL (default: `http://localhost:11434`) |
| `.with_timeout(duration)` | Request timeout (default: 300s) |
| `.with_keep_alive(duration_str)` | e.g. `"5m"`, `"1h"`, `"0"` to unload |

**Models:** Any model served by Ollama (e.g. `llama3.2`, `llama3.1`, `mistral`, `codellama`). Tool calls are model-dependent.

**Capabilities:** Tool calls (if model supports), streaming. No caching. No cost model (local/free).

### Embedding

```rust
use daimon::model::ollama_embed::OllamaEmbedding;

let embedding = OllamaEmbedding::new("nomic-embed-text")
    .with_base_url("http://localhost:11434")
    .with_dimensions(768);  // model-dependent; nomic-embed-text is 768
```

Uses `OLLAMA_HOST` env var if set; otherwise `http://localhost:11434`.

---

## Google Gemini (feature = "gemini")

**Feature:** `gemini` — enables `daimon-provider-gemini`

### Chat Model

```rust
use daimon::model::gemini::Gemini;
use std::time::Duration;

// From GOOGLE_API_KEY
let model = Gemini::new("gemini-2.0-flash");

let model = Gemini::with_api_key("gemini-2.0-flash", std::env::var("GOOGLE_API_KEY")?)
    .with_base_url("https://generativelanguage.googleapis.com/v1beta")  // or Vertex AI URL
    .with_timeout(Duration::from_secs(60))
    .with_max_retries(5)
    .with_bearer_token()  // for Vertex AI (OAuth2)
    .with_cached_content("cachedContents/<id>");  // pre-cached system/tools
```

**Configuration methods:**

| Method | Description |
|--------|-------------|
| `.with_api_key(model_id, key)` | Explicit key (otherwise `GOOGLE_API_KEY`) |
| `.with_base_url(url)` | Custom URL (e.g. Vertex AI) |
| `.with_timeout(duration)` | Timeout for non-streaming requests (default: 120s; streams use a connect timeout only) |
| `.with_max_retries(n)` | Retries for 429, 5xx |
| `.with_bearer_token()` | Use `Authorization: Bearer` (Vertex AI; API keys are sent via the `x-goog-api-key` header) |
| `.with_cached_content(name)` | Reference pre-created cached content |

**Models:** `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.0-flash`

**Capabilities:** Tool calls, streaming. Caching: system instruction caching via `with_cached_content` or Gemini Caching API.

### Embedding

```rust
use daimon::model::gemini::GeminiEmbedding;

let embedding = GeminiEmbedding::new("text-embedding-004")
    .with_api_key(api_key)
    .with_base_url("https://generativelanguage.googleapis.com/v1beta")
    .with_dimensions(768)
    .with_bearer_token();
```

### Cloud Broker

With feature `pubsub`: `PubSubBroker` for Google Cloud Pub/Sub task distribution.

---

## Azure OpenAI (feature = "azure")

**Feature:** `azure` — enables `daimon-provider-azure`

### Chat Model

```rust
use daimon::model::azure::AzureOpenAi;
use std::time::Duration;

// From AZURE_OPENAI_API_KEY
let model = AzureOpenAi::new(
    "https://my-resource.openai.azure.com",
    "gpt-4o",  // deployment name
);

let model = AzureOpenAi::with_api_key(
    "https://my-resource.openai.azure.com",
    "gpt-4o",
    std::env::var("AZURE_OPENAI_API_KEY")?,
)
    .with_api_version("2024-10-21")
    .with_timeout(Duration::from_secs(60))
    .with_max_retries(5)
    .with_bearer_token();  // for Microsoft Entra ID
```

**Configuration methods:**

| Method | Description |
|--------|-------------|
| `.with_api_key(resource_url, deployment, key)` | Explicit key (otherwise `AZURE_OPENAI_API_KEY`) |
| `.with_api_version(version)` | API version (default: `2024-10-21`) |
| `.with_timeout(duration)` | Timeout for non-streaming requests (default: 120s; streams use a connect timeout only) |
| `.with_max_retries(n)` | Retries |
| `.with_bearer_token()` | Microsoft Entra ID (Azure AD) auth |

**Endpoint format:** `{resource_url}/openai/deployments/{deployment}/chat/completions?api-version=...`

**Capabilities:** Tool calls and streaming, like OpenAI — but note there is no `.with_response_format()` / JSON-mode knob and no `.with_parallel_tool_calls()` on this provider (unlike OpenAI). Caching: system message caching (same as OpenAI).

### Embedding

```rust
use daimon::model::azure::AzureOpenAiEmbedding;

let embedding = AzureOpenAiEmbedding::new(
    "https://my-resource.openai.azure.com",
    "text-embedding-3-small",
)
    .with_api_key(key)
    .with_api_version("2024-10-21")
    .with_dimensions(1536)
    .with_bearer_token();
```

### Cloud Broker

With feature `servicebus`: `ServiceBusBroker` for Azure Service Bus task distribution.

---

## AWS Bedrock (feature = "bedrock")

**Feature:** `bedrock` — enables `daimon-provider-bedrock`

### Chat Model

```rust
use daimon::model::bedrock::Bedrock;
use std::time::Duration;

let model = Bedrock::new("us.anthropic.claude-sonnet-5")
    .with_region("us-east-1")
    .with_max_retries(5)
    .with_guardrail("guardrail-id", "DRAFT")
    .with_prompt_caching();  // CachePoint for system + tools (Claude)
```

**Configuration methods:**

| Method | Description |
|--------|-------------|
| `.with_client(client)` | Use a pre-built Bedrock client |
| `.with_region(region)` | AWS region (otherwise from env/config) |
| `.with_max_retries(n)` | Retries for throttling/5xx |
| `.with_guardrail(id, version)` | Content filtering guardrail |
| `.with_prompt_caching()` | CachePoint blocks for system and tools (Claude models) |

**Authentication:** Uses AWS SDK default credential chain (env vars, `~/.aws/credentials`, IAM roles).

**Models:** Anthropic Claude (`us.anthropic.claude-*`), Amazon Titan, Meta Llama, Cohere, AI21 — use full Bedrock model IDs.

**Capabilities:** Tool calls, streaming, guardrails. Caching: native system/tool caching for Claude models via `with_prompt_caching()`.

### Embedding

```rust
use daimon::model::bedrock::BedrockEmbedding;

let embedding = BedrockEmbedding::new("amazon.titan-embed-text-v2:0")
    .with_region("us-east-1")
    .with_dimensions(1024)
    .with_normalize(true);
```

### Cloud Broker

With feature `sqs`: `SqsBroker` for AWS SQS task distribution.

---

## OpenRouter (feature = "openrouter")

**Feature:** `openrouter`

[OpenRouter](https://openrouter.ai) is an OpenAI-compatible gateway that routes to hundreds of models — OpenAI, Anthropic, Google, Meta, and more — behind a single API key. Model ids use OpenRouter's `vendor/model` form (e.g. `openai/gpt-4o`, `anthropic/claude-sonnet-4`).

### Chat Model

```rust
use daimon::model::openrouter::OpenRouter;

// Reads OPENROUTER_API_KEY from the environment
let model = OpenRouter::new("openai/gpt-4o");

// Full configuration
let model = OpenRouter::with_api_key("anthropic/claude-sonnet-4", "sk-or-...")
    .with_timeout(Duration::from_secs(60))
    .with_max_retries(3)
    .with_response_format("json_object")   // if the routed model supports it
    .with_parallel_tool_calls(true)
    .with_site_url("https://your-app.com") // HTTP-Referer header (rankings attribution)
    .with_app_name("your-app");            // X-Title header (rankings display name)
```

**Capabilities:** Tool calls and streaming via the OpenAI-compatible Chat Completions API. No `EmbeddingModel`. Note that capabilities are only as strong as the routed model — e.g. `response_format` and parallel tool calls depend on the upstream provider. `max_tokens` is sent (not the OpenAI-specific `max_completion_tokens`) because OpenRouter normalizes it across upstreams.

---

## Switching Providers at Runtime

### SharedModel and Arc&lt;dyn ErasedModel&gt;

All providers implement the `Model` trait. Use `Arc<dyn ErasedModel>` (aliased as `SharedModel`) for dynamic dispatch when the provider is chosen at runtime:

```rust
use daimon::model::SharedModel;
use daimon::model::openai::OpenAi;
use daimon::model::anthropic::Anthropic;
use std::sync::Arc;

fn select_model(use_openai: bool) -> SharedModel {
    if use_openai {
        Arc::new(OpenAi::new("gpt-4o"))
    } else {
        Arc::new(Anthropic::new("claude-sonnet-5"))
    }
}

let agent = Agent::builder()
    .shared_model(select_model(std::env::var("USE_OPENAI").is_ok()))
    .build()?;
```

### HotSwapAgent for Runtime Model Swapping

`HotSwapAgent` wraps an agent behind a `RwLock`, allowing you to swap the model (or tools, system prompt, memory) at runtime without rebuilding:

```rust
use daimon::prelude::*;
use daimon::agent::hot_swap::HotSwapAgent;
use daimon::model::openai::OpenAi;
use daimon::model::anthropic::Anthropic;
use std::sync::Arc;

let agent = Agent::builder()
    .model(OpenAi::new("gpt-4o"))
    .system_prompt("You are helpful.")
    .build()?;

let hot = HotSwapAgent::new(agent);

// Use normally
let response = hot.prompt("Hello").await?;

// Swap model at runtime
hot.swap_model(Anthropic::new("claude-sonnet-5")).await;

// Next prompt uses the new model
let response = hot.prompt("Hello again").await?;

// Or swap with a pre-boxed SharedModel
hot.swap_shared_model(Arc::new(OpenAi::new("gpt-4o-mini"))).await;
```

### A/B Testing with fork_builder

Use `Agent::fork_builder()` to create mutated copies for A/B testing or specialized variants:

```rust
let base = Agent::builder()
    .model(OpenAi::new("gpt-4o"))
    .system_prompt("You are a helpful assistant.")
    .tool(SearchTool)
    .build()?;

// Variant A: different model
let variant_a = base.fork_builder()
    .model(Anthropic::new("claude-sonnet-5"))
    .build()?;

// Variant B: different system prompt
let variant_b = base.fork_builder()
    .system_prompt("You are a code reviewer. Be strict.")
    .remove_tool("search")
    .tool(ReviewTool)
    .build()?;

// Run both and compare
let resp_a = variant_a.prompt("Review this code").await?;
let resp_b = variant_b.prompt("Review this code").await?;
```

---

## Provider-Specific Caching

| Provider | Caching Mechanism | How to Enable |
|----------|-------------------|---------------|
| **OpenAI** | System message caching | Automatic via API; `usage.cached_tokens` reports reads |
| **Anthropic** | `cache_control` breakpoints | `.with_prompt_caching()` — system + tools |
| **Bedrock** | CachePoint blocks (Claude) | `.with_prompt_caching()` — system + tools |
| **Gemini** | System instruction / Caching API | `.with_cached_content("cachedContents/<id>")` or create via API |
| **Azure OpenAI** | Same as OpenAI | Automatic; `usage.cached_tokens` |
| **Ollama** | — | No native caching |

---

## Environment Variables

All providers support reading API keys from the environment. Set these before running:

| Provider | Variable |
|----------|----------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Ollama | `OLLAMA_HOST` (embedding provider only; default `http://localhost:11434` — chat uses `.with_base_url()`) |
| Gemini | `GOOGLE_API_KEY` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Bedrock | AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`) or IAM role |

Example:

```rust
// Typical pattern: read from env, fallback for tests
let api_key = std::env::var("OPENAI_API_KEY")
    .unwrap_or_else(|_| "sk-test-placeholder".to_string());

let model = OpenAi::with_api_key("gpt-4o", api_key);
```

---

## Cargo.toml Feature Selection

```toml
[dependencies]
# Default: openai, anthropic, ollama, macros
daimon = "0.23"

# Minimal: only OpenAI
daimon = { version = "0.23", default-features = false, features = ["openai"] }

# Add Gemini and Azure
daimon = { version = "0.23", features = ["gemini", "azure"] }

# Add OpenRouter
daimon = { version = "0.23", features = ["openrouter"] }

# Full: all providers + MCP, SQLite, Redis, etc.
daimon = { version = "0.23", features = ["full"] }
```

| Feature | Enables |
|---------|---------|
| `openai` | OpenAI chat + embedding |
| `anthropic` | Anthropic Claude |
| `ollama` | Ollama chat + embedding |
| `gemini` | Gemini chat + embedding |
| `azure` | Azure OpenAI chat + embedding |
| `bedrock` | Bedrock chat + embedding |
| `sqs` | Bedrock + SqsBroker |
| `pubsub` | Gemini + PubSubBroker |
| `servicebus` | Azure + ServiceBusBroker |
