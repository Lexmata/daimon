# Daimon

A Rust-native AI agent framework for building LLM-powered agents with tool use, memory, and streaming.

Daimon implements the **ReAct** (Reason-Act-Observe) pattern: the agent calls a model, optionally invokes tools, observes results, and repeats until it produces a final response. It is designed to be easy to use while leveraging Rust's type system, async runtime, and performance characteristics.

## Features

- **ReAct agent loop** with configurable iteration limits
- **Multiple LLM providers** behind feature flags — OpenAI, Anthropic, AWS Bedrock
- **Tool system** with async execution, parallel tool calls, and a typed registry
- **Streaming** with full ReAct loop support (tool calls accumulate and re-invoke within a single stream)
- **Conversation memory** with pluggable backends (sliding window included), plus an optional tiered memory subsystem (core/archival/episodic) for longer-lived agents
- **Lifecycle hooks** for observability and control
- **Cancellation** via `tokio_util::CancellationToken`
- **Tracing** instrumentation on all agent and provider operations
- **Retry logic** with exponential backoff for transient provider errors

## Quick Start

Add Daimon to your `Cargo.toml`:

```toml
[dependencies]
daimon = "0.22"
tokio = { version = "1", features = ["full"] }
```

Create an agent and prompt it:

```rust
use daimon::prelude::*;

#[tokio::main]
async fn main() -> daimon::Result<()> {
    let agent = Agent::builder()
        .model(daimon::model::openai::OpenAi::new("gpt-4o"))
        .system_prompt("You are a helpful assistant.")
        .build()?;

    let response = agent.prompt("What is Rust?").await?;
    println!("{}", response.text());
    Ok(())
}
```

## Tools

Define tools by implementing the `Tool` trait:

```rust
use daimon::prelude::*;

struct Calculator;

impl Tool for Calculator {
    fn name(&self) -> &str { "calculator" }
    fn description(&self) -> &str { "Evaluate math expressions" }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "operation": { "type": "string", "enum": ["add", "subtract", "multiply", "divide"] },
                "a": { "type": "number" },
                "b": { "type": "number" }
            },
            "required": ["operation", "a", "b"]
        })
    }

    async fn execute(&self, input: &Value) -> daimon::Result<ToolOutput> {
        let op = input["operation"].as_str().unwrap_or("add");
        let a = input["a"].as_f64().unwrap_or(0.0);
        let b = input["b"].as_f64().unwrap_or(0.0);

        let result = match op {
            "add" => a + b,
            "subtract" => a - b,
            "multiply" => a * b,
            "divide" if b != 0.0 => a / b,
            "divide" => return Ok(ToolOutput::error("Division by zero")),
            _ => return Ok(ToolOutput::error(format!("Unknown operation: {op}"))),
        };

        Ok(ToolOutput::text(format!("{result}")))
    }
}

#[tokio::main]
async fn main() -> daimon::Result<()> {
    let agent = Agent::builder()
        .model(daimon::model::openai::OpenAi::new("gpt-4o"))
        .system_prompt("Use the calculator tool to solve math problems.")
        .tool(Calculator)
        .build()?;

    let response = agent.prompt("What is 42 * 17 + 3?").await?;
    println!("{}", response.text());
    println!("Completed in {} iteration(s)", response.iterations);
    Ok(())
}
```

## Streaming

Stream responses token-by-token with the full ReAct loop. Streaming is not a
degraded path: it runs the same loop as `prompt()` — conversation memory is
loaded and persisted, lifecycle hooks fire, guardrails are enforced, and tool
calls accumulate and re-invoke the model within a single stream until a final
response is produced.

```rust
use daimon::prelude::*;

#[tokio::main]
async fn main() -> daimon::Result<()> {
    let agent = Agent::builder()
        .model(daimon::model::openai::OpenAi::new("gpt-4o"))
        .build()?;

    let mut stream = agent.prompt_stream("Explain quantum computing.").await?;

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::TextDelta(text) => print!("{text}"),
            StreamEvent::ToolResult { content, .. } => eprintln!("\n[tool result: {content}]"),
            StreamEvent::Done => { println!(); break; }
            _ => {}
        }
    }

    Ok(())
}
```

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `openai` | Yes | OpenAI Chat Completions API (via `daimon-provider-openai`) |
| `anthropic` | Yes | Anthropic Messages API (via `daimon-provider-anthropic`) |
| `macros` | Yes | `#[tool_fn]` proc macro for defining tools |
| `bedrock` | No | AWS Bedrock Converse API |
| `gemini` | No | Google Gemini / Vertex AI provider |
| `azure` | No | Azure OpenAI Service provider |
| `ollama` | Yes | Ollama local model provider (via `daimon-provider-local`) |
| `llamacpp` | No | llama.cpp (llama-server) provider |
| `llamars` | No | llama-rs provider |
| `local` | No | All local providers at once (Ollama, llama.cpp, llama-rs, generic OpenAI-compatible) |
| `a2a` | No | Agent-to-Agent (A2A) protocol client (`A2aClient`) |
| `mcp` | No | Model Context Protocol client & server |
| `sqlite` | No | SQLite memory backend |
| `redis` | No | Redis memory backend + task broker + checkpoint |
| `nats` | No | NATS JetStream task broker + checkpoint |
| `amqp` | No | RabbitMQ (AMQP) task broker |
| `qdrant` | No | Qdrant vector store retriever |
| `pgvector` | No | pgvector-backed vector store (via `daimon-plugin-pgvector`) |
| `opensearch` | No | OpenSearch k-NN vector store (via `daimon-plugin-opensearch`) |
| `otel` | No | OpenTelemetry OTLP span export |
| `http-server` | No | HTTP agent server (`AgentServer`) |
| `grpc` | No | gRPC transport for distributed execution |
| `full` | No | All providers + macros + MCP + SQLite + Redis + NATS + AMQP + OTel + HTTP server + Qdrant + pgvector + OpenSearch + gRPC + eval + SQS + Pub/Sub + Service Bus |

The core framework compiles with no features enabled. Enable only the providers you need:

```toml
# Only Anthropic
daimon = { version = "0.22", default-features = false, features = ["anthropic"] }

# All providers
daimon = { version = "0.22", features = ["full"] }

# Core only (bring your own Model impl)
daimon = { version = "0.22", default-features = false }
```

## Provider Configuration

All providers support configurable timeout, retries, and provider-specific options:

```rust
use std::time::Duration;

// OpenAI with custom settings
let model = daimon::model::openai::OpenAi::new("gpt-4o")
    .with_timeout(Duration::from_secs(30))
    .with_max_retries(5)
    .with_response_format("json_object")
    .with_parallel_tool_calls(true);

// Anthropic with prompt caching
let model = daimon::model::anthropic::Anthropic::new("claude-sonnet-4-20250514")
    .with_timeout(Duration::from_secs(60))
    .with_prompt_caching();

// AWS Bedrock with guardrails
let model = daimon::model::bedrock::Bedrock::new("anthropic.claude-3-5-sonnet-20241022-v2:0")
    .with_guardrail("my-guardrail-id", "DRAFT");
```

## Agent Configuration

```rust
use daimon::prelude::*;

let agent = Agent::builder()
    .model(model)                              // required
    .system_prompt("You are helpful.")         // optional system prompt
    .tool(Calculator)                          // register tools
    .tool(WebSearch)
    .memory(SlidingWindowMemory::new(100))     // custom memory (default: 50 messages)
    .hooks(MyObserver)                         // lifecycle hooks
    .max_iterations(10)                        // default: 25
    .temperature(0.7)                          // sampling temperature
    .max_tokens(4096)                          // max output tokens
    .build()?;

// Standard prompt
let response = agent.prompt("Hello").await?;
println!("{}", response.text());
println!("Tokens used: {}", response.usage.total_tokens());

// With cancellation
let cancel = CancellationToken::new();
let response = agent.prompt_with_cancellation("Hello", &cancel).await?;

// With pre-built messages
let messages = vec![Message::user("Hello")];
let response = agent.prompt_with_messages(messages).await?;
```

`SlidingWindowMemory` above only covers short-term conversation history. For
longer-lived agents, `TieredMemory` composes it with three optional
sub-memories — `CoreMemory` (small, always-in-context blocks, e.g. persona
or user preferences), `ArchivalMemory` (explicit write/search over a large
fact store, lexical or vector-backed), and `EpisodicMemory` (a queryable
structured event log) — and still drops straight into `.memory()` since it
implements the same `Memory` trait. See `daimon-core`'s `core_memory` /
`archival_memory` / `episodic_memory` modules or the CHANGELOG's DAIM-23
entry for details.

## Architecture

```
┌──────────────────────────────────────────────────┐
│                    Agent                          │
│  ┌────────────┐  ┌──────────┐  ┌──────────────┐ │
│  │   Model     │  │  Tools   │  │   Memory     │ │
│  │  (trait)    │  │ Registry │  │   (trait)    │ │
│  └─────┬──────┘  └────┬─────┘  └──────┬───────┘ │
│        │              │               │          │
│  ┌─────┴──────────────┴───────────────┴───────┐  │
│  │            ReAct Loop                      │  │
│  │  1. Load memory → build request            │  │
│  │  2. Call model                             │  │
│  │  3. Tool calls? → execute (parallel) → 2   │  │
│  │  4. Final response → save to memory        │  │
│  └────────────────────────────────────────────┘  │
│        │                                         │
│  ┌─────┴──────┐  ┌──────────┐                   │
│  │   Hooks    │  │ Streaming │                   │
│  │ (lifecycle)│  │  Events   │                   │
│  └────────────┘  └──────────┘                   │
└──────────────────────────────────────────────────┘
```

## Environment Variables

Each provider reads its API key from standard environment variables:

| Provider | Variable | Notes |
|----------|----------|-------|
| OpenAI | `OPENAI_API_KEY` | Required for `openai` feature |
| Anthropic | `ANTHROPIC_API_KEY` | Required for `anthropic` feature |
| AWS Bedrock | Standard AWS credentials | `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` or IAM role |
| Google Gemini | `GOOGLE_APPLICATION_CREDENTIALS` | Service account JSON path |
| Azure OpenAI | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` | Required for `azure` feature |
| Ollama | `OLLAMA_HOST` | Defaults to `http://localhost:11434` |

## Testing

```bash
# Default features (openai + anthropic + ollama + macros)
cargo test

# All features
cargo test --features full

# Core only (no providers)
cargo test --no-default-features

# Coverage (requires cargo-llvm-cov; CI enforces 80 on full, 84 on core)
cargo llvm-cov --features full --fail-under-lines 80
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full testing and development setup.

## Minimum Supported Rust Version

Rust **1.95** (edition 2024).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Related Repos

- [cardozo-ai](../cardozo-ai) -- Legal AI framework (Rust + Candle, shares Rust tooling patterns)
- [lexmata-initial-case-evaluation](../lexmata-initial-case-evaluation) -- Go AI service that could use Daimon's agent patterns
- [lexmata-app-backend](../lexmata-app-backend) -- Backend that dispatches AI work to Bedrock

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and contribution guidelines. Note that AI-assisted contributions must include proper attribution.
