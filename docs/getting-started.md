# Getting Started with Daimon

Daimon is a Rust-native AI agent framework. This guide takes you from zero to a working agent in minutes, then progressively adds tools, streaming, memory, and multiple providers.

## Installation

Add Daimon to your `Cargo.toml`:

```toml
[dependencies]
daimon = "0.22"  # includes openai, anthropic, ollama, macros
```

To minimize dependencies, enable only what you need:

```toml
[dependencies]
daimon = { version = "0.22", default-features = false, features = ["openai", "macros"] }
```

Optional providers and plugins:

| Feature   | Description                    |
|-----------|--------------------------------|
| `gemini`  | Google Gemini / Vertex AI      |
| `azure`   | Azure OpenAI Service           |
| `bedrock` | AWS Bedrock                    |
| `pgvector`| pgvector-backed vector store   |
| `opensearch` | OpenSearch k-NN vector store |
| `mcp`     | Model Context Protocol         |
| `sqlite`  | SQLite memory backend          |
| `redis`   | Redis memory + task broker    |

Example with Gemini and pgvector:

```toml
daimon = { version = "0.22", features = ["openai", "gemini", "pgvector"] }
```

---

## Your First Agent

Minimal example: create an OpenAI agent, send a prompt, print the response.

```rust
use daimon::model::openai::OpenAi;
use daimon::prelude::*;

#[tokio::main]
async fn main() -> daimon::Result<()> {
    let agent = Agent::builder()
        .model(OpenAi::new("gpt-4o"))
        .system_prompt("You are a helpful assistant. Be concise.")
        .build()?;

    let response = agent.prompt("What is Rust?").await?;
    println!("{}", response.text());
    Ok(())
}
```

Set `OPENAI_API_KEY` in your environment.

---

## Adding Tools

Tools let the agent perform actions (API calls, calculations, lookups). Implement the `Tool` trait or use the `#[tool_fn]` macro.

### Manual implementation

```rust
use daimon::model::openai::OpenAi;
use daimon::prelude::*;

struct Calculator;

impl Tool for Calculator {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Evaluate math: add, subtract, multiply, divide. Args: operation, a, b."
    }

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
            "divide" => {
                if b == 0.0 {
                    return Ok(ToolOutput::error("Division by zero"));
                }
                a / b
            }
            _ => return Ok(ToolOutput::error(format!("Unknown operation: {op}"))),
        };

        Ok(ToolOutput::text(format!("{result}")))
    }
}

#[tokio::main]
async fn main() -> daimon::Result<()> {
    let agent = Agent::builder()
        .model(OpenAi::new("gpt-4o"))
        .system_prompt("You are a math tutor. Use the calculator tool to solve problems.")
        .tool(Calculator)
        .max_iterations(10)
        .build()?;

    let response = agent.prompt("What is 42 * 17 + 3?").await?;
    println!("{}", response.text());
    Ok(())
}
```

### Using `#[tool_fn]`

With the `macros` feature, derive a tool from an async function:

```rust
use daimon::prelude::*;

/// Adds two numbers and returns the sum.
#[tool_fn]
async fn add(
    /// The first number.
    a: f64,
    /// The second number.
    b: f64,
) -> daimon::Result<ToolOutput> {
    Ok(ToolOutput::text(format!("{}", a + b)))
}

#[tokio::main]
async fn main() -> daimon::Result<()> {
    let agent = Agent::builder()
        .model(daimon::model::openai::OpenAi::new("gpt-4o"))
        .system_prompt("You are a helpful assistant. Use tools when needed.")
        .tool(Add)  // PascalCase struct generated from `add`
        .build()?;

    let response = agent.prompt("What is 42 + 58?").await?;
    println!("{}", response.text());
    Ok(())
}
```

The macro generates a struct (`Add`), JSON Schema from parameter types, and doc comments become the tool description.

---

## Streaming Responses

Use `prompt_stream` to receive events as the model generates. Consume with `StreamExt`:

```rust
use daimon::model::openai::OpenAi;
use daimon::prelude::*;

#[tokio::main]
async fn main() -> daimon::Result<()> {
    let agent = Agent::builder()
        .model(OpenAi::new("gpt-4o"))
        .system_prompt("You are a helpful assistant.")
        .build()?;

    let mut stream = agent
        .prompt_stream("Explain quantum computing in 3 sentences.")
        .await?;

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::TextDelta(text) => print!("{text}"),
            StreamEvent::ToolCallStart { name, .. } => eprintln!("\n[calling tool: {name}]"),
            StreamEvent::ToolCallDelta { .. } => {}
            StreamEvent::ToolCallEnd { .. } => {}
            StreamEvent::ToolResult { content, .. } => eprintln!("[result: {content}]"),
            StreamEvent::Usage { iteration, input_tokens, output_tokens, estimated_cost } => {
                eprintln!("\n[iter {iteration}: {input_tokens} in, {output_tokens} out, ${estimated_cost:.4}]");
            }
            StreamEvent::Error(msg) => eprintln!("\n[error: {msg}]"),
            StreamEvent::Done => {
                println!();
                break;
            }
        }
    }

    Ok(())
}
```

| Event            | Meaning                                                |
|------------------|--------------------------------------------------------|
| `TextDelta`      | Chunk of generated text                                |
| `ToolCallStart`  | Tool call begins (id, name known)                      |
| `ToolCallDelta`  | JSON fragment of tool arguments                        |
| `ToolCallEnd`    | Arguments complete, tool will execute                  |
| `ToolResult`     | Tool output (content, is_error)                       |
| `Usage`          | Token counts and estimated cost for this iteration    |
| `Error`          | Non-fatal error; stream may continue                   |
| `Done`           | Stream finished                                        |

---

## Memory

By default, agents use `SlidingWindowMemory` (50 messages). Switch to `TokenWindowMemory` for a token budget:

```rust
use daimon::model::openai::OpenAi;
use daimon::prelude::*;

#[tokio::main]
async fn main() -> daimon::Result<()> {
    let agent = Agent::builder()
        .model(OpenAi::new("gpt-4o"))
        .system_prompt("You are a helpful assistant. Remember what the user tells you.")
        .memory(TokenWindowMemory::new(4096))  // ~4k token budget
        .build()?;

    // First turn
    let r1 = agent.prompt("My name is Alice.").await?;
    println!("1: {}", r1.text());

    // Second turn — agent remembers
    let r2 = agent.prompt("What's my name?").await?;
    println!("2: {}", r2.text());  // "Alice"

    Ok(())
}
```

`SlidingWindowMemory::new(n)` keeps the last `n` messages. `TokenWindowMemory::new(budget)` evicts oldest messages when the estimated token count exceeds the budget.

---

## Using Different Providers

### OpenAI (default)

```rust
use daimon::model::openai::OpenAi;

let model = OpenAi::new("gpt-4o");
// Or: OpenAi::with_api_key("gpt-4o", api_key)
```

### Anthropic

```rust
use daimon::model::anthropic::Anthropic;

let model = Anthropic::new("claude-sonnet-4-20250514");
// Set ANTHROPIC_API_KEY
```

### Ollama (local)

```rust
use daimon::model::ollama::Ollama;

let model = Ollama::new("llama3.2");
// Requires Ollama running at localhost:11434
```

### Gemini (feature = "gemini")

```rust
use daimon::model::gemini::Gemini;

let model = Gemini::new("gemini-2.0-flash");
// Or: Gemini::with_api_key("gemini-pro", api_key)
// Set GOOGLE_API_KEY
```

### Azure OpenAI (feature = "azure")

```rust
use daimon::model::azure::AzureOpenAi;

let model = AzureOpenAi::new(
    "https://my-resource.openai.azure.com",
    "gpt-4o",
);
// Or: AzureOpenAi::with_api_key(endpoint, deployment, api_key)
// Set AZURE_OPENAI_API_KEY
```

### Bedrock (feature = "bedrock")

```rust
use daimon::model::bedrock::Bedrock;

let model = Bedrock::new("us.anthropic.claude-sonnet-4-20250514")
    .with_region("us-east-1");
// Uses AWS credentials from env or default chain
```

---

## Structured Output

Use `prompt_structured` to get typed responses via serde:

```rust
use daimon::model::openai::OpenAi;
use daimon::prelude::*;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Sentiment {
    label: String,
    confidence: f64,
}

#[tokio::main]
async fn main() -> daimon::Result<()> {
    let agent = Agent::builder()
        .model(OpenAi::new("gpt-4o"))
        .build()?;

    let result: StructuredOutput<Sentiment> = agent
        .prompt_structured(
            "Analyze sentiment: 'Rust is amazing!'",
            "Sentiment",
        )
        .await?;

    println!("{:?}", result.data);      // Sentiment { label: "positive", confidence: 0.95 }
    println!("{}", result.raw_text);    // Raw model output
    Ok(())
}
```

The agent instructs the model to return JSON matching the schema. On parse failure, it retries with the error message — up to 3 total attempts.

---

## Error Handling

Match on `DaimonError` variants:

```rust
use daimon::prelude::*;

fn handle_error(e: &DaimonError) {
    match e {
        DaimonError::Model(msg) => eprintln!("Model error: {msg}"),
        DaimonError::ToolExecution { tool, message } => eprintln!("Tool {tool} failed: {message}"),
        DaimonError::ToolNotFound(name) => eprintln!("Tool '{name}' not found"),
        DaimonError::Builder(msg) => eprintln!("Builder error: {msg}"),
        DaimonError::MaxIterations(n) => eprintln!("Exceeded {n} iterations"),
        DaimonError::SchemaValidation { tool, errors } => eprintln!("Schema error for {tool}: {errors}"),
        DaimonError::BudgetExceeded { spent, limit } => eprintln!("Budget ${spent:.2} exceeded ${limit:.2}"),
        DaimonError::Cancelled => eprintln!("Operation cancelled"),
        _ => eprintln!("{e}"),
    }
}

#[tokio::main]
async fn main() -> daimon::Result<()> {
    let agent = Agent::builder()
        .model(daimon::model::openai::OpenAi::new("gpt-4o"))
        .build()?;

    match agent.prompt("Hello").await {
        Ok(r) => println!("{}", r.text()),
        Err(e) => handle_error(&e),
    }
    Ok(())
}
```

Use `?` with the `Result` type alias for propagation:

```rust
let response = agent.prompt("Hello").await?;
```

---

## Next Steps

- **[agents.md](agents.md)** — Agent builder, ReAct loop, multi-agent patterns, resumable runs
- **[tools.md](tools.md)** — Tool trait, registry, `#[tool_fn]`, retry policies
- **[orchestration.md](orchestration.md)** — Chain, Graph, DAG, Workflow
- **[architecture.md](architecture.md)** — Design philosophy, plugin boundary, workspace layout
