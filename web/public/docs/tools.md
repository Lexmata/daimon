# Tools

Tools are callable functions the agent invokes during the ReAct loop. Each tool declares a name, description, and JSON Schema for its parameters. The model uses this metadata to decide when and how to call tools.

---

## The Tool Trait

The core abstraction is the [`Tool`] trait:

```rust
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;

    fn execute(&self, input: &serde_json::Value)
        -> impl Future<Output = Result<ToolOutput>> + Send;

    fn retry_policy(&self) -> Option<ToolRetryPolicy> {
        None
    }
}
```

| Method | Purpose |
|--------|---------|
| `name()` | Unique identifier. The model uses this when requesting a tool call. |
| `description()` | Human-readable description. The model uses this to decide *when* to call the tool. |
| `parameters_schema()` | JSON Schema for the tool's parameters. Validates and guides argument generation. |
| `execute()` | Runs the tool with the given arguments. Returns `ToolOutput`. |
| `retry_policy()` | Optional per-tool retry policy. Overrides agent-level policy when `Some`. |

Tools must be `Send + Sync` because they are invoked across async boundaries. The model receives `name`, `description`, and `parameters` (the schema) for each tool when building the chat request.

---

## Building Tools Manually

Implement `Tool` for a struct. The schema drives both validation and model behavior.

### Step 1: Define the struct and implement `Tool`

```rust
use daimon::prelude::*;
use serde_json::json;

struct Calculator;

impl Tool for Calculator {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Evaluate simple math expressions. Supports add, subtract, multiply, divide."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation"
                },
                "a": { "type": "number", "description": "First operand" },
                "b": { "type": "number", "description": "Second operand" }
            },
            "required": ["operation", "a", "b"]
        })
    }

    async fn execute(&self, input: &serde_json::Value) -> daimon::Result<ToolOutput> {
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
```

### Step 2: Return `ToolOutput`

Use the appropriate constructor for the result:

| Constructor | Use case |
|-------------|----------|
| `ToolOutput::text(s)` | Plain text result. |
| `ToolOutput::json(&value)` | Structured data. Serializes to JSON. |
| `ToolOutput::error(s)` | Error message. `is_error` is set; the model sees it and can retry or adjust. |

```rust
// Success — plain text
Ok(ToolOutput::text("42"))

// Success — structured JSON
Ok(ToolOutput::json(&serde_json::json!({
    "result": 42,
    "operation": "add"
}))?)

// Error — model receives this and can correct
Ok(ToolOutput::error("Division by zero"))
```

Errors returned as `ToolOutput::error(...)` are passed back to the model as tool results, so the agent can recover. Use `Err(...)` for unrecoverable failures that abort the loop.

---

## The `#[tool_fn]` Macro

The `#[tool_fn]` proc macro derives a `Tool` implementation from an async function. Parameters become schema properties; doc comments become descriptions.

### Basic usage

```rust
use daimon::prelude::*;

/// Adds two numbers together and returns the sum.
#[tool_fn]
async fn add(
    /// The first number to add.
    a: f64,
    /// The second number to add.
    b: f64,
) -> daimon::Result<ToolOutput> {
    Ok(ToolOutput::text(format!("{}", a + b)))
}

/// Converts a string to uppercase.
#[tool_fn]
async fn to_uppercase(
    /// The text to convert.
    text: String,
) -> daimon::Result<ToolOutput> {
    Ok(ToolOutput::text(text.to_uppercase()))
}
```

The macro generates a PascalCase struct (`Add`, `ToUppercase`) that implements `Tool`. Use it with the agent:

```rust
let agent = Agent::builder()
    .model(model)
    .tool(Add)
    .tool(ToUppercase)
    .build()?;
```

### Supported parameter types

| Type | JSON Schema |
|------|-------------|
| `String`, `&str` | `{"type": "string"}` |
| `i8`–`i128`, `isize`, `u8`–`u128`, `usize` | `{"type": "integer"}` |
| `f32`, `f64` | `{"type": "number"}` |
| `bool` | `{"type": "boolean"}` |
| `Option<T>` | Same as `T`, but not in `required` |
| `Vec<T>` | `{"type": "array", "items": <T schema>}` |
| `serde_json::Value` | `{}` (accepts anything) |

### Attributes

```rust
#[tool_fn(name = "custom_name")]
#[tool_fn(description = "Override the description")]
#[tool_fn(crate_path = "::my_crate::daimon")]
async fn my_tool(query: String) -> daimon::Result<ToolOutput> {
    Ok(ToolOutput::text(query))
}
```

| Attribute | Purpose |
|-----------|---------|
| `name = "..."` | Override tool name (default: function name) |
| `description = "..."` | Override description (default: doc comments) |
| `crate_path = "..."` | Path to daimon crate (default: `::daimon`) |

### Multiple examples

```rust
/// Search the knowledge base. Returns top matching documents.
#[tool_fn(name = "search_docs")]
async fn search(
    /// The search query.
    query: String,
    /// Maximum number of results. Defaults to 5.
    top_k: Option<usize>,
) -> daimon::Result<ToolOutput> {
    let k = top_k.unwrap_or(5);
    // ... retrieval logic ...
    Ok(ToolOutput::text(results))
}

/// Validate an email address format.
#[tool_fn]
async fn validate_email(
    /// The email to validate.
    email: String,
) -> daimon::Result<ToolOutput> {
    let valid = email.contains('@') && email.contains('.');
    Ok(ToolOutput::json(&serde_json::json!({
        "valid": valid,
        "email": email
    }))?)
}
```

---

## Tool Registry

Tools are collected in a [`ToolRegistry`]. The agent uses it to look up tools by name and to build tool specs for the model.

### Registration

```rust
let mut registry = ToolRegistry::new();

registry.register(Calculator)?;
registry.register(Add)?;
registry.register_shared(Arc::new(some_tool))?;
```

- `register(T)` — Register a tool by value. Returns `Err(DaimonError::DuplicateTool(name))` if the name already exists.
- `register_shared(SharedTool)` — Register a pre-boxed `Arc<dyn ErasedTool>`.

### Lookup and listing

```rust
registry.get("calculator");        // Option<&SharedTool>
registry.list();                   // Vec<&str> — all tool names
registry.len();                    // usize
registry.is_empty();               // bool
```

### Duplicate detection

Registering a tool with an existing name fails:

```rust
registry.register(Add)?;
let err = registry.register(Add);
assert!(matches!(err, Err(DaimonError::DuplicateTool(name)) if name == "add"));
```

### Cache and `warm_cache`

The registry caches compiled JSON Schema validators and tool specs. Call `warm_cache()` after all tools are registered to avoid per-call compilation:

```rust
let mut registry = ToolRegistry::new();
registry.register(tool1)?;
registry.register(tool2)?;
registry.warm_cache();  // Pre-compiles validators and tool specs
```

`Agent::builder().build()` calls `warm_cache()` automatically.

### Generation-based cache invalidation

`tool_specs()` returns cached specs when the registry is unchanged. When tools are registered or unregistered, the generation counter increments and the cache is invalidated. Use `tool_specs_mut()` when you have `&mut self` to persist the computed specs into the cache.

### Unregistering

```rust
registry.unregister("calculator");  // Returns true if the tool was present
```

---

## JSON Schema Validation

When `validate_tool_inputs` is enabled (default), the agent validates each tool call's arguments against the tool's `parameters_schema()` before execution.

### How it works

`ToolRegistry::validate_input(tool_name, input)` returns:

- `None` — Input is valid (or tool not found; validation is skipped).
- `Some(errors)` — Validation failed. `errors` is a string describing the issues.

Validation uses the `jsonschema` crate. Validators are compiled once and cached when you call `warm_cache()` or `compile_validators()`.

### When validation fails

Invalid inputs are **not** executed. Instead, an error message is returned to the model as a tool result:

```
Invalid arguments for tool 'calculator': count: -1 is less than minimum 0; name: "" is too short
```

The model sees this and can correct its arguments on the next iteration.

### Disabling validation

```rust
let agent = Agent::builder()
    .model(model)
    .tool(Calculator)
    .validate_tool_inputs(false)
    .build()?;
```

### Custom schemas

Provide a strict schema in `parameters_schema()` for stronger validation:

```rust
fn parameters_schema(&self) -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "name": { "type": "string", "minLength": 1 },
            "count": { "type": "integer", "minimum": 0 }
        },
        "required": ["name", "count"],
        "additionalProperties": false
    })
}
```

---

## Tool Retry Policy

Transient failures (network timeouts, rate limits) can be retried automatically.

### Agent-level policy

```rust
use daimon::tool::ToolRetryPolicy;
use std::time::Duration;

let agent = Agent::builder()
    .model(model)
    .tool(flaky_tool)
    .tool_retry_policy(ToolRetryPolicy::exponential(3))
    .build()?;
```

### `ToolRetryPolicy::exponential(max_retries)`

Exponential backoff: 100ms base, 10s max.

```rust
ToolRetryPolicy::exponential(3)
// Attempts: 0ms, 100ms, 200ms, 400ms (up to 10s cap)
```

### `ToolRetryPolicy::fixed(max_retries, delay)`

Fixed delay between retries.

```rust
ToolRetryPolicy::fixed(3, Duration::from_secs(2))
```

### `retryable_on(patterns)`

Only retry if the error message contains one of the patterns:

```rust
ToolRetryPolicy::exponential(3)
    .retryable_on(vec!["timeout".into(), "rate limit".into()])
```

- `"connection timeout"` → retried
- `"rate limit exceeded"` → retried
- `"invalid arguments"` → not retried

Empty patterns (default) means retry on any error.

### Per-tool override

Implement `retry_policy()` on a tool to override the agent-level policy:

```rust
impl Tool for FlakyApiTool {
    // ...

    fn retry_policy(&self) -> Option<ToolRetryPolicy> {
        Some(
            ToolRetryPolicy::exponential(5)
                .retryable_on(vec!["503".into(), "timeout".into()])
        )
    }
}
```

Per-tool policy takes precedence over the agent's `tool_retry_policy`.

### Backoff strategies

- **Exponential**: `base * 2^attempt`, capped at `max`.
- **Fixed**: Same delay every attempt.

---

## RetrieverTool

[`RetrieverTool`] wraps a [`Retriever`] as a tool so agents can perform RAG-style retrieval.

### Creating a retriever tool

```rust
use daimon::retriever::{Retriever, RetrieverTool};

let retriever = MyVectorStoreRetriever::new(...);
let tool = RetrieverTool::new(
    retriever,
    "search",
    "Search the knowledge base for relevant documents"
);

// Optional: set default top_k
let tool = RetrieverTool::new(retriever, "search", "Search docs")
    .with_default_top_k(10);
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | The search query |
| `top_k` | integer | No | Max results (default: 5) |

### Output

Returns formatted text with document content, scores, and metadata. Empty results return `"No relevant documents found."`.

---

## MCP Tools

The MCP (Model Context Protocol) client discovers tools from external servers and exposes them as Daimon tools.

### Bridging MCP tools

```rust
use daimon::mcp::{McpClient, StdioTransport};

let transport = StdioTransport::new("npx", ["-y", "my-mcp-server"]).await?;
let client = McpClient::connect(transport).await?;

let mut builder = Agent::builder().model(model);
for tool in client.tools() {
    builder = builder.tool(tool);
}
let agent = builder.build()?;
```

`McpClient::tools()` returns `Vec<McpToolBridge>`. Each bridge implements `Tool` and forwards calls to the MCP server. Name, description, and input schema come from the server's `tools/list` response.

---

## AgentTool

[`AgentTool`] wraps an [`Agent`] as a tool so one agent can delegate to another.

### Creating an agent-as-tool

```rust
use daimon::agent::as_tool::AgentTool;
use std::sync::Arc;

let research_agent = Arc::new(
    Agent::builder()
        .model(model.clone())
        .system_prompt("You are a research specialist.")
        .tool(web_search)
        .build()?
);

let coordinator = Agent::builder()
    .model(model)
    .tool(AgentTool::new(
        research_agent,
        "research",
        "Perform deep research on a topic. Use for complex queries."
    ))
    .build()?;
```

### Input format

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | string | The task or question sent to the sub-agent |

### Output format

The sub-agent's `final_text` is returned as the tool output. On failure, `ToolOutput::error(...)` is returned so the caller can handle it.

---

## Best Practices

### 1. Keep tool descriptions clear and specific

The model uses `description` to decide when to call a tool. Be precise:

```rust
// Good
"Search the product catalog by name or SKU. Returns up to 10 matching products."

// Avoid
"Search for products."
```

### 2. Return structured JSON when possible

Structured output helps the model reason over results:

```rust
Ok(ToolOutput::json(&serde_json::json!({
    "products": [...],
    "total": 42,
    "query": query
}))?)
```

### 3. Use schema validation to catch LLM mistakes early

Enable `validate_tool_inputs` (default) and define strict schemas with `required`, `minLength`, `minimum`, etc. Invalid inputs are returned to the model as errors so it can correct.

### 4. Use retry policies for flaky external services

For tools that call APIs, databases, or other external services:

```rust
.tool_retry_policy(
    ToolRetryPolicy::exponential(3)
        .retryable_on(vec!["timeout".into(), "503".into(), "rate limit".into()])
)
```

Or implement `retry_policy()` on the tool for service-specific behavior.

### 5. Use `ToolOutput::error` for recoverable failures

When the model can correct (e.g., bad arguments, validation), return `Ok(ToolOutput::error(...))`. Reserve `Err(...)` for unrecoverable failures that should abort the loop.

### 6. Document parameters in schemas

Add `description` to schema properties so the model understands each parameter:

```rust
"properties": {
    "query": {
        "type": "string",
        "description": "Natural language search query. Use keywords for best results."
    }
}
```

---

[`Tool`]: https://docs.rs/daimon/latest/daimon/tool/trait.Tool.html
[`ToolRegistry`]: https://docs.rs/daimon/latest/daimon/tool/struct.ToolRegistry.html
[`RetrieverTool`]: https://docs.rs/daimon/latest/daimon/retriever/struct.RetrieverTool.html
[`Retriever`]: https://docs.rs/daimon/latest/daimon/retriever/trait.Retriever.html
[`Agent`]: https://docs.rs/daimon/latest/daimon/agent/struct.Agent.html
