# Troubleshooting

## Compilation Issues

### Feature flag conflicts

If you see errors about missing types or unresolved imports, you likely need to enable the correct feature flag. Each provider and backend is gated:

```toml
# Enable only what you need
daimon = { version = "0.22", features = ["anthropic", "bedrock", "pgvector"] }
```

To check which features are available:
```bash
cargo doc --open  # Look at the feature flags section
```

### `daimon-core` version mismatch

If you are developing a plugin crate that depends on `daimon-core` and you see trait mismatch errors, ensure your `daimon-core` version matches the one used by the main `daimon` crate. In a workspace, use a path dependency to avoid version drift.

### `full` feature takes too long to compile

The `full` feature enables every provider, storage backend, and protocol. For development, enable only the features you are testing:

```bash
# Instead of:
cargo test --features full

# Prefer:
cargo test --features "openai,anthropic"
```

## Runtime Errors

### Agent returns empty text

This usually means the provider returned a valid HTTP response but with no content. Common causes:

- The model name is incorrect (e.g., typo in `gpt-4o` or `claude-sonnet-4-20250514`)
- The API key is missing or invalid (check the relevant environment variable)
- The request exceeded the provider's context window
- The provider is experiencing an outage

### `tool '<name>' not found in registry`

The agent's tool registry is case-sensitive. Ensure the tool name returned by `Tool::name()` matches exactly what the model requested. If the model hallucinates tool names, add a system prompt that lists available tools.

### Infinite tool loops

If the agent keeps calling tools without producing a final response, check:

1. **max_iterations:** Set a reasonable limit (default is 25). Lower it if the task is simple.
2. **Tool output:** Ensure tools return meaningful output. An empty or ambiguous result may cause the model to retry.
3. **System prompt:** Include explicit instructions about when to stop using tools and produce a final answer.

### Streaming stops mid-response

- Check for network timeouts. Increase the provider timeout: `.with_timeout(Duration::from_secs(120))`
- If using cancellation tokens, ensure nothing is cancelling the stream prematurely
- Check provider rate limits -- throttled requests may drop mid-stream

### Bedrock "AccessDeniedException"

The AWS Bedrock provider requires:
1. An IAM role/user with `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` permissions
2. The model must be enabled in your AWS account (Bedrock console > Model access)
3. The correct AWS region (not all models are available in all regions)

```bash
# Verify your credentials
aws sts get-caller-identity
aws bedrock list-foundation-models --region us-east-1 | grep claude
```

### Memory keeps growing

If agent memory grows unbounded:
- Use `SlidingWindowMemory` with a reasonable limit (e.g., 50-100 messages)
- For long-running agents, use the built-in `SummaryMemory` (summarizes old messages via LLM) or `TokenWindowMemory` (token budget)
- Check that tool outputs are not excessively large (the full output is stored in memory)

## Testing

### Tests require specific feature flags

Some tests require specific feature flags:

```bash
# Run tests with default features (openai + anthropic + ollama + macros)
cargo test

# Run tests with all features
cargo test --features full

# Run tests with no default features (core only)
cargo test --no-default-features
```

### Async test panics

Ensure async tests use `#[tokio::test]`:

```rust
#[tokio::test]
async fn test_something() {
    // ...
}
```

If a test spawns background tasks, use `tokio::time::timeout` to prevent hanging:

```rust
tokio::time::timeout(Duration::from_secs(5), agent.prompt("test")).await??;
```

## Provider-Specific Issues

### OpenAI: 429 Rate Limit

Daimon includes built-in retry logic with exponential backoff. If you are still hitting rate limits:
- Increase `max_retries` on the provider
- Add a delay between sequential prompts
- Use a higher-tier API key

### Anthropic: "overloaded_error"

Anthropic returns this when the API is under heavy load. The retry logic handles transient cases. For persistent errors, check the [Anthropic status page](https://status.anthropic.com/).

### Gemini: Authentication failure

The Gemini provider authenticates with an API key sent via the `x-goog-api-key` header. Ensure:
- `GOOGLE_API_KEY` is set to a valid Gemini API key (or pass one explicitly with `.with_api_key(...)`)
- The key is enabled for the Generative Language API

(Vertex AI is only reachable as an advanced manual path: `.with_base_url(...)` plus `.with_bearer_token()` with an OAuth2 access token you obtain yourself — there is no service-account/ADC flow in the crate.)
