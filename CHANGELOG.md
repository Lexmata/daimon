# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/Lexmata/daimon/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Lexmata/daimon/releases/tag/v0.1.0
