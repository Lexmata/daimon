# MCP SSE + WebSocket transports

Date: 2026-07-08
Status: approved, ready for implementation plan

## Context

`local-code` (a downstream consumer of `daimon`) is adding an in-TUI stepper
for configuring MCP servers, and needs to support servers reachable over
the legacy MCP "HTTP+SSE" transport, in addition to the `Stdio` and `Http`
transports `daimon` already ships. It also still configures servers over
`WebSocketTransport`, which existed in `daimon` 0.16.0 but was deliberately
removed in 0.17.0 (commit `c31e03b`, "non-spec surface with no consumers").
That's no longer true — local-code is a real consumer — so it needs to come
back.

This is a `daimon`-side change: both transports are implementations of the
public `McpTransport` trait (`src/mcp/transport.rs`), the same extension
point `StdioTransport`/`HttpTransport` already use.

## Scope

1. Restore `WebSocketTransport` (client-side only — no `ws_server.rs`/gRPC;
   local-code never runs an MCP server, only connects to one).
2. Add a new `SseTransport` implementing the MCP "HTTP+SSE" transport.
3. Re-export both from `mcp/mod.rs`.
4. Handle the `opentelemetry` version-pin risk `tokio-tungstenite` may
   reintroduce.
5. Bump to 0.18.0 and publish to crates.io.

## 1. Restore `WebSocketTransport`

Restore `src/mcp/websocket.rs` verbatim from git history at
`c31e03b^:src/mcp/websocket.rs` (159 lines, previously tested, over
`tokio-tungstenite`). Re-add the `tokio-tungstenite` dependency (whatever
version resolves cleanly against current deps — see §4) and the `pub mod
websocket;` / `pub use websocket::WebSocketTransport;` lines in `mcp/mod.rs`,
gated the same way `HttpTransport`/`StdioTransport` are (under the `mcp`
feature). Restore its existing unit test (`test_transport_types_are_send_sync`)
unmodified.

Do **not** restore `ws_server.rs` or the gRPC transport — those served the
removed MCP *server* surface, which no consumer (including local-code) needs.

## 2. `SseTransport` (new)

Implements the MCP "HTTP+SSE" transport (the pre-Streamable-HTTP transport
still in wide use): a persistent GET request receives an `text/event-stream`
response; JSON-RPC requests are sent via separate POSTs; all responses and
server-initiated notifications arrive asynchronously as SSE frames on the
original GET stream.

```rust
pub struct SseTransport {
    post_url: Arc<Mutex<String>>,           // may be updated by an `endpoint` frame
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<JsonRpcResponse>>>>,
    client: reqwest::Client,
    headers: HashMap<String, String>,
    reader_task: tokio::task::JoinHandle<()>,
}

impl SseTransport {
    pub async fn connect(url: impl Into<String>, headers: HashMap<String, String>) -> Result<Self> { ... }
}
```

- **Connect**: GET `url` with `Accept: text/event-stream` plus any configured
  headers. Spawn a background task that reads the response body via
  `reqwest`'s streamed `bytes_stream()` (the `stream` feature is already
  enabled in `Cargo.toml`), buffering and splitting on blank-line-terminated
  SSE frames (`event: <name>\ndata: <payload>\n\n`).
- **Endpoint discovery**: the first `event: endpoint` frame's `data` is the
  URL to POST subsequent JSON-RPC messages to (per the MCP spec; may be
  relative to `url`, resolve against it). If no such frame ever arrives
  before the first `send`/`notify` call, fall back to POSTing to the
  original `url` — several simplified server implementations skip the
  `endpoint` indirection and just accept POSTs on the same path.
- **Correlation**: `send()` generates the request's `id` (already assigned
  by `McpClient` — see `protocol::JsonRpcRequest.id: u64`), registers a
  `oneshot::Sender` in `pending` keyed by that `id`, POSTs the request body,
  then awaits the oneshot (no timeout beyond whatever the caller imposes —
  matches `StdioTransport`/`HttpTransport`, neither of which have one
  either). The background reader task, on each `event: message` frame,
  deserializes it as a `JsonRpcResponse`; if its `id` matches a pending
  sender, resolves it; otherwise (a notification, or a response with no
  matching pending id) it's dropped — `McpTransport` has no receive-side
  notification hook today, matching the other transports' behavior.
- **notify()**: POSTs only, no pending-slot registered.
- **close()**: aborts the reader task, drops any still-pending senders (their
  awaiting `send()` calls get a "transport closed" error via the dropped
  oneshot).
- **Errors**: connect failure, a malformed frame, or the stream ending all
  surface as `DaimonError::Mcp(...)`, consistent with the other transports.

## 3. Re-exports

`mcp/mod.rs` gains `pub mod sse; pub use sse::SseTransport;` alongside the
restored websocket lines.

## 4. opentelemetry version-pin risk

commit `c31e03b` notes `tokio-tungstenite` was previously forcing
`opentelemetry_sdk` to 0.32 while `opentelemetry`/`opentelemetry-otlp` sat at
0.31 — a latent mismatch only exposed once `tokio-tungstenite` was removed
(it had been accidentally unifying the versions). Re-adding
`tokio-tungstenite`: if `cargo build`/`cargo tree` shows that mismatch
resurfacing, bump `opentelemetry`/`opentelemetry-otlp` to 0.32 to match
rather than re-pinning `opentelemetry_sdk` down — i.e. move forward to
whatever version resolves cleanly, not backward. If it doesn't resurface,
no action needed.

## 5. Versioning

Bump `Cargo.toml` version to `0.18.0`. Dry-run `cargo publish`, confirm with
the user, then publish for real (same flow used for `ntui` 0.1.1 → 0.1.2
earlier in this work) — publishing is irreversible, so always confirm first.

## Testing

- `SseTransport`: a hand-rolled local SSE server (`tokio::net::TcpListener`,
  writing a raw HTTP/SSE response by hand — no new test-only HTTP framework
  dependency) covering:
  - endpoint-discovery frame updates the POST target
  - no endpoint frame → falls back to POSTing the connect URL
  - request/response correlation by id, including out-of-order delivery
    (response B arrives before response A, both resolve correctly)
  - malformed frame is skipped, not fatal to the stream
  - connect failure (nothing listening) surfaces as `DaimonError::Mcp`
  - `close()` fails any still-pending `send()` calls cleanly
- `WebSocketTransport`: restore the prior test
  (`test_transport_types_are_send_sync`) and confirm it still passes against
  the current `tokio-tungstenite` version.
- Full workspace `cargo test`/`cargo clippy --all-targets -D warnings` must
  pass before publish (existing `cargo-husky` pre-commit hook already
  enforces clippy).

## Out of scope

- MCP server-side SSE/WebSocket support (`ws_server.rs`, `McpWsServer`) —
  not restored; no consumer needs it.
- A `timeout` on `SseTransport::send`/`notify` — matches existing
  transports' lack of one; can be added later as its own change if needed.
- Reconnect/retry on a dropped SSE stream — a fresh `connect()` is required;
  automatic reconnection is future work if it turns out to matter in
  practice.
