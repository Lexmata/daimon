//! Confirms the `daimon::a2a`/`daimon::prelude` facade re-export paths for
//! the `a2a` feature (DAIM-27) are not just compiling but functionally
//! complete — `A2aClient` can be constructed and its RPC methods round-trip
//! against a real HTTP server.
//!
//! DAIM-27 fixed a silent-partial-compile bug: before that ticket, the
//! `a2a` module was reachable through several unrelated feature flags
//! (openai/anthropic/ollama/mcp) rather than its own, so a build that
//! disabled all four could still type-check other code paths while quietly
//! never compiling `a2a` at all. `cargo hack check --each-feature` proves
//! the module compiles under `--features a2a` alone, but compiling is not
//! the same as working: this file exercises the client against a hermetic
//! mock server so a future regression that reintroduces silent gating (or
//! breaks the client's wiring while leaving it compilable) fails a test,
//! not just a review.
//!
//! No external HTTP-mocking crate is used (none is a workspace dependency
//! by convention — see `src/mcp/sse.rs`'s hand-rolled test server for
//! precedent). The mock server below is a minimal `tokio::net::TcpListener`
//! loop that reads one HTTP/1.1 request, ignores everything but the body,
//! and writes back a canned HTTP/1.1 response.

#![cfg(feature = "a2a")]

use std::net::SocketAddr;

use daimon::a2a::A2aClient;
use serde_json::json;
use tokio::io::AsyncBufReadExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};

/// Reads one HTTP/1.1 request's method, path, and (if `Content-Length` is
/// present) body off `stream`. Blocks until the request line and headers
/// have arrived.
async fn read_http_request(stream: &mut TcpStream) -> (String, String, String) {
    let mut reader = BufReader::new(&mut *stream);
    let mut request_line = String::new();
    reader.read_line(&mut request_line).await.unwrap();
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();

    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).await.unwrap();
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break;
        }
        if let Some((key, value)) = trimmed.split_once(':')
            && key.eq_ignore_ascii_case("content-length")
        {
            content_length = value.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body).await.unwrap();
    }
    (method, path, String::from_utf8_lossy(&body).into_owned())
}

/// Writes a `200 OK` response with a JSON body.
async fn write_json_response(stream: &mut TcpStream, body: &str) {
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(response.as_bytes()).await.unwrap();
}

/// Starts a one-shot mock A2A server: accepts a single connection, parses
/// the JSON-RPC request, hands the parsed body to `respond` to build a
/// reply, writes that reply back, then the task ends. Returns the address
/// to point an `A2aClient` at.
async fn start_mock_server(
    respond: impl FnOnce(serde_json::Value) -> serde_json::Value + Send + 'static,
) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        let (mut conn, _) = listener.accept().await.unwrap();
        let (_method, _path, body) = read_http_request(&mut conn).await;
        let request: serde_json::Value = serde_json::from_str(&body).unwrap();
        let reply = respond(request);
        write_json_response(&mut conn, &serde_json::to_string(&reply).unwrap()).await;
    });

    addr
}

#[tokio::test]
async fn a2a_client_constructs_and_discovers_against_mock_server() {
    let addr = start_mock_server(|request| {
        assert_eq!(request["method"], "agent/discover");
        json!({
            "jsonrpc": "2.0",
            "id": request["id"],
            "result": {
                "name": "MockAgent",
                "description": "A hermetic test agent",
                "version": "1.0.0",
                "url": "http://mock.local/a2a",
                "protocolVersion": "0.2",
            }
        })
    })
    .await;

    let client = A2aClient::new(format!("http://{addr}"));
    let card = client.discover().await.expect("discover should succeed");
    assert_eq!(card.name, "MockAgent");
    assert_eq!(card.version, "1.0.0");
}

#[tokio::test]
async fn a2a_client_send_text_round_trips_a_task() {
    let addr = start_mock_server(|request| {
        assert_eq!(request["method"], "tasks/send");
        assert_eq!(
            request["params"]["message"]["parts"][0]["text"],
            "hello there"
        );
        json!({
            "jsonrpc": "2.0",
            "id": request["id"],
            "result": {
                "id": "task-1",
                "status": { "state": "completed" },
            }
        })
    })
    .await;

    let client = A2aClient::new(format!("http://{addr}"));
    let task = client
        .send_text("hello there")
        .await
        .expect("send_text should succeed");
    assert_eq!(task.id, "task-1");
}

#[tokio::test]
async fn a2a_client_surfaces_json_rpc_error_as_err() {
    let addr = start_mock_server(|request| {
        json!({
            "jsonrpc": "2.0",
            "id": request["id"],
            "error": {
                "code": -32601,
                "message": "Method not found",
            }
        })
    })
    .await;

    let client = A2aClient::new(format!("http://{addr}"));
    let result = client.discover().await;
    let err = result.expect_err("a JSON-RPC error response must surface as Err, not Ok");
    let message = err.to_string();
    assert!(
        message.contains("Method not found"),
        "error message should include the JSON-RPC error text, got: {message}"
    );
}
