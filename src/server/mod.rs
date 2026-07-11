//! HTTP server wrapper for exposing an agent as a REST API.
//!
//! Provides `POST /prompt`, `POST /prompt/stream` (SSE), and `GET /health` endpoints.
//! Optionally requires API key authentication via `Authorization: Bearer <key>` or
//! `X-API-Key: <key>` headers.
//!
//! ```ignore
//! use daimon::server::AgentServer;
//!
//! let server = AgentServer::new(agent)
//!     .bind("0.0.0.0:8080")
//!     .api_key("my-secret-key");
//! server.serve().await?;
//! ```

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::sse::{Event, Sse},
    routing::{get, post},
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::agent::Agent;
use crate::distributed::SerializableStreamEvent;
use crate::error::Result;
use crate::stream::StreamEvent;

struct AppState {
    agent: Agent,
    api_key: Option<String>,
}

/// Request body for `POST /prompt`.
#[derive(Deserialize)]
pub struct PromptRequest {
    pub input: String,
}

/// Response body from `POST /prompt`.
#[derive(Serialize)]
pub struct PromptResponse {
    pub text: String,
    pub iterations: usize,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cost: f64,
}

/// HTTP server wrapper around an [`Agent`].
///
/// Configure with builder methods, then call [`serve`](AgentServer::serve).
pub struct AgentServer {
    agent: Agent,
    bind_addr: String,
    api_key: Option<String>,
}

impl AgentServer {
    /// Wraps an agent in a server.
    pub fn new(agent: Agent) -> Self {
        Self {
            agent,
            bind_addr: "0.0.0.0:8080".to_string(),
            api_key: None,
        }
    }

    /// Sets the bind address (e.g. `"0.0.0.0:3000"`).
    pub fn bind(mut self, addr: impl Into<String>) -> Self {
        self.bind_addr = addr.into();
        self
    }

    /// Requires API key authentication. Requests must include a matching
    /// `Authorization: Bearer <key>` or `X-API-Key: <key>` header, or they
    /// will receive a 401 Unauthorized response.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Starts the HTTP server. This blocks until the server shuts down.
    ///
    /// Logs a warning when no API key is configured: the default bind
    /// address is `0.0.0.0:8080`, which exposes the agent to every network
    /// interface without authentication.
    pub async fn serve(self) -> Result<()> {
        if self.api_key.is_none() {
            tracing::warn!(
                addr = %self.bind_addr,
                "agent server starting WITHOUT authentication — every endpoint is publicly \
                 callable on this address; call .api_key(...) to require an API key"
            );
        }

        let state = Arc::new(AppState {
            agent: self.agent,
            api_key: self.api_key,
        });

        let app = Router::new()
            .route("/health", get(health))
            .route("/prompt", post(prompt_handler))
            .route("/prompt/stream", post(prompt_stream_handler))
            .with_state(state);

        let listener = tokio::net::TcpListener::bind(&self.bind_addr)
            .await
            .map_err(|e| crate::error::DaimonError::Other(format!("bind error: {e}")))?;

        tracing::info!(addr = %self.bind_addr, "agent server listening");

        axum::serve(listener, app)
            .await
            .map_err(|e| crate::error::DaimonError::Other(format!("server error: {e}")))?;

        Ok(())
    }
}

fn check_api_key(
    state: &AppState,
    headers: &HeaderMap,
) -> std::result::Result<(), (StatusCode, String)> {
    let Some(expected) = &state.api_key else {
        return Ok(());
    };

    let provided = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok()));

    match provided {
        Some(key) if constant_time_eq(key.as_bytes(), expected.as_bytes()) => Ok(()),
        _ => Err((
            StatusCode::UNAUTHORIZED,
            "invalid or missing API key".to_string(),
        )),
    }
}

/// Compares two byte slices in constant time relative to their contents.
///
/// A naive `==` on the API key short-circuits at the first differing byte,
/// leaking — via response timing — how many leading bytes a guess got right and
/// letting an attacker recover the key byte-by-byte. This comparison always
/// inspects every byte of `expected` and folds the result together with a
/// bitwise OR so the running time depends only on the lengths, not on where the
/// first mismatch is.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    // A length mismatch is already observable and not secret-dependent per
    // byte, so returning early here is fine.
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

async fn health() -> &'static str {
    "ok"
}

async fn prompt_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<PromptRequest>,
) -> std::result::Result<Json<PromptResponse>, (StatusCode, String)> {
    check_api_key(&state, &headers)?;

    let response = state
        .agent
        .prompt(&req.input)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(PromptResponse {
        text: response.final_text,
        iterations: response.iterations,
        input_tokens: response.usage.input_tokens,
        output_tokens: response.usage.output_tokens,
        cost: response.cost,
    }))
}

async fn prompt_stream_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<PromptRequest>,
) -> std::result::Result<
    Sse<impl futures::Stream<Item = std::result::Result<Event, axum::Error>>>,
    (StatusCode, String),
> {
    check_api_key(&state, &headers)?;

    let stream = state
        .agent
        .prompt_stream(&req.input)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let sse_stream = stream.map(|event| {
        let event = event.map_err(axum::Error::new)?;
        Ok(Event::default().data(sse_event_json(&event)))
    });

    Ok(Sse::new(sse_stream))
}

/// Serializes a [`StreamEvent`] as a machine-parseable JSON SSE payload via
/// [`SerializableStreamEvent`] (previously the `Debug` string was sent, which
/// clients could not parse).
fn sse_event_json(event: &StreamEvent) -> String {
    let serializable = SerializableStreamEvent::from(event);
    serde_json::to_string(&serializable).unwrap_or_else(|e| {
        // These plain enums serialize infallibly in practice; if that ever
        // changes, still emit valid JSON rather than panicking mid-stream.
        serde_json::json!({ "Error": format!("event serialization failed: {e}") }).to_string()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Model;
    use crate::model::types::{ChatRequest, ChatResponse, Message, StopReason, Usage};
    use crate::stream::ResponseStream;

    struct EchoModel;

    impl Model for EchoModel {
        async fn generate(&self, _request: &ChatRequest) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message::assistant("ok"),
                stop_reason: StopReason::EndTurn,
                usage: Some(Usage::default()),
            })
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    fn state_with_key(key: Option<&str>) -> AppState {
        AppState {
            agent: Agent::builder().model(EchoModel).build().unwrap(),
            api_key: key.map(|k| k.to_string()),
        }
    }

    #[test]
    fn test_constant_time_eq() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secreT"));
        assert!(!constant_time_eq(b"secret", b"secret-longer"));
        assert!(!constant_time_eq(b"", b"x"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn test_no_key_configured_allows_all() {
        let state = state_with_key(None);
        let headers = HeaderMap::new();
        assert!(check_api_key(&state, &headers).is_ok());
    }

    #[test]
    fn test_correct_bearer_key_accepted() {
        let state = state_with_key(Some("s3cr3t"));
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer s3cr3t".parse().unwrap());
        assert!(check_api_key(&state, &headers).is_ok());
    }

    #[test]
    fn test_correct_x_api_key_accepted() {
        let state = state_with_key(Some("s3cr3t"));
        let mut headers = HeaderMap::new();
        headers.insert("x-api-key", "s3cr3t".parse().unwrap());
        assert!(check_api_key(&state, &headers).is_ok());
    }

    #[test]
    fn test_wrong_key_rejected() {
        let state = state_with_key(Some("s3cr3t"));
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer wrong".parse().unwrap());
        let err = check_api_key(&state, &headers).unwrap_err();
        assert_eq!(err.0, StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn test_missing_key_rejected() {
        let state = state_with_key(Some("s3cr3t"));
        let headers = HeaderMap::new();
        assert!(check_api_key(&state, &headers).is_err());
    }

    #[test]
    fn test_sse_event_payload_is_parseable_json() {
        let json = sse_event_json(&StreamEvent::TextDelta("hello".into()));
        let parsed: SerializableStreamEvent = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, SerializableStreamEvent::TextDelta(ref t) if t == "hello"));
    }

    #[test]
    fn test_sse_tool_call_event_round_trips() {
        let json = sse_event_json(&StreamEvent::ToolCallStart {
            id: "tc-1".into(),
            name: "calc".into(),
        });
        let parsed: SerializableStreamEvent = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            parsed,
            SerializableStreamEvent::ToolCallStart { ref id, ref name }
                if id == "tc-1" && name == "calc"
        ));
    }

    #[test]
    fn test_sse_done_event_round_trips() {
        let json = sse_event_json(&StreamEvent::Done);
        let parsed: SerializableStreamEvent = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, SerializableStreamEvent::Done));
    }
}
