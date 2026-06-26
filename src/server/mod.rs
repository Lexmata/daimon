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
use crate::error::Result;

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
    pub async fn serve(self) -> Result<()> {
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
        Some(key) if key == expected.as_str() => Ok(()),
        _ => Err((
            StatusCode::UNAUTHORIZED,
            "invalid or missing API key".to_string(),
        )),
    }
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
        let data = serde_json::to_string(&format!("{event:?}")).unwrap_or_default();
        Ok(Event::default().data(data))
    });

    Ok(Sse::new(sse_stream))
}
