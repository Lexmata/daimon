//! A2A HTTP client for calling remote A2A agents.

#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "azure",
    feature = "ollama",
    feature = "mcp",
))]
use std::time::Duration;

#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "azure",
    feature = "ollama",
    feature = "mcp",
))]
use crate::error::{DaimonError, Result};

#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "gemini",
    feature = "azure",
    feature = "ollama",
    feature = "mcp",
))]
use super::types::*;

/// HTTP client for interacting with remote A2A agents.
///
/// Implements the client side of the A2A protocol: discovery, task
/// creation, status polling, and cancellation.
pub struct A2aClient {
    #[cfg(any(
        feature = "openai",
        feature = "anthropic",
        feature = "gemini",
        feature = "azure",
        feature = "ollama",
        feature = "mcp",
    ))]
    http: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
}

impl A2aClient {
    /// Creates a new A2A client pointing at the given base URL.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            #[cfg(any(
                feature = "openai",
                feature = "anthropic",
                feature = "gemini",
                feature = "azure",
                feature = "ollama",
                feature = "mcp",
            ))]
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("failed to build HTTP client"),
            base_url: base_url.into(),
            api_key: None,
        }
    }

    /// Sets an API key for authentication.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Discovers the remote agent by fetching its Agent Card.
    #[cfg(any(
        feature = "openai",
        feature = "anthropic",
        feature = "gemini",
        feature = "azure",
        feature = "ollama",
        feature = "mcp",
    ))]
    pub async fn discover(&self) -> Result<AgentCard> {
        let resp = self
            .rpc_call("agent/discover", serde_json::json!({}))
            .await?;
        serde_json::from_value(resp)
            .map_err(|e| DaimonError::Other(format!("A2A parse error: {e}")))
    }

    /// Sends a task to the remote agent.
    #[cfg(any(
        feature = "openai",
        feature = "anthropic",
        feature = "gemini",
        feature = "azure",
        feature = "ollama",
        feature = "mcp",
    ))]
    pub async fn send_task(&self, params: TaskSendParams) -> Result<A2aTask> {
        let resp = self
            .rpc_call("tasks/send", serde_json::to_value(&params)?)
            .await?;
        serde_json::from_value(resp)
            .map_err(|e| DaimonError::Other(format!("A2A parse error: {e}")))
    }

    /// Gets the status of a task.
    #[cfg(any(
        feature = "openai",
        feature = "anthropic",
        feature = "gemini",
        feature = "azure",
        feature = "ollama",
        feature = "mcp",
    ))]
    pub async fn get_task(&self, task_id: &str) -> Result<A2aTask> {
        let params = TaskGetParams {
            id: task_id.to_string(),
        };
        let resp = self
            .rpc_call("tasks/get", serde_json::to_value(&params)?)
            .await?;
        serde_json::from_value(resp)
            .map_err(|e| DaimonError::Other(format!("A2A parse error: {e}")))
    }

    /// Cancels a task.
    #[cfg(any(
        feature = "openai",
        feature = "anthropic",
        feature = "gemini",
        feature = "azure",
        feature = "ollama",
        feature = "mcp",
    ))]
    pub async fn cancel_task(&self, task_id: &str) -> Result<A2aTask> {
        let params = TaskCancelParams {
            id: task_id.to_string(),
        };
        let resp = self
            .rpc_call("tasks/cancel", serde_json::to_value(&params)?)
            .await?;
        serde_json::from_value(resp)
            .map_err(|e| DaimonError::Other(format!("A2A parse error: {e}")))
    }

    /// Sends a simple text message as a task and returns the completed task.
    #[cfg(any(
        feature = "openai",
        feature = "anthropic",
        feature = "gemini",
        feature = "azure",
        feature = "ollama",
        feature = "mcp",
    ))]
    pub async fn send_text(&self, text: &str) -> Result<A2aTask> {
        self.send_task(TaskSendParams {
            id: None,
            message: A2aMessage {
                role: A2aRole::User,
                parts: vec![Part::Text {
                    text: text.to_string(),
                }],
                message_id: None,
                metadata: std::collections::HashMap::new(),
            },
            context_id: None,
            metadata: std::collections::HashMap::new(),
        })
        .await
    }

    #[cfg(any(
        feature = "openai",
        feature = "anthropic",
        feature = "gemini",
        feature = "azure",
        feature = "ollama",
        feature = "mcp",
    ))]
    async fn rpc_call(&self, method: &str, params: serde_json::Value) -> Result<serde_json::Value> {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: serde_json::json!(1),
            method: method.to_string(),
            params,
        };

        let mut http_req = self.http.post(&self.base_url).json(&request);

        if let Some(ref key) = self.api_key {
            http_req = http_req.header("X-API-Key", key);
        }

        let http_resp = http_req
            .send()
            .await
            .map_err(|e| DaimonError::Other(format!("A2A HTTP error: {e}")))?;

        let response: JsonRpcResponse = http_resp
            .json()
            .await
            .map_err(|e| DaimonError::Other(format!("A2A response parse error: {e}")))?;

        if let Some(err) = response.error {
            return Err(DaimonError::Other(format!(
                "A2A error {}: {}",
                err.code, err.message
            )));
        }

        response.result.ok_or_else(|| {
            DaimonError::Other("A2A response has neither result nor error".to_string())
        })
    }
}

impl std::fmt::Debug for A2aClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("A2aClient")
            .field("base_url", &self.base_url)
            .field("has_api_key", &self.api_key.is_some())
            .finish()
    }
}
