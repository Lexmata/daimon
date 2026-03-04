//! WebSocket transport for MCP.
//!
//! Connects to an MCP server over a persistent WebSocket connection
//! and exchanges JSON-RPC messages as text frames.
//!
//! ```ignore
//! use daimon::mcp::{McpClient, WebSocketTransport};
//!
//! let transport = WebSocketTransport::connect("ws://localhost:3000/mcp").await?;
//! let client = McpClient::connect(transport).await?;
//! ```

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use futures::{SinkExt, StreamExt};
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::Message as WsMessage;

use crate::error::{DaimonError, Result};
use crate::mcp::protocol::{JsonRpcNotification, JsonRpcRequest, JsonRpcResponse};
use crate::mcp::transport::McpTransport;

type WsStream = tokio_tungstenite::WebSocketStream<
    tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
>;

/// WebSocket transport for MCP communication.
///
/// Maintains a persistent WebSocket connection. Requests are serialized
/// and sent as text frames; responses are read from the next text frame.
///
/// Thread-safe: all access to the underlying stream is serialized via
/// an internal mutex.
pub struct WebSocketTransport {
    stream: Arc<Mutex<Option<WsStream>>>,
}

impl WebSocketTransport {
    /// Connects to an MCP server at the given WebSocket URL.
    ///
    /// Supports `ws://` and `wss://` schemes.
    pub async fn connect(url: impl AsRef<str>) -> Result<Self> {
        let (stream, _response) = tokio_tungstenite::connect_async(url.as_ref())
            .await
            .map_err(|e| DaimonError::Mcp(format!("WebSocket connect failed: {e}")))?;

        Ok(Self {
            stream: Arc::new(Mutex::new(Some(stream))),
        })
    }

    async fn send_and_receive(&self, body: &[u8]) -> Result<Vec<u8>> {
        let mut guard = self.stream.lock().await;
        let stream = guard
            .as_mut()
            .ok_or_else(|| DaimonError::Mcp("WebSocket transport closed".into()))?;

        let text = String::from_utf8_lossy(body).into_owned();
        stream
            .send(WsMessage::Text(text.into()))
            .await
            .map_err(|e| DaimonError::Mcp(format!("WebSocket send failed: {e}")))?;

        loop {
            match stream.next().await {
                Some(Ok(WsMessage::Text(text))) => {
                    return Ok(text.as_bytes().to_vec());
                }
                Some(Ok(WsMessage::Ping(data))) => {
                    stream
                        .send(WsMessage::Pong(data))
                        .await
                        .map_err(|e| DaimonError::Mcp(format!("WebSocket pong failed: {e}")))?;
                    continue;
                }
                Some(Ok(WsMessage::Close(_))) => {
                    return Err(DaimonError::Mcp("WebSocket server closed connection".into()));
                }
                Some(Ok(_)) => continue,
                Some(Err(e)) => {
                    return Err(DaimonError::Mcp(format!("WebSocket receive error: {e}")));
                }
                None => {
                    return Err(DaimonError::Mcp("WebSocket stream ended".into()));
                }
            }
        }
    }

    async fn send_fire_and_forget(&self, body: &[u8]) -> Result<()> {
        let mut guard = self.stream.lock().await;
        let stream = guard
            .as_mut()
            .ok_or_else(|| DaimonError::Mcp("WebSocket transport closed".into()))?;

        let text = String::from_utf8_lossy(body).into_owned();
        stream
            .send(WsMessage::Text(text.into()))
            .await
            .map_err(|e| DaimonError::Mcp(format!("WebSocket send failed: {e}")))?;

        Ok(())
    }
}

impl McpTransport for WebSocketTransport {
    fn send<'a>(
        &'a self,
        request: &'a JsonRpcRequest,
    ) -> Pin<Box<dyn Future<Output = Result<JsonRpcResponse>> + Send + 'a>> {
        Box::pin(async move {
            let body = serde_json::to_vec(request)
                .map_err(|e| DaimonError::Mcp(format!("serialize request: {e}")))?;

            let response_bytes = self.send_and_receive(&body).await?;

            let response: JsonRpcResponse = serde_json::from_slice(&response_bytes)
                .map_err(|e| DaimonError::Mcp(format!("deserialize response: {e}")))?;

            Ok(response)
        })
    }

    fn notify<'a>(
        &'a self,
        notification: &'a JsonRpcNotification,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let body = serde_json::to_vec(notification)
                .map_err(|e| DaimonError::Mcp(format!("serialize notification: {e}")))?;

            self.send_fire_and_forget(&body).await
        })
    }

    fn close<'a>(&'a self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let mut guard = self.stream.lock().await;
            if let Some(mut stream) = guard.take() {
                let _ = stream.close(None).await;
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transport_types_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<WebSocketTransport>();
    }
}
