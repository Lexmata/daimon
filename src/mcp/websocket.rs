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
use std::sync::atomic::{AtomicU64, Ordering};

use futures::{SinkExt, StreamExt};
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::Message as WsMessage;

use crate::error::{DaimonError, Result};
use crate::mcp::protocol::{JsonRpcNotification, JsonRpcRequest, JsonRpcResponse};
use crate::mcp::transport::{MAX_SKIPPED_MESSAGES, McpTransport};

type WsStream =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// WebSocket transport for MCP communication.
///
/// Maintains a persistent WebSocket connection. Requests are serialized and
/// sent as text frames; the response is the next text frame whose JSON-RPC
/// `id` matches the request — interleaved server notifications (which carry
/// no `id`) are skipped rather than misread as the response.
///
/// Thread-safe: all access to the underlying stream is serialized via
/// an internal mutex.
pub struct WebSocketTransport {
    stream: Arc<Mutex<Option<WsStream>>>,
    next_id: AtomicU64,
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
            next_id: AtomicU64::new(1),
        })
    }

    /// Sends `body` and reads frames until a JSON-RPC response whose `id`
    /// matches `expected_id` arrives. Notifications and other non-matching
    /// messages are skipped (logged at debug), bounded by
    /// [`MAX_SKIPPED_MESSAGES`] so a notification-flooding server can't pin
    /// this loop forever.
    async fn send_and_receive(&self, body: &[u8], expected_id: u64) -> Result<JsonRpcResponse> {
        let mut guard = self.stream.lock().await;
        let stream = guard
            .as_mut()
            .ok_or_else(|| DaimonError::Mcp("WebSocket transport closed".into()))?;

        let text = String::from_utf8_lossy(body).into_owned();
        stream
            .send(WsMessage::Text(text.into()))
            .await
            .map_err(|e| DaimonError::Mcp(format!("WebSocket send failed: {e}")))?;

        let mut skipped = 0usize;
        loop {
            match stream.next().await {
                Some(Ok(WsMessage::Text(text))) => {
                    let response: JsonRpcResponse = serde_json::from_str(&text)
                        .map_err(|e| DaimonError::Mcp(format!("deserialize response: {e}")))?;
                    match response.id {
                        Some(id) if id == expected_id => return Ok(response),
                        Some(id) => {
                            tracing::debug!(
                                expected = expected_id,
                                received = id,
                                "skipping WebSocket JSON-RPC message with non-matching id"
                            );
                        }
                        None => {
                            tracing::debug!(
                                expected = expected_id,
                                "skipping WebSocket JSON-RPC notification while awaiting response"
                            );
                        }
                    }
                    skipped += 1;
                    if skipped >= MAX_SKIPPED_MESSAGES {
                        return Err(DaimonError::Mcp(format!(
                            "no response for request id {expected_id} within {MAX_SKIPPED_MESSAGES} messages"
                        )));
                    }
                }
                Some(Ok(WsMessage::Ping(data))) => {
                    stream
                        .send(WsMessage::Pong(data))
                        .await
                        .map_err(|e| DaimonError::Mcp(format!("WebSocket pong failed: {e}")))?;
                }
                Some(Ok(WsMessage::Close(_))) => {
                    return Err(DaimonError::Mcp(
                        "WebSocket server closed connection".into(),
                    ));
                }
                Some(Ok(_)) => {}
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
    fn request<'a>(
        &'a self,
        method: &'a str,
        params: Option<serde_json::Value>,
    ) -> Pin<Box<dyn Future<Output = Result<JsonRpcResponse>> + Send + 'a>> {
        Box::pin(async move {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            let request = JsonRpcRequest::new(id, method, params);
            let body = serde_json::to_vec(&request)
                .map_err(|e| DaimonError::Mcp(format!("serialize request: {e}")))?;

            self.send_and_receive(&body, id).await
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
    use tokio::net::TcpListener;

    #[test]
    fn test_transport_types_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<WebSocketTransport>();
    }

    /// Starts a one-connection WebSocket server that, on receiving a text
    /// frame, replies with each of `replies` in order (with `{id}` in a
    /// reply substituted by the incoming request's JSON-RPC id).
    async fn start_ws_server(replies: Vec<&'static str>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (conn, _) = listener.accept().await.unwrap();
            let mut ws = tokio_tungstenite::accept_async(conn).await.unwrap();
            while let Some(Ok(msg)) = ws.next().await {
                if let WsMessage::Text(text) = msg {
                    let id = serde_json::from_str::<serde_json::Value>(&text)
                        .ok()
                        .and_then(|v| v.get("id").and_then(|i| i.as_u64()))
                        .unwrap_or(0);
                    for reply in &replies {
                        let body = reply.replace("{id}", &id.to_string());
                        ws.send(WsMessage::Text(body.into())).await.unwrap();
                    }
                    break;
                }
            }
        });
        format!("ws://{addr}")
    }

    #[tokio::test]
    async fn test_notification_before_response_is_skipped() {
        // The server interleaves a notification (no id) before the actual
        // response; the transport must skip it and return the response whose
        // id matches the request, not hand back the notification body.
        let url = start_ws_server(vec![
            r#"{"jsonrpc":"2.0","method":"notifications/progress","params":{"p":1}}"#,
            r#"{"jsonrpc":"2.0","id":{id},"result":{"ok":true}}"#,
        ])
        .await;

        let transport = WebSocketTransport::connect(&url).await.unwrap();
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            transport.request("tools/list", None),
        )
        .await
        .expect("request timed out (notification was not skipped)")
        .unwrap();

        assert_eq!(response.result.unwrap()["ok"], true);
        transport.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_non_matching_id_before_response_is_skipped() {
        let url = start_ws_server(vec![
            r#"{"jsonrpc":"2.0","id":999999,"result":{"ok":false}}"#,
            r#"{"jsonrpc":"2.0","id":{id},"result":{"ok":true}}"#,
        ])
        .await;

        let transport = WebSocketTransport::connect(&url).await.unwrap();
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            transport.request("tools/list", None),
        )
        .await
        .expect("request timed out")
        .unwrap();

        assert_eq!(response.result.unwrap()["ok"], true);
        transport.close().await.unwrap();
    }
}
