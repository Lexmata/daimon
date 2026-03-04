//! WebSocket MCP server: expose Daimon tools over WebSocket connections.
//!
//! [`McpWsServer`] listens on a TCP port and accepts WebSocket connections,
//! serving JSON-RPC 2.0 requests the same way [`McpServer`](super::McpServer)
//! does over stdio.
//!
//! ```ignore
//! use daimon::mcp::McpWsServer;
//! use daimon::tool::ToolRegistry;
//!
//! let mut registry = ToolRegistry::new();
//! registry.register(my_tool)?;
//!
//! McpWsServer::new(registry)
//!     .serve("127.0.0.1:9090")
//!     .await?;
//! ```

use std::sync::Arc;

use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::Message as WsMessage;

use crate::error::{DaimonError, Result};
use crate::mcp::server::McpServer;
use crate::tool::ToolRegistry;

/// Serves MCP tools over WebSocket connections.
///
/// Wraps a [`McpServer`] and accepts incoming WebSocket connections on a
/// given TCP address. Each connection processes JSON-RPC requests independently,
/// allowing multiple clients to connect simultaneously.
pub struct McpWsServer {
    inner: Arc<McpServer>,
}

impl McpWsServer {
    /// Creates a new WebSocket MCP server wrapping the given tool registry.
    pub fn new(tools: ToolRegistry) -> Self {
        Self {
            inner: Arc::new(McpServer::new(tools)),
        }
    }

    /// Creates a WebSocket server from an existing `McpServer`.
    pub fn from_server(server: McpServer) -> Self {
        Self {
            inner: Arc::new(server),
        }
    }

    /// Listens on the given address and serves WebSocket connections
    /// indefinitely. Each connection is handled in a separate task.
    pub async fn serve(self, addr: impl tokio::net::ToSocketAddrs) -> Result<()> {
        let listener = TcpListener::bind(addr)
            .await
            .map_err(|e| DaimonError::Mcp(format!("bind failed: {e}")))?;

        tracing::info!(
            addr = %listener.local_addr().unwrap_or_else(|_| "unknown".parse().unwrap()),
            "MCP WebSocket server listening"
        );

        loop {
            let (stream, peer) = listener
                .accept()
                .await
                .map_err(|e| DaimonError::Mcp(format!("accept failed: {e}")))?;

            let server = Arc::clone(&self.inner);
            tokio::spawn(async move {
                if let Err(e) = handle_connection(server, stream).await {
                    tracing::warn!(%peer, error = %e, "WebSocket connection error");
                }
            });
        }
    }

    /// Binds, accepts one connection, handles it, and returns.
    /// Useful for testing.
    pub async fn serve_one(self, addr: impl tokio::net::ToSocketAddrs) -> Result<()> {
        let listener = TcpListener::bind(addr)
            .await
            .map_err(|e| DaimonError::Mcp(format!("bind failed: {e}")))?;

        let (stream, _peer) = listener
            .accept()
            .await
            .map_err(|e| DaimonError::Mcp(format!("accept failed: {e}")))?;

        handle_connection(Arc::clone(&self.inner), stream).await
    }
}

async fn handle_connection(server: Arc<McpServer>, stream: TcpStream) -> Result<()> {
    use futures::{SinkExt, StreamExt};

    let ws_stream = tokio_tungstenite::accept_async(stream)
        .await
        .map_err(|e| DaimonError::Mcp(format!("websocket handshake: {e}")))?;

    let (mut sink, mut source) = ws_stream.split();

    while let Some(msg_result) = source.next().await {
        let msg = match msg_result {
            Ok(m) => m,
            Err(e) => {
                tracing::debug!("ws read error: {e}");
                break;
            }
        };

        let body = match &msg {
            WsMessage::Text(text) => text.to_string(),
            WsMessage::Binary(data) => {
                String::from_utf8(data.to_vec())
                    .map_err(|e| DaimonError::Mcp(format!("invalid utf-8: {e}")))?
            }
            WsMessage::Close(_) => break,
            WsMessage::Ping(_) | WsMessage::Pong(_) | WsMessage::Frame(_) => continue,
        };

        let response = match server.handle_request_raw(&body).await {
            Ok(r) => r,
            Err(e) => {
                let err_response = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": null,
                    "error": { "code": -32603, "message": e }
                });
                serde_json::to_string(&err_response).unwrap_or_default()
            }
        };

        sink.send(WsMessage::Text(response.into()))
            .await
            .map_err(|e| DaimonError::Mcp(format!("ws send: {e}")))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{Tool, ToolOutput};

    struct PingTool;

    impl Tool for PingTool {
        fn name(&self) -> &str {
            "ping"
        }
        fn description(&self) -> &str {
            "Returns pong"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(&self, _input: &serde_json::Value) -> crate::error::Result<ToolOutput> {
            Ok(ToolOutput::text("pong"))
        }
    }

    fn make_server() -> McpWsServer {
        let mut registry = ToolRegistry::new();
        registry.register(PingTool).unwrap();
        McpWsServer::new(registry)
    }

    #[tokio::test]
    async fn test_ws_server_initialize_and_call() {
        use futures::{SinkExt, StreamExt};
        use tokio_tungstenite::tungstenite::Message as WsMsg;

        let server = make_server();

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let server_handle = tokio::spawn(async move {
            server.serve_one(addr).await.unwrap();
        });

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let url = format!("ws://{addr}");
        let (ws_stream, _) = tokio_tungstenite::connect_async(&url).await.unwrap();
        let (mut sink, mut source) = ws_stream.split();

        let init_req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        });
        sink.send(WsMsg::Text(init_req.to_string().into()))
            .await
            .unwrap();

        let resp = source.next().await.unwrap().unwrap();
        let body: serde_json::Value =
            serde_json::from_str(&resp.into_text().unwrap()).unwrap();
        assert!(body["result"]["capabilities"]["tools"].is_object());

        let call_req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": { "name": "ping", "arguments": {} }
        });
        sink.send(WsMsg::Text(call_req.to_string().into()))
            .await
            .unwrap();

        let resp = source.next().await.unwrap().unwrap();
        let body: serde_json::Value =
            serde_json::from_str(&resp.into_text().unwrap()).unwrap();
        assert_eq!(body["result"]["content"][0]["text"], "pong");

        sink.send(WsMsg::Close(None)).await.unwrap();

        let _ = server_handle.await;
    }

    #[tokio::test]
    async fn test_ws_server_tools_list() {
        use futures::{SinkExt, StreamExt};
        use tokio_tungstenite::tungstenite::Message as WsMsg;

        let server = make_server();

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let server_handle = tokio::spawn(async move {
            server.serve_one(addr).await.unwrap();
        });

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let url = format!("ws://{addr}");
        let (ws_stream, _) = tokio_tungstenite::connect_async(&url).await.unwrap();
        let (mut sink, mut source) = ws_stream.split();

        let list_req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list"
        });
        sink.send(WsMsg::Text(list_req.to_string().into()))
            .await
            .unwrap();

        let resp = source.next().await.unwrap().unwrap();
        let body: serde_json::Value =
            serde_json::from_str(&resp.into_text().unwrap()).unwrap();
        let tools = body["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "ping");

        sink.send(WsMsg::Close(None)).await.unwrap();

        let _ = server_handle.await;
    }
}
