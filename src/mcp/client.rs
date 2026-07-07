//! MCP client: connect to a server, discover tools, call them.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{DaimonError, Result};
use crate::mcp::protocol::{
    JsonRpcNotification, JsonRpcRequest, McpContentBlock, McpToolCallResult, McpToolInfo,
};
use crate::mcp::transport::McpTransport;

/// An MCP client that communicates with an MCP server to discover and invoke tools.
pub struct McpClient {
    transport: Arc<dyn McpTransport>,
    tools: Vec<McpToolInfo>,
    next_id: AtomicU64,
}

impl McpClient {
    /// Connects to an MCP server via the given transport, performs the
    /// initialization handshake, and discovers available tools.
    pub async fn connect(transport: impl McpTransport + 'static) -> Result<Self> {
        let transport: Arc<dyn McpTransport> = Arc::new(transport);

        let mut client = Self {
            transport,
            tools: Vec::new(),
            next_id: AtomicU64::new(1),
        };

        client.initialize().await?;
        client.tools = client.list_tools().await?;

        Ok(client)
    }

    /// Returns the tools discovered from the MCP server.
    pub fn tool_infos(&self) -> &[McpToolInfo] {
        &self.tools
    }

    /// Creates [`McpToolBridge`](super::bridge::McpToolBridge) instances for all
    /// discovered tools. These implement [`Tool`](crate::tool::Tool) and can
    /// be registered with an agent.
    pub fn tools(&self) -> Vec<super::bridge::McpToolBridge> {
        self.tools
            .iter()
            .map(|info| super::bridge::McpToolBridge::new(self.transport.clone(), info.clone()))
            .collect()
    }

    fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    async fn initialize(&self) -> Result<()> {
        let request = JsonRpcRequest::new(
            self.next_id(),
            "initialize",
            Some(serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "daimon",
                    "version": env!("CARGO_PKG_VERSION")
                }
            })),
        );

        let response = self.transport.send(&request).await?;

        if let Some(err) = response.error {
            return Err(DaimonError::Mcp(format!(
                "initialize failed: {} (code {})",
                err.message, err.code
            )));
        }

        let notification = JsonRpcNotification::new("notifications/initialized");
        self.transport.notify(&notification).await?;

        Ok(())
    }

    async fn list_tools(&self) -> Result<Vec<McpToolInfo>> {
        let request = JsonRpcRequest::new(self.next_id(), "tools/list", None);
        let response = self.transport.send(&request).await?;

        if let Some(err) = response.error {
            return Err(DaimonError::Mcp(format!(
                "tools/list failed: {} (code {})",
                err.message, err.code
            )));
        }

        let result = response
            .result
            .ok_or_else(|| DaimonError::Mcp("tools/list returned no result".into()))?;

        let tools: Vec<McpToolInfo> = result
            .get("tools")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        Ok(tools)
    }

    /// Calls a tool on the MCP server and returns the text output.
    pub async fn call_tool(&self, name: &str, arguments: &serde_json::Value) -> Result<String> {
        let request = JsonRpcRequest::new(
            self.next_id(),
            "tools/call",
            Some(serde_json::json!({
                "name": name,
                "arguments": arguments,
            })),
        );

        let response = self.transport.send(&request).await?;

        if let Some(err) = response.error {
            return Err(DaimonError::Mcp(format!(
                "tools/call '{}' failed: {} (code {})",
                name, err.message, err.code
            )));
        }

        let result = response
            .result
            .ok_or_else(|| DaimonError::Mcp(format!("tools/call '{name}' returned no result")))?;

        let call_result: McpToolCallResult = serde_json::from_value(result)
            .map_err(|e| DaimonError::Mcp(format!("parse tools/call result: {e}")))?;

        if call_result.is_error {
            let text = extract_text(&call_result.content);
            return Err(DaimonError::Mcp(format!(
                "MCP tool '{name}' returned error: {text}"
            )));
        }

        Ok(extract_text(&call_result.content))
    }

    /// Closes the connection to the MCP server.
    pub async fn close(&self) -> Result<()> {
        self.transport.close().await
    }
}

fn extract_text(blocks: &[McpContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|b| {
            if b.content_type == "text" {
                b.text.clone()
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_text_single() {
        let blocks = vec![McpContentBlock {
            content_type: "text".into(),
            text: Some("hello".into()),
        }];
        assert_eq!(extract_text(&blocks), "hello");
    }

    #[test]
    fn test_extract_text_multiple() {
        let blocks = vec![
            McpContentBlock {
                content_type: "text".into(),
                text: Some("a".into()),
            },
            McpContentBlock {
                content_type: "image".into(),
                text: None,
            },
            McpContentBlock {
                content_type: "text".into(),
                text: Some("b".into()),
            },
        ];
        assert_eq!(extract_text(&blocks), "a\nb");
    }

    #[test]
    fn test_extract_text_empty() {
        assert_eq!(extract_text(&[]), "");
    }
}
