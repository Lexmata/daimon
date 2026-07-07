//! Bridges MCP tools to the Daimon [`Tool`] trait.

use std::sync::Arc;

use crate::error::{DaimonError, Result};
use crate::mcp::protocol::{JsonRpcRequest, McpContentBlock, McpToolCallResult, McpToolInfo};
use crate::mcp::transport::McpTransport;
use crate::tool::{Tool, ToolOutput};

/// Wraps a single MCP tool as a Daimon [`Tool`].
///
/// Created by [`McpClient::tools()`](super::client::McpClient::tools). Register
/// these with an agent's builder just like any other tool.
pub struct McpToolBridge {
    transport: Arc<dyn McpTransport>,
    info: McpToolInfo,
    call_counter: std::sync::atomic::AtomicU64,
}

impl McpToolBridge {
    /// Creates a bridge for a single MCP tool.
    pub fn new(transport: Arc<dyn McpTransport>, info: McpToolInfo) -> Self {
        Self {
            transport,
            info,
            call_counter: std::sync::atomic::AtomicU64::new(1000),
        }
    }

    fn next_id(&self) -> u64 {
        self.call_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }
}

impl Tool for McpToolBridge {
    fn name(&self) -> &str {
        &self.info.name
    }

    fn description(&self) -> &str {
        self.info.description.as_deref().unwrap_or("")
    }

    fn parameters_schema(&self) -> serde_json::Value {
        if self.info.input_schema.is_null() {
            serde_json::json!({"type": "object"})
        } else {
            self.info.input_schema.clone()
        }
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<ToolOutput> {
        let request = JsonRpcRequest::new(
            self.next_id(),
            "tools/call",
            Some(serde_json::json!({
                "name": self.info.name,
                "arguments": input,
            })),
        );

        let response = self.transport.send(&request).await?;

        if let Some(err) = response.error {
            return Ok(ToolOutput::error(format!(
                "MCP error: {} (code {})",
                err.message, err.code
            )));
        }

        let result_value = response
            .result
            .ok_or_else(|| DaimonError::Mcp("tools/call returned no result".into()))?;

        let call_result: McpToolCallResult = serde_json::from_value(result_value)
            .map_err(|e| DaimonError::Mcp(format!("parse result: {e}")))?;

        let text = extract_text(&call_result.content);

        if call_result.is_error {
            Ok(ToolOutput::error(text))
        } else {
            Ok(ToolOutput::text(text))
        }
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
    fn test_bridge_name() {
        let info = McpToolInfo {
            name: "read_file".into(),
            description: Some("Reads a file".into()),
            input_schema: serde_json::json!({"type": "object"}),
        };
        let bridge = McpToolBridge::new(Arc::new(NullTransport), info);
        assert_eq!(bridge.name(), "read_file");
        assert_eq!(bridge.description(), "Reads a file");
    }

    #[test]
    fn test_bridge_schema_null_fallback() {
        let info = McpToolInfo {
            name: "test".into(),
            description: None,
            input_schema: serde_json::Value::Null,
        };
        let bridge = McpToolBridge::new(Arc::new(NullTransport), info);
        assert_eq!(
            bridge.parameters_schema(),
            serde_json::json!({"type": "object"})
        );
    }

    #[test]
    fn test_bridge_description_none() {
        let info = McpToolInfo {
            name: "test".into(),
            description: None,
            input_schema: serde_json::json!({}),
        };
        let bridge = McpToolBridge::new(Arc::new(NullTransport), info);
        assert_eq!(bridge.description(), "");
    }

    struct NullTransport;

    impl McpTransport for NullTransport {
        fn send<'a>(
            &'a self,
            _request: &'a crate::mcp::protocol::JsonRpcRequest,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<crate::mcp::protocol::JsonRpcResponse>>
                    + Send
                    + 'a,
            >,
        > {
            Box::pin(async { Err(DaimonError::Mcp("null transport".into())) })
        }

        fn notify<'a>(
            &'a self,
            _notification: &'a crate::mcp::protocol::JsonRpcNotification,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
            Box::pin(async { Ok(()) })
        }

        fn close<'a>(
            &'a self,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
            Box::pin(async { Ok(()) })
        }
    }
}
