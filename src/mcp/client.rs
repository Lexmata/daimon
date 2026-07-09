//! MCP client: connect to a server, discover tools, call them.

use std::sync::Arc;

use crate::error::{DaimonError, Result};
use crate::mcp::protocol::{JsonRpcNotification, McpContentBlock, McpToolCallResult, McpToolInfo};
use crate::mcp::transport::McpTransport;

/// Maximum number of `tools/list` pages the client will follow. A server
/// that returns a `nextCursor` on every page forever (buggy or malicious)
/// would otherwise pin `connect()` in an infinite pagination loop.
const MAX_TOOL_LIST_PAGES: usize = 1024;

/// An MCP client that communicates with an MCP server to discover and invoke tools.
pub struct McpClient {
    transport: Arc<dyn McpTransport>,
    tools: Vec<McpToolInfo>,
}

impl McpClient {
    /// Connects to an MCP server via the given transport, performs the
    /// initialization handshake, and discovers available tools.
    pub async fn connect(transport: impl McpTransport + 'static) -> Result<Self> {
        let transport: Arc<dyn McpTransport> = Arc::new(transport);

        let mut client = Self {
            transport,
            tools: Vec::new(),
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

    async fn initialize(&self) -> Result<()> {
        let response = self
            .transport
            .request(
                "initialize",
                Some(serde_json::json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "daimon",
                        "version": env!("CARGO_PKG_VERSION")
                    }
                })),
            )
            .await?;

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

    /// Fetches the full tool list, following `nextCursor` pagination until
    /// the server stops returning a cursor.
    async fn list_tools(&self) -> Result<Vec<McpToolInfo>> {
        let mut tools = Vec::new();
        let mut cursor: Option<String> = None;

        for _ in 0..MAX_TOOL_LIST_PAGES {
            let params = cursor.as_ref().map(|c| serde_json::json!({ "cursor": c }));
            let response = self.transport.request("tools/list", params).await?;

            if let Some(err) = response.error {
                return Err(DaimonError::Mcp(format!(
                    "tools/list failed: {} (code {})",
                    err.message, err.code
                )));
            }

            let result = response
                .result
                .ok_or_else(|| DaimonError::Mcp("tools/list returned no result".into()))?;

            // A result we can't parse is an error, not an empty tool list —
            // silently returning no tools masks a broken or incompatible
            // server behind an agent that mysteriously has no MCP tools.
            let page = result
                .get("tools")
                .ok_or_else(|| DaimonError::Mcp("tools/list result missing 'tools'".into()))?;
            let page: Vec<McpToolInfo> = serde_json::from_value(page.clone())
                .map_err(|e| DaimonError::Mcp(format!("parse tools/list result: {e}")))?;
            tools.extend(page);

            match result.get("nextCursor").and_then(|v| v.as_str()) {
                Some(next) => cursor = Some(next.to_string()),
                None => return Ok(tools),
            }
        }

        Err(DaimonError::Mcp(format!(
            "tools/list pagination exceeded {MAX_TOOL_LIST_PAGES} pages"
        )))
    }

    /// Calls a tool on the MCP server and returns the text output.
    pub async fn call_tool(&self, name: &str, arguments: &serde_json::Value) -> Result<String> {
        let response = self
            .transport
            .request(
                "tools/call",
                Some(serde_json::json!({
                    "name": name,
                    "arguments": arguments,
                })),
            )
            .await?;

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
    use crate::mcp::protocol::JsonRpcResponse;
    use std::collections::VecDeque;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Mutex;

    /// A transport that replays a scripted sequence of `result` payloads
    /// (one per request, in order) and records every request it receives.
    struct ScriptedTransport {
        results: Mutex<VecDeque<serde_json::Value>>,
        requests: Mutex<Vec<(String, Option<serde_json::Value>)>>,
    }

    impl ScriptedTransport {
        fn new(results: impl IntoIterator<Item = serde_json::Value>) -> Self {
            Self {
                results: Mutex::new(results.into_iter().collect()),
                requests: Mutex::new(Vec::new()),
            }
        }
    }

    impl McpTransport for ScriptedTransport {
        fn request<'a>(
            &'a self,
            method: &'a str,
            params: Option<serde_json::Value>,
        ) -> Pin<Box<dyn Future<Output = Result<JsonRpcResponse>> + Send + 'a>> {
            Box::pin(async move {
                self.requests
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push((method.to_string(), params));
                let result = self
                    .results
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .pop_front()
                    .ok_or_else(|| DaimonError::Mcp("scripted transport exhausted".into()))?;
                Ok(JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: Some(1),
                    result: Some(result),
                    error: None,
                })
            })
        }

        fn notify<'a>(
            &'a self,
            _notification: &'a JsonRpcNotification,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
            Box::pin(async { Ok(()) })
        }

        fn close<'a>(&'a self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
            Box::pin(async { Ok(()) })
        }
    }

    fn init_result() -> serde_json::Value {
        serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "test", "version": "0.0.0"}
        })
    }

    #[tokio::test]
    async fn test_list_tools_follows_next_cursor_pagination() {
        let transport = ScriptedTransport::new([
            init_result(),
            serde_json::json!({
                "tools": [{"name": "tool_one", "inputSchema": {"type": "object"}}],
                "nextCursor": "page-2"
            }),
            serde_json::json!({
                "tools": [{"name": "tool_two", "inputSchema": {"type": "object"}}]
            }),
        ]);

        let client = McpClient::connect(transport).await.unwrap();

        let names: Vec<&str> = client
            .tool_infos()
            .iter()
            .map(|t| t.name.as_str())
            .collect();
        assert_eq!(names, ["tool_one", "tool_two"]);
    }

    #[tokio::test]
    async fn test_list_tools_passes_cursor_param_on_second_page() {
        let transport = Arc::new(ScriptedTransport::new([
            init_result(),
            serde_json::json!({"tools": [], "nextCursor": "abc"}),
            serde_json::json!({"tools": []}),
        ]));

        // McpClient::connect takes ownership; drive list_tools through a
        // client built around a second handle to the same script so the
        // recorded requests stay inspectable.
        struct SharedTransport(Arc<ScriptedTransport>);
        impl McpTransport for SharedTransport {
            fn request<'a>(
                &'a self,
                method: &'a str,
                params: Option<serde_json::Value>,
            ) -> Pin<Box<dyn Future<Output = Result<JsonRpcResponse>> + Send + 'a>> {
                self.0.request(method, params)
            }
            fn notify<'a>(
                &'a self,
                notification: &'a JsonRpcNotification,
            ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
                self.0.notify(notification)
            }
            fn close<'a>(&'a self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
                self.0.close()
            }
        }

        let _client = McpClient::connect(SharedTransport(transport.clone()))
            .await
            .unwrap();

        let requests = transport.requests.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(requests[0].0, "initialize");
        assert_eq!(requests[1], ("tools/list".to_string(), None));
        assert_eq!(
            requests[2],
            (
                "tools/list".to_string(),
                Some(serde_json::json!({"cursor": "abc"}))
            )
        );
    }

    #[tokio::test]
    async fn test_list_tools_malformed_result_is_an_error_not_empty() {
        // Historically `.ok().unwrap_or_default()` swallowed this into a
        // silently empty tool list.
        let transport =
            ScriptedTransport::new([init_result(), serde_json::json!({"tools": "not an array"})]);

        let err = match McpClient::connect(transport).await {
            Ok(_) => panic!("connect succeeded on a malformed tools/list result"),
            Err(e) => e,
        };
        assert!(matches!(err, DaimonError::Mcp(_)));
        assert!(err.to_string().contains("tools/list"));
    }

    #[tokio::test]
    async fn test_list_tools_missing_tools_key_is_an_error() {
        let transport = ScriptedTransport::new([init_result(), serde_json::json!({})]);

        let err = match McpClient::connect(transport).await {
            Ok(_) => panic!("connect succeeded on a tools/list result missing 'tools'"),
            Err(e) => e,
        };
        assert!(matches!(err, DaimonError::Mcp(_)));
    }

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
