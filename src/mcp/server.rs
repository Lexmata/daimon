//! MCP server: expose Daimon tools over the Model Context Protocol.
//!
//! [`McpServer`] reads JSON-RPC 2.0 requests from stdin, dispatches
//! `initialize`, `tools/list`, and `tools/call` to a [`ToolRegistry`],
//! and writes responses to stdout.
//!
//! ```ignore
//! use daimon::mcp::server::McpServer;
//! use daimon::tool::ToolRegistry;
//!
//! let mut registry = ToolRegistry::new();
//! registry.register(my_tool)?;
//!
//! McpServer::new(registry).serve_stdio().await?;
//! ```

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

use crate::error::{DaimonError, Result};
use crate::mcp::protocol::McpToolInfo;
use crate::tool::ToolRegistry;

/// Maximum size (in bytes) of a single framed request body the server will
/// accept. The Content-Length header comes from the (untrusted) client; without
/// a cap a bogus length forces a huge up-front allocation (OOM DoS). 32 MiB is
/// well above any legitimate MCP request.
const MAX_MESSAGE_SIZE: usize = 32 * 1024 * 1024;

/// An MCP-compliant tool server.
///
/// Exposes a [`ToolRegistry`] over JSON-RPC 2.0 using Content-Length framed
/// messages on stdin/stdout (the standard MCP stdio transport).
pub struct McpServer {
    tools: ToolRegistry,
    server_name: String,
    server_version: String,
}

#[derive(Debug, Deserialize)]
struct IncomingRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<serde_json::Value>,
    method: String,
    #[serde(default)]
    params: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct OutgoingResponse {
    jsonrpc: String,
    id: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<RpcError>,
}

#[derive(Debug, Serialize)]
struct RpcError {
    code: i64,
    message: String,
}

impl McpServer {
    /// Creates a new MCP server wrapping the given tool registry.
    pub fn new(tools: ToolRegistry) -> Self {
        Self {
            tools,
            server_name: "daimon".to_string(),
            server_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Sets the server name reported during initialization.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.server_name = name.into();
        self
    }

    /// Sets the server version reported during initialization.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.server_version = version.into();
        self
    }

    /// Runs the server, reading from stdin and writing to stdout using
    /// Content-Length framed JSON-RPC messages.
    pub async fn serve_stdio(self) -> Result<()> {
        let stdin = tokio::io::stdin();
        let stdout = tokio::io::stdout();
        let mut reader = tokio::io::BufReader::new(stdin);
        let mut writer = tokio::io::BufWriter::new(stdout);

        loop {
            let body = match read_message(&mut reader).await {
                Ok(Some(body)) => body,
                Ok(None) => break,
                Err(e) => {
                    tracing::warn!("failed to read message: {e}");
                    continue;
                }
            };

            let request: IncomingRequest = match serde_json::from_str(&body) {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!("invalid JSON-RPC: {e}");
                    continue;
                }
            };

            if request.id.is_none() {
                tracing::debug!(method = %request.method, "notification (ignored)");
                continue;
            }

            let id = request.id.unwrap();
            let response = self.handle_request(&request.method, request.params).await;

            let out = match response {
                Ok(result) => OutgoingResponse {
                    jsonrpc: "2.0".into(),
                    id,
                    result: Some(result),
                    error: None,
                },
                Err((code, msg)) => OutgoingResponse {
                    jsonrpc: "2.0".into(),
                    id,
                    result: None,
                    error: Some(RpcError { code, message: msg }),
                },
            };

            let body = serde_json::to_string(&out)
                .map_err(|e| DaimonError::Mcp(format!("serialize response: {e}")))?;

            write_message(&mut writer, &body)
                .await
                .map_err(|e| DaimonError::Mcp(format!("write response: {e}")))?;
        }

        Ok(())
    }

    /// Process a synchronous request from an in-memory buffer (for testing or embedding).
    pub async fn handle_request_raw(&self, body: &str) -> std::result::Result<String, String> {
        let request: IncomingRequest =
            serde_json::from_str(body).map_err(|e| format!("parse error: {e}"))?;

        let id = request.id.clone().unwrap_or(serde_json::Value::Null);

        let out = match self.handle_request(&request.method, request.params).await {
            Ok(result) => OutgoingResponse {
                jsonrpc: "2.0".into(),
                id,
                result: Some(result),
                error: None,
            },
            Err((code, msg)) => OutgoingResponse {
                jsonrpc: "2.0".into(),
                id,
                result: None,
                error: Some(RpcError { code, message: msg }),
            },
        };

        serde_json::to_string(&out).map_err(|e| format!("serialize: {e}"))
    }

    async fn handle_request(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> std::result::Result<serde_json::Value, (i64, String)> {
        match method {
            "initialize" => self.handle_initialize(),
            "tools/list" => self.handle_tools_list(),
            "tools/call" => self.handle_tools_call(params).await,
            _ => Err((-32601, format!("method '{method}' not found"))),
        }
    }

    fn handle_initialize(&self) -> std::result::Result<serde_json::Value, (i64, String)> {
        Ok(serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": self.server_name,
                "version": self.server_version
            }
        }))
    }

    fn handle_tools_list(&self) -> std::result::Result<serde_json::Value, (i64, String)> {
        let tools: Vec<McpToolInfo> = self
            .tools
            .tool_specs()
            .iter()
            .map(|spec| McpToolInfo {
                name: spec.name.clone(),
                description: Some(spec.description.clone()),
                input_schema: spec.parameters.clone(),
            })
            .collect();

        Ok(serde_json::json!({ "tools": tools }))
    }

    async fn handle_tools_call(
        &self,
        params: Option<serde_json::Value>,
    ) -> std::result::Result<serde_json::Value, (i64, String)> {
        let params = params.ok_or((-32602, "missing params".to_string()))?;

        let name = params
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or((-32602, "missing 'name' in params".to_string()))?;

        let arguments = params
            .get("arguments")
            .cloned()
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

        let tool = self
            .tools
            .get(name)
            .ok_or((-32602, format!("tool '{name}' not found")))?;

        match tool.execute_erased(&arguments).await {
            Ok(output) => Ok(serde_json::json!({
                "content": [{
                    "type": "text",
                    "text": output.content
                }],
                "isError": output.is_error
            })),
            Err(e) => Ok(serde_json::json!({
                "content": [{
                    "type": "text",
                    "text": e.to_string()
                }],
                "isError": true
            })),
        }
    }
}

async fn read_message<R: tokio::io::AsyncBufRead + Unpin>(
    reader: &mut R,
) -> std::result::Result<Option<String>, std::io::Error> {
    let mut header_line = String::new();
    let n = reader.read_line(&mut header_line).await?;
    if n == 0 {
        return Ok(None);
    }

    let content_length: usize = if header_line
        .trim()
        .to_lowercase()
        .starts_with("content-length:")
    {
        header_line
            .trim()
            .split(':')
            .nth(1)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
    } else {
        let content = header_line.trim().to_string();
        if content.is_empty() {
            return Ok(None);
        }
        return Ok(Some(content));
    };

    if content_length > MAX_MESSAGE_SIZE {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Content-Length {content_length} exceeds maximum allowed message size {MAX_MESSAGE_SIZE}"
            ),
        ));
    }

    let mut separator = String::new();
    reader.read_line(&mut separator).await?;

    let mut body = vec![0u8; content_length];
    tokio::io::AsyncReadExt::read_exact(reader, &mut body).await?;

    Ok(Some(String::from_utf8_lossy(&body).into_owned()))
}

async fn write_message<W: tokio::io::AsyncWrite + Unpin>(
    writer: &mut W,
    body: &str,
) -> std::result::Result<(), std::io::Error> {
    let header = format!("Content-Length: {}\r\n\r\n", body.len());
    writer.write_all(header.as_bytes()).await?;
    writer.write_all(body.as_bytes()).await?;
    writer.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{Tool, ToolOutput};

    struct GreetTool;

    impl Tool for GreetTool {
        fn name(&self) -> &str {
            "greet"
        }
        fn description(&self) -> &str {
            "Greets a person"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                },
                "required": ["name"]
            })
        }
        async fn execute(&self, input: &serde_json::Value) -> crate::error::Result<ToolOutput> {
            let name = input
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("World");
            Ok(ToolOutput::text(format!("Hello, {name}!")))
        }
    }

    fn make_server() -> McpServer {
        let mut registry = ToolRegistry::new();
        registry.register(GreetTool).unwrap();
        McpServer::new(registry)
    }

    #[tokio::test]
    async fn test_initialize() {
        let server = make_server();
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        })
        .to_string();

        let resp = server.handle_request_raw(&body).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert!(parsed["result"]["capabilities"]["tools"].is_object());
        assert_eq!(parsed["result"]["serverInfo"]["name"], "daimon");
    }

    #[tokio::test]
    async fn test_tools_list() {
        let server = make_server();
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        })
        .to_string();

        let resp = server.handle_request_raw(&body).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&resp).unwrap();
        let tools = parsed["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "greet");
    }

    #[tokio::test]
    async fn test_tools_call() {
        let server = make_server();
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "greet",
                "arguments": {"name": "Daimon"}
            }
        })
        .to_string();

        let resp = server.handle_request_raw(&body).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(parsed["result"]["content"][0]["text"], "Hello, Daimon!");
        assert_eq!(parsed["result"]["isError"], false);
    }

    #[tokio::test]
    async fn test_tools_call_unknown_tool() {
        let server = make_server();
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "nonexistent",
                "arguments": {}
            }
        })
        .to_string();

        let resp = server.handle_request_raw(&body).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert!(parsed["error"].is_object());
    }

    #[tokio::test]
    async fn test_read_message_rejects_oversized_content_length() {
        // A client that advertises a length beyond MAX_MESSAGE_SIZE must be
        // rejected before we allocate the buffer.
        let framed = format!("Content-Length: {}\r\n\r\n", MAX_MESSAGE_SIZE + 1);
        let mut reader = tokio::io::BufReader::new(framed.as_bytes());
        let err = read_message(&mut reader).await.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[tokio::test]
    async fn test_read_message_accepts_normal_content_length() {
        let payload = "{}";
        let framed = format!("Content-Length: {}\r\n\r\n{payload}", payload.len());
        let mut reader = tokio::io::BufReader::new(framed.as_bytes());
        let body = read_message(&mut reader).await.unwrap();
        assert_eq!(body.as_deref(), Some("{}"));
    }

    #[tokio::test]
    async fn test_unknown_method() {
        let server = make_server();
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "unknown/method"
        })
        .to_string();

        let resp = server.handle_request_raw(&body).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(parsed["error"]["code"], -32601);
    }
}
