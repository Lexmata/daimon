//! JSON-RPC 2.0 types for the MCP protocol.

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: u64,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl JsonRpcRequest {
    pub fn new(id: u64, method: impl Into<String>, params: Option<serde_json::Value>) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            method: method.into(),
            params,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct JsonRpcResponse {
    #[allow(dead_code)]
    pub jsonrpc: String,
    /// The request id this response answers, if any. Outgoing requests always
    /// use numeric ids, but the JSON-RPC spec also allows string ids and some
    /// servers echo the id back as a string (e.g. `"1000"`), so both forms
    /// are accepted and normalized to `u64` for correlation. A string that is
    /// not a valid `u64` cannot match any id we issued and maps to `None`.
    #[serde(default, deserialize_with = "deserialize_response_id")]
    pub id: Option<u64>,
    pub result: Option<serde_json::Value>,
    pub error: Option<JsonRpcError>,
}

fn deserialize_response_id<'de, D>(deserializer: D) -> std::result::Result<Option<u64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(match value {
        Some(serde_json::Value::Number(n)) => n.as_u64(),
        Some(serde_json::Value::String(s)) => s.parse().ok(),
        _ => None,
    })
}

#[derive(Debug, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[allow(dead_code)]
    pub data: Option<serde_json::Value>,
}

/// Notification (no id, no response expected).
#[derive(Debug, Serialize)]
pub struct JsonRpcNotification {
    pub jsonrpc: String,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl JsonRpcNotification {
    pub fn new(method: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            method: method.into(),
            params: None,
        }
    }
}

/// MCP tool definition returned by `tools/list`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpToolInfo {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default, rename = "inputSchema")]
    pub input_schema: serde_json::Value,
}

/// MCP tool call result content block.
#[derive(Debug, Deserialize)]
pub struct McpContentBlock {
    #[serde(rename = "type")]
    pub content_type: String,
    #[serde(default)]
    pub text: Option<String>,
}

/// MCP `tools/call` result.
#[derive(Debug, Deserialize)]
pub struct McpToolCallResult {
    pub content: Vec<McpContentBlock>,
    #[serde(default, rename = "isError")]
    pub is_error: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_serialization() {
        let req = JsonRpcRequest::new(1, "tools/list", None);
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"method\":\"tools/list\""));
        assert!(!json.contains("\"params\""));
    }

    #[test]
    fn test_request_with_params() {
        let req = JsonRpcRequest::new(
            2,
            "tools/call",
            Some(serde_json::json!({"name": "test", "arguments": {}})),
        );
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"params\""));
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json).unwrap();
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_error_response() {
        let json =
            r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32600,"message":"Invalid Request"}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json).unwrap();
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32600);
    }

    #[test]
    fn test_tool_info_deserialization() {
        let json = r#"{"name":"read_file","description":"Read a file","inputSchema":{"type":"object","properties":{"path":{"type":"string"}}}}"#;
        let info: McpToolInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info.name, "read_file");
        assert_eq!(info.description.as_deref(), Some("Read a file"));
    }

    #[test]
    fn test_response_string_id_normalized_to_u64() {
        // Some servers echo a numeric request id back as a string; it must
        // still correlate with the u64 id the client issued.
        let json = r#"{"jsonrpc":"2.0","id":"1000","result":{}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, Some(1000));
    }

    #[test]
    fn test_response_non_numeric_string_id_maps_to_none() {
        // A string id that isn't a u64 can't match any id we issued.
        let json = r#"{"jsonrpc":"2.0","id":"abc","result":{}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, None);
    }

    #[test]
    fn test_response_missing_and_null_id_map_to_none() {
        let json = r#"{"jsonrpc":"2.0","result":{}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, None);

        let json = r#"{"jsonrpc":"2.0","id":null,"result":{}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, None);
    }

    #[test]
    fn test_notification() {
        let n = JsonRpcNotification::new("notifications/initialized");
        let json = serde_json::to_string(&n).unwrap();
        assert!(json.contains("\"method\":\"notifications/initialized\""));
        assert!(!json.contains("\"id\""));
    }
}
