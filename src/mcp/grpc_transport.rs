//! gRPC transport for MCP: serve and consume MCP tools over gRPC.
//!
//! [`McpGrpcServer`] wraps an [`McpServer`] and exposes
//! it as a gRPC service. [`McpGrpcTransport`] implements [`McpTransport`]
//! by connecting to a remote gRPC MCP server.
//!
//! Requires both `grpc` and `mcp` features.
//!
//! ## Server
//!
//! ```ignore
//! use daimon::mcp::{McpServer, McpGrpcServer};
//!
//! let server = McpServer::new(registry);
//! McpGrpcServer::new(server).serve("[::1]:50052").await?;
//! ```
//!
//! ## Client
//!
//! ```ignore
//! use daimon::mcp::{McpClient, McpGrpcTransport};
//!
//! let transport = McpGrpcTransport::connect("http://[::1]:50052").await?;
//! let client = McpClient::connect_with(transport).await?;
//! ```

use std::pin::Pin;
use std::sync::Arc;

use tonic::{Request, Response, Status};

use crate::error::{DaimonError, Result};
use crate::mcp::protocol::{JsonRpcNotification, JsonRpcRequest, JsonRpcResponse};
use crate::mcp::server::McpServer;
use crate::mcp::transport::McpTransport;

pub mod proto {
    tonic::include_proto!("daimon.mcp");
}

use proto::mcp_service_client::McpServiceClient;
use proto::mcp_service_server::{McpService, McpServiceServer};

/// Wraps an [`McpServer`] and serves MCP tools over gRPC.
pub struct McpGrpcServer {
    inner: Arc<McpServer>,
}

impl McpGrpcServer {
    /// Creates a gRPC MCP server from an existing `McpServer`.
    pub fn new(server: McpServer) -> Self {
        Self {
            inner: Arc::new(server),
        }
    }

    /// Starts the gRPC server on the given address.
    pub async fn serve(self, addr: impl Into<String>) -> Result<()> {
        let addr = addr
            .into()
            .parse()
            .map_err(|e| DaimonError::Mcp(format!("invalid address: {e}")))?;

        let svc = McpGrpcSvc { server: self.inner };

        tonic::transport::Server::builder()
            .add_service(McpServiceServer::new(svc))
            .serve(addr)
            .await
            .map_err(|e| DaimonError::Mcp(format!("grpc mcp server: {e}")))?;

        Ok(())
    }
}

struct McpGrpcSvc {
    server: Arc<McpServer>,
}

#[tonic::async_trait]
impl McpService for McpGrpcSvc {
    async fn initialize(
        &self,
        _request: Request<proto::InitializeRequest>,
    ) -> std::result::Result<Response<proto::JsonRpcResult>, Status> {
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        })
        .to_string();

        let result = self
            .server
            .handle_request_raw(&body)
            .await
            .map_err(Status::internal)?;

        Ok(Response::new(proto::JsonRpcResult {
            result_json: result,
        }))
    }

    async fn tools_list(
        &self,
        _request: Request<proto::ToolsListRequest>,
    ) -> std::result::Result<Response<proto::JsonRpcResult>, Status> {
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list"
        })
        .to_string();

        let result = self
            .server
            .handle_request_raw(&body)
            .await
            .map_err(Status::internal)?;

        Ok(Response::new(proto::JsonRpcResult {
            result_json: result,
        }))
    }

    async fn tools_call(
        &self,
        request: Request<proto::ToolsCallRequest>,
    ) -> std::result::Result<Response<proto::JsonRpcResult>, Status> {
        let req = request.into_inner();

        let arguments: serde_json::Value = serde_json::from_str(&req.arguments_json)
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": req.name,
                "arguments": arguments
            }
        })
        .to_string();

        let result = self
            .server
            .handle_request_raw(&body)
            .await
            .map_err(Status::internal)?;

        Ok(Response::new(proto::JsonRpcResult {
            result_json: result,
        }))
    }

    async fn handle_raw(
        &self,
        request: Request<proto::RawJsonRpc>,
    ) -> std::result::Result<Response<proto::RawJsonRpc>, Status> {
        let body = request.into_inner().body;

        let result = self
            .server
            .handle_request_raw(&body)
            .await
            .map_err(Status::internal)?;

        Ok(Response::new(proto::RawJsonRpc { body: result }))
    }
}

/// An [`McpTransport`] that communicates with a remote gRPC MCP server.
///
/// Connects to a [`McpGrpcServer`] and translates JSON-RPC
/// requests/notifications into gRPC calls.
pub struct McpGrpcTransport {
    client: tokio::sync::Mutex<McpServiceClient<tonic::transport::Channel>>,
}

impl McpGrpcTransport {
    /// Connects to a remote gRPC MCP server.
    pub async fn connect(addr: impl Into<String>) -> Result<Self> {
        let client = McpServiceClient::connect(addr.into())
            .await
            .map_err(|e| DaimonError::Mcp(format!("grpc mcp connect: {e}")))?;

        Ok(Self {
            client: tokio::sync::Mutex::new(client),
        })
    }
}

impl McpTransport for McpGrpcTransport {
    fn send<'a>(
        &'a self,
        request: &'a JsonRpcRequest,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<JsonRpcResponse>> + Send + 'a>> {
        Box::pin(async move {
            let body = serde_json::to_string(request)
                .map_err(|e| DaimonError::Mcp(format!("serialize request: {e}")))?;

            let resp = self
                .client
                .lock()
                .await
                .handle_raw(Request::new(proto::RawJsonRpc { body }))
                .await
                .map_err(|e| DaimonError::Mcp(format!("grpc send: {e}")))?;

            let response: JsonRpcResponse = serde_json::from_str(&resp.into_inner().body)
                .map_err(|e| DaimonError::Mcp(format!("deserialize response: {e}")))?;

            Ok(response)
        })
    }

    fn notify<'a>(
        &'a self,
        _notification: &'a JsonRpcNotification,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }

    fn close<'a>(&'a self) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{Tool, ToolOutput, ToolRegistry};

    struct EchoTool;

    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "Echoes input"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": { "text": { "type": "string" } }
            })
        }
        async fn execute(&self, input: &serde_json::Value) -> crate::error::Result<ToolOutput> {
            let text = input.get("text").and_then(|v| v.as_str()).unwrap_or("?");
            Ok(ToolOutput::text(text))
        }
    }

    fn make_mcp_server() -> McpServer {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool).unwrap();
        McpServer::new(registry)
    }

    #[tokio::test]
    async fn test_grpc_mcp_server_and_client() {
        let mcp_server = make_mcp_server();

        let listener = tokio::net::TcpListener::bind("[::1]:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let _server_handle = tokio::spawn(async move {
            McpGrpcServer::new(mcp_server)
                .serve(addr.to_string())
                .await
                .ok();
        });

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let transport = McpGrpcTransport::connect(format!("http://{addr}"))
            .await
            .unwrap();

        let init_req = JsonRpcRequest::new(1, "initialize", Some(serde_json::json!({})));
        let resp = transport.send(&init_req).await.unwrap();
        assert!(resp.result.is_some());
        let result = resp.result.unwrap();
        assert!(result["capabilities"]["tools"].is_object());

        let list_req = JsonRpcRequest::new(2, "tools/list", None);
        let resp = transport.send(&list_req).await.unwrap();
        let result = resp.result.unwrap();
        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "echo");

        let call_req = JsonRpcRequest::new(
            3,
            "tools/call",
            Some(serde_json::json!({
                "name": "echo",
                "arguments": { "text": "hello grpc" }
            })),
        );
        let resp = transport.send(&call_req).await.unwrap();
        let result = resp.result.unwrap();
        assert_eq!(result["content"][0]["text"], "hello grpc");
    }

    #[test]
    fn test_proto_types_compile() {
        let _ = proto::InitializeRequest {};
        let _ = proto::ToolsListRequest {};
        let _ = proto::ToolsCallRequest {
            name: "test".into(),
            arguments_json: "{}".into(),
        };
        let _ = proto::RawJsonRpc { body: "{}".into() };
        let _ = proto::JsonRpcResult {
            result_json: "{}".into(),
        };
    }
}
