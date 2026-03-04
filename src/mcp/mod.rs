//! Model Context Protocol (MCP) client and server.
//!
//! ## Client
//!
//! Connect to external tool servers via stdio or HTTP to discover and call
//! tools that integrate seamlessly with Daimon agents.
//!
//! ```ignore
//! use daimon::mcp::{McpClient, StdioTransport};
//!
//! let transport = StdioTransport::new("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]);
//! let client = McpClient::connect(transport).await?;
//! ```
//!
//! ## Server
//!
//! Expose a [`ToolRegistry`](crate::tool::ToolRegistry) as an MCP-compliant
//! tool server over stdio.
//!
//! ```ignore
//! use daimon::mcp::McpServer;
//!
//! McpServer::new(registry).serve_stdio().await?;
//! ```

pub mod protocol;
pub mod transport;
pub mod client;
pub mod bridge;
pub mod server;
pub mod websocket;
pub mod ws_server;

#[cfg(feature = "grpc")]
pub mod grpc_transport;

pub use client::McpClient;
pub use bridge::McpToolBridge;
pub use server::McpServer;
pub use transport::{HttpTransport, McpTransport, StdioTransport};
pub use websocket::WebSocketTransport;
pub use ws_server::McpWsServer;

#[cfg(feature = "grpc")]
pub use grpc_transport::{McpGrpcServer, McpGrpcTransport};
