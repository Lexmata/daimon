//! Model Context Protocol (MCP) client and server.
//!
//! ## Client
//!
//! Connect to external tool servers via stdio, HTTP, SSE, or WebSocket to
//! discover and call tools that integrate seamlessly with Daimon agents.
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
//!
//! Daimon implements four client transports: stdio ([`StdioTransport`]), HTTP
//! ([`HttpTransport`]), SSE ([`SseTransport`], the pre-Streamable-HTTP
//! "HTTP+SSE" transport), and WebSocket ([`WebSocketTransport`]). A bespoke
//! gRPC transport was removed in 0.17.0 as non-spec surface with no
//! consumers and stays removed.

pub mod bridge;
pub mod client;
pub mod protocol;
pub mod server;
pub mod sse;
pub mod transport;
pub mod websocket;

pub use bridge::McpToolBridge;
pub use client::McpClient;
pub use server::McpServer;
pub use sse::SseTransport;
pub use transport::{HttpTransport, McpTransport, StdioTransport};
pub use websocket::WebSocketTransport;
