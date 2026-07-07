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
//!
//! Daimon implements the two transports defined by the MCP specification —
//! stdio ([`StdioTransport`]) and HTTP ([`HttpTransport`]). Bespoke WebSocket
//! and gRPC transports were removed in 0.17.0 as non-spec surface with no
//! consumers.

pub mod bridge;
pub mod client;
pub mod protocol;
pub mod server;
pub mod transport;

pub use bridge::McpToolBridge;
pub use client::McpClient;
pub use server::McpServer;
pub use transport::{HttpTransport, McpTransport, StdioTransport};
