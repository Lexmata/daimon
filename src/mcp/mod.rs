//! Model Context Protocol (MCP) client.
//!
//! Connect to external tool servers via stdio or HTTP to discover and call
//! tools that integrate seamlessly with Daimon agents.
//!
//! # Transports
//!
//! - [`StdioTransport`] — spawn a child process and communicate via stdin/stdout
//! - [`HttpTransport`] — send JSON-RPC requests over HTTP POST
//!
//! # Example
//!
//! ```ignore
//! use daimon::mcp::{McpClient, StdioTransport};
//!
//! let transport = StdioTransport::new("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]);
//! let client = McpClient::connect(transport).await?;
//!
//! // Register MCP tools with an agent:
//! let mut builder = Agent::builder().model(model);
//! for tool in client.tools() {
//!     builder = builder.tool(tool);
//! }
//! ```

pub mod protocol;
pub mod transport;
pub mod client;
pub mod bridge;

pub use client::McpClient;
pub use bridge::McpToolBridge;
pub use transport::{HttpTransport, McpTransport, StdioTransport};
