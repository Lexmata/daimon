//! Google Agent-to-Agent (A2A) protocol support.
//!
//! The A2A protocol enables peer-to-peer communication between AI agents,
//! complementing MCP (which connects agents to tools). A2A uses JSON-RPC 2.0
//! over HTTP with support for synchronous, streaming (SSE), and asynchronous
//! interaction modes.
//!
//! # Components
//!
//! - [`types`] — Agent Card, Task, Message, Artifact, and protocol types
//! - [`client`] — HTTP client for calling remote A2A agents
//! - [`server`] — Framework-agnostic request handler for exposing agents via A2A

pub mod client;
pub mod server;
pub mod types;

pub use client::A2aClient;
pub use server::A2aHandler;
pub use types::{A2aMessage, A2aTask, AgentCard};
