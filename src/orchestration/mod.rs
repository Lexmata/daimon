//! Multi-agent orchestration: chains and graphs.
//!
//! - [`Chain`] runs steps sequentially, passing context from one to the next.
//! - [`Graph`] runs nodes with conditional routing, cycles, and fan-out/fan-in.

pub mod chain;
pub mod graph;

pub use chain::{AgentStep, Chain, ChainBuilder, ChainContext, ChainStep, TransformStep};
pub use graph::{AgentNode, Edge, FnNode, Graph, GraphBuilder, GraphContext, GraphNode, NodeOutcome};
