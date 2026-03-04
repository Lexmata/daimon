//! Multi-agent orchestration: chains, graphs, DAGs, and workflows.
//!
//! - [`Chain`] runs steps sequentially, passing context from one to the next.
//! - [`Graph`] runs nodes with conditional routing, cycles, and fan-out/fan-in
//!   (Pregel / sequential walker model).
//! - [`Dag`] performs topological scheduling: independent nodes execute in
//!   parallel, with cycle detection at build time (AllPredecessor model).
//! - [`Workflow`] extends the DAG model with field-level data mapping between
//!   nodes, inspired by Eino's typed-field workflow engine.

pub mod chain;
pub mod dag;
pub mod graph;
pub mod workflow;

pub use chain::{AgentStep, Chain, ChainBuilder, ChainContext, ChainStep, TransformStep};
pub use dag::{AgentDagNode, Dag, DagBuilder, DagContext, DagNode, FnDagNode, END, START};
pub use graph::{AgentNode, Edge, FnNode, Graph, GraphBuilder, GraphContext, GraphNode, NodeOutcome};
pub use workflow::{
    AgentWorkflowNode, FnWorkflowNode, Workflow, WorkflowBuilder, WorkflowNode,
};
