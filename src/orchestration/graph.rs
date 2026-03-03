//! Graph-based orchestration with conditional routing, cycles, and fan-out/fan-in.
//!
//! A [`Graph`] is a directed graph of [`GraphNode`]s connected by edges.
//! Edges can be unconditional or conditional (predicate-based). Nodes return
//! a [`NodeOutcome`] that controls the traversal.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::agent::Agent;
use crate::error::{DaimonError, Result};

/// Shared state flowing through a graph. Nodes read and write entries.
#[derive(Debug, Clone, Default)]
pub struct GraphContext {
    /// Key-value state shared across all nodes.
    pub state: HashMap<String, serde_json::Value>,
}

impl GraphContext {
    /// Creates an empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets a state entry.
    pub fn set(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.state.insert(key.into(), value);
    }

    /// Gets a state entry.
    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.state.get(key)
    }

    /// Gets a state entry as a string.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.state.get(key).and_then(|v| v.as_str())
    }

    /// Convenience: set the "input" key.
    pub fn with_input(mut self, input: impl Into<String>) -> Self {
        self.set("input", serde_json::Value::String(input.into()));
        self
    }
}

/// The outcome of a node's execution, controlling graph traversal.
#[derive(Debug, Clone)]
pub enum NodeOutcome {
    /// Follow the edges from this node (check conditions in order).
    Continue,
    /// Route directly to a named node, ignoring edges.
    Route(String),
    /// Execute multiple branches in parallel, then continue from the merge node.
    FanOut {
        /// Nodes to execute in parallel.
        branches: Vec<String>,
        /// Node to continue from after all branches complete.
        merge: String,
    },
    /// The graph is done; stop execution.
    Done,
}

/// A node in a [`Graph`]. Receives the shared context and returns an outcome.
pub trait GraphNode: Send + Sync {
    /// Processes the context and returns the next traversal action.
    fn process<'a>(
        &'a self,
        ctx: &'a mut GraphContext,
    ) -> Pin<Box<dyn Future<Output = Result<NodeOutcome>> + Send + 'a>>;
}

/// Wraps an [`Agent`] as a [`GraphNode`]. Reads `state["input"]` as the prompt,
/// writes the response to `state["output"]`, and returns [`NodeOutcome::Continue`].
pub struct AgentNode {
    agent: Arc<Agent>,
    input_key: String,
    output_key: String,
}

impl AgentNode {
    /// Wraps an agent as a graph node. Reads from `input_key` and writes to `output_key`.
    pub fn new(agent: Arc<Agent>, input_key: impl Into<String>, output_key: impl Into<String>) -> Self {
        Self {
            agent,
            input_key: input_key.into(),
            output_key: output_key.into(),
        }
    }
}

impl GraphNode for AgentNode {
    fn process<'a>(
        &'a self,
        ctx: &'a mut GraphContext,
    ) -> Pin<Box<dyn Future<Output = Result<NodeOutcome>> + Send + 'a>> {
        Box::pin(async move {
            let input = ctx
                .get_str(&self.input_key)
                .unwrap_or("")
                .to_string();
            let response = self.agent.prompt(&input).await?;
            ctx.set(&self.output_key, serde_json::Value::String(response.final_text));
            Ok(NodeOutcome::Continue)
        })
    }
}

type BoxedGraphFn = Arc<
    dyn for<'a> Fn(
            &'a mut GraphContext,
        ) -> Pin<Box<dyn Future<Output = Result<NodeOutcome>> + Send + 'a>>
        + Send
        + Sync,
>;

/// A node built from an async closure.
pub struct FnNode {
    func: BoxedGraphFn,
}

impl FnNode {
    /// Creates a node from a closure that returns a boxed future.
    ///
    /// Use with `|ctx| Box::pin(async move { ... })` for ergonomic construction.
    pub fn new<F>(func: F) -> Self
    where
        F: for<'a> Fn(
                &'a mut GraphContext,
            ) -> Pin<Box<dyn Future<Output = Result<NodeOutcome>> + Send + 'a>>
            + Send
            + Sync
            + 'static,
    {
        Self {
            func: Arc::new(func),
        }
    }
}

impl GraphNode for FnNode {
    fn process<'a>(
        &'a self,
        ctx: &'a mut GraphContext,
    ) -> Pin<Box<dyn Future<Output = Result<NodeOutcome>> + Send + 'a>> {
        (self.func)(ctx)
    }
}

/// An edge connecting two nodes. Can be unconditional or conditional.
pub struct Edge {
    /// Target node name.
    pub target: String,
    /// Optional condition; if `None`, the edge is always taken.
    pub condition: Option<Arc<dyn Fn(&GraphContext) -> bool + Send + Sync>>,
}

/// Builder for constructing a [`Graph`].
pub struct GraphBuilder {
    nodes: HashMap<String, Arc<dyn GraphNode>>,
    edges: HashMap<String, Vec<Edge>>,
    entry: Option<String>,
    max_steps: usize,
}

impl GraphBuilder {
    fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            entry: None,
            max_steps: 100,
        }
    }

    /// Adds a node with the given name. The first node added becomes the entry
    /// unless [`entry`](GraphBuilder::entry) is called.
    pub fn node<N: GraphNode + 'static>(mut self, name: impl Into<String>, node: N) -> Self {
        let name = name.into();
        if self.entry.is_none() {
            self.entry = Some(name.clone());
        }
        self.nodes.insert(name, Arc::new(node));
        self
    }

    /// Adds an unconditional edge from `from` to `to`.
    pub fn edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        let from = from.into();
        self.edges.entry(from).or_default().push(Edge {
            target: to.into(),
            condition: None,
        });
        self
    }

    /// Adds a conditional edge from `from` to `to`.
    pub fn conditional_edge<F>(
        mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        predicate: F,
    ) -> Self
    where
        F: Fn(&GraphContext) -> bool + Send + Sync + 'static,
    {
        let from = from.into();
        self.edges.entry(from).or_default().push(Edge {
            target: to.into(),
            condition: Some(Arc::new(predicate)),
        });
        self
    }

    /// Sets which node to start execution from.
    pub fn entry(mut self, name: impl Into<String>) -> Self {
        self.entry = Some(name.into());
        self
    }

    /// Sets the maximum number of node executions before aborting (default: 100).
    pub fn max_steps(mut self, max: usize) -> Self {
        self.max_steps = max;
        self
    }

    /// Builds the graph. Fails if no nodes or no entry is defined.
    pub fn build(self) -> Result<Graph> {
        let entry = self.entry.ok_or_else(|| {
            DaimonError::Orchestration("graph must have at least one node".into())
        })?;

        if !self.nodes.contains_key(&entry) {
            return Err(DaimonError::Orchestration(format!(
                "entry node '{entry}' not found"
            )));
        }

        Ok(Graph {
            nodes: self.nodes,
            edges: self.edges,
            entry,
            max_steps: self.max_steps,
        })
    }
}

/// A directed graph of nodes with conditional routing, cycle support,
/// and fan-out/fan-in parallel execution.
pub struct Graph {
    nodes: HashMap<String, Arc<dyn GraphNode>>,
    edges: HashMap<String, Vec<Edge>>,
    entry: String,
    max_steps: usize,
}

impl Graph {
    /// Returns a new graph builder.
    pub fn builder() -> GraphBuilder {
        GraphBuilder::new()
    }

    /// Executes the graph starting from the entry node.
    #[tracing::instrument(skip_all, fields(entry = %self.entry, max_steps = self.max_steps))]
    pub async fn run(&self, ctx: GraphContext) -> Result<GraphContext> {
        let mut ctx = ctx;
        let mut current = self.entry.clone();
        let mut steps = 0;

        loop {
            steps += 1;
            if steps > self.max_steps {
                return Err(DaimonError::Orchestration(format!(
                    "graph exceeded max steps ({}) — possible infinite loop",
                    self.max_steps
                )));
            }

            let node = self.nodes.get(&current).ok_or_else(|| {
                DaimonError::Orchestration(format!("node '{current}' not found"))
            })?;

            let _span = tracing::info_span!("graph_node", name = %current, step = steps).entered();
            let outcome = node.process(&mut ctx).await?;

            match outcome {
                NodeOutcome::Done => return Ok(ctx),

                NodeOutcome::Route(target) => {
                    current = target;
                }

                NodeOutcome::FanOut { branches, merge } => {
                    ctx = self.execute_fan_out(ctx, &branches).await?;
                    current = merge;
                }

                NodeOutcome::Continue => {
                    current = self.follow_edges(&current, &ctx)?;
                }
            }
        }
    }

    fn follow_edges(&self, from: &str, ctx: &GraphContext) -> Result<String> {
        let edges = self.edges.get(from).ok_or_else(|| {
            DaimonError::Orchestration(format!("no edges from node '{from}'"))
        })?;

        for edge in edges {
            match &edge.condition {
                Some(predicate) if !predicate(ctx) => continue,
                _ => return Ok(edge.target.clone()),
            }
        }

        Err(DaimonError::Orchestration(format!(
            "no matching edge from node '{from}'"
        )))
    }

    async fn execute_fan_out(
        &self,
        ctx: GraphContext,
        branches: &[String],
    ) -> Result<GraphContext> {
        use tokio::task::JoinSet;

        let mut join_set = JoinSet::new();

        for branch_name in branches {
            let node = self.nodes.get(branch_name).cloned().ok_or_else(|| {
                DaimonError::Orchestration(format!("fan-out node '{branch_name}' not found"))
            })?;
            let mut branch_ctx = ctx.clone();
            join_set.spawn(async move {
                node.process(&mut branch_ctx).await?;
                Ok::<_, DaimonError>(branch_ctx)
            });
        }

        let mut merged = ctx;
        while let Some(result) = join_set.join_next().await {
            let branch_ctx = result
                .map_err(|e| DaimonError::Orchestration(format!("fan-out join: {e}")))?
                ?;
            for (key, value) in branch_ctx.state {
                merged.state.insert(key, value);
            }
        }

        Ok(merged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SetValueNode {
        key: String,
        value: serde_json::Value,
    }

    impl GraphNode for SetValueNode {
        fn process<'a>(
            &'a self,
            ctx: &'a mut GraphContext,
        ) -> Pin<Box<dyn Future<Output = Result<NodeOutcome>> + Send + 'a>> {
            Box::pin(async move {
                ctx.set(&self.key, self.value.clone());
                Ok(NodeOutcome::Continue)
            })
        }
    }

    struct DoneNode;

    impl GraphNode for DoneNode {
        fn process<'a>(
            &'a self,
            _ctx: &'a mut GraphContext,
        ) -> Pin<Box<dyn Future<Output = Result<NodeOutcome>> + Send + 'a>> {
            Box::pin(async { Ok(NodeOutcome::Done) })
        }
    }

    struct RouterNode;

    impl GraphNode for RouterNode {
        fn process<'a>(
            &'a self,
            ctx: &'a mut GraphContext,
        ) -> Pin<Box<dyn Future<Output = Result<NodeOutcome>> + Send + 'a>> {
            Box::pin(async move {
                let target = ctx
                    .get_str("route_to")
                    .unwrap_or("default")
                    .to_string();
                Ok(NodeOutcome::Route(target))
            })
        }
    }

    #[tokio::test]
    async fn test_graph_simple_linear() {
        let graph = Graph::builder()
            .node("start", SetValueNode {
                key: "x".into(),
                value: serde_json::json!(1),
            })
            .node("end", DoneNode)
            .edge("start", "end")
            .build()
            .unwrap();

        let result = graph.run(GraphContext::new()).await.unwrap();
        assert_eq!(result.get("x"), Some(&serde_json::json!(1)));
    }

    #[tokio::test]
    async fn test_graph_conditional_routing() {
        let graph = Graph::builder()
            .node("check", SetValueNode {
                key: "checked".into(),
                value: serde_json::json!(true),
            })
            .node("branch_a", SetValueNode {
                key: "branch".into(),
                value: serde_json::json!("a"),
            })
            .node("branch_b", SetValueNode {
                key: "branch".into(),
                value: serde_json::json!("b"),
            })
            .node("end", DoneNode)
            .conditional_edge("check", "branch_a", |ctx| {
                ctx.get_str("input").unwrap_or("") == "go_a"
            })
            .conditional_edge("check", "branch_b", |_ctx| true)
            .edge("branch_a", "end")
            .edge("branch_b", "end")
            .build()
            .unwrap();

        let ctx = GraphContext::new().with_input("go_a");
        let result = graph.run(ctx).await.unwrap();
        assert_eq!(result.get("branch"), Some(&serde_json::json!("a")));

        let ctx = GraphContext::new().with_input("anything_else");
        let result = graph.run(ctx).await.unwrap();
        assert_eq!(result.get("branch"), Some(&serde_json::json!("b")));
    }

    #[tokio::test]
    async fn test_graph_explicit_routing() {
        let graph = Graph::builder()
            .node("router", RouterNode)
            .node("target", DoneNode)
            .node("default", DoneNode)
            .build()
            .unwrap();

        let mut ctx = GraphContext::new();
        ctx.set("route_to", serde_json::json!("target"));
        let result = graph.run(ctx).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_graph_max_steps() {
        struct LoopNode;

        impl GraphNode for LoopNode {
            fn process<'a>(
                &'a self,
                _ctx: &'a mut GraphContext,
            ) -> Pin<Box<dyn Future<Output = Result<NodeOutcome>> + Send + 'a>> {
                Box::pin(async { Ok(NodeOutcome::Continue) })
            }
        }

        let graph = Graph::builder()
            .node("loop", LoopNode)
            .edge("loop", "loop")
            .max_steps(5)
            .build()
            .unwrap();

        let result = graph.run(GraphContext::new()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_graph_fan_out() {
        struct FanOutNode;

        impl GraphNode for FanOutNode {
            fn process<'a>(
                &'a self,
                _ctx: &'a mut GraphContext,
            ) -> Pin<Box<dyn Future<Output = Result<NodeOutcome>> + Send + 'a>> {
                Box::pin(async {
                    Ok(NodeOutcome::FanOut {
                        branches: vec!["a".into(), "b".into()],
                        merge: "merge".into(),
                    })
                })
            }
        }

        let graph = Graph::builder()
            .node("start", FanOutNode)
            .node("a", SetValueNode {
                key: "from_a".into(),
                value: serde_json::json!(true),
            })
            .node("b", SetValueNode {
                key: "from_b".into(),
                value: serde_json::json!(true),
            })
            .node("merge", DoneNode)
            .build()
            .unwrap();

        let result = graph.run(GraphContext::new()).await.unwrap();
        assert_eq!(result.get("from_a"), Some(&serde_json::json!(true)));
        assert_eq!(result.get("from_b"), Some(&serde_json::json!(true)));
    }

    #[tokio::test]
    async fn test_graph_cycle_with_counter() {
        struct CounterNode;

        impl GraphNode for CounterNode {
            fn process<'a>(
                &'a self,
                ctx: &'a mut GraphContext,
            ) -> Pin<Box<dyn Future<Output = Result<NodeOutcome>> + Send + 'a>> {
                Box::pin(async move {
                    let count = ctx
                        .get("count")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    ctx.set("count", serde_json::json!(count + 1));
                    Ok(NodeOutcome::Continue)
                })
            }
        }

        let graph = Graph::builder()
            .node("counter", CounterNode)
            .node("done", DoneNode)
            .conditional_edge("counter", "done", |ctx| {
                ctx.get("count").and_then(|v| v.as_u64()).unwrap_or(0) >= 3
            })
            .edge("counter", "counter")
            .build()
            .unwrap();

        let result = graph.run(GraphContext::new()).await.unwrap();
        assert_eq!(result.get("count"), Some(&serde_json::json!(3)));
    }

    #[tokio::test]
    async fn test_graph_empty_fails() {
        let result = Graph::builder().build();
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_graph_missing_entry() {
        let result = Graph::builder().entry("nonexistent").build();
        assert!(result.is_err());
    }
}
