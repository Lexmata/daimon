//! Directed Acyclic Graph (DAG) orchestration with parallel execution.
//!
//! Unlike [`Graph`](super::Graph), which walks one node at a time following
//! edges (Pregel / sequential model), a [`Dag`] uses topological scheduling:
//! nodes whose predecessors have **all** completed execute concurrently.
//! This makes fan-out/fan-in patterns natural — independent branches run in
//! parallel without any explicit `FanOut` return value.
//!
//! Inspired by Eino's `AllPredecessor` / DAG execution engine.
//!
//! # Sentinels
//!
//! Every DAG has implicit [`START`] and [`END`] nodes. Edges from `START`
//! define the entry points; edges into `END` define the exit points.
//!
//! # Example
//!
//! ```ignore
//! use daimon::orchestration::dag::{Dag, FnDagNode, DagContext, START, END};
//!
//! let dag = Dag::builder()
//!     .node("summarize", FnDagNode::new(|ctx| Box::pin(async move {
//!         let input = ctx.get_str("input").unwrap_or_default().to_string();
//!         ctx.set("summary", serde_json::json!(format!("Summary of: {input}")));
//!         Ok(())
//!     })))
//!     .node("translate", FnDagNode::new(|ctx| Box::pin(async move {
//!         let input = ctx.get_str("input").unwrap_or_default().to_string();
//!         ctx.set("translation", serde_json::json!(format!("[FR] {input}")));
//!         Ok(())
//!     })))
//!     .edge(START, "summarize")
//!     .edge(START, "translate")   // runs in parallel with summarize
//!     .edge("summarize", END)
//!     .edge("translate", END)     // END waits for both
//!     .build()
//!     .unwrap();
//!
//! let result = dag.run(DagContext::new().with_input("hello")).await.unwrap();
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::agent::Agent;
use crate::error::{DaimonError, Result};

/// Sentinel name for the DAG entry point. Edges from `START` define which
/// nodes execute first.
pub const START: &str = "__start__";

/// Sentinel name for the DAG exit point. Edges into `END` define which
/// nodes must complete before the DAG finishes.
pub const END: &str = "__end__";

/// Shared key-value state flowing through a DAG.
///
/// Nodes read predecessor outputs and write their own results into this
/// context. Parallel nodes within the same topological level each receive
/// a clone; their writes are merged back after the level completes.
#[derive(Debug, Clone, Default)]
pub struct DagContext {
    /// Key-value state shared across all nodes.
    pub state: HashMap<String, serde_json::Value>,
}

impl DagContext {
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

    /// Gets a state entry as a string slice.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.state.get(key).and_then(|v| v.as_str())
    }

    /// Convenience: set the `"input"` key.
    pub fn with_input(mut self, input: impl Into<String>) -> Self {
        self.set("input", serde_json::Value::String(input.into()));
        self
    }
}

// ---------------------------------------------------------------------------
// DagNode trait
// ---------------------------------------------------------------------------

/// A processing node in a [`Dag`].
///
/// Unlike [`GraphNode`](super::GraphNode), a `DagNode` has no routing control —
/// traversal is determined entirely by the declared edge topology. The node
/// simply reads from and writes to the shared [`DagContext`].
pub trait DagNode: Send + Sync {
    /// Processes the context. The future must be `Send`.
    fn process<'a>(
        &'a self,
        ctx: &'a mut DagContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
}

// ---------------------------------------------------------------------------
// AgentDagNode
// ---------------------------------------------------------------------------

/// Wraps an [`Agent`] as a [`DagNode`].
///
/// Reads `state[input_key]` as the prompt, writes the agent's response
/// text to `state[output_key]`.
pub struct AgentDagNode {
    agent: Arc<Agent>,
    input_key: String,
    output_key: String,
}

impl AgentDagNode {
    /// Wraps an agent, reading from `input_key` and writing to `output_key`.
    pub fn new(
        agent: Arc<Agent>,
        input_key: impl Into<String>,
        output_key: impl Into<String>,
    ) -> Self {
        Self {
            agent,
            input_key: input_key.into(),
            output_key: output_key.into(),
        }
    }
}

impl DagNode for AgentDagNode {
    fn process<'a>(
        &'a self,
        ctx: &'a mut DagContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let input = ctx.get_str(&self.input_key).unwrap_or("").to_string();
            let response = self.agent.prompt(&input).await?;
            ctx.set(
                &self.output_key,
                serde_json::Value::String(response.final_text),
            );
            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------
// FnDagNode
// ---------------------------------------------------------------------------

type BoxedDagFn = Arc<
    dyn for<'a> Fn(
            &'a mut DagContext,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>
        + Send
        + Sync,
>;

/// A DAG node built from an async closure.
///
/// ```ignore
/// FnDagNode::new(|ctx| Box::pin(async move {
///     ctx.set("greeting", serde_json::json!("hello"));
///     Ok(())
/// }))
/// ```
pub struct FnDagNode {
    func: BoxedDagFn,
}

impl FnDagNode {
    /// Creates a node from a closure returning a boxed, pinned future.
    pub fn new<F>(func: F) -> Self
    where
        F: for<'a> Fn(
                &'a mut DagContext,
            ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>
            + Send
            + Sync
            + 'static,
    {
        Self {
            func: Arc::new(func),
        }
    }
}

impl DagNode for FnDagNode {
    fn process<'a>(
        &'a self,
        ctx: &'a mut DagContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        (self.func)(ctx)
    }
}

// ---------------------------------------------------------------------------
// Branch
// ---------------------------------------------------------------------------

type BranchFn = Arc<dyn Fn(&DagContext) -> Result<Vec<String>> + Send + Sync>;

// ---------------------------------------------------------------------------
// DagBuilder
// ---------------------------------------------------------------------------

/// Builder for constructing a [`Dag`].
///
/// Nodes and edges are declared, then `build()` validates acyclicity via
/// topological sort and produces the executable DAG.
pub struct DagBuilder {
    nodes: HashMap<String, Arc<dyn DagNode>>,
    edges: Vec<(String, String)>,
    branches: HashMap<String, BranchFn>,
}

impl DagBuilder {
    fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            branches: HashMap::new(),
        }
    }

    /// Adds a named processing node.
    pub fn node<N: DagNode + 'static>(mut self, name: impl Into<String>, node: N) -> Self {
        self.nodes.insert(name.into(), Arc::new(node));
        self
    }

    /// Adds a directed edge from `from` to `to`.
    ///
    /// Use [`START`] and [`END`] for the DAG entry/exit sentinels.
    pub fn edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.edges.push((from.into(), to.into()));
        self
    }

    /// Attaches a single-select branch to a node.
    ///
    /// When the branched node completes, the condition is evaluated. Only the
    /// returned successor is activated; all other successors of this node
    /// are skipped.
    pub fn branch<F>(mut self, from: impl Into<String>, condition: F) -> Self
    where
        F: Fn(&DagContext) -> Result<String> + Send + Sync + 'static,
    {
        let from = from.into();
        self.branches.insert(
            from,
            Arc::new(move |ctx| {
                let selected = condition(ctx)?;
                Ok(vec![selected])
            }),
        );
        self
    }

    /// Attaches a multi-select branch to a node.
    ///
    /// When the branched node completes, the condition returns which
    /// successors to activate. Unselected successors are skipped.
    pub fn multi_branch<F>(mut self, from: impl Into<String>, condition: F) -> Self
    where
        F: Fn(&DagContext) -> Result<Vec<String>> + Send + Sync + 'static,
    {
        self.branches.insert(from.into(), Arc::new(condition));
        self
    }

    /// Builds the DAG. Fails if cycles are detected or no edges are declared.
    pub fn build(self) -> Result<Dag> {
        if self.edges.is_empty() {
            return Err(DaimonError::Orchestration(
                "DAG must have at least one edge".into(),
            ));
        }

        let mut all_nodes: HashSet<String> = HashSet::new();
        all_nodes.insert(START.to_string());
        all_nodes.insert(END.to_string());
        for name in self.nodes.keys() {
            all_nodes.insert(name.clone());
        }

        let mut successors: HashMap<String, Vec<String>> = HashMap::new();
        let mut predecessors: HashMap<String, Vec<String>> = HashMap::new();

        for (from, to) in &self.edges {
            if from != START && !self.nodes.contains_key(from) {
                return Err(DaimonError::Orchestration(format!(
                    "edge references unknown node '{from}'"
                )));
            }
            if to != END && !self.nodes.contains_key(to) {
                return Err(DaimonError::Orchestration(format!(
                    "edge references unknown node '{to}'"
                )));
            }
            successors
                .entry(from.clone())
                .or_default()
                .push(to.clone());
            predecessors
                .entry(to.clone())
                .or_default()
                .push(from.clone());
        }

        if !successors.contains_key(START) {
            return Err(DaimonError::Orchestration(
                "no edges from START".into(),
            ));
        }
        if !predecessors.contains_key(END) {
            return Err(DaimonError::Orchestration(
                "no edges into END".into(),
            ));
        }

        let levels = topological_levels(&all_nodes, &successors, &predecessors)?;

        Ok(Dag {
            nodes: self.nodes,
            levels,
            successors,
            predecessors,
            branches: self.branches,
        })
    }
}

// ---------------------------------------------------------------------------
// Dag
// ---------------------------------------------------------------------------

/// A compiled Directed Acyclic Graph for parallel orchestration.
///
/// Nodes are grouped into topological levels. Within each level, all nodes
/// execute concurrently. A node only runs once **all** its predecessors have
/// completed.
impl std::fmt::Debug for Dag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dag")
            .field("levels", &self.levels)
            .field("node_count", &self.nodes.len())
            .finish()
    }
}

pub struct Dag {
    nodes: HashMap<String, Arc<dyn DagNode>>,
    levels: Vec<Vec<String>>,
    successors: HashMap<String, Vec<String>>,
    predecessors: HashMap<String, Vec<String>>,
    branches: HashMap<String, BranchFn>,
}

impl Dag {
    /// Returns a new DAG builder.
    pub fn builder() -> DagBuilder {
        DagBuilder::new()
    }

    /// Executes the DAG.
    ///
    /// Nodes in the same topological level run in parallel. Branch conditions
    /// are evaluated after each level to determine which successors are
    /// activated for subsequent levels.
    #[tracing::instrument(skip_all, fields(levels = self.levels.len()))]
    pub async fn run(&self, ctx: DagContext) -> Result<DagContext> {
        let mut ctx = ctx;
        let mut active_edges: HashSet<(String, String)> = HashSet::new();

        self.activate_successors(START, &ctx, &mut active_edges)?;

        for level in &self.levels {
            let mut runnable: Vec<&str> = Vec::new();

            for name in level {
                if name == START || name == END {
                    continue;
                }
                let preds = self.predecessors.get(name);
                let has_active_incoming = preds.is_some_and(|ps| {
                    ps.iter().any(|p| active_edges.contains(&(p.clone(), name.clone())))
                });
                if has_active_incoming {
                    runnable.push(name);
                }
            }

            if runnable.is_empty() {
                continue;
            }

            if runnable.len() == 1 {
                let name = runnable[0];
                let node = self.nodes.get(name).ok_or_else(|| {
                    DaimonError::Orchestration(format!("node '{name}' not found"))
                })?;
                let _span =
                    tracing::info_span!("dag_node", name = %name).entered();
                node.process(&mut ctx).await?;
                self.activate_successors(name, &ctx, &mut active_edges)?;
            } else {
                let mut join_set = tokio::task::JoinSet::new();

                for &name in &runnable {
                    let node = self.nodes.get(name).cloned().ok_or_else(|| {
                        DaimonError::Orchestration(format!("node '{name}' not found"))
                    })?;
                    let mut branch_ctx = ctx.clone();
                    let owned_name = name.to_string();
                    join_set.spawn(async move {
                        node.process(&mut branch_ctx).await?;
                        Ok::<_, DaimonError>((owned_name, branch_ctx))
                    });
                }

                while let Some(result) = join_set.join_next().await {
                    let (_, branch_ctx) = result.map_err(|e| {
                        DaimonError::Orchestration(format!("dag join error: {e}"))
                    })??;
                    for (key, value) in branch_ctx.state {
                        ctx.state.insert(key, value);
                    }
                }

                for &name in &runnable {
                    self.activate_successors(name, &ctx, &mut active_edges)?;
                }
            }
        }

        let end_reached = self
            .predecessors
            .get(END)
            .is_some_and(|ps| {
                ps.iter()
                    .any(|p| active_edges.contains(&(p.clone(), END.to_string())))
            });

        if !end_reached {
            return Err(DaimonError::Orchestration(
                "DAG execution did not reach END — all paths were skipped by branches".into(),
            ));
        }

        Ok(ctx)
    }

    fn activate_successors(
        &self,
        node: &str,
        ctx: &DagContext,
        active_edges: &mut HashSet<(String, String)>,
    ) -> Result<()> {
        let succs = match self.successors.get(node) {
            Some(s) => s,
            None => return Ok(()),
        };

        if let Some(branch_fn) = self.branches.get(node) {
            let selected = branch_fn(ctx)?;
            let selected_set: HashSet<&str> =
                selected.iter().map(|s| s.as_str()).collect();
            for succ in succs {
                if selected_set.contains(succ.as_str()) {
                    active_edges.insert((node.to_string(), succ.clone()));
                }
            }
        } else {
            for succ in succs {
                active_edges.insert((node.to_string(), succ.clone()));
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Topological sort
// ---------------------------------------------------------------------------

fn topological_levels(
    all_nodes: &HashSet<String>,
    successors: &HashMap<String, Vec<String>>,
    predecessors: &HashMap<String, Vec<String>>,
) -> Result<Vec<Vec<String>>> {
    let mut in_degree: HashMap<String, usize> = HashMap::new();
    for node in all_nodes {
        in_degree.insert(
            node.clone(),
            predecessors.get(node).map(|p| p.len()).unwrap_or(0),
        );
    }

    let mut queue: VecDeque<String> = VecDeque::new();
    for (node, &degree) in &in_degree {
        if degree == 0 {
            queue.push_back(node.clone());
        }
    }

    let mut levels: Vec<Vec<String>> = Vec::new();
    let mut visited = 0usize;

    while !queue.is_empty() {
        let level: Vec<String> = queue.drain(..).collect();
        visited += level.len();

        let mut next: VecDeque<String> = VecDeque::new();
        for node in &level {
            if let Some(succs) = successors.get(node) {
                for succ in succs {
                    let deg = in_degree.get_mut(succ).expect("node in in_degree map");
                    *deg -= 1;
                    if *deg == 0 {
                        next.push_back(succ.clone());
                    }
                }
            }
        }

        levels.push(level);
        queue = next;
    }

    if visited != all_nodes.len() {
        return Err(DaimonError::Orchestration(
            "cycle detected in DAG — use Graph for cyclic orchestration".into(),
        ));
    }

    Ok(levels)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct SetNode {
        key: String,
        value: serde_json::Value,
    }

    impl DagNode for SetNode {
        fn process<'a>(
            &'a self,
            ctx: &'a mut DagContext,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
            Box::pin(async move {
                ctx.set(&self.key, self.value.clone());
                Ok(())
            })
        }
    }

    fn set_node(key: &str, value: serde_json::Value) -> SetNode {
        SetNode {
            key: key.to_string(),
            value,
        }
    }

    #[tokio::test]
    async fn test_linear_dag() {
        let dag = Dag::builder()
            .node("a", set_node("x", serde_json::json!(1)))
            .node("b", set_node("y", serde_json::json!(2)))
            .edge(START, "a")
            .edge("a", "b")
            .edge("b", END)
            .build()
            .unwrap();

        let result = dag.run(DagContext::new()).await.unwrap();
        assert_eq!(result.get("x"), Some(&serde_json::json!(1)));
        assert_eq!(result.get("y"), Some(&serde_json::json!(2)));
    }

    #[tokio::test]
    async fn test_fan_out_fan_in() {
        let dag = Dag::builder()
            .node("a", set_node("from_a", serde_json::json!("A")))
            .node("b", set_node("from_b", serde_json::json!("B")))
            .node("c", set_node("from_c", serde_json::json!("C")))
            .node(
                "merge",
                FnDagNode::new(|ctx| {
                    Box::pin(async move {
                        let a = ctx.get_str("from_a").unwrap_or("").to_string();
                        let b = ctx.get_str("from_b").unwrap_or("").to_string();
                        let c = ctx.get_str("from_c").unwrap_or("").to_string();
                        ctx.set("merged", serde_json::json!(format!("{a}+{b}+{c}")));
                        Ok(())
                    })
                }),
            )
            .edge(START, "a")
            .edge(START, "b")
            .edge(START, "c")
            .edge("a", "merge")
            .edge("b", "merge")
            .edge("c", "merge")
            .edge("merge", END)
            .build()
            .unwrap();

        let result = dag.run(DagContext::new()).await.unwrap();
        assert_eq!(result.get("from_a"), Some(&serde_json::json!("A")));
        assert_eq!(result.get("from_b"), Some(&serde_json::json!("B")));
        assert_eq!(result.get("from_c"), Some(&serde_json::json!("C")));
        let merged = result.get_str("merged").unwrap();
        assert!(merged.contains('A'));
        assert!(merged.contains('B'));
        assert!(merged.contains('C'));
    }

    #[tokio::test]
    async fn test_diamond_dag() {
        let dag = Dag::builder()
            .node("left", set_node("left", serde_json::json!(10)))
            .node("right", set_node("right", serde_json::json!(20)))
            .node(
                "join",
                FnDagNode::new(|ctx| {
                    Box::pin(async move {
                        let l = ctx.get("left").and_then(|v| v.as_i64()).unwrap_or(0);
                        let r = ctx.get("right").and_then(|v| v.as_i64()).unwrap_or(0);
                        ctx.set("sum", serde_json::json!(l + r));
                        Ok(())
                    })
                }),
            )
            .edge(START, "left")
            .edge(START, "right")
            .edge("left", "join")
            .edge("right", "join")
            .edge("join", END)
            .build()
            .unwrap();

        let result = dag.run(DagContext::new()).await.unwrap();
        assert_eq!(result.get("sum"), Some(&serde_json::json!(30)));
    }

    #[tokio::test]
    async fn test_branch_single_select() {
        let dag = Dag::builder()
            .node(
                "router",
                FnDagNode::new(|ctx| {
                    Box::pin(async move {
                        let input = ctx.get_str("input").unwrap_or("").to_string();
                        ctx.set("routed", serde_json::json!(input));
                        Ok(())
                    })
                }),
            )
            .node("path_a", set_node("path", serde_json::json!("A")))
            .node("path_b", set_node("path", serde_json::json!("B")))
            .edge(START, "router")
            .edge("router", "path_a")
            .edge("router", "path_b")
            .edge("path_a", END)
            .edge("path_b", END)
            .branch("router", |ctx| {
                let choice = ctx.get_str("input").unwrap_or("a");
                if choice == "b" {
                    Ok("path_b".to_string())
                } else {
                    Ok("path_a".to_string())
                }
            })
            .build()
            .unwrap();

        let result = dag
            .run(DagContext::new().with_input("a"))
            .await
            .unwrap();
        assert_eq!(result.get("path"), Some(&serde_json::json!("A")));

        let result = dag
            .run(DagContext::new().with_input("b"))
            .await
            .unwrap();
        assert_eq!(result.get("path"), Some(&serde_json::json!("B")));
    }

    #[tokio::test]
    async fn test_branch_skip_propagation() {
        let dag = Dag::builder()
            .node("gate", set_node("gate", serde_json::json!(true)))
            .node("skipped_a", set_node("a_ran", serde_json::json!(true)))
            .node("active_b", set_node("b_ran", serde_json::json!(true)))
            .node(
                "only_after_a",
                set_node("after_a_ran", serde_json::json!(true)),
            )
            .edge(START, "gate")
            .edge("gate", "skipped_a")
            .edge("gate", "active_b")
            .edge("skipped_a", "only_after_a")
            .edge("only_after_a", END)
            .edge("active_b", END)
            .branch("gate", |_ctx| Ok("active_b".to_string()))
            .build()
            .unwrap();

        let result = dag.run(DagContext::new()).await.unwrap();
        assert_eq!(result.get("b_ran"), Some(&serde_json::json!(true)));
        assert!(result.get("a_ran").is_none(), "skipped_a should not have run");
        assert!(
            result.get("after_a_ran").is_none(),
            "only_after_a should be skipped because its only predecessor was skipped"
        );
    }

    #[tokio::test]
    async fn test_cycle_detection() {
        let result = Dag::builder()
            .node("a", set_node("x", serde_json::json!(1)))
            .node("b", set_node("y", serde_json::json!(2)))
            .edge(START, "a")
            .edge("a", "b")
            .edge("b", "a")
            .edge("b", END)
            .build();

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("cycle"), "expected cycle error, got: {err}");
    }

    #[tokio::test]
    async fn test_no_edges_fails() {
        let result = Dag::builder()
            .node("a", set_node("x", serde_json::json!(1)))
            .build();
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_unknown_node_in_edge() {
        let result = Dag::builder()
            .node("a", set_node("x", serde_json::json!(1)))
            .edge(START, "nonexistent")
            .edge("nonexistent", END)
            .build();
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_no_start_edges() {
        let result = Dag::builder()
            .node("a", set_node("x", serde_json::json!(1)))
            .edge("a", END)
            .build();
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_no_end_edges() {
        let result = Dag::builder()
            .node("a", set_node("x", serde_json::json!(1)))
            .edge(START, "a")
            .build();
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_context_with_input() {
        let dag = Dag::builder()
            .node(
                "echo",
                FnDagNode::new(|ctx| {
                    Box::pin(async move {
                        let input = ctx.get_str("input").unwrap_or("").to_string();
                        ctx.set("output", serde_json::json!(format!("echo: {input}")));
                        Ok(())
                    })
                }),
            )
            .edge(START, "echo")
            .edge("echo", END)
            .build()
            .unwrap();

        let result = dag
            .run(DagContext::new().with_input("hello"))
            .await
            .unwrap();
        assert_eq!(
            result.get_str("output"),
            Some("echo: hello")
        );
    }

    #[tokio::test]
    async fn test_deep_pipeline() {
        let dag = Dag::builder()
            .node("s1", set_node("step", serde_json::json!(1)))
            .node(
                "s2",
                FnDagNode::new(|ctx| {
                    Box::pin(async move {
                        let prev = ctx.get("step").and_then(|v| v.as_i64()).unwrap_or(0);
                        ctx.set("step", serde_json::json!(prev + 1));
                        Ok(())
                    })
                }),
            )
            .node(
                "s3",
                FnDagNode::new(|ctx| {
                    Box::pin(async move {
                        let prev = ctx.get("step").and_then(|v| v.as_i64()).unwrap_or(0);
                        ctx.set("step", serde_json::json!(prev + 1));
                        Ok(())
                    })
                }),
            )
            .edge(START, "s1")
            .edge("s1", "s2")
            .edge("s2", "s3")
            .edge("s3", END)
            .build()
            .unwrap();

        let result = dag.run(DagContext::new()).await.unwrap();
        assert_eq!(result.get("step"), Some(&serde_json::json!(3)));
    }

    #[tokio::test]
    async fn test_multi_branch() {
        let dag = Dag::builder()
            .node("gate", set_node("gate", serde_json::json!(true)))
            .node("a", set_node("a_ran", serde_json::json!(true)))
            .node("b", set_node("b_ran", serde_json::json!(true)))
            .node("c", set_node("c_ran", serde_json::json!(true)))
            .edge(START, "gate")
            .edge("gate", "a")
            .edge("gate", "b")
            .edge("gate", "c")
            .edge("a", END)
            .edge("b", END)
            .edge("c", END)
            .multi_branch("gate", |_ctx| Ok(vec!["a".to_string(), "c".to_string()]))
            .build()
            .unwrap();

        let result = dag.run(DagContext::new()).await.unwrap();
        assert_eq!(result.get("a_ran"), Some(&serde_json::json!(true)));
        assert!(result.get("b_ran").is_none(), "b should be skipped");
        assert_eq!(result.get("c_ran"), Some(&serde_json::json!(true)));
    }
}
