//! Eino-style Workflow orchestration with field-level mapping.
//!
//! A [`Workflow`] is a DAG where each edge carries explicit **field mappings**
//! that wire specific output fields of one node to specific input fields of the
//! next. This gives precise control over data flow — instead of sharing an
//! unstructured bag of key-value pairs, each node receives a well-defined JSON
//! object assembled from its predecessors' outputs.
//!
//! Inspired by Eino's `Workflow` / typed-field mapping model.
//!
//! # Example
//!
//! ```ignore
//! use daimon::orchestration::workflow::{Workflow, FnWorkflowNode, START, END};
//! use serde_json::json;
//!
//! let wf = Workflow::builder()
//!     .node("fetch", FnWorkflowNode::new(|input| Box::pin(async move {
//!         let url = input["url"].as_str().unwrap_or_default();
//!         Ok(json!({ "body": format!("fetched {url}"), "status": 200 }))
//!     })))
//!     .node("parse", FnWorkflowNode::new(|input| Box::pin(async move {
//!         let body = input["raw_body"].as_str().unwrap_or_default();
//!         Ok(json!({ "parsed": format!("parsed: {body}") }))
//!     })))
//!     .edge(START, "fetch", &[("url", "url")])
//!     .edge("fetch", "parse", &[("body", "raw_body")])
//!     .edge("parse", END, &[("parsed", "result")])
//!     .build()
//!     .unwrap();
//!
//! let output = wf.run(json!({ "url": "https://example.com" })).await.unwrap();
//! assert!(output["result"].as_str().unwrap().contains("parsed"));
//! ```

use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::error::{DaimonError, Result};

/// Sentinel name for the workflow entry point.
pub const START: &str = "__wf_start__";

/// Sentinel name for the workflow exit point.
pub const END: &str = "__wf_end__";

// ---------------------------------------------------------------------------
// WorkflowNode trait
// ---------------------------------------------------------------------------

/// A processing node in a [`Workflow`].
///
/// Receives a JSON object assembled from predecessor outputs (via field
/// mappings) and produces a JSON object whose fields can be mapped to
/// successor inputs.
pub trait WorkflowNode: Send + Sync {
    /// Processes the assembled input and returns an output JSON object.
    fn process<'a>(
        &'a self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value>> + Send + 'a>>;
}

// ---------------------------------------------------------------------------
// FnWorkflowNode
// ---------------------------------------------------------------------------

type BoxedWorkflowFn = Arc<
    dyn Fn(serde_json::Value) -> Pin<Box<dyn Future<Output = Result<serde_json::Value>> + Send>>
        + Send
        + Sync,
>;

/// A workflow node built from an async closure.
pub struct FnWorkflowNode {
    func: BoxedWorkflowFn,
}

impl FnWorkflowNode {
    /// Creates a workflow node from a closure.
    pub fn new<F>(func: F) -> Self
    where
        F: Fn(serde_json::Value) -> Pin<Box<dyn Future<Output = Result<serde_json::Value>> + Send>>
            + Send
            + Sync
            + 'static,
    {
        Self {
            func: Arc::new(func),
        }
    }
}

impl WorkflowNode for FnWorkflowNode {
    fn process<'a>(
        &'a self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value>> + Send + 'a>> {
        (self.func)(input)
    }
}

// ---------------------------------------------------------------------------
// AgentWorkflowNode
// ---------------------------------------------------------------------------

/// Wraps an [`Agent`](crate::agent::Agent) as a [`WorkflowNode`].
///
/// Reads `input["prompt"]` as the text to send to the agent and produces
/// `{ "text": "<response>" }` as output.
pub struct AgentWorkflowNode {
    agent: Arc<crate::agent::Agent>,
    input_field: String,
}

impl AgentWorkflowNode {
    /// Creates a new agent workflow node.
    ///
    /// `input_field` is the JSON field name to read the prompt text from.
    pub fn new(agent: Arc<crate::agent::Agent>, input_field: impl Into<String>) -> Self {
        Self {
            agent,
            input_field: input_field.into(),
        }
    }
}

impl WorkflowNode for AgentWorkflowNode {
    fn process<'a>(
        &'a self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value>> + Send + 'a>> {
        Box::pin(async move {
            let prompt = input[&self.input_field].as_str().unwrap_or("").to_string();
            let response = self.agent.prompt(&prompt).await?;
            Ok(serde_json::json!({ "text": response.final_text }))
        })
    }
}

// ---------------------------------------------------------------------------
// FieldMapping
// ---------------------------------------------------------------------------

/// Maps one output field from a source node to one input field of a target node.
#[derive(Debug, Clone)]
pub struct FieldMapping {
    /// Field name in the source node's output.
    pub source_field: String,
    /// Field name in the target node's assembled input.
    pub target_field: String,
}

// ---------------------------------------------------------------------------
// WorkflowEdge
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct WorkflowEdge {
    from: String,
    to: String,
    mappings: Vec<FieldMapping>,
}

// ---------------------------------------------------------------------------
// WorkflowBuilder
// ---------------------------------------------------------------------------

/// Builder for constructing a [`Workflow`].
pub struct WorkflowBuilder {
    nodes: HashMap<String, Arc<dyn WorkflowNode>>,
    edges: Vec<WorkflowEdge>,
}

impl WorkflowBuilder {
    fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
        }
    }

    /// Adds a named processing node.
    pub fn node<N: WorkflowNode + 'static>(mut self, name: impl Into<String>, node: N) -> Self {
        self.nodes.insert(name.into(), Arc::new(node));
        self
    }

    /// Adds a directed edge with field mappings.
    ///
    /// `mappings` is a slice of `(source_field, target_field)` pairs
    /// describing how to wire the source node's output fields into the
    /// target node's input fields.
    pub fn edge(
        mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        mappings: &[(&str, &str)],
    ) -> Self {
        self.edges.push(WorkflowEdge {
            from: from.into(),
            to: to.into(),
            mappings: mappings
                .iter()
                .map(|(s, t)| FieldMapping {
                    source_field: s.to_string(),
                    target_field: t.to_string(),
                })
                .collect(),
        });
        self
    }

    /// Adds a directed edge that passes all fields through unchanged
    /// (identity mapping, useful when the field names already match).
    pub fn edge_passthrough(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.edges.push(WorkflowEdge {
            from: from.into(),
            to: to.into(),
            mappings: Vec::new(),
        });
        self
    }

    /// Builds the workflow. Validates acyclicity and edge references.
    pub fn build(self) -> Result<Workflow> {
        if self.edges.is_empty() {
            return Err(DaimonError::Orchestration(
                "workflow must have at least one edge".into(),
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
        let mut edge_mappings: HashMap<(String, String), Vec<FieldMapping>> = HashMap::new();

        for edge in &self.edges {
            if edge.from != START && !self.nodes.contains_key(&edge.from) {
                return Err(DaimonError::Orchestration(format!(
                    "edge references unknown node '{}'",
                    edge.from
                )));
            }
            if edge.to != END && !self.nodes.contains_key(&edge.to) {
                return Err(DaimonError::Orchestration(format!(
                    "edge references unknown node '{}'",
                    edge.to
                )));
            }

            successors
                .entry(edge.from.clone())
                .or_default()
                .push(edge.to.clone());
            predecessors
                .entry(edge.to.clone())
                .or_default()
                .push(edge.from.clone());
            edge_mappings
                .entry((edge.from.clone(), edge.to.clone()))
                .or_default()
                .extend(edge.mappings.clone());
        }

        if !successors.contains_key(START) {
            return Err(DaimonError::Orchestration("no edges from START".into()));
        }
        if !predecessors.contains_key(END) {
            return Err(DaimonError::Orchestration("no edges into END".into()));
        }

        let levels = super::toposort::topological_levels(
            &all_nodes,
            &successors,
            &predecessors,
            "cycle detected in workflow — workflows must be acyclic",
        )?;

        Ok(Workflow {
            nodes: self.nodes,
            levels,
            successors,
            predecessors,
            edge_mappings,
        })
    }
}

// ---------------------------------------------------------------------------
// Workflow
// ---------------------------------------------------------------------------

/// A compiled Workflow DAG with field-level data mapping.
///
/// Nodes are grouped into topological levels and run in parallel within each
/// level. Each node receives a JSON object assembled by applying field
/// mappings from its predecessors' outputs.
pub struct Workflow {
    nodes: HashMap<String, Arc<dyn WorkflowNode>>,
    levels: Vec<Vec<String>>,
    successors: HashMap<String, Vec<String>>,
    predecessors: HashMap<String, Vec<String>>,
    edge_mappings: HashMap<(String, String), Vec<FieldMapping>>,
}

impl std::fmt::Debug for Workflow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Workflow")
            .field("levels", &self.levels)
            .field("node_count", &self.nodes.len())
            .field(
                "edge_count",
                &self.successors.values().map(|v| v.len()).sum::<usize>(),
            )
            .finish()
    }
}

impl Workflow {
    /// Returns a new workflow builder.
    pub fn builder() -> WorkflowBuilder {
        WorkflowBuilder::new()
    }

    /// Assembles the input for a node by applying field mappings from all
    /// predecessors' outputs.
    fn assemble_input(
        &self,
        node: &str,
        outputs: &HashMap<String, serde_json::Value>,
    ) -> serde_json::Value {
        let mut input = serde_json::Map::new();

        let preds = match self.predecessors.get(node) {
            Some(p) => p,
            None => return serde_json::Value::Object(input),
        };

        for pred in preds {
            let pred_output = match outputs.get(pred) {
                Some(o) => o,
                None => continue,
            };

            let mappings = self.edge_mappings.get(&(pred.clone(), node.to_string()));

            match mappings {
                Some(maps) if !maps.is_empty() => {
                    for m in maps {
                        if let Some(val) = pred_output.get(&m.source_field) {
                            input.insert(m.target_field.clone(), val.clone());
                        }
                    }
                }
                _ => {
                    if let serde_json::Value::Object(map) = pred_output {
                        for (k, v) in map {
                            input.insert(k.clone(), v.clone());
                        }
                    }
                }
            }
        }

        serde_json::Value::Object(input)
    }

    /// Executes the workflow.
    ///
    /// `initial_input` is the JSON value that the `START` sentinel "outputs".
    /// Field mappings from `START` edges assemble the first nodes' inputs.
    ///
    /// Returns the assembled output from all edges pointing to `END`.
    #[tracing::instrument(skip_all, fields(levels = self.levels.len()))]
    pub async fn run(&self, initial_input: serde_json::Value) -> Result<serde_json::Value> {
        let mut outputs: HashMap<String, serde_json::Value> = HashMap::new();
        outputs.insert(START.to_string(), initial_input);

        for level in &self.levels {
            let mut runnable: Vec<&str> = Vec::new();

            for name in level {
                if name == START || name == END {
                    continue;
                }
                if self.predecessors.contains_key(name) {
                    let preds = &self.predecessors[name];
                    let all_ready = preds.iter().all(|p| outputs.contains_key(p));
                    if all_ready {
                        runnable.push(name);
                    }
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
                let input = self.assemble_input(name, &outputs);
                let _span = tracing::info_span!("workflow_node", name = %name).entered();
                let output = node.process(input).await?;
                outputs.insert(name.to_string(), output);
            } else {
                let mut join_set = tokio::task::JoinSet::new();

                for &name in &runnable {
                    let node = self.nodes.get(name).cloned().ok_or_else(|| {
                        DaimonError::Orchestration(format!("node '{name}' not found"))
                    })?;
                    let input = self.assemble_input(name, &outputs);
                    let owned_name = name.to_string();
                    join_set.spawn(async move {
                        let output = node.process(input).await?;
                        Ok::<_, DaimonError>((owned_name, output))
                    });
                }

                while let Some(result) = join_set.join_next().await {
                    let (name, output) = result.map_err(|e| {
                        DaimonError::Orchestration(format!("workflow join error: {e}"))
                    })??;
                    outputs.insert(name, output);
                }
            }
        }

        let end_input = self.assemble_input(END, &outputs);
        Ok(end_input)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_linear_field_mapping() {
        let wf = Workflow::builder()
            .node(
                "double",
                FnWorkflowNode::new(|input| {
                    Box::pin(async move {
                        let n = input["n"].as_i64().unwrap_or(0);
                        Ok(json!({ "result": n * 2 }))
                    })
                }),
            )
            .edge(START, "double", &[("value", "n")])
            .edge("double", END, &[("result", "answer")])
            .build()
            .unwrap();

        let out = wf.run(json!({ "value": 21 })).await.unwrap();
        assert_eq!(out["answer"], 42);
    }

    #[tokio::test]
    async fn test_parallel_merge() {
        let wf = Workflow::builder()
            .node(
                "upper",
                FnWorkflowNode::new(|input| {
                    Box::pin(async move {
                        let text = input["text"].as_str().unwrap_or("").to_uppercase();
                        Ok(json!({ "uppercased": text }))
                    })
                }),
            )
            .node(
                "length",
                FnWorkflowNode::new(|input| {
                    Box::pin(async move {
                        let text = input["text"].as_str().unwrap_or("");
                        Ok(json!({ "len": text.len() }))
                    })
                }),
            )
            .node(
                "combine",
                FnWorkflowNode::new(|input| {
                    Box::pin(async move {
                        let upper = input["upper_text"].as_str().unwrap_or("").to_string();
                        let len = input["text_len"].as_i64().unwrap_or(0);
                        Ok(json!({ "summary": format!("{upper} ({len} chars)") }))
                    })
                }),
            )
            .edge(START, "upper", &[("input", "text")])
            .edge(START, "length", &[("input", "text")])
            .edge("upper", "combine", &[("uppercased", "upper_text")])
            .edge("length", "combine", &[("len", "text_len")])
            .edge("combine", END, &[("summary", "result")])
            .build()
            .unwrap();

        let out = wf.run(json!({ "input": "hello" })).await.unwrap();
        assert_eq!(out["result"], "HELLO (5 chars)");
    }

    #[tokio::test]
    async fn test_passthrough_edge() {
        let wf = Workflow::builder()
            .node(
                "echo",
                FnWorkflowNode::new(|input| Box::pin(async move { Ok(input) })),
            )
            .edge_passthrough(START, "echo")
            .edge_passthrough("echo", END)
            .build()
            .unwrap();

        let out = wf.run(json!({ "a": 1, "b": 2 })).await.unwrap();
        assert_eq!(out["a"], 1);
        assert_eq!(out["b"], 2);
    }

    #[tokio::test]
    async fn test_diamond_with_field_mapping() {
        let wf = Workflow::builder()
            .node(
                "left",
                FnWorkflowNode::new(|input| {
                    Box::pin(async move {
                        let n = input["n"].as_i64().unwrap_or(0);
                        Ok(json!({ "val": n + 10 }))
                    })
                }),
            )
            .node(
                "right",
                FnWorkflowNode::new(|input| {
                    Box::pin(async move {
                        let n = input["n"].as_i64().unwrap_or(0);
                        Ok(json!({ "val": n * 2 }))
                    })
                }),
            )
            .node(
                "join",
                FnWorkflowNode::new(|input| {
                    Box::pin(async move {
                        let l = input["left_val"].as_i64().unwrap_or(0);
                        let r = input["right_val"].as_i64().unwrap_or(0);
                        Ok(json!({ "sum": l + r }))
                    })
                }),
            )
            .edge(START, "left", &[("number", "n")])
            .edge(START, "right", &[("number", "n")])
            .edge("left", "join", &[("val", "left_val")])
            .edge("right", "join", &[("val", "right_val")])
            .edge("join", END, &[("sum", "total")])
            .build()
            .unwrap();

        let out = wf.run(json!({ "number": 5 })).await.unwrap();
        assert_eq!(out["total"], 5 + 10 + 5 * 2);
    }

    #[tokio::test]
    async fn test_cycle_detection() {
        let result = Workflow::builder()
            .node(
                "a",
                FnWorkflowNode::new(|_| Box::pin(async { Ok(json!({})) })),
            )
            .node(
                "b",
                FnWorkflowNode::new(|_| Box::pin(async { Ok(json!({})) })),
            )
            .edge(START, "a", &[])
            .edge("a", "b", &[])
            .edge("b", "a", &[])
            .edge("b", END, &[])
            .build();

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("cycle"), "expected cycle error, got: {err}");
    }

    #[tokio::test]
    async fn test_unknown_node_fails() {
        let result = Workflow::builder()
            .edge(START, "nonexistent", &[])
            .edge("nonexistent", END, &[])
            .build();
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_no_edges_fails() {
        let result = Workflow::builder()
            .node(
                "a",
                FnWorkflowNode::new(|_| Box::pin(async { Ok(json!({})) })),
            )
            .build();
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_selective_field_mapping() {
        let wf = Workflow::builder()
            .node(
                "producer",
                FnWorkflowNode::new(|_| Box::pin(async { Ok(json!({ "a": 1, "b": 2, "c": 3 })) })),
            )
            .node(
                "consumer",
                FnWorkflowNode::new(|input| {
                    Box::pin(async move {
                        let keys: Vec<_> = input
                            .as_object()
                            .map(|m| m.keys().cloned().collect())
                            .unwrap_or_default();
                        Ok(json!({ "received_keys": keys }))
                    })
                }),
            )
            .edge(START, "producer", &[])
            .edge("producer", "consumer", &[("a", "x"), ("c", "z")])
            .edge("consumer", END, &[("received_keys", "keys")])
            .build()
            .unwrap();

        let out = wf.run(json!({})).await.unwrap();
        let keys = out["keys"].as_array().unwrap();
        let key_strs: Vec<&str> = keys.iter().filter_map(|v| v.as_str()).collect();
        assert!(key_strs.contains(&"x"));
        assert!(key_strs.contains(&"z"));
        assert!(!key_strs.contains(&"b"));
    }
}
