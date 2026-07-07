# Daimon Orchestration Guide

This document covers the four orchestration primitives in Daimon: **Chain**, **Graph**, **DAG**, and **Workflow**. Each serves a distinct purpose. Choose the right one based on your data flow, branching needs, and parallelism requirements.

---

## Table of Contents

1. [When to Use Which](#when-to-use-which)
2. [Chain](#chain)
3. [Graph](#graph)
4. [DAG](#dag)
5. [Workflow](#workflow)
6. [Combining Orchestration with Multi-Agent](#combining-orchestration-with-multi-agent)

---

## When to Use Which

### Quick Decision Matrix

| Primitive | Use When | Key Characteristics |
|-----------|----------|----------------------|
| **Chain** | Linear A → B → C. No branching. | Sequential steps, single data path |
| **Graph** | Conditional routing, cycles, retry loops, fan-out/fan-in with explicit merge | One node at a time, `NodeOutcome` controls flow |
| **DAG** | Topological execution, no cycles, parallel levels | Independent nodes at same level run concurrently |
| **Workflow** | DAG with field-level data mapping between nodes | Nodes receive JSON assembled from predecessor outputs |

### When to Use Each

- **Chain**: ETL pipelines, sequential processing, translation pipelines, summarize → translate → format. Simplest model; each step receives the previous step's output.

- **Graph**: Complex workflows with routing logic (classify → route to specialist), retry loops (validate → retry or done), parallel branches that merge (fan-out → merge). Supports cycles; use `max_steps` to prevent infinite loops.

- **DAG**: Data pipelines where order matters but parallelism is safe. Fetch data in parallel → merge → analyze. Topological sort at build time; cycle detection fails `build()`.

- **Workflow**: When different nodes need different fields from predecessor outputs. Extract → enrich → format with explicit field mappings (e.g., `body` → `raw_body`). DAG execution with typed data flow.

---

## Chain

A **Chain** executes a sequence of steps sequentially. Each step receives a [`ChainContext`] and produces a new context for the next step. No branching, no parallelism — just A → B → C.

### API Overview

- **Builder**: `Chain::builder()`, `.step()`, `.agent()`, `.transform()`
- **Traits**: [`ChainStep`], [`AgentStep`], [`TransformStep`]
- **Context**: [`ChainContext`] — `text`, `metadata`

### ChainContext

```rust
pub struct ChainContext {
    pub text: String,           // Primary payload; steps read/write this
    pub metadata: HashMap<String, serde_json::Value>,  // Arbitrary key-value data
}
```

- `ChainContext::new(text)` — create with initial text
- `with_metadata(key, value)` — add metadata
- Each step receives the context and returns a new (or mutated) context

### run vs run_with_context

- **`run(input)`** — creates `ChainContext::new(input)` and runs all steps. Use when you only need to pass text.
- **`run_with_context(ctx)`** — runs with a pre-built context. Use when you need to seed metadata or start with a custom state.

### Full Example: Summarize → Translate → Format

```rust
use daimon::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> daimon::Result<()> {
    let summarizer = Arc::new(
        Agent::builder()
            .model(daimon::model::openai::OpenAi::new("gpt-4o"))
            .system_prompt("Summarize the following text in 2-3 sentences.")
            .build()?,
    );

    let translator = Arc::new(
        Agent::builder()
            .model(daimon::model::openai::OpenAi::new("gpt-4o"))
            .system_prompt("Translate the following text to French.")
            .build()?,
    );

    let chain = Chain::builder()
        .name("summarize_and_translate")
        .agent(summarizer)
        .agent(translator)
        .transform(|mut ctx| async move {
            ctx.text = format!("=== Final Output ===\n{}", ctx.text);
            Ok(ctx)
        })
        .build()?;

    let result = chain
        .run("Rust is a systems programming language focused on safety, speed, and concurrency.")
        .await?;

    println!("{}", result.text);
    Ok(())
}
```

### Custom ChainStep

Implement [`ChainStep`] for custom logic:

```rust
use daimon::orchestration::{Chain, ChainContext, ChainStep};
use std::future::Future;
use std::pin::Pin;

struct UppercaseStep;

impl ChainStep for UppercaseStep {
    fn process<'a>(
        &'a self,
        mut ctx: ChainContext,
    ) -> Pin<Box<dyn Future<Output = daimon::Result<ChainContext>> + Send + 'a>> {
        Box::pin(async move {
            ctx.text = ctx.text.to_uppercase();
            Ok(ctx)
        })
    }
}

let chain = Chain::builder()
    .step(UppercaseStep)
    .build()?;
```

---

## Graph

A **Graph** is a directed graph of nodes with conditional routing, cycles, and fan-out/fan-in. Execution walks one node at a time; each node returns a [`NodeOutcome`] that controls the next step. Use `max_steps` to prevent infinite loops in cyclic graphs.

### API Overview

- **Builder**: `Graph::builder()`, `.node()`, `.edge()`, `.conditional_edge()`, `.entry()`, `.max_steps()`
- **Traits**: [`GraphNode`], [`AgentNode`], [`FnNode`]
- **Context**: [`GraphContext`] — `state` (HashMap)
- **Outcome**: [`NodeOutcome`] — `Continue`, `Route(target)`, `FanOut { branches, merge }`, `Done`

### GraphContext

```rust
pub struct GraphContext {
    pub state: HashMap<String, serde_json::Value>,  // Shared key-value state
}
```

- `GraphContext::new()` — empty context
- `with_input(text)` — sets `state["input"]`
- `set(key, value)` / `get(key)` / `get_str(key)` — read/write state

### NodeOutcome

| Variant | Behavior |
|---------|----------|
| `Continue` | Follow edges from this node; first matching conditional edge wins |
| `Route(target)` | Jump directly to `target`, ignoring edges |
| `FanOut { branches, merge }` | Run `branches` in parallel, merge their state, then continue from `merge` |
| `Done` | Stop execution; return current context |

### Full Example: Classify → Route to Specialist → Validate → Retry or Done

```rust
use daimon::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> daimon::Result<()> {
    let classifier = Arc::new(
        Agent::builder()
            .model(daimon::model::openai::OpenAi::new("gpt-4o"))
            .system_prompt("Classify the user query as 'technical' or 'general'. Reply with only that word.")
            .build()?,
    );

    let technical_specialist = Arc::new(
        Agent::builder()
            .model(daimon::model::openai::OpenAi::new("gpt-4o"))
            .system_prompt("You are a technical support specialist. Answer concisely.")
            .build()?,
    );

    let general_specialist = Arc::new(
        Agent::builder()
            .model(daimon::model::openai::OpenAi::new("gpt-4o"))
            .system_prompt("You are a general assistant. Answer concisely.")
            .build()?,
    );

    use daimon::orchestration::{AgentNode, FnNode, Graph, GraphContext, NodeOutcome};

    let graph = Graph::builder()
        .node("classify", AgentNode::new(
            Arc::clone(&classifier),
            "input",
            "classification",
        ))
        .node("technical", AgentNode::new(
            Arc::clone(&technical_specialist),
            "input",
            "output",
        ))
        .node("general", AgentNode::new(
            Arc::clone(&general_specialist),
            "input",
            "output",
        ))
        .node("validate", FnNode::new(|ctx| {
            Box::pin(async move {
                let out = ctx.get_str("output").unwrap_or("");
                let valid = out.len() > 10;
                ctx.set("valid", serde_json::json!(valid));
                if valid {
                    Ok(NodeOutcome::Done)
                } else {
                    Ok(NodeOutcome::Route("classify".to_string()))
                }
            })
        }))
        .conditional_edge("classify", "technical", |ctx| {
            ctx.get_str("classification").unwrap_or("").trim().eq_ignore_ascii_case("technical")
        })
        .conditional_edge("classify", "general", |_| true)
        .edge("technical", "validate")
        .edge("general", "validate")
        .entry("classify")
        .max_steps(25)
        .build()?;

    let ctx = GraphContext::new().with_input("How do I fix a borrow checker error?");
    let result = graph.run(ctx).await?;
    println!("Output: {:?}", result.get("output"));
    Ok(())
}
```

### Fan-Out Example

```rust
// A node that fans out to parallel branches, then merges
let fan_out_node = FnNode::new(|_ctx| {
    Box::pin(async {
        Ok(NodeOutcome::FanOut {
            branches: vec!["branch_a".into(), "branch_b".into()],
            merge: "merge".into(),
        })
    })
});
```

---

## DAG

A **DAG** (Directed Acyclic Graph) uses topological scheduling: nodes whose predecessors have **all** completed run concurrently. No cycles; cycle detection at `build()` time fails. Use [`START`] and [`END`] sentinels for entry/exit.

### API Overview

- **Builder**: `Dag::builder()`, `.node()`, `.edge()`, `.branch()`, `.multi_branch()`
- **Traits**: [`DagNode`], [`AgentDagNode`], [`FnDagNode`]
- **Context**: [`DagContext`] — `state` (HashMap)
- **Sentinels**: `daimon::orchestration::{START, END}`

### DagContext

Same shape as `GraphContext`: `state: HashMap<String, serde_json::Value>`, with `with_input()`, `set()`, `get()`, `get_str()`.

### branch vs multi_branch

- **`branch(from, condition)`** — single-select: condition returns one successor; others are skipped.
- **`multi_branch(from, condition)`** — multi-select: condition returns `Vec<String>` of successors to activate.

### Full Example: Fetch Data (Parallel) → Merge → Analyze

```rust
use daimon::orchestration::{Dag, DagContext, FnDagNode, START, END};

#[tokio::main]
async fn main() -> daimon::Result<()> {
    let dag = Dag::builder()
        .node(
            "fetch_a",
            FnDagNode::new(|ctx| {
                Box::pin(async move {
                    let input = ctx.get_str("input").unwrap_or_default().to_string();
                    ctx.set("data_a", serde_json::json!(format!("data_a:{input}")));
                    Ok(())
                })
            }),
        )
        .node(
            "fetch_b",
            FnDagNode::new(|ctx| {
                Box::pin(async move {
                    let input = ctx.get_str("input").unwrap_or_default().to_string();
                    ctx.set("data_b", serde_json::json!(format!("data_b:{input}")));
                    Ok(())
                })
            }),
        )
        .node(
            "merge",
            FnDagNode::new(|ctx| {
                Box::pin(async move {
                    let a = ctx.get_str("data_a").unwrap_or("").to_string();
                    let b = ctx.get_str("data_b").unwrap_or("").to_string();
                    ctx.set("merged", serde_json::json!(format!("{a} + {b}")));
                    Ok(())
                })
            }),
        )
        .node(
            "analyze",
            FnDagNode::new(|ctx| {
                Box::pin(async move {
                    let merged = ctx.get_str("merged").unwrap_or("").to_string();
                    ctx.set("analysis", serde_json::json!(format!("Analyzed: {merged}")));
                    Ok(())
                })
            }),
        )
        .edge(START, "fetch_a")
        .edge(START, "fetch_b")   // fetch_a and fetch_b run in parallel
        .edge("fetch_a", "merge")
        .edge("fetch_b", "merge")
        .edge("merge", "analyze")
        .edge("analyze", END)
        .build()?;

    let result = dag
        .run(DagContext::new().with_input("query"))
        .await?;

    println!("Analysis: {:?}", result.get("analysis"));
    Ok(())
}
```

### Conditional Branching

```rust
// Route based on context after "router" completes
Dag::builder()
    .node("router", router_node)
    .node("path_a", node_a)
    .node("path_b", node_b)
    .edge(START, "router")
    .edge("router", "path_a")
    .edge("router", "path_b")
    .edge("path_a", END)
    .edge("path_b", END)
    .branch("router", |ctx| {
        if ctx.get_str("input").unwrap_or("") == "b" {
            Ok("path_b".to_string())
        } else {
            Ok("path_a".to_string())
        }
    })
    .build()?;
```

---

## Workflow

A **Workflow** is a DAG with **field-level data mapping** between nodes. Each edge specifies `(source_field, target_field)` pairs. Nodes receive a JSON object assembled from predecessor outputs; they return JSON whose fields can be mapped to successor inputs. Use `workflow::START` and `workflow::END` (different from DAG sentinels).

### API Overview

- **Builder**: `Workflow::builder()`, `.node()`, `.edge(from, to, &[(src_field, dst_field)])`, `.edge_passthrough()`
- **Traits**: [`WorkflowNode`], [`AgentWorkflowNode`], [`FnWorkflowNode`]
- **Data**: Nodes receive and return `serde_json::Value`

### Field Mapping

- `edge(from, to, &[("body", "raw_body"), ("status", "code")])` — source's `body` → target's `raw_body`, source's `status` → target's `code`
- `edge_passthrough(from, to)` — pass all fields through unchanged (identity mapping)

### Full Example: Extract → Enrich → Format with Field Mapping

```rust
use daimon::orchestration::workflow::{Workflow, FnWorkflowNode, START, END};
use serde_json::json;

#[tokio::main]
async fn main() -> daimon::Result<()> {
    let wf = Workflow::builder()
        .node(
            "extract",
            FnWorkflowNode::new(|input| {
                Box::pin(async move {
                    let url = input["url"].as_str().unwrap_or_default();
                    let body = format!("fetched content from {url}");
                    Ok(json!({ "body": body, "status": 200 }))
                })
            }),
        )
        .node(
            "enrich",
            FnWorkflowNode::new(|input| {
                Box::pin(async move {
                    let raw_body = input["raw_body"].as_str().unwrap_or_default();
                    let enriched = format!("[ENRICHED] {raw_body}");
                    Ok(json!({ "enriched": enriched, "length": enriched.len() }))
                })
            }),
        )
        .node(
            "format",
            FnWorkflowNode::new(|input| {
                Box::pin(async move {
                    let text = input["enriched_text"].as_str().unwrap_or_default();
                    let len = input["text_len"].as_i64().unwrap_or(0);
                    Ok(json!({
                        "result": format!("{text} ({len} chars)")
                    }))
                })
            }),
        )
        .edge(START, "extract", &[("url", "url")])
        .edge("extract", "enrich", &[("body", "raw_body")])
        .edge("enrich", "format", &[
            ("enriched", "enriched_text"),
            ("length", "text_len"),
        ])
        .edge("format", END, &[("result", "output")])
        .build()?;

    let output = wf
        .run(json!({ "url": "https://example.com" }))
        .await?;

    println!("Output: {}", output["output"]);
    Ok(())
}
```

### AgentWorkflowNode

`AgentWorkflowNode` reads a field from the input as the prompt and produces `{ "text": "<response>" }`:

```rust
use daimon::orchestration::workflow::{AgentWorkflowNode, Workflow, START, END};

let agent = Arc::new(Agent::builder().model(model).build()?);

let wf = Workflow::builder()
    .node("summarize", AgentWorkflowNode::new(Arc::clone(&agent), "prompt"))
    .edge(START, "summarize", &[("input_text", "prompt")])
    .edge("summarize", END, &[("text", "summary")])
    .build()?;
```

---

## Combining Orchestration with Multi-Agent

You can use agents, supervisors, or handoff networks **as nodes** in orchestration graphs.

### Agents as Chain Steps

```rust
let summarizer = Arc::new(Agent::builder().model(model).build()?);
let chain = Chain::builder()
    .agent(summarizer)
    .build()?;
```

### Agents as Graph Nodes

```rust
use daimon::orchestration::AgentNode;

let specialist = Arc::new(Agent::builder().model(model).build()?);
Graph::builder()
    .node("specialist", AgentNode::new(specialist, "input", "output"))
    .edge("classify", "specialist")
    .build()?;
```

### Supervisor as a Graph Node

Wrap a `Supervisor` in a custom `GraphNode`:

```rust
use daimon::orchestration::{FnNode, Graph, GraphContext, NodeOutcome};
use std::sync::Arc;

let supervisor = Arc::new(
    Supervisor::builder()
        .model(model)
        .agent("researcher", research_agent, "Performs research")
        .agent("writer", writer_agent, "Writes prose")
        .build()?,
);

let supervisor_node = FnNode::new(move |ctx| {
    let sup = Arc::clone(&supervisor);
    Box::pin(async move {
        let input = ctx.get_str("input").unwrap_or("").to_string();
        let response = sup.run(&input).await?;
        ctx.set("output", serde_json::Value::String(response.final_text));
        Ok(NodeOutcome::Continue)
    })
});

Graph::builder()
    .node("supervisor", supervisor_node)
    .edge("router", "supervisor")
    .build()?;
```

### Agent-as-Tool in Orchestration

A coordinator agent can use `AgentTool` to delegate to sub-agents. That coordinator can then be used as a Chain step or Graph node:

```rust
use daimon::agent::as_tool::AgentTool;

let research_agent = Arc::new(Agent::builder().model(model).build()?);
let coordinator = Agent::builder()
    .model(model)
    .tool(AgentTool::new(research_agent, "research", "Perform research on a topic"))
    .build()?;

let chain = Chain::builder()
    .agent(Arc::new(coordinator))
    .build()?;
```

### Handoff Network as a Node

A `HandoffNetwork` routes between agents. Wrap it in a `FnNode` to use it inside a Graph:

```rust
use daimon::agent::handoff::HandoffNetwork;

let handoff = HandoffNetwork::builder()
    .agent("a", agent_a)
    .agent("b", agent_b)
    .default_agent("a")
    .build()?;

let handoff_node = FnNode::new(move |ctx| {
    let h = handoff.clone();
    Box::pin(async move {
        let input = ctx.get_str("input").unwrap_or("").to_string();
        let response = h.run(&input).await?;
        ctx.set("output", serde_json::Value::String(response.final_text));
        Ok(NodeOutcome::Done)
    })
});
```

---

## Summary

| Primitive | Execution Model | Branching | Parallelism | Data Flow |
|-----------|-----------------|-----------|-------------|-----------|
| **Chain** | Sequential steps | None | None | `ChainContext` (text + metadata) |
| **Graph** | One node at a time, outcome-driven | Conditional edges, `Route`, `FanOut` | Explicit `FanOut` | `GraphContext.state` |
| **DAG** | Topological levels | `branch` / `multi_branch` | Same-level nodes in parallel | `DagContext.state` |
| **Workflow** | Topological levels | None (acyclic) | Same-level nodes in parallel | Field-mapped JSON per node |

Choose **Chain** for simple pipelines, **Graph** for routing and cycles, **DAG** for parallel data pipelines, and **Workflow** when you need explicit field mappings between nodes.
