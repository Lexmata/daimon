//! Benchmarks for the Daimon agent framework.
//!
//! Run with: `cargo bench`

use criterion::{Criterion, criterion_group, criterion_main};

use daimon::agent::Agent;
use daimon::error::Result;
use daimon::memory::{SlidingWindowMemory, TokenWindowMemory};
use daimon::model::Model;
use daimon::model::types::{ChatRequest, ChatResponse, Message, Role, StopReason, Usage};
use daimon::orchestration::chain::{Chain, TransformStep};
use daimon::orchestration::dag::{Dag, DagContext, END, FnDagNode, START};
use daimon::stream::ResponseStream;
use daimon::tool::{Tool, ToolOutput, ToolRegistry};

// ---------------------------------------------------------------------------
// Mock model (instant response, no network)
// ---------------------------------------------------------------------------

struct InstantModel;

impl Model for InstantModel {
    async fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
        let reply = if !request.tools.is_empty()
            && request
                .messages
                .last()
                .is_some_and(|m| m.role == Role::User)
        {
            ChatResponse {
                message: Message::assistant("done"),
                stop_reason: StopReason::EndTurn,
                usage: Some(Usage {
                    input_tokens: 100,
                    output_tokens: 50,
                    cached_tokens: 0,
                }),
            }
        } else {
            ChatResponse {
                message: Message::assistant("response"),
                stop_reason: StopReason::EndTurn,
                usage: Some(Usage {
                    input_tokens: 100,
                    output_tokens: 50,
                    cached_tokens: 0,
                }),
            }
        };
        Ok(reply)
    }

    async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

struct NoOpTool;

impl Tool for NoOpTool {
    fn name(&self) -> &str {
        "noop"
    }
    fn description(&self) -> &str {
        "does nothing"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object", "properties": {} })
    }
    async fn execute(&self, _input: &serde_json::Value) -> Result<ToolOutput> {
        Ok(ToolOutput::text("ok"))
    }
}

// ---------------------------------------------------------------------------
// Agent benchmarks
// ---------------------------------------------------------------------------

fn bench_agent_prompt(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let agent = Agent::builder()
        .model(InstantModel)
        .system_prompt("You are helpful.")
        .build()
        .unwrap();

    c.bench_function("agent_prompt_simple", |b| {
        b.iter(|| {
            rt.block_on(async {
                let _ = agent.prompt("hello").await.unwrap();
            })
        })
    });
}

fn bench_agent_with_tools(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let agent = Agent::builder()
        .model(InstantModel)
        .tool(NoOpTool)
        .build()
        .unwrap();

    c.bench_function("agent_prompt_with_tools", |b| {
        b.iter(|| {
            rt.block_on(async {
                let _ = agent.prompt("hello").await.unwrap();
            })
        })
    });
}

// ---------------------------------------------------------------------------
// Memory benchmarks
// ---------------------------------------------------------------------------

fn bench_sliding_window_memory(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("memory_sliding_window_50", |b| {
        b.iter(|| {
            rt.block_on(async {
                use daimon::memory::Memory;
                let mem = SlidingWindowMemory::new(50);
                for i in 0..100 {
                    mem.add_message(&Message::user(format!("msg {i}")))
                        .await
                        .unwrap();
                }
                let msgs = mem.get_messages().await.unwrap();
                assert_eq!(msgs.len(), 50);
            })
        })
    });
}

fn bench_token_window_memory(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("memory_token_window_1000", |b| {
        b.iter(|| {
            rt.block_on(async {
                use daimon::memory::Memory;
                let mem = TokenWindowMemory::new(1000);
                for i in 0..100 {
                    mem.add_message(&Message::user(format!(
                        "message number {i} with some content"
                    )))
                    .await
                    .unwrap();
                }
                let msgs = mem.get_messages().await.unwrap();
                assert!(msgs.len() <= 100);
            })
        })
    });
}

// ---------------------------------------------------------------------------
// Tool registry benchmarks
// ---------------------------------------------------------------------------

fn bench_tool_registry(c: &mut Criterion) {
    let mut registry = ToolRegistry::new();
    for i in 0..50 {
        registry.register(NamedTool(format!("tool_{i}"))).unwrap();
    }

    c.bench_function("tool_registry_lookup_50", |b| {
        b.iter(|| {
            let _ = registry.get("tool_25");
        })
    });

    c.bench_function("tool_registry_specs_50_uncached", |b| {
        b.iter(|| {
            let _ = registry.tool_specs();
        })
    });

    registry.warm_cache();

    c.bench_function("tool_registry_specs_50_cached", |b| {
        b.iter(|| {
            let _ = registry.tool_specs();
        })
    });

    c.bench_function("tool_registry_validate_cached", |b| {
        let input = serde_json::json!({"key": "value"});
        b.iter(|| {
            let _ = registry.validate_input("tool_25", &input);
        })
    });
}

struct NamedTool(String);

impl Tool for NamedTool {
    fn name(&self) -> &str {
        &self.0
    }
    fn description(&self) -> &str {
        "a tool"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object" })
    }
    async fn execute(&self, _input: &serde_json::Value) -> Result<ToolOutput> {
        Ok(ToolOutput::text("ok"))
    }
}

// ---------------------------------------------------------------------------
// Orchestration benchmarks
// ---------------------------------------------------------------------------

fn bench_dag_fan_out(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let dag = Dag::builder()
        .node(
            "a",
            FnDagNode::new(|ctx| {
                Box::pin(async move {
                    ctx.set("a", serde_json::json!(1));
                    Ok(())
                })
            }),
        )
        .node(
            "b",
            FnDagNode::new(|ctx| {
                Box::pin(async move {
                    ctx.set("b", serde_json::json!(2));
                    Ok(())
                })
            }),
        )
        .node(
            "c",
            FnDagNode::new(|ctx| {
                Box::pin(async move {
                    ctx.set("c", serde_json::json!(3));
                    Ok(())
                })
            }),
        )
        .node(
            "merge",
            FnDagNode::new(|ctx| {
                Box::pin(async move {
                    let sum = ctx.get("a").and_then(|v| v.as_i64()).unwrap_or(0)
                        + ctx.get("b").and_then(|v| v.as_i64()).unwrap_or(0)
                        + ctx.get("c").and_then(|v| v.as_i64()).unwrap_or(0);
                    ctx.set("sum", serde_json::json!(sum));
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

    c.bench_function("dag_fan_out_3_merge", |b| {
        b.iter(|| {
            rt.block_on(async {
                let _ = dag.run(DagContext::new()).await.unwrap();
            })
        })
    });
}

fn bench_chain_3_steps(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let chain = Chain::builder()
        .step(TransformStep::new(|mut ctx| {
            Box::pin(async move {
                ctx.metadata
                    .insert("step1".to_string(), serde_json::json!(true));
                Ok(ctx)
            })
        }))
        .step(TransformStep::new(|mut ctx| {
            Box::pin(async move {
                ctx.metadata
                    .insert("step2".to_string(), serde_json::json!(true));
                Ok(ctx)
            })
        }))
        .step(TransformStep::new(|mut ctx| {
            Box::pin(async move {
                ctx.metadata
                    .insert("step3".to_string(), serde_json::json!(true));
                Ok(ctx)
            })
        }))
        .build()
        .unwrap();

    c.bench_function("chain_3_transforms", |b| {
        b.iter(|| {
            rt.block_on(async {
                let _ = chain.run("input").await.unwrap();
            })
        })
    });
}

// ---------------------------------------------------------------------------
// Token counting benchmark
// ---------------------------------------------------------------------------

fn bench_token_estimation(c: &mut Criterion) {
    let text = "The quick brown fox jumps over the lazy dog. ".repeat(100);

    c.bench_function("token_estimate_4500_chars", |b| {
        b.iter(|| {
            let _estimate = text.len().div_ceil(4);
        })
    });
}

// ---------------------------------------------------------------------------
// Hot-swap agent benchmarks
// ---------------------------------------------------------------------------

fn bench_hot_swap_prompt(c: &mut Criterion) {
    use daimon::agent::hot_swap::HotSwapAgent;

    let rt = tokio::runtime::Runtime::new().unwrap();

    let agent = Agent::builder()
        .model(InstantModel)
        .system_prompt("You are helpful.")
        .build()
        .unwrap();
    let hot = HotSwapAgent::new(agent);

    c.bench_function("hot_swap_prompt_simple", |b| {
        b.iter(|| {
            rt.block_on(async {
                let _ = hot.prompt("hello").await.unwrap();
            })
        })
    });
}

fn bench_hot_swap_swap_model(c: &mut Criterion) {
    use daimon::agent::hot_swap::HotSwapAgent;

    let rt = tokio::runtime::Runtime::new().unwrap();

    let agent = Agent::builder().model(InstantModel).build().unwrap();
    let hot = HotSwapAgent::new(agent);

    c.bench_function("hot_swap_swap_model", |b| {
        b.iter(|| {
            rt.block_on(async {
                hot.swap_model(InstantModel).await;
            })
        })
    });
}

// ---------------------------------------------------------------------------
// InProcessBroker benchmarks
// ---------------------------------------------------------------------------

fn bench_in_process_broker(c: &mut Criterion) {
    use daimon::distributed::{AgentTask, InProcessBroker, TaskBroker, TaskResult};

    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("broker_submit_receive_complete", |b| {
        b.iter(|| {
            rt.block_on(async {
                let broker = InProcessBroker::new(64);
                let task = AgentTask::new("bench input");
                let id = broker.submit(task).await.unwrap();
                let received = broker.receive().await.unwrap().unwrap();
                assert_eq!(received.task_id, id);
                let result = TaskResult {
                    task_id: id,
                    output: "done".into(),
                    iterations: 1,
                    cost: 0.0,
                    error: None,
                };
                broker.complete(&received.task_id, result).await.unwrap();
            })
        })
    });
}

// ---------------------------------------------------------------------------
// Streaming event bus benchmarks
// ---------------------------------------------------------------------------

fn bench_event_bus(c: &mut Criterion) {
    use daimon::distributed::streaming::{
        InProcessEventBus, SerializableStreamEvent, TaskEventBus, TaskStreamEvent,
    };

    let rt = tokio::runtime::Runtime::new().unwrap();

    let bus = InProcessEventBus::new(1024);
    let mut rx = bus.subscribe();

    c.bench_function("event_bus_publish_receive", |b| {
        b.iter(|| {
            rt.block_on(async {
                bus.publish(TaskStreamEvent {
                    task_id: "t1".into(),
                    event: SerializableStreamEvent::TextDelta("chunk".into()),
                })
                .await
                .unwrap();
                let _ = rx.recv().await.unwrap();
            })
        })
    });
}

// ---------------------------------------------------------------------------
// Checkpoint benchmarks
// ---------------------------------------------------------------------------

fn bench_checkpoint_memory(c: &mut Criterion) {
    use daimon::checkpoint::{Checkpoint, CheckpointState, InMemoryCheckpoint};
    use daimon::model::types::Message;

    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("checkpoint_save_load_memory", |b| {
        b.iter(|| {
            rt.block_on(async {
                let cp = InMemoryCheckpoint::new();
                let state = CheckpointState::new(
                    "run1",
                    vec![Message::user("hello"), Message::assistant("hi")],
                    1,
                );
                cp.save(&state).await.unwrap();
                let loaded = cp.load("run1").await.unwrap().unwrap();
                assert_eq!(loaded.messages.len(), 2);
            })
        })
    });
}

// ---------------------------------------------------------------------------
// Serialization benchmarks
// ---------------------------------------------------------------------------

fn bench_serializable_stream_event(c: &mut Criterion) {
    use daimon::distributed::streaming::SerializableStreamEvent;

    let event = SerializableStreamEvent::Usage {
        iteration: 3,
        input_tokens: 512,
        output_tokens: 128,
        estimated_cost: 0.0023,
    };

    c.bench_function("stream_event_serialize", |b| {
        b.iter(|| {
            let _ = serde_json::to_string(&event).unwrap();
        })
    });

    let json = serde_json::to_string(&event).unwrap();
    c.bench_function("stream_event_deserialize", |b| {
        b.iter(|| {
            let _: SerializableStreamEvent = serde_json::from_str(&json).unwrap();
        })
    });
}

// ---------------------------------------------------------------------------
// Groups
// ---------------------------------------------------------------------------

criterion_group!(agent_benches, bench_agent_prompt, bench_agent_with_tools,);

criterion_group!(
    memory_benches,
    bench_sliding_window_memory,
    bench_token_window_memory,
);

criterion_group!(tool_benches, bench_tool_registry,);

criterion_group!(
    orchestration_benches,
    bench_dag_fan_out,
    bench_chain_3_steps,
);

criterion_group!(misc_benches, bench_token_estimation,);

criterion_group!(
    new_component_benches,
    bench_hot_swap_prompt,
    bench_hot_swap_swap_model,
    bench_in_process_broker,
    bench_event_bus,
    bench_checkpoint_memory,
    bench_serializable_stream_event,
);

criterion_main!(
    agent_benches,
    memory_benches,
    tool_benches,
    orchestration_benches,
    misc_benches,
    new_component_benches,
);
