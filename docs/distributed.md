# Distributed Execution

Daimon supports distributed agent execution across multiple processes via a broker-worker pattern. Tasks are submitted to a queue, workers pull and execute them, and results are reported back. Checkpointing persists agent state across process boundaries for resumable runs and time-travel debugging.

---

## Architecture Overview

Distributed execution in Daimon uses four core abstractions:

| Component | Role |
|-----------|------|
| **TaskBroker** | Queue for agent tasks — submit, receive, complete, fail |
| **TaskWorker** | Pulls tasks, runs agent, reports results |
| **Checkpoint** | Persists agent state across process boundaries |
| **TaskEventBus** | Streams events from distributed workers (optional) |

```
┌─────────────┐  submit    ┌─────────────┐  receive   ┌─────────────┐
│  Producer   │ ────────►  │ TaskBroker  │  ───────►  │ TaskWorker  │
└─────────────┘            └─────────────┘            └─────────────┘
                                  ▲                          │
                                  │     complete / fail      │
                                  └──────────────────────────┘
```

Producers submit `AgentTask` instances; workers block on `receive()`, run the agent, and call `complete()` or `fail()`. Status can be polled via `status(task_id)`.

---

## Task Broker Implementations

### InProcessBroker

Tokio MPSC channels. Single process. Ideal for testing and single-process parallelism.

```rust
use daimon::distributed::{InProcessBroker, TaskBroker, AgentTask};

let broker = InProcessBroker::new(64);
let task_id = broker.submit(AgentTask::new("Summarize this")).await?;
```

`InProcessBroker::new(capacity)` — capacity is the channel buffer size. Clone-friendly; all clones share the same underlying state.

---

### RedisBroker (feature = "redis")

Redis Lists for the queue, Redis Hashes for status and results. Multi-process.

```rust
use daimon::distributed::{RedisBroker, TaskBroker, AgentTask};

let broker = RedisBroker::new("redis://127.0.0.1/", "daimon:tasks").await?;
let task_id = broker.submit(AgentTask::new("Summarize this")).await?;
```

- `RedisBroker::new(url, prefix)` — `url` is the Redis connection URL; `prefix` is the key prefix (e.g. `daimon:tasks`).
- Keys: `{prefix}:queue`, `{prefix}:status`, `{prefix}:results`.

---

### NatsBroker (feature = "nats")

NATS JetStream with durable pull consumers. At-least-once delivery.

```rust
use daimon::distributed::{NatsBroker, TaskBroker, AgentTask};

let broker = NatsBroker::connect("nats://127.0.0.1:4222", "daimon-tasks").await?;
let task_id = broker.submit(AgentTask::new("Summarize this")).await?;
```

- `NatsBroker::connect(url, stream_name)` — creates or reuses a JetStream stream with WorkQueue retention.
- Durable consumer `daimon-worker` with explicit ack.

---

### AmqpBroker (feature = "amqp")

RabbitMQ via AMQP 0-9-1. Durable queue, manual ack.

```rust
use daimon::distributed::{AmqpBroker, TaskBroker, AgentTask};

let broker = AmqpBroker::connect("amqp://guest:guest@127.0.0.1:5672", "daimon-tasks").await?;
let task_id = broker.submit(AgentTask::new("Summarize this")).await?;
```

- `AmqpBroker::connect(url, queue_name)` — declares a durable queue if it doesn't exist.

---

### GrpcBrokerServer / GrpcBrokerClient (feature = "grpc")

gRPC transport. Server wraps any broker; client connects remotely.

**Server:**

```rust
use daimon::distributed::{InProcessBroker, GrpcBrokerServer};

let broker = InProcessBroker::new(64);
GrpcBrokerServer::new(broker)
    .serve("[::1]:50051")
    .await?;
```

**Client:**

```rust
use daimon::distributed::{GrpcBrokerClient, TaskBroker, AgentTask};

let client = GrpcBrokerClient::connect("http://[::1]:50051").await?;
let task_id = client.submit(AgentTask::new("Hello")).await?;
let status = client.status(&task_id).await?;
```

Note: `receive()` is not supported on `GrpcBrokerClient`. Workers must run on the server side, connected to the underlying broker.

---

### Cloud-Native Brokers

| Broker | Crate | Feature | Constructor |
|--------|-------|---------|-------------|
| **SqsBroker** | daimon-provider-bedrock | sqs | `SqsBroker::new(queue_url).await?` |
| **PubSubBroker** | daimon-provider-gemini | pubsub | `PubSubBroker::with_api_key(project, topic, sub, api_key)` or `with_bearer_token(...)` |
| **ServiceBusBroker** | daimon-provider-azure | servicebus | `ServiceBusBroker::new(namespace_url, queue_name, sas_token)` |

**SqsBroker** — AWS SQS. Uses visibility timeout for in-flight tracking. For FIFO queues, uses `message_group_id`.

```rust
use daimon_provider_bedrock::SqsBroker;
use daimon_core::distributed::{TaskBroker, AgentTask};

let broker = SqsBroker::new("https://sqs.us-east-1.amazonaws.com/123456789/daimon-tasks").await?;
broker.submit(AgentTask::new("Summarize")).await?;
```

**PubSubBroker** — Google Cloud Pub/Sub. Base64-encoded JSON in message bodies.

```rust
use daimon_provider_gemini::PubSubBroker;

let broker = PubSubBroker::with_api_key(
    "my-project",
    "daimon-tasks",
    "daimon-worker-sub",
    "api-key",
);
```

**ServiceBusBroker** — Azure Service Bus REST API. Peek-lock for receive; delete on complete.

```rust
use daimon_provider_azure::ServiceBusBroker;

let broker = ServiceBusBroker::new(
    "https://my-ns.servicebus.windows.net",
    "daimon-tasks",
    "SharedAccessKey...",
);
```

---

## TaskWorker

`TaskWorker` pulls tasks from a broker and executes them using agent instances from a factory.

```rust
use daimon::distributed::{TaskWorker, InProcessBroker, AgentTask};
use daimon::Agent;

let broker = InProcessBroker::new(64);
let worker = TaskWorker::new(broker.clone(), || {
    Agent::builder()
        .model(my_model)
        .build()
        .unwrap()
});

// Single task
let result = worker.run_once().await?;

// Indefinite loop
worker.run().await?;

// Parallel workers (up to N concurrent tasks)
worker.run_parallel(4).await?;
```

### run_once()

Waits for one task, executes it, reports the result. Returns `Ok(None)` if the broker is closed.

### Agent factory pattern

The factory closure `|| Agent { ... }` is called **once per task**. Each task gets a fresh agent so:

- Conversations do not bleed across tasks
- Memory is isolated per task
- No shared mutable state between tasks

### Worker loop example

```rust
loop {
    match worker.run_once().await? {
        Some(result) => {
            tracing::info!(task_id = %result.task_id, "completed: {} iterations", result.iterations);
        }
        None => {
            tracing::info!("broker closed, exiting");
            break;
        }
    }
}
```

---

## StreamingTaskWorker

`StreamingTaskWorker` uses `Agent::prompt_stream()` and publishes events through a `TaskEventBus`. Use when you need real-time visibility into agent execution across processes.

```rust
use daimon::distributed::streaming::*;
use daimon::distributed::{StreamingTaskWorker, InProcessBroker, AgentTask};

let broker = InProcessBroker::new(64);
let bus = InProcessEventBus::new(64);

let worker = StreamingTaskWorker::new(broker.clone(), bus.clone(), || {
    Agent::builder().model(my_model).build().unwrap()
});

// Subscribe before submitting
let mut rx = bus.subscribe();

// Run worker in background
tokio::spawn(async move { worker.run().await });

// Submit and receive live events
let task_id = broker.submit(AgentTask::new("Summarize")).await?;

while let Ok(evt) = rx.recv().await {
    if evt.task_id == task_id {
        match &evt.event {
            SerializableStreamEvent::TextDelta(t) => print!("{t}"),
            SerializableStreamEvent::ToolCallStart { name, .. } => println!("Calling {name}"),
            SerializableStreamEvent::Done => break,
            _ => {}
        }
    }
}
```

### InProcessEventBus

`InProcessEventBus::new(capacity)` — tokio broadcast channel. All subscribers receive every event. For cross-process streaming, implement `TaskEventBus` over Redis Pub/Sub, NATS, etc.

### SerializableStreamEvent

Wire format for stream events:

- `TextDelta(String)`
- `ToolCallStart { id, name }`
- `ToolCallDelta { id, arguments_delta }`
- `ToolCallEnd { id }`
- `ToolResult { id, content, is_error }`
- `Usage { iteration, input_tokens, output_tokens, estimated_cost }`
- `Error(String)`
- `Done`

---

## Checkpoint & State Persistence

### Checkpoint Trait

```rust
pub trait Checkpoint: Send + Sync {
    async fn save(&self, state: &CheckpointState) -> Result<()>;
    async fn load(&self, run_id: &str) -> Result<Option<CheckpointState>>;
    async fn list_runs(&self) -> Result<Vec<String>>;
    async fn delete(&self, run_id: &str) -> Result<()>;
}
```

### Implementations

| Implementation | Feature | Use Case |
|----------------|---------|----------|
| `InMemoryCheckpoint` | (built-in) | Ephemeral, testing |
| `FileCheckpoint::new(dir)` | (built-in) | JSON files on disk |
| `RedisCheckpoint::new(url, prefix)` | redis | Shared, multi-process |
| `NatsKvCheckpoint::connect(url, bucket)` | nats | JetStream KV |

```rust
use daimon::checkpoint::{InMemoryCheckpoint, FileCheckpoint, RedisCheckpoint, NatsKvCheckpoint};

let mem = InMemoryCheckpoint::new();
let file = FileCheckpoint::new("/var/daimon/checkpoints");
let redis = RedisCheckpoint::new("redis://127.0.0.1/", "daimon:checkpoints").await?;
let nats = NatsKvCheckpoint::connect("nats://127.0.0.1:4222", "daimon-checkpoints").await?;
```

### CheckpointState

```rust
pub struct CheckpointState {
    pub run_id: String,
    pub messages: Vec<Message>,
    pub iteration: usize,
    pub completed: bool,
    pub metadata: HashMap<String, serde_json::Value>,
    pub created_at: u64,
    /// Cost accumulated so far — restored on resume so budgets survive restarts.
    #[serde(default)]
    pub cumulative_cost: f64,
    /// Token usage accumulated so far.
    #[serde(default)]
    pub usage: Usage,
}
```

### CheckpointSync

Write-through to local + remote. Load prefers local, falls back to remote and backfills local.

```rust
use daimon::checkpoint::{CheckpointSync, ErasedCheckpoint, FileCheckpoint, InMemoryCheckpoint};
use std::sync::Arc;

let local = InMemoryCheckpoint::new();
let remote = FileCheckpoint::new("/shared/nfs/checkpoints");
let synced = CheckpointSync::new(local, remote);

// Use synced as the checkpoint backend. Checkpoints are supplied per run
// (there is no builder-level `.checkpoint(...)`):
let agent = Agent::builder()
    .model(model)
    .build()?;

let checkpoint: Arc<dyn ErasedCheckpoint> = Arc::new(synced);
let response = agent
    .prompt_resumable("Summarize the quarter", "run-42", &checkpoint)
    .await?;
```

### CheckpointReplicator

Background task that periodically pulls from remote into local.

```rust
use daimon::checkpoint::{CheckpointReplicator, ErasedCheckpoint, FileCheckpoint, InMemoryCheckpoint};
use std::sync::Arc;

let local = Arc::new(InMemoryCheckpoint::new());
let remote = Arc::new(FileCheckpoint::new("/shared/checkpoints"));

let replicator = CheckpointReplicator::new(
    local.clone() as Arc<dyn ErasedCheckpoint>,
    remote.clone() as Arc<dyn ErasedCheckpoint>,
    std::time::Duration::from_secs(30),
);

tokio::spawn(replicator.run());
```

### Bulk sync

- `CheckpointSync::pull_all()` — pull all remote checkpoints into local
- `CheckpointSync::push_all()` — push all local checkpoints to remote

---

## Time-Travel Debugging

### inspect_run

Reconstructs an execution trace from checkpoint data.

```rust
use daimon::checkpoint::{inspect_run, ExecutionTrace, RunSummary};

let trace: ExecutionTrace = inspect_run(checkpoint, run_id).await?;

println!("Run {}: {} iterations, completed={}", trace.run_id, trace.total_iterations, trace.completed);
for step in &trace.steps {
    println!("  Iteration {}: {} tool calls", step.iteration, step.tool_calls.len());
}
```

### list_runs

Lists all checkpointed runs with metadata.

```rust
let summaries: Vec<RunSummary> = list_runs(checkpoint).await?;
for s in &summaries {
    println!("{}: iter={}, completed={}, messages={}", s.run_id, s.iteration, s.completed, s.message_count);
}
```

### ExecutionTrace

- `run_id`, `steps`, `completed`, `total_iterations`
- `TraceStep`: `iteration`, `messages`, `tool_calls`, `response_text`, `usage`
- `final_text()`, `total_tool_calls()` helpers

---

## Agent Replay

Re-run an agent from a previous checkpoint with the current agent config (model, tools, system prompt). Useful for "what-if" debugging.

```rust
// Replay from the beginning of the checkpoint
let response = agent.replay(run_id, &checkpoint, None).await?;

// Replay from a specific iteration (truncate messages to that point)
let response = agent.replay(run_id, &checkpoint, Some(3)).await?;
```

- `from_iteration: None` — replay from the start of the checkpoint's message history
- `from_iteration: Some(n)` — truncate to iteration `n` and re-run

Modify the agent (tools, system prompt, model) before calling `replay` to see how the outcome changes.

---

## Choosing a Broker

| Scenario | Broker |
|---------|--------|
| Single process / testing | `InProcessBroker` |
| Redis already in stack | `RedisBroker` |
| Event streaming, durable delivery | `NatsBroker` |
| Enterprise messaging | `AmqpBroker` |
| Microservices, RPC-style | `GrpcBrokerClient` / `GrpcBrokerServer` |
| AWS native | `SqsBroker` |
| GCP native | `PubSubBroker` |
| Azure native | `ServiceBusBroker` |

---

## Full Example: Redis

```rust
use daimon::distributed::{RedisBroker, TaskWorker, TaskBroker, AgentTask, TaskStatus};
use daimon::Agent;

#[tokio::main]
async fn main() -> daimon::error::Result<()> {
    tracing_subscriber::fmt::init();

    let broker = RedisBroker::new("redis://127.0.0.1/", "daimon:tasks").await?;
    let broker_clone = broker.clone();

    let worker = TaskWorker::new(broker_clone, || {
        Agent::builder()
            .model(/* your model */)
            .build()
            .unwrap()
    });

    // Producer: submit tasks
    let task_id = broker
        .submit(AgentTask::new("Summarize the key points of distributed systems"))
        .await?;
    println!("Submitted task: {}", task_id);

    // Worker: process (run in same process for demo; in production, run in separate process)
    if let Some(result) = worker.run_once().await? {
        println!("Completed: {} iterations, output: {}", result.iterations, result.output);
    }

    // Poll status
    let status = broker.status(&task_id).await?;
    match &status {
        TaskStatus::Completed(r) => println!("Result: {}", r.output),
        TaskStatus::Failed(e) => eprintln!("Failed: {}", e),
        _ => println!("Still pending or running"),
    }

    Ok(())
}
```

---

## Full Example: NATS

```rust
use daimon::distributed::{NatsBroker, TaskWorker, TaskBroker, AgentTask};
use daimon::Agent;

#[tokio::main]
async fn main() -> daimon::error::Result<()> {
    tracing_subscriber::fmt::init();

    let broker = NatsBroker::connect("nats://127.0.0.1:4222", "daimon-tasks").await?;
    let broker_clone = broker.clone();

    let worker = TaskWorker::new(broker_clone, || {
        Agent::builder()
            .model(/* your model */)
            .build()
            .unwrap()
    });

    // Submit
    let task_id = broker.submit(AgentTask::new("What is NATS?")).await?;
    println!("Submitted: {}", task_id);

    // Worker loop
    loop {
        match worker.run_once().await? {
            Some(result) => {
                println!("Task {}: {} iterations", result.task_id, result.iterations);
                if let Some(e) = &result.error {
                    eprintln!("Error: {}", e);
                }
            }
            None => {
                tracing::info!("No more tasks (broker closed)");
                break;
            }
        }
    }

    Ok(())
}
```

---

## Cargo Features

Enable the broker you need:

```toml
[dependencies]
daimon = { version = "0.23", features = ["redis"] }   # RedisBroker
daimon = { version = "0.23", features = ["nats"] }     # NatsBroker
daimon = { version = "0.23", features = ["amqp"] }     # AmqpBroker
daimon = { version = "0.23", features = ["grpc"] }     # GrpcBrokerServer/Client
daimon = { version = "0.23", features = ["sqs"] }      # SqsBroker (via daimon-provider-bedrock)
daimon = { version = "0.23", features = ["pubsub"] }   # PubSubBroker (via daimon-provider-gemini)
daimon = { version = "0.23", features = ["servicebus"] }  # ServiceBusBroker (via daimon-provider-azure)
```

InProcessBroker is always available (no feature).
