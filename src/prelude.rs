//! Convenience re-exports for common Daimon types.
//!
//! Use `use daimon::prelude::*` to bring in [`Agent`], [`AgentResponse`], [`Model`],
//! [`Tool`], [`Memory`], [`StreamEvent`], and related types without qualifying paths.

pub use crate::a2a::{A2aClient, A2aHandler, A2aMessage, A2aTask, AgentCard};
pub use crate::agent::as_tool::AgentTool;
pub use crate::agent::handoff::HandoffNetwork;
pub use crate::agent::structured::StructuredOutput;
pub use crate::agent::supervisor::Supervisor;
pub use crate::agent::{Agent, AgentResponse};
pub use crate::checkpoint::{
    Checkpoint, CheckpointReplicator, CheckpointState, CheckpointSync, ExecutionTrace,
    FileCheckpoint, InMemoryCheckpoint, inspect_run, list_runs,
};
pub use crate::cost::{AnthropicCostModel, CostModel, CostTracker, OpenAiCostModel};
pub use crate::agent::fork::ForkBuilder;
pub use crate::agent::hot_swap::HotSwapAgent;
pub use crate::distributed::{
    AgentTask, InProcessBroker, InProcessEventBus, StreamingTaskWorker, TaskBroker, TaskEventBus,
    TaskResult, TaskStatus, TaskStreamEvent, TaskWorker,
};
pub use crate::error::{DaimonError, Result};
pub use crate::guardrails::{
    ContentPolicyGuardrail, GuardrailResult, InputGuardrail, MaxTokenGuardrail,
    OutputGuardrail, RegexFilterGuardrail,
};
pub use crate::hooks::AgentHook;
pub use crate::memory::{Memory, SlidingWindowMemory, SummaryMemory, TokenWindowMemory};
pub use crate::middleware::{Middleware, MiddlewareAction, MiddlewareStack};
pub use crate::model::{EmbeddingModel, Model};
pub use crate::model::types::{ChatRequest, ChatResponse, Message, Role, Usage};
pub use crate::orchestration::{
    Chain, ChainContext, ChainStep, Dag, DagContext, DagNode, Graph, GraphContext, GraphNode,
    NodeOutcome, Workflow, WorkflowNode, END, START,
};
pub use crate::prompt::{DynamicContext, FewShotTemplate, PromptBuilder, PromptTemplate};
pub use crate::retriever::{
    Document, InMemoryVectorStore, InMemoryVectorStoreBackend, KnowledgeBase, Retriever,
    RetrieverTool, ScoredDocument, SimpleKnowledgeBase, VectorStore,
};
pub use crate::stream::{ResponseStream, StreamEvent};
pub use crate::tool::{Tool, ToolOutput, ToolRegistry, ToolRetryPolicy};

pub use futures::StreamExt;
pub use serde_json::Value;
pub use serde_json::json;
pub use tokio_util::sync::CancellationToken;

#[cfg(feature = "macros")]
pub use crate::tool_fn;

#[cfg(feature = "sqlite")]
pub use crate::memory::SqliteMemory;

#[cfg(feature = "redis")]
pub use crate::memory::RedisMemory;

#[cfg(feature = "redis")]
pub use crate::checkpoint::RedisCheckpoint;

#[cfg(feature = "redis")]
pub use crate::distributed::RedisBroker;

#[cfg(feature = "nats")]
pub use crate::checkpoint::NatsKvCheckpoint;

#[cfg(feature = "nats")]
pub use crate::distributed::NatsBroker;

#[cfg(feature = "amqp")]
pub use crate::distributed::AmqpBroker;

#[cfg(feature = "grpc")]
pub use crate::distributed::{GrpcBrokerClient, GrpcBrokerServer};

#[cfg(feature = "mcp")]
pub use crate::mcp::McpWsServer;

#[cfg(all(feature = "mcp", feature = "grpc"))]
pub use crate::mcp::{McpGrpcServer, McpGrpcTransport};

#[cfg(feature = "qdrant")]
pub use crate::retriever::qdrant::QdrantRetriever;

#[cfg(feature = "pgvector")]
pub use daimon_plugin_pgvector::{DistanceMetric, PgVectorStore, PgVectorStoreBuilder};

#[cfg(feature = "eval")]
pub use crate::eval::{EvalResult, EvalRunner, EvalScenario, Scorer};

#[cfg(feature = "http-server")]
pub use crate::server::AgentServer;

#[cfg(feature = "sqs")]
pub use daimon_provider_bedrock::SqsBroker;

#[cfg(feature = "pubsub")]
pub use daimon_provider_gemini::PubSubBroker;

#[cfg(feature = "servicebus")]
pub use daimon_provider_azure::ServiceBusBroker;
