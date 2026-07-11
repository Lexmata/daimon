//! # Daimon
//!
//! A Rust-native AI agent framework for building LLM-powered agents with tool use,
//! memory, and streaming. Daimon implements the ReAct (Reason-Act-Observe) pattern:
//! the agent calls a model, optionally invokes tools, observes results, and repeats
//! until it produces a final response.
//!
//! ## Quick Start
//!
//! ```ignore
//! use daimon::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> daimon::Result<()> {
//!     let agent = Agent::builder()
//!         .model(daimon::model::openai::OpenAi::new("gpt-4o"))
//!         .system_prompt("You are a helpful assistant.")
//!         .build()?;
//!
//!     let response = agent.prompt("What is Rust?").await?;
//!     println!("{}", response.text());
//!     Ok(())
//! }
//! ```
//!
//! ## Feature Flags
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `openai` | OpenAI API provider (default) |
//! | `anthropic` | Anthropic Claude API provider (default) |
//! | `macros` | `#[tool_fn]` proc macro (default) |
//! | `gemini` | Google Gemini / Vertex AI provider (via `daimon-provider-gemini`) |
//! | `azure` | Azure OpenAI Service provider (via `daimon-provider-azure`) |
//! | `bedrock` | AWS Bedrock provider (via `daimon-provider-bedrock`) |
//! | `ollama` | Ollama local model provider |
//! | `llamacpp` | llama.cpp (llama-server) provider (via `daimon-provider-llamacpp`) |
//! | `sqlite` | SQLite memory backend |
//! | `redis` | Redis memory backend + task broker |
//! | `nats` | NATS JetStream task broker |
//! | `amqp` | RabbitMQ (AMQP) task broker |
//! | `sqs` | AWS SQS task broker (via `daimon-provider-bedrock`) |
//! | `pubsub` | Google Cloud Pub/Sub task broker (via `daimon-provider-gemini`) |
//! | `servicebus` | Azure Service Bus task broker (via `daimon-provider-azure`) |
//! | `mcp` | Model Context Protocol client & server |
//! | `otel` | OpenTelemetry OTLP span export |
//! | `qdrant` | Qdrant vector store retriever |
//! | `pgvector` | pgvector-backed vector store (via `daimon-plugin-pgvector`) |
//! | `opensearch` | OpenSearch k-NN vector store (via `daimon-plugin-opensearch`) |
//! | `grpc` | gRPC transport for distributed execution |
//! | `full` | All providers + macros + MCP + SQLite + Redis + NATS + AMQP + OTel + HTTP server + Qdrant + pgvector + OpenSearch + gRPC + eval + SQS + Pub/Sub + Service Bus |
//!
//! The core framework compiles with no features; enable providers as needed.
//!
//! ## Plugin Interface
//!
//! The [`Model`](crate::model::Model) trait (from [`daimon_core`]) is the plugin interface. To create
//! a new LLM provider, depend on `daimon-core` and implement `Model`. See the
//! `daimon-provider-*` crates for examples.
//!
//! ## Module Overview
//!
//! - [`agent`] — Agent builder, ReAct loop, multi-agent patterns, resumable runs
//! - [`model`] — LLM provider trait and implementations
//! - [`tool`] — Tool trait, registry, and execution
//! - [`memory`] — Conversation memory implementations
//! - [`stream`] — Streaming response types
//! - [`hooks`] — Lifecycle hooks for observability and control
//! - [`orchestration`] — Chain, graph, DAG, and workflow orchestration
//! - [`retriever`] — RAG retriever trait and tool adapter
//! - [`checkpoint`] — Checkpointing and state persistence
//! - [`a2a`] — Google Agent-to-Agent protocol support
//! - [`distributed`] — Distributed agent execution across processes
//! - [`mcp`] — Model Context Protocol client and server (stdio, HTTP)
//! - [`telemetry`] — OpenTelemetry OTLP export (feature = "otel")
//! - [`server`] — HTTP agent server (feature = "http-server")
//! - [`eval`] — evaluation harness for agent behavior (feature = "eval")

pub mod a2a;
pub mod agent;
pub mod checkpoint;
pub mod cost;
pub mod distributed;
pub mod error;
pub mod guardrails;
pub mod hooks;
pub mod memory;
pub mod middleware;
pub mod model;
pub mod orchestration;
pub mod prelude;
pub mod prompt;
pub mod retriever;
pub mod stream;
pub mod tool;

#[cfg(feature = "mcp")]
pub mod mcp;

#[cfg(feature = "otel")]
pub mod telemetry;

#[cfg(feature = "http-server")]
pub mod server;

#[cfg(feature = "eval")]
pub mod eval;

#[cfg(feature = "macros")]
pub use daimon_macros::tool_fn;

pub use error::{DaimonError, Result};
