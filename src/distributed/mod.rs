//! Distributed agent execution across multiple processes.
//!
//! Provides a [`TaskBroker`] trait for distributing agent tasks, a
//! [`TaskWorker`] for consuming and executing tasks, and multiple broker
//! implementations:
//!
//! - [`InProcessBroker`] — tokio channels, for testing or single-process parallelism
//! - [`RedisBroker`] — Redis Lists + Hashes, for multi-process execution (`feature = "redis"`)
//! - [`NatsBroker`] — NATS JetStream, durable at-least-once delivery (`feature = "nats"`)
//! - [`AmqpBroker`] — RabbitMQ via AMQP 0-9-1 (`feature = "amqp"`)
//! - [`GrpcBrokerServer`] / [`GrpcBrokerClient`] — gRPC transport (`feature = "grpc"`)
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────┐  submit   ┌────────────┐  receive   ┌────────────┐
//! │ Producer │ ───────►  │ TaskBroker │  ───────►  │ TaskWorker │
//! └─────────┘           └────────────┘            └────────────┘
//!                              ▲                        │
//!                              │      complete          │
//!                              └────────────────────────┘
//! ```
//!
//! To use a different message broker (RabbitMQ, NATS, etc.), implement
//! [`TaskBroker`] for your transport and pass it to [`TaskWorker::new`].
//!
//! ```ignore
//! use daimon::distributed::{InProcessBroker, TaskWorker, AgentTask};
//!
//! let broker = InProcessBroker::new(64);
//! let worker = TaskWorker::new(broker.clone(), || {
//!     Agent::builder().model(my_model).build().unwrap()
//! });
//!
//! // Submit work
//! broker.submit(AgentTask::new("Summarize this article")).await?;
//!
//! // Worker loop (run in a background task)
//! worker.run_once().await?;
//! ```

mod types;
mod broker;
pub mod streaming;
mod worker;

#[cfg(feature = "redis")]
pub mod redis_broker;

#[cfg(feature = "nats")]
pub mod nats_broker;

#[cfg(feature = "amqp")]
pub mod amqp_broker;

#[cfg(feature = "grpc")]
pub mod grpc;

pub use types::{AgentTask, TaskResult, TaskStatus};
pub use broker::{TaskBroker, ErasedTaskBroker, InProcessBroker};
pub use streaming::{
    InProcessEventBus, SerializableStreamEvent, StreamingTaskWorker, TaskEventBus,
    TaskStreamEvent,
};
pub use worker::TaskWorker;

#[cfg(feature = "redis")]
pub use redis_broker::RedisBroker;

#[cfg(feature = "nats")]
pub use nats_broker::NatsBroker;

#[cfg(feature = "amqp")]
pub use amqp_broker::AmqpBroker;

#[cfg(feature = "grpc")]
pub use grpc::{GrpcBrokerClient, GrpcBrokerServer};
