//! Streaming distributed execution — stream agent events across process boundaries.
//!
//! While [`TaskWorker`](super::TaskWorker) returns only the final result,
//! [`StreamingTaskWorker`] uses `Agent::prompt_stream()` and publishes each
//! [`StreamEvent`] through a [`TaskEventBus`], allowing remote consumers to
//! observe the agent's progress in real time.
//!
//! ```ignore
//! use daimon::distributed::streaming::*;
//!
//! let bus = InProcessEventBus::new(64);
//! let worker = StreamingTaskWorker::new(broker, bus.clone(), || {
//!     Agent::builder().model(my_model).build().unwrap()
//! });
//!
//! // Subscribe before submitting
//! let mut rx = bus.subscribe();
//!
//! // Worker loop (background task)
//! tokio::spawn(async move { worker.run().await });
//!
//! // Receive live events
//! while let Ok(evt) = rx.recv().await {
//!     println!("{}: {:?}", evt.task_id, evt.event);
//! }
//! ```

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::agent::Agent;
use crate::error::{DaimonError, Result};
use crate::stream::StreamEvent;

use super::broker::ErasedTaskBroker;
use super::types::{AgentTask, TaskResult};
use super::worker::AgentFactory;

/// A serializable wrapper around [`StreamEvent`] tagged with a task ID.
///
/// This is the unit of data that flows through a [`TaskEventBus`],
/// allowing consumers to correlate events with their originating task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStreamEvent {
    /// The task this event belongs to.
    pub task_id: String,
    /// The serializable event payload.
    pub event: SerializableStreamEvent,
}

/// Serializable version of [`StreamEvent`] for cross-process transport.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SerializableStreamEvent {
    TextDelta(String),
    ToolCallStart {
        id: String,
        name: String,
    },
    ToolCallDelta {
        id: String,
        arguments_delta: String,
    },
    ToolCallEnd {
        id: String,
    },
    ToolResult {
        id: String,
        content: String,
        is_error: bool,
    },
    Usage {
        iteration: usize,
        input_tokens: u32,
        output_tokens: u32,
        estimated_cost: f64,
    },
    Error(String),
    Done,
}

impl From<&StreamEvent> for SerializableStreamEvent {
    fn from(event: &StreamEvent) -> Self {
        match event {
            StreamEvent::TextDelta(t) => Self::TextDelta(t.clone()),
            StreamEvent::ToolCallStart { id, name } => Self::ToolCallStart {
                id: id.clone(),
                name: name.clone(),
            },
            StreamEvent::ToolCallDelta {
                id,
                arguments_delta,
            } => Self::ToolCallDelta {
                id: id.clone(),
                arguments_delta: arguments_delta.clone(),
            },
            StreamEvent::ToolCallEnd { id } => Self::ToolCallEnd { id: id.clone() },
            StreamEvent::ToolResult {
                id,
                content,
                is_error,
            } => Self::ToolResult {
                id: id.clone(),
                content: content.clone(),
                is_error: *is_error,
            },
            StreamEvent::Usage {
                iteration,
                input_tokens,
                output_tokens,
                estimated_cost,
            } => Self::Usage {
                iteration: *iteration,
                input_tokens: *input_tokens,
                output_tokens: *output_tokens,
                estimated_cost: *estimated_cost,
            },
            StreamEvent::Error(e) => Self::Error(e.clone()),
            StreamEvent::Done => Self::Done,
        }
    }
}

/// Trait for publishing and subscribing to task stream events.
///
/// Implement this for your transport layer (Redis pub/sub, NATS, etc.)
/// to stream agent events across process boundaries.
pub trait TaskEventBus: Send + Sync {
    /// Publishes an event for a given task.
    fn publish<'a>(
        &'a self,
        event: TaskStreamEvent,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
}

/// Object-safe erased version of [`TaskEventBus`].
pub trait ErasedTaskEventBus: Send + Sync {
    fn publish_erased<'a>(
        &'a self,
        event: TaskStreamEvent,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
}

impl<T: TaskEventBus> ErasedTaskEventBus for T {
    fn publish_erased<'a>(
        &'a self,
        event: TaskStreamEvent,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        self.publish(event)
    }
}

/// In-process event bus backed by a `tokio::sync::broadcast` channel.
///
/// All subscribers receive every event. Suitable for single-process use
/// and testing. For cross-process streaming, implement [`TaskEventBus`]
/// over your message broker.
pub struct InProcessEventBus {
    tx: broadcast::Sender<TaskStreamEvent>,
}

impl InProcessEventBus {
    /// Creates a new in-process event bus with the given channel capacity.
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }

    /// Returns a receiver that will get all future events.
    pub fn subscribe(&self) -> broadcast::Receiver<TaskStreamEvent> {
        self.tx.subscribe()
    }
}

impl Clone for InProcessEventBus {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
        }
    }
}

impl TaskEventBus for InProcessEventBus {
    fn publish<'a>(
        &'a self,
        event: TaskStreamEvent,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let _ = self.tx.send(event);
            Ok(())
        })
    }
}

/// A worker that executes tasks with streaming and publishes events
/// through a [`TaskEventBus`].
///
/// Unlike [`TaskWorker`](super::TaskWorker), this worker uses
/// `Agent::prompt_stream()` to emit real-time events during execution.
pub struct StreamingTaskWorker {
    broker: Arc<dyn ErasedTaskBroker>,
    bus: Arc<dyn ErasedTaskEventBus>,
    factory: AgentFactory,
}

impl StreamingTaskWorker {
    /// Creates a new streaming worker.
    pub fn new<B, E, F>(broker: B, bus: E, factory: F) -> Self
    where
        B: super::broker::TaskBroker + 'static,
        E: TaskEventBus + 'static,
        F: Fn() -> Agent + Send + Sync + 'static,
    {
        Self {
            broker: Arc::new(broker),
            bus: Arc::new(bus),
            factory: Arc::new(factory),
        }
    }

    /// Creates a streaming worker from erased broker/bus and factory.
    pub fn from_erased(
        broker: Arc<dyn ErasedTaskBroker>,
        bus: Arc<dyn ErasedTaskEventBus>,
        factory: AgentFactory,
    ) -> Self {
        Self {
            broker,
            bus,
            factory,
        }
    }

    /// Processes a single task with streaming events. Returns `Ok(None)` only
    /// if the broker is permanently closed; an idle queue is polled until a
    /// task arrives.
    ///
    /// A task whose agent errored is reported to the broker via `fail` (its
    /// status becomes `Failed`), matching [`TaskWorker`](super::TaskWorker).
    pub async fn run_once(&self) -> Result<Option<TaskResult>> {
        let task = match super::worker::next_task(&self.broker).await? {
            Some(t) => t,
            None => return Ok(None),
        };

        let result = self.execute_streaming(&task).await;

        match &result {
            // Route agent errors to `fail` so the broker records `Failed`
            // rather than `Completed` with an embedded error, keeping status
            // semantics consistent with TaskWorker's paths.
            Ok(tr) => match &tr.error {
                Some(err) => {
                    self.broker.fail_erased(&task.task_id, err.clone()).await?;
                }
                None => {
                    self.broker
                        .complete_erased(&task.task_id, tr.clone())
                        .await?;
                }
            },
            Err(e) => {
                self.broker
                    .fail_erased(&task.task_id, e.to_string())
                    .await?;
            }
        }

        result.map(Some)
    }

    /// Runs the streaming worker loop indefinitely, until the broker is
    /// permanently closed. Idle polls do not stop the loop.
    pub async fn run(&self) -> Result<()> {
        loop {
            match self.run_once().await? {
                Some(_) => continue,
                None => {
                    tracing::info!("streaming worker: broker closed, exiting");
                    return Ok(());
                }
            }
        }
    }

    async fn execute_streaming(&self, task: &AgentTask) -> Result<TaskResult> {
        use futures::StreamExt;

        let agent = (self.factory)();

        let stream = match agent.prompt_stream(&task.input).await {
            Ok(s) => s,
            Err(e) => {
                self.bus
                    .publish_erased(TaskStreamEvent {
                        task_id: task.task_id.clone(),
                        event: SerializableStreamEvent::Error(e.to_string()),
                    })
                    .await?;
                return Ok(TaskResult {
                    task_id: task.task_id.clone(),
                    output: String::new(),
                    iterations: 0,
                    cost: 0.0,
                    error: Some(e.to_string()),
                });
            }
        };

        tokio::pin!(stream);

        let mut full_text = String::new();
        let mut iterations = 0;
        let mut cost = 0.0;

        while let Some(event_result) = stream.next().await {
            match event_result {
                Ok(ref event) => {
                    if let StreamEvent::TextDelta(t) = event {
                        full_text.push_str(t);
                    }
                    if let StreamEvent::Usage {
                        iteration,
                        estimated_cost,
                        ..
                    } = event
                    {
                        iterations = *iteration;
                        cost = *estimated_cost;
                    }

                    let serializable = SerializableStreamEvent::from(event);
                    let _ = self
                        .bus
                        .publish_erased(TaskStreamEvent {
                            task_id: task.task_id.clone(),
                            event: serializable,
                        })
                        .await;
                }
                Err(e) => {
                    let _ = self
                        .bus
                        .publish_erased(TaskStreamEvent {
                            task_id: task.task_id.clone(),
                            event: SerializableStreamEvent::Error(e.to_string()),
                        })
                        .await;
                    return Err(DaimonError::Other(format!("stream error: {e}")));
                }
            }
        }

        Ok(TaskResult {
            task_id: task.task_id.clone(),
            output: full_text,
            iterations,
            cost,
            error: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::{InProcessBroker, TaskBroker};
    use crate::error::Result as DResult;
    use crate::model::Model;
    use crate::model::types::*;
    use crate::stream::ResponseStream;

    struct EchoModel;

    impl Model for EchoModel {
        async fn generate(&self, request: &ChatRequest) -> DResult<ChatResponse> {
            let last = request
                .messages
                .last()
                .and_then(|m| m.content.as_deref())
                .unwrap_or("none");
            Ok(ChatResponse {
                message: Message::assistant(format!("Echo: {last}")),
                stop_reason: StopReason::EndTurn,
                usage: Some(Usage::default()),
            })
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> DResult<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    #[tokio::test]
    async fn test_streaming_worker_executes_task() {
        let broker = InProcessBroker::new(16);
        let bus = InProcessEventBus::new(64);
        let mut rx = bus.subscribe();

        let worker = StreamingTaskWorker::new(broker.clone(), bus, || {
            Agent::builder().model(EchoModel).build().unwrap()
        });

        let task = AgentTask::new("streaming test");
        let id = broker.submit(task).await.unwrap();

        let result = worker.run_once().await.unwrap().unwrap();
        assert_eq!(result.task_id, id);
        assert!(result.error.is_none());

        let mut got_done = false;
        while let Ok(evt) = rx.try_recv() {
            assert_eq!(evt.task_id, id);
            if matches!(evt.event, SerializableStreamEvent::Done) {
                got_done = true;
            }
        }
        assert!(got_done);
    }

    #[tokio::test]
    async fn test_streaming_worker_run_exits_on_broker_close() {
        let broker = InProcessBroker::new(16);
        let bus = InProcessEventBus::new(64);

        let worker = StreamingTaskWorker::new(broker.clone(), bus, || {
            Agent::builder().model(EchoModel).build().unwrap()
        });

        let id = broker.submit(AgentTask::new("drain me")).await.unwrap();
        broker.close().await;

        // The worker must drain the queued task and then exit promptly once
        // the closed channel signals end-of-stream.
        tokio::time::timeout(std::time::Duration::from_secs(5), worker.run())
            .await
            .expect("streaming worker.run() must exit after the broker closes")
            .unwrap();

        let status = broker.status(&id).await.unwrap();
        assert!(matches!(
            status,
            crate::distributed::TaskStatus::Completed(_)
        ));
    }

    /// Model whose streaming entry point always errors.
    struct FailingStreamModel;

    impl Model for FailingStreamModel {
        async fn generate(&self, _request: &ChatRequest) -> DResult<ChatResponse> {
            Err(DaimonError::Model("boom".into()))
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> DResult<ResponseStream> {
            Err(DaimonError::Model("boom".into()))
        }
    }

    #[tokio::test]
    async fn test_streaming_worker_agent_error_marks_task_failed() {
        let broker = InProcessBroker::new(16);
        let bus = InProcessEventBus::new(64);

        let worker = StreamingTaskWorker::new(broker.clone(), bus, || {
            Agent::builder().model(FailingStreamModel).build().unwrap()
        });

        let id = broker.submit(AgentTask::new("will fail")).await.unwrap();

        // A mid-stream model error surfaces as `Err` from `run_once` (the
        // stream broke), but only after the failure has been reported to the
        // broker.
        let result = worker.run_once().await;
        assert!(result.is_err());

        // The broker status must be Failed, not Completed with an embedded
        // error, matching TaskWorker's unified error routing.
        let status = broker.status(&id).await.unwrap();
        assert!(
            matches!(status, crate::distributed::TaskStatus::Failed(ref msg) if msg.contains("boom")),
            "agent error must mark the task Failed, got {status:?}"
        );
    }

    #[tokio::test]
    async fn test_event_bus_subscribe_after_publish() {
        let bus = InProcessEventBus::new(16);
        let mut rx = bus.subscribe();

        bus.publish(TaskStreamEvent {
            task_id: "t1".into(),
            event: SerializableStreamEvent::TextDelta("hello".into()),
        })
        .await
        .unwrap();

        let evt = rx.recv().await.unwrap();
        assert_eq!(evt.task_id, "t1");
        assert!(matches!(evt.event, SerializableStreamEvent::TextDelta(ref t) if t == "hello"));
    }

    #[test]
    fn test_serializable_stream_event_roundtrip() {
        let events = vec![
            SerializableStreamEvent::TextDelta("hi".into()),
            SerializableStreamEvent::ToolCallStart {
                id: "tc1".into(),
                name: "search".into(),
            },
            SerializableStreamEvent::ToolCallDelta {
                id: "tc1".into(),
                arguments_delta: "{\"q\":".into(),
            },
            SerializableStreamEvent::ToolCallEnd { id: "tc1".into() },
            SerializableStreamEvent::ToolResult {
                id: "tc1".into(),
                content: "found it".into(),
                is_error: false,
            },
            SerializableStreamEvent::Usage {
                iteration: 1,
                input_tokens: 100,
                output_tokens: 50,
                estimated_cost: 0.001,
            },
            SerializableStreamEvent::Error("oops".into()),
            SerializableStreamEvent::Done,
        ];

        for event in &events {
            let json = serde_json::to_string(event).unwrap();
            let deser: SerializableStreamEvent = serde_json::from_str(&json).unwrap();
            let json2 = serde_json::to_string(&deser).unwrap();
            assert_eq!(json, json2);
        }
    }

    #[test]
    fn test_task_stream_event_roundtrip() {
        let evt = TaskStreamEvent {
            task_id: "task-123".into(),
            event: SerializableStreamEvent::TextDelta("chunk".into()),
        };

        let json = serde_json::to_string(&evt).unwrap();
        let deser: TaskStreamEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.task_id, "task-123");
    }

    #[test]
    fn test_from_stream_event() {
        let events = vec![
            StreamEvent::TextDelta("text".into()),
            StreamEvent::ToolCallStart {
                id: "1".into(),
                name: "fn".into(),
            },
            StreamEvent::Done,
        ];

        for event in &events {
            let serializable = SerializableStreamEvent::from(event);
            let json = serde_json::to_string(&serializable).unwrap();
            assert!(!json.is_empty());
        }
    }
}
