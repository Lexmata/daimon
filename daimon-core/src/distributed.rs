//! Core types and trait for distributed agent execution.
//!
//! Provider crates implement [`TaskBroker`] for their cloud-native message
//! service. The main `daimon` crate re-exports everything from here.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use serde::{Deserialize, Serialize};

use crate::error::Result;

/// A unit of work submitted to a [`TaskBroker`].
///
/// Each task carries a unique ID and the input text for an agent prompt.
/// Optional metadata lets callers tag tasks with routing hints, priority,
/// or any application-specific data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTask {
    /// Unique identifier for this task (generated on creation).
    pub task_id: String,
    /// The user input to prompt the agent with.
    pub input: String,
    /// Optional run ID for resumable execution via checkpoints.
    pub run_id: Option<String>,
    /// Arbitrary key-value metadata (routing hints, priority, etc.).
    pub metadata: HashMap<String, serde_json::Value>,
}

impl AgentTask {
    /// Creates a new task with a timestamp-based ID.
    pub fn new(input: impl Into<String>) -> Self {
        Self {
            task_id: Self::generate_id(),
            input: input.into(),
            run_id: None,
            metadata: HashMap::new(),
        }
    }

    /// Assigns a checkpoint run ID for resumable execution.
    pub fn with_run_id(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = Some(run_id.into());
        self
    }

    /// Adds a metadata key-value pair.
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    fn generate_id() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        format!("task-{ts:x}")
    }
}

/// The result of a completed agent task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    /// The task ID this result corresponds to.
    pub task_id: String,
    /// The agent's final text output.
    pub output: String,
    /// Number of ReAct iterations the agent performed.
    pub iterations: usize,
    /// Estimated cost in USD (if a cost model was configured).
    pub cost: f64,
    /// Error message if the task failed.
    pub error: Option<String>,
}

/// Current status of a distributed task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskStatus {
    /// Submitted but not yet picked up by a worker.
    Pending,
    /// Currently being executed by a worker.
    Running,
    /// Completed successfully.
    Completed(TaskResult),
    /// Failed with an error.
    Failed(String),
}

/// Trait for distributing agent tasks across workers.
///
/// Implement this for your message broker (AWS SQS, Google Pub/Sub,
/// Azure Service Bus, Redis, NATS, RabbitMQ, etc.) to enable
/// multi-process agent execution.
pub trait TaskBroker: Send + Sync {
    /// Submits a task for execution. Returns the task ID.
    fn submit(&self, task: AgentTask) -> impl Future<Output = Result<String>> + Send;

    /// Queries the current status of a task.
    fn status(&self, task_id: &str) -> impl Future<Output = Result<TaskStatus>> + Send;

    /// Blocks until a task is available and returns it.
    /// Returns `None` if the broker is closed.
    fn receive(&self) -> impl Future<Output = Result<Option<AgentTask>>> + Send;

    /// Marks a task as completed with the given result.
    fn complete(&self, task_id: &str, result: TaskResult) -> impl Future<Output = Result<()>> + Send;

    /// Marks a task as failed with an error message.
    fn fail(&self, task_id: &str, error: String) -> impl Future<Output = Result<()>> + Send;
}

/// Object-safe wrapper for [`TaskBroker`], enabling `Arc<dyn ErasedTaskBroker>`.
pub trait ErasedTaskBroker: Send + Sync {
    fn submit_erased<'a>(
        &'a self,
        task: AgentTask,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>>;

    fn status_erased<'a>(
        &'a self,
        task_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<TaskStatus>> + Send + 'a>>;

    fn receive_erased(&self) -> Pin<Box<dyn Future<Output = Result<Option<AgentTask>>> + Send + '_>>;

    fn complete_erased<'a>(
        &'a self,
        task_id: &'a str,
        result: TaskResult,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    fn fail_erased<'a>(
        &'a self,
        task_id: &'a str,
        error: String,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
}

impl<T: TaskBroker> ErasedTaskBroker for T {
    fn submit_erased<'a>(
        &'a self,
        task: AgentTask,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        Box::pin(self.submit(task))
    }

    fn status_erased<'a>(
        &'a self,
        task_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<TaskStatus>> + Send + 'a>> {
        Box::pin(self.status(task_id))
    }

    fn receive_erased(&self) -> Pin<Box<dyn Future<Output = Result<Option<AgentTask>>> + Send + '_>> {
        Box::pin(self.receive())
    }

    fn complete_erased<'a>(
        &'a self,
        task_id: &'a str,
        result: TaskResult,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(self.complete(task_id, result))
    }

    fn fail_erased<'a>(
        &'a self,
        task_id: &'a str,
        error: String,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(self.fail(task_id, error))
    }
}
