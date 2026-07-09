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
    /// Creates a new task with a unique ID derived from the current time and
    /// a process-wide counter.
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

    /// Generates a task ID of the form `task-{nanos:x}-{counter:x}`.
    ///
    /// The nanosecond timestamp alone is not collision-safe: two tasks created
    /// within the same clock tick (coarse clocks, concurrent callers) would get
    /// identical IDs. A process-wide atomic counter is appended so every call
    /// in a process yields a distinct ID, while the timestamp keeps IDs unique
    /// across processes and restarts.
    fn generate_id() -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::{SystemTime, UNIX_EPOCH};

        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("task-{ts:x}-{seq:x}")
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

    /// Waits for a task and returns it.
    ///
    /// Returns `Ok(None)` when no task was obtained. What that means depends
    /// on [`none_means_closed`](Self::none_means_closed):
    ///
    /// - `none_means_closed() == true` — the broker is permanently closed and
    ///   no further tasks will ever arrive; callers should stop polling.
    /// - `none_means_closed() == false` (the default) — the queue was merely
    ///   idle for one poll interval (e.g. a blocking-pop timeout or an empty
    ///   fetch); callers should keep polling.
    fn receive(&self) -> impl Future<Output = Result<Option<AgentTask>>> + Send;

    /// Whether [`receive`](Self::receive) returning `Ok(None)` means the
    /// broker is permanently closed rather than momentarily idle.
    ///
    /// Network brokers (Redis, NATS, SQS, Pub/Sub, Service Bus, …) poll with
    /// a timeout and legitimately come up empty when the queue is idle, so the
    /// default is `false`: an empty receive is a transient condition and the
    /// caller should retry. Brokers with a real end-of-stream signal (e.g. an
    /// in-process channel whose senders are gone, or a cancelled AMQP
    /// consumer) override this to `true` so workers can shut down promptly.
    fn none_means_closed(&self) -> bool {
        false
    }

    /// Marks a task as completed with the given result.
    fn complete(
        &self,
        task_id: &str,
        result: TaskResult,
    ) -> impl Future<Output = Result<()>> + Send;

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

    fn receive_erased(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Option<AgentTask>>> + Send + '_>>;

    /// Object-safe mirror of [`TaskBroker::none_means_closed`].
    fn none_means_closed_erased(&self) -> bool {
        false
    }

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

    fn receive_erased(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Option<AgentTask>>> + Send + '_>> {
        Box::pin(self.receive())
    }

    fn none_means_closed_erased(&self) -> bool {
        self.none_means_closed()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_id_unique_across_concurrent_calls() {
        // Nanosecond timestamps alone can collide under concurrency; the
        // appended process-wide counter must make every ID distinct.
        let handles: Vec<_> = (0..8)
            .map(|_| {
                std::thread::spawn(|| {
                    (0..100)
                        .map(|_| AgentTask::new("x").task_id)
                        .collect::<Vec<_>>()
                })
            })
            .collect();

        let mut ids = std::collections::HashSet::new();
        for handle in handles {
            for id in handle.join().expect("thread panicked") {
                assert!(ids.insert(id.clone()), "duplicate task id generated: {id}");
            }
        }
        assert_eq!(ids.len(), 800);
    }
}
