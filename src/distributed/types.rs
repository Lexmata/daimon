//! Serializable task and result types for distributed execution.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// A unit of work submitted to a [`TaskBroker`](super::TaskBroker).
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
    /// Creates a new task with a random UUID.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_creation() {
        let task = AgentTask::new("Hello, world!");
        assert!(task.task_id.starts_with("task-"));
        assert_eq!(task.input, "Hello, world!");
        assert!(task.run_id.is_none());
        assert!(task.metadata.is_empty());
    }

    #[test]
    fn test_task_with_run_id() {
        let task = AgentTask::new("test").with_run_id("run-42");
        assert_eq!(task.run_id.as_deref(), Some("run-42"));
    }

    #[test]
    fn test_task_with_metadata() {
        let task = AgentTask::new("test")
            .with_metadata("priority", serde_json::json!(1));
        assert_eq!(task.metadata["priority"], serde_json::json!(1));
    }

    #[test]
    fn test_task_unique_ids() {
        let a = AgentTask::new("a");
        let b = AgentTask::new("b");
        assert_ne!(a.task_id, b.task_id);
    }

    #[test]
    fn test_task_serialization() {
        let task = AgentTask::new("serialize me")
            .with_run_id("r1")
            .with_metadata("key", serde_json::json!("val"));

        let json = serde_json::to_string(&task).unwrap();
        let deser: AgentTask = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.input, "serialize me");
        assert_eq!(deser.run_id.as_deref(), Some("r1"));
    }

    #[test]
    fn test_result_serialization() {
        let result = TaskResult {
            task_id: "t-1".into(),
            output: "done".into(),
            iterations: 3,
            cost: 0.001,
            error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let deser: TaskResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.task_id, "t-1");
        assert_eq!(deser.iterations, 3);
    }

    #[test]
    fn test_status_variants() {
        let pending = TaskStatus::Pending;
        let running = TaskStatus::Running;
        let completed = TaskStatus::Completed(TaskResult {
            task_id: "t".into(),
            output: "ok".into(),
            iterations: 1,
            cost: 0.0,
            error: None,
        });
        let failed = TaskStatus::Failed("boom".into());

        assert!(matches!(pending, TaskStatus::Pending));
        assert!(matches!(running, TaskStatus::Running));
        assert!(matches!(completed, TaskStatus::Completed(_)));
        assert!(matches!(failed, TaskStatus::Failed(_)));
    }
}
