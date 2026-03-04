//! Serializable task and result types for distributed execution.
//!
//! These types are defined in [`daimon_core::distributed`] and re-exported here.

pub use daimon_core::distributed::{AgentTask, TaskResult, TaskStatus};

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
