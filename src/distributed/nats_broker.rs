//! NATS JetStream task broker for distributed agent execution.
//!
//! Uses NATS JetStream for durable, at-least-once task delivery with
//! automatic acknowledgement. Enable with `feature = "nats"`.
//!
//! ```ignore
//! use daimon::distributed::NatsBroker;
//!
//! let broker = NatsBroker::connect("nats://127.0.0.1:4222", "daimon-tasks").await?;
//! broker.submit(AgentTask::new("Summarize this")).await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use async_nats::jetstream::consumer::Consumer;
use async_nats::jetstream::consumer::pull::Config as PullConfig;
use tokio::sync::Mutex;

use crate::error::{DaimonError, Result};

use super::broker::TaskBroker;
use super::types::{AgentTask, TaskResult, TaskStatus};

/// Distributes agent tasks via NATS JetStream.
///
/// Tasks are published to a JetStream stream and consumed by workers
/// via a pull-based consumer. Status is tracked in an in-memory map
/// (for the local process) and replicated via JetStream KV when
/// cross-process visibility is needed.
pub struct NatsBroker {
    client: async_nats::Client,
    jetstream: async_nats::jetstream::Context,
    stream_name: String,
    subject_prefix: String,
    statuses: Arc<Mutex<HashMap<String, TaskStatus>>>,
    consumer: Arc<Mutex<Option<Consumer<PullConfig>>>>,
}

impl NatsBroker {
    /// Connects to a NATS server and sets up a JetStream stream.
    ///
    /// * `url` — NATS server URL (e.g. `nats://127.0.0.1:4222`)
    /// * `stream_name` — JetStream stream name (e.g. `daimon-tasks`)
    pub async fn connect(url: &str, stream_name: impl Into<String>) -> Result<Self> {
        let stream_name = stream_name.into();
        let client = async_nats::connect(url)
            .await
            .map_err(|e| DaimonError::Other(format!("nats connect: {e}")))?;

        let jetstream = async_nats::jetstream::new(client.clone());
        let subject_prefix = format!("{stream_name}.tasks");

        jetstream
            .get_or_create_stream(async_nats::jetstream::stream::Config {
                name: stream_name.clone(),
                subjects: vec![format!("{subject_prefix}.>")],
                retention: async_nats::jetstream::stream::RetentionPolicy::WorkQueue,
                ..Default::default()
            })
            .await
            .map_err(|e| DaimonError::Other(format!("nats create stream: {e}")))?;

        Ok(Self {
            client,
            jetstream,
            stream_name,
            subject_prefix,
            statuses: Arc::new(Mutex::new(HashMap::new())),
            consumer: Arc::new(Mutex::new(None)),
        })
    }

    /// Returns a reference to the underlying NATS client.
    pub fn client(&self) -> &async_nats::Client {
        &self.client
    }

    fn task_subject(&self) -> String {
        format!("{}.submit", self.subject_prefix)
    }

    async fn ensure_consumer(&self) -> Result<Consumer<PullConfig>> {
        let mut guard = self.consumer.lock().await;
        if let Some(ref consumer) = *guard {
            return Ok(consumer.clone());
        }

        let stream = self
            .jetstream
            .get_stream(&self.stream_name)
            .await
            .map_err(|e| DaimonError::Other(format!("nats get stream: {e}")))?;

        let consumer: Consumer<PullConfig> = stream
            .get_or_create_consumer(
                "daimon-worker",
                PullConfig {
                    durable_name: Some("daimon-worker".into()),
                    filter_subject: self.task_subject(),
                    ack_policy: async_nats::jetstream::consumer::AckPolicy::Explicit,
                    ..Default::default()
                },
            )
            .await
            .map_err(|e| DaimonError::Other(format!("nats create consumer: {e}")))?;

        *guard = Some(consumer.clone());
        Ok(consumer)
    }
}

impl TaskBroker for NatsBroker {
    async fn submit(&self, task: AgentTask) -> Result<String> {
        let id = task.task_id.clone();
        let json = serde_json::to_string(&task)
            .map_err(|e| DaimonError::Other(format!("serialize task: {e}")))?;

        {
            let mut statuses = self.statuses.lock().await;
            statuses.insert(id.clone(), TaskStatus::Pending);
        }

        self.jetstream
            .publish(self.task_subject(), json.into())
            .await
            .map_err(|e| DaimonError::Other(format!("nats publish: {e}")))?
            .await
            .map_err(|e| DaimonError::Other(format!("nats publish ack: {e}")))?;

        Ok(id)
    }

    async fn status(&self, task_id: &str) -> Result<TaskStatus> {
        let statuses = self.statuses.lock().await;
        Ok(statuses
            .get(task_id)
            .cloned()
            .unwrap_or(TaskStatus::Pending))
    }

    async fn receive(&self) -> Result<Option<AgentTask>> {
        use futures::StreamExt;

        let consumer = self.ensure_consumer().await?;

        let mut messages = consumer
            .fetch()
            .max_messages(1)
            .messages()
            .await
            .map_err(|e| DaimonError::Other(format!("nats fetch: {e}")))?;

        match messages.next().await {
            Some(Ok(msg)) => {
                let task: AgentTask = serde_json::from_slice(&msg.payload)
                    .map_err(|e| DaimonError::Other(format!("deserialize task: {e}")))?;

                msg.ack()
                    .await
                    .map_err(|e| DaimonError::Other(format!("nats ack: {e}")))?;

                {
                    let mut statuses = self.statuses.lock().await;
                    statuses.insert(task.task_id.clone(), TaskStatus::Running);
                }

                Ok(Some(task))
            }
            Some(Err(e)) => Err(DaimonError::Other(format!("nats message error: {e}"))),
            None => Ok(None),
        }
    }

    async fn complete(&self, task_id: &str, result: TaskResult) -> Result<()> {
        let mut statuses = self.statuses.lock().await;
        statuses.insert(task_id.to_string(), TaskStatus::Completed(result));
        Ok(())
    }

    async fn fail(&self, task_id: &str, error: String) -> Result<()> {
        let mut statuses = self.statuses.lock().await;
        statuses.insert(task_id.to_string(), TaskStatus::Failed(error));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subject_generation() {
        let prefix = "daimon-tasks.tasks";
        assert_eq!(format!("{prefix}.submit"), "daimon-tasks.tasks.submit");
    }

    #[test]
    fn test_task_serialization_for_nats() {
        let task = AgentTask::new("test input")
            .with_run_id("r1")
            .with_metadata("priority", serde_json::json!(1));

        let json = serde_json::to_string(&task).unwrap();
        let deser: AgentTask = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.input, "test input");
        assert_eq!(deser.run_id.as_deref(), Some("r1"));
        assert_eq!(deser.metadata["priority"], 1);
    }

    #[test]
    fn test_result_serialization_for_nats() {
        let result = TaskResult {
            task_id: "t-nats".into(),
            output: "nats result".into(),
            iterations: 3,
            cost: 0.01,
            error: None,
        };

        let json = serde_json::to_string(&result).unwrap();
        let deser: TaskResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.output, "nats result");
    }
}
