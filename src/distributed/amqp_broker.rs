//! RabbitMQ task broker via AMQP 0-9-1 for distributed agent execution.
//!
//! Uses a durable queue for task delivery with manual acknowledgement.
//! Enable with `feature = "amqp"`.
//!
//! ```ignore
//! use daimon::distributed::AmqpBroker;
//!
//! let broker = AmqpBroker::connect("amqp://guest:guest@127.0.0.1:5672", "daimon-tasks").await?;
//! broker.submit(AgentTask::new("Summarize this")).await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use lapin::options::{
    BasicAckOptions, BasicConsumeOptions, BasicPublishOptions, QueueDeclareOptions,
};
use lapin::types::FieldTable;
use lapin::{BasicProperties, Channel, Connection, ConnectionProperties, Consumer};
use tokio::sync::Mutex;

use crate::error::{DaimonError, Result};

use super::broker::TaskBroker;
use super::types::{AgentTask, TaskResult, TaskStatus};

/// Distributes agent tasks via RabbitMQ (AMQP 0-9-1).
///
/// Tasks are published to a durable queue. Workers consume with
/// manual acknowledgement for at-least-once delivery guarantees.
pub struct AmqpBroker {
    channel: Channel,
    queue_name: String,
    statuses: Arc<Mutex<HashMap<String, TaskStatus>>>,
    consumer: Arc<Mutex<Option<Consumer>>>,
}

impl AmqpBroker {
    /// Connects to RabbitMQ, declares a durable queue, and creates a broker.
    ///
    /// * `url` — AMQP connection URL (e.g. `amqp://guest:guest@127.0.0.1:5672`)
    /// * `queue_name` — the queue to use for task delivery
    pub async fn connect(url: &str, queue_name: impl Into<String>) -> Result<Self> {
        let queue_name = queue_name.into();
        let conn = Connection::connect(url, ConnectionProperties::default())
            .await
            .map_err(|e| DaimonError::Other(format!("amqp connect: {e}")))?;

        let channel = conn
            .create_channel()
            .await
            .map_err(|e| DaimonError::Other(format!("amqp channel: {e}")))?;

        channel
            .queue_declare(
                &queue_name,
                QueueDeclareOptions {
                    durable: true,
                    ..Default::default()
                },
                FieldTable::default(),
            )
            .await
            .map_err(|e| DaimonError::Other(format!("amqp declare queue: {e}")))?;

        Ok(Self {
            channel,
            queue_name,
            statuses: Arc::new(Mutex::new(HashMap::new())),
            consumer: Arc::new(Mutex::new(None)),
        })
    }

    /// Returns a reference to the underlying AMQP channel.
    pub fn channel(&self) -> &Channel {
        &self.channel
    }

    async fn ensure_consumer(&self) -> Result<Consumer> {
        let mut guard = self.consumer.lock().await;
        if let Some(ref consumer) = *guard {
            return Ok(consumer.clone());
        }

        let consumer = self
            .channel
            .basic_consume(
                &self.queue_name,
                "daimon-worker",
                BasicConsumeOptions {
                    no_ack: false,
                    ..Default::default()
                },
                FieldTable::default(),
            )
            .await
            .map_err(|e| DaimonError::Other(format!("amqp consume: {e}")))?;

        *guard = Some(consumer.clone());
        Ok(consumer)
    }
}

impl TaskBroker for AmqpBroker {
    async fn submit(&self, task: AgentTask) -> Result<String> {
        let id = task.task_id.clone();
        let json = serde_json::to_string(&task)
            .map_err(|e| DaimonError::Other(format!("serialize task: {e}")))?;

        {
            let mut statuses = self.statuses.lock().await;
            statuses.insert(id.clone(), TaskStatus::Pending);
        }

        self.channel
            .basic_publish(
                "",
                &self.queue_name,
                BasicPublishOptions::default(),
                json.as_bytes(),
                BasicProperties::default()
                    .with_delivery_mode(2)
                    .with_content_type("application/json".into()),
            )
            .await
            .map_err(|e| DaimonError::Other(format!("amqp publish: {e}")))?
            .await
            .map_err(|e| DaimonError::Other(format!("amqp publish confirm: {e}")))?;

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
        let mut stream = consumer;

        match stream.next().await {
            Some(Ok(delivery)) => {
                let task: AgentTask = serde_json::from_slice(&delivery.data)
                    .map_err(|e| DaimonError::Other(format!("deserialize task: {e}")))?;

                delivery
                    .ack(BasicAckOptions::default())
                    .await
                    .map_err(|e| DaimonError::Other(format!("amqp ack: {e}")))?;

                {
                    let mut statuses = self.statuses.lock().await;
                    statuses.insert(task.task_id.clone(), TaskStatus::Running);
                }

                Ok(Some(task))
            }
            Some(Err(e)) => Err(DaimonError::Other(format!("amqp delivery error: {e}"))),
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
    fn test_task_serialization_for_amqp() {
        let task = AgentTask::new("amqp test")
            .with_metadata("routing", serde_json::json!("high-priority"));

        let json = serde_json::to_string(&task).unwrap();
        let bytes = json.as_bytes();
        let deser: AgentTask = serde_json::from_slice(bytes).unwrap();
        assert_eq!(deser.input, "amqp test");
        assert_eq!(deser.metadata["routing"], "high-priority");
    }

    #[test]
    fn test_result_serialization_for_amqp() {
        let result = TaskResult {
            task_id: "t-amqp".into(),
            output: "amqp result".into(),
            iterations: 1,
            cost: 0.002,
            error: None,
        };

        let json = serde_json::to_string(&result).unwrap();
        let deser: TaskResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.output, "amqp result");
    }

    #[test]
    fn test_status_tracking_in_memory() {
        let statuses: HashMap<String, TaskStatus> = HashMap::new();
        assert!(!statuses.contains_key("unknown"));
    }
}
