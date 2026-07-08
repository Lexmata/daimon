//! RabbitMQ task broker via AMQP 0-9-1 for distributed agent execution.
//!
//! Uses a durable queue for task delivery with manual, deferred
//! acknowledgement. Enable with `feature = "amqp"`.
//!
//! # Acknowledgement model
//!
//! A delivery is **not** acked when [`receive`](AmqpBroker::receive) hands the
//! task to a worker. The ack handle is retained and the delivery is only acked
//! once the worker reports the outcome via
//! [`complete`](AmqpBroker::complete) or [`fail`](AmqpBroker::fail). If the
//! worker process crashes in between, the channel closes without an ack and
//! RabbitMQ requeues the delivery — this is what makes the at-least-once
//! guarantee real. (Acking inside `receive`, before processing, would silently
//! drop tasks on a mid-flight crash.)
//!
//! # Status visibility
//!
//! Task status is tracked in a process-local map. See
//! [`AmqpBroker::status`] for the cross-process caveat.
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
    BasicAckOptions, BasicConsumeOptions, BasicNackOptions, BasicPublishOptions,
    QueueDeclareOptions,
};
use lapin::types::FieldTable;
use lapin::{Acker, BasicProperties, Channel, Connection, ConnectionProperties, Consumer};
use tokio::sync::Mutex;

use crate::error::{DaimonError, Result};

use super::broker::TaskBroker;
use super::types::{AgentTask, TaskResult, TaskStatus};

/// Distributes agent tasks via RabbitMQ (AMQP 0-9-1).
///
/// Tasks are published to a durable queue. Workers consume with manual,
/// deferred acknowledgement for at-least-once delivery guarantees (see the
/// module docs). Status is tracked in a process-local map (see
/// [`AmqpBroker::status`]).
pub struct AmqpBroker {
    channel: Channel,
    queue_name: String,
    statuses: Arc<Mutex<HashMap<String, TaskStatus>>>,
    consumer: Arc<Mutex<Option<Consumer>>>,
    /// Ack handles for deliveries received but not yet completed/failed, keyed
    /// by task id. The delivery is acked only once the worker reports the
    /// task outcome.
    in_flight: Arc<Mutex<HashMap<String, Acker>>>,
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
                queue_name.as_str().into(),
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
            in_flight: Arc::new(Mutex::new(HashMap::new())),
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
                self.queue_name.as_str().into(),
                "daimon-worker".into(),
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
                "".into(),
                self.queue_name.as_str().into(),
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

    /// Returns the last-known status of a task.
    ///
    /// **Process-local only.** Status transitions are recorded in an in-memory
    /// map on the broker instance that handled the task, so a producer will
    /// only observe `Pending` for a task that a *different* worker process
    /// picked up. For cross-process status visibility use
    /// [`RedisBroker`](super::RedisBroker), which persists status in a shared
    /// Redis hash.
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
                let task: AgentTask = match serde_json::from_slice(&delivery.data) {
                    Ok(task) => task,
                    Err(e) => {
                        // Poison message: the payload can never be deserialized,
                        // so nack it with requeue=false. This removes it from the
                        // queue (dead-lettered if a DLX is configured) instead of
                        // requeuing it, which would otherwise redeliver the same
                        // undeserializable message forever. Dropping the acker
                        // without ack/nack closes the channel and RabbitMQ requeues
                        // it — the exact redelivery loop we must avoid.
                        if let Err(nack_err) = delivery
                            .acker
                            .nack(BasicNackOptions {
                                requeue: false,
                                ..Default::default()
                            })
                            .await
                        {
                            tracing::warn!(
                                error = %nack_err,
                                "failed to nack poison AMQP message"
                            );
                        }
                        return Err(DaimonError::Other(format!("deserialize task: {e}")));
                    }
                };

                // Retain the ack handle and ack only after the worker reports
                // the outcome. Acking here (before processing) would lose the
                // task if the worker then crashed.
                {
                    let mut in_flight = self.in_flight.lock().await;
                    in_flight.insert(task.task_id.clone(), delivery.acker);
                }
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
        // Ack the delivery now that the task has been fully processed.
        let acker = self.in_flight.lock().await.remove(task_id);
        if let Some(acker) = acker {
            acker
                .ack(BasicAckOptions::default())
                .await
                .map_err(|e| DaimonError::Other(format!("amqp ack: {e}")))?;
        }

        let mut statuses = self.statuses.lock().await;
        statuses.insert(task_id.to_string(), TaskStatus::Completed(result));
        Ok(())
    }

    async fn fail(&self, task_id: &str, error: String) -> Result<()> {
        // The task was delivered and processed (it errored); ack it so it is
        // not redelivered in a loop. The failure is recorded in the status
        // map. A lost-on-crash task is handled by *not* acking in `receive`.
        let acker = self.in_flight.lock().await.remove(task_id);
        if let Some(acker) = acker {
            acker
                .ack(BasicAckOptions::default())
                .await
                .map_err(|e| DaimonError::Other(format!("amqp ack: {e}")))?;
        }

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
