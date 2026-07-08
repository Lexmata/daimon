//! AWS SQS task broker for distributed agent execution.
//!
//! Uses Amazon Simple Queue Service as a cloud-native alternative to
//! RabbitMQ, leveraging SQS visibility timeouts for in-flight task
//! tracking and SQS message attributes for status management.
//!
//! Enable with `feature = "sqs"` on `daimon-provider-bedrock`.
//!
//! ```ignore
//! use daimon_provider_bedrock::SqsBroker;
//!
//! let broker = SqsBroker::new("https://sqs.us-east-1.amazonaws.com/123456789/daimon-tasks").await?;
//! broker.submit(AgentTask::new("Summarize this")).await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use aws_sdk_sqs::Client as SqsClient;
use tokio::sync::Mutex;

use daimon_core::distributed::{AgentTask, TaskBroker, TaskResult, TaskStatus};
use daimon_core::{DaimonError, Result};

/// Distributes agent tasks via Amazon SQS.
///
/// Tasks are serialized as JSON message bodies. SQS visibility timeout
/// prevents other workers from receiving a message while it's being
/// processed. Status tracking uses an in-memory map on the local process.
pub struct SqsBroker {
    client: SqsClient,
    queue_url: String,
    statuses: Arc<Mutex<HashMap<String, TaskStatus>>>,
    receipt_handles: Arc<Mutex<HashMap<String, String>>>,
    visibility_timeout: i32,
}

impl SqsBroker {
    /// Creates a new SQS broker using default AWS credentials.
    ///
    /// * `queue_url` — the full SQS queue URL
    pub async fn new(queue_url: impl Into<String>) -> Result<Self> {
        let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        let client = SqsClient::new(&config);
        Ok(Self {
            client,
            queue_url: queue_url.into(),
            statuses: Arc::new(Mutex::new(HashMap::new())),
            receipt_handles: Arc::new(Mutex::new(HashMap::new())),
            visibility_timeout: 300,
        })
    }

    /// Creates a broker with a specific AWS region.
    pub async fn with_region(queue_url: impl Into<String>, region: &str) -> Result<Self> {
        let config = aws_config::defaults(aws_config::BehaviorVersion::latest())
            .http_client(crate::modern_https_client())
            .region(aws_config::Region::new(region.to_string()))
            .load()
            .await;
        let client = SqsClient::new(&config);
        Ok(Self {
            client,
            queue_url: queue_url.into(),
            statuses: Arc::new(Mutex::new(HashMap::new())),
            receipt_handles: Arc::new(Mutex::new(HashMap::new())),
            visibility_timeout: 300,
        })
    }

    /// Creates a broker from an existing SQS client.
    pub fn from_client(client: SqsClient, queue_url: impl Into<String>) -> Self {
        Self {
            client,
            queue_url: queue_url.into(),
            statuses: Arc::new(Mutex::new(HashMap::new())),
            receipt_handles: Arc::new(Mutex::new(HashMap::new())),
            visibility_timeout: 300,
        }
    }

    /// Sets the SQS visibility timeout in seconds (default: 300).
    ///
    /// This controls how long a message stays invisible after being
    /// received, giving the worker time to process it.
    pub fn with_visibility_timeout(mut self, seconds: i32) -> Self {
        self.visibility_timeout = seconds;
        self
    }

    /// Returns the SQS queue URL.
    pub fn queue_url(&self) -> &str {
        &self.queue_url
    }
}

impl TaskBroker for SqsBroker {
    async fn submit(&self, task: AgentTask) -> Result<String> {
        let id = task.task_id.clone();
        let json = serde_json::to_string(&task)
            .map_err(|e| DaimonError::Other(format!("sqs serialize task: {e}")))?;

        {
            let mut statuses = self.statuses.lock().await;
            statuses.insert(id.clone(), TaskStatus::Pending);
        }

        self.client
            .send_message()
            .queue_url(&self.queue_url)
            .message_body(&json)
            .message_group_id("daimon-tasks")
            .send()
            .await
            .map_err(|e| DaimonError::Other(format!("sqs send: {e}")))?;

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
        let output = self
            .client
            .receive_message()
            .queue_url(&self.queue_url)
            .max_number_of_messages(1)
            .wait_time_seconds(5)
            .visibility_timeout(self.visibility_timeout)
            .send()
            .await
            .map_err(|e| DaimonError::Other(format!("sqs receive: {e}")))?;

        let messages = output.messages();
        if messages.is_empty() {
            return Ok(None);
        }

        let msg = &messages[0];
        let body = msg
            .body()
            .ok_or_else(|| DaimonError::Other("sqs message has no body".into()))?;

        let task: AgentTask = serde_json::from_str(body)
            .map_err(|e| DaimonError::Other(format!("sqs deserialize task: {e}")))?;

        if let Some(receipt) = msg.receipt_handle() {
            let mut handles = self.receipt_handles.lock().await;
            handles.insert(task.task_id.clone(), receipt.to_string());
        }

        {
            let mut statuses = self.statuses.lock().await;
            statuses.insert(task.task_id.clone(), TaskStatus::Running);
        }

        Ok(Some(task))
    }

    async fn complete(&self, task_id: &str, result: TaskResult) -> Result<()> {
        let receipt = {
            let mut handles = self.receipt_handles.lock().await;
            handles.remove(task_id)
        };

        if let Some(receipt_handle) = receipt {
            self.client
                .delete_message()
                .queue_url(&self.queue_url)
                .receipt_handle(&receipt_handle)
                .send()
                .await
                .map_err(|e| DaimonError::Other(format!("sqs delete: {e}")))?;
        }

        let mut statuses = self.statuses.lock().await;
        statuses.insert(task_id.to_string(), TaskStatus::Completed(result));
        Ok(())
    }

    async fn fail(&self, task_id: &str, error: String) -> Result<()> {
        let receipt = {
            let mut handles = self.receipt_handles.lock().await;
            handles.remove(task_id)
        };

        if let Some(receipt_handle) = receipt {
            let _ = self
                .client
                .change_message_visibility()
                .queue_url(&self.queue_url)
                .receipt_handle(&receipt_handle)
                .visibility_timeout(0)
                .send()
                .await;
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
    fn test_task_serialization_roundtrip() {
        let task = AgentTask::new("sqs test")
            .with_run_id("r1")
            .with_metadata("priority", serde_json::json!(1));

        let json = serde_json::to_string(&task).unwrap();
        let deser: AgentTask = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.input, "sqs test");
        assert_eq!(deser.run_id.as_deref(), Some("r1"));
    }

    #[test]
    fn test_result_serialization_roundtrip() {
        let result = TaskResult {
            task_id: "t-sqs".into(),
            output: "sqs result".into(),
            iterations: 2,
            cost: 0.005,
            error: None,
        };

        let json = serde_json::to_string(&result).unwrap();
        let deser: TaskResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.output, "sqs result");
    }

    #[test]
    fn test_default_visibility_timeout() {
        let broker = SqsBroker {
            client: {
                let config = aws_sdk_sqs::Config::builder()
                    .behavior_version(aws_sdk_sqs::config::BehaviorVersion::latest())
                    .region(aws_sdk_sqs::config::Region::new("us-east-1"))
                    .build();
                SqsClient::from_conf(config)
            },
            queue_url: "https://sqs.us-east-1.amazonaws.com/123/test".into(),
            statuses: Arc::new(Mutex::new(HashMap::new())),
            receipt_handles: Arc::new(Mutex::new(HashMap::new())),
            visibility_timeout: 300,
        };
        assert_eq!(broker.visibility_timeout, 300);

        let broker = broker.with_visibility_timeout(600);
        assert_eq!(broker.visibility_timeout, 600);
    }
}
