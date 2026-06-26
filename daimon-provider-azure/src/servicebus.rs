//! Azure Service Bus task broker for distributed agent execution.
//!
//! Uses the Azure Service Bus REST API as a cloud-native alternative to
//! RabbitMQ, providing enterprise-grade messaging with dead-letter queues,
//! sessions, and duplicate detection.
//!
//! Enable with `feature = "servicebus"` on `daimon-provider-azure`.
//!
//! ```ignore
//! use daimon_provider_azure::ServiceBusBroker;
//!
//! let broker = ServiceBusBroker::new(
//!     "https://my-namespace.servicebus.windows.net",
//!     "daimon-tasks",
//!     "SharedAccessKey...",
//! );
//! broker.submit(AgentTask::new("Summarize this")).await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use reqwest::Client;
use serde::Deserialize;
use tokio::sync::Mutex;

use daimon_core::distributed::{AgentTask, TaskBroker, TaskResult, TaskStatus};
use daimon_core::{DaimonError, Result};

/// Distributes agent tasks via Azure Service Bus REST API.
///
/// Tasks are sent as JSON message bodies to a Service Bus queue.
/// Workers receive messages with peek-lock and delete them after
/// processing. Failed tasks have their lock released so they can
/// be retried.
pub struct ServiceBusBroker {
    client: Client,
    namespace_url: String,
    queue_name: String,
    sas_token: String,
    statuses: Arc<Mutex<HashMap<String, TaskStatus>>>,
    lock_tokens: Arc<Mutex<HashMap<String, LockInfo>>>,
    lock_duration: u32,
}

struct LockInfo {
    lock_token: String,
    message_id: String,
}

impl ServiceBusBroker {
    /// Creates a new Service Bus broker.
    ///
    /// * `namespace_url` — Service Bus namespace URL (e.g. `https://my-ns.servicebus.windows.net`)
    /// * `queue_name` — the queue to send/receive messages
    /// * `sas_token` — a Shared Access Signature token for authentication
    ///
    /// Generate a SAS token from the Azure portal or via the Azure CLI:
    /// ```ignore
    /// az servicebus queue authorization-rule keys list \
    ///     --resource-group my-rg --namespace my-ns --queue my-queue \
    ///     --name my-rule --query primaryKey -o tsv
    /// ```
    pub fn new(
        namespace_url: impl Into<String>,
        queue_name: impl Into<String>,
        sas_token: impl Into<String>,
    ) -> Self {
        Self {
            client: Client::new(),
            namespace_url: namespace_url.into().trim_end_matches('/').to_string(),
            queue_name: queue_name.into(),
            sas_token: sas_token.into(),
            statuses: Arc::new(Mutex::new(HashMap::new())),
            lock_tokens: Arc::new(Mutex::new(HashMap::new())),
            lock_duration: 30,
        }
    }

    /// Sets the lock duration in seconds (default: 30).
    pub fn with_lock_duration(mut self, seconds: u32) -> Self {
        self.lock_duration = seconds;
        self
    }

    fn send_url(&self) -> String {
        format!("{}/{}/messages", self.namespace_url, self.queue_name)
    }

    fn receive_url(&self) -> String {
        format!(
            "{}/{}/messages/head?timeout={}",
            self.namespace_url, self.queue_name, self.lock_duration
        )
    }

    fn message_url(&self, message_id: &str, lock_token: &str) -> String {
        format!(
            "{}/{}/messages/{}/{}",
            self.namespace_url, self.queue_name, message_id, lock_token
        )
    }

    fn auth_header(&self) -> String {
        self.sas_token.clone()
    }
}

#[derive(Deserialize)]
struct BrokerProperties {
    #[serde(rename = "MessageId")]
    message_id: Option<String>,
    #[serde(rename = "LockToken")]
    lock_token: Option<String>,
}

impl TaskBroker for ServiceBusBroker {
    async fn submit(&self, task: AgentTask) -> Result<String> {
        let id = task.task_id.clone();
        let json = serde_json::to_string(&task)
            .map_err(|e| DaimonError::Other(format!("servicebus serialize: {e}")))?;

        {
            let mut statuses = self.statuses.lock().await;
            statuses.insert(id.clone(), TaskStatus::Pending);
        }

        let resp = self
            .client
            .post(self.send_url())
            .header("Authorization", self.auth_header())
            .header("Content-Type", "application/json")
            .body(json)
            .send()
            .await
            .map_err(|e| DaimonError::Other(format!("servicebus send: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(DaimonError::Other(format!(
                "servicebus send failed ({status}): {text}"
            )));
        }

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
        let resp = self
            .client
            .post(self.receive_url())
            .header("Authorization", self.auth_header())
            .send()
            .await
            .map_err(|e| DaimonError::Other(format!("servicebus receive: {e}")))?;

        if resp.status().as_u16() == 204 {
            return Ok(None);
        }

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(DaimonError::Other(format!(
                "servicebus receive failed ({status}): {text}"
            )));
        }

        let broker_props_header = resp
            .headers()
            .get("BrokerProperties")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("{}");

        let broker_props: BrokerProperties =
            serde_json::from_str(broker_props_header).unwrap_or(BrokerProperties {
                message_id: None,
                lock_token: None,
            });

        let body = resp
            .text()
            .await
            .map_err(|e| DaimonError::Other(format!("servicebus body: {e}")))?;

        let task: AgentTask = serde_json::from_str(&body)
            .map_err(|e| DaimonError::Other(format!("servicebus deserialize: {e}")))?;

        if let (Some(msg_id), Some(lock_token)) = (broker_props.message_id, broker_props.lock_token)
        {
            let mut locks = self.lock_tokens.lock().await;
            locks.insert(
                task.task_id.clone(),
                LockInfo {
                    lock_token,
                    message_id: msg_id,
                },
            );
        }

        {
            let mut statuses = self.statuses.lock().await;
            statuses.insert(task.task_id.clone(), TaskStatus::Running);
        }

        Ok(Some(task))
    }

    async fn complete(&self, task_id: &str, result: TaskResult) -> Result<()> {
        let lock_info = {
            let mut locks = self.lock_tokens.lock().await;
            locks.remove(task_id)
        };

        if let Some(info) = lock_info {
            let url = self.message_url(&info.message_id, &info.lock_token);
            self.client
                .delete(&url)
                .header("Authorization", self.auth_header())
                .send()
                .await
                .map_err(|e| DaimonError::Other(format!("servicebus delete: {e}")))?;
        }

        let mut statuses = self.statuses.lock().await;
        statuses.insert(task_id.to_string(), TaskStatus::Completed(result));
        Ok(())
    }

    async fn fail(&self, task_id: &str, error: String) -> Result<()> {
        let lock_info = {
            let mut locks = self.lock_tokens.lock().await;
            locks.remove(task_id)
        };

        if let Some(info) = lock_info {
            let url = self.message_url(&info.message_id, &info.lock_token);
            let _ = self
                .client
                .put(&url)
                .header("Authorization", self.auth_header())
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
    fn test_url_construction() {
        let broker = ServiceBusBroker::new(
            "https://my-ns.servicebus.windows.net",
            "my-queue",
            "sas-token",
        );

        assert_eq!(
            broker.send_url(),
            "https://my-ns.servicebus.windows.net/my-queue/messages"
        );
        assert!(broker.receive_url().contains("/my-queue/messages/head"));
    }

    #[test]
    fn test_trailing_slash_stripped() {
        let broker = ServiceBusBroker::new("https://my-ns.servicebus.windows.net/", "q", "token");
        assert_eq!(
            broker.send_url(),
            "https://my-ns.servicebus.windows.net/q/messages"
        );
    }

    #[test]
    fn test_task_serialization_roundtrip() {
        let task =
            AgentTask::new("servicebus test").with_metadata("region", serde_json::json!("eastus"));

        let json = serde_json::to_string(&task).unwrap();
        let deser: AgentTask = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.input, "servicebus test");
    }

    #[test]
    fn test_result_serialization_roundtrip() {
        let result = TaskResult {
            task_id: "t-sb".into(),
            output: "sb result".into(),
            iterations: 2,
            cost: 0.004,
            error: None,
        };

        let json = serde_json::to_string(&result).unwrap();
        let deser: TaskResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.output, "sb result");
    }

    #[test]
    fn test_lock_duration_config() {
        let broker = ServiceBusBroker::new("https://ns.servicebus.windows.net", "q", "tok")
            .with_lock_duration(60);
        assert_eq!(broker.lock_duration, 60);
    }

    #[test]
    fn test_message_url() {
        let broker = ServiceBusBroker::new("https://ns.servicebus.windows.net", "q", "tok");
        let url = broker.message_url("msg-1", "lock-abc");
        assert_eq!(
            url,
            "https://ns.servicebus.windows.net/q/messages/msg-1/lock-abc"
        );
    }
}
