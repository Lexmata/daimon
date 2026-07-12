//! Google Cloud Pub/Sub task broker for distributed agent execution.
//!
//! Uses the Pub/Sub REST API as a cloud-native alternative to RabbitMQ,
//! providing global, durable message delivery with automatic scaling.
//!
//! Enable with `feature = "pubsub"` on `daimon-provider-gemini`.
//!
//! ```ignore
//! use daimon_provider_gemini::PubSubBroker;
//!
//! let broker = PubSubBroker::new("my-project", "daimon-tasks", "daimon-worker-sub").await?;
//! broker.submit(AgentTask::new("Summarize this")).await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use daimon_core::distributed::{AgentTask, TaskBroker, TaskResult, TaskStatus};
use daimon_core::{DaimonError, Result};

const PUBSUB_BASE_URL: &str = "https://pubsub.googleapis.com/v1";

/// Default whole-request timeout for Pub/Sub REST calls.
///
/// Pub/Sub `pull` may long-poll when no messages are available; a `receive`
/// call that hits this deadline is treated as an empty poll (`Ok(None)`)
/// rather than an error, so the bound only protects against wedged
/// connections on publish/ack paths.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

fn build_client() -> Client {
    Client::builder()
        .timeout(DEFAULT_TIMEOUT)
        .build()
        .expect("failed to build HTTP client")
}

/// Distributes agent tasks via Google Cloud Pub/Sub.
///
/// Tasks are published as base64-encoded JSON to a Pub/Sub topic.
/// Workers pull messages from a subscription and acknowledge them
/// after processing.
pub struct PubSubBroker {
    client: Client,
    project: String,
    topic: String,
    subscription: String,
    api_key: Option<String>,
    bearer_token: Option<Arc<Mutex<String>>>,
    statuses: Arc<Mutex<HashMap<String, TaskStatus>>>,
    ack_ids: Arc<Mutex<HashMap<String, String>>>,
}

impl std::fmt::Debug for PubSubBroker {
    /// Hand-written to avoid leaking the plaintext API key or bearer token in
    /// logs or panic output; a derived `Debug` would print them verbatim.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PubSubBroker")
            .field("client", &self.client)
            .field("project", &self.project)
            .field("topic", &self.topic)
            .field("subscription", &self.subscription)
            .field("api_key", &self.api_key.as_ref().map(|_| "[redacted]"))
            .field(
                "bearer_token",
                &self.bearer_token.as_ref().map(|_| "[redacted]"),
            )
            .field("statuses", &self.statuses)
            .field("ack_ids", &self.ack_ids)
            .finish()
    }
}

impl PubSubBroker {
    /// Creates a new Pub/Sub broker with an API key.
    ///
    /// * `project` — GCP project ID
    /// * `topic` — Pub/Sub topic name (e.g. `daimon-tasks`)
    /// * `subscription` — Pub/Sub subscription name (e.g. `daimon-worker-sub`)
    pub fn with_api_key(
        project: impl Into<String>,
        topic: impl Into<String>,
        subscription: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self {
            client: build_client(),
            project: project.into(),
            topic: topic.into(),
            subscription: subscription.into(),
            api_key: Some(api_key.into()),
            bearer_token: None,
            statuses: Arc::new(Mutex::new(HashMap::new())),
            ack_ids: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Creates a new Pub/Sub broker with a bearer token (OAuth2 / service account).
    pub fn with_bearer_token(
        project: impl Into<String>,
        topic: impl Into<String>,
        subscription: impl Into<String>,
        token: impl Into<String>,
    ) -> Self {
        Self {
            client: build_client(),
            project: project.into(),
            topic: topic.into(),
            subscription: subscription.into(),
            api_key: None,
            bearer_token: Some(Arc::new(Mutex::new(token.into()))),
            statuses: Arc::new(Mutex::new(HashMap::new())),
            ack_ids: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Updates the bearer token (for token refresh).
    pub async fn set_bearer_token(&self, token: impl Into<String>) {
        if let Some(ref t) = self.bearer_token {
            let mut guard = t.lock().await;
            *guard = token.into();
        }
    }

    fn topic_path(&self) -> String {
        format!("projects/{}/topics/{}", self.project, self.topic)
    }

    fn subscription_path(&self) -> String {
        format!(
            "projects/{}/subscriptions/{}",
            self.project, self.subscription
        )
    }

    /// Attaches credentials to a request.
    ///
    /// API keys ride in the `x-goog-api-key` header rather than a `?key=`
    /// query parameter, which would leak into logs via reqwest error messages
    /// that include the full URL.
    async fn auth_request(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(ref key) = self.api_key {
            builder.header("x-goog-api-key", key.as_str())
        } else if let Some(ref token_lock) = self.bearer_token {
            let token = token_lock.lock().await;
            builder.bearer_auth(token.clone())
        } else {
            builder
        }
    }
}

#[derive(Serialize)]
struct PublishRequest {
    messages: Vec<PubSubMessage>,
}

#[derive(Serialize)]
struct PubSubMessage {
    data: String,
}

#[derive(Deserialize)]
struct PullResponse {
    #[serde(default)]
    received_messages: Vec<ReceivedMessage>,
}

#[derive(Deserialize)]
struct ReceivedMessage {
    ack_id: String,
    message: PulledMessage,
}

#[derive(Deserialize)]
struct PulledMessage {
    data: String,
}

#[derive(Serialize)]
struct AcknowledgeRequest {
    ack_ids: Vec<String>,
}

impl TaskBroker for PubSubBroker {
    async fn submit(&self, task: AgentTask) -> Result<String> {
        use base64::Engine;

        let id = task.task_id.clone();
        let json = serde_json::to_string(&task)
            .map_err(|e| DaimonError::Other(format!("pubsub serialize: {e}")))?;

        let encoded = base64::engine::general_purpose::STANDARD.encode(json.as_bytes());

        {
            let mut statuses = self.statuses.lock().await;
            statuses.insert(id.clone(), TaskStatus::Pending);
        }

        let url = format!("{}/{}:publish", PUBSUB_BASE_URL, self.topic_path());
        let body = PublishRequest {
            messages: vec![PubSubMessage { data: encoded }],
        };

        let req = self.client.post(&url).json(&body);
        let req = self.auth_request(req).await;

        let resp = req
            .send()
            .await
            .map_err(|e| DaimonError::Other(format!("pubsub publish: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(DaimonError::Other(format!(
                "pubsub publish failed ({status}): {text}"
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
        use base64::Engine;

        let url = format!("{}/{}:pull", PUBSUB_BASE_URL, self.subscription_path());
        let body = serde_json::json!({ "maxMessages": 1 });

        let req = self.client.post(&url).json(&body);
        let req = self.auth_request(req).await;

        let resp = match req.send().await {
            Ok(resp) => resp,
            // A pull that idles past the client timeout is an empty poll, not
            // a failure; callers poll `receive` in a loop.
            Err(e) if e.is_timeout() => return Ok(None),
            Err(e) => return Err(DaimonError::Other(format!("pubsub pull: {e}"))),
        };

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(DaimonError::Other(format!(
                "pubsub pull failed ({status}): {text}"
            )));
        }

        let pull_resp: PullResponse = resp
            .json()
            .await
            .map_err(|e| DaimonError::Other(format!("pubsub pull parse: {e}")))?;

        if pull_resp.received_messages.is_empty() {
            return Ok(None);
        }

        let received = &pull_resp.received_messages[0];
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&received.message.data)
            .map_err(|e| DaimonError::Other(format!("pubsub base64 decode: {e}")))?;

        let task: AgentTask = serde_json::from_slice(&decoded)
            .map_err(|e| DaimonError::Other(format!("pubsub deserialize: {e}")))?;

        {
            let mut ack_ids = self.ack_ids.lock().await;
            ack_ids.insert(task.task_id.clone(), received.ack_id.clone());
        }

        {
            let mut statuses = self.statuses.lock().await;
            statuses.insert(task.task_id.clone(), TaskStatus::Running);
        }

        Ok(Some(task))
    }

    async fn complete(&self, task_id: &str, result: TaskResult) -> Result<()> {
        let ack_id = {
            let mut ack_ids = self.ack_ids.lock().await;
            ack_ids.remove(task_id)
        };

        if let Some(ack_id) = ack_id {
            let url = format!(
                "{}/{}:acknowledge",
                PUBSUB_BASE_URL,
                self.subscription_path()
            );
            let body = AcknowledgeRequest {
                ack_ids: vec![ack_id],
            };

            let req = self.client.post(&url).json(&body);
            let req = self.auth_request(req).await;

            req.send()
                .await
                .map_err(|e| DaimonError::Other(format!("pubsub ack: {e}")))?;
        }

        let mut statuses = self.statuses.lock().await;
        statuses.insert(task_id.to_string(), TaskStatus::Completed(result));
        Ok(())
    }

    async fn fail(&self, task_id: &str, error: String) -> Result<()> {
        {
            let mut ack_ids = self.ack_ids.lock().await;
            ack_ids.remove(task_id);
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
    fn test_topic_path() {
        let broker = PubSubBroker::with_api_key("my-project", "my-topic", "my-sub", "key123");
        assert_eq!(broker.topic_path(), "projects/my-project/topics/my-topic");
    }

    #[test]
    fn test_subscription_path() {
        let broker = PubSubBroker::with_api_key("my-project", "my-topic", "my-sub", "key123");
        assert_eq!(
            broker.subscription_path(),
            "projects/my-project/subscriptions/my-sub"
        );
    }

    #[test]
    fn test_api_key_sent_as_header_not_query_param() {
        let broker = PubSubBroker::with_api_key("proj", "topic", "sub", "pubsub-secret-key");
        let builder = broker.client.post("https://example.com/v1");
        let req = futures::executor::block_on(broker.auth_request(builder))
            .build()
            .unwrap();
        assert!(
            !req.url().as_str().contains("pubsub-secret-key"),
            "API key must not appear in the request URL: {}",
            req.url()
        );
        assert_eq!(
            req.headers().get("x-goog-api-key").unwrap(),
            "pubsub-secret-key"
        );
    }

    #[test]
    fn test_task_serialization_roundtrip() {
        let task =
            AgentTask::new("pubsub test").with_metadata("region", serde_json::json!("us-central1"));

        let json = serde_json::to_string(&task).unwrap();
        let deser: AgentTask = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.input, "pubsub test");
    }

    #[test]
    fn test_base64_roundtrip() {
        use base64::Engine;

        let task = AgentTask::new("encode me");
        let json = serde_json::to_string(&task).unwrap();
        let encoded = base64::engine::general_purpose::STANDARD.encode(json.as_bytes());
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&encoded)
            .unwrap();
        let deser: AgentTask = serde_json::from_slice(&decoded).unwrap();
        assert_eq!(deser.input, "encode me");
    }

    #[test]
    fn test_result_serialization_roundtrip() {
        let result = TaskResult {
            task_id: "t-pubsub".into(),
            output: "pubsub result".into(),
            iterations: 1,
            cost: 0.003,
            error: None,
        };

        let json = serde_json::to_string(&result).unwrap();
        let deser: TaskResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.output, "pubsub result");
    }

    #[test]
    fn test_debug_redacts_api_key() {
        let broker = PubSubBroker::with_api_key(
            "my-project",
            "my-topic",
            "my-sub",
            "pubsub-supersecret-api-key",
        );
        let dbg = format!("{broker:?}");
        assert!(
            !dbg.contains("pubsub-supersecret-api-key"),
            "Debug output must not contain the plaintext API key: {dbg}"
        );
        assert!(dbg.contains("[redacted]"));
    }

    #[test]
    fn test_debug_redacts_bearer_token() {
        let broker = PubSubBroker::with_bearer_token(
            "my-project",
            "my-topic",
            "my-sub",
            "pubsub-supersecret-bearer-token",
        );
        let dbg = format!("{broker:?}");
        assert!(
            !dbg.contains("pubsub-supersecret-bearer-token"),
            "Debug output must not contain the plaintext bearer token: {dbg}"
        );
        assert!(dbg.contains("[redacted]"));
    }
}
