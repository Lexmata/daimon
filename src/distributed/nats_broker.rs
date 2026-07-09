//! NATS JetStream task broker for distributed agent execution.
//!
//! Uses NATS JetStream for durable, at-least-once task delivery with
//! explicit, deferred acknowledgement. Enable with `feature = "nats"`.
//!
//! # Acknowledgement model
//!
//! A message is **not** acked when [`receive`](NatsBroker::receive) hands the
//! task to a worker. The ack handle is retained and the message is only acked
//! once the worker reports the outcome via
//! [`complete`](NatsBroker::complete) or [`fail`](NatsBroker::fail). If the
//! worker process crashes in between, the message is never acked and JetStream
//! redelivers it after the consumer's ack-wait window — this is what makes the
//! at-least-once guarantee real. (Acking inside `receive`, before processing,
//! would silently drop tasks on a mid-flight crash.)
//!
//! # Status visibility
//!
//! Task status is tracked in a JetStream key-value bucket, so transitions
//! made by one process (e.g. a worker marking a task `running`/`completed`)
//! are visible to every other process holding a broker for the same stream
//! (e.g. the producer that submitted the task). The bucket is replicated by
//! JetStream just like the task stream itself. See [`NatsBroker::status`].
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
use async_nats::jetstream::kv::Store;
use async_nats::jetstream::message::Acker;
use tokio::sync::Mutex;

use crate::error::{DaimonError, Result};

use super::broker::TaskBroker;
use super::types::{AgentTask, TaskResult, TaskStatus};

/// Distributes agent tasks via NATS JetStream.
///
/// Tasks are published to a JetStream stream and consumed by workers
/// via a pull-based consumer. Status is tracked in a replicated JetStream
/// key-value bucket, so it is visible across processes (see
/// [`NatsBroker::status`]). Acknowledgement is deferred until a task
/// is completed or failed (see the module docs).
pub struct NatsBroker {
    client: async_nats::Client,
    jetstream: async_nats::jetstream::Context,
    stream_name: String,
    subject_prefix: String,
    /// Cross-process task status, backed by a JetStream KV bucket. Each task
    /// stores its state under its `task_id` and, when completed, the result
    /// JSON under `{task_id}.result`.
    status_kv: Store,
    consumer: Arc<Mutex<Option<Consumer<PullConfig>>>>,
    /// Ack handles for messages received but not yet completed/failed, keyed
    /// by task id. The handle is dropped (and the message acked) only once the
    /// worker reports the task outcome.
    in_flight: Arc<Mutex<HashMap<String, Acker>>>,
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

        // Create-or-open the KV bucket that holds cross-process task status.
        // `create_key_value` fails if the underlying KV stream already exists,
        // so on error we fall back to opening the existing bucket. `history: 1`
        // keeps only the latest state per key — we never need old revisions.
        let bucket = format!("{stream_name}-status");
        let status_kv = match jetstream
            .create_key_value(async_nats::jetstream::kv::Config {
                bucket: bucket.clone(),
                history: 1,
                ..Default::default()
            })
            .await
        {
            Ok(store) => store,
            Err(_) => jetstream
                .get_key_value(&bucket)
                .await
                .map_err(|e| DaimonError::Other(format!("nats open kv bucket: {e}")))?,
        };

        Ok(Self {
            client,
            jetstream,
            stream_name,
            subject_prefix,
            status_kv,
            consumer: Arc::new(Mutex::new(None)),
            in_flight: Arc::new(Mutex::new(HashMap::new())),
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

        self.status_kv
            .put(&id, status_marker(&TaskStatus::Pending).into())
            .await
            .map_err(|e| DaimonError::Other(format!("nats kv put pending: {e}")))?;

        self.jetstream
            .publish(self.task_subject(), json.into())
            .await
            .map_err(|e| DaimonError::Other(format!("nats publish: {e}")))?
            .await
            .map_err(|e| DaimonError::Other(format!("nats publish ack: {e}")))?;

        Ok(id)
    }

    /// Returns the last-known status of a task.
    ///
    /// **Cross-process.** Status transitions are persisted to a replicated
    /// JetStream KV bucket, so this reflects updates made by any process
    /// holding a broker for the same stream — a producer sees `running`,
    /// `completed`, and `failed` transitions made by a separate worker
    /// process. A task with no recorded state (e.g. never submitted, or the
    /// bucket entry expired) reports [`TaskStatus::Pending`].
    async fn status(&self, task_id: &str) -> Result<TaskStatus> {
        let raw = self
            .status_kv
            .get(task_id)
            .await
            .map_err(|e| DaimonError::Other(format!("nats kv get status: {e}")))?;

        let Some(bytes) = raw else {
            return Ok(TaskStatus::Pending);
        };
        let status_str = String::from_utf8_lossy(&bytes);

        match parse_status_marker(&status_str) {
            StatusMarker::Pending => Ok(TaskStatus::Pending),
            StatusMarker::Running => Ok(TaskStatus::Running),
            StatusMarker::Failed(msg) => Ok(TaskStatus::Failed(msg)),
            StatusMarker::Completed => {
                let result_json = self
                    .status_kv
                    .get(result_key(task_id))
                    .await
                    .map_err(|e| DaimonError::Other(format!("nats kv get result: {e}")))?;

                match result_json {
                    Some(bytes) => {
                        let result: TaskResult = serde_json::from_slice(&bytes)
                            .map_err(|e| DaimonError::Other(format!("deserialize result: {e}")))?;
                        Ok(TaskStatus::Completed(result))
                    }
                    None => Ok(TaskStatus::Completed(TaskResult {
                        task_id: task_id.to_string(),
                        output: String::new(),
                        iterations: 0,
                        cost: 0.0,
                        error: None,
                    })),
                }
            }
        }
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
                // Split the ack handle off so we can retain it and ack only
                // after the worker reports the outcome. Acking here (before
                // processing) would lose the task if the worker then crashed.
                let (msg, acker) = msg.split();

                let task: AgentTask = match serde_json::from_slice(&msg.payload) {
                    Ok(task) => task,
                    Err(e) => {
                        // Poison message: the payload can never be deserialized,
                        // so terminate it (AckKind::Term) to stop JetStream from
                        // redelivering it forever. Dropping the acker here without
                        // acking would leave the message un-acked and it would be
                        // redelivered on every ack-wait expiry in an endless loop.
                        if let Err(term_err) =
                            acker.ack_with(async_nats::jetstream::AckKind::Term).await
                        {
                            tracing::warn!(
                                error = %term_err,
                                "failed to terminate poison NATS message"
                            );
                        }
                        return Err(DaimonError::Other(format!("deserialize task: {e}")));
                    }
                };

                {
                    let mut in_flight = self.in_flight.lock().await;
                    in_flight.insert(task.task_id.clone(), acker);
                }
                self.status_kv
                    .put(&task.task_id, status_marker(&TaskStatus::Running).into())
                    .await
                    .map_err(|e| DaimonError::Other(format!("nats kv put running: {e}")))?;

                Ok(Some(task))
            }
            Some(Err(e)) => Err(DaimonError::Other(format!("nats message error: {e}"))),
            None => Ok(None),
        }
    }

    async fn complete(&self, task_id: &str, result: TaskResult) -> Result<()> {
        // Persist the outcome to KV *before* acking. If the process crashed
        // between an early ack and the KV writes, JetStream would consider the
        // message done and never redeliver it while the status bucket still
        // said "running" — the task would be lost forever. Writing first means
        // a crash between write and ack merely causes a redelivery, and the
        // duplicate KV writes on reprocessing are idempotent.
        //
        // Store the result JSON first, then flip the status marker to
        // "completed" — a reader that observes "completed" is then guaranteed
        // to find the result already present.
        let json = serde_json::to_string(&result)
            .map_err(|e| DaimonError::Other(format!("serialize result: {e}")))?;
        self.status_kv
            .put(result_key(task_id), json.into())
            .await
            .map_err(|e| DaimonError::Other(format!("nats kv put result: {e}")))?;
        self.status_kv
            .put(
                task_id,
                status_marker(&TaskStatus::Completed(result)).into(),
            )
            .await
            .map_err(|e| DaimonError::Other(format!("nats kv put completed: {e}")))?;

        // Ack last, now that the outcome is durably recorded, so JetStream
        // can advance and won't redeliver the message.
        let acker = self.in_flight.lock().await.remove(task_id);
        if let Some(acker) = acker {
            acker
                .ack()
                .await
                .map_err(|e| DaimonError::Other(format!("nats ack: {e}")))?;
        }
        Ok(())
    }

    async fn fail(&self, task_id: &str, error: String) -> Result<()> {
        // Record the failure in KV *before* acking, for the same reason as
        // `complete`: an early ack followed by a crash would drop the message
        // from JetStream while the status bucket still read "running".
        self.status_kv
            .put(task_id, status_marker(&TaskStatus::Failed(error)).into())
            .await
            .map_err(|e| DaimonError::Other(format!("nats kv put failed: {e}")))?;

        // The task was delivered and processed (it errored); ack it so it is
        // not redelivered in a loop. A lost-on-crash task is handled by *not*
        // acking in `receive`.
        let acker = self.in_flight.lock().await.remove(task_id);
        if let Some(acker) = acker {
            acker
                .ack()
                .await
                .map_err(|e| DaimonError::Other(format!("nats ack: {e}")))?;
        }
        Ok(())
    }
}

/// KV key under which a completed task's [`TaskResult`] JSON is stored.
///
/// Kept separate from the status marker key (`task_id`) so the small status
/// string and the larger result payload can be written and read independently
/// — mirroring [`RedisBroker`](super::RedisBroker)'s split status/results
/// hashes.
fn result_key(task_id: &str) -> String {
    format!("{task_id}.result")
}

/// Parsed form of a KV status marker string. `Completed` carries no payload
/// here: the [`TaskResult`] lives under a separate key and is fetched only
/// when the marker says the task is done (see [`NatsBroker::status`]).
#[derive(Debug, PartialEq, Eq)]
enum StatusMarker {
    Pending,
    Running,
    Completed,
    Failed(String),
}

/// Encodes a [`TaskStatus`] to the marker string stored in KV. Uses the same
/// scheme as [`RedisBroker`](super::RedisBroker):
/// `"pending"` / `"running"` / `"completed"` / `"failed:{msg}"`. The
/// `Completed` result payload is stored separately under [`result_key`].
fn status_marker(status: &TaskStatus) -> String {
    match status {
        TaskStatus::Pending => "pending".to_string(),
        TaskStatus::Running => "running".to_string(),
        TaskStatus::Completed(_) => "completed".to_string(),
        TaskStatus::Failed(msg) => format!("failed:{msg}"),
    }
}

/// Decodes a KV status marker string. An unrecognised marker is treated as
/// [`StatusMarker::Pending`], matching the Redis broker's fallback.
fn parse_status_marker(s: &str) -> StatusMarker {
    match s {
        "pending" => StatusMarker::Pending,
        "running" => StatusMarker::Running,
        "completed" => StatusMarker::Completed,
        other => match other.strip_prefix("failed:") {
            Some(msg) => StatusMarker::Failed(msg.to_string()),
            None => StatusMarker::Pending,
        },
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

    #[test]
    fn test_result_key_derivation() {
        assert_eq!(result_key("task-abc"), "task-abc.result");
    }

    #[test]
    fn test_status_marker_encode() {
        assert_eq!(status_marker(&TaskStatus::Pending), "pending");
        assert_eq!(status_marker(&TaskStatus::Running), "running");
        assert_eq!(
            status_marker(&TaskStatus::Completed(TaskResult {
                task_id: "t".into(),
                output: "ok".into(),
                iterations: 1,
                cost: 0.0,
                error: None,
            })),
            "completed"
        );
        assert_eq!(
            status_marker(&TaskStatus::Failed("boom".into())),
            "failed:boom"
        );
    }

    #[test]
    fn test_status_marker_parse() {
        assert_eq!(parse_status_marker("pending"), StatusMarker::Pending);
        assert_eq!(parse_status_marker("running"), StatusMarker::Running);
        assert_eq!(parse_status_marker("completed"), StatusMarker::Completed);
        assert_eq!(
            parse_status_marker("failed:something broke"),
            StatusMarker::Failed("something broke".into())
        );
        // Unknown markers fall back to Pending, matching the Redis broker.
        assert_eq!(parse_status_marker("garbage"), StatusMarker::Pending);
    }

    #[test]
    fn test_status_marker_roundtrip_failed_with_colon() {
        // A failure message containing a colon must survive round-trip: only
        // the first "failed:" prefix is stripped, the rest is the message.
        let msg = "http error: 500: upstream";
        let encoded = status_marker(&TaskStatus::Failed(msg.into()));
        assert_eq!(
            parse_status_marker(&encoded),
            StatusMarker::Failed(msg.into())
        );
    }
}
