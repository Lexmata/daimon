//! Task worker that consumes tasks from a broker and runs them through agents.

use std::sync::Arc;

use crate::agent::Agent;
use crate::error::Result;

use super::broker::ErasedTaskBroker;
use super::types::{AgentTask, TaskResult};

/// Factory function type for creating agent instances per-task.
///
/// Each task gets a fresh agent so conversations don't bleed across tasks.
pub type AgentFactory = Arc<dyn Fn() -> Agent + Send + Sync>;

/// How long workers sleep between empty polls of an idle broker.
///
/// Applies only to brokers whose `receive` legitimately comes up empty when
/// the queue is idle (`none_means_closed() == false`, e.g. Redis BRPOP with a
/// timeout or a NATS fetch). The delay stops such brokers with fast-returning
/// empty polls from being hammered in a hot loop.
const IDLE_POLL_DELAY: std::time::Duration = std::time::Duration::from_millis(100);

/// Waits for the next task from an erased broker, distinguishing "idle" from
/// "closed" via [`ErasedTaskBroker::none_means_closed_erased`].
///
/// Returns `Ok(Some(task))` when a task arrives and `Ok(None)` only when the
/// broker is permanently closed. An empty poll from a broker that merely timed
/// out (Redis, NATS, SQS, …) is retried after [`IDLE_POLL_DELAY`] instead of
/// being misread as a shutdown signal — previously a worker on such a broker
/// exited for good the first time its queue sat idle for one poll interval.
pub(super) async fn next_task(broker: &Arc<dyn ErasedTaskBroker>) -> Result<Option<AgentTask>> {
    loop {
        match broker.receive_erased().await? {
            Some(task) => return Ok(Some(task)),
            None if broker.none_means_closed_erased() => return Ok(None),
            None => tokio::time::sleep(IDLE_POLL_DELAY).await,
        }
    }
}

/// A worker that pulls tasks from a [`TaskBroker`](super::TaskBroker) and
/// executes them using agent instances from a factory.
///
/// Run the worker in a background `tokio::spawn` loop, or call
/// [`run_once`](Self::run_once) for single-task processing.
pub struct TaskWorker {
    broker: Arc<dyn ErasedTaskBroker>,
    factory: AgentFactory,
}

impl TaskWorker {
    /// Creates a new worker.
    ///
    /// The `factory` closure is called once per task to produce a fresh
    /// agent. This ensures each task gets independent memory.
    pub fn new<B, F>(broker: B, factory: F) -> Self
    where
        B: super::broker::TaskBroker + 'static,
        F: Fn() -> Agent + Send + Sync + 'static,
    {
        Self {
            broker: Arc::new(broker),
            factory: Arc::new(factory),
        }
    }

    /// Creates a worker from an already-erased broker and a factory.
    pub fn from_erased(broker: Arc<dyn ErasedTaskBroker>, factory: AgentFactory) -> Self {
        Self { broker, factory }
    }

    /// Waits for a single task, executes it, and reports the result back
    /// to the broker. Returns `Ok(None)` only if the broker is permanently
    /// closed; an idle queue is polled until a task arrives.
    ///
    /// A task whose agent errored is reported to the broker via `fail` (its
    /// status becomes `Failed`), matching [`run_parallel`](Self::run_parallel).
    /// The returned [`TaskResult`] still carries the error message in its
    /// `error` field.
    pub async fn run_once(&self) -> Result<Option<TaskResult>> {
        let task = match next_task(&self.broker).await? {
            Some(t) => t,
            None => return Ok(None),
        };

        let result = self.execute_task(&task).await;

        match &result {
            // Route agent errors to `fail` so the broker records `Failed`,
            // exactly as `run_parallel` does — previously this path marked
            // the task `Completed` with the error merely embedded in the
            // result, so the same failure produced two different statuses
            // depending on which worker loop picked it up.
            Ok(tr) => match &tr.error {
                Some(err) => {
                    self.broker.fail_erased(&task.task_id, err.clone()).await?;
                }
                None => {
                    self.broker
                        .complete_erased(&task.task_id, tr.clone())
                        .await?;
                }
            },
            Err(e) => {
                self.broker
                    .fail_erased(&task.task_id, e.to_string())
                    .await?;
            }
        }

        result.map(Some)
    }

    /// Runs the worker loop indefinitely, processing tasks until the broker
    /// is permanently closed. Idle polls (e.g. a Redis/NATS poll timeout on an
    /// empty queue) do not stop the loop.
    pub async fn run(&self) -> Result<()> {
        loop {
            match self.run_once().await? {
                Some(_) => continue,
                None => {
                    tracing::info!("broker closed, worker exiting");
                    return Ok(());
                }
            }
        }
    }

    /// Runs up to `n` tasks concurrently using `tokio::JoinSet`, until the
    /// broker is permanently closed. Idle polls do not stop the loop.
    pub async fn run_parallel(&self, concurrency: usize) -> Result<()> {
        use tokio::task::JoinSet;

        let mut join_set = JoinSet::new();

        loop {
            while join_set.len() < concurrency {
                let task = match next_task(&self.broker).await? {
                    Some(t) => t,
                    None => {
                        while let Some(result) = join_set.join_next().await {
                            if let Err(e) = result {
                                tracing::warn!("worker task panicked: {e}");
                            }
                        }
                        return Ok(());
                    }
                };

                let broker = Arc::clone(&self.broker);
                let factory = Arc::clone(&self.factory);
                join_set.spawn(async move {
                    let worker = TaskWorker { broker, factory };
                    if let Err(e) = worker.execute_and_report(&task).await {
                        tracing::warn!(task_id = %task.task_id, "task failed: {e}");
                    }
                });
            }

            if let Some(Err(e)) = join_set.join_next().await {
                tracing::warn!("worker task panicked: {e}");
            }
        }
    }

    async fn execute_task(&self, task: &AgentTask) -> Result<TaskResult> {
        let agent = (self.factory)();

        match agent.prompt(&task.input).await {
            Ok(response) => Ok(TaskResult {
                task_id: task.task_id.clone(),
                output: response.final_text,
                iterations: response.iterations,
                cost: response.cost,
                error: None,
            }),
            Err(e) => Ok(TaskResult {
                task_id: task.task_id.clone(),
                output: String::new(),
                iterations: 0,
                cost: 0.0,
                error: Some(e.to_string()),
            }),
        }
    }

    async fn execute_and_report(&self, task: &AgentTask) -> Result<()> {
        let result = self.execute_task(task).await?;
        if result.error.is_some() {
            self.broker
                .fail_erased(&task.task_id, result.error.unwrap_or_default())
                .await?;
        } else {
            self.broker.complete_erased(&task.task_id, result).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::{InProcessBroker, TaskBroker};
    use crate::error::Result as DResult;
    use crate::model::Model;
    use crate::model::types::*;
    use crate::stream::ResponseStream;

    struct EchoModel;

    impl Model for EchoModel {
        async fn generate(&self, request: &ChatRequest) -> DResult<ChatResponse> {
            let last = request
                .messages
                .last()
                .and_then(|m| m.content.as_deref())
                .unwrap_or("none");
            Ok(ChatResponse {
                message: Message::assistant(format!("Echo: {last}")),
                stop_reason: StopReason::EndTurn,
                usage: Some(Usage::default()),
            })
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> DResult<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    fn make_broker_and_worker() -> (InProcessBroker, TaskWorker) {
        let broker = InProcessBroker::new(16);
        let worker = TaskWorker::new(broker.clone(), || {
            Agent::builder().model(EchoModel).build().unwrap()
        });
        (broker, worker)
    }

    #[tokio::test]
    async fn test_run_once() {
        let (broker, worker) = make_broker_and_worker();

        let task = AgentTask::new("hello distributed");
        let id = broker.submit(task).await.unwrap();

        let result = worker.run_once().await.unwrap().unwrap();
        assert_eq!(result.task_id, id);
        assert!(result.output.contains("hello distributed"));
        assert!(result.error.is_none());
    }

    #[tokio::test]
    async fn test_status_updates() {
        let (broker, worker) = make_broker_and_worker();

        let task = AgentTask::new("track me");
        let id = broker.submit(task).await.unwrap();

        let _ = worker.run_once().await.unwrap();

        let status = broker.status(&id).await.unwrap();
        assert!(matches!(status, super::super::TaskStatus::Completed(_)));
    }

    #[tokio::test]
    async fn test_multiple_tasks_sequential() {
        let (broker, worker) = make_broker_and_worker();

        let id1 = broker.submit(AgentTask::new("first")).await.unwrap();
        let id2 = broker.submit(AgentTask::new("second")).await.unwrap();

        let r1 = worker.run_once().await.unwrap().unwrap();
        let r2 = worker.run_once().await.unwrap().unwrap();

        assert_eq!(r1.task_id, id1);
        assert_eq!(r2.task_id, id2);
    }

    #[tokio::test]
    async fn test_worker_processes_task_metadata() {
        let (broker, worker) = make_broker_and_worker();

        let task = AgentTask::new("with meta").with_metadata("priority", serde_json::json!(5));
        let id = broker.submit(task).await.unwrap();

        let result = worker.run_once().await.unwrap().unwrap();
        assert_eq!(result.task_id, id);
        assert!(result.output.contains("with meta"));
    }

    /// Model whose `generate` always errors, driving the agent-error path.
    struct FailingModel;

    impl Model for FailingModel {
        async fn generate(&self, _request: &ChatRequest) -> DResult<ChatResponse> {
            Err(crate::error::DaimonError::Model("boom".into()))
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> DResult<ResponseStream> {
            Err(crate::error::DaimonError::Model("boom".into()))
        }
    }

    fn make_failing_worker() -> (InProcessBroker, TaskWorker) {
        let broker = InProcessBroker::new(16);
        let worker = TaskWorker::new(broker.clone(), || {
            Agent::builder().model(FailingModel).build().unwrap()
        });
        (broker, worker)
    }

    #[tokio::test]
    async fn test_run_exits_on_broker_close() {
        let (broker, worker) = make_broker_and_worker();

        let id = broker.submit(AgentTask::new("last one")).await.unwrap();
        broker.close().await;

        // The worker must drain the queued task and then exit promptly once
        // the closed channel signals end-of-stream.
        tokio::time::timeout(std::time::Duration::from_secs(5), worker.run())
            .await
            .expect("worker.run() must exit after the broker closes")
            .unwrap();

        let status = broker.status(&id).await.unwrap();
        assert!(matches!(status, super::super::TaskStatus::Completed(_)));
    }

    /// Broker that scripts its `receive` responses: `None` entries model idle
    /// poll timeouts (Redis BRPOP / NATS fetch coming up empty), `Some` entries
    /// deliver a task. `none_means_closed` is `false`, matching the polling
    /// network brokers.
    struct IdleThenTaskBroker {
        polls: Mutex<std::collections::VecDeque<Option<AgentTask>>>,
        receive_calls: std::sync::atomic::AtomicUsize,
        statuses: Mutex<std::collections::HashMap<String, super::super::TaskStatus>>,
    }

    use tokio::sync::Mutex;

    impl IdleThenTaskBroker {
        fn new(polls: Vec<Option<AgentTask>>) -> Self {
            Self {
                polls: Mutex::new(polls.into()),
                receive_calls: std::sync::atomic::AtomicUsize::new(0),
                statuses: Mutex::new(std::collections::HashMap::new()),
            }
        }
    }

    impl TaskBroker for IdleThenTaskBroker {
        async fn submit(&self, task: AgentTask) -> DResult<String> {
            let id = task.task_id.clone();
            self.polls.lock().await.push_back(Some(task));
            Ok(id)
        }

        async fn status(&self, task_id: &str) -> DResult<super::super::TaskStatus> {
            Ok(self
                .statuses
                .lock()
                .await
                .get(task_id)
                .cloned()
                .unwrap_or(super::super::TaskStatus::Pending))
        }

        async fn receive(&self) -> DResult<Option<AgentTask>> {
            self.receive_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(self.polls.lock().await.pop_front().flatten())
        }

        async fn complete(&self, task_id: &str, result: TaskResult) -> DResult<()> {
            self.statuses.lock().await.insert(
                task_id.to_string(),
                super::super::TaskStatus::Completed(result),
            );
            Ok(())
        }

        async fn fail(&self, task_id: &str, error: String) -> DResult<()> {
            self.statuses
                .lock()
                .await
                .insert(task_id.to_string(), super::super::TaskStatus::Failed(error));
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_worker_keeps_polling_through_idle_receives() {
        // Two empty polls (idle broker) followed by a real task: the worker
        // must retry past the idle polls instead of treating the first `None`
        // as a shutdown signal and exiting.
        let task = AgentTask::new("after the lull");
        let id = task.task_id.clone();
        let broker = Arc::new(IdleThenTaskBroker::new(vec![None, None, Some(task)]));

        let worker = TaskWorker::from_erased(
            Arc::clone(&broker) as Arc<dyn ErasedTaskBroker>,
            Arc::new(|| Agent::builder().model(EchoModel).build().unwrap()),
        );

        let result = worker.run_once().await.unwrap();
        let result = result.expect("idle polls must not be misread as broker closure");
        assert_eq!(result.task_id, id);

        assert!(
            broker
                .receive_calls
                .load(std::sync::atomic::Ordering::SeqCst)
                >= 3,
            "worker must have polled through both idle receives"
        );
    }

    #[tokio::test]
    async fn test_run_once_agent_error_marks_task_failed() {
        let (broker, worker) = make_failing_worker();

        let id = broker.submit(AgentTask::new("will fail")).await.unwrap();

        let result = worker.run_once().await.unwrap().unwrap();
        assert!(result.error.is_some());

        // The broker status must be Failed — not Completed with an embedded
        // error — matching what run_parallel records for the same failure.
        let status = broker.status(&id).await.unwrap();
        assert!(
            matches!(status, super::super::TaskStatus::Failed(ref msg) if msg.contains("boom")),
            "agent error must mark the task Failed, got {status:?}"
        );
    }

    #[tokio::test]
    async fn test_run_parallel_agent_error_marks_task_failed() {
        let (broker, worker) = make_failing_worker();

        let id = broker
            .submit(AgentTask::new("parallel fail"))
            .await
            .unwrap();
        broker.close().await;

        tokio::time::timeout(std::time::Duration::from_secs(5), worker.run_parallel(2))
            .await
            .expect("run_parallel must exit after the broker closes")
            .unwrap();

        let status = broker.status(&id).await.unwrap();
        assert!(
            matches!(status, super::super::TaskStatus::Failed(ref msg) if msg.contains("boom")),
            "agent error must mark the task Failed, got {status:?}"
        );
    }
}
