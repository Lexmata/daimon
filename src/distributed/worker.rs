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
    /// to the broker. Returns `Ok(None)` if the broker is closed.
    pub async fn run_once(&self) -> Result<Option<TaskResult>> {
        let task = match self.broker.receive_erased().await? {
            Some(t) => t,
            None => return Ok(None),
        };

        let result = self.execute_task(&task).await;

        match &result {
            Ok(tr) => {
                self.broker
                    .complete_erased(&task.task_id, tr.clone())
                    .await?;
            }
            Err(e) => {
                self.broker
                    .fail_erased(&task.task_id, e.to_string())
                    .await?;
            }
        }

        result.map(Some)
    }

    /// Runs the worker loop indefinitely, processing tasks until the broker
    /// channel closes.
    pub async fn run(&self) -> Result<()> {
        loop {
            match self.run_once().await? {
                Some(_) => continue,
                None => {
                    tracing::info!("broker channel closed, worker exiting");
                    return Ok(());
                }
            }
        }
    }

    /// Runs up to `n` tasks concurrently using `tokio::JoinSet`.
    pub async fn run_parallel(&self, concurrency: usize) -> Result<()> {
        use tokio::task::JoinSet;

        let mut join_set = JoinSet::new();

        loop {
            while join_set.len() < concurrency {
                let task = match self.broker.receive_erased().await? {
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

            if let Some(result) = join_set.join_next().await
                && let Err(e) = result
            {
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
}
