//! Task broker trait and in-process implementation.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use tokio::sync::{Mutex, mpsc};

use crate::error::Result;

use super::types::{AgentTask, TaskResult, TaskStatus};

/// Trait for distributing agent tasks across workers.
///
/// Implement this for your message broker (Redis Streams, RabbitMQ,
/// NATS JetStream, etc.) to enable multi-process agent execution.
pub trait TaskBroker: Send + Sync {
    /// Submits a task for execution. Returns the task ID.
    fn submit(&self, task: AgentTask) -> impl Future<Output = Result<String>> + Send;

    /// Queries the current status of a task.
    fn status(&self, task_id: &str) -> impl Future<Output = Result<TaskStatus>> + Send;

    /// Blocks until a task is available and returns it.
    /// Returns `None` if the broker is closed.
    fn receive(&self) -> impl Future<Output = Result<Option<AgentTask>>> + Send;

    /// Marks a task as completed with the given result.
    fn complete(&self, task_id: &str, result: TaskResult) -> impl Future<Output = Result<()>> + Send;

    /// Marks a task as failed with an error message.
    fn fail(&self, task_id: &str, error: String) -> impl Future<Output = Result<()>> + Send;
}

/// Object-safe wrapper for [`TaskBroker`].
pub trait ErasedTaskBroker: Send + Sync {
    fn submit_erased<'a>(
        &'a self,
        task: AgentTask,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>>;

    fn status_erased<'a>(
        &'a self,
        task_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<TaskStatus>> + Send + 'a>>;

    fn receive_erased(&self) -> Pin<Box<dyn Future<Output = Result<Option<AgentTask>>> + Send + '_>>;

    fn complete_erased<'a>(
        &'a self,
        task_id: &'a str,
        result: TaskResult,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    fn fail_erased<'a>(
        &'a self,
        task_id: &'a str,
        error: String,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
}

impl<T: TaskBroker> ErasedTaskBroker for T {
    fn submit_erased<'a>(
        &'a self,
        task: AgentTask,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        Box::pin(self.submit(task))
    }

    fn status_erased<'a>(
        &'a self,
        task_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<TaskStatus>> + Send + 'a>> {
        Box::pin(self.status(task_id))
    }

    fn receive_erased(&self) -> Pin<Box<dyn Future<Output = Result<Option<AgentTask>>> + Send + '_>> {
        Box::pin(self.receive())
    }

    fn complete_erased<'a>(
        &'a self,
        task_id: &'a str,
        result: TaskResult,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(self.complete(task_id, result))
    }

    fn fail_erased<'a>(
        &'a self,
        task_id: &'a str,
        error: String,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(self.fail(task_id, error))
    }
}

/// In-process task broker backed by tokio MPSC channels.
///
/// Suitable for single-process parallelism and testing. Clone-friendly:
/// all clones share the same underlying channels and state.
pub struct InProcessBroker {
    tx: mpsc::Sender<AgentTask>,
    rx: Arc<Mutex<mpsc::Receiver<AgentTask>>>,
    statuses: Arc<Mutex<HashMap<String, TaskStatus>>>,
}

impl InProcessBroker {
    /// Creates a new in-process broker with the given channel capacity.
    pub fn new(capacity: usize) -> Self {
        let (tx, rx) = mpsc::channel(capacity);
        Self {
            tx,
            rx: Arc::new(Mutex::new(rx)),
            statuses: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl Clone for InProcessBroker {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
            rx: Arc::clone(&self.rx),
            statuses: Arc::clone(&self.statuses),
        }
    }
}

impl TaskBroker for InProcessBroker {
    async fn submit(&self, task: AgentTask) -> Result<String> {
        let id = task.task_id.clone();
        {
            let mut statuses = self.statuses.lock().await;
            statuses.insert(id.clone(), TaskStatus::Pending);
        }
        self.tx
            .send(task)
            .await
            .map_err(|e| crate::error::DaimonError::Other(format!("broker send failed: {e}")))?;
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
        let mut rx = self.rx.lock().await;
        match rx.recv().await {
            Some(task) => {
                let mut statuses = self.statuses.lock().await;
                statuses.insert(task.task_id.clone(), TaskStatus::Running);
                Ok(Some(task))
            }
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

    #[tokio::test]
    async fn test_submit_and_receive() {
        let broker = InProcessBroker::new(16);

        let task = AgentTask::new("hello");
        let id = broker.submit(task).await.unwrap();

        let status = broker.status(&id).await.unwrap();
        assert!(matches!(status, TaskStatus::Pending));

        let received = broker.receive().await.unwrap().unwrap();
        assert_eq!(received.task_id, id);

        let status = broker.status(&id).await.unwrap();
        assert!(matches!(status, TaskStatus::Running));
    }

    #[tokio::test]
    async fn test_complete() {
        let broker = InProcessBroker::new(16);

        let task = AgentTask::new("work");
        let id = broker.submit(task).await.unwrap();
        let _ = broker.receive().await.unwrap();

        let result = TaskResult {
            task_id: id.clone(),
            output: "done".into(),
            iterations: 1,
            cost: 0.0,
            error: None,
        };
        broker.complete(&id, result).await.unwrap();

        let status = broker.status(&id).await.unwrap();
        assert!(matches!(status, TaskStatus::Completed(_)));
    }

    #[tokio::test]
    async fn test_fail() {
        let broker = InProcessBroker::new(16);

        let task = AgentTask::new("boom");
        let id = broker.submit(task).await.unwrap();
        let _ = broker.receive().await.unwrap();

        broker.fail(&id, "something broke".into()).await.unwrap();

        let status = broker.status(&id).await.unwrap();
        assert!(matches!(status, TaskStatus::Failed(ref msg) if msg == "something broke"));
    }

    #[tokio::test]
    async fn test_clone_shares_state() {
        let broker = InProcessBroker::new(16);
        let clone = broker.clone();

        let task = AgentTask::new("shared");
        let id = broker.submit(task).await.unwrap();

        let received = clone.receive().await.unwrap().unwrap();
        assert_eq!(received.task_id, id);
    }

    #[tokio::test]
    async fn test_multiple_tasks() {
        let broker = InProcessBroker::new(16);

        let id1 = broker.submit(AgentTask::new("a")).await.unwrap();
        let id2 = broker.submit(AgentTask::new("b")).await.unwrap();
        let id3 = broker.submit(AgentTask::new("c")).await.unwrap();

        let t1 = broker.receive().await.unwrap().unwrap();
        let t2 = broker.receive().await.unwrap().unwrap();
        let t3 = broker.receive().await.unwrap().unwrap();

        assert_eq!(t1.task_id, id1);
        assert_eq!(t2.task_id, id2);
        assert_eq!(t3.task_id, id3);
    }
}
