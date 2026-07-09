//! Task broker trait and in-process implementation.
//!
//! The [`TaskBroker`] and [`ErasedTaskBroker`] traits are defined in
//! [`daimon_core`] and re-exported here.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{Mutex, mpsc};

use crate::error::Result;

pub use daimon_core::distributed::{ErasedTaskBroker, TaskBroker};

use super::types::{AgentTask, TaskResult, TaskStatus};

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

    /// Closes the broker: no further submissions are accepted, but already
    /// queued tasks can still be received and drained.
    ///
    /// Once the queue is drained, [`receive`](TaskBroker::receive) returns
    /// `Ok(None)`, which for this broker means "closed forever" (see
    /// [`TaskBroker::none_means_closed`]) — workers running against it exit
    /// their loops. Every clone shares the same channel, so closing any clone
    /// closes them all.
    pub async fn close(&self) {
        self.rx.lock().await.close();
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

    /// An in-process channel has a real end-of-stream signal: `recv()` only
    /// yields `None` once the channel is closed and drained, never on an idle
    /// poll. `None` therefore means "closed forever" here.
    fn none_means_closed(&self) -> bool {
        true
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

    #[tokio::test]
    async fn test_close_drains_then_returns_none() {
        let broker = InProcessBroker::new(16);

        let id = broker.submit(AgentTask::new("queued")).await.unwrap();
        broker.close().await;

        // Already-queued tasks are still delivered after close…
        let received = broker.receive().await.unwrap().unwrap();
        assert_eq!(received.task_id, id);

        // …then the drained, closed channel signals end-of-stream.
        assert!(broker.receive().await.unwrap().is_none());

        // And new submissions are rejected.
        assert!(broker.submit(AgentTask::new("late")).await.is_err());
    }

    #[test]
    fn test_none_means_closed_for_in_process() {
        let broker = InProcessBroker::new(1);
        assert!(broker.none_means_closed());
    }
}
