//! Task broker trait and in-process implementation.
//!
//! The [`TaskBroker`] and [`ErasedTaskBroker`] traits are defined in
//! [`daimon_core`] and re-exported here.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use tokio::sync::{Mutex, mpsc};

use crate::error::Result;

pub use daimon_core::distributed::{ErasedTaskBroker, TaskBroker};

use super::types::{AgentTask, TaskResult, TaskStatus};

/// Maximum number of terminal (`Completed`/`Failed`) statuses retained by
/// [`InProcessBroker`]; the oldest are evicted beyond this.
const MAX_TERMINAL_STATUSES: usize = 1024;

/// Status map plus the insertion order of terminal entries, kept under one
/// lock so eviction and lookup stay consistent.
#[derive(Default)]
struct StatusMap {
    statuses: HashMap<String, TaskStatus>,
    terminal_order: VecDeque<String>,
}

impl StatusMap {
    /// Records a terminal status and evicts the oldest terminal entries once
    /// more than [`MAX_TERMINAL_STATUSES`] are retained. Non-terminal
    /// (`Pending`/`Running`) statuses are never evicted.
    fn insert_terminal(&mut self, task_id: String, status: TaskStatus) {
        // Re-completing a task must not enqueue its id twice, or a later
        // eviction of the first occurrence would drop the live entry early.
        let was_terminal = matches!(
            self.statuses.insert(task_id.clone(), status),
            Some(TaskStatus::Completed(_) | TaskStatus::Failed(_))
        );
        if !was_terminal {
            self.terminal_order.push_back(task_id);
        }
        while self.terminal_order.len() > MAX_TERMINAL_STATUSES {
            if let Some(oldest) = self.terminal_order.pop_front() {
                self.statuses.remove(&oldest);
            }
        }
    }
}

/// In-process task broker backed by tokio MPSC channels.
///
/// Suitable for single-process parallelism and testing. Clone-friendly:
/// all clones share the same underlying channels and state.
///
/// Terminal task statuses (which embed the full [`TaskResult`]) are retained
/// for the most recent `MAX_TERMINAL_STATUSES` tasks only — a long-running
/// process no longer accumulates every result forever. Querying an evicted
/// task returns [`TaskStatus::Pending`], the same answer given for unknown
/// task ids.
pub struct InProcessBroker {
    tx: mpsc::Sender<AgentTask>,
    rx: Arc<Mutex<mpsc::Receiver<AgentTask>>>,
    statuses: Arc<Mutex<StatusMap>>,
}

impl InProcessBroker {
    /// Creates a new in-process broker with the given channel capacity.
    pub fn new(capacity: usize) -> Self {
        let (tx, rx) = mpsc::channel(capacity);
        Self {
            tx,
            rx: Arc::new(Mutex::new(rx)),
            statuses: Arc::new(Mutex::new(StatusMap::default())),
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
            statuses.statuses.insert(id.clone(), TaskStatus::Pending);
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
            .statuses
            .get(task_id)
            .cloned()
            .unwrap_or(TaskStatus::Pending))
    }

    async fn receive(&self) -> Result<Option<AgentTask>> {
        let mut rx = self.rx.lock().await;
        match rx.recv().await {
            Some(task) => {
                let mut statuses = self.statuses.lock().await;
                statuses
                    .statuses
                    .insert(task.task_id.clone(), TaskStatus::Running);
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
        statuses.insert_terminal(task_id.to_string(), TaskStatus::Completed(result));
        Ok(())
    }

    async fn fail(&self, task_id: &str, error: String) -> Result<()> {
        let mut statuses = self.statuses.lock().await;
        statuses.insert_terminal(task_id.to_string(), TaskStatus::Failed(error));
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

    #[tokio::test]
    async fn test_terminal_statuses_evicted_beyond_cap() {
        let broker = InProcessBroker::new(1);

        for i in 0..(MAX_TERMINAL_STATUSES + 10) {
            let id = format!("t-{i}");
            broker
                .complete(
                    &id,
                    TaskResult {
                        task_id: id.clone(),
                        output: "done".into(),
                        iterations: 1,
                        cost: 0.0,
                        error: None,
                    },
                )
                .await
                .unwrap();
        }

        // The oldest terminal statuses are gone (evicted ids read as Pending,
        // the same as unknown ids); the newest are retained.
        let oldest = broker.status("t-0").await.unwrap();
        assert!(matches!(oldest, TaskStatus::Pending));
        let newest = broker
            .status(&format!("t-{}", MAX_TERMINAL_STATUSES + 9))
            .await
            .unwrap();
        assert!(matches!(newest, TaskStatus::Completed(_)));
        assert_eq!(
            broker.statuses.lock().await.terminal_order.len(),
            MAX_TERMINAL_STATUSES
        );
    }

    #[tokio::test]
    async fn test_recompleting_task_does_not_double_track() {
        let broker = InProcessBroker::new(1);
        let result = TaskResult {
            task_id: "t-1".into(),
            output: "done".into(),
            iterations: 1,
            cost: 0.0,
            error: None,
        };
        broker.complete("t-1", result.clone()).await.unwrap();
        broker.complete("t-1", result).await.unwrap();
        assert_eq!(broker.statuses.lock().await.terminal_order.len(), 1);
    }
}
