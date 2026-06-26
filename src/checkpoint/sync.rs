//! Distributed checkpoint synchronization across processes.
//!
//! [`CheckpointSync`] wraps two [`Checkpoint`] backends — a fast local
//! store and a shared remote store — and keeps them in sync. Saves
//! write-through to both; loads prefer local with remote fallback.
//!
//! [`CheckpointReplicator`] runs as a background task, periodically
//! pulling new checkpoints from the remote into the local store.
//!
//! ```ignore
//! use daimon::checkpoint::{InMemoryCheckpoint, FileCheckpoint, CheckpointSync};
//!
//! let local = InMemoryCheckpoint::new();
//! let remote = FileCheckpoint::new("/shared/nfs/checkpoints")?;
//! let synced = CheckpointSync::new(local, remote);
//!
//! // Use `synced` as the checkpoint backend — writes go to both stores.
//! let agent = Agent::builder()
//!     .model(model)
//!     .checkpoint(synced)
//!     .build()?;
//! ```

use std::sync::Arc;

use crate::error::Result;

use super::traits::{Checkpoint, ErasedCheckpoint};
use super::types::CheckpointState;

/// Write-through checkpoint that synchronizes a local and remote store.
///
/// - `save` writes to both local and remote (write-through).
/// - `load` checks local first, falls back to remote, and backfills local on miss.
/// - `list_runs` returns the union of both stores' run IDs.
/// - `delete` removes from both stores.
pub struct CheckpointSync {
    local: Arc<dyn ErasedCheckpoint>,
    remote: Arc<dyn ErasedCheckpoint>,
}

impl CheckpointSync {
    /// Creates a new synced checkpoint from a local and remote backend.
    pub fn new<L: Checkpoint + 'static, R: Checkpoint + 'static>(local: L, remote: R) -> Self {
        Self {
            local: Arc::new(local),
            remote: Arc::new(remote),
        }
    }

    /// Creates from pre-erased checkpoint backends.
    pub fn from_erased(
        local: Arc<dyn ErasedCheckpoint>,
        remote: Arc<dyn ErasedCheckpoint>,
    ) -> Self {
        Self { local, remote }
    }

    /// Pulls all checkpoints from the remote store into the local store.
    ///
    /// Returns the number of run IDs that were synced (i.e. missing locally).
    pub async fn pull_all(&self) -> Result<usize> {
        let remote_runs = self.remote.list_runs_erased().await?;
        let local_runs = self.local.list_runs_erased().await?;

        let local_set: std::collections::HashSet<&str> =
            local_runs.iter().map(|s| s.as_str()).collect();

        let mut synced = 0;
        for run_id in &remote_runs {
            if !local_set.contains(run_id.as_str())
                && let Some(state) = self.remote.load_erased(run_id).await?
            {
                self.local.save_erased(&state).await?;
                synced += 1;
            }
        }

        Ok(synced)
    }

    /// Pushes all checkpoints from the local store to the remote store.
    ///
    /// Returns the number of run IDs that were pushed (i.e. missing remotely).
    pub async fn push_all(&self) -> Result<usize> {
        let local_runs = self.local.list_runs_erased().await?;
        let remote_runs = self.remote.list_runs_erased().await?;

        let remote_set: std::collections::HashSet<&str> =
            remote_runs.iter().map(|s| s.as_str()).collect();

        let mut pushed = 0;
        for run_id in &local_runs {
            if !remote_set.contains(run_id.as_str())
                && let Some(state) = self.local.load_erased(run_id).await?
            {
                self.remote.save_erased(&state).await?;
                pushed += 1;
            }
        }

        Ok(pushed)
    }
}

impl Checkpoint for CheckpointSync {
    async fn save(&self, state: &CheckpointState) -> Result<()> {
        self.local.save_erased(state).await?;
        self.remote.save_erased(state).await?;
        Ok(())
    }

    async fn load(&self, run_id: &str) -> Result<Option<CheckpointState>> {
        if let Some(state) = self.local.load_erased(run_id).await? {
            return Ok(Some(state));
        }

        if let Some(state) = self.remote.load_erased(run_id).await? {
            self.local.save_erased(&state).await?;
            return Ok(Some(state));
        }

        Ok(None)
    }

    async fn list_runs(&self) -> Result<Vec<String>> {
        let mut local_runs = self.local.list_runs_erased().await?;
        let remote_runs = self.remote.list_runs_erased().await?;

        let local_set: std::collections::HashSet<String> = local_runs.iter().cloned().collect();

        for run_id in remote_runs {
            if !local_set.contains(&run_id) {
                local_runs.push(run_id);
            }
        }

        Ok(local_runs)
    }

    async fn delete(&self, run_id: &str) -> Result<()> {
        self.local.delete_erased(run_id).await?;
        self.remote.delete_erased(run_id).await?;
        Ok(())
    }
}

/// Periodically replicates checkpoints from a remote store to a local store.
///
/// Runs as a background task. Useful for keeping a local in-memory cache
/// warm with checkpoints from a shared persistent store.
///
/// ```ignore
/// use daimon::checkpoint::{InMemoryCheckpoint, FileCheckpoint, CheckpointReplicator};
///
/// let local = Arc::new(InMemoryCheckpoint::new());
/// let remote = Arc::new(FileCheckpoint::new("/shared/checkpoints")?);
///
/// let replicator = CheckpointReplicator::new(
///     local.clone() as Arc<dyn ErasedCheckpoint>,
///     remote.clone() as Arc<dyn ErasedCheckpoint>,
///     std::time::Duration::from_secs(30),
/// );
///
/// // Run in background
/// tokio::spawn(replicator.run());
/// ```
pub struct CheckpointReplicator {
    local: Arc<dyn ErasedCheckpoint>,
    remote: Arc<dyn ErasedCheckpoint>,
    interval: std::time::Duration,
}

impl CheckpointReplicator {
    /// Creates a new replicator.
    ///
    /// * `local` — the local (fast) checkpoint store to populate
    /// * `remote` — the remote (shared) checkpoint store to pull from
    /// * `interval` — how often to poll for new remote checkpoints
    pub fn new(
        local: Arc<dyn ErasedCheckpoint>,
        remote: Arc<dyn ErasedCheckpoint>,
        interval: std::time::Duration,
    ) -> Self {
        Self {
            local,
            remote,
            interval,
        }
    }

    /// Runs the replicator loop indefinitely, pulling new checkpoints
    /// from the remote store at the configured interval.
    pub async fn run(&self) -> Result<()> {
        loop {
            match self.pull_once().await {
                Ok(count) => {
                    if count > 0 {
                        tracing::info!(synced = count, "checkpoint replicator: pulled new runs");
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "checkpoint replicator: pull failed");
                }
            }

            tokio::time::sleep(self.interval).await;
        }
    }

    /// Performs a single pull, returning the number of new runs synced.
    pub async fn pull_once(&self) -> Result<usize> {
        let remote_runs = self.remote.list_runs_erased().await?;
        let local_runs = self.local.list_runs_erased().await?;

        let local_set: std::collections::HashSet<String> = local_runs.into_iter().collect();

        let mut synced = 0;
        for run_id in &remote_runs {
            if !local_set.contains(run_id)
                && let Some(state) = self.remote.load_erased(run_id).await?
            {
                self.local.save_erased(&state).await?;
                synced += 1;
            }
        }

        Ok(synced)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::InMemoryCheckpoint;
    use crate::model::types::Message;

    #[tokio::test]
    async fn test_write_through() {
        let local = InMemoryCheckpoint::new();
        let remote = InMemoryCheckpoint::new();
        let sync = CheckpointSync::new(local, remote);

        let state = CheckpointState::new("run-1", vec![Message::user("hello")], 1);
        sync.save(&state).await.unwrap();

        let loaded = sync.load("run-1").await.unwrap().unwrap();
        assert_eq!(loaded.run_id, "run-1");
        assert_eq!(loaded.messages.len(), 1);
    }

    #[tokio::test]
    async fn test_load_local_first() {
        let local = Arc::new(InMemoryCheckpoint::new());
        let remote = Arc::new(InMemoryCheckpoint::new());

        let local_state = CheckpointState::new("run-1", vec![Message::user("local")], 2);
        local.save(&local_state).await.unwrap();

        let remote_state = CheckpointState::new("run-1", vec![Message::user("remote")], 1);
        remote.save(&remote_state).await.unwrap();

        let sync = CheckpointSync::from_erased(local, remote);
        let loaded = sync.load("run-1").await.unwrap().unwrap();
        assert_eq!(loaded.iteration, 2);
    }

    #[tokio::test]
    async fn test_load_falls_back_to_remote() {
        let local = Arc::new(InMemoryCheckpoint::new());
        let remote = Arc::new(InMemoryCheckpoint::new());

        let state = CheckpointState::new("run-remote", vec![Message::user("hi")], 1);
        remote.save(&state).await.unwrap();

        let sync = CheckpointSync::from_erased(local.clone(), remote);
        let loaded = sync.load("run-remote").await.unwrap().unwrap();
        assert_eq!(loaded.run_id, "run-remote");

        let backfilled = local.load("run-remote").await.unwrap();
        assert!(backfilled.is_some());
    }

    #[tokio::test]
    async fn test_load_returns_none_when_missing() {
        let sync = CheckpointSync::new(InMemoryCheckpoint::new(), InMemoryCheckpoint::new());
        assert!(sync.load("nonexistent").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_list_runs_union() {
        let local = Arc::new(InMemoryCheckpoint::new());
        let remote = Arc::new(InMemoryCheckpoint::new());

        local
            .save(&CheckpointState::new("local-only", vec![], 1))
            .await
            .unwrap();
        remote
            .save(&CheckpointState::new("remote-only", vec![], 1))
            .await
            .unwrap();
        local
            .save(&CheckpointState::new("both", vec![], 1))
            .await
            .unwrap();
        remote
            .save(&CheckpointState::new("both", vec![], 1))
            .await
            .unwrap();

        let sync = CheckpointSync::from_erased(local, remote);
        let mut runs = sync.list_runs().await.unwrap();
        runs.sort();
        assert_eq!(runs, vec!["both", "local-only", "remote-only"]);
    }

    #[tokio::test]
    async fn test_delete_from_both() {
        let sync = CheckpointSync::new(InMemoryCheckpoint::new(), InMemoryCheckpoint::new());

        let state = CheckpointState::new("run-del", vec![], 1);
        sync.save(&state).await.unwrap();
        assert!(sync.load("run-del").await.unwrap().is_some());

        sync.delete("run-del").await.unwrap();
        assert!(sync.load("run-del").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_pull_all() {
        let local = Arc::new(InMemoryCheckpoint::new());
        let remote = Arc::new(InMemoryCheckpoint::new());

        remote
            .save(&CheckpointState::new("r1", vec![], 1))
            .await
            .unwrap();
        remote
            .save(&CheckpointState::new("r2", vec![], 1))
            .await
            .unwrap();
        local
            .save(&CheckpointState::new("r1", vec![], 1))
            .await
            .unwrap();

        let sync = CheckpointSync::from_erased(local.clone(), remote);
        let synced = sync.pull_all().await.unwrap();
        assert_eq!(synced, 1);

        assert!(local.load("r2").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn test_push_all() {
        let local = Arc::new(InMemoryCheckpoint::new());
        let remote = Arc::new(InMemoryCheckpoint::new());

        local
            .save(&CheckpointState::new("l1", vec![], 1))
            .await
            .unwrap();
        local
            .save(&CheckpointState::new("l2", vec![], 1))
            .await
            .unwrap();
        remote
            .save(&CheckpointState::new("l1", vec![], 1))
            .await
            .unwrap();

        let sync = CheckpointSync::from_erased(local, remote.clone());
        let pushed = sync.push_all().await.unwrap();
        assert_eq!(pushed, 1);

        assert!(remote.load("l2").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn test_replicator_pull_once() {
        let local = Arc::new(InMemoryCheckpoint::new());
        let remote = Arc::new(InMemoryCheckpoint::new());

        remote
            .save(&CheckpointState::new("rep-1", vec![], 1))
            .await
            .unwrap();

        let replicator =
            CheckpointReplicator::new(local.clone(), remote, std::time::Duration::from_secs(60));

        let synced = replicator.pull_once().await.unwrap();
        assert_eq!(synced, 1);

        assert!(local.load("rep-1").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn test_replicator_pull_once_no_new() {
        let local = Arc::new(InMemoryCheckpoint::new());
        let remote = Arc::new(InMemoryCheckpoint::new());

        let state = CheckpointState::new("shared", vec![], 1);
        local.save(&state).await.unwrap();
        remote.save(&state).await.unwrap();

        let replicator =
            CheckpointReplicator::new(local, remote, std::time::Duration::from_secs(60));

        let synced = replicator.pull_once().await.unwrap();
        assert_eq!(synced, 0);
    }
}
