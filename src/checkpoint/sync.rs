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
//! use daimon::checkpoint::{CheckpointSync, ErasedCheckpoint, FileCheckpoint, InMemoryCheckpoint};
//! use std::sync::Arc;
//!
//! let local = InMemoryCheckpoint::new();
//! let remote = FileCheckpoint::new("/shared/nfs/checkpoints");
//! let synced = CheckpointSync::new(local, remote);
//!
//! // Use `synced` as the checkpoint backend — writes go to both stores.
//! let agent = Agent::builder().model(model).build()?;
//! let checkpoint: Arc<dyn ErasedCheckpoint> = Arc::new(synced);
//! let response = agent.prompt_resumable("...", "run-1", &checkpoint).await?;
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
    /// Returns the number of runs that were copied — those missing locally
    /// plus those whose remote checkpoint has advanced past the local copy.
    pub async fn pull_all(&self) -> Result<usize> {
        replicate(&*self.remote, &*self.local).await
    }

    /// Pushes all checkpoints from the local store to the remote store.
    ///
    /// Returns the number of runs that were copied — those missing remotely
    /// plus those whose local checkpoint has advanced past the remote copy.
    pub async fn push_all(&self) -> Result<usize> {
        replicate(&*self.local, &*self.remote).await
    }
}

/// Returns whether `source` represents more recent progress than `target`
/// for the same run.
///
/// A higher iteration always wins. At equal iterations a completed
/// checkpoint supersedes an in-progress one, and a later `created_at`
/// breaks the remaining ties. Replication previously only copied run IDs
/// missing on the target, so a run that advanced on the source was never
/// refreshed once the target held any (however stale) copy of it.
fn source_is_newer(source: &CheckpointState, target: &CheckpointState) -> bool {
    if source.iteration != target.iteration {
        return source.iteration > target.iteration;
    }
    if source.completed != target.completed {
        return source.completed;
    }
    source.created_at > target.created_at
}

/// Copies every run from `source` to `target` that is missing on the target
/// or newer on the source (see [`source_is_newer`]). Returns the number of
/// runs copied.
async fn replicate(source: &dyn ErasedCheckpoint, target: &dyn ErasedCheckpoint) -> Result<usize> {
    let source_runs = source.list_runs_erased().await?;

    let mut copied = 0;
    for run_id in &source_runs {
        let Some(source_state) = source.load_erased(run_id).await? else {
            continue;
        };

        let should_copy = match target.load_erased(run_id).await? {
            None => true,
            Some(target_state) => source_is_newer(&source_state, &target_state),
        };

        if should_copy {
            target.save_erased(&source_state).await?;
            copied += 1;
        }
    }

    Ok(copied)
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
        // Delete from the remote first. If the local copy were removed first
        // and the remote delete then failed, the run would survive on the
        // remote and a replicator (or a `load` fallback) would resurrect it
        // locally — the delete would silently un-happen. Failing after the
        // remote delete instead leaves only a stale local copy, which a
        // retried delete removes for good.
        self.remote.delete_erased(run_id).await?;
        self.local.delete_erased(run_id).await?;
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

    /// Runs the replicator loop indefinitely, pulling new and advanced
    /// checkpoints from the remote store at the configured interval.
    pub async fn run(&self) -> Result<()> {
        loop {
            match self.pull_once().await {
                Ok(count) => {
                    if count > 0 {
                        tracing::info!(synced = count, "checkpoint replicator: pulled runs");
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "checkpoint replicator: pull failed");
                }
            }

            tokio::time::sleep(self.interval).await;
        }
    }

    /// Performs a single pull, returning the number of runs synced — those
    /// missing locally plus those whose remote checkpoint has advanced past
    /// the local copy (see [`CheckpointSync::pull_all`], which shares the
    /// same replication logic).
    pub async fn pull_once(&self) -> Result<usize> {
        replicate(&*self.remote, &*self.local).await
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

    #[test]
    fn test_source_is_newer_ordering() {
        let base = CheckpointState::new("r", vec![], 2);

        // Higher iteration wins in either direction.
        let advanced = CheckpointState::new("r", vec![], 5);
        assert!(source_is_newer(&advanced, &base));
        assert!(!source_is_newer(&base, &advanced));

        // At equal iterations, completion supersedes in-progress.
        let done = CheckpointState::new("r", vec![], 2).mark_completed();
        assert!(source_is_newer(&done, &base));
        assert!(!source_is_newer(&base, &done));

        // Identical progress is not "newer" — no pointless copies.
        assert!(!source_is_newer(&base, &base.clone()));
    }

    #[tokio::test]
    async fn test_pull_all_refreshes_stale_local_run() {
        let local = Arc::new(InMemoryCheckpoint::new());
        let remote = Arc::new(InMemoryCheckpoint::new());

        // The run exists on both sides, but the remote has advanced. A
        // missing-only sync would skip it and the local copy would stay
        // stale forever.
        local
            .save(&CheckpointState::new("run-1", vec![], 2))
            .await
            .unwrap();
        remote
            .save(&CheckpointState::new(
                "run-1",
                vec![Message::user("progress")],
                7,
            ))
            .await
            .unwrap();

        let sync = CheckpointSync::from_erased(local.clone(), remote);
        let synced = sync.pull_all().await.unwrap();
        assert_eq!(synced, 1);

        let refreshed = local.load("run-1").await.unwrap().unwrap();
        assert_eq!(refreshed.iteration, 7);
        assert_eq!(refreshed.messages.len(), 1);
    }

    #[tokio::test]
    async fn test_pull_all_does_not_clobber_newer_local_run() {
        let local = Arc::new(InMemoryCheckpoint::new());
        let remote = Arc::new(InMemoryCheckpoint::new());

        local
            .save(&CheckpointState::new("run-1", vec![], 9))
            .await
            .unwrap();
        remote
            .save(&CheckpointState::new("run-1", vec![], 3))
            .await
            .unwrap();

        let sync = CheckpointSync::from_erased(local.clone(), remote);
        let synced = sync.pull_all().await.unwrap();
        assert_eq!(synced, 0, "a stale remote copy must not overwrite local");

        let kept = local.load("run-1").await.unwrap().unwrap();
        assert_eq!(kept.iteration, 9);
    }

    #[tokio::test]
    async fn test_push_all_refreshes_stale_remote_run() {
        let local = Arc::new(InMemoryCheckpoint::new());
        let remote = Arc::new(InMemoryCheckpoint::new());

        local
            .save(&CheckpointState::new("run-1", vec![], 6).mark_completed())
            .await
            .unwrap();
        remote
            .save(&CheckpointState::new("run-1", vec![], 6))
            .await
            .unwrap();

        let sync = CheckpointSync::from_erased(local, remote.clone());
        let pushed = sync.push_all().await.unwrap();
        assert_eq!(
            pushed, 1,
            "a completed local run must refresh the in-progress remote copy"
        );

        let refreshed = remote.load("run-1").await.unwrap().unwrap();
        assert!(refreshed.completed);
    }

    #[tokio::test]
    async fn test_replicator_pull_once_refreshes_stale_run() {
        let local = Arc::new(InMemoryCheckpoint::new());
        let remote = Arc::new(InMemoryCheckpoint::new());

        local
            .save(&CheckpointState::new("rep-1", vec![], 1))
            .await
            .unwrap();
        remote
            .save(&CheckpointState::new("rep-1", vec![], 4))
            .await
            .unwrap();

        let replicator =
            CheckpointReplicator::new(local.clone(), remote, std::time::Duration::from_secs(60));

        let synced = replicator.pull_once().await.unwrap();
        assert_eq!(synced, 1);

        let refreshed = local.load("rep-1").await.unwrap().unwrap();
        assert_eq!(refreshed.iteration, 4);
    }

    /// Checkpoint whose `delete` always fails, delegating everything else to
    /// an in-memory store. Models an unreachable remote during deletion.
    struct FailingDeleteCheckpoint {
        inner: InMemoryCheckpoint,
    }

    impl Checkpoint for FailingDeleteCheckpoint {
        async fn save(&self, state: &CheckpointState) -> Result<()> {
            self.inner.save(state).await
        }

        async fn load(&self, run_id: &str) -> Result<Option<CheckpointState>> {
            self.inner.load(run_id).await
        }

        async fn list_runs(&self) -> Result<Vec<String>> {
            self.inner.list_runs().await
        }

        async fn delete(&self, _run_id: &str) -> Result<()> {
            Err(crate::error::DaimonError::Other(
                "remote delete unavailable".into(),
            ))
        }
    }

    #[tokio::test]
    async fn test_delete_keeps_local_when_remote_delete_fails() {
        let local = Arc::new(InMemoryCheckpoint::new());
        let remote = FailingDeleteCheckpoint {
            inner: InMemoryCheckpoint::new(),
        };

        let state = CheckpointState::new("run-del", vec![], 1);
        local.save(&state).await.unwrap();
        remote.save(&state).await.unwrap();

        let sync = CheckpointSync::from_erased(local.clone(), Arc::new(remote));
        assert!(sync.delete("run-del").await.is_err());

        // The remote delete failed, so the local copy must still exist:
        // deleting local-first would have left the run only on the remote,
        // from where a replicator would resurrect it.
        assert!(
            local.load("run-del").await.unwrap().is_some(),
            "local copy must survive a failed remote delete"
        );
    }
}
