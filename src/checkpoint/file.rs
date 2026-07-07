//! File-based checkpoint backend that persists to JSON files.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{DaimonError, Result};

/// Monotonic counter used to give each in-flight temp checkpoint file a
/// unique name so concurrent `save` calls to the same `run_id` never collide.
static COUNTER: AtomicU64 = AtomicU64::new(0);

use super::traits::Checkpoint;
use super::types::CheckpointState;

/// Persists checkpoints as JSON files in a directory.
///
/// Each run gets a file named `<run_id>.json`. The directory is created
/// automatically if it doesn't exist.
pub struct FileCheckpoint {
    dir: PathBuf,
}

impl FileCheckpoint {
    /// Creates a new file-based checkpoint backend.
    ///
    /// `dir` is the directory where checkpoint files will be stored.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// Validates that `run_id` is a safe filename component before it is
    /// joined into a filesystem path.
    ///
    /// Without this check a caller-supplied `run_id` like `../../etc/passwd`
    /// or an absolute path such as `/etc/passwd` would escape the checkpoint
    /// directory when passed to `PathBuf::join`, letting an attacker read or
    /// clobber arbitrary files. We restrict ids to a conservative allowlist of
    /// `[A-Za-z0-9._-]` and explicitly reject the empty string and any `..`
    /// component.
    fn validate_run_id(run_id: &str) -> Result<()> {
        if run_id.is_empty() {
            return Err(DaimonError::Other(
                "invalid run_id: must not be empty".to_string(),
            ));
        }
        if run_id == ".." || run_id == "." {
            return Err(DaimonError::Other(format!(
                "invalid run_id '{run_id}': reserved path component"
            )));
        }
        if !run_id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
        {
            return Err(DaimonError::Other(format!(
                "invalid run_id '{run_id}': only [A-Za-z0-9._-] characters are allowed"
            )));
        }
        Ok(())
    }

    fn run_path(&self, run_id: &str) -> Result<PathBuf> {
        Self::validate_run_id(run_id)?;
        Ok(self.dir.join(format!("{run_id}.json")))
    }
}

impl Checkpoint for FileCheckpoint {
    async fn save(&self, state: &CheckpointState) -> Result<()> {
        let dir = self.dir.clone();
        let path = self.run_path(&state.run_id)?;
        let json = serde_json::to_string_pretty(state)?;

        tokio::task::spawn_blocking(move || {
            std::fs::create_dir_all(&dir)
                .map_err(|e| DaimonError::Other(format!("failed to create checkpoint dir: {e}")))?;

            // Write to a temp file in the *same* directory, then atomically
            // rename it into place. A direct `fs::write` can be interrupted by
            // a crash mid-write, leaving a truncated/corrupt JSON file that
            // `load` would then fail to parse forever. `rename` is atomic on a
            // single filesystem, so a reader always sees either the old file or
            // the fully-written new one.
            let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
            let tmp_path = path.with_extension(format!("json.tmp.{}.{unique}", std::process::id()));

            // Write and fsync the temp file before renaming so the bytes are
            // durable on disk prior to the rename becoming visible. Any failure
            // in create/write/flush/sync leaves an orphaned temp file behind, so
            // the whole sequence is wrapped and the temp file is removed on any
            // error before propagating (previously only a failed `rename` was
            // cleaned up, leaking a temp file on every write-side failure).
            let write_result = (|| {
                use std::io::Write;
                let mut f = std::fs::File::create(&tmp_path).map_err(|e| {
                    DaimonError::Other(format!("failed to create temp checkpoint: {e}"))
                })?;
                f.write_all(json.as_bytes())
                    .map_err(|e| DaimonError::Other(format!("failed to write checkpoint: {e}")))?;
                f.flush()
                    .map_err(|e| DaimonError::Other(format!("failed to flush checkpoint: {e}")))?;
                // A sync_all failure means the bytes may not be durable on disk.
                // Surface it (rather than silently discarding the error) so an
                // fsync problem is at least observable in logs instead of a lost
                // write masquerading as a committed checkpoint.
                if let Err(e) = f.sync_all() {
                    tracing::warn!(
                        error = %e,
                        path = %tmp_path.display(),
                        "failed to fsync checkpoint temp file"
                    );
                }
                Ok::<_, DaimonError>(())
            })();

            if let Err(e) = write_result {
                let _ = std::fs::remove_file(&tmp_path);
                return Err(e);
            }

            if let Err(e) = std::fs::rename(&tmp_path, &path) {
                let _ = std::fs::remove_file(&tmp_path);
                return Err(DaimonError::Other(format!(
                    "failed to commit checkpoint: {e}"
                )));
            }
            Ok::<_, DaimonError>(())
        })
        .await
        .map_err(|e| DaimonError::Other(format!("checkpoint spawn error: {e}")))?
    }

    async fn load(&self, run_id: &str) -> Result<Option<CheckpointState>> {
        let path = self.run_path(run_id)?;

        tokio::task::spawn_blocking(move || {
            if !path.exists() {
                return Ok(None);
            }
            let json = std::fs::read_to_string(&path)
                .map_err(|e| DaimonError::Other(format!("failed to read checkpoint: {e}")))?;
            let state: CheckpointState = serde_json::from_str(&json)?;
            Ok(Some(state))
        })
        .await
        .map_err(|e| DaimonError::Other(format!("checkpoint spawn error: {e}")))?
    }

    async fn list_runs(&self) -> Result<Vec<String>> {
        let dir = self.dir.clone();

        tokio::task::spawn_blocking(move || {
            if !dir.exists() {
                return Ok(Vec::new());
            }
            let mut runs = Vec::new();
            let entries = std::fs::read_dir(&dir)
                .map_err(|e| DaimonError::Other(format!("failed to read checkpoint dir: {e}")))?;
            for entry in entries {
                let entry = entry
                    .map_err(|e| DaimonError::Other(format!("failed to read dir entry: {e}")))?;
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "json")
                    && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                {
                    runs.push(stem.to_string());
                }
            }
            Ok(runs)
        })
        .await
        .map_err(|e| DaimonError::Other(format!("checkpoint spawn error: {e}")))?
    }

    async fn delete(&self, run_id: &str) -> Result<()> {
        let path = self.run_path(run_id)?;

        tokio::task::spawn_blocking(move || {
            if path.exists() {
                std::fs::remove_file(&path)
                    .map_err(|e| DaimonError::Other(format!("failed to delete checkpoint: {e}")))?;
            }
            Ok::<_, DaimonError>(())
        })
        .await
        .map_err(|e| DaimonError::Other(format!("checkpoint spawn error: {e}")))?
    }
}

impl std::fmt::Debug for FileCheckpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileCheckpoint")
            .field("dir", &self.dir)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::types::Message;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir() -> PathBuf {
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("daimon_cp_test_{}_{id}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[tokio::test]
    async fn test_file_save_load() {
        let dir = temp_dir();
        let cp = FileCheckpoint::new(&dir);
        let state = CheckpointState::new("run-1", vec![Message::user("hello")], 1);

        cp.save(&state).await.unwrap();

        assert!(dir.join("run-1.json").exists());

        let loaded = cp.load("run-1").await.unwrap().unwrap();
        assert_eq!(loaded.run_id, "run-1");
        assert_eq!(loaded.iteration, 1);
        assert_eq!(loaded.messages.len(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_file_load_nonexistent() {
        let dir = temp_dir();
        let cp = FileCheckpoint::new(&dir);
        assert!(cp.load("nope").await.unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_file_list_runs() {
        let dir = temp_dir();
        let cp = FileCheckpoint::new(&dir);

        cp.save(&CheckpointState::new("alpha", vec![], 0))
            .await
            .unwrap();
        cp.save(&CheckpointState::new("beta", vec![], 0))
            .await
            .unwrap();

        let mut runs = cp.list_runs().await.unwrap();
        runs.sort();
        assert_eq!(runs, vec!["alpha", "beta"]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_file_delete() {
        let dir = temp_dir();
        let cp = FileCheckpoint::new(&dir);

        cp.save(&CheckpointState::new("run-1", vec![], 0))
            .await
            .unwrap();
        assert!(dir.join("run-1.json").exists());

        cp.delete("run-1").await.unwrap();
        assert!(!dir.join("run-1.json").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_validate_run_id_accepts_safe_ids() {
        assert!(FileCheckpoint::validate_run_id("run-1").is_ok());
        assert!(FileCheckpoint::validate_run_id("Run_2.v3").is_ok());
        assert!(FileCheckpoint::validate_run_id("abc123").is_ok());
    }

    #[test]
    fn test_validate_run_id_rejects_traversal() {
        assert!(FileCheckpoint::validate_run_id("..").is_err());
        assert!(FileCheckpoint::validate_run_id("../etc/passwd").is_err());
        assert!(FileCheckpoint::validate_run_id("foo/bar").is_err());
        assert!(FileCheckpoint::validate_run_id("foo\\bar").is_err());
        assert!(FileCheckpoint::validate_run_id("/etc/passwd").is_err());
        assert!(FileCheckpoint::validate_run_id("").is_err());
    }

    #[tokio::test]
    async fn test_save_rejects_path_traversal() {
        let dir = temp_dir();
        let cp = FileCheckpoint::new(&dir);

        // A run_id that tries to escape the checkpoint directory must be
        // rejected rather than writing outside `dir`.
        let state = CheckpointState::new("../escape", vec![], 0);
        assert!(cp.save(&state).await.is_err());
        assert!(cp.load("../escape").await.is_err());
        assert!(cp.delete("/etc/passwd").await.is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn temp_file_count(dir: &PathBuf) -> usize {
        std::fs::read_dir(dir)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter(|e| e.file_name().to_string_lossy().contains(".json.tmp."))
                    .count()
            })
            .unwrap_or(0)
    }

    #[tokio::test]
    async fn test_file_save_leaves_no_temp_files() {
        // The happy path must consume its temp file via the atomic rename and
        // never leave a `.json.tmp.*` artifact behind.
        let dir = temp_dir();
        let cp = FileCheckpoint::new(&dir);
        cp.save(&CheckpointState::new(
            "run-tmp",
            vec![Message::user("x")],
            1,
        ))
        .await
        .unwrap();

        assert!(dir.join("run-tmp.json").exists());
        assert_eq!(
            temp_file_count(&dir),
            0,
            "no temp file may remain after save"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn test_file_save_error_removes_temp() {
        use std::os::unix::fs::PermissionsExt;

        // Make the checkpoint directory read-only so creating the temp file
        // fails after `create_dir_all`. Regardless of whether the save fails
        // (unprivileged user) or unexpectedly succeeds (root can write into a
        // read-only dir), no orphaned temp file may be left behind — that is
        // the invariant FIX 3 guarantees.
        let dir = temp_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let mut perms = std::fs::metadata(&dir).unwrap().permissions();
        perms.set_mode(0o555);
        std::fs::set_permissions(&dir, perms).unwrap();

        let cp = FileCheckpoint::new(&dir);
        let _ = cp.save(&CheckpointState::new("run-1", vec![], 1)).await;

        // Restore write permissions so we can inspect and clean up.
        let mut perms = std::fs::metadata(&dir).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&dir, perms).unwrap();

        assert_eq!(
            temp_file_count(&dir),
            0,
            "a failed save must not orphan a temp file"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_file_overwrite() {
        let dir = temp_dir();
        let cp = FileCheckpoint::new(&dir);

        cp.save(&CheckpointState::new("run-1", vec![], 1))
            .await
            .unwrap();
        cp.save(&CheckpointState::new(
            "run-1",
            vec![Message::user("updated")],
            5,
        ))
        .await
        .unwrap();

        let loaded = cp.load("run-1").await.unwrap().unwrap();
        assert_eq!(loaded.iteration, 5);
        assert_eq!(loaded.messages.len(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
