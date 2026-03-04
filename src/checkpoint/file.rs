//! File-based checkpoint backend that persists to JSON files.

use std::path::PathBuf;

use crate::error::{DaimonError, Result};

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

    fn run_path(&self, run_id: &str) -> PathBuf {
        self.dir.join(format!("{run_id}.json"))
    }
}

impl Checkpoint for FileCheckpoint {
    async fn save(&self, state: &CheckpointState) -> Result<()> {
        let dir = self.dir.clone();
        let path = self.run_path(&state.run_id);
        let json = serde_json::to_string_pretty(state)?;

        tokio::task::spawn_blocking(move || {
            std::fs::create_dir_all(&dir).map_err(|e| {
                DaimonError::Other(format!("failed to create checkpoint dir: {e}"))
            })?;
            std::fs::write(&path, json).map_err(|e| {
                DaimonError::Other(format!("failed to write checkpoint: {e}"))
            })?;
            Ok::<_, DaimonError>(())
        })
        .await
        .map_err(|e| DaimonError::Other(format!("checkpoint spawn error: {e}")))?
    }

    async fn load(&self, run_id: &str) -> Result<Option<CheckpointState>> {
        let path = self.run_path(run_id);

        tokio::task::spawn_blocking(move || {
            if !path.exists() {
                return Ok(None);
            }
            let json = std::fs::read_to_string(&path).map_err(|e| {
                DaimonError::Other(format!("failed to read checkpoint: {e}"))
            })?;
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
            let entries = std::fs::read_dir(&dir).map_err(|e| {
                DaimonError::Other(format!("failed to read checkpoint dir: {e}"))
            })?;
            for entry in entries {
                let entry = entry.map_err(|e| {
                    DaimonError::Other(format!("failed to read dir entry: {e}"))
                })?;
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "json") {
                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                        runs.push(stem.to_string());
                    }
                }
            }
            Ok(runs)
        })
        .await
        .map_err(|e| DaimonError::Other(format!("checkpoint spawn error: {e}")))?
    }

    async fn delete(&self, run_id: &str) -> Result<()> {
        let path = self.run_path(run_id);

        tokio::task::spawn_blocking(move || {
            if path.exists() {
                std::fs::remove_file(&path).map_err(|e| {
                    DaimonError::Other(format!("failed to delete checkpoint: {e}"))
                })?;
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
        let dir = std::env::temp_dir().join(format!(
            "daimon_cp_test_{}_{id}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[tokio::test]
    async fn test_file_save_load() {
        let dir = temp_dir();
        let cp = FileCheckpoint::new(&dir);
        let state = CheckpointState::new(
            "run-1",
            vec![Message::user("hello")],
            1,
        );

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
