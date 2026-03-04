//! Checkpoint trait and object-safe wrapper.

use std::future::Future;
use std::pin::Pin;

use crate::error::Result;

use super::CheckpointState;

/// Trait for persisting and loading agent run checkpoints.
///
/// Implementations must be `Send + Sync` so they can be shared across async tasks.
pub trait Checkpoint: Send + Sync {
    /// Saves a checkpoint. Overwrites any existing checkpoint with the same `run_id`.
    fn save(&self, state: &CheckpointState) -> impl Future<Output = Result<()>> + Send;

    /// Loads the most recent checkpoint for the given run.
    /// Returns `None` if no checkpoint exists.
    fn load(&self, run_id: &str) -> impl Future<Output = Result<Option<CheckpointState>>> + Send;

    /// Lists all stored run IDs.
    fn list_runs(&self) -> impl Future<Output = Result<Vec<String>>> + Send;

    /// Deletes all checkpoints for the given run.
    fn delete(&self, run_id: &str) -> impl Future<Output = Result<()>> + Send;
}

/// Object-safe wrapper for [`Checkpoint`], enabling `Arc<dyn ErasedCheckpoint>`.
pub trait ErasedCheckpoint: Send + Sync {
    /// Saves a checkpoint.
    fn save_erased<'a>(
        &'a self,
        state: &'a CheckpointState,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    /// Loads a checkpoint.
    fn load_erased<'a>(
        &'a self,
        run_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<CheckpointState>>> + Send + 'a>>;

    /// Lists all run IDs.
    fn list_runs_erased(&self) -> Pin<Box<dyn Future<Output = Result<Vec<String>>> + Send + '_>>;

    /// Deletes checkpoints for a run.
    fn delete_erased<'a>(
        &'a self,
        run_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
}

impl<T: Checkpoint> ErasedCheckpoint for T {
    fn save_erased<'a>(
        &'a self,
        state: &'a CheckpointState,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(self.save(state))
    }

    fn load_erased<'a>(
        &'a self,
        run_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<CheckpointState>>> + Send + 'a>> {
        Box::pin(self.load(run_id))
    }

    fn list_runs_erased(&self) -> Pin<Box<dyn Future<Output = Result<Vec<String>>> + Send + '_>> {
        Box::pin(self.list_runs())
    }

    fn delete_erased<'a>(
        &'a self,
        run_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(self.delete(run_id))
    }
}
