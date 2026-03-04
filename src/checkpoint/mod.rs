//! Checkpointing and state persistence for resumable agent runs.
//!
//! When an agent run is interrupted (crash, timeout, manual cancellation),
//! checkpoints allow resuming from the last saved state instead of replaying
//! from scratch.
//!
//! Implement [`Checkpoint`] for custom backends. Built-in implementations:
//! - [`InMemoryCheckpoint`] — ephemeral, useful for testing
//! - [`FileCheckpoint`] — persists to JSON files in a directory

mod file;
mod memory;
pub mod replay;
pub mod sync;
mod traits;
mod types;

pub use file::FileCheckpoint;
pub use memory::InMemoryCheckpoint;
pub use replay::{ExecutionTrace, RunSummary, TraceStep, inspect_run, list_runs};
pub use sync::{CheckpointReplicator, CheckpointSync};
pub use traits::{Checkpoint, ErasedCheckpoint};
pub use types::CheckpointState;
