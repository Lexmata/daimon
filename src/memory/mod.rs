//! Conversation memory for persisting message history across agent turns.
//!
//! Implement [`Memory`] for custom backends. Built-in implementations:
//! - [`SlidingWindowMemory`] — keeps the most recent N messages
//! - [`TokenWindowMemory`] — keeps messages within a token budget
//! - [`SummaryMemory`] — summarizes old messages using an LLM
//! - [`SqliteMemory`] — persists to SQLite (feature = "sqlite")

mod sliding_window;
mod summary;
mod token_window;
mod traits;

pub use sliding_window::SlidingWindowMemory;
pub use summary::SummaryMemory;
pub use token_window::TokenWindowMemory;
pub use traits::{ErasedMemory, Memory, SharedMemory};

#[cfg(feature = "sqlite")]
mod sqlite;

#[cfg(feature = "sqlite")]
pub use sqlite::SqliteMemory;
