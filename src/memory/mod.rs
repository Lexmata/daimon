//! Conversation memory for persisting message history across agent turns.
//!
//! Implement [`Memory`] for custom backends. [`SlidingWindowMemory`] keeps the most recent N messages in memory.

mod sliding_window;
mod traits;

pub use sliding_window::SlidingWindowMemory;
pub use traits::{ErasedMemory, Memory, SharedMemory};
