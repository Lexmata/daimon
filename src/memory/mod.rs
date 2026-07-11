//! Conversation memory for persisting message history across agent turns.
//!
//! Implement [`Memory`] for custom backends. Built-in implementations:
//! - [`SlidingWindowMemory`] — keeps the most recent N messages
//! - [`TokenWindowMemory`] — keeps messages within a token budget
//! - [`SummaryMemory`] — summarizes old messages using an LLM
//! - [`SqliteMemory`] — persists to SQLite (feature = "sqlite")
//! - [`RedisMemory`] — persists to Redis (feature = "redis")

mod sliding_window;
mod summary;
mod token_window;

pub use daimon_core::{ErasedMemory, Memory, SharedMemory};
pub use sliding_window::SlidingWindowMemory;
pub use summary::SummaryMemory;
pub use token_window::TokenWindowMemory;

#[cfg(feature = "sqlite")]
mod sqlite;

#[cfg(feature = "sqlite")]
pub use sqlite::SqliteMemory;

#[cfg(feature = "redis")]
mod redis;

#[cfg(feature = "redis")]
pub use self::redis::RedisMemory;

/// Returns the number of messages forming the eviction group at the front of
/// the queue.
///
/// An assistant message carrying tool calls and its contiguous following
/// [`Role::Tool`](crate::model::types::Role::Tool) result messages form an
/// atomic group: evicting the assistant message while keeping the tool
/// results would leave orphaned tool results, which OpenAI- and
/// Anthropic-style APIs reject. Any other message is a group of one.
pub(crate) fn eviction_group_len(
    messages: &std::collections::VecDeque<crate::model::types::Message>,
) -> usize {
    use crate::model::types::Role;

    let Some(first) = messages.front() else {
        return 0;
    };

    if first.role != Role::Assistant || first.tool_calls.is_empty() {
        return 1;
    }

    let mut len = 1;
    while messages.get(len).is_some_and(|m| m.role == Role::Tool) {
        len += 1;
    }
    len
}
