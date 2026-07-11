//! Conversation memory for persisting message history across agent turns,
//! plus optional tiered memory subsystems that generalize the
//! MemGPT/Letta-style memory model (core, archival, episodic).
//!
//! Implement [`Memory`] for custom conversation backends. Built-in
//! implementations:
//! - [`SlidingWindowMemory`] — keeps the most recent N messages
//! - [`TokenWindowMemory`] — keeps messages within a token budget
//! - [`SummaryMemory`] — summarizes old messages using an LLM
//! - [`SqliteMemory`] — persists to SQLite (feature = "sqlite")
//! - [`RedisMemory`] — persists to Redis (feature = "redis")
//!
//! ## Tiered memory subsystems
//!
//! These are additive and optional — the base [`Memory`] trait and its
//! implementations above are unchanged. Each subsystem is its own trait with
//! its own built-in implementations, and [`TieredMemory`] composes them
//! behind a single type that still implements [`Memory`] for drop-in use
//! with [`AgentBuilder::memory`](crate::agent::AgentBuilder::memory):
//!
//! - [`CoreMemory`] — a small, bounded, always-in-context block of editable
//!   facts (persona, user preferences) rendered into the system prompt.
//!   Built-ins: [`InMemoryCoreMemory`], [`SqliteCoreMemory`] (feature = "sqlite").
//! - [`ArchivalMemory`] — explicit write/search over long-term facts,
//!   decoupled from the turn-by-turn conversation log. Built-ins:
//!   [`InMemoryArchivalMemory`] (lexical), [`VectorArchivalMemory`] (adapts
//!   any [`VectorStore`](crate::retriever::VectorStore) +
//!   [`EmbeddingModel`](crate::model::EmbeddingModel) for semantic search),
//!   [`SqliteArchivalMemory`] (FTS5, feature = "sqlite").
//! - [`EpisodicMemory`] — a structured, timestamped event log, distinct from
//!   the conversation's chat messages. Built-ins: [`InMemoryEpisodicMemory`],
//!   [`SqliteEpisodicMemory`] (feature = "sqlite").
//!
//! Knowledge-graph memory (entities/relationships with graph traversal) and
//! LLM-driven archival consolidation are intentionally out of scope here —
//! see the project changelog for rationale. [`SummaryMemory`] already covers
//! the analogous LLM-summarization need on the conversation side.

mod archival_memory;
mod core_memory;
mod episodic_memory;
mod sliding_window;
mod summary;
mod tiered;
mod token_window;

pub use archival_memory::{InMemoryArchivalMemory, VectorArchivalMemory};
pub use core_memory::InMemoryCoreMemory;
pub use daimon_core::{
    ArchivalMemory, ArchivalRecord, CoreMemory, CoreMemoryBlock, EpisodicEvent, EpisodicMemory,
    EpisodicQuery, ErasedArchivalMemory, ErasedCoreMemory, ErasedEpisodicMemory, ErasedMemory,
    Memory, SharedArchivalMemory, SharedCoreMemory, SharedEpisodicMemory, SharedMemory,
};
pub use episodic_memory::InMemoryEpisodicMemory;
pub use sliding_window::SlidingWindowMemory;
pub use summary::SummaryMemory;
pub use tiered::TieredMemory;
pub use token_window::TokenWindowMemory;

#[cfg(feature = "sqlite")]
mod sqlite;

#[cfg(feature = "sqlite")]
pub use sqlite::SqliteMemory;

#[cfg(feature = "sqlite")]
mod sqlite_tiered;

#[cfg(feature = "sqlite")]
pub use sqlite_tiered::{SqliteArchivalMemory, SqliteCoreMemory, SqliteEpisodicMemory};

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
