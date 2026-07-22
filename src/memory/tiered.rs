//! [`TieredMemory`]: composes conversation, core, archival, and episodic
//! memory into a single type.
//!
//! `TieredMemory` implements the base [`Memory`] trait by delegating to its
//! conversation sub-memory, so it drops in wherever a plain `Memory` is
//! expected (e.g. [`AgentBuilder::memory`](crate::agent::AgentBuilder::memory)).
//! The richer sub-APIs ([`core`](TieredMemory::core),
//! [`archival`](TieredMemory::archival), [`episodic`](TieredMemory::episodic))
//! are available to consumers who want them directly — the agent runner
//! itself does not reach into them, so wiring core memory into a system
//! prompt or exposing archival/episodic search as tools is the host
//! application's responsibility. See [`TieredMemory::system_prompt_block`].

use std::sync::Arc;

use crate::error::{DaimonError, Result};
use crate::memory::{
    ArchivalMemory, CoreMemory, EpisodicMemory, ErasedArchivalMemory, ErasedCoreMemory,
    ErasedEpisodicMemory, Memory, SharedArchivalMemory, SharedCoreMemory, SharedEpisodicMemory,
    SharedMemory, SlidingWindowMemory,
};
use crate::model::types::Message;

/// Composes the base conversation [`Memory`] with optional core, archival,
/// and episodic sub-memories.
///
/// Only the conversation sub-memory participates in the [`Memory`] trait
/// (`add_message`/`get_messages`/`clear`); the other tiers are written to
/// and queried explicitly through their own APIs. In particular,
/// [`Memory::clear`] clears *only* the conversation log — archival facts and
/// episodic events are long-term by design and are not affected.
pub struct TieredMemory {
    conversation: SharedMemory,
    core: Option<SharedCoreMemory>,
    archival: Option<SharedArchivalMemory>,
    episodic: Option<SharedEpisodicMemory>,
}

impl TieredMemory {
    /// Creates a `TieredMemory` with the given conversation backend and no
    /// core, archival, or episodic tiers configured.
    pub fn new(conversation: impl Memory + 'static) -> Self {
        Self {
            conversation: Arc::new(conversation),
            core: None,
            archival: None,
            episodic: None,
        }
    }

    /// Attaches a core memory tier.
    pub fn with_core(mut self, core: impl CoreMemory + 'static) -> Self {
        self.core = Some(Arc::new(core));
        self
    }

    /// Attaches an archival memory tier.
    pub fn with_archival(mut self, archival: impl ArchivalMemory + 'static) -> Self {
        self.archival = Some(Arc::new(archival));
        self
    }

    /// Attaches an episodic memory tier.
    pub fn with_episodic(mut self, episodic: impl EpisodicMemory + 'static) -> Self {
        self.episodic = Some(Arc::new(episodic));
        self
    }

    /// The conversation (short-term, windowed) sub-memory.
    pub fn conversation(&self) -> &SharedMemory {
        &self.conversation
    }

    /// The core memory tier, if configured.
    pub fn core(&self) -> Option<&Arc<dyn ErasedCoreMemory>> {
        self.core.as_ref()
    }

    /// The archival memory tier, if configured.
    pub fn archival(&self) -> Option<&Arc<dyn ErasedArchivalMemory>> {
        self.archival.as_ref()
    }

    /// The episodic memory tier, if configured.
    pub fn episodic(&self) -> Option<&Arc<dyn ErasedEpisodicMemory>> {
        self.episodic.as_ref()
    }

    /// Renders the core memory tier (if configured) for splicing into a
    /// system prompt. Returns `Ok(None)` when no core memory is attached, so
    /// callers can write `if let Some(block) = tiered.system_prompt_block().await? { ... }`
    /// without special-casing the unconfigured case.
    pub async fn system_prompt_block(&self) -> Result<Option<String>> {
        match &self.core {
            Some(core) => Ok(Some(core.render_erased().await?)),
            None => Ok(None),
        }
    }
}

impl Default for TieredMemory {
    /// Conversation defaults to [`SlidingWindowMemory`] with 50 messages, no
    /// core/archival/episodic tiers attached.
    fn default() -> Self {
        Self::new(SlidingWindowMemory::default())
    }
}

impl Memory for TieredMemory {
    async fn add_message(&self, message: &Message) -> Result<()> {
        self.conversation.add_message_erased(message).await
    }

    async fn get_messages(&self) -> Result<Vec<Message>> {
        self.conversation.get_messages_erased().await
    }

    /// Forwards to the conversation sub-memory's erased visitor, so a
    /// conversation backend that overrides [`Memory::with_messages`] keeps
    /// its no-clone path when wrapped in `TieredMemory`.
    async fn with_messages<R, F>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&[Message]) -> R + Send,
        R: Send,
    {
        let mut out = None;
        self.conversation
            .with_messages_erased(Box::new(|messages| out = Some(f(messages))))
            .await?;
        out.ok_or_else(|| {
            DaimonError::Other(
                "ErasedMemory::with_messages_erased completed without invoking the visitor".into(),
            )
        })
    }

    async fn clear(&self) -> Result<()> {
        self.conversation.clear_erased().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{
        CoreMemoryBlock, EpisodicQuery, InMemoryArchivalMemory, InMemoryCoreMemory,
        InMemoryEpisodicMemory,
    };

    #[tokio::test]
    async fn implements_base_memory_via_conversation() {
        let mem = TieredMemory::new(SlidingWindowMemory::new(10));
        mem.add_message(&Message::user("hi")).await.unwrap();
        assert_eq!(mem.get_messages().await.unwrap().len(), 1);
        mem.clear().await.unwrap();
        assert!(mem.get_messages().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn default_uses_sliding_window() {
        let mem = TieredMemory::default();
        for i in 0..60 {
            mem.add_message(&Message::user(format!("m{i}")))
                .await
                .unwrap();
        }
        // Default sliding window caps at 50.
        assert_eq!(mem.get_messages().await.unwrap().len(), 50);
    }

    #[tokio::test]
    async fn tiers_are_independent_of_conversation_clear() {
        let core = InMemoryCoreMemory::new();
        core.put_block(CoreMemoryBlock::new("persona", "a helpful bot"))
            .await
            .unwrap();
        let archival = InMemoryArchivalMemory::new();
        archival
            .insert("the moon is round", Default::default())
            .await
            .unwrap();
        let episodic = InMemoryEpisodicMemory::new();
        episodic
            .record("started", serde_json::json!({}))
            .await
            .unwrap();

        let mem = TieredMemory::new(SlidingWindowMemory::new(10))
            .with_core(core)
            .with_archival(archival)
            .with_episodic(episodic);

        mem.add_message(&Message::user("hi")).await.unwrap();
        mem.clear().await.unwrap();

        assert!(mem.get_messages().await.unwrap().is_empty());
        assert_eq!(mem.core().unwrap().blocks_erased().await.unwrap().len(), 1);
        assert_eq!(mem.archival().unwrap().count_erased().await.unwrap(), 1);
        assert_eq!(
            mem.episodic()
                .unwrap()
                .query_erased(EpisodicQuery::all())
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn system_prompt_block_reflects_core_memory() {
        let mem = TieredMemory::new(SlidingWindowMemory::new(10));
        assert!(mem.system_prompt_block().await.unwrap().is_none());

        let core = InMemoryCoreMemory::new();
        core.put_block(CoreMemoryBlock::new("persona", "concise assistant"))
            .await
            .unwrap();
        let mem = TieredMemory::new(SlidingWindowMemory::new(10)).with_core(core);

        let block = mem.system_prompt_block().await.unwrap().unwrap();
        assert_eq!(block, "## persona\nconcise assistant");
    }

    #[tokio::test]
    async fn with_messages_forwards_to_conversation() {
        let mem = TieredMemory::new(SlidingWindowMemory::new(10));
        mem.add_message(&Message::user("hi")).await.unwrap();
        mem.add_message(&Message::assistant("hello")).await.unwrap();

        let owned = mem.get_messages().await.unwrap();
        let borrowed = mem
            .with_messages(|messages| messages.to_vec())
            .await
            .unwrap();

        assert_eq!(borrowed.len(), owned.len());
        for (b, o) in borrowed.iter().zip(&owned) {
            assert_eq!(b.role, o.role);
            assert_eq!(b.content, o.content);
        }
    }

    #[tokio::test]
    async fn drops_in_as_a_plain_memory() {
        fn accepts_memory<M: Memory>(_m: &M) {}
        let mem = TieredMemory::default();
        accepts_memory(&mem);
    }
}
