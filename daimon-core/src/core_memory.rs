//! Core (always-in-context) memory trait.
//!
//! Unlike [`Memory`](crate::memory::Memory), which holds the rolling
//! conversation log, [`CoreMemory`] holds a small, bounded set of
//! self-editable facts (persona, user preferences, standing instructions)
//! that a host application renders into the system prompt on every turn —
//! the MemGPT/Letta "core memory" pattern. Built-in implementations live in
//! the `daimon` facade crate.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::error::Result;

/// A single named, optionally size-limited block of core memory.
///
/// `limit` is a character count; `None` means unbounded. Implementations of
/// [`CoreMemory::put_block`] and [`CoreMemory::append_block`] must reject
/// writes that would push `value` past `limit`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreMemoryBlock {
    /// Short identifier for the block (e.g. `"persona"`, `"user"`).
    pub label: String,
    /// The block's current text content.
    pub value: String,
    /// Maximum length of `value` in characters, or `None` for unbounded.
    pub limit: Option<usize>,
}

impl CoreMemoryBlock {
    /// Creates a new block with no size limit.
    pub fn new(label: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            value: value.into(),
            limit: None,
        }
    }

    /// Sets a character limit on this block.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }
}

/// Trait for always-in-context core memory backends.
///
/// Core memory is distinct from [`Memory`](crate::memory::Memory): it is not
/// a chronological log, it is a small set of labeled blocks that are
/// rendered as a whole and re-injected into the system prompt every turn.
/// The agent itself (via a tool) or the host application can read and edit
/// blocks between turns.
pub trait CoreMemory: Send + Sync {
    /// Returns all blocks currently stored.
    fn blocks(&self) -> impl Future<Output = Result<Vec<CoreMemoryBlock>>> + Send;

    /// Returns a single block by label, if it exists.
    fn get_block(
        &self,
        label: &str,
    ) -> impl Future<Output = Result<Option<CoreMemoryBlock>>> + Send {
        async move { Ok(self.blocks().await?.into_iter().find(|b| b.label == label)) }
    }

    /// Creates a block (if `block.label` is new) or overwrites the value and
    /// limit of an existing one. Fails if `block.value` exceeds `block.limit`.
    fn put_block(&self, block: CoreMemoryBlock) -> impl Future<Output = Result<()>> + Send;

    /// Appends `text` to an existing block's value (creating an unbounded
    /// block first if `label` is unknown). Fails without modifying the block
    /// if the result would exceed the block's limit.
    fn append_block(&self, label: &str, text: &str) -> impl Future<Output = Result<()>> + Send;

    /// Removes a block. Returns `true` if it existed.
    fn remove_block(&self, label: &str) -> impl Future<Output = Result<bool>> + Send;

    /// Renders all blocks into a single string suitable for splicing into a
    /// system prompt. Default rendering is `"## {label}\n{value}"` per
    /// block, joined by blank lines, in the order returned by [`blocks`](CoreMemory::blocks).
    fn render(&self) -> impl Future<Output = Result<String>> + Send {
        async move { Ok(render_blocks(&self.blocks().await?)) }
    }
}

/// Shared rendering logic used by [`CoreMemory::render`]'s default
/// implementation and available to custom implementations that want the
/// same format.
pub fn render_blocks(blocks: &[CoreMemoryBlock]) -> String {
    blocks
        .iter()
        .map(|b| format!("## {}\n{}", b.label, b.value))
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Object-safe wrapper for the `CoreMemory` trait, enabling dynamic dispatch
/// via `Arc<dyn ErasedCoreMemory>`.
pub trait ErasedCoreMemory: Send + Sync {
    fn blocks_erased(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<CoreMemoryBlock>>> + Send + '_>>;

    fn get_block_erased<'a>(
        &'a self,
        label: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<CoreMemoryBlock>>> + Send + 'a>>;

    fn put_block_erased(
        &self,
        block: CoreMemoryBlock,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>>;

    fn append_block_erased<'a>(
        &'a self,
        label: &'a str,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    fn remove_block_erased<'a>(
        &'a self,
        label: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<bool>> + Send + 'a>>;

    fn render_erased(&self) -> Pin<Box<dyn Future<Output = Result<String>> + Send + '_>>;
}

impl<T: CoreMemory> ErasedCoreMemory for T {
    fn blocks_erased(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<CoreMemoryBlock>>> + Send + '_>> {
        Box::pin(self.blocks())
    }

    fn get_block_erased<'a>(
        &'a self,
        label: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<CoreMemoryBlock>>> + Send + 'a>> {
        Box::pin(self.get_block(label))
    }

    fn put_block_erased(
        &self,
        block: CoreMemoryBlock,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        Box::pin(self.put_block(block))
    }

    fn append_block_erased<'a>(
        &'a self,
        label: &'a str,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(self.append_block(label, text))
    }

    fn remove_block_erased<'a>(
        &'a self,
        label: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<bool>> + Send + 'a>> {
        Box::pin(self.remove_block(label))
    }

    fn render_erased(&self) -> Pin<Box<dyn Future<Output = Result<String>> + Send + '_>> {
        Box::pin(self.render())
    }
}

/// Shared ownership of core memory via `Arc<dyn ErasedCoreMemory>`.
pub type SharedCoreMemory = Arc<dyn ErasedCoreMemory>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct VecCoreMemory(Mutex<Vec<CoreMemoryBlock>>);

    impl CoreMemory for VecCoreMemory {
        async fn blocks(&self) -> Result<Vec<CoreMemoryBlock>> {
            Ok(self.0.lock().unwrap().clone())
        }

        async fn put_block(&self, block: CoreMemoryBlock) -> Result<()> {
            if let Some(limit) = block.limit
                && block.value.chars().count() > limit
            {
                return Err(crate::error::DaimonError::Other(format!(
                    "block '{}' exceeds limit of {limit} characters",
                    block.label
                )));
            }
            let mut blocks = self.0.lock().unwrap();
            if let Some(existing) = blocks.iter_mut().find(|b| b.label == block.label) {
                *existing = block;
            } else {
                blocks.push(block);
            }
            Ok(())
        }

        async fn append_block(&self, label: &str, text: &str) -> Result<()> {
            let mut blocks = self.0.lock().unwrap();
            if let Some(existing) = blocks.iter_mut().find(|b| b.label == label) {
                let candidate = format!("{}{}", existing.value, text);
                if let Some(limit) = existing.limit
                    && candidate.chars().count() > limit
                {
                    return Err(crate::error::DaimonError::Other(format!(
                        "block '{label}' exceeds limit of {limit} characters"
                    )));
                }
                existing.value = candidate;
            } else {
                blocks.push(CoreMemoryBlock::new(label, text));
            }
            Ok(())
        }

        async fn remove_block(&self, label: &str) -> Result<bool> {
            let mut blocks = self.0.lock().unwrap();
            let before = blocks.len();
            blocks.retain(|b| b.label != label);
            Ok(blocks.len() != before)
        }
    }

    #[tokio::test]
    async fn core_memory_is_implementable_from_core_alone() {
        let mem = VecCoreMemory(Mutex::new(Vec::new()));
        mem.put_block(CoreMemoryBlock::new("persona", "helpful assistant"))
            .await
            .unwrap();
        mem.append_block("persona", " who is concise")
            .await
            .unwrap();

        let block = mem.get_block("persona").await.unwrap().unwrap();
        assert_eq!(block.value, "helpful assistant who is concise");

        let rendered = mem.render().await.unwrap();
        assert_eq!(rendered, "## persona\nhelpful assistant who is concise");

        assert!(mem.remove_block("persona").await.unwrap());
        assert!(mem.get_block("persona").await.unwrap().is_none());

        let shared: SharedCoreMemory = Arc::new(VecCoreMemory(Mutex::new(Vec::new())));
        shared
            .put_block_erased(CoreMemoryBlock::new("user", "likes Rust"))
            .await
            .unwrap();
        assert_eq!(shared.blocks_erased().await.unwrap().len(), 1);
        assert_eq!(shared.render_erased().await.unwrap(), "## user\nlikes Rust");
    }

    #[tokio::test]
    async fn put_block_rejects_over_limit_value() {
        let mem = VecCoreMemory(Mutex::new(Vec::new()));
        let err = mem
            .put_block(CoreMemoryBlock::new("persona", "way too long").with_limit(4))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("exceeds limit"));
    }
}
