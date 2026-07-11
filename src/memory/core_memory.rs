//! Core (always-in-context) memory implementations.

use tokio::sync::RwLock;

use crate::error::{DaimonError, Result};
use crate::memory::{CoreMemory, CoreMemoryBlock};

/// In-process [`CoreMemory`] backed by a `HashMap`. Data is lost when the
/// process exits; use [`SqliteCoreMemory`](super::SqliteCoreMemory)
/// (feature = "sqlite") for persistence.
///
/// Thread-safe via internal `RwLock`. Block order returned by
/// [`blocks`](CoreMemory::blocks) is insertion order.
#[derive(Default)]
pub struct InMemoryCoreMemory {
    blocks: RwLock<Vec<CoreMemoryBlock>>,
}

impl InMemoryCoreMemory {
    /// Creates an empty core memory store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a store pre-seeded with the given blocks. Later blocks with a
    /// label matching an earlier one overwrite it, mirroring [`put_block`](CoreMemory::put_block).
    pub fn with_blocks(blocks: impl IntoIterator<Item = CoreMemoryBlock>) -> Result<Self> {
        let mut seeded: Vec<CoreMemoryBlock> = Vec::new();
        for block in blocks {
            check_limit(&block.label, &block.value, block.limit)?;
            if let Some(existing) = seeded.iter_mut().find(|b| b.label == block.label) {
                *existing = block;
            } else {
                seeded.push(block);
            }
        }
        Ok(Self {
            blocks: RwLock::new(seeded),
        })
    }
}

fn check_limit(label: &str, value: &str, limit: Option<usize>) -> Result<()> {
    if let Some(limit) = limit
        && value.chars().count() > limit
    {
        return Err(DaimonError::Other(format!(
            "core memory block '{label}' exceeds limit of {limit} characters"
        )));
    }
    Ok(())
}

impl CoreMemory for InMemoryCoreMemory {
    async fn blocks(&self) -> Result<Vec<CoreMemoryBlock>> {
        Ok(self.blocks.read().await.clone())
    }

    async fn put_block(&self, block: CoreMemoryBlock) -> Result<()> {
        check_limit(&block.label, &block.value, block.limit)?;
        let mut blocks = self.blocks.write().await;
        if let Some(existing) = blocks.iter_mut().find(|b| b.label == block.label) {
            *existing = block;
        } else {
            blocks.push(block);
        }
        Ok(())
    }

    async fn append_block(&self, label: &str, text: &str) -> Result<()> {
        let mut blocks = self.blocks.write().await;
        if let Some(existing) = blocks.iter_mut().find(|b| b.label == label) {
            let candidate = format!("{}{}", existing.value, text);
            check_limit(label, &candidate, existing.limit)?;
            existing.value = candidate;
        } else {
            blocks.push(CoreMemoryBlock::new(label, text));
        }
        Ok(())
    }

    async fn remove_block(&self, label: &str) -> Result<bool> {
        let mut blocks = self.blocks.write().await;
        let before = blocks.len();
        blocks.retain(|b| b.label != label);
        Ok(blocks.len() != before)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::ErasedCoreMemory;
    use std::sync::Arc;

    #[tokio::test]
    async fn put_and_get_block() {
        let mem = InMemoryCoreMemory::new();
        mem.put_block(CoreMemoryBlock::new("persona", "a helpful assistant"))
            .await
            .unwrap();

        let blocks = mem.blocks().await.unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].value, "a helpful assistant");
    }

    #[tokio::test]
    async fn put_block_overwrites_existing_label() {
        let mem = InMemoryCoreMemory::new();
        mem.put_block(CoreMemoryBlock::new("persona", "v1"))
            .await
            .unwrap();
        mem.put_block(CoreMemoryBlock::new("persona", "v2"))
            .await
            .unwrap();

        let blocks = mem.blocks().await.unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].value, "v2");
    }

    #[tokio::test]
    async fn append_block_creates_when_absent() {
        let mem = InMemoryCoreMemory::new();
        mem.append_block("notes", "first note").await.unwrap();
        let blocks = mem.blocks().await.unwrap();
        assert_eq!(blocks[0].value, "first note");
    }

    #[tokio::test]
    async fn append_block_respects_limit() {
        let mem = InMemoryCoreMemory::new();
        mem.put_block(CoreMemoryBlock::new("notes", "1234").with_limit(5))
            .await
            .unwrap();
        mem.append_block("notes", "5").await.unwrap();
        assert!(mem.append_block("notes", "6").await.is_err());

        let blocks = mem.blocks().await.unwrap();
        assert_eq!(
            blocks[0].value, "12345",
            "rejected append must not mutate value"
        );
    }

    #[tokio::test]
    async fn remove_block() {
        let mem = InMemoryCoreMemory::new();
        mem.put_block(CoreMemoryBlock::new("a", "x")).await.unwrap();
        assert!(mem.remove_block("a").await.unwrap());
        assert!(!mem.remove_block("a").await.unwrap());
        assert!(mem.blocks().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn render_joins_blocks_in_order() {
        let mem = InMemoryCoreMemory::new();
        mem.put_block(CoreMemoryBlock::new("persona", "helpful"))
            .await
            .unwrap();
        mem.put_block(CoreMemoryBlock::new("user", "likes cats"))
            .await
            .unwrap();

        let rendered = mem.render().await.unwrap();
        assert_eq!(rendered, "## persona\nhelpful\n\n## user\nlikes cats");
    }

    #[tokio::test]
    async fn with_blocks_seeds_store() {
        let mem = InMemoryCoreMemory::with_blocks([
            CoreMemoryBlock::new("a", "1"),
            CoreMemoryBlock::new("b", "2"),
        ])
        .unwrap();
        assert_eq!(mem.blocks().await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn erased_wrapper_works() {
        let mem: Arc<dyn ErasedCoreMemory> = Arc::new(InMemoryCoreMemory::new());
        mem.put_block_erased(CoreMemoryBlock::new("a", "1"))
            .await
            .unwrap();
        assert_eq!(mem.render_erased().await.unwrap(), "## a\n1");
    }
}
