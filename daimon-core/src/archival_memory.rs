//! Archival memory trait: explicit write/search over long-term facts.
//!
//! Unlike [`Memory`](crate::memory::Memory) (a linear, append-only
//! conversation log) or [`VectorStore`](crate::vector_store::VectorStore) (a
//! low-level embedding index), [`ArchivalMemory`] models a fact store that
//! consumers write to explicitly and retrieve by relevance rather than by
//! recency. Implementations may be lexical (full-text search), or compose an
//! existing [`VectorStore`](crate::vector_store::VectorStore) +
//! [`EmbeddingModel`](crate::embedding::EmbeddingModel) for semantic search.
//! Built-in implementations live in the `daimon` facade crate.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use crate::error::Result;

/// A fact stored in archival memory, returned from a search.
#[derive(Debug, Clone)]
pub struct ArchivalRecord {
    /// Backend-assigned unique identifier.
    pub id: String,
    /// The fact's text content.
    pub text: String,
    /// Arbitrary metadata attached at insert time.
    pub metadata: HashMap<String, Value>,
    /// Relevance score from the search backend, if any (higher = more
    /// relevant). `None` for retrieval methods that don't score.
    pub score: Option<f64>,
}

/// Trait for long-term archival fact storage, decoupled from the turn-by-turn
/// conversation log.
///
/// Facts are written explicitly via [`insert`](ArchivalMemory::insert) and
/// retrieved by relevance via [`search`](ArchivalMemory::search) — never by
/// simply replaying everything in insertion order.
pub trait ArchivalMemory: Send + Sync {
    /// Stores a fact and returns its assigned id.
    fn insert(
        &self,
        text: &str,
        metadata: HashMap<String, Value>,
    ) -> impl Future<Output = Result<String>> + Send;

    /// Returns up to `top_k` facts most relevant to `query`, ordered by
    /// descending relevance.
    fn search(
        &self,
        query: &str,
        top_k: usize,
    ) -> impl Future<Output = Result<Vec<ArchivalRecord>>> + Send;

    /// Deletes a fact by id. Returns `true` if it existed.
    fn delete(&self, id: &str) -> impl Future<Output = Result<bool>> + Send;

    /// Returns the total number of stored facts.
    fn count(&self) -> impl Future<Output = Result<usize>> + Send;
}

/// Object-safe wrapper for the `ArchivalMemory` trait, enabling dynamic
/// dispatch via `Arc<dyn ErasedArchivalMemory>`.
pub trait ErasedArchivalMemory: Send + Sync {
    fn insert_erased<'a>(
        &'a self,
        text: &'a str,
        metadata: HashMap<String, Value>,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>>;

    fn search_erased<'a>(
        &'a self,
        query: &'a str,
        top_k: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<ArchivalRecord>>> + Send + 'a>>;

    fn delete_erased<'a>(
        &'a self,
        id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<bool>> + Send + 'a>>;

    fn count_erased(&self) -> Pin<Box<dyn Future<Output = Result<usize>> + Send + '_>>;
}

impl<T: ArchivalMemory> ErasedArchivalMemory for T {
    fn insert_erased<'a>(
        &'a self,
        text: &'a str,
        metadata: HashMap<String, Value>,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        Box::pin(self.insert(text, metadata))
    }

    fn search_erased<'a>(
        &'a self,
        query: &'a str,
        top_k: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<ArchivalRecord>>> + Send + 'a>> {
        Box::pin(self.search(query, top_k))
    }

    fn delete_erased<'a>(
        &'a self,
        id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<bool>> + Send + 'a>> {
        Box::pin(self.delete(id))
    }

    fn count_erased(&self) -> Pin<Box<dyn Future<Output = Result<usize>> + Send + '_>> {
        Box::pin(self.count())
    }
}

/// Shared ownership of archival memory via `Arc<dyn ErasedArchivalMemory>`.
pub type SharedArchivalMemory = Arc<dyn ErasedArchivalMemory>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct VecArchivalMemory(Mutex<Vec<ArchivalRecord>>);

    impl ArchivalMemory for VecArchivalMemory {
        async fn insert(&self, text: &str, metadata: HashMap<String, Value>) -> Result<String> {
            let mut records = self.0.lock().unwrap();
            let id = format!("rec-{}", records.len());
            records.push(ArchivalRecord {
                id: id.clone(),
                text: text.to_string(),
                metadata,
                score: None,
            });
            Ok(id)
        }

        async fn search(&self, query: &str, top_k: usize) -> Result<Vec<ArchivalRecord>> {
            let records = self.0.lock().unwrap();
            Ok(records
                .iter()
                .filter(|r| r.text.contains(query))
                .take(top_k)
                .cloned()
                .collect())
        }

        async fn delete(&self, id: &str) -> Result<bool> {
            let mut records = self.0.lock().unwrap();
            let before = records.len();
            records.retain(|r| r.id != id);
            Ok(records.len() != before)
        }

        async fn count(&self) -> Result<usize> {
            Ok(self.0.lock().unwrap().len())
        }
    }

    #[tokio::test]
    async fn archival_memory_is_implementable_from_core_alone() {
        let mem = VecArchivalMemory(Mutex::new(Vec::new()));
        let id = mem.insert("the sky is blue", HashMap::new()).await.unwrap();
        mem.insert("water is wet", HashMap::new()).await.unwrap();

        assert_eq!(mem.count().await.unwrap(), 2);
        let results = mem.search("sky", 5).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, id);

        assert!(mem.delete(&id).await.unwrap());
        assert_eq!(mem.count().await.unwrap(), 1);

        let shared: SharedArchivalMemory = Arc::new(VecArchivalMemory(Mutex::new(Vec::new())));
        shared
            .insert_erased("erased fact", HashMap::new())
            .await
            .unwrap();
        assert_eq!(shared.count_erased().await.unwrap(), 1);
    }
}
