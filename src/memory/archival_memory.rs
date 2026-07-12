//! Archival memory implementations.
//!
//! [`InMemoryArchivalMemory`] provides lexical (substring/keyword) search
//! with no extra dependencies. [`VectorArchivalMemory`] adapts any existing
//! [`VectorStore`] + [`EmbeddingModel`] pair into [`ArchivalMemory`] for
//! semantic search, reusing the retrieval stack instead of reinventing it.
//! For full-text search over SQLite (FTS5) see
//! [`SqliteArchivalMemory`](super::SqliteArchivalMemory) (feature = "sqlite").

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::Value;
use tokio::sync::RwLock;

use crate::error::Result;
use crate::memory::{ArchivalMemory, ArchivalRecord};
use crate::model::EmbeddingModel;
use crate::retriever::vector_store::VectorStore;
use crate::retriever::{Document, ScoredDocument};

#[derive(Clone)]
struct StoredFact {
    id: String,
    text: String,
    /// Lowercased form of `text`, computed once at insert time so `search`
    /// doesn't reallocate a fresh lowercase copy of every fact on every
    /// query.
    text_lower: String,
    metadata: HashMap<String, Value>,
}

/// In-process [`ArchivalMemory`] with naive case-insensitive substring
/// scoring: a fact's score is the number of query terms (whitespace-split)
/// it contains. No embedding model or vector store required. Data is lost
/// when the process exits.
#[derive(Default)]
pub struct InMemoryArchivalMemory {
    facts: RwLock<Vec<StoredFact>>,
    next_id: AtomicU64,
}

impl InMemoryArchivalMemory {
    /// Creates an empty archival store.
    pub fn new() -> Self {
        Self::default()
    }
}

impl ArchivalMemory for InMemoryArchivalMemory {
    async fn insert(&self, text: &str, metadata: HashMap<String, Value>) -> Result<String> {
        let id = format!("archival-{}", self.next_id.fetch_add(1, Ordering::Relaxed));
        self.facts.write().await.push(StoredFact {
            id: id.clone(),
            text: text.to_string(),
            text_lower: text.to_lowercase(),
            metadata,
        });
        Ok(id)
    }

    async fn search(&self, query: &str, top_k: usize) -> Result<Vec<ArchivalRecord>> {
        let query_lower = query.to_lowercase();
        let terms: Vec<&str> = query_lower.split_whitespace().collect();
        if terms.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

        let facts = self.facts.read().await;
        let mut scored: Vec<(f64, &StoredFact)> = facts
            .iter()
            .filter_map(|fact| {
                let hits = terms
                    .iter()
                    .filter(|t| fact.text_lower.contains(*t))
                    .count();
                (hits > 0).then_some((hits as f64, fact))
            })
            .collect();

        // Stable sort keeps insertion order among equal scores.
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scored.truncate(top_k);

        Ok(scored
            .into_iter()
            .map(|(score, fact)| ArchivalRecord {
                id: fact.id.clone(),
                text: fact.text.clone(),
                metadata: fact.metadata.clone(),
                score: Some(score),
            })
            .collect())
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        let mut facts = self.facts.write().await;
        let before = facts.len();
        facts.retain(|f| f.id != id);
        Ok(facts.len() != before)
    }

    async fn count(&self) -> Result<usize> {
        Ok(self.facts.read().await.len())
    }
}

/// Adapts an existing [`VectorStore`] + [`EmbeddingModel`] pair into
/// [`ArchivalMemory`], so consumers who already have a vector database wired
/// up (pgvector, Qdrant, OpenSearch, ...) get semantic archival search
/// without a separate storage layer.
///
/// IDs are generated locally (`archival-{n}`) unless the store's `upsert`
/// contract requires caller-supplied IDs, which this adapter satisfies by
/// always generating one before calling `upsert`.
pub struct VectorArchivalMemory<V, E> {
    store: V,
    embedder: E,
    next_id: AtomicU64,
}

impl<V: VectorStore, E: EmbeddingModel> VectorArchivalMemory<V, E> {
    /// Wraps a vector store and embedding model as archival memory.
    pub fn new(store: V, embedder: E) -> Self {
        Self {
            store,
            embedder,
            next_id: AtomicU64::new(0),
        }
    }

    /// Borrows the underlying vector store.
    pub fn store(&self) -> &V {
        &self.store
    }
}

impl<V: VectorStore, E: EmbeddingModel> ArchivalMemory for VectorArchivalMemory<V, E> {
    async fn insert(&self, text: &str, metadata: HashMap<String, Value>) -> Result<String> {
        let id = format!("archival-{}", self.next_id.fetch_add(1, Ordering::Relaxed));
        let embedding = self
            .embedder
            .embed(&[text])
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| {
                crate::error::DaimonError::Other("embedding model returned no vectors".into())
            })?;

        let mut document = Document::new(text);
        document.metadata = metadata;
        self.store.upsert(&id, embedding, document).await?;
        Ok(id)
    }

    async fn search(&self, query: &str, top_k: usize) -> Result<Vec<ArchivalRecord>> {
        let embedding = self
            .embedder
            .embed(&[query])
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| {
                crate::error::DaimonError::Other("embedding model returned no vectors".into())
            })?;

        let results: Vec<ScoredDocument> = self.store.query(embedding, top_k).await?;
        Ok(results
            .into_iter()
            .map(|scored| ArchivalRecord {
                // `ScoredDocument::id` is the same stable id passed to
                // `VectorStore::upsert` at insert time, so it round-trips
                // directly into `delete` (see `VectorStore::query`'s
                // contract in daimon-core).
                id: scored.id,
                text: scored.document.content,
                metadata: scored.document.metadata,
                score: Some(scored.score),
            })
            .collect())
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        self.store.delete(id).await
    }

    async fn count(&self) -> Result<usize> {
        self.store.count().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::ErasedArchivalMemory;
    use std::sync::Arc;

    #[tokio::test]
    async fn insert_and_search_by_substring() {
        let mem = InMemoryArchivalMemory::new();
        let id = mem.insert("the sky is blue", HashMap::new()).await.unwrap();
        mem.insert("grass is green", HashMap::new()).await.unwrap();

        let results = mem.search("sky blue", 5).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, id);
        assert_eq!(results[0].score, Some(2.0));
    }

    #[tokio::test]
    async fn search_respects_top_k() {
        let mem = InMemoryArchivalMemory::new();
        for i in 0..5 {
            mem.insert(&format!("fact number {i}"), HashMap::new())
                .await
                .unwrap();
        }
        let results = mem.search("fact", 2).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn search_with_no_matches_is_empty() {
        let mem = InMemoryArchivalMemory::new();
        mem.insert("apples and oranges", HashMap::new())
            .await
            .unwrap();
        assert!(mem.search("bananas", 5).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn delete_and_count() {
        let mem = InMemoryArchivalMemory::new();
        let id = mem.insert("fact one", HashMap::new()).await.unwrap();
        mem.insert("fact two", HashMap::new()).await.unwrap();
        assert_eq!(mem.count().await.unwrap(), 2);

        assert!(mem.delete(&id).await.unwrap());
        assert!(!mem.delete(&id).await.unwrap());
        assert_eq!(mem.count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn metadata_round_trips() {
        let mem = InMemoryArchivalMemory::new();
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), Value::String("wiki".into()));
        mem.insert("a fact", metadata).await.unwrap();

        let results = mem.search("fact", 1).await.unwrap();
        assert_eq!(results[0].metadata["source"], Value::String("wiki".into()));
    }

    #[tokio::test]
    async fn erased_wrapper_works() {
        let mem: Arc<dyn ErasedArchivalMemory> = Arc::new(InMemoryArchivalMemory::new());
        mem.insert_erased("erased fact", HashMap::new())
            .await
            .unwrap();
        assert_eq!(mem.count_erased().await.unwrap(), 1);
    }

    // --- VectorArchivalMemory ---

    struct FakeEmbedder;

    impl EmbeddingModel for FakeEmbedder {
        async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
            // Deterministic 1-dim "embedding": text length. Good enough to
            // exercise the adapter without a real model.
            Ok(texts.iter().map(|t| vec![t.len() as f32]).collect())
        }

        fn dimensions(&self) -> usize {
            1
        }
    }

    #[tokio::test]
    async fn vector_archival_memory_inserts_and_searches() {
        use crate::retriever::InMemoryVectorStoreBackend;

        let adapter = VectorArchivalMemory::new(InMemoryVectorStoreBackend::new(), FakeEmbedder);
        adapter.insert("hello", HashMap::new()).await.unwrap();
        adapter.insert("hello world", HashMap::new()).await.unwrap();

        assert_eq!(adapter.count().await.unwrap(), 2);
        let results = adapter.search("hello", 2).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn vector_archival_memory_search_ids_round_trip_to_delete() {
        use crate::retriever::InMemoryVectorStoreBackend;

        let adapter = VectorArchivalMemory::new(InMemoryVectorStoreBackend::new(), FakeEmbedder);
        let inserted_id = adapter.insert("hello", HashMap::new()).await.unwrap();

        let results = adapter.search("hello", 1).await.unwrap();
        assert_eq!(results.len(), 1);
        // The id returned by search must be the real, stable id assigned at
        // insert time, not a rank-derived placeholder.
        assert_eq!(results[0].id, inserted_id);

        // And it must actually work with delete: the fact should be gone
        // afterward.
        assert!(adapter.delete(&results[0].id).await.unwrap());
        assert_eq!(adapter.count().await.unwrap(), 0);
        let remaining = adapter.search("hello", 2).await.unwrap();
        assert!(remaining.is_empty());
    }
}
