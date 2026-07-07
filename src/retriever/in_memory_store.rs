//! In-memory vector store backend implementing [`VectorStore`].
//!
//! This is a simple brute-force implementation for development and testing.
//! For production use, implement [`VectorStore`] for your preferred vector
//! database (Qdrant, Pinecone, Chroma, Weaviate, Milvus, etc.).

use std::collections::HashMap;

use tokio::sync::RwLock;

use crate::error::Result;
use crate::retriever::types::Document;
use crate::retriever::vector_store::{ScoredDocument, VectorStore};

struct StoredEntry {
    embedding: Vec<f32>,
    document: Document,
}

/// Brute-force in-memory vector store for development and testing.
///
/// Compose with [`SimpleKnowledgeBase`](super::SimpleKnowledgeBase) for a
/// complete embedding+search pipeline:
///
/// ```ignore
/// use daimon::retriever::{InMemoryVectorStoreBackend, SimpleKnowledgeBase};
///
/// let store = InMemoryVectorStoreBackend::new();
/// let kb = SimpleKnowledgeBase::new(embedding_model, store);
/// ```
pub struct InMemoryVectorStoreBackend {
    entries: RwLock<HashMap<String, StoredEntry>>,
}

impl InMemoryVectorStoreBackend {
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryVectorStoreBackend {
    fn default() -> Self {
        Self::new()
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}

impl VectorStore for InMemoryVectorStoreBackend {
    async fn upsert(&self, id: &str, embedding: Vec<f32>, document: Document) -> Result<()> {
        self.entries.write().await.insert(
            id.to_string(),
            StoredEntry {
                embedding,
                document,
            },
        );
        Ok(())
    }

    async fn query(&self, embedding: Vec<f32>, top_k: usize) -> Result<Vec<ScoredDocument>> {
        let entries = self.entries.read().await;
        let mut scored: Vec<ScoredDocument> = entries
            .values()
            .map(|entry| {
                let sim = cosine_similarity(&embedding, &entry.embedding) as f64;
                ScoredDocument::new(entry.document.clone(), sim)
            })
            .collect();

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);
        Ok(scored)
    }

    async fn delete(&self, id: &str) -> Result<bool> {
        Ok(self.entries.write().await.remove(id).is_some())
    }

    async fn count(&self) -> Result<usize> {
        Ok(self.entries.read().await.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_upsert_and_query() {
        let store = InMemoryVectorStoreBackend::new();
        store
            .upsert("a", vec![1.0, 0.0, 0.0], Document::new("doc a"))
            .await
            .unwrap();
        store
            .upsert("b", vec![0.0, 1.0, 0.0], Document::new("doc b"))
            .await
            .unwrap();

        let results = store.query(vec![1.0, 0.0, 0.0], 2).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].document.content, "doc a");
        assert!((results[0].score - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_upsert_overwrites() {
        let store = InMemoryVectorStoreBackend::new();
        store
            .upsert("a", vec![1.0, 0.0], Document::new("old"))
            .await
            .unwrap();
        store
            .upsert("a", vec![0.0, 1.0], Document::new("new"))
            .await
            .unwrap();

        assert_eq!(store.count().await.unwrap(), 1);
        let results = store.query(vec![0.0, 1.0], 1).await.unwrap();
        assert_eq!(results[0].document.content, "new");
    }

    #[tokio::test]
    async fn test_delete() {
        let store = InMemoryVectorStoreBackend::new();
        store
            .upsert("a", vec![1.0], Document::new("doc"))
            .await
            .unwrap();
        assert!(store.delete("a").await.unwrap());
        assert!(!store.delete("nonexistent").await.unwrap());
        assert_eq!(store.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_count() {
        let store = InMemoryVectorStoreBackend::new();
        assert_eq!(store.count().await.unwrap(), 0);
        store
            .upsert("a", vec![1.0], Document::new("a"))
            .await
            .unwrap();
        store
            .upsert("b", vec![0.0], Document::new("b"))
            .await
            .unwrap();
        assert_eq!(store.count().await.unwrap(), 2);
    }
}
