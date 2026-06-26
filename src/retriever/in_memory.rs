//! In-memory vector store using brute-force cosine similarity.

use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::Result;
use crate::retriever::traits::Retriever;
use crate::retriever::types::Document;

/// A stored document with its precomputed embedding vector.
struct StoredEntry {
    document: Document,
    embedding: Vec<f32>,
}

/// Brute-force in-memory vector store for development and testing.
///
/// Stores documents with their embeddings and retrieves the most similar ones
/// via cosine similarity. Not suitable for large-scale production use.
pub struct InMemoryVectorStore {
    embedding_model: Arc<dyn crate::model::ErasedEmbeddingModel>,
    entries: RwLock<Vec<StoredEntry>>,
}

impl InMemoryVectorStore {
    pub fn new(embedding_model: Arc<dyn crate::model::ErasedEmbeddingModel>) -> Self {
        Self {
            embedding_model,
            entries: RwLock::new(Vec::new()),
        }
    }

    /// Adds a document, computing its embedding automatically.
    pub async fn add(&self, doc: Document) -> Result<()> {
        let texts = [doc.content.as_str()];
        let embeddings = self.embedding_model.embed_erased(&texts).await?;
        let embedding = embeddings.into_iter().next().unwrap_or_default();

        self.entries.write().await.push(StoredEntry {
            document: doc,
            embedding,
        });
        Ok(())
    }

    /// Adds multiple documents in a single batch.
    pub async fn add_many(&self, docs: Vec<Document>) -> Result<()> {
        let texts: Vec<&str> = docs.iter().map(|d| d.content.as_str()).collect();
        let embeddings = self.embedding_model.embed_erased(&texts).await?;

        let mut entries = self.entries.write().await;
        for (doc, embedding) in docs.into_iter().zip(embeddings) {
            entries.push(StoredEntry {
                document: doc,
                embedding,
            });
        }
        Ok(())
    }

    /// Returns the number of stored documents.
    pub async fn len(&self) -> usize {
        self.entries.read().await.len()
    }

    /// Returns true if no documents are stored.
    pub async fn is_empty(&self) -> bool {
        self.entries.read().await.is_empty()
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

impl Retriever for InMemoryVectorStore {
    async fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<Document>> {
        let texts = [query];
        let query_embeddings = self.embedding_model.embed_erased(&texts).await?;
        let query_vec = query_embeddings.into_iter().next().unwrap_or_default();

        let entries = self.entries.read().await;
        let mut scored: Vec<(f64, &StoredEntry)> = entries
            .iter()
            .map(|entry| {
                let sim = cosine_similarity(&query_vec, &entry.embedding) as f64;
                (sim, entry)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored
            .into_iter()
            .take(top_k)
            .map(|(score, entry)| entry.document.clone().with_score(score))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }
}
