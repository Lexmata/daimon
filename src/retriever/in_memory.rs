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
    /// L2 norm of `embedding`, precomputed at insert time so each query pays
    /// only a dot product per entry.
    norm: f32,
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

        let norm = l2_norm(&embedding);
        self.entries.write().await.push(StoredEntry {
            document: doc,
            embedding,
            norm,
        });
        Ok(())
    }

    /// Adds multiple documents in a single batch.
    pub async fn add_many(&self, docs: Vec<Document>) -> Result<()> {
        let texts: Vec<&str> = docs.iter().map(|d| d.content.as_str()).collect();
        let embeddings = self.embedding_model.embed_erased(&texts).await?;

        let mut entries = self.entries.write().await;
        for (doc, embedding) in docs.into_iter().zip(embeddings) {
            let norm = l2_norm(&embedding);
            entries.push(StoredEntry {
                document: doc,
                embedding,
                norm,
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

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[cfg(test)]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let denom = l2_norm(a) * l2_norm(b);
    if denom == 0.0 { 0.0 } else { dot(a, b) / denom }
}

impl Retriever for InMemoryVectorStore {
    async fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<Document>> {
        let texts = [query];
        let query_embeddings = self.embedding_model.embed_erased(&texts).await?;
        let query_vec = query_embeddings.into_iter().next().unwrap_or_default();
        let query_norm = l2_norm(&query_vec);

        let entries = self.entries.read().await;
        let mut scored: Vec<(f64, &StoredEntry)> = entries
            .iter()
            .map(|entry| {
                // Entry norms are precomputed at insert; only the dot product
                // is paid per entry. Zero norms and dimension mismatches score
                // 0.0, matching the previous full cosine computation.
                let denom = query_norm * entry.norm;
                let sim = if denom == 0.0 || query_vec.len() != entry.embedding.len() {
                    0.0
                } else {
                    f64::from(dot(&query_vec, &entry.embedding) / denom)
                };
                (sim, entry)
            })
            .collect();

        let k = top_k.min(scored.len());
        if k == 0 {
            return Ok(Vec::new());
        }
        if k < scored.len() {
            // Partition the top k to the front in O(n), then sort only that
            // prefix — O(n + k log k) instead of sorting every entry. Ties may
            // resolve differently than a full sort, as with any unstable
            // selection.
            scored.select_nth_unstable_by(k - 1, |a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(k);
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored
            .into_iter()
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
