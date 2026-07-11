//! Trait-based plugin system for vector stores.
//!
//! Implement [`VectorStore`] for your storage backend (pgvector, Qdrant,
//! Pinecone, Chroma, Weaviate, Milvus, etc.), then compose with a knowledge
//! base for a complete embedding-backed retrieval pipeline.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::document::{Document, ScoredDocument};
use crate::error::Result;

/// Low-level vector storage interface.
///
/// Implement this trait for your vector database. Operations deal in
/// pre-computed embeddings — combine with an `EmbeddingModel` and knowledge
/// base for automatic embedding computation.
pub trait VectorStore: Send + Sync {
    /// Inserts or updates a document with a pre-computed embedding vector.
    fn upsert(
        &self,
        id: &str,
        embedding: Vec<f32>,
        document: Document,
    ) -> impl Future<Output = Result<()>> + Send;

    /// Inserts or updates a batch of documents with pre-computed embeddings.
    ///
    /// The default implementation calls [`VectorStore::upsert`] once per
    /// item. Backends with a bulk write API (multi-row `INSERT`, `_bulk`
    /// endpoints) should override it to collapse the batch into a single
    /// roundtrip.
    fn upsert_many(
        &self,
        items: Vec<(String, Vec<f32>, Document)>,
    ) -> impl Future<Output = Result<()>> + Send {
        async move {
            for (id, embedding, document) in items {
                self.upsert(&id, embedding, document).await?;
            }
            Ok(())
        }
    }

    /// Queries the store for the `top_k` most similar documents to the
    /// given embedding vector. Returns results sorted by descending similarity.
    fn query(
        &self,
        embedding: Vec<f32>,
        top_k: usize,
    ) -> impl Future<Output = Result<Vec<ScoredDocument>>> + Send;

    /// Deletes a document by ID. Returns `true` if the document was found
    /// and deleted.
    fn delete(&self, id: &str) -> impl Future<Output = Result<bool>> + Send;

    /// Returns the number of documents in the store.
    fn count(&self) -> impl Future<Output = Result<usize>> + Send;
}

/// Object-safe wrapper for [`VectorStore`].
pub trait ErasedVectorStore: Send + Sync {
    fn upsert_erased<'a>(
        &'a self,
        id: &'a str,
        embedding: Vec<f32>,
        document: Document,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    fn upsert_many_erased(
        &self,
        items: Vec<(String, Vec<f32>, Document)>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>>;

    fn query_erased<'a>(
        &'a self,
        embedding: Vec<f32>,
        top_k: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<ScoredDocument>>> + Send + 'a>>;

    fn delete_erased<'a>(
        &'a self,
        id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<bool>> + Send + 'a>>;

    fn count_erased(&self) -> Pin<Box<dyn Future<Output = Result<usize>> + Send + '_>>;
}

impl<T: VectorStore> ErasedVectorStore for T {
    fn upsert_erased<'a>(
        &'a self,
        id: &'a str,
        embedding: Vec<f32>,
        document: Document,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(self.upsert(id, embedding, document))
    }

    fn upsert_many_erased(
        &self,
        items: Vec<(String, Vec<f32>, Document)>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        Box::pin(self.upsert_many(items))
    }

    fn query_erased<'a>(
        &'a self,
        embedding: Vec<f32>,
        top_k: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<ScoredDocument>>> + Send + 'a>> {
        Box::pin(self.query(embedding, top_k))
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

/// Shared ownership of a vector store.
pub type SharedVectorStore = Arc<dyn ErasedVectorStore>;

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    struct CountingStore {
        upserts: AtomicUsize,
    }

    impl VectorStore for CountingStore {
        async fn upsert(&self, _id: &str, _embedding: Vec<f32>, _document: Document) -> Result<()> {
            self.upserts.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn query(&self, _embedding: Vec<f32>, _top_k: usize) -> Result<Vec<ScoredDocument>> {
            Ok(Vec::new())
        }

        async fn delete(&self, _id: &str) -> Result<bool> {
            Ok(false)
        }

        async fn count(&self) -> Result<usize> {
            Ok(self.upserts.load(Ordering::SeqCst))
        }
    }

    #[test]
    fn test_upsert_many_default_delegates_per_item() {
        futures::executor::block_on(async {
            let store = CountingStore {
                upserts: AtomicUsize::new(0),
            };
            let items = (0..3)
                .map(|i| (format!("id-{i}"), vec![0.0], Document::new("doc")))
                .collect();
            store.upsert_many(items).await.unwrap();
            assert_eq!(store.upserts.load(Ordering::SeqCst), 3);

            // The erased wrapper forwards to the same default implementation.
            let shared: SharedVectorStore = Arc::new(store);
            shared
                .upsert_many_erased(vec![("id-3".into(), vec![0.0], Document::new("doc"))])
                .await
                .unwrap();
            assert_eq!(shared.count_erased().await.unwrap(), 4);
        });
    }
}
