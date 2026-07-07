//! RAG retrieval backed by an OpenSearch k-NN [`VectorStore`].
//!
//! Stores a couple of documents (with pre-computed embeddings) in an
//! OpenSearch index using the native k-NN plugin, then queries for the most
//! similar ones.
//!
//! Run against a local OpenSearch cluster with the k-NN plugin enabled:
//!
//! ```sh
//! cargo run --example rag_opensearch --features opensearch
//! ```
//!
//! In CI this is `cargo check`-only (no live cluster), so nothing here
//! requires a running server at compile time — the connection is only
//! established inside `build().await`.

use daimon::prelude::*;

#[tokio::main]
async fn main() -> daimon::Result<()> {
    // Vector dimensionality — must match your embedding model. Kept small
    // here for readability; real models emit 768/1024/1536-dim vectors.
    const DIM: usize = 4;

    // Connect and auto-create the k-NN index on first use.
    let store = OpenSearchVectorStoreBuilder::new("http://localhost:9200", DIM)
        .index("rag_docs")
        .space_type(OpenSearchSpaceType::CosineSimilarity)
        .engine(OpenSearchEngine::Lucene)
        .build()
        .await?;

    // Upsert documents with pre-computed embeddings. In a real pipeline these
    // vectors come from an `EmbeddingModel`; here they are hand-written.
    store
        .upsert(
            "doc-1",
            vec![0.1, 0.2, 0.3, 0.4],
            Document::new("Rust is a systems programming language focused on safety.")
                .with_metadata("source", json!("handbook")),
        )
        .await?;

    store
        .upsert(
            "doc-2",
            vec![0.9, 0.1, 0.0, 0.2],
            Document::new("Tokio is an asynchronous runtime for Rust."),
        )
        .await?;

    // Retrieve the top-k most similar documents to a query embedding. Note:
    // OpenSearch scores are backend-raw and only comparable within one space
    // type — use them for ranking, not as calibrated similarities.
    let query_embedding = vec![0.12, 0.18, 0.28, 0.42];
    let results = store.query(query_embedding, 2).await?;

    println!("Top {} matches:", results.len());
    for (rank, scored) in results.iter().enumerate() {
        println!(
            "{}. (score {:.4}) {}",
            rank + 1,
            scored.score,
            scored.document.content
        );
    }

    Ok(())
}
