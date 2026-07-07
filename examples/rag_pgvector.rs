//! RAG retrieval backed by a pgvector [`VectorStore`].
//!
//! Stores a couple of documents (with pre-computed embeddings) in PostgreSQL
//! via the pgvector extension, then queries for the most similar ones.
//!
//! Run against a Postgres instance that has the `pgvector` extension:
//!
//! ```sh
//! cargo run --example rag_pgvector --features pgvector
//! ```
//!
//! In CI this is `cargo check`-only (no live database), so nothing here
//! requires a running server at compile time — the connection is only
//! established inside `build().await`.

use daimon::prelude::*;

#[tokio::main]
async fn main() -> daimon::Result<()> {
    // Vector dimensionality — must match your embedding model. Kept small
    // here for readability; real models emit 768/1024/1536-dim vectors.
    const DIM: usize = 4;

    // Connect and auto-create the table + HNSW index on first use.
    let store =
        PgVectorStoreBuilder::new("postgresql://postgres:postgres@localhost:5432/daimon", DIM)
            .table("rag_docs")
            .distance_metric(DistanceMetric::Cosine)
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

    // Retrieve the top-k most similar documents to a query embedding.
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
