//! Document types shared across the Daimon framework.

use std::collections::HashMap;

/// A retrieved document fragment with optional metadata and relevance score.
#[derive(Debug, Clone)]
pub struct Document {
    /// The text content of the document.
    pub content: String,
    /// Arbitrary key-value metadata (e.g., source URL, page number, title).
    pub metadata: HashMap<String, serde_json::Value>,
    /// Relevance score assigned by the retrieval backend (higher = more relevant).
    /// `None` if the backend does not provide scores.
    pub score: Option<f64>,
}

impl Document {
    /// Creates a document with only text content.
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            metadata: HashMap::new(),
            score: None,
        }
    }

    /// Adds a metadata entry.
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Sets the relevance score.
    pub fn with_score(mut self, score: f64) -> Self {
        self.score = Some(score);
        self
    }
}

/// A document paired with a similarity score from a vector query.
#[derive(Debug, Clone)]
pub struct ScoredDocument {
    pub document: Document,
    pub score: f64,
}

impl ScoredDocument {
    pub fn new(document: Document, score: f64) -> Self {
        Self { document, score }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_new() {
        let doc = Document::new("hello world");
        assert_eq!(doc.content, "hello world");
        assert!(doc.metadata.is_empty());
        assert!(doc.score.is_none());
    }

    #[test]
    fn test_document_with_metadata_and_score() {
        let doc = Document::new("text")
            .with_metadata("source", serde_json::json!("wiki"))
            .with_score(0.95);
        assert_eq!(doc.metadata["source"], "wiki");
        assert_eq!(doc.score, Some(0.95));
    }

    #[test]
    fn test_scored_document() {
        let doc = Document::new("content");
        let scored = ScoredDocument::new(doc, 0.87);
        assert_eq!(scored.document.content, "content");
        assert_eq!(scored.score, 0.87);
    }
}
