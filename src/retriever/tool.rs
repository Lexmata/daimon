//! Wraps a [`Retriever`] as a [`Tool`] for agent use.

use std::sync::Arc;

use crate::error::Result;
use crate::retriever::traits::ErasedRetriever;
use crate::tool::{Tool, ToolOutput};

/// Exposes a [`Retriever`](super::Retriever) as a callable [`Tool`].
///
/// When the agent invokes this tool with `{"query": "...", "top_k": N}`,
/// it performs retrieval and returns the documents as formatted text.
pub struct RetrieverTool {
    retriever: Arc<dyn ErasedRetriever>,
    name: String,
    description: String,
    default_top_k: usize,
}

impl RetrieverTool {
    /// Creates a new retriever tool.
    pub fn new<R: super::Retriever + 'static>(
        retriever: R,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            retriever: Arc::new(retriever),
            name: name.into(),
            description: description.into(),
            default_top_k: 5,
        }
    }

    /// Creates from a shared retriever.
    pub fn from_shared(
        retriever: Arc<dyn ErasedRetriever>,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            retriever,
            name: name.into(),
            description: description.into(),
            default_top_k: 5,
        }
    }

    /// Sets the default number of results when `top_k` is not specified.
    pub fn with_default_top_k(mut self, top_k: usize) -> Self {
        self.default_top_k = top_k;
        self
    }
}

impl Tool for RetrieverTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": self.default_top_k
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<ToolOutput> {
        let query = input
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let top_k = input
            .get("top_k")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(self.default_top_k);

        let documents = self.retriever.retrieve_erased(query, top_k).await?;

        if documents.is_empty() {
            return Ok(ToolOutput::text("No relevant documents found."));
        }

        let mut output = String::new();
        for (i, doc) in documents.iter().enumerate() {
            output.push_str(&format!("--- Document {} ---\n", i + 1));
            if let Some(score) = doc.score {
                output.push_str(&format!("Score: {score:.4}\n"));
            }
            for (key, value) in &doc.metadata {
                output.push_str(&format!("{key}: {value}\n"));
            }
            output.push_str(&doc.content);
            output.push_str("\n\n");
        }

        Ok(ToolOutput::text(output.trim()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retriever::{Document, Retriever};

    struct FakeRetriever {
        docs: Vec<Document>,
    }

    impl Retriever for FakeRetriever {
        async fn retrieve(&self, _query: &str, top_k: usize) -> Result<Vec<Document>> {
            Ok(self.docs.iter().take(top_k).cloned().collect())
        }
    }

    #[tokio::test]
    async fn test_retriever_tool_basic() {
        let retriever = FakeRetriever {
            docs: vec![
                Document::new("Rust is a systems language").with_score(0.95),
                Document::new("Go is a compiled language").with_score(0.80),
            ],
        };

        let tool = RetrieverTool::new(retriever, "search", "Search knowledge base");
        let input = serde_json::json!({"query": "what is rust?"});
        let output = tool.execute(&input).await.unwrap();

        assert!(!output.is_error);
        assert!(output.content.contains("Rust is a systems language"));
        assert!(output.content.contains("Go is a compiled language"));
        assert!(output.content.contains("0.95"));
    }

    #[tokio::test]
    async fn test_retriever_tool_respects_top_k() {
        let retriever = FakeRetriever {
            docs: vec![
                Document::new("doc1"),
                Document::new("doc2"),
                Document::new("doc3"),
            ],
        };

        let tool = RetrieverTool::new(retriever, "search", "Search");
        let input = serde_json::json!({"query": "test", "top_k": 1});
        let output = tool.execute(&input).await.unwrap();

        assert!(output.content.contains("doc1"));
        assert!(!output.content.contains("doc2"));
    }

    #[tokio::test]
    async fn test_retriever_tool_empty_results() {
        let retriever = FakeRetriever { docs: vec![] };
        let tool = RetrieverTool::new(retriever, "search", "Search");
        let input = serde_json::json!({"query": "nothing"});
        let output = tool.execute(&input).await.unwrap();

        assert!(output.content.contains("No relevant documents"));
    }

    #[tokio::test]
    async fn test_retriever_tool_metadata() {
        let doc = Document::new("content")
            .with_metadata("source", serde_json::json!("wiki"))
            .with_score(0.9);
        let retriever = FakeRetriever { docs: vec![doc] };

        let tool = RetrieverTool::new(retriever, "search", "Search");
        let input = serde_json::json!({"query": "test"});
        let output = tool.execute(&input).await.unwrap();

        assert!(output.content.contains("source"));
        assert!(output.content.contains("wiki"));
    }

    #[test]
    fn test_retriever_tool_schema() {
        let retriever = FakeRetriever { docs: vec![] };
        let tool = RetrieverTool::new(retriever, "search", "Search knowledge");
        let schema = tool.parameters_schema();

        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["query"].is_object());
        assert_eq!(schema["required"][0], "query");
    }
}
