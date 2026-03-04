use std::sync::Arc;

use crate::model::{SharedEmbeddingModel, SharedModel};
use crate::model::types::{ChatRequest, Message};

/// A scoring strategy that evaluates agent output.
pub enum Scorer {
    /// Output must exactly match the expected string.
    ExactMatch(String),
    /// Output must contain the given substring.
    Contains(String),
    /// Output must match the regex pattern.
    Regex(String),
    /// Custom scoring function returning true for pass.
    Custom(Arc<dyn Fn(&str) -> bool + Send + Sync>),
    /// Cosine similarity between output and expected text must exceed threshold.
    SemanticSimilarity {
        expected: String,
        embedding_model: SharedEmbeddingModel,
        threshold: f64,
    },
    /// An LLM grades the output against a rubric.
    LlmJudge {
        rubric: String,
        model: SharedModel,
    },
}

impl Scorer {
    /// Creates a semantic similarity scorer.
    pub fn semantic(
        expected: impl Into<String>,
        embedding_model: SharedEmbeddingModel,
        threshold: f64,
    ) -> Self {
        Scorer::SemanticSimilarity {
            expected: expected.into(),
            embedding_model,
            threshold,
        }
    }

    /// Creates an LLM-as-judge scorer.
    pub fn llm_judge(rubric: impl Into<String>, model: SharedModel) -> Self {
        Scorer::LlmJudge {
            rubric: rubric.into(),
            model,
        }
    }

    /// Evaluates the output against this scorer. Returns true if it passes.
    pub async fn evaluate(&self, output: &str) -> bool {
        match self {
            Scorer::ExactMatch(expected) => output == expected,
            Scorer::Contains(substring) => output.contains(substring.as_str()),
            Scorer::Regex(pattern) => regex_lite::Regex::new(pattern)
                .map(|re| re.is_match(output))
                .unwrap_or(false),
            Scorer::Custom(f) => f(output),
            Scorer::SemanticSimilarity {
                expected,
                embedding_model,
                threshold,
            } => {
                let texts = [expected.as_str(), output];
                match embedding_model.embed_erased(&texts).await {
                    Ok(embeddings) if embeddings.len() == 2 => {
                        let sim = cosine_similarity(&embeddings[0], &embeddings[1]);
                        sim as f64 >= *threshold
                    }
                    _ => false,
                }
            }
            Scorer::LlmJudge { rubric, model } => {
                let prompt = format!(
                    "You are an evaluation judge. Grade the following output against the rubric.\n\n\
                     Rubric: {rubric}\n\n\
                     Output to evaluate:\n{output}\n\n\
                     Respond with EXACTLY one word: PASS or FAIL."
                );
                let request = ChatRequest::new(vec![Message::user(&prompt)]);
                match model.generate_erased(&request).await {
                    Ok(response) => response.text().trim().to_uppercase().starts_with("PASS"),
                    Err(_) => false,
                }
            }
        }
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

impl std::fmt::Debug for Scorer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Scorer::ExactMatch(s) => write!(f, "ExactMatch({s:?})"),
            Scorer::Contains(s) => write!(f, "Contains({s:?})"),
            Scorer::Regex(s) => write!(f, "Regex({s:?})"),
            Scorer::Custom(_) => write!(f, "Custom(...)"),
            Scorer::SemanticSimilarity { expected, threshold, .. } => {
                write!(f, "SemanticSimilarity({expected:?}, threshold={threshold})")
            }
            Scorer::LlmJudge { rubric, .. } => write!(f, "LlmJudge({rubric:?})"),
        }
    }
}

impl Clone for Scorer {
    fn clone(&self) -> Self {
        match self {
            Scorer::ExactMatch(s) => Scorer::ExactMatch(s.clone()),
            Scorer::Contains(s) => Scorer::Contains(s.clone()),
            Scorer::Regex(s) => Scorer::Regex(s.clone()),
            Scorer::Custom(f) => Scorer::Custom(Arc::clone(f)),
            Scorer::SemanticSimilarity { expected, embedding_model, threshold } => {
                Scorer::SemanticSimilarity {
                    expected: expected.clone(),
                    embedding_model: Arc::clone(embedding_model),
                    threshold: *threshold,
                }
            }
            Scorer::LlmJudge { rubric, model } => Scorer::LlmJudge {
                rubric: rubric.clone(),
                model: Arc::clone(model),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_exact_match() {
        assert!(Scorer::ExactMatch("hello".into()).evaluate("hello").await);
        assert!(!Scorer::ExactMatch("hello".into()).evaluate("Hello").await);
    }

    #[tokio::test]
    async fn test_contains() {
        assert!(Scorer::Contains("world".into()).evaluate("hello world").await);
        assert!(!Scorer::Contains("xyz".into()).evaluate("hello world").await);
    }

    #[tokio::test]
    async fn test_regex() {
        assert!(Scorer::Regex(r"\d+".into()).evaluate("answer is 42").await);
        assert!(!Scorer::Regex(r"^\d+$".into()).evaluate("answer is 42").await);
    }

    #[tokio::test]
    async fn test_custom() {
        let scorer = Scorer::Custom(Arc::new(|s| s.len() > 5));
        assert!(scorer.evaluate("long enough").await);
        assert!(!scorer.evaluate("short").await);
    }
}
