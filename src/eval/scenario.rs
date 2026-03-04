use std::sync::Arc;

use crate::eval::scoring::Scorer;
use crate::model::{SharedEmbeddingModel, SharedModel};

/// A test scenario for evaluating agent behaviour.
#[derive(Debug, Clone)]
pub struct EvalScenario {
    /// The user input to send to the agent.
    pub input: String,
    /// Scoring strategies that determine pass/fail.
    pub scorers: Vec<Scorer>,
    /// Maximum iterations allowed for this scenario.
    pub max_iterations: Option<usize>,
    /// Maximum cost in USD allowed for this scenario.
    pub max_cost: Option<f64>,
}

impl EvalScenario {
    /// Creates a scenario with only an input prompt.
    pub fn new(input: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            scorers: Vec::new(),
            max_iterations: None,
            max_cost: None,
        }
    }

    /// Adds a scorer that checks if the output contains the given substring.
    pub fn expect_contains(mut self, substring: impl Into<String>) -> Self {
        self.scorers.push(Scorer::Contains(substring.into()));
        self
    }

    /// Adds a scorer that checks for exact match.
    pub fn expect_exact(mut self, expected: impl Into<String>) -> Self {
        self.scorers.push(Scorer::ExactMatch(expected.into()));
        self
    }

    /// Adds a scorer that checks against a regex pattern.
    pub fn expect_regex(mut self, pattern: impl Into<String>) -> Self {
        self.scorers.push(Scorer::Regex(pattern.into()));
        self
    }

    /// Adds a custom scoring function.
    pub fn expect_custom<F>(mut self, scorer: F) -> Self
    where
        F: Fn(&str) -> bool + Send + Sync + 'static,
    {
        self.scorers.push(Scorer::Custom(Arc::new(scorer)));
        self
    }

    /// Adds a semantic similarity scorer that passes if cosine similarity
    /// between the output and expected text exceeds the threshold.
    pub fn expect_semantic(
        mut self,
        expected: impl Into<String>,
        embedding_model: SharedEmbeddingModel,
        threshold: f64,
    ) -> Self {
        self.scorers
            .push(Scorer::semantic(expected, embedding_model, threshold));
        self
    }

    /// Adds an LLM-as-judge scorer that evaluates output against a rubric.
    pub fn expect_llm_judge(
        mut self,
        rubric: impl Into<String>,
        model: SharedModel,
    ) -> Self {
        self.scorers.push(Scorer::llm_judge(rubric, model));
        self
    }

    /// Sets the maximum iterations for this scenario.
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = Some(max);
        self
    }

    /// Sets the maximum cost for this scenario.
    pub fn with_max_cost(mut self, max: f64) -> Self {
        self.max_cost = Some(max);
        self
    }
}
