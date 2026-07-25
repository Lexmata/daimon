//! Task difficulty scoring for routing decisions.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::error::Result;
use crate::model::types::{ChatRequest, Role};

/// Scores the difficulty of a task (one ReAct iteration's request) on a
/// 0.0 (trivial) to 1.0 (hardest) scale.
///
/// Implement this trait to plug custom scoring into a
/// [`ModelRouter`](crate::routing::ModelRouter). Object-safe via
/// [`ErasedTaskScorer`].
pub trait TaskScorer: Send + Sync {
    /// Scores the difficulty of the given request.
    fn score(&self, request: &ChatRequest) -> impl Future<Output = Result<f64>> + Send;
}

/// Object-safe wrapper for [`TaskScorer`].
pub trait ErasedTaskScorer: Send + Sync {
    /// Object-safe version of [`TaskScorer::score`].
    fn score_erased<'a>(
        &'a self,
        request: &'a ChatRequest,
    ) -> Pin<Box<dyn Future<Output = Result<f64>> + Send + 'a>>;
}

impl<T: TaskScorer> ErasedTaskScorer for T {
    fn score_erased<'a>(
        &'a self,
        request: &'a ChatRequest,
    ) -> Pin<Box<dyn Future<Output = Result<f64>> + Send + 'a>> {
        Box::pin(self.score(request))
    }
}

/// Shared ownership of a scorer.
pub type SharedTaskScorer = Arc<dyn ErasedTaskScorer>;

/// Keyword signals of harder tasks, matched case-insensitively against the
/// last user message.
const KEYWORDS: &[&str] = &[
    "step by step",
    "step-by-step",
    "prove",
    "analyze",
    "analyse",
    "refactor",
    "debug",
    "architecture",
    "optimize",
    "compare and contrast",
    "tradeoff",
    "trade-off",
    "```",
];

/// Deterministic, zero-cost scorer driven by weighted request signals.
///
/// Signals: a base score, keyword hits in the last user message, last user
/// message length, tool activity in the conversation (tool calls or tool
/// results — tool-use iterations are harder), and the number of tool specs
/// in play. Weights are public and tunable; defaults are calibrated so a
/// one-word greeting scores ~0.1 and a long multi-turn tool conversation
/// with analytical keywords approaches 1.0.
#[derive(Debug, Clone)]
pub struct HeuristicScorer {
    /// Score floor for any request.
    pub base: f64,
    /// Bonus per distinct keyword hit.
    pub keyword_weight: f64,
    /// Cap on total keyword bonus.
    pub max_keyword_bonus: f64,
    /// Weight scaled by last user message length (saturating at 2000 chars).
    pub length_weight: f64,
    /// Flat bonus when the conversation contains tool calls or tool results.
    pub tool_history_weight: f64,
    /// Weight scaled by tool-spec count (saturating at 10 tools).
    pub tool_spec_weight: f64,
}

impl Default for HeuristicScorer {
    fn default() -> Self {
        Self {
            base: 0.1,
            keyword_weight: 0.08,
            max_keyword_bonus: 0.4,
            length_weight: 0.25,
            tool_history_weight: 0.2,
            tool_spec_weight: 0.15,
        }
    }
}

impl TaskScorer for HeuristicScorer {
    async fn score(&self, request: &ChatRequest) -> Result<f64> {
        let last_user = request
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .and_then(|m| m.content.as_deref())
            .unwrap_or("");
        let lower = last_user.to_lowercase();

        let keyword_hits = KEYWORDS.iter().filter(|k| lower.contains(**k)).count();
        let keyword_bonus = (keyword_hits as f64 * self.keyword_weight).min(self.max_keyword_bonus);
        let length_score = (last_user.len() as f64 / 2000.0).min(1.0) * self.length_weight;
        let has_tool_activity = request
            .messages
            .iter()
            .any(|m| m.role == Role::Tool || !m.tool_calls.is_empty());
        let tool_history = if has_tool_activity {
            self.tool_history_weight
        } else {
            0.0
        };
        let tool_specs = (request.tools.len() as f64 / 10.0).min(1.0) * self.tool_spec_weight;

        let score = self.base + keyword_bonus + length_score + tool_history + tool_specs;
        Ok(score.clamp(0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::types::{Message, ToolSpec};

    fn request_with(messages: Vec<Message>, tools: Vec<ToolSpec>) -> ChatRequest {
        ChatRequest {
            messages,
            tools,
            temperature: None,
            max_tokens: None,
        }
    }

    #[tokio::test]
    async fn trivial_prompt_scores_low() {
        let scorer = HeuristicScorer::default();
        let req = request_with(vec![Message::user("hi")], vec![]);
        let score = scorer.score(&req).await.unwrap();
        assert!(score < 0.2, "trivial prompt scored {score}");
    }

    #[tokio::test]
    async fn long_prompt_scores_higher_than_short() {
        let scorer = HeuristicScorer::default();
        let short = scorer
            .score(&request_with(vec![Message::user("hi")], vec![]))
            .await
            .unwrap();
        let long = scorer
            .score(&request_with(vec![Message::user("x".repeat(3000))], vec![]))
            .await
            .unwrap();
        assert!(long > short);
    }

    #[tokio::test]
    async fn keywords_raise_score() {
        let scorer = HeuristicScorer::default();
        let plain = scorer
            .score(&request_with(
                vec![Message::user("tell me about rust")],
                vec![],
            ))
            .await
            .unwrap();
        let analytical = scorer
            .score(&request_with(
                vec![Message::user(
                    "analyze and compare and contrast these designs step by step",
                )],
                vec![],
            ))
            .await
            .unwrap();
        assert!(analytical > plain);
    }

    #[tokio::test]
    async fn tool_activity_raises_score() {
        let scorer = HeuristicScorer::default();
        let no_tools = scorer
            .score(&request_with(vec![Message::user("hi")], vec![]))
            .await
            .unwrap();
        let with_tools = scorer
            .score(&request_with(
                vec![Message::user("hi"), Message::tool_result("call_1", "42")],
                vec![],
            ))
            .await
            .unwrap();
        assert!(with_tools > no_tools);
    }

    #[tokio::test]
    async fn scoring_is_deterministic() {
        let scorer = HeuristicScorer::default();
        let req = request_with(vec![Message::user("analyze this")], vec![]);
        let a = scorer.score(&req).await.unwrap();
        let b = scorer.score(&req).await.unwrap();
        assert_eq!(a, b);
    }

    #[tokio::test]
    async fn score_is_clamped_to_one() {
        let scorer = HeuristicScorer {
            base: 0.9,
            ..Default::default()
        };
        let req = request_with(
            vec![Message::user("analyze prove debug ".repeat(500))],
            vec![],
        );
        let score = scorer.score(&req).await.unwrap();
        assert!(score <= 1.0);
    }
}
