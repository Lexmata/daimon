//! Task difficulty scoring for routing decisions.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::error::Result;
use crate::model::types::{ChatRequest, Message, Role};
use crate::model::{Model, SharedModel};

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
    /// Weight scaled by last user message length (saturating at 2000 bytes).
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

/// Difficulty scored by a judge model: the request snapshot is summarized
/// into a classification prompt and the judge's numeric reply is parsed.
///
/// Scoring is fail-soft: a judge error or unparseable reply yields 0.5
/// (middle band) with a warning, so a scorer outage never breaks a run.
pub struct LlmScorer {
    judge: SharedModel,
}

impl LlmScorer {
    /// Uses an owned model as the difficulty judge (typically a cheap,
    /// Small-tier model).
    pub fn new<M: Model + 'static>(judge: M) -> Self {
        Self {
            judge: Arc::new(judge),
        }
    }

    /// Uses a shared model as the difficulty judge.
    pub fn shared(judge: SharedModel) -> Self {
        Self { judge }
    }
}

const JUDGE_SYSTEM: &str = "You are a task-difficulty classifier. Given a \
    conversation snapshot, reply with ONLY a number from 0.0 to 1.0: 0.0 \
    trivial (chit-chat, simple lookup), 0.5 moderate (multi-step reasoning, \
    tool use), 1.0 very hard (deep analysis, proofs, complex code). No \
    explanation.";

/// Difficulty used when the judge fails or replies with no parseable number.
const FALLBACK_DIFFICULTY: f64 = 0.5;

/// Maximum characters of the last user message sent to the judge.
const JUDGE_SNAPSHOT_CHARS: usize = 4000;

impl TaskScorer for LlmScorer {
    async fn score(&self, request: &ChatRequest) -> Result<f64> {
        let last_user = request
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .and_then(|m| m.content.as_deref())
            .unwrap_or("");
        let snapshot: String = last_user.chars().take(JUDGE_SNAPSHOT_CHARS).collect();
        let judge_prompt = format!(
            "Conversation length: {} messages\nTools available: {}\nLast user message:\n{snapshot}",
            request.messages.len(),
            request.tools.len(),
        );
        let judge_request = ChatRequest {
            messages: vec![Message::system(JUDGE_SYSTEM), Message::user(judge_prompt)],
            tools: Vec::new(),
            temperature: Some(0.0),
            max_tokens: Some(16),
        };

        let text = match self.judge.generate_erased(&judge_request).await {
            Ok(response) => response.text().to_string(),
            Err(e) => {
                tracing::warn!(error = %e, "difficulty judge call failed; using fallback 0.5");
                return Ok(FALLBACK_DIFFICULTY);
            }
        };

        match text.split_whitespace().find_map(|tok| {
            tok.trim_matches(|c: char| !c.is_ascii_digit() && c != '.')
                .trim_end_matches('.')
                .parse::<f64>()
                .ok()
        }) {
            Some(d) => Ok(d.clamp(0.0, 1.0)),
            None => {
                tracing::warn!(reply = %text, "judge reply had no parseable number; using fallback 0.5");
                Ok(FALLBACK_DIFFICULTY)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::types::{Message, ToolSpec};

    use crate::model::Model;
    use crate::model::types::{ChatResponse, StopReason};
    use crate::stream::ResponseStream;

    struct JudgeModel {
        reply: &'static str,
        fail: bool,
    }

    impl Model for JudgeModel {
        async fn generate(&self, _request: &ChatRequest) -> Result<ChatResponse> {
            if self.fail {
                return Err(crate::error::DaimonError::Model("judge down".into()));
            }
            Ok(ChatResponse {
                message: Message::assistant(self.reply),
                stop_reason: StopReason::EndTurn,
                usage: None,
            })
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }

        fn model_id(&self) -> &str {
            "judge"
        }
    }

    #[tokio::test]
    async fn llm_scorer_parses_plain_number() {
        let scorer = LlmScorer::new(JudgeModel {
            reply: "0.85",
            fail: false,
        });
        let req = request_with(vec![Message::user("anything")], vec![]);
        assert_eq!(scorer.score(&req).await.unwrap(), 0.85);
    }

    #[tokio::test]
    async fn llm_scorer_parses_number_from_prose() {
        let scorer = LlmScorer::new(JudgeModel {
            reply: "I'd say 0.7.",
            fail: false,
        });
        let req = request_with(vec![Message::user("anything")], vec![]);
        assert_eq!(scorer.score(&req).await.unwrap(), 0.7);
    }

    #[tokio::test]
    async fn llm_scorer_falls_back_on_garbage() {
        let scorer = LlmScorer::new(JudgeModel {
            reply: "hard to say",
            fail: false,
        });
        let req = request_with(vec![Message::user("anything")], vec![]);
        assert_eq!(scorer.score(&req).await.unwrap(), 0.5);
    }

    #[tokio::test]
    async fn llm_scorer_falls_back_on_error() {
        let scorer = LlmScorer::new(JudgeModel {
            reply: "",
            fail: true,
        });
        let req = request_with(vec![Message::user("anything")], vec![]);
        assert_eq!(scorer.score(&req).await.unwrap(), 0.5);
    }

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
