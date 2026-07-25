//! The model router: scores difficulty, selects the cheapest competent
//! registration, and escalates tiers on failure.

use std::sync::Arc;

use crate::cost::CostModel;
use crate::error::{DaimonError, Result};
use crate::model::types::ChatRequest;
use crate::model::{Model, SharedModel};

use super::registry::ModelRegistration;
use super::scorer::{HeuristicScorer, SharedTaskScorer};
use super::tier::ModelTier;

/// Difficulty→tier mapping. Difficulty below `small_below` routes Small,
/// below `medium_below` routes Medium, otherwise Large.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TierBands {
    /// Exclusive upper bound of the Small band.
    pub small_below: f64,
    /// Exclusive upper bound of the Medium band.
    pub medium_below: f64,
}

impl Default for TierBands {
    fn default() -> Self {
        Self {
            small_below: 0.33,
            medium_below: 0.66,
        }
    }
}

impl TierBands {
    /// Maps a difficulty score to the tier required to handle it.
    pub fn tier_for(&self, difficulty: f64) -> ModelTier {
        if difficulty < self.small_below {
            ModelTier::Small
        } else if difficulty < self.medium_below {
            ModelTier::Medium
        } else {
            ModelTier::Large
        }
    }
}

/// Plain-data record of one routing decision. Stored on
/// [`AgentResponse`](crate::agent::AgentResponse) and passed to the
/// `on_route_decision` hook.
#[derive(Debug, Clone)]
pub struct RouteDecision {
    /// 1-based ReAct iteration this decision served.
    pub iteration: usize,
    /// Scored difficulty that drove the decision.
    pub difficulty: f64,
    /// Tier the difficulty mapped to.
    pub required_tier: ModelTier,
    /// Tier actually selected (higher than required on fall-up or
    /// escalation; lower only in best-effort fallback).
    pub selected_tier: ModelTier,
    /// `model_id()` of the selected model.
    pub selected_model_id: String,
    /// Set when this decision is an escalation retry: the tier that failed.
    pub escalated_from: Option<ModelTier>,
}

/// A selected model plus the decision record.
pub struct RoutedModel {
    /// The model to call.
    pub handle: SharedModel,
    /// The decision record.
    pub decision: RouteDecision,
}

/// Routes model calls to the cheapest competent registration.
///
/// Construct via [`ModelRouter::builder`]. Cloning is cheap (`Arc`-backed).
#[derive(Clone)]
pub struct ModelRouter {
    registrations: Arc<Vec<ModelRegistration>>,
    scorer: SharedTaskScorer,
    cost_fallback: Option<Arc<dyn CostModel>>,
    bands: TierBands,
}

impl ModelRouter {
    /// Starts building a router.
    pub fn builder() -> ModelRouterBuilder {
        ModelRouterBuilder::default()
    }

    /// Scores the request and selects the cheapest adequate registration.
    ///
    /// A scorer failure degrades to difficulty 0.5 with a warning rather
    /// than failing the run.
    pub async fn route(&self, iteration: usize, request: &ChatRequest) -> Result<RoutedModel> {
        let difficulty = match self.scorer.score_erased(request).await {
            Ok(d) => d.clamp(0.0, 1.0),
            Err(e) => {
                tracing::warn!(error = %e, "task scorer failed; using fallback 0.5");
                0.5
            }
        };
        let required_tier = self.bands.tier_for(difficulty);
        self.select(iteration, difficulty, required_tier, None)
    }

    /// Selects the cheapest registration in the next populated tier strictly
    /// above the failed decision's tier. `None` when no higher tier is
    /// populated (escalation exhausted).
    pub(crate) fn escalate(&self, decision: &RouteDecision) -> Option<RoutedModel> {
        let next_tier = [ModelTier::Small, ModelTier::Medium, ModelTier::Large]
            .into_iter()
            .filter(|t| *t > decision.selected_tier)
            .find(|t| self.registrations.iter().any(|r| r.tier == *t))?;
        Some(
            self.select(
                decision.iteration,
                decision.difficulty,
                next_tier,
                Some(decision.selected_tier),
            )
            .expect("escalation tier is populated"),
        )
    }

    /// Core selection: smallest populated tier ≥ `required_tier` (best-effort
    /// highest populated tier when none qualifies), then cheapest by cost
    /// ordering key within the tier, ties broken by registration order.
    fn select(
        &self,
        iteration: usize,
        difficulty: f64,
        required_tier: ModelTier,
        escalated_from: Option<ModelTier>,
    ) -> Result<RoutedModel> {
        let tier = [ModelTier::Small, ModelTier::Medium, ModelTier::Large]
            .into_iter()
            .filter(|t| *t >= required_tier)
            .find(|t| self.registrations.iter().any(|r| r.tier == *t));

        let tier = match tier {
            Some(t) => t,
            None => {
                let highest = self
                    .registrations
                    .iter()
                    .map(|r| r.tier)
                    .max()
                    .ok_or_else(|| {
                        DaimonError::Builder("model router has no registrations".into())
                    })?;
                tracing::warn!(
                    ?required_tier,
                    ?highest,
                    "no registration at or above required tier; using highest populated tier"
                );
                highest
            }
        };

        let reg = self
            .registrations
            .iter()
            .enumerate()
            .filter(|(_, r)| r.tier == tier)
            .min_by(|(a_idx, a), (b_idx, b)| {
                let ka = a.effective_cost(self.cost_fallback.as_ref()).ordering_key();
                let kb = b.effective_cost(self.cost_fallback.as_ref()).ordering_key();
                ka.partial_cmp(&kb)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a_idx.cmp(b_idx))
            })
            .expect("selected tier is populated")
            .1;

        let decision = RouteDecision {
            iteration,
            difficulty,
            required_tier,
            selected_tier: reg.tier,
            selected_model_id: reg.model.model_id_erased().to_string(),
            escalated_from,
        };
        tracing::debug!(
            iteration,
            difficulty,
            ?required_tier,
            selected_tier = ?decision.selected_tier,
            selected_model_id = %decision.selected_model_id,
            "route selected model"
        );

        Ok(RoutedModel {
            handle: reg.model.clone(),
            decision,
        })
    }
}

/// Builder for [`ModelRouter`].
#[derive(Default)]
pub struct ModelRouterBuilder {
    registrations: Vec<ModelRegistration>,
    scorer: Option<SharedTaskScorer>,
    cost_fallback: Option<Arc<dyn CostModel>>,
    bands: TierBands,
}

impl ModelRouterBuilder {
    /// Registers an owned model at a tier.
    pub fn register<M: Model + 'static>(mut self, tier: ModelTier, model: M) -> Self {
        self.registrations.push(ModelRegistration::new(tier, model));
        self
    }

    /// Registers an owned model at a tier with explicit cost.
    pub fn register_with_cost<M: Model + 'static>(
        mut self,
        tier: ModelTier,
        model: M,
        cost: super::registry::ModelCost,
    ) -> Self {
        self.registrations
            .push(ModelRegistration::new(tier, model).with_cost(cost));
        self
    }

    /// Registers a shared model at a tier.
    pub fn register_shared(mut self, tier: ModelTier, model: SharedModel) -> Self {
        self.registrations
            .push(ModelRegistration::shared(tier, model));
        self
    }

    /// Sets the difficulty scorer. Defaults to [`HeuristicScorer`].
    pub fn scorer<S: super::scorer::TaskScorer + 'static>(mut self, scorer: S) -> Self {
        self.scorer = Some(Arc::new(scorer));
        self
    }

    /// Sets a pre-built shared scorer.
    pub fn shared_scorer(mut self, scorer: SharedTaskScorer) -> Self {
        self.scorer = Some(scorer);
        self
    }

    /// Sets the [`CostModel`] used to price registrations without explicit
    /// cost.
    pub fn cost_model<C: CostModel + 'static>(mut self, cost_model: C) -> Self {
        self.cost_fallback = Some(Arc::new(cost_model));
        self
    }

    /// Overrides the difficulty→tier bands.
    pub fn bands(mut self, bands: TierBands) -> Self {
        self.bands = bands;
        self
    }

    /// Builds the router. Fails when no models were registered.
    pub fn build(self) -> Result<ModelRouter> {
        if self.registrations.is_empty() {
            return Err(DaimonError::Builder(
                "model router requires at least one registration".into(),
            ));
        }
        Ok(ModelRouter {
            registrations: Arc::new(self.registrations),
            scorer: self
                .scorer
                .unwrap_or_else(|| Arc::new(HeuristicScorer::default())),
            cost_fallback: self.cost_fallback,
            bands: self.bands,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::TokenDirection;
    use crate::model::types::{ChatResponse, Message, StopReason};
    use crate::routing::{ModelCost, TaskScorer};
    use crate::stream::ResponseStream;

    struct StubModel {
        id: &'static str,
    }

    impl Model for StubModel {
        async fn generate(&self, _request: &ChatRequest) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message::assistant(format!("from {}", self.id)),
                stop_reason: StopReason::EndTurn,
                usage: None,
            })
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }

        fn model_id(&self) -> &str {
            self.id
        }
    }

    struct StaticScorer(f64);

    impl TaskScorer for StaticScorer {
        async fn score(&self, _request: &ChatRequest) -> Result<f64> {
            Ok(self.0)
        }
    }

    struct FailingScorer;

    impl TaskScorer for FailingScorer {
        async fn score(&self, _request: &ChatRequest) -> Result<f64> {
            Err(DaimonError::Model("scorer down".into()))
        }
    }

    fn cost(input: f64, output: f64) -> ModelCost {
        ModelCost {
            input_per_token: input,
            output_per_token: output,
        }
    }

    fn empty_request() -> ChatRequest {
        ChatRequest {
            messages: vec![Message::user("test")],
            tools: vec![],
            temperature: None,
            max_tokens: None,
        }
    }

    #[test]
    fn tier_bands_edges() {
        let bands = TierBands::default();
        assert_eq!(bands.tier_for(0.32), ModelTier::Small);
        assert_eq!(bands.tier_for(0.33), ModelTier::Medium);
        assert_eq!(bands.tier_for(0.65), ModelTier::Medium);
        assert_eq!(bands.tier_for(0.66), ModelTier::Large);
    }

    #[test]
    fn empty_router_fails_build() {
        let result = ModelRouter::builder().build();
        assert!(matches!(result, Err(DaimonError::Builder(_))));
    }

    #[tokio::test]
    async fn selects_cheapest_in_adequate_tier() {
        let router = ModelRouter::builder()
            .register_with_cost(
                ModelTier::Small,
                StubModel {
                    id: "small-expensive",
                },
                cost(9e-6, 9e-6),
            )
            .register_with_cost(
                ModelTier::Small,
                StubModel { id: "small-cheap" },
                cost(1e-6, 1e-6),
            )
            .register_with_cost(
                ModelTier::Large,
                StubModel { id: "large" },
                cost(50e-6, 50e-6),
            )
            .scorer(StaticScorer(0.1))
            .build()
            .unwrap();
        let routed = router.route(1, &empty_request()).await.unwrap();
        assert_eq!(routed.decision.selected_model_id, "small-cheap");
        assert_eq!(routed.decision.selected_tier, ModelTier::Small);
        assert_eq!(routed.decision.escalated_from, None);
    }

    #[tokio::test]
    async fn falls_up_when_tier_unpopulated() {
        let router = ModelRouter::builder()
            .register_with_cost(
                ModelTier::Medium,
                StubModel { id: "medium" },
                cost(1e-6, 1e-6),
            )
            .scorer(StaticScorer(0.1))
            .build()
            .unwrap();
        let routed = router.route(1, &empty_request()).await.unwrap();
        assert_eq!(routed.decision.selected_tier, ModelTier::Medium);
    }

    #[tokio::test]
    async fn best_effort_when_required_tier_exceeds_all() {
        let router = ModelRouter::builder()
            .register_with_cost(
                ModelTier::Small,
                StubModel { id: "small" },
                cost(1e-6, 1e-6),
            )
            .scorer(StaticScorer(0.95))
            .build()
            .unwrap();
        let routed = router.route(1, &empty_request()).await.unwrap();
        assert_eq!(routed.decision.required_tier, ModelTier::Large);
        assert_eq!(routed.decision.selected_tier, ModelTier::Small);
    }

    #[tokio::test]
    async fn ties_break_by_registration_order() {
        let router = ModelRouter::builder()
            .register_with_cost(
                ModelTier::Small,
                StubModel { id: "first" },
                cost(1e-6, 1e-6),
            )
            .register_with_cost(
                ModelTier::Small,
                StubModel { id: "second" },
                cost(1e-6, 1e-6),
            )
            .scorer(StaticScorer(0.1))
            .build()
            .unwrap();
        let routed = router.route(1, &empty_request()).await.unwrap();
        assert_eq!(routed.decision.selected_model_id, "first");
    }

    #[tokio::test]
    async fn explicit_cost_beats_fallback_for_ordering() {
        struct FallbackPricing;
        impl CostModel for FallbackPricing {
            fn cost_per_token(&self, _model_id: &str, _direction: TokenDirection) -> f64 {
                100e-6 // fallback would make both equally expensive
            }
        }
        let router = ModelRouter::builder()
            .register(
                ModelTier::Small,
                StubModel {
                    id: "fallback-priced",
                },
            )
            .register_with_cost(
                ModelTier::Small,
                StubModel {
                    id: "explicit-cheap",
                },
                cost(1e-6, 1e-6),
            )
            .cost_model(FallbackPricing)
            .scorer(StaticScorer(0.1))
            .build()
            .unwrap();
        let routed = router.route(1, &empty_request()).await.unwrap();
        assert_eq!(routed.decision.selected_model_id, "explicit-cheap");
    }

    #[tokio::test]
    async fn scorer_failure_degrades_to_default_difficulty() {
        let router = ModelRouter::builder()
            .register_with_cost(
                ModelTier::Small,
                StubModel { id: "small" },
                cost(1e-6, 1e-6),
            )
            .register_with_cost(
                ModelTier::Medium,
                StubModel { id: "medium" },
                cost(2e-6, 2e-6),
            )
            .scorer(FailingScorer)
            .build()
            .unwrap();
        let routed = router.route(1, &empty_request()).await.unwrap();
        // A scorer outage must not fail the run: difficulty falls back to
        // 0.5, which maps to Medium under the default bands.
        assert_eq!(routed.decision.difficulty, 0.5);
        assert_eq!(routed.decision.required_tier, ModelTier::Medium);
        assert_eq!(routed.decision.selected_tier, ModelTier::Medium);
        assert_eq!(routed.decision.selected_model_id, "medium");
    }

    #[tokio::test]
    async fn escalate_moves_to_next_populated_tier() {
        let router = ModelRouter::builder()
            .register_with_cost(
                ModelTier::Small,
                StubModel { id: "small" },
                cost(1e-6, 1e-6),
            )
            .register_with_cost(
                ModelTier::Large,
                StubModel { id: "large" },
                cost(50e-6, 50e-6),
            )
            .scorer(StaticScorer(0.1))
            .build()
            .unwrap();
        let routed = router.route(1, &empty_request()).await.unwrap();
        let escalated = router.escalate(&routed.decision).unwrap();
        // Medium is unpopulated, so escalation skips straight to Large.
        assert_eq!(escalated.decision.selected_tier, ModelTier::Large);
        assert_eq!(escalated.decision.escalated_from, Some(ModelTier::Small));
        // No tier above Large: escalation is exhausted.
        assert!(router.escalate(&escalated.decision).is_none());
    }
}
