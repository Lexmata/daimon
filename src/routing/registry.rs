//! Model registrations: a model plus its competence tier and cost metadata.

use std::sync::Arc;

use crate::cost::{CostModel, TokenDirection};
use crate::model::{Model, SharedModel};

use super::ModelTier;

/// Per-token USD cost for a registered model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelCost {
    /// Cost per input (prompt) token.
    pub input_per_token: f64,
    /// Cost per output (completion) token.
    pub output_per_token: f64,
}

impl ModelCost {
    /// Ordering key for cheapest-first selection within a tier: the fixed
    /// blend `input + output` per token. The blend is fixed so ordering is
    /// deterministic.
    pub(crate) fn ordering_key(&self) -> f64 {
        self.input_per_token + self.output_per_token
    }
}

/// One model registered with the router, at a competence tier, with optional
/// explicit cost. When `cost` is `None`, cost is resolved through the
/// router's [`CostModel`] fallback using the model's `model_id()`.
pub(crate) struct ModelRegistration {
    /// The model to call when this registration is selected.
    pub model: SharedModel,
    /// Competence tier this model is registered at.
    pub tier: ModelTier,
    /// Explicit per-token cost; wins over the `CostModel` fallback.
    pub cost: Option<ModelCost>,
    /// Optional provider-group label. When set, escalation stays within
    /// registrations bearing the same group label.
    pub group: Option<String>,
}

impl ModelRegistration {
    /// Registers an owned model at the given tier (cost via fallback).
    pub(crate) fn new<M: Model + 'static>(tier: ModelTier, model: M) -> Self {
        Self {
            model: Arc::new(model),
            tier,
            cost: None,
            group: None,
        }
    }

    /// Registers a shared model at the given tier (cost via fallback).
    pub(crate) fn shared(tier: ModelTier, model: SharedModel) -> Self {
        Self {
            model,
            tier,
            cost: None,
            group: None,
        }
    }

    /// Sets an explicit per-token cost, overriding any `CostModel` fallback.
    pub(crate) fn with_cost(mut self, cost: ModelCost) -> Self {
        self.cost = Some(cost);
        self
    }

    /// Sets a provider-group label. Escalation from a grouped registration
    /// stays within registrations bearing the same label.
    pub(crate) fn with_group(mut self, group: impl Into<String>) -> Self {
        self.group = Some(group.into());
        self
    }

    /// Resolves this registration's cost: explicit wins; otherwise the
    /// `CostModel` fallback is queried by `model_id()`. With neither, the
    /// registration sorts last within its tier (infinite cost).
    pub(crate) fn effective_cost(&self, fallback: Option<&Arc<dyn CostModel>>) -> ModelCost {
        if let Some(cost) = self.cost {
            return cost;
        }
        let id = self.model.model_id_erased();
        match fallback {
            Some(cm) => ModelCost {
                input_per_token: cm.cost_per_token(id, TokenDirection::Input),
                output_per_token: cm.cost_per_token(id, TokenDirection::Output),
            },
            None => {
                tracing::debug!(
                    model_id = id,
                    "no cost metadata for registration; it will sort last in its tier"
                );
                ModelCost {
                    input_per_token: f64::INFINITY,
                    output_per_token: f64::INFINITY,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;
    use crate::model::types::{ChatRequest, ChatResponse, Message, StopReason};
    use crate::stream::ResponseStream;

    struct StubModel;

    impl Model for StubModel {
        async fn generate(&self, _request: &ChatRequest) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message::assistant("stub"),
                stop_reason: StopReason::EndTurn,
                usage: None,
            })
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }

        fn model_id(&self) -> &str {
            "stub-model"
        }
    }

    struct FixedCostModel;

    impl CostModel for FixedCostModel {
        fn cost_per_token(&self, _model_id: &str, direction: TokenDirection) -> f64 {
            match direction {
                TokenDirection::Input => 1.0e-6,
                TokenDirection::Output => 2.0e-6,
            }
        }
    }

    #[test]
    fn explicit_cost_wins_over_fallback() {
        let fallback: Arc<dyn CostModel> = Arc::new(FixedCostModel);
        let reg = ModelRegistration::new(ModelTier::Small, StubModel).with_cost(ModelCost {
            input_per_token: 9.0e-6,
            output_per_token: 9.0e-6,
        });
        let cost = reg.effective_cost(Some(&fallback));
        assert_eq!(cost.input_per_token, 9.0e-6);
    }

    #[test]
    fn fallback_used_when_no_explicit_cost() {
        let fallback: Arc<dyn CostModel> = Arc::new(FixedCostModel);
        let reg = ModelRegistration::new(ModelTier::Small, StubModel);
        let cost = reg.effective_cost(Some(&fallback));
        assert_eq!(cost.input_per_token, 1.0e-6);
        assert_eq!(cost.output_per_token, 2.0e-6);
    }

    #[test]
    fn no_cost_metadata_sorts_last() {
        let reg = ModelRegistration::new(ModelTier::Small, StubModel);
        assert!(reg.effective_cost(None).ordering_key().is_infinite());
    }
}
