//! Difficulty-based dynamic routing across multiple registered models.
//!
//! A [`ModelRouter`] scores each ReAct iteration's difficulty and dispatches
//! the model call to the cheapest registered model competent to handle it.

mod registry;
mod router;
mod scorer;
mod tier;

pub use registry::ModelCost;
pub use router::{ModelRouter, ModelRouterBuilder, RouteDecision, RoutedModel, TierBands};
pub use scorer::{ErasedTaskScorer, HeuristicScorer, LlmScorer, SharedTaskScorer, TaskScorer};
pub use tier::ModelTier;

#[cfg(test)]
pub mod test_utils;
