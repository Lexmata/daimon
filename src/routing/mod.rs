//! Difficulty-based dynamic routing across multiple registered models.
//!
//! A [`ModelRouter`] scores each ReAct iteration's difficulty and dispatches
//! the model call to the cheapest registered model competent to handle it.

mod registry;
mod scorer;
mod tier;

pub use registry::{ModelCost, ModelRegistration};
pub use scorer::{ErasedTaskScorer, HeuristicScorer, SharedTaskScorer, TaskScorer};
pub use tier::ModelTier;
