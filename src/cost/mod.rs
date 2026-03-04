//! Token cost tracking and budget enforcement.
//!
//! Attach a [`CostModel`] to an agent to track cumulative spend. Set
//! [`AgentBuilder::max_budget`](crate::agent::AgentBuilder::max_budget) to
//! abort when a dollar limit is reached.
//!
//! ```ignore
//! use daimon::cost::OpenAiCostModel;
//!
//! let agent = Agent::builder()
//!     .model(my_model)
//!     .cost_model(OpenAiCostModel)
//!     .max_budget(0.50) // $0.50 per prompt
//!     .build()?;
//! ```

mod tracker;
mod models;

pub use tracker::CostTracker;
pub use models::{CostModel, TokenDirection, OpenAiCostModel, AnthropicCostModel};
