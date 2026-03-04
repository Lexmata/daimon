use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::cost::models::{CostModel, TokenDirection};
use crate::model::types::Usage;

/// Thread-safe accumulator for token costs across iterations.
///
/// Stores the cumulative cost as an atomic fixed-point integer (millionths of a
/// dollar) for lock-free updates.
pub struct CostTracker {
    pub(crate) cost_model: Arc<dyn CostModel>,
    /// Cumulative cost in micro-dollars (1e-6 USD).
    cumulative_microdollars: AtomicU64,
}

impl CostTracker {
    pub fn new(cost_model: Arc<dyn CostModel>) -> Self {
        Self {
            cost_model,
            cumulative_microdollars: AtomicU64::new(0),
        }
    }

    /// Records usage from one model call and returns the incremental cost in USD.
    pub fn record(&self, model_id: &str, usage: &Usage) -> f64 {
        let input_cost =
            usage.input_tokens as f64 * self.cost_model.cost_per_token(model_id, TokenDirection::Input);
        let output_cost =
            usage.output_tokens as f64 * self.cost_model.cost_per_token(model_id, TokenDirection::Output);
        let total = input_cost + output_cost;
        let microdollars = (total * 1_000_000.0) as u64;
        self.cumulative_microdollars
            .fetch_add(microdollars, Ordering::Relaxed);
        total
    }

    /// Returns the cumulative cost in USD.
    pub fn cumulative_cost(&self) -> f64 {
        self.cumulative_microdollars.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }

    /// Resets the cumulative cost to zero.
    pub fn reset(&self) {
        self.cumulative_microdollars.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::models::OpenAiCostModel;

    #[test]
    fn test_tracker_accumulates() {
        let tracker = CostTracker::new(Arc::new(OpenAiCostModel));

        let usage = Usage {
            input_tokens: 1000,
            output_tokens: 500,
            cached_tokens: 0,
        };

        let cost1 = tracker.record("gpt-4o", &usage);
        assert!(cost1 > 0.0);

        let cost2 = tracker.record("gpt-4o", &usage);
        assert!((tracker.cumulative_cost() - (cost1 + cost2)).abs() < 1e-6);
    }

    #[test]
    fn test_tracker_reset() {
        let tracker = CostTracker::new(Arc::new(OpenAiCostModel));

        tracker.record("gpt-4o", &Usage {
            input_tokens: 1000,
            output_tokens: 500,
            cached_tokens: 0,
        });

        assert!(tracker.cumulative_cost() > 0.0);
        tracker.reset();
        assert_eq!(tracker.cumulative_cost(), 0.0);
    }
}
