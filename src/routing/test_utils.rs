//! Shared test utilities for routing tests.

use crate::error::Result;
use crate::model::types::ChatRequest;
use crate::routing::TaskScorer;

/// A scorer that always returns a fixed difficulty.
pub struct StaticScorer(pub f64);

impl TaskScorer for StaticScorer {
    async fn score(&self, _request: &ChatRequest) -> Result<f64> {
        Ok(self.0)
    }
}
