/// Whether a token was input (prompt) or output (completion).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenDirection {
    Input,
    Output,
}

/// Maps (model_id, direction) to a per-token dollar cost.
pub trait CostModel: Send + Sync {
    /// Returns the cost in USD per token for the given model and direction.
    fn cost_per_token(&self, model_id: &str, direction: TokenDirection) -> f64;
}

/// Approximate pricing for OpenAI models (as of early 2026).
pub struct OpenAiCostModel;

impl CostModel for OpenAiCostModel {
    fn cost_per_token(&self, model_id: &str, direction: TokenDirection) -> f64 {
        match (model_id, direction) {
            (m, TokenDirection::Input) if m.starts_with("gpt-4o") => 2.5e-6,
            (m, TokenDirection::Output) if m.starts_with("gpt-4o") => 10.0e-6,
            (m, TokenDirection::Input) if m.starts_with("gpt-4") => 30.0e-6,
            (m, TokenDirection::Output) if m.starts_with("gpt-4") => 60.0e-6,
            (m, TokenDirection::Input) if m.starts_with("gpt-3.5") => 0.5e-6,
            (m, TokenDirection::Output) if m.starts_with("gpt-3.5") => 1.5e-6,
            (m, TokenDirection::Input) if m.contains("o1") || m.contains("o3") => 15.0e-6,
            (m, TokenDirection::Output) if m.contains("o1") || m.contains("o3") => 60.0e-6,
            (_, TokenDirection::Input) => 5.0e-6,
            (_, TokenDirection::Output) => 15.0e-6,
        }
    }
}

/// Approximate pricing for Anthropic Claude models (as of early 2026).
pub struct AnthropicCostModel;

impl CostModel for AnthropicCostModel {
    fn cost_per_token(&self, model_id: &str, direction: TokenDirection) -> f64 {
        match (model_id, direction) {
            (m, TokenDirection::Input) if m.contains("opus") => 15.0e-6,
            (m, TokenDirection::Output) if m.contains("opus") => 75.0e-6,
            (m, TokenDirection::Input) if m.contains("sonnet") => 3.0e-6,
            (m, TokenDirection::Output) if m.contains("sonnet") => 15.0e-6,
            (m, TokenDirection::Input) if m.contains("haiku") => 0.25e-6,
            (m, TokenDirection::Output) if m.contains("haiku") => 1.25e-6,
            (_, TokenDirection::Input) => 3.0e-6,
            (_, TokenDirection::Output) => 15.0e-6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_pricing() {
        let m = OpenAiCostModel;
        let cost = m.cost_per_token("gpt-4o-mini", TokenDirection::Input);
        assert!(cost > 0.0);
        assert!(cost < m.cost_per_token("gpt-4o-mini", TokenDirection::Output));
    }

    #[test]
    fn test_anthropic_pricing() {
        let m = AnthropicCostModel;
        let haiku_in = m.cost_per_token("claude-3-haiku", TokenDirection::Input);
        let opus_in = m.cost_per_token("claude-3-opus", TokenDirection::Input);
        assert!(haiku_in < opus_in);
    }
}
