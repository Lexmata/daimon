//! Named competence tiers for routed models.

/// Competence tier of a registered model.
///
/// Ordered `Small < Medium < Large` (derived `Ord`), so "at least as
/// competent as" is a simple `>=` comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ModelTier {
    /// Cheap, fast models (haiku, gpt-4o-mini, local models).
    Small,
    /// Mid-tier workhorses (sonnet, gpt-4o).
    Medium,
    /// Frontier models (opus, o-class).
    Large,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiers_are_ordered() {
        assert!(ModelTier::Small < ModelTier::Medium);
        assert!(ModelTier::Medium < ModelTier::Large);
        assert!(ModelTier::Small < ModelTier::Large);
    }
}
