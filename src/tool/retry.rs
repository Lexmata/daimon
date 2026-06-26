//! Self-healing tool retry with configurable backoff.

use std::time::Duration;

/// Strategy for computing delay between retries.
#[derive(Debug, Clone)]
pub enum BackoffStrategy {
    /// Fixed delay between retries.
    Fixed(Duration),
    /// Exponential backoff: base * 2^attempt, capped at max.
    Exponential { base: Duration, max: Duration },
}

impl BackoffStrategy {
    pub fn delay_for(&self, attempt: usize) -> Duration {
        match self {
            BackoffStrategy::Fixed(d) => *d,
            BackoffStrategy::Exponential { base, max } => {
                let millis = base.as_millis() as u64 * 2u64.saturating_pow(attempt as u32);
                Duration::from_millis(millis).min(*max)
            }
        }
    }
}

/// Policy controlling when and how tool execution is retried on failure.
#[derive(Debug, Clone)]
pub struct ToolRetryPolicy {
    /// Maximum number of retry attempts (0 = no retries).
    pub max_retries: usize,
    /// Backoff strategy between attempts.
    pub backoff: BackoffStrategy,
    /// If set, only errors whose message contains one of these substrings are retried.
    pub retryable_patterns: Vec<String>,
}

impl ToolRetryPolicy {
    /// Creates a policy with exponential backoff (100ms base, 10s max).
    pub fn exponential(max_retries: usize) -> Self {
        Self {
            max_retries,
            backoff: BackoffStrategy::Exponential {
                base: Duration::from_millis(100),
                max: Duration::from_secs(10),
            },
            retryable_patterns: Vec::new(),
        }
    }

    /// Creates a policy with fixed delay between retries.
    pub fn fixed(max_retries: usize, delay: Duration) -> Self {
        Self {
            max_retries,
            backoff: BackoffStrategy::Fixed(delay),
            retryable_patterns: Vec::new(),
        }
    }

    /// Only retry if the error message contains one of these substrings.
    pub fn retryable_on(mut self, patterns: Vec<String>) -> Self {
        self.retryable_patterns = patterns;
        self
    }

    /// Returns true if the given error message is eligible for retry.
    pub fn is_retryable(&self, error_msg: &str) -> bool {
        if self.retryable_patterns.is_empty() {
            return true;
        }
        self.retryable_patterns
            .iter()
            .any(|p| error_msg.contains(p.as_str()))
    }
}

impl Default for ToolRetryPolicy {
    fn default() -> Self {
        Self::exponential(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_backoff() {
        let strategy = BackoffStrategy::Exponential {
            base: Duration::from_millis(100),
            max: Duration::from_secs(5),
        };
        assert_eq!(strategy.delay_for(0), Duration::from_millis(100));
        assert_eq!(strategy.delay_for(1), Duration::from_millis(200));
        assert_eq!(strategy.delay_for(2), Duration::from_millis(400));
        assert_eq!(strategy.delay_for(10), Duration::from_secs(5));
    }

    #[test]
    fn test_fixed_backoff() {
        let strategy = BackoffStrategy::Fixed(Duration::from_secs(1));
        assert_eq!(strategy.delay_for(0), Duration::from_secs(1));
        assert_eq!(strategy.delay_for(5), Duration::from_secs(1));
    }

    #[test]
    fn test_retryable_patterns() {
        let policy = ToolRetryPolicy::exponential(3)
            .retryable_on(vec!["timeout".into(), "rate limit".into()]);
        assert!(policy.is_retryable("connection timeout"));
        assert!(policy.is_retryable("rate limit exceeded"));
        assert!(!policy.is_retryable("invalid arguments"));
    }

    #[test]
    fn test_empty_patterns_retries_everything() {
        let policy = ToolRetryPolicy::exponential(3);
        assert!(policy.is_retryable("any error at all"));
    }
}
