//! Tool trait, output/choice types, and retry policy. Implement [`Tool`] for
//! callable functions an agent can invoke; the registry lives in the `daimon`
//! facade crate.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;

use crate::error::Result;

/// The result of executing a tool.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    /// The output content (text or serialized JSON).
    pub content: String,
    /// Whether this output represents an error.
    pub is_error: bool,
}

impl ToolOutput {
    /// Create a successful text output.
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }

    /// Create a successful output from a serializable value, encoded as JSON.
    pub fn json(value: &impl Serialize) -> Result<Self> {
        Ok(Self {
            content: serde_json::to_string(value)?,
            is_error: false,
        })
    }

    /// Create an error output.
    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
        }
    }
}

/// Controls which tools the model is allowed to use.
#[derive(Debug, Clone, Default)]
pub enum ToolChoice {
    /// Let the model decide whether to call tools.
    #[default]
    Auto,
    /// Prevent the model from calling any tools.
    None,
    /// Force the model to call at least one tool.
    Required,
    /// Force the model to call a specific tool by name.
    Specific(String),
}

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

/// Trait for tools the agent can invoke. Tools must have unique names and declare a JSON Schema for parameters.
pub trait Tool: Send + Sync {
    /// Unique identifier for the tool. Used by the model when requesting a call.
    fn name(&self) -> &str;
    /// Human-readable description. The model uses this to decide when to call the tool.
    fn description(&self) -> &str;
    /// JSON Schema for the tool's parameters. Validates and guides the model's argument generation.
    fn parameters_schema(&self) -> serde_json::Value;

    /// Executes the tool with the given arguments. Arguments are validated by the model; implementors may still validate.
    fn execute(&self, input: &serde_json::Value)
    -> impl Future<Output = Result<ToolOutput>> + Send;

    /// Per-tool retry policy. If `Some`, overrides the agent-level retry policy
    /// for this tool. Return `None` to use the agent's default.
    fn retry_policy(&self) -> Option<ToolRetryPolicy> {
        None
    }
}

/// Object-safe wrapper for the `Tool` trait, enabling dynamic dispatch via `Arc<dyn ErasedTool>`.
pub trait ErasedTool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;

    fn execute_erased<'a>(
        &'a self,
        input: &'a serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput>> + Send + 'a>>;

    fn retry_policy(&self) -> Option<ToolRetryPolicy>;
}

impl<T: Tool> ErasedTool for T {
    fn name(&self) -> &str {
        Tool::name(self)
    }

    fn description(&self) -> &str {
        Tool::description(self)
    }

    fn parameters_schema(&self) -> serde_json::Value {
        Tool::parameters_schema(self)
    }

    fn execute_erased<'a>(
        &'a self,
        input: &'a serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput>> + Send + 'a>> {
        Box::pin(Tool::execute(self, input))
    }

    fn retry_policy(&self) -> Option<ToolRetryPolicy> {
        Tool::retry_policy(self)
    }
}

/// Shared ownership of a tool via `Arc<dyn ErasedTool>`. Used by registry and agent.
pub type SharedTool = Arc<dyn ErasedTool>;

#[cfg(test)]
mod tests {
    use super::BackoffStrategy;
    use super::{SharedTool, Tool, ToolOutput, ToolRetryPolicy};
    use crate::Result;
    use std::sync::Arc;
    use std::time::Duration;

    /// A provider-crate-style Tool impl using only daimon_core items.
    struct Echo;

    impl Tool for Echo {
        fn name(&self) -> &str {
            "echo"
        }

        fn description(&self) -> &str {
            "Echoes its input back."
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": { "text": { "type": "string" } },
                "required": ["text"]
            })
        }

        async fn execute(&self, input: &serde_json::Value) -> Result<ToolOutput> {
            Ok(ToolOutput::text(input["text"].as_str().unwrap_or_default()))
        }
    }

    #[tokio::test]
    async fn tool_is_implementable_from_core_alone() {
        let tool = Echo;
        assert_eq!(tool.name(), "echo");
        assert!(tool.retry_policy().is_none());

        let out = tool
            .execute(&serde_json::json!({ "text": "hi" }))
            .await
            .unwrap();
        assert_eq!(out.content, "hi");
        assert!(!out.is_error);

        let shared: SharedTool = Arc::new(Echo);
        let out = shared
            .execute_erased(&serde_json::json!({ "text": "yo" }))
            .await
            .unwrap();
        assert_eq!(out.content, "yo");

        let policy = ToolRetryPolicy::exponential(3).retryable_on(vec!["timeout".into()]);
        assert!(policy.is_retryable("connection timeout"));
        assert!(!policy.is_retryable("invalid arguments"));
    }

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
