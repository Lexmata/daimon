//! Runtime-resolved context providers for prompt templates.

use std::future::Future;
use std::pin::Pin;

/// Provides a named variable value resolved at runtime (e.g. current date,
/// user profile, database lookup).
///
/// ```ignore
/// use daimon::prompt::DynamicContext;
///
/// struct CurrentDate;
///
/// impl DynamicContext for CurrentDate {
///     fn key(&self) -> &str { "date" }
///     async fn resolve(&self) -> String {
///         chrono::Local::now().format("%Y-%m-%d").to_string()
///     }
/// }
/// ```
pub trait DynamicContext: Send + Sync {
    /// The template variable name this context provides (e.g. `"date"`).
    fn key(&self) -> &str;

    /// Resolves the variable value. Called once per render.
    fn resolve(&self) -> impl Future<Output = String> + Send;
}

/// Object-safe wrapper for [`DynamicContext`].
pub trait ErasedDynamicContext: Send + Sync {
    fn key(&self) -> &str;

    fn resolve_erased(&self) -> Pin<Box<dyn Future<Output = String> + Send + '_>>;
}

impl<T: DynamicContext> ErasedDynamicContext for T {
    fn key(&self) -> &str {
        DynamicContext::key(self)
    }

    fn resolve_erased(&self) -> Pin<Box<dyn Future<Output = String> + Send + '_>> {
        Box::pin(self.resolve())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StaticContext {
        key: String,
        value: String,
    }

    impl DynamicContext for StaticContext {
        fn key(&self) -> &str {
            &self.key
        }
        async fn resolve(&self) -> String {
            self.value.clone()
        }
    }

    #[tokio::test]
    async fn test_dynamic_context_resolve() {
        let ctx = StaticContext {
            key: "name".into(),
            value: "Alice".into(),
        };
        assert_eq!(DynamicContext::key(&ctx), "name");
        assert_eq!(ctx.resolve().await, "Alice");
    }

    #[tokio::test]
    async fn test_erased_dynamic_context() {
        let ctx: Box<dyn ErasedDynamicContext> = Box::new(StaticContext {
            key: "role".into(),
            value: "developer".into(),
        });
        assert_eq!(ctx.key(), "role");
        assert_eq!(ctx.resolve_erased().await, "developer");
    }
}
