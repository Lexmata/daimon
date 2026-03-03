//! Linear pipeline orchestration.
//!
//! A [`Chain`] executes a sequence of steps, passing the output of each step
//! as input to the next. Steps implement [`ChainStep`] — agents, transforms,
//! or any async function.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::agent::Agent;
use crate::error::{DaimonError, Result};

/// Shared context flowing through a chain. Each step receives the context
/// produced by the previous step and returns a new (or mutated) context.
#[derive(Debug, Clone, Default)]
pub struct ChainContext {
    /// The primary text payload. Each step typically reads this and
    /// overwrites it with its output.
    pub text: String,

    /// Arbitrary metadata carried between steps. Steps can read/write
    /// any key-value pairs here.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ChainContext {
    /// Creates a new context with the given text.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            metadata: HashMap::new(),
        }
    }

    /// Inserts a metadata entry.
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// A single step in a [`Chain`]. Receives a [`ChainContext`] and produces
/// a new context for the next step.
pub trait ChainStep: Send + Sync {
    /// Processes the context and returns a new context.
    fn process<'a>(
        &'a self,
        ctx: ChainContext,
    ) -> Pin<Box<dyn Future<Output = Result<ChainContext>> + Send + 'a>>;
}

/// Wraps an [`Agent`] as a [`ChainStep`]. The agent receives
/// `ctx.text` as a prompt and the response text becomes the new `ctx.text`.
pub struct AgentStep {
    agent: Arc<Agent>,
}

impl AgentStep {
    /// Wraps an agent as a chain step.
    pub fn new(agent: Arc<Agent>) -> Self {
        Self { agent }
    }
}

impl ChainStep for AgentStep {
    fn process<'a>(
        &'a self,
        mut ctx: ChainContext,
    ) -> Pin<Box<dyn Future<Output = Result<ChainContext>> + Send + 'a>> {
        Box::pin(async move {
            let response = self.agent.prompt(&ctx.text).await?;
            ctx.text = response.final_text;
            ctx.metadata.insert(
                "iterations".into(),
                serde_json::Value::Number(response.iterations.into()),
            );
            Ok(ctx)
        })
    }
}

type BoxedTransformFn =
    Arc<dyn Fn(ChainContext) -> Pin<Box<dyn Future<Output = Result<ChainContext>> + Send>> + Send + Sync>;

/// A [`ChainStep`] built from an async closure.
pub struct TransformStep {
    func: BoxedTransformFn,
}

impl TransformStep {
    /// Creates a transform step from an async closure.
    pub fn new<F, Fut>(func: F) -> Self
    where
        F: Fn(ChainContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<ChainContext>> + Send + 'static,
    {
        Self {
            func: Arc::new(move |ctx| Box::pin(func(ctx))),
        }
    }
}

impl ChainStep for TransformStep {
    fn process<'a>(
        &'a self,
        ctx: ChainContext,
    ) -> Pin<Box<dyn Future<Output = Result<ChainContext>> + Send + 'a>> {
        (self.func)(ctx)
    }
}

/// Builder for constructing a [`Chain`].
pub struct ChainBuilder {
    steps: Vec<Arc<dyn ChainStep>>,
    name: Option<String>,
}

impl ChainBuilder {
    fn new() -> Self {
        Self {
            steps: Vec::new(),
            name: None,
        }
    }

    /// Appends a step to the chain.
    pub fn step<S: ChainStep + 'static>(mut self, step: S) -> Self {
        self.steps.push(Arc::new(step));
        self
    }

    /// Appends an agent as a chain step.
    pub fn agent(self, agent: Arc<Agent>) -> Self {
        self.step(AgentStep::new(agent))
    }

    /// Appends an async transform closure as a chain step.
    pub fn transform<F, Fut>(self, f: F) -> Self
    where
        F: Fn(ChainContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<ChainContext>> + Send + 'static,
    {
        self.step(TransformStep::new(f))
    }

    /// Sets a name for the chain (used in tracing spans).
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Builds the chain. Fails if no steps were added.
    pub fn build(self) -> Result<Chain> {
        if self.steps.is_empty() {
            return Err(DaimonError::Orchestration(
                "chain must have at least one step".into(),
            ));
        }
        Ok(Chain {
            steps: self.steps,
            name: self.name,
        })
    }
}

/// A linear pipeline that executes steps sequentially.
pub struct Chain {
    steps: Vec<Arc<dyn ChainStep>>,
    name: Option<String>,
}

impl Chain {
    /// Returns a new chain builder.
    pub fn builder() -> ChainBuilder {
        ChainBuilder::new()
    }

    /// Runs the chain with the given input text.
    #[tracing::instrument(skip_all, fields(chain_name = self.name.as_deref().unwrap_or("unnamed"), steps = self.steps.len()))]
    pub async fn run(&self, input: impl Into<String>) -> Result<ChainContext> {
        let mut ctx = ChainContext::new(input);
        for (i, step) in self.steps.iter().enumerate() {
            let _span = tracing::info_span!("chain_step", index = i).entered();
            ctx = step.process(ctx).await?;
        }
        Ok(ctx)
    }

    /// Runs the chain with a pre-built context.
    pub async fn run_with_context(&self, mut ctx: ChainContext) -> Result<ChainContext> {
        for (i, step) in self.steps.iter().enumerate() {
            let _span = tracing::info_span!("chain_step", index = i).entered();
            ctx = step.process(ctx).await?;
        }
        Ok(ctx)
    }

    /// Returns the number of steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Returns true if the chain has no steps.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct UppercaseStep;

    impl ChainStep for UppercaseStep {
        fn process<'a>(
            &'a self,
            mut ctx: ChainContext,
        ) -> Pin<Box<dyn Future<Output = Result<ChainContext>> + Send + 'a>> {
            Box::pin(async move {
                ctx.text = ctx.text.to_uppercase();
                Ok(ctx)
            })
        }
    }

    struct AppendStep {
        suffix: String,
    }

    impl ChainStep for AppendStep {
        fn process<'a>(
            &'a self,
            mut ctx: ChainContext,
        ) -> Pin<Box<dyn Future<Output = Result<ChainContext>> + Send + 'a>> {
            Box::pin(async move {
                ctx.text.push_str(&self.suffix);
                Ok(ctx)
            })
        }
    }

    #[tokio::test]
    async fn test_chain_single_step() {
        let chain = Chain::builder()
            .step(UppercaseStep)
            .build()
            .unwrap();

        let result = chain.run("hello").await.unwrap();
        assert_eq!(result.text, "HELLO");
    }

    #[tokio::test]
    async fn test_chain_multiple_steps() {
        let chain = Chain::builder()
            .step(UppercaseStep)
            .step(AppendStep {
                suffix: "!".into(),
            })
            .build()
            .unwrap();

        let result = chain.run("hello").await.unwrap();
        assert_eq!(result.text, "HELLO!");
    }

    #[tokio::test]
    async fn test_chain_transform() {
        let chain = Chain::builder()
            .transform(|mut ctx| async move {
                ctx.text = format!("[{}]", ctx.text);
                Ok(ctx)
            })
            .build()
            .unwrap();

        let result = chain.run("test").await.unwrap();
        assert_eq!(result.text, "[test]");
    }

    #[tokio::test]
    async fn test_chain_metadata_propagation() {
        let chain = Chain::builder()
            .transform(|mut ctx| async move {
                ctx.metadata
                    .insert("step1".into(), serde_json::json!(true));
                Ok(ctx)
            })
            .transform(|ctx| async move {
                assert_eq!(ctx.metadata.get("step1"), Some(&serde_json::json!(true)));
                Ok(ctx)
            })
            .build()
            .unwrap();

        chain.run("test").await.unwrap();
    }

    #[tokio::test]
    async fn test_chain_empty_fails() {
        let result = Chain::builder().build();
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_chain_with_name() {
        let chain = Chain::builder()
            .name("my_chain")
            .step(UppercaseStep)
            .build()
            .unwrap();
        assert_eq!(chain.len(), 1);
    }

    #[tokio::test]
    async fn test_chain_run_with_context() {
        let chain = Chain::builder()
            .step(UppercaseStep)
            .build()
            .unwrap();

        let ctx = ChainContext::new("hello").with_metadata("key", serde_json::json!("val"));
        let result = chain.run_with_context(ctx).await.unwrap();
        assert_eq!(result.text, "HELLO");
        assert_eq!(
            result.metadata.get("key"),
            Some(&serde_json::json!("val"))
        );
    }

    #[tokio::test]
    async fn test_chain_error_propagation() {
        struct FailStep;

        impl ChainStep for FailStep {
            fn process<'a>(
                &'a self,
                _ctx: ChainContext,
            ) -> Pin<Box<dyn Future<Output = Result<ChainContext>> + Send + 'a>> {
                Box::pin(async { Err(DaimonError::Other("step failed".into())) })
            }
        }

        let chain = Chain::builder()
            .step(UppercaseStep)
            .step(FailStep)
            .step(UppercaseStep) // should never execute
            .build()
            .unwrap();

        let result = chain.run("hello").await;
        assert!(result.is_err());
    }
}
