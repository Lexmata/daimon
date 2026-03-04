use std::future::Future;
use std::pin::Pin;

use crate::error::Result;
use crate::model::types::{ChatResponse, Message};

/// The outcome of a guardrail check.
#[derive(Debug, Clone)]
pub enum GuardrailResult {
    /// Input/output is acceptable — proceed normally.
    Pass,
    /// Block the request and return this error message to the caller.
    Block(String),
    /// Replace the content with a transformed version.
    Transform(String),
}

/// Validates user input before it reaches the model.
pub trait InputGuardrail: Send + Sync {
    fn check(
        &self,
        input: &str,
        messages: &[Message],
    ) -> impl Future<Output = Result<GuardrailResult>> + Send;
}

/// Validates model output before it is returned to the caller.
pub trait OutputGuardrail: Send + Sync {
    fn check(
        &self,
        response: &ChatResponse,
    ) -> impl Future<Output = Result<GuardrailResult>> + Send;
}

/// Object-safe wrapper for [`InputGuardrail`].
pub trait ErasedInputGuardrail: Send + Sync {
    fn check_erased<'a>(
        &'a self,
        input: &'a str,
        messages: &'a [Message],
    ) -> Pin<Box<dyn Future<Output = Result<GuardrailResult>> + Send + 'a>>;
}

impl<T: InputGuardrail> ErasedInputGuardrail for T {
    fn check_erased<'a>(
        &'a self,
        input: &'a str,
        messages: &'a [Message],
    ) -> Pin<Box<dyn Future<Output = Result<GuardrailResult>> + Send + 'a>> {
        Box::pin(self.check(input, messages))
    }
}

/// Object-safe wrapper for [`OutputGuardrail`].
pub trait ErasedOutputGuardrail: Send + Sync {
    fn check_erased<'a>(
        &'a self,
        response: &'a ChatResponse,
    ) -> Pin<Box<dyn Future<Output = Result<GuardrailResult>> + Send + 'a>>;
}

impl<T: OutputGuardrail> ErasedOutputGuardrail for T {
    fn check_erased<'a>(
        &'a self,
        response: &'a ChatResponse,
    ) -> Pin<Box<dyn Future<Output = Result<GuardrailResult>> + Send + 'a>> {
        Box::pin(self.check(response))
    }
}
