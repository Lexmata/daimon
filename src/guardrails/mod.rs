//! Input and output guardrails for validating and transforming agent I/O.
//!
//! Guardrails run **before** the agent processes input (input guardrails) and
//! **after** the model produces a response (output guardrails). They can pass,
//! block with an error, or transform the content.
//!
//! ```ignore
//! use daimon::guardrails::{InputGuardrail, GuardrailResult};
//!
//! struct ProfanityFilter;
//!
//! impl InputGuardrail for ProfanityFilter {
//!     async fn check(&self, input: &str, _messages: &[Message]) -> daimon::Result<GuardrailResult> {
//!         if input.contains("badword") {
//!             Ok(GuardrailResult::Block("profanity detected".into()))
//!         } else {
//!             Ok(GuardrailResult::Pass)
//!         }
//!     }
//! }
//! ```

mod builtin;
mod content_policy;
mod traits;

pub use builtin::{MaxTokenGuardrail, RegexFilterGuardrail};
pub use content_policy::ContentPolicyGuardrail;
pub use traits::{
    ErasedInputGuardrail, ErasedOutputGuardrail, GuardrailResult, InputGuardrail, OutputGuardrail,
};
