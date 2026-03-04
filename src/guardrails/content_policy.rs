//! LLM-as-judge guardrail for content safety.

use crate::error::Result;
use crate::guardrails::traits::{GuardrailResult, InputGuardrail, OutputGuardrail};
use crate::model::SharedModel;
use crate::model::types::{ChatRequest, ChatResponse, Message};

/// Uses an LLM to evaluate whether content violates a policy.
///
/// The model is prompted with the policy description and the content to evaluate.
/// It must respond with `PASS` or `BLOCK: <reason>`.
///
/// ```ignore
/// use daimon::guardrails::ContentPolicyGuardrail;
///
/// let guard = ContentPolicyGuardrail::new(model.clone(), "No hate speech, threats, or illegal content.");
/// let agent = Agent::builder()
///     .model(model)
///     .input_guardrail(guard)
///     .build()?;
/// ```
pub struct ContentPolicyGuardrail {
    model: SharedModel,
    policy: String,
}

impl ContentPolicyGuardrail {
    pub fn new(model: SharedModel, policy: impl Into<String>) -> Self {
        Self {
            model,
            policy: policy.into(),
        }
    }

    async fn evaluate(&self, content: &str) -> Result<GuardrailResult> {
        let prompt = format!(
            "You are a content safety classifier. Evaluate whether the following content \
             violates the content policy.\n\n\
             Policy: {}\n\n\
             Content to evaluate:\n{}\n\n\
             Respond with EXACTLY one of:\n\
             PASS\n\
             BLOCK: <brief reason>\n\n\
             Your response:",
            self.policy, content
        );

        let request = ChatRequest::new(vec![Message::user(&prompt)]);
        let response = self.model.generate_erased(&request).await?;
        let text = response.text().trim().to_string();
        let upper = text.to_uppercase();

        if upper.starts_with("PASS") {
            Ok(GuardrailResult::Pass)
        } else if upper.starts_with("BLOCK") {
            let reason = text
                .get(5..)
                .map(|s| s.trim_start_matches(':').trim())
                .filter(|s| !s.is_empty())
                .unwrap_or("content policy violation")
                .to_string();
            Ok(GuardrailResult::Block(reason))
        } else {
            Ok(GuardrailResult::Block(format!(
                "content policy classifier returned ambiguous result: {text}"
            )))
        }
    }
}

impl InputGuardrail for ContentPolicyGuardrail {
    async fn check(&self, input: &str, _messages: &[Message]) -> Result<GuardrailResult> {
        self.evaluate(input).await
    }
}

impl OutputGuardrail for ContentPolicyGuardrail {
    async fn check(&self, response: &ChatResponse) -> Result<GuardrailResult> {
        self.evaluate(response.text()).await
    }
}
