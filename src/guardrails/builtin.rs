//! Built-in guardrail implementations.

use crate::error::{DaimonError, Result};
use crate::guardrails::traits::{GuardrailResult, InputGuardrail};
use crate::model::types::Message;

/// Rejects inputs whose estimated token count exceeds a limit.
pub struct MaxTokenGuardrail {
    max_tokens: usize,
}

impl MaxTokenGuardrail {
    pub fn new(max_tokens: usize) -> Self {
        Self { max_tokens }
    }
}

impl InputGuardrail for MaxTokenGuardrail {
    async fn check(&self, input: &str, _messages: &[Message]) -> Result<GuardrailResult> {
        let estimated = input.len().div_ceil(4);
        if estimated > self.max_tokens {
            Ok(GuardrailResult::Block(format!(
                "input too long: ~{estimated} tokens exceeds limit of {}",
                self.max_tokens
            )))
        } else {
            Ok(GuardrailResult::Pass)
        }
    }
}

/// Blocks or transforms input that matches any of the configured regex patterns.
pub struct RegexFilterGuardrail {
    patterns: Vec<(regex_lite::Regex, FilterAction)>,
}

/// What to do when a regex matches.
#[derive(Debug, Clone)]
pub enum FilterAction {
    /// Block the entire input with this message.
    Block(String),
    /// Replace matched text with the given string.
    Redact(String),
}

impl RegexFilterGuardrail {
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }

    /// Adds a pattern that blocks the input when matched.
    ///
    /// # Errors
    ///
    /// Returns [`DaimonError::Builder`] if `pattern` is not a valid regex.
    /// A filter pattern that fails to compile must never be silently
    /// dropped — that would turn a security filter into no filter at all.
    pub fn block(mut self, pattern: &str, message: impl Into<String>) -> Result<Self> {
        let re = compile_pattern(pattern)?;
        self.patterns
            .push((re, FilterAction::Block(message.into())));
        Ok(self)
    }

    /// Adds a pattern that redacts matched text with a replacement.
    ///
    /// # Errors
    ///
    /// Returns [`DaimonError::Builder`] if `pattern` is not a valid regex.
    /// A filter pattern that fails to compile must never be silently
    /// dropped — that would turn a security filter into no filter at all.
    pub fn redact(mut self, pattern: &str, replacement: impl Into<String>) -> Result<Self> {
        let re = compile_pattern(pattern)?;
        self.patterns
            .push((re, FilterAction::Redact(replacement.into())));
        Ok(self)
    }
}

fn compile_pattern(pattern: &str) -> Result<regex_lite::Regex> {
    regex_lite::Regex::new(pattern).map_err(|e| {
        DaimonError::Builder(format!("invalid guardrail regex pattern '{pattern}': {e}"))
    })
}

impl Default for RegexFilterGuardrail {
    fn default() -> Self {
        Self::new()
    }
}

impl InputGuardrail for RegexFilterGuardrail {
    async fn check(&self, input: &str, _messages: &[Message]) -> Result<GuardrailResult> {
        let mut current = input.to_string();
        for (re, action) in &self.patterns {
            match action {
                FilterAction::Block(msg) => {
                    if re.is_match(&current) {
                        return Ok(GuardrailResult::Block(msg.clone()));
                    }
                }
                FilterAction::Redact(replacement) => {
                    let replaced = re.replace_all(&current, replacement.as_str()).to_string();
                    if replaced != current {
                        current = replaced;
                    }
                }
            }
        }
        if current != input {
            Ok(GuardrailResult::Transform(current))
        } else {
            Ok(GuardrailResult::Pass)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_max_token_guardrail_pass() {
        let guard = MaxTokenGuardrail::new(100);
        let result = guard.check("short input", &[]).await.unwrap();
        assert!(matches!(result, GuardrailResult::Pass));
    }

    #[tokio::test]
    async fn test_max_token_guardrail_block() {
        let guard = MaxTokenGuardrail::new(5);
        let long_input = "a".repeat(100);
        let result = guard.check(&long_input, &[]).await.unwrap();
        assert!(matches!(result, GuardrailResult::Block(_)));
    }

    #[tokio::test]
    async fn test_regex_filter_block() {
        let guard = RegexFilterGuardrail::new()
            .block(r"(?i)password\s*[:=]", "potential credential leak")
            .unwrap();
        let result = guard.check("my password: secret123", &[]).await.unwrap();
        assert!(matches!(result, GuardrailResult::Block(_)));
    }

    #[tokio::test]
    async fn test_regex_filter_redact() {
        let guard = RegexFilterGuardrail::new()
            .redact(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN REDACTED]")
            .unwrap();
        let result = guard.check("my ssn is 123-45-6789", &[]).await.unwrap();
        match result {
            GuardrailResult::Transform(s) => {
                assert_eq!(s, "my ssn is [SSN REDACTED]");
            }
            _ => panic!("expected Transform"),
        }
    }

    #[tokio::test]
    async fn test_regex_filter_pass() {
        let guard = RegexFilterGuardrail::new()
            .block(r"badword", "blocked")
            .unwrap();
        let result = guard.check("totally fine input", &[]).await.unwrap();
        assert!(matches!(result, GuardrailResult::Pass));
    }

    #[test]
    fn test_regex_filter_block_invalid_pattern_errors() {
        let err = RegexFilterGuardrail::new()
            .block(r"(unclosed", "blocked")
            .err()
            .expect("invalid pattern must produce an error");
        assert!(matches!(err, DaimonError::Builder(_)));
        assert!(err.to_string().contains("(unclosed"));
    }

    #[test]
    fn test_regex_filter_redact_invalid_pattern_errors() {
        let err = RegexFilterGuardrail::new()
            .redact(r"[a-", "[REDACTED]")
            .err()
            .expect("invalid pattern must produce an error");
        assert!(matches!(err, DaimonError::Builder(_)));
        assert!(err.to_string().contains("[a-"));
    }

    #[tokio::test]
    async fn test_regex_filter_chained_builders() {
        let guard = RegexFilterGuardrail::new()
            .block(r"badword", "blocked")
            .unwrap()
            .redact(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN REDACTED]")
            .unwrap();

        let blocked = guard.check("this has a badword in it", &[]).await.unwrap();
        assert!(matches!(blocked, GuardrailResult::Block(_)));

        let redacted = guard.check("ssn 123-45-6789", &[]).await.unwrap();
        assert!(matches!(redacted, GuardrailResult::Transform(_)));
    }
}
