//! Few-shot example templates for prompt construction.

/// A collection of input/output example pairs injected into a prompt to guide
/// the model's behaviour through demonstration.
///
/// ```ignore
/// use daimon::prompt::FewShotTemplate;
///
/// let tpl = FewShotTemplate::new()
///     .example("What is 2+2?", "4")
///     .example("What is the capital of France?", "Paris")
///     .with_prefix("Here are some examples of how to respond:");
///
/// let rendered = tpl.render();
/// ```
#[derive(Debug, Clone, Default)]
pub struct FewShotTemplate {
    examples: Vec<(String, String)>,
    prefix: Option<String>,
    input_label: String,
    output_label: String,
}

impl FewShotTemplate {
    pub fn new() -> Self {
        Self {
            examples: Vec::new(),
            prefix: None,
            input_label: "Input".to_string(),
            output_label: "Output".to_string(),
        }
    }

    /// Adds an example input/output pair.
    pub fn example(mut self, input: impl Into<String>, output: impl Into<String>) -> Self {
        self.examples.push((input.into(), output.into()));
        self
    }

    /// Sets text rendered before the examples.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = Some(prefix.into());
        self
    }

    /// Customises the label used for inputs (default: `"Input"`).
    pub fn with_input_label(mut self, label: impl Into<String>) -> Self {
        self.input_label = label.into();
        self
    }

    /// Customises the label used for outputs (default: `"Output"`).
    pub fn with_output_label(mut self, label: impl Into<String>) -> Self {
        self.output_label = label.into();
        self
    }

    /// Renders the examples into a formatted string.
    pub fn render(&self) -> String {
        let mut parts = Vec::new();

        if let Some(prefix) = &self.prefix {
            parts.push(prefix.clone());
            parts.push(String::new());
        }

        for (i, (input, output)) in self.examples.iter().enumerate() {
            if i > 0 {
                parts.push(String::new());
            }
            parts.push(format!("{}: {input}", self.input_label));
            parts.push(format!("{}: {output}", self.output_label));
        }

        parts.join("\n")
    }

    /// Returns the number of examples.
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Returns true if no examples have been added.
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_few_shot() {
        let tpl = FewShotTemplate::new()
            .example("What is 2+2?", "4")
            .example("Capital of France?", "Paris");
        let rendered = tpl.render();
        assert!(rendered.contains("Input: What is 2+2?"));
        assert!(rendered.contains("Output: 4"));
        assert!(rendered.contains("Input: Capital of France?"));
        assert!(rendered.contains("Output: Paris"));
    }

    #[test]
    fn test_with_prefix() {
        let tpl = FewShotTemplate::new()
            .with_prefix("Examples:")
            .example("hi", "hello");
        let rendered = tpl.render();
        assert!(rendered.starts_with("Examples:"));
    }

    #[test]
    fn test_custom_labels() {
        let tpl = FewShotTemplate::new()
            .with_input_label("Q")
            .with_output_label("A")
            .example("question", "answer");
        let rendered = tpl.render();
        assert!(rendered.contains("Q: question"));
        assert!(rendered.contains("A: answer"));
    }

    #[test]
    fn test_empty() {
        let tpl = FewShotTemplate::new();
        assert!(tpl.is_empty());
        assert_eq!(tpl.len(), 0);
        assert_eq!(tpl.render(), "");
    }
}
