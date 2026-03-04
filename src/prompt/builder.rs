use crate::prompt::PromptTemplate;

/// Fluent builder for composing prompts from sections.
///
/// ```ignore
/// use daimon::prompt::PromptBuilder;
///
/// let tpl = PromptBuilder::new()
///     .persona("You are an expert Rust developer.")
///     .instruction("Answer concisely.")
///     .constraint("Never reveal internal implementation details.")
///     .example("Q: What is ownership?\nA: Ownership is Rust's memory management model.")
///     .build();
/// ```
#[derive(Debug, Default)]
pub struct PromptBuilder {
    persona: Option<String>,
    instructions: Vec<String>,
    constraints: Vec<String>,
    examples: Vec<String>,
    extra_sections: Vec<(String, String)>,
}

impl PromptBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the persona / role description.
    pub fn persona(mut self, persona: impl Into<String>) -> Self {
        self.persona = Some(persona.into());
        self
    }

    /// Adds an instruction line.
    pub fn instruction(mut self, instruction: impl Into<String>) -> Self {
        self.instructions.push(instruction.into());
        self
    }

    /// Adds a constraint the model should follow.
    pub fn constraint(mut self, constraint: impl Into<String>) -> Self {
        self.constraints.push(constraint.into());
        self
    }

    /// Adds a few-shot example (input/output pair as a single string).
    pub fn example(mut self, example: impl Into<String>) -> Self {
        self.examples.push(example.into());
        self
    }

    /// Adds a custom named section.
    pub fn section(mut self, title: impl Into<String>, content: impl Into<String>) -> Self {
        self.extra_sections.push((title.into(), content.into()));
        self
    }

    /// Builds the prompt template by concatenating all sections.
    pub fn build(self) -> PromptTemplate {
        let mut parts = Vec::new();

        if let Some(persona) = &self.persona {
            parts.push(persona.clone());
        }

        if !self.instructions.is_empty() {
            parts.push(String::new());
            parts.push("## Instructions".to_string());
            for inst in &self.instructions {
                parts.push(format!("- {inst}"));
            }
        }

        if !self.constraints.is_empty() {
            parts.push(String::new());
            parts.push("## Constraints".to_string());
            for c in &self.constraints {
                parts.push(format!("- {c}"));
            }
        }

        if !self.examples.is_empty() {
            parts.push(String::new());
            parts.push("## Examples".to_string());
            for (i, ex) in self.examples.iter().enumerate() {
                if i > 0 {
                    parts.push(String::new());
                }
                parts.push(ex.clone());
            }
        }

        for (title, content) in &self.extra_sections {
            parts.push(String::new());
            parts.push(format!("## {title}"));
            parts.push(content.clone());
        }

        PromptTemplate::new(parts.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_persona_only() {
        let tpl = PromptBuilder::new()
            .persona("You are a bot.")
            .build();
        assert_eq!(tpl.render_static(), "You are a bot.");
    }

    #[test]
    fn test_builder_full() {
        let tpl = PromptBuilder::new()
            .persona("You are helpful.")
            .instruction("Be concise.")
            .constraint("No profanity.")
            .example("Q: Hi\nA: Hello!")
            .build();

        let rendered = tpl.render_static();
        assert!(rendered.contains("You are helpful."));
        assert!(rendered.contains("Be concise."));
        assert!(rendered.contains("No profanity."));
        assert!(rendered.contains("Q: Hi\nA: Hello!"));
    }

    #[test]
    fn test_builder_with_variables() {
        let tpl = PromptBuilder::new()
            .persona("You are {role}.")
            .build()
            .var("role", "a teacher");
        assert_eq!(tpl.render_static(), "You are a teacher.");
    }
}
