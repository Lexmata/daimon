use std::collections::HashMap;

/// A prompt template with `{variable}` interpolation.
///
/// Variables are replaced at render time. Unknown variables are left as-is.
#[derive(Debug, Clone)]
pub struct PromptTemplate {
    template: String,
    variables: HashMap<String, String>,
}

impl PromptTemplate {
    /// Creates a new template from a format string containing `{name}` placeholders.
    pub fn new(template: impl Into<String>) -> Self {
        Self {
            template: template.into(),
            variables: HashMap::new(),
        }
    }

    /// Sets a variable value. Chainable.
    pub fn var(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.variables.insert(key.into(), value.into());
        self
    }

    /// Sets multiple variables from an iterator.
    pub fn vars<I, K, V>(mut self, vars: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        for (k, v) in vars {
            self.variables.insert(k.into(), v.into());
        }
        self
    }

    /// Renders the template by replacing all `{name}` placeholders with their
    /// values. Unset variables remain as literal `{name}`.
    pub fn render_static(&self) -> String {
        let mut result = self.template.clone();
        for (key, value) in &self.variables {
            let placeholder = format!("{{{key}}}");
            result = result.replace(&placeholder, value);
        }
        result
    }

    /// Renders with additional runtime variables that override stored ones.
    pub fn render_with(&self, overrides: &HashMap<String, String>) -> String {
        let mut merged = self.variables.clone();
        for (k, v) in overrides {
            merged.insert(k.clone(), v.clone());
        }
        let mut result = self.template.clone();
        for (key, value) in &merged {
            let placeholder = format!("{{{key}}}");
            result = result.replace(&placeholder, value);
        }
        result
    }

    /// Renders the template, resolving [`DynamicContext`](super::DynamicContext) providers
    /// for any remaining `{variable}` placeholders.
    pub async fn render_dynamic(
        &self,
        contexts: &[&dyn super::ErasedDynamicContext],
    ) -> String {
        let mut overrides = HashMap::new();
        for ctx in contexts {
            let value = ctx.resolve_erased().await;
            overrides.insert(ctx.key().to_string(), value);
        }
        self.render_with(&overrides)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_interpolation() {
        let tpl = PromptTemplate::new("Hello {name}, you are a {role}.")
            .var("name", "Alice")
            .var("role", "developer");
        assert_eq!(tpl.render_static(), "Hello Alice, you are a developer.");
    }

    #[test]
    fn test_missing_variable_preserved() {
        let tpl = PromptTemplate::new("Hello {name}, {unknown}.")
            .var("name", "Bob");
        assert_eq!(tpl.render_static(), "Hello Bob, {unknown}.");
    }

    #[test]
    fn test_render_with_overrides() {
        let tpl = PromptTemplate::new("{greeting} {name}")
            .var("greeting", "Hello")
            .var("name", "Alice");

        let mut overrides = HashMap::new();
        overrides.insert("name".to_string(), "Bob".to_string());

        assert_eq!(tpl.render_with(&overrides), "Hello Bob");
    }

    #[test]
    fn test_empty_template() {
        let tpl = PromptTemplate::new("");
        assert_eq!(tpl.render_static(), "");
    }

    #[test]
    fn test_no_placeholders() {
        let tpl = PromptTemplate::new("Just plain text.");
        assert_eq!(tpl.render_static(), "Just plain text.");
    }

    #[test]
    fn test_vars_bulk() {
        let tpl = PromptTemplate::new("{a} {b}")
            .vars([("a", "1"), ("b", "2")]);
        assert_eq!(tpl.render_static(), "1 2");
    }
}
