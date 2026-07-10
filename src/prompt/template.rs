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
        interpolate(&self.template, |key| {
            self.variables.get(key).map(String::as_str)
        })
    }

    /// Renders with additional runtime variables that override stored ones.
    pub fn render_with(&self, overrides: &HashMap<String, String>) -> String {
        interpolate(&self.template, |key| {
            overrides
                .get(key)
                .or_else(|| self.variables.get(key))
                .map(String::as_str)
        })
    }

    /// Renders the template, resolving [`DynamicContext`](super::DynamicContext) providers
    /// for any remaining `{variable}` placeholders.
    pub async fn render_dynamic(&self, contexts: &[&dyn super::ErasedDynamicContext]) -> String {
        let mut overrides = HashMap::new();
        for ctx in contexts {
            let value = ctx.resolve_erased().await;
            overrides.insert(ctx.key().to_string(), value);
        }
        self.render_with(&overrides)
    }
}

/// Single-pass `{name}` interpolation.
///
/// Each placeholder is resolved through `lookup`; unknown names stay literal.
/// Values are inserted verbatim — a placeholder-shaped substring inside a
/// substituted value is never expanded again. The previous per-variable
/// `String::replace` loop rescanned the whole string once per variable and
/// could re-expand placeholders that appeared inside earlier substitutions,
/// with the outcome depending on `HashMap` iteration order.
fn interpolate<'a>(template: &str, lookup: impl Fn(&str) -> Option<&'a str>) -> String {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        match after.find(['{', '}']) {
            // A well-formed `{name}` span: substitute or keep literal.
            Some(end) if after.as_bytes()[end] == b'}' => {
                let name = &after[..end];
                match lookup(name) {
                    Some(value) => out.push_str(value),
                    None => {
                        out.push('{');
                        out.push_str(name);
                        out.push('}');
                    }
                }
                rest = &after[end + 1..];
            }
            // Unclosed brace, or another `{` before any `}`: emit the `{`
            // literally and rescan from the next character.
            _ => {
                out.push('{');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
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
        let tpl = PromptTemplate::new("Hello {name}, {unknown}.").var("name", "Bob");
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
        let tpl = PromptTemplate::new("{a} {b}").vars([("a", "1"), ("b", "2")]);
        assert_eq!(tpl.render_static(), "1 2");
    }

    #[test]
    fn test_value_containing_placeholder_not_reexpanded() {
        // A substituted value that itself looks like a placeholder must be
        // inserted verbatim. The old per-variable replace loop could expand
        // it depending on HashMap iteration order.
        let tpl = PromptTemplate::new("{outer}")
            .var("outer", "{inner}")
            .var("inner", "surprise");
        assert_eq!(tpl.render_static(), "{inner}");
    }

    #[test]
    fn test_unclosed_brace_preserved() {
        let tpl = PromptTemplate::new("a {name and {b}").var("b", "2");
        assert_eq!(tpl.render_static(), "a {name and 2");
    }

    #[test]
    fn test_doubled_braces_inner_placeholder() {
        // `{{name}}` contains a valid `{name}` span; braces around it stay
        // literal, matching the old substring-replace behavior.
        let tpl = PromptTemplate::new("{{name}}").var("name", "Alice");
        assert_eq!(tpl.render_static(), "{Alice}");
    }
}
