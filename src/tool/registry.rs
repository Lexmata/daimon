use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{DaimonError, Result};
use crate::model::types::ToolSpec;
use crate::tool::traits::{SharedTool, Tool};

/// A registry holding named tools that the agent can invoke.
#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, SharedTool>,
}

impl ToolRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a tool by name. Returns an error if a tool with the same name already exists.
    pub fn register<T: Tool + 'static>(&mut self, tool: T) -> Result<()> {
        let name = tool.name().to_string();
        if self.tools.contains_key(&name) {
            return Err(DaimonError::DuplicateTool(name));
        }
        self.tools.insert(name, Arc::new(tool));
        Ok(())
    }

    /// Registers a pre-boxed shared tool. Returns an error if a tool with the same name already exists.
    pub fn register_shared(&mut self, tool: SharedTool) -> Result<()> {
        let name = tool.name().to_string();
        if self.tools.contains_key(&name) {
            return Err(DaimonError::DuplicateTool(name));
        }
        self.tools.insert(name, tool);
        Ok(())
    }

    /// Returns the tool with the given name, or `None` if not found.
    pub fn get(&self, name: &str) -> Option<&SharedTool> {
        self.tools.get(name)
    }

    /// Returns the names of all registered tools.
    pub fn list(&self) -> Vec<&str> {
        self.tools.keys().map(String::as_str).collect()
    }

    /// Returns the number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Returns true if no tools are registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Returns tool specs for the model (name, description, parameters schema). Used when building chat requests.
    pub fn tool_specs(&self) -> Vec<ToolSpec> {
        self.tools
            .values()
            .map(|tool| ToolSpec {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                parameters: tool.parameters_schema(),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::types::ToolOutput;

    struct MockTool {
        tool_name: String,
    }

    impl MockTool {
        fn new(name: &str) -> Self {
            Self {
                tool_name: name.to_string(),
            }
        }
    }

    impl Tool for MockTool {
        fn name(&self) -> &str {
            &self.tool_name
        }

        fn description(&self) -> &str {
            "A mock tool for testing"
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "input": { "type": "string" }
                }
            })
        }

        async fn execute(&self, _input: &serde_json::Value) -> Result<ToolOutput> {
            Ok(ToolOutput::text("mock result"))
        }
    }

    #[test]
    fn test_registry_new_is_empty() {
        let registry = ToolRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("calculator")).unwrap();

        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());
        assert!(registry.get("calculator").is_some());
        assert_eq!(registry.get("calculator").unwrap().name(), "calculator");
    }

    #[test]
    fn test_registry_register_duplicate_returns_error() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("calc")).unwrap();

        let result = registry.register(MockTool::new("calc"));
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), DaimonError::DuplicateTool(name) if name == "calc"));
    }

    #[test]
    fn test_registry_get_nonexistent_returns_none() {
        let registry = ToolRegistry::new();
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_list_returns_all_names() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("a")).unwrap();
        registry.register(MockTool::new("b")).unwrap();
        registry.register(MockTool::new("c")).unwrap();

        let mut names = registry.list();
        names.sort();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_registry_tool_specs() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("calc")).unwrap();

        let specs = registry.tool_specs();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "calc");
        assert_eq!(specs[0].description, "A mock tool for testing");
        assert!(specs[0].parameters.is_object());
    }

    #[test]
    fn test_registry_register_shared() {
        let mut registry = ToolRegistry::new();
        let tool: SharedTool = Arc::new(MockTool::new("shared_tool"));
        registry.register_shared(tool).unwrap();

        assert!(registry.get("shared_tool").is_some());
    }

    #[test]
    fn test_registry_register_shared_duplicate_returns_error() {
        let mut registry = ToolRegistry::new();
        let tool1: SharedTool = Arc::new(MockTool::new("dup"));
        let tool2: SharedTool = Arc::new(MockTool::new("dup"));
        registry.register_shared(tool1).unwrap();

        let result = registry.register_shared(tool2);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_tool_execute_through_registry() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("mock")).unwrap();

        let tool = registry.get("mock").unwrap();
        let result = tool.execute_erased(&serde_json::json!({})).await.unwrap();
        assert_eq!(result.content, "mock result");
        assert!(!result.is_error);
    }
}
