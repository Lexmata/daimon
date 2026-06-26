use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{DaimonError, Result};
use crate::model::types::ToolSpec;
use crate::tool::traits::{SharedTool, Tool};

/// A registry holding named tools that the agent can invoke.
///
/// Caches compiled JSON Schema validators and tool specs to avoid
/// repeated allocations in the hot path.
#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, SharedTool>,
    cached_specs: Option<Arc<[ToolSpec]>>,
    cached_validators: HashMap<String, Arc<jsonschema::Validator>>,
    generation: u64,
    specs_generation: u64,
}

impl ToolRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    fn invalidate_caches(&mut self) {
        self.cached_specs = None;
        self.cached_validators.clear();
        self.generation = self.generation.wrapping_add(1);
    }

    /// Registers a tool by name. Returns an error if a tool with the same name already exists.
    pub fn register<T: Tool + 'static>(&mut self, tool: T) -> Result<()> {
        let name = tool.name().to_string();
        if self.tools.contains_key(&name) {
            return Err(DaimonError::DuplicateTool(name));
        }
        self.tools.insert(name, Arc::new(tool));
        self.invalidate_caches();
        Ok(())
    }

    /// Registers a pre-boxed shared tool. Returns an error if a tool with the same name already exists.
    pub fn register_shared(&mut self, tool: SharedTool) -> Result<()> {
        let name = tool.name().to_string();
        if self.tools.contains_key(&name) {
            return Err(DaimonError::DuplicateTool(name));
        }
        self.tools.insert(name, tool);
        self.invalidate_caches();
        Ok(())
    }

    /// Removes a tool by name. Returns `true` if the tool was present.
    pub fn unregister(&mut self, name: &str) -> bool {
        let removed = self.tools.remove(name).is_some();
        if removed {
            self.invalidate_caches();
        }
        removed
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

    /// Validates tool input against the tool's declared JSON Schema.
    ///
    /// Uses a cached compiled validator when available. Returns `None` if
    /// the input is valid, or a description of errors otherwise.
    pub fn validate_input(&self, tool_name: &str, input: &serde_json::Value) -> Option<String> {
        if let Some(validator) = self.cached_validators.get(tool_name) {
            return run_validator(validator, input);
        }

        let tool = self.tools.get(tool_name)?;
        let schema = tool.parameters_schema();

        let validator = match jsonschema::validator_for(&schema) {
            Ok(v) => v,
            Err(e) => {
                return Some(format!("invalid schema for tool '{tool_name}': {e}"));
            }
        };

        run_validator(&validator, input)
    }

    /// Pre-compiles and caches JSON Schema validators for all registered tools.
    ///
    /// Call this after all tools are registered and before entering the agent
    /// loop to avoid per-call validator compilation overhead.
    pub fn compile_validators(&mut self) {
        for (name, tool) in &self.tools {
            if !self.cached_validators.contains_key(name) {
                let schema = tool.parameters_schema();
                if let Ok(v) = jsonschema::validator_for(&schema) {
                    self.cached_validators.insert(name.clone(), Arc::new(v));
                }
            }
        }
    }

    /// Returns tool specs for the model. The result is cached and shared via
    /// `Arc`, so repeated calls return the same allocation.
    ///
    /// The cache is lazily populated on first call and invalidated when
    /// tools are registered/unregistered. For `&self` access without
    /// interior mutability, call [`warm_cache`](Self::warm_cache) after
    /// registration is complete to ensure the cache is pre-built.
    pub fn tool_specs(&self) -> Arc<[ToolSpec]> {
        if let Some(ref cached) = self.cached_specs
            && self.specs_generation == self.generation
        {
            return Arc::clone(cached);
        }

        let specs: Arc<[ToolSpec]> = self
            .tools
            .values()
            .map(|tool| ToolSpec {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                parameters: tool.parameters_schema(),
            })
            .collect::<Vec<_>>()
            .into();

        specs
    }

    /// Mutable version of `tool_specs` that stores the result in the cache.
    pub fn tool_specs_mut(&mut self) -> Arc<[ToolSpec]> {
        if let Some(ref cached) = self.cached_specs
            && self.specs_generation == self.generation
        {
            return Arc::clone(cached);
        }

        let specs: Arc<[ToolSpec]> = self
            .tools
            .values()
            .map(|tool| ToolSpec {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                parameters: tool.parameters_schema(),
            })
            .collect::<Vec<_>>()
            .into();

        self.cached_specs = Some(Arc::clone(&specs));
        self.specs_generation = self.generation;
        specs
    }

    /// Ensures the tool_specs cache is populated. Call after registration is complete.
    pub fn warm_cache(&mut self) {
        self.tool_specs_mut();
        self.compile_validators();
    }
}

fn run_validator(validator: &jsonschema::Validator, input: &serde_json::Value) -> Option<String> {
    let result = validator.validate(input);
    if result.is_ok() {
        None
    } else {
        let errors: Vec<String> = validator
            .iter_errors(input)
            .map(|e| {
                let path = e.instance_path().to_string();
                if path.is_empty() {
                    e.to_string()
                } else {
                    format!("{path}: {e}")
                }
            })
            .collect();
        Some(errors.join("; "))
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

    struct StrictTool;

    impl Tool for StrictTool {
        fn name(&self) -> &str {
            "strict"
        }
        fn description(&self) -> &str {
            "Tool with strict schema"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "minLength": 1 },
                    "count": { "type": "integer", "minimum": 0 }
                },
                "required": ["name", "count"],
                "additionalProperties": false
            })
        }
        async fn execute(&self, _input: &serde_json::Value) -> Result<ToolOutput> {
            Ok(ToolOutput::text("ok"))
        }
    }

    #[test]
    fn test_validate_input_valid() {
        let mut registry = ToolRegistry::new();
        registry.register(StrictTool).unwrap();

        let input = serde_json::json!({"name": "foo", "count": 5});
        assert!(registry.validate_input("strict", &input).is_none());
    }

    #[test]
    fn test_validate_input_missing_required_field() {
        let mut registry = ToolRegistry::new();
        registry.register(StrictTool).unwrap();

        let input = serde_json::json!({"name": "foo"});
        let err = registry.validate_input("strict", &input);
        assert!(err.is_some());
        assert!(err.unwrap().contains("count"));
    }

    #[test]
    fn test_validate_input_wrong_type() {
        let mut registry = ToolRegistry::new();
        registry.register(StrictTool).unwrap();

        let input = serde_json::json!({"name": "foo", "count": "not_a_number"});
        let err = registry.validate_input("strict", &input);
        assert!(err.is_some());
    }

    #[test]
    fn test_validate_input_additional_properties() {
        let mut registry = ToolRegistry::new();
        registry.register(StrictTool).unwrap();

        let input = serde_json::json!({"name": "foo", "count": 1, "extra": true});
        let err = registry.validate_input("strict", &input);
        assert!(err.is_some());
    }

    #[test]
    fn test_validate_input_constraint_violation() {
        let mut registry = ToolRegistry::new();
        registry.register(StrictTool).unwrap();

        let input = serde_json::json!({"name": "", "count": -1});
        let err = registry.validate_input("strict", &input);
        assert!(err.is_some());
    }

    #[test]
    fn test_validate_input_nonexistent_tool() {
        let registry = ToolRegistry::new();
        assert!(
            registry
                .validate_input("missing", &serde_json::json!({}))
                .is_none()
        );
    }

    #[test]
    fn test_validate_input_permissive_schema() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("mock")).unwrap();

        let input = serde_json::json!({"anything": "goes", "extra": 42});
        assert!(registry.validate_input("mock", &input).is_none());
    }
}
