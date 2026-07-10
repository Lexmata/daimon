use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use crate::error::{DaimonError, Result};
use crate::model::types::ToolSpec;
use crate::tool::traits::{SharedTool, Tool};

/// A tool-spec snapshot tagged with the registry generation it was built from.
#[derive(Clone)]
struct CachedSpecs {
    generation: u64,
    specs: Arc<[ToolSpec]>,
}

/// A registry holding named tools that the agent can invoke.
///
/// Caches compiled JSON Schema validators and tool specs to avoid
/// repeated allocations in the hot path. The spec cache uses interior
/// mutability so it is populated lazily even through `&self` access.
#[derive(Default)]
pub struct ToolRegistry {
    tools: HashMap<String, SharedTool>,
    cached_specs: Mutex<Option<CachedSpecs>>,
    cached_validators: HashMap<String, Arc<jsonschema::Validator>>,
    generation: u64,
}

impl Clone for ToolRegistry {
    fn clone(&self) -> Self {
        let cached = self
            .cached_specs
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        Self {
            tools: self.tools.clone(),
            cached_specs: Mutex::new(cached),
            cached_validators: self.cached_validators.clone(),
            generation: self.generation,
        }
    }
}

impl ToolRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    fn invalidate_caches(&mut self) {
        *self
            .cached_specs
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner) = None;
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
    ///
    /// A `tool_name` that is not registered is reported as a validation
    /// error (`Some(...)`), never silently treated as valid — there is no
    /// schema to validate against, so the input cannot be trusted.
    pub fn validate_input(&self, tool_name: &str, input: &serde_json::Value) -> Option<String> {
        if let Some(validator) = self.cached_validators.get(tool_name) {
            return run_validator(validator, input);
        }

        let Some(tool) = self.tools.get(tool_name) else {
            return Some(format!("tool '{tool_name}' not found in registry"));
        };
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
    /// The cache is lazily populated on first call (including through
    /// `&self`, via interior mutability) and invalidated when tools are
    /// registered or unregistered.
    pub fn tool_specs(&self) -> Arc<[ToolSpec]> {
        {
            let cache = self
                .cached_specs
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if let Some(cached) = cache.as_ref()
                && cached.generation == self.generation
            {
                return Arc::clone(&cached.specs);
            }
        }

        // Build outside the lock: `parameters_schema()` may be non-trivial,
        // and a concurrent builder for the same generation is harmless (the
        // last writer wins with an identical snapshot).
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

        *self
            .cached_specs
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(CachedSpecs {
            generation: self.generation,
            specs: Arc::clone(&specs),
        });

        specs
    }

    /// Equivalent to [`tool_specs`](Self::tool_specs); retained for backward
    /// compatibility from when only the `&mut self` path populated the cache.
    pub fn tool_specs_mut(&mut self) -> Arc<[ToolSpec]> {
        self.tool_specs()
    }

    /// Ensures the tool_specs cache is populated. Call after registration is complete.
    pub fn warm_cache(&mut self) {
        self.tool_specs();
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
    fn test_validate_input_nonexistent_tool_is_error() {
        let registry = ToolRegistry::new();
        let err = registry.validate_input("missing", &serde_json::json!({}));
        assert!(
            err.is_some(),
            "a missing tool must be a validation error, not silently valid"
        );
        assert!(err.unwrap().contains("'missing' not found"));
    }

    #[test]
    fn test_tool_specs_cached_across_shared_calls() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("calc")).unwrap();

        let first = registry.tool_specs();
        let second = registry.tool_specs();
        assert!(
            Arc::ptr_eq(&first, &second),
            "repeated &self calls must return the same cached allocation"
        );
    }

    #[test]
    fn test_tool_specs_cache_invalidated_on_register() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("a")).unwrap();

        let before = registry.tool_specs();
        registry.register(MockTool::new("b")).unwrap();
        let after = registry.tool_specs();

        assert!(
            !Arc::ptr_eq(&before, &after),
            "registering a tool must invalidate the spec cache"
        );
        assert_eq!(before.len(), 1);
        assert_eq!(after.len(), 2);
    }

    #[test]
    fn test_tool_specs_cache_invalidated_on_unregister() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("a")).unwrap();
        registry.register(MockTool::new("b")).unwrap();

        let before = registry.tool_specs();
        assert!(registry.unregister("a"));
        let after = registry.tool_specs();

        assert!(!Arc::ptr_eq(&before, &after));
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].name, "b");
    }

    #[test]
    fn test_clone_preserves_cache() {
        let mut registry = ToolRegistry::new();
        registry.register(MockTool::new("a")).unwrap();

        let original = registry.tool_specs();
        let cloned = registry.clone();
        let from_clone = cloned.tool_specs();

        assert!(
            Arc::ptr_eq(&original, &from_clone),
            "cloning must carry the populated cache over"
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
