mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use daimon::agent::Agent;
use daimon::error::{DaimonError, Result};
use daimon::hooks::{AgentHook, AgentState};
use daimon::memory::SlidingWindowMemory;
use daimon::model::Model;
use daimon::model::types::*;
use daimon::stream::ResponseStream;
use daimon::tool::{Tool, ToolCall, ToolOutput};

use common::MockModel;

struct Adder;

impl Tool for Adder {
    fn name(&self) -> &str {
        "adder"
    }
    fn description(&self) -> &str {
        "Adds two numbers a and b"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        })
    }
    async fn execute(&self, input: &serde_json::Value) -> Result<ToolOutput> {
        let a = input["a"].as_f64().unwrap_or(0.0);
        let b = input["b"].as_f64().unwrap_or(0.0);
        Ok(ToolOutput::text(format!("{}", a + b)))
    }
}

struct FailingTool;

impl Tool for FailingTool {
    fn name(&self) -> &str {
        "failing"
    }
    fn description(&self) -> &str {
        "Always fails"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object"})
    }
    async fn execute(&self, _input: &serde_json::Value) -> Result<ToolOutput> {
        Err(DaimonError::ToolExecution {
            tool: "failing".into(),
            message: "intentional failure".into(),
        })
    }
}

#[tokio::test]
async fn test_full_react_loop_with_tool() {
    let model = MockModel::new(vec![
        ChatResponse {
            message: Message::assistant_with_tool_calls(vec![ToolCall {
                id: "tc_1".into(),
                name: "adder".into(),
                arguments: serde_json::json!({"a": 10, "b": 32}),
            }]),
            stop_reason: StopReason::ToolUse,
            usage: Some(Usage::default()),
        },
        ChatResponse {
            message: Message::assistant("The answer is 42"),
            stop_reason: StopReason::EndTurn,
            usage: Some(Usage::default()),
        },
    ]);

    let agent = Agent::builder()
        .model(model)
        .system_prompt("You are a calculator assistant.")
        .tool(Adder)
        .build()
        .unwrap();

    let response = agent.prompt("What is 10 + 32?").await.unwrap();
    assert_eq!(response.text(), "The answer is 42");
    assert_eq!(response.iterations, 2);
}

#[tokio::test]
async fn test_multi_tool_calls_in_single_iteration() {
    let model = MockModel::new(vec![
        ChatResponse {
            message: Message::assistant_with_tool_calls(vec![
                ToolCall {
                    id: "tc_1".into(),
                    name: "adder".into(),
                    arguments: serde_json::json!({"a": 1, "b": 2}),
                },
                ToolCall {
                    id: "tc_2".into(),
                    name: "adder".into(),
                    arguments: serde_json::json!({"a": 3, "b": 4}),
                },
            ]),
            stop_reason: StopReason::ToolUse,
            usage: None,
        },
        ChatResponse {
            message: Message::assistant("3 and 7"),
            stop_reason: StopReason::EndTurn,
            usage: None,
        },
    ]);

    let response = Agent::builder()
        .model(model)
        .tool(Adder)
        .build()
        .unwrap()
        .prompt("add 1+2 and 3+4")
        .await
        .unwrap();

    assert_eq!(response.text(), "3 and 7");
    assert_eq!(response.iterations, 2);
}

#[tokio::test]
async fn test_failing_tool_recovers() {
    let model = MockModel::new(vec![
        ChatResponse {
            message: Message::assistant_with_tool_calls(vec![ToolCall {
                id: "tc_1".into(),
                name: "failing".into(),
                arguments: serde_json::json!({}),
            }]),
            stop_reason: StopReason::ToolUse,
            usage: None,
        },
        ChatResponse {
            message: Message::assistant("The tool failed, sorry."),
            stop_reason: StopReason::EndTurn,
            usage: None,
        },
    ]);

    let response = Agent::builder()
        .model(model)
        .tool(FailingTool)
        .build()
        .unwrap()
        .prompt("try it")
        .await
        .unwrap();

    assert_eq!(response.text(), "The tool failed, sorry.");
}

#[tokio::test]
async fn test_max_iterations_stops_loop() {
    struct AlwaysToolModel;
    impl Model for AlwaysToolModel {
        async fn generate(&self, _request: &ChatRequest) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message::assistant_with_tool_calls(vec![ToolCall {
                    id: "tc".into(),
                    name: "adder".into(),
                    arguments: serde_json::json!({"a": 1, "b": 1}),
                }]),
                stop_reason: StopReason::ToolUse,
                usage: None,
            })
        }
        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    let result = Agent::builder()
        .model(AlwaysToolModel)
        .tool(Adder)
        .max_iterations(2)
        .build()
        .unwrap()
        .prompt("infinite loop")
        .await;

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), DaimonError::MaxIterations(2)));
}

#[tokio::test]
async fn test_no_tools_simple_response() {
    let agent = Agent::builder()
        .model(MockModel::single_text("Hello!"))
        .build()
        .unwrap();

    let response = agent.prompt("hi").await.unwrap();
    assert_eq!(response.text(), "Hello!");
    assert_eq!(response.iterations, 1);
}

#[tokio::test]
async fn test_memory_window_persists_across_prompts() {
    let agent = Agent::builder()
        .model(MockModel::single_text("ok"))
        .memory(SlidingWindowMemory::new(4))
        .build()
        .unwrap();

    agent.prompt("first").await.unwrap();
    agent.prompt("second").await.unwrap();
    agent.prompt("third").await.unwrap();

    let messages = agent.memory().get_messages_erased().await.unwrap();
    assert_eq!(messages.len(), 4);
}

#[tokio::test]
async fn test_hooks_are_called() {
    struct CountingHook {
        iterations_started: AtomicUsize,
        iterations_ended: AtomicUsize,
        tool_calls_seen: AtomicUsize,
        tool_results_seen: AtomicUsize,
    }

    impl AgentHook for CountingHook {
        async fn on_iteration_start(&self, _state: &AgentState) -> Result<()> {
            self.iterations_started.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        async fn on_iteration_end(&self, _state: &AgentState) -> Result<()> {
            self.iterations_ended.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        async fn on_tool_call(&self, _call: &ToolCall) -> Result<()> {
            self.tool_calls_seen.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        async fn on_tool_result(&self, _call: &ToolCall, _result: &ToolOutput) -> Result<()> {
            self.tool_results_seen.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    let hook = Arc::new(CountingHook {
        iterations_started: AtomicUsize::new(0),
        iterations_ended: AtomicUsize::new(0),
        tool_calls_seen: AtomicUsize::new(0),
        tool_results_seen: AtomicUsize::new(0),
    });

    let model = MockModel::new(vec![
        ChatResponse {
            message: Message::assistant_with_tool_calls(vec![ToolCall {
                id: "tc_1".into(),
                name: "adder".into(),
                arguments: serde_json::json!({"a": 1, "b": 2}),
            }]),
            stop_reason: StopReason::ToolUse,
            usage: None,
        },
        ChatResponse {
            message: Message::assistant("done"),
            stop_reason: StopReason::EndTurn,
            usage: None,
        },
    ]);

    let hook_clone = hook.clone();

    struct HookForwarder {
        inner: Arc<CountingHook>,
    }

    impl AgentHook for HookForwarder {
        async fn on_iteration_start(&self, state: &AgentState) -> Result<()> {
            self.inner.on_iteration_start(state).await
        }
        async fn on_iteration_end(&self, state: &AgentState) -> Result<()> {
            self.inner.on_iteration_end(state).await
        }
        async fn on_tool_call(&self, call: &ToolCall) -> Result<()> {
            self.inner.on_tool_call(call).await
        }
        async fn on_tool_result(&self, call: &ToolCall, result: &ToolOutput) -> Result<()> {
            self.inner.on_tool_result(call, result).await
        }
    }

    Agent::builder()
        .model(model)
        .tool(Adder)
        .hooks(HookForwarder { inner: hook_clone })
        .build()
        .unwrap()
        .prompt("go")
        .await
        .unwrap();

    assert_eq!(hook.iterations_started.load(Ordering::SeqCst), 2);
    assert_eq!(hook.iterations_ended.load(Ordering::SeqCst), 2);
    assert_eq!(hook.tool_calls_seen.load(Ordering::SeqCst), 1);
    assert_eq!(hook.tool_results_seen.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn test_missing_tool_sends_error_back_to_model() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let cc = call_count.clone();

    struct ErrorCheckModel {
        count: Arc<AtomicUsize>,
    }

    impl Model for ErrorCheckModel {
        async fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
            let c = self.count.fetch_add(1, Ordering::SeqCst);
            if c == 0 {
                Ok(ChatResponse {
                    message: Message::assistant_with_tool_calls(vec![ToolCall {
                        id: "tc_1".into(),
                        name: "nonexistent".into(),
                        arguments: serde_json::json!({}),
                    }]),
                    stop_reason: StopReason::ToolUse,
                    usage: None,
                })
            } else {
                let tool_msg = request.messages.iter().rev().find(|m| m.role == Role::Tool);
                let content = tool_msg
                    .and_then(|m| m.content.as_deref())
                    .unwrap_or("no tool result");
                Ok(ChatResponse {
                    message: Message::assistant(format!("Error received: {content}")),
                    stop_reason: StopReason::EndTurn,
                    usage: None,
                })
            }
        }
        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    let agent = Agent::builder()
        .model(ErrorCheckModel { count: cc })
        .build()
        .unwrap();

    let response = agent.prompt("use unknown tool").await.unwrap();
    assert!(response.text().contains("not found"));
}

#[tokio::test]
async fn test_system_prompt_in_request() {
    struct InspectingModel;

    impl Model for InspectingModel {
        async fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
            let has_system = request.messages.iter().any(|m| m.role == Role::System);
            let system_content = request
                .messages
                .iter()
                .find(|m| m.role == Role::System)
                .and_then(|m| m.content.as_deref())
                .unwrap_or("none");

            Ok(ChatResponse {
                message: Message::assistant(format!(
                    "system={has_system},content={system_content}"
                )),
                stop_reason: StopReason::EndTurn,
                usage: None,
            })
        }
        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    let agent = Agent::builder()
        .model(InspectingModel)
        .system_prompt("Be helpful")
        .build()
        .unwrap();

    let response = agent.prompt("test").await.unwrap();
    assert!(response.text().contains("system=true"));
    assert!(response.text().contains("content=Be helpful"));
}

#[tokio::test]
async fn test_tool_specs_passed_to_model() {
    struct ToolInspectingModel;

    impl Model for ToolInspectingModel {
        async fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
            let tool_count = request.tools.len();
            let tool_names: Vec<&str> = request.tools.iter().map(|t| t.name.as_str()).collect();
            Ok(ChatResponse {
                message: Message::assistant(format!(
                    "tools={tool_count},names={}",
                    tool_names.join(",")
                )),
                stop_reason: StopReason::EndTurn,
                usage: None,
            })
        }
        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    let agent = Agent::builder()
        .model(ToolInspectingModel)
        .tool(Adder)
        .build()
        .unwrap();

    let response = agent.prompt("test").await.unwrap();
    assert!(response.text().contains("tools=1"));
    assert!(response.text().contains("names=adder"));
}
