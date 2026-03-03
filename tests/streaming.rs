use std::sync::atomic::{AtomicUsize, Ordering};

use futures::StreamExt;

use daimon::agent::Agent;
use daimon::error::Result;
use daimon::model::Model;
use daimon::model::types::*;
use daimon::stream::{ResponseStream, StreamEvent};
use daimon::tool::{Tool, ToolOutput};

struct StreamingModel;

impl Model for StreamingModel {
    async fn generate(&self, _request: &ChatRequest) -> Result<ChatResponse> {
        Ok(ChatResponse {
            message: Message::assistant("not streaming"),
            stop_reason: StopReason::EndTurn,
            usage: None,
        })
    }

    async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
        let events = vec![
            Ok(StreamEvent::TextDelta("Hello ".into())),
            Ok(StreamEvent::TextDelta("world".into())),
            Ok(StreamEvent::Done),
        ];
        Ok(Box::pin(futures::stream::iter(events)))
    }
}

struct StreamingToolModel {
    call_count: AtomicUsize,
}

impl StreamingToolModel {
    fn new() -> Self {
        Self {
            call_count: AtomicUsize::new(0),
        }
    }
}

impl Model for StreamingToolModel {
    async fn generate(&self, _request: &ChatRequest) -> Result<ChatResponse> {
        unreachable!()
    }

    async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
        let count = self.call_count.fetch_add(1, Ordering::SeqCst);
        if count == 0 {
            let events = vec![
                Ok(StreamEvent::ToolCallStart {
                    id: "tc_1".into(),
                    name: "calculator".into(),
                }),
                Ok(StreamEvent::ToolCallDelta {
                    id: "tc_1".into(),
                    arguments_delta: r#"{"expr":"#.into(),
                }),
                Ok(StreamEvent::ToolCallDelta {
                    id: "tc_1".into(),
                    arguments_delta: r#""2+2"}"#.into(),
                }),
                Ok(StreamEvent::ToolCallEnd { id: "tc_1".into() }),
                Ok(StreamEvent::Done),
            ];
            Ok(Box::pin(futures::stream::iter(events)))
        } else {
            let events = vec![
                Ok(StreamEvent::TextDelta("The answer is 4.".into())),
                Ok(StreamEvent::Done),
            ];
            Ok(Box::pin(futures::stream::iter(events)))
        }
    }
}

struct CalcTool;

impl Tool for CalcTool {
    fn name(&self) -> &str {
        "calculator"
    }
    fn description(&self) -> &str {
        "Evaluate an expression"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {"expr": {"type": "string"}}})
    }
    async fn execute(&self, _input: &serde_json::Value) -> Result<ToolOutput> {
        Ok(ToolOutput::text("4"))
    }
}

#[tokio::test]
async fn test_streaming_text_events() {
    let agent = Agent::builder().model(StreamingModel).build().unwrap();

    let mut stream = agent.prompt_stream("hi").await.unwrap();
    let mut collected = String::new();
    let mut got_done = false;

    while let Some(event) = stream.next().await {
        match event.unwrap() {
            StreamEvent::TextDelta(text) => collected.push_str(&text),
            StreamEvent::Done => {
                got_done = true;
                break;
            }
            _ => {}
        }
    }

    assert_eq!(collected, "Hello world");
    assert!(got_done);
}

#[tokio::test]
async fn test_streaming_react_with_tools() {
    let agent = Agent::builder()
        .model(StreamingToolModel::new())
        .tool(CalcTool)
        .build()
        .unwrap();

    let mut stream = agent.prompt_stream("calc").await.unwrap();
    let mut event_types = Vec::new();

    while let Some(event) = stream.next().await {
        let name = match event.unwrap() {
            StreamEvent::TextDelta(_) => "text",
            StreamEvent::Done => "done",
            StreamEvent::ToolCallStart { .. } => "tool_start",
            StreamEvent::ToolCallDelta { .. } => "tool_delta",
            StreamEvent::ToolCallEnd { .. } => "tool_end",
            StreamEvent::ToolResult { .. } => "tool_result",
            StreamEvent::Error(_) => "error",
        };
        event_types.push(name);
    }

    // First model call: tool_start, tool_delta x2, tool_end
    // Then tool execution: tool_result
    // Second model call: text
    // Finally: done
    assert_eq!(
        event_types,
        vec![
            "tool_start",
            "tool_delta",
            "tool_delta",
            "tool_end",
            "tool_result",
            "text",
            "done"
        ]
    );
}

#[tokio::test]
async fn test_empty_stream() {
    struct EmptyStreamModel;
    impl Model for EmptyStreamModel {
        async fn generate(&self, _: &ChatRequest) -> Result<ChatResponse> {
            unreachable!()
        }
        async fn generate_stream(&self, _: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    let agent = Agent::builder().model(EmptyStreamModel).build().unwrap();

    let mut stream = agent.prompt_stream("hi").await.unwrap();
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event.unwrap());
    }
    // Empty model stream still yields Done from the ReAct wrapper
    assert_eq!(events.len(), 1);
    assert!(matches!(&events[0], StreamEvent::Done));
}

#[tokio::test]
async fn test_stream_event_ordering() {
    let agent = Agent::builder().model(StreamingModel).build().unwrap();

    let mut stream = agent.prompt_stream("hi").await.unwrap();
    let mut event_types = Vec::new();

    while let Some(event) = stream.next().await {
        let name = match event.unwrap() {
            StreamEvent::TextDelta(_) => "text",
            StreamEvent::Done => "done",
            StreamEvent::ToolCallStart { .. } => "tool_start",
            StreamEvent::ToolCallDelta { .. } => "tool_delta",
            StreamEvent::ToolCallEnd { .. } => "tool_end",
            StreamEvent::ToolResult { .. } => "tool_result",
            StreamEvent::Error(_) => "error",
        };
        event_types.push(name);
    }

    assert_eq!(event_types, vec!["text", "text", "done"]);
}
