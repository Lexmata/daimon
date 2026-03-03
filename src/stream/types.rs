//! Streaming event types for progressive agent responses.

use std::pin::Pin;

use futures::Stream;

use crate::error::Result;

/// An event emitted during a streaming agent response.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A chunk of generated text.
    TextDelta(String),
    /// A tool call is starting (name and ID known, arguments pending).
    ToolCallStart { id: String, name: String },
    /// A chunk of tool call arguments (JSON fragment).
    ToolCallDelta { id: String, arguments_delta: String },
    /// A tool call's arguments are complete and the tool will be executed.
    ToolCallEnd { id: String },
    /// A tool has produced its result.
    ToolResult {
        id: String,
        content: String,
        is_error: bool,
    },
    /// An error occurred during streaming (non-fatal; the stream may continue).
    Error(String),
    /// The stream is complete.
    Done,
}

/// A boxed, pinned stream of [`StreamEvent`] results.
pub type ResponseStream = Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>;
