//! Streaming event types for progressive model responses.

use std::pin::Pin;

use futures::Stream;

use crate::error::Result;

/// An event emitted during a streaming model response.
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
    /// Token usage and cost for the current iteration.
    ///
    /// Emitted after each model invocation completes (once per ReAct
    /// iteration). During streaming, token counts are estimated from
    /// character length (~4 chars/token).
    Usage {
        iteration: usize,
        input_tokens: u32,
        output_tokens: u32,
        estimated_cost: f64,
    },
    /// An error occurred during streaming (non-fatal; the stream may continue).
    Error(String),
    /// The stream is complete.
    Done,
}

/// A boxed, pinned stream of [`StreamEvent`] results.
pub type ResponseStream = Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>;
