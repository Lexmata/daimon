//! Streaming response types for token-by-token or event-by-event model output.
//!
//! Use [`ResponseStream`] from [`Agent::prompt_stream`](crate::agent::Agent::prompt_stream) and
//! consume [`StreamEvent`] variants (e.g. `TextDelta`, `ToolCallStart`, `Done`).

mod types;

pub use types::{ResponseStream, StreamEvent};
