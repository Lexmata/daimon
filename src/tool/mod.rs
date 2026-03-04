//! Tool abstraction and registry.
//!
//! Implement [`Tool`] for callable functions the agent can invoke. Tools declare a JSON Schema
//! for parameters; the model uses this to generate valid arguments. Use [`ToolRegistry`] to
//! collect and look up tools by name.

pub mod registry;
pub mod retry;
mod traits;
pub mod types;

pub use registry::ToolRegistry;
pub use retry::ToolRetryPolicy;
pub use traits::{ErasedTool, SharedTool, Tool};
pub use types::{ToolCall, ToolChoice, ToolOutput};
