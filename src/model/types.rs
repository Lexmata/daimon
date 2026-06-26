//! Core message and request types shared across all model providers.
//!
//! These types are defined in [`daimon_core`] and re-exported here for
//! backward compatibility.

pub use daimon_core::{ChatRequest, ChatResponse, Message, Role, StopReason, ToolSpec, Usage};
