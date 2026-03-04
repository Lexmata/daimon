//! Error types for the Daimon agent framework.
//!
//! All fallible operations in Daimon return [`Result<T>`], which is an alias
//! for `std::result::Result<T, DaimonError>`.
//!
//! The error type is defined in [`daimon_core`] and re-exported here so that
//! provider crates and the main framework share a single error enum.

pub use daimon_core::{DaimonError, Result};
