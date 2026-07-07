//! A byte-accurate line buffer for chunked HTTP streams (SSE and NDJSON).
//!
//! The implementation lives in [`daimon_core::stream_util`] so the built-in
//! providers and the external provider crates share a single copy. This module
//! re-exports it under the historical path the built-in providers use.

pub(crate) use daimon_core::stream_util::LineBuffer;
