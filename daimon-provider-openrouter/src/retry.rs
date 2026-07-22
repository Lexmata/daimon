//! Shared retry/backoff helpers for the reqwest-based HTTP providers.
//!
//! The reqwest-free policy (exponential backoff, full jitter, numeric
//! `Retry-After` parsing) lives in [`daimon_core::stream_util`] so every
//! provider — built-in and external — shares one copy. This module re-exports
//! [`backoff_delay`] and keeps the reqwest-aware [`parse_retry_after`] wrapper
//! that reads the header off a [`reqwest::header::HeaderMap`] and delegates the
//! actual parsing to the core helper.

use std::time::Duration;

pub(crate) use daimon_core::stream_util::backoff_delay;

/// Parses a `Retry-After` header expressed in integer seconds.
///
/// HTTP also permits an absolute-date form; providers only emit the delta form
/// on 429, so we parse seconds and ignore anything we cannot interpret.
pub(crate) fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(daimon_core::stream_util::parse_retry_after_secs)
        .map(Duration::from_secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_retry_after_seconds() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::RETRY_AFTER, "12".parse().unwrap());
        assert_eq!(parse_retry_after(&headers), Some(Duration::from_secs(12)));
    }

    #[test]
    fn test_parse_retry_after_absent() {
        let headers = reqwest::header::HeaderMap::new();
        assert_eq!(parse_retry_after(&headers), None);
    }
}
