//! Retry-After header parsing for the Gemini HTTP provider.
//!
//! The reqwest-free streaming and backoff helpers ([`LineBuffer`],
//! [`backoff_delay`]) live in [`daimon_core::stream_util`] and are shared by
//! every provider. This module re-exports them and adds the reqwest-aware
//! `Retry-After` wrapper, which cannot live in `daimon-core` because that crate
//! deliberately does not depend on reqwest.

use std::time::Duration;

pub(crate) use daimon_core::stream_util::{LineBuffer, backoff_delay};

/// Parses a `Retry-After` header expressed in integer seconds.
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
        assert_eq!(parse_retry_after(&reqwest::header::HeaderMap::new()), None);
    }
}
