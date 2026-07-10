//! Shared, dependency-light helpers for chunked HTTP streaming and retry
//! backoff, used by every reqwest-based provider.
//!
//! These helpers were previously copy-pasted into the `daimon` crate's built-in
//! providers and into each external provider crate (`daimon-provider-gemini`,
//! `daimon-provider-azure`). They could not live in a shared location before
//! because the copies were tangled together with reqwest-specific header
//! parsing, and `daimon-core` deliberately does not depend on reqwest.
//!
//! Everything here is std-only. The reqwest-aware `Retry-After` header wrapper
//! stays in each provider crate and delegates to [`parse_retry_after_secs`].

use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Base delay for the first retry, in milliseconds.
pub const BASE_DELAY_MS: u64 = 100;

/// Upper bound on the exponential term (before jitter), in milliseconds.
///
/// Without a cap, `2^attempt` overflows and, well before that, produces
/// absurd multi-hour sleeps for large retry counts.
pub const MAX_DELAY_MS: u64 = 30_000;

/// Computes the delay to wait before the next retry attempt.
///
/// Uses exponential backoff (`BASE_DELAY_MS * 2^attempt`, capped at
/// `MAX_DELAY_MS`) combined with full jitter: the returned delay is a value in
/// `[0, backoff]`, which decorrelates retries across concurrent clients.
///
/// When the server supplied a `Retry-After` value, that instruction wins — the
/// delay is at least `retry_after` (clamped to `MAX_DELAY_MS`, the same ceiling
/// the exponential branch respects, so a hostile or buggy server cannot make us
/// sleep for hours), with a small jittered margin added so a fleet of clients
/// that all received the same `Retry-After` do not retry in lockstep.
pub fn backoff_delay(attempt: u32, retry_after: Option<Duration>) -> Duration {
    if let Some(ra) = retry_after {
        let capped = ra.min(Duration::from_millis(MAX_DELAY_MS));
        return capped + Duration::from_millis(full_jitter(BASE_DELAY_MS));
    }

    let exp = BASE_DELAY_MS
        .saturating_mul(2u64.saturating_pow(attempt))
        .min(MAX_DELAY_MS);
    Duration::from_millis(full_jitter(exp))
}

/// Returns a pseudo-random value in `[0, cap_ms]`.
///
/// We deliberately avoid pulling in the `rand` crate for a jitter source: the
/// sub-nanosecond component of the wall clock is more than enough entropy to
/// stagger retries, which is all full jitter needs.
pub fn full_jitter(cap_ms: u64) -> u64 {
    if cap_ms == 0 {
        return 0;
    }
    let entropy = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0);
    entropy % (cap_ms + 1)
}

/// Parses a `Retry-After` value expressed in integer seconds.
///
/// HTTP also permits an absolute-date form; providers only emit the delta form
/// on 429, so we parse the numeric-seconds form and return `None` for anything
/// we cannot interpret (including the HTTP-date form).
pub fn parse_retry_after_secs(value: &str) -> Option<u64> {
    value.trim().parse::<u64>().ok()
}

/// Accumulates raw stream bytes and yields complete newline-terminated lines.
///
/// Streaming responses arrive as arbitrary byte chunks: a single multi-byte
/// UTF-8 character (e.g. `é`, or an emoji) can be split across two chunks. The
/// naive `buffer.push_str(&String::from_utf8_lossy(&chunk))` decodes each raw
/// chunk in isolation, so the leading bytes of a split character decode to
/// U+FFFD (the replacement character) and the trailing bytes decode to another
/// — permanently corrupting the text.
///
/// `LineBuffer` accumulates raw bytes and only decodes complete,
/// newline-terminated lines. A `\n` (0x0A) byte can never appear inside a
/// multi-byte UTF-8 sequence, so decoding a whole line is always safe even when
/// the underlying chunk boundary fell mid-character.
///
/// Consumed bytes are tracked with a cursor rather than drained per line:
/// draining left-shifts every remaining byte, which turns a chunk carrying K
/// lines into O(K·chunk) work. The buffer compacts once the consumed prefix
/// grows past a threshold, so memory stays bounded over a long stream.
#[derive(Default)]
pub struct LineBuffer {
    buf: Vec<u8>,
    /// Bytes before this offset have already been yielded as lines.
    cursor: usize,
}

/// Consumed-prefix size that triggers compaction in [`LineBuffer::next_line`].
const COMPACT_THRESHOLD: usize = 8 * 1024;

impl LineBuffer {
    /// Creates an empty buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a raw byte chunk from the stream.
    pub fn push(&mut self, chunk: &[u8]) {
        self.buf.extend_from_slice(chunk);
    }

    /// Removes and returns the next complete line (trailing `\n` stripped),
    /// or `None` when no complete line is buffered yet.
    pub fn next_line(&mut self) -> Option<String> {
        let rel = self.buf[self.cursor..].iter().position(|&b| b == b'\n')?;
        let end = self.cursor + rel;
        // Decode straight from the slice; a complete line never splits a
        // code point, and the trailing newline is excluded by `end`.
        let line = String::from_utf8_lossy(&self.buf[self.cursor..end]).into_owned();
        self.cursor = end + 1;

        if self.cursor == self.buf.len() {
            self.buf.clear();
            self.cursor = 0;
        } else if self.cursor >= COMPACT_THRESHOLD {
            self.buf.drain(..self.cursor);
            self.cursor = 0;
        }

        Some(line)
    }

    /// Drains and returns whatever bytes remain after the last complete line.
    ///
    /// A stream can end without a trailing newline, leaving a final,
    /// otherwise-complete record buffered that `next_line` will never yield.
    /// Call this once at end-of-stream to recover it. Returns `None` when the
    /// buffer is empty or the remainder is only whitespace.
    pub fn take_remaining(&mut self) -> Option<String> {
        let bytes = std::mem::take(&mut self.buf);
        let start = std::mem::take(&mut self.cursor);
        if start >= bytes.len() {
            return None;
        }
        let trimmed = String::from_utf8_lossy(&bytes[start..]).trim().to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backoff_within_bounds() {
        for attempt in 0..8 {
            let base = (BASE_DELAY_MS * 2u64.pow(attempt)).min(MAX_DELAY_MS);
            for _ in 0..50 {
                let d = backoff_delay(attempt, None).as_millis() as u64;
                assert!(d <= base, "attempt {attempt}: {d} exceeded cap {base}");
            }
        }
    }

    #[test]
    fn test_backoff_caps_large_attempts() {
        // Would overflow if 2^attempt were computed without saturation.
        let d = backoff_delay(100, None);
        assert!(d.as_millis() as u64 <= MAX_DELAY_MS);
    }

    #[test]
    fn test_retry_after_is_honored_as_floor() {
        let ra = Duration::from_secs(5);
        let d = backoff_delay(0, Some(ra));
        assert!(
            d >= ra,
            "delay {d:?} should be at least the Retry-After {ra:?}"
        );
        // The jitter margin is bounded by the base delay.
        assert!(d < ra + Duration::from_millis(BASE_DELAY_MS + 1));
    }

    #[test]
    fn test_retry_after_clamped_to_cap() {
        // A server sending an absurd Retry-After must not make us sleep for
        // hours; the value is clamped to the same ceiling as the exponential
        // branch (plus the bounded jitter margin).
        let d = backoff_delay(0, Some(Duration::from_secs(999_999)));
        assert!(
            d <= Duration::from_millis(MAX_DELAY_MS + BASE_DELAY_MS + 1),
            "delay {d:?} exceeded the clamped ceiling"
        );
    }

    #[test]
    fn test_full_jitter_zero_cap() {
        assert_eq!(full_jitter(0), 0);
    }

    #[test]
    fn test_parse_retry_after_secs_numeric() {
        assert_eq!(parse_retry_after_secs("12"), Some(12));
        assert_eq!(parse_retry_after_secs("  12  "), Some(12));
        assert_eq!(parse_retry_after_secs("0"), Some(0));
    }

    #[test]
    fn test_parse_retry_after_secs_http_date_ignored() {
        assert_eq!(
            parse_retry_after_secs("Wed, 21 Oct 2015 07:28:00 GMT"),
            None
        );
    }

    #[test]
    fn test_parse_retry_after_secs_garbage() {
        assert_eq!(parse_retry_after_secs(""), None);
        assert_eq!(parse_retry_after_secs("abc"), None);
        assert_eq!(parse_retry_after_secs("1.5"), None);
        assert_eq!(parse_retry_after_secs("-3"), None);
    }

    #[test]
    fn test_ascii_lines() {
        let mut lb = LineBuffer::new();
        lb.push(b"hello\nworld\n");
        assert_eq!(lb.next_line().as_deref(), Some("hello"));
        assert_eq!(lb.next_line().as_deref(), Some("world"));
        assert_eq!(lb.next_line(), None);
    }

    #[test]
    fn test_incomplete_line_retained() {
        let mut lb = LineBuffer::new();
        lb.push(b"partial");
        assert_eq!(lb.next_line(), None);
        lb.push(b" line\n");
        assert_eq!(lb.next_line().as_deref(), Some("partial line"));
    }

    #[test]
    fn test_multibyte_char_split_across_chunks() {
        // "café\n" — the 'é' is two bytes (0xC3 0xA9). Split the stream right
        // in the middle of that character.
        let full = "café\n".as_bytes();
        let split = 4; // 'c','a','f','é'-first-byte
        let mut lb = LineBuffer::new();
        lb.push(&full[..split]);
        assert_eq!(lb.next_line(), None, "no complete line yet");
        lb.push(&full[split..]);
        let line = lb.next_line().expect("line completes after second chunk");
        assert_eq!(line, "café");
        assert!(!line.contains('\u{FFFD}'), "must not contain U+FFFD");
    }

    #[test]
    fn test_emoji_split_across_chunks() {
        // A 4-byte emoji split byte-by-byte across chunks must not corrupt.
        let full = "🎉done\n".as_bytes();
        let mut lb = LineBuffer::new();
        for b in full {
            lb.push(&[*b]);
        }
        let line = lb.next_line().expect("line completes");
        assert_eq!(line, "🎉done");
        assert!(!line.contains('\u{FFFD}'));
    }

    #[test]
    fn test_multiple_lines_one_chunk() {
        let mut lb = LineBuffer::new();
        lb.push(b"a\nb\nc");
        assert_eq!(lb.next_line().as_deref(), Some("a"));
        assert_eq!(lb.next_line().as_deref(), Some("b"));
        assert_eq!(lb.next_line(), None);
    }

    #[test]
    fn test_take_remaining_returns_unterminated_final_line() {
        // A stream that ends without a trailing newline leaves the last record
        // buffered; next_line never yields it, take_remaining recovers it.
        let mut lb = LineBuffer::new();
        lb.push(b"data: {\"ok\":true}");
        assert_eq!(lb.next_line(), None, "no complete line without a newline");
        assert_eq!(lb.take_remaining().as_deref(), Some("data: {\"ok\":true}"));
        assert_eq!(lb.take_remaining(), None, "buffer drained after take");
    }

    #[test]
    fn test_many_lines_one_chunk_and_compaction() {
        // Push enough newline-terminated lines in one chunk to cross the
        // compaction threshold mid-drain, with a trailing partial line that
        // must survive compaction intact.
        let mut data = Vec::new();
        let line_count = COMPACT_THRESHOLD / 8 + 16;
        for i in 0..line_count {
            data.extend_from_slice(format!("line-{i:04}\n").as_bytes());
        }
        data.extend_from_slice(b"partial");

        let mut lb = LineBuffer::new();
        lb.push(&data);
        for i in 0..line_count {
            assert_eq!(
                lb.next_line().as_deref(),
                Some(format!("line-{i:04}").as_str())
            );
        }
        assert_eq!(lb.next_line(), None);
        lb.push(b" tail\n");
        assert_eq!(lb.next_line().as_deref(), Some("partial tail"));
    }

    #[test]
    fn test_take_remaining_after_consumed_lines() {
        let mut lb = LineBuffer::new();
        lb.push(b"first\nleftover");
        assert_eq!(lb.next_line().as_deref(), Some("first"));
        assert_eq!(lb.take_remaining().as_deref(), Some("leftover"));
        assert_eq!(lb.take_remaining(), None);
        assert_eq!(lb.next_line(), None);
    }

    #[test]
    fn test_take_remaining_trims_and_none_on_empty() {
        let mut lb = LineBuffer::new();
        assert_eq!(lb.take_remaining(), None, "empty buffer yields None");
        lb.push(b"  \r\n");
        // The trailing "\r\n" completes a line; next_line consumes it.
        assert_eq!(lb.next_line().as_deref(), Some("  \r"));
        assert_eq!(lb.take_remaining(), None);
        lb.push(b"   ");
        assert_eq!(
            lb.take_remaining(),
            None,
            "whitespace-only remainder is None"
        );
    }
}
