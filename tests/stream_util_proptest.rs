//! Property-based tests for [`daimon_core::stream_util::LineBuffer`].
//!
//! `LineBuffer` is the shared primitive every reqwest-based provider's
//! SSE/NDJSON streaming parser is built on top of, and it consumes bytes
//! straight off the wire from a remote server — untrusted input by
//! definition. These tests don't assert anything about the *content* of a
//! response; they assert the parser cannot be made to panic or to corrupt
//! line boundaries no matter how the byte stream happens to be chunked by
//! the transport.

use daimon_core::stream_util::LineBuffer;
use proptest::prelude::*;

/// Feeds `bytes` into a fresh [`LineBuffer`] split into arbitrary chunks
/// (sized via `chunk_sizes`, cycled and clamped to at least 1 byte so the
/// loop always makes progress), draining every complete line after each
/// push and finishing with `take_remaining`.
fn feed_in_chunks(bytes: &[u8], chunk_sizes: &[usize]) -> Vec<String> {
    let mut buf = LineBuffer::new();
    let mut collected = Vec::new();
    let mut pos = 0;
    let mut sizes = chunk_sizes.iter().cycle();

    while pos < bytes.len() {
        let remaining = bytes.len() - pos;
        let take = sizes.next().copied().unwrap_or(1).clamp(1, remaining);
        buf.push(&bytes[pos..pos + take]);
        pos += take;
        while let Some(line) = buf.next_line() {
            collected.push(line);
        }
    }
    if let Some(rest) = buf.take_remaining() {
        collected.push(rest);
    }
    collected
}

proptest! {
    /// Regardless of how a known-good stream (arbitrary lines, each
    /// terminated by `\n`) is chopped into chunks, `LineBuffer` must
    /// reassemble exactly the original lines, in order, with no corruption
    /// and no panic.
    #[test]
    fn reassembles_lines_regardless_of_split_points(
        lines in prop::collection::vec("[^\\n\\r]{0,24}", 0..10),
        chunk_sizes in prop::collection::vec(1usize..7, 1..20),
    ) {
        let mut full = lines.join("\n");
        if !lines.is_empty() {
            full.push('\n');
        }
        let bytes = full.into_bytes();

        let collected = feed_in_chunks(&bytes, &chunk_sizes);
        prop_assert_eq!(collected, lines);
    }

    /// Multi-byte UTF-8 characters (including ones outside the ASCII/Latin
    /// range) split arbitrarily across chunk boundaries must round-trip
    /// exactly, with no panic — this is the property the hand-written
    /// `test_multibyte_char_split_across_chunks` /
    /// `test_emoji_split_across_chunks` unit tests pin for two fixed
    /// inputs; here the input and split points are both arbitrary.
    #[test]
    fn multibyte_lines_never_corrupt_or_panic(
        lines in prop::collection::vec(".{0,16}", 0..6),
        chunk_sizes in prop::collection::vec(1usize..5, 1..30),
    ) {
        let filtered: Vec<String> = lines
            .into_iter()
            .map(|l| l.replace(['\n', '\r'], ""))
            .collect();
        let mut full = filtered.join("\n");
        if !filtered.is_empty() {
            full.push('\n');
        }
        let bytes = full.into_bytes();

        // Exact round-trip equality is a stronger guarantee than merely
        // checking for the absence of U+FFFD: the generated `lines` can
        // legitimately already contain U+FFFD as ordinary input, so the
        // only sound check is that decoding never *introduces* corruption —
        // i.e. every line comes back byte-for-byte identical to what went
        // in, regardless of where chunk boundaries fell.
        let collected = feed_in_chunks(&bytes, &chunk_sizes);
        prop_assert_eq!(collected, filtered);
    }

    /// Completely arbitrary bytes — not necessarily valid UTF-8, not
    /// necessarily containing any newline — must never panic `LineBuffer`
    /// regardless of chunking. This is the untrusted-input case: a
    /// misbehaving or malicious server sending raw garbage over the wire.
    #[test]
    fn arbitrary_bytes_never_panic(
        chunks in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..32), 0..24),
    ) {
        let mut buf = LineBuffer::new();
        for chunk in &chunks {
            buf.push(chunk);
            while buf.next_line().is_some() {}
        }
        let _ = buf.take_remaining();
        // No panic occurred: that is the property under test.
    }
}
