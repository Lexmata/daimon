# DAIM-29 memory tier hardening — task report

## Finding 1 — `CoreMemory::render()` header-injection hardening

File: `daimon-core/src/core_memory.rs`

Chose option (a) from the brief: escape forged headers within `value` rather
than switching to a fenced/structurally-immune format, because it keeps the
rendered output human-readable Markdown (unchanged for the common case) and
requires no changes to callers or the `CoreMemoryBlock` type.

Added `escape_headers(&str) -> String`, called from `render_blocks` on each
block's `value` before formatting. It splits `value` on `'\n'` (not
`.lines()`, to preserve a trailing newline exactly rather than silently
dropping it) and prefixes any line that starts with `#` with a backslash
(`## foo` -> `\## foo`). The real block boundaries — the `"## {label}\n"`
prefix `render_blocks` itself emits — are never escaped, so splitting the
rendered string on `"\n## "` still yields exactly the real blocks, even if a
stored value tries to forge one.

Documented the rationale in a doc comment on `render_blocks` (and pointed to
it from `CoreMemory::render`'s doc comment).

Added test `render_blocks_escapes_forged_header_in_value`: builds two blocks
where the second's `value` contains an embedded `"## persona\n..."` line,
asserts the real boundaries are intact, the fake header line is escaped
(`\## persona`), and splitting the rendered string on the real boundary
marker (`"\n## "`) yields exactly the two legitimate blocks (not three).

## Finding 2 — `VectorArchivalMemory::search()` id round-trip

Files: `daimon-core/src/document.rs`, `daimon-core/src/vector_store.rs`
(doc-only), `src/retriever/in_memory_store.rs`,
`daimon-plugin-pgvector/src/store.rs`, `daimon-plugin-opensearch/src/store.rs`,
`src/memory/archival_memory.rs`, plus `docs/rag.md`, `docs/plugin-development.md`,
`docs/architecture.md`.

**Investigation**: traced the `VectorStore::query` contract in
`daimon-core/src/vector_store.rs` — it returns `Vec<ScoredDocument>`
(`daimon-core/src/document.rs`), and `ScoredDocument` had **no `id` field at
all**. This isn't just the adapter discarding an id it had — the trait's
result type genuinely had no place to carry one. Checked all three real
implementations:

- `PgVectorStore::query` (pgvector plugin): the SQL already did
  `SELECT id, content, metadata, ... AS score`, but the code never read the
  `id` column off the row before building `ScoredDocument`.
- `OpenSearchVectorStore::query` (opensearch plugin): every OpenSearch hit
  carries `_id` at the top level regardless of `_source` filtering, but the
  code only read `hit["_source"]["content"/"metadata"]` and `hit["_score"]`,
  never `hit["_id"]`.
- `InMemoryVectorStoreBackend::query`: iterated `entries.values()` (a
  `HashMap<String, StoredEntry>`), discarding the key (the real id) that was
  right there.

So in every backend the real id was available at the point of construction
and simply never made it into the result. Root-caused this to the
`ScoredDocument` type itself being incomplete.

**Fix**: added `pub id: String` to `ScoredDocument`, changed
`ScoredDocument::new` to `new(id: impl Into<String>, document: Document, score: f64)`,
and updated all four call sites (`in_memory_store.rs`, the two plugin
stores, plus the `daimon-core` unit test) to pass the real, already-known id
through. `VectorArchivalMemory::search` in `src/memory/archival_memory.rs`
now returns `scored.id` directly instead of `format!("result-{i}")`.

Doc comment added on `ScoredDocument` explaining the id-stability contract
(must be the same id passed to `upsert`, so callers can round-trip search
results into `delete`). Updated `docs/rag.md`, `docs/plugin-development.md`,
and `docs/architecture.md`'s field-table references to `ScoredDocument` for
consistency (plugin-development.md's example SQL now also selects/reads
`id`).

Test: `vector_archival_memory_search_ids_round_trip_to_delete` in
`src/memory/archival_memory.rs` — inserts one fact, searches, asserts the
returned id equals the id `insert` returned, deletes using that id, and
confirms both `count()` and a follow-up `search()` show it's gone. (Note:
the test's `FakeEmbedder` produces 1-dimensional embeddings, where cosine
similarity between any two same-signed scalars is always 1.0 regardless of
magnitude — so the test uses a single stored fact rather than asserting
rank order between two facts, to keep the assertion deterministic regardless
of the backing `HashMap`'s iteration order.)

Also updated `src/retriever/in_memory_store.rs`'s existing
`test_upsert_and_query` to assert `results[0].id == "a"`.

## Finding 3 — `InMemoryArchivalMemory::search` redundant lowercase allocation

File: `src/memory/archival_memory.rs`

Added `text_lower: String` to `StoredFact`, computed once in `insert()`
(`text.to_lowercase()`). `search()` now reads `fact.text_lower` directly
instead of calling `fact.text.to_lowercase()` per query per fact. No other
behavior change — the linear scan structure and per-term substring matching
are untouched, only the source of the lowercase text changed from
"recomputed every query" to "cached at insert time".

No new test was strictly required since existing search tests already
exercise the read path and pass; the change is a pure internal
optimization with an identical observable contract.

## Finding 4 — `InMemoryEpisodicMemory` unbounded growth

File: `src/memory/episodic_memory.rs`

Added `max_events: Option<usize>` field (defaults to `None` via `#[derive(Default)]`,
preserving the existing unbounded behavior for all current callers) and a
builder method `with_max_events(self, max_events: usize) -> Self`. `record()`
now evicts the oldest event(s) FIFO (`events.drain(0..excess)`) after
pushing, whenever `max_events` is set and exceeded — mirroring the
`InProcessBroker` retention pattern from DAIM-18.

Documented the default-unbounded / opt-in-cap behavior in a doc comment on
the struct, pointing at `with_max_events`.

Tests added:
- `with_max_events_evicts_oldest_on_overflow`: caps at 3, records 5 events
  (payload `i` 0..5), asserts only 3 remain and (newest-first, per `query`'s
  existing ordering) they are `[4, 3, 2]` — i.e. the oldest two were evicted.
- `without_max_events_growth_is_unbounded`: records 10 events with no cap
  set, asserts all 10 are still present (regression guard against
  accidentally changing the default).

## Verification

```
$ cargo test --workspace --features full 2>&1 | grep -E "^test result:|FAILED"
test result: ok. 528 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.51s
test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
test result: ok. 3 passed; 0 failed; 8 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
test result: ok. 35 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
test result: ok. 38 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.06s
test result: ok. 31 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 45 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.06s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 52 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.07s
test result: ok. 0 passed; 0 failed; 48 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 1 passed; 0 failed; 3 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 1 passed; 0 failed; 3 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 0 passed; 0 failed; 3 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 0 passed; 0 failed; 3 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 0 passed; 0 failed; 3 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
test result: ok. 0 passed; 0 failed; 5 ignored; 0 measured; 0 filtered out; finished in 0.00s

$ cargo clippy --workspace --features full --all-targets -- -D warnings
Finished `dev` profile [unoptimized + debuginfo] target(s) in 1m 12s
(clean, zero warnings)

$ cargo clippy --workspace --no-default-features --all-targets -- -D warnings
Finished `dev` profile [unoptimized + debuginfo] target(s) in 10.91s
(clean, zero warnings)

$ cargo fmt --all --check
FMT_OK
(exit 0, no diff)
```

## Files changed

- `daimon-core/src/core_memory.rs` — Finding 1 (escape_headers, render_blocks doc, test)
- `daimon-core/src/document.rs` — Finding 2 (ScoredDocument.id field + doc, test update)
- `daimon-core/src/vector_store.rs` — no functional change (VectorStore trait doc context only; verified, not modified)
- `src/retriever/in_memory_store.rs` — Finding 2 (thread real id through query, test update)
- `daimon-plugin-pgvector/src/store.rs` — Finding 2 (read `id` column)
- `daimon-plugin-opensearch/src/store.rs` — Finding 2 (read `hit["_id"]`)
- `src/memory/archival_memory.rs` — Finding 2 (VectorArchivalMemory::search uses real id) + Finding 3 (StoredFact.text_lower cache) + new round-trip test
- `src/memory/episodic_memory.rs` — Finding 4 (with_max_events, FIFO eviction, tests)
- `docs/rag.md`, `docs/plugin-development.md`, `docs/architecture.md` — doc updates reflecting the new `ScoredDocument.id` field

## Concerns

None outstanding. All four findings fixed with tests; full workspace test
suite, clippy (both feature configurations), and fmt are clean.

The `ScoredDocument::new` signature change (added a leading `id` parameter)
is a breaking API change for anyone outside this repo who calls it directly
against a pre-release version, but `ScoredDocument` is newly-added
surface area within an unreleased/pre-1.0 crate cycle per the repo's own
CHANGELOG conventions, and all in-repo call sites were updated.

## Follow-up remediation (adversarial review of commit 2193482)

Two gaps found in the prior pass, both fixed and folded into the same commit
via `git commit --amend`.

### Gap 1 — missing CHANGELOG breaking-change entry

`ScoredDocument` shipped in the already-tagged, already-published `v0.21.0`
(per this repo's process, crates.io publish happens before tagging), so the
`ScoredDocument::new` signature change and the new required `id` field are a
real breaking change for external consumers, not just an in-repo concern —
contrary to what the "Concerns" note above implied.

Added a `**Breaking:**` bullet to `CHANGELOG.md` under `## [Unreleased]`
(new `### Changed` section, following the exact style of the two existing
`**Breaking:**` bullets under the 0.20.0 entry), documenting: the new
required `id: String` field, the new leading `id: impl Into<String>`
parameter on `ScoredDocument::new`, why (so `VectorArchivalMemory::search()`
results round-trip to `delete()`), and what external `VectorStore`
implementors / direct callers must change. Also documented the
`escape_headers` indentation-tolerance fix (Gap 2) in the same entry since
both landed in this commit.

### Gap 2 — `escape_headers` leading-whitespace bypass

`daimon-core/src/core_memory.rs`'s `escape_headers` only matched exact
column-0 `line.starts_with('#')`, but CommonMark (and lenient-Markdown LLM
readers — exactly this function's threat model) treats up to 3 leading
spaces before `#` as still forming a valid ATX header. A stored block
`value` containing `"  ## persona\n<attacker text>"` sailed through
unescaped.

Fix: compute `leading_spaces = line.len() - line.trim_start_matches(' ').len()`
and escape whenever `leading_spaces <= 3 && line.trim_start_matches(' ').starts_with('#')`,
reusing the same backslash-prefix escaping the column-0 case already used
(`format!("\\{line}")` — prefixing the *original, unstripped* line so the
leading spaces are preserved and the backslash lands at column 0, which is
sufficient to break the "0-3 spaces then #" pattern regardless of where the
`#` ends up relative to the backslash). Lines with 4+ leading spaces
(CommonMark indented code blocks, not headers) are left untouched, per the
brief's own reasoning about not over-broadening the check.

Tests added:
- `render_blocks_escapes_indented_forged_header_in_value`: a value with
  1/2/3-leading-space forged `## ` lines, asserts each is escaped and only
  the two real block boundaries remain splittable on `"\n## "`.
- `render_blocks_does_not_escape_four_space_indented_code_block`: a
  4-space-indented `## ...` line is asserted to render untouched (verifies
  the fix doesn't over-broaden into legitimate code-block content).

### Verification (post-amendment)

```
$ cargo test --workspace --features full 2>&1 | grep -E "^test result:|FAILED"
(22 test-result lines, all "0 failed"; daimon-core lib now 37 passed, up from 35)

$ cargo clippy --workspace --features full --all-targets -- -D warnings
Finished `dev` profile [unoptimized + debuginfo] target(s) in 5.83s
(clean, zero warnings)

$ cargo clippy --workspace --no-default-features --all-targets -- -D warnings
Finished `dev` profile [unoptimized + debuginfo] target(s) in 3.33s
(clean, zero warnings)

$ cargo fmt --all --check
(exit 0, no diff)
```

Amended into commit `e9c053c` (same logical change as `2193482`, closing the
two gaps the first pass left open). Pre-commit hook (`cargo fmt --check`,
`cargo clippy --features full`) passed on the amend.

No outstanding concerns.
