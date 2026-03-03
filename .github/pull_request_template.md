## Summary

<!-- Describe your changes in 1-3 sentences. What does this PR do and why? -->

## Related Issues

<!-- Link to related issues or Jira tickets. Use "Closes #123" to auto-close. -->

## Changes

<!-- Bullet list of notable changes. -->

-

## Test Plan

<!-- How was this tested? What scenarios were covered? -->

- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed (describe below)

## Checklist

- [ ] `cargo fmt --check` passes
- [ ] `cargo clippy --features full -- -D warnings` passes
- [ ] `cargo test --no-default-features` passes
- [ ] `cargo test --features full` passes
- [ ] Coverage ≥90% (`cargo llvm-cov --no-default-features --fail-under-lines 90`)
- [ ] Rustdoc added for all new public items
- [ ] CHANGELOG.md updated (if user-facing change)

## AI Disclosure

<!-- If AI tools were used in producing this PR, you MUST disclose it per CONTRIBUTING.md. -->

- [ ] No AI tools were used in this PR
- [ ] AI tools were used (specify below)

<!-- If AI was used, state which tool/model and for what parts: -->
<!-- e.g., "Claude claude-sonnet-4-20250514 was used to generate unit tests for the memory module." -->
