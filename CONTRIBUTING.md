# Contributing to Daimon

Thank you for your interest in contributing to Daimon. This document outlines the standards and processes for contributing to the project.

## Getting Started

1. Clone the repository and check out the `develop` branch.
2. Install Rust 1.85+ (edition 2024).
3. Install development tools:
   ```bash
   cargo install cargo-commitlint cargo-llvm-cov cargo-deny
   ```
4. Link the pre-commit hook:
   ```bash
   ln -sf ../../scripts/pre-commit.sh .git/hooks/pre-commit
   ```

## Branch Workflow

All work targets the `develop` branch. The `main` branch is reserved for production releases.

- Create a feature branch from `develop`: `git checkout -b feature/your-feature develop`
- Open a pull request targeting `develop`.
- Direct commits and PRs to `main` are prohibited (except tagged releases).

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/) enforced by `cargo-commitlint`.

**Format:** `type(scope): description`

Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`, `release`.

- Use imperative mood ("Add feature" not "Added feature").
- Keep the subject line under 72 characters and at least 10 characters.
- Reference a Jira ticket when applicable: `feat(agent): BACK-123 Add cancellation support`.

## Code Standards

- Rust 2024 edition. No `unsafe` without justification.
- `cargo fmt` and `cargo clippy --features full -- -D warnings` must pass.
- No `unwrap()` or `expect()` in library code. Use `?` and `DaimonError`.
- All public items must have rustdoc comments.
- Feature flags for optional providers (`openai`, `anthropic`, `bedrock`).

## Testing

- Write tests for all new functionality.
- Run: `cargo test --no-default-features && cargo test --features full`
- Maintain ≥90% line coverage: `cargo llvm-cov --no-default-features --fail-under-lines 90`
- Use `#[tokio::test]` for async tests.

## Pull Request Process

1. Ensure all checks pass (fmt, clippy, test, coverage).
2. Provide a clear description of the change and its motivation.
3. Reference related issues or tickets.
4. Request review from a maintainer.

## AI-Assisted Contributions

Contributions produced in whole or in part by AI tools (such as Claude, GPT, Copilot, Cursor, or any other AI model or coding assistant) **must** include proper attribution. This is a firm requirement, not optional.

### Attribution Rules

1. **Git trailer required.** Every commit that contains AI-generated or AI-assisted code must include a `Co-authored-by` or `Assisted-by` trailer identifying the AI model used. Examples:

   ```
   feat(agent): Add streaming cancellation support

   Assisted-by: Claude (Anthropic, claude-sonnet-4-20250514)
   ```

   ```
   fix(bedrock): Handle throttling with exponential backoff

   Co-authored-by: Claude <noreply@anthropic.com>
   Assisted-by: Claude (Anthropic, claude-sonnet-4-20250514)
   ```

2. **Be specific about the model.** Use the model name and version/identifier when known (e.g., `claude-sonnet-4-20250514`, `gpt-4o-2024-08-06`, `GitHub Copilot`). If the exact version is unknown, use the best identifier available.

3. **Scope of disclosure.** If AI was used for only part of a commit (e.g., generating tests but not the implementation), note this in the commit body:

   ```
   test(memory): Add property-based tests for SlidingWindowMemory

   Tests generated with AI assistance; implementation is human-authored.

   Assisted-by: Claude (Anthropic, claude-sonnet-4-20250514)
   ```

4. **Pull request description.** When opening a PR that contains any AI-assisted commits, state this clearly in the PR description along with which AI tool was used.

5. **Review responsibility.** The human author is fully responsible for reviewing, understanding, and validating all AI-generated code before submitting it. AI attribution does not reduce the author's accountability for correctness, security, or adherence to project standards.

### Why We Require This

- **Transparency.** Reviewers and future maintainers deserve to know how code was produced.
- **Auditability.** For licensing, security, and compliance purposes, the provenance of code matters.
- **Accountability.** Clear attribution ensures the human contributor takes ownership of the final result.

## License

By contributing, you agree that your contributions will be dual-licensed under the MIT and Apache 2.0 licenses, the same as the project.
