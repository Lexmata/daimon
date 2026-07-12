//! Confirms the `daimon::model::openai`/`daimon::model::anthropic` facade
//! re-export paths (not `daimon_provider_openai`'s or `daimon_provider_anthropic`'s
//! own crate-internal paths, which are already exercised by those crates' own test
//! suites) actually produce a type usable with `AgentBuilder::model()`.
//!
//! No live network call is made — construction plus a successful
//! `Agent::builder().model(...).build()` is sufficient to prove the facade
//! wiring (feature flag -> `pub mod` re-export -> `Model` trait bound) is
//! intact end to end.

#![cfg(any(feature = "openai", feature = "anthropic"))]

use daimon::agent::Agent;

#[cfg(feature = "openai")]
#[test]
fn openai_facade_reexport_builds_an_agent() {
    use daimon::model::openai::OpenAi;

    let model = OpenAi::with_api_key("gpt-4o", "test-key");
    let agent = Agent::builder().model(model).build();
    assert!(agent.is_ok());
}

#[cfg(feature = "anthropic")]
#[test]
fn anthropic_facade_reexport_builds_an_agent() {
    use daimon::model::anthropic::Anthropic;

    let model = Anthropic::with_api_key("claude-sonnet-4-20250514", "test-key");
    let agent = Agent::builder().model(model).build();
    assert!(agent.is_ok());
}
