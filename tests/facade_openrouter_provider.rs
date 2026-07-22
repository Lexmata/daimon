//! Confirms the `daimon::model::openrouter` facade re-export path (not
//! `daimon_provider_openrouter`'s own crate-internal paths, which are already
//! exercised by that crate's own test suite) actually produces a type usable
//! with `AgentBuilder::model()`.
//!
//! No live network call is made — construction plus a successful
//! `Agent::builder().model(...).build()` is sufficient to prove the facade
//! wiring (feature flag -> `pub mod` re-export -> `Model` trait bound) is
//! intact end to end.

#![cfg(feature = "openrouter")]

use daimon::agent::Agent;

#[test]
fn openrouter_facade_reexport_builds_an_agent() {
    use daimon::model::openrouter::OpenRouter;

    let model = OpenRouter::with_api_key("openai/gpt-4o", "test-key");
    let agent = Agent::builder().model(model).build();
    assert!(agent.is_ok());
}
