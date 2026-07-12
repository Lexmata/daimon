//! Confirms the `daimon::model::local`/`daimon::model::ollama` facade
//! re-export paths (not `daimon_provider_local`'s own crate-internal paths,
//! which are already exercised by that crate's own test suite) actually
//! produce a type usable with `AgentBuilder::model()`.
//!
//! No live network call is made — construction plus a successful
//! `Agent::builder().model(...).build()` is sufficient to prove the facade
//! wiring (feature flag -> `pub mod` re-export -> `Model` trait bound) is
//! intact end to end.

#![cfg(any(feature = "ollama", feature = "local"))]

use daimon::agent::Agent;

#[cfg(feature = "ollama")]
#[test]
fn ollama_facade_reexport_builds_an_agent() {
    use daimon::model::ollama::Ollama;

    let model = Ollama::new("llama3.1").with_base_url("http://localhost:11434");
    let agent = Agent::builder().model(model).build();
    assert!(agent.is_ok());
}

#[cfg(feature = "local")]
#[test]
fn local_facade_reexport_builds_an_agent() {
    use daimon::model::local::OpenAiCompatible;

    let model = OpenAiCompatible::new("http://localhost:8000").with_model("my-model");
    let agent = Agent::builder().model(model).build();
    assert!(agent.is_ok());
}
