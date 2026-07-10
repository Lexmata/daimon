//! Agent evaluation and testing harness.
//!
//! Define [`EvalScenario`]s with expected outcomes, run them through an agent
//! via [`EvalRunner`], and collect [`EvalResult`]s with pass/fail, latency,
//! cost, and scoring.
//!
//! ```ignore
//! use daimon::eval::{EvalScenario, EvalRunner, Scorer};
//!
//! let suite = vec![
//!     EvalScenario::new("What is 2+2?").expect_contains("4"),
//!     EvalScenario::new("Say hello").expect_contains("hello"),
//! ];
//!
//! let results = EvalRunner::new(&agent).run(&suite).await;
//! for r in &results {
//!     println!("{}: {}", r.scenario_input, if r.passed { "PASS" } else { "FAIL" });
//! }
//! ```

mod runner;
mod scenario;
mod scoring;

pub use runner::{EvalResult, EvalRunner};
pub use scenario::EvalScenario;
pub use scoring::{CompiledRegex, Scorer};
