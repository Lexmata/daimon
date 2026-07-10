use std::time::Instant;

use futures::future::join_all;

use crate::agent::Agent;
use crate::eval::scenario::EvalScenario;

/// The result of running a single evaluation scenario.
#[derive(Debug)]
pub struct EvalResult {
    /// The input that was sent to the agent.
    pub scenario_input: String,
    /// Whether all scorers passed.
    pub passed: bool,
    /// The agent's output text.
    pub output: String,
    /// Number of ReAct iterations used.
    pub iterations: usize,
    /// Wall-clock time for this scenario.
    pub latency: std::time::Duration,
    /// Estimated cost in USD (0.0 if no cost model).
    pub cost: f64,
    /// Details of individual scorer results.
    pub scorer_results: Vec<bool>,
    /// Error message if the agent failed.
    pub error: Option<String>,
}

/// Runs evaluation scenarios against an agent.
pub struct EvalRunner<'a> {
    agent: &'a Agent,
    concurrency: usize,
}

impl<'a> EvalRunner<'a> {
    pub fn new(agent: &'a Agent) -> Self {
        Self {
            agent,
            concurrency: 1,
        }
    }

    /// Sets the number of scenarios to run concurrently.
    pub fn concurrency(mut self, n: usize) -> Self {
        self.concurrency = n.max(1);
        self
    }

    /// Runs all scenarios and returns results, in scenario order.
    ///
    /// Scenarios run `concurrency` at a time: futures are lazy, so they must
    /// be polled together (`join_all`) to actually overlap — awaiting them
    /// one-by-one would run the whole suite serially regardless of the
    /// configured concurrency.
    pub async fn run(&self, scenarios: &[EvalScenario]) -> Vec<EvalResult> {
        let mut results = Vec::with_capacity(scenarios.len());
        for chunk in scenarios.chunks(self.concurrency) {
            results.extend(join_all(chunk.iter().map(|s| self.run_one(s))).await);
        }
        results
    }

    async fn run_one(&self, scenario: &EvalScenario) -> EvalResult {
        let start = Instant::now();
        let result = self.agent.prompt(&scenario.input).await;
        let latency = start.elapsed();

        match result {
            Ok(response) => {
                // Network-bound scorers (LLM judge, semantic similarity) run
                // concurrently; join_all preserves scorer order.
                let scorer_results = join_all(
                    scenario
                        .scorers
                        .iter()
                        .map(|scorer| scorer.evaluate(&response.final_text)),
                )
                .await;
                let passed = scorer_results.iter().all(|r| *r);

                EvalResult {
                    scenario_input: scenario.input.clone(),
                    passed,
                    output: response.final_text,
                    iterations: response.iterations,
                    latency,
                    cost: response.cost,
                    scorer_results,
                    error: None,
                }
            }
            Err(e) => EvalResult {
                scenario_input: scenario.input.clone(),
                passed: false,
                output: String::new(),
                iterations: 0,
                latency,
                cost: 0.0,
                scorer_results: Vec::new(),
                error: Some(e.to_string()),
            },
        }
    }
}
