use std::time::Instant;

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

    /// Runs all scenarios and returns results.
    pub async fn run(&self, scenarios: &[EvalScenario]) -> Vec<EvalResult> {
        let mut results = Vec::with_capacity(scenarios.len());
        for chunk in scenarios.chunks(self.concurrency) {
            let mut handles = Vec::with_capacity(chunk.len());
            for scenario in chunk {
                handles.push(self.run_one(scenario));
            }
            for handle in handles {
                results.push(handle.await);
            }
        }
        results
    }

    async fn run_one(&self, scenario: &EvalScenario) -> EvalResult {
        let start = Instant::now();
        let result = self.agent.prompt(&scenario.input).await;
        let latency = start.elapsed();

        match result {
            Ok(response) => {
                let mut scorer_results = Vec::with_capacity(scenario.scorers.len());
                for scorer in &scenario.scorers {
                    scorer_results.push(scorer.evaluate(&response.final_text).await);
                }
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
