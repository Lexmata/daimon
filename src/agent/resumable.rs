//! Resumable agent execution backed by checkpoints.
//!
//! Provides [`Agent::prompt_resumable`], which saves the agent state after
//! each iteration. If the run is interrupted, it can be resumed from the
//! last checkpoint instead of replaying from scratch.

use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::agent::Agent;
use crate::agent::runner::{AgentResponse, StepOutcome};
use crate::checkpoint::{CheckpointState, ErasedCheckpoint};
use crate::error::{DaimonError, Result};
use crate::model::types::{Message, Usage};

/// Runs a prompt with checkpoint-based persistence.
///
/// If a checkpoint exists for `run_id`, the agent resumes from the saved
/// iteration. Otherwise it starts fresh.
impl Agent {
    /// Send a prompt that can be resumed from a checkpoint.
    ///
    /// After each ReAct iteration the message state is checkpointed.
    /// If a previous checkpoint exists for `run_id`, the run resumes
    /// from the saved state.
    #[tracing::instrument(skip_all, fields(run_id = %run_id, input_len = input.len()))]
    pub async fn prompt_resumable(
        &self,
        input: &str,
        run_id: &str,
        checkpoint: &Arc<dyn ErasedCheckpoint>,
    ) -> Result<AgentResponse> {
        self.prompt_resumable_with_cancellation(
            input,
            run_id,
            checkpoint,
            &CancellationToken::new(),
        )
        .await
    }

    /// Resumable prompt with cancellation support.
    #[tracing::instrument(skip_all, fields(run_id = %run_id, input_len = input.len()))]
    pub async fn prompt_resumable_with_cancellation(
        &self,
        input: &str,
        run_id: &str,
        checkpoint: &Arc<dyn ErasedCheckpoint>,
        cancel: &CancellationToken,
    ) -> Result<AgentResponse> {
        let existing = checkpoint.load_erased(run_id).await?;

        let (mut messages, start_iteration, resuming, initial_usage, initial_cost) =
            if let Some(cp) = existing {
                if cp.completed {
                    tracing::info!(run_id, "checkpoint already completed, replaying result");
                    let final_text = cp
                        .messages
                        .last()
                        .and_then(|m| m.content.clone())
                        .unwrap_or_default();
                    return Ok(AgentResponse {
                        messages: cp.messages,
                        final_text,
                        iterations: cp.iteration,
                        usage: cp.usage,
                        cost: cp.cumulative_cost,
                    });
                }
                tracing::info!(
                    run_id,
                    iteration = cp.iteration,
                    messages = cp.messages.len(),
                    cumulative_cost = cp.cumulative_cost,
                    "resuming from checkpoint"
                );
                (
                    cp.messages,
                    cp.iteration,
                    true,
                    cp.usage,
                    cp.cumulative_cost,
                )
            } else {
                let actual_input = self.run_input_guardrails(input).await?;
                let history = self.memory.get_messages_erased().await?;

                let mut msgs = Vec::new();
                if let Some(system) = &self.system_prompt {
                    msgs.push(Message::system(system));
                }
                msgs.extend(history);
                msgs.push(Message::user(&actual_input));

                self.memory
                    .add_message_erased(Message::user(&actual_input))
                    .await?;
                (msgs, 0, false, Usage::default(), 0.0f64)
            };

        // On a fresh run, reset the cumulative cost. On a genuine resume,
        // reseed the tracker with the spend recorded in the checkpoint so that
        // `max_budget` is enforced against the combined pre- and post-resume
        // total, and the returned cost reflects the whole run.
        if let Some(ref tracker) = self.cost_tracker {
            if resuming {
                tracker.reseed(initial_cost);
            } else {
                tracker.reset();
            }
        }

        let mut tool_specs_vec: Vec<crate::model::types::ToolSpec> =
            self.tools.tool_specs().to_vec();
        let mut iteration = start_iteration;
        let mut total_usage = initial_usage;
        let mut total_cost = initial_cost;

        loop {
            if cancel.is_cancelled() {
                checkpoint
                    .save_erased(
                        &CheckpointState::new(run_id, messages.clone(), iteration)
                            .with_cost_usage(total_cost, total_usage.clone()),
                    )
                    .await?;
                return Err(DaimonError::Cancelled);
            }

            iteration += 1;

            let outcome = match self
                .run_iteration(
                    iteration,
                    &mut messages,
                    &mut tool_specs_vec,
                    &mut total_usage,
                    &mut total_cost,
                    cancel,
                )
                .await
            {
                Ok(outcome) => outcome,
                Err(DaimonError::Cancelled) => {
                    checkpoint
                        .save_erased(
                            &CheckpointState::new(run_id, messages.clone(), iteration)
                                .with_cost_usage(total_cost, total_usage.clone()),
                        )
                        .await?;
                    return Err(DaimonError::Cancelled);
                }
                Err(e) => return Err(e),
            };

            match outcome {
                StepOutcome::Continue => {
                    checkpoint
                        .save_erased(
                            &CheckpointState::new(run_id, messages.clone(), iteration)
                                .with_cost_usage(total_cost, total_usage.clone()),
                        )
                        .await?;
                    if iteration >= self.max_iterations {
                        return Err(DaimonError::MaxIterations(self.max_iterations));
                    }
                }
                StepOutcome::Final(final_text) => {
                    let mut response = AgentResponse {
                        messages,
                        final_text,
                        iterations: iteration,
                        usage: total_usage,
                        cost: total_cost,
                    };

                    // Run output guardrails BEFORE persisting a completed
                    // checkpoint. If a guardrail blocks (or errors), the run did
                    // not actually produce an acceptable answer, so the
                    // checkpoint must not be marked completed — otherwise a
                    // later resume would short-circuit and replay a response the
                    // guardrail rejected. On error we return without saving a
                    // completed checkpoint.
                    self.run_output_guardrails(&mut response).await?;

                    let completed_state =
                        CheckpointState::new(run_id, response.messages.clone(), iteration)
                            .with_cost_usage(response.cost, response.usage.clone())
                            .mark_completed();
                    checkpoint.save_erased(&completed_state).await?;

                    return Ok(response);
                }
            }
        }
    }

    /// Re-runs an agent from a previous checkpoint, optionally starting from
    /// a specific iteration.
    ///
    /// If `from_iteration` is `None`, replays from the beginning of the
    /// checkpoint's message history (but uses the current agent config).
    /// If `Some(n)`, truncates messages to iteration `n` and re-runs.
    ///
    /// This is useful for "what-if" debugging: modify the agent's tools,
    /// system prompt, or model, then replay a previous run to see how the
    /// outcome changes.
    #[tracing::instrument(skip_all, fields(run_id = %run_id))]
    pub async fn replay(
        &self,
        run_id: &str,
        checkpoint: &Arc<dyn ErasedCheckpoint>,
        from_iteration: Option<usize>,
    ) -> Result<AgentResponse> {
        let state = checkpoint
            .load_erased(run_id)
            .await?
            .ok_or_else(|| DaimonError::Other(format!("no checkpoint for run '{run_id}'")))?;

        let messages = if let Some(target) = from_iteration {
            truncate_to_iteration(state.messages, target)
        } else {
            state.messages
        };

        tracing::info!(
            run_id,
            message_count = messages.len(),
            from_iteration,
            "replaying from checkpoint"
        );

        let mut response = self
            .run_react_loop(messages, &CancellationToken::new())
            .await?;
        self.run_output_guardrails(&mut response).await?;
        Ok(response)
    }
}

/// Truncates messages to those produced up to (and including) the given
/// iteration. Iterations are counted by the number of assistant messages.
fn truncate_to_iteration(messages: Vec<Message>, target_iteration: usize) -> Vec<Message> {
    let mut result = Vec::new();
    let mut assistant_count = 0;
    for msg in messages {
        result.push(msg.clone());
        if msg.role == crate::model::types::Role::Assistant {
            assistant_count += 1;
            if assistant_count >= target_iteration {
                break;
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::InMemoryCheckpoint;
    use crate::error::Result;
    use crate::model::Model;
    use crate::model::types::{ChatRequest, ChatResponse, Message, StopReason, Usage};
    use crate::stream::ResponseStream;
    use crate::tool::{Tool, ToolOutput};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Echoes the last user/tool message back as the final answer in one turn.
    struct EchoModel;

    impl Model for EchoModel {
        async fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
            let last = request
                .messages
                .last()
                .and_then(|m| m.content.as_deref())
                .unwrap_or("empty");
            Ok(ChatResponse {
                message: Message::assistant(format!("Echo: {last}")),
                stop_reason: StopReason::EndTurn,
                usage: Some(Usage::default()),
            })
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    struct NoopTool;

    impl Tool for NoopTool {
        fn name(&self) -> &str {
            "noop"
        }
        fn description(&self) -> &str {
            "Does nothing"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(&self, _input: &serde_json::Value) -> Result<ToolOutput> {
            Ok(ToolOutput::text("ok"))
        }
    }

    /// First call emits a tool call (forcing a second iteration) and cancels the
    /// supplied token mid-run; the second call would return a final answer.
    struct CancelMidRunModel {
        token: CancellationToken,
        calls: AtomicUsize,
    }

    impl Model for CancelMidRunModel {
        async fn generate(&self, _request: &ChatRequest) -> Result<ChatResponse> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                self.token.cancel();
                Ok(ChatResponse {
                    message: Message::assistant_with_tool_calls(vec![crate::tool::ToolCall {
                        id: "c1".into(),
                        name: "noop".into(),
                        arguments: serde_json::json!({}),
                    }]),
                    stop_reason: StopReason::ToolUse,
                    usage: Some(Usage::default()),
                })
            } else {
                Ok(ChatResponse {
                    message: Message::assistant("done"),
                    stop_reason: StopReason::EndTurn,
                    usage: Some(Usage::default()),
                })
            }
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    /// First call emits a tool call (so the loop runs a second iteration where
    /// the budget pre-check trips), reporting non-zero token usage so a cost
    /// model records a positive cost.
    struct TwoStepModel {
        calls: AtomicUsize,
    }

    impl Model for TwoStepModel {
        async fn generate(&self, _request: &ChatRequest) -> Result<ChatResponse> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            let usage = Some(Usage {
                input_tokens: 1000,
                output_tokens: 1000,
                cached_tokens: 0,
            });
            if n == 0 {
                Ok(ChatResponse {
                    message: Message::assistant_with_tool_calls(vec![crate::tool::ToolCall {
                        id: "c1".into(),
                        name: "noop".into(),
                        arguments: serde_json::json!({}),
                    }]),
                    stop_reason: StopReason::ToolUse,
                    usage,
                })
            } else {
                Ok(ChatResponse {
                    message: Message::assistant("done"),
                    stop_reason: StopReason::EndTurn,
                    usage,
                })
            }
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    /// Always returns a final answer in one turn, reporting fixed non-zero
    /// token usage so a cost model records a positive per-call cost.
    struct FinalWithUsageModel;

    impl Model for FinalWithUsageModel {
        async fn generate(&self, _request: &ChatRequest) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message::assistant("done"),
                stop_reason: StopReason::EndTurn,
                usage: Some(Usage {
                    input_tokens: 1000,
                    output_tokens: 1000,
                    cached_tokens: 0,
                }),
            })
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    /// Emits a tool call (so the iteration is a `Continue`) reporting non-zero
    /// usage, then cancels the supplied token so the run stops with a saved,
    /// non-completed checkpoint carrying the spend from that first call.
    struct CancelAfterUsageModel {
        token: CancellationToken,
    }

    impl Model for CancelAfterUsageModel {
        async fn generate(&self, _request: &ChatRequest) -> Result<ChatResponse> {
            self.token.cancel();
            Ok(ChatResponse {
                message: Message::assistant_with_tool_calls(vec![crate::tool::ToolCall {
                    id: "c1".into(),
                    name: "noop".into(),
                    arguments: serde_json::json!({}),
                }]),
                stop_reason: StopReason::ToolUse,
                usage: Some(Usage {
                    input_tokens: 1000,
                    output_tokens: 1000,
                    cached_tokens: 0,
                }),
            })
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    fn cp() -> Arc<dyn ErasedCheckpoint> {
        Arc::new(InMemoryCheckpoint::new())
    }

    #[tokio::test]
    async fn test_run_to_completion_saves_completed_checkpoint() {
        let checkpoint = cp();
        let agent = Agent::builder().model(EchoModel).build().unwrap();

        let resp = agent
            .prompt_resumable("hello", "run-a", &checkpoint)
            .await
            .unwrap();
        assert_eq!(resp.final_text, "Echo: hello");

        let saved = checkpoint.load_erased("run-a").await.unwrap().unwrap();
        assert!(
            saved.completed,
            "a finished run must persist a completed checkpoint"
        );
    }

    #[tokio::test]
    async fn test_resume_completed_replays_stored_final_text() {
        // A completed checkpoint must be replayed verbatim: the stored final
        // answer is returned and the model is never re-invoked (EchoModel would
        // otherwise produce "Echo: ...").
        let checkpoint = cp();
        let state = CheckpointState::new(
            "run-b",
            vec![Message::user("hi"), Message::assistant("stored answer")],
            2,
        )
        .mark_completed();
        checkpoint.save_erased(&state).await.unwrap();

        let agent = Agent::builder().model(EchoModel).build().unwrap();
        let resp = agent
            .prompt_resumable("ignored input", "run-b", &checkpoint)
            .await
            .unwrap();

        assert_eq!(resp.final_text, "stored answer");
        assert_eq!(resp.iterations, 2);
    }

    #[tokio::test]
    async fn test_cancellation_mid_run_returns_cancelled_and_saves_checkpoint() {
        let token = CancellationToken::new();
        let checkpoint = cp();
        let agent = Agent::builder()
            .model(CancelMidRunModel {
                token: token.clone(),
                calls: AtomicUsize::new(0),
            })
            .tool(NoopTool)
            .build()
            .unwrap();

        let result = agent
            .prompt_resumable_with_cancellation("go", "run-c", &checkpoint, &token)
            .await;
        assert!(matches!(result, Err(DaimonError::Cancelled)));

        let saved = checkpoint.load_erased("run-c").await.unwrap();
        let saved = saved.expect("cancellation must persist a checkpoint to resume from");
        assert!(
            !saved.completed,
            "a cancelled run must not be marked completed"
        );
    }

    #[tokio::test]
    async fn test_resume_noncompleted_continues_without_readding_user_message() {
        // A non-completed checkpoint resumes from the stored messages and must
        // NOT re-run input guardrails or append the new prompt to memory — that
        // only happens on a fresh run.
        let checkpoint = cp();
        let state = CheckpointState::new("run-d", vec![Message::user("original")], 1);
        checkpoint.save_erased(&state).await.unwrap();

        let agent = Agent::builder().model(EchoModel).build().unwrap();
        let resp = agent
            .prompt_resumable("new input to ignore", "run-d", &checkpoint)
            .await
            .unwrap();

        assert_eq!(resp.final_text, "Echo: original");

        // A fresh run records the prompt as a User message in memory; a resume
        // must not. (The assistant reply is still persisted by run_iteration, so
        // we assert specifically that no User message was appended.)
        let mem = agent.memory.get_messages_erased().await.unwrap();
        assert!(
            !mem.iter()
                .any(|m| m.role == crate::model::types::Role::User),
            "resume must not append the new user message to memory"
        );
    }

    #[tokio::test]
    async fn test_budget_exceeded_propagates() {
        let checkpoint = cp();
        let agent = Agent::builder()
            .model(TwoStepModel {
                calls: AtomicUsize::new(0),
            })
            .tool(NoopTool)
            .cost_model(crate::cost::OpenAiCostModel)
            .max_budget(1e-9)
            .build()
            .unwrap();

        let result = agent.prompt_resumable("go", "run-e", &checkpoint).await;
        assert!(matches!(result, Err(DaimonError::BudgetExceeded { .. })));
    }

    #[tokio::test]
    async fn test_resume_restores_prior_cost_and_usage() {
        // Phase 1: run one iteration that spends, then get cancelled. The saved
        // (non-completed) checkpoint must carry the spend so a later, possibly
        // cross-process, resume can restore it rather than starting at zero.
        let token = CancellationToken::new();
        let checkpoint = cp();

        let agent1 = Agent::builder()
            .model(CancelAfterUsageModel {
                token: token.clone(),
            })
            .tool(NoopTool)
            .cost_model(crate::cost::OpenAiCostModel)
            .build()
            .unwrap();

        let r1 = agent1
            .prompt_resumable_with_cancellation("go", "run-cost", &checkpoint, &token)
            .await;
        assert!(matches!(r1, Err(DaimonError::Cancelled)));

        let saved = checkpoint.load_erased("run-cost").await.unwrap().unwrap();
        assert!(!saved.completed);
        let prior_cost = saved.cumulative_cost;
        assert!(
            prior_cost > 0.0,
            "the cancellation checkpoint must record the spend from iteration 1"
        );
        assert_eq!(
            saved.usage.input_tokens, 1000,
            "the checkpoint must record the token usage from iteration 1"
        );

        // Phase 2: resume with a fresh agent (simulating a new process) that
        // completes in one step, also spending. The returned cost must reflect
        // prior + new spend, not just the post-resume portion.
        let agent2 = Agent::builder()
            .model(FinalWithUsageModel)
            .cost_model(crate::cost::OpenAiCostModel)
            .build()
            .unwrap();

        let r2 = agent2
            .prompt_resumable("go", "run-cost", &checkpoint)
            .await
            .unwrap();

        assert!(
            r2.cost > prior_cost,
            "resumed cost {} must exceed the pre-resume spend {prior_cost}",
            r2.cost
        );
        // The finalizing call costs the same as the first, so total ~= 2x prior.
        assert!(
            (r2.cost - prior_cost * 2.0).abs() < 1e-6,
            "resumed cost {} should be prior ({prior_cost}) + new (~{prior_cost})",
            r2.cost
        );
        assert_eq!(
            r2.usage.input_tokens, 2000,
            "usage must aggregate pre- and post-resume tokens"
        );
        assert_eq!(r2.usage.output_tokens, 2000);

        // The completed checkpoint must also persist the full-run totals.
        let done = checkpoint.load_erased("run-cost").await.unwrap().unwrap();
        assert!(done.completed);
        assert!((done.cumulative_cost - r2.cost).abs() < 1e-6);
        assert_eq!(done.usage.input_tokens, 2000);
    }

    #[tokio::test]
    async fn test_budget_exceeded_accounts_for_pre_resume_spend() {
        // A checkpoint that already spent $0.02 is resumed with a budget of
        // $0.01. Because the tracker is reseeded with the prior spend, the very
        // first resumed iteration's budget pre-check must trip — before the
        // model is ever called. (A fresh run under the same budget would spend
        // once and finish, since the budget is only checked at iteration start;
        // this asserts the pre-resume spend is genuinely counted.)
        let checkpoint = cp();
        let state = CheckpointState::new("run-budget", vec![Message::user("orig")], 1)
            .with_cost_usage(
                0.02,
                Usage {
                    input_tokens: 1000,
                    output_tokens: 1000,
                    cached_tokens: 0,
                },
            );
        checkpoint.save_erased(&state).await.unwrap();

        let agent = Agent::builder()
            .model(FinalWithUsageModel)
            .cost_model(crate::cost::OpenAiCostModel)
            .max_budget(0.01)
            .build()
            .unwrap();

        let result = agent
            .prompt_resumable("go", "run-budget", &checkpoint)
            .await;
        match result {
            Err(DaimonError::BudgetExceeded { spent, limit }) => {
                assert!(
                    spent >= 0.02 - 1e-9,
                    "budget check must see the reseeded pre-resume spend, saw {spent}"
                );
                assert!((limit - 0.01).abs() < 1e-9);
            }
            other => panic!("expected BudgetExceeded, got {other:?}"),
        }
    }
}
