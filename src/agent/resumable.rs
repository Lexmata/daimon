//! Resumable agent execution backed by checkpoints.
//!
//! Provides [`Agent::prompt_resumable`], which saves the agent state after
//! each iteration. If the run is interrupted, it can be resumed from the
//! last checkpoint instead of replaying from scratch.

use std::sync::Arc;

use tokio_util::sync::CancellationToken;
use tracing::Instrument;

use crate::agent::runner::AgentResponse;
use crate::agent::Agent;
use crate::checkpoint::{CheckpointState, ErasedCheckpoint};
use crate::error::{DaimonError, Result};
use crate::hooks::AgentState;
use crate::model::types::{ChatRequest, Message, Usage};

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

        let (mut messages, start_iteration, mut total_usage) = if let Some(cp) = existing {
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
                    usage: Usage::default(),
                    cost: 0.0,
                });
            }
            tracing::info!(
                run_id,
                iteration = cp.iteration,
                messages = cp.messages.len(),
                "resuming from checkpoint"
            );
            (cp.messages, cp.iteration, Usage::default())
        } else {
            let history = self.memory.get_messages_erased().await?;

            let mut msgs = Vec::new();
            if let Some(system) = &self.system_prompt {
                msgs.push(Message::system(system));
            }
            msgs.extend(history);
            msgs.push(Message::user(input));

            self.memory.add_message_erased(Message::user(input)).await?;
            (msgs, 0, Usage::default())
        };

        let mut tool_specs_vec: Vec<crate::model::types::ToolSpec> =
            self.tools.tool_specs().to_vec();
        let mut iteration = start_iteration;

        loop {
            if cancel.is_cancelled() {
                checkpoint
                    .save_erased(&CheckpointState::new(
                        run_id,
                        messages.clone(),
                        iteration,
                    ))
                    .await?;
                return Err(DaimonError::Cancelled);
            }

            iteration += 1;
            let state = AgentState {
                iteration,
                max_iterations: self.max_iterations,
            };

            self.hooks.on_iteration_start_erased(&state).await?;

            let mut request = ChatRequest {
                messages: std::mem::take(&mut messages),
                tools: std::mem::take(&mut tool_specs_vec),
                temperature: self.temperature,
                max_tokens: self.max_tokens,
            };

            let result = {
                tracing::debug!(iteration, "calling model (resumable)");
                self.model
                    .generate_erased(&request)
                    .instrument(tracing::info_span!("model_generate", iteration))
                    .await
            };

            messages = std::mem::take(&mut request.messages);
            tool_specs_vec = std::mem::take(&mut request.tools);
            let response = result?;

            if let Some(ref usage) = response.usage {
                total_usage.accumulate(usage);
            }

            self.hooks.on_model_response_erased(&response).await?;

            if response.has_tool_calls() {
                let tool_calls = response.tool_calls().to_vec();
                let assistant_msg =
                    Message::assistant_with_tool_calls(tool_calls.clone());
                messages.push(assistant_msg.clone());
                self.memory.add_message_erased(assistant_msg).await?;

                let tool_results = self.execute_tools_parallel(&tool_calls).await;

                for (call, tool_result) in tool_calls.iter().zip(tool_results) {
                    let result_msg =
                        Message::tool_result(&call.id, &tool_result.content);
                    messages.push(result_msg.clone());
                    self.memory.add_message_erased(result_msg).await?;
                }

                checkpoint
                    .save_erased(&CheckpointState::new(
                        run_id,
                        messages.clone(),
                        iteration,
                    ))
                    .await?;

                self.hooks.on_iteration_end_erased(&state).await?;

                if iteration >= self.max_iterations {
                    return Err(DaimonError::MaxIterations(self.max_iterations));
                }

                continue;
            }

            let final_text = response.text().to_string();
            messages.push(response.message.clone());
            self.memory.add_message_erased(response.message).await?;

            let completed_state = CheckpointState::new(
                run_id,
                messages.clone(),
                iteration,
            )
            .mark_completed();
            checkpoint.save_erased(&completed_state).await?;

            self.hooks.on_iteration_end_erased(&state).await?;

            return Ok(AgentResponse {
                messages,
                final_text,
                iterations: iteration,
                usage: total_usage,
                cost: 0.0,
            });
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

        self.run_react_loop(messages, &CancellationToken::new())
            .await
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
