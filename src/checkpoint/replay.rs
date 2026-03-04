//! Time-travel debugging: inspect and replay agent runs from checkpoints.

use crate::checkpoint::{CheckpointState, ErasedCheckpoint};
use crate::error::{DaimonError, Result};
use crate::model::types::{Message, Usage};
use crate::tool::ToolCall;

/// A single step in an agent's execution trace.
#[derive(Debug, Clone)]
pub struct TraceStep {
    /// The iteration this step corresponds to (1-based).
    pub iteration: usize,
    /// Messages at the start of this iteration.
    pub messages: Vec<Message>,
    /// Tool calls made during this iteration (empty if none).
    pub tool_calls: Vec<ToolCall>,
    /// The assistant's response text (if any).
    pub response_text: Option<String>,
    /// Token usage for this iteration.
    pub usage: Usage,
}

/// A complete execution trace reconstructed from checkpoint data.
#[derive(Debug, Clone)]
pub struct ExecutionTrace {
    /// The run ID this trace belongs to.
    pub run_id: String,
    /// Ordered steps from the execution.
    pub steps: Vec<TraceStep>,
    /// Whether the run completed successfully.
    pub completed: bool,
    /// Total iterations in the run.
    pub total_iterations: usize,
}

impl ExecutionTrace {
    /// Returns the final response text, if the run completed.
    pub fn final_text(&self) -> Option<&str> {
        self.steps
            .last()
            .and_then(|s| s.response_text.as_deref())
    }

    /// Returns the total number of tool calls across all steps.
    pub fn total_tool_calls(&self) -> usize {
        self.steps.iter().map(|s| s.tool_calls.len()).sum()
    }
}

/// Inspects a completed run by reconstructing its execution trace from
/// the checkpoint's message history.
pub async fn inspect_run(
    checkpoint: &dyn ErasedCheckpoint,
    run_id: &str,
) -> Result<ExecutionTrace> {
    let state = checkpoint
        .load_erased(run_id)
        .await?
        .ok_or_else(|| DaimonError::Other(format!("no checkpoint found for run '{run_id}'")))?;

    let trace = reconstruct_trace(&state);
    Ok(trace)
}

/// Lists all checkpointed runs with their metadata.
pub async fn list_runs(
    checkpoint: &dyn ErasedCheckpoint,
) -> Result<Vec<RunSummary>> {
    let run_ids = checkpoint.list_runs_erased().await?;
    let mut summaries = Vec::with_capacity(run_ids.len());

    for run_id in run_ids {
        if let Some(state) = checkpoint.load_erased(&run_id).await? {
            summaries.push(RunSummary {
                run_id: state.run_id,
                iteration: state.iteration,
                completed: state.completed,
                message_count: state.messages.len(),
                created_at: state.created_at,
            });
        }
    }

    Ok(summaries)
}

/// Summary of a checkpointed run.
#[derive(Debug, Clone)]
pub struct RunSummary {
    pub run_id: String,
    pub iteration: usize,
    pub completed: bool,
    pub message_count: usize,
    pub created_at: u64,
}

fn reconstruct_trace(state: &CheckpointState) -> ExecutionTrace {
    let mut steps = Vec::new();
    let mut current_messages = Vec::new();
    let mut iteration = 0;

    for msg in &state.messages {
        current_messages.push(msg.clone());

        match msg.role {
            crate::model::types::Role::Assistant => {
                iteration += 1;
                let tool_calls = msg.tool_calls.clone();
                let response_text = msg.content.clone();

                steps.push(TraceStep {
                    iteration,
                    messages: current_messages.clone(),
                    tool_calls,
                    response_text,
                    usage: Usage::default(),
                });
            }
            _ => {}
        }
    }

    ExecutionTrace {
        run_id: state.run_id.clone(),
        steps,
        completed: state.completed,
        total_iterations: iteration,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::InMemoryCheckpoint;
    use crate::checkpoint::Checkpoint;
    use crate::model::types::Message;

    #[tokio::test]
    async fn test_inspect_run() {
        let cp = InMemoryCheckpoint::new();
        let state = CheckpointState::new(
            "run-1",
            vec![
                Message::user("hello"),
                Message::assistant("hi there"),
            ],
            1,
        ).mark_completed();
        cp.save(&state).await.unwrap();

        let trace = inspect_run(&cp, "run-1").await.unwrap();
        assert_eq!(trace.run_id, "run-1");
        assert!(trace.completed);
        assert_eq!(trace.steps.len(), 1);
        assert_eq!(trace.final_text(), Some("hi there"));
    }

    #[tokio::test]
    async fn test_list_runs() {
        let cp = InMemoryCheckpoint::new();
        cp.save(&CheckpointState::new("a", vec![], 1)).await.unwrap();
        cp.save(&CheckpointState::new("b", vec![], 2)).await.unwrap();

        let runs = list_runs(&cp).await.unwrap();
        assert_eq!(runs.len(), 2);
    }

    #[tokio::test]
    async fn test_inspect_nonexistent() {
        let cp = InMemoryCheckpoint::new();
        let result = inspect_run(&cp, "nope").await;
        assert!(result.is_err());
    }
}
