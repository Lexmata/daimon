//! The ReAct agent loop implementation.

use tokio_util::sync::CancellationToken;
use tracing::Instrument;

use crate::agent::Agent;
use crate::cost::CostTracker;
use crate::error::{DaimonError, Result};
use crate::guardrails::{ErasedOutputGuardrail, GuardrailResult};
use crate::hooks::AgentState;
use crate::model::types::{ChatRequest, ChatResponse, Message, Usage};
use crate::stream::ResponseStream;
use crate::tool::{ErasedTool, ToolRetryPolicy};

/// Executes a tool with optional retry policy.
async fn execute_tool_with_retry(
    tool: &std::sync::Arc<dyn ErasedTool>,
    arguments: &serde_json::Value,
    retry_policy: Option<&ToolRetryPolicy>,
) -> crate::tool::ToolOutput {
    let max_attempts = retry_policy.map_or(1, |p| 1 + p.max_retries);

    for attempt in 0..max_attempts {
        match tool.execute_erased(arguments).await {
            Ok(output) if !output.is_error => return output,
            Ok(output) => {
                if attempt + 1 >= max_attempts {
                    return output;
                }
                if let Some(policy) = retry_policy {
                    if !policy.is_retryable(&output.content) {
                        return output;
                    }
                    let delay = policy.backoff.delay_for(attempt);
                    tracing::debug!(
                        tool = tool.name(),
                        attempt = attempt + 1,
                        delay_ms = delay.as_millis() as u64,
                        "retrying tool after error"
                    );
                    tokio::time::sleep(delay).await;
                }
            }
            Err(e) => {
                if attempt + 1 >= max_attempts {
                    return crate::tool::ToolOutput::error(e.to_string());
                }
                if let Some(policy) = retry_policy {
                    if !policy.is_retryable(&e.to_string()) {
                        return crate::tool::ToolOutput::error(e.to_string());
                    }
                    let delay = policy.backoff.delay_for(attempt);
                    tracing::debug!(
                        tool = tool.name(),
                        attempt = attempt + 1,
                        delay_ms = delay.as_millis() as u64,
                        "retrying tool after error"
                    );
                    tokio::time::sleep(delay).await;
                }
            }
        }
    }

    crate::tool::ToolOutput::error("max retries exhausted")
}

/// The result of an agent prompt, including the full message history,
/// final text, iteration count, and aggregated token usage.
#[derive(Debug, Clone)]
pub struct AgentResponse {
    /// The full message log for this prompt (system + history + user + all iterations).
    pub messages: Vec<Message>,
    /// The final text response from the model.
    pub final_text: String,
    /// How many model invocations were made.
    pub iterations: usize,
    /// Aggregated token usage across all iterations (if providers reported it).
    pub usage: Usage,
    /// Estimated cost in USD for this prompt (requires a CostModel on the agent).
    pub cost: f64,
}

impl AgentResponse {
    /// Get the final text response.
    pub fn text(&self) -> &str {
        &self.final_text
    }
}

/// Outcome of a single ReAct iteration ([`Agent::run_iteration`]).
pub(crate) enum StepOutcome {
    /// The loop should stop with this text — either the model produced a final
    /// answer or middleware short-circuited with a replacement response. Both
    /// terminate the loop identically (final message already pushed).
    Final(String),
    /// Tool calls were executed and appended; the loop should continue.
    Continue,
}

/// Cross-cutting budget check shared by the non-streaming ([`Agent::run_iteration`])
/// and streaming ([`Agent::prompt_stream`]) ReAct loops.
///
/// Returns `Some((spent, limit))` when a tracker and a limit are both present
/// and the tracker's cumulative cost has reached or exceeded the limit;
/// otherwise `None` (no tracker, no limit, or still under budget). Keeping this
/// in one place means a change to budget semantics can't silently drift between
/// the two loops — the exact divergence this branch closes.
fn budget_exceeded(tracker: Option<&CostTracker>, limit: Option<f64>) -> Option<(f64, f64)> {
    match (tracker, limit) {
        (Some(tracker), Some(limit)) => {
            let spent = tracker.cumulative_cost();
            (spent >= limit).then_some((spent, limit))
        }
        _ => None,
    }
}

/// Outcome of folding the output-guardrail sequence over a candidate text.
pub(crate) enum GuardrailDecision {
    /// No guardrail altered or rejected the text.
    Pass,
    /// A guardrail rejected the text; carries the block message.
    Block(String),
    /// One or more guardrails rewrote the text; carries the final rewrite.
    Transform(String),
}

/// Folds an output-guardrail sequence over `text`, short-circuiting on the
/// first `Block`. Each guard sees the text as rewritten by the guards before it
/// (matching the historical in-place fold in both loops).
///
/// This is the single source of truth for output-guardrail semantics. Both the
/// non-streaming [`Agent::run_output_guardrails`] (via
/// [`Agent::evaluate_output_guardrails`]) and the streaming loop route through
/// it, so the two paths can never diverge. It is a free function rather than a
/// `&self` method because the streaming loop moves cloned guardrail handles into
/// a `'static` stream and cannot borrow `self`.
async fn evaluate_output_guardrails_over(
    guardrails: &[std::sync::Arc<dyn ErasedOutputGuardrail>],
    text: &str,
    usage: &Usage,
) -> Result<GuardrailDecision> {
    let mut current = text.to_string();
    let mut transformed = false;
    for guard in guardrails {
        let chat_resp = ChatResponse {
            message: Message::assistant(&current),
            stop_reason: crate::model::types::StopReason::EndTurn,
            usage: Some(usage.clone()),
        };
        match guard.check_erased(&chat_resp).await? {
            GuardrailResult::Pass => {}
            GuardrailResult::Block(msg) => {
                return Ok(GuardrailDecision::Block(msg));
            }
            GuardrailResult::Transform(new_text) => {
                current = new_text;
                transformed = true;
            }
        }
    }
    Ok(if transformed {
        GuardrailDecision::Transform(current)
    } else {
        GuardrailDecision::Pass
    })
}

impl Agent {
    /// Send a text prompt to the agent and get a complete response.
    ///
    /// This runs the full ReAct loop: call model, check for tool calls,
    /// execute tools, append results, repeat until the model produces a
    /// final text response or max iterations is reached.
    #[tracing::instrument(skip_all, fields(input_len = input.len()))]
    pub async fn prompt(&self, input: &str) -> Result<AgentResponse> {
        let actual_input = self.run_input_guardrails(input).await?;
        let history = self.memory.get_messages_erased().await?;

        let mut messages = Vec::new();
        if let Some(system) = &self.system_prompt {
            messages.push(Message::system(system));
        }
        messages.extend(history);
        messages.push(Message::user(&actual_input));

        self.memory
            .add_message_erased(Message::user(&actual_input))
            .await?;

        let mut response = self
            .run_react_loop(messages, &CancellationToken::new())
            .await?;

        self.run_output_guardrails(&mut response).await?;
        Ok(response)
    }

    /// Send a text prompt with an explicit cancellation token.
    ///
    /// The agent loop will check the token before each iteration and abort
    /// with [`DaimonError::Cancelled`] if it has been cancelled.
    #[tracing::instrument(skip_all, fields(input_len = input.len()))]
    pub async fn prompt_with_cancellation(
        &self,
        input: &str,
        cancel: &CancellationToken,
    ) -> Result<AgentResponse> {
        let actual_input = self.run_input_guardrails(input).await?;
        let history = self.memory.get_messages_erased().await?;

        let mut messages = Vec::new();
        if let Some(system) = &self.system_prompt {
            messages.push(Message::system(system));
        }
        messages.extend(history);
        messages.push(Message::user(&actual_input));

        self.memory
            .add_message_erased(Message::user(&actual_input))
            .await?;

        let mut response = self.run_react_loop(messages, cancel).await?;
        self.run_output_guardrails(&mut response).await?;
        Ok(response)
    }

    /// Send pre-built messages to the agent and get a complete response.
    ///
    /// This bypasses the system prompt and memory loading -- you provide the
    /// full message history yourself. Useful for advanced scenarios like
    /// replaying conversations or injecting custom context.
    #[tracing::instrument(skip_all, fields(message_count = messages.len()))]
    pub async fn prompt_with_messages(&self, mut messages: Vec<Message>) -> Result<AgentResponse> {
        // Apply input guardrails to the final user message, if any, so this
        // entry point enforces the same policy as `prompt`.
        if let Some(last_user) = messages
            .iter_mut()
            .rev()
            .find(|m| m.role == crate::model::types::Role::User)
            && let Some(content) = &last_user.content
        {
            let checked = self.run_input_guardrails(content).await?;
            last_user.content = Some(checked);
        }

        let mut response = self
            .run_react_loop(messages, &CancellationToken::new())
            .await?;
        self.run_output_guardrails(&mut response).await?;
        Ok(response)
    }

    /// Core ReAct loop shared by all non-streaming prompt methods.
    ///
    /// Delegates each iteration to [`Agent::run_iteration`] so that budget
    /// enforcement, middleware, hooks, cost tracking, and memory persistence
    /// stay identical across `prompt`, `prompt_with_cancellation`,
    /// `prompt_with_messages`, `prompt_resumable*`, and `replay`. The resumable
    /// variant wraps the same helper with per-iteration checkpoint saves.
    pub(crate) async fn run_react_loop(
        &self,
        mut messages: Vec<Message>,
        cancel: &CancellationToken,
    ) -> Result<AgentResponse> {
        let mut tool_specs_vec: Vec<crate::model::types::ToolSpec> =
            self.tools.tool_specs().to_vec();
        let mut iteration = 0;
        let mut total_usage = Usage::default();
        let mut total_cost = 0.0f64;

        if let Some(ref tracker) = self.cost_tracker {
            tracker.reset();
        }

        loop {
            iteration += 1;
            let outcome = self
                .run_iteration(
                    iteration,
                    &mut messages,
                    &mut tool_specs_vec,
                    &mut total_usage,
                    &mut total_cost,
                    cancel,
                )
                .await?;

            match outcome {
                StepOutcome::Final(final_text) => {
                    return Ok(AgentResponse {
                        messages,
                        final_text,
                        iterations: iteration,
                        usage: total_usage,
                        cost: total_cost,
                    });
                }
                StepOutcome::Continue => {
                    if iteration >= self.max_iterations {
                        return Err(DaimonError::MaxIterations(self.max_iterations));
                    }
                }
            }
        }
    }

    /// Runs a single ReAct iteration: budget check → middleware → model call →
    /// usage/cost accounting → tool execution (or final response). Appends all
    /// produced messages to both `messages` (the working request log) and the
    /// agent's memory, exactly once, so every caller shares identical
    /// cross-cutting behavior.
    ///
    /// `iteration` is 1-based and already incremented by the caller. The caller
    /// is responsible for the `max_iterations` guard after a
    /// [`StepOutcome::Continue`] and for any checkpointing.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn run_iteration(
        &self,
        iteration: usize,
        messages: &mut Vec<Message>,
        tool_specs_vec: &mut Vec<crate::model::types::ToolSpec>,
        total_usage: &mut Usage,
        total_cost: &mut f64,
        cancel: &CancellationToken,
    ) -> Result<StepOutcome> {
        use crate::middleware::MiddlewareAction;

        if cancel.is_cancelled() {
            return Err(DaimonError::Cancelled);
        }

        if let Some((spent, limit)) = budget_exceeded(self.cost_tracker.as_ref(), self.max_budget) {
            return Err(DaimonError::BudgetExceeded { spent, limit });
        }

        let state = AgentState {
            iteration,
            max_iterations: self.max_iterations,
        };

        self.hooks.on_iteration_start_erased(&state).await?;

        let mut request = ChatRequest {
            messages: std::mem::take(messages),
            tools: std::mem::take(tool_specs_vec),
            temperature: self.temperature,
            max_tokens: self.max_tokens,
        };

        match self.middleware.run_on_request(&mut request).await? {
            MiddlewareAction::ShortCircuit(resp) => {
                *messages = std::mem::take(&mut request.messages);
                *tool_specs_vec = std::mem::take(&mut request.tools);
                let final_text = resp.text().to_string();
                messages.push(resp.message);
                // Middleware short-circuit terminates the loop like a final answer.
                return Ok(StepOutcome::Final(final_text));
            }
            MiddlewareAction::Continue => {}
        }

        let result = {
            tracing::debug!(iteration, "calling model");
            self.model
                .generate_erased(&request)
                .instrument(tracing::info_span!("model_generate", iteration))
                .await
        };

        *messages = std::mem::take(&mut request.messages);
        *tool_specs_vec = std::mem::take(&mut request.tools);

        let mut response = result?;

        if let Some(ref usage) = response.usage {
            tracing::debug!(
                input_tokens = usage.input_tokens,
                output_tokens = usage.output_tokens,
                "model usage"
            );
            total_usage.accumulate(usage);
            if let Some(ref tracker) = self.cost_tracker {
                *total_cost += tracker.record("default", usage);
            }
        }

        match self.middleware.run_on_response(&mut response).await? {
            MiddlewareAction::ShortCircuit(replaced) => {
                let final_text = replaced.text().to_string();
                messages.push(replaced.message);
                // Middleware short-circuit terminates the loop like a final answer.
                return Ok(StepOutcome::Final(final_text));
            }
            MiddlewareAction::Continue => {}
        }

        self.hooks.on_model_response_erased(&response).await?;

        if response.has_tool_calls() {
            let tool_calls = std::mem::take(&mut response.message.tool_calls);
            // Preserve any assistant text emitted alongside the tool calls.
            let assistant_msg = match response.message.content.take() {
                Some(text) if !text.is_empty() => {
                    Message::assistant_with_text_and_tool_calls(text, tool_calls.clone())
                }
                _ => Message::assistant_with_tool_calls(tool_calls.clone()),
            };
            self.memory
                .add_message_erased(assistant_msg.clone())
                .await?;
            messages.push(assistant_msg);

            let tool_results = self.execute_tools_parallel(&tool_calls).await;

            for (call, tool_result) in tool_calls.iter().zip(tool_results) {
                let result_msg = Message::tool_result(&call.id, &tool_result.content);
                self.memory.add_message_erased(result_msg.clone()).await?;
                messages.push(result_msg);
            }

            self.hooks.on_iteration_end_erased(&state).await?;
            return Ok(StepOutcome::Continue);
        }

        let final_text = response.text().to_string();
        self.memory
            .add_message_erased(response.message.clone())
            .await?;
        messages.push(response.message);

        self.hooks.on_iteration_end_erased(&state).await?;

        Ok(StepOutcome::Final(final_text))
    }

    /// Execute multiple tool calls concurrently with `tokio::spawn`.
    ///
    /// When `validate_tool_inputs` is enabled, each call's arguments are
    /// validated against the tool's JSON Schema before execution. Invalid
    /// inputs are returned as error messages to the model.
    pub(crate) async fn execute_tools_parallel(
        &self,
        tool_calls: &[crate::tool::ToolCall],
    ) -> Vec<crate::tool::ToolOutput> {
        use tokio::task::JoinSet;

        let mut join_set = JoinSet::new();
        let mut order = Vec::with_capacity(tool_calls.len());
        let validate = self.validate_tool_inputs;

        for (idx, call) in tool_calls.iter().enumerate() {
            let mut call_mut = call.clone();

            match self.middleware.run_on_tool_call(&mut call_mut).await {
                Ok(crate::middleware::MiddlewareAction::ShortCircuit(_)) => {
                    order.push(idx);
                    let idx_copy = idx;
                    let output = crate::tool::ToolOutput::text("skipped by middleware");
                    join_set.spawn(async move { (idx_copy, output) });
                    continue;
                }
                Ok(crate::middleware::MiddlewareAction::Continue) => {}
                Err(e) => {
                    order.push(idx);
                    let idx_copy = idx;
                    let output = crate::tool::ToolOutput::error(e.to_string());
                    join_set.spawn(async move { (idx_copy, output) });
                    continue;
                }
            }

            self.hooks.on_tool_call_erased(&call_mut).await.ok();

            if validate
                && let Some(errors) = self
                    .tools
                    .validate_input(&call_mut.name, &call_mut.arguments)
            {
                tracing::warn!(
                    tool = %call_mut.name,
                    id = %call_mut.id,
                    "tool input schema validation failed: {errors}"
                );
                let output = crate::tool::ToolOutput::error(format!(
                    "Invalid arguments for tool '{}': {errors}",
                    call_mut.name
                ));
                let err = DaimonError::SchemaValidation {
                    tool: call_mut.name.clone(),
                    errors: errors.clone(),
                };
                self.hooks.on_error_erased(&err).await.ok();
                order.push(idx);
                let idx_copy = idx;
                join_set.spawn(async move { (idx_copy, output) });
                continue;
            }

            let tool_opt = self.tools.get(&call_mut.name).cloned();
            let call_name = call_mut.name.clone();
            let call_id = call_mut.id.clone();
            let per_tool_policy = tool_opt.as_ref().and_then(|t| t.retry_policy());
            let agent_policy = self.tool_retry_policy.clone();
            let effective_policy = per_tool_policy.or(agent_policy);

            order.push(idx);

            let span = tracing::info_span!(
                "tool_execute",
                tool = %call_name,
                id = %call_id,
            );

            join_set.spawn(
                async move {
                    let result = match tool_opt {
                        Some(tool) => {
                            execute_tool_with_retry(
                                &tool,
                                &call_mut.arguments,
                                effective_policy.as_ref(),
                            )
                            .await
                        }
                        None => crate::tool::ToolOutput::error(format!(
                            "tool '{}' not found",
                            call_name
                        )),
                    };
                    (idx, result)
                }
                .instrument(span),
            );
        }

        let mut results: Vec<Option<crate::tool::ToolOutput>> =
            (0..tool_calls.len()).map(|_| None).collect();

        while let Some(Ok((idx, output))) = join_set.join_next().await {
            if let Some(call) = tool_calls.get(idx) {
                if output.is_error {
                    let err = DaimonError::ToolExecution {
                        tool: call.name.clone(),
                        message: output.content.clone(),
                    };
                    self.hooks.on_error_erased(&err).await.ok();
                } else {
                    self.hooks.on_tool_result_erased(call, &output).await.ok();
                }
            }
            results[idx] = Some(output);
        }

        results
            .into_iter()
            .map(|r| r.unwrap_or_else(|| crate::tool::ToolOutput::error("task panicked")))
            .collect()
    }

    /// Start a streaming response from the agent.
    ///
    /// Returns a [`ResponseStream`] that emits [`StreamEvent`](crate::stream::StreamEvent)s as the model
    /// generates its response. The stream runs the full ReAct loop: tool-call
    /// deltas are accumulated, tools are executed (with the agent's retry
    /// policy), and the model is re-invoked, all within the same stream.
    ///
    /// Like [`Agent::prompt`], the streaming loop persists every assistant and
    /// tool message to the agent's [`memory`](Agent::memory), fires lifecycle
    /// [`hooks`](crate::hooks), enforces `max_budget`, and applies input/output
    /// guardrails. Two differences from the non-streaming path are inherent to
    /// streaming: usage/cost is *estimated* from character counts (providers do
    /// not report exact token counts mid-stream), and output guardrails act on
    /// the message persisted to memory — text deltas are emitted live and
    /// cannot be retracted, so a `Block` verdict surfaces as a
    /// [`StreamEvent::Error`](crate::stream::StreamEvent::Error) rather than suppressing already-sent tokens.
    #[tracing::instrument(skip_all, fields(input_len = input.len()))]
    pub async fn prompt_stream(&self, input: &str) -> Result<ResponseStream> {
        use crate::hooks::AgentState;
        use crate::stream::StreamEvent;
        use futures::StreamExt;
        use std::collections::HashMap;

        let actual_input = self.run_input_guardrails(input).await?;
        let history = self.memory.get_messages_erased().await?;

        let mut messages = Vec::new();
        if let Some(system) = &self.system_prompt {
            messages.push(Message::system(system));
        }
        messages.extend(history);
        messages.push(Message::user(&actual_input));

        self.memory
            .add_message_erased(Message::user(&actual_input))
            .await?;

        let mut tool_specs_vec: Vec<crate::model::types::ToolSpec> =
            self.tools.tool_specs().to_vec();
        let max_iterations = self.max_iterations;

        let model = self.model.clone();
        let tools = self.tools.clone();
        let memory = self.memory.clone();
        let hooks = self.hooks.clone();
        let output_guardrails = self.output_guardrails.clone();
        let agent_retry_policy = self.tool_retry_policy.clone();
        let temperature = self.temperature;
        let max_tokens = self.max_tokens;
        let validate = self.validate_tool_inputs;
        let max_budget = self.max_budget;
        // Budget is enforced per prompt call (the non-streaming loop resets its
        // tracker each call), so a fresh per-stream tracker seeded from the same
        // cost model correctly bounds spend across this stream's iterations.
        let cost_tracker = self.cost_tracker.as_ref().map(|t| {
            std::sync::Arc::new(crate::cost::CostTracker::new(std::sync::Arc::clone(
                &t.cost_model,
            )))
        });

        let out_stream = async_stream::try_stream! {
            let mut iteration = 0;

            loop {
                iteration += 1;
                if iteration > max_iterations {
                    yield StreamEvent::Error(format!("max iterations ({max_iterations}) exceeded"));
                    yield StreamEvent::Done;
                    break;
                }

                if let Some((spent, limit)) =
                    budget_exceeded(cost_tracker.as_deref(), max_budget)
                {
                    yield StreamEvent::Error(format!(
                        "budget exceeded: spent ${spent:.4} of ${limit:.4}"
                    ));
                    yield StreamEvent::Done;
                    break;
                }

                let state = AgentState { iteration, max_iterations };
                hooks.on_iteration_start_erased(&state).await?;

                let input_chars: usize = messages.iter().map(|m| {
                    m.content.as_ref().map_or(0, |c| c.len())
                }).sum();

                let mut request = ChatRequest {
                    messages: std::mem::take(&mut messages),
                    tools: std::mem::take(&mut tool_specs_vec),
                    temperature,
                    max_tokens,
                };

                let stream_result = model.generate_stream_erased(&request).await;
                messages = std::mem::take(&mut request.messages);
                tool_specs_vec = std::mem::take(&mut request.tools);
                let mut inner = stream_result?;

                let mut pending_tool_calls: HashMap<String, (String, String)> = HashMap::new();
                let mut tool_call_order: Vec<String> = Vec::new();
                let mut had_tool_calls = false;
                let mut text_buf = String::new();

                while let Some(event) = inner.next().await {
                    let event = event?;
                    match &event {
                        StreamEvent::ToolCallStart { id, name } => {
                            if !pending_tool_calls.contains_key(id) {
                                tool_call_order.push(id.clone());
                            }
                            pending_tool_calls.insert(id.clone(), (name.clone(), String::new()));
                            had_tool_calls = true;
                            yield event;
                        }
                        StreamEvent::ToolCallDelta { id, arguments_delta } => {
                            if let Some((_name, args)) = pending_tool_calls.get_mut(id) {
                                args.push_str(arguments_delta);
                            }
                            yield event;
                        }
                        StreamEvent::ToolCallEnd { id } => {
                            yield StreamEvent::ToolCallEnd { id: id.clone() };
                        }
                        StreamEvent::TextDelta(text) => {
                            text_buf.push_str(text);
                            yield event;
                        }
                        StreamEvent::Done => {
                            // Don't yield Done yet if we have tool calls to process
                        }
                        _ => {
                            yield event;
                        }
                    }
                }

                let est_input_tokens = (input_chars / 4).max(1) as u32;
                let est_output_tokens = (text_buf.len() / 4).max(1) as u32;
                let est_usage = Usage {
                    input_tokens: est_input_tokens,
                    output_tokens: est_output_tokens,
                    cached_tokens: 0,
                };
                let estimated_cost = cost_tracker
                    .as_ref()
                    .map(|t| t.record("default", &est_usage))
                    .unwrap_or(0.0);

                yield StreamEvent::Usage {
                    iteration,
                    input_tokens: est_input_tokens,
                    output_tokens: est_output_tokens,
                    estimated_cost,
                };

                if !had_tool_calls {
                    // Synthesize the final assistant response so hooks and
                    // output guardrails see the same shape as the non-streaming path.
                    let synthesized = ChatResponse {
                        message: Message::assistant(&text_buf),
                        stop_reason: crate::model::types::StopReason::EndTurn,
                        usage: Some(est_usage.clone()),
                    };
                    hooks.on_model_response_erased(&synthesized).await?;

                    // Route through the same guardrail fold the non-streaming
                    // path uses, so budget/guardrail semantics never diverge.
                    let decision =
                        evaluate_output_guardrails_over(&output_guardrails, &text_buf, &est_usage)
                            .await?;

                    hooks.on_iteration_end_erased(&state).await?;

                    // Deltas are already emitted live and cannot be retracted, so
                    // a Block surfaces as an Error event; Transform/Pass differ only
                    // in which text is persisted to memory.
                    let final_text = match decision {
                        GuardrailDecision::Block(msg) => {
                            yield StreamEvent::Error(format!("output blocked by guardrail: {msg}"));
                            yield StreamEvent::Done;
                            break;
                        }
                        GuardrailDecision::Transform(new_text) => new_text,
                        GuardrailDecision::Pass => text_buf.clone(),
                    };

                    if !final_text.is_empty() {
                        let assistant = Message::assistant(&final_text);
                        memory.add_message_erased(assistant.clone()).await?;
                        messages.push(assistant);
                    }
                    yield StreamEvent::Done;
                    break;
                }

                // Rebuild tool calls in the order they were announced.
                let mut completed_calls = Vec::with_capacity(tool_call_order.len());
                for id in &tool_call_order {
                    if let Some((name, args_str)) = pending_tool_calls.get(id) {
                        let arguments: serde_json::Value = serde_json::from_str(args_str)
                            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                        completed_calls.push(crate::tool::ToolCall {
                            id: id.clone(),
                            name: name.clone(),
                            arguments,
                        });
                    }
                }

                let assistant_msg = if text_buf.is_empty() {
                    Message::assistant_with_tool_calls(completed_calls.clone())
                } else {
                    Message::assistant_with_text_and_tool_calls(&text_buf, completed_calls.clone())
                };
                memory.add_message_erased(assistant_msg.clone()).await?;
                messages.push(assistant_msg);

                for call in &completed_calls {
                    hooks.on_tool_call_erased(call).await.ok();

                    let validation_error = if validate {
                        tools.validate_input(&call.name, &call.arguments)
                    } else {
                        None
                    };

                    let tool_result = if let Some(errors) = validation_error {
                        crate::tool::ToolOutput::error(format!(
                            "Invalid arguments for tool '{}': {errors}",
                            call.name
                        ))
                    } else {
                        match tools.get(&call.name).cloned() {
                            Some(tool) => {
                                let policy = tool
                                    .retry_policy()
                                    .or_else(|| agent_retry_policy.clone());
                                execute_tool_with_retry(&tool, &call.arguments, policy.as_ref())
                                    .await
                            }
                            None => crate::tool::ToolOutput::error(format!(
                                "tool '{}' not found",
                                call.name
                            )),
                        }
                    };

                    if tool_result.is_error {
                        let err = DaimonError::ToolExecution {
                            tool: call.name.clone(),
                            message: tool_result.content.clone(),
                        };
                        hooks.on_error_erased(&err).await.ok();
                    } else {
                        hooks.on_tool_result_erased(call, &tool_result).await.ok();
                    }

                    yield StreamEvent::ToolResult {
                        id: call.id.clone(),
                        content: tool_result.content.clone(),
                        is_error: tool_result.is_error,
                    };

                    let result_msg = Message::tool_result(&call.id, &tool_result.content);
                    memory.add_message_erased(result_msg.clone()).await?;
                    messages.push(result_msg);
                }

                hooks.on_iteration_end_erased(&state).await?;
            }
        };

        Ok(Box::pin(out_stream))
    }

    /// Runs input guardrails. Returns the (potentially transformed) input or an error.
    pub(crate) async fn run_input_guardrails(&self, input: &str) -> Result<String> {
        let mut current = input.to_string();
        for guard in &self.input_guardrails {
            match guard.check_erased(&current, &[]).await? {
                crate::guardrails::GuardrailResult::Pass => {}
                crate::guardrails::GuardrailResult::Block(msg) => {
                    return Err(DaimonError::GuardrailBlocked(msg));
                }
                crate::guardrails::GuardrailResult::Transform(new_input) => {
                    current = new_input;
                }
            }
        }
        Ok(current)
    }

    /// Evaluates this agent's output guardrails over `text`, returning the
    /// [`GuardrailDecision`]. Thin `&self` wrapper over the shared
    /// [`evaluate_output_guardrails_over`] fold so the non-streaming path and
    /// the streaming path share identical guardrail semantics.
    pub(crate) async fn evaluate_output_guardrails(
        &self,
        text: &str,
        usage: &Usage,
    ) -> Result<GuardrailDecision> {
        evaluate_output_guardrails_over(&self.output_guardrails, text, usage).await
    }

    /// Runs output guardrails on the final response.
    pub(crate) async fn run_output_guardrails(&self, response: &mut AgentResponse) -> Result<()> {
        match self
            .evaluate_output_guardrails(&response.final_text, &response.usage)
            .await?
        {
            GuardrailDecision::Pass => {}
            GuardrailDecision::Block(msg) => return Err(DaimonError::GuardrailBlocked(msg)),
            GuardrailDecision::Transform(new_text) => response.final_text = new_text,
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::agent::Agent;
    use crate::error::Result;
    use crate::model::Model;
    use crate::model::types::*;
    use crate::stream::ResponseStream;
    use crate::tool::{Tool, ToolOutput};

    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct EchoModel;

    impl Model for EchoModel {
        async fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
            let last_msg = request
                .messages
                .last()
                .and_then(|m| m.content.as_deref())
                .unwrap_or("no input");

            Ok(ChatResponse {
                message: Message::assistant(format!("Echo: {last_msg}")),
                stop_reason: StopReason::EndTurn,
                usage: Some(Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    cached_tokens: 0,
                }),
            })
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    struct ToolCallingModel {
        call_count: AtomicUsize,
    }

    impl ToolCallingModel {
        fn new() -> Self {
            Self {
                call_count: AtomicUsize::new(0),
            }
        }
    }

    impl Model for ToolCallingModel {
        async fn generate(&self, _request: &ChatRequest) -> Result<ChatResponse> {
            let count = self.call_count.fetch_add(1, Ordering::SeqCst);
            if count == 0 {
                Ok(ChatResponse {
                    message: Message::assistant_with_tool_calls(vec![crate::tool::ToolCall {
                        id: "call_1".into(),
                        name: "adder".into(),
                        arguments: serde_json::json!({"a": 2, "b": 3}),
                    }]),
                    stop_reason: StopReason::ToolUse,
                    usage: Some(Usage {
                        input_tokens: 20,
                        output_tokens: 10,
                        cached_tokens: 0,
                    }),
                })
            } else {
                Ok(ChatResponse {
                    message: Message::assistant("The sum is 5"),
                    stop_reason: StopReason::EndTurn,
                    usage: Some(Usage {
                        input_tokens: 30,
                        output_tokens: 8,
                        cached_tokens: 0,
                    }),
                })
            }
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    struct InfiniteToolModel;

    impl Model for InfiniteToolModel {
        async fn generate(&self, _request: &ChatRequest) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message::assistant_with_tool_calls(vec![crate::tool::ToolCall {
                    id: "call_loop".into(),
                    name: "noop".into(),
                    arguments: serde_json::json!({}),
                }]),
                stop_reason: StopReason::ToolUse,
                usage: Some(Usage::default()),
            })
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    struct AdderTool;

    impl Tool for AdderTool {
        fn name(&self) -> &str {
            "adder"
        }
        fn description(&self) -> &str {
            "Adds two numbers"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            })
        }
        async fn execute(&self, input: &serde_json::Value) -> Result<ToolOutput> {
            let a = input["a"].as_i64().unwrap_or(0);
            let b = input["b"].as_i64().unwrap_or(0);
            Ok(ToolOutput::text(format!("{}", a + b)))
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

    #[tokio::test]
    async fn test_simple_prompt() {
        let agent = Agent::builder().model(EchoModel).build().unwrap();

        let response = agent.prompt("hello").await.unwrap();
        assert_eq!(response.text(), "Echo: hello");
        assert_eq!(response.iterations, 1);
    }

    #[tokio::test]
    async fn test_prompt_with_system_prompt() {
        let agent = Agent::builder()
            .model(EchoModel)
            .system_prompt("You are helpful")
            .build()
            .unwrap();

        let response = agent.prompt("test").await.unwrap();
        assert_eq!(response.text(), "Echo: test");
    }

    #[tokio::test]
    async fn test_prompt_with_tool_calling() {
        let agent = Agent::builder()
            .model(ToolCallingModel::new())
            .tool(AdderTool)
            .build()
            .unwrap();

        let response = agent.prompt("add 2 and 3").await.unwrap();
        assert_eq!(response.text(), "The sum is 5");
        assert_eq!(response.iterations, 2);
    }

    #[tokio::test]
    async fn test_usage_aggregation() {
        let agent = Agent::builder()
            .model(ToolCallingModel::new())
            .tool(AdderTool)
            .build()
            .unwrap();

        let response = agent.prompt("add 2 and 3").await.unwrap();
        assert_eq!(response.usage.input_tokens, 50); // 20 + 30
        assert_eq!(response.usage.output_tokens, 18); // 10 + 8
    }

    #[tokio::test]
    async fn test_max_iterations_exceeded() {
        let agent = Agent::builder()
            .model(InfiniteToolModel)
            .tool(NoopTool)
            .max_iterations(3)
            .build()
            .unwrap();

        let result = agent.prompt("loop forever").await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            crate::error::DaimonError::MaxIterations(3)
        ));
    }

    #[tokio::test]
    async fn test_missing_tool_returns_error_to_model() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        struct MissingToolModel {
            call_count: Arc<AtomicUsize>,
        }

        impl Model for MissingToolModel {
            async fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
                let count = self.call_count.fetch_add(1, Ordering::SeqCst);
                if count == 0 {
                    Ok(ChatResponse {
                        message: Message::assistant_with_tool_calls(vec![crate::tool::ToolCall {
                            id: "call_1".into(),
                            name: "nonexistent".into(),
                            arguments: serde_json::json!({}),
                        }]),
                        stop_reason: StopReason::ToolUse,
                        usage: None,
                    })
                } else {
                    let last_content = request
                        .messages
                        .last()
                        .and_then(|m| m.content.as_deref())
                        .unwrap_or("");
                    Ok(ChatResponse {
                        message: Message::assistant(format!("Got error: {last_content}")),
                        stop_reason: StopReason::EndTurn,
                        usage: None,
                    })
                }
            }

            async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
                Ok(Box::pin(futures::stream::empty()))
            }
        }

        let agent = Agent::builder()
            .model(MissingToolModel {
                call_count: call_count_clone,
            })
            .build()
            .unwrap();

        let response = agent.prompt("call nonexistent").await.unwrap();
        assert!(response.text().contains("not found"));
    }

    #[tokio::test]
    async fn test_memory_persists_across_prompts() {
        let agent = Agent::builder().model(EchoModel).build().unwrap();

        agent.prompt("first message").await.unwrap();
        agent.prompt("second message").await.unwrap();

        let messages = agent.memory.get_messages_erased().await.unwrap();
        assert_eq!(messages.len(), 4); // user, assistant, user, assistant
    }

    #[tokio::test]
    async fn test_tool_call_messages_saved_to_memory() {
        let agent = Agent::builder()
            .model(ToolCallingModel::new())
            .tool(AdderTool)
            .build()
            .unwrap();

        agent.prompt("add").await.unwrap();

        let messages = agent.memory.get_messages_erased().await.unwrap();
        // user + assistant(tool_calls) + tool_result + assistant(final)
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].role, crate::model::types::Role::User);
        assert!(!messages[1].tool_calls.is_empty());
        assert_eq!(messages[2].role, crate::model::types::Role::Tool);
        assert_eq!(messages[3].role, crate::model::types::Role::Assistant);
    }

    #[tokio::test]
    async fn test_prompt_with_messages() {
        let agent = Agent::builder().model(EchoModel).build().unwrap();

        let response = agent
            .prompt_with_messages(vec![Message::user("custom")])
            .await
            .unwrap();
        assert_eq!(response.text(), "Echo: custom");
    }

    #[tokio::test]
    async fn test_cancellation() {
        let cancel = tokio_util::sync::CancellationToken::new();
        cancel.cancel();

        let agent = Agent::builder().model(EchoModel).build().unwrap();
        let result = agent.prompt_with_cancellation("hi", &cancel).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            crate::error::DaimonError::Cancelled
        ));
    }

    struct StrictAdderTool;

    impl Tool for StrictAdderTool {
        fn name(&self) -> &str {
            "adder"
        }
        fn description(&self) -> &str {
            "Adds two numbers"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "a": { "type": "integer" },
                    "b": { "type": "integer" }
                },
                "required": ["a", "b"]
            })
        }
        async fn execute(&self, input: &serde_json::Value) -> Result<ToolOutput> {
            let a = input["a"].as_i64().unwrap_or(0);
            let b = input["b"].as_i64().unwrap_or(0);
            Ok(ToolOutput::text(format!("{}", a + b)))
        }
    }

    struct InvalidThenValidModel {
        call_count: AtomicUsize,
    }

    impl InvalidThenValidModel {
        fn new() -> Self {
            Self {
                call_count: AtomicUsize::new(0),
            }
        }
    }

    impl Model for InvalidThenValidModel {
        async fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
            let count = self.call_count.fetch_add(1, Ordering::SeqCst);
            if count == 0 {
                Ok(ChatResponse {
                    message: Message::assistant_with_tool_calls(vec![crate::tool::ToolCall {
                        id: "call_1".into(),
                        name: "adder".into(),
                        arguments: serde_json::json!({"a": "not_a_number", "b": 3}),
                    }]),
                    stop_reason: StopReason::ToolUse,
                    usage: None,
                })
            } else if count == 1 {
                let last = request
                    .messages
                    .last()
                    .and_then(|m| m.content.as_deref())
                    .unwrap_or("");
                assert!(
                    last.contains("Invalid arguments"),
                    "model should see validation error: {last}"
                );
                Ok(ChatResponse {
                    message: Message::assistant_with_tool_calls(vec![crate::tool::ToolCall {
                        id: "call_2".into(),
                        name: "adder".into(),
                        arguments: serde_json::json!({"a": 2, "b": 3}),
                    }]),
                    stop_reason: StopReason::ToolUse,
                    usage: None,
                })
            } else {
                Ok(ChatResponse {
                    message: Message::assistant("The sum is 5"),
                    stop_reason: StopReason::EndTurn,
                    usage: None,
                })
            }
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    #[tokio::test]
    async fn test_schema_validation_rejects_invalid_input() {
        let agent = Agent::builder()
            .model(InvalidThenValidModel::new())
            .tool(StrictAdderTool)
            .build()
            .unwrap();

        let response = agent.prompt("add 2 and 3").await.unwrap();
        assert_eq!(response.text(), "The sum is 5");
        assert_eq!(response.iterations, 3);
    }

    #[tokio::test]
    async fn test_schema_validation_disabled_allows_invalid_input() {
        struct AlwaysInvalidModel;

        impl Model for AlwaysInvalidModel {
            async fn generate(&self, _request: &ChatRequest) -> Result<ChatResponse> {
                Ok(ChatResponse {
                    message: Message::assistant_with_tool_calls(vec![crate::tool::ToolCall {
                        id: "call_1".into(),
                        name: "adder".into(),
                        arguments: serde_json::json!({"a": "string", "b": "string"}),
                    }]),
                    stop_reason: StopReason::ToolUse,
                    usage: None,
                })
            }
            async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
                Ok(Box::pin(futures::stream::empty()))
            }
        }

        let agent = Agent::builder()
            .model(AlwaysInvalidModel)
            .tool(StrictAdderTool)
            .validate_tool_inputs(false)
            .max_iterations(1)
            .build()
            .unwrap();

        let result = agent.prompt("test").await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            crate::error::DaimonError::MaxIterations(1)
        ));
    }

    // ---- Shared-core parity tests (DAIM-1) ----

    /// A model whose stream emits a couple of text deltas then completes.
    struct StreamingTextModel;

    impl Model for StreamingTextModel {
        async fn generate(&self, _request: &ChatRequest) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message::assistant("hi there"),
                stop_reason: StopReason::EndTurn,
                usage: None,
            })
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            use crate::stream::StreamEvent;
            let events = vec![
                Ok(StreamEvent::TextDelta("hi ".into())),
                Ok(StreamEvent::TextDelta("there".into())),
                Ok(StreamEvent::Done),
            ];
            Ok(Box::pin(futures::stream::iter(events)))
        }
    }

    struct BlockingInputGuardrail;

    impl crate::guardrails::InputGuardrail for BlockingInputGuardrail {
        async fn check(
            &self,
            _input: &str,
            _messages: &[Message],
        ) -> Result<crate::guardrails::GuardrailResult> {
            Ok(crate::guardrails::GuardrailResult::Block("nope".into()))
        }
    }

    #[tokio::test]
    async fn test_streaming_persists_to_memory() {
        use crate::stream::StreamEvent;
        use futures::StreamExt;

        let agent = Agent::builder().model(StreamingTextModel).build().unwrap();

        let mut stream = agent.prompt_stream("hello").await.unwrap();
        while let Some(ev) = stream.next().await {
            if matches!(ev.unwrap(), StreamEvent::Done) {
                break;
            }
        }

        // The user message and the assembled assistant message must both be saved,
        // so a following turn sees the full context (regression: streaming used to
        // persist only the user message).
        let messages = agent.memory.get_messages_erased().await.unwrap();
        assert_eq!(messages.len(), 2, "expected user + assistant in memory");
        assert_eq!(messages[0].role, crate::model::types::Role::User);
        assert_eq!(messages[1].role, crate::model::types::Role::Assistant);
        assert_eq!(messages[1].content.as_deref(), Some("hi there"));
    }

    #[tokio::test]
    async fn test_streaming_enforces_input_guardrail() {
        let agent = Agent::builder()
            .model(StreamingTextModel)
            .input_guardrail(BlockingInputGuardrail)
            .build()
            .unwrap();

        let result = agent.prompt_stream("hello").await;
        assert!(matches!(
            result.err(),
            Some(crate::error::DaimonError::GuardrailBlocked(_))
        ));
    }

    #[tokio::test]
    async fn test_cancellation_variant_enforces_input_guardrail() {
        let cancel = tokio_util::sync::CancellationToken::new();
        let agent = Agent::builder()
            .model(EchoModel)
            .input_guardrail(BlockingInputGuardrail)
            .build()
            .unwrap();

        let result = agent.prompt_with_cancellation("hi", &cancel).await;
        assert!(matches!(
            result.err(),
            Some(crate::error::DaimonError::GuardrailBlocked(_))
        ));
    }

    #[tokio::test]
    async fn test_prompt_with_messages_enforces_input_guardrail() {
        let agent = Agent::builder()
            .model(EchoModel)
            .input_guardrail(BlockingInputGuardrail)
            .build()
            .unwrap();

        let result = agent.prompt_with_messages(vec![Message::user("hi")]).await;
        assert!(matches!(
            result.err(),
            Some(crate::error::DaimonError::GuardrailBlocked(_))
        ));
    }

    #[tokio::test]
    async fn test_cancellation_variant_enforces_budget() {
        // ToolCallingModel runs 2 iterations; a tiny budget must trip on the
        // second iteration's pre-check (regression: budget was only checked in
        // prompt(), not prompt_with_cancellation()).
        let cancel = tokio_util::sync::CancellationToken::new();
        let agent = Agent::builder()
            .model(ToolCallingModel::new())
            .tool(AdderTool)
            .cost_model(crate::cost::OpenAiCostModel)
            .max_budget(1e-9)
            .build()
            .unwrap();

        let result = agent.prompt_with_cancellation("add 2 and 3", &cancel).await;
        assert!(matches!(
            result.err(),
            Some(crate::error::DaimonError::BudgetExceeded { .. })
        ));
    }

    // ---- Streaming ReAct regression tests (DAIM-1) ----

    /// A streaming model that always emits a single tool call (no final text),
    /// forcing the ReAct loop to iterate indefinitely. Used to exercise the
    /// per-stream budget guard, which trips on the second iteration's pre-check.
    struct StreamingToolLoopModel;

    impl Model for StreamingToolLoopModel {
        async fn generate(&self, _request: &ChatRequest) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message::assistant("unused"),
                stop_reason: StopReason::EndTurn,
                usage: None,
            })
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            use crate::stream::StreamEvent;
            let events = vec![
                Ok(StreamEvent::ToolCallStart {
                    id: "loop".into(),
                    name: "noop".into(),
                }),
                Ok(StreamEvent::ToolCallDelta {
                    id: "loop".into(),
                    arguments_delta: "{}".into(),
                }),
                Ok(StreamEvent::Done),
            ];
            Ok(Box::pin(futures::stream::iter(events)))
        }
    }

    /// A streaming model that announces two tool calls (ids "a" then "b") on the
    /// first iteration and returns final text on the second. Exercises that the
    /// streaming loop preserves the announced tool-call order rather than the
    /// arbitrary iteration order of the pending-calls `HashMap`.
    struct StreamingMultiToolModel {
        call_count: AtomicUsize,
    }

    impl StreamingMultiToolModel {
        fn new() -> Self {
            Self {
                call_count: AtomicUsize::new(0),
            }
        }
    }

    impl Model for StreamingMultiToolModel {
        async fn generate(&self, _request: &ChatRequest) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message::assistant("unused"),
                stop_reason: StopReason::EndTurn,
                usage: None,
            })
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            use crate::stream::StreamEvent;
            let count = self.call_count.fetch_add(1, Ordering::SeqCst);
            let events: Vec<Result<StreamEvent>> = if count == 0 {
                vec![
                    Ok(StreamEvent::ToolCallStart {
                        id: "a".into(),
                        name: "tool_a".into(),
                    }),
                    Ok(StreamEvent::ToolCallDelta {
                        id: "a".into(),
                        arguments_delta: "{}".into(),
                    }),
                    Ok(StreamEvent::ToolCallStart {
                        id: "b".into(),
                        name: "tool_b".into(),
                    }),
                    Ok(StreamEvent::ToolCallDelta {
                        id: "b".into(),
                        arguments_delta: "{}".into(),
                    }),
                    Ok(StreamEvent::Done),
                ]
            } else {
                vec![
                    Ok(StreamEvent::TextDelta("all done".into())),
                    Ok(StreamEvent::Done),
                ]
            };
            Ok(Box::pin(futures::stream::iter(events)))
        }
    }

    struct ToolA;

    impl Tool for ToolA {
        fn name(&self) -> &str {
            "tool_a"
        }
        fn description(&self) -> &str {
            "Tool A"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(&self, _input: &serde_json::Value) -> Result<ToolOutput> {
            Ok(ToolOutput::text("result_a"))
        }
    }

    struct ToolB;

    impl Tool for ToolB {
        fn name(&self) -> &str {
            "tool_b"
        }
        fn description(&self) -> &str {
            "Tool B"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(&self, _input: &serde_json::Value) -> Result<ToolOutput> {
            Ok(ToolOutput::text("result_b"))
        }
    }

    struct BlockingOutputGuardrail;

    impl crate::guardrails::OutputGuardrail for BlockingOutputGuardrail {
        async fn check(
            &self,
            _response: &ChatResponse,
        ) -> Result<crate::guardrails::GuardrailResult> {
            Ok(crate::guardrails::GuardrailResult::Block("nope".into()))
        }
    }

    struct TransformOutputGuardrail;

    impl crate::guardrails::OutputGuardrail for TransformOutputGuardrail {
        async fn check(
            &self,
            _response: &ChatResponse,
        ) -> Result<crate::guardrails::GuardrailResult> {
            Ok(crate::guardrails::GuardrailResult::Transform(
                "REPLACED".into(),
            ))
        }
    }

    #[tokio::test]
    async fn test_streaming_enforces_budget() {
        use crate::stream::StreamEvent;
        use futures::StreamExt;

        // The model never terminates on its own (always a tool call), so only the
        // per-stream budget guard can stop the loop. With a sub-nanodollar budget,
        // the first iteration records a non-zero estimated cost and the second
        // iteration's pre-check must trip. The streaming path surfaces this as a
        // StreamEvent::Error (it does not return Err from prompt_stream).
        let agent = Agent::builder()
            .model(StreamingToolLoopModel)
            .tool(NoopTool)
            .cost_model(crate::cost::OpenAiCostModel)
            .max_budget(1e-9)
            .max_iterations(100)
            .build()
            .unwrap();

        let mut stream = agent.prompt_stream("go").await.unwrap();

        let mut saw_budget_error = false;
        let mut saw_done = false;
        while let Some(ev) = stream.next().await {
            match ev.unwrap() {
                StreamEvent::Error(msg) => {
                    assert!(
                        msg.contains("budget"),
                        "expected a budget error, got: {msg}"
                    );
                    // The budget Error must precede Done.
                    assert!(!saw_done, "budget Error should be emitted before Done");
                    saw_budget_error = true;
                }
                StreamEvent::Done => {
                    saw_done = true;
                    break;
                }
                _ => {}
            }
        }

        assert!(saw_budget_error, "expected a budget StreamEvent::Error");
        assert!(saw_done, "stream must terminate with Done");
    }

    #[tokio::test]
    async fn test_streaming_output_guardrail_blocks() {
        use crate::stream::StreamEvent;
        use futures::StreamExt;

        let agent = Agent::builder()
            .model(StreamingTextModel)
            .output_guardrail(BlockingOutputGuardrail)
            .build()
            .unwrap();

        let mut stream = agent.prompt_stream("hello").await.unwrap();

        let mut saw_guardrail_error = false;
        while let Some(ev) = stream.next().await {
            match ev.unwrap() {
                StreamEvent::Error(msg) => {
                    assert!(
                        msg.contains("guardrail"),
                        "expected a guardrail error, got: {msg}"
                    );
                    saw_guardrail_error = true;
                }
                StreamEvent::Done => break,
                _ => {}
            }
        }

        assert!(
            saw_guardrail_error,
            "expected a guardrail StreamEvent::Error"
        );

        // A blocked output must NOT be persisted: memory holds only the user
        // message, never the assistant response the guardrail rejected.
        let messages = agent.memory().get_messages_erased().await.unwrap();
        assert_eq!(
            messages.len(),
            1,
            "blocked assistant output must not be saved to memory"
        );
        assert_eq!(messages[0].role, crate::model::types::Role::User);
    }

    #[tokio::test]
    async fn test_streaming_output_guardrail_transforms() {
        use crate::stream::StreamEvent;
        use futures::StreamExt;

        let agent = Agent::builder()
            .model(StreamingTextModel)
            .output_guardrail(TransformOutputGuardrail)
            .build()
            .unwrap();

        let mut stream = agent.prompt_stream("hello").await.unwrap();
        while let Some(ev) = stream.next().await {
            if matches!(ev.unwrap(), StreamEvent::Done) {
                break;
            }
        }

        // The transform must affect the persisted assistant message, not just the
        // live deltas: history should reflect the transformed text.
        let messages = agent.memory().get_messages_erased().await.unwrap();
        assert_eq!(messages.len(), 2, "expected user + assistant in memory");
        assert_eq!(messages[1].role, crate::model::types::Role::Assistant);
        assert_eq!(messages[1].content.as_deref(), Some("REPLACED"));
    }

    #[tokio::test]
    async fn test_streaming_preserves_multi_tool_order() {
        use crate::stream::StreamEvent;
        use futures::StreamExt;

        let agent = Agent::builder()
            .model(StreamingMultiToolModel::new())
            .tool(ToolA)
            .tool(ToolB)
            .build()
            .unwrap();

        let mut stream = agent.prompt_stream("run both").await.unwrap();

        let mut result_ids = Vec::new();
        while let Some(ev) = stream.next().await {
            match ev.unwrap() {
                StreamEvent::ToolResult { id, content, .. } => {
                    // Sanity: each id maps to its own tool's output.
                    match id.as_str() {
                        "a" => assert_eq!(content, "result_a"),
                        "b" => assert_eq!(content, "result_b"),
                        other => panic!("unexpected tool result id: {other}"),
                    }
                    result_ids.push(id);
                }
                StreamEvent::Done => break,
                _ => {}
            }
        }

        // Both tools ran, and results are emitted in the announced order (a before
        // b) rather than an arbitrary HashMap iteration order.
        assert_eq!(
            result_ids,
            vec!["a".to_string(), "b".to_string()],
            "tool results must preserve announced order"
        );
    }
}
