//! The ReAct agent loop implementation.

use tokio_util::sync::CancellationToken;
use tracing::Instrument;

use crate::agent::Agent;
use crate::error::{DaimonError, Result};
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
        let history = self.memory.get_messages_erased().await?;

        let mut messages = Vec::new();
        if let Some(system) = &self.system_prompt {
            messages.push(Message::system(system));
        }
        messages.extend(history);
        messages.push(Message::user(input));

        self.memory.add_message_erased(Message::user(input)).await?;

        self.run_react_loop(messages, cancel).await
    }

    /// Send pre-built messages to the agent and get a complete response.
    ///
    /// This bypasses the system prompt and memory loading -- you provide the
    /// full message history yourself. Useful for advanced scenarios like
    /// replaying conversations or injecting custom context.
    #[tracing::instrument(skip_all, fields(message_count = messages.len()))]
    pub async fn prompt_with_messages(&self, messages: Vec<Message>) -> Result<AgentResponse> {
        self.run_react_loop(messages, &CancellationToken::new())
            .await
    }

    /// Core ReAct loop shared by all prompt methods.
    ///
    /// Uses `std::mem::take` to move messages and tool specs in and out of
    /// each `ChatRequest` without cloning, then moves them back after the
    /// model call completes.
    pub(crate) async fn run_react_loop(
        &self,
        mut messages: Vec<Message>,
        cancel: &CancellationToken,
    ) -> Result<AgentResponse> {
        use crate::middleware::MiddlewareAction;

        let mut tool_specs_vec: Vec<crate::model::types::ToolSpec> =
            self.tools.tool_specs().to_vec();
        let mut iteration = 0;
        let mut total_usage = Usage::default();
        let mut total_cost = 0.0f64;

        if let Some(ref tracker) = self.cost_tracker {
            tracker.reset();
        }

        loop {
            if cancel.is_cancelled() {
                return Err(DaimonError::Cancelled);
            }

            if let (Some(tracker), Some(limit)) = (&self.cost_tracker, self.max_budget) {
                let spent = tracker.cumulative_cost();
                if spent >= limit {
                    return Err(DaimonError::BudgetExceeded { spent, limit });
                }
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

            match self.middleware.run_on_request(&mut request).await? {
                MiddlewareAction::ShortCircuit(resp) => {
                    let mut msgs = std::mem::take(&mut request.messages);
                    let final_text = resp.text().to_string();
                    msgs.push(resp.message);
                    return Ok(AgentResponse {
                        messages: msgs,
                        final_text,
                        iterations: iteration,
                        usage: total_usage,
                        cost: total_cost,
                    });
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

            messages = std::mem::take(&mut request.messages);
            tool_specs_vec = std::mem::take(&mut request.tools);

            let mut response = result?;

            if let Some(ref usage) = response.usage {
                tracing::debug!(
                    input_tokens = usage.input_tokens,
                    output_tokens = usage.output_tokens,
                    "model usage"
                );
                total_usage.accumulate(usage);
                if let Some(ref tracker) = self.cost_tracker {
                    total_cost += tracker.record("default", usage);
                }
            }

            match self.middleware.run_on_response(&mut response).await? {
                MiddlewareAction::ShortCircuit(replaced) => {
                    let final_text = replaced.text().to_string();
                    messages.push(replaced.message);
                    return Ok(AgentResponse {
                        messages,
                        final_text,
                        iterations: iteration,
                        usage: total_usage,
                        cost: total_cost,
                    });
                }
                MiddlewareAction::Continue => {}
            }

            self.hooks.on_model_response_erased(&response).await?;

            if response.has_tool_calls() {
                let tool_calls = std::mem::take(&mut response.message.tool_calls);
                let assistant_msg = Message::assistant_with_tool_calls(tool_calls.clone());
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

                if iteration >= self.max_iterations {
                    return Err(DaimonError::MaxIterations(self.max_iterations));
                }

                continue;
            }

            let final_text = response.text().to_string();
            self.memory
                .add_message_erased(response.message.clone())
                .await?;
            messages.push(response.message);

            self.hooks.on_iteration_end_erased(&state).await?;

            return Ok(AgentResponse {
                messages,
                final_text,
                iterations: iteration,
                usage: total_usage,
                cost: total_cost,
            });
        }
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
    /// generates its response. The stream runs the full ReAct loop:
    /// tool call deltas are accumulated and when complete, tools are executed
    /// and the model is re-invoked, all within the same stream.
    #[tracing::instrument(skip_all, fields(input_len = input.len()))]
    pub async fn prompt_stream(&self, input: &str) -> Result<ResponseStream> {
        use crate::stream::StreamEvent;
        use futures::StreamExt;
        use std::collections::HashMap;

        let history = self.memory.get_messages_erased().await?;

        let mut messages = Vec::new();
        if let Some(system) = &self.system_prompt {
            messages.push(Message::system(system));
        }
        messages.extend(history);
        messages.push(Message::user(input));

        self.memory.add_message_erased(Message::user(input)).await?;

        let mut tool_specs_vec: Vec<crate::model::types::ToolSpec> =
            self.tools.tool_specs().to_vec();
        let max_iterations = self.max_iterations;

        let model = self.model.clone();
        let tools = self.tools.clone();
        let temperature = self.temperature;
        let max_tokens = self.max_tokens;
        let validate = self.validate_tool_inputs;
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
                let mut had_tool_calls = false;
                let mut text_buf = String::new();

                while let Some(event) = inner.next().await {
                    let event = event?;
                    match &event {
                        StreamEvent::ToolCallStart { id, name } => {
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
                    if !text_buf.is_empty() {
                        messages.push(Message::assistant(&text_buf));
                    }
                    yield StreamEvent::Done;
                    break;
                }

                let mut completed_calls = Vec::new();
                for (id, (name, args_str)) in &pending_tool_calls {
                    let arguments: serde_json::Value = serde_json::from_str(args_str)
                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                    completed_calls.push(crate::tool::ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments,
                    });
                }

                messages.push(Message::assistant_with_tool_calls(completed_calls.clone()));

                for call in &completed_calls {
                    let tool_result = if validate {
                        if let Some(errors) = tools.validate_input(&call.name, &call.arguments) {
                            crate::tool::ToolOutput::error(format!(
                                "Invalid arguments for tool '{}': {errors}",
                                call.name
                            ))
                        } else {
                            match tools.get(&call.name) {
                                Some(tool) => match tool.execute_erased(&call.arguments).await {
                                    Ok(output) => output,
                                    Err(e) => crate::tool::ToolOutput::error(e.to_string()),
                                },
                                None => crate::tool::ToolOutput::error(format!(
                                    "tool '{}' not found",
                                    call.name
                                )),
                            }
                        }
                    } else {
                        match tools.get(&call.name) {
                            Some(tool) => match tool.execute_erased(&call.arguments).await {
                                Ok(output) => output,
                                Err(e) => crate::tool::ToolOutput::error(e.to_string()),
                            },
                            None => crate::tool::ToolOutput::error(format!(
                                "tool '{}' not found",
                                call.name
                            )),
                        }
                    };

                    yield StreamEvent::ToolResult {
                        id: call.id.clone(),
                        content: tool_result.content.clone(),
                        is_error: tool_result.is_error,
                    };

                    messages.push(Message::tool_result(&call.id, &tool_result.content));
                }
            }
        };

        Ok(Box::pin(out_stream))
    }

    /// Runs input guardrails. Returns the (potentially transformed) input or an error.
    async fn run_input_guardrails(&self, input: &str) -> Result<String> {
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

    /// Runs output guardrails on the final response.
    async fn run_output_guardrails(&self, response: &mut AgentResponse) -> Result<()> {
        for guard in &self.output_guardrails {
            let chat_resp = ChatResponse {
                message: Message::assistant(&response.final_text),
                stop_reason: crate::model::types::StopReason::EndTurn,
                usage: Some(response.usage.clone()),
            };
            match guard.check_erased(&chat_resp).await? {
                crate::guardrails::GuardrailResult::Pass => {}
                crate::guardrails::GuardrailResult::Block(msg) => {
                    return Err(DaimonError::GuardrailBlocked(msg));
                }
                crate::guardrails::GuardrailResult::Transform(new_text) => {
                    response.final_text = new_text;
                }
            }
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
}
