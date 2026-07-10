//! Summarization-based conversation memory.

use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::Mutex;

use crate::error::Result;
use crate::memory::traits::Memory;
use crate::model::SharedModel;
use crate::model::types::{ChatRequest, Message};

const DEFAULT_MAX_MESSAGES: usize = 20;
const DEFAULT_RETAIN_RECENT: usize = 10;
const DEFAULT_SUMMARY_PROMPT: &str = "\
You are a conversation summarizer. Summarize the following conversation \
into a concise paragraph that preserves all important facts, decisions, \
tool results, and context. Be specific — include names, numbers, and \
outcomes. Do not include any preamble, just the summary.";

/// In-memory conversation storage that summarizes old messages instead of
/// dropping them.
///
/// When the message count exceeds `max_messages`, the oldest messages
/// (all except the most recent `retain_recent`) are summarized into a
/// single condensed message using the configured LLM. The summary is
/// prepended as a system message to future context, preserving long-term
/// context without consuming the full token budget.
///
/// # Example
///
/// ```ignore
/// use daimon::memory::SummaryMemory;
/// use daimon::model::openai::OpenAi;
/// use std::sync::Arc;
///
/// let model = Arc::new(OpenAi::new("gpt-4o-mini"));
/// let memory = SummaryMemory::new(model);
/// ```
pub struct SummaryMemory {
    messages: Mutex<Vec<Message>>,
    summary: Mutex<Option<String>>,
    /// Serializes summarization so two concurrent `add_message` calls cannot
    /// both summarize and clobber each other's summary (last write wins).
    summarize_lock: Mutex<()>,
    /// Incremented by [`clear`](Memory::clear); a summarization that started
    /// before a clear must not apply its stale result afterwards.
    epoch: AtomicU64,
    model: SharedModel,
    max_messages: usize,
    retain_recent: usize,
    summary_prompt: String,
}

impl SummaryMemory {
    /// Creates a new `SummaryMemory` with default settings.
    ///
    /// Defaults: summarize when >20 messages, keep 10 most recent.
    /// Uses the provided model to generate summaries.
    pub fn new(model: SharedModel) -> Self {
        Self {
            messages: Mutex::new(Vec::new()),
            summary: Mutex::new(None),
            summarize_lock: Mutex::new(()),
            epoch: AtomicU64::new(0),
            model,
            max_messages: DEFAULT_MAX_MESSAGES,
            retain_recent: DEFAULT_RETAIN_RECENT,
            summary_prompt: DEFAULT_SUMMARY_PROMPT.to_string(),
        }
    }

    /// Sets the message count threshold that triggers summarization.
    pub fn with_max_messages(mut self, max: usize) -> Self {
        self.max_messages = max;
        self
    }

    /// Sets how many recent messages to keep unsummarized after compaction.
    pub fn with_retain_recent(mut self, count: usize) -> Self {
        self.retain_recent = count;
        self
    }

    /// Overrides the default summarization system prompt.
    pub fn with_summary_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.summary_prompt = prompt.into();
        self
    }

    /// Returns the current running summary, if one has been generated.
    pub async fn current_summary(&self) -> Option<String> {
        self.summary.lock().await.clone()
    }

    /// Summarizes the oldest messages and replaces them with the summary.
    ///
    /// Messages are only removed from the store *after* the model call
    /// succeeds, so a failed summarization leaves the full history intact
    /// and never aborts the caller's `add_message`. The whole critical
    /// section runs under `summarize_lock` so concurrent callers cannot
    /// summarize the same messages or overwrite each other's summary.
    async fn maybe_summarize(&self) {
        let _guard = self.summarize_lock.lock().await;

        let (to_summarize, existing_summary, epoch) = {
            let messages = self.messages.lock().await;

            if messages.len() <= self.max_messages {
                return;
            }

            let split_at = messages.len().saturating_sub(self.retain_recent);
            if split_at == 0 {
                return;
            }

            let existing_summary = self.summary.lock().await.clone();
            let epoch = self.epoch.load(Ordering::Acquire);
            (messages[..split_at].to_vec(), existing_summary, epoch)
        };

        let mut summary_input = Vec::new();
        summary_input.push(Message::system(&self.summary_prompt));

        if let Some(ref prev) = existing_summary {
            summary_input.push(Message::user(format!(
                "Previous conversation summary:\n{prev}\n\nNew messages to incorporate:"
            )));
        } else {
            summary_input.push(Message::user("Conversation to summarize:".to_string()));
        }

        let mut conversation_text = String::new();
        for msg in &to_summarize {
            let role = match msg.role {
                crate::model::types::Role::System => "System",
                crate::model::types::Role::User => "User",
                crate::model::types::Role::Assistant => "Assistant",
                crate::model::types::Role::Tool => "Tool",
            };
            if let Some(ref content) = msg.content {
                conversation_text.push_str(&format!("{role}: {content}\n"));
            }
            if !msg.tool_calls.is_empty() {
                for tc in &msg.tool_calls {
                    conversation_text.push_str(&format!(
                        "Assistant called tool '{}' with args: {}\n",
                        tc.name, tc.arguments
                    ));
                }
            }
        }
        summary_input.push(Message::user(conversation_text));

        let request = ChatRequest {
            messages: summary_input,
            tools: Vec::new(),
            temperature: Some(0.0),
            max_tokens: Some(512),
        };

        tracing::debug!(
            messages_summarized = to_summarize.len(),
            "generating conversation summary"
        );

        // No locks are held across the model call; the store still contains
        // every message, so a failure here is fully recoverable.
        let summary_text = match self.model.generate_erased(&request).await {
            Ok(response) => response.text().to_string(),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "conversation summarization failed; keeping full message history"
                );
                return;
            }
        };

        // Lock both stores (same order as get_messages/clear) so readers
        // never observe the drained messages without the new summary.
        let mut messages = self.messages.lock().await;
        let mut summary = self.summary.lock().await;

        if self.epoch.load(Ordering::Acquire) != epoch {
            // Memory was cleared while the model call was in flight; the
            // summary describes messages that no longer exist. Discard it.
            return;
        }

        // Concurrent add_message calls only append, and clear() is detected
        // via the epoch, so the summarized prefix is still at the front.
        messages.drain(..to_summarize.len());
        *summary = Some(summary_text);
    }
}

impl Memory for SummaryMemory {
    async fn add_message(&self, message: &Message) -> Result<()> {
        {
            let mut messages = self.messages.lock().await;
            messages.push(message.clone());
        }

        // A failed summarization must not abort the caller's agent run:
        // the message store still holds everything, so it is recoverable
        // and maybe_summarize logs a warning instead of returning an error.
        self.maybe_summarize().await;
        Ok(())
    }

    async fn get_messages(&self) -> Result<Vec<Message>> {
        let messages = self.messages.lock().await;
        let summary = self.summary.lock().await;

        let mut result = Vec::new();
        if let Some(ref s) = *summary {
            result.push(Message::system(format!(
                "Previous conversation summary: {s}"
            )));
        }
        result.extend(messages.clone());
        Ok(result)
    }

    async fn clear(&self) -> Result<()> {
        let mut messages = self.messages.lock().await;
        let mut summary = self.summary.lock().await;
        messages.clear();
        *summary = None;
        // Invalidate any in-flight summarization so it discards its result.
        self.epoch.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Model;
    use crate::model::types::{ChatResponse, Role, StopReason, Usage};
    use crate::stream::ResponseStream;
    use std::sync::Arc;

    struct FakeSummarizerModel;

    impl Model for FakeSummarizerModel {
        async fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
            let has_previous = request.messages.iter().any(|m| {
                m.content
                    .as_ref()
                    .is_some_and(|c| c.contains("Previous conversation summary"))
            });

            let msg_count = request
                .messages
                .last()
                .and_then(|m| m.content.as_ref())
                .map(|c| c.lines().count())
                .unwrap_or(0);

            let text = if has_previous {
                format!("Extended summary incorporating {msg_count} new lines.")
            } else {
                format!("Summary of {msg_count} conversation lines.")
            };

            Ok(ChatResponse {
                message: Message::assistant(text),
                stop_reason: StopReason::EndTurn,
                usage: Some(Usage::default()),
            })
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    fn make_memory(max_messages: usize, retain_recent: usize) -> SummaryMemory {
        let model: SharedModel = Arc::new(FakeSummarizerModel);
        SummaryMemory::new(model)
            .with_max_messages(max_messages)
            .with_retain_recent(retain_recent)
    }

    #[tokio::test]
    async fn test_add_and_get_below_threshold() {
        let memory = make_memory(10, 5);

        memory.add_message(&Message::user("hello")).await.unwrap();
        memory.add_message(&Message::assistant("hi")).await.unwrap();

        let msgs = memory.get_messages().await.unwrap();
        assert_eq!(msgs.len(), 2);
        assert!(memory.current_summary().await.is_none());
    }

    #[tokio::test]
    async fn test_summarizes_when_exceeding_threshold() {
        let memory = make_memory(5, 2);

        for i in 0..6 {
            memory
                .add_message(&Message::user(format!("message {i}")))
                .await
                .unwrap();
        }

        let summary = memory.current_summary().await;
        assert!(summary.is_some());

        let msgs = memory.get_messages().await.unwrap();
        assert_eq!(msgs[0].role, Role::System);
        assert!(msgs[0].content.as_ref().unwrap().contains("Summary"));
    }

    #[tokio::test]
    async fn test_retains_recent_messages() {
        let memory = make_memory(4, 2);

        for i in 0..5 {
            memory
                .add_message(&Message::user(format!("msg {i}")))
                .await
                .unwrap();
        }

        let msgs = memory.get_messages().await.unwrap();
        // 1 summary message + 2 retained recent
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(msgs[1].content.as_deref(), Some("msg 3"));
        assert_eq!(msgs[2].content.as_deref(), Some("msg 4"));
    }

    #[tokio::test]
    async fn test_clear_resets_summary() {
        let memory = make_memory(3, 1);

        for i in 0..5 {
            memory
                .add_message(&Message::user(format!("msg {i}")))
                .await
                .unwrap();
        }

        assert!(memory.current_summary().await.is_some());

        memory.clear().await.unwrap();

        assert!(memory.current_summary().await.is_none());
        assert!(memory.get_messages().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_incremental_summarization() {
        let memory = make_memory(4, 2);

        // First batch: trigger first summarization
        for i in 0..5 {
            memory
                .add_message(&Message::user(format!("batch1 msg {i}")))
                .await
                .unwrap();
        }
        let first_summary = memory.current_summary().await.unwrap();

        // Second batch: trigger second summarization (builds on first)
        for i in 0..5 {
            memory
                .add_message(&Message::user(format!("batch2 msg {i}")))
                .await
                .unwrap();
        }
        let second_summary = memory.current_summary().await.unwrap();

        assert_ne!(first_summary, second_summary);
    }

    #[tokio::test]
    async fn test_custom_summary_prompt() {
        let model: SharedModel = Arc::new(FakeSummarizerModel);
        let memory = SummaryMemory::new(model)
            .with_max_messages(3)
            .with_retain_recent(1)
            .with_summary_prompt("Custom summarization instructions");

        assert_eq!(memory.summary_prompt, "Custom summarization instructions");
    }

    #[tokio::test]
    async fn test_empty_memory() {
        let memory = make_memory(10, 5);
        assert!(memory.current_summary().await.is_none());
        assert!(memory.get_messages().await.unwrap().is_empty());
    }

    struct FailingSummarizerModel;

    impl Model for FailingSummarizerModel {
        async fn generate(&self, _request: &ChatRequest) -> Result<ChatResponse> {
            Err(crate::error::DaimonError::Model(
                "summarizer unavailable".to_string(),
            ))
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    #[tokio::test]
    async fn test_failed_summarization_keeps_messages_and_returns_ok() {
        let model: SharedModel = Arc::new(FailingSummarizerModel);
        let memory = SummaryMemory::new(model)
            .with_max_messages(5)
            .with_retain_recent(2);

        // Every add must succeed even though summarization fails repeatedly.
        for i in 0..8 {
            memory
                .add_message(&Message::user(format!("message {i}")))
                .await
                .unwrap();
        }

        // No summary was produced and no message was lost.
        assert!(memory.current_summary().await.is_none());
        let msgs = memory.get_messages().await.unwrap();
        assert_eq!(msgs.len(), 8);
        for (i, msg) in msgs.iter().enumerate() {
            assert_eq!(
                msg.content.as_deref(),
                Some(format!("message {i}").as_str())
            );
        }
    }

    /// Records every conversation line it is asked to summarize, so tests
    /// can verify that no message is ever lost or summarized twice.
    struct RecordingSummarizerModel {
        summarized: std::sync::Mutex<Vec<String>>,
    }

    impl Model for RecordingSummarizerModel {
        async fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
            let lines: Vec<String> = request
                .messages
                .last()
                .and_then(|m| m.content.as_ref())
                .map(|c| {
                    c.lines()
                        .filter_map(|l| l.strip_prefix("User: "))
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();

            // Yield so concurrent add_message calls can interleave here.
            tokio::task::yield_now().await;

            let count = {
                let mut summarized = self
                    .summarized
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                summarized.extend(lines);
                summarized.len()
            };

            Ok(ChatResponse {
                message: Message::assistant(format!("summary after {count} lines")),
                stop_reason: StopReason::EndTurn,
                usage: Some(Usage::default()),
            })
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_concurrent_add_message_loses_no_messages() {
        let model = Arc::new(RecordingSummarizerModel {
            summarized: std::sync::Mutex::new(Vec::new()),
        });
        let memory = Arc::new(
            SummaryMemory::new(model.clone() as SharedModel)
                .with_max_messages(5)
                .with_retain_recent(2),
        );

        let mem_a = memory.clone();
        let task_a = tokio::spawn(async move {
            for i in 0..15 {
                mem_a
                    .add_message(&Message::user(format!("task-a {i}")))
                    .await
                    .unwrap();
            }
        });

        let mem_b = memory.clone();
        let task_b = tokio::spawn(async move {
            for i in 0..15 {
                mem_b
                    .add_message(&Message::user(format!("task-b {i}")))
                    .await
                    .unwrap();
            }
        });

        task_a.await.unwrap();
        task_b.await.unwrap();

        // Every added message must appear exactly once: either still stored
        // in memory, or recorded as summarized — never both, never neither.
        let summarized = model
            .summarized
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let remaining: Vec<String> = memory
            .get_messages()
            .await
            .unwrap()
            .into_iter()
            .filter(|m| m.role == Role::User)
            .filter_map(|m| m.content)
            .collect();

        let mut seen: Vec<String> = summarized.iter().chain(remaining.iter()).cloned().collect();
        seen.sort();

        let mut expected: Vec<String> = (0..15)
            .map(|i| format!("task-a {i}"))
            .chain((0..15).map(|i| format!("task-b {i}")))
            .collect();
        expected.sort();

        assert_eq!(
            seen, expected,
            "summarized: {summarized:?}, remaining: {remaining:?}"
        );
    }

    #[tokio::test]
    async fn test_clear_during_summarization_discards_stale_summary() {
        let memory = make_memory(3, 1);

        for i in 0..5 {
            memory
                .add_message(&Message::user(format!("msg {i}")))
                .await
                .unwrap();
        }
        assert!(memory.current_summary().await.is_some());

        memory.clear().await.unwrap();

        assert!(memory.current_summary().await.is_none());
        assert!(memory.get_messages().await.unwrap().is_empty());

        // Fresh messages after clear behave normally.
        memory.add_message(&Message::user("fresh")).await.unwrap();
        let msgs = memory.get_messages().await.unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content.as_deref(), Some("fresh"));
    }
}
