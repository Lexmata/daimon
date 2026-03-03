//! Token-budget-based conversation memory.

use tokio::sync::Mutex;

use crate::error::Result;
use crate::memory::traits::Memory;
use crate::model::types::Message;

/// Estimates the token count of a message using a simple heuristic.
///
/// Counts the text content plus a rough estimate for tool call payloads.
/// Uses ~1 token per 4 characters, which is a reasonable approximation
/// for GPT-style tokenizers on English text.
fn default_estimate_tokens(msg: &Message) -> usize {
    let mut chars = 0usize;

    if let Some(ref content) = msg.content {
        chars += content.len();
    }

    for tc in &msg.tool_calls {
        chars += tc.name.len();
        chars += tc.arguments.to_string().len();
    }

    if let Some(ref id) = msg.tool_call_id {
        chars += id.len();
    }

    // role label overhead (~6 chars)
    chars += 6;

    // ~4 chars per token
    chars.div_ceil(4)
}

/// A token counter function: takes a message, returns estimated token count.
type TokenCounterFn = Box<dyn Fn(&Message) -> usize + Send + Sync>;

/// In-memory conversation storage that keeps messages within a token budget.
///
/// Unlike [`SlidingWindowMemory`](super::SlidingWindowMemory) which counts
/// messages, this implementation estimates token usage and evicts the oldest
/// messages when the total exceeds the configured budget.
///
/// # Token Estimation
///
/// By default, tokens are estimated at ~4 characters per token (a reasonable
/// heuristic for GPT-style tokenizers). Use [`with_token_counter`](Self::with_token_counter)
/// to plug in a precise tokenizer like `tiktoken-rs`.
///
/// # Example
///
/// ```ignore
/// use daimon::memory::TokenWindowMemory;
///
/// let memory = TokenWindowMemory::new(4096);
/// ```
pub struct TokenWindowMemory {
    messages: Mutex<Vec<Message>>,
    token_counts: Mutex<Vec<usize>>,
    max_tokens: usize,
    total_tokens: Mutex<usize>,
    token_counter: TokenCounterFn,
}

impl TokenWindowMemory {
    /// Creates a new token window with the given maximum token budget.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            messages: Mutex::new(Vec::new()),
            token_counts: Mutex::new(Vec::new()),
            max_tokens,
            total_tokens: Mutex::new(0),
            token_counter: Box::new(default_estimate_tokens),
        }
    }

    /// Replaces the default token estimator with a custom function.
    ///
    /// Use this to plug in a precise tokenizer (e.g. `tiktoken-rs`):
    ///
    /// ```ignore
    /// let memory = TokenWindowMemory::new(4096)
    ///     .with_token_counter(|msg| {
    ///         my_tokenizer.count_tokens(msg.content.as_deref().unwrap_or(""))
    ///     });
    /// ```
    pub fn with_token_counter<F>(mut self, counter: F) -> Self
    where
        F: Fn(&Message) -> usize + Send + Sync + 'static,
    {
        self.token_counter = Box::new(counter);
        self
    }

    /// Returns the current total estimated token count.
    pub async fn current_tokens(&self) -> usize {
        *self.total_tokens.lock().await
    }
}

impl Memory for TokenWindowMemory {
    async fn add_message(&self, message: Message) -> Result<()> {
        let tokens = (self.token_counter)(&message);

        let mut messages = self.messages.lock().await;
        let mut counts = self.token_counts.lock().await;
        let mut total = self.total_tokens.lock().await;

        messages.push(message);
        counts.push(tokens);
        *total += tokens;

        while *total > self.max_tokens && messages.len() > 1 {
            let removed_tokens = counts.remove(0);
            messages.remove(0);
            *total -= removed_tokens;
        }

        Ok(())
    }

    async fn get_messages(&self) -> Result<Vec<Message>> {
        let messages = self.messages.lock().await;
        Ok(messages.clone())
    }

    async fn clear(&self) -> Result<()> {
        let mut messages = self.messages.lock().await;
        let mut counts = self.token_counts.lock().await;
        let mut total = self.total_tokens.lock().await;
        messages.clear();
        counts.clear();
        *total = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::types::Role;
    use crate::tool::ToolCall;

    #[tokio::test]
    async fn test_add_and_get_messages() {
        let memory = TokenWindowMemory::new(10_000);
        memory.add_message(Message::user("hello")).await.unwrap();
        memory
            .add_message(Message::assistant("hi"))
            .await
            .unwrap();

        let msgs = memory.get_messages().await.unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::User);
        assert_eq!(msgs[1].role, Role::Assistant);
    }

    #[tokio::test]
    async fn test_evicts_old_messages_when_over_budget() {
        // Use a custom counter: 1 token per character for predictability
        let memory = TokenWindowMemory::new(20).with_token_counter(|msg| {
            msg.content.as_ref().map_or(0, |c| c.len())
        });

        // "aaaaaaaaaa" = 10 tokens
        memory
            .add_message(Message::user("aaaaaaaaaa"))
            .await
            .unwrap();
        // "bbbbbbbbbb" = 10 tokens, total 20 (at limit)
        memory
            .add_message(Message::user("bbbbbbbbbb"))
            .await
            .unwrap();
        assert_eq!(memory.get_messages().await.unwrap().len(), 2);

        // "cccccccccc" = 10 tokens, total would be 30 > 20, evict "aaa..."
        memory
            .add_message(Message::user("cccccccccc"))
            .await
            .unwrap();

        let msgs = memory.get_messages().await.unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].content.as_deref(), Some("bbbbbbbbbb"));
        assert_eq!(msgs[1].content.as_deref(), Some("cccccccccc"));
    }

    #[tokio::test]
    async fn test_evicts_multiple_to_fit() {
        let memory = TokenWindowMemory::new(15).with_token_counter(|msg| {
            msg.content.as_ref().map_or(0, |c| c.len())
        });

        memory.add_message(Message::user("aaa")).await.unwrap(); // 3
        memory.add_message(Message::user("bbb")).await.unwrap(); // 3
        memory.add_message(Message::user("ccc")).await.unwrap(); // 3, total 9

        // Adding 8 tokens: total would be 17 > 15, evict "aaa" (3) -> 14, fits
        memory
            .add_message(Message::user("dddddddd"))
            .await
            .unwrap();

        let msgs = memory.get_messages().await.unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0].content.as_deref(), Some("bbb"));
        assert_eq!(msgs[1].content.as_deref(), Some("ccc"));
        assert_eq!(msgs[2].content.as_deref(), Some("dddddddd"));
    }

    #[tokio::test]
    async fn test_clear_resets_tokens() {
        let memory = TokenWindowMemory::new(100);
        memory.add_message(Message::user("hello")).await.unwrap();
        assert!(memory.current_tokens().await > 0);

        memory.clear().await.unwrap();

        assert_eq!(memory.current_tokens().await, 0);
        assert!(memory.get_messages().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_default_estimator_counts_tool_calls() {
        let memory = TokenWindowMemory::new(10_000);

        let msg = Message::assistant_with_tool_calls(vec![ToolCall {
            id: "tc_1".into(),
            name: "calculator".into(),
            arguments: serde_json::json!({"expression": "2+2"}),
        }]);
        memory.add_message(msg).await.unwrap();

        assert!(memory.current_tokens().await > 0);
    }

    #[tokio::test]
    async fn test_custom_token_counter() {
        let memory = TokenWindowMemory::new(5).with_token_counter(|_| 1);

        for i in 0..7 {
            memory
                .add_message(Message::user(format!("msg{i}")))
                .await
                .unwrap();
        }

        let msgs = memory.get_messages().await.unwrap();
        assert_eq!(msgs.len(), 5);
        assert_eq!(msgs[0].content.as_deref(), Some("msg2"));
    }

    #[tokio::test]
    async fn test_single_message_exceeds_budget() {
        let memory = TokenWindowMemory::new(5).with_token_counter(|msg| {
            msg.content.as_ref().map_or(0, |c| c.len())
        });

        memory
            .add_message(Message::user("short"))
            .await
            .unwrap();
        // "this is a very long message" = 27 tokens, exceeds budget but still kept as last msg
        memory
            .add_message(Message::user("this is a very long message"))
            .await
            .unwrap();

        let msgs = memory.get_messages().await.unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(
            msgs[0].content.as_deref(),
            Some("this is a very long message")
        );
    }

    #[tokio::test]
    async fn test_empty_memory() {
        let memory = TokenWindowMemory::new(100);
        assert_eq!(memory.current_tokens().await, 0);
        assert!(memory.get_messages().await.unwrap().is_empty());
    }
}
