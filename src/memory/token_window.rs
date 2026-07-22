//! Token-budget-based conversation memory.

use std::collections::VecDeque;

use tokio::sync::Mutex;

use crate::error::Result;
use crate::memory::Memory;
use crate::model::types::Message;

/// Estimates the byte-length of a JSON value without allocating a String.
fn json_value_len(v: &serde_json::Value) -> usize {
    match v {
        serde_json::Value::Null => 4,
        serde_json::Value::Bool(b) => {
            if *b {
                4
            } else {
                5
            }
        }
        serde_json::Value::Number(n) => {
            // fast path: count digits without allocation
            let mut buf = itoa::Buffer::new();
            if let Some(i) = n.as_i64() {
                buf.format(i).len()
            } else if let Some(u) = n.as_u64() {
                buf.format(u).len()
            } else {
                // f64: use ryu for allocation-free formatting
                let mut fbuf = ryu::Buffer::new();
                fbuf.format(n.as_f64().unwrap_or(0.0)).len()
            }
        }
        serde_json::Value::String(s) => s.len() + 2,
        serde_json::Value::Array(arr) => {
            2 + arr.iter().map(|v| json_value_len(v) + 1).sum::<usize>()
        }
        serde_json::Value::Object(map) => {
            2 + map
                .iter()
                .map(|(k, v)| k.len() + 3 + json_value_len(v) + 1)
                .sum::<usize>()
        }
    }
}

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
        chars += json_value_len(&tc.arguments);
    }

    if let Some(ref id) = msg.tool_call_id {
        chars += id.len();
    }

    chars += 6;

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
/// Eviction is group-aware: an assistant message carrying tool calls and its
/// contiguous following tool result messages are evicted together, never
/// split, so the history never contains orphaned tool results. If such a
/// group spans the entire window it is kept even while over budget, matching
/// the existing guarantee that the newest message is never evicted.
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
    inner: Mutex<TokenWindowInner>,
    max_tokens: usize,
    token_counter: TokenCounterFn,
}

struct TokenWindowInner {
    messages: VecDeque<Message>,
    token_counts: VecDeque<usize>,
    total_tokens: usize,
}

impl TokenWindowMemory {
    /// Creates a new token window with the given maximum token budget.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            inner: Mutex::new(TokenWindowInner {
                messages: VecDeque::new(),
                token_counts: VecDeque::new(),
                total_tokens: 0,
            }),
            max_tokens,
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
        self.inner.lock().await.total_tokens
    }
}

impl Memory for TokenWindowMemory {
    async fn add_message(&self, message: &Message) -> Result<()> {
        let tokens = (self.token_counter)(message);
        let mut inner = self.inner.lock().await;

        inner.messages.push_back(message.clone());
        inner.token_counts.push_back(tokens);
        inner.total_tokens += tokens;

        while inner.total_tokens > self.max_tokens {
            let group = crate::memory::eviction_group_len(&inner.messages);
            if group == 0 || group >= inner.messages.len() {
                // Never evict the entire history: keep at least the newest
                // message, or the atomic tool-call group spanning the window,
                // even when it exceeds the budget.
                break;
            }
            for _ in 0..group {
                inner.messages.pop_front();
                if let Some(removed_tokens) = inner.token_counts.pop_front() {
                    inner.total_tokens -= removed_tokens;
                }
            }
        }

        Ok(())
    }

    async fn get_messages(&self) -> Result<Vec<Message>> {
        let mut inner = self.inner.lock().await;
        Ok(inner.messages.make_contiguous().to_vec())
    }

    /// Borrows the storage under the lock — no per-call clone of the
    /// history. The closure is sync and non-blocking, so holding the async
    /// lock across it is fine.
    async fn with_messages<R, F>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&[Message]) -> R + Send,
        R: Send,
    {
        let mut inner = self.inner.lock().await;
        Ok(f(inner.messages.make_contiguous()))
    }

    async fn clear(&self) -> Result<()> {
        let mut inner = self.inner.lock().await;
        inner.messages.clear();
        inner.token_counts.clear();
        inner.total_tokens = 0;
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
        memory.add_message(&Message::user("hello")).await.unwrap();
        memory.add_message(&Message::assistant("hi")).await.unwrap();

        let msgs = memory.get_messages().await.unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::User);
        assert_eq!(msgs[1].role, Role::Assistant);
    }

    #[tokio::test]
    async fn test_evicts_old_messages_when_over_budget() {
        // Use a custom counter: 1 token per character for predictability
        let memory = TokenWindowMemory::new(20)
            .with_token_counter(|msg| msg.content.as_ref().map_or(0, |c| c.len()));

        // "aaaaaaaaaa" = 10 tokens
        memory
            .add_message(&Message::user("aaaaaaaaaa"))
            .await
            .unwrap();
        // "bbbbbbbbbb" = 10 tokens, total 20 (at limit)
        memory
            .add_message(&Message::user("bbbbbbbbbb"))
            .await
            .unwrap();
        assert_eq!(memory.get_messages().await.unwrap().len(), 2);

        // "cccccccccc" = 10 tokens, total would be 30 > 20, evict "aaa..."
        memory
            .add_message(&Message::user("cccccccccc"))
            .await
            .unwrap();

        let msgs = memory.get_messages().await.unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].content.as_deref(), Some("bbbbbbbbbb"));
        assert_eq!(msgs[1].content.as_deref(), Some("cccccccccc"));
    }

    #[tokio::test]
    async fn test_evicts_multiple_to_fit() {
        let memory = TokenWindowMemory::new(15)
            .with_token_counter(|msg| msg.content.as_ref().map_or(0, |c| c.len()));

        memory.add_message(&Message::user("aaa")).await.unwrap(); // 3
        memory.add_message(&Message::user("bbb")).await.unwrap(); // 3
        memory.add_message(&Message::user("ccc")).await.unwrap(); // 3, total 9

        // Adding 8 tokens: total would be 17 > 15, evict "aaa" (3) -> 14, fits
        memory
            .add_message(&Message::user("dddddddd"))
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
        memory.add_message(&Message::user("hello")).await.unwrap();
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
        memory.add_message(&msg).await.unwrap();

        assert!(memory.current_tokens().await > 0);
    }

    #[tokio::test]
    async fn test_custom_token_counter() {
        let memory = TokenWindowMemory::new(5).with_token_counter(|_| 1);

        for i in 0..7 {
            memory
                .add_message(&Message::user(format!("msg{i}")))
                .await
                .unwrap();
        }

        let msgs = memory.get_messages().await.unwrap();
        assert_eq!(msgs.len(), 5);
        assert_eq!(msgs[0].content.as_deref(), Some("msg2"));
    }

    #[tokio::test]
    async fn test_eviction_drops_tool_call_group_atomically() {
        // 1 token per message for predictable eviction boundaries.
        let memory = TokenWindowMemory::new(3).with_token_counter(|_| 1);
        let tool_call = ToolCall {
            id: "tc_1".into(),
            name: "calc".into(),
            arguments: serde_json::json!({"expr": "1+1"}),
        };

        memory
            .add_message(&Message::user("question"))
            .await
            .unwrap();
        memory
            .add_message(&Message::assistant_with_tool_calls(vec![tool_call]))
            .await
            .unwrap();
        memory
            .add_message(&Message::tool_result("tc_1", "result one"))
            .await
            .unwrap();
        // 4 tokens > 3: evicts the lone user message first.
        memory
            .add_message(&Message::tool_result("tc_1", "result two"))
            .await
            .unwrap();
        // 4 tokens > 3 again: the front group is assistant + 2 tool results,
        // which must be evicted as a whole.
        memory
            .add_message(&Message::assistant("final answer"))
            .await
            .unwrap();

        let msgs = memory.get_messages().await.unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, Role::Assistant);
        assert_eq!(msgs[0].content.as_deref(), Some("final answer"));
        assert!(msgs.iter().all(|m| m.role != Role::Tool));
        assert_eq!(memory.current_tokens().await, 1);
    }

    #[tokio::test]
    async fn test_tool_call_group_spanning_window_is_kept() {
        let memory = TokenWindowMemory::new(2).with_token_counter(|_| 1);
        let tool_call = ToolCall {
            id: "tc_1".into(),
            name: "calc".into(),
            arguments: serde_json::json!({"expr": "1+1"}),
        };

        memory
            .add_message(&Message::assistant_with_tool_calls(vec![tool_call]))
            .await
            .unwrap();
        memory
            .add_message(&Message::tool_result("tc_1", "result one"))
            .await
            .unwrap();
        memory
            .add_message(&Message::tool_result("tc_1", "result two"))
            .await
            .unwrap();

        // The group covers the whole window: it must not be split, so the
        // memory temporarily exceeds the token budget.
        let msgs = memory.get_messages().await.unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(memory.current_tokens().await, 3);
    }

    #[tokio::test]
    async fn test_single_message_exceeds_budget() {
        let memory = TokenWindowMemory::new(5)
            .with_token_counter(|msg| msg.content.as_ref().map_or(0, |c| c.len()));

        memory.add_message(&Message::user("short")).await.unwrap();
        // "this is a very long message" = 27 tokens, exceeds budget but still kept as last msg
        memory
            .add_message(&Message::user("this is a very long message"))
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
    async fn test_with_messages_matches_get_messages_after_eviction() {
        // 1 token per message: window of 3 forces eviction on the 5th add.
        let memory = TokenWindowMemory::new(3).with_token_counter(|_| 1);
        for i in 0..5 {
            memory
                .add_message(&Message::user(format!("msg{i}")))
                .await
                .unwrap();
        }

        let owned = memory.get_messages().await.unwrap();
        let borrowed = memory
            .with_messages(|messages| messages.to_vec())
            .await
            .unwrap();

        assert_eq!(borrowed.len(), owned.len());
        for (b, o) in borrowed.iter().zip(&owned) {
            assert_eq!(b.role, o.role);
            assert_eq!(b.content, o.content);
        }
        // Eviction happened, and the borrowed view reflects it.
        assert_eq!(borrowed.len(), 3);
        assert_eq!(borrowed[0].content.as_deref(), Some("msg2"));
    }

    #[tokio::test]
    async fn test_empty_memory() {
        let memory = TokenWindowMemory::new(100);
        assert_eq!(memory.current_tokens().await, 0);
        assert!(memory.get_messages().await.unwrap().is_empty());
    }
}
