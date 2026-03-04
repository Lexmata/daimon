use std::collections::VecDeque;

use tokio::sync::Mutex;

use crate::error::Result;
use crate::memory::traits::Memory;
use crate::model::types::Message;

/// In-memory conversation storage that keeps only the most recent N messages.
///
/// When the window is exceeded, the oldest messages are evicted. Thread-safe via internal mutex.
pub struct SlidingWindowMemory {
    messages: Mutex<VecDeque<Message>>,
    max_messages: usize,
}

impl SlidingWindowMemory {
    /// Creates a new sliding window with the given maximum message count.
    /// When exceeded, oldest messages are dropped.
    pub fn new(max_messages: usize) -> Self {
        Self {
            messages: Mutex::new(VecDeque::new()),
            max_messages,
        }
    }
}

impl Default for SlidingWindowMemory {
    /// Default window size of 50 messages.
    fn default() -> Self {
        Self::new(50)
    }
}

impl Memory for SlidingWindowMemory {
    async fn add_message(&self, message: Message) -> Result<()> {
        let mut messages = self.messages.lock().await;
        if messages.len() >= self.max_messages {
            messages.pop_front();
        }
        messages.push_back(message);
        Ok(())
    }

    async fn get_messages(&self) -> Result<Vec<Message>> {
        let mut messages = self.messages.lock().await;
        Ok(messages.make_contiguous().to_vec())
    }

    async fn clear(&self) -> Result<()> {
        let mut messages = self.messages.lock().await;
        messages.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::types::Role;

    #[tokio::test]
    async fn test_add_and_get_messages() {
        let memory = SlidingWindowMemory::new(10);
        memory.add_message(Message::user("hello")).await.unwrap();
        memory
            .add_message(Message::assistant("hi there"))
            .await
            .unwrap();

        let messages = memory.get_messages().await.unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::User);
        assert_eq!(messages[1].role, Role::Assistant);
    }

    #[tokio::test]
    async fn test_sliding_window_evicts_old_messages() {
        let memory = SlidingWindowMemory::new(3);

        for i in 0..5 {
            memory
                .add_message(Message::user(format!("msg {i}")))
                .await
                .unwrap();
        }

        let messages = memory.get_messages().await.unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].content.as_deref(), Some("msg 2"));
        assert_eq!(messages[1].content.as_deref(), Some("msg 3"));
        assert_eq!(messages[2].content.as_deref(), Some("msg 4"));
    }

    #[tokio::test]
    async fn test_clear_removes_all_messages() {
        let memory = SlidingWindowMemory::new(10);
        memory.add_message(Message::user("hello")).await.unwrap();
        memory.clear().await.unwrap();

        let messages = memory.get_messages().await.unwrap();
        assert!(messages.is_empty());
    }

    #[tokio::test]
    async fn test_default_window_size() {
        let memory = SlidingWindowMemory::default();
        assert_eq!(memory.max_messages, 50);
    }

    #[tokio::test]
    async fn test_empty_memory_returns_empty_vec() {
        let memory = SlidingWindowMemory::new(10);
        let messages = memory.get_messages().await.unwrap();
        assert!(messages.is_empty());
    }

    #[tokio::test]
    async fn test_window_size_of_one() {
        let memory = SlidingWindowMemory::new(1);
        memory.add_message(Message::user("first")).await.unwrap();
        memory.add_message(Message::user("second")).await.unwrap();

        let messages = memory.get_messages().await.unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content.as_deref(), Some("second"));
    }

    #[tokio::test]
    async fn test_messages_at_exact_capacity() {
        let memory = SlidingWindowMemory::new(3);
        memory.add_message(Message::user("a")).await.unwrap();
        memory.add_message(Message::user("b")).await.unwrap();
        memory.add_message(Message::user("c")).await.unwrap();

        let messages = memory.get_messages().await.unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].content.as_deref(), Some("a"));
    }
}
