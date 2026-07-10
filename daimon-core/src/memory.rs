//! Conversation memory trait. Implement [`Memory`] for custom backends;
//! built-in implementations live in the `daimon` facade crate.

use std::future::Future;
use std::pin::Pin;

use crate::error::Result;
use crate::types::Message;

/// Trait for conversation memory backends. Stores and retrieves messages for agent context.
pub trait Memory: Send + Sync {
    /// Appends a message to the history. Order is preserved.
    ///
    /// Takes a borrow so callers that keep the message (the agent runner
    /// appends it to its working log after persisting) don't clone it per
    /// iteration. Backends that store owned messages clone internally;
    /// serializing backends (Redis, SQLite) never need ownership at all.
    fn add_message(&self, message: &Message) -> impl Future<Output = Result<()>> + Send;

    /// Returns all stored messages in order. Used to build context for the model.
    fn get_messages(&self) -> impl Future<Output = Result<Vec<Message>>> + Send;

    /// Removes all messages. Use when starting a new conversation.
    fn clear(&self) -> impl Future<Output = Result<()>> + Send;
}

/// Object-safe wrapper for the `Memory` trait, enabling dynamic dispatch via `Arc<dyn ErasedMemory>`.
pub trait ErasedMemory: Send + Sync {
    fn add_message_erased<'a>(
        &'a self,
        message: &'a Message,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    fn get_messages_erased<'a>(
        &'a self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Message>>> + Send + 'a>>;

    fn clear_erased<'a>(&'a self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
}

impl<T: Memory> ErasedMemory for T {
    fn add_message_erased<'a>(
        &'a self,
        message: &'a Message,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(self.add_message(message))
    }

    fn get_messages_erased<'a>(
        &'a self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Message>>> + Send + 'a>> {
        Box::pin(self.get_messages())
    }

    fn clear_erased<'a>(&'a self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(self.clear())
    }
}

/// Shared ownership of memory via `Arc<dyn ErasedMemory>`. Used by the agent.
pub type SharedMemory = std::sync::Arc<dyn ErasedMemory>;

#[cfg(test)]
mod tests {
    use super::{Memory, SharedMemory};
    use crate::{Message, Result, Role};
    use std::sync::{Arc, Mutex};

    /// A provider-crate-style Memory impl using only daimon_core items.
    struct VecMemory(Mutex<Vec<Message>>);

    impl Memory for VecMemory {
        async fn add_message(&self, message: &Message) -> Result<()> {
            self.0.lock().unwrap().push(message.clone());
            Ok(())
        }

        async fn get_messages(&self) -> Result<Vec<Message>> {
            Ok(self.0.lock().unwrap().clone())
        }

        async fn clear(&self) -> Result<()> {
            self.0.lock().unwrap().clear();
            Ok(())
        }
    }

    #[tokio::test]
    async fn memory_is_implementable_from_core_alone() {
        let mem = VecMemory(Mutex::new(Vec::new()));
        mem.add_message(&Message::user("hi")).await.unwrap();
        assert_eq!(mem.get_messages().await.unwrap().len(), 1);
        assert_eq!(mem.get_messages().await.unwrap()[0].role, Role::User);
        mem.clear().await.unwrap();
        assert!(mem.get_messages().await.unwrap().is_empty());

        let shared: SharedMemory = Arc::new(VecMemory(Mutex::new(Vec::new())));
        shared
            .add_message_erased(&Message::user("x"))
            .await
            .unwrap();
        assert_eq!(shared.get_messages_erased().await.unwrap().len(), 1);
    }
}
