use std::future::Future;
use std::pin::Pin;

use crate::error::Result;
use crate::model::types::Message;

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
