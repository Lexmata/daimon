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

    /// Runs `f` over the stored messages without handing out an owned copy.
    ///
    /// The default implementation falls back to [`Memory::get_messages`]
    /// (one full clone); in-process backends override it to borrow their
    /// storage under the lock, so read-only consumers (serializing history
    /// to disk, measuring it, rendering it) don't pay an O(history) deep
    /// copy per call.
    fn with_messages<R, F>(&self, f: F) -> impl Future<Output = Result<R>> + Send
    where
        F: FnOnce(&[Message]) -> R + Send,
        R: Send,
    {
        async move { Ok(f(&self.get_messages().await?)) }
    }

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

    /// Object-safe counterpart of [`Memory::with_messages`]. Callers return
    /// values by writing into locals captured by `f`.
    ///
    /// Default-implemented (via [`ErasedMemory::get_messages_erased`], one
    /// full clone) so direct `ErasedMemory` implementors written before this
    /// method existed keep compiling. The blanket impl for `T: Memory`
    /// overrides it to route through [`Memory::with_messages`], so backends
    /// that override that get their no-clone path through [`SharedMemory`]
    /// too.
    fn with_messages_erased<'a>(
        &'a self,
        f: Box<dyn FnOnce(&[Message]) + Send + 'a>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let messages = self.get_messages_erased().await?;
            f(&messages);
            Ok(())
        })
    }

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

    fn with_messages_erased<'a>(
        &'a self,
        f: Box<dyn FnOnce(&[Message]) + Send + 'a>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(self.with_messages(f))
    }

    fn clear_erased<'a>(&'a self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(self.clear())
    }
}

/// Shared ownership of memory via `Arc<dyn ErasedMemory>`. Used by the agent.
pub type SharedMemory = std::sync::Arc<dyn ErasedMemory>;

#[cfg(test)]
mod tests {
    use super::{ErasedMemory, Memory, SharedMemory};
    use crate::{Message, Result, Role};
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};

    /// A provider-crate-style Memory impl using only daimon_core items.
    ///
    /// Deliberately does NOT override `with_messages`: it doubles as a
    /// compile-level regression test that an external `Memory` impl written
    /// before `with_messages` existed still compiles and works erased.
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

    #[tokio::test]
    async fn with_messages_default_matches_get_messages() {
        let mem = VecMemory(Mutex::new(Vec::new()));
        mem.add_message(&Message::user("one")).await.unwrap();
        mem.add_message(&Message::assistant("two")).await.unwrap();

        let owned = mem.get_messages().await.unwrap();
        let borrowed = mem
            .with_messages(|messages| {
                messages
                    .iter()
                    .map(|m| (m.role.clone(), m.content.clone()))
                    .collect::<Vec<_>>()
            })
            .await
            .unwrap();

        assert_eq!(borrowed.len(), owned.len());
        for (seen, expected) in borrowed.iter().zip(&owned) {
            assert_eq!(seen.0, expected.role);
            assert_eq!(seen.1, expected.content);
        }
    }

    #[tokio::test]
    async fn with_messages_erased_works_through_shared_memory() {
        let shared: SharedMemory = Arc::new(VecMemory(Mutex::new(Vec::new())));
        shared
            .add_message_erased(&Message::user("x"))
            .await
            .unwrap();
        shared
            .add_message_erased(&Message::assistant("y"))
            .await
            .unwrap();

        // The erased visitor returns () — values come back through a
        // captured local.
        let mut seen: Vec<Option<String>> = Vec::new();
        shared
            .with_messages_erased(Box::new(|messages| {
                seen.extend(messages.iter().map(|m| m.content.clone()));
            }))
            .await
            .unwrap();

        assert_eq!(seen, vec![Some("x".into()), Some("y".into())]);
    }

    /// A direct `ErasedMemory` implementor (no `Memory` impl) that predates
    /// `with_messages_erased`: relies on the trait's default body, proving
    /// such impls keep compiling and functioning.
    struct DirectErased(Mutex<Vec<Message>>);

    impl ErasedMemory for DirectErased {
        fn add_message_erased<'a>(
            &'a self,
            message: &'a Message,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
            Box::pin(async move {
                self.0.lock().unwrap().push(message.clone());
                Ok(())
            })
        }

        fn get_messages_erased<'a>(
            &'a self,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<Message>>> + Send + 'a>> {
            Box::pin(async move { Ok(self.0.lock().unwrap().clone()) })
        }

        fn clear_erased<'a>(&'a self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
            Box::pin(async move {
                self.0.lock().unwrap().clear();
                Ok(())
            })
        }
    }

    #[tokio::test]
    async fn direct_erased_impl_gets_default_with_messages_erased() {
        let shared: SharedMemory = Arc::new(DirectErased(Mutex::new(Vec::new())));
        shared
            .add_message_erased(&Message::user("hello"))
            .await
            .unwrap();

        let mut count = 0usize;
        shared
            .with_messages_erased(Box::new(|messages| count = messages.len()))
            .await
            .unwrap();
        assert_eq!(count, 1);
    }
}
