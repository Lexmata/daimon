//! Inter-session communication — message passing between concurrent agent
//! sessions.
//!
//! Where [`distributed`](crate::distributed) hands stateless *work* to a pool
//! of workers, this module lets long-lived, stateful sessions talk to one
//! another. Each session has an inbox; peers can [`send`](SessionBus::send) it
//! a directed message or [`broadcast`](SessionBus::broadcast) to everyone.
//!
//! The core abstraction is the [`SessionBus`] trait (defined in
//! [`daimon_core::session`] and re-exported here). This crate ships an
//! in-process implementation, [`InProcessSessionBus`], backed by bounded
//! per-session `tokio::sync::mpsc` FIFO queues: each session has an ordered
//! mailbox, so messages **queue behind whatever the session is currently
//! doing** and are delivered in send order, one at a time, with backpressure
//! when a busy session falls behind. Implement [`SessionBus`] over Redis,
//! NATS, etc. for cross-process messaging.
//!
//! Running agents can send messages via [`SendMessageTool`], which wraps a
//! [`SharedSessionBus`] as a callable tool.
//!
//! ```ignore
//! use daimon::session::{InProcessSessionBus, SessionBus, SessionId, SessionMessage};
//!
//! let bus = InProcessSessionBus::new();
//!
//! // Each session subscribes to its own inbox.
//! let mut researcher = bus.subscribe(SessionId::new("researcher")).await?;
//! let mut writer = bus.subscribe(SessionId::new("writer")).await?;
//!
//! // The researcher hands findings to the writer.
//! bus.send(SessionMessage::direct("researcher", "writer", "here are the facts")).await?;
//!
//! let msg = writer.recv().await?.expect("inbox open");
//! assert_eq!(msg.body, "here are the facts");
//! ```

mod bus;
mod tool;

pub use bus::{DEFAULT_INBOX_CAPACITY, InProcessSessionBus};
pub use daimon_core::session::{
    ErasedSessionBus, ReceiverStream, SessionBus, SessionId, SessionMessage, SessionReceiver,
    SharedSessionBus,
};
pub use tool::SendMessageTool;
