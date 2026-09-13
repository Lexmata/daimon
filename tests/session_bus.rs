//! Integration tests for inter-session communication (`daimon::session`).
//!
//! Exercises the public facade: two independent sessions exchanging directed
//! messages, broadcasts fanning out to all subscribers, and a running agent
//! sending a cross-session message via [`SendMessageTool`].

use std::sync::Arc;
use std::time::Duration;

use daimon::session::{
    InProcessSessionBus, SendMessageTool, SessionBus, SessionId, SessionMessage,
};
use daimon::tool::Tool;

/// Two sessions exchange a directed message end-to-end.
#[tokio::test]
async fn directed_message_flows_between_sessions() {
    let bus = InProcessSessionBus::new();

    let researcher = SessionId::new("researcher");
    let writer = SessionId::new("writer");

    let mut writer_inbox = bus.subscribe(writer.clone()).await.unwrap();
    let _researcher_inbox = bus.subscribe(researcher.clone()).await.unwrap();

    bus.send(SessionMessage::direct(
        researcher.clone(),
        writer.clone(),
        "here are the facts",
    ))
    .await
    .unwrap();

    let msg = tokio::time::timeout(Duration::from_secs(1), writer_inbox.recv())
        .await
        .expect("recv should not time out")
        .unwrap()
        .expect("inbox open");

    assert_eq!(msg.body, "here are the facts");
    assert_eq!(msg.from, researcher);
    assert_eq!(msg.to, Some(writer));
}

/// A broadcast reaches every subscribed session.
#[tokio::test]
async fn broadcast_reaches_every_session() {
    let bus = InProcessSessionBus::new();

    let mut a = bus.subscribe(SessionId::new("a")).await.unwrap();
    let mut b = bus.subscribe(SessionId::new("b")).await.unwrap();
    let mut c = bus.subscribe(SessionId::new("c")).await.unwrap();

    bus.broadcast(SessionMessage::broadcast("coordinator", "shutting down"))
        .await
        .unwrap();

    for inbox in [&mut a, &mut b, &mut c] {
        let msg = inbox.recv().await.unwrap().expect("inbox open");
        assert_eq!(msg.body, "shutting down");
        assert!(msg.is_broadcast());
    }
}

/// A message directed at a session nobody is listening on is dropped, not an
/// error, and does not disturb other inboxes.
#[tokio::test]
async fn directed_message_to_absent_session_is_dropped() {
    let bus = InProcessSessionBus::new();
    let mut present = bus.subscribe(SessionId::new("present")).await.unwrap();

    bus.send(SessionMessage::direct("x", "absent", "nobody home"))
        .await
        .unwrap();

    // The present session's inbox stays empty.
    let idle = tokio::time::timeout(Duration::from_millis(100), present.recv()).await;
    assert!(
        idle.is_err(),
        "present session must not receive foreign mail"
    );
}

/// A running-agent's tool sends a message that a peer session receives.
#[tokio::test]
async fn send_message_tool_delivers_to_peer_session() {
    let bus = Arc::new(InProcessSessionBus::new());

    let mut peer = bus.subscribe(SessionId::new("peer")).await.unwrap();

    // The tool sends on behalf of the "planner" session.
    let tool = SendMessageTool::new("planner", bus.clone());
    let output = tool
        .execute(&serde_json::json!({
            "to": "peer",
            "body": "please review the plan"
        }))
        .await
        .unwrap();
    assert!(!output.is_error);

    let msg = peer.recv().await.unwrap().expect("inbox open");
    assert_eq!(msg.body, "please review the plan");
    assert_eq!(msg.from, SessionId::new("planner"));
}

/// The tool can broadcast to all sessions too.
#[tokio::test]
async fn send_message_tool_broadcasts() {
    let bus = Arc::new(InProcessSessionBus::new());
    let mut a = bus.subscribe(SessionId::new("a")).await.unwrap();
    let mut b = bus.subscribe(SessionId::new("b")).await.unwrap();

    let tool = SendMessageTool::new("leader", bus.clone());
    let output = tool
        .execute(&serde_json::json!({ "body": "sync now", "broadcast": true }))
        .await
        .unwrap();
    assert!(!output.is_error);

    assert_eq!(a.recv().await.unwrap().unwrap().body, "sync now");
    assert_eq!(b.recv().await.unwrap().unwrap().body, "sync now");
}

/// The bus reports the set of registered sessions and clones share it, so a
/// message sent through one handle reaches subscribers created on another.
#[tokio::test]
async fn clones_share_the_session_registry() {
    let bus = InProcessSessionBus::new();
    let clone = bus.clone();

    let mut peer = clone.subscribe(SessionId::new("peer")).await.unwrap();

    let mut sessions = bus.sessions().await.unwrap();
    sessions.sort();
    assert_eq!(sessions, vec![SessionId::new("peer")]);

    bus.send(SessionMessage::direct("origin", "peer", "cross-handle"))
        .await
        .unwrap();
    assert_eq!(peer.recv().await.unwrap().unwrap().body, "cross-handle");
}

/// Messages sent to a session while it is busy (not draining its inbox) queue
/// in FIFO order and are all delivered, in order, once the session is free.
#[tokio::test]
async fn messages_queue_behind_busy_session_in_order() {
    let bus = InProcessSessionBus::new();
    let mut worker = bus.subscribe(SessionId::new("worker")).await.unwrap();

    // Simulate the worker being busy with current work: it does not recv yet.
    // Several messages arrive and must accumulate behind that work.
    for i in 0..8 {
        bus.send(SessionMessage::direct(
            "dispatcher",
            "worker",
            format!("job-{i}"),
        ))
        .await
        .unwrap();
    }

    // The worker finishes and drains its mailbox: strict send order, none lost.
    for i in 0..8 {
        let msg = worker.recv().await.unwrap().expect("inbox open");
        assert_eq!(msg.body, format!("job-{i}"));
    }
}
