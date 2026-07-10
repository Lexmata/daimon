//! Broker-contract integration suite.
//!
//! Every [`TaskBroker`] implementation must satisfy the same observable
//! contract: a submitted task is receivable exactly once, `complete`/`fail`
//! publish a terminal status readable by the producer, and `receive`'s
//! `Ok(None)` semantics match `none_means_closed()`. The audit's worst
//! distributed bug (workers exiting on an idle poll) lived precisely in the
//! gap between backends' divergent `None` semantics — this suite pins them.
//!
//! The in-process broker runs unconditionally. The network backends run the
//! identical contract but are `#[ignore]`d, expecting a live service:
//!
//! ```text
//! REDIS_URL=redis://127.0.0.1 cargo test --features redis --test broker_contract -- --ignored redis
//! NATS_URL=nats://127.0.0.1  cargo test --features nats  --test broker_contract -- --ignored nats
//! AMQP_URL=amqp://guest:guest@127.0.0.1 cargo test --features amqp --test broker_contract -- --ignored amqp
//! ```

use daimon::distributed::{AgentTask, TaskBroker, TaskResult, TaskStatus};

/// Submit → receive → complete: the result round-trips to the producer.
async fn contract_complete(broker: &impl TaskBroker) {
    let task = AgentTask::new("contract: complete me");
    let task_id = broker.submit(task).await.expect("submit failed");

    let received = broker
        .receive()
        .await
        .expect("receive errored")
        .expect("submitted task was not receivable");
    assert_eq!(received.task_id, task_id, "received a different task");
    assert_eq!(received.input, "contract: complete me");

    let result = TaskResult {
        task_id: task_id.clone(),
        output: "done".into(),
        iterations: 1,
        cost: 0.0,
        error: None,
    };
    broker
        .complete(&task_id, result)
        .await
        .expect("complete failed");

    match broker.status(&task_id).await.expect("status errored") {
        TaskStatus::Completed(r) => {
            assert_eq!(r.output, "done");
            assert_eq!(r.task_id, task_id);
        }
        other => panic!("expected Completed, got {other:?}"),
    }
}

/// Submit → receive → fail: the error round-trips to the producer.
async fn contract_fail(broker: &impl TaskBroker) {
    let task = AgentTask::new("contract: fail me");
    let task_id = broker.submit(task).await.expect("submit failed");

    broker
        .receive()
        .await
        .expect("receive errored")
        .expect("submitted task was not receivable");

    broker
        .fail(&task_id, "boom".into())
        .await
        .expect("fail failed");

    match broker.status(&task_id).await.expect("status errored") {
        TaskStatus::Failed(msg) => assert!(msg.contains("boom"), "lost error message: {msg}"),
        other => panic!("expected Failed, got {other:?}"),
    }
}

/// Polling brokers: an empty queue is an idle poll (`Ok(None)`), and
/// `none_means_closed()` must say so — otherwise `TaskWorker::run` exits
/// permanently the first moment the queue is quiet.
///
/// Only the feature-gated live suites exercise this (the in-process broker
/// blocks instead of idle-polling), so it is gated with them to stay out of
/// dead-code territory under `--no-default-features`.
#[cfg(any(feature = "redis", feature = "nats"))]
async fn contract_idle_poll(broker: &impl TaskBroker) {
    assert!(
        !broker.none_means_closed(),
        "polling broker must report none_means_closed() == false"
    );
    let idle = broker.receive().await.expect("idle receive errored");
    assert!(idle.is_none(), "empty queue should yield Ok(None)");
}

mod in_process {
    use super::*;
    use daimon::distributed::InProcessBroker;

    #[tokio::test]
    async fn satisfies_complete_contract() {
        contract_complete(&InProcessBroker::new(8)).await;
    }

    #[tokio::test]
    async fn satisfies_fail_contract() {
        contract_fail(&InProcessBroker::new(8)).await;
    }

    #[tokio::test]
    async fn close_means_closed() {
        // The in-process channel has a real end-of-stream signal, so None
        // must mean closed — and must only be delivered after close().
        let broker = InProcessBroker::new(8);
        assert!(broker.none_means_closed());

        broker
            .submit(AgentTask::new("drain me"))
            .await
            .expect("submit failed");
        broker.close().await;

        // Already-queued work drains before the close is observed.
        let drained = broker.receive().await.expect("receive errored");
        assert!(drained.is_some(), "queued task lost on close");
        let end = broker.receive().await.expect("receive errored");
        assert!(end.is_none(), "closed broker must yield None");
    }
}

#[cfg(feature = "redis")]
mod redis_live {
    use super::*;
    use daimon::distributed::RedisBroker;

    async fn broker() -> RedisBroker {
        let url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1".into());
        RedisBroker::new(&url, format!("contract-{}", std::process::id()))
            .await
            .expect("redis unavailable")
    }

    #[tokio::test]
    #[ignore = "requires a live Redis (set REDIS_URL)"]
    async fn satisfies_complete_contract() {
        contract_complete(&broker().await).await;
    }

    #[tokio::test]
    #[ignore = "requires a live Redis (set REDIS_URL)"]
    async fn satisfies_fail_contract() {
        contract_fail(&broker().await).await;
    }

    #[tokio::test]
    #[ignore = "requires a live Redis (set REDIS_URL)"]
    async fn satisfies_idle_poll_contract() {
        contract_idle_poll(&broker().await).await;
    }
}

#[cfg(feature = "nats")]
mod nats_live {
    use super::*;
    use daimon::distributed::NatsBroker;

    async fn broker() -> NatsBroker {
        let url = std::env::var("NATS_URL").unwrap_or_else(|_| "nats://127.0.0.1".into());
        NatsBroker::connect(&url, format!("contract{}", std::process::id()))
            .await
            .expect("nats unavailable")
    }

    #[tokio::test]
    #[ignore = "requires a live NATS with JetStream (set NATS_URL)"]
    async fn satisfies_complete_contract() {
        contract_complete(&broker().await).await;
    }

    #[tokio::test]
    #[ignore = "requires a live NATS with JetStream (set NATS_URL)"]
    async fn satisfies_fail_contract() {
        contract_fail(&broker().await).await;
    }

    #[tokio::test]
    #[ignore = "requires a live NATS with JetStream (set NATS_URL)"]
    async fn satisfies_idle_poll_contract() {
        contract_idle_poll(&broker().await).await;
    }
}

#[cfg(feature = "amqp")]
mod amqp_live {
    use super::*;
    use daimon::distributed::AmqpBroker;

    async fn broker() -> AmqpBroker {
        let url =
            std::env::var("AMQP_URL").unwrap_or_else(|_| "amqp://guest:guest@127.0.0.1".into());
        AmqpBroker::connect(&url, format!("contract-{}", std::process::id()))
            .await
            .expect("rabbitmq unavailable")
    }

    #[tokio::test]
    #[ignore = "requires a live RabbitMQ (set AMQP_URL)"]
    async fn satisfies_complete_contract() {
        contract_complete(&broker().await).await;
    }

    #[tokio::test]
    #[ignore = "requires a live RabbitMQ (set AMQP_URL)"]
    async fn satisfies_fail_contract() {
        contract_fail(&broker().await).await;
    }
}
