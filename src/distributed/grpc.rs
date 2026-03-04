//! gRPC transport for distributed agent execution.
//!
//! Provides [`GrpcBrokerServer`] to expose any [`TaskBroker`] over gRPC,
//! and [`GrpcBrokerClient`] to connect to a remote broker and implement
//! [`TaskBroker`] transparently.
//!
//! Enable with `feature = "grpc"`.
//!
//! ## Server
//!
//! ```ignore
//! use daimon::distributed::{InProcessBroker, GrpcBrokerServer};
//!
//! let broker = InProcessBroker::new(64);
//! GrpcBrokerServer::new(broker)
//!     .serve("[::1]:50051")
//!     .await?;
//! ```
//!
//! ## Client
//!
//! ```ignore
//! use daimon::distributed::{GrpcBrokerClient, TaskBroker, AgentTask};
//!
//! let client = GrpcBrokerClient::connect("http://[::1]:50051").await?;
//! let task_id = client.submit(AgentTask::new("Hello")).await?;
//! ```

use std::sync::Arc;

use tonic::{Request, Response, Status};

use crate::error::{DaimonError, Result};

use super::broker::{ErasedTaskBroker, TaskBroker};
use super::types::{AgentTask, TaskResult, TaskStatus};

pub mod proto {
    tonic::include_proto!("daimon.distributed");
}

use proto::task_broker_service_client::TaskBrokerServiceClient;
use proto::task_broker_service_server::{TaskBrokerService, TaskBrokerServiceServer};

/// Wraps a [`TaskBroker`] and serves it as a gRPC service.
pub struct GrpcBrokerServer {
    broker: Arc<dyn ErasedTaskBroker>,
}

impl GrpcBrokerServer {
    /// Creates a new server wrapping the given broker.
    pub fn new<B: TaskBroker + 'static>(broker: B) -> Self {
        Self {
            broker: Arc::new(broker),
        }
    }

    /// Creates a server from an already-erased broker.
    pub fn from_erased(broker: Arc<dyn ErasedTaskBroker>) -> Self {
        Self { broker }
    }

    /// Starts the gRPC server on the given address.
    pub async fn serve(self, addr: impl Into<String>) -> Result<()> {
        let addr = addr
            .into()
            .parse()
            .map_err(|e| DaimonError::Other(format!("invalid address: {e}")))?;

        let svc = GrpcBrokerSvc {
            broker: self.broker,
        };

        tonic::transport::Server::builder()
            .add_service(TaskBrokerServiceServer::new(svc))
            .serve(addr)
            .await
            .map_err(|e| DaimonError::Other(format!("grpc server: {e}")))?;

        Ok(())
    }
}

struct GrpcBrokerSvc {
    broker: Arc<dyn ErasedTaskBroker>,
}

#[tonic::async_trait]
impl TaskBrokerService for GrpcBrokerSvc {
    async fn submit(
        &self,
        request: Request<proto::SubmitRequest>,
    ) -> std::result::Result<Response<proto::SubmitResponse>, Status> {
        let req = request.into_inner();
        let task: AgentTask = serde_json::from_str(&req.task_json)
            .map_err(|e| Status::invalid_argument(format!("invalid task json: {e}")))?;

        let task_id = self
            .broker
            .submit_erased(task)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(proto::SubmitResponse { task_id }))
    }

    async fn get_status(
        &self,
        request: Request<proto::StatusRequest>,
    ) -> std::result::Result<Response<proto::StatusResponse>, Status> {
        let req = request.into_inner();
        let status = self
            .broker
            .status_erased(&req.task_id)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let status_json = serde_json::to_string(&status)
            .map_err(|e| Status::internal(format!("serialize status: {e}")))?;

        Ok(Response::new(proto::StatusResponse { status_json }))
    }

    async fn complete(
        &self,
        request: Request<proto::CompleteRequest>,
    ) -> std::result::Result<Response<proto::Empty>, Status> {
        let req = request.into_inner();
        let result: TaskResult = serde_json::from_str(&req.result_json)
            .map_err(|e| Status::invalid_argument(format!("invalid result json: {e}")))?;

        self.broker
            .complete_erased(&req.task_id, result)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(proto::Empty {}))
    }

    async fn fail(
        &self,
        request: Request<proto::FailRequest>,
    ) -> std::result::Result<Response<proto::Empty>, Status> {
        let req = request.into_inner();

        self.broker
            .fail_erased(&req.task_id, req.error)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(proto::Empty {}))
    }
}

/// A [`TaskBroker`] that delegates to a remote gRPC broker server.
pub struct GrpcBrokerClient {
    inner: tokio::sync::Mutex<TaskBrokerServiceClient<tonic::transport::Channel>>,
}

impl GrpcBrokerClient {
    /// Connects to a remote gRPC broker.
    ///
    /// ```ignore
    /// let client = GrpcBrokerClient::connect("http://[::1]:50051").await?;
    /// ```
    pub async fn connect(addr: impl Into<String>) -> Result<Self> {
        let addr = addr.into();
        let client = TaskBrokerServiceClient::connect(addr)
            .await
            .map_err(|e| DaimonError::Other(format!("grpc connect: {e}")))?;

        Ok(Self {
            inner: tokio::sync::Mutex::new(client),
        })
    }
}

impl TaskBroker for GrpcBrokerClient {
    async fn submit(&self, task: AgentTask) -> Result<String> {
        let task_json = serde_json::to_string(&task)
            .map_err(|e| DaimonError::Other(format!("serialize task: {e}")))?;

        let resp = self
            .inner
            .lock()
            .await
            .submit(Request::new(proto::SubmitRequest { task_json }))
            .await
            .map_err(|e| DaimonError::Other(format!("grpc submit: {e}")))?;

        Ok(resp.into_inner().task_id)
    }

    async fn status(&self, task_id: &str) -> Result<TaskStatus> {
        let resp = self
            .inner
            .lock()
            .await
            .get_status(Request::new(proto::StatusRequest {
                task_id: task_id.to_string(),
            }))
            .await
            .map_err(|e| DaimonError::Other(format!("grpc status: {e}")))?;

        let status: TaskStatus = serde_json::from_str(&resp.into_inner().status_json)
            .map_err(|e| DaimonError::Other(format!("deserialize status: {e}")))?;

        Ok(status)
    }

    async fn receive(&self) -> Result<Option<AgentTask>> {
        Err(DaimonError::Other(
            "receive() is not supported over gRPC; use TaskWorker on the server side".into(),
        ))
    }

    async fn complete(&self, task_id: &str, result: TaskResult) -> Result<()> {
        let result_json = serde_json::to_string(&result)
            .map_err(|e| DaimonError::Other(format!("serialize result: {e}")))?;

        self.inner
            .lock()
            .await
            .complete(Request::new(proto::CompleteRequest {
                task_id: task_id.to_string(),
                result_json,
            }))
            .await
            .map_err(|e| DaimonError::Other(format!("grpc complete: {e}")))?;

        Ok(())
    }

    async fn fail(&self, task_id: &str, error: String) -> Result<()> {
        self.inner
            .lock()
            .await
            .fail(Request::new(proto::FailRequest {
                task_id: task_id.to_string(),
                error,
            }))
            .await
            .map_err(|e| DaimonError::Other(format!("grpc fail: {e}")))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::InProcessBroker;

    #[tokio::test]
    async fn test_grpc_roundtrip() {
        let broker = InProcessBroker::new(32);
        let broker_clone = broker.clone();

        let server_handle = tokio::spawn(async move {
            GrpcBrokerServer::new(broker_clone)
                .serve("[::1]:0")
                .await
                .ok();
        });

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // The test just ensures the types compile and the server starts.
        // A real integration test would bind to a known port.
        server_handle.abort();
    }

    #[test]
    fn test_proto_types_compile() {
        let _ = proto::SubmitRequest {
            task_json: "{}".into(),
        };
        let _ = proto::SubmitResponse {
            task_id: "t-1".into(),
        };
        let _ = proto::StatusRequest {
            task_id: "t-1".into(),
        };
        let _ = proto::StatusResponse {
            status_json: "\"Pending\"".into(),
        };
        let _ = proto::CompleteRequest {
            task_id: "t-1".into(),
            result_json: "{}".into(),
        };
        let _ = proto::FailRequest {
            task_id: "t-1".into(),
            error: "oops".into(),
        };
        let _ = proto::Empty {};
    }

    #[tokio::test]
    async fn test_grpc_server_and_client() {
        let broker = InProcessBroker::new(32);
        let broker_for_server = broker.clone();

        let listener = tokio::net::TcpListener::bind("[::1]:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let _server_handle = tokio::spawn(async move {
            GrpcBrokerServer::new(broker_for_server)
                .serve(addr.to_string())
                .await
                .ok();
        });

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        let client = GrpcBrokerClient::connect(format!("http://{addr}"))
            .await
            .unwrap();

        let task = AgentTask::new("test via grpc");
        let task_id = client.submit(task).await.unwrap();
        assert!(!task_id.is_empty());

        let status = client.status(&task_id).await.unwrap();
        assert!(
            matches!(status, TaskStatus::Pending | TaskStatus::Running),
            "expected pending or running, got {status:?}"
        );
    }
}
