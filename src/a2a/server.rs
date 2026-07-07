//! Framework-agnostic A2A server handler.
//!
//! [`A2aHandler`] processes JSON-RPC 2.0 requests according to the A2A protocol
//! and delegates task execution to an agent. It manages task lifecycle (create,
//! get, cancel) using an in-memory task store.
//!
//! To expose this over HTTP, plug `handle_request` into your HTTP framework
//! of choice (axum, actix, warp, etc.).

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use tokio::sync::Mutex;

use super::types::*;
use crate::agent::Agent;

/// Default maximum number of tasks retained in the in-memory store before
/// terminal tasks start being evicted.
const DEFAULT_MAX_TASKS: usize = 1000;

/// Returns whether a task state is terminal (safe to evict).
fn is_terminal(state: &TaskState) -> bool {
    matches!(
        state,
        TaskState::Completed | TaskState::Failed | TaskState::Canceled
    )
}

/// Bounded in-memory task store with FIFO eviction of terminal tasks.
///
/// The A2A handler previously kept every task in an unbounded `HashMap` that
/// was only ever inserted into, so a long-running server leaked memory without
/// limit. This store caps the number of retained tasks and, when over the cap,
/// evicts the oldest tasks that have reached a terminal state (completed,
/// failed, or canceled). Non-terminal (in-flight) tasks are never evicted.
struct TaskStore {
    tasks: HashMap<String, A2aTask>,
    order: VecDeque<String>,
    max_tasks: usize,
}

impl TaskStore {
    fn new(max_tasks: usize) -> Self {
        Self {
            tasks: HashMap::new(),
            order: VecDeque::new(),
            max_tasks: max_tasks.max(1),
        }
    }

    fn insert(&mut self, id: String, task: A2aTask) {
        if !self.tasks.contains_key(&id) {
            self.order.push_back(id.clone());
        }
        self.tasks.insert(id, task);
        self.evict_if_needed();
    }

    fn get(&self, id: &str) -> Option<&A2aTask> {
        self.tasks.get(id)
    }

    fn get_mut(&mut self, id: &str) -> Option<&mut A2aTask> {
        self.tasks.get_mut(id)
    }

    /// Returns true if the store is at capacity and every retained task is
    /// non-terminal (in-flight). In that state a brand-new task cannot be
    /// admitted: eviction only reclaims terminal tasks, so inserting anyway
    /// would grow the map past `max_tasks` without bound. Callers use this to
    /// reject new inserts rather than dropping an in-flight task or leaking
    /// memory.
    fn is_full_of_nonterminal(&self) -> bool {
        self.tasks.len() >= self.max_tasks
            && self.tasks.values().all(|t| !is_terminal(&t.status.state))
    }

    fn evict_if_needed(&mut self) {
        while self.tasks.len() > self.max_tasks {
            // Evict the oldest task that has reached a terminal state. If none
            // of the retained tasks are terminal we stop rather than drop an
            // in-flight task.
            let mut evict_idx = None;
            for (i, id) in self.order.iter().enumerate() {
                if self
                    .tasks
                    .get(id)
                    .is_some_and(|t| is_terminal(&t.status.state))
                {
                    evict_idx = Some(i);
                    break;
                }
            }
            match evict_idx {
                Some(i) => {
                    if let Some(id) = self.order.remove(i) {
                        self.tasks.remove(&id);
                    }
                }
                None => break,
            }
        }
    }
}

/// A2A server that routes JSON-RPC requests to an agent.
///
/// Manages task lifecycle and stores tasks in a bounded in-memory store. Plug
/// this into any HTTP framework by calling
/// [`handle_request`](A2aHandler::handle_request) with the raw JSON body.
pub struct A2aHandler {
    agent: Arc<Agent>,
    card: AgentCard,
    tasks: Mutex<TaskStore>,
}

impl A2aHandler {
    /// Creates a new A2A handler with the default task-store cap
    /// (`DEFAULT_MAX_TASKS`).
    pub fn new(agent: Arc<Agent>, card: AgentCard) -> Self {
        Self::with_max_tasks(agent, card, DEFAULT_MAX_TASKS)
    }

    /// Creates a new A2A handler with a custom cap on retained tasks. Once the
    /// cap is exceeded, the oldest terminal tasks are evicted.
    pub fn with_max_tasks(agent: Arc<Agent>, card: AgentCard, max_tasks: usize) -> Self {
        Self {
            agent,
            card,
            tasks: Mutex::new(TaskStore::new(max_tasks)),
        }
    }

    /// Returns the Agent Card for discovery.
    pub fn agent_card(&self) -> &AgentCard {
        &self.card
    }

    /// Processes a raw JSON-RPC request body and returns a JSON-RPC response.
    ///
    /// Supports:
    /// - `agent/discover` — returns the Agent Card
    /// - `tasks/send` — creates or continues a task
    /// - `tasks/get` — retrieves task status
    /// - `tasks/cancel` — cancels a task
    pub async fn handle_request(&self, body: &str) -> String {
        let request: JsonRpcRequest = match serde_json::from_str(body) {
            Ok(r) => r,
            Err(e) => {
                let resp = JsonRpcResponse::error(
                    serde_json::Value::Null,
                    -32700,
                    format!("Parse error: {e}"),
                );
                return serde_json::to_string(&resp).unwrap_or_default();
            }
        };

        let response = match request.method.as_str() {
            "agent/discover" => self.handle_discover(&request).await,
            "tasks/send" => self.handle_task_send(&request).await,
            "tasks/get" => self.handle_task_get(&request).await,
            "tasks/cancel" => self.handle_task_cancel(&request).await,
            other => {
                JsonRpcResponse::error(request.id, -32601, format!("Method not found: {other}"))
            }
        };

        serde_json::to_string(&response).unwrap_or_default()
    }

    async fn handle_discover(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        match serde_json::to_value(&self.card) {
            Ok(v) => JsonRpcResponse::success(request.id.clone(), v),
            Err(e) => JsonRpcResponse::error(
                request.id.clone(),
                -32603,
                format!("Serialization error: {e}"),
            ),
        }
    }

    async fn handle_task_send(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let params: TaskSendParams = match serde_json::from_value(request.params.clone()) {
            Ok(p) => p,
            Err(e) => {
                return JsonRpcResponse::error(
                    request.id.clone(),
                    -32602,
                    format!("Invalid params: {e}"),
                );
            }
        };

        let task_id = params.id.unwrap_or_else(generate_id);
        let context_id = params.context_id.unwrap_or_else(generate_id);

        let user_text = extract_text_from_parts(&params.message.parts);

        let mut task = A2aTask {
            id: task_id.clone(),
            context_id: Some(context_id),
            status: TaskStatus {
                state: TaskState::Working,
                message: None,
            },
            artifacts: Vec::new(),
            history: vec![params.message],
            metadata: params.metadata,
        };

        {
            let mut tasks = self.tasks.lock().await;
            // Enforce a hard ceiling on in-flight tasks. Terminal-task eviction
            // cannot reclaim space when the store is full of non-terminal
            // (Working) tasks, so admitting another new task would grow the map
            // without bound (OOM). Reject the request instead. Continuing an
            // existing task (id already present) never grows the map, so it is
            // always allowed.
            if tasks.get(&task_id).is_none() && tasks.is_full_of_nonterminal() {
                return JsonRpcResponse::error(
                    request.id.clone(),
                    -32000,
                    "server busy: too many in-flight tasks".to_string(),
                );
            }
            tasks.insert(task_id.clone(), task.clone());
        }

        match self.agent.prompt(&user_text).await {
            Ok(response) => {
                let agent_message = A2aMessage {
                    role: A2aRole::Agent,
                    parts: vec![Part::Text {
                        text: response.final_text.clone(),
                    }],
                    message_id: Some(generate_id()),
                    metadata: HashMap::new(),
                };

                task.history.push(agent_message);
                task.artifacts.push(Artifact {
                    artifact_id: generate_id(),
                    name: Some("response".to_string()),
                    parts: vec![Part::Text {
                        text: response.final_text,
                    }],
                    metadata: HashMap::new(),
                });
                task.status = TaskStatus {
                    state: TaskState::Completed,
                    message: None,
                };
            }
            Err(e) => {
                task.status = TaskStatus {
                    state: TaskState::Failed,
                    message: Some(A2aMessage {
                        role: A2aRole::Agent,
                        parts: vec![Part::Text {
                            text: e.to_string(),
                        }],
                        message_id: None,
                        metadata: HashMap::new(),
                    }),
                };
            }
        }

        {
            let mut tasks = self.tasks.lock().await;
            tasks.insert(task_id, task.clone());
        }

        match serde_json::to_value(&task) {
            Ok(v) => JsonRpcResponse::success(request.id.clone(), v),
            Err(e) => JsonRpcResponse::error(
                request.id.clone(),
                -32603,
                format!("Serialization error: {e}"),
            ),
        }
    }

    async fn handle_task_get(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let params: TaskGetParams = match serde_json::from_value(request.params.clone()) {
            Ok(p) => p,
            Err(e) => {
                return JsonRpcResponse::error(
                    request.id.clone(),
                    -32602,
                    format!("Invalid params: {e}"),
                );
            }
        };

        let tasks = self.tasks.lock().await;
        match tasks.get(&params.id) {
            Some(task) => match serde_json::to_value(task) {
                Ok(v) => JsonRpcResponse::success(request.id.clone(), v),
                Err(e) => JsonRpcResponse::error(
                    request.id.clone(),
                    -32603,
                    format!("Serialization error: {e}"),
                ),
            },
            None => JsonRpcResponse::error(
                request.id.clone(),
                -32001,
                format!("Task not found: {}", params.id),
            ),
        }
    }

    async fn handle_task_cancel(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        let params: TaskCancelParams = match serde_json::from_value(request.params.clone()) {
            Ok(p) => p,
            Err(e) => {
                return JsonRpcResponse::error(
                    request.id.clone(),
                    -32602,
                    format!("Invalid params: {e}"),
                );
            }
        };

        let mut tasks = self.tasks.lock().await;
        match tasks.get_mut(&params.id) {
            Some(task) => {
                task.status = TaskStatus {
                    state: TaskState::Canceled,
                    message: None,
                };
                match serde_json::to_value(&*task) {
                    Ok(v) => JsonRpcResponse::success(request.id.clone(), v),
                    Err(e) => JsonRpcResponse::error(
                        request.id.clone(),
                        -32603,
                        format!("Serialization error: {e}"),
                    ),
                }
            }
            None => JsonRpcResponse::error(
                request.id.clone(),
                -32001,
                format!("Task not found: {}", params.id),
            ),
        }
    }
}

fn extract_text_from_parts(parts: &[Part]) -> String {
    parts
        .iter()
        .filter_map(|p| match p {
            Part::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn generate_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{nanos:x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;
    use crate::model::Model;
    use crate::model::types::{ChatRequest, ChatResponse, Message, StopReason, Usage};
    use crate::stream::ResponseStream;

    struct EchoModel;

    impl Model for EchoModel {
        async fn generate(&self, request: &ChatRequest) -> Result<ChatResponse> {
            let input = request
                .messages
                .last()
                .and_then(|m| m.content.as_deref())
                .unwrap_or("empty");
            Ok(ChatResponse {
                message: Message::assistant(format!("echo: {input}")),
                stop_reason: StopReason::EndTurn,
                usage: Some(Usage::default()),
            })
        }

        async fn generate_stream(&self, _request: &ChatRequest) -> Result<ResponseStream> {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    fn test_handler() -> A2aHandler {
        let agent = Arc::new(Agent::builder().model(EchoModel).build().unwrap());
        let card = AgentCard {
            name: "TestAgent".to_string(),
            description: "Test".to_string(),
            version: "0.1.0".to_string(),
            url: "http://localhost:8080".to_string(),
            capabilities: Vec::new(),
            authentication: None,
            protocol_version: "0.2".to_string(),
        };
        A2aHandler::new(agent, card)
    }

    #[tokio::test]
    async fn test_discover() {
        let handler = test_handler();
        let req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "agent/discover",
            "params": {}
        });
        let resp_str = handler.handle_request(&req.to_string()).await;
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        assert!(resp.error.is_none());
        let result = resp.result.unwrap();
        assert_eq!(result["name"], "TestAgent");
    }

    #[tokio::test]
    async fn test_task_send_and_get() {
        let handler = test_handler();
        let req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tasks/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "hello"}]
                }
            }
        });
        let resp_str = handler.handle_request(&req.to_string()).await;
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        assert!(resp.error.is_none());

        let task: A2aTask = serde_json::from_value(resp.result.unwrap()).unwrap();
        assert_eq!(task.status.state, TaskState::Completed);
        assert!(!task.artifacts.is_empty());

        let get_req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tasks/get",
            "params": { "id": task.id }
        });
        let get_resp_str = handler.handle_request(&get_req.to_string()).await;
        let get_resp: JsonRpcResponse = serde_json::from_str(&get_resp_str).unwrap();
        assert!(get_resp.error.is_none());
    }

    #[tokio::test]
    async fn test_task_cancel() {
        let handler = test_handler();
        let send_req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tasks/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "work"}]
                }
            }
        });
        let resp_str = handler.handle_request(&send_req.to_string()).await;
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        let task: A2aTask = serde_json::from_value(resp.result.unwrap()).unwrap();

        let cancel_req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tasks/cancel",
            "params": { "id": task.id }
        });
        let cancel_resp_str = handler.handle_request(&cancel_req.to_string()).await;
        let cancel_resp: JsonRpcResponse = serde_json::from_str(&cancel_resp_str).unwrap();
        assert!(cancel_resp.error.is_none());
        let cancelled: A2aTask = serde_json::from_value(cancel_resp.result.unwrap()).unwrap();
        assert_eq!(cancelled.status.state, TaskState::Canceled);
    }

    fn handler_with_cap(max_tasks: usize) -> A2aHandler {
        let agent = Arc::new(Agent::builder().model(EchoModel).build().unwrap());
        let card = AgentCard {
            name: "TestAgent".to_string(),
            description: "Test".to_string(),
            version: "0.1.0".to_string(),
            url: "http://localhost:8080".to_string(),
            capabilities: Vec::new(),
            authentication: None,
            protocol_version: "0.2".to_string(),
        };
        A2aHandler::with_max_tasks(agent, card, max_tasks)
    }

    async fn send_task(handler: &A2aHandler, text: &str) -> String {
        let req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tasks/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": text}]
                }
            }
        });
        let resp_str = handler.handle_request(&req.to_string()).await;
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        let task: A2aTask = serde_json::from_value(resp.result.unwrap()).unwrap();
        task.id
    }

    #[tokio::test]
    async fn test_task_store_evicts_over_cap() {
        let handler = handler_with_cap(2);

        // Each send runs the agent synchronously, so tasks are terminal
        // (Completed) by the time they land in the store.
        let id1 = send_task(&handler, "one").await;
        let id2 = send_task(&handler, "two").await;
        let id3 = send_task(&handler, "three").await;

        {
            let store = handler.tasks.lock().await;
            assert!(store.tasks.len() <= 2, "store should be capped at 2");
            // The oldest task must have been evicted; the two newest remain.
            assert!(store.get(&id1).is_none());
            assert!(store.get(&id2).is_some());
            assert!(store.get(&id3).is_some());
        }
    }

    #[tokio::test]
    async fn test_task_send_rejected_when_full_of_nonterminal() {
        let handler = handler_with_cap(2);

        // Pre-fill the store with in-flight (Working) tasks up to capacity.
        // These stand in for concurrent requests still awaiting their agent
        // response, which terminal-eviction cannot reclaim.
        {
            let mut store = handler.tasks.lock().await;
            for id in ["w1", "w2"] {
                store.insert(
                    id.to_string(),
                    A2aTask {
                        id: id.to_string(),
                        context_id: None,
                        status: TaskStatus {
                            state: TaskState::Working,
                            message: None,
                        },
                        artifacts: Vec::new(),
                        history: Vec::new(),
                        metadata: HashMap::new(),
                    },
                );
            }
        }

        // A new send must be rejected with -32000 rather than growing the map.
        let req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tasks/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "overflow"}]
                }
            }
        });
        let resp_str = handler.handle_request(&req.to_string()).await;
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        assert!(resp.error.is_some(), "over-capacity send must be rejected");
        assert_eq!(resp.error.unwrap().code, -32000);

        let store = handler.tasks.lock().await;
        assert_eq!(
            store.tasks.len(),
            2,
            "rejected send must not grow the store past its cap"
        );
    }

    #[test]
    fn test_task_store_keeps_nonterminal_over_cap() {
        // If everything retained is non-terminal, the store must not drop an
        // in-flight task even when over the cap.
        let mut store = TaskStore::new(1);
        let working = |id: &str| A2aTask {
            id: id.to_string(),
            context_id: None,
            status: TaskStatus {
                state: TaskState::Working,
                message: None,
            },
            artifacts: Vec::new(),
            history: Vec::new(),
            metadata: HashMap::new(),
        };
        store.insert("a".into(), working("a"));
        store.insert("b".into(), working("b"));
        assert_eq!(store.tasks.len(), 2, "non-terminal tasks are never evicted");
    }

    #[tokio::test]
    async fn test_method_not_found() {
        let handler = test_handler();
        let req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "nonexistent",
            "params": {}
        });
        let resp_str = handler.handle_request(&req.to_string()).await;
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32601);
    }

    #[tokio::test]
    async fn test_invalid_json() {
        let handler = test_handler();
        let resp_str = handler.handle_request("not json").await;
        let resp: JsonRpcResponse = serde_json::from_str(&resp_str).unwrap();
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32700);
    }
}
