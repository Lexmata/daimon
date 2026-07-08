//! SSE transport for MCP: the pre-Streamable-HTTP "HTTP+SSE" transport. A
//! persistent GET request receives a `text/event-stream` response; JSON-RPC
//! requests are sent via separate HTTP POSTs; all responses (and any
//! server-initiated notifications) arrive asynchronously as SSE frames on
//! the original GET stream.
//!
//! ```ignore
//! use daimon::mcp::{McpClient, SseTransport};
//!
//! let transport = SseTransport::connect("http://localhost:3000/sse", Default::default()).await?;
//! let client = McpClient::connect(transport).await?;
//! ```

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use futures::StreamExt;
use tokio::sync::{Mutex, oneshot};

use crate::error::{DaimonError, Result};
use crate::mcp::protocol::{JsonRpcNotification, JsonRpcRequest, JsonRpcResponse};
use crate::mcp::transport::McpTransport;

type PendingMap = Arc<Mutex<HashMap<u64, oneshot::Sender<JsonRpcResponse>>>>;

/// SSE transport for MCP communication.
///
/// Holds a persistent GET connection open via a background task; `send`
/// POSTs a request and awaits the matching response arriving asynchronously
/// on that stream, correlated by JSON-RPC `id`.
pub struct SseTransport {
    post_url: Arc<Mutex<String>>,
    pending: PendingMap,
    client: reqwest::Client,
    headers: HashMap<String, String>,
    reader_task: tokio::task::JoinHandle<()>,
    // ponytail: `close()` clears `pending` once; if a concurrent `send()` is
    // between creating its oneshot and inserting it into `pending`, that
    // insert could land *after* `close()`'s clear, leaving an entry no one
    // will ever remove (the reader task is aborted, `close()` already ran).
    // `send()` checks this flag right after inserting and bails out if set,
    // covering both orderings: either the flag is already true (caught
    // here) or `close()`'s clear() runs after and removes the entry.
    is_closed: Arc<AtomicBool>,
}

impl SseTransport {
    /// Connects to an MCP server's SSE endpoint at `url`, opening the
    /// persistent event stream and spawning a background task that reads
    /// it. `headers` are attached to both the initial GET and every
    /// subsequent POST (e.g. for authentication).
    pub async fn connect(url: impl Into<String>, headers: HashMap<String, String>) -> Result<Self> {
        let url = url.into();
        let client = reqwest::Client::new();

        let mut req = client.get(&url).header("Accept", "text/event-stream");
        for (key, value) in &headers {
            req = req.header(key.as_str(), value.as_str());
        }
        let resp = req
            .send()
            .await
            .map_err(|e| DaimonError::Mcp(format!("SSE connect failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            return Err(DaimonError::Mcp(format!(
                "SSE connect failed: HTTP {status}"
            )));
        }

        let post_url = Arc::new(Mutex::new(url.clone()));
        let pending: PendingMap = Arc::new(Mutex::new(HashMap::new()));

        let reader_task = {
            let post_url = post_url.clone();
            let pending = pending.clone();
            let base_url = url.clone();
            tokio::spawn(async move {
                let mut stream = resp.bytes_stream();
                let mut buf = String::new();
                while let Some(chunk) = stream.next().await {
                    let Ok(bytes) = chunk else { break };
                    buf.push_str(&String::from_utf8_lossy(&bytes).replace("\r\n", "\n"));
                    while let Some(idx) = buf.find("\n\n") {
                        let frame_raw = buf[..idx].to_string();
                        buf.drain(..idx + 2);
                        if let Some(frame) = parse_sse_frame(&frame_raw) {
                            handle_frame(frame, &post_url, &base_url, &pending).await;
                        }
                    }
                }
                // Stream ended (server closed it, or a network error broke
                // the read loop): drop every still-pending sender. Each
                // awaiting `send()` call's `rx.await` then fails with
                // `RecvError`, which `send()` maps to a clear "transport
                // closed" error instead of hanging forever.
                pending.lock().await.clear();
            })
        };

        Ok(Self {
            post_url,
            pending,
            client,
            headers,
            reader_task,
            is_closed: Arc::new(AtomicBool::new(false)),
        })
    }
}

/// One parsed SSE frame: its event name (defaulting to `"message"` if the
/// frame had no `event:` line) and its data payload (multiple `data:` lines
/// within one frame are joined with `\n`, per the SSE spec). Returns `None`
/// for a frame with no `data:` lines at all (e.g. a comment-only frame) —
/// there's nothing to act on.
struct SseFrame {
    event: String,
    data: String,
}

fn parse_sse_frame(raw: &str) -> Option<SseFrame> {
    let mut event = String::from("message");
    let mut data_lines = Vec::new();
    for line in raw.split('\n') {
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("event:") {
            event = rest.trim().to_string();
        } else if let Some(rest) = line.strip_prefix("data:") {
            data_lines.push(rest.trim_start().to_string());
        }
        // Any other field (id:, retry:) is ignored — not needed for MCP.
    }
    if data_lines.is_empty() {
        return None;
    }
    Some(SseFrame {
        event,
        data: data_lines.join("\n"),
    })
}

/// Resolves an `event: endpoint` frame's data against the original connect
/// URL (it may be a full URL or a path relative to it, per the MCP HTTP+SSE
/// spec). Falls back to the literal data on any parse failure.
fn resolve_endpoint_url(base_url: &str, endpoint_data: &str) -> String {
    match reqwest::Url::parse(base_url).and_then(|base| base.join(endpoint_data)) {
        Ok(resolved) => resolved.to_string(),
        Err(_) => endpoint_data.to_string(),
    }
}

async fn handle_frame(
    frame: SseFrame,
    post_url: &Arc<Mutex<String>>,
    base_url: &str,
    pending: &PendingMap,
) {
    if frame.event == "endpoint" {
        let resolved = resolve_endpoint_url(base_url, frame.data.trim());
        *post_url.lock().await = resolved;
        return;
    }

    // Any other event (typically "message") is expected to carry a
    // JSON-RPC response body. A malformed payload is skipped rather than
    // tearing down the whole stream — one bad frame from a buggy server
    // shouldn't kill every other in-flight request.
    let Ok(response) = serde_json::from_str::<JsonRpcResponse>(&frame.data) else {
        return;
    };
    // A JSON-RPC message with no `id` is a notification, not a response to
    // any pending `send()` — there's no sender to resolve.
    let Some(id) = response.id else {
        return;
    };
    if let Some(tx) = pending.lock().await.remove(&id) {
        let _ = tx.send(response);
    }
}

impl McpTransport for SseTransport {
    fn send<'a>(
        &'a self,
        request: &'a JsonRpcRequest,
    ) -> Pin<Box<dyn Future<Output = Result<JsonRpcResponse>> + Send + 'a>> {
        Box::pin(async move {
            let (tx, rx) = oneshot::channel();
            self.pending.lock().await.insert(request.id, tx);

            if self.is_closed.load(Ordering::SeqCst) {
                self.pending.lock().await.remove(&request.id);
                return Err(DaimonError::Mcp(
                    "SSE transport closed before a response arrived".into(),
                ));
            }

            let post_url = self.post_url.lock().await.clone();
            let mut req = self.client.post(&post_url).json(request);
            for (key, value) in &self.headers {
                req = req.header(key.as_str(), value.as_str());
            }

            let send_result = req
                .send()
                .await
                .map_err(|e| DaimonError::Mcp(format!("SSE POST failed: {e}")));

            let resp = match send_result {
                Ok(resp) => resp,
                Err(e) => {
                    self.pending.lock().await.remove(&request.id);
                    return Err(e);
                }
            };

            if !resp.status().is_success() {
                self.pending.lock().await.remove(&request.id);
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                return Err(DaimonError::Mcp(format!(
                    "SSE POST failed: HTTP {status}: {text}"
                )));
            }

            rx.await.map_err(|_| {
                DaimonError::Mcp("SSE transport closed before a response arrived".into())
            })
        })
    }

    fn notify<'a>(
        &'a self,
        notification: &'a JsonRpcNotification,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let post_url = self.post_url.lock().await.clone();
            let mut req = self.client.post(&post_url).json(notification);
            for (key, value) in &self.headers {
                req = req.header(key.as_str(), value.as_str());
            }

            let resp = req
                .send()
                .await
                .map_err(|e| DaimonError::Mcp(format!("SSE POST failed: {e}")))?;

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                return Err(DaimonError::Mcp(format!(
                    "SSE POST failed: HTTP {status}: {text}"
                )));
            }

            Ok(())
        })
    }

    fn close<'a>(&'a self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            self.reader_task.abort();
            self.is_closed.store(true, Ordering::SeqCst);
            self.pending.lock().await.clear();
            Ok(())
        })
    }
}
