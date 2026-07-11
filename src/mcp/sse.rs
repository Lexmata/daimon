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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use futures::StreamExt;
use tokio::sync::{Mutex, oneshot};

use crate::error::{DaimonError, Result};
use crate::mcp::protocol::{JsonRpcNotification, JsonRpcRequest, JsonRpcResponse};
use crate::mcp::transport::{DEFAULT_HTTP_TIMEOUT, McpTransport};

type PendingMap = Arc<Mutex<HashMap<u64, oneshot::Sender<JsonRpcResponse>>>>;

/// SSE transport for MCP communication.
///
/// Holds a persistent GET connection open via a background task; `request`
/// POSTs a JSON-RPC request and awaits the matching response arriving
/// asynchronously on that stream, correlated by JSON-RPC `id`.
pub struct SseTransport {
    post_url: Arc<Mutex<String>>,
    pending: PendingMap,
    client: reqwest::Client,
    headers: HashMap<String, String>,
    reader_task: tokio::task::JoinHandle<()>,
    next_id: AtomicU64,
    /// Whole-request timeout applied to every JSON-RPC POST. The persistent
    /// GET stream deliberately has no such timeout — it is long-lived by
    /// design and only bounded by a connect timeout.
    timeout: Duration,
    // ponytail: `close()` (and the reader task, on stream death) clears
    // `pending` once; if a concurrent `request()` is between creating its
    // oneshot and inserting it into `pending`, that insert could land
    // *after* the clear, leaving an entry no one will ever remove.
    // `request()` checks this flag right after inserting and bails out if
    // set, covering both orderings: either the flag is already true (caught
    // here) or the subsequent clear() removes the entry.
    is_closed: Arc<AtomicBool>,
    /// Why the transport closed (stream death, endpoint-origin violation,
    /// explicit `close()`), surfaced in the error every subsequent
    /// `request()` returns. Set exactly once by whichever closer ran first.
    close_reason: Arc<OnceLock<String>>,
}

impl SseTransport {
    /// Connects to an MCP server's SSE endpoint at `url`, opening the
    /// persistent event stream and spawning a background task that reads
    /// it. `headers` are attached to both the initial GET and every
    /// subsequent POST (e.g. for authentication).
    ///
    /// Uses a 30-second TCP connect timeout and a 30-second per-POST request
    /// timeout; use [`connect_with_timeout`](Self::connect_with_timeout) to
    /// change them.
    pub async fn connect(url: impl Into<String>, headers: HashMap<String, String>) -> Result<Self> {
        Self::connect_with_timeout(url, headers, DEFAULT_HTTP_TIMEOUT).await
    }

    /// Like [`connect`](Self::connect), with an explicit `timeout` used both
    /// as the TCP connect timeout for the persistent event stream and as the
    /// whole-request timeout for each JSON-RPC POST. The event stream read
    /// itself is never timed out — it stays open for the life of the
    /// transport.
    pub async fn connect_with_timeout(
        url: impl Into<String>,
        headers: HashMap<String, String>,
        timeout: Duration,
    ) -> Result<Self> {
        let url = url.into();
        let client = reqwest::Client::builder()
            .connect_timeout(timeout)
            .build()
            .map_err(|e| DaimonError::Mcp(format!("SSE client build failed: {e}")))?;

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
        let is_closed = Arc::new(AtomicBool::new(false));
        let close_reason: Arc<OnceLock<String>> = Arc::new(OnceLock::new());

        let reader_task = {
            let post_url = post_url.clone();
            let pending = pending.clone();
            let base_url = url.clone();
            let is_closed = is_closed.clone();
            let close_reason = close_reason.clone();
            tokio::spawn(async move {
                let mut stream = resp.bytes_stream();
                let mut buf = String::new();
                'read: while let Some(chunk) = stream.next().await {
                    let Ok(bytes) = chunk else { break };
                    buf.push_str(&String::from_utf8_lossy(&bytes).replace("\r\n", "\n"));
                    while let Some(idx) = buf.find("\n\n") {
                        let frame_raw = buf[..idx].to_string();
                        buf.drain(..idx + 2);
                        if let Some(frame) = parse_sse_frame(&frame_raw)
                            && let Err(fatal) =
                                handle_frame(frame, &post_url, &base_url, &pending).await
                        {
                            // A fatal frame (e.g. an endpoint event pointing
                            // at a different origin) poisons the session:
                            // stop reading entirely rather than keep a
                            // connection to a misbehaving server alive.
                            tracing::error!("SSE stream fatal: {fatal}");
                            let _ = close_reason.set(fatal);
                            break 'read;
                        }
                    }
                }
                // Stream ended (server closed it, a network error broke the
                // read loop, or a fatal frame above): mark the transport
                // closed so subsequent `request()` calls fail fast instead
                // of POSTing and awaiting a response that can never arrive,
                // then drop every still-pending sender so in-flight
                // `request()` calls' `rx.await` fail with a clear error
                // instead of hanging forever.
                let _ = close_reason.set("SSE stream ended".to_string());
                is_closed.store(true, Ordering::SeqCst);
                pending.lock().await.clear();
            })
        };

        Ok(Self {
            post_url,
            pending,
            client,
            headers,
            reader_task,
            next_id: AtomicU64::new(1),
            timeout,
            is_closed,
            close_reason,
        })
    }

    /// The error returned once the transport is closed, naming the reason
    /// the connection went away.
    fn closed_error(&self) -> DaimonError {
        let reason = self
            .close_reason
            .get()
            .map(String::as_str)
            .unwrap_or("transport closed");
        DaimonError::Mcp(format!("SSE transport closed: {reason}"))
    }
}

impl Drop for SseTransport {
    fn drop(&mut self) {
        // Without this, dropping the transport without calling `close()`
        // leaks the background reader task (and its half of the HTTP
        // connection) for the life of the runtime.
        self.reader_task.abort();
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
/// spec) and enforces same-origin: the resolved URL must share the connect
/// URL's scheme, host, and port.
///
/// The auth headers supplied at connect time are attached to every POST, so
/// accepting a cross-origin endpoint would let a malicious (or compromised)
/// server redirect JSON-RPC POSTs — bearer token included — to an attacker
/// origin. Likewise, data that fails to parse or join is an error, never
/// used verbatim as a POST target.
fn resolve_endpoint_url(base_url: &str, endpoint_data: &str) -> Result<String> {
    let base = reqwest::Url::parse(base_url)
        .map_err(|e| DaimonError::Mcp(format!("invalid SSE base URL '{base_url}': {e}")))?;
    let resolved = base
        .join(endpoint_data)
        .map_err(|e| DaimonError::Mcp(format!("invalid SSE endpoint '{endpoint_data}': {e}")))?;

    if resolved.scheme() != base.scheme()
        || resolved.host_str() != base.host_str()
        || resolved.port_or_known_default() != base.port_or_known_default()
    {
        return Err(DaimonError::Mcp(format!(
            "SSE endpoint '{resolved}' is not same-origin with '{base}'"
        )));
    }

    Ok(resolved.to_string())
}

/// Processes one SSE frame. Returns `Err(reason)` for a fatal frame that
/// must tear down the transport (an endpoint event failing same-origin
/// validation); all other malformed frames are skipped — one bad message
/// frame from a buggy server shouldn't kill every other in-flight request.
async fn handle_frame(
    frame: SseFrame,
    post_url: &Arc<Mutex<String>>,
    base_url: &str,
    pending: &PendingMap,
) -> std::result::Result<(), String> {
    if frame.event == "endpoint" {
        match resolve_endpoint_url(base_url, frame.data.trim()) {
            Ok(resolved) => {
                *post_url.lock().await = resolved;
                return Ok(());
            }
            Err(e) => return Err(e.to_string()),
        }
    }

    // Any other event (typically "message") is expected to carry a
    // JSON-RPC response body.
    let Ok(response) = serde_json::from_str::<JsonRpcResponse>(&frame.data) else {
        return Ok(());
    };
    // A JSON-RPC message with no `id` is a notification, not a response to
    // any pending `request()` — there's no sender to resolve.
    let Some(id) = response.id else {
        return Ok(());
    };
    if let Some(tx) = pending.lock().await.remove(&id) {
        let _ = tx.send(response);
    }
    Ok(())
}

impl McpTransport for SseTransport {
    fn request<'a>(
        &'a self,
        method: &'a str,
        params: Option<serde_json::Value>,
    ) -> Pin<Box<dyn Future<Output = Result<JsonRpcResponse>> + Send + 'a>> {
        Box::pin(async move {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            let request = JsonRpcRequest::new(id, method, params);

            let (tx, rx) = oneshot::channel();
            self.pending.lock().await.insert(id, tx);

            if self.is_closed.load(Ordering::SeqCst) {
                self.pending.lock().await.remove(&id);
                return Err(self.closed_error());
            }

            let post_url = self.post_url.lock().await.clone();
            let mut req = self
                .client
                .post(&post_url)
                .timeout(self.timeout)
                .json(&request);
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
                    self.pending.lock().await.remove(&id);
                    return Err(e);
                }
            };

            if !resp.status().is_success() {
                self.pending.lock().await.remove(&id);
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                return Err(DaimonError::Mcp(format!(
                    "SSE POST failed: HTTP {status}: {text}"
                )));
            }

            rx.await.map_err(|_| self.closed_error())
        })
    }

    fn notify<'a>(
        &'a self,
        notification: &'a JsonRpcNotification,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            if self.is_closed.load(Ordering::SeqCst) {
                return Err(self.closed_error());
            }

            let post_url = self.post_url.lock().await.clone();
            let mut req = self
                .client
                .post(&post_url)
                .timeout(self.timeout)
                .json(notification);
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
            let _ = self.close_reason.set("transport closed".to_string());
            self.is_closed.store(true, Ordering::SeqCst);
            self.pending.lock().await.clear();
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::SocketAddr;
    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
    use tokio::net::{TcpListener, TcpStream};
    use tokio::sync::mpsc;

    /// Writes one HTTP/1.1 chunked-transfer-encoding chunk.
    async fn write_chunk(stream: &mut TcpStream, data: &[u8]) -> std::io::Result<()> {
        stream
            .write_all(format!("{:x}\r\n", data.len()).as_bytes())
            .await?;
        stream.write_all(data).await?;
        stream.write_all(b"\r\n").await
    }

    /// Writes the terminating zero-length chunk, ending the response body.
    async fn write_chunk_end(stream: &mut TcpStream) -> std::io::Result<()> {
        stream.write_all(b"0\r\n\r\n").await
    }

    /// Writes one SSE frame (optionally with an explicit `event:` line) as a
    /// single chunked-transfer-encoding chunk.
    async fn write_sse_frame(
        stream: &mut TcpStream,
        event: Option<&str>,
        data: &str,
    ) -> std::io::Result<()> {
        let mut frame = String::new();
        if let Some(event) = event {
            frame.push_str(&format!("event: {event}\n"));
        }
        frame.push_str(&format!("data: {data}\n\n"));
        write_chunk(stream, frame.as_bytes()).await
    }

    /// Writes the HTTP/1.1 response line + headers that open a chunked
    /// `text/event-stream` response. Caller writes frames after this via
    /// `write_sse_frame`, and must eventually call `write_chunk_end`.
    async fn write_sse_response_head(stream: &mut TcpStream) -> std::io::Result<()> {
        stream
            .write_all(
                b"HTTP/1.1 200 OK\r\n\
                  Content-Type: text/event-stream\r\n\
                  Transfer-Encoding: chunked\r\n\
                  Connection: keep-alive\r\n\r\n",
            )
            .await
    }

    /// Reads one HTTP/1.1 request's method, path, and (if `Content-Length`
    /// is present) body off `stream`. Blocks until the request line and
    /// headers have arrived.
    async fn read_http_request(stream: &mut TcpStream) -> (String, String, String) {
        let mut reader = BufReader::new(&mut *stream);
        let mut request_line = String::new();
        reader.read_line(&mut request_line).await.unwrap();
        let mut parts = request_line.split_whitespace();
        let method = parts.next().unwrap_or("").to_string();
        let path = parts.next().unwrap_or("").to_string();

        let mut content_length = 0usize;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).await.unwrap();
            let trimmed = line.trim();
            if trimmed.is_empty() {
                break;
            }
            if let Some((key, value)) = trimmed.split_once(':')
                && key.eq_ignore_ascii_case("content-length")
            {
                content_length = value.trim().parse().unwrap_or(0);
            }
        }
        let mut body = vec![0u8; content_length];
        if content_length > 0 {
            reader.read_exact(&mut body).await.unwrap();
        }
        (method, path, String::from_utf8_lossy(&body).into_owned())
    }

    async fn write_202_accepted(stream: &mut TcpStream) -> std::io::Result<()> {
        stream
            .write_all(b"HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\n\r\n")
            .await
    }

    /// A running test SSE server: accepts exactly one long-lived GET
    /// connection (the SSE stream, lazily obtained the first time a
    /// `push_*`/`close_sse_stream` call needs it) and any number of POST
    /// connections (each just parsed for its JSON-RPC `id` and acknowledged
    /// with a 202 — the id and raw body are forwarded over `posted` so the
    /// test can react to them). The accept loop is fully spawned in the background so
    /// `start()` can return immediately, before any client has connected —
    /// otherwise `start()`'s own `.accept().await` would block forever,
    /// since nothing connects until the caller's subsequent
    /// `SseTransport::connect(...)` call runs, which can't happen until
    /// `start()` returns. (The very first version of this harness got this
    /// wrong and deadlocked every test that used it.)
    struct TestSseServer {
        addr: SocketAddr,
        sse_stream: Option<TcpStream>,
        sse_stream_rx: Option<tokio::sync::oneshot::Receiver<TcpStream>>,
        posted: mpsc::UnboundedReceiver<(u64, String)>,
    }

    impl TestSseServer {
        /// Starts the server in the background and returns immediately —
        /// does NOT wait for a client to connect. If `endpoint_frame` is
        /// `Some(data)`, an `event: endpoint` frame with that data is sent
        /// right after the SSE response head, as soon as the first (SSE)
        /// connection arrives.
        async fn start(endpoint_frame: Option<&str>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            let (posted_tx, posted_rx) = mpsc::unbounded_channel();
            let (sse_tx, sse_rx) = tokio::sync::oneshot::channel();
            let endpoint_frame = endpoint_frame.map(|s| s.to_string());

            tokio::spawn(async move {
                let mut sse_tx = Some(sse_tx);
                loop {
                    let Ok((mut conn, _)) = listener.accept().await else {
                        break;
                    };
                    if let Some(tx) = sse_tx.take() {
                        // First connection: this is the SSE GET.
                        let (_method, _path, _body) = read_http_request(&mut conn).await;
                        write_sse_response_head(&mut conn).await.unwrap();
                        if let Some(data) = &endpoint_frame {
                            write_sse_frame(&mut conn, Some("endpoint"), data)
                                .await
                                .unwrap();
                        }
                        let _ = tx.send(conn);
                        continue;
                    }
                    // Every later connection is a POST.
                    let (method, _path, body) = read_http_request(&mut conn).await;
                    if method == "POST" {
                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body)
                            && let Some(id) = json.get("id").and_then(|v| v.as_u64())
                        {
                            let _ = posted_tx.send((id, body));
                        }
                        let _ = write_202_accepted(&mut conn).await;
                    }
                }
            });

            Self {
                addr,
                sse_stream: None,
                sse_stream_rx: Some(sse_rx),
                posted: posted_rx,
            }
        }

        fn connect_url(&self) -> String {
            format!("http://{}/sse", self.addr)
        }

        /// Returns the SSE `TcpStream`, awaiting the client's connection the
        /// first time it's needed (by which point the test has already
        /// called `SseTransport::connect(...)`, so this resolves promptly).
        async fn sse_stream(&mut self) -> &mut TcpStream {
            if self.sse_stream.is_none() {
                let rx = self
                    .sse_stream_rx
                    .take()
                    .expect("sse_stream already consumed");
                let stream = tokio::time::timeout(std::time::Duration::from_secs(5), rx)
                    .await
                    .expect("timed out waiting for the SSE client to connect")
                    .expect("sse_stream sender dropped");
                self.sse_stream = Some(stream);
            }
            self.sse_stream.as_mut().unwrap()
        }

        /// Sends a JSON-RPC response as a `message` SSE frame over the held
        /// GET stream.
        async fn push_response(&mut self, response_json: &str) {
            let stream = self.sse_stream().await;
            write_sse_frame(stream, None, response_json).await.unwrap();
        }

        /// Sends a raw (possibly malformed) `data:` payload as a `message`
        /// frame.
        async fn push_raw(&mut self, raw_data: &str) {
            let stream = self.sse_stream().await;
            write_sse_frame(stream, None, raw_data).await.unwrap();
        }

        /// Waits for the next POSTed request's JSON-RPC id and raw body.
        async fn next_posted(&mut self) -> (u64, String) {
            tokio::time::timeout(std::time::Duration::from_secs(5), self.posted.recv())
                .await
                .expect("timed out waiting for a POST")
                .expect("posted channel closed")
        }

        /// Waits for the next POSTed request's JSON-RPC id.
        async fn next_posted_id(&mut self) -> u64 {
            self.next_posted().await.0
        }

        /// Closes the SSE stream (simulates the server hanging up).
        async fn close_sse_stream(mut self) {
            let stream = self.sse_stream().await;
            let _ = write_chunk_end(stream).await;
            let _ = stream.shutdown().await;
        }
    }

    #[tokio::test]
    async fn test_endpoint_discovery_updates_post_target() {
        let server = TestSseServer::start(Some("/messages?sessionId=abc")).await;
        let connect_url = server.connect_url();

        let transport = SseTransport::connect(connect_url.clone(), HashMap::new())
            .await
            .unwrap();

        // Give the reader task a moment to process the endpoint frame sent
        // during TestSseServer::start before we inspect post_url.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let post_url = transport.post_url.lock().await.clone();
        assert_eq!(
            post_url,
            format!("http://{}/messages?sessionId=abc", server.addr)
        );

        let _ = transport.close().await;
        server.close_sse_stream().await;
    }

    #[tokio::test]
    async fn test_falls_back_to_connect_url_without_endpoint_frame() {
        let server = TestSseServer::start(None).await;
        let connect_url = server.connect_url();

        let transport = SseTransport::connect(connect_url.clone(), HashMap::new())
            .await
            .unwrap();

        let post_url = transport.post_url.lock().await.clone();
        assert_eq!(post_url, connect_url);

        let _ = transport.close().await;
        server.close_sse_stream().await;
    }

    #[tokio::test]
    async fn test_correlates_responses_including_out_of_order() {
        let mut server = TestSseServer::start(None).await;
        let connect_url = server.connect_url();
        let transport = Arc::new(
            SseTransport::connect(connect_url, HashMap::new())
                .await
                .unwrap(),
        );

        // Fire two concurrent requests, A then B. The transport allocates
        // the ids; the methods distinguish which request is which on the
        // wire.
        let ta = transport.clone();
        let tb = transport.clone();
        let handle_a = tokio::spawn(async move { ta.request("a/method", None).await });
        let handle_b = tokio::spawn(async move { tb.request("b/method", None).await });

        // Wait for both POSTs to land server-side, then reply out of order:
        // B's response before A's.
        let first = server.next_posted().await;
        let second = server.next_posted().await;
        let (id_a, id_b) = if first.1.contains("a/method") {
            (first.0, second.0)
        } else {
            (second.0, first.0)
        };
        assert_ne!(id_a, id_b, "transport must allocate unique request ids");

        server
            .push_response(&format!(
                r#"{{"jsonrpc":"2.0","id":{id_b},"result":{{"ok":"b"}}}}"#
            ))
            .await;
        server
            .push_response(&format!(
                r#"{{"jsonrpc":"2.0","id":{id_a},"result":{{"ok":"a"}}}}"#
            ))
            .await;

        let resp_a = handle_a.await.unwrap().unwrap();
        let resp_b = handle_b.await.unwrap().unwrap();
        assert_eq!(resp_a.result.unwrap()["ok"], "a");
        assert_eq!(resp_b.result.unwrap()["ok"], "b");

        let _ = transport.close().await;
        server.close_sse_stream().await;
    }

    #[tokio::test]
    async fn test_malformed_frame_is_skipped_not_fatal() {
        let mut server = TestSseServer::start(None).await;
        let connect_url = server.connect_url();
        let transport = Arc::new(
            SseTransport::connect(connect_url, HashMap::new())
                .await
                .unwrap(),
        );

        let t = transport.clone();
        let handle = tokio::spawn(async move { t.request("tools/list", None).await });

        let posted_id = server.next_posted_id().await;
        assert_eq!(posted_id, 1);

        // A garbage frame first — must not crash the reader task or corrupt
        // subsequent frame parsing.
        server.push_raw("not valid json at all").await;
        server
            .push_response(r#"{"jsonrpc":"2.0","id":1,"result":{"ok":true}}"#)
            .await;

        let resp = handle.await.unwrap().unwrap();
        assert_eq!(resp.result.unwrap()["ok"], true);

        let _ = transport.close().await;
        server.close_sse_stream().await;
    }

    #[tokio::test]
    async fn test_connect_failure_surfaces_as_mcp_error() {
        // Port 1: nothing listens here (same convention local-code's
        // mcp/connect.rs tests use for HTTP).
        let result = SseTransport::connect("http://127.0.0.1:1/sse", HashMap::new()).await;
        assert!(matches!(result, Err(DaimonError::Mcp(_))));
    }

    #[tokio::test]
    async fn test_close_fails_pending_sends() {
        let mut server = TestSseServer::start(None).await;
        let connect_url = server.connect_url();
        let transport = Arc::new(
            SseTransport::connect(connect_url, HashMap::new())
                .await
                .unwrap(),
        );

        let t = transport.clone();
        let handle = tokio::spawn(async move { t.request("tools/list", None).await });

        // Make sure the POST actually landed (the pending sender is
        // registered) before closing, so this exercises "closed while a
        // request is in flight" rather than "closed before it started".
        let _ = server.next_posted_id().await;

        transport.close().await.unwrap();

        let result = handle.await.unwrap();
        assert!(matches!(result, Err(DaimonError::Mcp(_))));

        server.close_sse_stream().await;
    }

    /// Polls until the transport observes closure, bounded by a timeout so a
    /// regression hangs the assertion, not the whole test run.
    async fn wait_until_closed(transport: &SseTransport) {
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while !transport.is_closed.load(Ordering::SeqCst) {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("transport never marked itself closed");
    }

    #[tokio::test]
    async fn test_dead_stream_fails_subsequent_requests_fast() {
        let server = TestSseServer::start(None).await;
        let connect_url = server.connect_url();
        let transport = SseTransport::connect(connect_url, HashMap::new())
            .await
            .unwrap();

        // Kill the SSE stream: the reader task must mark the transport
        // closed, not just exit.
        server.close_sse_stream().await;
        wait_until_closed(&transport).await;

        // A send after stream death must fail fast, not hang forever
        // awaiting a response that can never arrive.
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            transport.request("tools/list", None),
        )
        .await
        .expect("request hung after the SSE stream died");
        assert!(matches!(result, Err(DaimonError::Mcp(_))));
    }

    #[tokio::test]
    async fn test_cross_origin_endpoint_event_closes_transport() {
        // A malicious server redirecting POSTs (which carry the connect-time
        // auth headers) to another origin must poison the session, and the
        // attacker URL must never become the POST target.
        let server = TestSseServer::start(Some("http://attacker.example.com/messages")).await;
        let connect_url = server.connect_url();
        let transport = SseTransport::connect(connect_url.clone(), HashMap::new())
            .await
            .unwrap();

        wait_until_closed(&transport).await;

        assert_eq!(*transport.post_url.lock().await, connect_url);
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            transport.request("tools/list", None),
        )
        .await
        .expect("request hung after a cross-origin endpoint event");
        assert!(matches!(result, Err(DaimonError::Mcp(_))));

        server.close_sse_stream().await;
    }

    #[tokio::test]
    async fn test_unparseable_endpoint_event_closes_transport() {
        // Endpoint data that fails to parse/join must be an error, never
        // used verbatim as the POST target.
        let server = TestSseServer::start(Some("http://[not a url")).await;
        let connect_url = server.connect_url();
        let transport = SseTransport::connect(connect_url.clone(), HashMap::new())
            .await
            .unwrap();

        wait_until_closed(&transport).await;
        assert_eq!(*transport.post_url.lock().await, connect_url);

        server.close_sse_stream().await;
    }

    #[test]
    fn test_resolve_endpoint_url_same_origin() {
        // Relative paths and absolute same-origin URLs are both fine.
        assert_eq!(
            resolve_endpoint_url("http://localhost:3000/sse", "/messages?sessionId=x").unwrap(),
            "http://localhost:3000/messages?sessionId=x"
        );
        assert_eq!(
            resolve_endpoint_url(
                "http://localhost:3000/sse",
                "http://localhost:3000/messages"
            )
            .unwrap(),
            "http://localhost:3000/messages"
        );
        // Explicit default port counts as the same origin.
        assert_eq!(
            resolve_endpoint_url("http://localhost/sse", "http://localhost:80/messages").unwrap(),
            "http://localhost/messages"
        );
    }

    #[test]
    fn test_resolve_endpoint_url_rejects_cross_origin() {
        // Different host.
        assert!(
            resolve_endpoint_url("http://localhost:3000/sse", "http://evil.example.com/m").is_err()
        );
        // Different port.
        assert!(
            resolve_endpoint_url("http://localhost:3000/sse", "http://localhost:4000/m").is_err()
        );
        // Different scheme (https -> http downgrade).
        assert!(
            resolve_endpoint_url("https://localhost:3000/sse", "http://localhost:3000/m").is_err()
        );
        // Unparseable data is an error, never used verbatim.
        assert!(resolve_endpoint_url("http://localhost:3000/sse", "http://[bad").is_err());
    }

    #[tokio::test]
    async fn test_drop_aborts_reader_task() {
        let server = TestSseServer::start(None).await;
        let connect_url = server.connect_url();
        let transport = SseTransport::connect(connect_url, HashMap::new())
            .await
            .unwrap();

        // The reader task holds a clone of `pending`; once the transport is
        // dropped (without close()) and Drop aborts the task, that clone is
        // released and the weak reference can no longer upgrade.
        let pending_weak = Arc::downgrade(&transport.pending);
        drop(transport);

        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while pending_weak.upgrade().is_some() {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("reader task leaked after drop (pending map still referenced)");

        server.close_sse_stream().await;
    }

    #[tokio::test]
    async fn test_two_bridges_sharing_one_transport_get_distinct_responses() {
        use crate::mcp::bridge::McpToolBridge;
        use crate::mcp::protocol::McpToolInfo;
        use crate::tool::Tool;

        let mut server = TestSseServer::start(None).await;
        let connect_url = server.connect_url();
        let transport: Arc<dyn McpTransport> = Arc::new(
            SseTransport::connect(connect_url, HashMap::new())
                .await
                .unwrap(),
        );

        // Two bridges over the SAME transport — the historical bug gave each
        // bridge its own id counter starting at 1000, so their concurrent
        // requests collided in the pending map and one hung or misrouted.
        let bridge_a = McpToolBridge::new(
            transport.clone(),
            McpToolInfo {
                name: "tool_a".into(),
                description: None,
                input_schema: serde_json::json!({"type": "object"}),
            },
        );
        let bridge_b = McpToolBridge::new(
            transport.clone(),
            McpToolInfo {
                name: "tool_b".into(),
                description: None,
                input_schema: serde_json::json!({"type": "object"}),
            },
        );

        // Both calls run concurrently over the shared transport.
        let handle_a = tokio::spawn(async move { bridge_a.execute(&serde_json::json!({})).await });
        let handle_b = tokio::spawn(async move { bridge_b.execute(&serde_json::json!({})).await });

        // Answer each POST as it arrives, keyed by the tool name in its
        // body, echoing the transport-allocated id back.
        let mut seen_ids = std::collections::HashSet::new();
        for _ in 0..2 {
            let (id, body) = server.next_posted().await;
            assert!(seen_ids.insert(id), "request id {id} was reused");
            let label = if body.contains("tool_a") { "a" } else { "b" };
            server
                .push_response(&format!(
                    r#"{{"jsonrpc":"2.0","id":{id},"result":{{"content":[{{"type":"text","text":"from-{label}"}}],"isError":false}}}}"#
                ))
                .await;
        }

        let output_a = tokio::time::timeout(std::time::Duration::from_secs(5), handle_a)
            .await
            .expect("bridge A call hung (id collision?)")
            .unwrap()
            .unwrap();
        let output_b = tokio::time::timeout(std::time::Duration::from_secs(5), handle_b)
            .await
            .expect("bridge B call hung (id collision?)")
            .unwrap()
            .unwrap();

        assert_eq!(output_a.content, "from-a");
        assert_eq!(output_b.content, "from-b");

        let _ = transport.close().await;
        server.close_sse_stream().await;
    }
}
