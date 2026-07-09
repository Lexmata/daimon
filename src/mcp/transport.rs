//! MCP transport implementations (stdio and HTTP).

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{DaimonError, Result};
use crate::mcp::protocol::{JsonRpcNotification, JsonRpcRequest, JsonRpcResponse};

/// Default timeout applied to each HTTP request the HTTP-based transports
/// make. Long enough for a slow tool call, short enough that a dead server
/// doesn't hang the agent loop indefinitely.
pub(crate) const DEFAULT_HTTP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Maximum number of non-matching JSON-RPC messages (server notifications,
/// server-initiated requests) a transport will skip while waiting for the
/// response to a specific request id. Without a bound, a peer that streams
/// notifications forever would pin `request()` in an infinite read loop.
pub(crate) const MAX_SKIPPED_MESSAGES: usize = 1024;

/// Maximum size (in bytes) of a single framed message body we will accept from
/// a peer. The Content-Length header is peer-supplied; without a cap a
/// malicious or buggy server could advertise a huge length and force us to
/// allocate that much memory up front (a trivial OOM DoS). 32 MiB is far larger
/// than any legitimate MCP message.
const MAX_MESSAGE_SIZE: usize = 32 * 1024 * 1024;

/// Rejects a peer-advertised message length that exceeds [`MAX_MESSAGE_SIZE`]
/// before we allocate a buffer of that size.
fn check_content_length(length: usize) -> Result<()> {
    if length > MAX_MESSAGE_SIZE {
        return Err(DaimonError::Mcp(format!(
            "Content-Length {length} exceeds maximum allowed message size {MAX_MESSAGE_SIZE}"
        )));
    }
    Ok(())
}

/// Trait for sending JSON-RPC messages to an MCP server.
///
/// Request ids are allocated by the transport, not the caller: every
/// transport owns a per-transport atomic counter, so any number of callers
/// (e.g. multiple [`McpToolBridge`](crate::mcp::McpToolBridge)s sharing one
/// transport) get unique, collision-free ids and correct response
/// correlation.
pub trait McpTransport: Send + Sync {
    /// Sends a JSON-RPC request for `method` with optional `params`,
    /// allocating this transport's next request id, and waits for the
    /// response correlated to that id.
    fn request<'a>(
        &'a self,
        method: &'a str,
        params: Option<serde_json::Value>,
    ) -> Pin<Box<dyn Future<Output = Result<JsonRpcResponse>> + Send + 'a>>;

    /// Sends a notification (fire-and-forget, no response expected).
    fn notify<'a>(
        &'a self,
        notification: &'a JsonRpcNotification,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    /// Closes the transport.
    fn close<'a>(&'a self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
}

/// Stdio transport: communicates with a child process via stdin/stdout.
///
/// Uses Content-Length framing (like LSP).
pub struct StdioTransport {
    child_stdin: tokio::sync::Mutex<Option<tokio::process::ChildStdin>>,
    child_stdout: tokio::sync::Mutex<Option<tokio::io::BufReader<tokio::process::ChildStdout>>>,
    child: tokio::sync::Mutex<Option<tokio::process::Child>>,
    // ponytail: stdio framing carries no JSON-RPC id correlation, so a bare
    // write-then-read pair lets concurrent `request` calls (the agent runs
    // tools in parallel over one shared transport) read each other's
    // responses. Rather than build a full demuxing reader task, we serialize
    // every write+read round trip behind this single lock — held for the
    // whole exchange — so a request and its response can never interleave
    // with another call's. Server-initiated messages (notifications) that
    // arrive between the write and the matching response are skipped by id
    // inside the read loop.
    request_lock: tokio::sync::Mutex<()>,
    next_id: AtomicU64,
}

impl StdioTransport {
    /// Spawns a new child process and creates a stdio transport.
    pub async fn new(
        program: impl AsRef<std::ffi::OsStr>,
        args: impl IntoIterator<Item = impl AsRef<std::ffi::OsStr>>,
    ) -> Result<Self> {
        use tokio::process::Command;

        let mut child = Command::new(program)
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| DaimonError::Mcp(format!("failed to spawn MCP server: {e}")))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| DaimonError::Mcp("failed to open stdin".into()))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| DaimonError::Mcp("failed to open stdout".into()))?;

        Ok(Self {
            child_stdin: tokio::sync::Mutex::new(Some(stdin)),
            child_stdout: tokio::sync::Mutex::new(Some(tokio::io::BufReader::new(stdout))),
            child: tokio::sync::Mutex::new(Some(child)),
            request_lock: tokio::sync::Mutex::new(()),
            next_id: AtomicU64::new(1),
        })
    }

    async fn write_message(&self, body: &[u8]) -> Result<()> {
        use tokio::io::AsyncWriteExt;

        let mut stdin_guard = self.child_stdin.lock().await;
        let stdin = stdin_guard
            .as_mut()
            .ok_or_else(|| DaimonError::Mcp("transport closed".into()))?;

        let header = format!("Content-Length: {}\r\n\r\n", body.len());
        stdin
            .write_all(header.as_bytes())
            .await
            .map_err(|e| DaimonError::Mcp(format!("write header: {e}")))?;
        stdin
            .write_all(body)
            .await
            .map_err(|e| DaimonError::Mcp(format!("write body: {e}")))?;
        stdin
            .flush()
            .await
            .map_err(|e| DaimonError::Mcp(format!("flush: {e}")))?;

        Ok(())
    }
}

/// Reads one Content-Length framed message body off `reader`.
async fn read_framed_message<R>(reader: &mut R) -> Result<Vec<u8>>
where
    R: tokio::io::AsyncBufRead + Unpin,
{
    use tokio::io::AsyncBufReadExt;

    let mut content_length: Option<usize> = None;
    let mut header_line = String::new();

    loop {
        header_line.clear();
        let bytes_read = reader
            .read_line(&mut header_line)
            .await
            .map_err(|e| DaimonError::Mcp(format!("read header: {e}")))?;

        if bytes_read == 0 {
            return Err(DaimonError::Mcp("server closed connection".into()));
        }

        let trimmed = header_line.trim();
        if trimmed.is_empty() {
            break;
        }

        if let Some(len_str) = trimmed.strip_prefix("Content-Length:") {
            content_length = Some(
                len_str
                    .trim()
                    .parse()
                    .map_err(|e| DaimonError::Mcp(format!("invalid Content-Length: {e}")))?,
            );
        }
    }

    let length =
        content_length.ok_or_else(|| DaimonError::Mcp("missing Content-Length header".into()))?;

    check_content_length(length)?;

    use tokio::io::AsyncReadExt;
    let mut body = vec![0u8; length];
    reader
        .read_exact(&mut body)
        .await
        .map_err(|e| DaimonError::Mcp(format!("read body: {e}")))?;

    Ok(body)
}

/// Reads framed messages off `reader` until one deserializes to a JSON-RPC
/// response whose `id` matches `expected_id`, skipping anything else.
///
/// A server may interleave notifications (no `id`) — or, in principle,
/// responses to other requests — before the response we're waiting for;
/// returning the next frame blindly would hand a notification body to the
/// caller as its "response" and desync every subsequent round trip. Skipped
/// messages are logged at debug level. Bounded by [`MAX_SKIPPED_MESSAGES`]
/// so a notification-flooding peer can't pin us in this loop forever.
async fn read_matching_response<R>(reader: &mut R, expected_id: u64) -> Result<JsonRpcResponse>
where
    R: tokio::io::AsyncBufRead + Unpin,
{
    for _ in 0..MAX_SKIPPED_MESSAGES {
        let bytes = read_framed_message(reader).await?;
        let response: JsonRpcResponse = serde_json::from_slice(&bytes)
            .map_err(|e| DaimonError::Mcp(format!("deserialize response: {e}")))?;

        match response.id {
            Some(id) if id == expected_id => return Ok(response),
            Some(id) => {
                tracing::debug!(
                    expected = expected_id,
                    received = id,
                    "skipping JSON-RPC message with non-matching id"
                );
            }
            None => {
                tracing::debug!(
                    expected = expected_id,
                    "skipping JSON-RPC server notification while awaiting response"
                );
            }
        }
    }

    Err(DaimonError::Mcp(format!(
        "no response for request id {expected_id} within {MAX_SKIPPED_MESSAGES} messages"
    )))
}

impl McpTransport for StdioTransport {
    fn request<'a>(
        &'a self,
        method: &'a str,
        params: Option<serde_json::Value>,
    ) -> Pin<Box<dyn Future<Output = Result<JsonRpcResponse>> + Send + 'a>> {
        Box::pin(async move {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            let request = JsonRpcRequest::new(id, method, params);
            let body = serde_json::to_vec(&request)
                .map_err(|e| DaimonError::Mcp(format!("serialize request: {e}")))?;

            // Hold the round-trip lock across both the write and the read so a
            // response can only be consumed by the request that produced it.
            let _guard = self.request_lock.lock().await;
            self.write_message(&body).await?;

            let mut stdout_guard = self.child_stdout.lock().await;
            let stdout = stdout_guard
                .as_mut()
                .ok_or_else(|| DaimonError::Mcp("transport closed".into()))?;

            read_matching_response(stdout, id).await
        })
    }

    fn notify<'a>(
        &'a self,
        notification: &'a JsonRpcNotification,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let body = serde_json::to_vec(notification)
                .map_err(|e| DaimonError::Mcp(format!("serialize notification: {e}")))?;
            // Take the same round-trip lock so a notification write can't
            // interleave between a concurrent request's write and its read.
            let _guard = self.request_lock.lock().await;
            self.write_message(&body).await
        })
    }

    fn close<'a>(&'a self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            self.child_stdin.lock().await.take();
            self.child_stdout.lock().await.take();
            if let Some(mut child) = self.child.lock().await.take() {
                let _ = child.kill().await;
            }
            Ok(())
        })
    }
}

/// HTTP transport: sends JSON-RPC requests via HTTP POST.
pub struct HttpTransport {
    url: String,
    client: reqwest::Client,
    headers: std::collections::HashMap<String, String>,
    timeout: std::time::Duration,
    next_id: AtomicU64,
}

impl HttpTransport {
    /// Creates an HTTP transport targeting the given URL.
    ///
    /// Each request is bounded by a 30-second timeout by default; use
    /// [`with_timeout`](Self::with_timeout) to change it.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            client: reqwest::Client::new(),
            headers: std::collections::HashMap::new(),
            timeout: DEFAULT_HTTP_TIMEOUT,
            next_id: AtomicU64::new(1),
        }
    }

    /// Adds a custom header to all requests (e.g. for authentication).
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    /// Sets the per-request timeout (default: 30 seconds).
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

impl McpTransport for HttpTransport {
    fn request<'a>(
        &'a self,
        method: &'a str,
        params: Option<serde_json::Value>,
    ) -> Pin<Box<dyn Future<Output = Result<JsonRpcResponse>> + Send + 'a>> {
        Box::pin(async move {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            let request = JsonRpcRequest::new(id, method, params);

            let mut req = self
                .client
                .post(&self.url)
                .timeout(self.timeout)
                .json(&request);
            for (key, value) in &self.headers {
                req = req.header(key.as_str(), value.as_str());
            }

            let resp = req
                .send()
                .await
                .map_err(|e| DaimonError::Mcp(format!("HTTP request failed: {e}")))?;

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                return Err(DaimonError::Mcp(format!("HTTP {status}: {text}")));
            }

            let response: JsonRpcResponse = resp
                .json()
                .await
                .map_err(|e| DaimonError::Mcp(format!("deserialize response: {e}")))?;

            Ok(response)
        })
    }

    fn notify<'a>(
        &'a self,
        notification: &'a JsonRpcNotification,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let mut req = self
                .client
                .post(&self.url)
                .timeout(self.timeout)
                .json(notification);
            for (key, value) in &self.headers {
                req = req.header(key.as_str(), value.as_str());
            }

            let resp = req
                .send()
                .await
                .map_err(|e| DaimonError::Mcp(format!("HTTP notify failed: {e}")))?;

            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                return Err(DaimonError::Mcp(format!("HTTP {status}: {text}")));
            }

            Ok(())
        })
    }

    fn close<'a>(&'a self) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_length_within_cap_ok() {
        assert!(check_content_length(0).is_ok());
        assert!(check_content_length(1024).is_ok());
        assert!(check_content_length(MAX_MESSAGE_SIZE).is_ok());
    }

    #[test]
    fn test_content_length_over_cap_rejected() {
        let err = check_content_length(MAX_MESSAGE_SIZE + 1).unwrap_err();
        assert!(matches!(err, DaimonError::Mcp(_)));
        // A wildly oversized advertised length must also be rejected before
        // any allocation happens.
        assert!(check_content_length(usize::MAX).is_err());
    }

    #[test]
    fn test_http_transport_new() {
        let t = HttpTransport::new("http://localhost:8080/mcp");
        assert_eq!(t.url, "http://localhost:8080/mcp");
    }

    #[test]
    fn test_http_transport_with_header() {
        let t = HttpTransport::new("http://localhost:8080")
            .with_header("Authorization", "Bearer token123");
        assert_eq!(
            t.headers.get("Authorization"),
            Some(&"Bearer token123".to_string())
        );
    }

    #[test]
    fn test_http_transport_default_timeout() {
        let t = HttpTransport::new("http://localhost:8080");
        assert_eq!(t.timeout, DEFAULT_HTTP_TIMEOUT);
    }

    #[test]
    fn test_http_transport_with_timeout() {
        let t = HttpTransport::new("http://localhost:8080")
            .with_timeout(std::time::Duration::from_secs(5));
        assert_eq!(t.timeout, std::time::Duration::from_secs(5));
    }

    /// Frames `body` with a Content-Length header, as an MCP stdio server
    /// would write it.
    fn frame(body: &str) -> String {
        format!("Content-Length: {}\r\n\r\n{body}", body.len())
    }

    #[tokio::test]
    async fn test_read_matching_response_skips_interleaved_notification() {
        // A server notification (no id) arrives before the actual response;
        // it must be skipped, not returned as the "response".
        let input = format!(
            "{}{}",
            frame(r#"{"jsonrpc":"2.0","method":"notifications/progress","params":{}}"#),
            frame(r#"{"jsonrpc":"2.0","id":1,"result":{"ok":true}}"#),
        );
        let mut reader = tokio::io::BufReader::new(input.as_bytes());
        let response = read_matching_response(&mut reader, 1).await.unwrap();
        assert_eq!(response.id, Some(1));
        assert_eq!(response.result.unwrap()["ok"], true);
    }

    #[tokio::test]
    async fn test_read_matching_response_skips_non_matching_id() {
        let input = format!(
            "{}{}",
            frame(r#"{"jsonrpc":"2.0","id":99,"result":{"ok":false}}"#),
            frame(r#"{"jsonrpc":"2.0","id":7,"result":{"ok":true}}"#),
        );
        let mut reader = tokio::io::BufReader::new(input.as_bytes());
        let response = read_matching_response(&mut reader, 7).await.unwrap();
        assert_eq!(response.id, Some(7));
        assert_eq!(response.result.unwrap()["ok"], true);
    }

    #[tokio::test]
    async fn test_read_matching_response_bounded_by_skip_limit() {
        // A peer that streams nothing but notifications must not pin the
        // read loop forever: after MAX_SKIPPED_MESSAGES frames it errors.
        let notification = frame(r#"{"jsonrpc":"2.0","method":"notifications/progress"}"#);
        let input = notification.repeat(MAX_SKIPPED_MESSAGES + 1);
        let mut reader = tokio::io::BufReader::new(input.as_bytes());
        let err = read_matching_response(&mut reader, 1).await.unwrap_err();
        assert!(matches!(err, DaimonError::Mcp(_)));
    }
}
