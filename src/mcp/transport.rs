//! MCP transport implementations (stdio and HTTP).

use std::future::Future;
use std::pin::Pin;

use crate::error::{DaimonError, Result};
use crate::mcp::protocol::{JsonRpcNotification, JsonRpcRequest, JsonRpcResponse};

/// Trait for sending JSON-RPC messages to an MCP server.
pub trait McpTransport: Send + Sync {
    /// Sends a request and waits for the response.
    fn send<'a>(
        &'a self,
        request: &'a JsonRpcRequest,
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

    async fn read_message(&self) -> Result<Vec<u8>> {
        use tokio::io::AsyncBufReadExt;

        let mut stdout_guard = self.child_stdout.lock().await;
        let stdout = stdout_guard
            .as_mut()
            .ok_or_else(|| DaimonError::Mcp("transport closed".into()))?;

        let mut content_length: Option<usize> = None;
        let mut header_line = String::new();

        loop {
            header_line.clear();
            let bytes_read = stdout
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

        let length = content_length
            .ok_or_else(|| DaimonError::Mcp("missing Content-Length header".into()))?;

        use tokio::io::AsyncReadExt;
        let mut body = vec![0u8; length];
        stdout
            .read_exact(&mut body)
            .await
            .map_err(|e| DaimonError::Mcp(format!("read body: {e}")))?;

        Ok(body)
    }
}

impl McpTransport for StdioTransport {
    fn send<'a>(
        &'a self,
        request: &'a JsonRpcRequest,
    ) -> Pin<Box<dyn Future<Output = Result<JsonRpcResponse>> + Send + 'a>> {
        Box::pin(async move {
            let body = serde_json::to_vec(request)
                .map_err(|e| DaimonError::Mcp(format!("serialize request: {e}")))?;
            self.write_message(&body).await?;

            let response_bytes = self.read_message().await?;
            let response: JsonRpcResponse = serde_json::from_slice(&response_bytes)
                .map_err(|e| DaimonError::Mcp(format!("deserialize response: {e}")))?;

            Ok(response)
        })
    }

    fn notify<'a>(
        &'a self,
        notification: &'a JsonRpcNotification,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let body = serde_json::to_vec(notification)
                .map_err(|e| DaimonError::Mcp(format!("serialize notification: {e}")))?;
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
}

impl HttpTransport {
    /// Creates an HTTP transport targeting the given URL.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            client: reqwest::Client::new(),
            headers: std::collections::HashMap::new(),
        }
    }

    /// Adds a custom header to all requests (e.g. for authentication).
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }
}

impl McpTransport for HttpTransport {
    fn send<'a>(
        &'a self,
        request: &'a JsonRpcRequest,
    ) -> Pin<Box<dyn Future<Output = Result<JsonRpcResponse>> + Send + 'a>> {
        Box::pin(async move {
            let mut req = self.client.post(&self.url).json(request);
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
            let mut req = self.client.post(&self.url).json(notification);
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
}
