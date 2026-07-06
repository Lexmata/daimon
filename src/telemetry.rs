//! OpenTelemetry integration for exporting `tracing` spans to OTLP collectors.
//!
//! Since Daimon already instruments the entire agent pipeline with the `tracing`
//! crate, enabling OpenTelemetry export is a matter of installing the right
//! subscriber layer. This module provides [`init_otel_tracing`] to set up the
//! pipeline and [`OtelGuard::shutdown`] for graceful shutdown.
//!
//! # Feature Flag
//!
//! Requires the `otel` feature to be enabled.
//!
//! # Example
//!
//! ```ignore
//! use daimon::telemetry::{OtelConfig, init_otel_tracing};
//!
//! #[tokio::main]
//! async fn main() {
//!     let guard = init_otel_tracing(OtelConfig::default()).expect("otel init");
//!
//!     // ... run your agent ...
//!
//!     guard.shutdown();
//! }
//! ```

use opentelemetry::trace::TracerProvider;
use opentelemetry_otlp::{SpanExporter, WithExportConfig};
use opentelemetry_sdk::trace::SdkTracerProvider;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// Configuration for the OpenTelemetry OTLP pipeline.
#[derive(Debug, Clone)]
pub struct OtelConfig {
    /// OTLP endpoint URL (default: `http://localhost:4318` for HTTP).
    pub endpoint: String,
    /// Service name reported in traces.
    pub service_name: String,
    /// Whether to also install a `tracing_subscriber::fmt` layer for
    /// local console output alongside OTLP export.
    pub with_fmt_layer: bool,
}

impl Default for OtelConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:4318".to_string(),
            service_name: "daimon-agent".to_string(),
            with_fmt_layer: true,
        }
    }
}

impl OtelConfig {
    /// Sets the OTLP endpoint.
    pub fn endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Sets the service name.
    pub fn service_name(mut self, name: impl Into<String>) -> Self {
        self.service_name = name.into();
        self
    }

    /// Enables or disables the local fmt layer.
    pub fn with_fmt_layer(mut self, enabled: bool) -> Self {
        self.with_fmt_layer = enabled;
        self
    }
}

/// Guard returned by [`init_otel_tracing`]. Holds the tracer provider
/// for graceful shutdown.
pub struct OtelGuard {
    provider: SdkTracerProvider,
}

impl OtelGuard {
    /// Shuts down the OpenTelemetry pipeline, flushing any pending spans.
    pub fn shutdown(self) {
        if let Err(e) = self.provider.shutdown() {
            eprintln!("OpenTelemetry shutdown error: {e}");
        }
    }
}

/// Initializes the `tracing` subscriber with an OpenTelemetry OTLP layer.
///
/// This wires up `tracing` spans → OpenTelemetry → OTLP exporter so all
/// existing `#[instrument]` annotations on agent, model, tool, and
/// orchestration code automatically appear in your observability backend
/// (Jaeger, Tempo, Datadog, etc.).
///
/// Returns an [`OtelGuard`] that should be kept alive for the duration of
/// the application. Call [`OtelGuard::shutdown`] before exit for clean
/// flush.
pub fn init_otel_tracing(config: OtelConfig) -> Result<OtelGuard, Box<dyn std::error::Error>> {
    let exporter = SpanExporter::builder()
        .with_http()
        .with_endpoint(config.endpoint.clone())
        .build()?;

    let resource = opentelemetry_sdk::Resource::builder()
        .with_service_name(config.service_name)
        .build();

    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(resource)
        .build();

    let tracer = provider.tracer("daimon");
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    if config.with_fmt_layer {
        let fmt_layer = tracing_subscriber::fmt::layer().compact();
        tracing_subscriber::registry()
            .with(otel_layer)
            .with(fmt_layer)
            .with(filter)
            .init();
    } else {
        tracing_subscriber::registry()
            .with(otel_layer)
            .with(filter)
            .init();
    }

    Ok(OtelGuard { provider })
}
