//! Deprecated: this crate has been absorbed into [`daimon_provider_local`].
//!
//! Depend on `daimon-provider-local` directly (`daimon_provider_local::llamacpp`)
//! instead — this crate will be dropped from future releases. Kept as a
//! re-export shim for one release so existing `Cargo.toml` pins keep working.

#[deprecated(
    since = "0.21.0",
    note = "moved to daimon_provider_local::llamacpp; depend on daimon-provider-local directly"
)]
pub use daimon_provider_local::llamacpp::LlamaCpp;

#[deprecated(
    since = "0.21.0",
    note = "moved to daimon_provider_local::llamacpp_embed; depend on daimon-provider-local directly"
)]
pub use daimon_provider_local::llamacpp_embed::LlamaCppEmbedding;
