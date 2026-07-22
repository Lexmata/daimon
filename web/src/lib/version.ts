// Injected at build time from the workspace-root Cargo.toml — see
// astro.config.mjs (`vite.define`). Never hardcode the crate version.
declare const __DAIMON_VERSION__: string;

/** The crate version (major.minor), e.g. "0.22". */
export const CRATE_VERSION: string = __DAIMON_VERSION__;
