// @ts-check
import { readFileSync } from 'node:fs';
import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';
import { rehypeDocLinks } from './src/lib/rehype-doc-links.mjs';

// GitHub Pages project hosting: https://lexmata.github.io/daimon/
// (matches the base-href the previous Angular build used).
const BASE = '/daimon';

// Crate version (major.minor) from the workspace-root Cargo.toml, injected at
// build time so the header badge can never drift from the release.
const CRATE_VERSION = (() => {
  try {
    const toml = readFileSync(new URL('../Cargo.toml', import.meta.url), 'utf8');
    // The first `version = "x.y.z"` line in the file is [workspace.package].version.
    return toml.match(/^version\s*=\s*"(\d+\.\d+)\.\d+"/m)?.[1] ?? '0.0';
  } catch {
    return '0.0';
  }
})();

// https://astro.build/config
export default defineConfig({
  site: 'https://lexmata.github.io',
  base: BASE,
  output: 'static',
  markdown: {
    // Closest Shiki match to the previous highlight.js `github-dark` stylesheet
    // (same #0d1117 code background).
    shikiConfig: {
      theme: 'github-dark-default',
    },
    rehypePlugins: [[rehypeDocLinks, { base: BASE }]],
  },
  vite: {
    plugins: [tailwindcss()],
    define: {
      __DAIMON_VERSION__: JSON.stringify(CRATE_VERSION),
    },
  },
});
