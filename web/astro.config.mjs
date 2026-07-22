// @ts-check
import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';
import { rehypeDocLinks } from './src/lib/rehype-doc-links.mjs';

// GitHub Pages project hosting: https://lexmata.github.io/daimon/
// (matches the base-href the previous Angular build used).
const BASE = '/daimon';

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
  },
});
