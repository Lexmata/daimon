// Single helper for base-path-aware internal URLs. `base` is `/daimon` for
// GitHub Pages project hosting (see astro.config.mjs); every internal link and
// asset reference must go through this so the prefix is applied consistently.
const base = import.meta.env.BASE_URL.replace(/\/+$/, '');

export function withBase(path: string): string {
  return `${base}${path.startsWith('/') ? path : `/${path}`}`;
}
