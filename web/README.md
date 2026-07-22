# Web

The Daimon documentation website: an [Astro](https://astro.build) 7 static
site, deployed to GitHub Pages at <https://lexmata.github.io/daimon/>.

Docs content is sourced straight from the repo-root [`docs/`](../docs)
directory via an Astro content collection (`src/content.config.ts`) — there is
no copy step. Any new `docs/*.md` file automatically becomes a
`/docs/<slug>/` page and appears in the sidebar (under "More" until added to
the fixed nav in `src/lib/docs.ts`).

## Prerequisites

- Node.js **>= 22.12.0**
- pnpm **10.29.2** (pinned via `packageManager` in `package.json`; use
  [Corepack](https://nodejs.org/api/corepack.html): `corepack enable`)

## Setup

```bash
pnpm install
```

## Development server

```bash
pnpm dev
```

Open <http://localhost:4321/daimon/> (the `/daimon` base path matches the
GitHub Pages deployment). The site reloads automatically on changes —
including edits to `../docs/*.md`.

## Build

```bash
pnpm build
```

Emits the static site to `dist/`. Preview the production build locally with:

```bash
pnpm preview
```

## Checks

```bash
pnpm check    # astro check (types + diagnostics)
```

## Deployment

`.github/workflows/pages.yml` builds and deploys the site to GitHub Pages on
pushes to `develop`/`main` that touch `web/**` or `docs/**`.
