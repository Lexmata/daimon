import { RenderMode, ServerRoute } from '@angular/ssr';

export const serverRoutes: ServerRoute[] = [
  {
    path: '',
    renderMode: RenderMode.Prerender,
  },
  {
    path: 'docs/:slug',
    renderMode: RenderMode.Prerender,
    async getPrerenderParams() {
      return [
        { slug: 'getting-started' },
        { slug: 'architecture' },
        { slug: 'agents' },
        { slug: 'tools' },
        { slug: 'orchestration' },
        { slug: 'multi-agent' },
        { slug: 'rag' },
        { slug: 'distributed' },
        { slug: 'providers' },
        { slug: 'performance' },
        { slug: 'plugin-development' },
      ];
    },
  },
  {
    path: '**',
    renderMode: RenderMode.Prerender,
  },
];
