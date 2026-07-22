import type { CollectionEntry } from 'astro:content';
import type { IconName } from './icons';

type Doc = CollectionEntry<'docs'>;

/**
 * Page title for a doc: frontmatter `title` if present, otherwise the first
 * `# heading` in the markdown, otherwise the slug title-cased. Mirrors the
 * Angular site's `document.title` behavior (first `<h1>` won).
 */
export function docTitle(doc: Doc): string {
  if (doc.data.title) return doc.data.title;
  const heading = doc.body?.match(/^#\s+(.+?)\s*$/m);
  if (heading) return heading[1];
  return doc.id
    .split('-')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

export interface NavLink {
  label: string;
  slug: string;
  icon: IconName;
}

export interface NavSection {
  title: string;
  links: NavLink[];
}

// Fixed sidebar order, mirroring the Angular sidebar's sections
// (getting-started first). Troubleshooting existed in docs/ but was missing
// from the old nav; it slots into Reference.
const NAV_SECTIONS: NavSection[] = [
  {
    title: 'Getting Started',
    links: [
      { label: 'Getting Started', slug: 'getting-started', icon: 'rocket' },
      { label: 'Architecture', slug: 'architecture', icon: 'sitemap' },
    ],
  },
  {
    title: 'Core Concepts',
    links: [
      { label: 'Agents', slug: 'agents', icon: 'robot' },
      { label: 'Tools', slug: 'tools', icon: 'wrench' },
      { label: 'Orchestration', slug: 'orchestration', icon: 'diagram-project' },
    ],
  },
  {
    title: 'Advanced',
    links: [
      { label: 'Multi-Agent', slug: 'multi-agent', icon: 'users' },
      { label: 'RAG', slug: 'rag', icon: 'book-open' },
      { label: 'Distributed', slug: 'distributed', icon: 'network-wired' },
    ],
  },
  {
    title: 'Reference',
    links: [
      { label: 'Providers', slug: 'providers', icon: 'plug' },
      { label: 'Performance', slug: 'performance', icon: 'gauge-high' },
      { label: 'Plugin Development', slug: 'plugin-development', icon: 'puzzle-piece' },
      { label: 'Troubleshooting', slug: 'troubleshooting', icon: 'circle-question' },
    ],
  },
];

/**
 * Sidebar sections: the fixed nav above, restricted to docs that actually
 * exist, plus a trailing "More" section for any `docs/*.md` file not yet
 * listed — so new docs show up in the sidebar automatically.
 */
export function buildNav(docs: Doc[]): NavSection[] {
  const bySlug = new Map(docs.map((doc) => [doc.id, doc]));
  const listed = new Set<string>();

  const sections: NavSection[] = [];
  for (const section of NAV_SECTIONS) {
    const links = section.links.filter((link) => bySlug.has(link.slug));
    links.forEach((link) => listed.add(link.slug));
    if (links.length > 0) sections.push({ ...section, links });
  }

  const extras = docs
    .filter((doc) => !listed.has(doc.id))
    .map((doc) => ({ label: docTitle(doc), slug: doc.id, icon: 'file-lines' as IconName }))
    .sort((a, b) => a.label.localeCompare(b.label));
  if (extras.length > 0) sections.push({ title: 'More', links: extras });

  return sections;
}
