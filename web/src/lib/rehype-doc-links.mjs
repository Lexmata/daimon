/**
 * Rehype plugin that rewrites relative cross-doc links written for the GitHub
 * repo view (e.g. `[agents.md](agents.md)` in `docs/*.md`) into site routes
 * (`/daimon/docs/agents/`), preserving any `#fragment`.
 *
 * @param {{ base?: string }} [options]
 * @returns {(tree: import('hast').Root) => void}
 */
export function rehypeDocLinks({ base = '' } = {}) {
  const prefix = base.replace(/\/+$/, '');
  const DOC_LINK = /^(?:\.\/)?([A-Za-z0-9_-]+)\.md(#.*)?$/;

  /** @param {any} node */
  const visit = (node) => {
    if (node.type === 'element' && node.tagName === 'a' && node.properties?.href) {
      const match = DOC_LINK.exec(String(node.properties.href));
      if (match) {
        node.properties.href = `${prefix}/docs/${match[1]}/${match[2] ?? ''}`;
      }
    }
    if (Array.isArray(node.children)) {
      node.children.forEach(visit);
    }
  };

  return (tree) => visit(tree);
}
