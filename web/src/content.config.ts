import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

// Docs are sourced straight from the repo-root `docs/` directory — no CI copy
// step. Any new `docs/*.md` file automatically becomes a `/docs/<slug>/` page.
const docs = defineCollection({
  loader: glob({ pattern: '**/*.md', base: '../docs' }),
  // The docs carry no frontmatter today; titles are derived from the first
  // `# heading` (see src/lib/docs.ts). A `title` frontmatter key wins if added.
  schema: z.object({
    title: z.string().optional(),
    description: z.string().optional(),
  }),
});

export const collections = { docs };
