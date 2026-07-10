import {
  Component,
  inject,
  signal,
  effect,
  PLATFORM_ID,
  afterNextRender,
} from '@angular/core';
import { isPlatformBrowser } from '@angular/common';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { marked } from 'marked';
import hljs from 'highlight.js/lib/core';
import rust from 'highlight.js/lib/languages/rust';
import bash from 'highlight.js/lib/languages/bash';
import json from 'highlight.js/lib/languages/json';
import ini from 'highlight.js/lib/languages/ini';
import plaintext from 'highlight.js/lib/languages/plaintext';
import type { Tokens } from 'marked';

// Core build + only the grammars the docs use (~200kB gzipped lighter than
// the full all-languages build). Unregistered languages fall back to
// plaintext below, same as before. `ini` declares the `toml` alias.
hljs.registerLanguage('rust', rust);
hljs.registerLanguage('bash', bash);
hljs.registerLanguage('json', json);
hljs.registerLanguage('ini', ini);
hljs.registerLanguage('plaintext', plaintext);

type DocState = 'loading' | 'ready' | 'not-found' | 'error';

@Component({
  selector: 'app-doc-page',
  standalone: true,
  imports: [RouterLink],
  template: `
    <article class="doc-page">
      @if (state() === 'loading') {
        <div class="doc-skeleton" aria-busy="true">
          <div class="skeleton-title"></div>
          <div class="skeleton-line"></div>
          <div class="skeleton-line short"></div>
          <div class="skeleton-line"></div>
          <div class="skeleton-line short"></div>
          <div class="skeleton-code"></div>
          <div class="skeleton-line"></div>
          <div class="skeleton-line"></div>
        </div>
      } @else if (state() === 'not-found') {
        <div class="doc-error">
          <h1>Page not found</h1>
          <p>The documentation page "{{ slug() }}" could not be found.</p>
          <a routerLink="/docs/getting-started" class="doc-error-link">
            Go to Getting Started
          </a>
        </div>
      } @else if (state() === 'error') {
        <div class="doc-error">
          <h1>Something went wrong</h1>
          <p>Failed to load the documentation page. Please try again.</p>
          <button type="button" (click)="loadDoc()" class="doc-retry-btn">
            Retry
          </button>
        </div>
      } @else {
        <div
          class="prose doc-content"
          [innerHTML]="sanitizedHtml()"
        ></div>
      }
    </article>
  `,
  styles: `
    :host {
      display: block;
    }

    .doc-page {
      padding: 2rem 1.5rem;
      max-width: 48rem;
      margin: 0 auto;
    }

    .doc-content {
      animation: fadeIn 0.2s ease-out;
    }

    @keyframes fadeIn {
      from {
        opacity: 0;
      }
      to {
        opacity: 1;
      }
    }

    .doc-skeleton {
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }

    .skeleton-title {
      height: 2.25rem;
      width: 70%;
      background: linear-gradient(
        90deg,
        var(--color-surface-lighter) 25%,
        var(--color-surface-light) 50%,
        var(--color-surface-lighter) 75%
      );
      background-size: 200% 100%;
      animation: shimmer 1.5s infinite;
      border-radius: 0.375rem;
    }

    .skeleton-line {
      height: 1rem;
      width: 100%;
      background: linear-gradient(
        90deg,
        var(--color-surface-lighter) 25%,
        var(--color-surface-light) 50%,
        var(--color-surface-lighter) 75%
      );
      background-size: 200% 100%;
      animation: shimmer 1.5s infinite;
      border-radius: 0.25rem;
    }

    .skeleton-line.short {
      width: 85%;
    }

    .skeleton-code {
      height: 8rem;
      width: 100%;
      background: linear-gradient(
        90deg,
        var(--color-surface-lighter) 25%,
        var(--color-surface-light) 50%,
        var(--color-surface-lighter) 75%
      );
      background-size: 200% 100%;
      animation: shimmer 1.5s infinite;
      border-radius: 0.5rem;
    }

    @keyframes shimmer {
      0% {
        background-position: 200% 0;
      }
      100% {
        background-position: -200% 0;
      }
    }

    .doc-error {
      text-align: center;
      padding: 4rem 2rem;
    }

    .doc-error h1 {
      font-size: 1.5rem;
      font-weight: 700;
      color: var(--color-text);
      margin: 0 0 0.5rem;
    }

    .doc-error p {
      color: var(--color-text-muted);
      margin: 0 0 1.5rem;
    }

    .doc-error-link {
      display: inline-block;
      color: var(--color-primary);
      font-weight: 600;
      text-decoration: none;
      transition: color 0.2s ease;
    }

    .doc-error-link:hover {
      color: var(--color-primary-light);
    }

    .doc-retry-btn {
      padding: 0.5rem 1.25rem;
      font-size: 0.9375rem;
      font-weight: 600;
      color: white;
      background: var(--color-primary);
      border: none;
      border-radius: 0.5rem;
      cursor: pointer;
      transition: background 0.2s ease;
    }

    .doc-retry-btn:hover {
      background: var(--color-primary-light);
    }
  `,
})
export class DocPageComponent {
  private readonly route = inject(ActivatedRoute);
  private readonly sanitizer = inject(DomSanitizer);
  private readonly platformId = inject(PLATFORM_ID);

  readonly slug = signal<string>('');
  readonly state = signal<DocState>('loading');
  readonly sanitizedHtml = signal<SafeHtml>('');

  constructor() {
    this.configureMarked();

    effect(() => {
      const slug = this.slug();
      if (slug && isPlatformBrowser(this.platformId)) {
        this.loadDoc();
      } else if (slug && !isPlatformBrowser(this.platformId)) {
        this.state.set('loading');
      }
    });

    afterNextRender(() => {
      this.route.paramMap.subscribe((params) => {
        const slug = params.get('slug') ?? '';
        this.slug.set(slug);
        if (isPlatformBrowser(this.platformId)) {
          window.scrollTo({ top: 0, behavior: 'smooth' });
        }
      });
    });
  }

  private configureMarked(): void {
    const renderer = new marked.Renderer();

    renderer.code = ({ text, lang }: Tokens.Code) => {
      const code = text;
      const language = lang ?? 'plaintext';

      if (language && hljs.getLanguage(language)) {
        const highlighted = hljs.highlight(code, { language }).value;
        return `<pre><code class="hljs language-${language}">${highlighted}</code></pre>`;
      }
      const escaped = hljs.highlight(code, { language: 'plaintext' }).value;
      return `<pre><code class="hljs">${escaped}</code></pre>`;
    };

    marked.use({ renderer });
  }

  async loadDoc(): Promise<void> {
    const slug = this.slug();
    if (!slug) {
      this.state.set('not-found');
      return;
    }

    this.state.set('loading');

    try {
      const url =
        typeof document !== 'undefined'
          ? new URL(`docs/${slug}.md`, document.baseURI).href
          : `docs/${slug}.md`;

      const response = await fetch(url);

      if (!response.ok) {
        if (response.status === 404) {
          this.state.set('not-found');
        } else {
          this.state.set('error');
        }
        return;
      }

      const markdown = await response.text();
      const html = marked.parse(markdown, { async: false }) as string;
      this.sanitizedHtml.set(this.sanitizer.bypassSecurityTrustHtml(html));
      this.state.set('ready');

      this.updatePageTitle(html, slug);
    } catch {
      this.state.set('error');
    }
  }

  private updatePageTitle(html: string, slug: string): void {
    if (typeof document === 'undefined') return;

    const match = html.match(/<h1[^>]*>([^<]+)<\/h1>/);
    const title = match
      ? `${match[1].trim()} | Daimon`
      : `${this.slugToTitle(slug)} | Daimon`;
    document.title = title;
  }

  private slugToTitle(slug: string): string {
    return slug
      .split('-')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  }
}
