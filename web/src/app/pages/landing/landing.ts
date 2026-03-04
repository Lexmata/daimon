import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';
import { FaIconComponent } from '@fortawesome/angular-fontawesome';
import {
  faPuzzlePiece,
  faBolt,
  faDatabase,
  faUsers,
  faServer,
  faPlug,
} from '@fortawesome/free-solid-svg-icons';
import { faGithub } from '@fortawesome/free-brands-svg-icons';

@Component({
  selector: 'app-landing',
  imports: [RouterLink, FaIconComponent],
  template: `
    <div class="landing">
      <!-- Hero Section -->
      <section class="hero">
        <div class="hero-bg">
          <div class="hero-gradient"></div>
          <div class="hero-grid"></div>
        </div>
        <div class="hero-content">
          <h1 class="hero-title">Daimon</h1>
          <p class="hero-subtitle">
            High-Performance AI Agent Framework for Rust
          </p>
          <p class="hero-description">
            Zero-cost abstractions, trait-based plugins, async-first design.
            Built for streaming, RAG, and distributed execution at scale.
          </p>
          <div class="hero-cta">
            <a routerLink="/docs/getting-started" class="btn btn-primary">
              Get Started
            </a>
            <a
              href="https://github.com/Lexmata/daimon"
              target="_blank"
              rel="noopener noreferrer"
              class="btn btn-secondary"
            >
              <fa-icon [icon]="faGithub" class="btn-icon" />
              View on GitHub
            </a>
          </div>
        </div>
      </section>

      <!-- Feature Grid -->
      <section class="features">
        <h2 class="section-title">Built for Production</h2>
        <p class="section-subtitle">
          Everything you need to build robust, scalable AI agents
        </p>
        <div class="feature-grid">
          <article class="feature-card">
            <div class="feature-icon">
              <fa-icon [icon]="faPuzzlePiece" />
            </div>
            <h3>Trait-Based Plugins</h3>
            <p>
              Add LLM providers, vector stores, and brokers by implementing
              traits. Compose your stack without lock-in.
            </p>
          </article>
          <article class="feature-card">
            <div class="feature-icon">
              <fa-icon [icon]="faBolt" />
            </div>
            <h3>Async Streaming</h3>
            <p>
              Real-time token streaming with async/await and zero-copy design.
              First-class cancellation and backpressure.
            </p>
          </article>
          <article class="feature-card">
            <div class="feature-icon">
              <fa-icon [icon]="faDatabase" />
            </div>
            <h3>RAG & Vector Stores</h3>
            <p>
              Built-in retrieval with pgvector, OpenSearch, Qdrant support.
              Semantic search and document ingestion out of the box.
            </p>
          </article>
          <article class="feature-card">
            <div class="feature-icon">
              <fa-icon [icon]="faUsers" />
            </div>
            <h3>Multi-Agent Patterns</h3>
            <p>
              Supervisor, handoff, agent-as-tool, and fork patterns. Compose
              agents into complex workflows.
            </p>
          </article>
          <article class="feature-card">
            <div class="feature-icon">
              <fa-icon [icon]="faServer" />
            </div>
            <h3>Distributed Execution</h3>
            <p>
              Task brokers for Redis, NATS, RabbitMQ, SQS, and more. Scale
              horizontally with durable queues.
            </p>
          </article>
          <article class="feature-card">
            <div class="feature-icon">
              <fa-icon [icon]="faPlug" />
            </div>
            <h3>MCP & A2A</h3>
            <p>
              Model Context Protocol client/server and Agent-to-Agent protocol.
              Interoperate with the broader AI ecosystem.
            </p>
          </article>
        </div>
      </section>

      <!-- Code Example -->
      <section class="code-section">
        <h2 class="section-title">Get Started in Minutes</h2>
        <p class="section-subtitle">
          A minimal agent with OpenAI in under 15 lines
        </p>
        <div class="code-block">
          <pre><code>{{ codeExample }}</code></pre>
        </div>
        <a routerLink="/docs/getting-started" class="code-cta">
          Read the full guide →
        </a>
      </section>

      <!-- Footer -->
      <footer class="footer">
        <div class="footer-content">
          <p class="footer-copyright">
            © {{ currentYear }} Lexmata LLC. Apache 2.0 + MIT dual-licensed.
          </p>
          <nav class="footer-links">
            <a
              href="https://github.com/Lexmata/daimon"
              target="_blank"
              rel="noopener noreferrer"
              >GitHub</a
            >
            <a
              href="https://crates.io/crates/daimon"
              target="_blank"
              rel="noopener noreferrer"
              >crates.io</a
            >
          </nav>
        </div>
      </footer>
    </div>
  `,
  styles: `
    :host {
      display: block;
    }

    .landing {
      min-height: 100vh;
      background-color: var(--color-surface);
    }

    /* Hero */
    .hero {
      position: relative;
      min-height: 85vh;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }

    .hero-bg {
      position: absolute;
      inset: 0;
      z-index: 0;
    }

    .hero-gradient {
      position: absolute;
      inset: 0;
      background: radial-gradient(
        ellipse 80% 50% at 50% -20%,
        rgba(249, 115, 22, 0.15) 0%,
        rgba(249, 115, 22, 0.05) 40%,
        transparent 70%
      );
    }

    .hero-grid {
      position: absolute;
      inset: 0;
      background-image: linear-gradient(
          rgba(249, 115, 22, 0.03) 1px,
          transparent 1px
        ),
        linear-gradient(
          90deg,
          rgba(249, 115, 22, 0.03) 1px,
          transparent 1px
        );
      background-size: 60px 60px;
    }

    .hero-content {
      position: relative;
      z-index: 1;
      max-width: 48rem;
      margin: 0 auto;
      padding: 2rem 1.5rem;
      text-align: center;
    }

    .hero-title {
      font-size: clamp(3.5rem, 10vw, 5.5rem);
      font-weight: 900;
      letter-spacing: -0.04em;
      line-height: 1;
      margin: 0 0 0.5rem;
      background: linear-gradient(
        135deg,
        #fff 0%,
        #f1f5f9 50%,
        var(--color-primary-light) 100%
      );
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }

    .hero-subtitle {
      font-size: clamp(1.25rem, 2.5vw, 1.5rem);
      font-weight: 600;
      color: var(--color-text);
      margin: 0 0 1rem;
      letter-spacing: -0.02em;
    }

    .hero-description {
      font-size: 1.125rem;
      color: var(--color-text-muted);
      line-height: 1.7;
      margin: 0 0 2.5rem;
      max-width: 36rem;
      margin-left: auto;
      margin-right: auto;
    }

    .hero-cta {
      display: flex;
      flex-wrap: wrap;
      gap: 1rem;
      justify-content: center;
    }

    .btn {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.875rem 1.75rem;
      font-size: 1rem;
      font-weight: 600;
      border-radius: 0.5rem;
      text-decoration: none;
      transition: all 0.2s ease;
      cursor: pointer;
    }

    .btn-primary {
      background: var(--color-primary);
      color: white;
      border: none;
      box-shadow: 0 4px 14px rgba(249, 115, 22, 0.35);
    }

    .btn-primary:hover {
      background: var(--color-primary-light);
      transform: translateY(-1px);
      box-shadow: 0 6px 20px rgba(249, 115, 22, 0.45);
    }

    .btn-secondary {
      background: var(--color-surface-light);
      color: var(--color-text);
      border: 1px solid var(--color-border);
    }

    .btn-secondary:hover {
      background: var(--color-surface-lighter);
      border-color: var(--color-primary);
      color: var(--color-primary);
    }

    .btn-icon {
      font-size: 1.125rem;
    }

    /* Features */
    .features {
      padding: 5rem 1.5rem;
      max-width: 72rem;
      margin: 0 auto;
    }

    .section-title {
      font-size: 2rem;
      font-weight: 800;
      text-align: center;
      margin: 0 0 0.5rem;
      color: var(--color-text);
      letter-spacing: -0.03em;
    }

    .section-subtitle {
      text-align: center;
      color: var(--color-text-muted);
      font-size: 1.125rem;
      margin: 0 0 3rem;
    }

    .feature-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 1.5rem;
    }

    .feature-card {
      background: var(--color-surface-light);
      border: 1px solid var(--color-border);
      border-radius: 0.75rem;
      padding: 1.75rem;
      transition: all 0.2s ease;
    }

    .feature-card:hover {
      border-color: rgba(249, 115, 22, 0.4);
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
    }

    .feature-icon {
      width: 3rem;
      height: 3rem;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(249, 115, 22, 0.15);
      color: var(--color-primary);
      border-radius: 0.5rem;
      font-size: 1.25rem;
      margin-bottom: 1rem;
    }

    .feature-card h3 {
      font-size: 1.125rem;
      font-weight: 700;
      color: var(--color-text);
      margin: 0 0 0.5rem;
    }

    .feature-card p {
      font-size: 0.9375rem;
      color: var(--color-text-muted);
      line-height: 1.6;
      margin: 0;
    }

    /* Code Section */
    .code-section {
      padding: 5rem 1.5rem;
      max-width: 56rem;
      margin: 0 auto;
    }

    .code-block {
      background: #0d1117;
      border: 1px solid var(--color-border);
      border-radius: 0.75rem;
      overflow: hidden;
      margin-bottom: 1.5rem;
      box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    }

    .code-block pre {
      margin: 0;
      padding: 1.5rem 1.75rem;
      overflow-x: auto;
    }

    .code-block code {
      font-family: var(--font-mono);
      font-size: 0.875rem;
      line-height: 1.7;
      color: #e6edf3;
    }

    .code-cta {
      display: inline-flex;
      align-items: center;
      color: var(--color-primary);
      font-weight: 600;
      text-decoration: none;
      transition: color 0.2s ease;
    }

    .code-cta:hover {
      color: var(--color-primary-light);
    }

    /* Footer */
    .footer {
      padding: 2.5rem 1.5rem;
      border-top: 1px solid var(--color-border);
    }

    .footer-content {
      max-width: 72rem;
      margin: 0 auto;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
    }

    .footer-copyright {
      font-size: 0.875rem;
      color: var(--color-text-muted);
      margin: 0;
    }

    .footer-links {
      display: flex;
      gap: 1.5rem;
    }

    .footer-links a {
      font-size: 0.875rem;
      color: var(--color-text-muted);
      text-decoration: none;
      transition: color 0.2s ease;
    }

    .footer-links a:hover {
      color: var(--color-primary);
    }

    @media (max-width: 640px) {
      .hero-cta {
        flex-direction: column;
      }

      .btn {
        width: 100%;
        justify-content: center;
      }
    }
  `,
})
export class LandingComponent {
  protected readonly codeExample = `use daimon::prelude::*;

#[tokio::main]
async fn main() -> Result<(), DaimonError> {
    let agent = Agent::builder()
        .name("assistant")
        .model(OpenAiModel::builder("gpt-4o").build()?)
        .system_prompt("You are a helpful assistant.")
        .build()?;

    let response = agent.prompt("Hello, world!").await?;
    println!("{}", response.text());
    Ok(())
}`;
  protected readonly faGithub = faGithub;
  protected readonly faPuzzlePiece = faPuzzlePiece;
  protected readonly faBolt = faBolt;
  protected readonly faDatabase = faDatabase;
  protected readonly faUsers = faUsers;
  protected readonly faServer = faServer;
  protected readonly faPlug = faPlug;
  protected readonly currentYear = new Date().getFullYear();
}
