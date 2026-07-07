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
  faArrowRight,
  faGauge,
  faCubes,
  faCloud,
  faShieldHalved,
} from '@fortawesome/free-solid-svg-icons';
import { faGithub, faRust } from '@fortawesome/free-brands-svg-icons';

@Component({
  selector: 'app-landing',
  imports: [RouterLink, FaIconComponent],
  template: `
    <div class="landing">
      <!-- Hero Section -->
      <section class="hero">
        <div class="hero-bg">
          <div class="orb orb-1"></div>
          <div class="orb orb-2"></div>
          <div class="orb orb-3"></div>
          <div class="grid-overlay"></div>
          <div class="noise-overlay"></div>
        </div>
        <div class="hero-content">
          <div class="hero-badge">
            <fa-icon [icon]="faRust" class="hero-badge-icon" />
            <span>Open Source Rust Framework</span>
            <span class="hero-badge-version">v0.16</span>
          </div>
          <h1 class="hero-title">
            <span class="hero-title-line">Build AI Agents</span>
            <span class="hero-title-accent">That Actually Ship</span>
          </h1>
          <p class="hero-description">
            Daimon is a <strong>high-performance Rust framework</strong> for AI agents.
            Trait-based plugins, zero-cost abstractions, async streaming, RAG,
            distributed execution, and multi-agent orchestration -- all compile-time safe.
          </p>
          <div class="hero-cta">
            <a routerLink="/docs/getting-started" class="btn btn-primary">
              Get Started
              <fa-icon [icon]="faArrowRight" class="btn-arrow" />
            </a>
            <a
              href="https://github.com/Lexmata/daimon"
              target="_blank"
              rel="noopener noreferrer"
              class="btn btn-ghost"
            >
              <fa-icon [icon]="faGithub" />
              Star on GitHub
            </a>
          </div>
          <div class="hero-stats">
            <div class="stat">
              <span class="stat-value">8</span>
              <span class="stat-label">Crates</span>
            </div>
            <div class="stat-divider"></div>
            <div class="stat">
              <span class="stat-value">6</span>
              <span class="stat-label">LLM Providers</span>
            </div>
            <div class="stat-divider"></div>
            <div class="stat">
              <span class="stat-value">5</span>
              <span class="stat-label">Task Brokers</span>
            </div>
            <div class="stat-divider"></div>
            <div class="stat">
              <span class="stat-value">3</span>
              <span class="stat-label">Vector Stores</span>
            </div>
          </div>
        </div>
      </section>

      <!-- Code Example -->
      <section class="code-showcase">
        <div class="code-showcase-inner">
          <div class="code-info">
            <h2 class="section-eyebrow">Dead Simple API</h2>
            <h3 class="section-heading">From zero to agent<br>in under 15 lines</h3>
            <p class="section-text">
              No boilerplate. No ceremony. Build an agent, give it tools,
              and start prompting. Streaming, memory, and hooks are all opt-in.
            </p>
            <a routerLink="/docs/getting-started" class="link-arrow">
              Read the full guide
              <fa-icon [icon]="faArrowRight" />
            </a>
          </div>
          <div class="code-window">
            <div class="code-titlebar">
              <div class="code-dot code-dot--red"></div>
              <div class="code-dot code-dot--yellow"></div>
              <div class="code-dot code-dot--green"></div>
              <span class="code-filename">main.rs</span>
            </div>
            <pre class="code-body"><code [innerHTML]="highlightedCode"></code></pre>
          </div>
        </div>
      </section>

      <!-- Feature Grid -->
      <section class="features">
        <div class="features-inner">
          <h2 class="section-eyebrow section-eyebrow--center">Capabilities</h2>
          <h3 class="section-heading section-heading--center">
            Everything you need for production AI
          </h3>
          <div class="feature-grid">
            @for (feature of features; track feature.title) {
              <article class="feature-card" [attr.data-index]="$index">
                <div class="feature-glow"></div>
                <div class="feature-content">
                  <div class="feature-icon-wrap">
                    <fa-icon [icon]="feature.icon" />
                  </div>
                  <h4 class="feature-title">{{ feature.title }}</h4>
                  <p class="feature-desc">{{ feature.desc }}</p>
                </div>
              </article>
            }
          </div>
        </div>
      </section>

      <!-- Providers Banner -->
      <section class="providers">
        <div class="providers-inner">
          <h2 class="section-eyebrow section-eyebrow--center">Integrations</h2>
          <h3 class="section-heading section-heading--center">
            Connect to any LLM provider
          </h3>
          <div class="provider-grid">
            @for (provider of providers; track provider) {
              <div class="provider-chip">{{ provider }}</div>
            }
          </div>
          <a routerLink="/docs/providers" class="link-arrow link-arrow--center">
            View all providers
            <fa-icon [icon]="faArrowRight" />
          </a>
        </div>
      </section>

      <!-- CTA Banner -->
      <section class="cta-banner">
        <div class="cta-banner-bg">
          <div class="cta-orb cta-orb-1"></div>
          <div class="cta-orb cta-orb-2"></div>
        </div>
        <div class="cta-banner-inner">
          <h2 class="cta-heading">Ready to build?</h2>
          <p class="cta-text">
            Add Daimon to your project and ship your first agent in minutes.
          </p>
          <div class="cta-actions">
            <div class="install-cmd">
              <code>cargo add daimon --features full</code>
            </div>
            <a routerLink="/docs/getting-started" class="btn btn-primary">
              Read the Docs
              <fa-icon [icon]="faArrowRight" class="btn-arrow" />
            </a>
          </div>
        </div>
      </section>

      <!-- Footer -->
      <footer class="footer">
        <div class="footer-inner">
          <div class="footer-brand">
            <span class="footer-logo">
              <fa-icon [icon]="faRust" />
              Daimon
            </span>
            <p class="footer-tagline">High-performance AI agents in Rust.</p>
          </div>
          <div class="footer-links-group">
            <h4 class="footer-col-title">Resources</h4>
            <a routerLink="/docs/getting-started">Getting Started</a>
            <a routerLink="/docs/architecture">Architecture</a>
            <a routerLink="/docs/plugin-development">Plugin Guide</a>
          </div>
          <div class="footer-links-group">
            <h4 class="footer-col-title">Community</h4>
            <a href="https://github.com/Lexmata/daimon" target="_blank" rel="noopener noreferrer">GitHub</a>
            <a href="https://crates.io/crates/daimon" target="_blank" rel="noopener noreferrer">crates.io</a>
            <a href="https://docs.rs/daimon" target="_blank" rel="noopener noreferrer">docs.rs</a>
          </div>
          <div class="footer-bottom">
            <p>&copy; {{ currentYear }} Lexmata LLC. Apache 2.0 + MIT dual-licensed.</p>
          </div>
        </div>
      </footer>
    </div>
  `,
  styles: `
    :host { display: block; }
    .landing { background: var(--color-surface); min-height: 100vh; overflow-x: hidden; }

    /* ---- HERO ---- */
    .hero { position: relative; min-height: 92vh; display: flex; align-items: center; justify-content: center; overflow: hidden; }
    .hero-bg { position: absolute; inset: 0; z-index: 0; }

    .orb {
      position: absolute; border-radius: 50%; filter: blur(80px); opacity: 0.5;
      animation: float 20s ease-in-out infinite;
    }
    .orb-1 { width: 600px; height: 600px; top: -15%; left: -10%; background: radial-gradient(circle, rgba(249,115,22,0.35), transparent 70%); animation-delay: 0s; }
    .orb-2 { width: 500px; height: 500px; bottom: -10%; right: -5%; background: radial-gradient(circle, rgba(139,92,246,0.2), transparent 70%); animation-delay: -7s; animation-duration: 25s; }
    .orb-3 { width: 400px; height: 400px; top: 30%; left: 50%; background: radial-gradient(circle, rgba(249,115,22,0.15), transparent 70%); animation-delay: -13s; animation-duration: 30s; }

    @keyframes float {
      0%, 100% { transform: translate(0, 0) scale(1); }
      33% { transform: translate(30px, -40px) scale(1.05); }
      66% { transform: translate(-20px, 20px) scale(0.95); }
    }

    .grid-overlay {
      position: absolute; inset: 0;
      background-image:
        linear-gradient(rgba(249,115,22,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(249,115,22,0.04) 1px, transparent 1px);
      background-size: 64px 64px;
      mask-image: radial-gradient(ellipse 70% 60% at 50% 50%, black, transparent);
    }

    .noise-overlay {
      position: absolute; inset: 0; opacity: 0.03;
      background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
    }

    .hero-content { position: relative; z-index: 1; max-width: 52rem; margin: 0 auto; padding: 2rem 1.5rem; text-align: center; }

    .hero-badge {
      display: inline-flex; align-items: center; gap: 0.5rem;
      padding: 0.375rem 0.875rem; border-radius: 9999px;
      font-size: 0.8125rem; font-weight: 500; color: var(--color-text-muted);
      background: rgba(249,115,22,0.08); border: 1px solid rgba(249,115,22,0.2);
      margin-bottom: 1.5rem;
      animation: fadeInUp 0.6s ease-out both;
    }
    .hero-badge-icon { color: var(--color-primary); font-size: 0.9rem; }
    .hero-badge-version {
      font-size: 0.6875rem; font-weight: 700; color: var(--color-primary);
      background: rgba(249,115,22,0.15); padding: 0.0625rem 0.375rem; border-radius: 9999px;
    }

    .hero-title {
      margin: 0 0 1.25rem; line-height: 1.05; letter-spacing: -0.045em;
      animation: fadeInUp 0.6s ease-out 0.1s both;
    }
    .hero-title-line {
      display: block; font-size: clamp(2.75rem, 8vw, 4.5rem); font-weight: 900; color: white;
    }
    .hero-title-accent {
      display: block; font-size: clamp(2.75rem, 8vw, 4.5rem); font-weight: 900;
      background: linear-gradient(135deg, var(--color-primary) 0%, #fbbf24 50%, var(--color-primary-light) 100%);
      background-size: 200% 200%;
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      background-clip: text;
      animation: shimmerText 5s ease-in-out infinite, fadeInUp 0.6s ease-out 0.1s both;
    }

    @keyframes shimmerText {
      0%, 100% { background-position: 0% 50%; }
      50% { background-position: 100% 50%; }
    }

    .hero-description {
      font-size: 1.125rem; line-height: 1.7; color: var(--color-text-muted);
      max-width: 38rem; margin: 0 auto 2rem;
      animation: fadeInUp 0.6s ease-out 0.2s both;
    }
    .hero-description strong { color: var(--color-text); }

    .hero-cta {
      display: flex; flex-wrap: wrap; gap: 0.75rem; justify-content: center;
      animation: fadeInUp 0.6s ease-out 0.3s both;
    }

    .btn {
      display: inline-flex; align-items: center; gap: 0.5rem;
      padding: 0.75rem 1.5rem; font-size: 0.9375rem; font-weight: 600;
      border-radius: 0.625rem; text-decoration: none; cursor: pointer;
      transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1); border: none;
    }
    .btn-primary {
      color: white;
      background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-primary-dark) 100%);
      box-shadow: 0 4px 16px rgba(249,115,22,0.3), inset 0 1px 0 rgba(255,255,255,0.1);
    }
    .btn-primary:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 30px rgba(249,115,22,0.45), inset 0 1px 0 rgba(255,255,255,0.1);
    }
    .btn-ghost {
      color: var(--color-text); background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.1);
    }
    .btn-ghost:hover { background: rgba(255,255,255,0.1); border-color: rgba(255,255,255,0.2); }
    .btn-arrow { font-size: 0.8rem; transition: transform 0.2s ease; }
    .btn:hover .btn-arrow { transform: translateX(3px); }

    .hero-stats {
      display: flex; align-items: center; justify-content: center; gap: 1.5rem;
      margin-top: 3rem; animation: fadeInUp 0.6s ease-out 0.4s both;
    }
    .stat { text-align: center; }
    .stat-value { display: block; font-size: 1.75rem; font-weight: 800; color: white; letter-spacing: -0.03em; }
    .stat-label { font-size: 0.75rem; font-weight: 500; color: var(--color-text-muted); text-transform: uppercase; letter-spacing: 0.05em; }
    .stat-divider { width: 1px; height: 2.5rem; background: var(--color-border); }

    @keyframes fadeInUp {
      from { opacity: 0; transform: translateY(20px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @media (max-width: 640px) {
      .hero-stats { gap: 1rem; flex-wrap: wrap; }
      .stat-divider { display: none; }
      .hero-cta { flex-direction: column; }
      .btn { width: 100%; justify-content: center; }
    }

    /* ---- CODE SHOWCASE ---- */
    .code-showcase {
      position: relative; padding: 6rem 1.5rem;
      background: linear-gradient(180deg, var(--color-surface) 0%, rgba(15,23,42,0.95) 100%);
    }
    .code-showcase-inner {
      max-width: 72rem; margin: 0 auto;
      display: grid; grid-template-columns: 1fr 1.2fr; gap: 4rem; align-items: center;
    }
    @media (max-width: 768px) {
      .code-showcase-inner { grid-template-columns: 1fr; gap: 2rem; }
    }

    .section-eyebrow {
      font-size: 0.8125rem; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.1em; color: var(--color-primary); margin: 0 0 0.75rem;
    }
    .section-eyebrow--center { text-align: center; }

    .section-heading {
      font-size: clamp(1.75rem, 4vw, 2.25rem); font-weight: 800;
      color: white; letter-spacing: -0.03em; line-height: 1.15; margin: 0 0 1rem;
    }
    .section-heading--center { text-align: center; }

    .section-text { font-size: 1rem; line-height: 1.7; color: var(--color-text-muted); margin: 0 0 1.5rem; }

    .link-arrow {
      display: inline-flex; align-items: center; gap: 0.375rem;
      font-size: 0.9375rem; font-weight: 600; color: var(--color-primary);
      text-decoration: none; transition: gap 0.2s ease;
    }
    .link-arrow:hover { gap: 0.625rem; color: var(--color-primary-light); }
    .link-arrow--center { display: flex; justify-content: center; margin-top: 2rem; }

    .code-window {
      border-radius: 0.75rem; overflow: hidden;
      border: 1px solid rgba(249,115,22,0.15);
      box-shadow: 0 8px 40px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.03), 0 0 60px rgba(249,115,22,0.08);
    }
    .code-titlebar {
      display: flex; align-items: center; gap: 0.5rem;
      padding: 0.75rem 1rem;
      background: rgba(30,41,59,0.8); border-bottom: 1px solid var(--color-border);
    }
    .code-dot { width: 12px; height: 12px; border-radius: 50%; }
    .code-dot--red { background: #ff5f57; }
    .code-dot--yellow { background: #febc2e; }
    .code-dot--green { background: #28c840; }
    .code-filename { margin-left: auto; font-size: 0.75rem; color: var(--color-text-muted); font-family: var(--font-mono); }

    .code-body {
      margin: 0; padding: 1.25rem 1.5rem; overflow-x: auto;
      background: #0d1117; font-size: 0.8125rem; line-height: 1.8;
    }
    .code-body code { font-family: var(--font-mono); color: #e6edf3; }

    /* ---- FEATURES ---- */
    .features { padding: 6rem 1.5rem; }
    .features-inner { max-width: 72rem; margin: 0 auto; }

    .feature-grid {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      gap: 1.25rem; margin-top: 3rem;
    }

    .feature-card {
      position: relative; border-radius: 0.875rem; overflow: hidden;
      background: var(--color-surface-light); border: 1px solid var(--color-border);
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .feature-card:hover {
      border-color: rgba(249,115,22,0.3);
      transform: translateY(-4px);
      box-shadow: 0 12px 40px rgba(0,0,0,0.3);
    }

    .feature-glow {
      position: absolute; top: -1px; left: 50%; width: 60%; height: 2px;
      transform: translateX(-50%); opacity: 0;
      background: linear-gradient(90deg, transparent, var(--color-primary), transparent);
      transition: opacity 0.3s ease;
    }
    .feature-card:hover .feature-glow { opacity: 1; }

    .feature-content { padding: 1.75rem; }
    .feature-icon-wrap {
      width: 2.75rem; height: 2.75rem;
      display: flex; align-items: center; justify-content: center;
      border-radius: 0.625rem; margin-bottom: 1rem;
      font-size: 1.125rem; color: var(--color-primary);
      background: rgba(249,115,22,0.1); border: 1px solid rgba(249,115,22,0.15);
    }
    .feature-title { font-size: 1.0625rem; font-weight: 700; color: white; margin: 0 0 0.5rem; }
    .feature-desc { font-size: 0.875rem; line-height: 1.6; color: var(--color-text-muted); margin: 0; }

    /* ---- PROVIDERS ---- */
    .providers { padding: 5rem 1.5rem; background: var(--color-surface-light); }
    .providers-inner { max-width: 56rem; margin: 0 auto; }
    .provider-grid {
      display: flex; flex-wrap: wrap; justify-content: center; gap: 0.75rem; margin-top: 2rem;
    }
    .provider-chip {
      padding: 0.625rem 1.25rem; border-radius: 9999px;
      font-size: 0.875rem; font-weight: 600; color: var(--color-text);
      background: var(--color-surface); border: 1px solid var(--color-border);
      transition: all 0.2s ease;
    }
    .provider-chip:hover {
      border-color: var(--color-primary);
      color: var(--color-primary);
      box-shadow: 0 0 12px rgba(249,115,22,0.15);
    }

    /* ---- CTA BANNER ---- */
    .cta-banner {
      position: relative; padding: 5rem 1.5rem; overflow: hidden;
    }
    .cta-banner-bg { position: absolute; inset: 0; }
    .cta-orb {
      position: absolute; border-radius: 50%; filter: blur(100px); opacity: 0.3;
    }
    .cta-orb-1 { width: 400px; height: 400px; top: -30%; left: 10%; background: radial-gradient(circle, rgba(249,115,22,0.4), transparent 70%); }
    .cta-orb-2 { width: 300px; height: 300px; bottom: -20%; right: 15%; background: radial-gradient(circle, rgba(139,92,246,0.25), transparent 70%); }

    .cta-banner-inner {
      position: relative; z-index: 1;
      max-width: 40rem; margin: 0 auto; text-align: center;
    }
    .cta-heading { font-size: 2.25rem; font-weight: 800; color: white; margin: 0 0 0.75rem; letter-spacing: -0.03em; }
    .cta-text { font-size: 1.0625rem; color: var(--color-text-muted); margin: 0 0 2rem; }
    .cta-actions { display: flex; flex-wrap: wrap; gap: 1rem; justify-content: center; align-items: center; }

    .install-cmd {
      padding: 0.625rem 1.25rem; border-radius: 0.625rem;
      background: rgba(13,17,23,0.8); border: 1px solid var(--color-border);
    }
    .install-cmd code {
      font-family: var(--font-mono); font-size: 0.875rem; color: var(--color-primary-light);
    }

    /* ---- FOOTER ---- */
    .footer { padding: 4rem 1.5rem 2rem; border-top: 1px solid var(--color-border); }
    .footer-inner {
      max-width: 72rem; margin: 0 auto;
      display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 3rem;
    }
    @media (max-width: 768px) {
      .footer-inner { grid-template-columns: 1fr; gap: 2rem; }
    }
    .footer-brand {}
    .footer-logo {
      display: flex; align-items: center; gap: 0.5rem;
      font-size: 1.25rem; font-weight: 800; color: white; margin-bottom: 0.5rem;
    }
    .footer-logo fa-icon { color: var(--color-primary); }
    .footer-tagline { font-size: 0.875rem; color: var(--color-text-muted); margin: 0; }

    .footer-links-group { display: flex; flex-direction: column; gap: 0.5rem; }
    .footer-col-title { font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: var(--color-text-muted); margin: 0 0 0.25rem; }
    .footer-links-group a {
      font-size: 0.875rem; color: var(--color-text-muted); text-decoration: none;
      transition: color 0.2s ease;
    }
    .footer-links-group a:hover { color: var(--color-primary); }

    .footer-bottom {
      grid-column: 1 / -1; padding-top: 2rem; margin-top: 1rem;
      border-top: 1px solid var(--color-border);
    }
    .footer-bottom p { font-size: 0.8125rem; color: var(--color-text-muted); margin: 0; }
  `,
})
export class LandingComponent {
  protected readonly faGithub = faGithub;
  protected readonly faRust = faRust;
  protected readonly faArrowRight = faArrowRight;
  protected readonly currentYear = new Date().getFullYear();

  protected readonly features = [
    { icon: faPuzzlePiece, title: 'Trait-Based Plugins', desc: 'Add LLM providers, vector stores, and task brokers by implementing traits. No lock-in, full composability.' },
    { icon: faBolt, title: 'Async Streaming', desc: 'Real-time token streaming with async/await. First-class cancellation, backpressure, and zero-copy design.' },
    { icon: faDatabase, title: 'RAG & Vector Stores', desc: 'Built-in retrieval with pgvector, OpenSearch, Qdrant. Semantic search and document ingestion out of the box.' },
    { icon: faUsers, title: 'Multi-Agent Orchestration', desc: 'Supervisor, handoff, agent-as-tool, fork patterns. Chain, Graph, and DAG workflows.' },
    { icon: faServer, title: 'Distributed Execution', desc: 'Task brokers for Redis, NATS, RabbitMQ, SQS, Pub/Sub, and Service Bus. Scale horizontally.' },
    { icon: faPlug, title: 'MCP & A2A Protocols', desc: 'Model Context Protocol client/server and Agent-to-Agent protocol. Full ecosystem interop.' },
    { icon: faShieldHalved, title: 'Guardrails & Middleware', desc: 'Input/output validation, content policies, regex filters. Composable middleware pipeline.' },
    { icon: faGauge, title: 'Cost Tracking & Budgets', desc: 'Per-token cost tracking, budget limits, streaming cost events. Know exactly what you spend.' },
    { icon: faCubes, title: 'Checkpointing & Replay', desc: 'Save and resume agent runs. Time-travel debugging. Fork agents from any checkpoint.' },
  ];

  protected readonly providers = [
    'OpenAI', 'Anthropic', 'AWS Bedrock', 'Google Gemini', 'Azure OpenAI', 'Ollama',
  ];

  protected readonly highlightedCode = `<span style="color:#ff7b72">use</span> <span style="color:#ffa657">daimon</span>::<span style="color:#ffa657">prelude</span>::*;

<span style="color:#d2a8ff">#[tokio::main]</span>
<span style="color:#ff7b72">async fn</span> <span style="color:#d2a8ff">main</span>() -> <span style="color:#ffa657">Result</span>&lt;(), <span style="color:#ffa657">DaimonError</span>&gt; {
    <span style="color:#ff7b72">let</span> agent = <span style="color:#ffa657">Agent</span>::builder()
        .name(<span style="color:#a5d6ff">"assistant"</span>)
        .model(<span style="color:#ffa657">OpenAiModel</span>::builder(<span style="color:#a5d6ff">"gpt-4o"</span>).build()?)
        .system_prompt(<span style="color:#a5d6ff">"You are a helpful assistant."</span>)
        .build()?;

    <span style="color:#ff7b72">let</span> response = agent.prompt(<span style="color:#a5d6ff">"Hello, world!"</span>).<span style="color:#ff7b72">await</span>?;
    <span style="color:#ffa657">println!</span>(<span style="color:#a5d6ff">"{}"</span>, response.text());
    <span style="color:#ffa657">Ok</span>(())
}`;
}
