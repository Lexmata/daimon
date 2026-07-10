import { Component, input, output } from '@angular/core';
import { RouterLink } from '@angular/router';
import { FaIconComponent } from '@fortawesome/angular-fontawesome';
import { faBars, faCube, faBook, faArrowRight } from '@fortawesome/free-solid-svg-icons';
import { faGithub, faRust } from '@fortawesome/free-brands-svg-icons';

@Component({
  selector: 'app-header',
  standalone: true,
  imports: [FaIconComponent, RouterLink],
  template: `
    <header
      class="header"
      [class.header--transparent]="transparent()"
      [class.header--solid]="!transparent()"
    >
      <div class="header-inner">
        <div class="header-left">
          @if (showMenuToggle()) {
            <button
              type="button"
              class="menu-toggle"
              (click)="toggleSidebar.emit()"
              aria-label="Toggle sidebar"
            >
              <fa-icon [icon]="faBars" />
            </button>
          }
          <a routerLink="/" class="brand">
            <span class="brand-icon">
              <fa-icon [icon]="faRust" />
            </span>
            <span class="brand-text">Daimon</span>
            <span class="version-badge">v0.16</span>
          </a>
        </div>

        <nav class="header-nav">
          <a routerLink="/docs/getting-started" class="nav-link">
            <fa-icon [icon]="faBook" class="nav-icon" />
            Docs
          </a>
          <a
            href="https://github.com/Lexmata/daimon"
            target="_blank"
            rel="noopener noreferrer"
            class="nav-link"
          >
            <fa-icon [icon]="faGithub" class="nav-icon" />
            GitHub
          </a>
          <a
            href="https://crates.io/crates/daimon"
            target="_blank"
            rel="noopener noreferrer"
            class="nav-link"
          >
            <fa-icon [icon]="faCube" class="nav-icon" />
            crates.io
          </a>
        </nav>

        <div class="header-right">
          <a routerLink="/docs/getting-started" class="cta-btn">
            Get Started
            <fa-icon [icon]="faArrowRight" class="cta-icon" />
          </a>
        </div>
      </div>
    </header>
  `,
  styles: `
    :host {
      display: block;
    }

    .header {
      position: sticky;
      top: 0;
      z-index: 50;
      height: 4rem;
      display: flex;
      align-items: center;
      transition: background-color 0.3s ease, border-color 0.3s ease, backdrop-filter 0.3s ease;
    }

    .header--solid {
      background-color: rgba(15, 23, 42, 0.95);
      backdrop-filter: blur(12px) saturate(180%);
      border-bottom: 1px solid var(--color-border);
    }

    .header--transparent {
      background-color: rgba(15, 23, 42, 0.6);
      backdrop-filter: blur(12px) saturate(180%);
      border-bottom: 1px solid rgba(51, 65, 85, 0.3);
    }

    .header-inner {
      display: flex;
      align-items: center;
      justify-content: space-between;
      width: 100%;
      max-width: 80rem;
      margin: 0 auto;
      padding: 0 1.5rem;
    }

    .header-left {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .menu-toggle {
      display: none;
      padding: 0.5rem;
      border-radius: 0.375rem;
      color: var(--color-text-muted);
      background: none;
      border: none;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .menu-toggle:hover {
      color: var(--color-text);
      background: var(--color-surface-light);
    }

    @media (max-width: 1023px) {
      .menu-toggle {
        display: flex;
      }
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      text-decoration: none;
      transition: opacity 0.2s ease;
    }

    .brand:hover {
      opacity: 0.85;
    }

    .brand-icon {
      font-size: 1.5rem;
      color: var(--color-primary);
      filter: drop-shadow(0 0 6px rgba(249, 115, 22, 0.4));
    }

    .brand-text {
      font-size: 1.25rem;
      font-weight: 800;
      letter-spacing: -0.03em;
      color: white;
    }

    .version-badge {
      font-size: 0.625rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      color: var(--color-primary);
      background: rgba(249, 115, 22, 0.12);
      border: 1px solid rgba(249, 115, 22, 0.25);
      padding: 0.125rem 0.375rem;
      border-radius: 9999px;
      line-height: 1.4;
    }

    .header-nav {
      display: flex;
      align-items: center;
      gap: 0.25rem;
    }

    @media (max-width: 639px) {
      .header-nav {
        display: none;
      }
    }

    .nav-link {
      display: flex;
      align-items: center;
      gap: 0.375rem;
      padding: 0.5rem 0.75rem;
      font-size: 0.875rem;
      font-weight: 500;
      color: var(--color-text-muted);
      text-decoration: none;
      border-radius: 0.375rem;
      transition: all 0.2s ease;
    }

    .nav-link:hover {
      color: var(--color-text);
      background: rgba(255, 255, 255, 0.05);
    }

    .nav-icon {
      font-size: 0.875rem;
    }

    .header-right {
      display: flex;
      align-items: center;
    }

    .cta-btn {
      display: inline-flex;
      align-items: center;
      gap: 0.375rem;
      padding: 0.5rem 1rem;
      font-size: 0.8125rem;
      font-weight: 600;
      color: white;
      background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-primary-dark) 100%);
      border-radius: 0.5rem;
      text-decoration: none;
      transition: all 0.25s ease;
      box-shadow: 0 2px 8px rgba(249, 115, 22, 0.25);
    }

    .cta-btn:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 16px rgba(249, 115, 22, 0.4);
    }

    .cta-icon {
      font-size: 0.75rem;
      transition: transform 0.2s ease;
    }

    .cta-btn:hover .cta-icon {
      transform: translateX(2px);
    }

    @media (max-width: 639px) {
      .cta-btn {
        display: none;
      }
    }
  `,
})
export class HeaderComponent {
  transparent = input<boolean>(false);
  showMenuToggle = input<boolean>(false);
  toggleSidebar = output<void>();

  protected readonly faBars = faBars;
  protected readonly faGithub = faGithub;
  protected readonly faRust = faRust;
  protected readonly faCube = faCube;
  protected readonly faBook = faBook;
  protected readonly faArrowRight = faArrowRight;
}
