import { Component, output } from '@angular/core';
import { RouterLink } from '@angular/router';
import { FontAwesomeModule } from '@fortawesome/angular-fontawesome';
import { faBars, faCube } from '@fortawesome/free-solid-svg-icons';
import { faGithub } from '@fortawesome/free-brands-svg-icons';

@Component({
  selector: 'app-header',
  standalone: true,
  imports: [FontAwesomeModule, RouterLink],
  template: `
    <header
      class="sticky top-0 z-50 flex h-16 items-center justify-between border-b border-border bg-surface px-4 lg:px-6"
    >
      <div class="flex items-center gap-4">
        <button
          type="button"
          class="lg:hidden p-2 rounded-md text-text-muted hover:text-text hover:bg-surface-light transition-colors"
          (click)="toggleSidebar.emit()"
          aria-label="Toggle sidebar"
        >
          <fa-icon [icon]="faBars" />
        </button>
        <a
          routerLink="/"
          class="text-xl font-bold text-text hover:text-primary transition-colors"
        >
          Daimon
        </a>
      </div>

      <div class="flex items-center gap-2">
        <a
          href="https://github.com/lexmata/daimon"
          target="_blank"
          rel="noopener noreferrer"
          class="p-2 rounded-md text-text-muted hover:text-text hover:bg-surface-light transition-colors"
          aria-label="GitHub"
        >
          <fa-icon [icon]="faGithub" class="w-5 h-5" />
        </a>
        <a
          href="https://crates.io/crates/daimon"
          target="_blank"
          rel="noopener noreferrer"
          class="p-2 rounded-md text-text-muted hover:text-text hover:bg-surface-light transition-colors"
          aria-label="crates.io"
        >
          <fa-icon [icon]="faCube" class="w-5 h-5" />
        </a>
      </div>
    </header>
  `,
  styles: [
    `
      :host {
        display: block;
      }
    `,
  ],
})
export class HeaderComponent {
  toggleSidebar = output<void>();

  protected readonly faBars = faBars;
  protected readonly faGithub = faGithub;
  protected readonly faCube = faCube;
}
