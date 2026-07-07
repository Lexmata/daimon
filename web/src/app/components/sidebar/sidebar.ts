import { Component, input, output } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';
import type { IconDefinition } from '@fortawesome/fontawesome-svg-core';
import { FontAwesomeModule } from '@fortawesome/angular-fontawesome';
import {
  faRocket,
  faSitemap,
  faRobot,
  faWrench,
  faDiagramProject,
  faUsers,
  faBookOpen,
  faNetworkWired,
  faPlug,
  faGaugeHigh,
  faPuzzlePiece,
  faXmark,
} from '@fortawesome/free-solid-svg-icons';

interface NavSection {
  title: string;
  links: { label: string; path: string; icon: IconDefinition }[];
}

@Component({
  selector: 'app-sidebar',
  standalone: true,
  imports: [FontAwesomeModule, RouterLink, RouterLinkActive],
  template: `
    <aside
      class="fixed inset-y-0 left-0 z-40 w-64 flex flex-col bg-surface-light border-r border-border transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:z-auto"
      [class.translate-x-0]="isOpen()"
      [class.-translate-x-full]="!isOpen()"
    >
      <div class="flex h-16 items-center justify-between px-6 border-b border-border lg:justify-center">
        <a routerLink="/" class="text-xl font-bold text-text hover:text-primary transition-colors">
          Daimon
        </a>
        <button
          type="button"
          class="lg:hidden p-2 rounded-md text-text-muted hover:text-text hover:bg-surface-lighter transition-colors"
          (click)="closeSidebar.emit()"
          aria-label="Close sidebar"
        >
          <fa-icon [icon]="faXmark" />
        </button>
      </div>

      <nav class="flex-1 overflow-y-auto py-4 px-3">
        @for (section of navSections; track section.title) {
          <div class="mb-6">
            <h3 class="px-3 mb-2 text-xs font-semibold uppercase tracking-wider text-text-muted">
              {{ section.title }}
            </h3>
            <ul class="space-y-1">
              @for (link of section.links; track link.path) {
                <li>
                  <a
                    [routerLink]="link.path"
                    routerLinkActive="bg-surface-lighter text-primary border-l-2 border-primary"
                    [routerLinkActiveOptions]="{ exact: false }"
                    class="flex items-center gap-3 px-3 py-2 rounded-md text-text hover:bg-surface-lighter hover:text-primary border-l-2 border-transparent transition-colors"
                    (click)="closeSidebar.emit()"
                  >
                    <fa-icon [icon]="link.icon" class="w-4 h-4 shrink-0" />
                    <span>{{ link.label }}</span>
                  </a>
                </li>
              }
            </ul>
          </div>
        }
      </nav>
    </aside>

    @if (isOpen()) {
      <div
        class="fixed inset-0 z-30 bg-black/50 lg:hidden"
        (click)="closeSidebar.emit()"
        aria-hidden="true"
      ></div>
    }
  `,
  styles: [
    `
      :host {
        display: contents;
      }
    `,
  ],
})
export class SidebarComponent {
  isOpen = input<boolean>(false);
  closeSidebar = output<void>();

  protected readonly faXmark = faXmark;
  protected readonly navSections: NavSection[] = [
    {
      title: 'Getting Started',
      links: [
        { label: 'Getting Started', path: '/docs/getting-started', icon: faRocket },
        { label: 'Architecture', path: '/docs/architecture', icon: faSitemap },
      ],
    },
    {
      title: 'Core Concepts',
      links: [
        { label: 'Agents', path: '/docs/agents', icon: faRobot },
        { label: 'Tools', path: '/docs/tools', icon: faWrench },
        { label: 'Orchestration', path: '/docs/orchestration', icon: faDiagramProject },
      ],
    },
    {
      title: 'Advanced',
      links: [
        { label: 'Multi-Agent', path: '/docs/multi-agent', icon: faUsers },
        { label: 'RAG', path: '/docs/rag', icon: faBookOpen },
        { label: 'Distributed', path: '/docs/distributed', icon: faNetworkWired },
      ],
    },
    {
      title: 'Reference',
      links: [
        { label: 'Providers', path: '/docs/providers', icon: faPlug },
        { label: 'Performance', path: '/docs/performance', icon: faGaugeHigh },
        { label: 'Plugin Development', path: '/docs/plugin-development', icon: faPuzzlePiece },
      ],
    },
  ];
}
