import { Component, signal } from '@angular/core';
import { RouterOutlet, Router, NavigationEnd } from '@angular/router';
import { HeaderComponent } from './components/header/header';
import { SidebarComponent } from './components/sidebar/sidebar';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, HeaderComponent, SidebarComponent],
  template: `
    @if (isDocPage()) {
      <div class="flex min-h-screen">
        <app-sidebar [isOpen]="sidebarOpen()" (closeSidebar)="sidebarOpen.set(false)" />
        <div class="flex-1 flex flex-col lg:ml-0">
          <app-header [transparent]="false" [showMenuToggle]="true" (toggleSidebar)="sidebarOpen.set(!sidebarOpen())" />
          <main class="flex-1">
            <router-outlet />
          </main>
        </div>
      </div>
    } @else {
      <app-header [transparent]="true" [showMenuToggle]="false" />
      <router-outlet />
    }
  `,
  styles: `
    :host {
      display: block;
    }
  `,
})
export class App {
  protected readonly sidebarOpen = signal(false);
  protected readonly isDocPage = signal(false);

  constructor(private router: Router) {
    this.router.events.subscribe((event) => {
      if (event instanceof NavigationEnd) {
        this.isDocPage.set(event.urlAfterRedirects.startsWith('/docs'));
        this.sidebarOpen.set(false);
      }
    });
  }
}
