import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    loadComponent: () =>
      import('./pages/landing/landing').then((m) => m.LandingComponent),
  },
  {
    path: 'docs/:slug',
    loadComponent: () =>
      import('./pages/doc-page/doc-page').then((m) => m.DocPageComponent),
  },
  {
    path: '**',
    redirectTo: '',
  },
];
