import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    loadComponent: () =>
      import('./pages/schedule-page/schedule-page.component').then(
        (m) => m.SchedulePageComponent
      ),
  },
  { path: '**', redirectTo: '' },
];
