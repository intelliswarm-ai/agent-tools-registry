import { Routes } from '@angular/router';
import { ToolsComponent } from './tools/tools.component';

export const routes: Routes = [
  { path: '', component: ToolsComponent },
  { path: '**', redirectTo: '' }
];
