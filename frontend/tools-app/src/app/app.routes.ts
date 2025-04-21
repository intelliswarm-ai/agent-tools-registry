import { Routes } from '@angular/router';
import { ToolsComponent } from './tools/tools.component';
import { DemoComponent } from './demo/demo.component';

export const routes: Routes = [
  { path: '', component: ToolsComponent },
  { path: 'demo', component: DemoComponent },
  { path: '**', redirectTo: '' }
];
