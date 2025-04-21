import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet } from '@angular/router';
import { ToolsComponent } from './tools/tools.component';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatIconModule } from '@angular/material/icon';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    RouterOutlet,
    ToolsComponent,
    MatToolbarModule,
    MatIconModule
  ],
  template: `
    <mat-toolbar color="primary" class="app-toolbar">
      <mat-icon>build</mat-icon>
      <span class="title">Agent Tools Registry</span>
    </mat-toolbar>
    <main class="content">
      <router-outlet></router-outlet>
    </main>
  `,
  styles: [`
    :host {
      display: block;
      height: 100vh;
    }
    .app-toolbar {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      z-index: 1000;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .title {
      margin-left: 12px;
      font-weight: 400;
    }
    .content {
      padding: 84px 20px 20px;
      max-width: 1200px;
      margin: 0 auto;
    }
  `]
})
export class AppComponent {}
