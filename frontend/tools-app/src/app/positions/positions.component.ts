import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatTableModule } from '@angular/material/table';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatCardModule } from '@angular/material/card';
import { MatChipsModule } from '@angular/material/chips';
import { MatExpansionModule } from '@angular/material/expansion';
import { ApiService, Tool } from '../services/api.service';

@Component({
  selector: 'app-tools',
  standalone: true,
  imports: [
    CommonModule,
    MatTableModule,
    MatButtonModule,
    MatIconModule,
    MatProgressSpinnerModule,
    MatCardModule,
    MatChipsModule,
    MatExpansionModule
  ],
  template: `
    <div class="tools-container">
      <mat-card>
        <mat-card-header>
          <mat-card-title>Available Tools</mat-card-title>
          <mat-card-subtitle>List of supported API tools and their specifications</mat-card-subtitle>
        </mat-card-header>
        
        <mat-card-content>
          <div *ngIf="loading" class="loading-spinner">
            <mat-spinner diameter="40"></mat-spinner>
          </div>

          <div *ngIf="error" class="error-message">
            <mat-icon color="warn">error</mat-icon>
            <span>{{ error }}</span>
            <button mat-button color="primary" (click)="fetchTools()">Retry</button>
          </div>

          <div *ngIf="!loading && !error" class="tools-list">
            <mat-accordion>
              <mat-expansion-panel *ngFor="let tool of tools">
                <mat-expansion-panel-header>
                  <mat-panel-title>
                    {{ tool.name }}
                  </mat-panel-title>
                  <mat-panel-description>
                    {{ tool.description }}
                  </mat-panel-description>
                </mat-expansion-panel-header>

                <div class="tool-details">
                  <div class="tool-section">
                    <h3>Tags</h3>
                    <mat-chip-set>
                      <mat-chip *ngFor="let tag of tool.tags">{{ tag }}</mat-chip>
                    </mat-chip-set>
                  </div>

                  <div class="tool-section">
                    <h3>Endpoint</h3>
                    <code>{{ tool.endpoint }}</code>
                  </div>

                  <div class="tool-section">
                    <h3>Inputs</h3>
                    <pre>{{ tool.inputs | json }}</pre>
                  </div>

                  <div class="tool-section">
                    <h3>Outputs</h3>
                    <pre>{{ tool.outputs | json }}</pre>
                  </div>
                </div>
              </mat-expansion-panel>
            </mat-accordion>
          </div>
        </mat-card-content>

        <mat-card-actions align="end">
          <button mat-button color="primary" (click)="fetchTools()" [disabled]="loading">
            <mat-icon>refresh</mat-icon>
            Refresh
          </button>
        </mat-card-actions>
      </mat-card>
    </div>
  `,
  styles: [`
    .tools-container {
      margin: 0 auto;
    }

    .tool-details {
      padding: 16px;
    }

    .tool-section {
      margin-bottom: 24px;
    }

    .tool-section h3 {
      margin: 0 0 8px 0;
      font-size: 16px;
      font-weight: 500;
      color: rgba(0, 0, 0, 0.87);
    }

    .tool-section pre {
      background: #f5f5f5;
      padding: 12px;
      border-radius: 4px;
      overflow-x: auto;
      margin: 0;
    }

    code {
      background: #f5f5f5;
      padding: 4px 8px;
      border-radius: 4px;
      font-family: 'Roboto Mono', monospace;
    }

    mat-expansion-panel {
      margin-bottom: 16px;
    }

    mat-panel-title {
      font-weight: 500;
    }

    mat-panel-description {
      font-size: 14px;
    }

    mat-chip-set {
      margin-bottom: 8px;
    }
  `]
})
export class ToolsComponent implements OnInit {
  tools: Tool[] = [];
  loading = true;
  error: string | null = null;

  constructor(private apiService: ApiService) {}

  ngOnInit(): void {
    this.fetchTools();
  }

  fetchTools(): void {
    this.loading = true;
    this.error = null;
    
    this.apiService.getTools().subscribe({
      next: (tools) => {
        this.tools = tools;
        this.loading = false;
      },
      error: (err) => {
        this.error = 'Failed to fetch tools. Please try again later.';
        this.loading = false;
        console.error('Error fetching tools:', err);
      }
    });
  }
}
