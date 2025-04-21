import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatTableModule } from '@angular/material/table';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatCardModule } from '@angular/material/card';
import { MatChipsModule } from '@angular/material/chips';
import { MatExpansionModule } from '@angular/material/expansion';
import { ApiService, Tool, ToolListResponse } from '../services/api.service';
import { catchError } from 'rxjs/operators';
import { of } from 'rxjs';

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
  templateUrl: './tools.component.html',
  styleUrls: ['./tools.component.css']
})
export class ToolsComponent implements OnInit {
  tools: Tool[] = [];
  loading = true;
  error: string | null = null;

  constructor(private apiService: ApiService) {
    console.log('API URL:', apiService['apiUrl']); // Debug log
  }

  ngOnInit(): void {
    console.log('ToolsComponent initialized');
    this.fetchTools();
  }

  fetchTools(): void {
    this.loading = true;
    this.error = null;
    console.log('Fetching tools...');
    
    this.apiService.getTools().pipe(
      catchError(err => {
        console.error('Error details:', err);
        this.error = `Failed to fetch tools: ${err.message || 'Unknown error'}`;
        return of({ tools: [] } as ToolListResponse);
      })
    ).subscribe({
      next: (response) => {
        console.log('Tools response:', response);
        this.tools = response.tools;
        this.loading = false;
      },
      error: (err) => {
        console.error('Error fetching tools:', err);
        this.error = `Failed to fetch tools: ${err.message || 'Unknown error'}`;
        this.loading = false;
      }
    });
  }

  refreshTools(): void {
    this.loading = true;
    this.error = null;
    console.log('Refreshing tools...');
    
    this.apiService.refreshTools().pipe(
      catchError(err => {
        console.error('Error refreshing tools:', err);
        this.error = `Failed to refresh tools: ${err.message || 'Unknown error'}`;
        return of({ tools: [] } as ToolListResponse);
      })
    ).subscribe({
      next: (response) => {
        console.log('Tools refreshed:', response);
        this.tools = response.tools;
        this.loading = false;
      },
      error: (err) => {
        console.error('Error refreshing tools:', err);
        this.error = `Failed to refresh tools: ${err.message || 'Unknown error'}`;
        this.loading = false;
      }
    });
  }
}
