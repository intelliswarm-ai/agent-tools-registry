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

  constructor(private apiService: ApiService) {}

  ngOnInit(): void {
    console.log('ToolsComponent initialized'); // Debug log
    this.fetchTools();
  }

  fetchTools(): void {
    this.loading = true;
    this.error = null;
    console.log('Fetching tools...'); // Debug log
    
    this.apiService.getTools().pipe(
      catchError(err => {
        console.error('Error details:', err); // Detailed error logging
        return of([]); // Return empty array on error
      })
    ).subscribe({
      next: (tools) => {
        console.log('Tools received:', tools); // Debug log
        this.tools = tools;
        this.loading = false;
      },
      error: (err) => {
        console.error('Error fetching tools:', err);
        this.error = `Failed to fetch tools: ${err.message || 'Unknown error'}`;
        this.loading = false;
      }
    });
  }
}
