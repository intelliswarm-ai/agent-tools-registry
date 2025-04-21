import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';

interface Position {
  symbol: string;
  qty: number;
  avg_entry_price: number;
  market_value: number;
}

@Component({
  selector: 'app-positions',
  templateUrl: './positions.component.html',
  styleUrls: ['./positions.component.css']
})
export class PositionsComponent implements OnInit {
  positions: Position[] = [];
  loading = true;
  error: string | null = null;

  constructor(private http: HttpClient) {}

  ngOnInit(): void {
    this.fetchPositions();
  }

  fetchPositions(): void {
    this.loading = true;
    this.error = null;
    
    this.http.get<{ positions: Position[] }>('http://localhost:8000/get_positions')
      .subscribe({
        next: (response) => {
          this.positions = response.positions;
          this.loading = false;
        },
        error: (err) => {
          this.error = 'Failed to fetch positions. Please try again later.';
          this.loading = false;
          console.error('Error fetching positions:', err);
        }
      });
  }

  formatCurrency(value: number): string {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(value);
  }
} 