import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

export interface Position {
  symbol: string;
  qty: number;
  avg_entry_price: number;
  market_value: number;
}

export interface ToolInput {
  [key: string]: string | number | object;
}

export interface ToolOutput {
  [key: string]: any;
}

export interface Tool {
  name: string;
  description: string;
  inputs: ToolInput;
  outputs: ToolOutput;
  tags: string[];
  endpoint: string;
}

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private apiUrl = environment.apiUrl;

  constructor(private http: HttpClient) {
    console.log('API URL:', this.apiUrl); // Debug log
  }

  getPositions(): Observable<{ positions: Position[] }> {
    return this.http.get<{ positions: Position[] }>(`${this.apiUrl}/get_positions`);
  }

  getTools(): Observable<Tool[]> {
    const url = `${this.apiUrl}/tools`;
    console.log('Fetching tools from:', url); // Debug log
    return this.http.get<Tool[]>(url);
  }
}
