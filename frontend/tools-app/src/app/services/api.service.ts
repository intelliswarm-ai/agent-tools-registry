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
  type: string;
  description: string;
  required?: boolean;
}

export interface ToolOutput {
  type: string;
  description: string;
}

export interface Tool {
  name: string;
  description: string;
  inputs?: { [key: string]: ToolInput };
  outputs?: { [key: string]: ToolOutput };
  tags: string[];
}

export interface ToolListResponse {
  tools: Tool[];
}

export interface ToolExecuteRequest {
  tool_name: string;
  inputs: { [key: string]: any };
}

export interface ToolExecuteResponse {
  success: boolean;
  tool_name: string;
  result: any;
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

  getTools(): Observable<ToolListResponse> {
    const url = `${this.apiUrl}/tools`;
    console.log('Fetching tools from:', url); // Debug log
    return this.http.get<ToolListResponse>(url);
  }

  executeTool(request: ToolExecuteRequest): Observable<ToolExecuteResponse> {
    const url = `${this.apiUrl}/tools/execute`;
    console.log('Executing tool at:', url);
    return this.http.post<ToolExecuteResponse>(url, request);
  }

  refreshTools(): Observable<ToolListResponse> {
    const url = `${this.apiUrl}/tools/refresh`;
    console.log('Refreshing tools from:', url);
    return this.http.post<ToolListResponse>(url, {});
  }
}
