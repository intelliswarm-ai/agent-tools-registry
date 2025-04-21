import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

export interface AgentRequest {
  message: string;
  context?: Record<string, any>;
}

export interface AgentResponse {
  success: boolean;
  message: string;
  data?: {
    result: string;
  };
  error?: {
    detail: string;
    type: string;
  };
}

@Injectable({
  providedIn: 'root'
})
export class AgentService {
  private apiUrl = environment.apiUrl;

  constructor(private http: HttpClient) {}

  executeAgent(message: string): Observable<AgentResponse> {
    const request: AgentRequest = { message };
    return this.http.post<AgentResponse>(`${this.apiUrl}/agent/execute`, request);
  }
} 