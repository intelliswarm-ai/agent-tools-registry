import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatIconModule } from '@angular/material/icon';
import { HttpClient } from '@angular/common/http';

interface Message {
  content: string;
  type: 'user' | 'agent';
  timestamp: Date;
}

@Component({
  selector: 'app-demo',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatCardModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatProgressSpinnerModule,
    MatIconModule
  ],
  template: `
    <div class="demo-container">
      <mat-card class="chat-card">
        <mat-card-header>
          <mat-card-title>Agent Tools Demo</mat-card-title>
          <mat-card-subtitle>
            Interact with the AI agent to explore available tools
          </mat-card-subtitle>
        </mat-card-header>

        <mat-card-content>
          <div class="chat-messages" #chatContainer>
            <div *ngFor="let message of messages" 
                 class="message" 
                 [ngClass]="message.type">
              <div class="message-content">
                <mat-icon *ngIf="message.type === 'agent'">smart_toy</mat-icon>
                <mat-icon *ngIf="message.type === 'user'">person</mat-icon>
                <div class="text">
                  <pre>{{ message.content }}</pre>
                  <small class="timestamp">
                    {{ message.timestamp | date:'short' }}
                  </small>
                </div>
              </div>
            </div>
          </div>

          <div *ngIf="loading" class="loading-indicator">
            <mat-spinner diameter="24"></mat-spinner>
            <span>Agent is thinking...</span>
          </div>
        </mat-card-content>

        <mat-card-actions>
          <form (ngSubmit)="sendMessage()" class="input-form">
            <mat-form-field appearance="outline" class="message-input">
              <mat-label>Type your message</mat-label>
              <input matInput
                     [(ngModel)]="userInput"
                     name="userInput"
                     [disabled]="loading"
                     placeholder="Ask the agent to use available tools...">
            </mat-form-field>
            <button mat-raised-button
                    color="primary"
                    type="submit"
                    [disabled]="loading || !userInput.trim()">
              <mat-icon>send</mat-icon>
              Send
            </button>
          </form>
        </mat-card-actions>
      </mat-card>
    </div>
  `,
  styles: [`
    .demo-container {
      padding: 20px;
      max-width: 800px;
      margin: 0 auto;
    }

    .chat-card {
      min-height: 600px;
      display: flex;
      flex-direction: column;
    }

    .chat-messages {
      height: 400px;
      overflow-y: auto;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .message {
      max-width: 80%;
      padding: 12px;
      border-radius: 8px;
      margin-bottom: 8px;
    }

    .message.user {
      align-self: flex-end;
      background-color: #e3f2fd;
    }

    .message.agent {
      align-self: flex-start;
      background-color: #f5f5f5;
    }

    .message-content {
      display: flex;
      gap: 8px;
      align-items: flex-start;
    }

    .message .text {
      flex: 1;
    }

    .message pre {
      white-space: pre-wrap;
      margin: 0;
      font-family: inherit;
    }

    .timestamp {
      display: block;
      color: #666;
      font-size: 0.8em;
      margin-top: 4px;
    }

    .loading-indicator {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px;
      color: #666;
    }

    .input-form {
      display: flex;
      gap: 16px;
      padding: 16px;
      align-items: flex-start;
    }

    .message-input {
      flex: 1;
    }
  `]
})
export class DemoComponent implements OnInit {
  messages: Message[] = [];
  userInput = '';
  loading = false;

  constructor(private http: HttpClient) {}

  ngOnInit() {
    // Add welcome message
    this.messages.push({
      content: 'Hello! I\'m an AI agent that can help you use various tools. Try asking me about available tools or give me a task to perform!',
      type: 'agent',
      timestamp: new Date()
    });
  }

  async sendMessage() {
    if (!this.userInput.trim() || this.loading) return;

    // Add user message
    this.messages.push({
      content: this.userInput,
      type: 'user',
      timestamp: new Date()
    });

    const userMessage = this.userInput;
    this.userInput = '';
    this.loading = true;

    try {
      // Send request to backend
      const response = await this.http.post<{response: string}>(
        'http://localhost:8000/agent/run',
        { input: userMessage }
      ).toPromise();

      // Add agent response
      this.messages.push({
        content: response?.response || 'Sorry, I encountered an error.',
        type: 'agent',
        timestamp: new Date()
      });
    } catch (error) {
      this.messages.push({
        content: 'Sorry, I encountered an error while processing your request.',
        type: 'agent',
        timestamp: new Date()
      });
    } finally {
      this.loading = false;
    }
  }
} 