import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatIconModule } from '@angular/material/icon';
import { AgentService } from '../services/agent.service';

interface Message {
  content: string;
  type: 'user' | 'agent';
  timestamp: Date;
}

const CHAT_HISTORY_KEY = 'agent_chat_history';

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
      max-width: 1200px;
      margin: 0 auto;
    }

    .chat-card {
      min-height: 600px;
      display: flex;
      flex-direction: column;
      width: 100%;
    }

    .chat-messages {
      height: 500px;
      overflow-y: auto;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .message {
      max-width: 90%;
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
      width: 100%;
    }

    .message pre {
      white-space: pre-wrap;
      margin: 0;
      font-family: inherit;
      width: 100%;
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
      width: 100%;
    }

    .message-input {
      flex: 1;
      width: 100%;
    }

    ::ng-deep .message-input .mat-mdc-form-field-infix {
      width: 100% !important;
    }

    mat-card-content {
      width: 100%;
    }

    mat-card-actions {
      width: 100%;
      padding: 0 !important;
    }
  `]
})
export class DemoComponent implements OnInit, OnDestroy {
  messages: Message[] = [];
  userInput = '';
  loading = false;

  constructor(private agentService: AgentService) {}

  ngOnInit() {
    // Load saved messages from localStorage
    const savedMessages = localStorage.getItem(CHAT_HISTORY_KEY);
    if (savedMessages) {
      this.messages = JSON.parse(savedMessages).map((msg: any) => ({
        ...msg,
        timestamp: new Date(msg.timestamp)
      }));
    } else {
      // Add welcome message if no saved messages
      this.messages.push({
        content: 'Hello! I\'m an AI agent that can help you use various tools. Try asking me about available tools or give me a task to perform!',
        type: 'agent',
        timestamp: new Date()
      });
      this.saveMessages();
    }
  }

  ngOnDestroy() {
    // Save messages when component is destroyed
    this.saveMessages();
  }

  private saveMessages() {
    localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(this.messages));
  }

  async sendMessage() {
    if (!this.userInput.trim() || this.loading) return;

    // Add user message
    this.messages.push({
      content: this.userInput,
      type: 'user',
      timestamp: new Date()
    });
    this.saveMessages();

    const userMessage = this.userInput;
    this.userInput = '';
    this.loading = true;

    try {
      // Send request to backend using agent service
      const response = await this.agentService.executeAgent(userMessage).toPromise();

      // Add agent response
      this.messages.push({
        content: response?.data?.result || response?.message || 'Sorry, I encountered an error.',
        type: 'agent',
        timestamp: new Date()
      });
      this.saveMessages();
    } catch (error: any) {
      console.error('Error:', error);
      this.messages.push({
        content: error?.error?.detail || 'Sorry, I encountered an error while processing your request.',
        type: 'agent',
        timestamp: new Date()
      });
      this.saveMessages();
    } finally {
      this.loading = false;
      // Scroll to bottom after a short delay to ensure new content is rendered
      setTimeout(() => this.scrollToBottom(), 100);
    }
  }

  private scrollToBottom(): void {
    const chatContainer = document.querySelector('.chat-messages');
    if (chatContainer) {
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }
  }
} 