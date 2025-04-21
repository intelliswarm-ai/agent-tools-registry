# Agent Tools Registry

A modern web application for managing and exploring agent tools, built with Angular and FastAPI.

## Overview

The Agent Tools Registry is a full-stack application that provides a user-friendly interface to browse, manage, and interact with various agent tools. The application consists of a FastAPI backend for tool management and an Angular frontend for the user interface.

## Features

- 📋 Comprehensive list of available agent tools
- 🔍 Detailed tool specifications and documentation
- 🎯 Interactive tool exploration through an expandable interface
- 🚀 Real-time tool status and updates
- 💫 Modern, responsive Material Design UI
- ⚡ Fast and efficient API endpoints
- 🔒 CORS-enabled secure communication

## Tech Stack

### Frontend
- Angular (Latest version)
- Angular Material UI
- TypeScript
- PNPM package manager

### Backend
- FastAPI
- Python
- SSL support for secure communications
- LangChain for AI agent capabilities
- OpenAI GPT models for natural language understanding

## Getting Started

### Prerequisites
- Node.js (Latest LTS version)
- PNPM package manager
- Python 3.8+
- pip (Python package manager)
- OpenAI API key

### Configuration

1. **Create Environment File**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   ```

2. **Set up OpenAI API Key**
   - Get your API key from [OpenAI's platform](https://platform.openai.com/api-keys)
   - Add it to your `.env` file:
     ```
     OPENAI_API_KEY=sk-your-api-key-here
     ```
   - Optionally configure other settings in the `.env` file:
     ```
     OPENAI_MODEL=gpt-3.5-turbo
     OPENAI_TEMPERATURE=0
     TOOLS_REGISTRY_URL=http://localhost:8000/tools
     ```

### Installation

1. **Clone the repository**
   ```bash
   git clone [repository-url]
   cd agent-tools-registry
   ```

2. **Backend Setup**
   ```bash
   # Create and activate virtual environment (optional but recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Start the FastAPI server
   uvicorn main:app --reload
   ```

3. **Frontend Setup**
   ```bash
   cd frontend/tools-app
   pnpm install
   pnpm start
   ```

The application will be available at:
- Frontend: http://localhost:4200
- Backend: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Application Structure

### Frontend Structure
```
frontend/tools-app/
├── src/
│   ├── app/
│   │   ├── tools/          # Tools component
│   │   ├── interceptors/   # HTTP interceptors
│   │   ├── services/       # API services
│   │   └── app.component.ts
│   ├── environments/       # Environment configurations
│   └── index.html
```

### Backend Structure
```
├── main.py                 # FastAPI application entry point
├── config.py              # Configuration management
├── dynamic_agent.py       # LangChain agent implementation
├── tools/                 # Tool implementations
└── tools_registry/        # Registry management
```

## Features in Detail

### Tools Component
- Displays a list of all available tools
- Expandable panels for detailed tool information
- Real-time loading states and error handling
- Refresh capability for updated tool information

### API Integration
- Secure HTTP interceptors for API communication
- Centralized error handling
- Typed interfaces for API responses
- Environment-based configuration

### AI Agent
- Dynamic tool discovery and usage
- Natural language understanding
- Secure API key management
- Configurable model parameters

## Security Notes

- Never commit your `.env` file to version control
- Keep your OpenAI API key secure and rotate it regularly
- Use environment variables in production
- Consider implementing rate limiting for API endpoints

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]
