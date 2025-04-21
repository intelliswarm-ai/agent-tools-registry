import sys
import ssl  # Ensure ssl module is explicitly imported if environment requires it

# Patch for environments missing SSL (some sandboxed ones)
if not hasattr(ssl, 'SSLContext'):
    print("SSL module is missing or improperly configured. HTTPS requests may fail.")
    sys.exit(1)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tools.place_trade.run import router as trade_router
from tools_registry import router as registry_router
from agent_router import router as agent_router

app = FastAPI(title="Agent Tools Registry")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Angular dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(registry_router)
app.include_router(trade_router)
app.include_router(agent_router)

# Add OpenAPI tags metadata
app.openapi_tags = [
    {
        "name": "agent",
        "description": "Operations with the AI agent and tools"
    }
]
