import sys
import ssl  # Ensure ssl module is explicitly imported if environment requires it
import logging
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tools.place_trade.run import router as trade_router
from tools_registry import router as registry_router, get_db, ToolDB
from agent_router import router as agent_router, initialize_agent
from sqlalchemy.orm import Session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Patch for environments missing SSL (some sandboxed ones)
if not hasattr(ssl, 'SSLContext'):
    logger.error("SSL module is missing or improperly configured. HTTPS requests may fail.")
    sys.exit(1)

app = FastAPI(title="Agent Tools Registry")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Angular dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    try:
        logger.info("Starting application initialization...")
        
        # Get database session
        db: Session = next(get_db())
        try:
            # Check if we have any tools in the database
            tools_count = db.query(ToolDB).count()
            logger.info(f"Found {tools_count} tools in database")
            
            if tools_count == 0:
                logger.warning("No tools found in database. Please register tools using register_tools.py")
            
            # Initialize the agent with the tools
            await initialize_agent()
            logger.info("Agent initialized successfully with available tools")
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error during application startup: {str(e)}", exc_info=True)
        # We don't want to prevent the app from starting if there's an error
        # but we want to log it clearly
        logger.error("Application started but tool initialization failed!")

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
