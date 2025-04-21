from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from dynamic_agent import DynamicAgent
import asyncio
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])

# Initialize agent as None, will be created on first request
agent = None

class AgentRequest(BaseModel):
    input: str

class AgentResponse(BaseModel):
    response: str

async def initialize_agent() -> None:
    """Initialize the agent during application startup."""
    global agent
    try:
        logger.info("Initializing agent during startup...")
        agent = DynamicAgent()
        # Force initial tools refresh
        agent.refresh_tools()
        logger.info("Agent initialized successfully during startup")
    except Exception as e:
        logger.error(f"Failed to initialize agent during startup: {str(e)}", exc_info=True)
        raise

def get_agent() -> DynamicAgent:
    """Get or create the agent instance."""
    global agent
    try:
        if agent is None:
            logger.info("Agent not initialized, creating new instance...")
            asyncio.create_task(initialize_agent())
            raise HTTPException(
                status_code=503,
                detail="Agent is initializing, please try again in a few seconds"
            )
        return agent
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        logger.error(f"Error accessing agent: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to access agent: {str(e)}"
        )

@router.post("/run", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    """
    Run the agent with the given input.
    """
    try:
        logger.info(f"Received agent run request with input: {request.input}")
        agent_instance = get_agent()
        logger.info("Starting agent execution")
        result = await agent_instance.run(request.input)
        logger.info(f"Agent execution completed with result: {result}")
        return AgentResponse(response=result)
    except Exception as e:
        error_msg = f"Error running agent: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )

@router.post("/refresh")
async def refresh_tools():
    """
    Refresh the available tools from the registry.
    """
    try:
        logger.info("Refreshing agent tools")
        agent_instance = get_agent()
        agent_instance.refresh_tools()
        logger.info("Tools refreshed successfully")
        return {"status": "success", "message": "Tools refreshed successfully"}
    except Exception as e:
        error_msg = f"Error refreshing tools: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=error_msg
        ) 