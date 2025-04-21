from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from dynamic_agent import DynamicAgent
import asyncio
import os

router = APIRouter(prefix="/agent", tags=["agent"])

# Initialize the agent
agent = DynamicAgent()

class AgentRequest(BaseModel):
    input: str

class AgentResponse(BaseModel):
    response: str

@router.post("/run", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    """
    Run the agent with the given input.
    """
    try:
        result = await agent.run(request.input)
        return AgentResponse(response=result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error running agent: {str(e)}"
        )

@router.post("/refresh")
async def refresh_tools():
    """
    Refresh the available tools from the registry.
    """
    try:
        agent.refresh_tools()
        return {"status": "success", "message": "Tools refreshed successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error refreshing tools: {str(e)}"
        ) 