from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional

from src.core.dynamic_agent import DynamicAgent
from src.core.logging import get_logger

router = APIRouter(
    prefix="/agent",
    tags=["agent"]
)

logger = get_logger(__name__)

class AgentRequest(BaseModel):
    """Request model for agent operations."""
    message: str
    context: Optional[Dict[str, Any]] = None

class AgentResponse(BaseModel):
    """Response model for agent operations."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

def get_agent():
    """Dependency to get DynamicAgent instance."""
    return DynamicAgent()

@router.post("/execute", response_model=AgentResponse)
async def execute_agent(
    request: AgentRequest,
    agent: DynamicAgent = Depends(get_agent)
):
    """Execute the agent with a given message and context."""
    try:
        logger.info(f"Executing agent with message: {request.message}")
        result = await agent.run(request.message, request.context or {})
        return AgentResponse(
            success=True,
            message="Agent execution completed successfully",
            data={"result": result}
        )
    except Exception as e:
        logger.error(f"Error executing agent: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=str(e)
        ) 