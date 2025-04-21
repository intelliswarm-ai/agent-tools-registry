from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional
from functools import lru_cache

from src.core.dynamic_agent import DynamicAgent
from src.core.logging import get_logger, setup_logging
from src.core.errors import ToolExecutionError, ToolNotFoundError, ToolValidationError, ToolTimeoutError

router = APIRouter(
    tags=["agent"]
)

# Set up logging
setup_logging()
logger = get_logger(__name__)

# Use lru_cache to ensure true singleton behavior
@lru_cache()
def get_agent() -> DynamicAgent:
    """Singleton dependency to get DynamicAgent instance."""
    logger.info("Getting DynamicAgent instance")
    try:
        agent = DynamicAgent()
        logger.info("DynamicAgent instance ready")
        return agent
    except Exception as e:
        logger.error(f"Failed to create DynamicAgent: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize agent: {str(e)}"
        )

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

@router.post("/execute", response_model=AgentResponse)
async def execute_agent(
    request: AgentRequest,
    agent: DynamicAgent = Depends(get_agent)
):
    """Execute the agent with a given message and context."""
    try:
        logger.info(f"Executing agent with message: {request.message}")
        logger.debug(f"Context: {request.context}")
        
        # Add more detailed logging around agent.run
        logger.debug("Calling agent.run method")
        result = await agent.run(request.message, request.context)
        logger.debug(f"Agent.run completed with result: {result}")
        
        # If result is None or empty, return an error
        if not result:
            logger.error("Agent returned empty result")
            return AgentResponse(
                success=False,
                message="Failed to process request - empty result",
                error={"detail": "Agent returned empty result"}
            )
        
        logger.info("Agent execution completed successfully")
        return AgentResponse(
            success=True,
            message="Agent execution completed successfully",
            data={"result": result}
        )
        
    except ToolNotFoundError as e:
        logger.error(f"Tool not found: {str(e)}")
        return AgentResponse(
            success=False,
            message="Tool not found",
            error={"detail": str(e), "type": "tool_not_found"}
        )
        
    except ToolValidationError as e:
        logger.error(f"Tool validation error: {str(e)}")
        return AgentResponse(
            success=False,
            message="Invalid tool parameters",
            error={"detail": str(e), "type": "validation_error"}
        )
        
    except ToolTimeoutError as e:
        logger.error(f"Tool timeout: {str(e)}")
        return AgentResponse(
            success=False,
            message="Tool execution timed out",
            error={"detail": str(e), "type": "timeout"}
        )
        
    except ToolExecutionError as e:
        logger.error(f"Tool execution error: {str(e)}")
        return AgentResponse(
            success=False,
            message="Tool execution failed",
            error={"detail": str(e), "type": "execution_error"}
        )
        
    except Exception as e:
        logger.error(f"Unexpected error executing agent: {str(e)}", exc_info=True)
        return AgentResponse(
            success=False,
            message="An unexpected error occurred",
            error={"detail": str(e), "type": "unexpected_error"}
        ) 