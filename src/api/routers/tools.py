from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status, Depends
from loguru import logger
from pydantic import BaseModel, Field

from core.dynamic_agent import DynamicAgent
from core.errors import (
    ToolExecutionError,
    ToolNotFoundError,
    ToolValidationError,
    ToolTimeoutError,
    ToolPermissionError
)

router = APIRouter(
    prefix="/tools",
    tags=["tools"],
)

class ToolExecuteRequest(BaseModel):
    """Model for tool execution requests."""
    tool_name: str
    parameters: Dict[str, Any]

class ToolResponse(BaseModel):
    """Model for standardized API responses."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

def get_agent():
    """Dependency to get DynamicAgent instance."""
    return DynamicAgent()

@router.get("/list", response_model=ToolResponse)
async def list_tools(agent: DynamicAgent = Depends(get_agent)):
    """List all available tools."""
    try:
        logger.info("Retrieving list of available tools")
        tools = agent.list_tools()
        return ToolResponse(
            success=True,
            message="Tools retrieved successfully",
            data={"tools": tools}
        )
    except Exception as e:
        logger.error(f"Error retrieving tools: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve tools"
        )

@router.post("/execute", response_model=ToolResponse)
async def execute_tool(
    request: ToolExecuteRequest,
    agent: DynamicAgent = Depends(get_agent)
):
    """Execute a specific tool with given parameters."""
    try:
        logger.info(f"Executing tool: {request.tool_name}")
        result = agent.execute_tool(request.tool_name, request.parameters)
        return ToolResponse(
            success=True,
            message=f"Tool '{request.tool_name}' executed successfully",
            data={"result": result}
        )
    except ToolNotFoundError as e:
        logger.warning(f"Tool not found: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except ToolValidationError as e:
        logger.warning(f"Tool validation error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "message": str(e),
                "validation_errors": e.details.get("validation_errors")
            }
        )
    except ToolPermissionError as e:
        logger.warning(f"Tool permission error: {str(e)}")
        raise HTTPException(
            status_code=403,
            detail={
                "message": str(e),
                "required_permissions": e.details.get("required_permissions")
            }
        )
    except ToolTimeoutError as e:
        logger.error(f"Tool execution timeout: {str(e)}")
        raise HTTPException(
            status_code=504,
            detail=str(e)
        )
    except ToolExecutionError as e:
        logger.error(f"Tool execution error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": str(e),
                "tool_name": e.tool_name,
                "details": e.details
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error executing tool: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred"
        )

@router.get("/refresh", response_model=ToolResponse)
async def refresh_tools(agent: DynamicAgent = Depends(get_agent)):
    """Refresh the list of available tools."""
    try:
        logger.info("Refreshing tools list")
        agent.refresh_tools()
        return ToolResponse(
            success=True,
            message="Tools refreshed successfully",
            data={"tools": agent.list_tools()}
        )
    except Exception as e:
        logger.error(f"Error refreshing tools: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to refresh tools"
        ) 