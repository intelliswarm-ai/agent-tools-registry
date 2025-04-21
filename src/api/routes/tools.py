from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional

from src.core.logging import get_logger, log_request_details, log_response_details, log_error
from src.core.config import settings
from src.core.dynamic_agent import DynamicAgent

router = APIRouter()
logger = get_logger(__name__)

class ToolExecuteRequest(BaseModel):
    """Request model for tool execution."""
    tool_name: str
    inputs: Dict[str, Any]

class ToolInfo(BaseModel):
    """Model for tool information."""
    name: str
    description: str
    tags: List[str] = []
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, Any]] = None

class ToolListResponse(BaseModel):
    """Response model for tool listing."""
    tools: List[ToolInfo]

@router.get("/", response_model=ToolListResponse)
async def list_tools():
    """List all available tools."""
    try:
        agent = DynamicAgent()
        agent.refresh_tools()
        
        tools = []
        for name, data in agent._tools_dict.items():
            tools.append(ToolInfo(
                name=name,
                description=data.get("description", "No description available"),
                tags=data.get("tags", []),
                inputs=data.get("inputs"),
                outputs=data.get("outputs")
            ))
        
        logger.info(f"Retrieved {len(tools)} tools")
        return ToolListResponse(tools=tools)
        
    except Exception as e:
        log_error(e, logger)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list tools: {str(e)}"
        )

@router.post("/execute")
async def execute_tool(request: ToolExecuteRequest):
    """Execute a specific tool."""
    try:
        log_request_details(request.dict(), logger)
        
        agent = DynamicAgent()
        result = await agent.execute_tool(request.tool_name, request.inputs)
        
        response_data = {
            "success": True,
            "tool_name": request.tool_name,
            "result": result
        }
        log_response_details(response_data, logger)
        
        return response_data
        
    except ValueError as e:
        log_error(e, logger)
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        log_error(e, logger)
        raise HTTPException(
            status_code=500,
            detail=f"Tool execution failed: {str(e)}"
        )

@router.post("/refresh")
async def refresh_tools():
    """Refresh the list of available tools."""
    try:
        agent = DynamicAgent()
        agent.refresh_tools()
        
        tools = []
        for name, data in agent._tools_dict.items():
            tools.append(ToolInfo(
                name=name,
                description=data.get("description", "No description available"),
                tags=data.get("tags", []),
                inputs=data.get("inputs"),
                outputs=data.get("outputs")
            ))
        
        response_data = {
            "success": True,
            "message": "Tools refreshed successfully",
            "tools": tools
        }
        log_response_details(response_data, logger)
        
        return response_data
        
    except Exception as e:
        log_error(e, logger)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to refresh tools: {str(e)}"
        ) 