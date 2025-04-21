from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import os

from src.core.logging import get_logger
from src.core.registry import ToolRegistry
from src.core.errors import ToolValidationError, ToolNotFoundError

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
    version: str = "1.0.0"
    args_schema: Dict[str, Any] = {}

class ToolListResponse(BaseModel):
    """Response model for tool listing."""
    tools: List[ToolInfo]

# Create a global registry instance
registry = ToolRegistry()

@router.get("/", response_model=ToolListResponse)
async def list_tools():
    """List all available tools."""
    try:
        logger.info("Listing available tools")
        
        # Get tools from registry
        tools = [
            ToolInfo(
                name=tool.name,
                description=tool.description,
                tags=tool.tags,
                version=tool.version,
                args_schema=tool.args_schema.schema() if tool.args_schema else {}
            )
            for tool in registry.get_tools_for_agent()
        ]
        
        logger.info(f"Retrieved {len(tools)} tools")
        return ToolListResponse(tools=tools)
        
    except Exception as e:
        logger.error(f"Failed to list tools: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list tools: {str(e)}"
        )

@router.post("/execute")
async def execute_tool(request: ToolExecuteRequest):
    """Execute a specific tool."""
    try:
        logger.info(f"Executing tool: {request.tool_name}")
        
        # Get tool from registry
        tool = registry.get_tool(request.tool_name)
        if not tool:
            raise ToolNotFoundError(request.tool_name)
            
        # Execute the tool
        if hasattr(tool, '_arun'):
            result = await tool._arun(**request.inputs)
        else:
            result = tool._run(**request.inputs)
            
        response_data = {
            "success": True,
            "tool_name": request.tool_name,
            "result": result
        }
        logger.info(f"Tool execution successful: {request.tool_name}")
        
        return response_data
        
    except ToolNotFoundError as e:
        logger.error(f"Tool not found: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except ToolValidationError as e:
        logger.error(f"Tool validation error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Tool execution failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Tool execution failed: {str(e)}"
        )

@router.post("/refresh")
async def refresh_tools():
    """Refresh the list of available tools."""
    try:
        logger.info("Refreshing tools list")
        
        # Clear and reload tools
        registry._tools.clear()
        # Use absolute path from workspace root
        tools_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "tools", "definitions")
        logger.info(f"Loading tools from: {tools_dir}")
        registry.load_tools_from_directory(tools_dir)
        
        # Get updated tool list
        tools = [
            ToolInfo(
                name=tool.name,
                description=tool.description,
                tags=tool.tags,
                version=tool.version,
                args_schema=tool.args_schema.schema() if tool.args_schema else {}
            )
            for tool in registry.get_tools_for_agent()
        ]
        
        response_data = {
            "success": True,
            "message": "Tools refreshed successfully",
            "tools": tools
        }
        logger.info(f"Refreshed {len(tools)} tools")
        return response_data
        
    except Exception as e:
        logger.error(f"Failed to refresh tools: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to refresh tools: {str(e)}"
        ) 