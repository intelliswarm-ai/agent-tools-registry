from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from pydantic import BaseModel
from core.dynamic_agent import DynamicAgent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
agent = DynamicAgent()

class ToolInput(BaseModel):
    tool_name: str
    inputs: Dict[str, Any]

class ToolResponse(BaseModel):
    result: Any
    error: str | None = None

@router.get("/list")
async def list_tools() -> List[Dict[str, Any]]:
    """
    List all available tools in the registry.
    """
    try:
        logger.info("Fetching list of available tools")
        tools = agent.refresh_tools()
        return tools
    except Exception as e:
        logger.error(f"Error fetching tools: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/execute")
async def execute_tool(tool_input: ToolInput) -> ToolResponse:
    """
    Execute a specific tool with the provided inputs.
    """
    try:
        logger.info(f"Executing tool: {tool_input.tool_name}")
        result = await agent.execute_tool(tool_input.tool_name, tool_input.inputs)
        return ToolResponse(result=result)
    except ValueError as e:
        logger.error(f"Invalid input for tool {tool_input.tool_name}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing tool {tool_input.tool_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/refresh")
async def refresh_tools() -> List[Dict[str, Any]]:
    """
    Refresh the list of available tools.
    """
    try:
        logger.info("Refreshing tools list")
        tools = agent.refresh_tools()
        return tools
    except Exception as e:
        logger.error(f"Error refreshing tools: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 