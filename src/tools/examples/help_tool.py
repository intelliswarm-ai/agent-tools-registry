from typing import Any, Dict, List, Type
from pydantic import BaseModel, Field

from src.tools.base import RegistryTool
from src.core.logging import get_logger
from src.core.registry import ToolRegistry

logger = get_logger(__name__)

class HelpToolInput(BaseModel):
    """Input schema for the help tool."""
    query: str = Field(
        default="",
        description="Optional query to filter tools by name or tag"
    )

class ToolInfo(BaseModel):
    """Schema for tool information."""
    name: str = Field(description="Name of the tool")
    description: str = Field(description="Description of what the tool does")
    tags: List[str] = Field(description="Tags associated with the tool")

class HelpToolOutput(BaseModel):
    """Output schema for the help tool."""
    tools: List[ToolInfo] = Field(description="List of available tools with their descriptions")
    total_tools: int = Field(description="Total number of tools found")

class HelpTool(RegistryTool):
    """Help tool that lists available tools and their descriptions."""
    
    name = "help"
    description = "Lists all available tools and their descriptions, optionally filtered by name or tag"
    tags = ["system", "utility"]
    version = "1.0.0"
    
    def __init__(self, registry: ToolRegistry = None, **kwargs):
        super().__init__(**kwargs)
        self._registry = registry or ToolRegistry()
    
    def _run(self, query: str = "") -> Dict[str, Any]:
        """List available tools, optionally filtered by query.
        
        Args:
            query: Optional string to filter tools by name or tag
            
        Returns:
            Dictionary containing list of tools and their details
        """
        logger.info(f"Listing tools with query: {query}")
        
        # Get all tools from registry
        all_tools = self._registry.list_tools()
        
        # Filter tools if query is provided
        if query:
            query = query.lower()
            filtered_tools = [
                tool for tool in all_tools
                if query in tool["name"].lower() or
                any(query in tag.lower() for tag in tool["tags"])
            ]
        else:
            filtered_tools = all_tools
            
        # Convert to output format
        tool_infos = [
            ToolInfo(
                name=tool["name"],
                description=tool["description"],
                tags=tool["tags"]
            )
            for tool in filtered_tools
        ]
        
        return HelpToolOutput(
            tools=tool_infos,
            total_tools=len(tool_infos)
        ).dict()
    
    async def _arun(self, query: str = "") -> Dict[str, Any]:
        """Async version of the help tool."""
        return self._run(query)

    @property
    def args_schema(self) -> Type[BaseModel]:
        """Return the input schema for this tool."""
        return HelpToolInput
        
    @property
    def return_schema(self) -> Type[BaseModel]:
        """Return the output schema for this tool."""
        return HelpToolOutput 