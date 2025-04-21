from typing import Any, Dict, Optional, Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from abc import ABC

class RegistryTool(BaseTool, ABC):
    """Base class for all tools in the Agent Tools Registry."""
    
    name: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    version: str = "1.0.0"
    
    @property
    def tool_type(self) -> str:
        """Return the type of the tool."""
        return self.__class__.__name__

    def to_json(self) -> Dict[str, Any]:
        """Convert tool to JSON format for registry."""
        return {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "version": self.version,
            "type": self.tool_type,
            "args_schema": self.args_schema.schema() if self.args_schema else {},
            "return_schema": self.return_schema.schema() if hasattr(self, 'return_schema') else {}
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "RegistryTool":
        """Create a tool instance from JSON data."""
        return cls(
            name=data.get("name"),
            description=data.get("description"),
            tags=data.get("tags", []),
            version=data.get("version", "1.0.0")
        )

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True 