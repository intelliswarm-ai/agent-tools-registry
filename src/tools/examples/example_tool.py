from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, Field

from src.tools.base import RegistryTool
from src.core.logging import get_logger

logger = get_logger(__name__)

class ExampleToolInput(BaseModel):
    """Input schema for the example tool."""
    message: str = Field(
        description="The message to process"
    )
    count: int = Field(
        default=1,
        description="Number of times to repeat the message"
    )

class ExampleToolOutput(BaseModel):
    """Output schema for the example tool."""
    result: str = Field(description="The processed result")
    count: int = Field(description="Number of times the message was repeated")

class ExampleTool(RegistryTool):
    """Example tool that demonstrates proper LangChain tool implementation."""
    
    name = "example"
    description = "An example tool that processes messages"
    tags = ["example", "demo"]
    version = "1.0.0"
    
    def _run(self, message: str, count: int = 1) -> Dict[str, Any]:
        """Run the tool's core logic.
        
        This is the method that LangChain will call when using the tool.
        It should implement the actual functionality of your tool.
        """
        logger.info(f"Running example tool with message: {message}, count: {count}")
        
        # Process the input
        result = message * count
        
        # Return in the expected output format
        return ExampleToolOutput(
            result=result,
            count=count
        ).dict()
    
    async def _arun(self, message: str, count: int = 1) -> Dict[str, Any]:
        """Async version of the run method.
        
        Implement this if your tool supports async execution.
        """
        return self._run(message, count)

    @property
    def args_schema(self) -> Type[BaseModel]:
        """Return the input schema for this tool."""
        return ExampleToolInput
        
    @property
    def return_schema(self) -> Type[BaseModel]:
        """Return the output schema for this tool."""
        return ExampleToolOutput 