from typing import Type, Dict, Any
from pydantic import BaseModel, Field
from src.tools.base import RegistryTool

class CalculatorInput(BaseModel):
    """Input schema for the Calculator tool."""
    expression: str = Field(
        ...,
        description="The mathematical expression to evaluate (e.g., '2 + 2')"
    )

class CalculatorOutput(BaseModel):
    """Output schema for the Calculator tool."""
    result: float = Field(..., description="The result of the calculation")

class CalculatorTool(RegistryTool):
    """A simple calculator tool that can perform basic arithmetic."""
    
    name: str = "calculator"
    description: str = "Performs basic arithmetic calculations"
    tags: list[str] = Field(default_factory=lambda: ["math", "utility"])
    version: str = "1.0.0"
    
    def _run(self, expression: str) -> Dict[str, Any]:
        """Execute the calculator tool."""
        try:
            # Using eval is not safe for production, this is just an example
            result = eval(expression, {"__builtins__": {}}, {})
            return {"result": float(result)}
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression: {str(e)}")

    args_schema: Type[BaseModel] = CalculatorInput
    return_schema: Type[BaseModel] = CalculatorOutput 