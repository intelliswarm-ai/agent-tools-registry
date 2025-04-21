from typing import Type
from pydantic import BaseModel, Field
from src.tools.base import RegistryTool
import sys
from io import StringIO
import contextlib

class PythonInput(BaseModel):
    """Input schema for the Python REPL tool."""
    code: str = Field(
        ...,
        description="The Python code to execute"
    )

class PythonOutput(BaseModel):
    """Output schema for the Python REPL tool."""
    output: str = Field(..., description="The output of the code execution")
    error: str = Field(default="", description="Any error messages")

class PythonREPLTool(RegistryTool):
    """A tool for executing Python code in a REPL environment."""
    
    name: str = "python_repl"
    description: str = "Execute Python code and return the output"
    tags: list[str] = ["development", "code", "python"]
    version: str = "1.0.0"
    
    def _run(self, code: str) -> dict:
        """Execute the Python code."""
        # Create string buffers for stdout and stderr
        stdout = StringIO()
        stderr = StringIO()
        
        # Capture output and errors
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                # Create a restricted globals dictionary
                globals_dict = {
                    '__builtins__': {
                        'print': print,
                        'len': len,
                        'str': str,
                        'int': int,
                        'float': float,
                        'list': list,
                        'dict': dict,
                        'tuple': tuple,
                        'range': range,
                        'sum': sum,
                        'min': min,
                        'max': max,
                    }
                }
                
                # Execute the code
                exec(code, globals_dict)
                
                return {
                    "output": stdout.getvalue().strip(),
                    "error": stderr.getvalue().strip()
                }
            except Exception as e:
                return {
                    "output": stdout.getvalue().strip(),
                    "error": str(e)
                }

    args_schema: Type[BaseModel] = PythonInput
    return_schema: Type[BaseModel] = PythonOutput 