from typing import Dict, List, Type, Optional, Any
from pathlib import Path
import json
import importlib.util
from pydantic import BaseModel
from src.tools.base import RegistryTool
from src.core.logging import get_logger
from src.core.errors import ToolValidationError

logger = get_logger(__name__)

class ToolDefinitionSchema(BaseModel):
    """Schema for validating tool definition JSON files."""
    name: str
    description: str
    version: str
    tags: List[str]
    module_path: str
    class_name: str
    args_schema: Dict[str, Any]

class ToolRegistry:
    """Registry for managing LangChain-compatible tools."""
    
    def __init__(self):
        self._tools: Dict[str, RegistryTool] = {}
        
    def register_tool(self, tool: RegistryTool) -> None:
        """Register a new tool in the registry."""
        # Validate tool has required LangChain attributes
        self._validate_langchain_tool(tool)
        logger.info(f"Registering tool: {tool.name}")
        self._tools[tool.name] = tool
        
    def get_tool(self, name: str) -> Optional[RegistryTool]:
        """Get a tool by name."""
        return self._tools.get(name)
        
    def list_tools(self) -> List[Dict]:
        """List all registered tools."""
        return [tool.to_json() for tool in self._tools.values()]

    def _validate_langchain_tool(self, tool: RegistryTool) -> None:
        """Validate that a tool meets all LangChain requirements."""
        # Check required attributes
        required_attrs = ['name', 'description', 'args_schema', 'return_schema']
        missing_attrs = [attr for attr in required_attrs if not hasattr(tool, attr)]
        if missing_attrs:
            raise ToolValidationError(
                tool_name=getattr(tool, 'name', 'unknown'),
                validation_errors={
                    'missing_attributes': missing_attrs
                }
            )

        # Validate args_schema and return_schema are Pydantic models
        if not (hasattr(tool.args_schema, '__bases__') and BaseModel in tool.args_schema.__bases__):
            raise ToolValidationError(
                tool_name=tool.name,
                validation_errors={
                    'args_schema': 'Must be a Pydantic BaseModel'
                }
            )

        if not (hasattr(tool.return_schema, '__bases__') and BaseModel in tool.return_schema.__bases__):
            raise ToolValidationError(
                tool_name=tool.name,
                validation_errors={
                    'return_schema': 'Must be a Pydantic BaseModel'
                }
            )

        # Validate required methods
        required_methods = ['_run']
        missing_methods = [method for method in required_methods 
                         if not hasattr(tool, method) or not callable(getattr(tool, method))]
        if missing_methods:
            raise ToolValidationError(
                tool_name=tool.name,
                validation_errors={
                    'missing_methods': missing_methods
                }
            )

    def _validate_tool_definition(self, tool_data: Dict[str, Any], tool_file: Path) -> None:
        """Validate tool definition JSON against schema."""
        try:
            ToolDefinitionSchema(**tool_data)
        except Exception as e:
            logger.error(f"Invalid tool definition in {tool_file}: {str(e)}")
            raise ToolValidationError(
                tool_name=tool_data.get('name', 'unknown'),
                validation_errors={'schema_validation': str(e)}
            )
        
    def load_tools_from_directory(self, directory: str) -> None:
        """Load tools from a directory containing tool definitions."""
        tool_dir = Path(directory)
        if not tool_dir.exists():
            logger.warning(f"Tool directory not found: {directory}")
            return
            
        # Get the workspace root (three levels up from tool_dir)
        workspace_root = tool_dir.parent.parent.parent
        logger.debug(f"Using workspace root: {workspace_root}")
        
        for tool_file in tool_dir.glob("**/*.json"):
            try:
                # Load and validate tool definition
                with open(tool_file) as f:
                    tool_data = json.load(f)
                
                # Validate tool definition schema
                self._validate_tool_definition(tool_data, tool_file)
                    
                # Import the tool class
                module_path = tool_data["module_path"]
                class_name = tool_data["class_name"]
                
                # Construct absolute path to the module
                module_file = workspace_root / module_path
                logger.debug(f"Loading module from: {module_file}.py")
                
                spec = importlib.util.spec_from_file_location(
                    module_path.replace("/", "."),
                    str(module_file) + ".py"
                )
                if not spec or not spec.loader:
                    raise ImportError(f"Could not load module: {module_path}")
                    
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                tool_class = getattr(module, class_name)
                if not issubclass(tool_class, RegistryTool):
                    raise TypeError(f"Tool class {class_name} must inherit from RegistryTool")
                
                # Create tool instance
                tool = tool_class.from_json(tool_data)
                
                # Register tool (includes LangChain validation)
                self.register_tool(tool)
                logger.info(f"Successfully loaded tool: {tool.name}")
                
            except ToolValidationError as e:
                logger.error(f"Tool validation failed for {tool_file}: {str(e)}")
            except Exception as e:
                logger.error(f"Error loading tool from {tool_file}: {str(e)}")
                
    def get_tools_for_agent(self) -> List[RegistryTool]:
        """Get all tools in a format suitable for LangChain agents."""
        return list(self._tools.values()) 