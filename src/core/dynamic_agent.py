from typing import List, Dict, Any, Optional
from langchain.agents import Tool, AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field
import requests
import json
from functools import partial
from src.core.config import get_settings, verify_api_key
import time
import logging
import os
import httpx
import asyncio
from datetime import datetime
from src.core.errors import ToolExecutionError, ToolNotFoundError, ToolValidationError, ToolTimeoutError

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ToolSpec(BaseModel):
    """Specification for a tool in the registry"""
    name: str
    description: str
    endpoint: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    tags: List[str]

class DynamicAgent:
    def __init__(
        self,
        registry_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        temperature: Optional[float] = None,
        verbose: bool = True,
        max_retries: int = 5,
        retry_delay: int = 2
    ):
        try:
            logger.info("Initializing DynamicAgent...")
            self.settings = get_settings()
            
            # Debug log the tools directory path
            logger.debug(f"Tools directory path from settings: {self.settings.TOOLS_DIR}")
            logger.debug(f"Absolute tools directory path: {os.path.abspath(self.settings.TOOLS_DIR)}")
            
            self.registry_url = registry_url or self.settings.TOOLS_REGISTRY_URL
            logger.info(f"Using registry URL: {self.registry_url}")
            
            # Initialize OpenAI if API key is available
            self.llm = None
            if verify_api_key():
                logger.info(f"Initializing ChatOpenAI with model: {llm_model or self.settings.OPENAI_MODEL}")
                self.llm = ChatOpenAI(
                    model=llm_model or self.settings.OPENAI_MODEL,
                    temperature=temperature if temperature is not None else self.settings.OPENAI_TEMPERATURE,
                    openai_api_key=self.settings.OPENAI_API_KEY
                )
            else:
                logger.warning("OpenAI API key not configured. Agent will operate in tools-only mode.")
            
            self.verbose = verbose
            self.max_retries = max_retries
            self.retry_delay = retry_delay
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            self._tools = None
            self._agent_executor = None
            self._tools_dict = {}
            
            # Load tools from local files
            self.refresh_tools()
            logger.info("DynamicAgent initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize DynamicAgent: {str(e)}")
            raise ToolExecutionError(
                message="Failed to initialize DynamicAgent",
                details={"error": str(e)}
            )

    @property
    def tools(self):
        """Get the list of available tools as LangChain Tool objects."""
        if self._tools is None:
            logger.info("Creating LangChain tools from local definitions...")
            tools = []
            
            for tool_name, tool_data in self._tools_dict.items():
                logger.info(f"Creating LangChain tool for: {tool_name}")
                
                if tool_name == "help":
                    # Special handling for the help tool
                    tools.append(
                        Tool(
                            name="help",
                            description="Lists all available tools and their descriptions. No input required.",
                            func=lambda _=None: self._get_tools_help(),  # Accept any input but ignore it
                            return_direct=True
                        )
                    )
                else:
                    def create_sync_tool(tool_name=tool_name):
                        async def async_execute(tool_input: str) -> str:
                            try:
                                input_dict = json.loads(tool_input) if tool_input else {}
                            except json.JSONDecodeError:
                                input_dict = {}  # Use empty dict if JSON parsing fails
                            return await self._execute_tool(tool_name, input_dict)
                            
                        def sync_execute(tool_input: str) -> str:
                            loop = asyncio.get_event_loop()
                            return loop.run_until_complete(async_execute(tool_input))
                        
                        return sync_execute
                    
                    tools.append(
                        Tool(
                            name=tool_name,
                            description=self._create_tool_description(ToolSpec(**tool_data)),
                            func=create_sync_tool(tool_name)
                        )
                    )
            self._tools = tools
            logger.info(f"Created {len(tools)} LangChain tools")
        return self._tools

    @property
    def agent_executor(self):
        """Lazy initialization of agent executor."""
        if self._agent_executor is None and self.llm is not None:
            logger.info("Creating agent executor...")
            self._agent_executor = self._create_agent_executor()
        return self._agent_executor

    def _create_tool_description(self, spec: ToolSpec) -> str:
        """Create a detailed description for a tool including its inputs and outputs."""
        desc = f"{spec.description}\n\n"
        
        # Add input parameters
        if spec.inputs:
            desc += "Required Inputs:\n"
            for name, details in spec.inputs.items():
                required = details.get('required', False)
                desc += f"- {name}: {details.get('description', 'No description')}"
                if required:
                    desc += " (Required)"
                desc += "\n"
        else:
            desc += "This tool requires no inputs.\n"
        
        # Add output description
        if spec.outputs:
            desc += "\nOutputs:\n"
            for name, details in spec.outputs.items():
                desc += f"- {name}: {details.get('description', 'No description')}\n"
        
        # Add tags if available
        if spec.tags:
            desc += f"\nCategories: {', '.join(spec.tags)}"
        
        return desc.strip()

    async def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Execute a tool by calling its endpoint with the provided input."""
        try:
            logger.info(f"Executing tool: {tool_name}")
            
            if tool_name not in self._tools_dict:
                error_msg = f"Tool not found: {tool_name}"
                logger.error(error_msg)
                raise Exception(error_msg)

            tool_data = self._tools_dict[tool_name]
            endpoint = tool_data.get('endpoint')
            required_inputs = tool_data.get('required_inputs', [])

            # Validate required inputs
            for required_input in required_inputs:
                if required_input not in tool_input:
                    error_msg = f"Missing required input: {required_input}"
                    logger.error(error_msg)
                    raise Exception(error_msg)

            logger.info(f"Making API call to endpoint: {endpoint}")
            async with httpx.AsyncClient() as client:
                response = await client.post(endpoint, json=json.loads(tool_input))
                response.raise_for_status()
                result = response.json()
                logger.info(f"Tool execution successful: {tool_name}")
                return json.dumps(result, indent=2)

        except httpx.HTTPError as e:
            error_msg = f"HTTP error executing tool {tool_name}: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            logger.error(error_msg)
            raise

    def _tool_to_openai_function(self, tool: Tool) -> dict:
        """Convert a Tool to an OpenAI Function format."""
        # Parse the tool description to get parameters
        spec = self._tools_dict[tool.name]
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for name, details in spec["inputs"].items():
            parameters["properties"][name] = {
                "type": details.get("type", "string"),
                "description": details.get("description", "")
            }
            if details.get("required", False):
                parameters["required"].append(name)
        
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters
        }

    def _create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor with the dynamic tools."""
        logger.info("Creating agent executor with prompt template")

        # Create the prompt template with better system message
        system_message = """You are a helpful AI assistant that can use various tools to accomplish tasks. 
You have access to several tools that can help users with different tasks.

When users ask about available tools or what you can do:
1. Use the 'help' tool to show them a list of all available tools
2. Explain the tools' capabilities in a clear, conversational way
3. If they ask about a specific task, recommend the most relevant tool(s) and explain why they would be useful

When using tools:
1. Choose the most appropriate tool for the task
2. Explain why you're using a particular tool
3. Format the inputs correctly according to the tool's requirements
4. Interpret the results in a user-friendly way

If a user asks what tool would be good for a specific task:
1. Analyze the task requirements
2. Check the available tools using the 'help' tool if needed
3. Recommend the most appropriate tool(s)
4. Explain why the recommended tool(s) would be suitable
5. Provide an example of how to use the tool if appropriate

Remember to be conversational and helpful while guiding users through the available tools and their usage."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Convert tools to OpenAI functions format
        openai_functions = [self._tool_to_openai_function(tool) for tool in self.tools]
        logger.info(f"Created {len(openai_functions)} OpenAI functions from tools")

        # Create the agent using OpenAI functions
        agent = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"],
                "agent_scratchpad": lambda x: format_to_openai_functions(x["intermediate_steps"])
            }
            | prompt
            | self.llm.bind(functions=openai_functions)
            | OpenAIFunctionsAgentOutputParser()
        )

        logger.info("Creating AgentExecutor")
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=self.verbose,
            handle_parsing_errors=True,
            max_iterations=3  # Limit the number of tool calls per request
        )

    async def run(self, input_text: str) -> str:
        """Run the agent with the given input."""
        try:
            logger.info(f"Starting agent run with input: {input_text}")
            
            if self.llm is None:
                error_msg = "OpenAI API key not configured. Cannot run agent in tools-only mode."
                logger.error(error_msg)
                return error_msg
                
            start_time = time.time()

            # Log the current state of tools
            logger.info(f"Available tools: {list(self._tools_dict.keys())}")

            # Create agent executor if needed
            if self._agent_executor is None:
                self._agent_executor = self._create_agent_executor()

            # Execute the agent
            logger.info("Executing agent with input...")
            result = await self._agent_executor.ainvoke({"input": input_text})
            
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"Agent run completed in {execution_time:.2f} seconds with result: {result}")
            
            return result.get("output", "I encountered an error processing your request.")
            
        except Exception as e:
            logger.error(f"Error in agent run: {str(e)}")
            raise

    def _get_tools_help(self) -> str:
        """Get a formatted list of available tools and their descriptions."""
        help_text = "Here are the available tools and their capabilities:\n\n"
        
        # Group tools by tags
        tools_by_tag = {}
        for name, data in self._tools_dict.items():
            tags = data.get("tags", ["uncategorized"])
            for tag in tags:
                if tag not in tools_by_tag:
                    tools_by_tag[tag] = []
                tools_by_tag[tag].append((name, data))
        
        # Output tools grouped by category
        for tag, tools in tools_by_tag.items():
            if tag != "system":  # Skip system tag in the main listing
                help_text += f"\n{tag.upper()} TOOLS:\n"
                for name, data in tools:
                    description = data.get("description", "No description available")
                    help_text += f"\n• {name}\n"
                    help_text += f"  Description: {description}\n"
                    
                    # Add input parameters if any
                    inputs = data.get("inputs", {})
                    if inputs:
                        help_text += "  Required Inputs:\n"
                        for input_name, input_data in inputs.items():
                            required = input_data.get("required", False)
                            input_desc = input_data.get("description", "No description")
                            if required:
                                help_text += f"    - {input_name}: {input_desc} (Required)\n"
                            else:
                                help_text += f"    - {input_name}: {input_desc} (Optional)\n"
                    
                    # Add example usage if available
                    if "example" in data:
                        help_text += f"  Example: {data['example']}\n"
                    
                    help_text += "\n"
        
        help_text += "\nTo use a tool, simply describe your task and I'll help you choose and use the most appropriate tool."
        return help_text

    def refresh_tools(self) -> List[Dict[str, Any]]:
        """
        Refresh the list of available tools from the tools directory.
        Returns a list of tool information dictionaries.
        """
        try:
            logger.info("Refreshing tools from directory")
            logger.debug(f"Tools directory path: {self.settings.TOOLS_DIR}")
            self._tools_dict.clear()
            tools_info = []

            if not os.path.exists(self.settings.TOOLS_DIR):
                logger.warning(f"Tools directory not found: {self.settings.TOOLS_DIR}")
                return []

            definitions_dir = os.path.join(self.settings.TOOLS_DIR, "definitions")
            logger.debug(f"Looking for tool definitions in: {definitions_dir}")
            
            if not os.path.exists(definitions_dir):
                logger.warning(f"Tool definitions directory not found: {definitions_dir}")
                return []

            for filename in os.listdir(definitions_dir):
                if filename.endswith('.json'):
                    try:
                        file_path = os.path.join(definitions_dir, filename)
                        logger.debug(f"Loading tool from: {file_path}")
                        
                        with open(file_path, 'r') as f:
                            tool_config = json.load(f)
                            
                        tool_info = self._load_tool(tool_config)
                        if tool_info:
                            tools_info.append(tool_info)
                            logger.debug(f"Successfully loaded tool: {tool_info['name']}")
                        else:
                            logger.warning(f"Failed to load tool from {filename}")
                            
                    except Exception as e:
                        logger.error(f"Error loading tool from {filename}: {str(e)}")
                        continue

            logger.info(f"Successfully loaded {len(tools_info)} tools")
            return tools_info
            
        except Exception as e:
            logger.error(f"Error refreshing tools: {str(e)}")
            raise ToolExecutionError(
                message="Failed to refresh tools",
                details={"error": str(e)}
            )

    async def execute_tool(self, tool_name: str, inputs: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
        """
        Execute a tool with the given inputs.
        
        Args:
            tool_name: Name of the tool to execute
            inputs: Dictionary of input parameters
            timeout: Maximum execution time in seconds
            
        Returns:
            Dictionary containing the tool execution results
            
        Raises:
            ToolNotFoundError: If the specified tool doesn't exist
            ToolValidationError: If the inputs are invalid
            ToolTimeoutError: If execution exceeds timeout
            ToolExecutionError: For other execution errors
        """
        logger.info(f"Executing tool: {tool_name}")
        
        # Check if tool exists
        if tool_name not in self.tools:
            raise ToolNotFoundError(
                message=f"Tool '{tool_name}' not found",
                tool_name=tool_name
            )
            
        tool_config = self.tools[tool_name]
        
        # Validate inputs against schema
        validation_errors = {}
        for param_name, param_schema in tool_config['inputs'].items():
            if param_name not in inputs:
                if param_schema.get('required', True):
                    validation_errors[param_name] = "Missing required parameter"
            else:
                # Add type validation here if needed
                pass
                
        if validation_errors:
            raise ToolValidationError(
                message="Invalid tool parameters",
                tool_name=tool_name,
                validation_errors=validation_errors
            )
            
        try:
            # Make API call to execute tool
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.settings.TOOLS_API_URL}/execute/{tool_name}",
                    json=inputs,
                    timeout=timeout
                )
                
                if response.status_code != 200:
                    raise ToolExecutionError(
                        message=f"Tool execution failed with status {response.status_code}",
                        tool_name=tool_name,
                        details=response.text
                    )
                    
                return response.json()
                
        except httpx.TimeoutException:
            raise ToolTimeoutError(
                message=f"Tool execution timed out after {timeout} seconds",
                tool_name=tool_name,
                timeout=timeout
            )
            
        except Exception as e:
            raise ToolExecutionError(
                message=str(e),
                tool_name=tool_name
            )

    def _load_tool(self, tool_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Load a tool from its configuration dictionary.
        Returns the tool information if successful, None otherwise.
        """
        try:
            logger.debug(f"Loading tool configuration: {tool_config}")
            
            # Validate required fields
            required_fields = ['name', 'description', 'inputs', 'outputs']
            for field in required_fields:
                if field not in tool_config:
                    logger.warning(f"Tool config missing required field: {field}")
                    return None

            # Add the tool to our registry
            tool_name = tool_config['name']
            logger.info(f"Loading tool: {tool_name}")
            
            # Store the tool in our dictionary
            self._tools_dict[tool_name] = tool_config
            logger.debug(f"Added tool {tool_name} to registry")
            
            return {
                'name': tool_name,
                'description': tool_config['description'],
                'inputs': tool_config['inputs'],
                'outputs': tool_config['outputs'],
                'tags': tool_config.get('tags', []),
                'example': tool_config.get('example', '')
            }
            
        except Exception as e:
            logger.error(f"Error loading tool configuration: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    async def main():
        # Will use settings from .env file
        agent = DynamicAgent()
        result = await agent.run("What tools are available and what can they do?")
        print(result)
    
    asyncio.run(main())