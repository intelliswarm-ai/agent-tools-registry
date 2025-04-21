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
from config import get_settings, verify_api_key
import time
import logging
import os
import httpx
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
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
        logger.info("Initializing DynamicAgent...")
        settings = get_settings()
        
        if not verify_api_key():
            logger.error("OpenAI API key not properly configured")
            raise ValueError(
                "OpenAI API key is not properly configured. "
                "Please set it in your .env file or environment variables."
            )
        
        self.registry_url = registry_url or settings.tools_registry_url
        logger.info(f"Using registry URL: {self.registry_url}")
        
        logger.info(f"Initializing ChatOpenAI with model: {llm_model or settings.openai_model}")
        self.llm = ChatOpenAI(
            model=llm_model or settings.openai_model,
            temperature=temperature if temperature is not None else settings.openai_temperature,
            openai_api_key=settings.openai_api_key
        )
        
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
        if self._agent_executor is None:
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

    def refresh_tools(self):
        """Refresh the available tools from the registry."""
        try:
            logger.info("Starting tools refresh")
            self._tools_dict = {}
            
            # Add a default help tool that's always available
            self._tools_dict["help"] = {
                "name": "help",
                "description": "Lists all available tools and their descriptions, grouped by category. Use this tool to discover what capabilities are available.",
                "endpoint": "/help",
                "inputs": {
                    "query": {
                        "type": "string",
                        "description": "Optional. Any text query about tools (can be empty).",
                        "required": False
                    }
                },
                "outputs": {
                    "tools": {
                        "type": "string",
                        "description": "Formatted list of available tools and their descriptions"
                    }
                },
                "tags": ["system"],
                "example": "Just ask 'What tools are available?' or 'What can you help me with?'"
            }
            
            # Ensure the URL ends with a trailing slash for consistency
            registry_url = self.registry_url.rstrip('/') + '/'
            logger.info(f"Fetching tools from: {registry_url}")
            
            try:
                response = requests.get(registry_url, timeout=10)
                logger.info(f"Registry response status: {response.status_code}")
                logger.info(f"Registry response headers: {response.headers}")
                
                if response.status_code == 307:  # Handle redirect
                    redirect_url = response.headers['Location']
                    logger.info(f"Following redirect to: {redirect_url}")
                    response = requests.get(redirect_url, timeout=10)
                
                if response.status_code != 200:
                    logger.error(f"Failed to fetch tools. Status code: {response.status_code}")
                    logger.error(f"Response content: {response.text}")
                else:
                    try:
                        tools_data = response.json()
                        logger.info(f"Fetched {len(tools_data)} tools from registry")
                        
                        for tool_data in tools_data:
                            tool_name = tool_data.get('name')
                            if tool_name:
                                logger.info(f"Adding tool: {tool_name}")
                                # Ensure the endpoint is properly formatted
                                if 'endpoint' in tool_data:
                                    if not tool_data['endpoint'].startswith('http'):
                                        base_url = self.registry_url.rstrip('/').rsplit('/', 1)[0]
                                        tool_data['endpoint'] = f"{base_url}{tool_data['endpoint']}"
                                self._tools_dict[tool_name] = tool_data
                            else:
                                logger.warning(f"Skipping tool - missing name: {tool_data}")
                    except ValueError as e:
                        logger.error(f"Failed to parse JSON response: {str(e)}")
                        logger.error(f"Response content: {response.text}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {str(e)}")

            # Reset the tools cache to force re-fetching
            self._tools = None
            logger.info(f"Tools refresh completed. Total tools: {len(self._tools_dict)}")
            
            # Log the available tools
            logger.info("Available tools:")
            for tool_name, tool_data in self._tools_dict.items():
                logger.info(f"- {tool_name}: {tool_data.get('description', 'No description')} (endpoint: {tool_data.get('endpoint', 'No endpoint')})")
                
        except Exception as e:
            logger.error(f"Error refreshing tools: {str(e)}", exc_info=True)
            # Don't raise the exception, just log it
            # This allows the agent to continue with at least the default help tool

    async def execute_tool(self, tool_name: str, inputs: Dict[str, Any]) -> Any:
        """Execute a specific tool with the given inputs."""
        try:
            logger.info(f"Executing tool: {tool_name}")
            
            if tool_name not in self._tools_dict:
                error_msg = f"Tool not found: {tool_name}"
                logger.error(error_msg)
                raise Exception(error_msg)

            tool_data = self._tools_dict[tool_name]
            
            # Handle the help tool specially
            if tool_name == "help":
                return {
                    "tools": [
                        {
                            "name": name,
                            "description": data.get("description", "No description available"),
                            "tags": data.get("tags", [])
                        }
                        for name, data in self._tools_dict.items()
                    ]
                }
            
            endpoint = tool_data.get('endpoint')
            required_inputs = tool_data.get('required_inputs', [])

            # Validate required inputs
            for required_input in required_inputs:
                if required_input not in inputs:
                    error_msg = f"Missing required input: {required_input}"
                    logger.error(error_msg)
                    raise Exception(error_msg)

            logger.info(f"Making API call to endpoint: {endpoint}")
            async with httpx.AsyncClient() as client:
                response = await client.post(endpoint, json=inputs)
                response.raise_for_status()
                result = response.json()
                logger.info(f"Tool execution successful: {tool_name}")
                return result

        except httpx.HTTPError as e:
            error_msg = f"HTTP error executing tool {tool_name}: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            logger.error(error_msg)
            raise

# Example usage
if __name__ == "__main__":
    async def main():
        # Will use settings from .env file
        agent = DynamicAgent()
        result = await agent.run("What tools are available and what can they do?")
        print(result)
    
    asyncio.run(main())