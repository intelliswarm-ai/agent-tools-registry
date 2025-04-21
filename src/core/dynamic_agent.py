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
from src.core.errors import ToolExecutionError, ToolNotFoundError, ToolValidationError, ToolTimeoutError, ToolPermissionError
from src.core.logging import get_logger, setup_logging

class ToolSpec(BaseModel):
    """Specification for a tool in the registry"""
    name: str
    description: str
    endpoint: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    tags: List[str]

class DynamicAgent:
    """Agent that dynamically loads and executes tools based on configuration."""
    
    def __init__(self):
        """Initialize the agent with tools and configuration."""
        self.logger = get_logger(__name__)
        self.logger.info("Initializing DynamicAgent")
        
        # Initialize tools dictionary
        self.tools = {}
        
        # Load tools from configuration
        self.refresh_tools()
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0
        )
        
        # Initialize agent
        self.agent = self._create_agent()
        self.logger.info("DynamicAgent initialization completed")
        
    def _create_agent(self):
        """Create the agent with tools and memory."""
        self.logger.info("Creating agent with tools and memory")
        tools = [
            Tool(
                name=tool["name"],
                func=partial(self.execute_tool, tool["name"]),
                description=tool["description"]
            )
            for tool in self.tools.values()
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant that can use tools to help users."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_functions(x["intermediate_steps"]),
            "chat_history": lambda x: x["chat_history"],
        } | prompt | self.llm | OpenAIFunctionsAgentOutputParser()
        
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=True
        )
        
    def refresh_tools(self):
        """Refresh the list of available tools from the tools directory."""
        self.logger.info("Starting tool refresh process")
        tools_dir = os.path.join(os.path.dirname(__file__), "..", "tools", "definitions")
        self.logger.debug(f"Looking for tools in directory: {tools_dir}")
        
        if not os.path.exists(tools_dir):
            self.logger.error(f"Tools directory not found: {tools_dir}")
            raise FileNotFoundError(f"Tools directory not found: {tools_dir}")
            
        self.tools = {}
        for filename in os.listdir(tools_dir):
            if filename.endswith(".json"):
                self.logger.info(f"Processing tool file: {filename}")
                try:
                    with open(os.path.join(tools_dir, filename), "r") as f:
                        tool_config = json.load(f)
                        self.logger.debug(f"Loaded tool config: {tool_config}")
                        self._load_tool(tool_config)
                except json.JSONDecodeError as e:
                    self.logger.error(f"Error parsing tool file {filename}: {str(e)}")
                    continue
                except Exception as e:
                    self.logger.error(f"Error loading tool from {filename}: {str(e)}")
                    continue
                    
        self.logger.info(f"Tool refresh completed. Loaded {len(self.tools)} tools")
        
    def _load_tool(self, tool_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Load a tool from its configuration dictionary."""
        try:
            self.logger.info(f"Loading tool: {tool_config.get('name', 'unknown')}")
            
            # Validate required fields
            required_fields = ["name", "description", "inputs", "outputs"]
            missing_fields = [field for field in required_fields if field not in tool_config]
            if missing_fields:
                self.logger.warning(f"Tool missing required fields: {missing_fields}")
                return None
                
            # Add tool to registry
            self.tools[tool_config["name"]] = tool_config
            self.logger.info(f"Successfully loaded tool: {tool_config['name']}")
            return tool_config
            
        except Exception as e:
            self.logger.error(f"Error loading tool: {str(e)}", exc_info=True)
            return None
            
    async def execute_tool(self, tool_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with the given inputs."""
        self.logger.info(f"Executing tool: {tool_name}")
        self.logger.debug(f"Tool inputs: {inputs}")
        
        if tool_name not in self.tools:
            self.logger.error(f"Tool not found: {tool_name}")
            raise ToolNotFoundError(f"Tool not found: {tool_name}", tool_name=tool_name)
            
        tool = self.tools[tool_name]
        if not tool.get("enabled", True):
            self.logger.warning(f"Attempted to execute disabled tool: {tool_name}")
            raise ToolPermissionError(
                f"Tool is disabled: {tool_name}",
                tool_name=tool_name,
                required_permissions=[]
            )
            
        try:
            # Validate inputs
            self.logger.debug("Validating tool inputs")
            self._validate_inputs(tool, inputs)
            
            # Special handling for help tool - process locally
            if tool_name == 'help':
                self.logger.info("Processing help tool request")
                tools_list = self.list_tools()
                return {
                    "status": "success",
                    "tools": tools_list
                }
            
            # For trading tool, provide mock response for demo
            if tool_name == 'trading':
                self.logger.info("Processing trading tool request")
                operation = inputs.get('operation')
                symbol = inputs.get('symbol')
                quantity = inputs.get('quantity')
                
                # Mock response
                response = {
                    "status": "success",
                    "operation": operation,
                    "symbol": symbol,
                    "quantity": quantity,
                    "message": f"Successfully processed {operation} order for {quantity if quantity else ''} {symbol}"
                }
                self.logger.info(f"Trading tool response: {response}")
                return response
            
            # For other tools that need HTTP execution
            endpoint = tool.get("endpoint")
            if not endpoint:
                self.logger.error(f"Tool endpoint not configured for {tool_name}")
                return {
                    "status": "error",
                    "message": "Tool endpoint not configured"
                }
            
            # Execute tool via HTTP request
            timeout = tool.get("timeout", 30)
            try:
                self.logger.info(f"Making HTTP request to {endpoint}")
                async with httpx.AsyncClient() as client:
                    self.logger.debug(f"Request details - URL: {endpoint}, Inputs: {inputs}, Timeout: {timeout}")
                    response = await client.post(
                        endpoint,
                        json=inputs,
                        timeout=timeout
                    )
                    
                    if response.status_code != 200:
                        self.logger.error(f"Tool execution failed with status {response.status_code}: {response.text}")
                        return {
                            "status": "error",
                            "message": f"Request failed with status {response.status_code}"
                        }
                        
                    result = response.json()
                    self.logger.info(f"Tool execution successful: {tool_name}")
                    self.logger.debug(f"Tool response: {result}")
                    return result
            except httpx.TimeoutException:
                self.logger.error(f"Request timed out for tool {tool_name}")
                return {
                    "status": "error",
                    "message": f"Request timed out after {timeout} seconds"
                }
            except Exception as e:
                self.logger.error(f"HTTP request failed for tool {tool_name}: {str(e)}", exc_info=True)
                return {
                    "status": "error",
                    "message": f"Failed to execute request: {str(e)}"
                }
                
        except ToolValidationError as e:
            self.logger.error(f"Validation error for tool {tool_name}: {str(e)}")
            return {
                "status": "error",
                "message": f"Invalid inputs: {str(e)}"
            }
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"Tool execution failed: {str(e)}"
            }
            
    def _validate_inputs(self, tool: Dict[str, Any], inputs: Dict[str, Any]) -> None:
        """Validate tool inputs against schema."""
        self.logger.debug(f"Validating inputs for tool {tool['name']}")
        validation_errors = {}
        required_inputs = {
            name: schema for name, schema in tool["inputs"].items()
            if schema.get("required", False)
        }
        
        # Check for missing required inputs
        for name in required_inputs:
            if name not in inputs:
                validation_errors[name] = "missing required input"
                
        if validation_errors:
            self.logger.error(f"Validation errors: {validation_errors}")
            raise ToolValidationError(
                "Invalid tool inputs",
                tool_name=tool["name"],
                validation_errors=validation_errors
            )
            
    async def run(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Process a message and return a response."""
        self.logger.info(f"Processing message: {message}")
        self.logger.debug(f"Context: {context}")
        
        try:
            # Handle simple greetings directly
            message_lower = message.lower().strip()
            if message_lower in ['hi', 'hello', 'hey']:
                self.logger.info("Processing greeting message")
                greeting = "Hello! I'm an AI agent that can help you use various tools. Here are the available tools:\n\n"
                tools_list = self.list_tools()
                for tool in tools_list:
                    greeting += f"- {tool['name']}: {tool['description']}\n"
                self.logger.debug(f"Greeting response: {greeting}")
                return greeting

            # If asking for help or about available tools
            if 'help' in message_lower or 'tools' in message_lower or 'what can you do' in message_lower:
                self.logger.info("Processing help request")
                result = await self.execute_tool('help', {})
                if result.get("status") == "error":
                    self.logger.warning("Help tool failed, falling back to direct tool listing")
                    tools_list = self.list_tools()
                    response = "Here are the available tools:\n\n"
                    for tool in tools_list:
                        response += f"- {tool['name']}: {tool['description']}\n"
                    self.logger.debug(f"Help response: {response}")
                    return response
                response = "Here are the available tools:\n" + json.dumps(result.get("tools", []), indent=2)
                self.logger.debug(f"Help response: {response}")
                return response

            # Handle trading commands
            if any(cmd in message_lower for cmd in ['buy', 'sell', 'trade', 'status']):
                self.logger.info("Processing trading command")
                # Extract trading parameters from message
                operation = 'status'  # default operation
                if 'buy' in message_lower:
                    operation = 'buy'
                elif 'sell' in message_lower:
                    operation = 'sell'

                # Simple regex to extract symbol and quantity
                import re
                symbol_match = re.search(r'(?:of|for|symbol)\s+([A-Z]+)', message.upper())
                quantity_match = re.search(r'(\d+)\s+(?:shares?|units?)?', message)

                symbol = symbol_match.group(1) if symbol_match else None
                quantity = float(quantity_match.group(1)) if quantity_match else None

                if not symbol and operation != 'status':
                    self.logger.warning("No symbol provided for trading operation")
                    return "Please specify a trading symbol (e.g., AAPL, GOOGL, etc.)"

                inputs = {
                    "operation": operation,
                    "symbol": symbol
                }
                if quantity:
                    inputs["quantity"] = quantity

                self.logger.debug(f"Executing trading tool with inputs: {inputs}")
                result = await self.execute_tool('trading', inputs)
                
                if result.get("status") == "error":
                    self.logger.error(f"Trading operation failed: {result.get('message', 'Unknown error')}")
                    return f"Trading operation failed: {result.get('message', 'Unknown error')}"
                    
                response = (
                    f"Trading operation completed:\n"
                    f"Operation: {result.get('operation')}\n"
                    f"Symbol: {result.get('symbol')}\n"
                    f"Quantity: {result.get('quantity', 'N/A')}\n"
                    f"Status: {result.get('status')}\n"
                    f"Message: {result.get('message', '')}"
                )
                self.logger.debug(f"Trading response: {response}")
                return response

            # For other messages, use the agent
            self.logger.info("Using agent to process message")
            result = await self.agent.ainvoke({"input": message})
            self.logger.debug(f"Agent response: {result}")
            return result["output"]
            
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}", exc_info=True)
            return f"An error occurred while processing your message: {str(e)}"
            
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools."""
        self.logger.info("Listing available tools")
        return [
            {
                "name": name,
                "description": data.get("description", "No description available"),
                "tags": data.get("tags", []),
                "inputs": data.get("inputs"),
                "outputs": data.get("outputs")
            }
            for name, data in self.tools.items()
        ]