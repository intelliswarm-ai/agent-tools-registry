from typing import List, Dict, Any, Optional
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import StringPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackManager
from pydantic import BaseModel, Field
import requests
import json
from functools import partial
from config import get_settings, verify_api_key
import time
import logging
import os
import httpx

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

class DynamicToolsPromptTemplate(StringPromptTemplate):
    """Custom prompt template that includes dynamic tools information."""
    
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps", [])
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"\nAction: {action}\nObservation: {observation}\n"
            
        tools_str = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        
        return f"""You are an AI agent that can use various tools to accomplish tasks.

Available tools:
{tools_str}

Previous actions and observations:
{thoughts}

Current task: {kwargs["input"]}

Think through this step-by-step:
1) First, analyze what needs to be done
2) Then, choose the most appropriate tool
3) Finally, use the tool with the correct parameters

Answer in the following format:
Thought: your thought process
Action: the tool to use
Action Input: the input to the tool
"""

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
        self.memory = ConversationBufferMemory()
        self._tools = None
        self._agent_executor = None
        self.tools = {}
        self.refresh_tools()
        logger.info("DynamicAgent initialization complete")

    @property
    def tools(self):
        """Lazy initialization of tools with retry mechanism."""
        if self._tools is None:
            logger.info("Fetching tools for the first time...")
            self._tools = self._fetch_and_create_tools()
        return self._tools

    @property
    def agent_executor(self):
        """Lazy initialization of agent executor."""
        if self._agent_executor is None:
            logger.info("Creating agent executor...")
            self._agent_executor = self._create_agent_executor()
        return self._agent_executor

    def _fetch_and_create_tools(self) -> List[Tool]:
        """Fetch tools from registry and convert them to LangChain Tool objects."""
        retries = 0
        last_exception = None
        
        while retries < self.max_retries:
            try:
                logger.info(f"Attempting to fetch tools (attempt {retries + 1}/{self.max_retries})")
                response = requests.get(self.registry_url)
                response.raise_for_status()
                tools_data = response.json()
                
                tools = []
                for tool_data in tools_data:
                    logger.info(f"Creating tool: {tool_data.get('name')}")
                    spec = ToolSpec(**tool_data)
                    tool_func = partial(self._execute_tool, spec.endpoint, spec.inputs)
                    
                    tools.append(
                        Tool(
                            name=spec.name,
                            description=self._create_tool_description(spec),
                            func=tool_func
                        )
                    )
                logger.info(f"Successfully created {len(tools)} tools")
                return tools
            except Exception as e:
                last_exception = e
                retries += 1
                logger.warning(f"Failed to fetch tools (attempt {retries}/{self.max_retries}): {str(e)}")
                if retries < self.max_retries:
                    logger.info(f"Waiting {self.retry_delay} seconds before retrying...")
                    time.sleep(self.retry_delay)
                    continue
                logger.error(f"Failed to fetch tools after {self.max_retries} attempts")
                raise Exception(f"Failed to fetch tools from registry after {self.max_retries} attempts: {str(last_exception)}")

    def _create_tool_description(self, spec: ToolSpec) -> str:
        """Create a detailed description for a tool including its inputs and outputs."""
        desc = f"{spec.description}\n"
        desc += "\nInputs:\n"
        for name, details in spec.inputs.items():
            desc += f"- {name}: {details.get('description', 'No description')}\n"
        desc += "\nOutputs:\n"
        for name, details in spec.outputs.items():
            desc += f"- {name}: {details.get('description', 'No description')}\n"
        return desc

    def _execute_tool(self, endpoint: str, input_spec: Dict[str, Any], tool_input: str) -> str:
        """Execute a tool by calling its endpoint with the provided input."""
        try:
            logger.info(f"Executing tool with endpoint: {endpoint}")
            # Parse the tool_input string into a dictionary matching the input_spec
            input_data = json.loads(tool_input) if isinstance(tool_input, str) else tool_input
            
            # Validate input against spec
            for required_input in input_spec.keys():
                if required_input not in input_data:
                    logger.error(f"Missing required input: {required_input}")
                    raise ValueError(f"Missing required input: {required_input}")
            
            # Make the API call
            logger.info(f"Making API call with input: {input_data}")
            response = requests.post(endpoint, json=input_data)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Tool execution successful: {result}")
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Error executing tool: {str(e)}")
            return f"Error executing tool: {str(e)}"

    def _create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor with the dynamic tools."""
        logger.info("Creating agent executor with prompt template")
        prompt = DynamicToolsPromptTemplate(
            template="",  # Template is defined in the class
            tools=self.tools,
            input_variables=["input", "intermediate_steps"]
        )

        logger.info("Creating LLMSingleActionAgent")
        llm_chain = LLMSingleActionAgent(
            llm_chain=self.llm,
            output_parser=self._parse_output,
            stop=["\nObservation:"],
            allowed_tools=[tool.name for tool in self.tools],
            prompt=prompt
        )

        logger.info("Creating AgentExecutor")
        return AgentExecutor.from_agent_and_tools(
            agent=llm_chain,
            tools=self.tools,
            verbose=self.verbose,
            memory=self.memory
        )

    def _parse_output(self, text: str) -> AgentAction | AgentFinish:
        """Parse the output of the LLM to determine the next action."""
        logger.info(f"Parsing LLM output: {text}")
        if "Action:" not in text:
            logger.info("No action found in output, finishing")
            return AgentFinish(
                return_values={"output": text},
                log=text
            )

        action_match = text.split("Action:")[1].strip()
        action_input_match = text.split("Action Input:")[1].strip()
        
        logger.info(f"Parsed action: {action_match} with input: {action_input_match}")
        return AgentAction(
            tool=action_match,
            tool_input=action_input_match,
            log=text
        )

    async def run(self, input_text: str) -> str:
        """Run the agent with the given input."""
        try:
            logger.info(f"Starting agent run with input: {input_text}")
            start_time = time.time()

            # Log the current state of tools
            logger.info(f"Available tools: {list(self.tools.keys())}")

            # Here you would typically have your agent's logic to:
            # 1. Parse the input
            logger.info("Parsing input...")
            
            # 2. Decide which tool to use
            logger.info("Determining appropriate tool...")
            
            # 3. Execute the tool
            logger.info("Executing tool...")
            
            # 4. Process the result
            logger.info("Processing result...")

            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"Agent run completed in {execution_time:.2f} seconds")
            
            return f"Processed input: {input_text}"
            
        except Exception as e:
            logger.error(f"Error in agent run: {str(e)}", exc_info=True)
            raise

    def refresh_tools(self):
        """Refresh the available tools from the registry."""
        try:
            logger.info("Starting tools refresh")
            tools_dir = "tools"
            self.tools = {}
            
            if not os.path.exists(tools_dir):
                logger.error(f"Tools directory not found: {tools_dir}")
                raise Exception(f"Tools directory not found: {tools_dir}")

            for tool_file in os.listdir(tools_dir):
                if tool_file.endswith(".json"):
                    logger.info(f"Processing tool file: {tool_file}")
                    with open(os.path.join(tools_dir, tool_file), 'r') as f:
                        tool_data = json.load(f)
                        tool_name = tool_data.get('name')
                        if tool_name:
                            logger.info(f"Adding tool: {tool_name}")
                            self.tools[tool_name] = tool_data
                        else:
                            logger.warning(f"Skipping tool file {tool_file} - missing name")

            logger.info(f"Tools refresh completed. Total tools: {len(self.tools)}")
        except Exception as e:
            logger.error(f"Error refreshing tools: {str(e)}")
            raise

    async def execute_tool(self, tool_name: str, inputs: Dict[str, Any]) -> Any:
        """Execute a specific tool with the given inputs."""
        try:
            logger.info(f"Executing tool: {tool_name}")
            
            if tool_name not in self.tools:
                error_msg = f"Tool not found: {tool_name}"
                logger.error(error_msg)
                raise Exception(error_msg)

            tool_data = self.tools[tool_name]
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
    import asyncio
    
    async def main():
        # Will use settings from .env file
        agent = DynamicAgent()
        result = await agent.run("What tools are available and what can they do?")
        print(result)
    
    asyncio.run(main())