from typing import List, Dict, Any, Optional
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import StringPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackManager
from pydantic import BaseModel, Field
import requests
import json
from functools import partial
from config import get_settings, verify_api_key

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
        verbose: bool = True
    ):
        settings = get_settings()
        
        if not verify_api_key():
            raise ValueError(
                "OpenAI API key is not properly configured. "
                "Please set it in your .env file or environment variables."
            )
        
        self.registry_url = registry_url or settings.tools_registry_url
        self.llm = ChatOpenAI(
            model=llm_model or settings.openai_model,
            temperature=temperature if temperature is not None else settings.openai_temperature,
            openai_api_key=settings.openai_api_key
        )
        self.tools = self._fetch_and_create_tools()
        self.memory = ConversationBufferMemory()
        self.verbose = verbose
        self.agent_executor = self._create_agent_executor()

    def _fetch_and_create_tools(self) -> List[Tool]:
        """Fetch tools from registry and convert them to LangChain Tool objects."""
        try:
            response = requests.get(self.registry_url)
            response.raise_for_status()
            tools_data = response.json()
            
            tools = []
            for tool_data in tools_data:
                spec = ToolSpec(**tool_data)
                tool_func = partial(self._execute_tool, spec.endpoint, spec.inputs)
                
                tools.append(
                    Tool(
                        name=spec.name,
                        description=self._create_tool_description(spec),
                        func=tool_func
                    )
                )
            return tools
        except Exception as e:
            raise Exception(f"Failed to fetch tools from registry: {str(e)}")

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
            # Parse the tool_input string into a dictionary matching the input_spec
            input_data = json.loads(tool_input) if isinstance(tool_input, str) else tool_input
            
            # Validate input against spec
            for required_input in input_spec.keys():
                if required_input not in input_data:
                    raise ValueError(f"Missing required input: {required_input}")
            
            # Make the API call
            response = requests.post(endpoint, json=input_data)
            response.raise_for_status()
            
            return json.dumps(response.json(), indent=2)
        except Exception as e:
            return f"Error executing tool: {str(e)}"

    def _create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor with the dynamic tools."""
        prompt = DynamicToolsPromptTemplate(
            template="",  # Template is defined in the class
            tools=self.tools,
            input_variables=["input", "intermediate_steps"]
        )

        llm_chain = LLMSingleActionAgent(
            llm_chain=self.llm,
            output_parser=self._parse_output,
            stop=["\nObservation:"],
            allowed_tools=[tool.name for tool in self.tools],
            prompt=prompt
        )

        return AgentExecutor.from_agent_and_tools(
            agent=llm_chain,
            tools=self.tools,
            verbose=self.verbose,
            memory=self.memory
        )

    def _parse_output(self, text: str) -> AgentAction | AgentFinish:
        """Parse the output of the LLM to determine the next action."""
        if "Action:" not in text:
            return AgentFinish(
                return_values={"output": text},
                log=text
            )

        action_match = text.split("Action:")[1].strip()
        action_input_match = text.split("Action Input:")[1].strip()
        
        return AgentAction(
            tool=action_match,
            tool_input=action_input_match,
            log=text
        )

    async def run(self, task: str) -> str:
        """Run the agent on a given task."""
        try:
            result = await self.agent_executor.arun(input=task)
            return result
        except Exception as e:
            return f"Error running agent: {str(e)}"

    def refresh_tools(self):
        """Refresh the available tools from the registry."""
        self.tools = self._fetch_and_create_tools()
        self.agent_executor = self._create_agent_executor()

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Will use settings from .env file
        agent = DynamicAgent()
        result = await agent.run("What tools are available and what can they do?")
        print(result)
    
    asyncio.run(main())