import pytest
import asyncio
from unittest.mock import Mock, patch
from dynamic_agent import DynamicAgent, ToolSpec
from config import Settings

# Sample tool data for testing
MOCK_TOOLS_DATA = [
    {
        "name": "calculator",
        "description": "Performs basic arithmetic operations",
        "endpoint": "http://localhost:8000/tools/calculator",
        "inputs": {
            "operation": {"type": "string", "description": "Operation to perform (add, subtract, multiply, divide)"},
            "numbers": {"type": "array", "description": "List of numbers to operate on"}
        },
        "outputs": {
            "result": {"type": "number", "description": "Result of the operation"}
        },
        "tags": ["math", "arithmetic"]
    },
    {
        "name": "weather",
        "description": "Gets weather information for a location",
        "endpoint": "http://localhost:8000/tools/weather",
        "inputs": {
            "location": {"type": "string", "description": "City name or coordinates"}
        },
        "outputs": {
            "temperature": {"type": "number", "description": "Current temperature"},
            "conditions": {"type": "string", "description": "Weather conditions"}
        },
        "tags": ["weather", "location"]
    }
]

# Mock settings for testing
MOCK_SETTINGS = {
    "openai_api_key": "sk-test-key-123",
    "openai_model": "gpt-3.5-turbo",
    "openai_temperature": 0,
    "tools_registry_url": "http://localhost:8000/tools"
}

@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    with patch('config.get_settings') as mock_get_settings:
        mock_get_settings.return_value = Settings(**MOCK_SETTINGS)
        yield mock_get_settings

@pytest.fixture
def mock_responses():
    """Mock HTTP responses for testing."""
    with patch('requests.get') as mock_get, patch('requests.post') as mock_post:
        # Mock tools registry response
        mock_get.return_value.json.return_value = MOCK_TOOLS_DATA
        mock_get.return_value.status_code = 200
        
        # Mock calculator tool response
        def mock_calculator_response(url, json):
            if 'calculator' in url:
                numbers = json.get('numbers', [])
                operation = json.get('operation')
                if operation == 'add':
                    result = sum(numbers)
                return {'result': result}
            return {}
        
        mock_post.side_effect = mock_calculator_response
        mock_post.return_value.status_code = 200
        
        yield mock_get, mock_post

@pytest.fixture
async def agent(mock_settings):
    """Create a DynamicAgent instance for testing."""
    agent = DynamicAgent(verbose=False)
    yield agent

@pytest.mark.asyncio
async def test_tool_fetching(agent, mock_responses, mock_settings):
    """Test that tools are correctly fetched and created."""
    tools = agent.tools
    assert len(tools) == len(MOCK_TOOLS_DATA)
    assert tools[0].name == "calculator"
    assert tools[1].name == "weather"

@pytest.mark.asyncio
async def test_tool_description_format(agent, mock_responses, mock_settings):
    """Test that tool descriptions are properly formatted."""
    tools = agent.tools
    calc_desc = tools[0].description
    
    assert "Performs basic arithmetic operations" in calc_desc
    assert "Inputs:" in calc_desc
    assert "Outputs:" in calc_desc
    assert "operation:" in calc_desc
    assert "numbers:" in calc_desc
    assert "result:" in calc_desc

@pytest.mark.asyncio
async def test_calculator_tool_execution(agent, mock_responses, mock_settings):
    """Test the execution of the calculator tool."""
    mock_get, mock_post = mock_responses
    
    task = 'Add the numbers 5 and 3'
    result = await agent.run(task)
    
    # Verify that the tool was called
    assert mock_post.called
    # The exact assertion for result depends on the LLM's response format
    # but we can check if the post was called with correct data
    call_args = mock_post.call_args_list[-1]
    assert 'calculator' in call_args[0][0]

@pytest.mark.asyncio
async def test_tool_refresh(agent, mock_responses, mock_settings):
    """Test that tools can be refreshed."""
    initial_tools = len(agent.tools)
    
    # Modify mock data to include a new tool
    modified_tools = MOCK_TOOLS_DATA + [{
        "name": "translator",
        "description": "Translates text between languages",
        "endpoint": "http://localhost:8000/tools/translate",
        "inputs": {"text": {"type": "string"}, "target_language": {"type": "string"}},
        "outputs": {"translated_text": {"type": "string"}},
        "tags": ["language", "translation"]
    }]
    
    mock_get, _ = mock_responses
    mock_get.return_value.json.return_value = modified_tools
    
    agent.refresh_tools()
    assert len(agent.tools) == len(modified_tools)
    assert agent.tools[-1].name == "translator"

@pytest.mark.asyncio
async def test_error_handling(agent, mock_responses, mock_settings):
    """Test error handling when a tool fails."""
    mock_get, mock_post = mock_responses
    mock_post.side_effect = Exception("Tool execution failed")
    
    result = await agent.run("Add 5 and 3")
    assert "Error" in result

@pytest.mark.asyncio
async def test_missing_api_key():
    """Test that the agent raises an error when API key is missing."""
    with patch('config.get_settings') as mock_get_settings:
        mock_get_settings.return_value = Settings(**{
            **MOCK_SETTINGS,
            "openai_api_key": ""  # Empty API key
        })
        
        with pytest.raises(ValueError) as exc_info:
            agent = DynamicAgent()
        
        assert "OpenAI API key is not properly configured" in str(exc_info.value)

if __name__ == "__main__":
    pytest.main(["-v", "test_dynamic_agent.py"]) 