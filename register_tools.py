import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample tools to register
tools = [
    {
        "name": "get_positions",
        "description": "Get current positions from the trading account",
        "endpoint": "/get_positions",  # Endpoint relative to base URL
        "inputs": {},
        "outputs": {
            "positions": {
                "type": "array",
                "description": "List of current positions"
            }
        },
        "tags": ["trading", "positions"]
    },
    {
        "name": "place_trade",
        "description": "Place a trade order",
        "endpoint": "/place_trade",  # Endpoint relative to base URL
        "inputs": {
            "symbol": {
                "type": "string",
                "description": "Trading symbol (e.g., AAPL)"
            },
            "side": {
                "type": "string",
                "description": "buy or sell"
            },
            "qty": {
                "type": "number",
                "description": "Quantity to trade"
            },
            "type": {
                "type": "string",
                "description": "Order type (market, limit)"
            },
            "limit_price": {
                "type": "number",
                "description": "Limit price for limit orders"
            }
        },
        "outputs": {
            "status": {
                "type": "integer",
                "description": "HTTP status code"
            },
            "order_id": {
                "type": "string",
                "description": "ID of the placed order"
            }
        },
        "tags": ["trading", "orders"]
    }
]

def register_tools():
    """Register tools with the registry."""
    base_url = "http://localhost:8000/tools"
    
    # First, let's clear existing tools to avoid duplicates
    try:
        existing_tools = requests.get(f"{base_url}/").json()
        for tool in existing_tools:
            if 'id' in tool:
                logger.info(f"Deleting existing tool: {tool['name']}")
                requests.delete(f"{base_url}/{tool['id']}")
    except Exception as e:
        logger.warning(f"Error clearing existing tools: {str(e)}")
    
    # Register new tools
    for tool in tools:
        try:
            logger.info(f"Registering tool: {tool['name']}")
            response = requests.post(f"{base_url}/", json=tool)
            response.raise_for_status()
            logger.info(f"Successfully registered tool: {tool['name']}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to register tool {tool['name']}: {str(e)}")
            logger.error(f"Response content: {e.response.content if hasattr(e, 'response') else 'No response content'}")

if __name__ == "__main__":
    register_tools() 