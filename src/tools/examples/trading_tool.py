from typing import Any, Dict, Optional, Type, Literal
from pydantic import BaseModel, Field
import httpx

from src.tools.base import RegistryTool
from src.core.logging import get_logger
from src.core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()

class TradingToolInput(BaseModel):
    """Input schema for the trading tool."""
    operation: Literal["buy", "sell", "status"] = Field(
        description="The trading operation to perform"
    )
    symbol: str = Field(
        description="The trading symbol"
    )
    quantity: Optional[float] = Field(
        default=None,
        description="The quantity to trade"
    )

class TradingToolOutput(BaseModel):
    """Output schema for the trading tool."""
    status: str = Field(description="The status of the operation")
    message: str = Field(description="Detailed message about the operation")

class TradingTool(RegistryTool):
    """Trading tool for executing trading operations."""
    
    name = "trading"
    description = "Execute trading operations like buy, sell, and check status"
    tags = ["trading", "finance"]
    version = "1.0.0"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_endpoint = f"{settings.api_base_url}/api/v1/tools/trading"
    
    def _run(
        self, 
        operation: Literal["buy", "sell", "status"],
        symbol: str,
        quantity: Optional[float] = None
    ) -> Dict[str, Any]:
        """Execute a trading operation.
        
        Args:
            operation: The type of operation (buy/sell/status)
            symbol: The trading symbol
            quantity: The quantity to trade (optional for status checks)
        
        Returns:
            Dict containing the operation status and message
        """
        logger.info(f"Executing {operation} operation for {symbol}")
        
        # Prepare the request payload
        payload = {
            "operation": operation,
            "symbol": symbol
        }
        if quantity is not None:
            payload["quantity"] = quantity
            
        try:
            # Make the API call
            with httpx.Client() as client:
                response = client.post(self.api_endpoint, json=payload)
                response.raise_for_status()
                
                result = response.json()
                return TradingToolOutput(
                    status=result.get("status", "error"),
                    message=result.get("message", "No message provided")
                ).dict()
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error during trading operation: {str(e)}")
            return TradingToolOutput(
                status="error",
                message=f"Trading operation failed: {str(e)}"
            ).dict()
            
        except Exception as e:
            logger.error(f"Unexpected error during trading operation: {str(e)}")
            return TradingToolOutput(
                status="error",
                message=f"Unexpected error: {str(e)}"
            ).dict()
    
    async def _arun(
        self,
        operation: Literal["buy", "sell", "status"],
        symbol: str,
        quantity: Optional[float] = None
    ) -> Dict[str, Any]:
        """Async version of the trading operation."""
        logger.info(f"Executing async {operation} operation for {symbol}")
        
        # Prepare the request payload
        payload = {
            "operation": operation,
            "symbol": symbol
        }
        if quantity is not None:
            payload["quantity"] = quantity
            
        try:
            # Make the async API call
            async with httpx.AsyncClient() as client:
                response = await client.post(self.api_endpoint, json=payload)
                response.raise_for_status()
                
                result = response.json()
                return TradingToolOutput(
                    status=result.get("status", "error"),
                    message=result.get("message", "No message provided")
                ).dict()
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error during async trading operation: {str(e)}")
            return TradingToolOutput(
                status="error",
                message=f"Trading operation failed: {str(e)}"
            ).dict()
            
        except Exception as e:
            logger.error(f"Unexpected error during async trading operation: {str(e)}")
            return TradingToolOutput(
                status="error",
                message=f"Unexpected error: {str(e)}"
            ).dict()

    @property
    def args_schema(self) -> Type[BaseModel]:
        """Return the input schema for this tool."""
        return TradingToolInput
        
    @property
    def return_schema(self) -> Type[BaseModel]:
        """Return the output schema for this tool."""
        return TradingToolOutput 