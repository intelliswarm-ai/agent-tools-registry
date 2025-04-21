from typing import Any, Dict, List, Type
from pydantic import BaseModel, Field
import httpx

from src.tools.base import RegistryTool
from src.core.logging import get_logger
from src.core.config import get_settings

logger = get_logger(__name__)
settings = get_settings()

class WebSearchInput(BaseModel):
    """Input schema for the web search tool."""
    query: str = Field(
        description="The search query to look up on the web"
    )

class SearchResult(BaseModel):
    """Schema for a single search result."""
    title: str = Field(description="Title of the search result")
    snippet: str = Field(description="Text snippet from the search result")
    url: str = Field(description="URL of the search result")

class WebSearchOutput(BaseModel):
    """Output schema for the web search tool."""
    results: List[SearchResult] = Field(description="List of search results")
    total_results: int = Field(description="Total number of results found")

class WebSearchTool(RegistryTool):
    """Tool for searching the web for current information."""
    
    name = "web_search"
    description = "Search the web for current information on any topic"
    tags = ["search", "web", "information"]
    version = "1.0.0"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.search_endpoint = f"{settings.api_base_url}/api/v1/search"
    
    def _run(self, query: str) -> Dict[str, Any]:
        """Execute a web search.
        
        Args:
            query: The search query to look up
            
        Returns:
            Dictionary containing search results
        """
        logger.info(f"Executing web search for: {query}")
        
        try:
            # Make the API call to search service
            with httpx.Client() as client:
                response = client.get(
                    self.search_endpoint,
                    params={"q": query},
                    timeout=10.0
                )
                response.raise_for_status()
                
                data = response.json()
                results = [
                    SearchResult(
                        title=result["title"],
                        snippet=result["snippet"],
                        url=result["url"]
                    )
                    for result in data.get("results", [])
                ]
                
                return WebSearchOutput(
                    results=results,
                    total_results=len(results)
                ).dict()
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error during web search: {str(e)}")
            return WebSearchOutput(
                results=[],
                total_results=0
            ).dict()
            
        except Exception as e:
            logger.error(f"Unexpected error during web search: {str(e)}")
            return WebSearchOutput(
                results=[],
                total_results=0
            ).dict()
    
    async def _arun(self, query: str) -> Dict[str, Any]:
        """Async version of the web search."""
        logger.info(f"Executing async web search for: {query}")
        
        try:
            # Make the async API call
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.search_endpoint,
                    params={"q": query},
                    timeout=10.0
                )
                response.raise_for_status()
                
                data = response.json()
                results = [
                    SearchResult(
                        title=result["title"],
                        snippet=result["snippet"],
                        url=result["url"]
                    )
                    for result in data.get("results", [])
                ]
                
                return WebSearchOutput(
                    results=results,
                    total_results=len(results)
                ).dict()
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error during async web search: {str(e)}")
            return WebSearchOutput(
                results=[],
                total_results=0
            ).dict()
            
        except Exception as e:
            logger.error(f"Unexpected error during async web search: {str(e)}")
            return WebSearchOutput(
                results=[],
                total_results=0
            ).dict()

    @property
    def args_schema(self) -> Type[BaseModel]:
        """Return the input schema for this tool."""
        return WebSearchInput
        
    @property
    def return_schema(self) -> Type[BaseModel]:
        """Return the output schema for this tool."""
        return WebSearchOutput 