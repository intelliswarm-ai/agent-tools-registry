from typing import Type, Optional
from pydantic import BaseModel, Field
from src.tools.base import RegistryTool
from langchain.utilities import WikipediaAPIWrapper

class WikipediaInput(BaseModel):
    """Input schema for the Wikipedia tool."""
    query: str = Field(
        ...,
        description="The topic to look up on Wikipedia"
    )
    language: str = Field(
        default="en",
        description="The language code for Wikipedia (e.g., 'en' for English)"
    )

class WikipediaOutput(BaseModel):
    """Output schema for the Wikipedia tool."""
    title: str = Field(..., description="The title of the Wikipedia article")
    summary: str = Field(..., description="Summary of the Wikipedia article")
    url: str = Field(..., description="URL of the Wikipedia article")

class WikipediaTool(RegistryTool):
    """A tool for looking up information on Wikipedia."""
    
    name: str = "wikipedia"
    description: str = "Look up information about a topic on Wikipedia"
    tags: list[str] = ["search", "knowledge", "encyclopedia"]
    version: str = "1.0.0"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wiki = WikipediaAPIWrapper()
    
    def _run(self, query: str, language: str = "en") -> dict:
        """Execute the Wikipedia lookup tool."""
        try:
            # Set the language
            self.wiki.wiki_client.language = language
            
            # Get the page
            page = self.wiki.run(query)
            if not page:
                raise ValueError(f"No Wikipedia article found for: {query}")
                
            # Get the URL
            url = f"https://{language}.wikipedia.org/wiki/{query.replace(' ', '_')}"
            
            return {
                "title": query,
                "summary": page,
                "url": url
            }
        except Exception as e:
            raise ValueError(f"Failed to fetch Wikipedia article: {str(e)}")

    args_schema: Type[BaseModel] = WikipediaInput
    return_schema: Type[BaseModel] = WikipediaOutput 