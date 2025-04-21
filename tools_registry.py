from fastapi import APIRouter
import json
import os
from pydantic import BaseModel
from typing import List

router = APIRouter()

TOOLS_DIR = "tools"

class ToolMetadata(BaseModel):
    name: str
    description: str
    inputs: dict
    outputs: dict
    tags: List[str]
    endpoint: str

@router.get("/tools", response_model=List[ToolMetadata])
def list_tools():
    tools = []
    for tool_name in os.listdir(TOOLS_DIR):
        meta_path = os.path.join(TOOLS_DIR, tool_name, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                tools.append(json.load(f))
    return tools
