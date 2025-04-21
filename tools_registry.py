from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy import create_engine, Column, String, JSON, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./tools.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ToolDB(Base):
    """Database model for tools"""
    __tablename__ = "tools"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(String)
    endpoint = Column(String)
    inputs = Column(JSON)
    outputs = Column(JSON)
    tags = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

class ToolCreate(BaseModel):
    """Schema for creating a tool"""
    name: str
    description: str
    endpoint: str
    inputs: dict
    outputs: dict
    tags: List[str]

class ToolResponse(ToolCreate):
    """Schema for tool response"""
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

router = APIRouter(prefix="/tools", tags=["tools"])

def get_db():
    """Dependency for database sessions"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/", response_model=ToolResponse)
async def create_tool(tool: ToolCreate, db: Session = Depends(get_db)):
    """Register a new tool"""
    try:
        db_tool = ToolDB(**tool.model_dump())
        db.add(db_tool)
        db.commit()
        db.refresh(db_tool)
        return db_tool
    except Exception as e:
        logger.error(f"Error creating tool: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/", response_model=List[ToolResponse])
async def list_tools(
    skip: int = 0,
    limit: int = 100,
    search: Optional[str] = None,
    tags: Optional[List[str]] = Query(None),
    db: Session = Depends(get_db)
):
    """List tools with optional filtering"""
    query = db.query(ToolDB)
    
    if search:
        query = query.filter(
            ToolDB.name.ilike(f"%{search}%") | 
            ToolDB.description.ilike(f"%{search}%")
        )
    
    if tags:
        # SQLite JSON filtering is limited, so we do it in Python
        tools = query.all()
        tools = [tool for tool in tools if any(tag in tool.tags for tag in tags)]
        return tools[skip:skip + limit]
    
    return query.offset(skip).limit(limit).all()

@router.get("/{tool_id}", response_model=ToolResponse)
async def get_tool(tool_id: int, db: Session = Depends(get_db)):
    """Get a specific tool by ID"""
    tool = db.query(ToolDB).filter(ToolDB.id == tool_id).first()
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    return tool

@router.put("/{tool_id}", response_model=ToolResponse)
async def update_tool(
    tool_id: int, 
    tool: ToolCreate, 
    db: Session = Depends(get_db)
):
    """Update a tool"""
    db_tool = db.query(ToolDB).filter(ToolDB.id == tool_id).first()
    if not db_tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    for key, value in tool.model_dump().items():
        setattr(db_tool, key, value)
    
    db.commit()
    db.refresh(db_tool)
    return db_tool

@router.delete("/{tool_id}")
async def delete_tool(tool_id: int, db: Session = Depends(get_db)):
    """Delete a tool"""
    db_tool = db.query(ToolDB).filter(ToolDB.id == tool_id).first()
    if not db_tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    db.delete(db_tool)
    db.commit()
    return {"status": "success", "message": f"Tool {tool_id} deleted"}
