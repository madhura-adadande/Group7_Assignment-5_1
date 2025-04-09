# backend/app.py

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# Import using absolute path from project root
from backend.langgraph.orchestrator import ResearchOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Rest of the code remains the same...

# Create FastAPI app
app = FastAPI(
    title="NVIDIA Research Assistant API",
    description="AI-powered research assistant for NVIDIA financial and technological insights"
)

# Request model for research query
class ResearchQuery(BaseModel):
    query: str
    year: Optional[int] = None
    quarter: Optional[int] = None
    agents: List[str] = ["rag", "snowflake", "websearch"]

# Research endpoint
@app.post("/research")
async def generate_research(request: ResearchQuery):
    """
    Generate a comprehensive research report about NVIDIA
    
    Parameters:
    - query: Research question
    - year (optional): Specific year to filter data
    - quarter (optional): Specific quarter to filter data
    - agents (optional): List of agents to use (rag, snowflake, websearch)
    """
    try:
        # Initialize orchestrator with selected agents
        orchestrator = ResearchOrchestrator(
            use_rag="rag" in request.agents,
            use_snowflake="snowflake" in request.agents,
            use_websearch="websearch" in request.agents
        )

        # Generate research report
        result = orchestrator.run(
            query=request.query,
            year=request.year,
            quarter=request.quarter
        )

        return result
    
    except Exception as e:
        # Detailed error handling
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating research report: {str(e)}"
        )

# Health check endpoint
@app.get("/health")
def health_check():
    """
    Simple health check endpoint
    """
    return {"status": "healthy", "message": "NVIDIA Research Assistant API is running"}

# Optional: Add more endpoints as needed
@app.get("/agents")
def list_available_agents():
    """
    List available research agents
    """
    return {
        "agents": [
            {
                "name": "rag",
                "description": "Historical data retrieval using Retrieval-Augmented Generation"
            },
            {
                "name": "snowflake",
                "description": "Financial metrics from Snowflake database"
            },
            {
                "name": "websearch",
                "description": "Latest news and insights from web sources"
            }
        ]
    }

# Run with: 
# uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000