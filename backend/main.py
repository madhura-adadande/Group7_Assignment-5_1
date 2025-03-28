from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from agents.langgraph_controller import run_agents
from fastapi.middleware.cors import CORSMiddleware
import traceback
import json
from langchain_core.messages import ToolMessage

app = FastAPI()

# 🛡️ Allow CORS for Streamlit frontend (optional but recommended)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to your frontend's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📦 Request Model for the unified research endpoint
class ResearchRequest(BaseModel):
    year: str
    quarter: List[str]
    query: str
    tools: List[str]  # ["vector_search", "snowflake_query", "web_search"]

@app.get("/")
def root():
    return {"message": "NVIDIA Research Assistant Backend is running."}

@app.post("/query_multiagent")
def query_research_agents(request: ResearchRequest):
    try:
        print(f"📥 Query: {request.query}")
        print(f"📅 Year: {request.year}, Quarter(s): {request.quarter}")
        print(f"🛠️ Selected Tools: {request.tools}")

        # Build the LangGraph DAG
        runnable = run_agents(
            tools=request.tools,
            year=request.year,
            quarter=request.quarter
        )

        # Run the agent workflow
        result = runnable.invoke({
            "input": request.query,
            "chat_history": [],
            "intermediate_steps": [],
            "year": request.year,
            "quarter": request.quarter,
            "agent_outcome": None
        })

        # Extract final ToolMessage
        final_message = result.get("intermediate_steps", [])[-1][1] if result.get("intermediate_steps") else None

        if isinstance(final_message, ToolMessage):
            content = final_message.content
            try:
                parsed = json.loads(content)  # Attempt to parse content as JSON
                final_answer = parsed.get("answer", content)
                sources = parsed.get("sources", [])
            except json.JSONDecodeError:
                final_answer = content  # If not JSON, just return the content
                sources = []
        else:
            final_answer = "⚠️ Unexpected message format"
            sources = []

        return {
            "query": request.query,
            "year": request.year,
            "quarter": request.quarter,
            "tools_used": request.tools,
            "final_answer": final_answer,
            "sources": sources
        }

    except Exception as e:
        print("❌ Error occurred in /query_multiagent")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
