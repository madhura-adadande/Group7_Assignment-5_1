from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Any
import traceback
import json

from agents.controller import run_research_agent
from langchain_core.messages import HumanMessage

app = FastAPI()

# === CORS for Streamlit ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Request & Response Schema ===
class NVIDIARequest(BaseModel):
    query: str
    year: Optional[str]
    quarter: Optional[List[str]]
    tools: List[str]

class ResearchResponse(BaseModel):
    answer: Any

@app.get("/")
def root():
    return {"message": "NVIDIA Research Assistant API is running."}

@app.post("/query_research_agent", response_model=ResearchResponse)
async def query_nvidia_documents(request: NVIDIARequest):
    try:
        # 🚀 Run LangGraph agent
        agent = run_research_agent(
            tools=request.tools,
            year=request.year,
            quarter=request.quarter
        )

        initial_state = {
            "input": request.query,
            "chat_history": [HumanMessage(content=request.query)],
            "intermediate_steps": [],
            "year": request.year,
            "quarter": request.quarter
        }

        final_state = agent.invoke(initial_state)

        # 🔍 DEBUG: Print final state with intermediate steps
        print("\n🧠 FINAL INTERMEDIATE STEPS:")
        for step in final_state["intermediate_steps"]:
            print(f"➡️ Tool: {step.tool}")
            print(f"📝 Input: {step.tool_input}")
            print(f"📤 Output: {step.log[:300]}..." if isinstance(step.log, str) else json_dump(step.log))
            print("---")

        # 🧠 Extract last step
        last_step = final_state["intermediate_steps"][-1]
        result = last_step.log

        # ✅ If final_answer tool was used, return the full structured output
        if last_step.tool == "final_answer":
            try:
                parsed_result = json.loads(result)
                return {"answer": parsed_result}
            except Exception:
                return {"answer": {"summary": str(result)}}

        # ✅ Fallback: return other tool output as plain summary
        return {"answer": {"summary": str(result)}}

    except Exception as e:
        print("❌ Exception in /query_research_agent:")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )

# === Optional JSON Dump Helper ===
def json_dump(obj):
    try:
        return json.dumps(obj, indent=2)
    except:
        return str(obj)