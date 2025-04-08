from typing import TypedDict, Annotated, Optional, List, Any
import operator
from functools import partial
from dotenv import load_dotenv
import os
import json
from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

# === Load Environment ===
load_dotenv()

# === Agent State ===
class AgentState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    intermediate_steps: Annotated[List[AgentAction], operator.add]
    year: Optional[str]
    quarter: Optional[List[str]]

# === Import Tools ===
from agents.rag_agent.rag_tool import retrieve_rag_chunks
from agents.snowflake_agent.snowflake_tool import snowflake_query_tool
from agents.web_agent.web_tool import web_search_tool
from langchain_core.tools import tool

@tool("final_answer")
def final_answer(
    research_steps: Any,
    historical_performance: str,
    financial_analysis: str,
    industry_insights: str,
    summary: str,
    sources: Any,
    analysis_type: str = "financial_summary"
):
    """
    Produces a full-length research report combining all agent outputs.
    """
    if isinstance(research_steps, list):
        steps = "\n".join(f"- {step}" for step in research_steps)
    else:
        steps = str(research_steps)

    if isinstance(sources, list):
        sources_str = "\n".join(f"- {src}" for src in sources)
    else:
        sources_str = str(sources)

    return {
        "research_steps": steps,
        "historical_performance": historical_performance,
        "financial_analysis": financial_analysis,
        "industry_insights": industry_insights,
        "summary": summary,
        "analysis_type": analysis_type,
        "sources": sources_str,
    }

# === Tool Mapping ===
tool_map = {
    "rag_retrieve_chunks": retrieve_rag_chunks,
    "snowflake_query": snowflake_query_tool,
    "web_search": web_search_tool,
    "final_answer": final_answer,
}

# === Build Scratchpad ===
def build_scratchpad(steps: List[AgentAction]) -> str:
    formatted = []
    for step in steps:
        formatted.append(
            f"Tool: {step.tool}\nInput: {step.tool_input}\nOutput: {step.log}"
        )
    return "\n---\n".join(formatted)[:12000]

# === Initialize Oracle Agent ===
def initialize_oracle(tools: List[str], year: Optional[str], quarter: Optional[List[str]]):
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a multi-agent NVIDIA research assistant.
You have access to tools for:
- `rag_retrieve_chunks`: Retrieve historical report chunks using Pinecone vector DB.
- `snowflake_query`: Get structured financial summaries from Snowflake.
- `web_search`: Find real-time trends about NVIDIA using the web.
- `final_answer`: Combine all tool outputs into a cohesive report.

Guidelines:
- Do NOT call the same tool twice with the same input.
- Do NOT use any tool more than twice.
- Use `final_answer` only after all useful tools are invoked.
- Prefer diversity in source types before concluding.
- Do NOT format numbers with bold or italics.

Context:
- Year: {year or "Not provided"}
- Quarter: {', '.join(quarter) if quarter else "Not provided"}
"""),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
        ("assistant", "scratchpad: {scratchpad}")
    ])

    selected_tools = [tool_map[name] for name in tools]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    oracle = (
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
            "scratchpad": lambda x: build_scratchpad(x["intermediate_steps"])
        }
        | prompt
        | llm.bind_tools(selected_tools, tool_choice="any")
    )
    return oracle

# === Run Oracle ===

def run_oracle(state: AgentState, oracle: Runnable) -> AgentState:
    output = oracle.invoke(state)
    tool_call = output.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    # Prevent duplicate tool calls with same input
    for step in state["intermediate_steps"]:
        if step.tool == tool_name and step.tool_input == tool_args:
            print(f"[ORACLE] ⚠️ Tool '{tool_name}' already called with same input. Forcing final_answer.")

            steps_formatted = [f"Tool: {s.tool}, input: {s.tool_input}" for s in state["intermediate_steps"]]
            logs = {s.tool: s.log for s in state["intermediate_steps"]}
            analysis_type = logs.get("analysis_type", "financial_summary")

            final_output = {
                "research_steps": steps_formatted,
                "historical_performance": logs.get("rag_retrieve_chunks", "N/A"),
                "financial_analysis": logs.get("snowflake_query", "N/A"),
                "industry_insights": logs.get("web_search", "N/A"),
                "summary": "Here is a summary based on the tools used so far.",
                "sources": ["Pinecone", "Snowflake", "Web Search"],
                "analysis_type": analysis_type,
            }

            return {
                **state,
                "intermediate_steps": state["intermediate_steps"] + [AgentAction(
                    tool="final_answer",
                    tool_input=final_output,
                    log=json.dumps(final_output)  # Ensure log is always a JSON string
                )]
            }

    return {
        **state,
        "intermediate_steps": state["intermediate_steps"] + [AgentAction(
            tool=tool_name,
            tool_input=tool_args,
            log="TBD"
        )]
    }


# === Run Tool ===


def run_tool(state: AgentState) -> AgentState:
    current_action = state["intermediate_steps"][-1]
    tool_fn = tool_map[current_action.tool]
    tool_args = current_action.tool_input

    if not isinstance(tool_args, dict):
        tool_args = {"query": str(tool_args)}

    if current_action.tool in ["rag_retrieve_chunks", "snowflake_query"]:
        tool_args["year"] = state.get("year")
        tool_args["quarter"] = state.get("quarter")

    if current_action.tool == "rag_retrieve_chunks" and isinstance(tool_args.get("quarter"), list):
        tool_args["quarter"] = tool_args["quarter"][0]

    if current_action.tool == "snowflake_query":
        if "input" in tool_args:
            tool_args["query"] = tool_args.pop("input")
        tool_args["query"] = tool_args.get("query", "Give me a financial analysis")

    # ✅ FIX: invoke with flat dictionary
    result = tool_fn.invoke(tool_args)

    log_str = result if isinstance(result, str) else json.dumps(result)
    if current_action.tool == "rag_retrieve_chunks" and isinstance(log_str, str) and len(log_str) > 8000:
        log_str = log_str[:8000]

    updated_action = AgentAction(
        tool=current_action.tool,
        tool_input=tool_args,
        log=log_str
    )

    return {
        **state,
        "intermediate_steps": state["intermediate_steps"] + [updated_action]
    }


# === Routing Logic ===
def route_agent(state: AgentState) -> str:
    try:
        return state["intermediate_steps"][-1].tool
    except Exception:
        return "final_answer"

# === LangGraph Assembly ===
def build_graph(oracle: Runnable) -> Runnable:
    tools = ["rag_retrieve_chunks", "snowflake_query", "web_search", "final_answer"]
    graph = StateGraph(AgentState)

    graph.add_node("oracle", partial(run_oracle, oracle=oracle))
    for tool in tools:
        graph.add_node(tool, run_tool)

    graph.set_entry_point("oracle")
    graph.add_conditional_edges("oracle", route_agent)

    for tool in tools:
        if tool == "final_answer":
            graph.add_edge(tool, END)
        else:
            graph.add_edge(tool, "oracle")

    return graph.compile()

# === Public Entry Point ===
def run_research_agent(tools: List[str], year: Optional[str] = None, quarter: Optional[List[str]] = None) -> Runnable:
    oracle = initialize_oracle(tools, year, quarter)
    return build_graph(oracle)