# Define the LangGraph state and agent logic
from typing import TypedDict, List, Optional, Any
from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.agents.output_parsers import ToolsAgentOutputParser
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.agent import AgentFinish
from langchain_core.messages import AIMessage

# ✅ Import tools
from agents.rag_agent.rag_tool import vector_search_tool
from agents.snowflake_agent.snowflake_tool import snowflake_query_tool
from agents.web_agent.web_tool import web_search_tool

# 🧱 1. Define the persistent Agent State
class AgentState(TypedDict):
    input: str
    chat_history: List
    intermediate_steps: List
    year: str
    quarter: List[str]
    agent_outcome: Optional[Any]

# 🧠 2. Final answer tool
@tool("final_answer")
def final_answer_tool(input: str) -> str:
    """Used to provide the final research answer after all tools are done."""
    return f"📝 Final Answer:\n\n{input}"

# 🧠 3. System prompt and agent logic
def init_research_agent(selected_tools):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, model_kwargs={"tool_choice": "auto"})

    tool_map = {
        "vector_search": vector_search_tool,
        "snowflake_query": snowflake_query_tool,
        "web_search": web_search_tool,
        "final_answer": final_answer_tool
    }

    tools = [tool_map[name] for name in selected_tools]

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You're a multi-agent NVIDIA research assistant.\n"
         "You have access to RAG (historical), Snowflake (valuation), and Web (real-time) tools.\n"
         "Make sure your answer is specific to the user's year and quarter filters: {year}, {quarter}.\n"
         "Do not use a tool more than twice."),
        ("user", "{input}"),
        MessagesPlaceholder("chat_history"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = (
        prompt
        | llm.bind_tools(tools)
        | ToolsAgentOutputParser()
    )

    return agent, tools

# 🔁 4. Agent decides next action
def run_oracle(state: AgentState, agent) -> AgentState:
    # Convert intermediate steps to proper OpenAI tool format
    tool_messages = format_to_openai_functions(state["intermediate_steps"])

    # Print out tool messages and chat history for debugging
    print(f"Tool Messages: {tool_messages}")
    print(f"Chat History: {state['chat_history']}")

    # Format messages for OpenAI - ensure proper sequencing of tool calls and responses
    formatted_messages = []

    # Add existing chat history to the formatted messages
    for msg in state["chat_history"]:
        if isinstance(msg, ToolMessage):
            formatted_messages.append(msg)  # Add any tool messages in chat history

    # Add tool calls in the proper order, followed by tool responses
    for msg in tool_messages:
        if isinstance(msg, ToolMessage):
            formatted_messages.append(msg)  # Add the tool responses after tool calls

    # Call the agent with the properly formatted chat history
    response = agent.invoke({
        "input": state["input"],
        "chat_history": formatted_messages,  # Ensure the properly formatted chat history
        "agent_scratchpad": tool_messages,  # Keep track of scratchpad for intermediate steps
        "year": state["year"],
        "quarter": state["quarter"]
    })

    return {
        **state,
        "agent_outcome": response  # This response now contains the correct tool_call-response sequence
    }

# 🔧 5. Run selected tool
def run_tool(state: AgentState) -> AgentState:
    if not state.get("agent_outcome"):
        raise ValueError("❌ Missing 'agent_outcome'. Ensure oracle runs before tools.")

    outcome = state["agent_outcome"]

    if isinstance(outcome, list):
        outcome = outcome[0]

    if isinstance(outcome, AgentFinish):
        return state

    tool_name = outcome.tool
    tool_input = outcome.tool_input
    tool_call_id = outcome.tool_call_id

    print(f"⚙️ Running Tool: {tool_name} with args {tool_input}")

    tool_map = {
        "vector_search": vector_search_tool,
        "snowflake_query": snowflake_query_tool,
        "web_search": web_search_tool,
        "final_answer": final_answer_tool
    }

    try:
        result = tool_map[tool_name].invoke(tool_input)
    except Exception as e:
        print(f"❌ Tool {tool_name} failed: {e}")
        result = f"Tool {tool_name} encountered an error: {str(e)}"

    # ✅ Construct correct messages for LangChain
    ai_msg = AIMessage(
        content=f"Tool '{tool_name}' result: {str(result)}",  # Properly passing the content
    )
    
    # Wrap the tool call properly in the expected format
    tool_response = ToolMessage(
        tool_call_id=tool_call_id,
        content=str(result)
    )

    print("🧠 chat_history now contains:", state["chat_history"] + [ai_msg, tool_response])

    return {
        **state,
        "intermediate_steps": state["intermediate_steps"] + [(outcome, tool_response)],
        "chat_history": state["chat_history"] + [ai_msg, tool_response],  # ✅ Add both
        "agent_outcome": None  # Reset for the next cycle
    }

# 🔀 6. Router logic to determine next node
def router(state: AgentState) -> str:
    outcome = state.get("agent_outcome")

    if isinstance(outcome, list):
        outcome = outcome[0]

    if isinstance(outcome, AgentFinish):
        return "final_answer"

    return outcome.tool

# 🕸️ 7. Build LangGraph DAG
def create_graph(tools_selected: List[str]) -> Runnable:
    agent, tools = init_research_agent(tools_selected)
    graph = StateGraph(AgentState)

    graph.add_node("oracle", lambda state: run_oracle(state, agent))
    graph.add_node("vector_search", run_tool)
    graph.add_node("snowflake_query", run_tool)
    graph.add_node("web_search", run_tool)
    graph.add_node("final_answer", run_tool)

    graph.set_entry_point("oracle")
    graph.add_conditional_edges("oracle", router)
    for name in ["vector_search", "snowflake_query", "web_search"]:
        graph.add_edge(name, "oracle")
    graph.add_edge("final_answer", END)

    return graph.compile()

# 🚀 8. FastAPI-compatible entrypoint for graph execution
def run_agents(tools: List[str], year: str, quarter: List[str]) -> Runnable:
    graph = create_graph(tools)
    return graph