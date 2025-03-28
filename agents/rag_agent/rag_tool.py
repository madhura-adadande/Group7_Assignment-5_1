from langchain.tools import tool
from agents.rag_agent.pinecone_utils import search_chunks
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

@tool("vector_search")
def vector_search_tool(input: str, year: str, quarter: list) -> str:
    """
    Use this tool to search NVIDIA historical report chunks based on a user query and year/quarter filters.
    """
    print(f"[RAG TOOL] Searching with input: {input}, year: {year}, quarter: {quarter}")
    results = []

    # Search for results for each quarter
    for q in quarter:
        chunks = search_chunks(input, year, q)
        results.extend(chunks)

    # If no results found, return a message indicating no relevant data
    if not results:
        return "No relevant information found in historical reports."

    # Join results to form the context
    context = "\n\n".join(results)

    # Return the context (we will process the response in langgraph_controller)
    return context