from typing import List
from agents.rag_agent.pinecone_utils import search_chunks
from langchain.tools import tool

@tool("rag_retrieve_chunks", return_direct=True)
def retrieve_rag_chunks(query: str, year: str, quarter: str, top_k: int = 8) -> str:
    """
    Retrieves relevant text chunks from Pinecone vector DB based on query, year, and quarter.
    Returns formatted chunks for LLM consumption.
    """
    print(f"[RAG] 🔍 Querying Pinecone for: year={year}, quarter={quarter}")
    results = search_chunks(query=query, year=year, quarter=quarter, top_k=top_k)

    if not results:
        return "No relevant data found for the selected year and quarter."

    formatted = "\n\n---\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(results)])
    print("✅ Final formatted chunks:\n", formatted[:500])
    return formatted