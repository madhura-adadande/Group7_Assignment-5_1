import sys
import os

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from agents.rag_agent.rag_tool import retrieve_rag_chunks

if __name__ == "__main__":
    query = "total revenue"
    year = "2024"
    quarter = "Q1"

    result = retrieve_rag_chunks.invoke({
        "query": query,
        "year": year,
        "quarter": quarter
    })

    print("\n🔍 Final Tool Output:\n")
    print(result[:1000])  # Truncate output
