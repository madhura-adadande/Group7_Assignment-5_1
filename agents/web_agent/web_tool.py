# agents/web_agent/web_tool.py

import os
from tavily import TavilyClient
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
client = TavilyClient(api_key=TAVILY_API_KEY)

@tool("web_search")
def web_search_tool(input: str) -> str:
    """
    Use this tool to search for real-time web information about NVIDIA.
    Useful for getting the latest financial news, trends, or product updates.
    """
    if "nvidia" not in input.lower():
        input = f"NVIDIA {input}"

    response = client.search(
        query=input,
        search_depth="advanced",
        max_results=5,
        include_answer=True,
        include_raw_content=True
    )

    summary = response.get("answer", "No summary available.")
    results = response.get("results", [])

    report = f"## 🧠 AI Summary\n{summary}\n\n## 🔍 Top Results:\n"
    for i, res in enumerate(results, 1):
        title = res.get("title", "Untitled")
        url = res.get("url", "#")
        content = res.get("content") or res.get("raw_content") or "No content available"
        report += f"### {i}. {title}\n🔗 [Source]({url})\n\n{content}\n\n"

    return report
