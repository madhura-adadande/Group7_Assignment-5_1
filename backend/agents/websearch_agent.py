# backend/agents/websearch_agent.py
import os
import logging
from typing import Dict, Any, List, Optional
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Rest of the code remains the same...

class WebSearchAgent:
    def __init__(self):
        # Initialize with Tavily API
        self.api_key = os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY environment variable not set")
        
        # Initialize Tavily client
        self.client = TavilyClient(api_key=self.api_key)
        
        # Initialize Language Model
        self.llm = ChatOpenAI(temperature=0)
        
    def query(self, query_text: str, year: Optional[int] = None, quarter: Optional[int] = None) -> Dict[str, Any]:
        """
        Query Tavily API for latest information on NVIDIA related to the query.
        
        Args:
            query_text: The query text
            year: Optional year filter
            quarter: Optional quarter filter
            
        Returns:
            Dictionary with search results and synthesized information
        """
        # Augment query to focus on NVIDIA and recent information
        augmented_query = f"NVIDIA {query_text}"
        if year or quarter:
            time_filter = f" in {year} Q{quarter}" if year and quarter else (f" in {year}" if year else f" in Q{quarter}")
            augmented_query += time_filter

        # Execute search with Tavily
        try:
            response = self.client.search(
                query=augmented_query,
                search_depth="advanced",
                max_results=5,
                include_domains=[
                    "forbes.com", 
                    "cnbc.com", 
                    "bloomberg.com", 
                    "reuters.com", 
                    "wsj.com",
                    "nvidia.com"
                ]
            )
            
            # Extract results
            search_results = response.get("results", [])
            
            # Format results
            formatted_results = []
            for result in search_results:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "published_date": result.get("published_date", "")
                })
            
            # Generate insights from the search results
            insights = self._generate_insights(formatted_results, query_text)
            
            # Return results
            return {
                "results": formatted_results,
                "response": insights,
                "sources": [result["url"] for result in formatted_results]
            }
        
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return {
                "results": [],
                "response": f"Error performing web search: {e}",
                "sources": []
            }
        
    def _generate_insights(self, results: List[Dict[str, Any]], query_text: str) -> str:
        """
        Generate insights from search results using LLM
        """
        # Prepare context from search results
        context = ""
        for i, result in enumerate(results, 1):
            context += f"{i}. Title: {result['title']}\n"
            context += f"   URL: {result['url']}\n"
            context += f"   Published Date: {result.get('published_date', 'N/A')}\n"
            # Include a snippet of content
            content_snippet = result['content'][:300] + "..." if len(result['content']) > 300 else result['content']
            context += f"   Content Snippet: {content_snippet}\n\n"
        
        # Create prompt for insights generation
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are a research analyst specializing in NVIDIA.
            Analyze the following recent news articles to provide insights.
            Focus on extracting the most relevant and recent information.
            Provide a balanced and objective summary.
            
            IMPORTANT: 
            - Do not quote directly from articles
            - Summarize key points in your own words
            - Highlight the most significant recent developments
            - Be concise and clear
            """),
            ("human", "Recent news articles about NVIDIA:\n{context}\n\nQuery: {query}")
        ])
        
        # Generate insights
        try:
            chain = prompt | self.llm
            response = chain.invoke({
                "context": context,
                "query": query_text
            })
            
            return response.content
        
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return "Unable to generate insights from the search results."

# Optional: Test the Web Search Agent
if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Create Web Search agent
    web_search_agent = WebSearchAgent()
    
    # Example query
    result = web_search_agent.query("recent developments in AI chips")
    print("\nWeb Search Result:")
    print("Response:", result.get("response", "No response"))
    print("\nSources:", result.get("sources", "No sources"))