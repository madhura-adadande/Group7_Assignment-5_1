# backend/agents/rag_agent.py
import os
import logging
from typing import Dict, Any, Optional
import pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class RagAgent:
    def __init__(self):
        # Initialize Pinecone client
        api_key = os.getenv("PINECONE_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Initialize Pinecone
        self.pc = pinecone.Pinecone(api_key=api_key)
        self.index_name = "nvidia-financial-reports"
        
        # Initialize embedding and LLM
        self.embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=1536,
            openai_api_key=openai_api_key
        )
        self.llm = ChatOpenAI(temperature=0, api_key=openai_api_key)
        
        # Detailed index verification
        self._verify_index()
        
    def _verify_index(self):
        """
        Comprehensive index verification and debugging
        """
        try:
            # List all indexes
            indexes = self.pc.list_indexes().names()
            logger.info(f"Available Pinecone indexes: {indexes}")
            
            # Check if our specific index exists
            if self.index_name not in indexes:
                logger.error(f"Index {self.index_name} not found!")
                logger.error(f"Available indexes: {indexes}")
                return False
            
            # Get index instance
            index = self.pc.Index(self.index_name)
            
            # Describe index stats
            stats = index.describe_index_stats()
            logger.info("Index Statistics:")
            logger.info(f"Dimension: {stats.get('dimension')}")
            logger.info(f"Index Fullness: {stats.get('index_fullness')}")
            logger.info(f"Total Vector Count: {stats.get('total_vector_count')}")
            
            # List a few vectors to verify content
            try:
                # Attempt to fetch a few vectors
                sample_vectors = index.fetch(ids=[f"doc_{i}" for i in range(3)])
                logger.info("Sample Vectors:")
                logger.info(sample_vectors)
            except Exception as fetch_error:
                logger.error(f"Error fetching sample vectors: {fetch_error}")
            
            return True
        except Exception as e:
            logger.error(f"Error verifying index: {e}")
            return False
    
    def query(self, query_text: str, year: Optional[int] = None, quarter: Optional[int] = None) -> Dict[str, Any]:
        """
        Query the RAG system with optional metadata filtering.
        """
        try:
            # Log query details
            logger.info(f"RAG Query: '{query_text}', Year: {year}, Quarter: {quarter}")
            
            # Generate embedding for the query
            query_embedding = self.embedding_model.embed_query(query_text)
            logger.info(f"Query Embedding Dimension: {len(query_embedding)}")
            
            # Prepare metadata filter
            filter_dict = {}
            if year is not None:
                filter_dict["year"] = str(year)
            if quarter is not None:
                filter_dict["quarter"] = f"q{quarter}"
            
            logger.info(f"Metadata Filter: {filter_dict}")
            
            # Connect to index
            index = self.pc.Index(self.index_name)
            
            # Perform hybrid search
            search_results = index.query(
                vector=query_embedding,
                filter=filter_dict if filter_dict else None,
                top_k=5,
                include_metadata=True
            )
            
            # Log search results
            logger.info(f"Search Results Matches: {len(search_results.matches)}")
            for i, match in enumerate(search_results.matches, 1):
                logger.info(f"Match {i}:")
                logger.info(f"  Score: {match.score}")
                logger.info(f"  Metadata: {match.metadata}")
            
            # Extract retrieved contexts
            contexts = []
            for match in search_results.matches:
                text = match.metadata.get("text", "")
                source = match.metadata.get("source", "Unknown source")
                contexts.append(f"[Source: {source}]\n{text}")
            
            # If no contexts, provide fallback
            if not contexts:
                logger.warning("No relevant information found")
                return {
                    "response": "No historical data found for the given query and filters.",
                    "sources": []
                }
            
            # Combine contexts
            combined_context = "\n\n".join(contexts)
            
            # Generate response
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Analyze the following historical information about NVIDIA."),
                ("human", "Context:\n{context}\n\nQuery: {query}")
            ])
            
            chain = prompt | self.llm
            response = chain.invoke({
                "context": combined_context,
                "query": query_text
            })
            
            return {
                "response": response.content,
                "sources": [match.metadata.get("source", "Unknown") for match in search_results.matches]
            }
        
        except Exception as e:
            logger.error(f"RAG Query Error: {e}")
            return {
                "response": f"Error retrieving historical data: {e}",
                "sources": []
            }

# Optional: Test the RAG Agent
if __name__ == "__main__":
    # Create RAG agent
    rag_agent = RagAgent()
    
    # Test query
    result = rag_agent.query("What are NVIDIA's key financial metrics?")
    print("\nRAG Query Result:")
    print("Response:", result.get("response", "No response"))
    print("Sources:", result.get("sources", "No sources"))