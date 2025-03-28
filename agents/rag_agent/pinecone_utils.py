import os
from dotenv import load_dotenv
from pinecone import Pinecone
import openai

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# ✅ Connect to Pinecone
def connect_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)

# ✅ Get embedding for a query using OpenAI
def get_openai_embedding(text: str) -> list:
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# ✅ Search Pinecone using OpenAI embedding, filtered by metadata (year + quarter)
def search_chunks(query: str, year: str, quarter: str, top_k: int = 8):
    print(f"[RAG TOOL] Searching with input: {query}, year: {year}, quarter: {quarter}")
    
    index = connect_pinecone_index()
    embedded_query = get_openai_embedding(query)

    namespace = "mistral_recursive"

    # ✅ Query Pinecone with metadata filter (no ID filtering needed)
    results = index.query(
        vector=embedded_query,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
        filter={
            "year": year,
            "quarter": quarter
        }
    )

    matches = results.get("matches", [])
    print(f"📦 Retrieved {len(matches)} matches from namespace: {namespace} for year={year}, quarter={quarter}")
    for match in matches:
        chunk_preview = match['metadata']['text'][:80].replace("\n", " ")
        print(f"🧠 {match['id']} => {chunk_preview}...")

    return [match["metadata"]["text"] for match in matches]
