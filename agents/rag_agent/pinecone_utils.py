import os
from dotenv import load_dotenv
from pinecone import Pinecone
import openai

# Load environment variables
load_dotenv()

# API keys & index info
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Connect to Pinecone index
def connect_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)

# Get embedding from OpenAI
def get_openai_embedding(text: str) -> list:
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Query Pinecone index using metadata filtering (year + quarter)
def search_chunks(query: str, year: str, quarter: str, top_k: int = 8):
    print(f"[RAG TOOL] 🧠 Searching with: year={year}, quarter={quarter}, query='{query}'")

    index = connect_pinecone_index()
    embedded_query = get_openai_embedding(query)

    results = index.query(
        vector=embedded_query,
        top_k=top_k,
        namespace="mistral_recursive",
        include_metadata=True,
        filter={
            "year": year,
            "quarter": quarter
        }
    )

    matches = results.get("matches", [])
    print(f"📦 Retrieved {len(matches)} matches")

    for match in matches:
        print("📦 MATCH METADATA:", match.get("metadata"))

    if not matches:
        print("❌ No matching vectors found. Debug this:")
        print("🔍 Check if your Pinecone vectors were upserted with correct metadata:")
        print(f" - year = '{year}'")
        print(f" - quarter = '{quarter}'")
        print(f" - namespace = 'mistral_recursive'")
        print("✅ You can use index.describe_index_stats() to check available metadata values.")

    return [
        match["metadata"]["text"]
        for match in matches
        if "text" in match.get("metadata", {})
    ]
