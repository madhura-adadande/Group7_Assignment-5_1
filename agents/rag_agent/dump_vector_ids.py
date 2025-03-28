# dump_vector_ids.py

import os
from dotenv import load_dotenv
from pinecone import Pinecone
import json
from collections import defaultdict
import re

# Load env variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")

# Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

namespace = "mistral_recursive"
print(f"🔍 Gathering vector IDs from namespace: {namespace}")
vector_id_map = defaultdict(list)

# Flatten the generator result
vector_batches = list(index.list(namespace=namespace))  # List of lists
flat_vector_ids = [vid for batch in vector_batches for vid in batch]  # Flattened

print(f"📦 Total vectors found: {len(flat_vector_ids)}")

# Group by "2022_Q2" prefix
for vector_id in flat_vector_ids:
    match = re.match(r"(\d{4}_Q\d+)_", vector_id)
    if match:
        prefix = match.group(1)
        vector_id_map[prefix].append(vector_id)

# Save mapping to JSON
with open("vector_id_map.json", "w") as f:
    json.dump(vector_id_map, f, indent=2)

print("✅ Saved vector_id_map.json with grouped IDs.")
