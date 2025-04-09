import os
import pinecone
from langchain_openai import OpenAIEmbeddings
import pandas as pd
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_historical_data():
    """
    Load historical data from CSV
    """
    # Path to the CSV file
    csv_path = os.path.join(os.path.dirname(__file__), 'nvidia_historical_financials.csv')
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    return df

def create_detailed_documents(df):
    """
    Create comprehensive documents for each quarter
    """
    documents = []
    
    for _, row in df.iterrows():
        # Extract year and quarter from the date
        date = pd.to_datetime(row['Date'])
        year = date.year
        quarter = (date.month - 1) // 3 + 1
        
        # Create a detailed narrative document
        document_text = f"""
        NVIDIA Quarterly Financial Snapshot: {year} Q{quarter}

        Comprehensive Financial Overview:
        - Date of Snapshot: {row['Date']}
        
        Market Valuation:
        - Market Capitalization: ${row['Market Cap']:,.2f}
        - Enterprise Value: ${row['Enterprise Value']:,.2f}
        
        Key Valuation Metrics:
        - Price-to-Earnings (PE) Ratio: {row['PE Ratio']:.2f}
        - Forward PE Ratio: {row['Forward PE']:.2f}
        - Price-to-Book Ratio: {row['Price to Book']:.2f}
        - Dividend Yield: {row['Dividend Yield']:.4f}
        
        Market Context:
        This financial snapshot provides insights into NVIDIA's performance during {year} Q{quarter}. 
        The data reflects the company's market valuation, earnings potential, and investment characteristics 
        at a specific point in time, showcasing the dynamic nature of NVIDIA's financial landscape.
        
        Strategic Implications:
        - The financial metrics suggest NVIDIA's position in the technology and AI semiconductor market.
        - Market capitalization and valuation ratios indicate investor sentiment and growth expectations.
        """
        
        documents.append({
            'text': document_text,
            'source': f'NVIDIA_Financial_Snapshot_{year}_Q{quarter}',
            'year': str(year),
            'quarter': f'q{quarter}'
        })
    
    return documents

def index_to_pinecone(documents):
    """
    Index documents to Pinecone
    """
    try:
        # Initialize Pinecone
        pc = pinecone.Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        
        # Initialize embedding model
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=1536
        )
        
        # Index name
        index_name = "nvidia-financial-reports"
        
        # Delete existing index if it exists
        if index_name in pc.list_indexes().names():
            pc.delete_index(index_name)
            logger.info(f"Deleted existing index: {index_name}")
        
        # Create new index
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=pinecone.ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        logger.info(f"Created new index: {index_name}")
        
        # Get the index
        index = pc.Index(index_name)
        
        # Embed and index documents
        for i, doc in enumerate(documents):
            # Embed the document
            embedding = embeddings.embed_documents([doc['text']])[0]
            
            # Upsert to Pinecone
            index.upsert([
                (
                    f"doc_{i}",  # unique ID
                    embedding,   # embedding vector
                    {            # metadata
                        "text": doc['text'],
                        "source": doc['source'],
                        "year": doc['year'],
                        "quarter": doc['quarter']
                    }
                )
            ])
            logger.info(f"Indexed document {i+1}: {doc['source']}")
        
        # Verify index stats
        stats = index.describe_index_stats()
        logger.info(f"Index Statistics: {stats}")
        
        logger.info("Pinecone indexing complete!")
    
    except Exception as e:
        logger.error(f"Pinecone indexing failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Load historical data
    historical_df = load_historical_data()
    
    # Create detailed documents
    documents = create_detailed_documents(historical_df)
    
    # Index to Pinecone
    index_to_pinecone(documents)

if __name__ == "__main__":
    main()