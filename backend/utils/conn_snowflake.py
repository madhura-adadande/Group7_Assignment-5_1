import os
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))

def populate_snowflake():
    """
    Read the CSV and populate Snowflake with historical NVIDIA financial data
    """
    # Path to the CSV file
    csv_path = os.path.join(os.path.dirname(__file__), 'nvidia_historical_financials.csv')
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Snowflake connection parameters
    conn_params = {
        'user': os.getenv('SNOWFLAKE_USER'),
        'password': os.getenv('SNOWFLAKE_PASSWORD'),
        'account': os.getenv('SNOWFLAKE_ACCOUNT'),
        'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE'),
        'database': os.getenv('SNOWFLAKE_DATABASE'),
        'schema': os.getenv('SNOWFLAKE_SCHEMA')
    }
    
    # Connect to Snowflake
    try:
        conn = snowflake.connector.connect(**conn_params)
        cursor = conn.cursor()
        
        # Drop existing table if it exists
        drop_table_sql = "DROP TABLE IF EXISTS NVIDIA_FINANCIALS"
        cursor.execute(drop_table_sql)
        
        # Create table with appropriate columns
        create_table_sql = """
        CREATE TABLE NVIDIA_FINANCIALS (
            Date VARCHAR(20),
            Market_Cap NUMBER(38,2),
            Enterprise_Value NUMBER(38,2),
            PE_Ratio FLOAT,
            Forward_PE FLOAT,
            Price_to_Book FLOAT,
            Dividend_Yield FLOAT
        )
        """
        cursor.execute(create_table_sql)
        
        # Prepare data for insertion
        insert_sql = """
        INSERT INTO NVIDIA_FINANCIALS 
        (Date, Market_Cap, Enterprise_Value, PE_Ratio, Forward_PE, Price_to_Book, Dividend_Yield)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        # Convert DataFrame to list of tuples for bulk insert
        data_to_insert = [
            (
                str(row['Date']),
                float(row['Market Cap']),
                float(row['Enterprise Value']),
                float(row['PE Ratio']),
                float(row['Forward PE']),
                float(row['Price to Book']),
                float(row['Dividend Yield'])
            ) for _, row in df.iterrows()
        ]
        
        # Bulk insert
        cursor.executemany(insert_sql, data_to_insert)
        
        # Verify insertion
        cursor.execute("SELECT COUNT(*) FROM NVIDIA_FINANCIALS")
        row_count = cursor.fetchone()[0]
        print(f"Total rows inserted: {row_count}")
        
        # Commit and close
        conn.commit()
        cursor.close()
        conn.close()
        
        print("Successfully populated Snowflake with NVIDIA financial data")
        
    except Exception as e:
        print(f"Error populating Snowflake: {e}")

def main():
    # Populate Snowflake
    populate_snowflake()

if __name__ == "__main__":
    main()