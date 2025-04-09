import os
import pandas as pd
import yfinance as yf
import snowflake.connector
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def fetch_nvidia_real_time_historical_data():
    """
    Fetch real-time historical financial data for NVIDIA
    Covers the last 5 years with quarterly granularity
    """
    # Create Ticker object for NVIDIA
    nvda = yf.Ticker("NVDA")
    
    # Fetch quarterly financial history
    quarterly_financials = nvda.quarterly_financials
    quarterly_balance_sheet = nvda.quarterly_balance_sheet
    
    # Prepare to store historical data
    historical_data = []
    
    # Iterate through years and quarters
    for year in range(2020, 2025):
        for quarter in range(1, 5):
            # Determine the date range for the quarter
            if quarter == 1:
                quarter_date = f"{year}-03-31"
            elif quarter == 2:
                quarter_date = f"{year}-06-30"
            elif quarter == 3:
                quarter_date = f"{year}-09-30"
            else:  # quarter 4
                quarter_date = f"{year}-12-31"
            
            try:
                # Try to get historical stock data
                stock_data = nvda.history(start=quarter_date, end=quarter_date, interval="1d")
                
                # Initialize data point
                data_point = {
                    'Date': quarter_date,
                    'Market Cap': 0,
                    'Enterprise Value': 0,
                    'PE Ratio': 0,
                    'Forward PE': 0,
                    'Price to Book': 0,
                    'Dividend Yield': 0
                }
                
                # Fetch current market data
                info = nvda.info
                
                # Populate data point with available information
                if not stock_data.empty:
                    # Market Cap calculation
                    data_point['Market Cap'] = (stock_data['Close'].iloc[-1] * 
                        info.get('sharesOutstanding', stock_data['Close'].iloc[-1]))
                
                # Additional financial metrics
                data_point['Enterprise Value'] = info.get('enterpriseValue', 0)
                data_point['PE Ratio'] = info.get('trailingPE', 0)
                data_point['Forward PE'] = info.get('forwardPE', 0)
                data_point['Price to Book'] = info.get('priceToBook', 0)
                data_point['Dividend Yield'] = info.get('dividendYield', 0)
                
                # Try to get more precise metrics from financial statements
                if not quarterly_financials.empty and quarter_date in quarterly_financials.columns:
                    col = quarterly_financials[quarter_date]
                    # You can add more financial metrics here if needed
                
                historical_data.append(data_point)
                print(f"Collected real-time data for {year} Q{quarter}")
            
            except Exception as e:
                print(f"Error fetching real-time data for {year} Q{quarter}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(historical_data)
    
    # Save to CSV
    csv_path = os.path.join(os.path.dirname(__file__), 'nvidia_real_time_financials.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved real-time financial data to {csv_path}")
    
    return df

def populate_snowflake(df):
    """
    Populate Snowflake with real-time historical financial data
    """
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
        
        # Commit and close
        conn.commit()
        cursor.close()
        conn.close()
        
        print("Successfully populated Snowflake with real-time NVIDIA financial data")
        
    except Exception as e:
        print(f"Error populating Snowflake: {e}")

def main():
    # Fetch real-time historical data
    historical_df = fetch_nvidia_real_time_historical_data()
    
    # Populate Snowflake
    populate_snowflake(historical_df)
    
    # Print data overview
    print("\nReal-Time Financial Data Overview:")
    print(historical_df)
    print("\nTotal Rows:", len(historical_df))

if __name__ == "__main__":
    main()