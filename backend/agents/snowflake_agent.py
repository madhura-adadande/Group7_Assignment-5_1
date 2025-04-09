# backend/agents/snowflake_agent.py
import os
import logging
from typing import Dict, Any, Optional
import pandas as pd
import snowflake.connector
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import base64
from io import BytesIO
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Rest of the code remains the same...

class SnowflakeAgent:
    def __init__(self):
        # Initialize with Snowflake credentials
        self.snowflake_user = os.getenv("SNOWFLAKE_USER")
        self.snowflake_password = os.getenv("SNOWFLAKE_PASSWORD")
        self.snowflake_account = os.getenv("SNOWFLAKE_ACCOUNT")
        self.snowflake_warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
        self.snowflake_database = os.getenv("SNOWFLAKE_DATABASE")
        self.snowflake_schema = os.getenv("SNOWFLAKE_SCHEMA")
        
        # Check for required environment variables
        required_vars = [
            "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD", "SNOWFLAKE_ACCOUNT",
            "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA"
        ]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.warning(f"Missing Snowflake environment variables: {missing_vars}")
        
        # Initialize OpenAI for generating insights
        self.llm = ChatOpenAI(temperature=0)
        
    def query(self, query_text: str, year: Optional[int] = None, quarter: Optional[int] = None) -> Dict[str, Any]:
        """
        Query Snowflake for NVIDIA financial data and generate insights.
        """
        try:
            # Connect to Snowflake
            conn = snowflake.connector.connect(
                user=self.snowflake_user,
                password=self.snowflake_password,
                account=self.snowflake_account,
                warehouse=self.snowflake_warehouse,
                database=self.snowflake_database,
                schema=self.snowflake_schema
            )
            
            # Build query with filtering
            sql_query = "SELECT * FROM NVIDIA_FINANCIALS"
            where_clauses = []
            
            if year:
                where_clauses.append(f"SUBSTRING(Date, 1, 4) = '{year}'")
            
            if quarter:
                quarter_map = {1: '03-31', 2: '06-30', 3: '09-30', 4: '12-31'}
                if quarter in quarter_map:
                    where_clauses.append(f"SUBSTRING(Date, 6) = '{quarter_map[quarter]}'")
            
            if where_clauses:
                sql_query += " WHERE " + " AND ".join(where_clauses)
            
            # Execute query
            df = pd.read_sql(sql_query, conn)
            conn.close()
            
            if df.empty:
                return {
                    "response": f"No financial data found for the specified filters.",
                    "chart": None,
                    "sources": []
                }
            
            # Rename columns to match the expected format
            column_mapping = {
                'DATE': 'Date',
                'MARKET_CAP': 'Market Cap',
                'ENTERPRISE_VALUE': 'Enterprise Value',
                'PE_RATIO': 'PE Ratio',
                'FORWARD_PE': 'Forward PE',
                'PRICE_TO_BOOK': 'Price to Book',
                'DIVIDEND_YIELD': 'Dividend Yield'
            }
            df.columns = [column_mapping.get(col.upper(), col) for col in df.columns]
            
            # Generate chart
            chart_path = self._generate_chart(df)
            
            # Format results for text response
            summary = self._generate_financial_summary(df, query_text)
            
            return {
                "response": summary,
                "chart": chart_path,
                "sources": ["NVIDIA Financial Data from Snowflake"]
            }
            
        except Exception as e:
            logger.error(f"Error in Snowflake query: {str(e)}", exc_info=True)
            return {
                "response": f"Error querying financial data: {str(e)}",
                "chart": None,
                "sources": []
            }
            
    def _generate_chart(self, df) -> str:
        """Generate a chart from financial data"""
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Plot Market Cap over time
            df['Date'] = pd.to_datetime(df['Date'])
            df_sorted = df.sort_values('Date')
            
            ax.plot(df_sorted['Date'], df_sorted['Market Cap'], marker="o", linewidth=2, color="#007acc")
            
            # Title and axes
            ax.set_title("NVIDIA Market Cap Over Time", fontsize=14)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Market Cap", fontsize=12)
            
            # Format axes
            plt.xticks(rotation=45)
            ax.grid(True, linestyle="--", alpha=0.6)
            
            # Format y-axis with billions/trillions
            def billions(x, pos):
                if x >= 1e12:
                    return f"${x*1.0/1e12:.1f}T"
                elif x >= 1e9:
                    return f"${x*1.0/1e9:.1f}B"
                else:
                    return f"${x:,.0f}"
                    
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(billions))
            
            # Save chart to bytes
            buf = BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Convert to base64 for embedding
            chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
            chart_data = f"data:image/png;base64,{chart_base64}"
            
            plt.close()
            return chart_data
            
        except Exception as e:
            logger.error(f"Error generating chart: {str(e)}", exc_info=True)
            return None
            
    def _generate_financial_summary(self, df, query_text) -> str:
        """Generate a textual summary of financial data"""
        try:
            # Basic statistics
            latest_row = df.iloc[-1]
            earliest_row = df.iloc[0]
            
            summary = f"""
            ## NVIDIA Financial Summary
            
            **Time Period**: {earliest_row['Date']} to {latest_row['Date']}
            
            **Latest Market Cap**: ${latest_row['Market Cap']:,.2f}
            
            **Growth since start of period**: {((latest_row['Market Cap'] - earliest_row['Market Cap']) / earliest_row['Market Cap'] * 100):.2f}%
            
            **Latest P/E Ratio**: {latest_row['PE Ratio']:.2f}
            
            **Latest Forward P/E**: {latest_row['Forward PE']:.2f}
            
            **Price to Book Ratio**: {latest_row['Price to Book']:.2f}
            
            **Dividend Yield**: {latest_row['Dividend Yield']:.4f}
            
            This data answers the user query: "{query_text}"
            """
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}", exc_info=True)
            return f"Error generating financial summary: {str(e)}"

# Optional: Test the Snowflake Agent
if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Create Snowflake agent
    snowflake_agent = SnowflakeAgent()
    
    # Test with different queries
    print("\n--- Query without filters ---")
    result = snowflake_agent.query("What are NVIDIA's key financial metrics?")
    print("Response:", result.get("response", "No response"))
    
    print("\n--- Query with year filter ---")
    result_year = snowflake_agent.query("NVIDIA financial performance", year=2024)
    print("Response:", result_year.get("response", "No response"))