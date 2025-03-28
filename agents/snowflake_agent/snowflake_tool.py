import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import snowflake.connector
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

# ✅ Helper: Connect to Snowflake and run a SQL query
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_REGION = os.getenv("SNOWFLAKE_REGION")
FULL_ACCOUNT = f"{SNOWFLAKE_ACCOUNT}.{SNOWFLAKE_REGION}"

# ✅ Query function with FULL_ACCOUNT
def query_snowflake(sql: str) -> pd.DataFrame:
    conn = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=FULL_ACCOUNT,  # ✅ Using full account URL
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA")
    )
    df = pd.read_sql(sql, conn)
    conn.close()
    return df

# ✅ Helper: Generate a chart from the DataFrame
def generate_chart(df: pd.DataFrame, metric: str = "Market_Cap") -> str:
    df = df.sort_values("Report_Date")
    df["Report_Date"] = pd.to_datetime(df["Report_Date"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Report_Date"], df[metric], marker="o", linewidth=2, color="#0a9396")
    ax.set_title(f"NVIDIA {metric} Over Time", fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    plt.xticks(rotation=45)
    ax.grid(True, linestyle="--", alpha=0.6)

    def format_billions(x, _):
        if x >= 1e12:
            return f"${x/1e12:.1f}T"
        elif x >= 1e9:
            return f"${x/1e9:.1f}B"
        return f"${x:,.0f}"
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_billions))

    for _, row in df.iterrows():
        ax.annotate(
            f"{row[metric]/1e9:.1f}B",
            (row["Report_Date"], row[metric]),
            textcoords="offset points",
            xytext=(0, 8),
            ha='center',
            fontsize=8,
            color='gray'
        )

    chart_path = f"{metric.lower()}_chart.png"
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    return f"📊 Chart saved as `{chart_path}`"

# ✅ LangChain Tool for LangGraph
@tool("snowflake_query")
def snowflake_query_tool(input: str, year: str, quarter: list) -> str:
    """
    Use this tool to retrieve NVIDIA valuation data from Snowflake for specific year/quarter.
    Returns a textual summary, a table, and a saved chart.
    """
    try:
        quarter_filters = ",".join([f"'{q}'" for q in quarter])
        query = f"""
            SELECT * FROM NVDA_FINANCIAL_DATA
            WHERE YEAR(Report_Date) = {year}
            AND QUARTER(Report_Date) IN ({','.join([q[-1] for q in quarter])})
        """
        df = query_snowflake(query)

        if df.empty:
            return f"No financial data found for {year}, Quarter(s): {', '.join(quarter)}."

        # Generate chart
        chart_msg = generate_chart(df, metric="Market_Cap")

        # Generate textual summary for first row
        row = df.iloc[0]
        summary = (
            f"📄 NVIDIA Valuation Summary — {year} Q{quarter[0]}:\n"
            f"- Enterprise Value: {row['Enterprise_Value']:,}\n"
            f"- Market Cap: {row['Market_Cap']:,}\n"
            f"- Forward PE: {row['Forward_PE']:.2f}\n"
            f"- Price-Earnings Ratio: {row['Price_Earnings_Ratio']:.2f}\n"
            f"- Price-Book Ratio: {row['Price_Book_Ratio']:.2f}\n"
            f"- PEG Ratio: {row['Price_Earnings_Growth_Ratio']:.3f}\n"
            f"- Price-Sales Ratio: {row['Price_Sales_Ratio']:.2f}"
        )

        return summary + f"\n\n🧮 Table Preview:\n{df.head().to_markdown()}\n\n{chart_msg}"

    except Exception as e:
        return f"❌ Snowflake Tool Error: {str(e)}"
