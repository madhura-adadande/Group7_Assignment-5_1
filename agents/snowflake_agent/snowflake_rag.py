import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import snowflake.connector
from dotenv import load_dotenv
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool

# ---------- Load .env ---------- #
dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

# ---------- Snowflake Configs ---------- #
SNOW_USER = os.getenv("SNOWFLAKE_USER")
SNOW_PASS = os.getenv("SNOWFLAKE_PASSWORD")
SNOW_ACC = os.getenv("SNOWFLAKE_ACCOUNT")
SNOW_DB = os.getenv("SNOWFLAKE_DATABASE")
SNOW_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")
SNOW_WH = os.getenv("SNOWFLAKE_WAREHOUSE")

# ---------- Snowflake Query Helper ---------- #
def run_snowflake_query(sql: str) -> pd.DataFrame:
    conn = snowflake.connector.connect(
        user=SNOW_USER,
        password=SNOW_PASS,
        account=SNOW_ACC,
        warehouse=SNOW_WH,
        database=SNOW_DB,
        schema=SNOW_SCHEMA,
    )
    df = pd.read_sql(sql, conn)
    conn.close()
    return df

# ---------- Chart Generator ---------- #
def plot_metric_over_time(df: pd.DataFrame, metric: str) -> str:
    df = df.sort_values("REPORT_DATE")
    df["REPORT_DATE"] = pd.to_datetime(df["REPORT_DATE"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["REPORT_DATE"], df[metric], marker="o", linewidth=2, color="#007acc")

    ax.set_title(f"NVIDIA {metric.replace('_', ' ').title()} Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel(metric)
    plt.xticks(rotation=45)
    ax.grid(True, linestyle="--", alpha=0.6)

    def format_billions(x, pos):
        if x >= 1e12:
            return f"${x/1e12:.1f}T"
        elif x >= 1e9:
            return f"${x/1e9:.1f}B"
        else:
            return f"${x:,.0f}"

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_billions))

    for i, row in df.iterrows():
        ax.annotate(
            f'{row[metric]/1e9:.1f}B',
            (row["REPORT_DATE"], row[metric]),
            textcoords="offset points",
            xytext=(0, 8),
            ha='center',
            fontsize=8,
            color='gray'
        )

    os.makedirs("charts", exist_ok=True)
    chart_path = f"charts/{metric.lower()}_chart.png"
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    return chart_path

# ---------- LangChain Tool ---------- #
@tool
def fetch_nvda_valuation(input: str) -> str:
    """
    Retrieves NVIDIA valuation data from Snowflake by year and quarter.
    Input format: "year=2024, quarter=4"
    """
    try:
        year = input.split("year=")[1].split(",")[0].strip()
        quarter = input.split("quarter=")[1].strip()

        query = f"""
        SELECT * FROM NVDA_FINANCIAL_DATA
        WHERE YEAR(REPORT_DATE) = {year} AND QUARTER(REPORT_DATE) = {quarter}
        """
        df = run_snowflake_query(query)

        if df.empty:
            return f"No data found for Q{quarter} {year}."

        row = df.iloc[0]
        summary = (
            f"NVIDIA Financials for Q{quarter} {year}:\n"
            f"- Report Date: {row['REPORT_DATE']}\n"
            f"- Enterprise Value: {row['ENTERPRISE_VALUE']:,}\n"
            f"- Market Cap: {row['MARKET_CAP']:,}\n"
            f"- PE Ratio: {row['PRICE_EARNINGS_RATIO']:.2f}\n"
            f"- Forward PE: {row['FORWARD_PE']:.2f}\n"
            f"- PB Ratio: {row['PRICE_BOOK_RATIO']:.2f}\n"
            f"- PS Ratio: {row['PRICE_SALES_RATIO']:.2f}\n"
            f"- PEG Ratio: {row['PRICE_EARNINGS_GROWTH_RATIO']:.4f}"
        )

        chart_path = plot_metric_over_time(df, metric="MARKET_CAP")
        return summary + f"\n\n📈 Chart saved to: {chart_path}"

    except Exception as e:
        return f"❌ Error: {str(e)}"

# ---------- LLM Agent ---------- #
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
snowflake_agent = initialize_agent(
    tools=[fetch_nvda_valuation],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ---------- FastAPI Helper Function ---------- #
def fetch_snowflake_summary_and_charts(year: str, quarter: str):
    input_str = f"year={year}, quarter={quarter}"
    result = snowflake_agent.invoke(input_str)
    output = result["output"]

    # Extract chart path
    chart_path = None
    for line in output.splitlines():
        if "charts/" in line and ".png" in line:
            chart_path = line.split("charts/")[-1].strip().replace("📈 Chart saved to: ", "")
            chart_path = os.path.join("charts", os.path.basename(chart_path))

    return output, [chart_path] if chart_path else []

# ---------- CLI Test ---------- #
if __name__ == "__main__":
    user_input = input("🔍 Ask about NVIDIA financials (e.g., year=2024, quarter=4): ")
    result = snowflake_agent.invoke(user_input)
    print("\n📊 Answer:\n", result["output"])
