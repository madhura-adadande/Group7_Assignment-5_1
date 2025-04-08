import os
import boto3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import snowflake.connector
from dotenv import load_dotenv
from langchain.tools import tool
from typing import List

# Load .env
load_dotenv()

# Snowflake setup
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_REGION = os.getenv("SNOWFLAKE_REGION")
FULL_ACCOUNT = f"{SNOWFLAKE_ACCOUNT}.{SNOWFLAKE_REGION}"

# AWS S3 setup
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET = os.getenv("AWS_BUCKET_NAME")

s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

def query_snowflake(sql: str) -> pd.DataFrame:
    conn = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=FULL_ACCOUNT,
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        role="ACCOUNTADMIN"
    )
    df = pd.read_sql(sql, conn)
    conn.close()
    return df

def format_billions(x, _):
    if x >= 1e12:
        return f"${x / 1e12:.1f}T"
    elif x >= 1e9:
        return f"${x / 1e9:.1f}B"
    return f"${x:,.0f}"

def upload_to_s3(filepath: str, s3_folder: str = "Reports/Charts") -> str:
    filename = os.path.basename(filepath)
    s3_key = f"{s3_folder}/{filename}"

    s3_client.upload_file(filepath, S3_BUCKET, s3_key)

    s3_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
    return s3_url

def save_chart(df: pd.DataFrame, columns: list, title: str, ylabel: str, filename: str, chart_type="line") -> str:
    df["PERIOD_END_DATE"] = pd.to_datetime(df["PERIOD_END_DATE"])
    df = df.sort_values("PERIOD_END_DATE")
    fig, ax = plt.subplots(figsize=(10, 5))

    plotted = False
    for col in columns:
        if col in df.columns:
            clean_df = df[["PERIOD_END_DATE", col]].dropna()
            if not clean_df.empty:
                if chart_type == "bar":
                    clean_df["Label"] = clean_df["PERIOD_END_DATE"].dt.strftime("%Y-%m")
                    ax.bar(clean_df["Label"], clean_df[col], label=col)
                else:
                    ax.plot(clean_df["PERIOD_END_DATE"], clean_df[col], label=col, marker='o')
                plotted = True

    if not plotted:
        return f"⚠️ Chart skipped: {filename} (not enough data)"

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)

    if "USD" in ylabel or "Value" in title:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_billions))

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    return upload_to_s3(filename)

@tool("snowflake_query")
def snowflake_query_tool(query: str, year: str = None, quarter: List[str] = None) -> str:
    """
    Generates charts and summary from Snowflake valuation data and uploads charts to S3.

    Args:
        query: User's financial query
        year: Optional year filter
        quarter: Optional list of quarters

    Returns:
        str: Summary with S3 chart URLs
    """
    user_query = query.lower()

    try:
        sql = """
            SELECT * FROM MULTI_AGENT_LLM.NVIDA_REPORTS_SCHEMA.QUICKFS_NVDA_VALUATION
            WHERE PERIOD_END_DATE >= '2021-01-01' AND PERIOD_END_DATE <= '2025-12-31'
            ORDER BY PERIOD_END_DATE
        """
        df = query_snowflake(sql)

        if df.empty:
            return "❌ No data found for 2021–2025."

        charts = []
        latest = df.iloc[-1]
        summary_parts = []

        # PEG
        if "peg" in user_query:
            charts.append(save_chart(df, ["PEG_RATIO"], "PEG Ratio", "Ratio", "peg_ratio_bar.png", chart_type="bar"))
            summary_parts.append(f"- PEG Ratio: {latest['PEG_RATIO']:.3f}")

        # Valuation
        if any(k in user_query for k in ["market cap", "valuation", "enterprise value"]):
            charts.append(save_chart(df, ["MARKET_CAP", "ENTERPRISE_VALUE"], "Market Cap vs Enterprise Value", "USD", "market_value_line.png", chart_type="line"))
            summary_parts.extend([
                f"- Market Cap: ${latest['MARKET_CAP']:,.0f}",
                f"- Enterprise Value: ${latest['ENTERPRISE_VALUE']:,.0f}"
            ])

        # Multiples
        if any(k in user_query for k in ["ev/ebitda", "multiple"]):
            charts.append(save_chart(df, ["ENTERPRISE_VALUE_TO_SALES", "EV_TO_EBITDA"], "Enterprise Multiples", "Multiple", "enterprise_multiples_bar.png", chart_type="bar"))
            summary_parts.extend([
                f"- EV/Sales: {latest['ENTERPRISE_VALUE_TO_SALES']:.2f}",
                f"- EV/EBITDA: {latest['EV_TO_EBITDA']:.2f}"
            ])

        if not summary_parts:
            charts.append(save_chart(df, ["MARKET_CAP", "ENTERPRISE_VALUE"], "Market Cap vs Enterprise Value", "USD", "market_value_line.png", chart_type="line"))
            charts.append(save_chart(df, ["PEG_RATIO"], "PEG Ratio", "Ratio", "peg_ratio_bar.png", chart_type="bar"))
            charts.append(save_chart(df, ["ENTERPRISE_VALUE_TO_SALES", "EV_TO_EBITDA"], "Enterprise Multiples", "Multiple", "enterprise_multiples_bar.png", chart_type="bar"))
            summary_parts = [
                f"- Market Cap: ${latest['MARKET_CAP']:,.0f}",
                f"- Enterprise Value: ${latest['ENTERPRISE_VALUE']:,.0f}",
                f"- PEG Ratio: {latest['PEG_RATIO']:.3f}",
                f"- EV/Sales: {latest['ENTERPRISE_VALUE_TO_SALES']:.2f}",
                f"- EV/EBITDA: {latest['EV_TO_EBITDA']:.2f}"
            ]

        summary = f"📊 NVIDIA Financial Summary (2021–2025):\n" + "\n".join(summary_parts)
        return summary + "\n\n🖼️ S3 Chart URLs:\n" + "\n".join(charts)

    except Exception as e:
        return f"❌ Error: {str(e)}"