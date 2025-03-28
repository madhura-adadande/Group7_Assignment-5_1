import os
from pathlib import Path
import pandas as pd
from yahooquery import Ticker
import snowflake.connector
from dotenv import load_dotenv

# ---------- LOAD .env FILE SAFELY ---------- #
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    raise FileNotFoundError(f".env file not found at {env_path}")

# ---------- LOAD CREDENTIALS ---------- #
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_REGION = os.getenv("SNOWFLAKE_REGION")
SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SNOWFLAKE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")
SNOWFLAKE_STAGE = os.getenv("SNOWFLAKE_STAGE")

FULL_ACCOUNT = f"{SNOWFLAKE_ACCOUNT}.{SNOWFLAKE_REGION}"
CSV_OUTPUT_FILE = "nvdia_transformed_dataset.csv"
TICKER_SYMBOL = "NVDA"

# ---------- DEBUG PRINT ---------- #
print("✅ ENV CHECK:")
print("USER:", SNOWFLAKE_USER)
print("ACCOUNT (full):", FULL_ACCOUNT)
print("ROLE:", SNOWFLAKE_ROLE)
print("WAREHOUSE:", SNOWFLAKE_WAREHOUSE)
print("DATABASE:", SNOWFLAKE_DATABASE)
print("SCHEMA:", SNOWFLAKE_SCHEMA)
print("STAGE:", SNOWFLAKE_STAGE)

# ---------- STEP 1: FETCH & CLEAN DATA ---------- #
def fetch_nvda_financials() -> pd.DataFrame:
    print("📈 Grabbing financial data for NVIDIA...")
    ticker = Ticker(TICKER_SYMBOL)
    stats = ticker.valuation_measures

    df = pd.DataFrame(stats).T.reset_index()
    df = df[df["index"] != "periodType"] if "periodType" in df["index"].values else df
    df = df.rename(columns={"index": "Metric"})
    df = df.set_index("Metric").T.reset_index().rename(columns={"index": "ReportDate"})

    if "symbol" in df.columns:
        df = df.drop(columns=["symbol"])
        print("🧹 Removed 'symbol' column")

    df.to_csv(CSV_OUTPUT_FILE, index=False)
    print(f"✅ Data saved to {CSV_OUTPUT_FILE}")
    return df

# ---------- STEP 2: CONNECT & UPLOAD TO SNOWFLAKE STAGE ---------- #
def upload_csv_to_stage(file_path: str):
    print("🔗 Connecting to Snowflake...")
    conn = snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=FULL_ACCOUNT,
        role=SNOWFLAKE_ROLE,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA
    )
    cursor = conn.cursor()

    try:
        print("📤 Uploading CSV to Snowflake stage...")
        cursor.execute(f"PUT file://{file_path} @{SNOWFLAKE_STAGE} AUTO_COMPRESS=FALSE")
        print("✅ Upload complete!")
    except Exception as e:
        print(f"❌ Failed to upload CSV: {e}")
    finally:
        cursor.close()
        conn.close()

# ---------- STEP 3: EXECUTE SETUP QUERIES IN SNOWFLAKE ---------- #
def run_sql_setup():
    sql_commands = [
        f"LIST @{SNOWFLAKE_STAGE}",
        """
        CREATE OR REPLACE TABLE NVDA_FINANCIAL_DATA (
            Report_Date TIMESTAMP,
            Enterprise_Value FLOAT,
            Enterprise_Value_By_EBITDA FLOAT,
            Enterprise_Value_By_Revenue FLOAT,
            Forward_PE FLOAT,
            Market_Cap FLOAT,
            Price_Book_Ratio FLOAT,
            Price_Earnings_Ratio FLOAT,
            Price_Earnings_Growth_Ratio FLOAT,
            Price_Sales_Ratio FLOAT
        )
        """,
        f"""
        COPY INTO NVDA_FINANCIAL_DATA
        FROM @{SNOWFLAKE_STAGE}/nvdia_transformed_dataset.csv
        FILE_FORMAT = (TYPE = CSV, SKIP_HEADER = 1)
        """
    ]

    print("⚙️ Executing SQL setup in Snowflake...")
    conn = snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=FULL_ACCOUNT,
        role=SNOWFLAKE_ROLE,
        warehouse=SNOWFLAKE_WAREHOUSE,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA
    )
    cursor = conn.cursor()

    try:
        for stmt in sql_commands:
            stmt_clean = stmt.strip()
            if stmt_clean:
                cursor.execute(stmt_clean)
                print(f"✅ Executed: {stmt_clean.splitlines()[0].strip()}")
    except Exception as e:
        print(f"❌ Error executing setup: {e}")
    finally:
        cursor.close()
        conn.close()

# ---------- MAIN EXECUTION ---------- #
if __name__ == "__main__":
    df = fetch_nvda_financials()
    upload_csv_to_stage(CSV_OUTPUT_FILE)
    run_sql_setup()
    print("🎯 All steps completed. Data is live in Snowflake!")
