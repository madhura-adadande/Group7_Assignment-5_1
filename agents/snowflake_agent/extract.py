import os
from pathlib import Path
import pandas as pd
import requests
import snowflake.connector
from datetime import datetime, timezone
from dotenv import load_dotenv

# ---------- LOAD .env FILE SAFELY ---------- #
env_path = Path(__file__).resolve().parent / ".env"
if not env_path.exists():
    raise FileNotFoundError(f"❌ .env file not found at {env_path}")
load_dotenv(dotenv_path=env_path)

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
QUICK_FS_API_KEY = os.getenv("QUICK_FS_API_KEY")

FULL_ACCOUNT = f"{SNOWFLAKE_ACCOUNT}.{SNOWFLAKE_REGION}"
CSV_OUTPUT_FILE = "nvdia_valuation_quarters.csv"  # ✅ CHANGED FILE NAME
TICKER_SYMBOL = "NVDA"

# ---------- STEP 1: FETCH DATA FROM QUICKFS ---------- #
def fetch_quickfs_data():
    print(f"📡 Fetching {TICKER_SYMBOL} data from QuickFS...")
    url = f"https://public-api.quickfs.net/v1/data/all-data/{TICKER_SYMBOL}?api_key={QUICK_FS_API_KEY}"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"❌ QuickFS API Error: {response.status_code} - {response.text}")

    raw = response.json()
    df_raw = pd.DataFrame(raw['data']['financials']['quarterly'])
    df_raw['period_end_date'] = pd.to_datetime(df_raw['period_end_date'])

    # Filter: 2021–2025
    df = df_raw[
        (df_raw['period_end_date'] >= '2021-01-01') &
        (df_raw['period_end_date'] <= '2025-12-31')
    ].copy()

    # Derived metrics
    df['trailing_pe'] = df.get('period_end_price') / df.get('eps_diluted').rolling(window=4).sum()
    df['ev_to_ebitda'] = df.get('enterprise_value') / df.get('ebitda')

    # Final DataFrame
    df_final = pd.DataFrame({
        "symbol": TICKER_SYMBOL,
        "period_end_date": df.get("period_end_date"),
        "market_cap": df.get("market_cap"),
        "enterprise_value": df.get("enterprise_value"),
        "trailing_pe": df.get("trailing_pe"),
        "forward_pe": df.get("forward_pe"),
        "peg_ratio": df.get("peg_ratio"),
        "price_to_sales": df.get("price_to_sales"),
        "price_to_book": df.get("price_to_book"),
        "enterprise_value_to_sales": df.get("enterprise_value_to_sales"),
        "ev_to_ebitda": df.get("ev_to_ebitda")
    })

    df_final = df_final.where(pd.notnull(df_final), None)
    df_final["period_end_date"] = pd.to_datetime(df_final["period_end_date"]).dt.strftime("%Y-%m-%d")

    df_final.to_csv(CSV_OUTPUT_FILE, index=False)
    print(f"✅ Saved cleaned CSV to {CSV_OUTPUT_FILE}")
    return df_final

# ---------- STEP 2: UPLOAD CSV TO STAGE ---------- #
def upload_csv_to_stage(file_path: str):
    print("🔗 Connecting to Snowflake for CSV upload...")
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
        print("📤 Uploading CSV to stage...")
        cursor.execute(f"PUT file://{file_path} @{SNOWFLAKE_STAGE} AUTO_COMPRESS=FALSE")
        print("✅ Upload to stage complete.")
    except Exception as e:
        print(f"❌ CSV upload failed: {e}")
    finally:
        cursor.close()
        conn.close()

# ---------- STEP 3: CREATE TABLE & COPY INTO ---------- #
def run_sql_setup():
    ddl = """
    CREATE OR REPLACE TABLE QUICKFS_NVDA_VALUATION (
        SYMBOL STRING,
        PERIOD_END_DATE DATE,
        MARKET_CAP FLOAT,
        ENTERPRISE_VALUE FLOAT,
        TRAILING_PE FLOAT,
        FORWARD_PE FLOAT,
        PEG_RATIO FLOAT,
        PRICE_TO_SALES FLOAT,
        PRICE_TO_BOOK FLOAT,
        ENTERPRISE_VALUE_TO_SALES FLOAT,
        EV_TO_EBITDA FLOAT
    );
    """
    copy = f"""
    COPY INTO QUICKFS_NVDA_VALUATION
    FROM @{SNOWFLAKE_STAGE}/{CSV_OUTPUT_FILE}
    FILE_FORMAT = (
        TYPE = CSV,
        SKIP_HEADER = 1,
        FIELD_OPTIONALLY_ENCLOSED_BY = '"',
        ERROR_ON_COLUMN_COUNT_MISMATCH = FALSE
    )
    ON_ERROR = 'SKIP_FILE'
    """

    print("📥 Executing DDL + COPY INTO Snowflake...")
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
        cursor.execute(ddl)
        cursor.execute(copy)
        print("✅ Table created and data copied.")
    except Exception as e:
        print(f"❌ Snowflake SQL Error: {e}")
    finally:
        cursor.close()
        conn.close()

# ---------- MAIN ---------- #
if __name__ == "__main__":
    fetch_quickfs_data()
    upload_csv_to_stage(CSV_OUTPUT_FILE)
    run_sql_setup()
    print("🎯 Done! QuickFS quarterly valuation data is now in Snowflake.")
