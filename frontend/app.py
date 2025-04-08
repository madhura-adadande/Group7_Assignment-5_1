import streamlit as st
import requests
import json
import time

st.set_page_config(page_title="NVIDIA Research Assistant", layout="wide")

API_URL = "http://localhost:8000/query_research_agent"  # ✅ FastAPI backend endpoint

# === PAGE 1: Landing ===
def landing_page():
    st.title("🧠 NVIDIA Research Assistant (Multi-Agent)")
    st.markdown("### Assignment Overview")

    st.markdown("""
    This research assistant uses **three AI agents** orchestrated via **LangGraph** to generate structured reports on NVIDIA:

    - 🧩 **RAG Agent (Pinecone)**: Retrieves historical financial report context using metadata filtering (Year + Quarter).
    - 📊 **Snowflake Agent**: Connects to Snowflake to query structured valuation data and generates charts + summaries.
    - 🌐 **Web Search Agent**: Fetches real-time news, trends, and updates from the internet (Tavily API).

    ---
    ### ✅ Key Objectives:
    - Combine multiple data modalities (structured + unstructured + real-time).
    - Enable Year/Quarter filtering for contextual insights.
    - Generate detailed research reports with summaries, visualizations, and sources.
    - Deploy using Docker with FastAPI backend and Streamlit frontend.
    """)

# === PAGE 2: Research Assistant ===
def research_assistant_page():
    st.title("🔎 Ask NVIDIA a Research Question")

    with st.form("query_form"):
        col1, col2 = st.columns(2)
        with col1:
            year = st.selectbox("Select Year", ["2021", "2022", "2023", "2024", "2025"])
        with col2:
            quarter = st.selectbox("Select Quarter", ["Q1", "Q2", "Q3", "Q4"])

        st.markdown("### 🛠️ Choose Agents to Use")
        use_rag = st.checkbox("RAG Agent (Historical)", value=True)
        use_snowflake = st.checkbox("Snowflake Agent (Structured Valuation)", value=True)
        use_web = st.checkbox("Web Search Agent (Real-Time)", value=True)

        query = st.text_area("💬 Enter your research question", placeholder="e.g., How did NVIDIA perform in Q2 2023?")
        submitted = st.form_submit_button("🚀 Run Research")

    if submitted:
        if not query.strip():
            st.warning("Please enter a research question.")
            return

        selected_tools = []
        if use_rag:
            selected_tools.append("rag_retrieve_chunks")
        if use_snowflake:
            selected_tools.append("snowflake_query")
        if use_web:
            selected_tools.append("web_search")

        if not selected_tools:
            st.warning("Please select at least one agent.")
            return

        selected_tools.append("final_answer")  # ✅ Final synthesis is mandatory

        with st.spinner("🔄 Querying agents and generating report..."):
            try:
                response = requests.post(API_URL, json={
                    "query": query,
                    "year": year,
                    "quarter": [quarter],
                    "tools": selected_tools
                })
                response.raise_for_status()
                answer = response.json().get("answer", {})
                time.sleep(0.3)
                display_report(answer)
            except Exception as e:
                st.error(f"❌ Failed to retrieve response: {e}")

# === Display Final Report ===
def display_report(result):
    st.markdown("## 📄 Research Report")

    if isinstance(result, str):
        st.markdown("### 🧾 Summary")
        st.markdown(result)
        return

    st.markdown("### 🔍 Research Steps")
    st.code(result.get("research_steps", "Not available"))

    st.markdown("### 🧠 Historical Performance")
    st.markdown(result.get("historical_performance", "Not available"))

    st.markdown("### 📊 Financial Analysis")
    financial_block = result.get("financial_analysis", "Not available")

    if isinstance(financial_block, str):
        lines = financial_block.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("https://") and "amazonaws.com" in line and line.endswith(".png"):
                # Dynamic captions based on filename
                if "peg" in line:
                    caption = "PEG Ratio"
                elif "market_value" in line:
                    caption = "Market Cap vs Enterprise Value"
                elif "multiples" in line:
                    caption = "Enterprise Multiples"
                else:
                    caption = "Financial Chart"
                st.image(line, caption=f"🖼️ {caption}", use_column_width=True)
            else:
                st.markdown(line)
    else:
        st.markdown(financial_block)

    st.markdown("### 🌍 Industry Insights")
    st.markdown(result.get("industry_insights", "Not available"))

    st.markdown("### 🧾 Final Summary")
    st.markdown(result.get("summary", "Not available"))

    st.markdown("### 🔗 Sources")
    sources = result.get("sources", "Not available")
    if isinstance(sources, list):
        for src in sources:
            st.markdown(f"- {src}")
    else:
        st.code(sources)

    with st.expander("🧪 Raw Result (Debug)", expanded=False):
        st.json(result)

# === Page Router ===
page = st.sidebar.radio("📁 Select Page", ["🏠 Landing Page", "🤖 Research Assistant"])
if page == "🏠 Landing Page":
    landing_page()
else:
    research_assistant_page()
