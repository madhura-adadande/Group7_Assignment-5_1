# frontend/app.py

import streamlit as st
import requests

# ✅ Backend FastAPI URL (adjust for deployment)
FASTAPI_URL = "http://localhost:8000"

# 🧭 Streamlit Config
st.set_page_config(page_title="NVIDIA Agentic Research Assistant", layout="wide")
st.sidebar.title("🔎 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Landing Page", "🧠 Multi-Agent Research"])

# --------------------------------------------------
# 🏠 LANDING PAGE
# --------------------------------------------------
if page == "🏠 Landing Page":
    st.title("💼 NVIDIA Agentic Research Assistant")
    st.markdown("---")
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/2/21/Nvidia_logo.svg/1920px-Nvidia_logo.svg.png", width=200)

    st.markdown("""
    Welcome to the **AI-Powered Research Assistant for NVIDIA** — a LangGraph-based multi-agent system that leverages:
    
    - 🤖 **RAG Agent** for historical report analysis (Pinecone + OpenAI)
    - 📊 **Snowflake Agent** for structured financial valuation & visualizations
    - 🌐 **Web Search Agent** for real-time insights using Tavily

    This project was built as part of a Data Engineering assignment, demonstrating LangGraph orchestration, agentic workflows, and hybrid data intelligence for financial analysis.

    ---
    #### 🚀 Technologies Used:
    - LangChain + LangGraph
    - FastAPI + Streamlit
    - Pinecone Vector DB
    - Snowflake Data Warehouse
    - OpenAI GPT-4o API
    - Tavily (Real-time Search API)
    - Docker-based Deployment

    ---
    👇 Head to the **Multi-Agent Research** tab to try it live!
    """)

# --------------------------------------------------
# 🧠 MULTI-AGENT RESEARCH
# --------------------------------------------------
elif page == "🧠 Multi-Agent Research":
    st.title("🧠 NVIDIA Multi-Agent Research Assistant")
    st.markdown("Configure the query below and run the agents for insights.")

    # Step 1: Year & Quarter Selection
    st.subheader("📅 Select Time Period")
    selected_year = st.selectbox("Select Year", ["2021", "2022", "2023", "2024", "2025"], index=3)
    selected_quarter = st.multiselect("Select Quarter(s)", ["Q1", "Q2", "Q3", "Q4"], default=["Q1"])

    # Step 2: Agent Selection
    st.subheader("🧠 Choose Agents to Activate")
    col1, col2, col3 = st.columns(3)
    with col1:
        rag = st.checkbox("📘 RAG Agent (Historical Reports)", value=True)
    with col2:
        snowflake = st.checkbox("📊 Snowflake Agent (Valuation Data)", value=True)
    with col3:
        web = st.checkbox("🌐 Web Search Agent (Live News)", value=True)

    selected_tools = []
    if rag:
        selected_tools.append("vector_search")
    if snowflake:
        selected_tools.append("snowflake_query")
    if web:
        selected_tools.append("web_search")

    # Step 3: Query Input
    st.subheader("💬 Research Question")
    user_query = st.text_input("Ask your question about NVIDIA:", placeholder="e.g., How did NVIDIA perform in Q1 2024 in terms of revenue and AI developments?")

    # Step 4: Submit
    if st.button("🚀 Run Research"):
        if not user_query or not selected_tools or not selected_quarter:
            st.warning("⚠️ Please enter a query, select at least one agent, and choose a quarter.")
        else:
            with st.spinner("🤖 Running LangGraph Agents..."):
                payload = {
                    "query": user_query,
                    "year": selected_year,
                    "quarter": selected_quarter,
                    "tools": selected_tools
                }

                try:
                    response = requests.post(f"{FASTAPI_URL}/query_multiagent", json=payload)

                    if response.status_code == 200:
                        result = response.json()

                        st.success("✅ Final Research Report Generated:")
                        st.markdown(f"**Query:** {result['query']}")
                        st.markdown(f"**Year:** {result['year']} &nbsp;&nbsp;&nbsp; **Quarter(s):** {', '.join(result['quarter'])}")
                        st.markdown(f"**Tools Used:** {', '.join(result['tools_used'])}")
                        st.markdown("---")
                        st.markdown(result["final_answer"])
                    else:
                        st.error(f"❌ Error from backend: {response.text}")
                except Exception as e:
                    st.error(f"⚠️ Could not connect to backend: {e}")
