# frontend/app.py
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import streamlit as st
import requests
import json
from typing import List, Dict, Any

# Import using full path
from backend.langgraph.orchestrator import ResearchOrchestrator

# API endpoint
API_URL = "http://localhost:8000"  # Local development

def generate_research_report(query: str, year: int, quarter: int, agents: List[str]) -> Dict[str, Any]:
    """
    Send research request to backend API
    """
    payload = {
        "query": query,
        "year": year,
        "quarter": quarter,
        "agents": agents
    }

    try:
        response = requests.post(f"{API_URL}/research", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to research API: {e}")
        return {}

def display_sources(sources: List[str], title: str):
    """
    Display sources in an expandable section
    """
    with st.expander(f"{title} Sources"):
        for source in sources:
            st.write(source)

def main():
    # Set page configuration
    st.set_page_config(
        page_title="NVIDIA Research Assistant",
        page_icon=":chart_with_upwards_trend:",
        layout="wide"
    )

    # Title and description
    st.title("🚀 NVIDIA Research Assistant")
    st.write("Unlock comprehensive insights about NVIDIA using AI-powered research")

    # Query input
    query = st.text_input(
        "Research Question", 
        placeholder="What are the latest developments in NVIDIA's AI technology?"
    )

    # Filtering options
    col1, col2 = st.columns(2)
    
    with col1:
        year = st.selectbox(
            "Select Year", 
            [None, 2020, 2021, 2022, 2023, 2024],
            format_func=lambda x: "All Years" if x is None else str(x)
        )
    
    with col2:
        quarter = st.selectbox(
            "Select Quarter", 
            [None, 1, 2, 3, 4],
            format_func=lambda x: "All Quarters" if x is None else f"Q{x}"
        )

    # Agent selection
    st.write("### Select Research Sources")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_rag = st.checkbox("Historical Data", value=True)
    with col2:
        use_snowflake = st.checkbox("Financial Metrics", value=True)
    with col3:
        use_websearch = st.checkbox("Latest News", value=True)

    # Generate report button
    if st.button("Generate Research Report", type="primary"):
        # Validate input
        if not query:
            st.warning("Please enter a research question")
            return

        # Prepare agents list
        selected_agents = []
        if use_rag:
            selected_agents.append("rag")
        if use_snowflake:
            selected_agents.append("snowflake")
        if use_websearch:
            selected_agents.append("websearch")

        # Show loading spinner
        with st.spinner("Generating comprehensive research report..."):
            try:
                # Send request to backend
                result = generate_research_report(
                    query, 
                    year, 
                    quarter, 
                    selected_agents
                )

                # Display report
                if result:
                    st.markdown("## Research Report")
                    st.write(result.get("content", "No report generated"))

                    # Display sources
                    if "historical_data" in result:
                        display_sources(
                            result["historical_data"].get("sources", []), 
                            "Historical Data"
                        )
                    
                    if "financial_metrics" in result:
                        display_sources(
                            result["financial_metrics"].get("sources", []), 
                            "Financial Metrics"
                        )
                    
                    if "latest_insights" in result:
                        display_sources(
                            result["latest_insights"].get("sources", []), 
                            "Latest Insights"
                        )

                    # Display chart if available
                    if "financial_metrics" in result and result["financial_metrics"].get("chart"):
                        st.image(result["financial_metrics"]["chart"], caption="NVIDIA Market Cap Trend")

            except Exception as e:
                st.error(f"Error generating report: {e}")

if __name__ == "__main__":
    main()