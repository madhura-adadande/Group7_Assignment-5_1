# Assignment 5.1 – Multi-Agent RAG Research Assistant

## Introduction

In this project, we developed an intelligent, multi-agent research assistant that integrates structured, unstructured, and real-time data sources to deliver comprehensive financial insights about NVIDIA. This assistant uses LangGraph to orchestrate three specialized agents:

- **Snowflake Agent** – handles structured financial valuation data  
- **RAG Agent** – processes quarterly reports using Pinecone  
- **Web Search Agent** – pulls live market updates using APIs

The frontend is built with Streamlit, and the backend uses FastAPI. Both are containerized with Docker and deployed to the cloud.

---

## Technologies Used

- **Orchestration:** LangGraph  
- **Vector DB:** Pinecone  
- **Structured DB:** Snowflake  
- **Frontend:** Streamlit  
- **Backend:** FastAPI  
- **Deployment:** Docker + Render / Railway  
- **Web Search APIs:** Tavily / SerpAPI  

---

## Problem Statement

The goal of this project is to create a system that can intelligently combine historical financial data, structured company metrics, and the latest market trends. By leveraging LangGraph’s agent orchestration, the assistant enables seamless collaboration between the Snowflake, RAG, and Web agents. The result is an integrated platform for answering research questions with real-time insights, document-based context, and structured financial data.

---

##  Proof of Concept

The system is broken down into three core components:

### 🔹 RAG Agent (Unstructured Data)
- NVIDIA quarterly reports are stored in S3
- Parsed and chunked into Pinecone with Year/Quarter metadata
- Enables semantic and filtered retrieval

### 🔹 Snowflake Agent (Structured Data)
- Financial metrics from Yahoo Finance are loaded into Snowflake
- Enables SQL querying and returns formatted text summaries and charts

### 🔹 Web Search Agent (Real-time Insights)
- Uses Tavily / SerpAPI to fetch current news related to NVIDIA
- Provides market sentiment and news context

LangGraph orchestrates the agent flow, FastAPI connects frontend to backend, and the whole project runs via Docker containers.

---
[View this project as a Codelab](https://codelabs-preview.appspot.com/?file_id=10JxYB2VVfDPeJeyB4UOgK0hLkW-VULxwzqYZYmiYE_g)
##  Walkthrough of the Application
- Enter a research query and filter by **Year** and **Quarter**
- Choose agents to trigger:
  -  Snowflake Agent  
  -  RAG Agent  
  -  Web Search Agent
- View the structured research report:
  - Summaries  
  - Financial visualizations  
  - Web insights


---

graph TD
    UI[Streamlit Frontend] -->|API Call| FastAPI
    FastAPI --> LangGraph[LangGraph Multi-Agent Orchestrator]
    LangGraph --> SnowflakeAgent[Snowflake Agent]
    LangGraph --> RAGAgent[Pinecone-based RAG Agent]
    LangGraph --> WebSearchAgent[Real-time Web Search Agent]
    SnowflakeAgent --> SnowflakeDB[Snowflake Database]
    RAGAgent --> Pinecone[Pinecone Vector DB]
    WebSearchAgent --> WebAPI[SerpAPI / Tavily / Bing]

