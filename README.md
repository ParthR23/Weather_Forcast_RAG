# AI Weather & RAG Agent

An intelligent AI agent built with **LangGraph**, **LangChain**, and **Streamlit**. It can fetch real-time weather data and answer questions from uploaded PDF documents using RAG (Qdrant).

## Features
- **Smart Routing:** The agent decides whether to use the Weather tool or the RAG tool based on the user's question.
- **RAG Pipeline:** Ingests PDFs, creates embeddings (HuggingFace), and stores them in a local Vector DB (Qdrant).
- **Tools:**
  - `get_weather`: Fetches live weather from OpenWeatherMap.
  - `search_knowledge_base`: Retrieves context from uploaded documents.
- **Evaluation:** Integrated with LangSmith for tracing and monitoring.

## Tech Stack
- **Framework:** LangChain, LangGraph
- **LLM:** Llama 3.3 70B (via Groq)
- **Vector DB:** Qdrant (In-Memory)
- **UI:** Streamlit
- **Testing:** Unittest

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <YOUR_REPO_URL>
   cd Weather_Forcast_RAG