import os

# LANGSMITH SETUP
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = ""

import streamlit as st
import os
import tempfile
from langchain_core.messages import HumanMessage, AIMessage

# Import our new RAG processor and the Graph
from src.rag import process_pdf
from src.graph import app as graph_app

st.set_page_config(page_title="AI Engineer Assignment", layout="wide")
st.title("🤖 AI Agent: Weather & RAG")
st.markdown("I can check the weather OR answer questions from your PDF.")

# --- SIDEBAR: PDF UPLOAD ---
with st.sidebar:
    st.header("📂 Knowledge Base")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    
    if uploaded_file:
        # Save file temporarily so we can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
            
        if st.button("Process PDF"):
            with st.spinner("Ingesting into Qdrant..."):
                status = process_pdf(tmp_path)
                st.success(status)
                os.remove(tmp_path)  # Cleanup

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        if msg.content:
            with st.chat_message("assistant"):
                st.write(msg.content)

# Handle Input
if prompt := st.chat_input("Ask about weather or the PDF..."):
    
    with st.chat_message("user"):
        st.write(prompt)
    
    current_messages = st.session_state.messages + [HumanMessage(content=prompt)]
    
    with st.spinner("Thinking..."):
        response_dict = graph_app.invoke({"messages": current_messages})
        updated_messages = response_dict["messages"]
        final_response = updated_messages[-1].content
    
    with st.chat_message("assistant"):
        st.write(final_response)
    
    st.session_state.messages = updated_messages