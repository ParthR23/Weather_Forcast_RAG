import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# 1. Initialize Embeddings (Free, runs locally)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Initialize Qdrant Client (In-Memory)
# FIX: Pass ":memory:" directly, NOT as 'url='
client = QdrantClient(":memory:")

# Create collection immediately
collection_name = "knowledge_base"
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

# 3. Create Vector Store wrapper
vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)

def process_pdf(file_path):
    """
    Reads a PDF, splits it into chunks, and stores them in Qdrant.
    """
    try:
        # Load PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # Split Text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        
        # Add to Qdrant
        vector_store.add_documents(documents=splits)
        
        return f"Successfully processed {len(splits)} chunks from the PDF."
    except Exception as e:
        return f"Error processing PDF: {e}"

def query_qdrant(query: str):
    """
    Searches the Qdrant database for relevant text chunks.
    """
    try:
        results = vector_store.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in results])
        
        if not context:
            return "No relevant information found in the document."
            
        return context
    except Exception as e:
        return f"Error querying database: {e}"