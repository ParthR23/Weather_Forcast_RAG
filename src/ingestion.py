import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Configuration
PDF_PATH = "./data/NIPS-2017-attention-is-all-you-need-Paper.pdf"
COLLECTION_NAME = "agent_knowledge"
QDRANT_PATH = "./qdrant_db"

def ingestion_document():
    """
    Loads the PDF, splits it into chunks, and stores it in Qdrant.
    """
    # 1. Check if the file exists
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}. Please add a file.")

    print(f"Starting ingestion for: {PDF_PATH}")

    # 2. Load the PDF
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")

    # 3. Split Text 
    # We use a chunk size of 1000 with overlap to preserve context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    print(f"Split into {len(splits)} chunks.")

    # 4. Initialize Qdrant Client (Local mode)
    #embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 5. Initialize Qdrant Client (Local mode)
    client = QdrantClient(path=QDRANT_PATH)

    # Check if collection exists, if not create it
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        print(f"Created Qdrant collection: {COLLECTION_NAME}")

    # 6. Store in Qdrant
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    vector_store.add_documents(documents=splits)
    print(f"Ingestion Complete! Embeddings stored in Qdrant")

    return vector_store

def get_retriever():
    """
    Returns the retriever object for the LangGraph agent to use it later.
    """
    #embeddings = GoogleGenerativeAIEmbeddings(
        #model="text-embedding-004"
    #)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    client = QdrantClient(path=QDRANT_PATH)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    # Return a retriever that fetches the top 3 most relevant chunks
    return vector_store.as_retriever(search_kwargs={"k": 3})

if __name__ == "__main__":
    ingestion_document()