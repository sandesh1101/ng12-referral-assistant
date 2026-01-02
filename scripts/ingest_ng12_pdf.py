import os
import sys
import time
import vertexai

# Add project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_vertexai import VertexAIEmbeddings

# Import config
import app.config as config  

PDF = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "data", "ng12.pdf"))
VECTOR_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "vectorstore", "chroma"))

def ingest():
    """
    Ingests the NG12 PDF into the Chroma vector store.
    Handles initialization, text splitting, and batched embedding with rate limiting.
    """
    print(f"Connecting to Google Cloud Project: {config.PROJECT_ID} in {config.LOCATION}")
    vertexai.init(project=config.PROJECT_ID, location=config.LOCATION)

    print(f"DEBUG: Using Embedding Model -> '{config.EMBEDDING_MODEL}'")

    if not os.path.exists(PDF):
        print(f"Error: PDF file not found at: {PDF}")
        return

    print("Loading PDF...")
    loader = PyPDFLoader(PDF)
    docs = loader.load()
    
    print("Splitting text...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} text chunks.")

    emb = VertexAIEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        project=config.PROJECT_ID,
        location=config.LOCATION
    )

    batch_size = 10
    vector_store = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=emb)

    print(f"Ingesting to {VECTOR_DB_PATH}...")
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vector_store.add_documents(batch)
        print(f"Processed batch {i // batch_size + 1} / {(len(chunks) // batch_size) + 1}")
        time.sleep(1.0)  # Increased sleep slightly for safety

    print("Done.")

if __name__ == "__main__":
    ingest()