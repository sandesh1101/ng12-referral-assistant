import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import EMBEDDING_MODEL, PROJECT_ID, LOCATION

BASE_DIR = Path(__file__).resolve().parent.parent
# Assuming structure is app/tools/guideline_rag.py -> up to app/ -> vectorstore is in app/vectorstore
VECTOR_DB_PATH = str(BASE_DIR / "vectorstore" / "chroma")

emb = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=os.getenv('GOOGLE_API_KEY'))
store = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=emb)

def search_guidelines(query, k=5):
    """Searches the vector store for relevant guideline chunks."""
    return store.similarity_search(query, k=k)
