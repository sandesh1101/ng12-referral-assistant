from pathlib import Path
from langchain_chroma import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
from app.config import EMBEDDING_MODEL, PROJECT_ID, LOCATION

BASE_DIR = Path(__file__).resolve().parent.parent
# Assuming structure is app/tools/guideline_rag.py -> up to app/ -> vectorstore is in app/vectorstore
VECTOR_DB_PATH = str(BASE_DIR / "vectorstore" / "chroma")

emb = VertexAIEmbeddings(model_name=EMBEDDING_MODEL, project=PROJECT_ID, location=LOCATION)
store = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=emb)

def search_guidelines(query, k=5):
    """Searches the vector store for relevant guideline chunks."""
    return store.similarity_search(query, k=k)
