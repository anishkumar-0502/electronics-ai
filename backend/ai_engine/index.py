import os
import logging
import faiss
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pathlib import Path
from datetime import datetime
from utils import get_chunked_documents
from config import configure_settings

# Constants
PERSIST_DIR = "data/faiss"  # Defined but not used for persistence
DATASHEET_DIR = "data/datasheets"
MEMORY_DIR = "data/docs"
os.makedirs(MEMORY_DIR, exist_ok=True)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure settings
configure_settings()

# Global index to cache in-memory instance
_index = None

def get_documents():
    """Load documents from memory and datasheet directories"""
    if not any(Path(MEMORY_DIR).iterdir()) and not any(Path(DATASHEET_DIR).iterdir()):
        with open(os.path.join(MEMORY_DIR, "memory.txt"), "w", encoding="utf-8") as f:
            f.write("The system has not yet learned anything.")

    documents = []
    if os.path.exists(MEMORY_DIR) and os.listdir(MEMORY_DIR):
        documents.extend(SimpleDirectoryReader(MEMORY_DIR).load_data())
    if os.path.exists(DATASHEET_DIR) and os.listdir(DATASHEET_DIR):
        documents.extend(get_chunked_documents(SimpleDirectoryReader(DATASHEET_DIR).load_data()))
    return documents

def create_index():
    """Create a new FAISS index (in-memory)"""
    global _index
    documents = get_documents()
    if not documents:
        raise ValueError("No documents available to create index")

    # Initialize FAISS index with the correct dimension (e.g., 384 for all-MiniLM-L6-v2)
    dimension = 384
    faiss_index = faiss.IndexFlatL2(dimension)
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    _index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
        store_nodes_override=True
    )
    
    logger.info(f"New index created at {datetime.now().isoformat()}")
    return _index

def load_or_build_index():
    """Load existing in-memory index or create new if not available"""
    global _index
    if _index is None:
        logger.info("Creating new in-memory index...")
        return create_index()
    logger.info("Reusing existing in-memory index")
    return _index

# Maintain backward compatibility alias
load_index = load_or_build_index