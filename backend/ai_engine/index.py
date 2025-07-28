import os
import logging
import faiss
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    SimpleDirectoryReader,
    Settings
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pathlib import Path
from datetime import datetime
from utils import get_chunked_documents
from config import configure_settings

# Constants
PERSIST_DIR = "data/faiss"
DATASHEET_DIR = "data/datasheets"
MEMORY_DIR = "data/docs"
os.makedirs(MEMORY_DIR, exist_ok=True)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure settings
configure_settings()

def get_documents():
    """Load documents from memory and datasheet directories"""
    if not any(Path(MEMORY_DIR).iterdir()) and not any(Path(DATASHEET_DIR).iterdir()):
        with open(os.path.join(MEMORY_DIR, "memory.txt"), "w") as f:
            f.write("The system has not yet learned anything.")

    documents = []
    if os.path.exists(MEMORY_DIR) and os.listdir(MEMORY_DIR):
        documents.extend(SimpleDirectoryReader(MEMORY_DIR).load_data())
    if os.path.exists(DATASHEET_DIR) and os.listdir(DATASHEET_DIR):
        documents.extend(get_chunked_documents(SimpleDirectoryReader(DATASHEET_DIR).load_data()))
    return documents

def create_index():
    """Create a new FAISS index with proper persistence"""
    documents = get_documents()
    if not documents:
        raise ValueError("No documents available to create index")

    # Initialize FAISS index
    dimension = 384
    faiss_index = faiss.IndexFlatL2(dimension)
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=PERSIST_DIR
    )

    # Create index with embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
        store_nodes_override=True
    )
    
    # Persist index
    index.storage_context.persist()
    logger.info(f"New index created at {datetime.now().isoformat()}")
    return index
# In index.py
def load_or_build_index():
    """Load existing index or create new if not found"""
    try:
        if not os.path.exists(os.path.join(PERSIST_DIR, "docstore.json")):
            raise FileNotFoundError("No existing index found")

        vector_store = FaissVectorStore.from_persist_dir(PERSIST_DIR)
        storage_context = StorageContext.from_defaults(
            persist_dir=PERSIST_DIR,
            vector_store=vector_store
        )
        
        index = VectorStoreIndex.load_from_storage(storage_context)
        logger.info("Existing index loaded successfully")
        return index
        
    except Exception as e:
        logger.warning(f"Failed to load index: {str(e)}")
        logger.info("Creating new index...")
        return create_index()

# Maintain backward compatibility alias
load_index = load_or_build_index
