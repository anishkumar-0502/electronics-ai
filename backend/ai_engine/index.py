import os
import logging
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader, Settings
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pathlib import Path
from datetime import datetime
from utils import get_chunked_documents
from config import configure_settings

# Constants
PERSIST_DIR = "data/simple"
DATASHEET_DIR = "data/datasheets"
MEMORY_DIR = "data/docs"
os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)  # Ensure directory exists

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
    """Create a new SimpleVectorStore index (in-memory with persistence option)"""
    global _index
    documents = get_documents()
    if not documents:
        raise ValueError("No documents available to create index")

    vector_store = SimpleVectorStore()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    try:
        logger.info("Creating index from documents...")
        _index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True,
            store_nodes_override=True
        )
        logger.info("Index created successfully")
        
        vector_store.persist(os.path.join(PERSIST_DIR, "vector_store.json"))
        logger.info(f"New index persisted to {os.path.join(PERSIST_DIR, 'vector_store.json')} at {datetime.now().isoformat()}")
    except Exception as e:
        logger.error(f"Failed to create or persist index: {str(e)}")
        raise
    
    return _index

def load_or_build_index():
    """Load existing persisted index or create new if not available"""
    global _index
    vector_store_path = os.path.join(PERSIST_DIR, "vector_store.json")
    logger.info(f"Attempting to load persisted index from {vector_store_path}")
    try:
        # Attempt to load from persisted storage
        if os.path.exists(vector_store_path):
            vector_store = SimpleVectorStore.from_persist_path(vector_store_path)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            # Workaround for missing docstore attribute
            if vector_store._index_struct.index.nodes:  # Check internal nodes
                _index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
                logger.info(f"Loaded existing persisted index from {vector_store_path}")
                return _index
            else:
                logger.warning(f"Persisted vector store at {vector_store_path} is empty or invalid. Rebuilding...")
                os.remove(vector_store_path)
        else:
            logger.info(f"No persisted index found at {vector_store_path}. Creating new...")
    except Exception as e:
        logger.warning(f"Failed to load persisted index: {str(e)}. Rebuilding...")

    # Fall back to cached or new index
    if _index is None:
        logger.info("Creating new in-memory index...")
        return create_index()
    logger.info("Reusing existing in-memory index")
    return _index

# Maintain backward compatibility alias
load_index = load_or_build_index