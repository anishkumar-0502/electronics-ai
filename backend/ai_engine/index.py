import os
import logging
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import TextNode
import faiss
from utils import get_chunked_documents
from config import configure_settings
from datetime import datetime

# Constants
STORAGE_DIR = "storage"
DATASHEET_DIR = "data/datasheets"
FAISS_INDEX_PATH = "data/faiss/faiss_index.index"

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure settings
configure_settings()

def build_index():
    """Build and persist a new index from datasheets"""
    if not os.path.exists(DATASHEET_DIR) or not os.listdir(DATASHEET_DIR):
        raise FileNotFoundError("No datasheets found in 'data/datasheets'.")
    
    documents = get_chunked_documents(SimpleDirectoryReader(DATASHEET_DIR).load_data())
    nodes = documents if isinstance(documents[0], TextNode) else SimpleNodeParser().get_nodes_from_documents(documents)

    faiss_index = faiss.IndexFlatL2(384)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build index with nodes and ensure text is stored
    index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    storage_context.persist(persist_dir=STORAGE_DIR)  # Corrected persist call
    logger.info(f"Index built and saved at {datetime.now().isoformat()}")
    return index

def load_or_build_index():
    """Load existing index or build a new one if not found"""
    try:
        if os.path.exists(FAISS_INDEX_PATH):
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR, vector_store=vector_store)
            index = VectorStoreIndex.from_vector_store(
                vector_store,
                storage_context=storage_context,
                show_progress=True
            )
            logger.info("Index loaded successfully")
            return index
        logger.warning("FAISS index not found, rebuilding...")
        return build_index()
    except Exception as e:
        logger.error(f"Failed to load index, rebuilding... Reason: {e}")
        return build_index()
