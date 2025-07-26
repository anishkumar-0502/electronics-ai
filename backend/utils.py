import time
import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

DATASHEET_LEARNED_PATH = "data/datasheets/learned"

def get_chunked_documents(documents):
    """Split documents into chunks/nodes for FAISS indexing."""
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    return splitter.get_nodes_from_documents(documents)

def load_or_build_index():
    """Load existing FAISS index or build a new one if it doesn't exist."""
    # Implementation of this function should be here
    pass

def learn_from_interaction(question: str, answer: str):
    """
    Save the user interaction as a file and add it to the FAISS index.
    """
    os.makedirs(DATASHEET_LEARNED_PATH, exist_ok=True)
    timestamp = int(time.time())
    file_path = os.path.join(DATASHEET_LEARNED_PATH, f"interaction_{timestamp}.txt")
    
    with open(file_path, "w") as f:
        f.write(f"Q: {question}\nA: {answer}")
    
    print(f"[🧠] Learning interaction saved to {file_path}")

    # Load the new file as document
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    
    # Split documents into nodes
    nodes = get_chunked_documents(documents)
    
    # Load index and insert the new nodes
    index = load_or_build_index()
    index.insert_nodes(nodes)
    
    # Save updated index
    index.storage_context.persist()
    print("[✅] Updated FAISS index with new interaction.")
