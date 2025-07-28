from pathlib import Path

# Define storage directory
STORAGE_DIR = Path(__file__).parent / ".." / "data" / "faiss"

from .ai_engine import ask_ai, ask_ai_streaming
from .index import load_or_build_index