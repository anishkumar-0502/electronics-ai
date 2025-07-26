from langchain_ollama import OllamaLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
import os
from dotenv import load_dotenv

load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.5))
def configure_settings():
    """Configure global settings for embeddings and LLM"""
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.embed_model = embed_model

    llm = OllamaLLM(
        model=OLLAMA_MODEL,
        temperature=TEMPERATURE,
        base_url=OLLAMA_URL,
    )
    Settings.llm = llm
    Settings.callback_manager = CallbackManager([LlamaDebugHandler()])