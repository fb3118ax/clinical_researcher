# config.py
# Central configuration for the Clinical Research Assistant pipeline
from dotenv import load_dotenv
import os
load_dotenv()

# --- Paths ---
DATA_FOLDER = "./pdf/"
VECTOR_STORE_DIR = "./research_asst_db"

# --- Chunking ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# --- Embedding & LLM ---
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"

# --- Retriever ---
TOP_K = 3

# --- Vector Store ---
COLLECTION_METADATA = {"hnsw:space": "cosine"}

# --- Keys ---
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]