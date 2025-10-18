# config/settings.py
import os
from pathlib import Path

# Project Root
BASE_DIR = Path(__file__).resolve().parent.parent

# Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistral:7b-instruct"  # Ollama model name

# Chunking Configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
SEPARATORS = ["\n\n", "\n", " ", ""]

# Retrieval Configuration
TOP_K = 4
SIMILARITY_THRESHOLD = None

# Paths
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
VECTOR_DB_DIR = BASE_DIR / "data" / "vector_db"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# File Configuration
ALLOWED_EXTENSIONS = ['.pdf', '.txt']
MAX_FILE_SIZE_MB = 50

# Streamlit Configuration
PAGE_TITLE = "Smart Document Q&A"
PAGE_ICON = "📚"
LAYOUT = "wide"

# LLM Configuration
TEMPERATURE = 0.1  # Lower = more deterministic
MAX_TOKENS = 512