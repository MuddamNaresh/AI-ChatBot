"""
Configuration module for the AI Chatbot application.
Handles environment variables, default model settings, and file paths.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory: root of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file if available
ENV_FILE = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE)

# Application Information
APP_NAME = "AuraGen AI"
APP_VERSION = "1.0.0"

# Directories for persistence
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
VECTORS_DIR = DATA_DIR / "vectors"
DB_PATH = DATA_DIR / "chatbot.db"
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"

# Ensure runtime directories exist
for directory in [DATA_DIR, UPLOADS_DIR, VECTORS_DIR, SAMPLE_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Google Gemini API Settings
# Supported default models:
# gemini-3.8-flash: Recommended for fast, high-quality conversational intelligence
# gemini-3.5-flash-lite: Ultra-fast and lightweight
# gemini-embedding-001: Standard text embedding model for RAG vector search
DEFAULT_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")
DEFAULT_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

# RAG & Chunking Hyperparameters
DEFAULT_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "750"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.25"))

# Server Settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

def get_gemini_api_key() -> str:
    """Retrieve Gemini API key from environment, re-reading .env if needed."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key and ENV_FILE.exists():
        load_dotenv(dotenv_path=ENV_FILE, override=True)
        key = os.getenv("GEMINI_API_KEY", "").strip()
    return key
