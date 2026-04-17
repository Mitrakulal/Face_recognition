from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "phi3"
RETRIEVAL_K = 3
CHAT_HISTORY_LEN = 5
RECOGNITION_THRESHOLD = 0.4

FAISS_INDEX_DIR = PROJECT_ROOT / "backend" / "rag" / "faiss_index"
EMBEDDINGS_DIR = PROJECT_ROOT / "backend" / "vision" / "embeddings"
DATA_DIR = PROJECT_ROOT / "data"
