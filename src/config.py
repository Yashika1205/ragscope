import os 
from dataclasses import dataclass 
from pathlib import Path 
 
from dotenv import load_dotenv 
 
load_dotenv() 
 
BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_DIR = BASE_DIR / "data" 
RESULTS_DIR = BASE_DIR / "results" 
CHROMA_DIR = BASE_DIR / "chroma_db" 
 
 
@dataclass(frozen=True) 
class Settings: 
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2") 
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini") 
    llm_api_key: str = os.getenv("LLM_API_KEY", "") 
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500")) 
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50")) 
    top_k: int = int(os.getenv("TOP_K", "4")) 
 
 
settings = Settings() 
