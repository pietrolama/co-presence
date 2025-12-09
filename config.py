"""
CO-PRESENCE Configuration Module
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))

# Ensure data directories exist
ENVIRONMENT_DIR = DATA_DIR / "environment"
WORLD_DIR = DATA_DIR / "world"
RAG_A_DIR = DATA_DIR / "rag_a"
RAG_B_DIR = DATA_DIR / "rag_b"
LOGS_DIR = DATA_DIR / "logs"
RUNS_DIR = DATA_DIR / "runs"

for d in [ENVIRONMENT_DIR, WORLD_DIR, RAG_A_DIR, RAG_B_DIR, LOGS_DIR, RUNS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# LLM Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Perturbation settings
PERTURBATION_MIN_CYCLES = int(os.getenv("PERTURBATION_MIN_CYCLES", 10))
PERTURBATION_MAX_CYCLES = int(os.getenv("PERTURBATION_MAX_CYCLES", 30))

# Agent names
AGENT_A_NAME = "Agent A"
AGENT_B_NAME = "Agent B"

