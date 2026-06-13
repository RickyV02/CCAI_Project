import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 🧠 LLM & AGENT SETTINGS
# ==========================================
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.5"))
POST_LENGTH_GUIDANCE = os.getenv("POST_LENGTH_GUIDANCE", "1000 parole circa")
MAX_QUALITY_RETRIES = int(os.getenv("MAX_QUALITY_RETRIES", "1"))
MAX_RESEARCHER_ITERATIONS = int(os.getenv("MAX_RESEARCHER_ITERATIONS", "5"))

# ==========================================
# 📚 RAG & SCRAPING SETTINGS
# ==========================================
RAG_PERSIST_DIRECTORY = os.getenv("RAG_PERSIST_DIRECTORY", "./chroma_db")
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RAG_RETRIEVE_K = int(os.getenv("RAG_RETRIEVE_K", "5"))
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))

# ==========================================
# 🕸️ KNOWLEDGE GRAPH (NEO4J) SETTINGS
# ==========================================
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
