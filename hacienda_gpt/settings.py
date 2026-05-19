import os


OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0"))
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", ".faiss")
MEMORY_KEY = os.environ.get("MEMORY_KEY", "chat_history")
TOP_K = int(os.environ.get("TOP_K", "3"))
