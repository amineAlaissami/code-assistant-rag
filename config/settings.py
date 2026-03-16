"""
settings.py
-----------
Configuration centralisée du projet.
Toutes les valeurs viennent du fichier .env — jamais hardcodées dans le code.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from enum import Enum


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"
    CODE_AWARE = "code_aware"


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    llm_provider: LLMProvider = Field(default=LLMProvider.OPENAI, env="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4o-mini", env="LLM_MODEL")
    llm_temperature: float = Field(default=0.2, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=2048, env="LLM_MAX_TOKENS")

    # Embeddings
    embedding_model: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    local_embedding_model: str = Field(default="all-MiniLM-L6-v2", env="LOCAL_EMBEDDING_MODEL")

    # ChromaDB
    chroma_persist_dir: str = Field(default="./chroma_db", env="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field(default="python_docs", env="CHROMA_COLLECTION_NAME")

    # RAG
    chunk_strategy: ChunkStrategy = Field(default=ChunkStrategy.FIXED, env="CHUNK_STRATEGY")
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    top_k_results: int = Field(default=5, env="TOP_K_RESULTS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Instance globale — importée dans tous les autres modules
settings = Settings()