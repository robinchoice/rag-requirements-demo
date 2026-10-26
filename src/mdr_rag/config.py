import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str = "postgresql://mdr:mdr@localhost:5432/mdr"
    # Any OpenAI-compatible endpoint: Ollama locally, vLLM in a cluster.
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "unused"
    embed_model: str = "bge-m3"
    embed_dim: int = 1024
    chat_model: str = "qwen2.5:7b-instruct"


def load_settings() -> Settings:
    defaults = Settings()
    return Settings(
        database_url=os.environ.get("DATABASE_URL", defaults.database_url),
        llm_base_url=os.environ.get("LLM_BASE_URL", defaults.llm_base_url),
        llm_api_key=os.environ.get("LLM_API_KEY", defaults.llm_api_key),
        embed_model=os.environ.get("EMBED_MODEL", defaults.embed_model),
        embed_dim=int(os.environ.get("EMBED_DIM", defaults.embed_dim)),
        chat_model=os.environ.get("CHAT_MODEL", defaults.chat_model),
    )
