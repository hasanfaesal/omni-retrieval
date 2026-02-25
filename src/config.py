"""
Loads YAML config files into validated Pydantic models.
Handles .env loading for API keys.
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

# Load .env file from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# Default config path
DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "base.yaml"


class _ApiKeyMixin(BaseModel):
    """Mixin for configs that resolve an API key from an environment variable."""

    api_key_env: str

    @property
    def api_key(self) -> str:
        key = os.getenv(self.api_key_env, "")
        if not key:
            raise ValueError(f"API key not found. Set {self.api_key_env}.")
        return key


class LLMConfig(_ApiKeyMixin):
    model_name: str
    api_base: str
    temperature: float
    max_tokens: int


class EmbeddingConfig(_ApiKeyMixin):
    model_name: str
    api_base: str
    dimensions: int


class ChunkingConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int


class QdrantConfig(BaseModel):
    host: str
    port: int
    collection_name: str


class RerankingConfig(BaseModel):
    model_name: str
    api_base: str
    top_n: int


class AppConfig(BaseModel):
    llm: LLMConfig
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    qdrant: QdrantConfig
    reranking: RerankingConfig


def load_config(config_path: str | Path | None = None) -> AppConfig:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    return AppConfig(**raw)
