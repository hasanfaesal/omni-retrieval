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
DASHSCOPE_API_KEY_ENV = "DASHSCOPE_API_KEY"


def get_dashscope_api_key() -> str:
    """Resolve the shared DashScope API key from environment."""
    key = os.getenv(DASHSCOPE_API_KEY_ENV, "")
    if not key:
        raise ValueError(f"API key not found. Set {DASHSCOPE_API_KEY_ENV}.")
    return key


class LLMConfig(BaseModel):
    model_name: str
    api_base: str
    temperature: float
    max_tokens: int


class EmbeddingConfig(BaseModel):
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
    collection_name_hybrid: str = "hotpotqa_hybrid"


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
