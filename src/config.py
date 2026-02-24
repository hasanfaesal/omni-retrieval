"""
Loads YAML config files into validated Pydantic models.
Handles .env loading for API keys.

values in YAML, schema in Pydantic 
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field

# Load .env file
load_dotenv(find_dotenv())

# Default config path
DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "base.yaml"


class LLMConfig(BaseModel):
    model_name: str
    api_base: str
    api_key_env: str
    temperature: float
    max_tokens: int

    @property
    def api_key(self) -> str:
        key = os.getenv(self.api_key_env, "")
        if not key:
            raise ValueError(f"API key not found. Set {self.api_key_env}.")
        return key


class EmbeddingConfig(BaseModel):
    model_name: str
    api_base: str
    api_key_env: str
    dimensions: int

    @property
    def api_key(self) -> str:
        key = os.getenv(self.api_key_env, "")
        if not key:
            raise ValueError(f"API key not found. Set {self.api_key_env}.")
        return key


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