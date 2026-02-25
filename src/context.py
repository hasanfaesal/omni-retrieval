"""
Application context: single object holding all initialized services.

Created once at each entry-point's startup, then passed through the call
graph — replacing scattered ``if config is None: config = load_config()``
checks and redundant ``configure_settings()`` calls.
"""

from dataclasses import dataclass, field

from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike
from qdrant_client import QdrantClient

from src.config import AppConfig
from src.llm import get_embed_model, get_llm


@dataclass
class AppContext:
    """Holds config and all initialized services. Created once at startup."""

    config: AppConfig

    llm: OpenAILike = field(init=False)
    embed_model: OpenAIEmbedding = field(init=False)
    qdrant_client: QdrantClient = field(init=False)

    def __post_init__(self):
        self.llm = get_llm(self.config.llm)
        self.embed_model = get_embed_model(self.config.embedding)
        self.qdrant_client = QdrantClient(
            host=self.config.qdrant.host,
            port=self.config.qdrant.port,
        )
        # Configure LlamaIndex global Settings once, reusing our instances
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = self.config.chunking.chunk_size
        Settings.chunk_overlap = self.config.chunking.chunk_overlap
