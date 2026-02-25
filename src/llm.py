"""
Creates LlamaIndex LLM and embedding instances configured for
DashScope (Qwen3) via the OpenAI-compatible API.
"""

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike

from src.config import EmbeddingConfig, LLMConfig


def get_llm(cfg: LLMConfig) -> OpenAILike:
    """Create a Qwen3 LLM instance via DashScope's OpenAI-compatible API.

    Args:
        cfg: LLM sub-config.

    Returns:
        Configured OpenAILike LLM instance.
    """
    return OpenAILike(
        model=cfg.model_name,
        api_base=cfg.api_base,
        api_key=cfg.api_key,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        is_chat_model=True,
    )


def get_embed_model(cfg: EmbeddingConfig) -> OpenAIEmbedding:
    """Create an embedding model instance via DashScope's OpenAI-compatible API.

    Args:
        cfg: Embedding sub-config.

    Returns:
        Configured OpenAIEmbedding instance.
    """
    return OpenAIEmbedding(
        model_name=cfg.model_name,
        api_base=cfg.api_base,
        api_key=cfg.api_key,
        dimensions=cfg.dimensions,
        embed_batch_size=10,
    )
