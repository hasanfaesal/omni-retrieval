"""
Simple document chunker using LlamaIndex's SentenceSplitter.

Phase 1: Flat sentence-based chunking only.
Hierarchical chunking (parent-child nodes) will be implemented later.
"""

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, Document

from src.config import ChunkingConfig


def get_chunker(cfg: ChunkingConfig) -> SentenceSplitter:
    """Create a SentenceSplitter with configured chunk size and overlap.

    Args:
        cfg: Chunking sub-config.

    Returns:
        Configured SentenceSplitter instance.
    """
    return SentenceSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )


def chunk_documents(documents: list[Document], cfg: ChunkingConfig) -> list[BaseNode]:
    """Chunk a list of documents into text nodes.

    Args:
        documents: List of LlamaIndex Document objects.
        cfg: Chunking sub-config.

    Returns:
        List of TextNode objects with metadata preserved from source documents.
    """
    chunker = get_chunker(cfg)
    nodes = chunker.get_nodes_from_documents(documents, show_progress=True)
    return nodes
