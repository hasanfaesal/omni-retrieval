"""
Vector indexer using Qdrant.

Phase 1: Dense-only
Phase 2: Supports hybrid search (dense + BM25/sparse) via enable_hybrid flag.
"""

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.vector_stores.qdrant import QdrantVectorStore

from src.context import AppContext
from src.indexing.chunker import chunk_documents


def _get_vector_store(
    ctx: AppContext, enable_hybrid: bool = False
) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=ctx.qdrant_client,
        collection_name=ctx.config.qdrant.collection_name,
        enable_hybrid=enable_hybrid,
        fastembed_sparse_model="Qdrant/bm25" if enable_hybrid else None,
    )


def index_documents(
    documents: list[Document],
    ctx: AppContext,
    enable_hybrid: bool = False,
) -> VectorStoreIndex:
    """Chunk documents, embed them, and store in Qdrant.

    Args:
        documents: List of LlamaIndex Document objects to index.
        ctx: Application context (config + initialized services).
        enable_hybrid: If True, enable hybrid search (dense + sparse BM25).

    Returns:
        VectorStoreIndex connected to the populated Qdrant collection.
    """
    # Chunk documents into nodes
    nodes = chunk_documents(documents, ctx.config.chunking)

    # Create Qdrant vector store and storage context
    vector_store = _get_vector_store(ctx, enable_hybrid=enable_hybrid)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build index from nodes (embeds and upserts into Qdrant)
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True,
    )
    return index


def load_index(ctx: AppContext, enable_hybrid: bool = False) -> VectorStoreIndex:
    """Load an existing Qdrant-backed index (reconnect to existing collection).

    Args:
        ctx: Application context (config + initialized services).
        enable_hybrid: If True, enable hybrid search mode for the collection.

    Returns:
        VectorStoreIndex connected to the existing Qdrant collection.
    """
    vector_store = _get_vector_store(ctx, enable_hybrid=enable_hybrid)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)
