"""
Vector indexer using Qdrant (dense-only for Phase 1).

Handles creating, populating, and loading a Qdrant-backed VectorStoreIndex.
Hybrid search (dense + BM25/sparse) will be enabled in Phase 2.
"""

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.vector_stores.qdrant import QdrantVectorStore

from src.context import AppContext
from src.indexing.chunker import chunk_documents


def _get_vector_store(ctx: AppContext) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=ctx.qdrant_client,
        collection_name=ctx.config.qdrant.collection_name,
    )


def index_documents(
    documents: list[Document],
    ctx: AppContext,
) -> VectorStoreIndex:
    """Chunk documents, embed them, and store in Qdrant.

    Args:
        documents: List of LlamaIndex Document objects to index.
        ctx: Application context (config + initialized services).

    Returns:
        VectorStoreIndex connected to the populated Qdrant collection.
    """
    # Chunk documents into nodes
    nodes = chunk_documents(documents, ctx.config.chunking)

    # Create Qdrant vector store and storage context
    vector_store = _get_vector_store(ctx)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build index from nodes (embeds and upserts into Qdrant)
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True,
    )
    return index


def load_index(ctx: AppContext) -> VectorStoreIndex:
    """Load an existing Qdrant-backed index (reconnect to existing collection).

    Args:
        ctx: Application context (config + initialized services).

    Returns:
        VectorStoreIndex connected to the existing Qdrant collection.
    """
    vector_store = _get_vector_store(ctx)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)
