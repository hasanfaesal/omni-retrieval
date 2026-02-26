"""
Hybrid search retriever using Qdrant (dense + sparse/BM25).

Phase 2: Combines dense vector similarity with BM25 sparse retrieval
for improved recall. Uses alpha to weight between dense and sparse.
"""

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore


class HybridRetriever:
    """Hybrid retriever combining dense and sparse (BM25) search.

    Uses Qdrant's hybrid search mode which combines dense embeddings
    with BM25 sparse vectors for improved retrieval quality.

    Args:
        index: Pre-built VectorStoreIndex with hybrid mode enabled.
        top_k: Number of top results to retrieve.
        alpha: Weight for dense vs sparse. 0.0 = pure sparse, 1.0 = pure dense.
        sparse_top_k: Number of sparse results to consider.
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        top_k: int = 10,
        alpha: float = 0.5,
        sparse_top_k: int = 10,
    ):
        self._index = index
        self._top_k = top_k
        self._alpha = alpha
        self._sparse_top_k = sparse_top_k
        self._retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=self._top_k,
            vector_store_query_mode="hybrid",
            alpha=self._alpha,
            sparse_top_k=self._sparse_top_k,
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[NodeWithScore]:
        """Retrieve the most similar nodes for a query using hybrid search.

        Args:
            query: The query string.
            top_k: Override the default top_k for this query.

        Returns:
            List of NodeWithScore objects ranked by hybrid similarity.
        """
        if top_k is not None and top_k != self._top_k:
            self._retriever = VectorIndexRetriever(
                index=self._index,
                similarity_top_k=top_k,
                vector_store_query_mode="hybrid",
                alpha=self._alpha,
                sparse_top_k=self._sparse_top_k,
            )
            self._top_k = top_k

        return self._retriever.retrieve(query)
