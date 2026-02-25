"""
Dense vector retriever using Qdrant.

Phase 1: Pure dense retrieval (cosine similarity).
Hybrid search (dense + BM25 sparse) will be added in Phase 2.
"""

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore


class DenseRetriever:
    """Dense vector similarity retriever backed by Qdrant.

    Wraps LlamaIndex's VectorIndexRetriever for pure dense retrieval.
    In Phase 2, this will be upgraded to hybrid (dense + sparse) mode.
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        top_k: int = 10,
    ):
        """Initialize the dense retriever.

        Args:
            index: Pre-built VectorStoreIndex.
            top_k: Number of top results to retrieve.
        """
        self._index = index
        self._top_k = top_k
        self._retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=self._top_k,
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[NodeWithScore]:
        """Retrieve the most similar nodes for a query.

        Args:
            query: The query string.
            top_k: Override the default top_k for this query.

        Returns:
            List of NodeWithScore objects ranked by similarity.
        """
        if top_k is not None and top_k != self._top_k:
            self._retriever = VectorIndexRetriever(
                index=self._index,
                similarity_top_k=top_k,
            )
            self._top_k = top_k

        return self._retriever.retrieve(query)
