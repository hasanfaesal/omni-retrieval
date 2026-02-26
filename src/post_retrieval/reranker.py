"""
Reranker post-processor using DashScope qwen3-rerank API.

Phase 2: Re-ranks retrieved nodes using Qwen3's reranking model
via DashScope's TextReRank API.
"""

import os

from dashscope import TextReRank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle


class DashScopeReranker(BaseNodePostprocessor):
    """Reranker using DashScope's qwen3-rerank model.

    Calls the DashScope TextReRank API to re-order retrieved nodes
    based on their relevance to the query.

    Args:
        model: Reranker model name (default: qwen3-rerank).
        top_n: Number of top results to return after reranking.
    """

    def __init__(
        self,
        model: str = "qwen3-rerank",
        top_n: int = 5,
    ):
        self._model = model
        self._top_n = top_n
        if not os.getenv("DASHSCOPE_API_KEY"):
            raise ValueError("DashScope API key not found. Set DASHSCOPE_API_KEY.")

    @classmethod
    def from_config(cls, model: str, top_n: int) -> "DashScopeReranker":
        """Create reranker from config values.

        Args:
            model: Model name from config.
            top_n: Top N from config.

        Returns:
            Configured DashScopeReranker instance.
        """
        return cls(model=model, top_n=top_n)

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        """Re-rank nodes based on relevance to the query.

        Args:
            nodes: List of nodes with scores to re-rank.
            query_bundle: The query with context.

        Returns:
            Re-ranked list of nodes (limited to top_n).
        """
        if not nodes or not query_bundle:
            return nodes

        query = query_bundle.query_str
        documents = [node.node.get_content() for node in nodes]

        try:
            response = TextReRank.call(
                model=self._model,
                query=query,
                documents=documents,
                top_n=self._top_n,
                return_doc=True,
            )

            if response.status_code != 200:
                print(f"Rerank API error: {response.code} - {response.message}")
                return nodes

            results = response.output.results
            reranked_nodes = []
            for item in results:
                idx = item.index
                relevance_score = item.relevance_score
                reranked_nodes.append(
                    NodeWithScore(
                        node=nodes[idx].node,
                        score=relevance_score,
                    )
                )

            return reranked_nodes

        except Exception as e:
            print(f"Reranking failed: {e}")
            return nodes
