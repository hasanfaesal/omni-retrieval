"""
Simple RAG pipeline for Phase 1.

Chains: retriever → response synthesis via RetrieverQueryEngine.
Full PipelineBuilder with registry-based module composition deferred to Phase 3.
"""

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode

from src.context import AppContext
from src.indexing.vector_indexer import load_index
from src.retrieval.dense_retriever import DenseRetriever


class SimplePipeline:
    """Minimal RAG pipeline: dense retrieval → LLM response generation.

    This is a thin wrapper around LlamaIndex's RetrieverQueryEngine
    configured with our Qdrant-backed dense retriever and Qwen3 LLM.
    """

    def __init__(
        self,
        ctx: AppContext,
        index: VectorStoreIndex | None = None,
        top_k: int = 10,
        response_mode: str = "compact",
    ):
        """Initialize the simple pipeline.

        Args:
            ctx: Application context (config + initialized services).
            index: Pre-built VectorStoreIndex. If None, loads from Qdrant.
            top_k: Number of documents to retrieve.
            response_mode: LlamaIndex response synthesis mode
                (compact, refine, tree_summarize, etc.).
        """
        if index is None:
            index = load_index(ctx)

        self._ctx = ctx
        self._retriever = DenseRetriever(index=index, top_k=top_k)

        # Build the query engine
        response_synthesizer = get_response_synthesizer(
            llm=ctx.llm,
            response_mode=ResponseMode(response_mode),
        )

        self._query_engine = RetrieverQueryEngine(
            retriever=self._retriever._retriever,
            response_synthesizer=response_synthesizer,
        )

    def run(self, query: str) -> str:
        """Run a query through the pipeline.

        Args:
            query: The user's question.

        Returns:
            Generated answer string.
        """
        response = self._query_engine.query(query)
        return str(response)

    def run_with_sources(self, query: str) -> dict:
        """Run a query and return the answer with source nodes.

        Args:
            query: The user's question.

        Returns:
            Dict with 'answer' (str) and 'sources' (list of source node dicts).
        """
        response = self._query_engine.query(query)
        sources = []
        for node in response.source_nodes:
            sources.append(
                {
                    "text": node.node.get_content()[:200] + "...",
                    "score": node.score,
                    "metadata": node.node.metadata,
                }
            )
        return {
            "answer": str(response),
            "sources": sources,
        }
