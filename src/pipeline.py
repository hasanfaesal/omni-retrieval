"""
RAG pipeline with support for multiple retrieval modes and post-processors.

Phase 1: Simple dense retrieval pipeline.
Phase 2: Added hybrid search and reranking support via config-driven composition.

Full PipelineBuilder with registry-based module composition deferred to Phase 3.
"""

from pathlib import Path

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode

from src.context import AppContext
from src.indexing.vector_indexer import load_index
from src.post_retrieval.reranker import DashScopeReranker
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_search import HybridRetriever


class SimplePipeline:
    """RAG pipeline with configurable retrieval and post-processing.

    Supports:
    - Dense retrieval (default)
    - Hybrid retrieval (dense + sparse/BM25)
    - Optional reranking post-processor

    Configuration can be passed directly or loaded from a YAML file.
    """

    def __init__(
        self,
        ctx: AppContext,
        index: VectorStoreIndex | None = None,
        top_k: int = 10,
        response_mode: str = "compact",
        retrieval_type: str = "dense",
        alpha: float = 0.5,
        sparse_top_k: int = 10,
        use_reranker: bool = False,
        rerank_top_n: int = 5,
    ):
        """Initialize the pipeline.

        Args:
            ctx: Application context (config + initialized services).
            index: Pre-built VectorStoreIndex. If None, loads from Qdrant.
            top_k: Number of documents to retrieve.
            response_mode: LlamaIndex response synthesis mode.
            retrieval_type: "dense" or "hybrid".
            alpha: Weight for hybrid search (0.0=sparse, 1.0=dense).
            sparse_top_k: Number of sparse results for hybrid search.
            use_reranker: Whether to apply reranking.
            rerank_top_n: Number of results to keep after reranking.
        """
        if index is None:
            enable_hybrid = retrieval_type == "hybrid"
            index = load_index(ctx, enable_hybrid=enable_hybrid)

        self._ctx = ctx
        self._retrieval_type = retrieval_type

        if retrieval_type == "hybrid":
            self._retriever = HybridRetriever(
                index=index,
                top_k=top_k,
                alpha=alpha,
                sparse_top_k=sparse_top_k,
            )
        else:
            self._retriever = DenseRetriever(index=index, top_k=top_k)

        # Build post-processors
        self._node_postprocessors = []
        if use_reranker:
            reranker = DashScopeReranker.from_config(
                model=ctx.config.reranking.model_name,
                top_n=rerank_top_n,
            )
            self._node_postprocessors.append(reranker)

        # Build the query engine
        response_synthesizer = get_response_synthesizer(
            llm=ctx.llm,
            response_mode=ResponseMode(response_mode),
        )

        self._query_engine = RetrieverQueryEngine(
            retriever=self._retriever._retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=self._node_postprocessors,
        )

    @classmethod
    def from_config(
        cls,
        ctx: AppContext,
        pipeline_config_path: str | Path | None = None,
        base_config_path: str | Path | None = None,
    ) -> "SimplePipeline":
        """Create pipeline from YAML config file.

        Args:
            ctx: Application context.
            pipeline_config_path: Path to pipeline YAML (e.g., configs/pipelines/hybrid_rag.yaml).
            base_config_path: Path to base config YAML.

        Returns:
            Configured SimplePipeline instance.
        """
        config_data = {}

        if base_config_path:
            import yaml

            with open(base_config_path) as f:
                base_data = yaml.safe_load(f) or {}
                config_data.update(base_data)

        if pipeline_config_path:
            with open(pipeline_config_path) as f:
                pipeline_data = yaml.safe_load(f) or {}
                config_data["pipeline"] = pipeline_data

        pipeline_cfg = config_data.get("pipeline", {})

        retrieval_cfg = (
            pipeline_cfg.get("retrieval", [{}])[0]
            if pipeline_cfg.get("retrieval")
            else {}
        )
        post_retrieval_cfg = (
            pipeline_cfg.get("post_retrieval", [{}])
            if pipeline_cfg.get("post_retrieval")
            else []
        )
        generation_cfg = pipeline_cfg.get("generation", {})

        retrieval_type = "dense"
        alpha = 0.5
        sparse_top_k = 10
        top_k = 10

        if retrieval_cfg:
            module = retrieval_cfg.get("module", "dense")
            params = retrieval_cfg.get("params", {})

            if module == "hybrid_search":
                retrieval_type = "hybrid"
                alpha = params.get("alpha", 0.5)
                sparse_top_k = params.get("sparse_top_k", 10)
                top_k = params.get("top_k", 10)

        use_reranker = False
        rerank_top_n = 5

        for post_proc in post_retrieval_cfg:
            if post_proc.get("module") == "reranker":
                use_reranker = True
                rerank_top_n = post_proc.get("params", {}).get("top_n", 5)
                break

        response_mode = generation_cfg.get("response_mode", "compact")

        return cls(
            ctx=ctx,
            top_k=top_k,
            response_mode=response_mode,
            retrieval_type=retrieval_type,
            alpha=alpha,
            sparse_top_k=sparse_top_k,
            use_reranker=use_reranker,
            rerank_top_n=rerank_top_n,
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
