"""
Query script: Run a single query through the RAG pipeline.

Phase 2: Added support for hybrid search and reranking.

Usage:
    python scripts/query.py --query "Were Scott Derrickson and Ed Wood of the same nationality?"
    python scripts/query.py --query "What government position was held by the woman ..." --top-k 5
    python scripts/query.py --query "..." --hybrid --use-reranker
"""

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from src.config import load_config
from src.context import AppContext
from src.pipeline import SimplePipeline

console = Console()


@click.command()
@click.option(
    "--query",
    "-q",
    required=True,
    help="The question to ask.",
)
@click.option(
    "--collection",
    type=str,
    default=None,
    help="Qdrant collection name. Defaults to config value.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to config YAML file. Defaults to configs/base.yaml.",
)
@click.option(
    "--pipeline-config",
    "pipeline_config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to pipeline config YAML (e.g., configs/pipelines/hybrid_rag.yaml).",
)
@click.option(
    "--top-k",
    type=int,
    default=10,
    help="Number of documents to retrieve.",
)
@click.option(
    "--show-sources",
    is_flag=True,
    default=False,
    help="Show retrieved source documents.",
)
@click.option(
    "--hybrid",
    is_flag=True,
    default=False,
    help="Use hybrid search (dense + sparse).",
)
@click.option(
    "--use-reranker",
    is_flag=True,
    default=False,
    help="Apply qwen3-rerank after retrieval.",
)
@click.option(
    "--alpha",
    type=float,
    default=0.5,
    help="Alpha weight for hybrid search (0.0=sparse, 1.0=dense).",
)
def main(
    query: str,
    collection: str | None,
    config_path: str | None,
    pipeline_config_path: str | None,
    top_k: int,
    show_sources: bool,
    hybrid: bool,
    use_reranker: bool,
    alpha: float,
):
    """Run a query through the RAG pipeline."""
    config = load_config(config_path)

    if hybrid:
        config.qdrant.collection_name = config.qdrant.collection_name_hybrid
    elif collection:
        config.qdrant.collection_name = collection

    ctx = AppContext(config=config)

    console.print(f"\n[bold blue]Query:[/] {query}\n")
    console.print(f"[dim]Mode:[/] {'hybrid' if hybrid else 'dense'}")
    if use_reranker:
        console.print("[dim]Reranker:[/] enabled")
    console.print()

    if pipeline_config_path:
        pipeline = SimplePipeline.from_config(
            ctx=ctx,
            pipeline_config_path=pipeline_config_path,
            base_config_path=config_path,
        )
    else:
        pipeline = SimplePipeline(
            ctx=ctx,
            top_k=top_k,
            retrieval_type="hybrid" if hybrid else "dense",
            alpha=alpha,
            use_reranker=use_reranker,
        )

    if show_sources:
        result = pipeline.run_with_sources(query)
        answer = result["answer"]

        console.print(Panel(Markdown(answer), title="Answer", border_style="green"))

        if result["sources"]:
            table = Table(title="Retrieved Sources", show_lines=True)
            table.add_column("Score", style="cyan", width=8)
            table.add_column("Title", style="yellow", width=20)
            table.add_column("Text Preview", style="white")

            for src in result["sources"]:
                title = src["metadata"].get("title", "N/A")
                score = f"{src['score']:.4f}" if src["score"] else "N/A"
                table.add_row(score, title, src["text"])

            console.print(table)
    else:
        answer = pipeline.run(query)
        console.print(Panel(Markdown(answer), title="Answer", border_style="green"))


if __name__ == "__main__":
    main()
