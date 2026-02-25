"""
Query script: Run a single query through the RAG pipeline.

Usage:
    python scripts/query.py --query "Were Scott Derrickson and Ed Wood of the same nationality?"
    python scripts/query.py --query "What government position was held by the woman ..." --top-k 5
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
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to config YAML file. Defaults to configs/base.yaml.",
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
def main(query: str, config_path: str | None, top_k: int, show_sources: bool):
    """Run a query through the RAG pipeline."""
    config = load_config(config_path)
    ctx = AppContext(config=config)

    console.print(f"\n[bold blue]Query:[/] {query}\n")

    pipeline = SimplePipeline(ctx=ctx, top_k=top_k)

    if show_sources:
        result = pipeline.run_with_sources(query)
        answer = result["answer"]

        # Show answer
        console.print(Panel(Markdown(answer), title="Answer", border_style="green"))

        # Show sources
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
