"""
Test script: Validate LLM and embedding model connectivity.

Run this FIRST before any other scripts to catch DashScope API issues early.

Usage:
    python scripts/test_llm.py
    python scripts/test_llm.py --config configs/base.yaml
"""

import sys
import time

import click
from rich.console import Console
from rich.panel import Panel

from src.config import load_config

console = Console()


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to config YAML file. Defaults to configs/base.yaml.",
)
def main(config_path: str | None):
    """Test LLM and embedding model connectivity."""
    config = load_config(config_path)

    console.print("[bold]Testing Omni-Retrieval LLM & Embedding setup[/]\n")
    console.print(f"  LLM model:       {config.llm.model_name}")
    console.print(f"  LLM API base:    {config.llm.api_base}")
    console.print(f"  Embed model:     {config.embedding.model_name}")
    console.print(f"  Embed API base:  {config.embedding.api_base}")
    console.print(f"  Embed dims:      {config.embedding.dimensions}")
    console.print()

    # Test 1: LLM completion
    console.print("[bold blue]Test 1: LLM completion...[/]")
    try:
        from src.llm import get_llm

        llm = get_llm(config.llm)
        start = time.time()
        response = llm.complete("Say 'Hello, world!' and nothing else.")
        elapsed = time.time() - start
        console.print(
            Panel(
                str(response),
                title=f"LLM Response ({elapsed:.2f}s)",
                border_style="green",
            )
        )
        console.print("[bold green]✓ LLM test passed[/]\n")
    except Exception as e:
        console.print(f"[bold red]✗ LLM test failed: {e}[/]\n")
        sys.exit(1)

    # Test 2: Embedding generation
    console.print("[bold blue]Test 2: Embedding generation...[/]")
    try:
        from src.llm import get_embed_model

        embed_model = get_embed_model(config.embedding)
        start = time.time()
        embedding = embed_model.get_text_embedding("This is a test sentence.")
        elapsed = time.time() - start
        console.print(
            f"  Embedding dimensions: {len(embedding)} "
            f"(expected: {config.embedding.dimensions})"
        )
        console.print(f"  First 5 values: {embedding[:5]}")
        console.print(f"  Time: {elapsed:.2f}s")

        if len(embedding) == config.embedding.dimensions:
            console.print("[bold green]✓ Embedding test passed[/]\n")
        else:
            console.print(
                f"[bold yellow]⚠ Dimension mismatch: got {len(embedding)}, "
                f"expected {config.embedding.dimensions}[/]\n"
            )
    except Exception as e:
        console.print(f"[bold red]✗ Embedding test failed: {e}[/]\n")
        sys.exit(1)

    # Test 3: Batch embedding
    console.print("[bold blue]Test 3: Batch embedding (2 texts)...[/]")
    try:
        start = time.time()
        embeddings = embed_model.get_text_embedding_batch(
            ["First test sentence.", "Second test sentence."]
        )
        elapsed = time.time() - start
        console.print(f"  Batch size: {len(embeddings)}")
        console.print(f"  Each dimension: {len(embeddings[0])}")
        console.print(f"  Time: {elapsed:.2f}s")
        console.print("[bold green]✓ Batch embedding test passed[/]\n")
    except Exception as e:
        console.print(f"[bold red]✗ Batch embedding test failed: {e}[/]\n")
        sys.exit(1)

    console.print("[bold green]All tests passed! ✓[/]")
    console.print(
        "[dim]You can now run: python scripts/ingest.py --dataset hotpotqa --limit 200[/]"
    )


if __name__ == "__main__":
    main()
