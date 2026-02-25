"""
Ingest script: Load datasets, chunk, and index into Qdrant.

Phase 1: HotpotQA distractor split, dense-only indexing.

Usage:
    python scripts/ingest.py --dataset hotpotqa --limit 200
    python scripts/ingest.py --dataset hotpotqa --limit 200 --config configs/base.yaml
"""

import os

import click
from llama_index.core.schema import Document
from rich.console import Console
from rich.progress import Progress

from src.config import load_config
from src.context import AppContext
from src.indexing.vector_indexer import index_documents

console = Console()


def load_hotpotqa(limit: int = 200) -> list[Document]:
    """Load HotpotQA distractor split and convert to LlamaIndex Documents.

    Each context passage (title + sentences) becomes a Document with metadata.
    Uses the 'distractor' split which includes 10 context paragraphs per question
    (much more manageable than 'fullwiki' which has ~5M paragraphs).

    Args:
        limit: Maximum number of samples to load.

    Returns:
        List of LlamaIndex Document objects.
    """
    from datasets import load_dataset

    console.print(f"[bold blue]Loading HotpotQA distractor split (limit={limit})...[/]")

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        console.print("[dim]Using Hugging Face token from HF_TOKEN.[/]")

    try:
        ds = load_dataset("hotpot_qa", "distractor", split="train", token=hf_token)
    except TypeError:
        ds = load_dataset(
            "hotpot_qa",
            "distractor",
            split="train",
            use_auth_token=hf_token,
        )
    ds = ds.select(range(min(limit, len(ds))))

    documents = []
    seen_titles = set()  # Avoid duplicate context passages

    with Progress() as progress:
        task = progress.add_task("Processing samples...", total=len(ds))

        for i, sample in enumerate(ds):
            question_id = sample.get("id", str(i))
            context_titles = sample["context"]["title"]
            context_sentences = sample["context"]["sentences"]

            for title, sentences in zip(context_titles, context_sentences):
                # Deduplicate by title (same passage can appear across questions)
                if title in seen_titles:
                    continue
                seen_titles.add(title)

                # Join sentences into a single passage
                passage_text = " ".join(sentences)
                if not passage_text.strip():
                    continue

                doc = Document(
                    text=f"{title}\n\n{passage_text}",
                    metadata={
                        "title": title,
                        "dataset": "hotpotqa",
                        "question_id": question_id,
                        "source_type": "context_passage",
                    },
                    excluded_llm_metadata_keys=["question_id", "dataset"],
                    excluded_embed_metadata_keys=["question_id", "dataset"],
                )
                documents.append(doc)

            progress.update(task, advance=1)

    console.print(
        f"[bold green]Loaded {len(documents)} unique context passages "
        f"from {len(ds)} samples.[/]"
    )
    return documents


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(["hotpotqa"]),
    default="hotpotqa",
    help="Dataset to ingest.",
)
@click.option(
    "--limit",
    type=int,
    default=200,
    help="Maximum number of samples to load.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to config YAML file. Defaults to configs/base.yaml.",
)
def main(dataset: str, limit: int, config_path: str | None):
    """Ingest a dataset into the Qdrant vector store."""
    config = load_config(config_path)
    ctx = AppContext(config=config)

    console.print(f"[bold]Dataset:[/] {dataset}")
    console.print(f"[bold]Limit:[/] {limit}")
    console.print(f"[bold]Qdrant collection:[/] {config.qdrant.collection_name}")
    console.print(f"[bold]Chunk size:[/] {config.chunking.chunk_size}")
    console.print()

    # Load documents
    if dataset == "hotpotqa":
        documents = load_hotpotqa(limit=limit)
    else:
        raise click.BadParameter(f"Unknown dataset: {dataset}")

    if not documents:
        console.print("[bold red]No documents loaded. Exiting.[/]")
        return

    # Index into Qdrant
    console.print(f"\n[bold blue]Indexing {len(documents)} documents into Qdrant...[/]")
    index_documents(documents, ctx)
    console.print("[bold green]Indexing complete![/]")
    console.print(
        f"[dim]Verify at: http://{config.qdrant.host}:{config.qdrant.port}/dashboard[/]"
    )


if __name__ == "__main__":
    main()
