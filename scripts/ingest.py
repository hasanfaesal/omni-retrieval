"""
Ingest script: Load datasets or local files, chunk, and index into Qdrant.

Phase 1: HotpotQA distractor split, dense-only indexing.
Supports local PDF files for custom data ingestion.

Usage:
    # Ingest from HuggingFace dataset
    python scripts/ingest.py --dataset hotpotqa --limit 200

    # Ingest from local PDF file
    python scripts/ingest.py --file data/tiny_aya_tech_report.pdf
    python scripts/ingest.py --file data/tiny_aya_tech_report.pdf --collection aya_report

    # With custom config
    python scripts/ingest.py --file data/tiny_aya_tech_report.pdf --config configs/base.yaml

    # Checking Qdrant Documents
    # List all collections
    curl -s http://localhost:6333/collections
    # View collection details
    curl -s http://localhost:6333/collections/aya_report
    # View sample documents
    curl -s "http://localhost:6333/collections/aya_report/points/scroll" \
    -H "Content-Type: application/json" \
    -d '{"limit": 5, "with_payload": true}'
    # Or open dashboard
    http://localhost:6333/dashboard
"""

import os
from pathlib import Path

import click
from llama_index.core.schema import Document
from llama_index.readers.file import PDFReader
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


def load_pdf_file(file_path: str) -> list[Document]:
    """Load a local PDF file using LlamaIndex's PDFReader.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of LlamaIndex Document objects.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    console.print(f"[bold blue]Loading PDF file: {file_path}[/]")

    reader = PDFReader()
    documents = reader.load_data(file_path)

    for doc in documents:
        doc.metadata = doc.metadata or {}
        doc.metadata["source_file"] = path.name
        doc.metadata["source_type"] = "pdf"
        doc.excluded_llm_metadata_keys = ["source_file"]
        doc.excluded_embed_metadata_keys = ["source_file"]

    console.print(f"[bold green]Loaded {len(documents)} pages from PDF.[/]")
    return documents


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(["hotpotqa"]),
    default=None,
    help="HuggingFace dataset to ingest.",
)
@click.option(
    "--file",
    "file_path",
    type=click.Path(exists=True),
    default=None,
    help="Local file to ingest (PDF, txt, etc.).",
)
@click.option(
    "--collection",
    type=str,
    default=None,
    help="Qdrant collection name. Defaults to config value.",
)
@click.option(
    "--limit",
    type=int,
    default=200,
    help="Maximum number of samples (for datasets).",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to config YAML file. Defaults to configs/base.yaml.",
)
@click.option(
    "--hybrid",
    is_flag=True,
    default=False,
    help="Use hybrid search mode (dense + sparse).",
)
@click.option(
    "--recreate",
    is_flag=True,
    default=False,
    help="Delete existing collection before indexing.",
)
def main(
    dataset: str | None,
    file_path: str | None,
    collection: str | None,
    limit: int,
    config_path: str | None,
    hybrid: bool,
    recreate: bool,
):
    """Ingest a dataset or local file into the Qdrant vector store."""
    if not dataset and not file_path:
        raise click.UsageError("Either --dataset or --file must be specified.")

    config = load_config(config_path)

    if hybrid:
        config.qdrant.collection_name = config.qdrant.collection_name_hybrid
    elif collection:
        config.qdrant.collection_name = collection

    ctx = AppContext(config=config)

    console.print(f"[bold]Source:[/] {dataset or file_path}")
    console.print(f"[bold]Qdrant collection:[/] {config.qdrant.collection_name}")
    console.print(f"[bold]Mode:[/] {'hybrid' if hybrid else 'dense'}")
    console.print(f"[bold]Chunk size:[/] {config.chunking.chunk_size}")
    console.print()

    # Handle recreate flag
    if recreate:
        console.print(
            f"[bold yellow]Deleting existing collection: {config.qdrant.collection_name}[/]"
        )
        try:
            ctx.qdrant_client.delete_collection(config.qdrant.collection_name)
            console.print("[bold green]Collection deleted.[/]")
        except Exception as e:
            console.print(f"[dim]Collection not found or error: {e}[/]")

    if file_path:
        documents = load_pdf_file(file_path)
    elif dataset == "hotpotqa":
        documents = load_hotpotqa(limit=limit)
    else:
        raise click.BadParameter(f"Unknown dataset: {dataset}")

    if not documents:
        console.print("[bold red]No documents loaded. Exiting.[/]")
        return

    # Index into Qdrant
    console.print(f"\n[bold blue]Indexing {len(documents)} documents into Qdrant...[/]")
    index_documents(documents, ctx, enable_hybrid=hybrid)
    console.print("[bold green]Indexing complete![/]")
    console.print(
        f"[dim]Verify at: http://{config.qdrant.host}:{config.qdrant.port}/dashboard[/]"
    )


if __name__ == "__main__":
    main()
