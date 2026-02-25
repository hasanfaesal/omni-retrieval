# Omni-Retrieval

Modular RAG pipeline with swappable stages for multi-hop question answering.

## Quick Start

### Prerequisites

- Python 3.10+
- uv (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Qdrant running locally (`docker run -p 6333:6333 qdrant/qdrant`)
- DashScope API key (for Qwen3 LLM + embeddings) or any other LLM/Embedding provider of your choice
- Optional: Hugging Face token (`HF_TOKEN`) for authenticated dataset access/rate limits

### Setup

```bash
# Install and lock project dependencies
uv sync --dev

# Configure API keys
cp .env.example .env
# Edit .env with your actual API keys
```

Dependencies are managed via `pyproject.toml` and `uv.lock` using uv project APIs.

## Architecture

The pipeline is structured as four swappable stage groups:

1. **Pre-retrieval** — Query transformation (HyDE, multi-query, routing)
2. **Indexing** — Document chunking and vector/graph/RAPTOR indexing
3. **Retrieval** — Dense, hybrid, graph, and fusion retrieval
4. **Post-retrieval** — Reranking, compression, correction loops (CRAG, Self-RAG)
