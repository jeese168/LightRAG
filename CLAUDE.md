# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LightRAG is a Retrieval-Augmented Generation (RAG) framework that uses graph-based knowledge representation for enhanced information retrieval. The system extracts entities and relationships from documents, builds a knowledge graph, and uses multi-modal retrieval (local, global, hybrid, mix, naive) for queries.

### Package Structure
- `lightrag/`: Core Python package
  - `lightrag.py`: Main LightRAG class
  - `operate.py`: Core extraction and query operations
  - `base.py`: Abstract base classes for storage backends
  - `kg/`: Storage implementations (JSON, NetworkX, Neo4j, PostgreSQL, MongoDB, Redis, Milvus, Qdrant, Faiss, Memgraph)
  - `llm/`: LLM provider bindings (OpenAI, Ollama, Azure, Gemini, Bedrock, Anthropic, etc.)
  - `api/`: FastAPI server with REST endpoints and WebUI
  - `utils*.py`: Helper utilities for various operations
- `lightrag_webui/`: React 19 + TypeScript web interface (Bun + Vite)
- `tests/`: Test suite (mirrors feature folders)
- Root-level `test_*.py`: Specific integration tests

## Core Architecture

### Key Components

- **lightrag.py**: Main orchestrator class (`LightRAG`) that coordinates document insertion, query processing, and storage management. Critical: Always call `await rag.initialize_storages()` after instantiation.

- **operate.py**: Core extraction and query operations including entity/relation extraction, chunking, and multi-mode retrieval logic.

- **base.py**: Abstract base classes for storage backends (`BaseKVStorage`, `BaseVectorStorage`, `BaseGraphStorage`, `BaseDocStatusStorage`).

- **kg/**: Storage implementations (JSON, NetworkX, Neo4j, PostgreSQL, MongoDB, Redis, Milvus, Qdrant, Faiss, Memgraph). Each storage type provides different trade-offs for production vs. development use.

- **llm/**: LLM provider bindings (OpenAI, Ollama, Azure, Gemini, Bedrock, Anthropic, etc.). All use async patterns with caching support.

- **api/**: FastAPI server (`lightrag_server.py`) with REST endpoints and Ollama-compatible API, plus React 19 + TypeScript WebUI.

### Storage Layer

LightRAG uses 4 storage types with pluggable backends:
- **KV_STORAGE**: LLM response cache, text chunks, document info
- **VECTOR_STORAGE**: Entity/relation/chunk embeddings
- **GRAPH_STORAGE**: Entity-relation graph structure
- **DOC_STATUS_STORAGE**: Document processing status tracking

Workspace isolation is implemented differently per storage type (subdirectories for file-based, prefixes for collections, fields for relational DBs).

### Query Modes

- **local**: Context-dependent retrieval focused on specific entities
- **global**: Community/summary-based broad knowledge retrieval
- **hybrid**: Combines local and global
- **naive**: Direct vector search without graph
- **mix**: Integrates KG and vector retrieval (recommended with reranker)

## Development Commands

### Setup
```bash
# Install core package (development mode)
uv sync
source .venv/bin/activate  # Or: .venv\Scripts\activate on Windows

# Alternative: Using pip/venv
python -m venv .venv && source .venv/bin/activate
pip install -e .              # Core package
pip install -e .[api]         # With API support

# Install with specific extras (uv)
uv sync --extra api               # API server and WebUI
uv sync --extra offline-storage   # Storage backends (Redis, Neo4j, PostgreSQL, MongoDB, Milvus, Qdrant)
uv sync --extra offline-llm       # LLM providers (OpenAI, Anthropic, Ollama, Gemini, Bedrock, etc.)
uv sync --extra offline           # Complete offline package (api + storage + llm)
uv sync --extra test              # Testing dependencies
uv sync --extra evaluation        # RAGAS evaluation framework
uv sync --extra observability     # Langfuse tracing
uv sync --extra docling           # Advanced document processing (not compatible with macOS gunicorn multi-worker)
```

### API Server
```bash
# Copy and configure environment
cp env.example .env  # Edit with your LLM/embedding configs

# Build WebUI
cd lightrag_webui
bun install --frozen-lockfile
bun run build
cd ..

# Run server
lightrag-server                                           # Production
uvicorn lightrag.api.lightrag_server:app --reload        # Development
lightrag-gunicorn                                         # Multi-worker (gunicorn)
```

### Testing
```bash
# Run offline tests (default)
python -m pytest tests

# Run integration tests (requires external services)
python -m pytest tests --run-integration
# Or set: LIGHTRAG_RUN_INTEGRATION=true

# Run specific test file
python test_graph_storage.py

# Keep artifacts for debugging
python -m pytest tests --keep-artifacts

# Run with custom workers
python -m pytest tests --test-workers 4
```

### Linting
```bash
ruff check .
ruff check . --fix  # Auto-fix issues
```

### Additional CLI Tools
```bash
# Download tiktoken cache for offline deployment
lightrag-download-cache

# Clean LLM query cache
lightrag-clean-llmqc
```

## Key Implementation Patterns

### LightRAG Initialization (Critical)

The most common error is forgetting to initialize storages:

```python
import asyncio
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

async def main():
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed
    )

    # REQUIRED: Initialize storage backends
    await rag.initialize_storages()

    # Now safe to use
    await rag.ainsert("Your text here")
    result = await rag.aquery("Your question", param=QueryParam(mode="hybrid"))

    # Cleanup
    await rag.finalize_storages()

asyncio.run(main())
```

### Custom Embedding Functions

Use `@wrap_embedding_func_with_attrs` decorator and call `.func` when wrapping:

```python
from lightrag.utils import wrap_embedding_func_with_attrs

@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
async def custom_embed(texts: list[str]) -> np.ndarray:
    # Call underlying function, not wrapped version
    return await openai_embed.func(texts, model="text-embedding-3-large")
```

### Storage Configuration

Configure via environment variables or constructor params:

```python
# Environment-based (recommended for production)
# See env.example for full list

# Constructor-based
rag = LightRAG(
    working_dir="./storage",
    workspace="project_name",  # For data isolation
    kv_storage="PGKVStorage",
    vector_storage="PGVectorStorage",
    graph_storage="Neo4JStorage",
    doc_status_storage="PGDocStatusStorage",
    vector_db_storage_cls_kwargs={
        "cosine_better_than_threshold": 0.2
    }
)
```

### Document Insertion

```python
# Single document
await rag.ainsert("Text content")

# Batch insertion
await rag.ainsert(["Text 1", "Text 2", ...])

# With custom IDs
await rag.ainsert("Text", ids=["doc-123"])

# With file paths (for citation)
await rag.ainsert(["Text 1", "Text 2"], file_paths=["doc1.pdf", "doc2.pdf"])

# Configure batch size
rag = LightRAG(..., max_parallel_insert=4)  # Default: 2, max recommended: 10
```

### Query Configuration

```python
from lightrag import QueryParam

result = await rag.aquery(
    "Your question",
    param=QueryParam(
        mode="mix",                    # Recommended with reranker
        top_k=60,                      # KG entities/relations to retrieve
        chunk_top_k=20,                # Text chunks to retrieve
        max_entity_tokens=6000,
        max_relation_tokens=8000,
        max_total_tokens=30000,
        enable_rerank=True,
        user_prompt="Additional instructions for LLM",
        stream=False
    )
)
```

## WebUI Development

### Structure
- `lightrag_webui/src/`: React components (TypeScript)
- Uses Vite + Bun build system
- Tailwind CSS for styling
- React 19 with functional components and hooks

### Commands
```bash
cd lightrag_webui
bun install --frozen-lockfile  # Install dependencies
bun run dev                    # Development server
bun run build                  # Production build
bun test                       # Run tests
```

## Common Issues

### 1. Storage Not Initialized
**Error**: `AttributeError: __aenter__` or `KeyError: 'history_messages'`
**Solution**: Always call `await rag.initialize_storages()` after creating LightRAG instance

### 2. Embedding Model Changes
When switching embedding models, you MUST clear the data directory (except optionally `kv_store_llm_response_cache.json` for LLM cache).

### 3. Nested Embedding Functions
Cannot wrap already-decorated embedding functions. Use `.func` to access underlying function:
```python
# Wrong: EmbeddingFunc(func=openai_embed)
# Right: EmbeddingFunc(func=openai_embed.func)
```

### 4. Context Length for Ollama
Ollama models default to 8k context; LightRAG requires 32k+. Configure via:
```python
llm_model_kwargs={"options": {"num_ctx": 32768}}
```

## Configuration Files

### .env Configuration
Primary configuration file for API server. Key sections:
- Server settings (HOST, PORT, CORS)
- Storage backends (connection strings via environment variables)
- Query parameters (TOP_K, MAX_TOTAL_TOKENS, etc.)
- Reranking configuration (RERANK_BINDING, RERANK_MODEL)
- Authentication (AUTH_ACCOUNTS, LIGHTRAG_API_KEY)

See `env.example` for comprehensive template.

### Workspace Isolation
Each LightRAG instance can use a `workspace` parameter for data isolation. Implementation varies by storage type:
- File-based: subdirectories
- Collection-based: collection name prefixes
- Relational DB: workspace column filtering
- Qdrant: payload-based partitioning

## Testing Guidelines

### Test Structure
- `tests/`: Main test suite (mirrors feature folders)
- `test_*.py` in root: Specific integration tests
- Markers: `offline`, `integration`, `requires_db`, `requires_api`

### Running Tests
```bash
# Default: runs only offline tests
pytest tests

# Include integration tests
pytest tests --run-integration

# Keep test artifacts for debugging
pytest tests --keep-artifacts

# Configure test workers
pytest tests --test-workers 4
```

### Environment Variables for Tests
Set `LIGHTRAG_*` variables for integration tests:
- `LIGHTRAG_RUN_INTEGRATION=true`
- `LIGHTRAG_KEEP_ARTIFACTS=true`
- `LIGHTRAG_TEST_WORKERS=4`
- Plus storage-specific connection strings

## Code Style

### Language
- Comment Language - Use English for comments and documentation
- Backend Language - Use English for backend code and messages
- Frontend Internationalization: i18next for multi-language support

### Python
- Follow PEP 8 with 4-space indentation
- Use type annotations
- Prefer dataclasses for state management
- Use `lightrag.utils.logger` instead of print
- Async/await patterns throughout
- Keep storage implementations in `kg/` with consistent base class inheritance

### TypeScript/React
- Functional components with hooks
- 2-space indentation
- PascalCase for components
- Tailwind utility-first styling

## Important Architectural Notes

### LLM Requirements
- Minimum 32B parameters recommended
- 32KB context minimum (64KB recommended)
- Avoid reasoning models during indexing
- Stronger models for query stage than indexing stage

### Embedding Models
- Must be consistent across indexing and querying
- Recommended: `BAAI/bge-m3`, `text-embedding-3-large`
- Changing models requires clearing vector storage and recreating with new dimensions

### Reranker Configuration
- Significantly improves retrieval quality
- Recommended models: `BAAI/bge-reranker-v2-m3`, Jina rerankers
- Use "mix" mode when reranker is enabled

### API Server Deployment
- Single-worker mode: Use `lightrag-server` or `uvicorn` (supports all features including docling on macOS)
- Multi-worker mode: Use `lightrag-gunicorn` (NOT compatible with docling extra on macOS due to fork-safety issues with PyTorch/Objective-C frameworks)
- Configure via `.env` file (copy from `env.example`)
- WebUI must be built with `bun run build` before server start in production
- API authentication via `X-API-Key` header or JWT tokens (configured in AUTH_ACCOUNTS)

### Document Processing
- Supported formats: TXT, PDF, DOCX, PPTX, XLSX
- Optional advanced processing: Install `docling` extra (not compatible with macOS multi-worker mode)
- Configure chunk size (500-1500 recommended), entity types, and summary settings in `.env`
- Processing is async and tracked via DOC_STATUS_STORAGE

## Learning Documentation Style Guide

The user is writing learning notes in `my_lightrag_learning/`. When writing or editing these documents, follow these rules:

1. **No Python source code blocks** — use ASCII flow diagrams and JSON data examples instead. The reader wants to understand design intent, not read code.
2. **ASCII diagrams for flows** — use `→`, `├──`, `└──`, `↓`, `▼` to show data flows and pipelines.
3. **Natural language for mechanisms** — explain "what it does" and "why it exists" in plain Chinese sentences. Never walk through code line by line.
4. **Source code references are rare** — only cite a function name (e.g. `process_chunks_unified`) as a location hint when truly necessary. Never cite line number ranges like `utils.py:2730-2738`.
5. **Tables for comparisons and configs** — use markdown tables instead of code blocks for parameter lists and comparisons.
6. **Section structure** — each storage/concept gets: what it stores → why it exists → role in insert/query/delete → relationship to other components.
7. **Feynman style** — explain with concrete examples (e.g. "糖尿病" as the entity name). Ask "why" before "how".
