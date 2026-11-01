# CLAUDE.md - multimodal-rag-poc

Guidance for Claude Code when working in this repository.

## Project Overview

A modular **multimodal Retrieval-Augmented Generation (RAG) proof of concept** in Python 3.11+.
It ingests PDFs and extracts three kinds of content: text chunks, tables (as Markdown) and rendered image figures. It embeds each kind, stores the vectors in Qdrant, retrieves across all three for a query, and answers with a vision-capable LLM routed through `litellm`.

This is a PoC. Favour clarity, small composable modules and swappable backends over premature optimisation. Every external dependency (embedding model, vector store, LLM) must sit behind a thin interface so it can be swapped through config without code changes.

## Architecture

```
PDF ──► ingestion ──► [TextChunk | TableChunk | ImageChunk] ──► embeddings ──► Qdrant
                                                                                │
Query ──► embed query ──► retrieve (text + table + image) ──► rerank/merge ──────┘
                                     │
                                     ▼
                      build multimodal prompt ──► litellm router ──► answer + citations
```

### Pipeline stages
1. **Ingestion** (`src/ingestion/`)
   - `pymupdf` (fitz) is the primary parser: page text, image extraction, and rendering figure regions to PNG (`page.get_pixmap(clip=bbox, dpi=...)`).
   - `unstructured[pdf]` handles layout-aware partitioning, mainly for table detection (`strategy="hi_res"`, `infer_table_structure=True`). Convert table HTML to Markdown.
   - `pypdf` is used for lightweight metadata and page-count checks, and as a fallback text extractor.
   - Text is chunked by token or character count with overlap. Each chunk keeps provenance: `doc_id`, `page`, `bbox`, `source_path`.
   - Rendered figures are saved to `data/processed/images/{doc_id}/p{page}_fig{n}.png`.
2. **Embeddings** (`src/embeddings/`)
   - **CLIP (local, default):** `sentence-transformers` `clip-ViT-B-32` (or ViT-L/14) embeds images and text into a **shared space**, which enables text→image retrieval.
   - **OpenAI (API):** OpenAI's embedding endpoint is **text-only**. The "OpenAI image" path is therefore *caption-then-embed*: a vision LLM (GPT-4o via litellm) describes the figure, and `text-embedding-3-small/large` embeds the caption. Store the caption in the payload either way, because it also helps the generator.
   - Text and table chunks use a text embedder (`sentence-transformers` model or OpenAI `text-embedding-3-*`).
3. **Vector store** (`src/vectorstore/`)
   - `qdrant-client` supports three modes, selected by config: `:memory:` (tests), local path (`QdrantClient(path="data/qdrant")`), or Docker server (`http://localhost:6333`).
   - Use one collection with **named vectors** (e.g. `text`, `clip`) or separate collections per modality. Keep the choice in config. Vector dimensions must come from the embedder, never be hard-coded.
   - Payload schema: `chunk_id, doc_id, modality (text|table|image), page, bbox, content (text/markdown/caption), image_path, source_path`.
4. **Retrieval** (`src/retrieval/`): query each modality, merge by score (normalise scores per modality first), apply an optional simple rerank, and return the top-k with provenance.
5. **Generation** (`src/llm/`): build a prompt with retrieved text and tables plus images (base64 `image_url` content parts) and call `litellm.acompletion`. The answer must cite `doc_id`/`page`.

## Repository Layout (target)

```
src/mmrag/
  __init__.py
  config.py            # pydantic-settings; loads .env + config.yaml
  logging_setup.py     # central logging config
  models.py            # Chunk dataclasses / pydantic models (TextChunk, TableChunk, ImageChunk)
  ingestion/
    pdf_loader.py      # pymupdf + pypdf
    tables.py          # unstructured table extraction -> markdown
    figures.py         # figure detection + rendering to PNG
    chunking.py
  embeddings/
    base.py            # Embedder Protocol
    clip_embedder.py   # sentence-transformers CLIP (MPS aware)
    openai_embedder.py # text-embedding-3-* via litellm.aembedding
    captioner.py       # vision-LLM figure captions
  vectorstore/
    qdrant_store.py
  retrieval/
    retriever.py
  llm/
    router.py          # litellm wrapper, model aliases, fallbacks
    prompts.py
  pipeline.py          # ingest() / query() orchestration
  cli.py               # `python -m mmrag ingest|query`
tests/
data/raw/              # input PDFs (gitignored)
data/processed/        # rendered images, caches (gitignored)
config.yaml
.env.example
pyproject.toml
```

## LLM Router (`litellm`)

All LLM and embedding API calls go through `src/mmrag/llm/router.py`. Never call provider SDKs directly.

| Alias | litellm model string (example) | Notes |
|---|---|---|
| `openai` | `gpt-4o` | Needs `OPENAI_API_KEY`. Vision-capable. |
| `gemini` | `gemini/gemini-2.5-flash` | Free tier via Google AI Studio. Needs `GEMINI_API_KEY`. Vision-capable. Check the current model names in AI Studio, because they change. |
| `local` | `ollama/llama3.2-vision` (or `ollama/qwen2.5vl`) | Needs `ollama serve` running. `api_base=http://localhost:11434`. |

- Model strings live in `config.yaml`, never in code.
- Use `litellm.acompletion` / `litellm.aembedding` (async).
- Configure fallbacks (e.g. `gemini → local`) and retries with backoff. The Gemini free tier is rate-limited, so expect 429s and handle them gracefully.
- Log model, latency and token usage for every call. Never log API keys or full base64 images.

## Code Style & Conventions

- **Python 3.11+.** Use modern typing: `list[str]`, `X | None`, `typing.Protocol`, `Self`.
- **Strict type hints** on every function, method and module-level variable. Code must pass `mypy --strict`. Avoid `Any` unless it is justified in a comment.
- **Async where it helps:** network-bound work (LLM calls, captioning, OpenAI embeddings, Qdrant server calls via `AsyncQdrantClient`) is `async`. Use `asyncio.gather` with a bounded `asyncio.Semaphore` for concurrency. CPU/GPU-bound work (PDF parsing, local CLIP) stays sync, and async callers wrap it with `asyncio.to_thread`.
- **Modular functions:** small, pure-where-possible functions in `src/`. No business logic in `cli.py` or notebooks. Depend on Protocols (`Embedder`, `VectorStore`, `LLMClient`), not concrete classes.
- **Explicit logging:** use `logging.getLogger(__name__)` in every module and configure it once in `logging_setup.py`. Never use `print()` in library code. Log stage boundaries, counts (pages, chunks, figures, vectors), timings and fallbacks. Use structured context (`doc_id`, `page`) in messages.
- **Data models:** pydantic v2 models or `@dataclass(slots=True, frozen=True)` for chunks and results.
- **Errors:** raise specific exceptions. A single bad page or figure must not abort a whole document: log a warning and continue.
- **Config:** use `pydantic-settings`. Secrets come only from `.env` (gitignored). Commit `.env.example` with placeholder keys.
- **Formatting/linting:** `ruff format`, `ruff check`, `mypy --strict`. Docstrings use Google style on public functions.
- **Tests:** `pytest` + `pytest-asyncio`. Use Qdrant `:memory:` and mock litellm in unit tests. Keep one small sample PDF in `tests/fixtures/`.

## Tech Stack

- **Document processing:** `pypdf`, `pymupdf`, `unstructured[pdf]`
- **Vector storage:** `qdrant-client` (in-memory, local path, or Docker)
- **Embeddings:** `sentence-transformers`, `torch`, `pillow`
- **LLM integration:** `litellm`
- **Support:** `pydantic`, `pydantic-settings`, `python-dotenv`, `tenacity` (retries), `typer` (CLI), `pytest`, `pytest-asyncio`, `ruff`, `mypy`

## Apple Silicon (macOS, M-series)

- Pick the device once in a helper:
  ```python
  def get_device() -> str:
      if torch.backends.mps.is_available():
          return "mps"
      if torch.cuda.is_available():
          return "cuda"
      return "cpu"
  ```
  Pass it to `SentenceTransformer(model_name, device=get_device())`.
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` so unsupported ops fall back to CPU instead of crashing.
- Use native arm64 Python (not Rosetta). Check with `python -c "import platform; print(platform.machine())"`, which should print `arm64`.
- `unstructured[pdf]` hi-res mode needs system packages: `brew install poppler tesseract libmagic`. If hi-res is too slow or heavy, fall back to `strategy="fast"` and to pymupdf's `page.find_tables()`.
- Qdrant via Docker: `docker run -p 6333:6333 -v $(pwd)/data/qdrant_storage:/qdrant/storage qdrant/qdrant`.

## Common Commands

```bash
# setup
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # then add OPENAI_API_KEY / GEMINI_API_KEY

# run
python -m mmrag ingest data/raw/sample.pdf
python -m mmrag query "What does Figure 3 show about latency?" --llm gemini

# quality
ruff format . && ruff check . && mypy src && pytest -q
```

## Working Rules for Claude

- Read this file and `config.yaml` before adding a new backend. Add it behind the existing Protocol and register it in config.
- Do not hard-code model names, embedding dimensions, file paths or API keys.
- Keep each change small and runnable. Update the tests and this file's layout section whenever you add a module.
- Do not commit anything under `data/`, `.env`, model caches or Qdrant storage.
- When adding a dependency, add it to `pyproject.toml` with a lower-bound version and say why in the PR or commit message.
