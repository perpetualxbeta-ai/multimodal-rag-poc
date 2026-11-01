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
1. **Ingestion** (`src/extractor.py`)
   - `pymupdf` does everything by default, with no system packages needed:
     - **Text:** page text blocks outside tables and figures are split into overlapping chunks (`chunk_size`/`chunk_overlap`, paragraph- then sentence-aware).
     - **Tables:** `page.find_tables()` → `Table.to_markdown()`.
     - **Tables (cell text):** rebuilt from the words inside each cell box, because pymupdf's own cell text can drop spaces in bold header rows.
     - **Figures:** raster images (`page.get_image_info`) plus vector-graphic clusters such as charts (`page.cluster_drawings`), rendered with `page.get_pixmap(clip=bbox, dpi=...)`. Vector boxes grow to include short nearby label text (axis ticks, legends). Parts closer than `figure_merge_gap` merge into one figure. Small items (logos, icons), full-page backgrounds and table borders are filtered out. The nearest caption becomes the image chunk's `content`; labelled captions ("Figure 2: ...") beat body text that merely starts with "Figure".
   - `table_backend="unstructured"` swaps in `unstructured[pdf]` hi-res table detection (HTML → Markdown). It needs poppler and tesseract, so it is optional.
   - Scanned pages (no text layer) are logged and skipped; OCR is not implemented yet.
   - Rendered figures are saved to `data/processed/images/{doc_id}/p{page}_fig{n}.png`. `doc_id` is the file stem plus a content hash, so chunk IDs stay stable across runs.
2. **Embeddings** (`src/embedder.py`)
   - **CLIP (local, default):** `sentence-transformers` `clip-ViT-B-32` (or ViT-L/14) embeds images and text into a **shared space**, which enables text→image retrieval.
   - **Caption-then-embed (optional):** a vision LLM (Gemini or a local model via litellm) describes each figure, and a text embedder embeds the caption. Store the caption in the payload either way, because it also helps the generator.
   - Text and table chunks use a text embedder: a local `sentence-transformers` model by default, or Gemini embeddings via `litellm.aembedding` (e.g. `gemini/gemini-embedding-001`).
3. **Vector store** (`src/store.py`)
   - `qdrant-client` supports three modes, selected by config: `:memory:` (tests), local path (`QdrantClient(path="data/qdrant")`), or Docker server (`http://localhost:6333`).
   - One collection with **named vectors** `text` and `clip` (the latter only when the image strategy is CLIP or BOTH). Vector dimensions come from the embedder, never hard-coded.
   - Payload schema: `chunk_id, chunk_type (text|table|image), page_number, content (text/markdown), summary (image caption), image_path`, plus any extractor metadata (`doc_id`, `bbox`, `source_path`).
4. **Retrieval** (`src/retriever.py`): `VectorStoreManager.search` queries each vector space and merges them with Reciprocal Rank Fusion (raw text and CLIP scores are not comparable). `Retriever` then optionally re-ranks (`KeywordReranker`, dependency-free; `CrossEncoderReranker`, sentence-transformers), drops near-duplicate chunks, enforces `max_images` and a `max_context_chars` budget, skips images missing on disk, and returns a `RetrievedContext` with text/table items separate from image items. Every item gets a citation label `S1`, `S2`, ...
5. **Generation** (`src/generator.py`): `MultimodalGenerator` sends a system prompt (grounding and citation rules) plus one user turn: labelled text sources, then each figure as a label/caption text part followed by a base64 `image_url` part (downscaled to `max_image_side`), then the question. It calls `litellm.acompletion` with retries on rate limits and timeouts, and falls back through `fallback_models`. Models litellm marks as non-vision get captions only (`send_images` overrides). Returns a `GenerationResult` with the answer, cited items (parsed from `[S1]`), token usage, cost (litellm price table, `None` if unknown), LLM/total/retrieval latency and attempt count. An empty context skips the LLM call.

## Repository Layout

A flat `src/` package. Modules are imported as `from src.embedder import ...` and use relative imports internally (`from .models import Chunk`).

```
src/
  __init__.py
  models.py        # Chunk dataclass + ChunkType; Chunk.from_any() normalises extractor output   [done]
  extractor.py     # PDF -> text chunks, markdown tables, rendered figure PNGs                    [done]
  embedder.py      # text embedders (MiniLM / litellm), ClipEmbedder, VisionSummarizer,
                   # MultimodalEmbedder with ImageStrategy (caption | clip | both)                [done]
  store.py         # VectorStoreManager: Qdrant collection, index_chunks(), search()              [done]
  retriever.py     # Retriever, RetrievedContext/ContextItem, KeywordReranker, CrossEncoderReranker [done]
  generator.py     # MultimodalGenerator (litellm vision prompts), GenerationResult, TokenUsage     [done]
  pipeline.py      # RAGPipeline: build components from Settings, ingest(), query()               [done]
  config.py        # Settings: config.yaml + MMRAG_* env + .env (pydantic-settings)              [done]
  logging_setup.py # setup_logging(): central logging config, quiets litellm/httpx               [done]
app.py             # CLI: --ingest PATH... / --query "..." / --reset / --model / --json          [done]
tests/
  test_store.py    # embedder + store tests with fake embedders and in-memory Qdrant
  test_extractor.py # builds a sample PDF on the fly (text, table, raster + vector figures)
  test_rag.py      # retriever + generator; litellm faked or run with mock_response (no network)
  test_pipeline.py # extraction units, mock retrieval, config, on-disk persistence, app.py CLI
samples/
  halcyon_fy2025_report.pdf # fictional 5-page report: text, 2 tables, raster + vector charts, diagram
scripts/
  make_sample_pdf.py        # regenerates the sample PDF (reportlab + matplotlib)
data/raw/          # input PDFs (gitignored)
data/processed/    # rendered images, caches (gitignored)
requirements.txt        # core deps for the default Gemini setup (no torch)
requirements-local.txt  # + sentence-transformers, torch, unstructured (local models)
requirements-dev.txt    # + pytest, mypy, ruff, sample-PDF generator deps
pytest.ini         # asyncio_mode = auto
mypy.ini           # strict; pydantic plugin; pymupdf untyped calls allowed
config.yaml        # all model names and settings (override with MMRAG_* env vars)
.env.example       # GEMINI_API_KEY, OPENAI_API_KEY, MMRAG_* examples
README.md          # user-facing quickstart and docs
```

Keep new modules flat in `src/` unless one grows past roughly 400 lines; then split it into a subpackage and update this section.

### Key interfaces
- `extract_pdf(path, ExtractorConfig(...)) -> list[Chunk]` (`src/extractor.py`); `extract_pdf_async` and `extract_many` wrap it for async callers. Every chunk's metadata holds `doc_id` and `source_path`; tables and figures also hold `bbox`.
- `Chunk` (`src/models.py`): `chunk_type`, `content`, `page_number`, `image_path`, `chunk_id`, `metadata`. `point_id` is a deterministic UUID, so re-indexing overwrites rather than duplicates. The extractor may return dicts or objects; `Chunk.from_any` maps common field-name aliases.
- `TextEmbedder` Protocol (`src/embedder.py`): `dimension` property + `async embed(texts)`. New embedding backends implement this.
- `MultimodalEmbedder.embed_chunks()` / `embed_query()` return named vectors keyed `text` / `clip`.
- `VectorStoreManager` (`src/store.py`): `in_memory()`, `from_url()`, `async index_chunks(chunks)`, `async search(query_text, top_k=5, chunk_types=..., page_numbers=...)`.
- `Retriever(store, top_k, max_images, max_context_chars, reranker)` → `async retrieve(query) -> RetrievedContext` (`.text_items`, `.image_items`, `.image_paths`, `.text_block()`).
- `MultimodalGenerator(model, fallback_models=...)` → `async generate(question, context) -> GenerationResult` (`.answer`, `.citations`, `.usage`, `.cost_usd`, `.llm_latency_ms`, `.end_to_end_latency_ms`). `build_messages()` exposes the exact prompt for debugging.
- `load_settings(config_path)` → `Settings`; `RAGPipeline.from_settings(settings, embedder=..., generator=...)` → `async ingest(paths, reset=False) -> list[IngestReport]`, `async query(question) -> QueryResult`. Pass `embedder`/`generator` to inject fakes. `PipelineError` carries user-facing messages (empty index, embedding mismatch → suggests `--reset`).
- `app.build_pipeline` is the CLI's factory; tests monkeypatch it (plus `app.missing_api_keys`, `app.setup_logging`).

## LLM Router (`litellm`)

All LLM and embedding API calls go through `litellm` (`src/embedder.py` for captions and API embeddings, `src/generator.py` for answers). Never call provider SDKs directly.

| Alias | litellm model string (example) | Notes |
|---|---|---|
| `gemini` | `gemini/gemini-3.8-flash` | Free tier via Google AI Studio. Needs `GEMINI_API_KEY`. Vision-capable. Google's deprecations page recommends it for new projects; `gemini-2.5-flash` is limited to existing users and the 1.5 models are retired. Check current names in AI Studio, because they change. |
| `local` | `ollama/llama3.2-vision` (or `ollama/qwen2.5vl`) | Needs `ollama serve` running. `api_base=http://localhost:11434`. |

- Model strings live in `config.yaml`, never in code.
- Use `litellm.acompletion` / `litellm.aembedding` (async).
- Configure fallbacks (e.g. `gemini → local`) and retries with backoff. The Gemini free tier is rate-limited, so expect 429s and handle them gracefully.
- Log model, latency and token usage for every call. Never log API keys or full base64 images.

## Code Style & Conventions

- **Python 3.11+.** Use modern typing: `list[str]`, `X | None`, `typing.Protocol`, `Self`.
- **Strict type hints** on every function, method and module-level variable. Code must pass `mypy --strict`. Avoid `Any` unless it is justified in a comment.
- **Async where it helps:** network-bound work (LLM calls, captioning, API embeddings, Qdrant server calls via `AsyncQdrantClient`) is `async`. Use `asyncio.gather` with a bounded `asyncio.Semaphore` for concurrency. CPU/GPU-bound work (PDF parsing, local CLIP) stays sync, and async callers wrap it with `asyncio.to_thread`.
- **Modular functions:** small, pure-where-possible functions in `src/`. No business logic in `cli.py` or notebooks. Depend on Protocols (`Embedder`, `VectorStore`, `LLMClient`), not concrete classes.
- **Explicit logging:** use `logging.getLogger(__name__)` in every module and configure it once in `logging_setup.py`. Never use `print()` in library code. Log stage boundaries, counts (pages, chunks, figures, vectors), timings and fallbacks. Use structured context (`doc_id`, `page`) in messages.
- **Data models:** pydantic v2 models or `@dataclass(slots=True, frozen=True)` for chunks and results.
- **Errors:** raise specific exceptions. A single bad page or figure must not abort a whole document: log a warning and continue.
- **Config:** use `pydantic-settings`. Secrets come only from `.env` (gitignored). Commit `.env.example` with placeholder keys.
- **Formatting/linting:** `ruff format`, `ruff check`, `mypy --strict`. Docstrings use Google style on public functions.
- **Tests:** `pytest` + `pytest-asyncio` (`asyncio_mode = auto`). Use Qdrant `:memory:` and fake embedders/summarizers (see `tests/test_store.py`); unit tests must not download models or call APIs. Generate test PDFs with pymupdf inside the test (see `tests/test_extractor.py`) rather than committing binary fixtures.

## Tech Stack

- **Document processing:** `pypdf`, `pymupdf`, `unstructured[pdf]`
- **Vector storage:** `qdrant-client` (in-memory, local path, or Docker)
- **Embeddings:** `sentence-transformers`, `torch`, `pillow`
- **LLM integration:** `litellm`
- **Config/CLI:** `pydantic-settings`, `pyyaml`, `python-dotenv`, stdlib `argparse`
- **Support:** `pytest`, `pytest-asyncio`, `ruff`, `mypy`

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
- The default pymupdf extractor needs no system packages. The optional `unstructured[pdf]` hi-res backend needs `brew install poppler tesseract libmagic`, which requires admin rights on the Mac.
- Qdrant via Docker: `docker run -p 6333:6333 -v $(pwd)/data/qdrant_storage:/qdrant/storage qdrant/qdrant`.

## Common Commands

```bash
# setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt     # or requirements.txt (core) / requirements-local.txt (torch)
cp .env.example .env   # then add GEMINI_API_KEY

# run
# extract a PDF and print a summary of its chunks
python -m src.extractor samples/halcyon_fy2025_report.pdf

# full pipeline
python app.py --ingest samples/halcyon_fy2025_report.pdf
python app.py --query "What is the Q3 revenue trend shown in the chart?"
python app.py --query "..." --model ollama/llama3.2-vision --json

# quality
ruff format --line-length 110 src tests app.py && ruff check --line-length 110 src tests app.py && python -m mypy && pytest -q
# run mypy as `python -m mypy` so it sees the project's installed packages (pydantic plugin, pymupdf)
```

## Working Rules for Claude

- Read this file and `config.yaml` before adding a new backend. Add it behind the existing Protocol and register it in config.
- Do not hard-code model names, embedding dimensions, file paths or API keys.
- Keep each change small and runnable. Update the tests and this file's layout section whenever you add a module.
- Do not commit anything under `data/`, `.env`, model caches or Qdrant storage.
- Keep `README.md` in sync with user-facing changes (CLI flags, config keys, env vars, model names).
- When adding a dependency, add it to the right requirements file (`requirements.txt` core, `-local` for torch-based, `-dev` for tooling) with a lower-bound version and say why in the PR or commit message.
