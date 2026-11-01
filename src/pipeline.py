"""End-to-end orchestration: build components from settings, ingest PDFs, answer questions.

Example:
    >>> settings = load_settings()
    >>> async with RAGPipeline.from_settings(settings) as rag:
    ...     await rag.ingest(["samples/halcyon_fy2025_report.pdf"])
    ...     result = await rag.query("What was the Q3 operating margin?")
    ...     print(result.answer)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Self

from .config import Settings
from .embedder import (
    ClipEmbedder,
    ImageStrategy,
    LiteLLMTextEmbedder,
    MultimodalEmbedder,
    SentenceTransformerTextEmbedder,
    TextEmbedder,
    VisionSummarizer,
)
from .extractor import ExtractorConfig, extract_pdf_async
from .generator import GenerationResult, MultimodalGenerator
from .models import ChunkType
from .retriever import CrossEncoderReranker, KeywordReranker, Reranker, RetrievedContext, Retriever
from .store import VectorStoreManager

logger = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """A user-facing pipeline failure (bad input, incompatible index, ...)."""


@dataclass(slots=True, frozen=True)
class IngestReport:
    """Outcome of ingesting one PDF."""

    path: str
    doc_id: str | None = None
    text_chunks: int = 0
    table_chunks: int = 0
    image_chunks: int = 0
    points_indexed: int = 0
    seconds: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(slots=True, frozen=True)
class QueryResult:
    """A question, the evidence retrieved for it, and the generated answer."""

    question: str
    context: RetrievedContext
    generation: GenerationResult
    images: list[str] = field(default_factory=list)

    @property
    def answer(self) -> str:
        return self.generation.answer


def build_embedder(settings: Settings) -> MultimodalEmbedder:
    """Create the embedder described by ``settings.embedding``."""
    cfg = settings.embedding
    text_embedder: TextEmbedder
    if cfg.text.provider == "local":
        text_embedder = SentenceTransformerTextEmbedder(cfg.text.model)
    else:
        if cfg.text.dimension is None:  # guarded by the settings validator too
            raise PipelineError("embedding.text.dimension is required for the litellm provider")
        text_embedder = LiteLLMTextEmbedder(cfg.text.model, cfg.text.dimension, api_base=cfg.text.api_base)

    strategy = ImageStrategy(cfg.image_strategy)
    summarizer = None
    clip = None
    if strategy in (ImageStrategy.CAPTION, ImageStrategy.BOTH):
        summarizer = VisionSummarizer(
            cfg.caption_model, max_concurrency=cfg.caption_concurrency, api_base=cfg.caption_api_base
        )
    if strategy in (ImageStrategy.CLIP, ImageStrategy.BOTH):
        clip = ClipEmbedder(cfg.clip_model)
    return MultimodalEmbedder(text_embedder, strategy, clip_embedder=clip, summarizer=summarizer)


def build_reranker(settings: Settings) -> Reranker | None:
    cfg = settings.retrieval
    if cfg.reranker == "keyword":
        return KeywordReranker(cfg.reranker_weight)
    if cfg.reranker == "cross-encoder":
        return CrossEncoderReranker(cfg.cross_encoder_model)
    return None


def build_generator(settings: Settings) -> MultimodalGenerator:
    cfg = settings.llm
    return MultimodalGenerator(
        cfg.model,
        fallback_models=cfg.fallback_models,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        api_base=cfg.api_base,
        send_images=cfg.send_images,
    )


def build_store(settings: Settings, embedder: MultimodalEmbedder) -> VectorStoreManager:
    cfg = settings.qdrant
    if cfg.url:
        return VectorStoreManager(embedder, cfg.collection, url=cfg.url, api_key=cfg.api_key)
    if cfg.path:
        Path(cfg.path).mkdir(parents=True, exist_ok=True)
        return VectorStoreManager(embedder, cfg.collection, path=cfg.path)
    return VectorStoreManager(embedder, cfg.collection, location=":memory:")


class RAGPipeline:
    """Wires extractor → embedder → Qdrant → retriever → generator.

    Use ``RAGPipeline.from_settings(settings)``; pass ``embedder`` / ``generator`` to substitute
    components (e.g. fakes in tests).
    """

    def __init__(
        self,
        settings: Settings,
        store: VectorStoreManager,
        retriever: Retriever,
        generator: MultimodalGenerator,
    ) -> None:
        self.settings = settings
        self.store = store
        self.retriever = retriever
        self.generator = generator
        ext = settings.extraction
        self.extractor_config = ExtractorConfig(
            output_dir=ext.output_dir,
            chunk_size=ext.chunk_size,
            chunk_overlap=ext.chunk_overlap,
            image_dpi=ext.image_dpi,
            table_backend=ext.table_backend,
        )

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        embedder: MultimodalEmbedder | None = None,
        generator: MultimodalGenerator | None = None,
    ) -> Self:
        embedder = embedder or build_embedder(settings)
        store = build_store(settings, embedder)
        cfg = settings.retrieval
        retriever = Retriever(
            store,
            top_k=cfg.top_k,
            max_images=cfg.max_images,
            max_context_chars=cfg.max_context_chars,
            reranker=build_reranker(settings),
        )
        return cls(settings, store, retriever, generator or build_generator(settings))

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self.close()

    async def close(self) -> None:
        await self.store.close()

    async def open_index(self, reset: bool = False) -> None:
        """Create or check the collection. ``reset=True`` deletes everything indexed so far.

        Raises:
            PipelineError: If the existing index was built with different embedding settings.
        """
        try:
            await self.store.ensure_collection(recreate=reset)
        except ValueError as exc:
            raise PipelineError(
                f"{exc}\nThe index was built with different embedding settings. "
                "Re-ingest with --reset to rebuild it."
            ) from exc

    async def ingest(self, paths: Sequence[str | Path], reset: bool = False) -> list[IngestReport]:
        """Extract and index PDFs (folders are searched for ``*.pdf``). One bad file does not stop the rest."""
        await self.open_index(reset=reset)
        files = expand_pdf_paths(paths)
        if not files:
            raise PipelineError(f"No PDF files found in: {', '.join(map(str, paths))}")

        reports: list[IngestReport] = []
        for path in files:
            start = time.perf_counter()
            try:
                chunks = await extract_pdf_async(path, self.extractor_config)
                points = await self.store.index_chunks(chunks)
            except Exception as exc:  # report per file and keep going
                logger.error("Failed to ingest %s: %s", path.name, exc)
                logger.debug("Ingest failure details", exc_info=True)
                reports.append(IngestReport(str(path), error=f"{type(exc).__name__}: {exc}"))
                continue
            counts = {t: sum(c.chunk_type is t for c in chunks) for t in ChunkType}
            report = IngestReport(
                path=str(path),
                doc_id=str(chunks[0].metadata.get("doc_id")) if chunks else None,
                text_chunks=counts[ChunkType.TEXT],
                table_chunks=counts[ChunkType.TABLE],
                image_chunks=counts[ChunkType.IMAGE],
                points_indexed=points,
                seconds=time.perf_counter() - start,
            )
            logger.info(
                "Ingested %s: %d text, %d table, %d image chunks -> %d points in %.1fs",
                path.name,
                report.text_chunks,
                report.table_chunks,
                report.image_chunks,
                points,
                report.seconds,
            )
            reports.append(report)
        return reports

    async def query(
        self,
        question: str,
        *,
        chunk_types: Sequence[ChunkType | str] | None = None,
        page_numbers: Sequence[int] | None = None,
    ) -> QueryResult:
        """Retrieve evidence for ``question`` and generate a cited answer.

        Raises:
            PipelineError: If nothing has been ingested yet or the index is incompatible.
        """
        if not question.strip():
            raise PipelineError("The question is empty.")
        await self.open_index()
        if await self.store.count() == 0:
            raise PipelineError(
                "The index is empty. Ingest a PDF first, e.g. --ingest samples/halcyon_fy2025_report.pdf"
            )
        context = await self.retriever.retrieve(question, chunk_types=chunk_types, page_numbers=page_numbers)
        generation = await self.generator.generate(question, context)
        return QueryResult(question, context, generation, context.image_paths)


def expand_pdf_paths(paths: Sequence[str | Path]) -> list[Path]:
    """Expand folders to the PDFs inside them (recursively); keep explicit files as given."""
    out: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            out.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() == ".pdf"))
        elif path.is_file():
            out.append(path)
        else:
            logger.warning("Not found: %s", path)
    return list(dict.fromkeys(out))
