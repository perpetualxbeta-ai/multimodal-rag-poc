"""Retrieval: search Qdrant, optionally re-rank, and assemble a context for the generator.

``Retriever.retrieve(query)`` returns a ``RetrievedContext`` that keeps text/table evidence
separate from image figures (file paths plus captions), with a citation label (``S1``, ``S2``, ...)
on every item so the generator can cite sources.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .models import ChunkType
from .store import SearchResult, VectorStoreManager

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9.%$-]*", re.IGNORECASE)
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
    ]
)


@dataclass(slots=True, frozen=True)
class ContextItem:
    """One piece of evidence handed to the generator.

    Attributes:
        label: Citation label, e.g. ``"S3"``; the answer cites it as ``[S3]``.
        content: Text, Markdown table, or the image's caption/summary.
        image_path: PNG path for image items, ``None`` otherwise.
        score: Raw similarity score from the vector store.
        rank_score: Score used for the final ordering (fused or re-ranked).
    """

    label: str
    chunk_type: ChunkType
    page_number: int
    content: str
    doc_id: str | None
    source: str | None
    image_path: str | None
    score: float
    rank_score: float
    point_id: str

    @property
    def is_image(self) -> bool:
        return self.chunk_type is ChunkType.IMAGE

    @property
    def citation(self) -> str:
        """Human-readable source, e.g. ``"halcyon_fy2025_report.pdf, page 2 (table)"``."""
        name = Path(self.source).name if self.source else (self.doc_id or "document")
        return f"{name}, page {self.page_number} ({self.chunk_type.value})"


@dataclass(slots=True, frozen=True)
class RetrievedContext:
    """Retrieval output, split into text evidence and image figures."""

    query: str
    text_items: list[ContextItem] = field(default_factory=list)
    image_items: list[ContextItem] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def items(self) -> list[ContextItem]:
        """All items in label order."""
        return sorted(self.text_items + self.image_items, key=lambda i: int(i.label[1:]))

    @property
    def image_paths(self) -> list[str]:
        return [i.image_path for i in self.image_items if i.image_path]

    @property
    def is_empty(self) -> bool:
        return not self.text_items and not self.image_items

    def by_label(self, label: str) -> ContextItem | None:
        return next((i for i in self.items if i.label == label), None)

    def text_block(self) -> str:
        """Text evidence formatted for a prompt, one labelled block per item."""
        return "\n\n".join(f"[{i.label}] {i.citation}\n{i.content}" for i in self.text_items)


@runtime_checkable
class Reranker(Protocol):
    """Re-orders search results for a query. Implementations must not drop or invent results."""

    async def rerank(
        self, query: str, results: Sequence[SearchResult]
    ) -> list[tuple[SearchResult, float]]: ...


class KeywordReranker:
    """Dependency-free re-ranker: blends retrieval rank with query-term overlap.

    Useful for exact facts (numbers, names, "Q3") that embedding search can rank too low.
    ``weight`` is how much keyword overlap counts against the original rank (0 = no change).
    """

    def __init__(self, weight: float = 0.5) -> None:
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be between 0 and 1")
        self.weight = weight

    async def rerank(self, query: str, results: Sequence[SearchResult]) -> list[tuple[SearchResult, float]]:
        terms = _terms(query)
        n = len(results)
        scored: list[tuple[SearchResult, float]] = []
        for rank, result in enumerate(results):
            rank_score = 1.0 - rank / n if n else 0.0
            text = f"{result.content} {result.summary or ''}"
            overlap = len(terms & _terms(text)) / len(terms) if terms else 0.0
            scored.append((result, (1 - self.weight) * rank_score + self.weight * overlap))
        return sorted(scored, key=lambda pair: pair[1], reverse=True)


class CrossEncoderReranker:
    """Cross-encoder re-ranker (sentence-transformers), e.g. ``cross-encoder/ms-marco-MiniLM-L-6-v2``.

    More accurate than embedding similarity but slower; runs locally (MPS on Apple Silicon).
    Image results are scored on their caption/summary text.
    """

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model: CrossEncoder | None = None

    def _load(self) -> CrossEncoder:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            from .embedder import get_device

            device = self.device or get_device()
            self._model = CrossEncoder(self.model_name, device=device)
            logger.info("Loaded cross-encoder %s on %s", self.model_name, device)
        return self._model

    def _score(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(s) for s in self._load().predict(pairs, show_progress_bar=False)]

    async def rerank(self, query: str, results: Sequence[SearchResult]) -> list[tuple[SearchResult, float]]:
        if not results:
            return []
        pairs = [(query, r.content or r.summary or "") for r in results]
        scores = await asyncio.to_thread(self._score, pairs)
        return sorted(zip(results, scores, strict=True), key=lambda pair: pair[1], reverse=True)


class Retriever:
    """Searches the vector store and assembles a citation-labelled context.

    Args:
        store: The vector store to search.
        top_k: Results to keep after re-ranking.
        candidate_k: Results fetched from the store before re-ranking (defaults to ``2 * top_k``
            when a re-ranker is set, otherwise ``top_k``).
        max_images: Maximum figures passed on; each image costs many prompt tokens.
        max_context_chars: Character budget for text/table evidence; lower-ranked items that
            do not fit are dropped.
        reranker: Optional ``Reranker`` (``KeywordReranker``, ``CrossEncoderReranker``, ...).
    """

    def __init__(
        self,
        store: VectorStoreManager,
        top_k: int = 6,
        candidate_k: int | None = None,
        max_images: int = 3,
        max_context_chars: int = 12_000,
        reranker: Reranker | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.store = store
        self.top_k = top_k
        self.candidate_k = candidate_k or (top_k * 2 if reranker else top_k)
        self.max_images = max_images
        self.max_context_chars = max_context_chars
        self.reranker = reranker

    async def retrieve(
        self,
        query: str,
        *,
        chunk_types: Sequence[ChunkType | str] | None = None,
        page_numbers: Sequence[int] | None = None,
    ) -> RetrievedContext:
        """Search, re-rank, de-duplicate and split results into text and image evidence."""
        start = time.perf_counter()
        results = await self.store.search(
            query, top_k=self.candidate_k, chunk_types=chunk_types, page_numbers=page_numbers
        )
        if self.reranker is not None and results:
            ranked = await self.reranker.rerank(query, results)
        else:
            ranked = [(r, r.fused_score) for r in results]

        text_items: list[ContextItem] = []
        image_items: list[ContextItem] = []
        kept_text: list[str] = []
        used_chars = 0
        label_no = 0
        dropped = {"duplicate": 0, "budget": 0, "image_cap": 0, "missing_image": 0}

        for result, rank_score in ranked:
            if len(text_items) + len(image_items) >= self.top_k:
                break
            if result.chunk_type is ChunkType.IMAGE:
                if len(image_items) >= self.max_images:
                    dropped["image_cap"] += 1
                    continue
                if not result.image_path or not Path(result.image_path).is_file():
                    dropped["missing_image"] += 1
                    logger.warning("Retrieved image is missing on disk: %s", result.image_path)
                    continue
                label_no += 1
                image_items.append(self._item(result, rank_score, label_no))
                continue

            content = result.content.strip()
            if _is_near_duplicate(content, kept_text):
                dropped["duplicate"] += 1
                continue
            if used_chars + len(content) > self.max_context_chars and text_items:
                dropped["budget"] += 1
                continue
            label_no += 1
            used_chars += len(content)
            kept_text.append(content)
            text_items.append(self._item(result, rank_score, label_no))

        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Retrieved %d text/table + %d image items for %r from %d candidates in %.0f ms (dropped: %s)",
            len(text_items),
            len(image_items),
            query[:80],
            len(results),
            latency_ms,
            {k: v for k, v in dropped.items() if v},
        )
        return RetrievedContext(query, text_items, image_items, latency_ms)

    @staticmethod
    def _item(result: SearchResult, rank_score: float, label_no: int) -> ContextItem:
        payload = result.payload
        content = result.content.strip()
        if result.chunk_type is ChunkType.IMAGE:
            # Prefer the vision summary, keep the printed caption too when both exist.
            parts = [p for p in (content, (result.summary or "").strip()) if p]
            content = "\n".join(dict.fromkeys(parts))
        return ContextItem(
            label=f"S{label_no}",
            chunk_type=result.chunk_type,
            page_number=result.page_number,
            content=content,
            doc_id=_str_or_none(payload.get("doc_id")),
            source=_str_or_none(payload.get("source_path")),
            image_path=result.image_path,
            score=result.score,
            rank_score=float(rank_score),
            point_id=result.point_id,
        )


def renumber(context: RetrievedContext) -> RetrievedContext:
    """Return a copy with labels renumbered S1..Sn in order (after manual filtering)."""
    items = context.items
    relabelled = {i.point_id: replace(i, label=f"S{n}") for n, i in enumerate(items, start=1)}
    return replace(
        context,
        text_items=[relabelled[i.point_id] for i in context.text_items],
        image_items=[relabelled[i.point_id] for i in context.image_items],
    )


def _terms(text: str) -> set[str]:
    return {w.lower().strip(".") for w in _WORD_RE.findall(text) if w.lower() not in _STOPWORDS}


def _is_near_duplicate(text: str, kept: list[str], threshold: float = 0.8) -> bool:
    """True if ``text`` is contained in, or mostly overlaps (word Jaccard), an already-kept chunk."""
    if not text:
        return True
    words = set(text.lower().split())
    for other in kept:
        if text in other or (other in text and len(other) > 0.8 * len(text)):
            return True
        other_words = set(other.lower().split())
        union = words | other_words
        if union and len(words & other_words) / len(union) >= threshold:
            return True
    return False


def _str_or_none(value: object) -> str | None:
    return str(value) if value is not None else None
