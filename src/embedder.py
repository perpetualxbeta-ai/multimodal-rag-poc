"""Embedding module: text/table embeddings plus two strategies for image figures.

Text and table chunks are embedded with a text embedder:
  * ``SentenceTransformerTextEmbedder`` - local ``all-MiniLM-L6-v2`` (default, MPS aware).
  * ``LiteLLMTextEmbedder`` - any embedding model litellm supports
    (e.g. ``gemini/gemini-embedding-001`` or ``text-embedding-3-small``).

Image figures use one (or both) of:
  * ``ImageStrategy.CAPTION`` - a vision LLM (``VisionSummarizer`` via litellm) writes a
    summary, which is embedded with the text embedder into the ``text`` vector space.
  * ``ImageStrategy.CLIP`` - ``ClipEmbedder`` embeds the image pixels into the ``clip``
    vector space; queries are embedded with CLIP's text tower for text-to-image search.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .models import Chunk, ChunkType

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

TEXT_VECTOR = "text"
CLIP_VECTOR = "clip"

DEFAULT_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CLIP_MODEL = "sentence-transformers/clip-ViT-B-32"

DEFAULT_SUMMARY_PROMPT = (
    "You are indexing a figure extracted from a PDF for a retrieval system. "
    "Describe it so it can be found by text search: state the figure type "
    "(chart, diagram, photo, table image, ...), its title or axis labels, the key "
    "entities, numbers, trends and conclusions it shows. Be factual and concise "
    "(under 150 words). Do not speculate beyond what is visible."
)


def get_device() -> str:
    """Return the best available torch device: ``mps`` (Apple Silicon), ``cuda`` or ``cpu``."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@runtime_checkable
class TextEmbedder(Protocol):
    """Anything that turns strings into fixed-size vectors."""

    @property
    def dimension(self) -> int: ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerTextEmbedder:
    """Local text embedder backed by sentence-transformers (default ``all-MiniLM-L6-v2``)."""

    def __init__(
        self,
        model_name: str = DEFAULT_TEXT_MODEL,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            device = self.device or get_device()
            start = time.perf_counter()
            self._model = SentenceTransformer(self.model_name, device=device)
            logger.info(
                "Loaded text embedder %s on %s in %.1fs",
                self.model_name,
                device,
                time.perf_counter() - start,
            )
        return self._model

    @property
    def dimension(self) -> int:
        dim = self._load().get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError(f"Model {self.model_name} does not report an embedding dimension")
        return int(dim)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._load().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in row] for row in vectors]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode, list(texts))


class LiteLLMTextEmbedder:
    """API text embedder routed through ``litellm.aembedding``.

    Args:
        model: litellm model string, e.g. ``gemini/gemini-embedding-001``.
        dimension: Vector size the model returns (from config; needed to create the collection).
        batch_size: Inputs per API request.
        api_base: Optional custom endpoint (e.g. a local OpenAI-compatible server).
    """

    def __init__(
        self,
        model: str,
        dimension: int,
        batch_size: int = 64,
        api_base: str | None = None,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self._dimension = dimension
        self.batch_size = batch_size
        self.api_base = api_base
        self.max_retries = max_retries

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        import litellm

        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            start = time.perf_counter()
            response = await _with_retries(
                lambda b=batch: litellm.aembedding(model=self.model, input=b, api_base=self.api_base),
                attempts=self.max_retries,
                what=f"embedding batch via {self.model}",
            )
            for item in response.data:
                vector = item["embedding"] if isinstance(item, dict) else item.embedding
                out.append([float(x) for x in vector])
            logger.debug(
                "Embedded %d texts via %s in %.2fs", len(batch), self.model, time.perf_counter() - start
            )
        if out and len(out[0]) != self._dimension:
            raise ValueError(
                f"{self.model} returned {len(out[0])}-dim vectors but dimension={self._dimension} was configured"
            )
        return out


class ClipEmbedder:
    """CLIP image + text embedder in a shared vector space (default ``clip-ViT-B-32``).

    Implements ``TextEmbedder`` (for queries) and adds ``embed_images``.
    CLIP's text tower handles at most 77 tokens, so keep queries short.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_MODEL,
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            device = self.device or get_device()
            start = time.perf_counter()
            self._model = SentenceTransformer(self.model_name, device=device)
            logger.info(
                "Loaded CLIP model %s on %s in %.1fs", self.model_name, device, time.perf_counter() - start
            )
        return self._model

    @property
    def dimension(self) -> int:
        dim = self._load().get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError(f"Model {self.model_name} does not report an embedding dimension")
        return int(dim)

    def _encode(self, items: list[Any]) -> list[list[float]]:
        vectors = self._load().encode(
            items,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in row] for row in vectors]

    def _encode_images(self, paths: list[str]) -> list[list[float]]:
        from PIL import Image

        images = []
        for path in paths:
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
        return self._encode(images)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode, list(texts))

    async def embed_images(self, image_paths: Sequence[str]) -> list[list[float]]:
        if not image_paths:
            return []
        start = time.perf_counter()
        vectors = await asyncio.to_thread(self._encode_images, list(image_paths))
        logger.info("CLIP-embedded %d images in %.2fs", len(vectors), time.perf_counter() - start)
        return vectors


class VisionSummarizer:
    """Generates text summaries of figures with a vision LLM via ``litellm.acompletion``.

    Args:
        model: litellm model string, e.g. ``gemini/gemini-2.5-flash`` or ``ollama/llama3.2-vision``.
        prompt: Instruction sent with each image.
        max_concurrency: Parallel requests (keep low for the Gemini free tier).
    """

    def __init__(
        self,
        model: str,
        prompt: str = DEFAULT_SUMMARY_PROMPT,
        max_concurrency: int = 4,
        max_tokens: int = 400,
        api_base: str | None = None,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.api_base = api_base
        self.max_retries = max_retries
        self._semaphore = asyncio.Semaphore(max_concurrency)

    @staticmethod
    def _data_url(image_path: str) -> str:
        mime = mimetypes.guess_type(image_path)[0] or "image/png"
        encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    async def summarize(self, image_path: str, context: str | None = None) -> str:
        """Return a text summary of one image. ``context`` (e.g. a caption) is added to the prompt."""
        import litellm

        prompt = self.prompt if not context else f"{self.prompt}\n\nNearby text from the PDF: {context}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": self._data_url(image_path)}},
                ],
            }
        ]
        async with self._semaphore:
            start = time.perf_counter()
            response = await _with_retries(
                lambda: litellm.acompletion(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    api_base=self.api_base,
                ),
                attempts=self.max_retries,
                what=f"summary of {image_path} via {self.model}",
            )
        summary = str(response.choices[0].message.content or "").strip()
        logger.info(
            "Summarised %s via %s in %.2fs (%d chars)",
            image_path,
            self.model,
            time.perf_counter() - start,
            len(summary),
        )
        return summary

    async def summarize_many(
        self, image_paths: Sequence[str], contexts: Sequence[str | None] | None = None
    ) -> list[str | None]:
        """Summarise images concurrently. A failed image yields ``None`` instead of aborting the batch."""
        ctx: Sequence[str | None] = contexts if contexts is not None else [None] * len(image_paths)

        async def one(path: str, context: str | None) -> str | None:
            try:
                return await self.summarize(path, context)
            except Exception:
                logger.warning("Vision summary failed for %s", path, exc_info=True)
                return None

        return list(await asyncio.gather(*(one(p, c) for p, c in zip(image_paths, ctx, strict=True))))


class ImageStrategy(StrEnum):
    """How image figures are represented in the vector store."""

    CAPTION = "caption"  # vision-LLM summary embedded as text
    CLIP = "clip"  # direct CLIP visual embedding
    BOTH = "both"  # store both vectors on the same point


@dataclass(slots=True, frozen=True)
class EmbeddedChunk:
    """A chunk with its named vectors (``text`` and/or ``clip``) and optional image summary."""

    chunk: Chunk
    vectors: dict[str, list[float]] = field(default_factory=dict)
    summary: str | None = None


class MultimodalEmbedder:
    """Routes each chunk type to the right embedder and produces named vectors.

    Vector spaces:
      * ``text`` - text chunks, Markdown tables and (CAPTION/BOTH) image summaries.
      * ``clip`` - (CLIP/BOTH) image pixels; queries use CLIP's text encoder.
    """

    def __init__(
        self,
        text_embedder: TextEmbedder,
        image_strategy: ImageStrategy = ImageStrategy.CAPTION,
        clip_embedder: ClipEmbedder | None = None,
        summarizer: VisionSummarizer | None = None,
    ) -> None:
        if image_strategy in (ImageStrategy.CAPTION, ImageStrategy.BOTH) and summarizer is None:
            raise ValueError(f"image_strategy={image_strategy.value} requires a VisionSummarizer")
        if image_strategy in (ImageStrategy.CLIP, ImageStrategy.BOTH) and clip_embedder is None:
            raise ValueError(f"image_strategy={image_strategy.value} requires a ClipEmbedder")
        self.text_embedder = text_embedder
        self.image_strategy = image_strategy
        self.clip_embedder = clip_embedder
        self.summarizer = summarizer

    @property
    def uses_clip(self) -> bool:
        return self.image_strategy in (ImageStrategy.CLIP, ImageStrategy.BOTH)

    @property
    def uses_captions(self) -> bool:
        return self.image_strategy in (ImageStrategy.CAPTION, ImageStrategy.BOTH)

    def vector_dimensions(self) -> dict[str, int]:
        """Named vector spaces and their sizes, used to create the Qdrant collection."""
        dims = {TEXT_VECTOR: self.text_embedder.dimension}
        if self.uses_clip and self.clip_embedder is not None:
            dims[CLIP_VECTOR] = self.clip_embedder.dimension
        return dims

    async def embed_chunks(self, chunks: Sequence[Chunk]) -> list[EmbeddedChunk]:
        """Embed a mixed batch of text, table and image chunks.

        Chunks that end up with no vector (empty text, unreadable image, failed summary with no
        fallback text) are skipped with a warning.
        """
        start = time.perf_counter()
        images = [c for c in chunks if c.chunk_type is ChunkType.IMAGE]
        missing = [c for c in images if not c.image_path or not Path(c.image_path).is_file()]
        for c in missing:
            logger.warning(
                "Image chunk on page %d has no readable image_path (%r)", c.page_number, c.image_path
            )
        missing_ids = {id(c) for c in missing}

        # 1. Vision summaries for images (CAPTION / BOTH).
        summaries: dict[int, str] = {}
        if self.uses_captions and self.summarizer is not None:
            to_sum = [c for c in images if id(c) not in missing_ids]
            results = await self.summarizer.summarize_many(
                [c.image_path or "" for c in to_sum], [c.content or None for c in to_sum]
            )
            for c, summary in zip(to_sum, results, strict=True):
                text = summary or c.content.strip()
                if text:
                    summaries[id(c)] = text

        # 2. Text-space inputs: text, tables, image summaries.
        text_inputs: list[tuple[Chunk, str]] = []
        for c in chunks:
            if c.chunk_type is ChunkType.IMAGE:
                if id(c) in summaries:
                    text_inputs.append((c, summaries[id(c)]))
            elif c.content.strip():
                text_inputs.append((c, c.content))
            else:
                logger.warning("Skipping empty %s chunk on page %d", c.chunk_type.value, c.page_number)
        text_vectors = await self.text_embedder.embed([t for _, t in text_inputs])

        vectors: dict[int, dict[str, list[float]]] = {id(c): {} for c in chunks}
        for (c, _), vec in zip(text_inputs, text_vectors, strict=True):
            vectors[id(c)][TEXT_VECTOR] = vec

        # 3. CLIP image vectors (CLIP / BOTH).
        if self.uses_clip and self.clip_embedder is not None:
            clip_targets = [c for c in images if id(c) not in missing_ids]
            try:
                clip_vectors = await self.clip_embedder.embed_images(
                    [c.image_path or "" for c in clip_targets]
                )
            except Exception:
                logger.exception("CLIP batch failed; retrying images one at a time")
                clip_vectors = []
                kept = []
                for c in clip_targets:
                    try:
                        clip_vectors.extend(await self.clip_embedder.embed_images([c.image_path or ""]))
                        kept.append(c)
                    except Exception:
                        logger.warning("CLIP embedding failed for %s", c.image_path, exc_info=True)
                clip_targets = kept
            for c, vec in zip(clip_targets, clip_vectors, strict=True):
                vectors[id(c)][CLIP_VECTOR] = vec

        embedded = [EmbeddedChunk(c, vectors[id(c)], summaries.get(id(c))) for c in chunks if vectors[id(c)]]
        skipped = len(chunks) - len(embedded)
        logger.info(
            "Embedded %d/%d chunks (text=%d, table=%d, image=%d, strategy=%s, skipped=%d) in %.2fs",
            len(embedded),
            len(chunks),
            sum(c.chunk_type is ChunkType.TEXT for c in chunks),
            sum(c.chunk_type is ChunkType.TABLE for c in chunks),
            len(images),
            self.image_strategy.value,
            skipped,
            time.perf_counter() - start,
        )
        return embedded

    async def embed_query(self, query: str) -> dict[str, list[float]]:
        """Embed a search query into every active vector space."""
        out = {TEXT_VECTOR: (await self.text_embedder.embed([query]))[0]}
        if self.uses_clip and self.clip_embedder is not None:
            out[CLIP_VECTOR] = (await self.clip_embedder.embed([query]))[0]
        return out


async def _with_retries(call: Any, attempts: int, what: str, base_delay: float = 2.0) -> Any:
    """Await ``call()`` with exponential backoff (handles rate limits such as Gemini free-tier 429s)."""
    for attempt in range(1, attempts + 1):
        try:
            return await call()
        except Exception as exc:
            if attempt == attempts:
                raise
            delay = base_delay * 2 ** (attempt - 1)
            logger.warning("%s failed (%s); retry %d/%d in %.0fs", what, exc, attempt, attempts - 1, delay)
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable")
