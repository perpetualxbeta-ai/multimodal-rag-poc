"""Qdrant vector store for text, table and image chunks."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Self

from qdrant_client import AsyncQdrantClient, models

from .embedder import CLIP_VECTOR, TEXT_VECTOR, EmbeddedChunk, MultimodalEmbedder
from .models import Chunk, ChunkType

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "mmrag_chunks"
_RRF_K = 60  # standard Reciprocal Rank Fusion constant


@dataclass(slots=True, frozen=True)
class SearchResult:
    """One retrieved chunk.

    Attributes:
        score: Raw cosine similarity in the vector space that matched (``vector_name``).
            Text-space and CLIP-space scores sit on different scales; CLIP text-to-image
            scores are typically ~0.2-0.35 even for good matches.
        fused_score: Reciprocal Rank Fusion score used to order mixed results.
    """

    point_id: str
    score: float
    fused_score: float
    vector_name: str
    chunk_type: ChunkType
    page_number: int
    content: str
    image_path: str | None
    summary: str | None
    payload: dict[str, Any] = field(default_factory=dict)


class VectorStoreManager:
    """Wraps an async Qdrant client: creates the collection, indexes chunks and searches them.

    The collection uses named vectors: ``text`` always, plus ``clip`` when the embedder's image
    strategy is CLIP or BOTH. Points only carry the vectors that apply to them.

    Connection (first match wins): ``client`` > ``url`` (Docker/server) > ``path`` (local on-disk)
    > ``location`` (default ``":memory:"``).

    Example:
        >>> async with VectorStoreManager(embedder) as store:
        ...     await store.index_chunks(chunks)
        ...     hits = await store.search("revenue by quarter", top_k=5)
    """

    def __init__(
        self,
        embedder: MultimodalEmbedder,
        collection_name: str = DEFAULT_COLLECTION,
        *,
        location: str = ":memory:",
        url: str | None = None,
        path: str | None = None,
        api_key: str | None = None,
        client: AsyncQdrantClient | None = None,
    ) -> None:
        self.embedder = embedder
        self.collection_name = collection_name
        if client is not None:
            self.client = client
            target = "provided client"
        elif url is not None:
            self.client = AsyncQdrantClient(url=url, api_key=api_key)
            target = url
        elif path is not None:
            self.client = AsyncQdrantClient(path=path)
            target = f"local path {path}"
        else:
            self.client = AsyncQdrantClient(location=location)
            target = location
        self._is_server = client is None and url is not None
        self._ready = False
        logger.info("VectorStoreManager using Qdrant at %s, collection %r", target, collection_name)

    @classmethod
    def in_memory(cls, embedder: MultimodalEmbedder, collection_name: str = DEFAULT_COLLECTION) -> Self:
        return cls(embedder, collection_name, location=":memory:")

    @classmethod
    def from_url(
        cls,
        embedder: MultimodalEmbedder,
        url: str = "http://localhost:6333",
        collection_name: str = DEFAULT_COLLECTION,
        api_key: str | None = None,
    ) -> Self:
        return cls(embedder, collection_name, url=url, api_key=api_key)

    async def __aenter__(self) -> Self:
        await self.ensure_collection()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        await self.client.close()

    async def ensure_collection(self, recreate: bool = False) -> None:
        """Create the collection if needed and check its vector sizes match the embedder.

        Raises:
            ValueError: If an existing collection has different vector names or sizes
                (e.g. you switched embedding models); pass ``recreate=True`` to rebuild it.
        """
        dims = self.embedder.vector_dimensions()
        exists = await self.client.collection_exists(self.collection_name)

        if exists and recreate:
            await self.client.delete_collection(self.collection_name)
            logger.info("Dropped collection %r for recreation", self.collection_name)
            exists = False

        if exists:
            info = await self.client.get_collection(self.collection_name)
            current = info.config.params.vectors
            current_dims = (
                {name: params.size for name, params in current.items()} if isinstance(current, dict) else {}
            )
            if current_dims != dims:
                raise ValueError(
                    f"Collection {self.collection_name!r} has vectors {current_dims}, embedder needs {dims}. "
                    "Use ensure_collection(recreate=True) or another collection name."
                )
        else:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    name: models.VectorParams(size=size, distance=models.Distance.COSINE)
                    for name, size in dims.items()
                },
            )
            if self._is_server:  # payload indexes only take effect on a Qdrant server
                for field_name, schema in (
                    ("chunk_type", models.PayloadSchemaType.KEYWORD),
                    ("page_number", models.PayloadSchemaType.INTEGER),
                ):
                    await self.client.create_payload_index(self.collection_name, field_name, schema)
            logger.info("Created collection %r with vectors %s", self.collection_name, dims)
        self._ready = True

    async def index_chunks(self, chunks: Iterable[Chunk | object], batch_size: int = 64) -> int:
        """Embed chunks and upsert them with their payload metadata.

        Accepts ``Chunk`` objects, dicts or extractor objects (see ``Chunk.from_any``). Re-indexing the
        same chunk overwrites it, because point IDs are deterministic.

        Returns:
            Number of points upserted.
        """
        if not self._ready:
            await self.ensure_collection()

        normalised: list[Chunk] = []
        for raw in chunks:
            try:
                normalised.append(Chunk.from_any(raw))
            except ValueError:
                logger.warning("Skipping malformed chunk", exc_info=True)
        if not normalised:
            logger.warning("index_chunks called with no valid chunks")
            return 0

        start = time.perf_counter()
        embedded = await self.embedder.embed_chunks(normalised)
        points = [self._to_point(e) for e in embedded]

        for i in range(0, len(points), batch_size):
            await self.client.upsert(self.collection_name, points=points[i : i + batch_size], wait=True)
        logger.info(
            "Indexed %d points into %r in %.2fs",
            len(points),
            self.collection_name,
            time.perf_counter() - start,
        )
        return len(points)

    @staticmethod
    def _to_point(item: EmbeddedChunk) -> models.PointStruct:
        chunk = item.chunk
        payload: dict[str, Any] = {
            **{k: v for k, v in chunk.metadata.items() if _is_json_safe(v)},
            "page_number": chunk.page_number,
            "chunk_type": chunk.chunk_type.value,
            "image_path": chunk.image_path,
            "content": chunk.content,
            "summary": item.summary,
            "chunk_id": chunk.chunk_id,
        }
        vector: dict[str, Any] = dict(item.vectors)
        return models.PointStruct(id=chunk.point_id, vector=vector, payload=payload)

    async def search(
        self,
        query_text: str,
        top_k: int = 5,
        *,
        chunk_types: Sequence[ChunkType | str] | None = None,
        page_numbers: Sequence[int] | None = None,
    ) -> list[SearchResult]:
        """Retrieve the top-k chunks for a text query across all active vector spaces.

        With a single vector space, results are ordered by cosine similarity. With both ``text`` and
        ``clip`` spaces, each is searched separately and merged with Reciprocal Rank Fusion, because
        their raw scores are not comparable. Each result keeps its raw ``score``.

        Args:
            query_text: Natural-language query.
            top_k: Number of results to return.
            chunk_types: Optional filter, e.g. ``["table", "image"]``.
            page_numbers: Optional filter on page numbers.
        """
        if not query_text.strip():
            raise ValueError("query_text must not be empty")
        if not self._ready:
            await self.ensure_collection()

        start = time.perf_counter()
        query_filter = self._build_filter(chunk_types, page_numbers)
        query_vectors = await self.embedder.embed_query(query_text)

        best: dict[str, tuple[float, float, str, dict[str, Any]]] = {}
        for vector_name, vector in query_vectors.items():
            vec_filter = query_filter
            if vector_name == CLIP_VECTOR:
                # Only image points carry a clip vector; skip the space if images are filtered out.
                if chunk_types and ChunkType.IMAGE.value not in {ChunkType(t).value for t in chunk_types}:
                    continue
                vec_filter = self._build_filter(None, page_numbers, images_only=True)
            response = await self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                using=vector_name,
                limit=top_k,
                query_filter=vec_filter,
                with_payload=True,
            )
            for rank, point in enumerate(response.points, start=1):
                pid = str(point.id)
                rrf = 1.0 / (_RRF_K + rank)
                prev = best.get(pid)
                if prev is None:
                    best[pid] = (rrf, point.score, vector_name, dict(point.payload or {}))
                else:
                    # Point matched in both spaces: sum fusion scores, keep the stronger raw match.
                    keep_text = prev[2] == TEXT_VECTOR
                    best[pid] = (
                        prev[0] + rrf,
                        prev[1] if keep_text else point.score,
                        prev[2] if keep_text else vector_name,
                        prev[3],
                    )

        multi_space = len(query_vectors) > 1
        ordered = sorted(best.items(), key=lambda kv: kv[1][0] if multi_space else kv[1][1], reverse=True)

        results = [
            SearchResult(
                point_id=pid,
                score=raw,
                fused_score=fused,
                vector_name=name,
                chunk_type=ChunkType(payload.get("chunk_type", ChunkType.TEXT.value)),
                page_number=int(payload.get("page_number", 0)),
                content=str(payload.get("content") or ""),
                image_path=payload.get("image_path"),
                summary=payload.get("summary"),
                payload=payload,
            )
            for pid, (fused, raw, name, payload) in ordered[:top_k]
        ]
        logger.info(
            "Search %r returned %d results (spaces=%s) in %.2fs",
            query_text[:80],
            len(results),
            list(query_vectors),
            time.perf_counter() - start,
        )
        return results

    @staticmethod
    def _build_filter(
        chunk_types: Sequence[ChunkType | str] | None,
        page_numbers: Sequence[int] | None,
        images_only: bool = False,
    ) -> models.Filter | None:
        must: list[models.Condition] = []
        types = [ChunkType(t).value for t in chunk_types] if chunk_types else []
        if images_only:
            types = [t for t in types if t == ChunkType.IMAGE.value] or [ChunkType.IMAGE.value]
        if types:
            must.append(models.FieldCondition(key="chunk_type", match=models.MatchAny(any=types)))
        if page_numbers:
            must.append(
                models.FieldCondition(key="page_number", match=models.MatchAny(any=list(page_numbers)))
            )
        return models.Filter(must=must) if must else None

    async def count(self) -> int:
        result = await self.client.count(self.collection_name, exact=True)
        return int(result.count)


def _is_json_safe(value: Any) -> bool:
    return isinstance(value, str | int | float | bool | type(None) | list | dict | tuple)
