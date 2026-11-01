"""Offline tests for src.embedder and src.store using fake embedders and in-memory Qdrant."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from pathlib import Path

import pytest
from PIL import Image

from src.embedder import (
    CLIP_VECTOR,
    TEXT_VECTOR,
    ClipEmbedder,
    ImageStrategy,
    MultimodalEmbedder,
    VisionSummarizer,
)
from src.models import Chunk, ChunkType
from src.store import VectorStoreManager

DIM = 64


def _bow(text: str, dim: int = DIM) -> list[float]:
    vec = [0.0] * dim
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class FakeTextEmbedder:
    @property
    def dimension(self) -> int:
        return DIM

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [_bow(t) for t in texts]


class FakeSummarizer(VisionSummarizer):
    def __init__(self) -> None:
        super().__init__(model="fake/vision")

    async def summarize(self, image_path: str, context: str | None = None) -> str:
        if "broken" in image_path:
            raise RuntimeError("vision model unavailable")
        return f"bar chart showing quarterly revenue growth {Path(image_path).stem}"


class FakeClip(ClipEmbedder):
    """Maps an image's dominant colour (red/blue) and query words to a shared 8-dim space."""

    @property
    def dimension(self) -> int:
        return 8

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0 if "red" in t else 0.0, 1.0 if "blue" in t else 0.0, 0.1, 0, 0, 0, 0, 0] for t in texts]

    async def embed_images(self, image_paths: Sequence[str]) -> list[list[float]]:
        out = []
        for p in image_paths:
            r, _, b = Image.open(p).convert("RGB").getpixel((0, 0))  # type: ignore[misc]
            out.append([r / 255, b / 255, 0.1, 0, 0, 0, 0, 0])
        return out


@pytest.fixture()
def images(tmp_path: Path) -> dict[str, str]:
    paths = {}
    for name, colour in (("red", (255, 0, 0)), ("blue", (0, 0, 255))):
        path = tmp_path / f"p3_fig_{name}.png"
        Image.new("RGB", (16, 16), colour).save(path)
        paths[name] = str(path)
    return paths


def _chunks(images: dict[str, str]) -> list[object]:
    # Mixed input shapes, as an extractor might produce.
    return [
        {
            "type": "text",
            "text": "The transformer architecture uses self attention.",
            "page": 1,
            "doc_id": "d1",
        },
        Chunk(ChunkType.TEXT, "Qdrant stores vectors with payload metadata.", 2),
        {"chunk_type": "table", "content": "| quarter | revenue |\n|---|---|\n| Q1 | 10 |", "page_number": 2},
        {"chunk_type": "figure", "image_path": images["red"], "page_number": 3},
        {"chunk_type": "image", "image_path": images["blue"], "page_number": 4},
        {"chunk_type": "text", "content": "   ", "page_number": 5},  # empty: skipped
    ]


def test_chunk_from_any_aliases_and_stable_ids() -> None:
    a = Chunk.from_any({"type": "Figure", "page": "3", "path": "x.png", "metadata": {"bbox": [0, 0, 1, 1]}})
    assert a.chunk_type is ChunkType.IMAGE and a.page_number == 3 and a.image_path == "x.png"
    assert a.metadata["bbox"] == [0, 0, 1, 1]
    assert a.point_id == Chunk.from_any({"type": "image", "page": 3, "path": "x.png"}).point_id
    with pytest.raises(ValueError):
        Chunk.from_any({"type": "video", "page": 1})


def test_strategy_requires_components() -> None:
    with pytest.raises(ValueError):
        MultimodalEmbedder(FakeTextEmbedder(), ImageStrategy.CAPTION)
    with pytest.raises(ValueError):
        MultimodalEmbedder(FakeTextEmbedder(), ImageStrategy.CLIP)


async def test_caption_strategy_index_and_search(images: dict[str, str]) -> None:
    embedder = MultimodalEmbedder(FakeTextEmbedder(), ImageStrategy.CAPTION, summarizer=FakeSummarizer())
    async with VectorStoreManager.in_memory(embedder) as store:
        assert await store.index_chunks(_chunks(images)) == 5
        assert await store.count() == 5

        hits = await store.search("quarterly revenue chart", top_k=3)
        assert hits[0].chunk_type is ChunkType.IMAGE
        assert hits[0].summary and "revenue" in hits[0].summary
        assert hits[0].image_path in images.values()
        assert all(h.vector_name == TEXT_VECTOR for h in hits)
        assert hits == sorted(hits, key=lambda h: h.score, reverse=True)

        tables = await store.search("revenue", chunk_types=["table"])
        assert {h.chunk_type for h in tables} == {ChunkType.TABLE}
        assert tables[0].payload["page_number"] == 2

        # Re-indexing is idempotent (deterministic point IDs).
        await store.index_chunks(_chunks(images))
        assert await store.count() == 5


async def test_clip_strategy_text_to_image(images: dict[str, str]) -> None:
    embedder = MultimodalEmbedder(FakeTextEmbedder(), ImageStrategy.CLIP, clip_embedder=FakeClip())
    async with VectorStoreManager.in_memory(embedder) as store:
        info = await store.client.get_collection(store.collection_name)
        assert set(info.config.params.vectors) == {TEXT_VECTOR, CLIP_VECTOR}  # type: ignore[arg-type]

        assert await store.index_chunks(_chunks(images)) == 5
        hits = await store.search("a red figure", top_k=5)
        image_hits = [h for h in hits if h.chunk_type is ChunkType.IMAGE]
        assert image_hits[0].image_path == images["red"]
        assert image_hits[0].vector_name == CLIP_VECTOR

        text_only = await store.search("attention", chunk_types=["text"])
        assert all(h.chunk_type is ChunkType.TEXT for h in text_only)


async def test_failed_summary_is_skipped_not_fatal(tmp_path: Path) -> None:
    broken = tmp_path / "broken.png"
    Image.new("RGB", (4, 4)).save(broken)
    embedder = MultimodalEmbedder(FakeTextEmbedder(), ImageStrategy.CAPTION, summarizer=FakeSummarizer())
    async with VectorStoreManager.in_memory(embedder) as store:
        n = await store.index_chunks(
            [
                {"type": "image", "image_path": str(broken), "page": 1},
                {"type": "image", "image_path": str(tmp_path / "missing.png"), "page": 1},
                {"type": "text", "text": "still indexed", "page": 1},
            ]
        )
        assert n == 1


async def test_dimension_mismatch_is_detected(images: dict[str, str]) -> None:
    caption = MultimodalEmbedder(FakeTextEmbedder(), ImageStrategy.CAPTION, summarizer=FakeSummarizer())
    store = VectorStoreManager.in_memory(caption)
    await store.ensure_collection()
    clip = MultimodalEmbedder(FakeTextEmbedder(), ImageStrategy.CLIP, clip_embedder=FakeClip())
    other = VectorStoreManager(clip, client=store.client)
    with pytest.raises(ValueError, match="recreate"):
        await other.ensure_collection()
    await other.ensure_collection(recreate=True)
    await store.close()
