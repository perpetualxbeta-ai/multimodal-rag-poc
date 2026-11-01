"""Tests for src.retriever and src.generator (no network: litellm is faked or uses mock_response)."""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import litellm
import pytest
from PIL import Image

from src import generator as gen_module
from src.embedder import ImageStrategy, MultimodalEmbedder
from src.generator import NO_CONTEXT_ANSWER, GenerationError, MultimodalGenerator, cited_items, encode_image
from src.models import Chunk, ChunkType
from src.retriever import KeywordReranker, RetrievedContext, Retriever, renumber
from src.store import VectorStoreManager
from tests.test_store import FakeSummarizer, FakeTextEmbedder

TABLE = "|Quarter|Revenue ($M)|Margin|\n|---|---|---|\n|Q1|42.1|27.6%|\n|Q3|55.3|32.0%|"


@pytest.fixture()
def chart(tmp_path: Path) -> str:
    path = tmp_path / "p3_fig1.png"
    Image.new("RGB", (2400, 1200), (224, 122, 31)).save(path)  # large: must be downscaled
    return str(path)


def _chunks(chart: str, tmp_path: Path) -> list[Chunk]:
    meta = {"doc_id": "halcyon-1", "source_path": "/docs/halcyon.pdf"}
    para = "Revenue grew in every quarter and operating margin expanded through the year."
    return [
        Chunk(ChunkType.TEXT, para, 1, chunk_id="t1", metadata=meta),
        Chunk(ChunkType.TEXT, para + " Margin", 1, chunk_id="t1-dup", metadata=meta),  # near duplicate
        Chunk(ChunkType.TABLE, TABLE, 2, chunk_id="tab1", metadata=meta),
        Chunk(
            ChunkType.TEXT,
            "Safety performance remained strong across all sites.",
            2,
            chunk_id="t2",
            metadata=meta,
        ),
        Chunk(
            ChunkType.IMAGE,
            "Figure 2: Revenue by quarter.",
            3,
            image_path=chart,
            chunk_id="f1",
            metadata=meta,
        ),
        Chunk(
            ChunkType.IMAGE,
            "Figure 9: Gone.",
            4,
            image_path=str(tmp_path / "x.png"),
            chunk_id="f2",
            metadata=meta,
        ),
    ]


async def _store(chart: str, tmp_path: Path) -> VectorStoreManager:
    embedder = MultimodalEmbedder(FakeTextEmbedder(), ImageStrategy.CAPTION, summarizer=FakeSummarizer())
    store = VectorStoreManager.in_memory(embedder)
    chunks = _chunks(chart, tmp_path)
    await store.index_chunks(chunks)
    Path(chunks[-1].image_path or "").unlink(missing_ok=True)  # image deleted after indexing
    return store


def _ok_response(text: str, prompt: int = 900, completion: int = 40) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
        usage=SimpleNamespace(
            prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
        ),
    )


# ---------------------------------------------------------------------------------- retriever


async def test_retriever_splits_text_and_images(chart: str, tmp_path: Path) -> None:
    Image.new("RGB", (8, 8)).save(tmp_path / "x.png")
    store = await _store(chart, tmp_path)
    retriever = Retriever(store, top_k=6, max_images=2)
    ctx = await retriever.retrieve("quarterly revenue chart margin")

    assert ctx.image_paths == [chart]  # missing image skipped
    assert all(not i.is_image for i in ctx.text_items)
    contents = [i.content for i in ctx.text_items]
    assert sum("Revenue grew" in c for c in contents) == 1  # near-duplicate removed
    assert [i.label for i in ctx.items] == [f"S{n}" for n in range(1, len(ctx.items) + 1)]
    img = ctx.image_items[0]
    assert img.content.startswith("Figure 2") and "bar chart" in img.content  # caption + vision summary
    assert img.citation == "halcyon.pdf, page 3 (image)"
    assert "[S" in ctx.text_block() and "halcyon.pdf, page 2 (table)" in ctx.text_block()
    assert ctx.latency_ms > 0
    await store.close()


async def test_retriever_limits_and_filters(chart: str, tmp_path: Path) -> None:
    store = await _store(chart, tmp_path)
    no_images = await Retriever(store, top_k=6, max_images=0).retrieve("revenue chart")
    assert not no_images.image_items
    tables = await Retriever(store).retrieve("revenue", chunk_types=["table"])
    assert [i.chunk_type for i in tables.items] == [ChunkType.TABLE]
    tight = await Retriever(store, top_k=6, max_context_chars=100, max_images=0).retrieve("revenue margin")
    assert sum(len(i.content) for i in tight.text_items[1:]) <= 100
    await store.close()


async def test_keyword_reranker_promotes_exact_terms(chart: str, tmp_path: Path) -> None:
    store = await _store(chart, tmp_path)
    plain = await Retriever(store, top_k=2, max_images=0).retrieve("safety Q3 32.0%")
    reranked = await Retriever(store, top_k=2, max_images=0, reranker=KeywordReranker(0.9)).retrieve(
        "safety Q3 32.0%"
    )
    assert reranked.text_items[0].chunk_type is ChunkType.TABLE
    assert plain.text_items  # baseline still works
    with pytest.raises(ValueError):
        KeywordReranker(1.5)
    await store.close()


def test_renumber_relabels_after_filtering(chart: str) -> None:
    ctx = RetrievedContext("q")
    assert renumber(ctx).items == []


# ---------------------------------------------------------------------------------- generator


async def _context(chart: str, tmp_path: Path) -> RetrievedContext:
    store = await _store(chart, tmp_path)
    ctx = await Retriever(store, top_k=4).retrieve("Q3 revenue chart margin")
    await store.close()
    return ctx


async def test_generator_sends_text_and_images(
    chart: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = await _context(chart, tmp_path)
    calls: list[dict[str, Any]] = []
    table = next(i for i in ctx.text_items if i.chunk_type is ChunkType.TABLE)
    image = ctx.image_items[0]

    async def fake(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return _ok_response(f"Q3 margin was 32.0% [{table.label}], shown in the chart [{image.label}, S99].")

    monkeypatch.setattr(litellm, "acompletion", fake)
    result = await MultimodalGenerator("gemini/gemini-test-model").generate("What was the Q3 margin?", ctx)

    messages = calls[0]["messages"]
    assert messages[0]["role"] == "system" and "ONLY the sources" in messages[0]["content"]
    parts = messages[1]["content"]
    images = [p for p in parts if p["type"] == "image_url"]
    assert len(images) == 1 and images[0]["image_url"]["url"].startswith("data:image/png;base64,")
    texts = " ".join(p["text"] for p in parts if p["type"] == "text")
    assert f"[{image.label}] {image.citation}" in texts and "Figure 2: Revenue by quarter." in texts
    assert "|Q3|55.3|32.0%|" in texts and texts.endswith("Question: What was the Q3 margin?")

    assert [c.label for c in result.citations] == [table.label, image.label]  # unknown S99 ignored
    assert result.usage.prompt_tokens == 900 and result.usage.total_tokens == 940
    assert result.images_sent == 1 and result.attempts == 1 and result.finish_reason == "stop"
    assert result.llm_latency_ms >= 0 and result.total_latency_ms >= result.llm_latency_ms
    assert result.end_to_end_latency_ms >= result.total_latency_ms
    assert table.citation in result.sources_text()


async def test_generator_retries_and_falls_back(
    chart: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = await _context(chart, tmp_path)
    seen: list[str] = []

    async def fake(**kwargs: Any) -> SimpleNamespace:
        seen.append(kwargs["model"])
        if kwargs["model"] == "primary":
            raise RuntimeError("429 rate limit exceeded")
        return _ok_response("From fallback [S1].", prompt=500)

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(litellm, "acompletion", fake)
    monkeypatch.setattr(gen_module.asyncio, "sleep", no_sleep)
    result = await MultimodalGenerator("primary", fallback_models=["backup"], max_retries=2).generate(
        "q", ctx
    )
    assert seen == ["primary", "primary", "backup"]
    assert result.model == "backup" and result.attempts == 3 and result.usage.prompt_tokens == 500

    async def always_fail(**kwargs: Any) -> SimpleNamespace:
        raise ValueError("invalid api key")

    monkeypatch.setattr(litellm, "acompletion", always_fail)
    with pytest.raises(GenerationError, match="invalid api key"):
        await MultimodalGenerator("primary", max_retries=3).generate("q", ctx)


async def test_text_only_model_gets_captions(
    chart: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = await _context(chart, tmp_path)
    captured: list[Any] = []

    async def fake(**kwargs: Any) -> SimpleNamespace:
        captured.append(kwargs["messages"])
        return _ok_response("ok")

    monkeypatch.setattr(litellm, "acompletion", fake)
    monkeypatch.setattr(gen_module, "_supports_vision", lambda model: False)
    result = await MultimodalGenerator("text-only").generate("q", ctx)
    parts = captured[0][1]["content"]
    assert not [p for p in parts if p["type"] == "image_url"] and result.images_sent == 0
    assert any("(image not shown)" in p.get("text", "") for p in parts)


async def test_empty_context_skips_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(**kwargs: Any) -> None:
        raise AssertionError("LLM should not be called")

    monkeypatch.setattr(litellm, "acompletion", boom)
    result = await MultimodalGenerator("m").generate("anything?", RetrievedContext("anything?"))
    assert result.answer == NO_CONTEXT_ANSWER and result.attempts == 0


async def test_real_litellm_response_shape(
    chart: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Uses litellm's built-in mock_response: real ModelResponse objects, no network."""
    ctx = await _context(chart, tmp_path)
    original = litellm.acompletion

    async def mocked(**kwargs: Any) -> Any:
        return await original(**kwargs, mock_response="Margin was 32.0% [S1].")

    monkeypatch.setattr(litellm, "acompletion", mocked)
    result = await MultimodalGenerator("gemini/gemini-2.5-flash").generate("Q3 margin?", ctx)
    assert result.answer == "Margin was 32.0% [S1]." and result.citations[0].label == "S1"
    assert result.usage.total_tokens > 0


def test_encode_image_downscales(chart: str) -> None:
    url = encode_image(chart, max_side=800)
    header, data = url.split(",", 1)
    assert header == "data:image/png;base64"
    import io

    with Image.open(io.BytesIO(base64.b64decode(data))) as img:
        assert max(img.size) == 800


def test_cited_items_parses_groups() -> None:
    ctx = RetrievedContext("q")
    assert cited_items("no sources [S1]", ctx) == []


def test_vision_detection() -> None:
    assert gen_module._supports_vision("gemini/gemini-2.5-flash") is True
    assert gen_module._supports_vision("some-provider/brand-new-model") is True  # unknown: assume yes
    assert MultimodalGenerator("m", send_images=False).send_images is False
