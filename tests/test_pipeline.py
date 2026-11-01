"""Pipeline-level tests: extraction units, mock retrieval, config, persistence and the app.py CLI.

No network or model downloads: embeddings and figure captions use fakes, and litellm.acompletion
is replaced with a stub that records the prompt it receives.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import litellm
import pymupdf
import pytest
from PIL import Image

import app
from src.config import Settings, load_settings
from src.embedder import ImageStrategy, MultimodalEmbedder
from src.extractor import ExtractorConfig, extract_pdf
from src.models import ChunkType
from src.pipeline import PipelineError, RAGPipeline, expand_pdf_paths
from src.retriever import Retriever
from tests.test_store import FakeSummarizer, FakeTextEmbedder

# ------------------------------------------------------------------------------------- fixtures


def _png(colour: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (320, 200), colour).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def report_pdf(tmp_path: Path) -> Path:
    """Two-page report: revenue text + quarterly table on page 1, a revenue chart on page 2."""
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_textbox(
        pymupdf.Rect(72, 60, 540, 160),
        "Revenue grew every quarter in FY2025. The Q3 revenue trend continued upward as summer output peaked.",
        fontsize=10,
    )
    rows = [["Quarter", "Revenue ($M)"], ["Q1", "42.1"], ["Q2", "47.8"], ["Q3", "55.3"], ["Q4", "61.0"]]
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            rect = pymupdf.Rect(72 + j * 150, 200 + i * 22, 72 + (j + 1) * 150, 200 + (i + 1) * 22)
            p1.draw_rect(rect, color=(0, 0, 0), width=0.8)
            p1.insert_text((rect.x0 + 5, rect.y1 - 7), cell, fontsize=10)
    p2 = doc.new_page()
    p2.insert_image(pymupdf.Rect(150, 100, 470, 300), stream=_png((40, 90, 200)))
    p2.insert_text((150, 320), "Figure 1: Quarterly revenue trend chart.", fontsize=9)
    path = tmp_path / "report.pdf"
    doc.save(path)
    return path


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "qdrant:\n"
        f"  path: {tmp_path / 'qdrant'}\n"
        "extraction:\n"
        f"  output_dir: {tmp_path / 'images'}\n"
        "llm:\n"
        "  model: gemini/gemini-test\n"
        "retrieval:\n"
        "  top_k: 4\n"
    )
    return load_settings(cfg, env_file=None)


def fake_embedder() -> MultimodalEmbedder:
    return MultimodalEmbedder(FakeTextEmbedder(), ImageStrategy.CAPTION, summarizer=FakeSummarizer())


@pytest.fixture()
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace litellm.acompletion; returns the list of recorded calls."""
    calls: list[dict[str, Any]] = []

    async def acompletion(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Revenue rose each quarter, reaching $55.3M in Q3 [S1]."),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1200, completion_tokens=30, total_tokens=1230),
        )

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    return calls


# ------------------------------------------------------------------------------ PDF extraction


def test_extraction_produces_all_three_chunk_types(report_pdf: Path, tmp_path: Path) -> None:
    chunks = extract_pdf(report_pdf, ExtractorConfig(output_dir=str(tmp_path / "img")))
    by_type = {t: [c for c in chunks if c.chunk_type is t] for t in ChunkType}

    (table,) = by_type[ChunkType.TABLE]
    assert table.page_number == 1 and "|Q3|55.3|" in table.content
    assert any("Q3 revenue trend" in c.content for c in by_type[ChunkType.TEXT])
    (figure,) = by_type[ChunkType.IMAGE]
    assert figure.page_number == 2 and figure.content == "Figure 1: Quarterly revenue trend chart."
    assert figure.image_path and Path(figure.image_path).is_file()
    assert all(c.metadata["source_path"] == str(report_pdf) for c in chunks)


def test_extraction_is_deterministic(report_pdf: Path, tmp_path: Path) -> None:
    cfg = ExtractorConfig(output_dir=str(tmp_path / "img"))
    assert [c.point_id for c in extract_pdf(report_pdf, cfg)] == [
        c.point_id for c in extract_pdf(report_pdf, cfg)
    ]


def test_expand_pdf_paths(report_pdf: Path, tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "B.PDF").write_bytes(report_pdf.read_bytes())
    (tmp_path / "notes.txt").write_text("x")
    found = expand_pdf_paths([tmp_path, report_pdf, tmp_path / "missing.pdf"])
    assert {p.name for p in found} == {"report.pdf", "B.PDF"} and len(found) == 2


# -------------------------------------------------------------------------------- mock retrieval


async def test_mock_retrieval_returns_table_text_and_figure(report_pdf: Path, settings: Settings) -> None:
    async with RAGPipeline.from_settings(settings, embedder=fake_embedder()) as rag:
        (report,) = await rag.ingest([report_pdf])
        assert report.ok and report.table_chunks == 1 and report.image_chunks == 1
        ctx = await rag.retriever.retrieve("Q3 revenue trend chart")

    types = {i.chunk_type for i in ctx.items}
    assert types == {ChunkType.TEXT, ChunkType.TABLE, ChunkType.IMAGE}
    assert ctx.image_paths and Path(ctx.image_paths[0]).name == "p2_fig1.png"
    assert "Figure 1: Quarterly revenue trend chart." in ctx.image_items[0].content
    assert all(i.citation.startswith("report.pdf, page ") for i in ctx.items)


async def test_retriever_on_stub_store_separates_images() -> None:
    """Retriever logic in isolation, with a stub in place of Qdrant."""
    from src.store import SearchResult

    img = Path(__file__)  # any existing file stands in for a PNG
    hits = [
        SearchResult("a", 0.9, 0.03, "text", ChunkType.TABLE, 1, "|Q3|55.3|", None, None, {"doc_id": "d"}),
        SearchResult(
            "b", 0.8, 0.02, "text", ChunkType.IMAGE, 2, "Figure 1", str(img), "chart", {"doc_id": "d"}
        ),
        SearchResult("c", 0.7, 0.01, "text", ChunkType.TEXT, 1, "Revenue grew.", None, None, {"doc_id": "d"}),
    ]

    class StubStore:
        async def search(self, query: str, top_k: int = 5, **_: Any) -> list[SearchResult]:
            return hits[:top_k]

    ctx = await Retriever(StubStore(), top_k=3).retrieve("q")  # type: ignore[arg-type]
    assert [i.label for i in ctx.items] == ["S1", "S2", "S3"]
    assert [i.chunk_type for i in ctx.text_items] == [ChunkType.TABLE, ChunkType.TEXT]
    assert ctx.image_paths == [str(img)] and ctx.image_items[0].content == "Figure 1\nchart"


# --------------------------------------------------------------------- pipeline + persistence


async def test_query_after_reopening_index(
    report_pdf: Path, settings: Settings, fake_llm: list[dict[str, Any]]
) -> None:
    async with RAGPipeline.from_settings(settings, embedder=fake_embedder()) as rag:
        await rag.ingest([report_pdf])

    # A fresh pipeline (a new CLI run) reads the same on-disk index.
    async with RAGPipeline.from_settings(settings, embedder=fake_embedder()) as rag:
        result = await rag.query("What is the Q3 revenue trend shown in the chart?")

    assert "$55.3M" in result.answer and result.generation.citations
    assert result.generation.usage.total_tokens == 1230 and result.generation.images_sent == 1
    parts = fake_llm[0]["messages"][1]["content"]
    assert any(p["type"] == "image_url" for p in parts)
    assert fake_llm[0]["model"] == "gemini/gemini-test"


async def test_reset_and_errors(report_pdf: Path, settings: Settings, fake_llm: list[dict[str, Any]]) -> None:
    async with RAGPipeline.from_settings(settings, embedder=fake_embedder()) as rag:
        with pytest.raises(PipelineError, match="index is empty"):
            await rag.query("anything")
        await rag.ingest([report_pdf])
        before = await rag.store.count()
        await rag.ingest([report_pdf])  # re-ingest overwrites, no duplicates
        assert await rag.store.count() == before
        await rag.ingest([report_pdf], reset=True)
        assert await rag.store.count() == before
        with pytest.raises(PipelineError, match="No PDF files"):
            await rag.ingest([report_pdf.parent / "nope.pdf"])
        bad = report_pdf.parent / "bad.pdf"
        bad.write_text("not a pdf")
        reports = await rag.ingest([bad, report_pdf])
        assert [r.ok for r in reports] == [False, True]

    # Different embedding size than the stored index -> clear error suggesting --reset.
    other = MultimodalEmbedder(_SmallEmbedder(), ImageStrategy.CAPTION, summarizer=FakeSummarizer())
    async with RAGPipeline.from_settings(settings, embedder=other) as rag:
        with pytest.raises(PipelineError, match="--reset"):
            await rag.query("q")


class _SmallEmbedder(FakeTextEmbedder):
    @property
    def dimension(self) -> int:
        return 16


# ------------------------------------------------------------------------------------- config


def test_config_yaml_env_and_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("llm:\n  model: gemini/a\nretrieval:\n  top_k: 3\n")
    assert load_settings(cfg, env_file=None).llm.model == "gemini/a"

    monkeypatch.setenv("MMRAG_LLM__MODEL", "ollama/llama3.2-vision")
    monkeypatch.setenv("MMRAG_RETRIEVAL__MAX_IMAGES", "1")
    s = load_settings(cfg, env_file=None)
    assert s.llm.model == "ollama/llama3.2-vision" and s.retrieval.max_images == 1 and s.retrieval.top_k == 3

    env = tmp_path / ".env"
    env.write_text("MMRAG_LOG_LEVEL=DEBUG\nSOME_PROVIDER_API_KEY=abc\n")
    monkeypatch.delenv("SOME_PROVIDER_API_KEY", raising=False)
    monkeypatch.delenv("MMRAG_LOG_LEVEL", raising=False)
    try:
        assert load_settings(cfg, env_file=env).log_level == "DEBUG"
        assert os.environ["SOME_PROVIDER_API_KEY"] == "abc"  # exported for litellm
    finally:  # load_dotenv writes os.environ directly; undo so other tests are unaffected
        os.environ.pop("SOME_PROVIDER_API_KEY", None)
        os.environ.pop("MMRAG_LOG_LEVEL", None)

    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("embedding:\n  text:\n    provider: litellm\n    dimension: null\n")
    with pytest.raises(ValueError, match="dimension"):
        load_settings(bad, env_file=None)


def test_repo_config_yaml_is_valid() -> None:
    s = load_settings(Path(__file__).parent.parent / "config.yaml", env_file=None)
    assert s.llm.model and s.embedding.text.dimension


# ------------------------------------------------------------------------------------ app.py CLI


@pytest.fixture()
def cli(settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    """Point app.py at the test config, inject fakes, and pretend API keys are present."""
    cfg = tmp_path / "config.yaml"
    monkeypatch.setattr(
        app, "build_pipeline", lambda s: RAGPipeline.from_settings(s, embedder=fake_embedder())
    )
    monkeypatch.setattr(app, "missing_api_keys", lambda models: {})
    monkeypatch.setattr(app, "setup_logging", lambda level: None)  # keep pytest's log capture intact
    return str(cfg)


def test_cli_ingest_then_query(
    cli: str, report_pdf: Path, fake_llm: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    assert app.main(["--config", cli, "--ingest", str(report_pdf)]) == 0
    out = capsys.readouterr().out
    assert "Indexed" in out and "1 table, 1 figure chunks" in out

    code = app.main(["--config", cli, "--query", "What is the Q3 revenue trend shown in the chart?"])
    out = capsys.readouterr().out
    assert code == 0
    assert "$55.3M in Q3 [S1]" in out and "Sources:" in out and "report.pdf, page" in out
    assert "Figures shown to the model (1)" in out and "p2_fig1.png" in out
    assert "tokens 1200 in / 30 out" in out and "gemini/gemini-test" in out


def test_cli_json_and_model_override(
    cli: str, report_pdf: Path, fake_llm: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    code = app.main(
        [
            "--config",
            cli,
            "--ingest",
            str(report_pdf),
            "--query",
            "Q3 revenue?",
            "--model",
            "ollama/x",
            "--json",
        ]
    )
    data = json.loads(capsys.readouterr().out)
    assert code == 0 and fake_llm[0]["model"] == "ollama/x"
    assert data["ingested"][0]["table_chunks"] == 1
    q = data["query"]
    assert q["usage"]["total_tokens"] == 1230 and q["citations"][0]["label"] == "S1"
    assert set(q["latency_ms"]) == {"retrieval", "llm", "generation_total", "end_to_end"}


def test_cli_errors(
    cli: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    with pytest.raises(SystemExit) as exc:
        app.main([])
    assert exc.value.code == 2
    with pytest.raises(SystemExit):
        app.main(["--reset", "--query", "x"])

    assert app.main(["--config", cli, "--query", "anything"]) == 1
    assert "index is empty" in capsys.readouterr().err

    assert app.main(["--config", str(tmp_path / "nope.yaml"), "--query", "x"]) == 2
    assert "Configuration error" in capsys.readouterr().err

    monkeypatch.setattr(app, "missing_api_keys", lambda models: {"gemini/gemini-test": ["GEMINI_API_KEY"]})
    assert app.main(["--config", cli, "--query", "x"]) == 2
    err = capsys.readouterr().err
    assert "GEMINI_API_KEY" in err and ".env" in err
