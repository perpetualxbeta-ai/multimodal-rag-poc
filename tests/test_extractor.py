"""Tests for src.extractor using a PDF generated on the fly (no fixtures or downloads needed)."""

from __future__ import annotations

import io
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from src.embedder import ImageStrategy, MultimodalEmbedder
from src.extractor import (
    ExtractorConfig,
    chunk_text,
    extract_many,
    extract_pdf,
    html_table_to_markdown,
)
from src.models import ChunkType
from src.store import VectorStoreManager
from tests.test_store import FakeSummarizer, FakeTextEmbedder

LOREM = (
    "Retrieval augmented generation combines a retriever with a language model. "
    "The retriever finds relevant passages and the model writes an answer grounded in them. "
)


def _png(colour: tuple[int, int, int], size: tuple[int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def sample_pdf(tmp_path: Path) -> Path:
    doc = pymupdf.open()

    # Page 1: text, a raster photo with a caption below, and a tiny logo that must be ignored.
    p1 = doc.new_page()
    p1.insert_textbox(pymupdf.Rect(72, 60, 540, 260), "Introduction\n\n" + LOREM * 6, fontsize=10)
    p1.insert_image(pymupdf.Rect(150, 300, 450, 500), stream=_png((200, 30, 30), (300, 200)))
    p1.insert_text((150, 520), "Figure 1: Sales by region in 2025.", fontsize=9)
    p1.insert_image(pymupdf.Rect(520, 20, 545, 45), stream=_png((0, 0, 0), (25, 25)))

    # Page 2: a ruled table plus a paragraph.
    p2 = doc.new_page()
    p2.insert_text((72, 80), "Quarterly results are summarised below.", fontsize=10)
    rows = [
        ["Quarter", "Revenue", "Margin"],
        ["Q1", "10.2", "31%"],
        ["Q2", "14.8", "34%"],
        ["Q3", "17.1", "36%"],
    ]
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            rect = pymupdf.Rect(72 + j * 110, 110 + i * 22, 72 + (j + 1) * 110, 110 + (i + 1) * 22)
            p2.draw_rect(rect, color=(0, 0, 0), width=0.8)
            p2.insert_text((rect.x0 + 5, rect.y1 - 7), cell, fontsize=10)
    p2.insert_textbox(pymupdf.Rect(72, 230, 540, 330), "Margins improved every quarter. " * 5, fontsize=10)

    # Page 3: a vector bar chart with a caption above.
    p3 = doc.new_page()
    p3.insert_text((100, 90), "Figure 2: Quarterly revenue growth.", fontsize=9)
    for k, height in enumerate([60, 90, 120, 150]):
        p3.draw_rect(
            pymupdf.Rect(130 + k * 50, 280 - height, 160 + k * 50, 280), color=None, fill=(0.2, 0.4, 0.8)
        )
    p3.draw_line((115, 280), (340, 280))
    p3.draw_line((115, 110), (115, 280))

    # Page 4: blank (should be skipped gracefully).
    doc.new_page()

    path = tmp_path / "sample report.pdf"
    doc.save(path)
    doc.close()
    return path


def test_extracts_text_tables_and_figures(sample_pdf: Path, tmp_path: Path) -> None:
    cfg = ExtractorConfig(output_dir=str(tmp_path / "images"), chunk_size=400, chunk_overlap=80)
    chunks = extract_pdf(sample_pdf, cfg)
    by_type = {t: [c for c in chunks if c.chunk_type is t] for t in ChunkType}

    # Text: chunked, page 1 split into several overlapping pieces, provenance kept.
    p1_text = [c for c in by_type[ChunkType.TEXT] if c.page_number == 1]
    assert len(p1_text) >= 2 and all(len(c.content) <= 400 for c in p1_text)
    assert all(c.metadata["doc_id"].startswith("sample_report-") for c in chunks)

    # Table: Markdown, and its cell text is not duplicated into page 2's text chunks.
    (table,) = by_type[ChunkType.TABLE]
    assert table.page_number == 2
    assert table.content.splitlines()[0].replace(" ", "") == "|Quarter|Revenue|Margin|"
    assert "|Q3|17.1|36%|" in table.content.replace(" ", "")
    p2_text = " ".join(c.content for c in by_type[ChunkType.TEXT] if c.page_number == 2)
    assert "Margins improved" in p2_text and "17.1" not in p2_text

    # Figures: raster photo (logo filtered) and vector chart, rendered to PNG with captions.
    figs = sorted(by_type[ChunkType.IMAGE], key=lambda c: c.page_number)
    assert [f.page_number for f in figs] == [1, 3]
    assert figs[0].content.startswith("Figure 1") and figs[1].content.startswith("Figure 2")
    for fig in figs:
        assert fig.image_path and Path(fig.image_path).is_file()
        with Image.open(fig.image_path) as img:
            assert img.width > 200 and img.height > 150
    red = Image.open(figs[0].image_path).convert("RGB").getpixel((img.width // 2, img.height // 2))
    assert red[0] > 150 and red[2] < 100  # type: ignore[index]

    # Blank page produces nothing; chunk IDs are unique and stable across runs.
    assert not any(c.page_number == 4 for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert [c.point_id for c in extract_pdf(sample_pdf, cfg)] == [c.point_id for c in chunks]


def test_toggles(sample_pdf: Path, tmp_path: Path) -> None:
    cfg = ExtractorConfig(output_dir=str(tmp_path), extract_tables=False, include_vector_figures=False)
    chunks = extract_pdf(sample_pdf, cfg)
    assert not [c for c in chunks if c.chunk_type is ChunkType.TABLE]
    assert [c.page_number for c in chunks if c.chunk_type is ChunkType.IMAGE] == [1]


def test_bad_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_pdf(tmp_path / "missing.pdf")
    bad = tmp_path / "bad.pdf"
    bad.write_text("not a pdf")
    with pytest.raises(ValueError):
        extract_pdf(bad)
    with pytest.raises(ValueError):
        ExtractorConfig(chunk_size=100, chunk_overlap=100)


def test_chunk_text_overlap_and_limits() -> None:
    text = ("Alpha beta gamma delta. " * 40) + "\n\n" + ("Second paragraph here. " * 40)
    pieces = chunk_text(text, chunk_size=300, chunk_overlap=60)
    assert len(pieces) > 3 and all(len(p) <= 300 for p in pieces)
    assert pieces[1].split()[0] in pieces[0]  # overlap carried over
    assert chunk_text("   ", 100, 10) == []
    assert chunk_text("short", 100, 10) == ["short"]


def test_html_table_to_markdown() -> None:
    html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>x | y</td></tr><tr><td>2</td></tr></table>"
    assert html_table_to_markdown(html) == "|A|B|\n|---|---|\n|1|x \\| y|\n|2||"
    assert html_table_to_markdown("<p>none</p>") == ""


async def test_extract_then_index_and_search(sample_pdf: Path, tmp_path: Path) -> None:
    results = await extract_many(
        [sample_pdf, tmp_path / "missing.pdf"], ExtractorConfig(output_dir=str(tmp_path))
    )
    chunks = results[str(sample_pdf)]
    assert results[str(tmp_path / "missing.pdf")] == []

    embedder = MultimodalEmbedder(FakeTextEmbedder(), ImageStrategy.CAPTION, summarizer=FakeSummarizer())
    async with VectorStoreManager.in_memory(embedder) as store:
        assert await store.index_chunks(chunks) == len(chunks)
        hits = await store.search("quarter revenue margin", top_k=3, chunk_types=["table"])
        assert hits[0].page_number == 2 and "Q3" in hits[0].content
        assert hits[0].payload["doc_id"] == chunks[0].metadata["doc_id"]


def test_vector_labels_wide_diagrams_captions_and_header_spacing(tmp_path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    # Body text that starts with "Figure" sits just above the chart; the real caption is further below.
    page.insert_text((80, 95), "Figure 1 compares capacity by region across the portfolio.", fontsize=10)
    for k, height in enumerate([60, 90, 120]):
        page.draw_rect(
            pymupdf.Rect(130 + k * 60, 260 - height, 170 + k * 60, 260), color=None, fill=(0.9, 0.5, 0.1)
        )
    page.draw_line((115, 260), (330, 260))
    for k, label in enumerate(["North", "Coast", "South"]):
        page.insert_text((132 + k * 60, 275), label, fontsize=9)  # axis labels drawn as text
    page.insert_text((150, 335), "Figure 1: Capacity by region.", fontsize=9)
    # A wide, short flow diagram (40pt tall) must not be filtered as an icon.
    for k in range(4):
        page.draw_rect(
            pymupdf.Rect(80 + k * 110, 400, 170 + k * 110, 440), color=(0, 0, 0), fill=(0.3, 0.6, 0.9)
        )
    page.insert_text((80, 460), "Figure 2: Data flow.", fontsize=9)
    # Table with a bold header whose cell text contains spaces.
    for i, row in enumerate([["Site name", "Peak output (MW)"], ["Kestrel Bay", "310"]]):
        for j, cell in enumerate(row):
            rect = pymupdf.Rect(80 + j * 150, 520 + i * 22, 80 + (j + 1) * 150, 520 + (i + 1) * 22)
            page.draw_rect(rect, color=(0, 0, 0), width=0.8)
            page.insert_text((rect.x0 + 5, rect.y1 - 7), cell, fontsize=10, fontname="helv" if i else "hebo")
    path = tmp_path / "edge.pdf"
    doc.save(path)

    chunks = extract_pdf(path, ExtractorConfig(output_dir=str(tmp_path / "img")))
    figs = [c for c in chunks if c.chunk_type is ChunkType.IMAGE]
    assert [f.content for f in figs] == ["Figure 1: Capacity by region.", "Figure 2: Data flow."]
    chart_bottom = figs[0].metadata["bbox"][3]
    assert chart_bottom >= 272  # grew to include the "North/Coast/South" labels below the axis
    text = " ".join(c.content for c in chunks if c.chunk_type is ChunkType.TEXT)
    assert "North" not in text and "compares capacity" in text
    (table,) = [c for c in chunks if c.chunk_type is ChunkType.TABLE]
    assert table.content.splitlines()[0] == "|Site name|Peak output (MW)|"
