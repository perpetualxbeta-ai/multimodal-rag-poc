"""PDF extraction: text chunks, Markdown tables and rendered figure images.

Uses pymupdf for everything by default (no system packages needed):
  * Tables: ``page.find_tables()`` -> ``Table.to_markdown()``.
  * Figures: embedded raster images (``page.get_image_info``) and vector graphics such as
    charts (``page.cluster_drawings``), rendered to PNG with ``page.get_pixmap(clip=...)``.
    A nearby "Figure N ..." caption is attached as the chunk's ``content``.
  * Text: page text blocks outside tables and figures, split into overlapping chunks.

``table_backend="unstructured"`` swaps table detection for ``unstructured``'s hi-res layout model
(needs ``unstructured[pdf]`` plus poppler/tesseract installed).

Output is a list of ``Chunk`` objects that ``VectorStoreManager.index_chunks`` accepts directly.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .models import Chunk, ChunkType

if TYPE_CHECKING:
    import pymupdf

logger = logging.getLogger(__name__)

BBox = tuple[float, float, float, float]

_CAPTION_RE = re.compile(r"^\s*(fig(?:ure)?\.?|chart|graph|diagram|exhibit)\s*\d*", re.IGNORECASE)


@dataclass(slots=True, frozen=True)
class ExtractorConfig:
    """Extraction settings.

    Attributes:
        output_dir: Root folder for rendered figures; images go to ``{output_dir}/{doc_id}/``.
        chunk_size: Target characters per text chunk.
        chunk_overlap: Characters of overlap between consecutive chunks on a page.
        image_dpi: Resolution for rendered figures.
        min_figure_size: Minimum width and height (PDF points, 72 per inch) for a figure.
            Filters out logos, icons and decorative rules.
        figure_padding: Points added around a figure's bounding box when rendering.
        extract_tables: Detect tables and emit them as Markdown chunks.
        extract_figures: Detect figures and render them to PNG.
        include_vector_figures: Also treat clusters of vector drawings (e.g. charts) as figures.
        table_backend: ``"pymupdf"`` (default) or ``"unstructured"``.
    """

    output_dir: str = "data/processed/images"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    image_dpi: int = 150
    min_figure_size: float = 60.0
    figure_padding: float = 4.0
    extract_tables: bool = True
    extract_figures: bool = True
    include_vector_figures: bool = True
    table_backend: Literal["pymupdf", "unstructured"] = "pymupdf"

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")


@dataclass(slots=True, frozen=True)
class _Table:
    bbox: BBox | None
    markdown: str


def make_doc_id(pdf_path: Path) -> str:
    """Stable document ID: file stem plus a short hash of the file contents."""
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:10]
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", pdf_path.stem).strip("_") or "doc"
    return f"{stem}-{digest}"


def extract_pdf(pdf_path: str | Path, config: ExtractorConfig | None = None) -> list[Chunk]:
    """Extract text, table and image chunks from a PDF.

    A failure on one page is logged and that page is skipped; the rest of the document still
    gets extracted.

    Args:
        pdf_path: Path to the PDF.
        config: Extraction settings (defaults to ``ExtractorConfig()``).

    Returns:
        Chunks in page order: per page, tables first, then figures, then text.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If the file cannot be opened as a PDF or is encrypted.
    """
    import pymupdf

    cfg = config or ExtractorConfig()
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    start = time.perf_counter()
    doc_id = make_doc_id(path)
    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        raise ValueError(f"Could not open {path} as a PDF: {exc}") from exc

    with doc:
        if doc.needs_pass:
            raise ValueError(f"{path} is password-protected")
        logger.info("Extracting %s (doc_id=%s, %d pages)", path.name, doc_id, doc.page_count)

        unstructured_tables: dict[int, list[_Table]] = {}
        if cfg.extract_tables and cfg.table_backend == "unstructured":
            sizes = {pg.number + 1: (float(pg.rect.width), float(pg.rect.height)) for pg in doc}
            unstructured_tables = _unstructured_tables(path, sizes)

        image_dir = Path(cfg.output_dir) / doc_id
        base_meta: dict[str, Any] = {"doc_id": doc_id, "source_path": str(path)}
        chunks: list[Chunk] = []
        for page in doc:
            page_number = page.number + 1
            try:
                if cfg.table_backend == "unstructured":
                    tables = unstructured_tables.get(page_number, [])
                elif cfg.extract_tables:
                    tables = _pymupdf_tables(page)
                else:
                    tables = []
                chunks.extend(_page_chunks(page, page_number, tables, cfg, image_dir, base_meta))
            except Exception:
                logger.warning(
                    "Failed to extract page %d of %s; skipping", page_number, path.name, exc_info=True
                )

    counts = {t: sum(c.chunk_type is t for c in chunks) for t in ChunkType}
    logger.info(
        "Extracted %s: %d text, %d table, %d image chunks in %.2fs",
        path.name,
        counts[ChunkType.TEXT],
        counts[ChunkType.TABLE],
        counts[ChunkType.IMAGE],
        time.perf_counter() - start,
    )
    return chunks


async def extract_pdf_async(pdf_path: str | Path, config: ExtractorConfig | None = None) -> list[Chunk]:
    """Async wrapper: runs the CPU-bound extraction in a worker thread."""
    return await asyncio.to_thread(extract_pdf, pdf_path, config)


async def extract_many(
    pdf_paths: Sequence[str | Path], config: ExtractorConfig | None = None, max_concurrency: int = 2
) -> dict[str, list[Chunk]]:
    """Extract several PDFs concurrently. A PDF that fails entirely maps to an empty list."""
    semaphore = asyncio.Semaphore(max_concurrency)

    async def one(p: str | Path) -> list[Chunk]:
        async with semaphore:
            try:
                return await extract_pdf_async(p, config)
            except Exception:
                logger.exception("Extraction failed for %s", p)
                return []

    results = await asyncio.gather(*(one(p) for p in pdf_paths))
    return {str(p): r for p, r in zip(pdf_paths, results, strict=True)}


# --------------------------------------------------------------------------- page processing


def _page_chunks(
    page: pymupdf.Page,
    page_number: int,
    tables: list[_Table],
    cfg: ExtractorConfig,
    image_dir: Path,
    base_meta: dict[str, Any],
) -> list[Chunk]:
    doc_id = str(base_meta["doc_id"])
    chunks: list[Chunk] = []
    table_boxes = [t.bbox for t in tables if t.bbox is not None]

    for i, table in enumerate(tables, start=1):
        chunks.append(
            Chunk(
                chunk_type=ChunkType.TABLE,
                content=table.markdown,
                page_number=page_number,
                chunk_id=f"{doc_id}:p{page_number}:table{i}",
                metadata={**base_meta, "bbox": list(table.bbox) if table.bbox else None, "table_index": i},
            )
        )

    blocks = _text_blocks(page)
    figure_boxes: list[BBox] = []
    if cfg.extract_figures:
        figure_boxes = _figure_boxes(page, table_boxes, cfg)
        if figure_boxes:
            image_dir.mkdir(parents=True, exist_ok=True)
        page_rect = page.rect
        for i, box in enumerate(figure_boxes, start=1):
            import pymupdf

            clip = pymupdf.Rect(*_pad(box, cfg.figure_padding)) & page_rect
            image_path = image_dir / f"p{page_number}_fig{i}.png"
            pixmap = page.get_pixmap(clip=clip, dpi=cfg.image_dpi)
            pixmap.save(image_path)
            caption = _find_caption(box, blocks)
            chunks.append(
                Chunk(
                    chunk_type=ChunkType.IMAGE,
                    content=caption,
                    page_number=page_number,
                    image_path=str(image_path),
                    chunk_id=f"{doc_id}:p{page_number}:fig{i}",
                    metadata={**base_meta, "bbox": list(box), "figure_index": i, "caption": caption or None},
                )
            )
        logger.debug("Page %d: rendered %d figures", page_number, len(figure_boxes))

    excluded = table_boxes + figure_boxes
    text = "\n\n".join(
        text for bbox, text in blocks if not any(_overlap_frac(bbox, ex) > 0.5 for ex in excluded)
    )
    pieces = chunk_text(text, cfg.chunk_size, cfg.chunk_overlap)
    if not pieces and not tables and not figure_boxes:
        logger.warning("Page %d has no extractable text (scanned page? OCR is not enabled)", page_number)
    for i, piece in enumerate(pieces, start=1):
        chunks.append(
            Chunk(
                chunk_type=ChunkType.TEXT,
                content=piece,
                page_number=page_number,
                chunk_id=f"{doc_id}:p{page_number}:text{i}",
                metadata={**base_meta, "chunk_index": i},
            )
        )
    return chunks


def _text_blocks(page: pymupdf.Page) -> list[tuple[BBox, str]]:
    """Text blocks in reading order, with whitespace inside each block normalised."""
    out: list[tuple[BBox, str]] = []
    for block in page.get_text("blocks", sort=True):
        x0, y0, x1, y1, text, _, block_type = block[:7]
        if block_type != 0:  # 1 = image block
            continue
        cleaned = re.sub(r"[ \t]+", " ", str(text)).strip()
        cleaned = re.sub(r"-\n(?=[a-z])", "", cleaned)  # re-join hyphenated line breaks
        cleaned = re.sub(r"\s*\n\s*", " ", cleaned)
        if cleaned:
            out.append(((float(x0), float(y0), float(x1), float(y1)), cleaned))
    return out


def _pymupdf_tables(page: pymupdf.Page) -> list[_Table]:
    tables: list[_Table] = []
    for table in page.find_tables().tables:
        markdown = table.to_markdown().strip()
        if markdown and table.row_count >= 2:
            x0, y0, x1, y1 = table.bbox
            tables.append(_Table((float(x0), float(y0), float(x1), float(y1)), markdown))
    return tables


def _figure_boxes(page: pymupdf.Page, table_boxes: list[BBox], cfg: ExtractorConfig) -> list[BBox]:
    """Bounding boxes of raster images and vector-graphic clusters, minus tables and tiny items."""
    candidates: list[BBox] = []
    for info in page.get_image_info():
        x0, y0, x1, y1 = info["bbox"]
        candidates.append((float(x0), float(y0), float(x1), float(y1)))
    if cfg.include_vector_figures:
        for rect in page.cluster_drawings():
            candidates.append((float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)))

    page_box: BBox = (float(page.rect.x0), float(page.rect.y0), float(page.rect.x1), float(page.rect.y1))
    page_area = _area(page_box)
    kept: list[BBox] = []
    for box in candidates:
        clipped = _intersection(box, page_box)
        if clipped is None:
            continue
        if (clipped[2] - clipped[0]) < cfg.min_figure_size or (clipped[3] - clipped[1]) < cfg.min_figure_size:
            continue
        if _area(clipped) > 0.9 * page_area:  # full-page backgrounds / scanned page images
            continue
        # Table borders show up as drawing clusters; drop clusters that are really a table.
        if any(_overlap_frac(t, clipped) > 0.5 or _overlap_frac(clipped, t) > 0.5 for t in table_boxes):
            continue
        kept.append(clipped)
    return _merge_boxes(kept)


def _find_caption(box: BBox, blocks: list[tuple[BBox, str]], max_gap: float = 40.0) -> str:
    """Closest text block just below (preferred) or above a figure that looks like a caption."""
    best: tuple[float, str] | None = None
    for (x0, y0, x1, y1), text in blocks:
        if not _CAPTION_RE.match(text):
            continue
        if x1 < box[0] or x0 > box[2]:  # no horizontal overlap
            continue
        if 0 <= y0 - box[3] <= max_gap:
            gap = y0 - box[3]
        elif 0 <= box[1] - y1 <= max_gap:
            gap = (box[1] - y1) + 0.5  # slight preference for captions below
        else:
            continue
        if best is None or gap < best[0]:
            best = (gap, text)
    return best[1] if best else ""


# --------------------------------------------------------------------------- text chunking


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into chunks of about ``chunk_size`` characters with ``chunk_overlap`` overlap.

    Splits on paragraph breaks first, then sentence ends, then whitespace, so chunks rarely cut
    mid-word. Overlap is taken from the end of the previous chunk, aligned to a word boundary.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    # (unit, starts_new_paragraph)
    units: list[tuple[str, bool]] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            units.append((para, True))
            continue
        first = True
        for sentence in re.split(r"(?<=[.!?])\s+", para):
            while len(sentence) > chunk_size:  # very long sentence: cut at the last space
                cut = sentence.rfind(" ", 0, chunk_size)
                cut = cut if cut > chunk_size // 2 else chunk_size
                units.append((sentence[:cut].strip(), first))
                first = False
                sentence = sentence[cut:].strip()
            if sentence:
                units.append((sentence, first))
                first = False

    chunks: list[str] = []
    current = ""
    for unit, new_para in units:
        candidate = f"{current}{chr(10) * 2 if new_para else ' '}{unit}" if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = _overlap_tail(current, chunk_overlap)
            current = f"{tail} {unit}" if tail and len(tail) + 1 + len(unit) <= chunk_size else unit
        else:
            current = unit
    if current:
        chunks.append(current)
    return chunks


def _overlap_tail(text: str, overlap: int) -> str:
    if overlap <= 0:
        return ""
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1 :] if 0 <= space < len(tail) - 1 else tail


# --------------------------------------------------------------------------- unstructured backend


def _unstructured_tables(path: Path, page_sizes: dict[int, tuple[float, float]]) -> dict[int, list[_Table]]:
    """Tables per page via unstructured's hi-res layout model, converted from HTML to Markdown."""
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError as exc:
        raise RuntimeError(
            "table_backend='unstructured' needs `pip install 'unstructured[pdf]'` "
            "plus poppler and tesseract; use table_backend='pymupdf' otherwise"
        ) from exc

    start = time.perf_counter()
    elements = partition_pdf(filename=str(path), strategy="hi_res", infer_table_structure=True)
    out: dict[int, list[_Table]] = {}
    for el in elements:
        if getattr(el, "category", "") != "Table":
            continue
        meta = el.metadata
        page_number = int(getattr(meta, "page_number", None) or 0)
        html = getattr(meta, "text_as_html", None)
        markdown = html_table_to_markdown(html) if html else str(el)
        bbox = _unstructured_bbox(meta, page_sizes.get(page_number))
        if page_number and markdown.strip():
            out.setdefault(page_number, []).append(_Table(bbox, markdown))
    logger.info(
        "unstructured found %d tables in %s in %.1fs",
        sum(len(v) for v in out.values()),
        path.name,
        time.perf_counter() - start,
    )
    return out


def _unstructured_bbox(meta: Any, page_size: tuple[float, float] | None) -> BBox | None:
    """Convert unstructured's page-image pixel coordinates to PDF points."""
    coords = getattr(meta, "coordinates", None)
    system = getattr(coords, "system", None)
    if coords is None or not getattr(coords, "points", None) or page_size is None:
        return None
    width, height = getattr(system, "width", None), getattr(system, "height", None)
    if not width or not height:
        return None
    sx, sy = page_size[0] / float(width), page_size[1] / float(height)
    xs = [float(pt[0]) * sx for pt in coords.points]
    ys = [float(pt[1]) * sy for pt in coords.points]
    return (min(xs), min(ys), max(xs), max(ys))


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._row is not None and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def html_table_to_markdown(html: str) -> str:
    """Convert a simple HTML table to a Markdown table (first row is the header)."""
    parser = _TableHTMLParser()
    parser.feed(html)
    rows = parser.rows
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    padded = [[c.replace("|", "\\|") for c in r] + [""] * (width - len(r)) for r in rows]
    lines = ["|" + "|".join(padded[0]) + "|", "|" + "|".join(["---"] * width) + "|"]
    lines += ["|" + "|".join(r) + "|" for r in padded[1:]]
    return "\n".join(lines)


# --------------------------------------------------------------------------- geometry helpers


def _area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection(a: BBox, b: BBox) -> BBox | None:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _overlap_frac(a: BBox, b: BBox) -> float:
    """Fraction of ``a``'s area that lies inside ``b``."""
    inter = _intersection(a, b)
    area = _area(a)
    return _area(inter) / area if inter and area else 0.0


def _pad(b: BBox, pad: float) -> BBox:
    return (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)


def _merge_boxes(boxes: list[BBox], gap: float = 6.0) -> list[BBox]:
    """Merge boxes that overlap or nearly touch (e.g. an image plus its vector overlay)."""
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        out: list[BBox] = []
        for box in merged:
            for i, other in enumerate(out):
                if _intersection(_pad(box, gap), other) is not None:
                    out[i] = (
                        min(box[0], other[0]),
                        min(box[1], other[1]),
                        max(box[2], other[2]),
                        max(box[3], other[3]),
                    )
                    changed = True
                    break
            else:
                out.append(box)
        merged = out
    return sorted(merged, key=lambda b: (round(b[1]), b[0]))


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Extract text, tables and figures from a PDF.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output-dir", default=ExtractorConfig().output_dir)
    parser.add_argument("--table-backend", choices=["pymupdf", "unstructured"], default="pymupdf")
    args = parser.parse_args()

    result = extract_pdf(
        args.pdf, ExtractorConfig(output_dir=args.output_dir, table_backend=args.table_backend)
    )
    for chunk in result:
        preview = (chunk.image_path or chunk.content).replace("\n", " ")[:90]
        print(f"p{chunk.page_number:<3} {chunk.chunk_type.value:<5} {preview}")
