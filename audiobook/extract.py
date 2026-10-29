"""Extract text blocks with typographic metadata from a PDF.

The extractor is deliberately dumb: it returns pages, blocks, lines and spans
with font/size/position, plus the native table of contents. Deciding what is
prose, code or a figure belongs to :mod:`audiobook.classify`.

A future fallback for scanned pages (LFM3-VL) would produce the same
``Block`` structures from page images and plug in here; nothing downstream
needs to know where the blocks came from.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pymupdf

MONO_FONT_HINTS = ("mono", "courier", "consol", "code", "menlo", "inconsolata")


@dataclass(frozen=True)
class Span:
    text: str
    font: str
    size: float
    flags: int = 0

    @property
    def mono(self) -> bool:
        name = self.font.lower()
        return any(hint in name for hint in MONO_FONT_HINTS)


@dataclass(frozen=True)
class Line:
    text: str
    bbox: tuple[float, float, float, float]
    spans: tuple[Span, ...]

    @property
    def body_size(self) -> float:
        """Size of the dominant text on the line (ignores superscripts)."""
        return max((s.size for s in self.spans), default=0.0)

    @property
    def font(self) -> str:
        """Font carrying most characters on the line."""
        if not self.spans:
            return ""
        return max(self.spans, key=lambda s: len(s.text)).font

    @property
    def size(self) -> float:
        if not self.spans:
            return 0.0
        return max(self.spans, key=lambda s: len(s.text)).size

    @property
    def mono_chars(self) -> int:
        return sum(len(s.text) for s in self.spans if s.mono)

    @property
    def text_chars(self) -> int:
        return sum(len(s.text) for s in self.spans)

    def speech_text(self) -> str:
        """Text with superscript footnote markers removed.

        The book marks footnotes with 6 pt digits inside 10.5 pt body text.
        A line that is entirely small text (the footnote itself) has its own
        body size, so its content survives.
        """
        body = self.body_size
        parts = []
        for span in self.spans:
            small_digit = (
                body > 0
                and span.size < body * 0.7
                and span.text.strip().isdigit()
            )
            if not small_digit:
                parts.append(span.text)
        return "".join(parts)


@dataclass(frozen=True)
class Block:
    page: int
    bbox: tuple[float, float, float, float]
    lines: tuple[Line, ...] = ()
    image: bool = False

    @property
    def text(self) -> str:
        return "\n".join(line.speech_text() for line in self.lines).strip()

    @property
    def text_chars(self) -> int:
        return sum(line.text_chars for line in self.lines)

    @property
    def mono_chars(self) -> int:
        return sum(line.mono_chars for line in self.lines)

    @property
    def mono_ratio(self) -> float:
        if not self.text_chars:
            return 0.0
        return self.mono_chars / self.text_chars

    @property
    def size(self) -> float:
        """Most common font size in the block, weighted by characters."""
        sizes: dict[float, int] = {}
        for line in self.lines:
            for span in line.spans:
                sizes[round(span.size, 1)] = sizes.get(round(span.size, 1), 0) + len(
                    span.text
                )
        if not sizes:
            return 0.0
        return max(sizes.items(), key=lambda item: item[1])[0]

    @property
    def font(self) -> str:
        fonts: dict[str, int] = {}
        for line in self.lines:
            for span in line.spans:
                fonts[span.font] = fonts.get(span.font, 0) + len(span.text)
        if not fonts:
            return ""
        return max(fonts.items(), key=lambda item: item[1])[0]

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def y1(self) -> float:
        return self.bbox[3]


@dataclass(frozen=True)
class Page:
    number: int
    width: float
    height: float
    blocks: tuple[Block, ...]


@dataclass(frozen=True)
class TocEntry:
    level: int
    title: str
    page: int


@dataclass(frozen=True)
class Document:
    path: Path
    title: str
    page_count: int
    toc: tuple[TocEntry, ...]
    sha256: str


@dataclass(frozen=True)
class ChapterRange:
    number: int
    title: str
    first_page: int
    last_page: int


@dataclass(frozen=True)
class Chapter:
    range: ChapterRange
    pages: tuple[Page, ...]


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_document(path: str | Path) -> Document:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    with pymupdf.open(path) as pdf:
        if pdf.needs_pass:
            raise ValueError("PDF is password protected")
        toc = tuple(
            TocEntry(level=level, title=title.strip(), page=page)
            for level, title, page in pdf.get_toc(simple=True)
        )
        return Document(
            path=path,
            title=(pdf.metadata or {}).get("title") or path.stem,
            page_count=pdf.page_count,
            toc=toc,
            sha256=file_sha256(path),
        )


_CHAPTER_RE = re.compile(r"^Chapter\s+(\d+)\s*[.:]\s*(.+)$", re.IGNORECASE)


def find_chapter(doc: Document, number: int) -> ChapterRange:
    """Locate a numbered chapter in the native TOC.

    The book's TOC page numbers are 1-based PDF page indexes, which is what
    PyMuPDF returns. The chapter ends before the next TOC entry at level 1 or
    2 (parts or chapters).
    """
    chapters: list[tuple[int, TocEntry]] = []
    for index, entry in enumerate(doc.toc):
        match = _CHAPTER_RE.match(entry.title)
        if entry.level == 2 and match:
            chapters.append((int(match.group(1)), entry))

    match = next(((i, entry) for i, entry in chapters if i == number), None)
    if match is None:
        available = ", ".join(str(i) for i, _ in chapters[:8])
        raise ValueError(
            f"Chapter {number} not found in the PDF TOC (found: {available}...)"
        )

    index, entry = match
    next_page = doc.page_count + 1
    position = doc.toc.index(entry)
    for later in doc.toc[position + 1 :]:
        if later.level <= 2 and later.page > entry.page:
            next_page = later.page
            break

    first_page = max(1, entry.page)
    last_page = min(doc.page_count, next_page - 1)
    return ChapterRange(
        number=number,
        title=entry.title,
        first_page=first_page,
        last_page=last_page,
    )


def extract_chapter(path: str | Path, number: int) -> tuple[Document, Chapter]:
    doc = open_document(path)
    chapter_range = find_chapter(doc, number)
    pages = []
    with pymupdf.open(doc.path) as pdf:
        for page_index in range(chapter_range.first_page - 1, chapter_range.last_page):
            pdf_page = pdf[page_index]
            pages.append(_extract_page(pdf_page))
    return doc, Chapter(range=chapter_range, pages=tuple(pages))


def _extract_page(page: pymupdf.Page) -> Page:
    blocks: list[Block] = []
    for raw in page.get_text("dict")["blocks"]:
        bbox = tuple(round(value, 2) for value in raw["bbox"])
        if raw["type"] == 1:  # image block
            blocks.append(Block(page=page.number + 1, bbox=bbox, image=True))
            continue

        lines: list[Line] = []
        for raw_line in raw["lines"]:
            spans = tuple(
                Span(
                    text=span["text"],
                    font=span["font"],
                    size=round(span["size"], 2),
                    flags=span.get("flags", 0),
                )
                for span in raw_line["spans"]
                if span["text"]
            )
            text = "".join(span.text for span in spans)
            if not text.strip():
                continue
            lines.append(
                Line(
                    text=text,
                    bbox=tuple(round(value, 2) for value in raw_line["bbox"]),
                    spans=spans,
                )
            )
        if lines:
            blocks.append(Block(page=page.number + 1, bbox=bbox, lines=tuple(lines)))

    # Content-stream order is not visual order: on several pages footnotes are
    # emitted before the body. Reading order is top-to-bottom, left-to-right.
    blocks.sort(key=lambda block: (round(block.y0, 1), round(block.x0, 1)))
    return Page(
        number=page.number + 1,
        width=round(page.rect.width, 2),
        height=round(page.rect.height, 2),
        blocks=tuple(blocks),
    )


def iter_lines(blocks: Iterable[Block]) -> Iterable[Line]:
    for block in blocks:
        yield from block.lines
