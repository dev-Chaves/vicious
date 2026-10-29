"""Classify extracted blocks into narration segments.

Rules use typography and geometry only: this book has a very consistent
design, so we do not need a layout model for milestone 1.

* prose / heading  : MinionPro body or MyriadPro heading fonts
* code             : UbuntuMono, either indented blocks or single expressions
* figure           : image blocks, or a ``Figure N-N.`` caption without an image
* table            : a ``Table N-N.`` caption plus the rows that follow it
* caption          : the figure/table caption itself (read aloud)

Furniture (running footer, page number, watermark, footnotes) is flagged with
``skip=True``; :mod:`audiobook.script` decides what to persist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .clean import page_furniture_kind
from .extract import Block, Chapter
from .types import SegmentType

HEADING_FONTS = ("MyriadPro-SemiboldCond", "MyriadPro-BoldCond")
TABLE_FONTS = ("MyriadPro-Cond", "MyriadPro-BoldCond")
CAPTION_RE = re.compile(
    r"^(?P<kind>Figure|Table)\s+(?P<label>\d+[-\u2013]\d+)\.\s*(?P<body>.*)$",
    re.DOTALL,
)
CODE_CHARS_RE = re.compile(r"[{}=;<>/\\\[\]#$]|::|\w\(\)|=>")

#: A figure caption sits directly under its artwork, but multi-column layouts
#: can leave a gap. Anything further away belongs to another figure.
CAPTION_DISTANCE = 90.0


@dataclass
class ClassifiedItem:
    type: SegmentType
    page: int
    blocks: list[Block] = field(default_factory=list)
    label: str | None = None
    caption_body: str | None = None
    skip: bool = False
    reason: str = ""

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.blocks).strip()

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        if not self.blocks:
            return (0.0, 0.0, 0.0, 0.0)
        return self.blocks[0].bbox

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def y1(self) -> float:
        return self.bbox[3]


def classify_chapter(chapter: Chapter) -> list[ClassifiedItem]:
    items: list[ClassifiedItem] = []
    table: ClassifiedItem | None = None

    for page in chapter.pages:
        for block in page.blocks:
            if block.image:
                items.append(ClassifiedItem(SegmentType.FIGURE, page.number, [block]))
                continue

            if not block.text:
                continue

            furniture = page_furniture_kind(
                block.text, block.y0, page.height, block.size
            )
            if furniture is not None:
                items.append(
                    ClassifiedItem(
                        SegmentType.PROSE,
                        page.number,
                        [block],
                        skip=True,
                        reason=furniture,
                    )
                )
                # A footnote interrupts the table? Treat it as the end of one.
                if furniture != "footnote":
                    table = None
                continue

            if _is_chapter_opener(block):
                items.append(
                    ClassifiedItem(
                        SegmentType.PROSE,
                        page.number,
                        [block],
                        skip=True,
                        reason="chapter-opener",
                    )
                )
                continue

            caption = CAPTION_RE.match(block.text)
            if caption:
                label = f"{caption.group('kind')} {caption.group('label')}"
                if caption.group("kind") == "Table":
                    table = ClassifiedItem(
                        SegmentType.TABLE, page.number, label=label
                    )
                    items.append(table)
                else:
                    table = None
                items.append(
                    ClassifiedItem(
                        SegmentType.CAPTION,
                        page.number,
                        [block],
                        label=label,
                        caption_body=caption.group("body").strip(),
                    )
                )
                continue

            if table is not None and _is_table_row(block):
                table.blocks.append(block)
                continue
            table = None

            if _is_code(block):
                items.append(ClassifiedItem(SegmentType.CODE, page.number, [block]))
            elif _is_heading(block):
                items.append(
                    ClassifiedItem(
                        SegmentType.PROSE,
                        page.number,
                        [block],
                        reason="heading",
                    )
                )
            else:
                items.append(ClassifiedItem(SegmentType.PROSE, page.number, [block]))

    return _attach_figure_labels(items)


def _attach_figure_labels(items: list[ClassifiedItem]) -> list[ClassifiedItem]:
    """Give image blocks their caption label.

    Figures put the artwork first and the caption below it. A caption without
    a nearby image is a vector drawing; a synthetic figure item is inserted so
    the listener still hears ``Figure X-Y, see the book`` before the caption.

    The caption is always kept: the narration announces the figure *and* reads
    the caption. When one caption covers several images (parts a/b/c), only the
    last image is announced as the figure; the others are marked as parts.
    """
    result: list[ClassifiedItem] = []
    pending: list[ClassifiedItem] = []

    for item in items:
        if item.type is SegmentType.FIGURE:
            pending.append(item)
            result.append(item)
            continue

        if item.type is SegmentType.CAPTION and (item.label or "").startswith(
            "Figure"
        ):
            owner = _figure_for_caption(pending, item)
            if owner is not None:
                owner.label = item.label
                for part in pending:
                    if part is not owner:
                        part.skip = True
                        part.reason = "figure-part"
            else:
                result.append(
                    ClassifiedItem(
                        SegmentType.FIGURE, item.page, label=item.label
                    )
                )
            pending = []
            result.append(item)
            continue

        if item.type is not SegmentType.CAPTION:
            pending = []
        elif not (item.label or "").startswith("Figure"):
            pending = []
        result.append(item)

    return result


def _figure_for_caption(
    pending: list[ClassifiedItem], caption: ClassifiedItem
) -> ClassifiedItem | None:
    for candidate in reversed(pending):
        if candidate.page != caption.page:
            continue
        gap = caption.y0 - candidate.y1
        if -CAPTION_DISTANCE <= gap <= CAPTION_DISTANCE:
            return candidate
    return None


def _is_chapter_opener(block: Block) -> bool:
    return block.size >= 20 and "CHAPTER" in block.text.upper()


def _is_table_row(block: Block) -> bool:
    return block.size <= 9.6 and block.font.startswith("MyriadPro-")


def _is_heading(block: Block) -> bool:
    return block.size >= 11.5 and any(
        block.font.startswith(font) for font in HEADING_FONTS
    )


def _is_code(block: Block) -> bool:
    if block.mono_ratio < 0.6 or block.text_chars < 8:
        return False
    text = block.text.strip()
    # "zone" style definition terms are isolated mono words, not code.
    if len(block.lines) == 1 and len(text) < 30 and not CODE_CHARS_RE.search(text):
        return False
    return True
