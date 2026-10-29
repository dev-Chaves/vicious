"""Pass 1: PDF → segments → ``script.json``.

Extraction, classification, cleanup and narration live here. The schema lives
in :mod:`audiobook.schema`, which pass 2 imports on its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .classify import ClassifiedItem, classify_chapter
from .clean import block_text, join_paragraphs, normalize_for_review
from .extract import Chapter, ChapterRange, Document, extract_chapter
from .narrate import NarrationRequest, Narrator, RuleNarrator
from .schema import SCHEMA_VERSION, Script, ScriptError, Segment, Source
from .types import Origin, SegmentType

__all__ = [
    "SCHEMA_VERSION",
    "Script",
    "ScriptError",
    "Segment",
    "Source",
    "build_script",
    "build_script_from_chapter",
    "build_segments",
]


def build_script(
    pdf_path: str | Path,
    chapter_number: int,
    narrator: Narrator | None = None,
) -> Script:
    document, chapter = extract_chapter(pdf_path, chapter_number)
    return build_script_from_chapter(
        document=document,
        chapter=chapter,
        source_pdf=str(pdf_path),
        narrator=narrator,
    )


def build_script_from_chapter(
    document: Document,
    chapter: Chapter,
    source_pdf: str | None = None,
    narrator: Narrator | None = None,
) -> Script:
    narrator = narrator or RuleNarrator()
    chapter_range = chapter.range
    source = Source(
        pdf=source_pdf or str(document.path),
        sha256=document.sha256,
        book=document.title,
        chapter=chapter_range.number,
        title=chapter_range.title,
        first_page=chapter_range.first_page,
        last_page=chapter_range.last_page,
    )
    segments = build_segments(chapter_range, chapter, narrator)
    return Script(source=source, segments=segments)


def build_segments(
    chapter_range: ChapterRange,
    chapter: Chapter,
    narrator: Narrator,
) -> list[Segment]:
    items = classify_chapter(chapter)
    segments: list[Segment] = []
    prefix = f"ch{chapter_range.number:02d}"

    def add(segment: Segment) -> None:
        segments.append(segment)

    last_prose: Segment | None = None

    title = normalize_for_review(chapter_range.title)
    add(
        Segment(
            id=f"{prefix}-{len(segments) + 1:04d}",
            type=SegmentType.PROSE,
            text_original=title,
            text_narrated=narrator.narrate(
                NarrationRequest(SegmentType.PROSE, title)
            ),
            origin=Origin.RULE,
            page=chapter_range.first_page,
        )
    )

    for item in items:
        segment_id = f"{prefix}-{len(segments) + 1:04d}"
        original = _original_text(item)
        if not original:
            if item.type in (SegmentType.FIGURE, SegmentType.TABLE):
                original = f"{item.label}." if item.label else ""
            else:
                continue
        if item.skip:
            add(
                Segment(
                    id=segment_id,
                    type=item.type,
                    text_original=original,
                    text_narrated="",
                    origin=Origin.SKIPPED,
                    page=item.page,
                    label=item.label,
                )
            )
            continue
        if (
            item.type is SegmentType.PROSE
            and last_prose is not None
            and _continues(last_prose.text_original, original)
        ):
            last_prose.text_original = join_paragraphs(
                last_prose.text_original, original
            )
            last_prose.text_narrated = narrator.narrate(
                NarrationRequest(SegmentType.PROSE, last_prose.text_original)
            )
            continue
        narrated = narrator.narrate(
            NarrationRequest(
                kind=item.type,
                text=original,
                label=item.label,
                caption=item.caption_body,
            )
        )
        add(
            Segment(
                id=segment_id,
                type=item.type,
                text_original=original,
                text_narrated=narrated,
                origin=Origin.RULE,
                page=item.page,
                label=item.label,
            )
        )
        last_prose = segments[-1] if item.type is SegmentType.PROSE else None
    return segments


def _continues(previous: str, following: str) -> bool:
    """Detect a paragraph interrupted by a page break.

    The last body block on a page can end mid-sentence (``…learn and``) and the
    next page opens with its continuation (``emulate.``). Merging them avoids
    an artificial full stop and a pause in the middle of the sentence.
    """
    previous = previous.rstrip()
    following = following.lstrip()
    if not previous or not following:
        return False
    if previous[-1] in ".!?\"'’”)]":
        return False
    return following[0].islower()


def _original_text(item: ClassifiedItem) -> str:
    texts: Iterable[str] = (block_text(block) for block in item.blocks if block.text)
    return "\n\n".join(text for text in texts if text).strip()
