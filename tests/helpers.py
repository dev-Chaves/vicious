"""Builders for synthetic extraction objects used by tests."""

from __future__ import annotations

from pathlib import Path

from audiobook.extract import (
    Block,
    Chapter,
    ChapterRange,
    Document,
    Line,
    Page,
    Span,
)

PAGE_HEIGHT = 661.5
PAGE_WIDTH = 504.0


def make_block(
    text: str,
    *,
    page: int = 1,
    y0: float = 100.0,
    x0: float = 72.0,
    width: float = 360.0,
    height: float = 14.0,
    font: str = "MinionPro-Regular",
    size: float = 10.5,
    image: bool = False,
    line_height: float = 13.0,
) -> Block:
    if image:
        return Block(page=page, bbox=(x0, y0, x0 + width, y0 + height), image=True)
    lines = []
    for index, line_text in enumerate(text.split("\n")):
        line_y = y0 + index * line_height
        lines.append(
            Line(
                text=line_text,
                bbox=(x0, line_y, x0 + width, line_y + height),
                spans=(Span(text=line_text, font=font, size=size),),
            )
        )
    return Block(page=page, bbox=(x0, y0, x0 + width, y0 + height), lines=tuple(lines))


def make_page(number: int, blocks: list[Block]) -> Page:
    return Page(
        number=number,
        width=PAGE_WIDTH,
        height=PAGE_HEIGHT,
        blocks=tuple(blocks),
    )


def make_chapter(
    pages: list[Page],
    *,
    number: int = 1,
    title: str = "Chapter 1. Test",
) -> Chapter:
    return Chapter(
        range=ChapterRange(
            number=number,
            title=title,
            first_page=pages[0].number,
            last_page=pages[-1].number,
        ),
        pages=tuple(pages),
    )


def make_document(path: str = "book.pdf") -> Document:
    return Document(
        path=Path(path),
        title="Test Book",
        page_count=100,
        toc=(),
        sha256="0" * 64,
    )
