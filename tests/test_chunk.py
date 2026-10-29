from __future__ import annotations

from audiobook.chunk import (
    chunk_script,
    estimate_seconds,
    slots_for,
    split_text,
)
from audiobook.narrate import RuleNarrator
from audiobook.script import build_segments
from audiobook.schema import Script, Source
from audiobook.types import Origin
from tests.helpers import make_block, make_chapter, make_page


def _script(*texts: str) -> Script:
    chapter = make_chapter(
        [make_page(1, [make_block(text, y0=100 + 40 * i) for i, text in enumerate(texts)])]
    )
    segments = build_segments(chapter.range, chapter, RuleNarrator())
    return Script(
        source=Source(
            pdf="book.pdf",
            sha256="0" * 64,
            book="Test Book",
            chapter=1,
            title="Chapter 1. Test",
            first_page=1,
            last_page=1,
        ),
        segments=segments,
    )


def test_every_chunk_respects_the_limit():
    text = " ".join(["This is a sentence with several words in it."] * 40)
    chunks = split_text(text, max_chars=200, min_chars=50)
    assert chunks
    assert all(len(chunk) <= 200 for chunk in chunks)
    assert " ".join(chunks).split() == text.split()


def test_long_sentence_is_split_at_clauses():
    sentence = (
        "The first clause explains one thing, the second clause explains another "
        "thing, and the third clause keeps going until it is comfortably longer "
        "than the configured limit."
    )
    chunks = split_text(sentence, max_chars=80, min_chars=10)
    assert len(chunks) > 1
    assert all(len(chunk) <= 80 for chunk in chunks)


def test_tail_shorter_than_minimum_is_merged():
    text = "First sentence is here. Second sentence is here. End."
    chunks = split_text(text, max_chars=200, min_chars=50)
    assert chunks == [text]


def test_chunk_ids_are_per_segment_and_stable():
    script = _script("A short paragraph.")
    first = chunk_script(script)
    second = chunk_script(script)
    assert [(chunk.id, chunk.text) for chunk in first] == [
        (chunk.id, chunk.text) for chunk in second
    ]
    assert first[0].id.endswith("-c0001")
    assert first[0].segment_id.startswith("ch01-")


def test_skipped_segments_are_not_chunked():
    chapter = make_chapter(
        [
            make_page(
                1,
                [
                    make_block("Real text.", y0=100),
                    make_block("www.it-ebooks.info", y0=640, size=8.0),
                ],
            )
        ]
    )
    segments = build_segments(chapter.range, chapter, RuleNarrator())
    script = Script(
        source=Source(
            pdf="book.pdf",
            sha256="0" * 64,
            chapter=1,
            title="Chapter 1. Test",
            first_page=1,
            last_page=1,
        ),
        segments=segments,
    )
    assert any(segment.origin is Origin.SKIPPED for segment in segments)
    chunks = chunk_script(script)
    assert all("it-ebooks" not in chunk.text for chunk in chunks)


def test_slots_budget_scales_with_text():
    short = slots_for("hello")
    long = slots_for("hello " * 300)
    assert short == 256
    assert long > short
    assert long <= 2048
    assert estimate_seconds("a" * 140) == 10.0
