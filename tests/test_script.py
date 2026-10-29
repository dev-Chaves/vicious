from __future__ import annotations

import json

import pytest

from audiobook.narrate import RuleNarrator
from audiobook.schema import SCHEMA_VERSION, Script, ScriptError
from audiobook.script import build_script_from_chapter, build_segments
from audiobook.types import Origin, SegmentType
from tests.helpers import make_block, make_chapter, make_document, make_page


def test_title_segment_comes_first_and_ids_are_chapter_scoped():
    chapter = make_chapter(
        [make_page(1, [make_block("Body text.", y0=100)])],
        title="Chapter 1. Introduction",
    )
    segments = build_segments(chapter.range, chapter, RuleNarrator())
    assert segments[0].id == "ch01-0001"
    assert segments[0].text_original == "Chapter 1. Introduction"
    assert segments[0].text_narrated == "Chapter one. Introduction."
    assert segments[1].id == "ch01-0002"
    assert segments[1].origin is Origin.RULE


def test_page_break_paragraph_is_merged():
    chapter = make_chapter(
        [
            make_page(
                1,
                [make_block("The redirection ends when the opera\u2010", y0=540)],
            ),
            make_page(
                2,
                [make_block("tional load drops back to normal.", y0=52)],
            ),
        ]
    )
    segments = build_segments(chapter.range, chapter, RuleNarrator())
    prose = [segment for segment in segments if segment.origin is Origin.RULE][1:]
    assert len(prose) == 1
    assert prose[0].text_original == (
        "The redirection ends when the operational load drops back to normal."
    )


def test_furniture_is_kept_as_skipped_segments():
    chapter = make_chapter(
        [
            make_page(
                1,
                [
                    make_block("Body text.", y0=100),
                    make_block("www.it-ebooks.info", y0=640, size=8.0),
                    make_block("1 A footnote.", y0=560, size=8.0),
                ],
            )
        ]
    )
    segments = build_segments(chapter.range, chapter, RuleNarrator())
    skipped = [segment for segment in segments if segment.origin is Origin.SKIPPED]
    assert len(skipped) == 2
    assert all(segment.text_narrated == "" for segment in skipped)
    assert [segment.text_original for segment in skipped] == [
        "www.it-ebooks.info",
        "1 A footnote.",
    ]


def test_figure_and_caption_narration():
    chapter = make_chapter(
        [
            make_page(
                1,
                [
                    make_block("", y0=200, height=100, image=True),
                    make_block(
                        "Figure 1-1. Requests flow through the load balancer.",
                        y0=310,
                        size=9.5,
                        font="MinionPro-It",
                    ),
                ],
            )
        ]
    )
    segments = build_segments(chapter.range, chapter, RuleNarrator())
    figure = next(segment for segment in segments if segment.type is SegmentType.FIGURE)
    caption = next(
        segment for segment in segments if segment.type is SegmentType.CAPTION
    )
    assert figure.text_narrated == "Figure one dash one, see the book."
    assert caption.text_narrated == "Requests flow through the load balancer."


def test_code_is_announced():
    chapter = make_chapter(
        [
            make_page(
                1,
                [
                    make_block(
                        "if err != nil {\n    return err\n}",
                        y0=100,
                        font="UbuntuMono-Regular",
                        size=8.5,
                    )
                ],
            )
        ]
    )
    segments = build_segments(chapter.range, chapter, RuleNarrator())
    code = next(segment for segment in segments if segment.type is SegmentType.CODE)
    assert code.text_narrated == "Code listing, see the book."


def test_script_roundtrip(tmp_path):
    from audiobook.schema import Source

    script = Script(
        source=Source(
            pdf="book.pdf",
            sha256="0" * 64,
            book="Test Book",
            chapter=1,
            title="Chapter 1. Test",
            first_page=10,
            last_page=20,
        ),
        segments=build_segments(
            make_chapter([make_page(10, [make_block("Body.", y0=100)])]).range,
            make_chapter([make_page(10, [make_block("Body.", y0=100)])]),
            RuleNarrator(),
        ),
    )
    path = script.save(tmp_path / "script.json")
    loaded = Script.load(path)
    assert loaded.to_dict() == script.to_dict()
    assert loaded.version == SCHEMA_VERSION


def test_load_rejects_unknown_version(tmp_path):
    path = tmp_path / "script.json"
    path.write_text(json.dumps({"version": 99, "source": {}, "segments": []}))
    with pytest.raises(ScriptError, match="unsupported script version"):
        Script.load(path)


def test_load_reports_bad_segment(tmp_path):
    path = tmp_path / "script.json"
    path.write_text(
        json.dumps(
            {
                "version": SCHEMA_VERSION,
                "source": {
                    "pdf": "x.pdf",
                    "sha256": "0",
                    "chapter": 1,
                    "title": "T",
                    "first_page": 1,
                    "last_page": 1,
                },
                "segments": [{"id": "a", "type": "nope"}],
            }
        )
    )
    with pytest.raises(ScriptError, match="segment 0"):
        Script.load(path)
