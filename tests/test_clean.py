from __future__ import annotations

import pytest

from audiobook.clean import (
    block_text,
    ensure_sentence,
    join_lines,
    join_paragraphs,
    normalize_for_review,
    normalize_for_speech,
    page_furniture_kind,
)
from tests.helpers import PAGE_HEIGHT, make_block


# ---------------------------------------------------------------------------
# Hyphenation and line joining
# ---------------------------------------------------------------------------


def test_soft_hyphen_at_line_end_joins_without_hyphen():
    assert join_lines(["computing is", "com\u2010", "puting"]) == "computing is computing"


def test_ascii_hyphen_at_line_end_is_kept():
    assert join_lines(["a blame-", "free process"]) == "a blame-free process"


def test_soft_hyphen_across_page_break_joins_without_space():
    assert (
        join_paragraphs("when the opera\u2010", "tional load drops")
        == "when the operational load drops"
    )


def test_hard_hyphen_across_page_break_joins_without_space():
    assert join_paragraphs("a blame-", "free postmortem") == "a blame-free postmortem"


def test_plain_paragraph_join_adds_one_space():
    assert join_paragraphs("ends here", "starts here") == "ends here starts here"


def test_block_text_uses_line_joining():
    block = make_block("a blame-\nfree process")
    assert block_text(block) == "a blame-free process"


# ---------------------------------------------------------------------------
# Page furniture
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "y0", "size", "expected"),
    [
        ("www.it-ebooks.info", 630, 8, "watermark"),
        ("4 | Chapter 1: Introduction", PAGE_HEIGHT * 0.93, 8.5, "footer"),
        ("3", PAGE_HEIGHT * 0.93, 8.5, "footer"),
        ("2 See \u201cDisaster Role Playing\u201d on page 401.", 560, 8, "footnote"),
        ("Regular body text.", 300, 10.5, None),
    ],
)
def test_page_furniture(text, y0, size, expected):
    assert page_furniture_kind(text, y0, PAGE_HEIGHT, size) == expected


def test_superscript_footnote_marker_is_dropped():
    block = make_block("time.\u200b", y0=300)
    line = block.lines[0]
    # Rebuild the line with a 6 pt footnote marker appended.
    from audiobook.extract import Line, Span

    marked = Line(
        text="time.1",
        bbox=line.bbox,
        spans=(
            Span(text="time.", font="MinionPro-Regular", size=10.5),
            Span(text="1", font="MinionPro-Regular", size=6.0),
        ),
    )
    assert marked.speech_text() == "time."


# ---------------------------------------------------------------------------
# Review normalization
# ---------------------------------------------------------------------------


def test_review_fixes_typography_without_rewriting_words():
    text = "A \ufb01le \u201cquoted\u201d \u2014 with an em dash\u2026"
    assert normalize_for_review(text) == 'A file "quoted", with an em dash...'


def test_review_drops_comma_created_by_em_dash_after_stop():
    text = "Hope is not a strategy.\u2014Traditional SRE saying"
    assert normalize_for_review(text) == "Hope is not a strategy. Traditional SRE saying"


def test_review_collapses_whitespace_and_space_before_punctuation():
    assert normalize_for_review("a  b , c .") == "a b, c."


# ---------------------------------------------------------------------------
# Speech normalization: acronyms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("An SRE owns reliability.", "An S R E owns reliability."),
        ("SREs own reliability.", "S R E's own reliability."),
        ("UNIX and NASA.", "UNIX and NASA."),
        ("JSON and YAML.", "jason and yaml."),
        ("SQL and SRE.", "sequel and S R E."),
    ],
)
def test_acronyms(text, expected):
    assert normalize_for_speech(text) == expected


# ---------------------------------------------------------------------------
# Speech normalization: numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("In 2014.", "In twenty fourteen."),
        ("the 1990s", "the nineteen nineties"),
        ("99.9% of requests", "ninety-nine point nine percent of requests"),
        ("1,234 requests", "one thousand two hundred thirty-four requests"),
        ("the 22nd time", "the twenty-second time"),
        ("a 3x improvement", "a three times improvement"),
        (
            "version 2.3.4",
            "version two point three point four",
        ),
        ("10.0.0.1", "one zero dot zero dot zero dot one"),
        ("fifty percent", "fifty percent"),
    ],
)
def test_numbers(text, expected):
    assert normalize_for_speech(text) == expected


# ---------------------------------------------------------------------------
# Speech normalization: identifiers and URLs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("use snake_case_names", "use snake case names"),
        ("camelCaseName", "camel Case Name"),
        ("the read-only mode", "the read only mode"),
        ("iPhone and macOS", "iPhone and macOS"),
    ],
)
def test_identifiers(text, expected):
    assert normalize_for_speech(text) == expected


def test_urls_are_verbalized():
    assert (
        normalize_for_speech("see https://example.com/sre-guide for details")
        == "see example dot com slash sre guide for details"
    )


def test_www_and_query_are_verbalized():
    assert (
        normalize_for_speech("visit www.example.org/search?q=sre")
        == "visit example dot org slash search query q equals sre"
    )


def test_abbreviations_are_expanded_before_sentence_splitting():
    assert (
        normalize_for_speech("e.g. one, i.e. two, etc.")
        == "for example one, that is two, et cetera."
    )


def test_editorial_brackets_do_not_reach_the_speaker():
    assert normalize_for_speech("[re]integrating teams") == "reintegrating teams"


def test_figure_labels_are_spoken_with_dash():
    assert (
        normalize_for_speech("Figure 1-1. Overview.")
        == "Figure one dash one. Overview."
    )


def test_citations_are_dropped():
    assert normalize_for_speech("as shown in [Bey16].") == "as shown in."


def test_ensure_sentence():
    assert ensure_sentence("hello") == "hello."
    assert ensure_sentence("hello!") == "hello!"
    assert ensure_sentence("") == ""
