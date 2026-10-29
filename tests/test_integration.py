"""End-to-end pass 1 on the sample book (needs the PDF in ebook-pdf/)."""

from __future__ import annotations

import re

import pytest

from audiobook.chunk import chunk_script
from audiobook.script import build_script

from .conftest import PDF_PATH


@pytest.mark.integration
def test_chapter_one_script_is_clean():
    script = build_script(PDF_PATH, 1)
    assert script.source.book.startswith("Site Reliability Engineering")
    assert (script.source.first_page, script.source.last_page) == (29, 38)
    assert len(script.narrated) > 60

    narrated = " ".join(segment.text_narrated for segment in script.narrated)
    assert not re.search(r"\d", narrated)
    assert "\u2010" not in narrated
    assert "www.it-ebooks.info" not in narrated
    assert "Chapter one. Introduction." in narrated
    assert "operational load drops back" in narrated
    assert "S R E" in narrated

    chunks = chunk_script(script)
    assert all(chunk.chars <= 600 for chunk in chunks)
    assert " ".join(chunk.text for chunk in chunks).split() == narrated.split()


@pytest.mark.integration
def test_script_json_roundtrip(tmp_path):
    from audiobook.schema import Script

    script = build_script(PDF_PATH, 1)
    path = script.save(tmp_path / "script.json")
    loaded = Script.load(path)
    assert loaded.to_dict() == script.to_dict()
