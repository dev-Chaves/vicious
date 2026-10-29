from __future__ import annotations

import json

from audiobook.schema import Script, Segment, Source
from audiobook.types import Origin, SegmentType


def make_script(text: str = "Hello there.") -> Script:
    return Script(
        source=Source(
            pdf="book.pdf",
            sha256="0" * 64,
            chapter=1,
            title="Chapter 1. Test",
            first_page=1,
            last_page=1,
        ),
        segments=[
            Segment(
                id="ch01-0001",
                type=SegmentType.PROSE,
                text_original=text,
                text_narrated=text,
                origin=Origin.RULE,
                page=1,
            )
        ],
    )


def test_save_is_atomic_and_leaves_no_temporary_files(tmp_path):
    path = tmp_path / "script.json"
    make_script("first").save(path)
    make_script("second").save(path)

    assert json.loads(path.read_text())["segments"][0]["text_narrated"] == "second"
    assert list(tmp_path.glob("*.tmp")) == []


def test_to_json_matches_the_saved_file(tmp_path):
    script = make_script()
    path = script.save(tmp_path / "script.json")
    assert path.read_text(encoding="utf-8") == script.to_json()
    assert Script.load(path).to_json() == script.to_json()
