from __future__ import annotations

import json
import shutil

import pytest

from audiobook.book import BookOptions, PdfBookSource, run_book
from audiobook.cli import _parse_chapters
from audiobook.engines import EngineOptions

from .helpers import make_block, make_chapter, make_document, make_page
from .test_pipeline import StubEngine

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)


class FakeSource:
    """In-memory stand-in for PdfBookSource."""

    def __init__(self, chapters):
        self.document = make_document()
        self._chapters = chapters

    def chapter(self, number):
        chapter = self._chapters[number]
        if isinstance(chapter, Exception):
            raise chapter
        return chapter

    def available(self):
        return sorted(self._chapters)


def chapter(number: int):
    return make_chapter(
        [
            make_page(
                number * 10,
                [
                    make_block(f"Chapter {number} opens with a sentence."),
                    make_block("And continues with another one."),
                ],
            )
        ],
        number=number,
        title=f"Chapter {number}. Test",
    )


def options(tmp_path, chapters=(1, 2), force=False):
    return BookOptions(
        pdf=tmp_path / "book.pdf",
        out_dir=tmp_path / "build",
        chapters=list(chapters),
        engine=EngineOptions(name="kokoro"),
        cache_dir=tmp_path / "cache",
        use_cache=False,
        force=force,
        normalize=False,
    )


@requires_ffmpeg
def test_run_book_builds_every_chapter(tmp_path):
    engine = StubEngine()
    source = FakeSource({1: chapter(1), 2: chapter(2)})
    result = run_book(options(tmp_path), engine=engine, source=source)

    assert [record.status for record in result.chapters] == ["done", "done"]
    assert [record.number for record in result.chapters] == [1, 2]
    assert all(
        (tmp_path / "build" / f"ch{record.number:02d}"
         / f"ch{record.number:02d}.mp3").exists()
        for record in result.chapters
    )
    assert result.manifest_path.exists()
    assert result.book_mp3 is not None and result.book_mp3.exists()
    assert result.manifest_path.with_name("book.json").exists()

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["version"] == 1
    assert [item["status"] for item in manifest["chapters"]] == ["done", "done"]


@requires_ffmpeg
def test_run_book_resumes_without_synthesising(tmp_path):
    source = FakeSource({1: chapter(1), 2: chapter(2)})
    first = StubEngine()
    run_book(options(tmp_path), engine=first, source=source)
    assert first.calls > 0

    second = StubEngine()
    result = run_book(options(tmp_path), engine=second, source=source)
    assert second.calls == 0
    assert [record.status for record in result.chapters] == ["skipped", "skipped"]
    assert all(record.duration_s > 0 for record in result.chapters)


@requires_ffmpeg
def test_run_book_force_regenerates(tmp_path):
    source = FakeSource({1: chapter(1)})
    run_book(options(tmp_path, chapters=(1,)), engine=StubEngine(), source=source)
    second = StubEngine()
    result = run_book(
        options(tmp_path, chapters=(1,), force=True),
        engine=second,
        source=source,
    )
    assert second.calls > 0
    assert result.chapters[0].status == "done"


@requires_ffmpeg
def test_run_book_isolates_a_failed_chapter(tmp_path):
    source = FakeSource(
        {1: chapter(1), 2: RuntimeError("boom"), 3: chapter(3)}
    )
    result = run_book(
        options(tmp_path, chapters=(1, 2, 3)),
        engine=StubEngine(),
        source=source,
    )
    statuses = [record.status for record in result.chapters]
    assert statuses == ["done", "failed", "done"]
    assert "boom" in result.chapters[1].error
    manifest = json.loads(result.manifest_path.read_text())
    assert len(manifest["chapters"]) == 3


@requires_ffmpeg
def test_run_book_rejects_unknown_chapter(tmp_path):
    source = FakeSource({1: chapter(1)})
    with pytest.raises(ValueError, match="not found in the PDF TOC"):
        run_book(options(tmp_path, chapters=(1, 9)), engine=StubEngine(), source=source)


def test_parse_chapters_accepts_ranges():
    assert _parse_chapters("all", [1, 2, 3]) is None
    assert _parse_chapters("2-4", [1, 2, 3, 4, 5]) == [2, 3, 4]
    assert _parse_chapters("1, 3, 5-6", [1, 2, 3, 4, 5, 6]) == [1, 3, 5, 6]
    with pytest.raises(ValueError, match="not found"):
        _parse_chapters("1, 9", [1, 2, 3])


def test_pdf_source_is_a_book_source():
    assert hasattr(PdfBookSource, "chapter")
