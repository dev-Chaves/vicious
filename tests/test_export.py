from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from audiobook.export import (
    build_ffmetadata,
    chapter_spans,
    export_m4b,
    probe_chapters,
)

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


def manifest_for(tmp_path: Path, durations=(1.0, 1.5, 2.0)) -> Path:
    chapters = []
    for index, duration in enumerate(durations, start=1):
        directory = tmp_path / f"ch{index:02d}"
        directory.mkdir()
        mp3 = directory / f"ch{index:02d}.mp3"
        mp3.write_bytes(b"")
        chapters.append(
            {
                "number": index,
                "title": f"Chapter {index}. Test",
                "status": "done",
                "duration_s": duration,
                "mp3": f"ch{index:02d}/ch{index:02d}.mp3",
            }
        )
    path = tmp_path / "book.json"
    path.write_text(json.dumps({"book": "Test Book", "chapters": chapters}))
    return path


def test_chapter_spans_accumulate_durations(tmp_path):
    manifest = json.loads(manifest_for(tmp_path).read_text())
    spans = chapter_spans(manifest, tmp_path)
    assert [(s.start_ms, s.end_ms) for s in spans] == [
        (0, 1000),
        (1000, 2500),
        (2500, 4500),
    ]
    assert spans[1].title == "Chapter 2. Test"


def test_chapter_spans_skip_unfinished_chapters(tmp_path):
    path = manifest_for(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["chapters"][1]["status"] = "failed"
    spans = chapter_spans(manifest, tmp_path)
    assert [s.title for s in spans] == ["Chapter 1. Test", "Chapter 3. Test"]


def test_ffmetadata_lists_one_block_per_chapter(tmp_path):
    manifest = json.loads(manifest_for(tmp_path).read_text())
    spans = chapter_spans(manifest, tmp_path)
    text = build_ffmetadata(manifest, spans)
    assert text.startswith(";FFMETADATA1\n")
    assert "genre=Audiobook" in text
    assert text.count("[CHAPTER]") == 3
    assert "START=1000" in text and "END=2500" in text
    assert "title=Chapter 2. Test" in text


def test_ffmetadata_escapes_syntax_characters(tmp_path):
    path = manifest_for(tmp_path, durations=(1.0,))
    manifest = json.loads(path.read_text())
    manifest["book"] = "A=B; C#D"
    manifest["chapters"][0]["title"] = "Chapter 1 = one; two"
    spans = chapter_spans(manifest, tmp_path)
    text = build_ffmetadata(manifest, spans)
    assert r"title=A\=B\; C\#D" in text
    assert r"title=Chapter 1 \= one\; two" in text


def _sine_mp3(path: Path, seconds: float) -> None:
    subprocess.run(
        [
            shutil.which("ffmpeg") or "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=220:duration={seconds}:sample_rate=24000",
            "-ac",
            "1",
            "-b:a",
            "96k",
            str(path),
        ],
        check=True,
    )


@requires_ffmpeg
def test_export_writes_chaptered_m4b(tmp_path):
    path = manifest_for(tmp_path, durations=(1.0, 1.0))
    manifest = json.loads(path.read_text())
    for record in manifest["chapters"]:
        _sine_mp3(tmp_path / record["mp3"], 1.0)

    result = export_m4b(path, tmp_path / "book.m4b")

    assert result.m4b.exists()
    assert result.chapters == 2
    assert result.duration_s == pytest.approx(2.0, abs=0.3)
    assert result.size_bytes > 0
    assert probe_chapters(result.m4b) == 2


@requires_ffmpeg
def test_export_fails_when_a_chapter_file_is_missing(tmp_path):
    path = manifest_for(tmp_path, durations=(1.0,))
    (tmp_path / "ch01" / "ch01.mp3").unlink()
    with pytest.raises(FileNotFoundError):
        export_m4b(path, tmp_path / "book.m4b")
