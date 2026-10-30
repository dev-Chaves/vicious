"""Export a finished book as ``.m4b`` with chapter markers.

Apple Books is the only native macOS/iOS player with audiobook behaviour
(position memory, sleep timer, chapter navigation), and it wants an ``.m4b``
with an MP4 chapter track. The chapter MP3s and ``book.json`` already carry
everything needed, so this module only rewrites the container once.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

DEFAULT_BITRATE = "64k"


@dataclass(frozen=True)
class ChapterSpan:
    title: str
    start_ms: int
    end_ms: int
    path: Path


@dataclass(frozen=True)
class ExportResult:
    m4b: Path
    chapters: int
    duration_s: float
    size_bytes: int
    bitrate: str


def load_manifest(path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"manifest not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON ({error})") from error
    if not isinstance(data, dict) or not isinstance(data.get("chapters"), list):
        raise ValueError(f"{path}: missing chapter list")
    return data, path.parent


def chapter_spans(manifest: dict[str, Any], root: Path) -> list[ChapterSpan]:
    """Chapter boundaries in playback order, in milliseconds."""
    records = [
        record
        for record in manifest["chapters"]
        if record.get("status") in ("done", "skipped")
        and float(record.get("duration_s", 0)) > 0
    ]
    if not records:
        raise ValueError("manifest has no finished chapters")
    records.sort(key=lambda record: int(record["number"]))

    spans: list[ChapterSpan] = []
    position = 0.0
    for record in records:
        duration = float(record["duration_s"])
        path = root / str(record["mp3"])
        if not path.exists():
            raise FileNotFoundError(f"chapter audio missing: {path}")
        spans.append(
            ChapterSpan(
                title=str(record.get("title") or f"Chapter {record['number']}"),
                start_ms=round(position * 1000),
                end_ms=round((position + duration) * 1000),
                path=path,
            )
        )
        position += duration
    return spans


def build_ffmetadata(
    manifest: dict[str, Any],
    spans: Sequence[ChapterSpan],
    genre: str = "Audiobook",
    artist: str = "vicious local TTS",
) -> str:
    """Render ffmpeg's ``;FFMETADATA1`` format with one ``[CHAPTER]`` block each."""
    title = str(manifest.get("book") or "Audiobook")
    lines = [
        ";FFMETADATA1",
        f"title={_escape(title)}",
        f"album={_escape(title)}",
        f"artist={_escape(artist)}",
        f"genre={_escape(genre)}",
    ]
    for span in spans:
        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={span.start_ms}",
            f"END={span.end_ms}",
            f"title={_escape(span.title)}",
        ]
    return "\n".join(lines) + "\n"


def export_m4b(
    manifest_path: str | Path,
    out_path: str | Path,
    bitrate: str = DEFAULT_BITRATE,
) -> ExportResult:
    """Concatenate the chapter MP3s and encode one chaptered AAC ``.m4b``."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg not found; install it (brew install ffmpeg) or add it to PATH"
        )
    manifest, root = load_manifest(manifest_path)
    spans = chapter_spans(manifest, root)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=out_path.parent, prefix=".export-") as tmp:
        directory = Path(tmp)
        listing = directory / "chapters.txt"
        listing.write_text(
            "".join(
                f"file '{span.path.resolve().as_posix()}'\n" for span in spans
            ),
            encoding="utf-8",
        )
        metadata = directory / "chapters.ffmetadata"
        metadata.write_text(
            build_ffmetadata(manifest, spans), encoding="utf-8"
        )

        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-i",
            str(metadata),
            "-map",
            "0:a",
            "-map_metadata",
            "1",
            "-map_chapters",
            "1",
            "-c:a",
            "aac",
            "-b:a",
            bitrate,
            "-ac",
            "1",
            "-ar",
            "24000",
            "-metadata:s:a:0",
            "language=eng",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed ({completed.returncode}): "
                f"{completed.stderr.strip()}"
            )

    duration = subprocess.run(
        [
            shutil.which("ffprobe") or "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        measured = float(duration.stdout.strip())
    except ValueError:
        measured = spans[-1].end_ms / 1000
    written = probe_chapters(out_path)
    if written != len(spans):
        raise RuntimeError(
            f"expected {len(spans)} chapters in {out_path.name}, found {written}"
        )
    return ExportResult(
        m4b=out_path,
        chapters=written,
        duration_s=measured,
        size_bytes=out_path.stat().st_size,
        bitrate=bitrate,
    )


def probe_chapters(path: str | Path) -> int:
    """Number of chapter markers actually written to the container."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return 0
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_chapters",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return len(json.loads(completed.stdout).get("chapters", []))
    except json.JSONDecodeError:
        return 0


def _escape(value: str) -> str:
    """Escape the characters ffmetadata treats as syntax."""
    for character in ("\\", "=", ";", "#"):
        value = value.replace(character, "\\" + character)
    return value.replace("\n", "\\\n")


__all__ = [
    "ChapterSpan",
    "ExportResult",
    "build_ffmetadata",
    "chapter_spans",
    "export_m4b",
    "load_manifest",
    "probe_chapters",
]
