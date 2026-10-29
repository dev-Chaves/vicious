"""Whole-book runner: chapter 1..N in one process, with resume.

The runner reuses the same engine instance for every chapter (the model is
loaded once), rebuilds ``script.json`` only when its content changed, and keeps
a manifest at ``<out>/book.json`` so an interrupted run can continue where it
stopped.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol, Sequence

from .chunk import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MIN_CHARS,
    chunk_script,
)
from .engines import EngineOptions, create_engine
from .extract import (
    Chapter,
    Document,
    chapter_numbers,
    extract_chapter_pages,
    find_chapter,
    open_document,
)
from .narrate import RuleNarrator
from .pipeline import SpeakOptions, run_speak
from .schema import Script
from .script import build_script_from_chapter
from .tts import AudioCache, TTSEngine

MANIFEST_VERSION = 1
MANIFEST_NAME = "book.json"
BOOK_MP3_NAME = "book.mp3"

_LOUDNESS_RE = re.compile(r"I:\s+(-?\d+(?:\.\d+)?) LUFS")
_PEAK_RE = re.compile(r"Peak:\s+(-?\d+(?:\.\d+)?) dBFS")


@dataclass(frozen=True)
class BookOptions:
    pdf: Path
    out_dir: Path = Path("build")
    chapters: Sequence[int] | None = None
    engine: EngineOptions = field(default_factory=EngineOptions)
    cache_dir: Path = Path(".cache/tts")
    use_cache: bool = True
    refresh: bool = False
    force: bool = False
    gap_within: float = 0.18
    gap_between: float = 0.45
    bitrate: str = "96k"
    max_chars: int = DEFAULT_MAX_CHARS
    min_chars: int = DEFAULT_MIN_CHARS
    normalize: bool = True
    target_lufs: float = -19.0
    true_peak: float = -2.0
    loudness_passes: int = 1

    @property
    def manifest_path(self) -> Path:
        return self.out_dir / MANIFEST_NAME

    @property
    def book_mp3(self) -> Path:
        return self.out_dir / BOOK_MP3_NAME


@dataclass
class ChapterRecord:
    number: int
    title: str
    first_page: int
    last_page: int
    status: str = "pending"
    chunks: int = 0
    duration_s: float = 0.0
    speech_s: float = 0.0
    generation_s: float = 0.0
    wall_s: float = 0.0
    loudness_lufs: float | None = None
    true_peak_dbtp: float | None = None
    peak_memory_gb: float = 0.0
    warnings: int = 0
    script: str = ""
    mp3: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "number": self.number,
            "title": self.title,
            "first_page": self.first_page,
            "last_page": self.last_page,
            "status": self.status,
            "chunks": self.chunks,
            "duration_s": round(self.duration_s, 3),
            "speech_s": round(self.speech_s, 3),
            "generation_s": round(self.generation_s, 3),
            "wall_s": round(self.wall_s, 3),
            "loudness_lufs": (
                None if self.loudness_lufs is None else round(self.loudness_lufs, 2)
            ),
            "true_peak_dbtp": (
                None if self.true_peak_dbtp is None else round(self.true_peak_dbtp, 2)
            ),
            "peak_memory_gb": round(self.peak_memory_gb, 2),
            "warnings": self.warnings,
            "script": self.script,
            "mp3": self.mp3,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ChapterRecord:
        return cls(
            number=int(data.get("number", 0)),
            title=str(data.get("title", "")),
            first_page=int(data.get("first_page", 0)),
            last_page=int(data.get("last_page", 0)),
            status=str(data.get("status", "pending")),
            chunks=int(data.get("chunks", 0)),
            duration_s=float(data.get("duration_s", 0.0)),
            speech_s=float(data.get("speech_s", 0.0)),
            generation_s=float(data.get("generation_s", 0.0)),
            wall_s=float(data.get("wall_s", 0.0)),
            loudness_lufs=_optional_float(data.get("loudness_lufs")),
            true_peak_dbtp=_optional_float(data.get("true_peak_dbtp")),
            peak_memory_gb=float(data.get("peak_memory_gb", 0.0)),
            warnings=int(data.get("warnings", 0)),
            script=str(data.get("script", "")),
            mp3=str(data.get("mp3", "")),
            error=str(data.get("error", "")),
        )


@dataclass
class BookResult:
    manifest_path: Path
    chapters: list[ChapterRecord]
    book_mp3: Path | None = None
    all_chapters: list[ChapterRecord] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return sum(record.duration_s for record in (self.all_chapters or self.chapters))

    @property
    def generated_s(self) -> float:
        return sum(record.generation_s for record in self.chapters)

    @property
    def warnings(self) -> int:
        return sum(record.warnings for record in self.chapters)

    @property
    def failed(self) -> list[ChapterRecord]:
        return [record for record in self.chapters if record.status == "failed"]


class BookSource(Protocol):
    """Where chapters come from; a PDF in production, synthetic text in tests."""

    document: Document

    def chapter(self, number: int) -> Chapter: ...


class PdfBookSource:
    """Opens the PDF once for metadata and hashes, then per chapter for pages."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.document = open_document(self.path)

    def chapter(self, number: int) -> Chapter:
        chapter_range = find_chapter(self.document, number)
        return extract_chapter_pages(self.path, chapter_range)

    def available(self) -> list[int]:
        return chapter_numbers(self.document)


def _available_chapters(source: BookSource, document: Document) -> list[int]:
    provider = getattr(source, "available", None)
    if callable(provider):
        return list(provider())
    return chapter_numbers(document)


class _EngineHolder:
    """Create the engine only when a chapter actually needs synthesis."""

    def __init__(
        self,
        options: BookOptions,
        engine: TTSEngine | None,
        label: str = "",
    ):
        self.options = options
        self.engine = engine
        self.label = label
        self.cache = (
            AudioCache(options.cache_dir) if options.use_cache else None
        )
        self.created_here = False

    def get(self) -> tuple[TTSEngine, str]:
        if self.engine is None:
            self.engine, self.label = create_engine(
                self.options.engine, cache=self.cache, refresh=self.options.refresh
            )
            self.created_here = True
        return self.engine, self.label

    def close(self) -> None:
        if self.engine is not None and self.created_here:
            self.engine.close()


def run_book(
    options: BookOptions,
    engine: TTSEngine | None = None,
    source: BookSource | None = None,
    on_chapter: Callable[[ChapterRecord, int, int], None] | None = None,
) -> BookResult:
    """Build scripts and MP3s for every requested chapter, resuming as needed."""
    options.out_dir.mkdir(parents=True, exist_ok=True)
    book_source = source or PdfBookSource(options.pdf)
    document = book_source.document
    known = _available_chapters(book_source, document)
    available = list(options.chapters) if options.chapters is not None else known
    if not available:
        raise ValueError("the PDF table of contents has no numbered chapters")
    missing = [number for number in available if number not in set(known)]
    if missing:
        raise ValueError(
            "chapter(s) not found in the PDF TOC: "
            + ", ".join(str(number) for number in missing)
        )

    previous = _load_manifest(options.manifest_path)
    merged: dict[int, ChapterRecord] = dict(previous)
    holder = _EngineHolder(options, engine)
    records: list[ChapterRecord] = []
    total = len(available)
    started_manifest = time.perf_counter()
    try:
        for index, number in enumerate(available, start=1):
            record = _process_chapter(
                number=number,
                options=options,
                source=book_source,
                document=document,
                previous=previous.get(number),
                holder=holder,
            )
            records.append(record)
            merged[number] = record
            _write_manifest(
                options.manifest_path,
                options,
                document,
                _ordered(merged),
                started_manifest,
            )
            if on_chapter is not None:
                on_chapter(record, index, total)
    finally:
        holder.close()

    all_records = _ordered(merged)
    book_mp3 = _concat_chapters(options, all_records, document.title)
    _write_manifest(
        options.manifest_path, options, document, all_records, started_manifest
    )
    return BookResult(
        manifest_path=options.manifest_path,
        chapters=records,
        book_mp3=book_mp3,
        all_chapters=all_records,
    )


def _ordered(records: dict[int, ChapterRecord]) -> list[ChapterRecord]:
    return [records[number] for number in sorted(records)]


def _process_chapter(
    number: int,
    options: BookOptions,
    source: BookSource,
    document: Document,
    previous: ChapterRecord | None,
    holder: _EngineHolder,
) -> ChapterRecord:
    chapter_dir = options.out_dir / f"ch{number:02d}"
    script_path = chapter_dir / "script.json"
    mp3_path = chapter_dir / f"ch{number:02d}.mp3"
    wall_started = time.perf_counter()
    try:
        chapter = source.chapter(number)
        script = build_script_from_chapter(
            document=document,
            chapter=chapter,
            source_pdf=str(document.path),
            narrator=RuleNarrator(),
        )
        script_changed = _write_script_if_changed(script_path, script)
        base = ChapterRecord(
            number=number,
            title=script.source.title,
            first_page=script.source.first_page,
            last_page=script.source.last_page,
            script=str(script_path.relative_to(options.out_dir)),
            mp3=str(mp3_path.relative_to(options.out_dir)),
        )
        if not options.force and not script_changed and mp3_path.exists():
            base.status = "skipped"
            if (
                previous is not None
                and previous.status in ("done", "skipped")
                and previous.duration_s > 0
                and previous.chunks > 0
            ):
                base.chunks = previous.chunks
                base.duration_s = previous.duration_s
                base.speech_s = previous.speech_s
                base.generation_s = previous.generation_s
                base.loudness_lufs = previous.loudness_lufs
                base.true_peak_dbtp = previous.true_peak_dbtp
                base.peak_memory_gb = previous.peak_memory_gb
                base.warnings = previous.warnings
            else:
                base.chunks = len(chunk_script(script))
                duration, loudness, peak = _probe_audio(mp3_path)
                base.duration_s = duration
                base.loudness_lufs = loudness
                base.true_peak_dbtp = peak
            base.wall_s = time.perf_counter() - wall_started
            return base

        generic_engine, _ = holder.get()
        result = run_speak(
            script_path,
            _speak_options(options, mp3_path),
            engine=generic_engine,
            cache=holder.cache,
        )
        base.status = "done"
        base.chunks = len(result.selected)
        base.duration_s = result.assembly.duration_s
        base.speech_s = result.speech_s
        base.generation_s = result.generated_s
        base.loudness_lufs = result.assembly.loudness_lufs
        base.true_peak_dbtp = result.assembly.true_peak_dbtp
        base.peak_memory_gb = result.peak_memory_gb
        base.warnings = len(result.warnings)
        base.wall_s = time.perf_counter() - wall_started
        return base
    except Exception as error:  # noqa: BLE001 - one bad chapter must not stop the book
        failed = previous or ChapterRecord(
            number=number,
            title=f"Chapter {number}",
            first_page=0,
            last_page=0,
        )
        failed.status = "failed"
        failed.error = str(error)
        failed.wall_s = time.perf_counter() - wall_started
        return failed


def _speak_options(options: BookOptions, mp3_path: Path) -> SpeakOptions:
    return SpeakOptions(
        out=mp3_path,
        cache_dir=options.cache_dir,
        use_cache=options.use_cache,
        refresh=options.refresh,
        gap_within=options.gap_within,
        gap_between=options.gap_between,
        bitrate=options.bitrate,
        max_chars=options.max_chars,
        min_chars=options.min_chars,
        normalize=options.normalize,
        target_lufs=options.target_lufs,
        true_peak=options.true_peak,
        loudness_passes=options.loudness_passes,
    )


def _write_script_if_changed(path: Path, script: Script) -> bool:
    new_text = script.to_json()
    if path.exists():
        try:
            if path.read_text(encoding="utf-8") == new_text:
                return False
        except OSError:
            pass
    script.save(path)
    return True


def _load_manifest(path: Path) -> dict[int, ChapterRecord]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    records: dict[int, ChapterRecord] = {}
    for item in data.get("chapters", []):
        if isinstance(item, dict):
            record = ChapterRecord.from_dict(item)
            records[record.number] = record
    return records


def _write_manifest(
    path: Path,
    options: BookOptions,
    document: Document,
    records: Sequence[ChapterRecord],
    started: float,
) -> None:
    payload = {
        "version": MANIFEST_VERSION,
        "pdf": str(document.path),
        "sha256": document.sha256,
        "book": document.title,
        "engine": options.engine.as_dict(),
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(time.perf_counter() - started, 1),
        "chapters": [record.to_dict() for record in records],
    }
    write_json_atomic(path, payload)


def write_json_atomic(path: str | Path, payload: object) -> Path:
    """Write JSON via a temporary file so an interrupt never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


def _concat_chapters(
    options: BookOptions, records: Sequence[ChapterRecord], title: str
) -> Path | None:
    """Join chapter MP3s into ``book.mp3`` without re-encoding."""
    chapters = [
        record
        for record in records
        if record.status in ("done", "skipped")
        and (options.out_dir / record.mp3).exists()
    ]
    if len(chapters) < 2:
        return None
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg not found; install it (brew install ffmpeg) or add it to PATH"
        )
    listing = options.out_dir / ".book-concat.txt"
    listing.write_text(
        "".join(
            f"file '{(options.out_dir / record.mp3).resolve().as_posix()}'\n"
            for record in chapters
        ),
        encoding="utf-8",
    )
    try:
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
            "-c",
            "copy",
            "-metadata",
            f"title={title}",
            "-metadata",
            f"album={title}",
            "-id3v2_version",
            "3",
            str(options.book_mp3),
        ]
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"ffmpeg concat failed: {completed.stderr.strip()}"
            )
        return options.book_mp3
    finally:
        listing.unlink(missing_ok=True)


def _probe_audio(path: Path) -> tuple[float, float | None, float | None]:
    """Duration, integrated loudness and true peak of an existing MP3."""
    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    duration = 0.0
    if ffprobe is not None:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            duration = float(completed.stdout.strip())
        except ValueError:
            duration = 0.0
    loudness: float | None = None
    peak: float | None = None
    if ffmpeg is not None:
        completed = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-nostats",
                "-i",
                str(path),
                "-af",
                "ebur128=peak=true",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            measured = _LOUDNESS_RE.findall(completed.stderr)
            true_peak = _PEAK_RE.findall(completed.stderr)
            if measured:
                loudness = float(measured[-1])
            if true_peak:
                peak = float(true_peak[-1])
    return duration, loudness, peak


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "BookOptions",
    "BookResult",
    "BookSource",
    "ChapterRecord",
    "PdfBookSource",
    "run_book",
    "write_json_atomic",
]
