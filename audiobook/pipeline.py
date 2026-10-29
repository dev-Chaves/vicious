"""Pass 2 as a plain function: ``script.json`` → chapter MP3.

The CLI is a thin wrapper around :func:`run_speak`, and the book runner calls
the same function once per chapter with a shared engine, so both paths share
chunking, caching, synthesising, assembly and reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .assemble import Assembly, assemble_mp3
from .chunk import DEFAULT_MAX_CHARS, DEFAULT_MIN_CHARS, Chunk, chunk_script, slots_for
from .schema import Script
from .tts import AudioCache, TTSEngine
from .tts.base import Synthesis

DEFAULT_GAP_WITHIN = 0.18
DEFAULT_GAP_BETWEEN = 0.45


@dataclass(frozen=True)
class SpeakOptions:
    """Everything ``run_speak`` needs besides the script and the engine."""

    out: Path
    work: Path | None = None
    cache_dir: Path = Path(".cache/tts")
    use_cache: bool = True
    refresh: bool = False
    start: int = 1
    limit: int = 0
    gap_within: float = DEFAULT_GAP_WITHIN
    gap_between: float = DEFAULT_GAP_BETWEEN
    bitrate: str = "96k"
    max_chars: int = DEFAULT_MAX_CHARS
    min_chars: int = DEFAULT_MIN_CHARS
    normalize: bool = True
    target_lufs: float = -19.0
    true_peak: float = -2.0
    loudness_passes: int = 1


@dataclass
class ChapterResult:
    script: Script
    chunks: list[Chunk]
    selected: list[Chunk]
    syntheses: list[Synthesis]
    assembly: Assembly
    load_peak_memory_gb: float = 0.0

    @property
    def speech_s(self) -> float:
        return sum(item.duration_s for item in self.syntheses)

    @property
    def generated_s(self) -> float:
        return sum(item.generation_s for item in self.syntheses)

    @property
    def cached(self) -> int:
        return sum(1 for item in self.syntheses if item.cached)

    @property
    def realtime_factor(self) -> float:
        generated = self.generated_s
        return self.speech_s / generated if generated > 0 else 0.0

    @property
    def peak_memory_gb(self) -> float:
        per_chunk = max(
            (item.peak_memory_gb for item in self.syntheses if not item.cached),
            default=0.0,
        )
        return max(self.load_peak_memory_gb, per_chunk)

    @property
    def warnings(self) -> list[tuple[str, str]]:
        return [
            (chunk.id, warning)
            for chunk, item in zip(self.selected, self.syntheses)
            for warning in item.warnings
        ]


def run_speak(
    script_path: str | Path,
    options: SpeakOptions,
    engine: TTSEngine,
    cache: AudioCache | None = None,
    on_chunk: Callable[[int, int, Chunk, Synthesis], None] | None = None,
) -> ChapterResult:
    """Chunk, synthesise and assemble one chapter.

    The engine is supplied by the caller and is **not** closed here: a book run
    loads the model once and reuses it for every chapter.
    """
    script = Script.load(script_path)
    chunks = chunk_script(
        script, max_chars=options.max_chars, min_chars=options.min_chars
    )
    if not chunks:
        raise ValueError("script has no narrated text")
    if options.start > len(chunks):
        raise ValueError(
            f"--start {options.start} is past the last chunk ({len(chunks)})"
        )
    selected = chunks[options.start - 1 :]
    if options.limit:
        selected = selected[: options.limit]

    if cache is None and options.use_cache:
        cache = AudioCache(options.cache_dir)

    work_dir = options.work or options.out.parent / f"{options.out.stem}.chunks"
    work_dir.mkdir(parents=True, exist_ok=True)

    syntheses: list[Synthesis] = []
    total = len(selected)
    for index, chunk in enumerate(selected, start=1):
        wav_path = work_dir / f"{chunk.id}.wav"
        synthesis = engine.synthesize(chunk.text, wav_path)
        syntheses.append(synthesis)
        if on_chunk is not None:
            on_chunk(index, total, chunk, synthesis)

    gaps = [
        options.gap_between
        if left.segment_id != right.segment_id
        else options.gap_within
        for left, right in zip(selected, selected[1:])
    ]
    assembly = assemble_mp3(
        [item.path for item in syntheses],
        options.out,
        gaps=gaps,
        title=script.source.title,
        album=script.source.book or None,
        bitrate=options.bitrate,
        target_lufs=options.target_lufs if options.normalize else None,
        true_peak=options.true_peak,
        loudness_passes=options.loudness_passes,
    )
    return ChapterResult(
        script=script,
        chunks=chunks,
        selected=selected,
        syntheses=syntheses,
        assembly=assembly,
        load_peak_memory_gb=getattr(engine, "load_peak_memory_gb", 0.0),
    )


def live_cache_keys(
    scripts: Iterable[Script],
    cache: AudioCache,
    fingerprint: str,
    engine_name: str,
) -> set[str]:
    """Keys a given engine configuration would look up for these scripts."""
    keys: set[str] = set()
    for script in scripts:
        for chunk in chunk_script(script):
            budget = slots_for(chunk.text) if engine_name == "lfm3" else 0
            keys.add(cache.key(fingerprint, chunk.text, budget))
    return keys


__all__ = [
    "ChapterResult",
    "SpeakOptions",
    "live_cache_keys",
    "run_speak",
]
