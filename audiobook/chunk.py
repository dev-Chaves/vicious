"""Split narrated text into TTS-sized chunks.

chunks are 10-40 seconds of speech: long enough for natural prosody, short
enough that a bad sample can be regenerated cheaply and that the model's
repetition failure mode stays contained. Chunk ids are stable so the cache
survives re-runs.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .schema import Script
from .types import Origin

DEFAULT_MAX_CHARS = 600
DEFAULT_MIN_CHARS = 120

_SENTENCE_SPLIT_RE = re.compile(
    r"(?:(?<=[.!?])|(?<=[.!?][\"')\]]))\s+"
)
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[,;:])\s+")


@dataclass(frozen=True)
class Chunk:
    id: str
    segment_id: str
    page: int
    text: str

    @property
    def chars(self) -> int:
        return len(self.text)


def chunk_script(
    script: Script,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for segment in script.segments:
        if segment.origin is Origin.SKIPPED:
            continue
        text = segment.text_narrated.strip()
        if not text:
            continue
        for index, piece in enumerate(
            split_text(text, max_chars=max_chars, min_chars=min_chars), start=1
        ):
            chunks.append(
                Chunk(
                    id=f"{segment.id}-c{index:04d}",
                    segment_id=segment.id,
                    page=segment.page,
                    text=piece,
                )
            )
    return chunks


def split_text(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[str]:
    units: list[str] = []
    for sentence in _sentences(text):
        if len(sentence) <= max_chars:
            units.append(sentence)
            continue
        for clause in _CLAUSE_SPLIT_RE.split(sentence):
            clause = clause.strip()
            if not clause:
                continue
            if len(clause) <= max_chars:
                units.append(clause)
            else:
                units.extend(_split_words(clause, max_chars))

    chunks = _pack(units, max_chars)
    return _merge_tail(chunks, max_chars, min_chars)


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]


def _split_words(text: str, max_chars: int) -> list[str]:
    words = text.split()
    pieces: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if current and len(candidate) > max_chars:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _pack(units: list[str], max_chars: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current} {unit}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _merge_tail(chunks: list[str], max_chars: int, min_chars: int) -> list[str]:
    if len(chunks) < 2:
        return chunks
    if len(chunks[-1]) >= min_chars:
        return chunks
    merged = f"{chunks[-2]} {chunks[-1]}"
    if len(merged) <= max_chars:
        return [*chunks[:-2], merged]
    return chunks


def estimate_seconds(text: str, chars_per_second: float = 14.0) -> float:
    """Rough duration estimate used to size the sampling budget."""
    return max(1.0, len(text) / chars_per_second)


def slots_for(text: str, chars_per_second: float = 14.0) -> int:
    """Speech frames (80 ms each) to allow, with headroom for slow delivery."""
    seconds = estimate_seconds(text, chars_per_second)
    return max(256, min(2048, math.ceil(seconds * 12.5 * 2.0) + 32))
