"""The ``script.json`` schema, shared by both passes.

This module deliberately depends on nothing but the standard library and
:mod:`audiobook.types`: pass 2 must be able to read a script without importing
PyMuPDF, classification or narration code.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import Origin, SegmentType

SCHEMA_VERSION = 1


class ScriptError(ValueError):
    """Raised when a script file cannot be read or is not valid."""


@dataclass
class Segment:
    id: str
    type: SegmentType
    text_original: str
    text_narrated: str
    origin: Origin
    page: int
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "type": self.type.value,
            "text_original": self.text_original,
            "text_narrated": self.text_narrated,
            "origin": self.origin.value,
            "page": self.page,
        }
        if self.label:
            data["label"] = self.label
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int) -> Segment:
        if not isinstance(data, dict):
            raise ScriptError(f"segment {index}: expected a JSON object")
        try:
            return cls(
                id=str(data["id"]),
                type=SegmentType(data["type"]),
                text_original=str(data["text_original"]),
                text_narrated=str(data["text_narrated"]),
                origin=Origin(data["origin"]),
                page=int(data["page"]),
                label=data.get("label"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ScriptError(f"segment {index}: invalid segment ({error})") from error


@dataclass
class Source:
    pdf: str
    sha256: str
    chapter: int
    title: str
    first_page: int
    last_page: int
    book: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pdf": self.pdf,
            "sha256": self.sha256,
            "book": self.book,
            "chapter": self.chapter,
            "title": self.title,
            "first_page": self.first_page,
            "last_page": self.last_page,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Source:
        if not isinstance(data, dict):
            raise ScriptError("invalid source block")
        try:
            return cls(
                pdf=str(data["pdf"]),
                sha256=str(data["sha256"]),
                chapter=int(data["chapter"]),
                title=str(data["title"]),
                first_page=int(data["first_page"]),
                last_page=int(data["last_page"]),
                book=str(data.get("book", "")),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ScriptError(f"invalid source block ({error})") from error


@dataclass
class Script:
    source: Source
    segments: list[Segment] = field(default_factory=list)
    version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source": self.source.to_dict(),
            "segments": [segment.to_dict() for segment in self.segments],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = self.to_json()
        handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(content)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        return path

    @classmethod
    def load(cls, path: str | Path) -> Script:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"script not found: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ScriptError(f"{path}: invalid JSON ({error})") from error
        if not isinstance(data, dict):
            raise ScriptError(f"{path}: expected a JSON object")
        version = data.get("version")
        if version != SCHEMA_VERSION:
            raise ScriptError(
                f"{path}: unsupported script version {version!r} "
                f"(expected {SCHEMA_VERSION})"
            )
        segments = data.get("segments")
        if not isinstance(segments, list):
            raise ScriptError(f"{path}: missing segment list")
        return cls(
            source=Source.from_dict(data.get("source") or {}),
            segments=[
                Segment.from_dict(segment, index)
                for index, segment in enumerate(segments)
            ],
        )

    @property
    def narrated(self) -> list[Segment]:
        return [
            segment
            for segment in self.segments
            if segment.origin is not Origin.SKIPPED and segment.text_narrated.strip()
        ]

    @property
    def word_count(self) -> int:
        return sum(len(segment.text_narrated.split()) for segment in self.narrated)

    @property
    def skipped_count(self) -> int:
        return sum(
            1 for segment in self.segments if segment.origin is Origin.SKIPPED
        )
