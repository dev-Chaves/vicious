"""Narration: turn classified items into the exact words the TTS will say.

Milestone 1 uses rules only. The :class:`Narrator` protocol is the extension
point for an LLM: a future ``LLMNarrator`` can rewrite code, figures and
awkward prose while everything else (cache keys, chunking, assembly) stays
the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .clean import ensure_sentence, normalize_for_speech
from .types import SegmentType

CODE_ANNOUNCEMENT = "Code listing, see the book."


@dataclass(frozen=True)
class NarrationRequest:
    kind: SegmentType
    text: str
    label: str | None = None
    caption: str | None = None


@runtime_checkable
class Narrator(Protocol):
    def narrate(self, request: NarrationRequest) -> str:
        """Return the sentence(s) to speak for one item."""
        ...


class RuleNarrator:
    """Deterministic narration for prose, headings, code, figures and captions."""

    def narrate(self, request: NarrationRequest) -> str:
        if request.kind is SegmentType.CODE:
            return CODE_ANNOUNCEMENT
        if request.kind in (SegmentType.FIGURE, SegmentType.TABLE):
            return self.reference(request.label)
        if request.kind is SegmentType.CAPTION:
            body = request.caption or request.text
            return ensure_sentence(normalize_for_speech(body))
        return ensure_sentence(normalize_for_speech(request.text))

    @staticmethod
    def reference(label: str | None) -> str:
        """``Figure 1-1, see the book.`` / ``Table 5-2, see the book.``"""
        name = label or "Figure"
        return f"{normalize_for_speech(name)}, see the book."
