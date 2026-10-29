"""Shared enums for the script schema."""

from __future__ import annotations

from enum import Enum


class SegmentType(str, Enum):
    PROSE = "prose"
    CODE = "code"
    FIGURE = "figure"
    TABLE = "table"
    CAPTION = "caption"


class Origin(str, Enum):
    RULE = "rule"
    LLM = "llm"
    SKIPPED = "skipped"
