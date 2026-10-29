"""Engine-independent TTS settings and result types."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

DEFAULT_MODEL = "LiquidAI/LFM3-Audio-2B-MLX-8bit"


@dataclass(frozen=True)
class TTSSettings:
    """Everything that can change the generated voice or its sampling."""

    model_id: str = DEFAULT_MODEL
    language: str = "American English"
    voice: str = "female"
    audio_temperature: float = 1.0
    audio_top_k: int = 128
    audio_repetition_penalty: float = 1.1
    audio_repetition_window: int = 4096
    max_tokens: int | None = None

    def system_prompt(self) -> str:
        return f"Perform TTS in {self.language} using a {self.voice} voice."

    def as_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "language": self.language,
            "voice": self.voice,
            "audio_temperature": self.audio_temperature,
            "audio_top_k": self.audio_top_k,
            "audio_repetition_penalty": self.audio_repetition_penalty,
            "audio_repetition_window": self.audio_repetition_window,
            "max_tokens": self.max_tokens,
        }

    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


@dataclass
class Synthesis:
    path: Path
    text: str
    duration_s: float
    generation_s: float
    sample_rate: int
    peak_memory_gb: float
    finish_reason: str = "stop"
    cached: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def realtime_factor(self) -> float:
        if self.generation_s <= 0:
            return 0.0
        return self.duration_s / self.generation_s


@runtime_checkable
class TTSEngine(Protocol):
    settings: TTSSettings
    cache: object | None
    sample_rate: int

    def synthesize(
        self,
        text: str,
        out_path: Path,
        max_tokens: int | None = None,
    ) -> Synthesis:
        """Generate speech for ``text`` and write a mono WAV to ``out_path``."""
        ...
