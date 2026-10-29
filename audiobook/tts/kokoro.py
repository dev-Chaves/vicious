"""Kokoro-82M speech synthesis on MLX.

Kokoro is deterministic and phoneme-based: no sampling, no temperature. It is
much smaller than LFM3-Audio, so it trades expressiveness for stability. The
engine implements the same :class:`~audiobook.tts.base.TTSEngine` contract, so
chunking, caching and assembly are unchanged and a chapter can be synthesized
with either engine from the same ``script.json``.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..wav import write_pcm16
from .base import Synthesis
from .cache import AudioCache

DEFAULT_KOKORO_MODEL = "mlx-community/Kokoro-82M-bf16"
DEFAULT_VOICE = "af_heart"


@dataclass(frozen=True)
class KokoroSettings:
    """Everything that can change Kokoro's voice or delivery."""

    model_id: str = DEFAULT_KOKORO_MODEL
    voice: str = DEFAULT_VOICE
    speed: float = 1.0
    lang_code: str = "a"

    def as_dict(self) -> dict[str, object]:
        return {
            "engine": "kokoro",
            "model_id": self.model_id,
            "voice": self.voice,
            "speed": self.speed,
            "lang_code": self.lang_code,
        }

    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


class KokoroEngine:
    """Speech engine backed by ``hexgrad/Kokoro-82M`` converted to MLX."""

    def __init__(
        self,
        settings: KokoroSettings | None = None,
        cache: AudioCache | None = None,
        refresh: bool = False,
    ):
        self.settings = settings or KokoroSettings()
        self.cache = cache
        self.refresh = refresh
        self.sample_rate = 24_000
        self.load_peak_memory_gb = 0.0
        self._model = None

    # ------------------------------------------------------------------ load

    def load(self) -> None:
        if self._model is not None:
            return
        import mlx.core as mx
        from mlx_audio.tts.utils import load

        mx.reset_peak_memory()
        self._model = load(self.settings.model_id)
        self.load_peak_memory_gb = mx.get_peak_memory() / 1e9

    def close(self) -> None:
        self._model = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:  # pragma: no cover - best effort cleanup
            pass

    # -------------------------------------------------------------- synthesis

    def synthesize(
        self,
        text: str,
        out_path: Path,
        max_tokens: int | None = None,
    ) -> Synthesis:
        text = " ".join(text.split())
        if not text:
            raise ValueError("cannot synthesize empty text")
        out_path = Path(out_path)

        key = None
        if self.cache is not None:
            # Kokoro has no sampling budget; the zero keeps the cache key
            # shape identical to the LFM3 engine's.
            key = self.cache.key(self.settings.fingerprint(), text, 0)
            entry = self.cache.lookup(key) if not self.refresh else None
            if entry is not None:
                self.cache.copy_from_cache(entry, out_path)
                return Synthesis(
                    path=out_path,
                    text=text,
                    duration_s=float(entry.meta.get("duration_s", 0.0)),
                    generation_s=0.0,
                    sample_rate=self.sample_rate,
                    peak_memory_gb=float(entry.meta.get("peak_memory_gb", 0.0)),
                    finish_reason=str(entry.meta.get("finish_reason", "stop")),
                    cached=True,
                )

        self.load()
        import mlx.core as mx

        mx.reset_peak_memory()
        started = time.perf_counter()
        parts = list(
            self._model.generate(
                text,
                voice=self.settings.voice,
                speed=self.settings.speed,
                lang_code=self.settings.lang_code,
            )
        )
        generation_s = time.perf_counter() - started
        if not parts:
            raise RuntimeError("Kokoro returned no audio")

        samples = np.concatenate([np.asarray(part.audio) for part in parts])
        sample_rate = int(parts[0].sample_rate or self.sample_rate)
        write_pcm16(out_path, samples, sample_rate)
        duration = samples.size / sample_rate
        peak_memory = mx.get_peak_memory() / 1e9
        warnings = _quality_warnings(text, duration)

        synthesis = Synthesis(
            path=out_path,
            text=text,
            duration_s=duration,
            generation_s=generation_s,
            sample_rate=sample_rate,
            peak_memory_gb=peak_memory,
            finish_reason="stop",
            cached=False,
            warnings=warnings,
        )
        if self.cache is not None and key is not None:
            self.cache.store(
                key,
                out_path,
                text=text,
                settings=self.settings.as_dict(),
                stats={
                    "duration_s": round(duration, 3),
                    "generation_s": round(generation_s, 3),
                    "sample_rate": sample_rate,
                    "peak_memory_gb": round(peak_memory, 3),
                    "finish_reason": "stop",
                },
            )
        return synthesis


def _quality_warnings(text: str, duration_s: float) -> list[str]:
    """Cheap signals that a sample may be broken (the report lists them)."""
    warnings: list[str] = []
    expected = max(1.0, len(text) / 14.0)
    if duration_s < expected * 0.45:
        warnings.append("audio is much shorter than expected; words may be missing")
    if duration_s > expected * 2.2:
        warnings.append("audio is much longer than expected; possible repetition")
    return warnings
