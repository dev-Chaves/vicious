"""Speech synthesis interfaces and settings.

Only this package knows about the model. ``audiobook.tts.lfm3`` imports MLX
lazily so ``script``, ``chunk`` and tests stay importable without the model
stack installed.
"""

from __future__ import annotations

from .base import DEFAULT_MODEL, Synthesis, TTSEngine, TTSSettings
from .cache import AudioCache
from .kokoro import DEFAULT_KOKORO_MODEL, KokoroEngine, KokoroSettings

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_KOKORO_MODEL",
    "AudioCache",
    "KokoroEngine",
    "KokoroSettings",
    "Synthesis",
    "TTSEngine",
    "TTSSettings",
]
