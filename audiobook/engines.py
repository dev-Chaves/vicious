"""Engine selection, shared by the CLI and the book runner."""

from __future__ import annotations

from dataclasses import dataclass

from .tts import (
    DEFAULT_KOKORO_MODEL,
    DEFAULT_MODEL,
    AudioCache,
    KokoroEngine,
    KokoroSettings,
    TTSEngine,
    TTSSettings,
)
from .tts.kokoro import DEFAULT_VOICE, lang_code_for, resolve_voice

ENGINE_NAMES = ("kokoro", "lfm3")


@dataclass(frozen=True)
class EngineOptions:
    """User-facing engine choice, independent of the concrete classes."""

    name: str = "kokoro"
    voice: str = ""
    speed: float = 1.0
    lfm3_model: str = DEFAULT_MODEL
    kokoro_model: str = DEFAULT_KOKORO_MODEL
    language: str = "American English"

    def validate(self) -> str:
        name = self.name.strip().lower()
        if name not in ENGINE_NAMES:
            raise ValueError(f"unknown engine {self.name!r} (use kokoro or lfm3)")
        if name == "kokoro":
            resolve_voice(self.voice or DEFAULT_VOICE)
        return name

    def describe(self) -> str:
        name = self.validate()
        if name == "kokoro":
            voice = resolve_voice(self.voice or DEFAULT_VOICE)
            return f"kokoro {voice} at {self.speed:.2f}x"
        return f"lfm3 {self.voice or 'female'}"

    def as_dict(self) -> dict[str, object]:
        name = self.validate()
        if name == "kokoro":
            voice = resolve_voice(self.voice or DEFAULT_VOICE)
        else:
            voice = self.voice or "female"
        return {"name": name, "voice": voice, "speed": self.speed}

    def fingerprint(self) -> str:
        name = self.validate()
        if name == "kokoro":
            voice = resolve_voice(self.voice or DEFAULT_VOICE)
            return KokoroSettings(
                model_id=self.kokoro_model,
                voice=voice,
                speed=self.speed,
                lang_code=lang_code_for(voice),
            ).fingerprint()
        return TTSSettings(
            model_id=self.lfm3_model,
            language=self.language,
            voice=self.voice or "female",
        ).fingerprint()


def create_engine(
    options: EngineOptions,
    cache: AudioCache | None = None,
    refresh: bool = False,
) -> tuple[TTSEngine, str]:
    """Build the engine plus a short human-readable label for reports."""
    name = options.validate()
    if name == "kokoro":
        voice = resolve_voice(options.voice or DEFAULT_VOICE)
        settings = KokoroSettings(
            model_id=options.kokoro_model,
            voice=voice,
            speed=options.speed,
            lang_code=lang_code_for(voice),
        )
        engine = KokoroEngine(settings, cache=cache, refresh=refresh)
        return engine, f"kokoro {settings.voice} at {settings.speed:.2f}x"

    settings = TTSSettings(
        model_id=options.lfm3_model,
        language=options.language,
        voice=options.voice or "female",
    )
    from .tts.lfm3 import LFM3Engine

    engine = LFM3Engine(settings, cache=cache, refresh=refresh)
    return engine, f"lfm3 {settings.voice}"


__all__ = ["ENGINE_NAMES", "EngineOptions", "create_engine"]
