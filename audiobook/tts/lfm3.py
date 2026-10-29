"""LFM3-Audio-2B speech synthesis on MLX.

Imports of ``mlx_vlm``/``mlx`` happen inside methods: pass 1 must run without
the model stack and the CLI stays fast to start.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..chunk import slots_for
from ..wav import write_pcm16
from .base import Synthesis, TTSSettings
from .cache import AudioCache

TTS_SYSTEM_PROMPT = "Perform TTS in {language} using a {voice} voice."


class LFM3Engine:
    """Speech engine backed by ``LiquidAI/LFM3-Audio-2B`` (MLX)."""

    def __init__(
        self,
        settings: TTSSettings | None = None,
        cache: AudioCache | None = None,
        refresh: bool = False,
    ):
        self.settings = settings or TTSSettings()
        self.cache = cache
        self.refresh = refresh
        self.sample_rate = 24_000
        self.load_peak_memory_gb = 0.0
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------ load

    def load(self) -> None:
        if self._model is not None:
            return
        import mlx.core as mx
        from mlx_vlm import load

        mx.reset_peak_memory()
        self._model, self._processor = load(self.settings.model_id)
        self.load_peak_memory_gb = mx.get_peak_memory() / 1e9

    def close(self) -> None:
        self._model = None
        self._processor = None
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
        budget = (
            max_tokens
            or self.settings.max_tokens
            or slots_for(text)
        )
        key = None
        if self.cache is not None:
            key = self.cache.key(
                self.settings.fingerprint(), text, budget
            )
            entry = self.cache.lookup(key) if not self.refresh else None
            if entry is not None:
                self.cache.copy_from_cache(entry, out_path)
                duration = _duration(entry.wav)
                return Synthesis(
                    path=out_path,
                    text=text,
                    duration_s=duration,
                    generation_s=0.0,
                    sample_rate=self.sample_rate,
                    peak_memory_gb=float(entry.meta.get("peak_memory_gb", 0.0)),
                    finish_reason=str(entry.meta.get("finish_reason", "stop")),
                    cached=True,
                )

        self.load()
        import mlx.core as mx
        from mlx_vlm import generate_audio

        prompt = self._prompt(text)
        mx.reset_peak_memory()
        started = time.perf_counter()
        result = generate_audio(
            self._model,
            self._processor,
            prompt,
            generation_mode="audio",
            max_tokens=budget,
            audio_temperature=self.settings.audio_temperature,
            audio_top_k=self.settings.audio_top_k,
            audio_repetition_penalty=self.settings.audio_repetition_penalty,
            audio_repetition_window=self.settings.audio_repetition_window,
        )
        generation_s = time.perf_counter() - started

        samples = np.asarray(result.audio, dtype=np.float32).squeeze()
        sample_rate = int(result.sample_rate or self.sample_rate)
        write_pcm16(out_path, samples, sample_rate)
        duration = samples.size / sample_rate
        peak_memory = mx.get_peak_memory() / 1e9
        warnings = _quality_warnings(text, duration, result.finish_reason, budget)
        synthesis = Synthesis(
            path=out_path,
            text=text,
            duration_s=duration,
            generation_s=generation_s,
            sample_rate=sample_rate,
            peak_memory_gb=peak_memory,
            finish_reason=result.finish_reason,
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
                    "finish_reason": result.finish_reason,
                },
            )
        return synthesis

    def _prompt(self, text: str) -> str:
        messages = [
            {"role": "system", "content": self.settings.system_prompt()},
            {"role": "user", "content": text},
            {"role": "assistant", "content": "<|audio_start|>"},
        ]
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        return tokenizer.apply_chat_template(
            messages, continue_final_message=True, tokenize=False
        )


def _duration(path: Path) -> float:
    from ..wav import duration

    return duration(path)


def _quality_warnings(
    text: str, duration_s: float, finish_reason: str, budget: int
) -> list[str]:
    """Cheap signals that a sample may be broken (the report lists them)."""
    warnings: list[str] = []
    if finish_reason != "stop":
        warnings.append(
            f"generation hit the {budget}-slot limit; audio may be cut off"
        )
    expected = max(1.0, len(text) / 14.0)
    if duration_s < expected * 0.45:
        warnings.append("audio is much shorter than expected; words may be missing")
    if duration_s > expected * 2.2:
        warnings.append("audio is much longer than expected; possible repetition")
    return warnings
