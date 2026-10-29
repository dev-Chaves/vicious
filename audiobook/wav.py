"""Small WAV helpers: PCM16 chunks in, one concatenated WAV out."""

from __future__ import annotations

import os
import tempfile
import wave
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

SAMPLE_WIDTH = 2  # 16-bit PCM


def write_pcm16(path: str | Path, samples: np.ndarray, sample_rate: int) -> Path:
    """Write a mono float waveform as 16-bit PCM, atomically."""
    path = Path(path)
    data = np.asarray(samples, dtype=np.float32).squeeze()
    if data.ndim != 1:
        raise ValueError(f"expected a mono waveform, got shape {data.shape}")
    data = np.clip(data, -1.0, 1.0)
    pcm = np.round(data * 32767.0).astype(np.int16)

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".wav.tmp")
    os.close(handle)
    try:
        with wave.open(temporary, "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(SAMPLE_WIDTH)
            stream.setframerate(sample_rate)
            stream.writeframes(pcm.tobytes())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


def read_info(path: str | Path) -> tuple[int, int, int]:
    """Return ``(channels, sample_rate, frames)`` for a PCM WAV."""
    with wave.open(str(path), "rb") as stream:
        return stream.getnchannels(), stream.getframerate(), stream.getnframes()


def duration(path: str | Path) -> float:
    _, rate, frames = read_info(path)
    return frames / rate if rate else 0.0


def validate_pcm16(path: str | Path) -> tuple[int, int]:
    channels, rate, _ = read_info(path)
    with wave.open(str(path), "rb") as stream:
        width = stream.getsampwidth()
    if channels != 1 or width != SAMPLE_WIDTH:
        raise ValueError(
            f"{path}: expected mono 16-bit PCM, got {channels} channel(s), "
            f"{width * 8}-bit"
        )
    return rate, channels


def concat_wavs(
    paths: Sequence[str | Path],
    out_path: str | Path,
    gap_seconds: float | Sequence[float] = 0.0,
) -> float:
    """Concatenate mono PCM16 WAVs, optionally inserting silence.

    ``gap_seconds`` is either one gap for every boundary or a sequence with
    ``len(paths) - 1`` values. Returns the duration in seconds. Files are
    streamed, so a thirty-minute chapter never has to be resident in memory.
    """
    if not paths:
        raise ValueError("no WAV files to concatenate")
    rates = {validate_pcm16(path)[0] for path in paths}
    if len(rates) != 1:
        raise ValueError(f"mixed sample rates: {sorted(rates)}")
    rate = rates.pop()
    boundaries = len(paths) - 1
    if isinstance(gap_seconds, (int, float)):
        gaps = [float(gap_seconds)] * boundaries
    else:
        gaps = [float(value) for value in gap_seconds]
        if len(gaps) != boundaries:
            raise ValueError(
                f"expected {boundaries} gaps, got {len(gaps)}"
            )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=out_path.parent, suffix=".wav.tmp")
    os.close(handle)
    total_frames = 0
    try:
        with wave.open(temporary, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(SAMPLE_WIDTH)
            output.setframerate(rate)
            for index, path in enumerate(paths):
                if index:
                    gap_frames = int(round(gaps[index - 1] * rate))
                    if gap_frames:
                        output.writeframes(
                            np.zeros(gap_frames, dtype=np.int16).tobytes()
                        )
                        total_frames += gap_frames
                with wave.open(str(path), "rb") as source:
                    frames = source.getnframes()
                    while True:
                        block = source.readframes(65536)
                        if not block:
                            break
                        output.writeframes(block)
                    total_frames += frames
        os.replace(temporary, out_path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return total_frames / rate


def silence_wav(
    path: str | Path, seconds: float, sample_rate: int = 24_000
) -> Path:
    return write_pcm16(
        path, np.zeros(int(round(seconds * sample_rate)), dtype=np.float32), sample_rate
    )


def total_duration(paths: Iterable[str | Path]) -> float:
    return sum(duration(path) for path in paths)
