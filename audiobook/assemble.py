"""Assemble chunk WAVs into the final MP3 with ffmpeg."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .wav import concat_wavs

DEFAULT_BITRATE = "96k"
DEFAULT_TARGET_LUFS = -19.0
DEFAULT_TRUE_PEAK = -2.0
DEFAULT_LOUDNESS_RANGE = 11.0

_JSON_OBJECT = re.compile(r"\{[^{}]*\}")


@dataclass(frozen=True)
class Assembly:
    mp3: Path
    duration_s: float
    size_bytes: int
    loudness_lufs: float | None = None
    true_peak_dbtp: float | None = None


def assemble_mp3(
    wav_paths: Sequence[str | Path],
    mp3_path: str | Path,
    gaps: float | Sequence[float] = 0.2,
    title: str | None = None,
    album: str | None = None,
    artist: str = "vicious local TTS",
    bitrate: str = DEFAULT_BITRATE,
    sample_rate: int = 24_000,
    target_lufs: float | None = DEFAULT_TARGET_LUFS,
    true_peak: float = DEFAULT_TRUE_PEAK,
    loudness_range: float = DEFAULT_LOUDNESS_RANGE,
) -> Assembly:
    """Concatenate WAVs, normalise loudness and encode a mono MP3.

    A temporary WAV is used instead of ffmpeg's concat demuxer so mixed WAV
    headers never matter; the chunks come from our own writer anyway.

    ``target_lufs=None`` skips normalisation. The default ceiling is -2 dBTP
    rather than the usual -1.5, because MP3 encoding overshoots the sample
    peaks by roughly half a dB. Otherwise ffmpeg's ``loudnorm`` runs twice:
    once to measure the programme, then once to apply a linear gain with a
    true-peak ceiling, falling back to dynamic mode when a linear gain would
    clip. Engines produce very different levels (LFM3 peaks above 0 dBFS,
    Kokoro sits 5 dB lower), and the chapter should not depend on that.
    """
    if not wav_paths:
        raise ValueError("no audio to assemble")
    mp3_path = Path(mp3_path)
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg not found; install it (brew install ffmpeg) or add it to PATH"
        )

    combined = mp3_path.with_name(f".{mp3_path.stem}.assembled.wav")
    loudness: float | None = None
    peak: float | None = None
    try:
        duration = concat_wavs(wav_paths, combined, gaps)
        measured = None
        if target_lufs is not None:
            measured = _measure(
                ffmpeg, combined, target_lufs, true_peak, loudness_range
            )

        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "info" if measured else "error",
            "-y",
            "-i",
            str(combined),
        ]
        if measured is not None:
            command += [
                "-af",
                _loudnorm_filter(target_lufs, true_peak, loudness_range, measured),
            ]
        command += [
            "-codec:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
        ]
        if title:
            command += ["-metadata", f"title={title}"]
        if album:
            command += ["-metadata", f"album={album}"]
        if artist:
            command += ["-metadata", f"artist={artist}"]
        command += ["-id3v2_version", "3", str(mp3_path)]
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed ({completed.returncode}): {completed.stderr.strip()}"
            )
        if measured is not None:
            applied = _parse_loudnorm(completed.stderr)
            if applied:
                loudness = _as_float(applied.get("output_i"))
                peak = _as_float(applied.get("output_tp"))
    finally:
        combined.unlink(missing_ok=True)

    return Assembly(
        mp3=mp3_path,
        duration_s=duration,
        size_bytes=mp3_path.stat().st_size,
        loudness_lufs=loudness,
        true_peak_dbtp=peak,
    )


def _loudnorm_filter(
    target_lufs: float,
    true_peak: float,
    loudness_range: float,
    measured: dict[str, Any],
) -> str:
    base = f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={loudness_range}"
    return (
        f"{base}:measured_I={measured['input_i']}"
        f":measured_TP={measured['input_tp']}"
        f":measured_LRA={measured['input_lra']}"
        f":measured_thresh={measured['input_thresh']}"
        f":offset={measured['target_offset']}"
        ":linear=true:print_format=json"
    )


def _measure(
    ffmpeg: str,
    source: Path,
    target_lufs: float,
    true_peak: float,
    loudness_range: float,
) -> dict[str, Any] | None:
    """First loudnorm pass: measure the programme without writing audio."""
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostats",
        "-i",
        str(source),
        "-af",
        f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={loudness_range}"
        ":print_format=json",
        "-f",
        "null",
        "-",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg loudness measurement failed: {completed.stderr.strip()}"
        )
    measured = _parse_loudnorm(completed.stderr)
    if not measured:
        return None
    if _as_float(measured.get("input_i")) is None:
        return None
    return measured


def _parse_loudnorm(stderr: str) -> dict[str, Any] | None:
    matches = _JSON_OBJECT.findall(stderr)
    if not matches:
        return None
    try:
        return json.loads(matches[-1])
    except json.JSONDecodeError:
        return None


def _as_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number
