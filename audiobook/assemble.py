"""Assemble chunk WAVs into the final MP3 with ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .wav import concat_wavs

DEFAULT_BITRATE = "96k"


@dataclass(frozen=True)
class Assembly:
    mp3: Path
    duration_s: float
    size_bytes: int


def assemble_mp3(
    wav_paths: Sequence[str | Path],
    mp3_path: str | Path,
    gaps: float | Sequence[float] = 0.2,
    title: str | None = None,
    album: str | None = None,
    artist: str = "vicious local TTS",
    bitrate: str = DEFAULT_BITRATE,
    sample_rate: int = 24_000,
) -> Assembly:
    """Concatenate WAVs and encode a mono MP3.

    A temporary WAV is used instead of ffmpeg's concat demuxer so mixed WAV
    headers never matter; the chunks come from our own writer anyway.
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
    try:
        duration = concat_wavs(wav_paths, combined, gaps)
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(combined),
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
    finally:
        combined.unlink(missing_ok=True)

    return Assembly(
        mp3=mp3_path,
        duration_s=duration,
        size_bytes=mp3_path.stat().st_size,
    )
