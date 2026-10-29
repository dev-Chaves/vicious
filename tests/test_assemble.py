from __future__ import annotations

import shutil

import numpy as np
import pytest

from audiobook.assemble import assemble_mp3
from audiobook.wav import write_pcm16

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)


@requires_ffmpeg
def test_assemble_mp3(tmp_path):
    paths = [
        write_pcm16(
            tmp_path / f"{index}.wav",
            np.zeros(12_000, dtype=np.float32),
            24_000,
        )
        for index in range(2)
    ]
    out = tmp_path / "chapter.mp3"
    result = assemble_mp3(
        paths,
        out,
        gaps=0.2,
        title="Chapter 1. Test",
        album="Test Book",
    )
    assert out.exists()
    assert out.stat().st_size > 0
    assert result.duration_s == pytest.approx(1.2)
    assert result.size_bytes == out.stat().st_size


def _tone(path, seconds=6.0, amplitude=0.05, rate=24_000):
    t = np.arange(int(rate * seconds)) / rate
    samples = amplitude * np.sin(2 * np.pi * 220 * t)
    return write_pcm16(path, samples.astype(np.float32), rate)


@requires_ffmpeg
def test_assemble_normalises_loudness(tmp_path):
    source = _tone(tmp_path / "tone.wav")
    result = assemble_mp3(
        [source],
        tmp_path / "tone.mp3",
        gaps=0,
        target_lufs=-19.0,
    )
    assert result.loudness_lufs == pytest.approx(-19.0, abs=1.0)
    assert result.true_peak_dbtp is not None
    assert result.true_peak_dbtp <= -1.5


@requires_ffmpeg
def test_assemble_can_skip_normalisation(tmp_path):
    source = _tone(tmp_path / "tone.wav")
    result = assemble_mp3(
        [source],
        tmp_path / "tone.mp3",
        gaps=0,
        target_lufs=None,
    )
    assert result.loudness_lufs is None
    assert result.true_peak_dbtp is None
