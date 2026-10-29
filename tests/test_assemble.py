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
