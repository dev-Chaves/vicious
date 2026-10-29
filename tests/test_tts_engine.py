from __future__ import annotations

import os

import pytest

from audiobook.tts.base import TTSSettings
from audiobook.tts.lfm3 import LFM3Engine

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("RUN_MLX") != "1",
        reason="set RUN_MLX=1 to load the speech model",
    ),
]


def test_lfm3_generates_a_wav(tmp_path):
    engine = LFM3Engine(TTSSettings(), cache=None)
    try:
        result = engine.synthesize(
            "Testing one two three.", tmp_path / "test.wav", max_tokens=128
        )
    finally:
        engine.close()
    assert result.path.exists()
    assert result.duration_s > 0.5
    assert result.sample_rate == 24_000
    assert result.finish_reason == "stop"
    assert result.peak_memory_gb > 0
