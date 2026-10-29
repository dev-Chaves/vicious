from __future__ import annotations

import os

import numpy as np
import pytest

from audiobook.tts.kokoro import KokoroSettings, trim_edges


def test_fingerprint_tracks_voice_and_speed():
    base = KokoroSettings()
    assert base.fingerprint() == KokoroSettings().fingerprint()
    assert base.fingerprint() != KokoroSettings(voice="af_bella").fingerprint()
    assert base.fingerprint() != KokoroSettings(speed=1.1).fingerprint()
    assert base.fingerprint() != KokoroSettings(edge_pad_ms=0).fingerprint()
    assert base.as_dict()["engine"] == "kokoro"


def test_trim_edges_keeps_only_the_requested_pad():
    rate = 24_000
    tone = 0.4 * np.sin(2 * np.pi * 220 * np.arange(rate) / rate)
    padded = np.concatenate(
        [np.zeros(rate // 2), tone, np.zeros(rate // 2)]  # 500 ms each side
    )
    trimmed = trim_edges(padded, rate, pad_ms=120)
    expected = rate + 2 * int(rate * 0.120)
    assert abs(trimmed.size - expected) <= 2 * int(rate * 0.010)


def test_trim_edges_leaves_pure_silence_alone():
    silence = np.zeros(24_000, dtype=np.float32)
    assert trim_edges(silence, 24_000).size == silence.size


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("RUN_KOKORO") != "1",
    reason="set RUN_KOKORO=1 to load the Kokoro model",
)
def test_kokoro_generates_a_wav(tmp_path):
    from audiobook.tts.kokoro import KokoroEngine

    engine = KokoroEngine(KokoroSettings(), cache=None)
    try:
        result = engine.synthesize("Testing one two three.", tmp_path / "test.wav")
    finally:
        engine.close()
    assert result.path.exists()
    assert result.duration_s > 0.5
    assert result.sample_rate == 24_000
    assert result.generation_s > 0
    assert result.peak_memory_gb > 0
