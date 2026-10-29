from __future__ import annotations

import os

import pytest

from audiobook.tts.kokoro import KokoroSettings


def test_fingerprint_tracks_voice_and_speed():
    base = KokoroSettings()
    assert base.fingerprint() == KokoroSettings().fingerprint()
    assert base.fingerprint() != KokoroSettings(voice="af_bella").fingerprint()
    assert base.fingerprint() != KokoroSettings(speed=1.1).fingerprint()
    assert base.as_dict()["engine"] == "kokoro"


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
