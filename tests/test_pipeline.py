from __future__ import annotations

import shutil

import numpy as np
import pytest

from audiobook.chunk import chunk_script
from audiobook.pipeline import SpeakOptions, run_speak
from audiobook.schema import Script
from audiobook.script import build_script_from_chapter
from audiobook.tts.base import Synthesis
from audiobook.wav import write_pcm16

from .helpers import make_block, make_chapter, make_document, make_page

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)


class StubEngine:
    """Deterministic stand-in: one short tone per chunk, no model."""

    sample_rate = 24_000
    cache = None
    load_peak_memory_gb = 0.75

    def __init__(self, seconds: float = 0.3, amplitude: float = 0.2):
        self.seconds = seconds
        self.amplitude = amplitude
        self.calls = 0

    def synthesize(self, text, out_path, max_tokens=None) -> Synthesis:
        self.calls += 1
        rate = self.sample_rate
        count = int(self.seconds * rate)
        t = np.arange(count) / rate
        # Speech-like bursts: loudnorm ignores a perfectly steady tone, and the
        # tests must exercise the same path a real chapter takes.
        envelope = (np.sin(2 * np.pi * 1.2 * t) > 0).astype(np.float32)
        samples = self.amplitude * np.sin(2 * np.pi * 180 * t) * envelope
        write_pcm16(out_path, samples.astype(np.float32), rate)
        return Synthesis(
            path=out_path,
            text=text,
            duration_s=self.seconds,
            generation_s=0.05,
            sample_rate=rate,
            peak_memory_gb=0.5,
        )


@pytest.fixture
def script_path(tmp_path):
    chapter = make_chapter(
        [
            make_page(
                1,
                [
                    make_block("It is a truth universally acknowledged."),
                    make_block("How, then, should a system be run?"),
                ],
            )
        ]
    )
    script = build_script_from_chapter(make_document(), chapter)
    return script.save(tmp_path / "script.json")


@requires_ffmpeg
def test_run_speak_produces_an_mp3(tmp_path, script_path):
    engine = StubEngine()
    options = SpeakOptions(
        out=tmp_path / "chapter.mp3",
        cache_dir=tmp_path / "cache",
        use_cache=False,
        normalize=False,
    )
    result = run_speak(script_path, options, engine=engine)

    assert options.out.exists()
    assert engine.calls == len(chunk_script(Script.load(script_path)))
    assert len(result.selected) == engine.calls
    assert result.assembly.duration_s >= engine.calls * 0.3
    assert result.load_peak_memory_gb == pytest.approx(0.75)


@requires_ffmpeg
def test_run_speak_reports_progress(tmp_path, script_path):
    seen: list[tuple[int, int, str]] = []

    def on_chunk(index, total, chunk, synthesis):
        seen.append((index, total, chunk.id))

    run_speak(
        script_path,
        SpeakOptions(
            out=tmp_path / "chapter.mp3",
            cache_dir=tmp_path / "cache",
            use_cache=False,
            normalize=False,
        ),
        engine=StubEngine(),
        on_chunk=on_chunk,
    )
    assert [item[0] for item in seen] == list(range(1, len(seen) + 1))
    assert all(item[1] == len(seen) for item in seen)


@requires_ffmpeg
def test_run_speak_normalises_loudness(tmp_path, script_path):
    options = SpeakOptions(
        out=tmp_path / "quiet.mp3",
        cache_dir=tmp_path / "cache",
        use_cache=False,
        target_lufs=-19.0,
        true_peak=-2.0,
        loudness_passes=1,
    )
    result = run_speak(
        script_path, options, engine=StubEngine(seconds=3.0, amplitude=0.02)
    )
    assert result.assembly.loudness_lufs == pytest.approx(-19.0, abs=1.5)
    assert result.assembly.true_peak_dbtp is not None


@requires_ffmpeg
def test_run_speak_rejects_start_past_the_end(tmp_path, script_path):
    with pytest.raises(ValueError, match="past the last chunk"):
        run_speak(
            script_path,
            SpeakOptions(
                out=tmp_path / "chapter.mp3",
                cache_dir=tmp_path / "cache",
                use_cache=False,
                start=999,
            ),
            engine=StubEngine(),
        )
