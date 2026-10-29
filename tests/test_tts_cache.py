from __future__ import annotations

import numpy as np
import pytest

from audiobook.tts import AudioCache
from audiobook.tts.base import TTSSettings
from audiobook.wav import (
    concat_wavs,
    duration,
    read_info,
    silence_wav,
    write_pcm16,
)


def _tone(seconds: float, rate: int = 24_000, value: float = 0.1) -> np.ndarray:
    samples = int(seconds * rate)
    return np.full(samples, value, dtype=np.float32)


def test_write_and_read_wav(tmp_path):
    path = write_pcm16(tmp_path / "a.wav", _tone(0.5), 24_000)
    channels, rate, frames = read_info(path)
    assert (channels, rate, frames) == (1, 24_000, 12_000)
    assert duration(path) == pytest.approx(0.5)


def test_concat_with_uniform_gap(tmp_path):
    first = write_pcm16(tmp_path / "a.wav", _tone(0.5), 24_000)
    second = write_pcm16(tmp_path / "b.wav", _tone(0.5), 24_000)
    out = tmp_path / "out.wav"
    total = concat_wavs([first, second], out, gap_seconds=0.2)
    assert total == pytest.approx(1.2)
    assert duration(out) == pytest.approx(1.2)


def test_concat_with_per_boundary_gaps(tmp_path):
    paths = [
        write_pcm16(tmp_path / f"{index}.wav", _tone(0.5), 24_000)
        for index in range(3)
    ]
    out = tmp_path / "out.wav"
    total = concat_wavs(paths, out, gap_seconds=[0.1, 0.3])
    assert total == pytest.approx(1.9)


def test_concat_rejects_wrong_gap_count(tmp_path):
    paths = [
        write_pcm16(tmp_path / f"{index}.wav", _tone(0.2), 24_000)
        for index in range(2)
    ]
    with pytest.raises(ValueError, match="expected 1 gaps"):
        concat_wavs(paths, tmp_path / "out.wav", gap_seconds=[0.1, 0.2])


def test_silence_wav(tmp_path):
    path = silence_wav(tmp_path / "silence.wav", 0.25)
    assert duration(path) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_cache_key_covers_text_settings_and_budget():
    cache = AudioCache("/tmp/does-not-matter")
    fingerprint = TTSSettings().fingerprint()
    a = cache.key(fingerprint, "hello", 256)
    b = cache.key(fingerprint, "hello", 256)
    c = cache.key(fingerprint, "hello!", 256)
    d = cache.key(fingerprint, "hello", 512)
    e = cache.key(TTSSettings(voice="male").fingerprint(), "hello", 256)
    assert a == b
    assert len({a, c, d, e}) == 4


def test_cache_store_and_lookup(tmp_path):
    cache = AudioCache(tmp_path)
    wav = write_pcm16(tmp_path / "source.wav", _tone(0.1), 24_000)
    key = cache.key("fingerprint", "hello", 256)
    assert cache.lookup(key) is None
    cache.store(
        key,
        wav,
        text="hello",
        settings={"voice": "female"},
        stats={"duration_s": 0.1},
    )
    entry = cache.lookup(key)
    assert entry is not None
    assert entry.meta["text"] == "hello"
    assert entry.meta["duration_s"] == 0.1
    target = tmp_path / "copy" / "chunk.wav"
    cache.copy_from_cache(entry, target)
    assert target.exists()
    assert duration(target) == pytest.approx(0.1)


def test_cache_miss_when_wav_is_missing(tmp_path):
    cache = AudioCache(tmp_path)
    key = cache.key("fingerprint", "hello", 256)
    cache.meta_path(key).parent.mkdir(parents=True)
    cache.meta_path(key).write_text("{}")
    assert cache.lookup(key) is None


def test_prune_keeps_only_live_keys(tmp_path):
    cache = AudioCache(tmp_path / "cache")
    wav = write_pcm16(tmp_path / "source.wav", _tone(0.1), 24_000)
    live = cache.key("fingerprint", "keep me", 0)
    stale = cache.key("fingerprint", "drop me", 0)
    for key, text in ((live, "keep me"), (stale, "drop me")):
        cache.store(key, wav, text=text, settings={}, stats={"duration_s": 0.1})

    result = cache.prune({live})

    assert result.kept == 1
    assert result.removed == 1
    assert result.freed_bytes > 0
    assert cache.lookup(live) is not None
    assert cache.lookup(stale) is None
    assert cache.meta_path(stale).exists() is False


def test_prune_dry_run_changes_nothing(tmp_path):
    cache = AudioCache(tmp_path / "cache")
    wav = write_pcm16(tmp_path / "source.wav", _tone(0.1), 24_000)
    stale = cache.key("fingerprint", "drop me", 0)
    cache.store(stale, wav, text="drop me", settings={}, stats={})

    result = cache.prune(set(), dry_run=True)

    assert result.removed == 1
    assert result.freed_bytes > 0
    assert cache.lookup(stale) is not None


def test_live_cache_keys_match_the_engine_lookup(tmp_path):
    from audiobook.pipeline import live_cache_keys
    from audiobook.schema import Script, Segment, Source
    from audiobook.tts.kokoro import KokoroSettings
    from audiobook.types import Origin, SegmentType

    cache = AudioCache(tmp_path / "cache")
    script = Script(
        source=Source(
            pdf="book.pdf",
            sha256="0" * 64,
            chapter=1,
            title="Chapter 1. Test",
            first_page=1,
            last_page=1,
        ),
        segments=[
            Segment(
                id="ch01-0001",
                type=SegmentType.PROSE,
                text_original="Hello there.",
                text_narrated="Hello there.",
                origin=Origin.RULE,
                page=1,
            )
        ],
    )
    fingerprint = KokoroSettings().fingerprint()
    keys = live_cache_keys([script], cache, fingerprint, "kokoro")
    assert keys == {cache.key(fingerprint, "Hello there.", 0)}
