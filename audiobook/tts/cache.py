"""Content-addressed cache for generated speech.

The key covers the model, sampling settings, the maximum length and the exact
text, so changing any of them produces new audio while re-running the same
script reuses every chunk. Cached files are written atomically.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CacheEntry:
    key: str
    wav: Path
    meta: dict[str, Any]


@dataclass(frozen=True)
class PruneResult:
    """Outcome of :meth:`AudioCache.prune`."""

    kept: int
    removed: int
    freed_bytes: int


class AudioCache:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def key(
        self,
        settings_fingerprint: str,
        text: str,
        max_tokens: int,
    ) -> str:
        payload = json.dumps(
            {
                "settings": settings_fingerprint,
                "text": text,
                "max_tokens": max_tokens,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def wav_path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.wav"

    def meta_path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def lookup(self, key: str) -> CacheEntry | None:
        wav = self.wav_path(key)
        if not wav.exists():
            return None
        meta: dict[str, Any] = {}
        meta_file = self.meta_path(key)
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                meta = {}
        return CacheEntry(key=key, wav=wav, meta=meta)

    def store(
        self,
        key: str,
        wav: Path,
        text: str,
        settings: dict[str, Any],
        stats: dict[str, Any],
    ) -> Path:
        target = self.wav_path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        _copy_atomic(wav, target)
        meta = {
            "text": text,
            "settings": settings,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **stats,
        }
        _write_atomic(
            self.meta_path(key),
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        )
        return target

    def copy_from_cache(self, entry: CacheEntry, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if entry.wav.resolve() != out_path.resolve():
            _copy_atomic(entry.wav, out_path)

    # ------------------------------------------------------------- maintenance

    def keys(self) -> set[str]:
        """Every cache key currently on disk."""
        if not self.directory.exists():
            return set()
        return {path.stem for path in self.directory.glob("*/*.wav")}

    def prune(self, keep: set[str], dry_run: bool = False) -> PruneResult:
        """Delete entries whose key is not in ``keep``.

        Old model settings, voices and edited text leave entries behind; a book
        run with a few tuning iterations can waste gigabytes. ``keep`` is
        normally rebuilt from the current ``script.json`` files.
        """
        if not self.directory.exists():
            return PruneResult(kept=0, removed=0, freed_bytes=0)
        removed = 0
        freed = 0
        kept = 0
        for wav in sorted(self.directory.glob("*/*.wav")):
            if wav.stem in keep:
                kept += 1
                continue
            meta = wav.with_suffix(".json")
            freed += wav.stat().st_size
            if meta.exists():
                freed += meta.stat().st_size
            removed += 1
            if not dry_run:
                wav.unlink(missing_ok=True)
                meta.unlink(missing_ok=True)
        if not dry_run:
            for shard in self.directory.iterdir():
                if shard.is_dir() and not any(shard.iterdir()):
                    shard.rmdir()
            for leftover in self.directory.glob("*/*.tmp"):
                leftover.unlink(missing_ok=True)
        return PruneResult(kept=kept, removed=removed, freed_bytes=freed)


def _copy_atomic(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    os.close(handle)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _write_atomic(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
