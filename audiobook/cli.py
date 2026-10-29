"""Command line interface: ``script`` (pass 1) and ``speak`` (pass 2)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .schema import SCHEMA_VERSION, Script, ScriptError
from .tts import (
    DEFAULT_KOKORO_MODEL,
    DEFAULT_MODEL,
    AudioCache,
    KokoroSettings,
    TTSSettings,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Local, two-pass PDF-to-audiobook pipeline.",
)

CHARS_PER_SECOND = 14.0


@app.command()
def script(
    pdf: Path = typer.Argument(
        ..., exists=True, readable=True, help="Source PDF."
    ),
    chapter: int = typer.Option(1, "--chapter", "-c", min=1, help="Chapter number."),
    out: Path = typer.Option(
        Path("build/ch01/script.json"), "--out", "-o", help="Where to write the script."
    ),
    preview: int = typer.Option(
        6, min=0, help="Show the first N narrated segments."
    ),
) -> None:
    """Pass 1: extract one chapter, classify, clean and narrate into JSON."""
    from .script import build_script

    try:
        result = build_script(pdf, chapter)
    except (FileNotFoundError, ValueError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    result.save(out)
    narrated = result.narrated
    estimated = sum(len(segment.text_narrated) for segment in narrated)
    estimated_minutes = estimated / CHARS_PER_SECOND / 60

    typer.echo(result.source.title)
    typer.echo(
        f"  pages       {result.source.first_page}-{result.source.last_page}"
    )
    typer.echo(
        f"  segments    {len(result.segments)} "
        f"({len(narrated)} narrated, {result.skipped_count} skipped)"
    )
    typer.echo(f"  words       {result.word_count} narrated")
    typer.echo(f"  estimate    ~{estimated_minutes:.1f} min of speech")
    typer.echo(f"  output      {out}")

    if preview:
        typer.echo("\nFirst segments:")
        for segment in narrated[:preview]:
            text = " ".join(segment.text_narrated.split())
            if len(text) > 110:
                text = text[:107] + "..."
            typer.echo(f"  {segment.id}  {segment.type.value:<7}  {text}")
        typer.echo(
            "\nReview text_original vs text_narrated in the JSON; "
            "edit text_narrated and re-run speak."
        )


@app.command()
def speak(
    script_path: Path = typer.Argument(
        ..., exists=True, readable=True, help="script.json from pass 1."
    ),
    out: Path = typer.Option(
        Path("audiobook.mp3"), "--out", "-o", help="Output MP3."
    ),
    work: Optional[Path] = typer.Option(
        None, "--work", help="Directory for chunk WAVs (default: next to the MP3)."
    ),
    cache_dir: Path = typer.Option(
        Path(".cache/tts"), "--cache", help="Audio cache directory."
    ),
    engine_name: str = typer.Option(
        "lfm3", "--engine", help="Speech engine: lfm3 or kokoro."
    ),
    model: str = typer.Option(DEFAULT_MODEL, help="LFM3 MLX checkpoint."),
    kokoro_model: str = typer.Option(
        DEFAULT_KOKORO_MODEL, help="Kokoro MLX checkpoint."
    ),
    voice: str = typer.Option(
        "", help="LFM3: female/male. Kokoro: af_heart, af_bella, am_michael..."
    ),
    speed: float = typer.Option(
        1.0, "--speed", min=0.5, max=2.0, help="Kokoro speaking rate."
    ),
    language: str = typer.Option("American English", help="LFM3 TTS voice locale."),
    start: int = typer.Option(1, min=1, help="First chunk (1-based)."),
    limit: int = typer.Option(0, min=0, help="Synthesize only N chunks (0 = all)."),
    gap_within: float = typer.Option(
        0.18, min=0.0, help="Silence between chunks of one segment."
    ),
    gap_between: float = typer.Option(
        0.45, min=0.0, help="Silence between segments."
    ),
    bitrate: str = typer.Option("96k", help="MP3 bitrate."),
    max_chars: int = typer.Option(600, min=100, help="Chunk size limit."),
    min_chars: int = typer.Option(120, min=0, help="Merge shorter trailing chunks."),
    refresh: bool = typer.Option(
        False, "--refresh", help="Regenerate chunks even if cached."
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Do not read or write the audio cache."
    ),
) -> None:
    """Pass 2: chunk, synthesize, assemble. Reads only script.json."""
    from .assemble import assemble_mp3
    from .chunk import chunk_script

    engine_name = engine_name.strip().lower()
    if engine_name not in ("lfm3", "kokoro"):
        typer.echo(
            f"error: unknown engine {engine_name!r} (use lfm3 or kokoro)", err=True
        )
        raise typer.Exit(code=1)

    try:
        loaded = Script.load(script_path)
    except (FileNotFoundError, ScriptError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    if loaded.version != SCHEMA_VERSION:
        typer.echo("error: unsupported script version", err=True)
        raise typer.Exit(code=1)

    chunks = chunk_script(loaded, max_chars=max_chars, min_chars=min_chars)
    if not chunks:
        typer.echo("error: script has no narrated text", err=True)
        raise typer.Exit(code=1)
    if start > len(chunks):
        typer.echo(
            f"error: --start {start} is past the last chunk ({len(chunks)})",
            err=True,
        )
        raise typer.Exit(code=1)
    selected = chunks[start - 1 :]
    if limit:
        selected = selected[:limit]

    work_dir = work or out.parent / f"{out.stem}.chunks"
    work_dir.mkdir(parents=True, exist_ok=True)
    cache = None if no_cache else AudioCache(cache_dir)
    if engine_name == "kokoro":
        from .tts.kokoro import KokoroEngine

        kokoro_settings = KokoroSettings(
            model_id=kokoro_model,
            voice=voice or "af_heart",
            speed=speed,
        )
        engine = KokoroEngine(kokoro_settings, cache=cache, refresh=refresh)
        engine_label = f"kokoro {kokoro_settings.voice} at {speed:.2f}x"
    else:
        from .tts.lfm3 import LFM3Engine

        settings = TTSSettings(
            model_id=model, language=language, voice=voice or "female"
        )
        engine = LFM3Engine(settings, cache=cache, refresh=refresh)
        engine_label = f"lfm3 {settings.voice}"

    typer.echo(f"{loaded.source.title}")
    typer.echo(
        f"  chunks      {len(selected)} of {len(chunks)} "
        f"(chars {min_chars}-{max_chars})"
    )
    typer.echo(f"  engine      {engine_label}")
    typer.echo(f"  output      {out}")
    typer.echo(f"  work        {work_dir}\n")

    results = []
    total_chunks = len(selected)
    try:
        for index, chunk in enumerate(selected, start=1):
            wav_path = work_dir / f"{chunk.id}.wav"
            synthesis = engine.synthesize(chunk.text, wav_path)
            results.append(synthesis)
            detail = (
                "cached"
                if synthesis.cached
                else f"{synthesis.generation_s:5.1f}s "
                f"{synthesis.realtime_factor:4.2f}x"
            )
            flags = ""
            if synthesis.warnings:
                flags = f"  ! {synthesis.warnings[0]}"
            typer.echo(
                f"[{index:>4}/{total_chunks}] {chunk.id}  "
                f"{chunk.chars:>3} ch  {synthesis.duration_s:6.1f}s  "
                f"{detail}{flags}"
            )
    except KeyboardInterrupt:
        typer.echo("\ninterrupted; cached chunks are kept", err=True)
        raise typer.Exit(code=130)
    except Exception as error:  # noqa: BLE001 - surface the real message
        typer.echo(f"\nerror while synthesizing: {error}", err=True)
        raise typer.Exit(code=1) from error
    finally:
        engine.close()

    gaps = [
        gap_between if left.segment_id != right.segment_id else gap_within
        for left, right in zip(selected, selected[1:])
    ]
    try:
        assembly = assemble_mp3(
            [result.path for result in results],
            out,
            gaps=gaps,
            title=loaded.source.title,
            album=loaded.source.book or None,
            bitrate=bitrate,
        )
    except (RuntimeError, ValueError) as error:
        typer.echo(f"error while assembling: {error}", err=True)
        raise typer.Exit(code=1) from error

    _report(loaded, results, assembly, engine.load_peak_memory_gb)


def _report(loaded: Script, results, assembly, load_peak: float) -> None:
    speech = sum(result.duration_s for result in results)
    generated = sum(result.generation_s for result in results)
    cached = sum(1 for result in results if result.cached)
    warnings = [
        (result.path.stem, warning)
        for result in results
        for warning in result.warnings
    ]
    generated_peak = max(
        (result.peak_memory_gb for result in results if not result.cached),
        default=0.0,
    )
    peak = max(load_peak, generated_peak)
    speed = speech / generated if generated > 0 else 0.0

    typer.echo("\nDone.")
    typer.echo(
        f"  audio       {_clock(assembly.duration_s)} total "
        f"({assembly.duration_s:.1f}s, {_size(assembly.size_bytes)})"
    )
    typer.echo(
        f"  speech      {_clock(speech)} in {len(results)} chunks "
        f"({cached} cached)"
    )
    if generated > 0:
        typer.echo(
            f"  generated   {_clock(generated)} this run at {speed:.2f}x real time"
        )
    typer.echo(f"  peak memory {peak:.2f} GB (model load {load_peak:.2f} GB)")
    typer.echo(f"  file        {assembly.mp3}")
    if warnings:
        typer.echo(f"  warnings    {len(warnings)}")
        for chunk_id, warning in warnings[:20]:
            typer.echo(f"    {chunk_id}: {warning}")
        if len(warnings) > 20:
            typer.echo(f"    ... and {len(warnings) - 20} more")


def _clock(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, rest = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


if __name__ == "__main__":
    app()
