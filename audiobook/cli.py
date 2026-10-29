"""Command line interface: ``script`` (pass 1), ``speak`` (pass 2), ``book``."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .engines import EngineOptions
from .schema import SCHEMA_VERSION, Script, ScriptError
from .tts import DEFAULT_KOKORO_MODEL, DEFAULT_MODEL, AudioCache

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Local, two-pass PDF-to-audiobook pipeline.",
)
cache_app = typer.Typer(no_args_is_help=True, help="Inspect or clean the audio cache.")
app.add_typer(cache_app, name="cache")

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
        "kokoro", "--engine", help="Speech engine: kokoro or lfm3."
    ),
    model: str = typer.Option(DEFAULT_MODEL, help="LFM3 MLX checkpoint."),
    kokoro_model: str = typer.Option(
        DEFAULT_KOKORO_MODEL, help="Kokoro MLX checkpoint."
    ),
    voice: str = typer.Option(
        "",
        help="Kokoro pack (af_heart, af_bella, am_michael) or female/male for LFM3.",
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
    normalize: bool = typer.Option(
        True,
        "--normalize/--no-normalize",
        help="Bring the chapter to a fixed loudness with ffmpeg loudnorm.",
    ),
    target_lufs: float = typer.Option(
        -19.0, "--target-lufs", help="Integrated loudness target (LUFS)."
    ),
    true_peak: float = typer.Option(
        -2.0, "--true-peak", help="True peak ceiling (dBTP)."
    ),
    loudness_passes: int = typer.Option(
        1, "--loudness-passes", min=1, max=2, help="1 is faster, 2 is exact."
    ),
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
    from .engines import create_engine
    from .pipeline import SpeakOptions, run_speak
    from .tts.base import Synthesis
    from .chunk import Chunk

    engine_options = EngineOptions(
        name=engine_name,
        voice=voice,
        speed=speed,
        lfm3_model=model,
        kokoro_model=kokoro_model,
        language=language,
    )
    try:
        script_data = Script.load(script_path)
    except (FileNotFoundError, ScriptError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    if script_data.version != SCHEMA_VERSION:
        typer.echo("error: unsupported script version", err=True)
        raise typer.Exit(code=1)

    cache = None if no_cache else AudioCache(cache_dir)
    try:
        engine, label = create_engine(engine_options, cache=cache, refresh=refresh)
    except ValueError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    options = SpeakOptions(
        out=out,
        work=work,
        cache_dir=cache_dir,
        use_cache=not no_cache,
        refresh=refresh,
        start=start,
        limit=limit,
        gap_within=gap_within,
        gap_between=gap_between,
        bitrate=bitrate,
        max_chars=max_chars,
        min_chars=min_chars,
        normalize=normalize,
        target_lufs=target_lufs,
        true_peak=true_peak,
        loudness_passes=loudness_passes,
    )

    def progress(index: int, total: int, chunk: Chunk, synthesis: Synthesis) -> None:
        detail = (
            "cached"
            if synthesis.cached
            else f"{synthesis.generation_s:5.1f}s "
            f"{synthesis.realtime_factor:4.2f}x"
        )
        flags = f"  ! {synthesis.warnings[0]}" if synthesis.warnings else ""
        typer.echo(
            f"[{index:>4}/{total}] {chunk.id}  "
            f"{chunk.chars:>3} ch  {synthesis.duration_s:6.1f}s  "
            f"{detail}{flags}"
        )

    typer.echo(script_data.source.title)
    typer.echo(f"  engine      {label}")
    typer.echo(f"  output      {out}")
    typer.echo(f"  work        {work or out.parent / f'{out.stem}.chunks'}\n")
    try:
        result = run_speak(
            script_path, options, engine=engine, cache=cache, on_chunk=progress
        )
    except KeyboardInterrupt:
        typer.echo("\ninterrupted; cached chunks are kept", err=True)
        raise typer.Exit(code=130)
    except ValueError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    except Exception as error:  # noqa: BLE001 - surface the real message
        typer.echo(f"\nerror while synthesizing: {error}", err=True)
        raise typer.Exit(code=1) from error
    finally:
        engine.close()

    _print_report(result)


@app.command()
def book(
    pdf: Path = typer.Argument(..., exists=True, readable=True, help="Source PDF."),
    out_dir: Path = typer.Option(
        Path("build"), "--out", "-o", help="Book directory (scripts, MP3s, manifest)."
    ),
    chapters: str = typer.Option(
        "all", "--chapters", help="all, 1-34 or a list such as 2,4,6-8."
    ),
    engine_name: str = typer.Option(
        "kokoro", "--engine", help="Speech engine: kokoro or lfm3."
    ),
    model: str = typer.Option(DEFAULT_MODEL, help="LFM3 MLX checkpoint."),
    kokoro_model: str = typer.Option(
        DEFAULT_KOKORO_MODEL, help="Kokoro MLX checkpoint."
    ),
    voice: str = typer.Option("", help="Kokoro pack or female/male for LFM3."),
    speed: float = typer.Option(
        1.0, "--speed", min=0.5, max=2.0, help="Kokoro speaking rate."
    ),
    language: str = typer.Option("American English", help="LFM3 TTS voice locale."),
    force: bool = typer.Option(
        False, "--force", help="Regenerate chapters even when the MP3 exists."
    ),
    refresh: bool = typer.Option(
        False, "--refresh", help="Ignore cached audio for every chunk."
    ),
    cache_dir: Path = typer.Option(
        Path(".cache/tts"), "--cache", help="Audio cache directory."
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Do not read or write the audio cache."
    ),
    gap_within: float = typer.Option(0.18, min=0.0),
    gap_between: float = typer.Option(0.45, min=0.0),
    bitrate: str = typer.Option("96k", help="MP3 bitrate."),
    normalize: bool = typer.Option(
        True, "--normalize/--no-normalize", help="Normalise each chapter's loudness."
    ),
    target_lufs: float = typer.Option(-19.0, "--target-lufs"),
    true_peak: float = typer.Option(-2.0, "--true-peak"),
    loudness_passes: int = typer.Option(
        1, "--loudness-passes", min=1, max=2, help="1 is faster, 2 is exact."
    ),
    max_chars: int = typer.Option(600, min=100),
    min_chars: int = typer.Option(120, min=0),
) -> None:
    """Generate every chapter of a book in one run, resuming when interrupted."""
    from .book import BookOptions, run_book
    from .extract import chapter_numbers, open_document

    engine_options = EngineOptions(
        name=engine_name,
        voice=voice,
        speed=speed,
        lfm3_model=model,
        kokoro_model=kokoro_model,
        language=language,
    )
    try:
        document = open_document(pdf)
        selected = _parse_chapters(chapters, chapter_numbers(document))
    except (FileNotFoundError, ValueError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    options = BookOptions(
        pdf=pdf,
        out_dir=out_dir,
        chapters=selected,
        engine=engine_options,
        cache_dir=cache_dir,
        use_cache=not no_cache,
        refresh=refresh,
        force=force,
        gap_within=gap_within,
        gap_between=gap_between,
        bitrate=bitrate,
        max_chars=max_chars,
        min_chars=min_chars,
        normalize=normalize,
        target_lufs=target_lufs,
        true_peak=true_peak,
        loudness_passes=loudness_passes,
    )

    typer.echo(f"{document.title}")
    typer.echo(f"  chapters    {len(selected)} of {len(chapter_numbers(document))}")
    typer.echo(f"  output      {out_dir}")
    typer.echo(f"  engine      {engine_options.describe()}")
    typer.echo("")

    started = _now()
    state = {"done": 0, "wall": 0.0}

    def progress(record, index: int, total: int) -> None:
        state["done"] += 1
        state["wall"] += record.wall_s
        if record.status == "failed":
            typer.echo(
                f"[{index:>3}/{total}] ch{record.number:02d} FAILED  {record.error}"
            )
            return
        audio = _clock(record.duration_s)
        loudness = (
            f"{record.loudness_lufs:6.1f} LUFS"
            if record.loudness_lufs is not None
            else "     unknown"
        )
        if record.status == "skipped":
            detail = f"{record.chunks:4} chunks  {audio}  {loudness}  (cached)"
        else:
            detail = (
                f"{record.chunks:4} chunks  {audio}  {loudness}  "
                f"gen {_clock(record.generation_s)}"
            )
        eta = _eta(state["wall"] / state["done"], total - state["done"])
        typer.echo(
            f"[{index:>3}/{total}] ch{record.number:02d} "
            f"{record.status:<7} {detail}  ETA {eta}"
        )

    try:
        result = run_book(options, on_chapter=progress)
    except KeyboardInterrupt:
        typer.echo("\ninterrupted; rerun the same command to resume", err=True)
        raise typer.Exit(code=130)
    except (ValueError, RuntimeError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    elapsed = _now() - started
    typer.echo("\nDone.")
    typer.echo(
        f"  audio       {_clock(result.duration_s)} in "
        f"{len(result.all_chapters or result.chapters)} chapters"
    )
    typer.echo(f"  generated   {_clock(result.generated_s)} of speech")
    typer.echo(f"  wall clock  {_clock(elapsed)}")
    if result.warnings:
        typer.echo(f"  warnings    {result.warnings} (see the manifest)")
    if result.failed:
        typer.echo(f"  failed      {len(result.failed)} chapter(s): "
                   + ", ".join(f"ch{r.number:02d}" for r in result.failed))
    typer.echo(f"  manifest    {result.manifest_path}")
    if result.book_mp3 is not None:
        typer.echo(f"  book mp3    {result.book_mp3}")


@cache_app.command("gc")
def cache_gc(
    scripts: Optional[list[Path]] = typer.Option(
        None, "--scripts", "-s", help="script.json files to keep (default: build/ch*/)."
    ),
    cache_dir: Path = typer.Option(
        Path(".cache/tts"), "--cache", help="Audio cache directory."
    ),
    engine_name: str = typer.Option(
        "kokoro", "--engine", help="Engine whose entries must survive."
    ),
    voice: str = typer.Option("", help="Kokoro pack or female/male for LFM3."),
    speed: float = typer.Option(1.0, "--speed", min=0.5, max=2.0),
    model: str = typer.Option(DEFAULT_MODEL),
    kokoro_model: str = typer.Option(DEFAULT_KOKORO_MODEL),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report what would be removed."
    ),
) -> None:
    """Delete cache entries that no current script + engine configuration needs."""
    from .pipeline import live_cache_keys

    paths = list(scripts or [])
    if not paths:
        paths = sorted(Path("build").glob("ch*/script.json"))
    if not paths:
        typer.echo("error: no scripts given and none found under build/ch*/", err=True)
        raise typer.Exit(code=1)

    engine_options = EngineOptions(
        name=engine_name,
        voice=voice,
        speed=speed,
        lfm3_model=model,
        kokoro_model=kokoro_model,
    )
    cache = AudioCache(cache_dir)
    try:
        fingerprint = engine_options.fingerprint()
        scripts_data = [Script.load(path) for path in paths]
    except (ValueError, ScriptError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error

    keep = live_cache_keys(scripts_data, cache, fingerprint, engine_name)
    result = cache.prune(keep, dry_run=dry_run)
    verb = "would free" if dry_run else "freed"
    typer.echo(f"{len(paths)} script(s), {len(keep)} live keys")
    typer.echo(
        f"kept {result.kept}, removed {result.removed} entries, "
        f"{verb} {_size(result.freed_bytes)}"
    )


def _parse_chapters(spec: str, available: list[int]) -> list[int] | None:
    text = spec.strip().lower()
    if text in ("", "all", "todos"):
        return None
    numbers: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first, _, last = part.partition("-")
            start, end = int(first), int(last)
            if start > end:
                raise ValueError(f"invalid chapter range {part!r}")
            numbers.extend(range(start, end + 1))
        else:
            numbers.append(int(part))
    unique = sorted(dict.fromkeys(numbers))
    missing = [number for number in unique if number not in available]
    if missing:
        raise ValueError(
            "chapter(s) not found in the PDF TOC: "
            + ", ".join(str(number) for number in missing)
        )
    return unique


def _print_report(result) -> None:
    assembly = result.assembly
    generated = result.generated_s
    typer.echo("\nDone.")
    typer.echo(
        f"  audio       {_clock(assembly.duration_s)} total "
        f"({assembly.duration_s:.1f}s, {_size(assembly.size_bytes)})"
    )
    typer.echo(
        f"  speech      {_clock(result.speech_s)} in {len(result.selected)} chunks "
        f"({result.cached} cached)"
    )
    if generated > 0:
        typer.echo(
            f"  generated   {_clock(generated)} this run at "
            f"{result.realtime_factor:.2f}x real time"
        )
    if result.peak_memory_gb > 0:
        typer.echo(
            f"  peak memory {result.peak_memory_gb:.2f} GB "
            f"(model load {result.load_peak_memory_gb:.2f} GB)"
        )
    else:
        typer.echo("  peak memory not measured (every chunk was cached)")
    if assembly.loudness_lufs is not None:
        peak_text = (
            f", peak {assembly.true_peak_dbtp:.1f} dBTP"
            if assembly.true_peak_dbtp is not None
            else ""
        )
        typer.echo(f"  loudness    {assembly.loudness_lufs:.1f} LUFS{peak_text}")
    typer.echo(f"  file        {assembly.mp3}")
    if result.warnings:
        typer.echo(f"  warnings    {len(result.warnings)}")
        for chunk_id, warning in result.warnings[:20]:
            typer.echo(f"    {chunk_id}: {warning}")
        if len(result.warnings) > 20:
            typer.echo(f"    ... and {len(result.warnings) - 20} more")


def _clock(seconds: float) -> str:
    seconds = max(0.0, seconds)
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
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


def _eta(seconds_per_chapter: float, remaining: int) -> str:
    if remaining <= 0 or seconds_per_chapter <= 0:
        return "--:--"
    return _clock(seconds_per_chapter * remaining)


def _now() -> float:
    import time

    return time.perf_counter()
