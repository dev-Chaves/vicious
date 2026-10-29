# vicious

Local PDF → audiobook pipeline for technical books, built for Apple Silicon.
Two passes with a reviewable script in between:

```text
PDF → extract → classify → clean → script.json      (pass 1, text)
script.json → chunk → TTS → assemble → mp3          (pass 2, audio)
```

Pass 1 is fast and deterministic; pass 2 is slow and stochastic. Keeping them
apart means you review (and edit) every spoken word before spending minutes on
speech synthesis. Pass 2 reads **only** `script.json` — it never opens the PDF
or imports PyMuPDF.

Speech is generated locally on the GPU with
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by default and
[LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B) (MLX, 8-bit) as
the alternative. No API keys, no network at synthesis time.

Milestone 1: one chapter in, one MP3 out. No LLM narration, no translation, no
web UI.

## Requirements

- macOS on Apple Silicon (tested on an M4 with 16 GB)
- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- ffmpeg with libmp3lame (`brew install ffmpeg`)
- ~390 MB of disk for the default Kokoro model (or ~3 GB for LFM3)
- ~1 GB of RAM while speaking with Kokoro, ~2.5 GB with LFM3

## Install

```bash
uv sync
brew install ffmpeg     # if you do not have it
```

The first synthesis downloads `mlx-community/Kokoro-82M-bf16` (~390 MB) plus a
spaCy model into `~/.cache/huggingface`. `--engine lfm3` downloads
`LiquidAI/LFM3-Audio-2B-MLX-8bit` (~3 GB) instead. After that, everything is
offline.

Kokoro's G2P comes from `misaki[en]`, which pulls PyTorch and spaCy into the
virtualenv; that is the price of correct numbers and acronyms.

## Quick start

```bash
# Pass 1: chapter 1 → build/ch01/script.json
uv run python -m audiobook script \
  "ebook-pdf/Site Reliability Engineering_ How Google Runs Production Systems ( PDFDrive ).pdf" \
  --chapter 1 --out build/ch01/script.json

# Read the JSON, fix anything you dislike (text_narrated is what is spoken).

# Pass 2: script.json → build/ch01/ch01.mp3
uv run python -m audiobook speak build/ch01/script.json --out build/ch01/ch01.mp3
```

Or generate a whole book in one run, resuming when interrupted:

```bash
uv run python -m audiobook book "ebook-pdf/....pdf" --out build
```

`book` writes `build/chNN/script.json`, `build/chNN/chNN.mp3`, a manifest at
`build/book.json` and a single `build/book.mp3`.

`uv run audiobook ...` is equivalent once the environment is synced.

### Pass 1: `script`

```text
audiobook script PDF [--chapter N] [--out PATH] [--preview N]
```

Locates the chapter in the PDF's table of contents, extracts its pages,
classifies blocks, cleans the text and writes `script.json`. It prints a
summary (segments, narrated words, estimated duration).

### Pass 2: `speak`

```text
audiobook speak SCRIPT [--out FILE] [--engine kokoro|lfm3] [--voice ...] [--limit N]
                        [--start N] [--cache DIR] [--refresh] [--no-cache] [--work DIR]
                        [--bitrate 96k] [--gap-within 0.18] [--gap-between 0.45]
                        [--target-lufs -19.0] [--true-peak -2.0] [--no-normalize]
                        [--loudness-passes 1]
```

Chunks are synthesized one at a time, written as 24 kHz mono PCM WAVs, cached,
then concatenated, normalised and encoded to MP3. The command prints per-chunk
timings and a final report (audio duration, generation time, real-time factor,
peak memory, loudness, quality warnings).

Useful options:

- `--engine lfm3` — use the larger LFM3-Audio engine instead of Kokoro.
- `--voice` — Kokoro takes a voice pack such as `af_heart`, `af_bella`,
  `af_nicole`, `am_michael`; `female`/`male` are aliases. LFM3 takes
  `female`/`male`.
- `--speed 0.95` — Kokoro speaking rate (LFM3 ignores it).
- `--no-normalize` — skip loudness normalisation and keep the engine's level.
- `--loudness-passes 2` — measure first, then apply a linear gain. Slower
  (~37 s per chapter instead of ~21 s) and only worth it for archival copies.
- `--limit 3` — smoke-test the first three chunks, then inspect the MP3.
- `--start 40 --limit 5` — regenerate a specific region.
- `--refresh` — ignore cached audio (after changing the voice, for example).
- `--no-cache` — do not read or write the cache.
- `--work DIR` — keep chunk WAVs somewhere other than `<mp3>.chunks/`.

Chunks are cached by a hash of model, sampling settings, max length and exact
text. Editing a segment in `script.json` invalidates only the affected chunks.
Tuning the voice or the chunker leaves old entries behind; prune them with:

```bash
uv run python -m audiobook cache gc -s build/ch*/script.json --dry-run
uv run python -m audiobook cache gc -s build/ch*/script.json
```

### Whole book: `book`

```text
audiobook book PDF [--chapters all|1-34|2,4,6-8] [--out build] [--engine kokoro]
                   [--force] [--refresh] [--cache DIR] [--no-cache]
                   [--normalize/--no-normalize] [--loudness-passes 1|2]
```

One process loads the PDF once, hashes it once and loads the model once for
every chapter. For each chapter it:

1. rebuilds `script.json` only when the text changed (atomic write);
2. skips the chapter when `chNN.mp3` exists and the script is unchanged;
3. otherwise chunks, synthesises, normalises and encodes;

and after each chapter it rewrites `build/book.json` atomically, so an
interrupted run resumes with the same command. The manifest records status,
pages, chunks, duration, loudness, peak memory and warning count per chapter.

At the end the finished chapters are concatenated with `-c copy` into
`build/book.mp3` (no re-encode). MP3 chapter markers and m4b output are still
out of scope.

## `script.json`

```json
{
  "version": 1,
  "source": {
    "pdf": "ebook-pdf/....pdf",
    "sha256": "6cba1924...",
    "book": "Site Reliability Engineering",
    "chapter": 1,
    "title": "Chapter 1. Introduction",
    "first_page": 29,
    "last_page": 38
  },
  "segments": [
    {
      "id": "ch01-0001",
      "type": "prose",
      "text_original": "Chapter 1. Introduction",
      "text_narrated": "Chapter one. Introduction.",
      "origin": "rule",
      "page": 29
    },
    {
      "id": "ch01-0004",
      "type": "figure",
      "text_original": "Figure 1-1.",
      "text_narrated": "Figure one dash one, see the book.",
      "origin": "rule",
      "page": 29,
      "label": "Figure 1-1"
    },
    {
      "id": "ch01-0005",
      "type": "caption",
      "text_original": "Figure 1-1. How a load balancer routes requests.",
      "text_narrated": "How a load balancer routes requests.",
      "origin": "rule",
      "page": 29,
      "label": "Figure 1-1"
    }
  ]
}
```

- `type`: `prose`, `code`, `figure`, `table`, `caption`.
- `origin`: `rule` (generated), `llm` (reserved), `skipped` (kept for review,
  never spoken).
- `text_original` is the cleaned book text (line-break hyphenation resolved);
  `text_narrated` is the exact string sent to the TTS.
- Skipped furniture (running footers, page numbers, watermarks, footnotes)
  stays in the file with `origin: skipped` and an empty `text_narrated`, so the
  review shows what was dropped.

You can edit `text_narrated` by hand and re-run `speak`. `text_original` is
there as a reference; it is not read aloud.

## How it works

### 1. Extract (`audiobook/extract.py`)

PyMuPDF gives blocks, lines and spans with font, size and position, plus the
native TOC. Chapter boundaries come from the TOC (`Chapter N. Title` entries),
not from guessing page numbers. Blocks are sorted top-to-bottom; the PDF's
content-stream order is unreliable (footnotes often come first).

Superscript footnote markers are dropped at the span level: a 6 pt digit inside
10.5 pt body text is a reference, not speech.

A future fallback for scanned pages (LFM3-VL) only has to produce the same
`Block` structures; nothing downstream changes.

### 2. Classify (`audiobook/classify.py`)

Typography-driven rules for this book's design:

- prose/headings: MinionPro body, MyriadPro condensed headings;
- code: UbuntuMono blocks, excluding isolated mono words used as inline terms;
- figures: image blocks, plus a synthetic item for vector figures that only
  have a caption;
- tables: a `Table N-N.` caption plus the following rows in table fonts;
- captions: figure/table captions, read aloud after the announcement;
- furniture: running footer, page number, watermark, footnotes, chapter opener.

The chapter opener is skipped because pass 1 emits a synthetic title segment
(`Chapter one. Introduction.`). Footnote content is skipped in milestone 1.

### 3. Clean (`audiobook/clean.py`)

The same block produces two strings:

- **review**: keeps the author's words, fixes ligatures, quotes, dashes,
  whitespace and line-break hyphenation;
- **speech**: additionally verbalizes acronyms, numbers, identifiers and URLs.

Hyphenation follows the PDF: a line ending in U+2010 is a typeset break inside
a word (`com‐` + `puting` → `computing`); an ASCII hyphen at the end belongs to
a compound (`blame-` + `free` → `blame-free`). Paragraphs split by a page break
are merged, including the hyphen case (`opera‐` + `tional` → `operational`).

Speech normalization (examples):

| input | spoken |
| --- | --- |
| `SRE` | `S R E` |
| `UNIX`, `NASA` | `UNIX`, `NASA` (word) |
| `JSON` | `jason` |
| `2014` | `twenty fourteen` |
| `99.9%` | `ninety-nine point nine percent` |
| `3x` | `three times` |
| `Figure 1-1` | `Figure one dash one` |
| `https://example.com/sre-guide` | `example dot com slash sre guide` |
| `snake_case` | `snake case` |
| `e.g.` | `for example` |

### 4. Narrate (`audiobook/narrate.py`)

`RuleNarrator` turns classified items into speech:

- prose/heading: cleaned text, sentence-terminated;
- code: `Code listing, see the book.`;
- figure/table: `Figure 1-1, see the book.` / `Table 5-2, see the book.`;
- caption: reads the caption body (the figure was already announced).

`Narrator` is a protocol. An `LLMNarrator` can be added later without touching
chunking, caching or assembly; `origin` already distinguishes `rule` from
`llm`. For milestone 1 there is no LLM.

### 5. Chunk (`audiobook/chunk.py`)

Narrated text is split on sentence boundaries into 10–40 second chunks
(120–600 characters), with clause and word fallbacks for long sentences. Short
tails are merged. Chunk ids are `segment-id-cNNNN`, stable across runs.

### 6. Speak (`audiobook/tts/`)

Two engines implement the same `TTSEngine` contract (`load`, `close`,
`synthesize`, `load_peak_memory_gb`), so chunking, caching, warning checks and
assembly are shared. Both write 24 kHz mono 16-bit PCM.

#### Kokoro-82M (default)

`KokoroEngine` runs `mlx_audio.tts` with `mlx-community/Kokoro-82M-bf16` and
takes phonemes from `misaki[en]`, which is declared as a dependency and bundles
its own espeak loader.

```bash
uv run python -m audiobook speak build/ch01/script.json --voice af_heart --speed 0.95
```

What matters:

- **Deterministic.** No temperature, top-k or repetition penalty; `--speed` is
  the only delivery knob, so there is no sampling drift or repetition risk.
  Running the same chunk twice produces the same audio.
- **It reads phonemes.** The G2P stage expands numbers and decides acronyms:
  `S R E`, `SRE` and `S.R.E.` all become "ess ar ee", `UNIX` becomes "yoo-niks",
  `NASA` becomes "nasa", `SQL` becomes "sequel", `96 percent` becomes "ninety
  six percent". `clean.py` needs no engine-specific rules.
- **It pads every generation.** Kokoro adds ~300 ms of lead and ~500 ms of trail
  silence, and anything over ~510 phonemes is split internally at punctuation,
  with each part padded again. `audiobook/tts/kokoro.py` trims each part to
  60 ms, joins parts with a 400 ms pause, then trims the outer edges to 120 ms,
  so the assembler stays in charge of pacing instead of stacking dead air.
- **Voices.** `af_heart` is the default; `af_bella`, `af_nicole`, `am_michael`
  and the British `bf_*`/`bm_*` packs also work. `female`/`male` are aliases,
  and the language code is inferred from the pack prefix.
- **Cache key** includes voice, speed, language and the trimming settings.

#### LFM3-Audio-2B (alternative)

`--engine lfm3` builds the prompt from the model card:

```text
system:    Perform TTS in American English using a female voice.
user:      <chunk text>
assistant: <|audio_start|>
```

and calls `mlx_vlm.generate_audio(..., generation_mode="audio")` with the
speech sampling recommended by Liquid:

```text
audio_temperature          1.0
audio_top_k                128
audio_repetition_penalty   1.1
audio_repetition_window    4096
```

`max_tokens` is measured in 80 ms speech frames: the budget is sized from the
chunk length with 2× headroom, capped at 2048 frames (164 s). The model stops
at its end-of-speech token; the cap only prevents runaway repetition.

> Sampling values from the LFM3-VL card (`temperature=0.1`, `top_p`, ...) are
> for text generation and must **not** be used here.

### 7. Assemble (`audiobook/assemble.py`)

Chunk WAVs are streamed into one WAV with 0.18 s of silence inside a segment
and 0.45 s between segments, then normalised and encoded with ffmpeg:

```text
loudnorm:   I=-19 LUFS, TP=-2 dBTP, LRA=11 (one pass by default)
libmp3lame: 96 kbps, 24 kHz, mono, ID3v2.3 (title/chapter, album/book)
```

By default `loudnorm` runs once and adapts as it goes (~21 s for a 26 min
chapter). `--loudness-passes 2` measures first and then applies a linear gain
with a true-peak ceiling (~37 s), which is exact but rarely needed. The ceiling
is -2 dBTP rather than the usual -1.5 because MP3 encoding overshoots sample
peaks by about half a dB; the finished chapters measure -19.4 LUFS with peaks
of -2.2 dBTP (Kokoro) and -1.3 dBTP (LFM3). Without it, Kokoro sat 4.7 dB lower
and LFM3 clipped at +2.7 dBTP. `--no-normalize` keeps the engine's own level.

The chapter MP3 and the chunk WAVs stay next to each other in `build/`.

## Measured on chapter 1

Harness: MacBook Air M4, 16 GB, MLX 0.32.4, mlx-vlm 0.7.11, mlx-audio 0.6.3.

| metric | LFM3-Audio-2B | Kokoro-82M |
| --- | --- | --- |
| chapter | 1, pages 29–38 | same `script.json` |
| narrated segments | 78 (24 skipped) | same |
| narrated words | 4,146 | same |
| chunks | 97 | 97 |
| audio duration | 26:38 | 26:24 |
| generation time | 13:39 | 2:01 |
| real-time factor | 1.90× | 12.8× |
| peak MLX memory | 3.07 GB (load 2.33 GB) | 3.11 GB (load 0.33 GB) |
| file | 96 kbps, 24 kHz mono, 18.3 MB | same, 18.1 MB |

Both engines were run from the byte-identical `script.json` (sha256
`f2a1eadb…`) with the default gaps, so the difference is the voice alone.

Objective checks on the finished chapters:

| metric | LFM3 | Kokoro |
| --- | --- | --- |
| median F0 | 180 Hz | 195 Hz |
| F0 spread across chunks | 5.5% CV | 4.0% CV |
| within-chunk F0 range | 48% of median | 46% |
| pace | 15.2 chars/s (CV 18.1%) | 16.0 chars/s (CV 10.6%) |
| edge silence | 180 ms lead / 120 ms trail | 120 / 120 ms |
| integrated loudness | −19.4 LUFS (raw −21.6) | −19.4 LUFS (raw −26.3) |
| true peak | −1.3 dBTP (raw +2.7, clipped) | −2.2 dBTP (raw −2.1) |
| ASR round-trip WER (4 chunks, LFM3 as recognizer) | 13.9% | 5.5% |

The ASR test is biased toward LFM3's acoustics, so treat it as a check for
dropped or repeated words rather than an absolute score: LFM3's worst chunk
lost a whole clause, Kokoro's worst dropped a possessive.

## Tests

```bash
uv run pytest                       # unit + integration (skips if the PDF is absent)
uv run pytest -m "not integration"
RUN_MLX=1 uv run pytest -m slow     # loads LFM3
RUN_KOKORO=1 uv run pytest -m slow  # loads Kokoro
```

The book runner and the pass-2 pipeline are tested with a stub engine, so the
suite never loads a model.

The sample PDF is not committed; integration tests are skipped when
`ebook-pdf/` is empty.

## Limitations / next steps

- Code and figures are announced, not described: `see the book` is the rule
  announcement. An LLM narrator can fill this in later.
- Tables are announced and their captions read; rows are not.
- Footnotes are skipped, not read.
- Speech is deterministic with Kokoro but sampled at temperature 1.0 with LFM3,
  whose voice can drift slightly between chunks. The cache makes it cheap to
  regenerate a region (`--start/--limit --refresh`). Chunk-level voice
  verification is not implemented yet.
- Numbers and identifiers are approximated by rules; unusual notations may need
  a manual edit in `script.json`.
- One chapter per script and one MP3 per chapter. Book-wide concatenation and
  m4b chapters are out of scope for milestone 1.
