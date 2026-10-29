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

Speech is generated locally with
[LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B) (MLX, 8-bit) on
the GPU. No API keys, no network at synthesis time.

Milestone 1: one chapter in, one MP3 out. No LLM narration, no translation, no
web UI.

## Requirements

- macOS on Apple Silicon (tested on an M4 with 16 GB)
- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- ffmpeg with libmp3lame (`brew install ffmpeg`)
- ~4 GB of disk for the model, ~2.5 GB of RAM while speaking

## Install

```bash
uv sync
brew install ffmpeg     # if you do not have it
```

The first synthesis downloads `LiquidAI/LFM3-Audio-2B-MLX-8bit` from Hugging
Face (~3 GB) into `~/.cache/huggingface`. After that, everything is offline.

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
audiobook speak SCRIPT [--out FILE] [--voice female|male] [--limit N] [--start N]
                        [--cache DIR] [--refresh] [--no-cache] [--work DIR]
                        [--bitrate 96k] [--gap-within 0.18] [--gap-between 0.45]
```

Chunks are synthesized one at a time, written as 24 kHz mono PCM WAVs, cached,
then concatenated and encoded to MP3. The command prints per-chunk timings and
a final report (audio duration, generation time, real-time factor, peak memory,
quality warnings).

Useful options:

- `--limit 3` — smoke-test the first three chunks, then inspect the MP3.
- `--start 40 --limit 5` — regenerate a specific region.
- `--refresh` — ignore cached audio (after changing the voice, for example).
- `--no-cache` — do not read or write the cache.
- `--work DIR` — keep chunk WAVs somewhere other than `<mp3>.chunks/`.

Chunks are cached by a hash of model, sampling settings, max length and exact
text. Editing a segment in `script.json` invalidates only the affected chunks.

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

`LFM3Engine` builds the prompt from the model card:

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

Output is 24 kHz mono, written as 16-bit PCM WAV.

> Sampling values from the LFM3-VL card (`temperature=0.1`, `top_p`, ...) are
> for text generation and must **not** be used here.

### 7. Assemble (`audiobook/assemble.py`)

Chunk WAVs are streamed into one WAV with 0.18 s of silence inside a segment
and 0.45 s between segments, then encoded with ffmpeg:

```text
libmp3lame, 96 kbps, 24 kHz, mono, ID3v2.3 (title/chapter, album/book)
```

The chapter MP3 and the chunk WAVs stay next to each other in `build/`.

## Measured on chapter 1

Harness: MacBook Air M4, 16 GB, MLX 0.32.4, mlx-vlm 0.7.11.

| metric | value |
| --- | --- |
| chapter | 1, pages 29–38 |
| narrated segments | 78 (24 skipped) |
| narrated words | 4,146 |
| chunks | 97 |
| audio duration | _pending_ |
| generation time | _pending_ |
| real-time factor | _pending_ |
| peak MLX memory | _pending_ |

Quality checks (ASR round-trip, duration outliers, F0 drift) are reported in
the milestone review.

## Tests

```bash
uv run pytest                 # unit + integration (skips if the PDF is absent)
uv run pytest -m "not integration"
RUN_MLX=1 uv run pytest -m slow   # actually loads the speech model
```

The sample PDF is not committed; integration tests are skipped when
`ebook-pdf/` is empty.

## Limitations / next steps

- Code and figures are announced, not described: `see the book` is the rule
  announcement. An LLM narrator can fill this in later.
- Tables are announced and their captions read; rows are not.
- Footnotes are skipped, not read.
- Speech is sampled at temperature 1.0, so the voice can drift slightly between
  chunks. The cache makes it cheap to regenerate a region (`--start/--limit
  --refresh`). Chunk-level voice verification is not implemented yet.
- Numbers and identifiers are approximated by rules; unusual notations may need
  a manual edit in `script.json`.
- One chapter per script and one MP3 per chapter. Book-wide concatenation and
  m4b chapters are out of scope for milestone 1.
