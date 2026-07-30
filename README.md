# gpt-from-scratch

A GPT-2 implementation built from scratch in PyTorch ([src/gpt.py](src/gpt.py)), loaded with
OpenAI's original pretrained GPT-2 weights and then fine-tuned two different ways:

1. **Instruction fine-tuning** — general instruction-following (Alpaca-style Q&A).
2. **Lyrics fine-tuning** — generating song verses/lyrics in a particular artist's style.

Both fine-tunes start from the same pretrained checkpoint and are independent of each other
(neither is chained on top of the other). A third script lets you chat interactively with any
resulting checkpoint.

## Layout

| Path | Purpose |
|---|---|
| [src/gpt.py](src/gpt.py) | `GPTModel` architecture, `generate()` sampling loop, `train_model_simple()` training loop, OpenAI checkpoint weight loader |
| [src/utils.py](src/utils.py) | device selection, GPT-2 weight download, loss helpers, checkpoint save, loss plotting |
| [src/fine_tune_instruction_response.py](src/fine_tune_instruction_response.py) | fine-tune script 1: instruction following |
| [src/fine_tune_lyrics_response.py](src/fine_tune_lyrics_response.py) | fine-tune script 2: lyrics generation |
| [src/load_execute_model.py](src/load_execute_model.py) | interactive prompt loop for any fine-tuned checkpoint |
| [src/scrape_lyrics_data.py](src/scrape_lyrics_data.py) | optional data-collection script — scrapes raw song lyrics |
| `data/` | datasets (`*.json`) and checkpoints (`*.pth`, git-ignored) |
| `gpt2/` | pretrained OpenAI GPT-2 weights, downloaded on first run and cached by size |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install torch tiktoken tensorflow numpy requests matplotlib tqdm beautifulsoup4
```

`beautifulsoup4` is only needed for the optional scraper. Tested with `torch 2.6.0+cu124`,
`tensorflow 2.21.0` (used solely to read OpenAI's original TensorFlow checkpoint format),
`tiktoken 0.13.0`, `numpy 2.5.1`.

**Run everything from the repository root** (`python src/<script>.py`). The fine-tune scripts
use paths like `data/...` and `gpt2/...` relative to the current working directory, so running
from elsewhere will re-download weights or write files to the wrong place.

`configure_device()` in [src/utils.py](src/utils.py) picks CUDA if available, otherwise MPS
(on `torch >= 2.9`), otherwise CPU — no flags needed.

## Fine-tuning script 1: instructions ([src/fine_tune_instruction_response.py](src/fine_tune_instruction_response.py))

Fine-tunes GPT-2 to follow generic instructions, in the `### Instruction / ### Input / ### Response`
format popularized by Alpaca.

- **Data**: `data/instruction-data-basic.json` (1,100 instruction/input/output triples). If the
  file is missing, it's auto-downloaded from the [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
  ch07 dataset. Split 85% train / 10% test / 5% validation.
- **Base model**: pretrained `gpt2-medium (355M)` OpenAI weights, downloaded into `gpt2/355M/`
  on first run (~1.4 GB, cached after).
- **Training**: batch size 8, 2 epochs, AdamW at `lr=5e-5`, `weight_decay=0.1`.
- **Output**: `data/gpt2-medium355M-sft.pth` and `loss-plot.pdf` (training/validation loss curves).

Run:

```bash
python src/fine_tune_instruction_response.py
```

## Fine-tuning script 2: lyrics ([src/fine_tune_lyrics_response.py](src/fine_tune_lyrics_response.py))

Fine-tunes GPT-2 to write song lyrics/verses on a given topic, using the same
`### Instruction / ### Response` prompt format but with lyric-writing instructions as input and
full verses as output.

- **Data**: `data/instruction-data-lyrics-large-large.json` (1,065 lyric-writing prompt/output
  pairs), shuffled with a fixed seed before the 85/10/5 split.
  > This file must already exist — if it's missing, the script's download fallback fetches the
  > same **generic** instruction dataset used by script 1, not lyrics data.
- **Base model**: pretrained `gpt2-medium (355M)` OpenAI weights (fresh from OpenAI, not from
  the instruction-tuned checkpoint above).
- **Training**: micro-batch size 1 with 16-step gradient accumulation (effective batch size 16)
  to fit longer lyric sequences in memory, dropout 0.05, 2 epochs, AdamW at `lr=2e-5`.
- **Output**: `data/gpt2-medium355M-lyrics-sft.pth` and `loss-plot.pdf`.

Run:

```bash
python src/fine_tune_lyrics_response.py
```

### Building the lyrics dataset

[src/scrape_lyrics_data.py](src/scrape_lyrics_data.py) collects raw song lyrics (by default,
Drake's solo tracks, via MusicBrainz for the song list and AZLyrics/LRCLIB for lyrics text) into
a JSON file. Its output has empty `instruction`/`input` fields and raw lyrics as `output` — a
separate step (not included here) turns that raw text into the prompt/verse pairs consumed by
`fine_tune_lyrics_response.py`. Example:

```bash
python src/scrape_lyrics_data.py --source lrclib --output data/instruction-data-lyrics-raw.json
```

## Running a fine-tuned model ([src/load_execute_model.py](src/load_execute_model.py))

Interactive REPL that loads any checkpoint from `data/`, infers the model size/config from the
checkpoint's tensor shapes, and auto-selects a prompt style (`lyrics` vs `instruction`) based on
whether `"lyrics"` appears in the filename.

```bash
python src/load_execute_model.py                                   # defaults to data/gpt2-medium355M-sft.pth
python src/load_execute_model.py --model gpt2-medium355M-lyrics-sft-2.pth
```

Optional flags override the style's defaults:

| Flag | Meaning |
|---|---|
| `--temperature` | `0.0` = greedy decoding; higher = more random |
| `--top-k` | sample only from the k most likely next tokens |
| `--max-new-tokens` | response length cap |
| `--repetition-penalty` | `> 1.0` discourages repeating tokens; `1.0` disables |

Once running, type a prompt and press Enter:

```
Prompt > Write a verse about summer nights in the city
```

Use `instruction :: input` to fill the optional `### Input:` field, e.g.:

```
Prompt > Rewrite the following sentence in passive voice :: The cat chased the mouse.
```

Type `exit`, `quit`, or `:q` (or Ctrl-C) to leave.

## Notes

- Checkpoints are written atomically ([save_checkpoint](src/utils.py) writes to a temp file,
  verifies every tensor round-trips, then renames into place), so an interrupted save can't
  silently leave a corrupt `.pth`. If `load_execute_model.py` ever hits one anyway, it'll tell
  you to delete it and re-run the fine-tune script.
- `*.pth` checkpoints and the `gpt2/` weights directory are git-ignored — expect to regenerate
  or re-download them after a fresh clone.
