# yomu

A TUI for turning a plain-text script into narration, one line at a time.

Each line of `script.txt` is zero-shot cloned from the reference audio in
`refs/` and written to `output/`. Only lines that actually changed are
regenerated. Pick a line, reroll its seed until the delivery is right, adopt the
take you like, and the seed is written back to the script so the result is
reproducible.

The model is
[`mlx-community/Irodori-TTS-v4.1-Small-8bit`](https://huggingface.co/mlx-community/Irodori-TTS-v4.1-Small-8bit).
Compared against the fp16 build, 8bit is good enough (waveform correlation 0.96,
log-spectral distance 3.1 dB) while generating in 0.78× the time with 0.6 GB
less peak memory, so 8bit is what we use.

macOS only — playback goes through `afplay`, and the model runs on MLX.

## Setup

```sh
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python -r requirements.txt
```

On the first run the model (1.3 GB) is downloaded from Hugging Face into
`~/.cache/huggingface`.

You also need at least one reference wav in `refs/` and a `script.txt` to read.
Neither is part of the repository.

## Usage

```sh
./yomu
```

The model is loaded once and stays resident. Move to a line, hit `r` a few times
to hear alternative takes, and `Enter` to adopt the one you want.

| Key | Action |
|---|---|
| `j` `k` `↑` `↓` | Move between lines |
| `space` | Play the current selection from the top (cuts off what is sounding) |
| `r` | Generate a take with an untried seed and play it |
| `←` `→` `0` | Cycle takes (`0` returns to the adopted one) |
| `Enter` | Adopt the take on screen |
| `g` `f` | Generate / regenerate even if up to date |
| `a` | Generate every line that has no wav yet |
| `e` `i` | Edit the line text / edit the instruct prompt |
| `s` | Set the line's duration scale (empty to clear it) |
| `x` `c` | Discard takes / cancel queued jobs |
| `u` `l` `q` | Reload the script / toggle the log / quit |

Adopting writes the seed to `script.txt` and moves the wav into
`output/lineNN-HASH.wav`. Candidate takes live in `output/takes/` and are
**session-only**: that directory is wiped on startup and on exit.

If you edit `script.txt` in an external editor, the status line warns you. Press
`u` to pick the change up — it is never reloaded automatically, so it cannot
clobber an edit made inside the TUI.

## Script format

```
#!duration-scale 0.85

Unity CLI is a program that drives Unity from the command line.
12234, So really, there is nothing it cannot do.
12234, x0.7, And this one is read faster than the rest.
x1.1, Slower, on the default seed.
```

An integer followed by a comma at the start of a line is used as that line's
seed. Lines without one use `--seed` (default 0), so results never change
between runs. Blank lines and lines starting with `#` are ignored. The utterance
number (`lineNN` in the file name) counts only the lines that are not ignored.

`x0.7,` (or `×0.7,`) sets that line's duration scale, overriding the default for
that line alone. It may come before or after the seed, and the value has to be
between 0.1 and 3.0 — anything else is ignored with a warning.

`#!duration-scale 0.85` sets the default for the whole script. It is a comment,
so it does not shift the utterance numbering, and it survives edits made in the
TUI. The first one in the file wins. An explicit `--duration-scale` on the
command line outranks it.

## Output files

`output/lineNN-HASH.wav`, where `HASH` is the first 8 characters of a SHA-256
over **the line text, its seed, and its own duration scale if it has one**.
Because of that:

- only lines whose text, seed or own duration scale changed get regenerated
- wavs carrying a stale hash are deleted automatically
- when inserting or deleting a line shifts the numbering, the affected files are
  renamed rather than regenerated

Giving a line a duration scale, changing it, or clearing it therefore marks that
line — and only that line — for regeneration. A line with no scale of its own
hashes exactly as it did before the feature existed, so existing wavs are not
invalidated by upgrading. Writing `x0.9,` on a line while the default is already
0.9 still costs one regeneration, since the hash cannot depend on the default.

`--instruct`, `--ref` and the *default* duration scale are *not* part of the
hash. After changing one of those, use `f` to force a line to be redone —
including after editing `#!duration-scale`, which `u` will warn you about.

## Options

```sh
./yomu --ref refs/ref_01.wav            # narrow the reference down to one clip
./yomu --duration-scale 1.0             # undo the default speed-up
./yomu --instruct "calm, matter-of-fact"
./yomu --script other.txt --out other   # work on a different script
```

Passing a whole directory as `--ref` enables v4 multi-clip cloning (each clip is
encoded separately, then concatenated). The default is every wav under `refs/`.

The default duration scale is **0.9**, since the model's own pacing tends to
drag. It applies to every line that does not set its own with `x0.7,`. Set it
per script with `#!duration-scale`, or per run with `--duration-scale`, which
wins when both are present. Valid values run from 0.1 to 3.0.

Also available: `--model --seed --seq-len --cfg-mode --num-steps`.

## Measured (M-series 64 GB, 3 reference clips, seed 0)

10 lines / 39.04 s of audio in about 15 s (RTF 2.6). Peak memory 4.2–4.7 GB.

Peak inference memory exceeds 4 GB against 1.3 GB of weights because the default
`cfg_guidance_mode="independent"` runs the CFG branches in parallel. To cut
memory use `--cfg-mode alternating`, and lower `--seq-len` on top of that
(750 ≈ 30 s, 300 ≈ 12 s).

## Notes

- Script I/O, hashing, reconciliation and synthesis all live in `tts_core.py`;
  `tui.py` is a thin shell over it.
- The wavs in `refs/` are already pinned at a peak of 1.0, and the output touches
  1.0 for a few samples as well. It is not audible distortion.
- Feeding a short sentence (under roughly 7 tokens) with no reference audio makes
  the model repeat itself — a known quirk. Not applicable here, since reference
  audio is always passed.
- Other kwargs worth tuning: `t_schedule_mode="sway"` with `sway_coeff`, and
  `cfg_scale_text` / `cfg_scale_speaker` / `cfg_scale_caption` (defaults 3.0 /
  5.0 / 3.0), plus `max_ref_seconds`. Upstream implementation notes are in
  `.venv/lib/python3.12/site-packages/mlx_audio/tts/models/irodori_tts/README.md`.
- The launcher sets `TEXTUAL_DISABLE_KITTY_KEY=1`. Terminals speaking the Kitty
  keyboard protocol attach IME-committed text to the key event, and Textual
  gives up on escape sequences longer than 32 characters -- which a commit of
  5 or more Japanese characters exceeds, leaking the raw sequence into the `e`
  prompt. Run through `./yomu` rather than `tui.py` directly so the opt-out
  applies.
- Streaming generation is not implemented (`stream=True` raises
  NotImplementedError).
