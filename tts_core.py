#!/usr/bin/env python
"""Script parsing and speech synthesis core (Irodori-TTS v4.1, MLX 8bit).

Output files follow the naming convention lineNN-HASH.wav. HASH is the first
8 characters of a SHA-256 over "text + seed", so only lines whose text or seed
changed need to be regenerated; when inserting or deleting lines merely shifts
the numbering, a rename is enough.
"""

from __future__ import annotations

import hashlib
import io
import re
import time
import wave
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np

from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.utils import load

DEFAULT_MODEL = "mlx-community/Irodori-TTS-v4.1-Small-8bit"
DEFAULT_SEED = 0
# Slightly faster than the model's own pacing, which tends to drag.
DEFAULT_DURATION_SCALE = 0.9
HASH_LENGTH = 8

# Leading seed override, e.g. "12234, some text"
SEED_PREFIX = re.compile(r"^(\d+)\s*,\s*(.+)$")
# Only files we wrote ourselves are eligible for cleanup.
OUTPUT_NAME = re.compile(r"^line(\d+)-([0-9a-f]{%d})\.wav$" % HASH_LENGTH)


@dataclass
class Line:
    index: int  # Utterance number (blank and comment lines are not counted)
    text: str
    seed: int

    @property
    def digest(self) -> str:
        """Hash over text and seed. The HASH part of the file name."""
        payload = f"{self.seed}\n{self.text}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:HASH_LENGTH]

    @property
    def filename(self) -> str:
        return f"line{self.index:02d}-{self.digest}.wav"


@dataclass
class Take:
    """One generation result. Used both for candidates and adopted wavs."""

    seed: int
    path: Path
    seconds: float
    wall: float = 0.0


class Script:
    """Reads and writes script.txt while preserving its line structure.

    Comment and blank lines are kept in `raw`, so writing a seed or a text
    edit back from the TUI does not disturb the layout.
    """

    def __init__(self, path: str | Path, default_seed: int = DEFAULT_SEED):
        self.path = Path(path)
        self.default_seed = default_seed
        self.reload()

    def reload(self) -> list[Line]:
        text = self.path.read_text(encoding="utf-8")
        self.trailing_newline = text.endswith("\n") or not text
        self.raw = text.splitlines()
        self.lines = []
        self._row_of: list[int] = []  # utterance number - 1 -> row in `raw`
        for row, raw in enumerate(self.raw):
            body = raw.strip()
            if not body or body.startswith("#"):
                continue
            seed = self.default_seed
            if match := SEED_PREFIX.match(body):
                seed, body = int(match.group(1)), match.group(2).strip()
            self.lines.append(Line(index=len(self.lines) + 1, text=body, seed=seed))
            self._row_of.append(row)
        self._mtime = self._disk_mtime()
        return self.lines

    def _disk_mtime(self) -> int:
        try:
            return self.path.stat().st_mtime_ns
        except OSError:
            return -1

    def changed_on_disk(self) -> bool:
        """Whether an external editor touched the file since our last save()."""
        return self._disk_mtime() != self._mtime

    def _format(self, line: Line) -> str:
        # Omit the prefix for the default seed, matching how scripts are written.
        if line.seed == self.default_seed:
            return line.text
        return f"{line.seed},{line.text}"

    def _write_row(self, index: int) -> None:
        line = self.lines[index - 1]
        self.raw[self._row_of[index - 1]] = self._format(line)

    def set_seed(self, index: int, seed: int) -> None:
        self.lines[index - 1].seed = seed
        self._write_row(index)

    def set_text(self, index: int, text: str) -> None:
        self.lines[index - 1].text = text.strip()
        self._write_row(index)

    def save(self) -> None:
        body = "\n".join(self.raw) + ("\n" if self.trailing_newline else "")
        self.path.write_text(body, encoding="utf-8")
        self._mtime = self._disk_mtime()


def collect_refs(paths: Iterable[str]) -> list[str]:
    """Normalize a mix of file and directory arguments into a list of wavs."""
    refs: list[str] = []
    for entry in paths:
        path = Path(entry)
        if path.is_dir():
            refs.extend(str(p) for p in sorted(path.glob("*.wav")))
        else:
            refs.append(str(path))
    return refs


def reconcile(out_dir: Path, lines: list[Line]) -> tuple[list[str], list[str]]:
    """Bring the output directory in line with the script.

    Since the hash depends only on text and seed, a file whose contents still
    match is renamed rather than regenerated when inserting or deleting lines
    shifts the numbering. Otherwise every line after the edit would be redone.
    Files left over after that are deleted.
    """
    existing = {
        path.name: OUTPUT_NAME.match(path.name).group(2)
        for path in sorted(out_dir.glob("line*.wav"))
        if OUTPUT_NAME.match(path.name)
    }
    expected = {line.filename for line in lines}
    renamed, removed = [], []

    for line in lines:
        if line.filename in existing:
            continue
        stale = [
            name
            for name, digest in existing.items()
            if digest == line.digest and name not in expected
        ]
        if stale:
            (out_dir / stale[0]).rename(out_dir / line.filename)
            existing.pop(stale[0])
            existing[line.filename] = line.digest
            renamed.append(f"{stale[0]} -> {line.filename}")

    for name in sorted(existing):
        if name not in expected:
            (out_dir / name).unlink()
            removed.append(name)

    return renamed, removed


def read_take(path: Path, seed: int) -> Take:
    """Read the duration of an existing wav from its header (no decoding)."""
    with wave.open(str(path), "rb") as wav:
        seconds = wav.getnframes() / wav.getframerate()
    return Take(seed=seed, path=path, seconds=seconds)


class _LineWriter(io.TextIOBase):
    """Feeds printed text to a callback one line at a time."""

    def __init__(self, sink: Callable[[str], None]):
        self._sink = sink
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._sink(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            self._sink(self._buffer)
        self._buffer = ""


@contextmanager
def _captured(log: Callable[[str], None] | None) -> Iterator[None]:
    """Intercept the [INFO] chatter mlx-audio prints. Pass-through if log is None.

    Essential for the TUI: anything written to stdout corrupts the screen.
    """
    if log is None:
        yield
        return
    writer = _LineWriter(log)
    with redirect_stdout(writer), redirect_stderr(writer):
        try:
            yield
        finally:
            writer.flush()


class Synth:
    """Holds the model and synthesizes one line at a time.

    Loading is deferred until the first generation.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        refs: Iterable[str] = (),
        gen_kwargs: dict | None = None,
        instruct: str | None = None,
    ):
        self.model_id = model_id
        self.refs = list(refs)
        self.gen_kwargs = dict(gen_kwargs or {})
        self.instruct = instruct
        self.model = None

    @property
    def ref_audio(self) -> str | list[str]:
        # Passing several clips enables v4 multi-clip cloning.
        return self.refs if len(self.refs) > 1 else self.refs[0]

    def load(self, log: Callable[[str], None] | None = None) -> float:
        """Load the model and return the elapsed seconds. 0.0 if already loaded."""
        if self.model is not None:
            return 0.0
        t0 = time.perf_counter()
        with _captured(log):
            self.model = load(self.model_id)
        return time.perf_counter() - t0

    def generate(
        self,
        text: str,
        seed: int,
        out_path: Path,
        log: Callable[[str], None] | None = None,
        **overrides,
    ) -> Take:
        self.load(log)
        kwargs = {**self.gen_kwargs, **overrides}
        t0 = time.perf_counter()
        with _captured(log):
            result = next(
                self.model.generate(
                    text=text,
                    ref_audio=self.ref_audio,
                    instruct=self.instruct,
                    rng_seed=seed,
                    **kwargs,
                )
            )
        wall = time.perf_counter() - t0

        audio = np.asarray(result.audio, dtype=np.float32)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        audio_write(out_path, audio, result.sample_rate)
        seconds = audio.size / result.sample_rate
        return Take(seed=seed, path=out_path, seconds=seconds, wall=wall)
