#!/usr/bin/env python
"""yomu -- a TUI for synthesizing, auditioning and adopting script lines.

Backed by Irodori-TTS v4.1 (MLX 8bit). The model stays resident in the process
while you pick a line, reroll its seed with r, and adopt the take you like with
Enter. Adopting writes the seed back to script.txt and commits the wav to
output/lineNN-HASH.wav, so the state on disk always matches the script.

    ./yomu
"""

from __future__ import annotations

import argparse
import queue
import random
import shutil
import subprocess
import threading
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Header, Input, Label, RichLog, Static

from tts_core import (
    DEFAULT_DURATION_SCALE,
    DEFAULT_MODEL,
    DEFAULT_SEED,
    Line,
    SCALE_MAX,
    SCALE_MIN,
    Script,
    Synth,
    Take,
    collect_refs,
    read_take,
    reconcile,
)

TAKES_DIR = "takes"
SEED_MAX = 10000  # Written back to the script, so keep it to four digits


def _width(text: str) -> int:
    """Display width in the terminal (full-width characters count as 2)."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _fit(text: str, limit: int) -> str:
    if _width(text) <= limit:
        return text
    out, used = "", 0
    for char in text:
        step = 2 if unicodedata.east_asian_width(char) in "WF" else 1
        if used + step > limit - 1:
            break
        out, used = out + char, used + step
    return out + "…"


@dataclass
class Job:
    """One entry on the generation queue."""

    index: int  # Utterance number
    seed: int
    text: str
    path: Path
    scale: float  # Resolved duration scale; never None, mlx casts it with float()
    kind: str  # "final" (committed to output/) | "take" (candidate in takes/)
    generation: int  # Script revision; results from before a reload are dropped
    autoplay: bool = False


@dataclass
class Row:
    """One table row = one utterance of the script."""

    line: Line
    adopted: Take | None = None  # Metrics of the wav in output/
    takes: list[Take] = field(default_factory=list)
    take_idx: int | None = None  # None means the adopted take is shown
    busy: bool = False
    queued: bool = False

    @property
    def viewing(self) -> Take | None:
        if self.take_idx is not None:
            return self.takes[self.take_idx]
        return self.adopted


class PromptScreen(ModalScreen[str | None]):
    """Single-line input modal. Esc cancels."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, title: str, value: str = "") -> None:
        super().__init__()
        self.title_text = title
        self.value = value

    def compose(self) -> ComposeResult:
        with Vertical(id="prompt"):
            yield Label(self.title_text)
            yield Input(value=self.value, id="prompt-input")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        self.dismiss(None)


class LineTable(DataTable):
    """The script table.

    Enter arrives as RowSelected and is wired to adopt. Left and right override
    the DataTable default of moving the column cursor and cycle takes instead
    (the cursor is row-based, so column movement is useless here).
    """

    BINDINGS = [
        Binding("left", "app.cycle(-1)", "◀ take"),
        Binding("right", "app.cycle(1)", "take ▶"),
    ]


class TtsApp(App):
    TITLE = "yomu"

    CSS = """
    #status { height: 1; padding: 0 1; background: $panel; }
    LineTable { height: 1fr; }
    RichLog { height: 10; border-top: solid $primary; padding: 0 1; }
    RichLog.hidden { display: none; }
    PromptScreen { align: center middle; }
    #prompt {
        width: 80%; max-width: 110; height: auto;
        border: round $accent; background: $surface; padding: 1 2;
    }
    """

    BINDINGS = [
        Binding("j", "move(1)", "Move", show=False),
        Binding("k", "move(-1)", "Move", show=False),
        Binding("space", "play", "Play"),
        Binding("r", "take", "🎲 Take"),
        Binding("0", "cycle(0)", "Adopted", show=False),
        Binding("enter", "adopt", "Adopt"),
        Binding("g", "generate(False)", "Generate"),
        Binding("f", "generate(True)", "Force"),
        Binding("a", "generate_all", "Generate all"),
        Binding("e", "edit_text", "Edit text"),
        Binding("i", "edit_instruct", "Instruct", show=False),
        Binding("s", "edit_scale", "Scale", show=False),
        Binding("x", "drop_takes", "Drop takes", show=False),
        Binding("c", "cancel_queue", "Cancel queue", show=False),
        Binding("u", "reload", "Reload", show=False),
        Binding("l", "toggle_log", "Log", show=False),
        Binding("q", "quit", "Quit"),
    ]

    COLUMNS = [
        ("no", "#", 3),
        ("status", "status", 12),
        ("seed", "seed", 6),
        ("sec", "sec", 6),
        ("dur", "dur", 5),
        ("text", "text", 48),
    ]

    def __init__(self, args: argparse.Namespace, refs: list[str]) -> None:
        super().__init__()
        self.args = args
        self.script = Script(args.script, args.seed, args.duration_scale)
        self.out_dir = Path(args.out)
        self.takes_dir = self.out_dir / TAKES_DIR
        self.rows: list[Row] = []
        self.generation = 0
        self.dirty_on_disk = False

        gen_kwargs = {
            "sequence_length": args.seq_len,
            "cfg_guidance_mode": args.cfg_mode,
            # Every job passes its own resolved scale, which wins. This is only
            # the fallback that keeps Synth safe for a caller that does not.
            "duration_scale": self.script.default_scale,
        }
        if args.num_steps:
            gen_kwargs["num_steps"] = args.num_steps
        self.synth = Synth(args.model, refs, gen_kwargs, args.instruct)

        self._jobs: queue.Queue[Job] = queue.Queue()
        self._stopping = threading.Event()
        self._player: subprocess.Popen | None = None
        self._model_ready = False
        self._text_width = 48

    # ---------------------------------------------------------------- layout

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="status")
        yield LineTable(cursor_type="row", zebra_stripes=True)
        yield RichLog(id="log", markup=True, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(LineTable)
        for key, label, width in self.COLUMNS:
            table.add_column(label, width=width, key=key)
        table.focus()

        self.out_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(self.takes_dir, ignore_errors=True)
        self.build_rows(first=True)

        threading.Thread(target=self._engine, name="synth", daemon=True).start()
        self.set_interval(1.0, self._poll_disk)
        self.update_status()

    def on_resize(self) -> None:
        # Only the text column follows the terminal width; the rest are fixed.
        fixed = sum(width for _, _, width in self.COLUMNS[:-1]) + 2 * len(self.COLUMNS)
        self._text_width = max(20, self.size.width - fixed)
        table = self.query_one(LineTable)
        if table.columns:
            table.columns["text"].width = self._text_width
            for row in self.rows:
                self.refresh_row(row)

    # ------------------------------------------------------------- row state

    def build_rows(self, first: bool = False) -> None:
        """Rebuild the table from the script, filling in metrics of existing wavs."""
        renamed, removed = reconcile(self.out_dir, self.script.lines)
        for entry in renamed:
            self.log_line(f"[dim]renamed {entry}")
        for name in removed:
            self.log_line(f"[dim]removed {name}")
        for warning in self.script.warnings:
            self.log_line(f"[yellow]{warning}")
        self.script.warnings.clear()

        self.rows = [Row(line=line) for line in self.script.lines]
        for row in self.rows:
            path = self.out_dir / row.line.filename
            if path.exists():
                try:
                    row.adopted = read_take(path, row.line.seed)
                except Exception as exc:  # Still show the row for a broken wav
                    self.log_line(f"[yellow]cannot read {path.name}: {exc}")

        table = self.query_one(LineTable)
        cursor = table.cursor_row if not first else 0
        table.clear()
        for row in self.rows:
            table.add_row(*self.cells(row), key=f"L{row.line.index}")
        if self.rows:
            table.move_cursor(row=min(cursor, len(self.rows) - 1))
        self.dirty_on_disk = False

    def cells(self, row: Row) -> list:
        # Seed and duration show whatever is currently being auditioned, which
        # is the candidate's values while cycling through takes. The scale
        # belongs to the line, so takes inherit it and it never varies here.
        # ASCII "x", not the multiplication sign, whose width is ambiguous.
        take = row.viewing
        return [
            str(row.line.index),
            self.status_cell(row),
            str(take.seed if take else row.line.seed),
            f"{take.seconds:.2f}" if take else "-",
            f"x{row.line.scale:g}" if row.line.scale is not None else "-",
            _fit(row.line.text, self._text_width),
        ]

    def status_cell(self, row: Row) -> Text:
        if row.busy:
            return Text("⋯ working", style="bold yellow")
        if row.queued:
            return Text("… queued", style="yellow")
        if row.take_idx is not None:
            return Text(f"🎲 {row.take_idx + 1}/{len(row.takes)}", style="bold cyan")
        if (self.out_dir / row.line.filename).exists():
            label = "✓ ready" + (f" +{len(row.takes)}" if row.takes else "")
            return Text(label, style="green")
        return Text("● pending", style="red")

    def refresh_row(self, row: Row) -> None:
        table = self.query_one(LineTable)
        key = f"L{row.line.index}"
        for (column, _, _), value in zip(self.COLUMNS, self.cells(row), strict=True):
            table.update_cell(key, column, value)

    @property
    def current(self) -> Row | None:
        table = self.query_one(LineTable)
        if not self.rows:
            return None
        return self.rows[min(table.cursor_row, len(self.rows) - 1)]

    def update_status(self) -> None:
        model = "resident" if self._model_ready else "loading…"
        parts = [
            f"model {model}",
            f"refs {len(self.synth.refs)}",
            f"instruct {self.synth.instruct or '-'}",
            f"dur×{self.script.default_scale:g}",
            f"queue {self._jobs.qsize()}",
        ]
        if self.dirty_on_disk:
            parts.append("[b yellow]script.txt changed on disk (u to reload)")
        self.query_one("#status", Static).update("  ".join(parts))

    def log_line(self, message: str) -> None:
        self.query_one("#log", RichLog).write(message)

    # ---------------------------------------------------------------- engine

    def _log_from_thread(self, message: str) -> None:
        try:
            self.call_from_thread(self.log_line, f"[dim]{message}")
        except Exception:
            pass  # Shutting down

    def _engine(self) -> None:
        """Dedicated generation thread. Processes one job at a time."""
        try:
            wall = self.synth.load(self._log_from_thread)
            self.call_from_thread(self._on_model_ready, wall)
        except Exception as exc:
            self._log_from_thread(f"[red]failed to load the model: {exc}")
            return

        while not self._stopping.is_set():
            try:
                job = self._jobs.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self.call_from_thread(self._on_job_start, job)
                take = self.synth.generate(
                    job.text,
                    job.seed,
                    job.path,
                    self._log_from_thread,
                    duration_scale=job.scale,
                )
            except Exception as exc:
                self.call_from_thread(self._on_job_fail, job, str(exc))
            else:
                self.call_from_thread(self._on_job_done, job, take)

    def _on_model_ready(self, wall: float) -> None:
        self._model_ready = True
        self.log_line(f"[green]model is resident ({wall:.1f}s)")
        self.update_status()

    def _row_for(self, job: Job) -> Row | None:
        """Check that the script has not changed since the job was queued."""
        if job.generation != self.generation or job.index > len(self.rows):
            return None
        row = self.rows[job.index - 1]
        if row.line.text != job.text:
            return None
        # A scale change moves the digest, so the wav no longer belongs to this
        # row -- committing it would leave a filename that lies about its rate.
        return row if self.script.effective_scale(row.line) == job.scale else None

    def _on_job_start(self, job: Job) -> None:
        row = self._row_for(job)
        if row:
            row.busy, row.queued = True, False
            self.refresh_row(row)
        self.update_status()

    def _on_job_fail(self, job: Job, message: str) -> None:
        row = self._row_for(job)
        if row:
            row.busy = False
            self.refresh_row(row)
        self.log_line(f"[red][{job.index:2d}] generation failed: {message}")
        self.update_status()

    def _on_job_done(self, job: Job, take: Take) -> None:
        row = self._row_for(job)
        if row is None:
            self.log_line(f"[yellow][{job.index:2d}] script changed, result discarded")
            take.path.unlink(missing_ok=True)
            self.update_status()
            return

        row.busy = False
        self.log_line(
            f"[{job.index:2d}] seed {take.seed:<6} {take.seconds:5.2f}s / {take.wall:5.2f}s"
            f"  {take.path.name}"
        )
        if job.kind == "final":
            self.prune_line_files(job.index, keep=take.path.name)
            self.drop_takes(row)  # Committed, so the old candidates are moot
            row.adopted = take
        else:
            row.takes.append(take)
            row.take_idx = len(row.takes) - 1
        self.refresh_row(row)
        if job.autoplay:
            self.play(take.path)
        self.update_status()

    def enqueue(self, job: Job) -> None:
        row = self.rows[job.index - 1]
        row.queued = True
        self.refresh_row(row)
        self._jobs.put(job)
        self.update_status()

    # -------------------------------------------------------------- playback

    def play(self, path: Path) -> None:
        self.stop_playback()
        self._player = subprocess.Popen(
            ["afplay", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def stop_playback(self) -> None:
        if self._player and self._player.poll() is None:
            self._player.terminate()
        self._player = None

    # --------------------------------------------------------------- actions

    def action_move(self, delta: int) -> None:
        table = self.query_one(LineTable)
        table.move_cursor(row=max(0, min(len(self.rows) - 1, table.cursor_row + delta)))

    def action_play(self) -> None:
        """Play the selection from the top, cutting off anything still sounding."""
        row = self.current
        if row is None:
            return
        take = row.viewing
        path = take.path if take else self.out_dir / row.line.filename
        if path.exists():
            self.play(path)
        else:
            self.notify("not generated yet (g to generate)", severity="warning")

    def action_generate(self, force: bool) -> None:
        row = self.current
        if row is None:
            return
        path = self.out_dir / row.line.filename
        if path.exists() and not force:
            self.notify("already up to date (f to force)")
            return
        self.enqueue(
            Job(
                index=row.line.index,
                seed=row.line.seed,
                text=row.line.text,
                path=path,
                scale=self.script.effective_scale(row.line),
                kind="final",
                generation=self.generation,
            )
        )

    def action_generate_all(self) -> None:
        pending = [
            row
            for row in self.rows
            if not (self.out_dir / row.line.filename).exists()
            and not (row.busy or row.queued)
        ]
        for row in pending:
            self.enqueue(
                Job(
                    index=row.line.index,
                    seed=row.line.seed,
                    text=row.line.text,
                    path=self.out_dir / row.line.filename,
                    scale=self.script.effective_scale(row.line),
                    kind="final",
                    generation=self.generation,
                )
            )
        self.notify(f"queued {len(pending)} lines")

    def action_take(self) -> None:
        """Make a candidate take with a seed that has not been tried yet."""
        row = self.current
        if row is None:
            return
        tried = {row.line.seed} | {take.seed for take in row.takes}
        seed = random.randrange(1, SEED_MAX)
        for _ in range(50):
            if seed not in tried:
                break
            seed = random.randrange(1, SEED_MAX)
        # The candidate carries the line's scale so the take is auditioned at
        # the same rate it will be adopted at.
        candidate = Line(
            index=row.line.index, text=row.line.text, seed=seed, scale=row.line.scale
        )
        self.enqueue(
            Job(
                index=row.line.index,
                seed=seed,
                text=row.line.text,
                path=self.takes_dir / candidate.filename,
                scale=self.script.effective_scale(row.line),
                kind="take",
                generation=self.generation,
                autoplay=True,
            )
        )

    def action_cycle(self, delta: int) -> None:
        row = self.current
        if row is None or not row.takes:
            return
        if delta == 0:
            row.take_idx = None
        elif row.take_idx is None:
            row.take_idx = 0 if delta > 0 else len(row.takes) - 1
        else:
            nxt = row.take_idx + delta
            row.take_idx = None if not 0 <= nxt < len(row.takes) else nxt
        self.refresh_row(row)
        take = row.viewing
        if take and take.path.exists():
            self.play(take.path)

    def action_adopt(self) -> None:
        row = self.current
        if row is None:
            return
        if row.take_idx is None:
            self.notify("pick a take with ← → first", severity="warning")
            return
        take = row.takes.pop(row.take_idx)
        row.take_idx = None

        self.script.set_seed(row.line.index, take.seed)
        self.script.save()
        dest = self.out_dir / row.line.filename  # Seed updated, so the digest matches
        self.prune_line_files(row.line.index, keep=dest.name)
        shutil.move(str(take.path), dest)
        take.path = dest
        row.adopted = take
        self.refresh_row(row)
        self.log_line(f"[green][{row.line.index:2d}] adopted seed {take.seed} -> {dest.name}")

    def action_edit_text(self) -> None:
        row = self.current
        if row is None:
            return

        def done(value: str | None) -> None:
            if value is None or not value.strip() or value.strip() == row.line.text:
                return
            self.script.set_text(row.line.index, value)
            self.script.save()
            self.drop_takes(row)
            row.adopted = None
            self.refresh_row(row)
            self.log_line(f"[{row.line.index:2d}] text updated (g to generate)")

        self.push_screen(PromptScreen(f"[{row.line.index}] text", row.line.text), done)

    def action_edit_instruct(self) -> None:
        def done(value: str | None) -> None:
            if value is None:
                return
            self.synth.instruct = value.strip() or None
            self.update_status()
            self.log_line("instruct changed (f to redo existing wavs)")

        self.push_screen(PromptScreen("instruct (empty to clear)", self.synth.instruct or ""), done)

    def action_edit_scale(self) -> None:
        row = self.current
        if row is None:
            return

        def done(value: str | None) -> None:
            if value is None:
                return
            body = value.strip().lstrip("x\u00d7").strip()
            if not body:
                scale = None  # Back to inheriting the script default
            else:
                try:
                    scale = float(f"{float(body):g}")
                except ValueError:
                    self.notify(f"not a number: {value.strip()}", severity="warning")
                    return
                if not SCALE_MIN <= scale <= SCALE_MAX:
                    self.notify(
                        f"duration scale must be {SCALE_MIN}-{SCALE_MAX}",
                        severity="warning",
                    )
                    return
            if scale == row.line.scale:
                return
            self.script.set_scale(row.line.index, scale)
            self.script.save()
            self.drop_takes(row)
            row.adopted = None  # The digest moved, so the old wav is stale
            self.refresh_row(row)
            shown = f"x{scale:g}" if scale is not None else "default"
            self.log_line(f"[{row.line.index:2d}] duration scale {shown} (g to generate)")

        self.push_screen(
            PromptScreen(
                f"[{row.line.index}] duration scale "
                f"(empty for the default {self.script.default_scale:g})",
                f"{row.line.scale:g}" if row.line.scale is not None else "",
            ),
            done,
        )

    def action_drop_takes(self) -> None:
        row = self.current
        if row:
            self.drop_takes(row)
            self.refresh_row(row)

    def action_cancel_queue(self) -> None:
        dropped = 0
        while True:
            try:
                job = self._jobs.get_nowait()
            except queue.Empty:
                break
            self.rows[job.index - 1].queued = False
            self.refresh_row(self.rows[job.index - 1])
            dropped += 1
        self.notify(f"cancelled {dropped} queued jobs")
        self.update_status()

    def action_reload(self) -> None:
        self.action_cancel_queue()
        before = self.script.default_scale
        self.script.reload()
        self.generation += 1
        shutil.rmtree(self.takes_dir, ignore_errors=True)
        self.build_rows()
        self.log_line("reloaded script.txt")
        if self.script.default_scale != before:
            # The default is outside the hash, so nothing above is marked stale
            # by it -- every row still reads "ready" at the old rate. Say so.
            message = (
                f"default duration scale {before:g} -> {self.script.default_scale:g};"
                " existing wavs keep the old rate (f to redo a line)"
            )
            self.log_line(f"[yellow]{message}")
            self.notify(message, severity="warning")
        self.update_status()

    def action_toggle_log(self) -> None:
        self.query_one("#log", RichLog).toggle_class("hidden")

    def action_quit(self) -> None:
        self.exit()

    # ----------------------------------------------------------------- misc

    def on_data_table_row_selected(self) -> None:
        self.action_adopt()  # Enter

    def prune_line_files(self, index: int, keep: str) -> None:
        """Delete wavs of the same utterance number carrying an older hash."""
        for path in self.out_dir.glob(f"line{index:02d}-*.wav"):
            if path.name != keep:
                path.unlink()

    def drop_takes(self, row: Row) -> None:
        for take in row.takes:
            take.path.unlink(missing_ok=True)
        row.takes, row.take_idx = [], None

    def _poll_disk(self) -> None:
        changed = self.script.changed_on_disk()
        if changed != self.dirty_on_disk:
            self.dirty_on_disk = changed
            self.update_status()

    def on_unmount(self) -> None:
        self._stopping.set()
        self.stop_playback()
        shutil.rmtree(self.takes_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(prog="yomu", description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--script", default="script.txt")
    parser.add_argument(
        "--ref",
        nargs="+",
        default=["refs"],
        metavar="PATH",
        help="reference wav, or a directory holding them (default: refs)",
    )
    parser.add_argument("--out", default="output", help="wav output directory (default: output)")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"seed for lines without one of their own (default: {DEFAULT_SEED})",
    )
    parser.add_argument("--instruct", help="style prompt passed as a VoiceDesign caption")
    parser.add_argument("--seq-len", type=int, default=750)
    parser.add_argument(
        "--cfg-mode",
        default="independent",
        choices=["independent", "joint", "alternating"],
    )
    parser.add_argument("--num-steps", type=int, help="default 40; lower is faster but coarser")
    parser.add_argument(
        "--duration-scale",
        type=float,
        help="speaking rate multiplier; lower is faster. Outranks a "
        "#!duration-scale directive in the script "
        f"(default: {DEFAULT_DURATION_SCALE})",
    )
    args = parser.parse_args()

    if args.duration_scale is not None and not SCALE_MIN <= args.duration_scale <= SCALE_MAX:
        parser.error(f"--duration-scale must be between {SCALE_MIN} and {SCALE_MAX}")

    if not Path(args.script).exists():
        parser.error(f"script not found: {args.script}")
    refs = collect_refs(args.ref)
    if not refs:
        parser.error(f"no reference wav found in: {', '.join(args.ref)}")
    if missing := [r for r in refs if not Path(r).exists()]:
        parser.error(f"reference audio not found: {', '.join(missing)}")

    TtsApp(args, refs).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
