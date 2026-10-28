"""The effects, one module each, and what each hands the runner.

An effect is two things, found by its id (say `words-behind`):

- `effects/words-behind/effect.json`: its manifest entry (title, line, options and their defaults,
  what to type, the AskEditor ask). `tools/build_manifest.py` assembles every entry into
  `effects.json`, the README's gallery and the table in SKILL.md;
- `fx/effects/words_behind.py`: `NAME`, `OPTIONS` (`{name: kind}` from `options.py`, the same
  names as the entry's options) and `prepare(clip, opts, plan=None) -> Job`, everything decided
  before a frame is drawn. A `plan` (the JSON a previous run wrote) is checked and used instead of
  deciding again. `run.py` renders the job, or draws its stills, through the same code.
  An effect that writes a file instead of a video (`post-copy`) sets `Job.writes`, and names the
  file's ending as `WRITES` (`".post.md"`), so its file sits beside the video as `<video>.post.md`.
  An effect that makes several files at once (`reframe` in every shape) returns a list of jobs, each
  with a `label`; one that only reads a video and reports (`qc-check`) sets `REPORTS = True` and
  gives `report(video, opts, out, still)` instead, which returns the exit code.

Nothing else needs touching to add an effect.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ..core.clip import Clip
from ..core.framing import Framing
from ..core.paths import PACK
from ..core.render import Draw
from ..core.timeline import Timeline

EFFECTS = PACK / "effects"


class PlanError(ValueError):
    """A plan that cannot be rendered; the message says what is wrong with it."""


@dataclass
class Job:
    effect: str
    clip: Clip
    timeline: Timeline
    framing: Framing
    draw: Optional[Draw]
    plan: dict
    checks: dict = field(default_factory=dict)
    still_times: Callable[[], list[float]] = lambda: []
    after: Optional[Callable[[dict, Path], dict]] = None  # runs once the video is written
    # Sounds the effect lays under the clip's own (a click on a press): given the output's sound
    # (samples, 48 kHz stereo float32) and its rate, returns the sound to write, the same length.
    sound: Optional[Callable] = None
    # What a `--still` run shows besides the frames, for an effect whose work is not in a frame (a
    # sound effect draws the planned mix as a picture): given the output path, writes its files and
    # returns {what: where} to print. Nothing is rendered.
    preview: Optional[Callable[[Path], dict]] = None
    # An effect whose work is words, not pictures (the copy to post with a video): given the output
    # path, writes its file there and returns {what: value} to print. No video is rendered and
    # `--still` draws nothing. A `failed` key that is true makes the command exit with 1.
    writes: Optional[Callable[[Path], dict]] = None
    # When `prepare` hands back several jobs (one file each), what tells this one's file from the
    # others: `talk-reframe-4x5.mp4` for the label "4x5". The first job's file is the output itself.
    label: str = ""


def names() -> list[str]:
    """Every effect with a manifest entry, by id."""
    return sorted(p.parent.name for p in EFFECTS.glob("*/effect.json"))


def manifest() -> list[dict]:
    """Every effect's manifest entry, read from its own folder (the source `effects.json` is built from)."""
    return [json.loads(p.read_text()) for p in sorted(EFFECTS.glob("*/effect.json"))]


def entry(name: str) -> dict:
    path = EFFECTS / name / "effect.json"
    if not path.exists():
        raise KeyError(name)
    return json.loads(path.read_text())


def module_name(name: str) -> str:
    return "fx.effects." + name.replace("-", "_")


def load(name: str):
    entry(name)  # an effect is only runnable once it has a manifest entry
    try:
        module = importlib.import_module(module_name(name))
    except ModuleNotFoundError as e:
        if e.name == module_name(name):
            raise KeyError(name) from None
        raise
    spec = set(module.OPTIONS)
    defaults = set(entry(name)["options"])
    if spec != defaults:
        raise RuntimeError(f"{name}: effects/{name}/effect.json and {module_name(name)} disagree on the options "
                           f"({', '.join(sorted(spec ^ defaults))})")
    return module
