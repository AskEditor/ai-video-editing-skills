"""Effects laid on one clip together: another effect shown inside a render, from a moment on.

An effect that combines others (`prompt-box` answers each prompt with another effect of the pack)
goes through here rather than through their modules directly:

- `split("a | b | c")` -> ["a", "b", "c"]: a list a person types, where a `|` inside quotes stays
  (`headline text="Edit less. | Post more."` is one item).
- `parse("grade look=moody")` -> Spec("grade", {"look": "moody"}): an effect by its id and the
  options given to it, written as on the command line.
- `prepare(clip, spec, start, end, base)` -> Layer: the effect told when to begin and end, prepared
  on the same clip, and checked to fit the render it joins (it must keep the clip's timeline and
  its shape: an effect that cuts the clip or reframes it cannot be laid inside another).
- `Layer.draw(frame, j, t, seen)`: the effect where it is on, blended by `seen` from 0 to 1 (a layer
  can step aside for a while for something laid over it).
- `changes(layer, size, fps, frames)`: what the layer changes on a plain grey frame at those output
  frames: the box round the change and its share of the frame. From it `first_change` (the first
  frame it shows on) and where it sits (so other things can keep clear of it).
- `kind(name)`: `picture`, `behind` or `over`, read from the effect's category in its manifest:
  the order layers are drawn in (a colour look repaints the whole frame, so it goes first; a word
  behind the person next; text and cards over everything).

**Inside a layout.** A layout (`split-layout`) moves the person into a panel, so an effect laid in
it is prepared on a `View` of the clip (`core/view.py`): the canvas or one panel of it, with the
face and the person where the layout puts them and the panel's own safe part, which `prepare`
puts in force while the effect decides where it goes. The other way round is refused: an effect
that moves the person (its module says `MOVES_PERSON = True`) is never laid inside another, since
everything laid with it would still keep clear of where the person was; and `guess` never answers
with one.

**Telling an effect when.** It is told through the first of these options it has, the names the
pack's effects already use:

- `start` (and `end`), in seconds, as `grade` takes them;
- `when` (and `duration` or `hold`, in seconds from that moment), as `headline`, `words-behind` and
  `cta-card` take them;
- `at`, as `before-after` takes it.

An effect told the moment enters by its own animation. An effect with none of these options (such
as `captions`) is shown from the moment with a short fade in, and hidden after the end with a short
fade out. Either way nothing of it shows before the moment. Options the person gave the effect
themselves are kept as given.
"""

from __future__ import annotations

import contextlib
import math
import re
import shlex
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from . import Job, entry, load
from .options import OptionError, parse_all
from ..core.layout import Rect, zone_for

FADE = 0.12  # seconds: how a layer that is not told the moment comes in and goes out
PICTURE = {"colour", "camera", "layout"}  # categories whose effects repaint the whole frame
BEHIND = {"behind-you"}
START_OPTIONS = ("start", "when", "at")
END_OPTIONS = ("end", "duration", "hold")
GREY = 128  # the plain frame a layer is drawn on to see what it changes
SEEN = 3  # a pixel changed by more than this many levels (of 255) counts as changed


@dataclass(frozen=True)
class Spec:
    name: str
    given: dict  # option name -> value, as typed

    def __str__(self) -> str:
        opts = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in self.given.items())
        return f"{self.name} {opts}".strip()


def split(text: str, sep: str = "|") -> list[str]:
    """`text` cut at each `sep` that is not inside quotes, each piece trimmed."""
    out, cur, quote = [], [], None
    for ch in str(text):
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'“”":
            quote = "”" if ch == "“" else ch
            cur.append(ch)
        elif ch == sep:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur).strip())
    return out


def parse(text: str) -> Spec:
    """An effect and its options as typed: `grade look=moody`, `headline text="Big title" style=serif`."""
    raw = str(text).strip().replace("“", '"').replace("”", '"').replace("’", "'")
    try:
        parts = shlex.split(raw)
    except ValueError:
        raise OptionError(f"{text!r} has a quote that is never closed") from None
    if not parts:
        raise OptionError("name an effect, such as grade look=moody")
    name, given = parts[0].strip().lower(), {}
    for p in parts[1:]:
        if "=" not in p:
            raise OptionError(f"after the effect's name, options go as name=value (got {p!r} in {text!r})")
        k, v = p.split("=", 1)
        given[k.strip().replace("-", "_")] = v
    return Spec(name, given)


def kind(name: str) -> str:
    category = entry(name).get("category", "")
    if category in PICTURE:
        return "picture"
    if category in BEHIND:
        return "behind"
    return "over"


def timing(options: dict, start: float, end: Optional[float], given: dict) -> dict:
    """The options that tell an effect with these options when to begin and end (see the module's
    notes). Options the person gave are left as they are."""
    out: dict = {}
    if "start" in options:
        out["start"] = f"{start:.3f}"
        if end is not None and "end" in options:
            out["end"] = f"{end:.3f}"
    elif "when" in options:
        out["when"] = f"{start:.3f}"
        if end is not None:
            if "duration" in options:
                out["duration"] = f"{max(0.0, end - start):.3f}"
            elif "hold" in options:
                out["hold"] = f"{max(0.0, end - start):.3f}"
    elif "at" in options:
        out["at"] = f"{start:.3f}"
    return {k: v for k, v in out.items() if k not in given}


def _smooth(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


@dataclass
class Layer:
    spec: Spec
    opts: dict  # the options it was prepared with, parsed
    job: Job
    kind: str
    start: float  # output seconds: nothing of it shows before
    end: Optional[float]  # nothing after; None: for as long as the effect itself shows it
    fade_in: float  # 0 when the effect was told the moment and enters by itself
    fade_out: float
    told: dict = field(default_factory=dict)  # the options that told it when

    @property
    def name(self) -> str:
        return self.spec.name

    def weight(self, t: float) -> float:
        """How much of the layer shows at `t`, from its window alone."""
        if t < self.start - 1e-9 or (self.end is not None and t >= self.end):
            return 0.0
        w = 1.0
        if self.fade_in > 0:
            w = min(w, _smooth((t - self.start) / self.fade_in))
        if self.end is not None and self.fade_out > 0:
            w = min(w, _smooth((self.end - t) / self.fade_out))
        return w

    def draw(self, frame: np.ndarray, j: int, t: float, seen: float = 1.0) -> None:
        if self.job.draw is None:
            return
        w = self.weight(t) * seen
        if w <= 0.002:
            return
        if w >= 0.998:
            self.job.draw(frame, j, t)
            return
        before = frame.copy()
        self.job.draw(frame, j, t)
        frame[:] = np.clip(before + (frame.astype(np.float32) - before) * w + 0.5, 0, 255).astype(np.uint8)

    def describe(self) -> str:
        told = ", ".join(f"{k}={v}" for k, v in self.told.items())
        how = f"told {told}" if told else (f"shown from {self.start:.2f} s" + (f" to {self.end:.2f} s" if self.end else ""))
        return f"{self.spec} ({how})"


def prepare(clip, spec: Spec, start: Optional[float], end: Optional[float], base, fix: str = "",
            plan: Optional[dict] = None) -> Layer:
    """`spec` prepared on `clip` to show from `start` (output seconds) until `end` (None: as long as
    the effect shows it), in a render framed by `base` (a `Framing`). `start` None tells the effect
    no moment at all: it keeps its own (a headline on the line it picks), and shows whenever it
    shows itself. `plan` is the effect's own plan from an earlier run, used instead of deciding
    again. Raises OptionError in plain words when the effect cannot be laid there."""
    try:
        module = load(spec.name)
    except KeyError:
        from . import names

        raise OptionError(f"there is no effect called {spec.name!r}; the effects are {', '.join(names())}") from None
    if getattr(module, "MOVES_PERSON", False):
        raise OptionError(f"{spec.name} moves you within the frame, so it cannot be laid inside another effect: "
                          f"lay the other effect inside it instead")
    defaults = entry(spec.name)["options"]
    given = dict(spec.given)
    unknown = [k for k in given if k not in module.OPTIONS]
    if unknown:
        raise OptionError(f"{spec.name} has no option {', '.join(unknown)}; it takes {', '.join(module.OPTIONS)}")
    if fix and "fix" in module.OPTIONS and "fix" not in given:
        given["fix"] = fix
    told = timing(module.OPTIONS, start, end, given) if start is not None else {}
    tries = [told, {k: v for k, v in told.items() if k in START_OPTIONS}, {}]
    opts, last = None, None
    for attempt in tries:
        try:
            opts = parse_all(module.OPTIONS, defaults, {**given, **attempt})
            told = attempt
            break
        except OptionError as e:
            last = e
    if opts is None:
        raise OptionError(f"{spec.name}: {last}")
    zone = getattr(clip, "zone", None)  # a view of one panel: its own safe part, in force while it decides
    with zone_for(clip.info.width, clip.info.height, zone) if zone is not None else contextlib.nullcontext():
        job = module.prepare(clip, opts, plan)
    duration = clip.info.frames / clip.info.rate
    if not job.timeline.is_identity(duration):
        raise OptionError(f"{spec.name} changes the clip's length, so it cannot be laid inside another effect; "
                          f"run it first, then this effect on its result")
    fr = job.framing
    if (fr.out_w, fr.out_h) != (base.out_w, base.out_h) or fr.reframed != base.reframed:
        raise OptionError(f"{spec.name} would frame the clip differently ({fr.note}); laid inside another effect "
                          f"it must keep the clip's shape (leave frame at keep)")
    if start is None:  # its own moment: nothing of the layer's window is laid over it
        return Layer(spec=spec, opts=opts, job=job, kind=kind(spec.name), start=0.0, end=None,
                     fade_in=0.0, fade_out=0.0, told={})
    by_itself_in = any(k in told or k in spec.given for k in START_OPTIONS if k in module.OPTIONS)
    by_itself_out = end is None or any(k in told for k in END_OPTIONS)
    return Layer(spec=spec, opts=opts, job=job, kind=kind(spec.name), start=float(start),
                 end=None if by_itself_out else float(end),
                 fade_in=0.0 if by_itself_in else FADE, fade_out=0.0 if by_itself_out else FADE, told=told)


# ---- what a layer changes ---------------------------------------------------------------------

@dataclass(frozen=True)
class Change:
    rect: Optional[Rect]  # the box round every pixel it changes (output pixels); None: nothing
    share: float  # the share of the frame it changes, 0..1


def change_on(layer: Layer, size: tuple[int, int], j: int, fps: float, gated: bool = False) -> Change:
    """What `layer` changes when drawn on a plain grey frame at output frame `j` (the effect's own
    drawing; `gated` applies the layer's window too). Measured on every other pixel."""
    W, H = size
    frame = np.full((H, W, 3), GREY, np.uint8)
    t = j / fps
    if gated:
        layer.draw(frame, j, t)
    elif layer.job.draw is not None:
        layer.job.draw(frame, j, t)
    small = frame[::2, ::2]
    moved = (np.abs(small.astype(np.int16) - GREY) > SEEN).any(axis=2)
    share = float(moved.mean())
    if share <= 0:
        return Change(None, 0.0)
    rows = np.flatnonzero(moved.any(axis=1))
    cols = np.flatnonzero(moved.any(axis=0))
    rect = Rect(float(cols[0] * 2), float(rows[0] * 2), float(min(W, cols[-1] * 2 + 2)), float(min(H, rows[-1] * 2 + 2)))
    return Change(rect, share)


def changes(layer: Layer, size: tuple[int, int], fps: float, frames) -> dict[int, Change]:
    return {int(j): change_on(layer, size, int(j), fps) for j in frames}


def first_change(layer: Layer, size: tuple[int, int], fps: float, n_frames: int, t0: float, t1: float,
                 least: float = 0.0004, gated: bool = False) -> Optional[float]:
    """The first output time between t0 and t1 at which the layer changes at least `least` of the
    frame: the effect's own drawing, or with `gated` as it will be shown (its window applied).
    Looked for every third frame, then frame by frame."""
    a = int(max(0, math.floor(t0 * fps)))
    b = int(min(n_frames - 1, math.ceil(t1 * fps)))
    hit = None
    for j in range(a, b + 1, 3):
        if change_on(layer, size, j, fps, gated).share >= least:
            hit = j
            break
    if hit is None:
        return None
    for j in range(max(a, hit - 2), hit + 1):
        if change_on(layer, size, j, fps, gated).share >= least:
            return j / fps
    return hit / fps


def overlap(a: Optional[Rect], b: Optional[Rect], margin: float = 0.0) -> bool:
    return a is not None and b is not None and a.intersects(b, margin=margin)


# ---- an effect named in a person's own words -----------------------------------------------------

_WORD = re.compile(r"[a-z0-9&']+")
# Words that say nothing about which effect is meant.
_FILLER = set("""a an the it its it's this that these my me i to of in on at for and or with please make
put add give turn set some can could would you your just now then show do get let's lets also more bit
little""".split())
# Effects that cut the clip change its length, so they never answer from inside another effect.
NEVER_GUESSED = {"talking", "cutting", "repurposing"}


def words_of(text: str) -> list[str]:
    return _WORD.findall(str(text).lower().replace("’", "'"))


def _stem(w: str) -> str:
    for end in ("ing", "er", "ed", "s"):
        if len(w) > len(end) + 2 and w.endswith(end):
            return w[: -len(end)]
    return w


def _has(words: list[str], phrase: list[str]) -> bool:
    n = len(phrase)
    return n > 0 and any(words[i:i + n] == phrase for i in range(len(words) - n + 1))


def _named_values(kind: Any) -> list[tuple[list[str], str]]:
    """The values an option takes, each as the words a person would write for it: the value itself
    (`black-and-white` as "black and white"), any plain words the option knows for it, and the
    colours by name for a colour."""
    from ..core.text import COLOURS
    from .options import Colour

    out = []
    for v in list(getattr(kind, "values", ()) or ()) + list(getattr(kind, "names", ()) or ()):
        out.append((words_of(str(v).replace("-", " ")), str(v)))
    for word, v in getattr(kind, "aliases", ()) or ():
        out.append((words_of(word), str(v)))
    if isinstance(kind, Colour):
        out += [([name], name) for name in COLOURS]
    return [(w, v) for w, v in out if w and v not in ("auto", "keep")]


def guess(text: str, exclude: tuple = ()) -> tuple[Optional[Spec], str]:
    """The effect a short sentence asks for, and why: matched against what people say to ask for each
    effect (`say` in its manifest), with any option value named in it set ("make it black and white"
    is grade with look=black-and-white). None, with the reason, when nothing fits well."""
    from . import manifest

    said = words_of(text)
    want = {_stem(w) for w in said if w not in _FILLER}
    if not want:
        return None, "the prompt has no words that name an effect"
    best: Optional[tuple[float, Spec, str]] = None
    for e in manifest():
        if e["name"] in exclude or e.get("category") in NEVER_GUESSED:
            continue
        try:
            module = load(e["name"])
        except Exception:  # an effect that cannot load is never guessed
            continue
        if getattr(module, "MOVES_PERSON", False):  # never laid inside another (see the notes above)
            continue
        top, top_phrase = 0.0, ""
        for phrase in list(e.get("say", [])) + [e.get("title", "")]:
            have = {_stem(w) for w in words_of(phrase) if w not in _FILLER}
            if have:
                score = len(want & have) / len(want | have)
                if score > top:
                    top, top_phrase = score, phrase
        given: dict = {}
        for opt, k in module.OPTIONS.items():
            longest = None
            for phrase, value in _named_values(k):
                if _has(said, phrase) and (longest is None or len(phrase) > len(longest[0])):
                    longest = (phrase, value)
            if longest is not None:
                given[opt] = longest[1]
        score = top + 0.5 * len(given)
        if top > 0 and score > 0.3 and (best is None or score > best[0]):
            why = f"“{text}” is like “{top_phrase}”" if top_phrase else f"“{text}” names its {', '.join(given)}"
            best = (score, Spec(e["name"], given), why)
    if best is None:
        return None, f"no effect of the pack is asked for by “{text}”"
    return best[1], best[2]
