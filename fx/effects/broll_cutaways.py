"""broll-cutaways: your own B-roll cut in over the words that name it, while your voice runs on.

Three layouts (`mode`):

- **full**: the B-roll fills the frame for a moment, then you are back;
- **split**: the B-roll above, you below, for as long as it is up;
- **inset**: the B-roll fills the frame and you stay in a small window in a top corner.

How it decides, in order:

1. **Which clip goes where: by name and tag, never by looking.** `map` (a word said -> a clip)
   first. Then the words in each clip's file name (`lake.mp4`) and in a `tags.txt` beside the
   clips, matched to the words said, a plural matching its singular. A clip named only with a
   camera's number (`IMG_5922.MOV`) says nothing about what it shows, so those go, in folder order,
   on the strongest words left: the things named (nouns, places, things, found by rules: never
   grammar, filler or a doing word, never a word said again). A clip whose name or tags are never
   said is left out and listed, rather than laid on a word it does not show.
2. **When.** A cutaway starts on the first frame of its word, timed from when the voice begins
   (the transcriber marks words late; the core pulls each start back and checks it against the
   sound). It runs one to three seconds and comes back to you between two words, never inside
   one: at a pause or the end of a clause when there is one near, else where one word ends and the
   next begins. Two things named close together cut from one B-roll straight to the next. The film
   opens on you, you are on screen at least a second between cutaways, and B-roll never covers
   more than half the film unless `max_share` allows it.
3. **The framing.** An upright clip is shown as filmed. A wide one is cropped to the frame round
   where it moves and has the most detail, and pushes in slowly while it is up, as a photo does. The
   stretch of a clip used is its steadiest, sharpest one with something moving in it, never the
   first half second, when the button is still being pressed. An HDR clip is brought down to normal
   range as the core reads the talking clip, so it does not look washed out.
4. **The sound.** Your voice runs on under every cutaway, untouched. The B-roll's own sound is off,
   or with `broll_audio=duck` laid well under your voice, coming in just before the picture and
   trailing just after it.
5. **Captions stay on top.** `captions=on` (or a look's name) adds them over everything, so they
   carry across each cutaway; in split and inset they keep clear of where your face now is. A clip
   that already has this pack's captions keeps them on screen across full-frame cutaways.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from ..core import match, moments
from ..core.layout import Rect, SafeZone
from ..core.match import keywords, name_words, read_tags, stem  # noqa: F401 (read by tests and other effects)
from ..core.paths import CACHE
from ..core.words import HESITATIONS, Word, pause_after
from . import captions as cap
from . import options as O

NAME = "broll-cutaways"
MODES = ("full", "split", "inset")
TRANSITIONS = ("cut", "dip", "slide")
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
PHOTO_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
BROLL = match.Kind(frozenset(VIDEO_EXT | PHOTO_EXT), "clips or photos", "B-roll",
                   "a clip or a photo this pack reads (clips: mp4, mov; photos: jpg, png, webp)", "your B-roll",
                   "give your B-roll: broll=\"~/Movies/broll\" (a folder) or files separated by commas")

# ---- the options -------------------------------------------------------------------------------


def parse_map(text: str) -> list[tuple[str, str]]:
    """`{"lake": "lake.mp4"}` or `lake=lake.mp4; city=rooftops.mp4` -> [("lake", "lake.mp4"), …]."""
    try:
        return match.parse_map(text)
    except ValueError as e:
        raise O.OptionError(str(e)) from None


@dataclass(frozen=True)
class _Files:
    help: str

    def parse(self, raw) -> str:
        return str(raw).strip()


@dataclass(frozen=True)
class _Map:
    help: str

    def parse(self, raw) -> str:
        return "; ".join(f"{w}={c}" for w, c in parse_map(str(raw)))


@dataclass(frozen=True)
class _On:
    help: str

    def parse(self, raw) -> str:
        v = str(raw).strip().strip("\"'").lower()
        if v in ("", "auto", "keywords", "keyword", "words"):
            return "keywords"
        if v in ("map", "names", "tags", "named"):
            return "map"
        if v.startswith("times"):
            v = v[5:].lstrip(" =:")
        parts = [p for p in re.split(r"[,;\s]+", v) if p]
        try:
            times = [float(p.rstrip("s")) for p in parts]
        except ValueError:
            raise O.OptionError(f"say keywords, map, or seconds such as 2.5, 6, 11 (got {raw!r})") from None
        if not times:
            raise O.OptionError("say keywords, map, or seconds such as 2.5, 6, 11")
        if any(t < 0 for t in times):
            raise O.OptionError("times start at 0 seconds")
        return ", ".join(f"{t:g}" for t in sorted(times))


@dataclass(frozen=True)
class _Duration:
    help: str

    def parse(self, raw) -> str:
        v = str(raw).strip().lower().rstrip("s")
        if v in ("", "auto"):
            return "auto"
        try:
            d = float(v)
        except ValueError:
            raise O.OptionError(f"give auto, or seconds from 1 to 3 (got {raw!r})") from None
        if not 1.0 <= d <= 3.0:
            raise O.OptionError(f"give seconds from 1 to 3 (got {raw!r})")
        return f"{d:g}"


CAPTIONS = ("auto", "off", "on") + tuple(cap.LOOKS)

OPTIONS = {
    "broll": _Files("your B-roll: a folder, or clips and photos separated by commas. A clip named for what it "
                    "shows (lake.mp4), or listed in a tags.txt beside it, lands where you say those words"),
    "map": _Map("which clip goes on which word you say: word=clip pairs separated by ; (lake=lake.mp4; "
                "city=rooftops.mp4) or JSON ({\"lake\": \"lake.mp4\"}); a bare name is looked for among your B-roll"),
    "mode": O.Choice(MODES, "full: the B-roll fills the frame · split: the B-roll above, you below · inset: the "
                            "B-roll fills the frame and you stay in a small window in a top corner"),
    "on": _On("keywords: clips land on the words that name them, and clips with only a camera's number for a "
              "name on the strongest things you say, in folder order · map: only on words named by map, a file "
              "name or tags.txt · or seconds, such as 2.5, 6, 11, for the clips in folder order"),
    "duration": _Duration("auto: each cutaway runs 1 to 3 seconds, coming back to you at a pause or the end of "
                          "a clause near two seconds · or seconds from 1 to 3"),
    "max_share": O.Number(0.05, 1.0, "the most of the film the B-roll may cover, from 0.05 to 1 (0.5 is half)"),
    "transition": O.Choice(TRANSITIONS, "cut: straight to the B-roll on the word · dip: through black · slide: "
                                        "the B-roll pushes in (in split it slides down, in inset you shrink "
                                        "into your corner)"),
    "broll_audio": O.Choice(("mute", "duck"), "mute: only your voice · duck: the B-roll's own sound well "
                                              "under your voice, in just before its picture and out just after"),
    "captions": O.Choice(CAPTIONS, "auto: a clip that already has this pack's captions keeps them over "
                                   "full-frame cutaways · off · on (stacked) or a look: stacked, one-word, "
                                   "serif-chip, fitted, drawn over everything"),
    "frame": O.Choice(("keep", "9:16"), "keep the clip's own shape, or make it 9:16 following your face"),
    "fix": O.Text("misheard words as wrong=right, separated by ;  e.g. \"Jon=John\""),
}

# The runs `tests/matrix.py broll-cutaways` makes, on our own B-roll (links in the lab's
# inputs; `hua-hin/` carries a tags.txt, `named/` is named for what each clip shows, `airport/` and
# `scenery/` keep the cameras' numbers).
_IN = "{lab}/broll-cutaways/inputs"
MATRIX = [
    # the places named: town, city and hotel by tags.txt (the hotel a photo), captions carried across
    ("D", "D", {"broll": f"{_IN}/hua-hin", "captions": "on"}),
    ("D-split", "D", {"broll": f"{_IN}/hua-hin", "mode": "split", "captions": "on"}),
    # a close shot split: the hardest framing for the panel; map puts two clips on two words
    ("A-split", "A", {"broll": f"{_IN}/named", "mode": "split", "map": "hooks=plane.mp4; campaign=station.mov",
                      "transition": "slide"}),
    # camera-numbered clips on the strongest words: a wide clip cropped and pushed in, you in a corner
    ("B-inset", "B", {"broll": f"{_IN}/airport", "mode": "inset", "transition": "slide", "captions": "on"}),
    # the hard clip: trees and people passing; dips through black
    ("C", "C", {"broll": f"{_IN}/scenery", "transition": "dip"}),
    # 44 s of real talk: many cutaways, the share held, captions on top, the B-roll's sound ducked
    ("33", "33", {"broll": f"{_IN}/scenery", "captions": "on", "broll_audio": "duck"}),
]

# ---- the rules, in seconds (chosen for this pack) ------------------------------------------------

MIN_LEN, MAX_LEN = 1.0, 3.0  # how long one cutaway runs
TARGET = 2.0  # `duration=auto` aims here, and a pause or a clause end nearby pulls it
MIN_SPEAKER = 1.2  # you are on screen at least this long between two cutaways
OPENING = 1.0  # the film opens on you: no cutaway on a keyword before this
OPENING_NAMED = 0.4  # a word you name, or a clip named for it, may cut in from here
END_ROOM = 0.6  # never come back to you for less than this at the very end
CHAIN_MAX = 3.5  # one B-roll runs straight into the next only if it is up no longer than this
HEAD_SKIP, TAIL_SKIP = 0.5, 0.3  # never used of a clip: the press of the button, and the stop
PUSH = 0.06  # how far a wide clip or a photo pushes in while it is up
DIP = 0.10  # dip: to black and back, each way
SLIDE = 0.28  # slide: how long the move takes, centred on the word
SEAM = (0.40, 0.50)  # split: the B-roll's share of the height, fitted to how much of you there is to show
FACE_FLOOR = 0.83  # split: your chin stays above this share of the height (the app's caption and buttons)
INSET_W, INSET_ASPECT = 0.34, 1.25  # inset: the window's width as a share of the frame's, and its height to width
DUCK_UNDER = 18.0  # dB the B-roll's own sound sits under your voice
J_LEAD, L_TRAIL = 0.15, 0.30  # its sound starts this long before its picture and trails this long after
SNAP = 0.3  # a time given in `on` within this of a word's start moves onto it

# Which clip goes on which word (map, file names, tags.txt, the things named) is the core's
# matcher, `fx/core/match.py`, shared with the other effects that lay a person's own files on words.


# ---- the B-roll ----------------------------------------------------------------------------------

@dataclass
class Broll:
    path: str
    kind: str  # "video" or "photo"
    width: int
    height: int
    seconds: float  # how long a clip runs (a photo: forever)
    info: object = None  # a clip's VideoInfo
    image: Optional[np.ndarray] = None  # a photo, RGB
    names: tuple = ()  # the words in its file name
    tags: tuple = ()  # its phrases in tags.txt
    transfer: str = ""  # "hlg" or "pq" for an HDR clip
    has_audio: bool = False

    @property
    def label(self) -> str:
        return Path(self.path).name

    @property
    def named(self) -> bool:
        return bool(self.names or self.tags)

    @property
    def upright(self) -> bool:
        return self.height > self.width

    @property
    def usable(self) -> float:
        """The longest cutaway it can carry."""
        return math.inf if self.kind == "photo" else max(0.0, self.seconds - HEAD_SKIP - TAIL_SKIP)


def broll_paths(spec: str) -> list[Path]:
    """The files `spec` names: a folder's clips and photos in name order, or a list of files."""
    return match.files(spec, BROLL)


def _transfer(path: str) -> str:
    run = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=color_transfer", "-of", "csv=p=0", path], capture_output=True, text=True)
    t = run.stdout.strip().lower()
    return "hlg" if t.startswith("arib-std-b67") else "pq" if t.startswith("smpte2084") else ""


def load_broll(path: Path) -> Broll:
    from ..core import video

    if path.suffix.lower() in VIDEO_EXT:
        info = video.probe(path)
        return Broll(str(path), "video", info.width, info.height, info.frames / info.rate, info=info,
                     names=name_words(path), tags=match.tags_for(path), transfer=_transfer(str(path)),
                     has_audio=info.has_audio)
    from PIL import Image, ImageOps

    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA")
                bg = Image.new("RGBA", im.size, (0, 0, 0, 255))
                im = Image.alpha_composite(bg, im)
            rgb = np.asarray(im.convert("RGB"))
    except OSError:
        raise ValueError(f"{path.name} could not be read as a photo") from None
    return Broll(str(path), "photo", rgb.shape[1], rgb.shape[0], math.inf, image=rgb,
                 names=name_words(path), tags=match.tags_for(path))


def load_pool(spec: str, mapping: list[tuple[str, str]]) -> tuple[list[Broll], dict[str, int]]:
    """The B-roll in folder order, and which clip each mapped word names (word -> index)."""
    return match.pool(spec, mapping, BROLL, load_broll)


# ---- the words: where you may come back -----------------------------------------------------------

def boundaries(words: list[Word], quiet, env, seconds: float) -> list[tuple[float, int]]:
    """Every moment between two words where the picture may come back to you, with how good a place
    it is: 2 or 3 in a real pause (its quietest point, and every tenth of a second across a long
    one, so a cutaway can hold through the pause and come back as you speak again), 1 at the end of
    a clause, 0 where one word runs straight into the next."""
    out = []
    for i, w in enumerate(words):
        a = w.end
        b = words[i + 1].start if i + 1 < len(words) else seconds
        if b - a >= 0.06 and env is not None:
            t = min(b - 0.01, max(a + 0.01, env.quietest(a, b)))
        else:
            t = a
        gap = pause_after(words, i, quiet) if quiet is not None else max(0.0, b - a)
        clause = 1 if w.ends_clause else 0
        q = (2 if gap >= 0.25 else 1 if gap >= 0.12 else 0) + clause
        out.append((float(t), q))
        if gap >= 0.3 and b - a >= 0.3:
            for u in np.arange(a + 0.15, b - 0.12 + 1e-6, 0.1):
                if abs(u - t) > 0.05:
                    out.append((float(u), 2 + clause - 1))
    return sorted(out)


def inside_word(t: float, words: list[Word], slack: float = 0.03) -> Optional[Word]:
    """The word being said at `t`, if `t` falls inside one."""
    for w in words:
        if w.start + slack < t < w.end - slack:
            return w
    return None


# ---- deciding the cutaways --------------------------------------------------------------------------

@dataclass
class Cut:
    start: float
    end: float = 0.0
    clip: Optional[int] = None  # the B-roll, by its place in folder order
    word: str = ""  # what is being said as it lands
    why: str = ""  # map, tags, name, keyword, time, spread
    priority: int = 0
    score: float = 0.0
    chained: bool = False  # runs straight into the next cutaway
    after: bool = False  # follows straight on from the one before
    in_point: float = 0.0  # seconds into the B-roll where it starts
    crop: tuple = (0.5, 0.5)  # the point of the B-roll the frame is centred on (shares)
    push: float = 0.0  # how far it pushes in while up
    fit: str = ""  # how it was framed, in words
    want: Optional[float] = None  # the length it aims for, when not the usual


class Timing:
    """Where a cutaway may end, and whether a run of them fits."""

    def __init__(self, bounds: list[tuple[float, int]], seconds: float, duration: str, budget: float,
                 usable):
        self.bounds, self.seconds, self.budget, self.usable = bounds, seconds, budget, usable
        if duration == "auto":
            self.target, self.weight = TARGET, 0.35
        else:
            self.target, self.weight = float(duration), 0.12

    def end_for(self, start: float, lo: float, hi: float, least: int = 0,
                want: Optional[float] = None) -> Optional[float]:
        """The best place to come back to you between `lo` and `hi` seconds after `start`, among
        places at least `least` good (1: a clause end or a pause)."""
        cands = [(t, q) for t, q in self.bounds if start + lo - 1e-6 <= t <= start + hi + 1e-6 and q >= least]
        if not cands:
            return None
        aim = start + min(hi, max(lo, want if want is not None else self.target))
        return min(cands, key=lambda tq: abs(tq[0] - aim) - self.weight * tq[1])[0]

    def lay(self, cuts: list[Cut]) -> Optional[list[Cut]]:
        """The cutaways in time order with their ends, or None when they cannot all be laid (`why`
        says which and why): a clip too short, a word too near the end, or two too close together
        either to show you between them or to run one straight into the other."""
        cuts = sorted(cuts, key=lambda c: c.start)
        out: list[Cut] = []
        self.why = ""
        for i, c in enumerate(cuts):
            if self.seconds - c.start < MIN_LEN - 1e-6:
                self.why = f"“{c.word}” at {c.start:.2f} s is too near the end for a cutaway of {MIN_LEN:g} s"
                return None
            hi = min(MAX_LEN, self.usable(c), self.seconds - c.start)
            if hi < MIN_LEN - 1e-6:
                self.why = "too short to carry a cutaway"
                return None
            after = bool(out and out[-1].chained)
            end = self.end_for(c.start, MIN_LEN, hi, want=c.want)
            if end is None:  # no words after it: run its length, or to the end
                end = min(self.seconds, c.start + min(hi, self.target))
            nxt = cuts[i + 1] if i + 1 < len(cuts) else None
            chained = False
            if nxt is not None and nxt.start < end + MIN_SPEAKER:
                # Either it comes back to you sooner, at a clause end or a pause, so you are seen
                # between them; or it runs straight into the next (never three in a row on keywords
                # alone, so your face comes back; things you named one after another may run on);
                # or, failing both, it comes back between two words.
                room = nxt.start - MIN_SPEAKER - c.start
                gap = nxt.start - c.start
                can_chain = (MIN_LEN <= gap <= min(CHAIN_MAX, self.usable(c))
                             and (not after or (c.priority >= 2 and nxt.priority >= 2)))
                shorter = self.end_for(c.start, MIN_LEN, room, least=1) if room >= MIN_LEN else None
                if shorter is None and can_chain:
                    end, chained = nxt.start, True
                else:
                    if shorter is None and room >= MIN_LEN:
                        shorter = self.end_for(c.start, MIN_LEN, room)
                    if shorter is None:
                        self.why = f"“{c.word}” at {c.start:.2f} s and “{nxt.word}” at {nxt.start:.2f} s are too close"
                        return None
                    end = shorter
            if not chained and self.seconds - end < END_ROOM:
                # The film ends on you when it can; else on the B-roll, never on a sliver of you.
                shorter = self.end_for(c.start, MIN_LEN, self.seconds - END_ROOM - c.start)
                if shorter is not None:
                    end = shorter
                elif self.seconds - c.start <= min(MAX_LEN + END_ROOM, self.usable(c)):
                    end = self.seconds
                else:
                    self.why = f"“{c.word}” at {c.start:.2f} s is too near the end"
                    return None
            out.append(replace(c, end=float(end), chained=chained, after=after))
        return out

    @staticmethod
    def total(cuts: list[Cut]) -> float:
        return sum(c.end - c.start for c in cuts)


def decide(words: list[Word], quiet, env, pool: list[Broll], named: dict[str, int], on: str, duration: str,
           max_share: float, seconds: float) -> tuple[list[Cut], dict[int, str], list[str]]:
    """The cutaways, why each clip not used was left out, and notes for the person. `words` are the
    clip's whole words, fillers and all, so what the speaker took back is found (`moments.taken_back`)
    and no clip lands on it."""
    said = match.said(words)
    back = moments.taken_back(words, quiet)
    timing = Timing(boundaries(words, quiet, env, seconds), seconds, duration, max_share * seconds,
                    lambda c: pool[c.clip].usable if c.clip is not None else MAX_LEN)
    left: dict[int, str] = {}
    notes: list[str] = []
    cands: list[Cut] = []
    mapped = set(named.values())
    for i, b in enumerate(pool):
        if b.usable < MIN_LEN:
            left[i] = f"only {b.seconds:.1f} s long; a cutaway needs at least {MIN_LEN + HEAD_SKIP + TAIL_SKIP:.1f} s"
    if on not in ("keywords", "map"):
        times = [float(x) for x in on.split(",")]
        order = [i for i in range(len(pool)) if i not in left]
        for k, t in enumerate(times):
            if k >= len(order):
                notes.append(f"{len(times)} times for {len(order)} clips: the last {len(times) - k} not used")
                break
            near = [w for w in said if abs(w.start - t) <= SNAP]
            w = min(near, key=lambda w: abs(w.start - t)) if near else None
            start = float(w.start) if w else t
            if start >= seconds - MIN_LEN:
                raise ValueError(f"{t:g} s is too near the end of the clip ({seconds:.2f} s) for a cutaway")
            cands.append(Cut(start, clip=order[k], word=cap.bare(w) if w else "", why="time", priority=3))
        unnamed = [i for i, b in enumerate(pool) if not b.named and i not in mapped and i not in left]
    else:
        found, unnamed = match.candidates(said, quiet, pool, named, on, left, back)
        cands += [Cut(c.start, clip=c.item, word=c.word, why=c.why, priority=c.priority, score=c.score)
                  for c in found]
    if on == "keywords" and unnamed and not said:
        notes.append("no speech was found, so the B-roll is spread evenly; on=2,6,10 sets the times")
        n = max(1, min(len(unnamed), int(max_share * seconds / TARGET)))
        for k in range(n):
            cands.append(Cut(seconds * (k + 1) / (n + 1), why="spread", priority=1))

    # The strongest first: a word you mapped, then a clip named for a word, then the things named.
    placed: list[Cut] = []
    used: set[int] = set()
    keyword_slots = len(unnamed)
    for c in sorted(cands, key=lambda c: (-c.priority, -c.score, c.start)):
        if c.clip is not None and c.clip in used:
            continue
        if c.clip is None and keyword_slots <= 0:
            continue
        if c.start < (OPENING if c.priority <= 1 else OPENING_NAMED):
            if c.clip is not None and c.clip not in used:
                left.setdefault(c.clip, f"its word “{c.word}” comes in the opening second, which stays on you")
            continue
        laid = timing.lay(placed + [c])
        if laid is None:
            if c.clip is not None:
                left.setdefault(c.clip, timing.why or f"“{c.word}” at {c.start:.2f} s does not fit")
            continue
        if Timing.total(laid) > timing.budget + 1e-6:  # it may still fit if it is shorter
            short = replace(c, want=MIN_LEN + 0.2)
            laid = timing.lay(placed + [short])
            if laid is not None and Timing.total(laid) <= timing.budget + 1e-6:
                c = short
        if laid is None or Timing.total(laid) > timing.budget + 1e-6:
            if c.clip is not None:
                left.setdefault(c.clip, f"one more cutaway would cover more than {max_share:.0%} of the film")
            else:
                notes.append(f"more of the things you name would take the B-roll past {max_share:.0%} of the film")
            continue
        placed.append(c)
        if c.clip is not None:
            used.add(c.clip)
            left.pop(c.clip, None)
        else:
            keyword_slots -= 1
    # The clips with only a number for a name take the things named, in folder order.
    picked = [c for c in placed if c.clip is None]
    for c, idx in match.folder_order(picked, unnamed):
        c.clip = idx
    if on == "keywords":
        for idx in unnamed[len(picked):]:
            left.setdefault(idx, "no strong word was left for it")
    final = timing.lay(placed) or []
    while placed and not final:  # a clip shorter than the moment it was given: drop the weakest
        placed.remove(min(placed, key=lambda c: (c.priority, c.score)))
        final = timing.lay(placed) or []
    used = {c.clip for c in final}
    left = {i: why for i, why in left.items() if i not in used}
    for idx in range(len(pool)):
        if idx not in used:
            left.setdefault(idx, "not used")
    return final, left, notes


# ---- looking at a clip: its best stretch, and where in it to look ----------------------------------

ANALYSIS_VERSION = 2


def _cache_file(path: str, what: str) -> Path:
    st = os.stat(path)
    key = hashlib.sha1(f"{os.path.realpath(path)}|{st.st_size}|{int(st.st_mtime)}|{what}|{ANALYSIS_VERSION}"
                       .encode()).hexdigest()[:20]
    folder = CACHE / "broll"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{key}.json"


def _small(b: Broll, long_side: int = 192) -> tuple[int, int]:
    if b.width >= b.height:
        return long_side, max(2, int(round(long_side * b.height / b.width / 2)) * 2)
    return max(2, int(round(long_side * b.width / b.height / 2)) * 2), long_side


def _gray_frames(b: Broll, size: tuple[int, int], start: Optional[float] = None, length: Optional[float] = None,
                 fps: Optional[float] = None, keys_only: bool = False) -> tuple[np.ndarray, list[float]]:
    """Small grey frames of a clip and their times: every key frame (fast: nothing else is decoded),
    or `fps` frames a second over a stretch."""
    sw, sh = size
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "info"]
    if keys_only:
        cmd += ["-skip_frame", "nokey"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", b.path]
    if length:
        cmd += ["-t", f"{length:.3f}"]
    vf = (f"fps={fps}," if fps else "") + f"scale={sw}:{sh}:flags=area,format=gray,showinfo"
    cmd += ["-an", "-sn", "-map", "0:v:0", "-vf", vf, "-vsync", "vfr", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    run = subprocess.run(cmd, capture_output=True)
    n = len(run.stdout) // (sw * sh)
    frames = np.frombuffer(run.stdout[: n * sw * sh], np.uint8).reshape(n, sh, sw)
    times = [float(x) for x in re.findall(rb"pts_time:\s*(-?[0-9.]+)", run.stderr)][:n]
    if len(times) < n:
        times += [(start or 0.0) + k / (fps or 1.0) for k in range(len(times), n)]
    if start and times and times[0] < start - 0.5:  # times reported from the seek point, not the file's start
        times = [t + start for t in times]
    return frames, times


def stretch_stats(b: Broll) -> dict:
    """For each key frame of a clip: how sharp it is, how bright, how far the camera moved since the
    one before (a share of the frame a second) and how much moved within the picture. Kept in the
    pack's own cache, never beside your clip."""
    import cv2

    cache = _cache_file(b.path, "stretch")
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except ValueError:
            pass
    size = _small(b)
    frames, times = _gray_frames(b, size, keys_only=True)
    if len(frames) < 3 or (len(times) > 1 and max(np.diff(times)) > 2.5):
        frames, times = _gray_frames(b, size, fps=2)
    sw, sh = size
    f = frames.astype(np.float32)
    sharp = [float(cv2.Laplacian(x, cv2.CV_32F).var()) for x in f]
    bright = [float(x.mean() / 255.0) for x in f]
    cam, moved = [0.0], [0.0]
    hann = cv2.createHanningWindow((sw, sh), cv2.CV_32F)
    m = max(1, int(0.1 * min(sw, sh)))
    for i in range(1, len(f)):
        # copies: the correlation multiplies what it is given by the window, in place
        (dx, dy), _ = cv2.phaseCorrelate(f[i - 1].copy(), f[i].copy(), hann)
        dt = max(1e-3, times[i] - times[i - 1])
        cam.append(float(math.hypot(dx, dy) / max(sw, sh) / dt))
        shifted = cv2.warpAffine(f[i], np.float32([[1, 0, -dx], [0, 1, -dy]]), (sw, sh), borderMode=cv2.BORDER_REFLECT)
        moved.append(float(np.abs(shifted[m:-m, m:-m] - f[i - 1][m:-m, m:-m]).mean() / 255.0))
    out = {"t": [round(t, 3) for t in times], "sharp": sharp, "bright": bright, "cam": cam, "moved": moved}
    tmp = cache.with_suffix(f".{os.getpid()}.part")
    tmp.write_text(json.dumps(out))
    tmp.replace(cache)
    return out


def best_stretch(b: Broll, length: float) -> float:
    """Where in a clip to start a cutaway `length` long: the stretch where the camera is steadiest
    and the picture sharpest, with something moving in it; never the first half second, nor the
    last moments, when the camera is being started or stopped."""
    if b.kind == "photo":
        return 0.0
    lo, hi = HEAD_SKIP, b.seconds - TAIL_SKIP - length - SLIDE
    if hi <= lo:
        return max(0.0, min(lo, (b.seconds - length) / 2))
    st = stretch_stats(b)
    t = np.array(st["t"], np.float64)
    if len(t) < 2:
        return lo
    sharp = np.log1p(np.array(st["sharp"]))
    sharp -= np.median(sharp)
    cam = np.array(st["cam"])
    moved = np.array(st["moved"])
    bright = np.array(st["bright"])
    best, best_score = lo, -math.inf
    for s in np.arange(lo, hi + 1e-6, 0.25):
        sel = (t >= s - 0.35) & (t <= s + length + 0.35)
        if not sel.any():
            sel = np.abs(t - (s + length / 2)) == np.abs(t - (s + length / 2)).min()
        score = (sharp[sel].mean() - 6.0 * cam[sel].mean() + 0.6 * min(1.0, 12.0 * moved[sel].mean())
                 - 0.8 * max(0.0, abs(bright[sel].mean() - 0.5) - 0.25))
        if score > best_score + 1e-9:
            best, best_score = float(s), score
    return best


def interest(b: Broll, start: float, length: float) -> np.ndarray:
    """Where a clip is worth looking at over a stretch: its detail (edges) and what moves in it, as a
    small map from 0 to 1."""
    import cv2

    size = _small(b, 256)
    if b.kind == "photo":
        g = cv2.cvtColor(cv2.resize(b.image, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        frames = g[None].astype(np.float32)
    else:
        raw, _ = _gray_frames(b, size, start=start, length=max(0.5, length), fps=8)
        frames = raw.astype(np.float32) if len(raw) else np.zeros((1, size[1], size[0]), np.float32)
    detail = np.zeros(frames.shape[1:], np.float32)
    for x in frames:
        gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
        detail += np.sqrt(gx * gx + gy * gy)
    detail /= len(frames)
    motion = np.abs(np.diff(frames, axis=0)).mean(axis=0) if len(frames) > 1 else np.zeros_like(detail)

    def norm(a: np.ndarray) -> np.ndarray:
        p = float(np.percentile(a, 99))
        return np.clip(a / p, 0, 1) if p > 1e-6 else np.zeros_like(a)

    has_motion = float(np.percentile(motion, 99)) > 4.0
    m = 0.55 * norm(detail) + 0.45 * norm(motion) if has_motion else norm(detail)
    sigma = 0.03 * max(size)
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigma)


def crop_centre(imap: np.ndarray, src_aspect: float, target_aspect: float) -> tuple[tuple[float, float], str]:
    """The point a crop of `target_aspect` from a picture of `src_aspect` is centred on: the window
    that holds the most of the map, leaning a little toward the middle, where people frame things."""
    h, w = imap.shape
    if src_aspect > target_aspect * 1.01:
        share, prof, axis = target_aspect / src_aspect, imap.sum(axis=0), "across"
    elif src_aspect < target_aspect / 1.01:
        share, prof, axis = src_aspect / target_aspect, imap.sum(axis=1), "up and down"
    else:
        return (0.5, 0.5), "whole"
    n = len(prof)
    x = (np.arange(n) + 0.5) / n
    # Where nothing stands out clearly, the middle wins: that is where the person filming put it.
    prof = prof * (0.35 + 0.65 * np.exp(-0.5 * ((x - 0.5) / 0.24) ** 2))
    win = max(1, int(round(share * n)))
    if win >= n:
        return (0.5, 0.5), "whole"
    sums = np.convolve(prof, np.ones(win), mode="valid")
    k = int(np.argmax(sums))
    c = (k + win / 2) / n
    return ((c, 0.5) if axis == "across" else (0.5, c)), axis


# ---- drawing the B-roll ------------------------------------------------------------------------------

def _even(v: float) -> int:
    return max(2, int(round(v / 2)) * 2)


class Reel:
    """One cutaway's B-roll, drawn at `size` for any moment it is up: the crop round its centre of
    interest, read at the output's frame rate from its in-point, pushing in when it should."""

    def __init__(self, b: Broll, cut: Cut, size: tuple[int, int], fps: float, tonemap: bool):
        self.b, self.cut, self.size, self.fps, self.tonemap = b, cut, size, fps, tonemap
        tw, th = size
        k1 = max(tw / b.width, th / b.height)  # the scale at which the clip just covers the frame
        cw, ch = min(b.width, tw / k1), min(b.height, th / k1)
        cx, cy = cut.crop[0] * b.width, cut.crop[1] * b.height
        x0 = float(np.clip(cx - cw / 2, 0, b.width - cw))
        y0 = float(np.clip(cy - ch / 2, 0, b.height - ch))
        self.region = (x0, y0, cw, ch)
        self.read = (_even(tw * (1 + cut.push)), _even(th * (1 + cut.push)))
        self.centre = ((cx - x0) / cw, (cy - y0) / ch)  # where the push heads, within the region
        self._proc = None
        self._n = -1  # the last frame number read
        self._open_at = 0.0
        self._last = None
        self._still = None
        if b.kind == "photo":
            import cv2

            img = b.image[int(y0):int(round(y0 + ch)), int(x0):int(round(x0 + cw))]
            self._still = cv2.resize(img, self.read, interpolation=cv2.INTER_AREA)

    def _open(self, at: float) -> None:
        self.close()
        b = self.b
        x0, y0, cw, ch = self.region
        crop = f"crop={_even(cw)}:{_even(ch)}:{int(x0) // 2 * 2}:{int(y0) // 2 * 2}"
        chain = [f"fps={self.fps:.6f}", crop]
        matrix = getattr(b.info, "color_matrix", "bt709")
        rw, rh = self.read
        chain.append(f"scale={rw}:{rh}:flags=bicubic+accurate_rnd+full_chroma_int:in_color_matrix={matrix}")
        cwd = None
        if self.tonemap:  # the core's own conversion, so the B-roll matches the talking clip beside it
            from ..core import hdr

            to_sdr, cwd = hdr.reader_filter(b.transfer, None)
            chain += ["format=rgb48le", to_sdr]
        chain.append("format=rgb24")
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{max(0.0, at):.4f}", "-i", b.path, "-map", "0:v:0",
               "-vf", ",".join(chain), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=rw * rh * 3 * 2,
                                      cwd=cwd)
        self._open_at = max(0.0, at)
        self._n = -1
        self._last = None

    def _raw(self, tb: float) -> np.ndarray:
        if self._still is not None:
            return self._still
        tb = float(np.clip(tb, 0.0, max(0.0, self.b.seconds - 1.0 / self.fps)))
        want = int(math.floor((tb - self._open_at) * self.fps + 1e-6))
        if self._proc is None or want < self._n or want > self._n + int(self.fps):
            self._open(tb)
            want = 0
        rw, rh = self.read
        while self._n < want:
            buf = self._proc.stdout.read(rw * rh * 3) if self._proc and self._proc.stdout else b""
            if len(buf) < rw * rh * 3:
                break
            self._last = np.frombuffer(buf, np.uint8).reshape(rh, rw, 3)
            self._n += 1
        if self._last is None:
            self._last = np.zeros((rh, rw, 3), np.uint8)
        return self._last

    def frame(self, t: float) -> np.ndarray:
        """The B-roll at output time `t` (seconds since the film began)."""
        import cv2

        c = self.cut
        img = self._raw(c.in_point + (t - c.start))
        tw, th = self.size
        if c.push <= 0:
            return img if img.shape[1] == tw and img.shape[0] == th else cv2.resize(img, (tw, th),
                                                                                  interpolation=cv2.INTER_AREA)
        u = float(np.clip((t - c.start) / max(1e-3, c.end - c.start), 0.0, 1.0))
        z = 1.0 + c.push * u
        rw, rh = self.read
        ww, wh = rw / z, rh / z
        x0 = float(np.clip(self.centre[0] * rw - ww / 2, 0, rw - ww))
        y0 = float(np.clip(self.centre[1] * rh - wh / 2, 0, rh - wh))
        m = np.float32([[tw / ww, 0, -x0 * tw / ww], [0, th / wh, -y0 * th / wh]])
        return cv2.warpAffine(img, m, (tw, th), flags=cv2.INTER_AREA if ww > tw * 1.02 else cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)

    def close(self) -> None:
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.kill()
            self._proc.wait()
            self._proc = None


def frame_cut(b: Broll, cut: Cut, size: tuple[int, int], length: float) -> Cut:
    """The cut with its stretch of the B-roll, the point it is centred on, and whether it pushes in."""
    tw, th = size
    target = tw / th
    in_point = best_stretch(b, length)
    src = b.width / b.height
    if b.upright and b.kind == "video" and abs(src - target) < 0.02:
        return replace(cut, in_point=in_point, crop=(0.5, 0.5), push=0.0, fit="upright, shown as filmed")
    imap = interest(b, in_point, length)
    centre, axis = crop_centre(imap, src, target)
    push = PUSH if (not b.upright or b.kind == "photo") else 0.0
    if b.kind == "photo":
        fit = f"a photo, centred on its detail, pushing in {push:.0%}"
    elif axis == "whole":
        fit = "shown whole" + (f", pushing in {push:.0%}" if push else "")
    else:
        where = centre[0] if axis == "across" else centre[1]
        what = "wide" if not b.upright else "upright"
        fit = (f"{what}, cropped {axis} round its detail and motion at {where:.0%}"
               + (f", pushing in {push:.0%}" if push else ""))
    return replace(cut, in_point=in_point, crop=centre, push=push, fit=fit)


# ---- a contact sheet of the B-roll, for a person or an agent to look at ------------------------------

def _short(why: str) -> str:
    """Why a clip was left out, in the few words a sheet's label holds."""
    for part, short in (("never said", "never said"), ("no strong word", "no word left"),
                        ("long", "too short"), ("on=map", "on=map"), ("opening", "in the opening"),
                        ("near the end", "too near the end"), ("too close", "no room there"),
                        ("cover more", "share full")):
        if part in why:
            return short
    return "not used"


def broll_sheet(pool: list[Broll], cuts: list[Cut], left: dict[int, str]) -> Optional[Path]:
    """One frame of every clip (its middle; a photo whole), numbered and named, with the word it
    landed on or why it was left out. The pack itself never looks at the pictures; this sheet is for
    the person, or the agent running the pack, to look at and write a `map` from. Kept in the pack's
    own cache, never beside the clips."""
    from PIL import Image, ImageDraw

    from ..core import text as textmod
    from ..core import video

    if not pool:
        return None
    key = hashlib.sha1("|".join(f"{b.path}:{c.start if (c := next((x for x in cuts if x.clip == i), None)) else ''}"
                                for i, b in enumerate(pool)).encode()).hexdigest()[:16]
    folder = CACHE / "broll"
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"sheet-{key}.jpg"
    tw, th, pad, cols = 216, 384, 8, 6
    try:
        font = textmod.font("poppins-semibold", 15)
    except Exception:  # a missing face never stops a render for the sake of a sheet
        font = None
    tiles = []
    for i, b in enumerate(pool):
        try:
            if b.kind == "photo":
                img = Image.fromarray(b.image)
            else:
                img = Image.fromarray(video.read_frame(b.info, b.info.frame_at(b.seconds / 2)))
        except Exception:
            img = Image.new("RGB", (tw, th), (60, 60, 60))
        img.thumbnail((tw, th), Image.LANCZOS)
        tile = Image.new("RGB", (tw, th + 52), (24, 24, 24))
        tile.paste(img, ((tw - img.width) // 2, (th - img.height) // 2))
        d = ImageDraw.Draw(tile)
        cut = next((c for c in cuts if c.clip == i), None)
        line2 = f"on “{cut.word}” at {cut.start:.1f} s"[:34] if cut else "left out: " + _short(left.get(i, ""))
        name = b.label if len(b.label) <= 24 else "…" + b.label[-23:]  # a camera's names differ at the end
        d.text((6, th + 6), f"{i + 1}. {name}", fill=(240, 240, 240), font=font)
        d.text((6, th + 28), line2, fill=(140, 220, 140) if cut else (200, 160, 120), font=font)
        tiles.append(tile)
    rows = math.ceil(len(tiles) / cols)
    n = min(cols, len(tiles))
    sheet = Image.new("RGB", (n * tw + (n - 1) * pad, rows * (th + 52) + (rows - 1) * pad), (40, 40, 40))
    for k, tile in enumerate(tiles):
        sheet.paste(tile, ((k % cols) * (tw + pad), (k // cols) * (th + 52 + pad)))
    sheet.save(out, quality=82)
    return out


# ---- you, when the B-roll is up: the layouts -------------------------------------------------------

def _smooth(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def _lerp_rect(a: Rect, b: Rect, e: float) -> Rect:
    return Rect(a.x0 + (b.x0 - a.x0) * e, a.y0 + (b.y0 - a.y0) * e, a.x1 + (b.x1 - a.x1) * e, a.y1 + (b.y1 - a.y1) * e)


@lru_cache(maxsize=32)
def rounded_alpha(w: int, h: int, r: float) -> np.ndarray:
    """A w×h mask with round corners of radius r, smooth at the edge."""
    a = np.ones((h, w), np.float32)
    r = float(min(r, w / 2, h / 2))
    if r < 0.5:
        return a
    k = int(math.ceil(r))
    yy, xx = np.mgrid[0:k, 0:k].astype(np.float32) + 0.5
    d = np.sqrt((xx - r) ** 2 + (yy - r) ** 2) - r
    corner = np.clip(0.5 - d, 0, 1)
    corner[(xx > r) | (yy > r)] = 1.0
    a[:k, :k] = np.minimum(a[:k, :k], corner)
    a[:k, w - k:] = np.minimum(a[:k, w - k:], corner[:, ::-1])
    a[h - k:, :k] = np.minimum(a[h - k:, :k], corner[::-1, :])
    a[h - k:, w - k:] = np.minimum(a[h - k:, w - k:], corner[::-1, ::-1])
    return a


@dataclass
class View:
    """Where you are shown while a cutaway is up: `src`, a window of your frame, drawn into `dst`."""
    src: Rect
    dst: Rect
    radius: float = 0.0


class Layout:
    """How you and the B-roll share the frame in one `mode`."""

    def __init__(self, mode: str, size: tuple[int, int]):
        self.mode = mode
        self.W, self.H = size
        self.full = Rect(0, 0, self.W, self.H)
        self.seam = _even(SEAM[1] * self.H)
        iw = _even(INSET_W * min(self.W, self.H * 9 / 16))  # the same size on a wide frame as a tall one
        ih = _even(iw * INSET_ASPECT)
        self.inset_size = (iw, ih)
        self.inset = Rect(0, 0, iw, ih)  # placed in its corner by `place_inset`
        self.corner = ""
        self.radius = 0.085 * iw
        self.border = max(2.0, 0.0035 * self.W)

    @property
    def broll_size(self) -> tuple[int, int]:
        return (self.W, self.seam) if self.mode == "split" else (self.W, self.H)

    def fit_seam(self, heads: list[Rect], faces: list[Rect]) -> None:
        """Split: where the B-roll meets your panel, for the whole film. Your panel gets the room your
        head needs (hair to chin, as you sit in the frame) above the app's bar, between half and three
        fifths of the height; a smaller head leaves the B-roll more."""
        if not heads or not faces:
            return
        hh = float(np.median([h.h for h in heads]))
        fh = float(np.median([f.h for f in faces]))
        z = self._zoom(fh, FACE_FLOOR * self.H - SEAM[1] * self.H)
        need = hh * z + 0.03 * self.H
        seam = FACE_FLOOR * self.H - need
        self.seam = _even(float(np.clip(seam, SEAM[0] * self.H, SEAM[1] * self.H)))

    @staticmethod
    def _zoom(face_h: float, band: float) -> float:
        """How much closer a small face is brought in your panel, so it still reads."""
        return float(np.clip(0.45 * band / max(1.0, face_h), 1.0, 1.6))

    def place_inset(self, side: str, level: str = "top") -> None:
        """The inset's corner: inside the safe zone, at the top unless your captions sit up there."""
        iw, ih = self.inset_size
        zone = SafeZone(self.W, self.H).rect
        y0 = zone.y0 + 0.015 * self.H if level == "top" else zone.y1 - 0.015 * self.H - ih
        x0 = zone.x0 if side == "left" else zone.x1 - iw
        self.inset = Rect(x0, y0, x0 + iw, y0 + ih)
        self.corner = f"{level} {side}"

    def target_view(self, face: Optional[Rect], head: Optional[Rect] = None) -> View:
        """Your window when the cutaway is fully up, from your face's box (brows to chin) and your
        head's (hair to chin) over it, in output pixels."""
        W, H = self.W, self.H
        if self.mode == "split":
            dst = Rect(0, self.seam, W, H)
            floor = FACE_FLOOR * H
            band = floor - self.seam  # the part of your panel above the app's caption and buttons
            if face is None:
                return View(Rect(0, H - dst.h, W, H), dst)
            head = head or face
            z = self._zoom(face.h, band)
            sw, sh = W / z, dst.h / z
            margin = 0.012 * H
            # Your head's middle a little above the middle of the band, then: your chin above the
            # app's bar first, and as much of your hair under the seam as there is room for.
            y0 = head.cy - 0.47 * band / z
            lo = face.y1 - (floor - margin - self.seam) / z  # chin at the bar
            hi = head.y0 - margin / z  # hair just under the seam
            y0 = max(lo, min(y0, hi)) if lo <= hi else lo
            y0 = float(np.clip(y0, 0, H - sh))
            x0 = float(np.clip(face.cx - sw / 2, 0, W - sw))
            return View(Rect(x0, y0, x0 + sw, y0 + sh), dst)
        if self.mode == "inset":
            dst = self.inset
            if face is None:
                return View(Rect(0, 0, W, W * dst.h / dst.w), dst, self.radius)
            sh = min(H, face.h / 0.40)
            sw = sh * dst.w / dst.h
            if sw > W:
                sw, sh = W, W * dst.h / dst.w
            x0 = float(np.clip(face.cx - sw / 2, 0, W - sw))
            y0 = float(np.clip(face.cy - 0.47 * sh, 0, H - sh))
            return View(Rect(x0, y0, x0 + sw, y0 + sh), dst, self.radius)
        return View(self.full, self.full)

    def view_at(self, target: View, e: float) -> View:
        """Your window part of the way (e from 0 to 1) from the whole frame to `target`. The window
        keeps the shape it is drawn into, so you are never squashed on the way."""
        if e >= 0.999:
            return target
        dst = _lerp_rect(self.full, target.dst, e)
        src = _lerp_rect(self.full, target.src, e)
        h = src.w * dst.h / max(1e-6, dst.w)
        cy = float(np.clip(src.cy, h / 2, self.H - h / 2))
        src = Rect(src.x0, cy - h / 2, src.x1, cy + h / 2)
        return View(src, dst, target.radius * e)


def paint_view(frame: np.ndarray, speaker: np.ndarray, v: View, border: float = 0.0, shadow: float = 0.0) -> None:
    """Draw `v.src` of your frame into `v.dst` of `frame`, with round corners, a light border and a
    soft shadow when asked."""
    import cv2

    H, W = frame.shape[:2]
    sx0, sy0 = int(round(max(0, v.src.x0))), int(round(max(0, v.src.y0)))
    sx1, sy1 = int(round(min(W, v.src.x1))), int(round(min(H, v.src.y1)))
    dx0, dy0 = int(round(v.dst.x0)), int(round(v.dst.y0))
    dx1, dy1 = int(round(v.dst.x1)), int(round(v.dst.y1))
    dw, dh = dx1 - dx0, dy1 - dy0
    if dw < 2 or dh < 2 or sx1 - sx0 < 2 or sy1 - sy0 < 2:
        return
    crop = speaker[sy0:sy1, sx0:sx1]
    interp = cv2.INTER_AREA if crop.shape[1] > dw else cv2.INTER_LINEAR
    img = cv2.resize(crop, (dw, dh), interpolation=interp)
    if v.radius < 0.5 and border <= 0 and shadow <= 0:
        frame[dy0:dy1, dx0:dx1] = img
        return
    if shadow > 0:
        pad = int(0.05 * W)
        a = np.zeros((dh + 2 * pad, dw + 2 * pad), np.float32)
        a[pad:pad + dh, pad:pad + dw] = rounded_alpha(dw, dh, v.radius)
        small = cv2.resize(a, (max(1, a.shape[1] // 4), max(1, a.shape[0] // 4)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 0.018 * W / 4)
        sh = cv2.resize(small, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR) * (0.38 * shadow)
        off = int(round(0.006 * H))
        y0, x0 = dy0 - pad + off, dx0 - pad
        ys0, xs0 = max(0, -y0), max(0, -x0)
        ye, xe = min(sh.shape[0], H - y0), min(sh.shape[1], W - x0)
        if ye > ys0 and xe > xs0:
            reg = frame[y0 + ys0:y0 + ye, x0 + xs0:x0 + xe].astype(np.float32)
            reg *= 1.0 - sh[ys0:ye, xs0:xe, None]
            frame[y0 + ys0:y0 + ye, x0 + xs0:x0 + xe] = np.clip(reg + 0.5, 0, 255).astype(np.uint8)
    a = rounded_alpha(dw, dh, v.radius)
    reg = frame[dy0:dy1, dx0:dx1].astype(np.float32)
    out = reg * (1.0 - a[..., None]) + img.astype(np.float32) * a[..., None]
    if border > 0:
        b = max(1, int(round(0.0035 * W)))
        inner = np.zeros_like(a)
        if dw > 2 * b and dh > 2 * b:
            inner[b:dh - b, b:dw - b] = rounded_alpha(dw - 2 * b, dh - 2 * b, max(0.0, v.radius - b))
        ring = np.clip(a - inner, 0, 1) * (0.9 * border)
        out = out * (1.0 - ring[..., None]) + 255.0 * ring[..., None]
    frame[dy0:dy1, dx0:dx1] = np.clip(out + 0.5, 0, 255).astype(np.uint8)


# ---- the painter: every frame ------------------------------------------------------------------------

class Painter:
    def __init__(self, layout: Layout, transition: str, cuts: list[Cut], pool: list[Broll], fps: float,
                 views: list[View], tonemap: list[bool]):
        self.layout, self.transition, self.cuts, self.pool, self.fps = layout, transition, cuts, pool, fps
        self.views, self.tonemap = views, tonemap
        self.reels: dict[int, Reel] = {}
        self.captions = None  # drawn over everything
        self.carried = None  # the clip's own captions, drawn again where a cutaway hides them

    def window(self, c: Cut) -> tuple[float, float]:
        a = 0.0 if c.after else (SLIDE / 2 if self.transition == "slide" else DIP if self.transition == "dip" else 0.0)
        b = 0.0 if c.chained else (SLIDE / 2 if self.transition == "slide" else DIP if self.transition == "dip" else 0.0)
        return c.start - a, c.end + b

    def state(self, t: float) -> tuple[Optional[int], float, float]:
        """Which cutaway is up at `t`, how far it has arrived (0 to 1), and how dark the dip is."""
        for k, c in enumerate(self.cuts):
            a, b = self.window(c)
            if not (a - 1e-9 <= t < b):
                continue
            if self.transition == "slide":
                e = 1.0
                if not c.after:
                    e = min(e, _smooth((t - (c.start - SLIDE / 2)) / SLIDE))
                if not c.chained:
                    e = min(e, 1.0 - _smooth((t - (c.end - SLIDE / 2)) / SLIDE))
                return k, e, 0.0
            e = 1.0 if c.start - 1e-9 <= t < c.end else 0.0
            dark = 0.0
            if self.transition == "dip":
                if not c.after and t < c.start + DIP:
                    dark = max(dark, 1.0 - abs(t - c.start) / DIP)
                if not c.chained and t >= c.end - DIP:
                    dark = max(dark, 1.0 - abs(t - c.end) / DIP)
            return k, e, max(0.0, dark)
        return None, 0.0, 0.0

    def reel(self, k: int) -> Reel:
        if k not in self.reels:
            for old in [i for i in self.reels if abs(i - k) > 1]:
                self.reels.pop(old).close()
            c = self.cuts[k]
            self.reels[k] = Reel(self.pool[c.clip], c, self.layout.broll_size, self.fps, self.tonemap[k])
        return self.reels[k]

    def view(self, k: int, e: float) -> View:
        return self.layout.view_at(self.views[k], e)

    def draw(self, frame: np.ndarray, j: int, t: float) -> None:
        k, e, dark = self.state(t)
        if k is not None and e > 0:
            L = self.layout
            broll = self.reel(k).frame(t)
            if L.mode == "full":
                if e >= 0.999:
                    frame[:] = broll
                else:  # slide: the B-roll pushes you out to the left, and pulls back the way it came
                    off = int(round(e * L.W))
                    if off > 0:
                        frame[:, :L.W - off] = frame[:, off:].copy()
                        frame[:, L.W - off:] = broll[:, :off]
            else:
                speaker = frame.copy()
                if L.mode == "split":
                    hb = int(round(L.seam * e))
                    frame[:] = 0
                    if hb > 0:
                        frame[:hb] = broll[L.seam - hb:L.seam]
                else:
                    frame[:] = broll
                v = self.view(k, e)
                paint_view(frame, speaker, v, border=e if L.mode == "inset" else 0.0,
                           shadow=e * e if L.mode == "inset" else 0.0)
                if L.mode == "split" and e > 0:
                    self._seam(frame, int(round(v.dst.y0)), e)
        if dark > 0:
            frame[:] = (frame.astype(np.float32) * (1.0 - dark) + 0.5).astype(np.uint8)
        if self.carried is not None and (e > 0 or dark > 0):
            self.carried.draw(frame, j, t)
        if self.captions is not None:
            self.captions.draw(frame, j, t)

    def _seam(self, frame: np.ndarray, y: int, e: float) -> None:
        """A soft shadow where the B-roll meets your panel, so the two read as layers."""
        H = frame.shape[0]
        n = max(2, int(round(0.012 * H)))
        y1 = min(H, y + n)
        if y1 <= y:
            return
        ramp = (1.0 - np.linspace(0, 1, y1 - y, dtype=np.float32)) ** 2 * (0.30 * e)
        reg = frame[y:y1].astype(np.float32) * (1.0 - ramp[:, None, None])
        frame[y:y1] = np.clip(reg + 0.5, 0, 255).astype(np.uint8)

    def close(self) -> None:
        for r in self.reels.values():
            r.close()
        self.reels.clear()


# ---- captions over it all ----------------------------------------------------------------------------

class _Captions(cap.Captions):
    """The pack's captions, with no card on screen across a change of layout: a card ends where the
    layout changes, so each one is placed for one layout."""

    def __init__(self, *args, breaks: tuple = (), **kwargs):
        self._breaks = tuple(sorted(breaks))
        super().__init__(*args, **kwargs)

    def _time(self) -> None:
        super()._time()
        for c in self.cards:
            for b in self._breaks:
                if c.start < b - 1e-6 < c.end:
                    c.end = max(c.start + 1.0 / self.fps, b)
                    break


def caption_cards(words: list[Word], quiet, most: int, breaks: list[float]) -> list[dict]:
    """The phrases the captions show, each cut where the layout changes, each with the captions' own
    key word (never one the speaker took back, read from `words`, the clip's whole words)."""
    back = moments.taken_back(words, quiet)
    ws = [w for w in words if w.key not in HESITATIONS and not getattr(w, "unsure", "")]
    out = []
    for ph in cap.phrases(ws, quiet, most):
        piece: list[Word] = []
        for i in ph:
            w = ws[i]
            if piece and any(piece[-1].start < b - 1e-6 <= w.start + 1e-6 for b in breaks):
                out.append(piece)
                piece = []
            piece.append(w)
        if piece:
            out.append(piece)
    return [{"words": [{"w": w.text, "s": w.start, "e": w.end} for w in p], "key": cap.key_word(p, quiet, back)} for p in out]


def carried_captions(clip, size: tuple[int, int], fps: float, n_out: int) -> tuple[Optional[cap.Captions], str]:
    """The pack's own captions burned into this clip, read from the plan beside it, to draw again over
    full-frame cutaways in the same places."""
    plan_path = Path(clip.path).with_suffix(".plan.json")
    if not plan_path.exists():
        return None, ""
    try:
        plan = json.loads(plan_path.read_text())
    except (OSError, ValueError):
        return None, ""
    cards = plan.get("cards") or []
    out = plan.get("output") or {}
    if not cards or (out.get("width"), out.get("height")) != size:
        return None, ""
    opts = plan.get("options") or {}
    look = opts.get("look") if opts.get("look") in cap.LOOKS else "stacked"
    try:
        caps = cap.Captions([], [], size, fps, n_out, lambda j: None, look=look, color=opts.get("color", "auto"),
                            cards=cards)
    except (KeyError, TypeError, ValueError):
        return None, ""
    for c, given in zip(caps.cards, cards):
        r = given.get("rect")
        if not r:
            continue
        x0, y0, x1, _ = (float(v) for v in r)
        for _ in range(8):  # a card the clip's captions shrank to fit is shrunk the same way
            if c.w <= (x1 - x0) + 2:
                break
            caps._layout(c, c.scale * 0.88)
        c.x, c.y = x0, y0
    return caps, f"the clip's own captions ({plan.get('effect')}, {look}), drawn again over each full-frame cutaway"


# ---- the B-roll's own sound --------------------------------------------------------------------------

def _rms_db(x: np.ndarray, rate: int, pct: float = 75) -> float:
    if x.size == 0:
        return -120.0
    m = x.mean(axis=1) if x.ndim == 2 else x
    n = max(1, int(0.01 * rate))
    k = len(m) // n
    if k == 0:
        return -120.0
    frames = m[: k * n].reshape(k, n)
    lv = 10 * np.log10((frames ** 2).mean(axis=1) + 1e-12)
    return float(np.percentile(lv, pct))


def _read_sound(path: str, start: float, length: float, rate: int) -> np.ndarray:
    run = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{max(0.0, start):.4f}", "-t", f"{length:.4f}",
                          "-i", path, "-map", "0:a:0", "-ac", "2", "-ar", str(rate), "-f", "f32le", "-"],
                         capture_output=True)
    if run.returncode != 0:
        return np.zeros((0, 2), np.float32)
    return np.frombuffer(run.stdout, np.float32).reshape(-1, 2).copy()


def ducked(cuts: list[Cut], pool: list[Broll], report: list[str]):
    """Lays each clip's own sound under the voice (`DUCK_UNDER` dB under it, never louder than it was
    filmed), from a moment before its picture to a moment after."""

    def mix(samples: np.ndarray, rate: int) -> np.ndarray:
        out = np.array(samples, np.float32, copy=True)
        added = np.zeros_like(out)
        n = len(out)
        for c in cuts:
            b = pool[c.clip]
            if b.kind != "video" or not b.has_audio:
                continue
            lead = 0.04 if c.after else J_LEAD
            trail = 0.04 if c.chained else L_TRAIL
            a0, a1 = max(0.0, c.start - lead), min(n / rate, c.end + trail)
            seg = _read_sound(b.path, c.in_point - (c.start - a0), a1 - a0, rate)
            i0 = int(round(a0 * rate))
            m = min(len(seg), n - i0)
            if m <= 0:
                continue
            seg = seg[:m]
            voice = _rms_db(samples[i0:i0 + m], rate)
            own = _rms_db(seg, rate)
            if own < -70:
                continue
            gain_db = float(np.clip(voice - DUCK_UNDER - own, -40.0, 0.0))  # lowered, never raised
            env = np.ones(m, np.float32)
            fi, fo = max(1, int(lead * rate)), max(1, int(trail * rate))
            env[:fi] = np.linspace(0, 1, fi, dtype=np.float32) if fi <= m else env[:fi]
            env[m - min(fo, m):] *= np.linspace(1, 0, min(fo, m), dtype=np.float32)
            added[i0:i0 + m] += seg * (10 ** (gain_db / 20)) * env[:, None]
            report.append(f"{b.label} {gain_db:+.0f} dB")
        peak_sum = np.abs(out + added).max() if n else 0.0
        if peak_sum > 0.98 and np.abs(added).max() > 0:
            room = np.clip(0.98 - np.abs(out), 0, None)
            k = float(np.clip(np.min(np.where(np.abs(added) > 1e-6, room / np.maximum(np.abs(added), 1e-6), 1.0)),
                              0.0, 1.0))
            added *= k
        return out + added

    return mix


# ---- the effect --------------------------------------------------------------------------------------

def _face_rects(clip, fr, timeline, n_out: int):
    """Your face's box (from brows to chin) and your head's (with hair) at each output frame, in
    output pixels, smoothed; None where no face is known."""
    face = clip.face
    fps, n_src = clip.info.rate, clip.info.frames

    def at(j: int) -> tuple[Optional[Rect], Optional[Rect]]:
        k = timeline.source_frame(j, fps, n_src)
        b = face.box(k)
        h = face.head(k)
        return (fr.to_output(b, k) if b is not None else None, fr.to_output(h, k) if h is not None else None)

    return at


def _median_face(face_at, j0: int, j1: int, which: int = 0) -> Optional[Rect]:
    """Your face's box (which=0) or your head's (1) over output frames j0..j1: where it mostly is,
    and as big as it gets."""
    boxes = [b for j in range(j0, j1 + 1, 2) if (b := face_at(j)[which]) is not None]
    if not boxes:
        return None
    arr = np.array([[b.cx, b.cy, b.w, b.h] for b in boxes])
    cx, cy = np.median(arr[:, 0]), np.median(arr[:, 1])
    w, h = np.percentile(arr[:, 2], 90), np.percentile(arr[:, 3], 90)
    return Rect(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _map_rect(r: Rect, v: View) -> Rect:
    sx, sy = v.dst.w / max(1e-6, v.src.w), v.dst.h / max(1e-6, v.src.h)
    return Rect(v.dst.x0 + (r.x0 - v.src.x0) * sx, v.dst.y0 + (r.y0 - v.src.y0) * sy,
                v.dst.x0 + (r.x1 - v.src.x0) * sx, v.dst.y0 + (r.y1 - v.src.y0) * sy)


def _cuts_from_plan(plan: dict, seconds: float) -> list[dict]:
    from . import PlanError

    items = plan.get("cutaways")
    if not isinstance(items, list):
        raise PlanError("the plan has no cutaways; run once without --plan to get one to edit")
    out = []
    for it in items:
        try:
            s, e, p = float(it["start"]), float(it["end"]), str(it["broll"])
        except (KeyError, TypeError, ValueError):
            raise PlanError("each cutaway in the plan needs start, end and broll") from None
        if not (0 <= s < e <= seconds + 1e-3):
            raise PlanError(f"the cutaway {s:.2f}-{e:.2f} s is not inside the clip (0-{seconds:.2f} s)")
        if not Path(p).exists():
            raise PlanError(f"the plan's B-roll {p} is not there any more")
        out.append(dict(it, start=s, end=e, broll=p))
    out.sort(key=lambda it: it["start"])
    for a, b in zip(out, out[1:]):
        if b["start"] < a["end"] - 1e-6:
            raise PlanError(f"two cutaways overlap at {b['start']:.2f} s; move one")
    return out


def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from ..core import framing as framing_mod
    from ..core.timeline import Timeline
    from . import Job, PlanError

    info = clip.info
    fps = info.rate
    if plan is not None:
        kept = plan.get("options") or {}
        opts = dict(opts, **{k: kept[k] for k in ("mode", "transition", "broll_audio", "captions", "frame")
                             if k in kept})
    fr = framing_mod.plan(info.width, info.height, fps, info.frames, clip.face, shape=opts["frame"])
    tl = Timeline.whole(info.frames / fps)
    W, H = fr.out_w, fr.out_h
    seconds = tl.duration
    n_out = tl.output_frames(fps)
    words = clip.words
    warnings: list[str] = []
    if opts["mode"] == "split" and W > H:
        raise ValueError("split puts the B-roll above you, which needs a tall frame; add frame=9:16 to make "
                         "this video vertical first, or use mode=full or mode=inset")
    layout = Layout(opts["mode"], (W, H))

    # what goes where, and when
    if plan is not None:
        items = _cuts_from_plan(plan, seconds)
        pool, cuts, index = [], [], {}
        for it in items:
            if it["broll"] not in index:
                index[it["broll"]] = len(pool)
                pool.append(load_broll(Path(it["broll"])))
            cuts.append(Cut(it["start"], it["end"], index[it["broll"]], str(it.get("word", "")),
                            str(it.get("why", "plan")), 3, in_point=float(it.get("in", HEAD_SKIP)),
                            crop=tuple(float(x) for x in it.get("crop", (0.5, 0.5))), push=float(it.get("push", 0.0)),
                            fit=str(it.get("fit", "as the plan says"))))
        for a, b in zip(cuts, cuts[1:]):
            if abs(a.end - b.start) < 1e-3:
                a.chained, b.after = True, True
        left, notes = {}, []
        # a return given inside a word moves to the nearest place between words
        bounds = [t for t, _ in boundaries(words, clip.quiet, clip.envelope, seconds)]
        for c in cuts:
            w = None if c.chained else inside_word(c.end, words)
            if w is not None and bounds:
                new = min(bounds, key=lambda t: abs(t - c.end))
                if c.start + 0.5 <= new <= seconds:
                    warnings.append(f"the return at {c.end:.2f} s fell inside “{cap.bare(w)}”; moved to {new:.2f} s")
                    c.end = new
        how = "the plan's cutaways"
    else:
        named = {}
        mapping = parse_map(opts["map"])
        pool, named = load_pool(opts["broll"], mapping)
        cuts, left, notes = decide(words, clip.quiet, clip.envelope, pool, named, opts["on"], opts["duration"],
                                   float(opts["max_share"]), seconds)
        cuts = [frame_cut(pool[c.clip], c, layout.broll_size, c.end - c.start) for c in cuts]
        how = {"keywords": "names, tags and the things you name", "map": "map, names and tags"}.get(
            opts["on"], "the times given")
    if not cuts:
        warnings.append("no cutaway was placed: " + ("; ".join(f"{pool[i].label}: {why}" for i, why in left.items())
                                                     if left else "nothing matched"))

    # HDR B-roll is brought to normal range, as the core reads the talking clip (HDR or not): the
    # film is written as SDR, so an HDR clip left as it is would look flat and grey beside you
    tonemap = [bool(pool[c.clip].transfer) for c in cuts]

    # where you go while the B-roll is up
    face_at = _face_rects(clip, fr, tl, n_out)
    cap_choice = opts["captions"]
    look = "stacked" if cap_choice == "on" else cap_choice
    if layout.mode == "inset":
        # Up or down: away from where your captions sit over you (above your head, or below it).
        level = "top"
        if look in cap.LOOKS:
            trial = cap.Captions(words, clip.quiet, (W, H), fps, n_out, lambda j: face_at(j)[1], look=look)
            if sum(c.where == "top" for c in trial.cards) > len(trial.cards) / 2:
                level = "bottom"
        # Left or right: the side of the B-roll with less going on, so the window hides the least.
        left_mass = right_mass = 0.0
        for c in cuts:
            imap = interest(pool[c.clip], c.in_point, c.end - c.start)
            h, w = imap.shape
            band = imap[: max(1, int(h * 0.45))] if level == "top" else imap[int(h * 0.4): int(h * 0.85)]
            left_mass += float(band[:, : w // 2].sum())
            right_mass += float(band[:, w - w // 2:].sum())
        layout.place_inset("left" if left_mass <= right_mass else "right", level)
    spans = [(int(math.floor(c.start * fps)), min(n_out - 1, int(math.ceil(c.end * fps)))) for c in cuts]
    if layout.mode == "split":
        layout.fit_seam([h for a, b in spans if (h := _median_face(face_at, a, b, 1)) is not None],
                        [f for a, b in spans if (f := _median_face(face_at, a, b, 0)) is not None])
    views = [layout.target_view(_median_face(face_at, a, b, 0), _median_face(face_at, a, b, 1)) for a, b in spans]
    painter = Painter(layout, opts["transition"], cuts, pool, fps, views, tonemap)

    # captions on top
    caps, cap_note = None, ""
    if cap_choice not in ("auto", "off"):

        def head_at(j: int) -> Optional[Rect]:
            t = j / fps
            k, e, _ = painter.state(t)
            _, head = face_at(j)
            if k is None or e <= 0 or layout.mode == "full":
                # Over a full-frame cutaway the words keep the place they had over you, so they
                # never jump when the picture changes under them.
                return head
            v = painter.view(k, e)
            if layout.mode == "inset" and e >= 0.999:  # the whole window is kept clear, not just the face
                return v.dst.grow(0.02 * W, 0.012 * H, 0.02 * W, 0.012 * H)
            return _map_rect(head, v) if head is not None else None  # on the way: your face as it moves

        breaks = []
        if layout.mode != "full":
            for c in cuts:
                if not c.after:
                    breaks.append(c.start)
                if not c.chained:
                    breaks.append(c.end)
        cards = caption_cards(words, clip.quiet, cap.LOOK[look].most, breaks) if breaks else None
        caps = _Captions(words, clip.quiet, (W, H), fps, n_out, head_at, look=look, breaks=tuple(breaks),
                         cards=cards) if cards is not None else cap.Captions(words, clip.quiet, (W, H), fps, n_out,
                                                                            head_at, look=look)
        painter.captions = caps
        cap_note = f"{look}, {len(caps.cards)} phrases over everything, carried across every cutaway"
    elif cap_choice == "auto":
        carried, why = carried_captions(clip, (W, H), fps, n_out)
        if carried is not None:
            if layout.mode == "full":
                painter.carried = carried
                cap_note = why
            else:
                raise PlanError("this clip already has its captions burned in, and a split or an inset would cut "
                                "through them. Run broll-cutaways on the clip before captions with captions=on, or "
                                "use mode=full, or say captions=off to go ahead anyway")
        else:
            cap_note = "none added (captions=on adds them over the cutaways)"
    else:
        cap_note = "off"

    # the B-roll's own sound
    sound_report: list[str] = []
    mix = ducked(cuts, pool, sound_report) if opts["broll_audio"] == "duck" else None

    # checks
    sheet_path = broll_sheet(pool, cuts, left)
    total = Timing.total(cuts)
    returns = [c for c in cuts if not c.chained and c.end < seconds - 1e-3]
    mid = [c for c in returns if inside_word(c.end, words) is not None]
    off_word = [c for c in cuts if c.why not in ("time", "spread", "plan") and not any(
        abs(w.start - c.start) < 0.5 / fps for w in words)]
    framing_lines = []
    for c in cuts:
        line = f"{pool[c.clip].label}: {c.fit}"
        if tonemap[cuts.index(c)]:
            line += ", HDR brought to normal range"
        if line not in framing_lines:
            framing_lines.append(line)
    face_line = ""
    if layout.mode in ("split", "inset") and cuts:
        bad = checked = 0
        for k, c in enumerate(cuts):
            v = views[k]
            band = Rect(v.dst.x0, v.dst.y0, v.dst.x1, min(v.dst.y1, FACE_FLOOR * H) if layout.mode == "split" else v.dst.y1)
            for j in range(int(math.ceil(c.start * fps)), min(n_out, int(math.floor(c.end * fps))), 3):
                box = face_at(j)[0]
                if box is None:
                    continue
                checked += 1
                if not _map_rect(box, v).inside(band, slack=2.0):
                    bad += 1
        face_line = (f"your face (brows to chin) stays whole in its {'panel above the app' if layout.mode == 'split' else 'window'}"
                     f" on {checked - bad} of {checked} frames checked")
    rep = caps.report() if caps is not None else {}
    outside = int(rep.get("frames_outside_safe_zone", 0))
    if layout.mode == "inset" and cuts and not SafeZone(W, H).contains(layout.inset):
        outside += sum(int(round((c.end - c.start) * fps)) for c in cuts)
    lands = ("each starts on the first frame of its word" if not off_word else
             f"{len(off_word)} start between words: " + ", ".join(f"{c.start:.2f} s" for c in off_word))
    if opts["on"] not in ("keywords", "map") and plan is None:
        lands = f"at the times given, each moved onto a word starting within {SNAP:g} s of it"
    checks = {
        "cutaways": (f"{len(cuts)} on {how} ({total:.1f} s of {seconds:.1f} s, {total / max(1e-6, seconds):.0%}; "
                     f"at most {float(opts['max_share']):.0%}): " if cuts else "none: ")
                    + "; ".join(f"{c.start:.2f}-{c.end:.2f} s “{c.word}” → {pool[c.clip].label} ({c.why}"
                                + (", straight into the next" if c.chained else "") + ")" for c in cuts),
        "lands": lands,
        "returns": f"{len(returns)} back to you, {len(mid)} inside a word"
                   + (": " + ", ".join(f"{c.end:.2f} s" for c in mid) if mid else ""),
        "framing": "; ".join(framing_lines) or "-",
        "not_used": "; ".join(f"{pool[i].label}: {why}" for i, why in sorted(left.items())) or "none",
        "sound": ("your voice as filmed; the B-roll's own sound off" if mix is None else
                  f"your voice as filmed; the B-roll's own sound {DUCK_UNDER:g} dB under it, in {J_LEAD:g} s "
                  f"before each picture and out {L_TRAIL:g} s after"),
        "captions": cap_note,
        "frames_over_face": int(rep.get("frames_over_face", 0)),
        "frames_outside_safe_zone": outside,
        "broll_sheet": (f"{sheet_path} (one frame of each clip and where it landed: look, then name or map "
                        "clips whose picture should match a word)" if sheet_path else "-"),
        "placed": {"full": "the B-roll fills the frame while it is up",
                   "split": f"the B-roll above ({layout.seam / H:.0%} of the height), you below",
                   "inset": f"the B-roll fills the frame; you in a window at the {layout.corner}"}[layout.mode]
                  + (f"; {face_line}" if face_line else ""),
        "warnings": warnings + notes + list(rep.get("warnings", [])),
    }

    # the plan
    def sheet_times() -> list[float]:
        ts = []
        if cuts:
            ts.append(max(0.0, min(cuts[0].start - 0.6, cuts[0].start / 2 + 0.4)))
        for c in cuts[:4]:
            ts.append(c.start + 0.55 * (c.end - c.start))
        if cuts:
            back = next((c for c in cuts if not c.chained), cuts[0])
            ts.append(min(seconds - 1.0 / fps, back.end + 0.35))
        while len(ts) < 6:
            ts.append(seconds * (len(ts) + 0.5) / 6)
        return sorted(round(min(seconds - 1.0 / fps, max(0.0, t)), 3) for t in ts)[:6]

    out_plan = {
        "effect": NAME, "clip": clip.path, "options": opts,
        "output": {"width": W, "height": H, "fps": round(fps, 3), "seconds": round(seconds, 3), "framing": fr.note},
        "mode": layout.mode, "transition": opts["transition"],
        "cutaways": [{"start": round(c.start, 3), "end": round(c.end, 3), "broll": pool[c.clip].path,
                      "in": round(c.in_point, 3), "word": c.word, "why": c.why,
                      "crop": [round(c.crop[0], 4), round(c.crop[1], 4)], "push": c.push, "fit": c.fit,
                      "chained": c.chained} for c in cuts],
        "not_used": [{"broll": pool[i].path, "why": why} for i, why in sorted(left.items())],
        "inset": layout.inset.as_list() if layout.mode == "inset" else None,
        "seam": layout.seam if layout.mode == "split" else None,
        "captions": cap_note,
        "cards": caps.plan() if caps is not None else [],
        "sheet_times": sheet_times(),
    }

    def still_times() -> list[float]:
        if not cuts:
            return [seconds / 3, 2 * seconds / 3]
        ts = [max(0.0, cuts[0].start - 0.5), cuts[0].start + 0.5 * (cuts[0].end - cuts[0].start)]
        if len(cuts) > 1:
            ts.append(cuts[1].start + 0.5 * (cuts[1].end - cuts[1].start))
        back = next((c for c in cuts if not c.chained), cuts[0])
        ts.append(min(seconds - 1.0 / fps, back.end + 0.3))
        return sorted(round(t, 3) for t in ts)

    def after(result: dict, out_path) -> dict:
        painter.close()
        extra = {}
        if sound_report:
            extra["B-roll sound"] = ", ".join(sound_report)
        return extra

    return Job(NAME, clip, tl, fr, painter.draw, out_plan, checks=checks, still_times=still_times, after=after,
               sound=mix)
