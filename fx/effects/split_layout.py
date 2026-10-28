"""split-layout: you share the frame with your own B-roll, pictures, screen recordings or a line of
text, held through a section of the film and changing on its cuts or between its sentences.

The layouts:

- `panel`: you in a rounded panel at the bottom, what you show in a rounded panel above you;
- `split`: two halves edge to edge (you on top in a tall film, on the right in a wide one);
- `side`: you beside what you show, two rounded panels (made for wide films; in a tall one, two
  cards side by side);
- `pip`: what you show fills the frame, and you sit small in a corner over it;
- and between them, you alone filling the frame (`full`).

How it decides:

1. **Sections.** The film is cut where the layout may change: at the clip's own cuts (`cuts`,
   found by `core/shots.py`), between sentences (`words`: sections of about four seconds, never
   under two and a half), or at the seconds given. `full` says which sections are you alone; the
   rest show the content, each item in turn, an item held across sections when there are more
   sections than items.
2. **Your panel follows your face.** It shows a window of your picture that keeps your forehead
   and chin always and your hair when there is room, your face across its middle and a little
   above its centre, moving like a calm camera operator: still while you stay near, easing after
   you when you drift, jumping with the clip at its own cuts. The window never runs past the edge
   of your picture, so there are never bars. When your shot is too close for your whole head to fit
   a panel that wide, the panel is made taller until it does (your share grows), and the run says
   so. With you at the bottom, your face is kept above the bottom fifth where the app's caption
   sits, as far as the panel allows.
3. **What you show fills its panel well.** A clip fills it, cropped about its middle, unless that
   would cut away more than half of it (a wide clip in a tall panel): then it is shown whole over a
   blurred copy of itself. Black bars in it are taken off first. A picture is cropped the same way
   and moves slowly closer at one steady rate for as long as it is up. A screen recording or a
   screenshot (told apart by its flat colours, its name, or `kind`) is never cropped or moved: it is
   scaled once, sharply, and shown whole as a window on a card. A line of text is set on a card, its
   key word marked; with no content given, each section's card shows the strongest line said in it.
4. **Changes land on the moment.** A new section starts moving on its first frame (the cut, or the
   first word of the sentence) and settles in under half a second: `slide` pushes the content in
   from its edge as your panel moves to its place, `scale` grows and shrinks the panels into place,
   `cut` changes on the frame. A soft swish sits under each move (`sound`).
5. **Other effects of the pack sit inside it.** `over` lays effects over the whole frame, prepared on
   a view of the film as composed (core/view.py), so their words keep clear of your face where it
   now is; `inside` lays them inside the content panel, keeping to that panel's safe part; and
   `captions` puts the pack's captions on the seam between you and your content, inside your panel,
   or anywhere clear of your face.

It is never laid inside another effect (`MOVES_PERSON`): everything laid with it would still keep
clear of where you were.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..core import moments, video
from ..core import text as textmod
from ..core.compose import Sprite, over
from ..core.face import HEAD
from ..core.framing import Framing, output_size
from ..core.layout import Rect, SafeZone
from ..core.words import HESITATIONS, Word, pause_after
from . import layers
from . import options as O

NAME = "split-layout"
MOVES_PERSON = True  # see layers.py: other effects are laid inside it, never it inside them
LAYOUTS = ("panel", "split", "side", "pip")
CORNERS = ("auto", "top", "bottom", "left", "right", "top-left", "top-right", "bottom-left", "bottom-right")

OPTIONS = {
    "layout": O.Text("panel: you in a rounded panel at the bottom, your content above · split: two halves "
                     "edge to edge · side: you beside your content (for wide films) · pip: you small in a corner "
                     "over your content. Several, such as \"panel, split\", take turns, one a section"),
    "content": O.Text("what shares the frame with you: a folder, or files and lines of text separated by | "
                      "(\"walk.mp4 | view.jpg | text: Three steps\"), or a .json list of {path, from, to}; empty "
                      "shows the strongest line said in each section"),
    "switch": O.Text("when the layout changes: auto (the clip's own cuts, else between sentences) · cuts · words "
                     "(between sentences) · times (the from and to of each item) · or seconds such as 4.2, 9, 15"),
    "full": O.Choice(("first", "between", "ends", "none"),
                     "when you fill the frame alone: first (the opening section, then the layout holds) · between "
                     "(back to you alone between sections) · ends (the first and last sections) · none"),
    "ratio": O.Number(0, 0.8, "your share of the frame: 0 lets the layout decide (about half; a third for pip)"),
    "corner": O.Choice(CORNERS, "where you sit: auto (panel: bottom · split: top · side: right · pip: a lower corner "
                                "clear of the app's buttons), or top, bottom, left, right, or a corner for pip"),
    "transition": O.Choice(("slide", "scale", "cut"), "how a layout changes: slide · scale · cut"),
    "style": O.Choice(("dark", "light", "blur"), "the ground round the panels: dark · light · blur (your own "
                                                  "picture, blurred and dimmed)"),
    "border": O.Switch("on draws a thin line round each panel"),
    "captions": O.Choice(("off", "seam", "panel", "frame"),
                         "the pack's captions: off · seam (on the line between you and your content) · panel (inside "
                         "your panel) · frame (anywhere clear of your face)"),
    "inside": O.Text("effects of the pack drawn inside the content panel, separated by | "
                     "(\"chart values=2K,197K\")"),
    "over": O.Text("effects of the pack drawn over the whole frame, clear of your face where you now are "
                   "(\"headline\")"),
    "sound": O.Switch("on lays a soft swish under each change of layout; off keeps the clip's sound exactly"),
    "frame": O.Choice(("keep", "9:16", "16:9"), "keep the clip's own shape, or make the film 9:16 or 16:9"),
    "size": O.Size("auto (1080x1920 at most, never enlarged) or a size such as 1080x1920"),
    "fix": O.Text("misheard words as wrong=right, separated by ;  e.g. \"Jon=John; ask editor=AskEditor\""),
}

_IN = "{lab}/split-layout/inputs/"
MATRIX = [
    # the plain wall: you in the bottom panel under your B-roll, then a picture; captions on the seam
    ("D", "D", {"content": f"{_IN}broll-lake.mp4 | {_IN}photo-rooftops.jpg", "captions": "seam"}),
    # the close shot: the product's own screen recording above you, whole and sharp
    ("A", "A", {"content": f"{_IN}screen-editor.mp4"}),
    # room above the head: two halves, you on top, a wide clip below, growing into place
    ("B", "B", {"layout": "split", "content": f"{_IN}broll-lounge-wide.mp4", "transition": "scale"}),
    # the hard clip: you small in a corner over your B-roll, the light ground, a border
    ("C", "C", {"layout": "pip", "content": f"{_IN}broll-plane.mp4", "style": "light", "border": "on"}),
    # nothing given: the strongest line said in each section, on a card
    ("A-lines", "A", {}),
    # 44 s with ten joins: switching on its own cuts, back to you alone between, every layout in turn
    ("33", "33", {"content": f"{_IN}broll-lake.mp4 | {_IN}broll-lounge-wide.mp4 | {_IN}broll-plane.mp4 | "
                             f"{_IN}screen-editor.mp4", "switch": "cuts", "full": "between",
                  "layout": "panel, split, pip", "style": "blur"}),
    # another effect inside the content panel: a chart over the B-roll, as its values are said
    ("D-chart", "D", {"content": f"{_IN}broll-rooftops.mp4", "full": "none",
                      "inside": "chart values=\"2K, 15K, 197K\" label=Followers when=\"the lively city\""}),
    # a wide film: you beside the product's home screen
    ("D-side", "D", {"frame": "16:9", "layout": "side", "full": "none", "content": f"{_IN}screen-home.png"}),
]

TRANSITION_S = {"slide": 0.45, "scale": 0.40, "cut": 0.0}
MIN_SECTION = {"cuts": 1.6, "words": 2.4, "times": 0.6}  # seconds: no section is shorter
SCREEN_FLAT = 0.42  # a picture this flat (share of pixels with no grain at all) is a screen
FILL_KEEPS = 0.34  # a clip or picture fills its panel when that keeps at least this share of it, else it is shown whole
LOOP_DISSOLVE = 0.35  # seconds: a clip shorter than its time on screen starts again through a dissolve this long
SCREEN_NAMES = re.compile(r"screen|recording|capture|screenshot|record|demo", re.I)
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
PICTURE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _even(v: float) -> int:
    return max(2, int(round(v / 2)) * 2)


def _smooth(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * u * (u * (u * 6 - 15) + 10)


def _out(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return 1.0 - (1.0 - u) ** 3


def _lerp(a: float, b: float, e: float) -> float:
    return a + (b - a) * e


def _lerp_rect(a: Rect, b: Rect, e: float) -> Rect:
    return Rect(_lerp(a.x0, b.x0, e), _lerp(a.y0, b.y0, e), _lerp(a.x1, b.x1, e), _lerp(a.y1, b.y1, e))


def _scaled(r: Rect, k: float) -> Rect:
    return Rect(r.cx - r.w * k / 2, r.cy - r.h * k / 2, r.cx + r.w * k / 2, r.cy + r.h * k / 2)


def _inter(a: Rect, b: Rect) -> Optional[Rect]:
    x0, y0, x1, y1 = max(a.x0, b.x0), max(a.y0, b.y0), min(a.x1, b.x1), min(a.y1, b.y1)
    return Rect(x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


# ---- the look -----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Look:
    ground: tuple  # round the panels
    card: tuple  # a text card's ground
    ink: tuple  # a text card's words
    matte: tuple  # what a screen sits on
    mark: tuple  # the block behind a card's key word
    mark_ink: tuple
    shadow: float  # how dark a panel's shadow is
    edge: tuple  # the thin border: colour and opacity
    blur: bool = False


LOOKS = {
    "dark": Look(ground=(14, 14, 17), card=(243, 240, 233), ink=(20, 20, 24), matte=(31, 32, 38),
                 mark=(255, 211, 77), mark_ink=(20, 20, 24), shadow=0.6, edge=(255, 255, 255, 0.26)),
    "light": Look(ground=(238, 235, 228), card=(24, 24, 29), ink=(245, 242, 236), matte=(222, 218, 209),
                  mark=(255, 211, 77), mark_ink=(20, 20, 24), shadow=0.24, edge=(0, 0, 0, 0.14)),
    "blur": Look(ground=(16, 16, 20), card=(243, 240, 233), ink=(20, 20, 24), matte=(26, 27, 33),
                 mark=(255, 211, 77), mark_ink=(20, 20, 24), shadow=0.5, edge=(255, 255, 255, 0.32), blur=True),
}


# ---- where each layout puts things ----------------------------------------------------------------

@dataclass(frozen=True)
class Shape:
    layout: str
    speaker: Rect
    content: Optional[Rect]
    r_speaker: float
    r_content: float
    lifted: float  # 1: floating panels with a shadow and the border; 0: edge to edge
    seam: Optional[Rect] = None  # the line between you and your content, where seam captions sit
    share: float = 0.0
    note: str = ""


class Grid:
    """The panels of each layout on a W x H frame: one margin, the same as the gutter between panels,
    one corner radius, and your share of the frame."""

    def __init__(self, W: int, H: int, corner: str, ratio: float):
        self.W, self.H = W, H
        self.u = min(W, H) / 1080.0
        self.m = _even(30 * self.u)
        self.r = 34 * self.u
        self.tall = H >= W
        self.corner, self.ratio = corner, ratio
        self.notes: list[str] = []

    def share(self, layout: str) -> float:
        if self.ratio > 0:
            return self.ratio
        return {"panel": 0.5, "split": 0.5, "side": 0.5 if self.tall else 0.4,
                "pip": 0.34 if self.tall else 0.24}.get(layout, 0.5)

    def where(self, layout: str) -> str:
        c = self.corner
        if layout in ("panel", "split") and self.tall:
            default = "bottom" if layout == "panel" else "top"
            return c if c in ("top", "bottom") else default
        if layout in ("panel", "split", "side"):
            return c if c in ("left", "right") else "right"
        # pip
        v_default, h_default = "bottom", ("left" if self.tall else "right")
        if c in ("top-left", "top-right", "bottom-left", "bottom-right"):
            return c
        if c in ("top", "bottom"):
            return f"{c}-{h_default}"
        if c in ("left", "right"):
            return f"{v_default}-{c}"
        return f"{v_default}-{h_default}"

    def shape(self, layout: str, share: Optional[float] = None) -> Shape:
        W, H, m, r = self.W, self.H, self.m, self.r
        s = self.share(layout) if share is None else share
        if layout == "full":
            return Shape("full", Rect(0, 0, W, H), None, 0.0, 0.0, 0.0)
        where = self.where(layout)
        if layout == "split":
            if self.tall:
                cut = _even(H * s) if where == "top" else _even(H * (1 - s))
                a, b = Rect(0, 0, W, cut), Rect(0, cut, W, H)
                sp, ct = (a, b) if where == "top" else (b, a)
                seam = Rect(0, cut - 1, W, cut + 1)
            else:
                cut = _even(W * (1 - s)) if where == "right" else _even(W * s)
                a, b = Rect(0, 0, cut, H), Rect(cut, 0, W, H)
                sp, ct = (b, a) if where == "right" else (a, b)
                seam = Rect(cut - 1, 0, cut + 1, H)
            return Shape("split", sp, ct, 0.0, 0.0, 0.0, seam, s)
        if layout == "pip":
            ws = _even(W * s)
            hs = _even(ws * 1.25)
            v, h = where.split("-")
            x0 = m if h == "left" else W - m - ws
            if self.tall:
                y0 = _even(0.10 * H) if v == "top" else _even(0.78 * H) - hs
            else:
                y0 = m if v == "top" else H - m - hs
            return Shape("pip", Rect(x0, y0, x0 + ws, y0 + hs), Rect(0, 0, W, H), 28 * self.u, 0.0, 1.0, None, s)
        if layout == "panel" and self.tall:  # one above the other
            avail = H - 3 * m
            hs = _even(avail * s)
            hc = avail - hs
            if where == "bottom":
                ct, sp = Rect(m, m, W - m, m + hc), Rect(m, H - m - hs, W - m, H - m)
                seam = Rect(m, m + hc, W - m, m + hc + m)
            else:
                sp, ct = Rect(m, m, W - m, m + hs), Rect(m, 2 * m + hs, W - m, H - m)
                seam = Rect(m, m + hs, W - m, m + hs + m)
            return Shape("panel", sp, ct, r, r, 1.0, seam, s)
        # side by side: `side` anywhere, and `panel` on a wide frame
        avail = W - 3 * m
        ws = _even(avail * s)
        wc = avail - ws
        if self.tall:
            hcol = _even(min(H - 2 * m, max(ws, wc) / 0.62))
            y0 = int(min(max(m, 0.45 * H - hcol / 2), H - m - hcol))
        else:
            hcol, y0 = H - 2 * m, m
        if where == "right":
            ct, sp = Rect(m, y0, m + wc, y0 + hcol), Rect(2 * m + wc, y0, W - m, y0 + hcol)
            seam = Rect(m + wc, y0, 2 * m + wc, y0 + hcol)
        else:
            sp, ct = Rect(m, y0, m + ws, y0 + hcol), Rect(2 * m + ws, y0, W - m, y0 + hcol)
            seam = Rect(m + ws, y0, 2 * m + ws, y0 + hcol)
        return Shape(layout, sp, ct, r, r, 1.0, seam, s)

    def adapt(self, sh: Shape, aspect: float, pad: float, allowance: float) -> Shape:
        """`sh` with its content panel shaped for a picture of `aspect` (width over height) shown whole:
        as tall as the picture needs at the panel's width (as wide, in a wide film), with `pad` round it
        and `allowance` for the part of the panel the app covers. Your share takes the rest: in a tall
        film never less than your head needs (`sh.share`) and at most two thirds."""
        if sh.content is None or sh.layout == "pip":
            return sh
        m = self.m
        if self.tall and sh.layout in ("panel", "split"):
            need = sh.content.w / aspect + 2 * pad + allowance
            if sh.layout == "panel":
                avail = self.H - 3 * m
                s = (avail - need) / avail
            else:
                s = (self.H - need) / self.H
            lo, hi = sh.share, 0.66
        elif not self.tall:
            need = sh.content.h * aspect + 2 * pad
            if sh.layout == "split":
                s = (self.W - need) / self.W
            else:
                s = (self.W - 3 * m - need) / (self.W - 3 * m)
            lo, hi = 0.28, max(0.6, sh.share)
        else:
            return sh
        s = float(min(hi, max(lo, s)))
        if abs(s - sh.share) < 0.005:
            return sh
        new = self.shape(sh.layout, s)
        return Shape(new.layout, new.speaker, new.content, new.r_speaker, new.r_content, new.lifted, new.seam, s,
                     sh.note)

    def fitted(self, layout: str, verdict: Callable[[Shape], str]) -> Shape:
        """The layout with your share grown (by a point at a time, at most to 0.64) until `verdict`
        says your shot works in it ("good"); otherwise the one that came closest, with a note."""
        sh = self.shape(layout)
        if layout not in ("panel", "split") or not self.tall:
            return sh
        why = verdict(sh)
        if why == "good" or self.ratio > 0:
            return sh if why == "good" else Shape(sh.layout, sh.speaker, sh.content, sh.r_speaker, sh.r_content,
                                                  sh.lifted, sh.seam, sh.share, FIT_NOTES[why] + " (ratio is yours)")
        best, s = sh, sh.share
        while s < 0.64 - 1e-9:
            s = min(0.64, s + 0.01)
            best = self.shape(layout, s)
            why = verdict(best)
            if why == "good":
                break
        note = f"your panel takes {best.share:.0%} of the frame rather than {sh.share:.0%}, so " + FIT_REASONS[layout]
        if why != "good":
            note += "; even so " + FIT_NOTES[why]
        return Shape(best.layout, best.speaker, best.content, best.r_speaker, best.r_content, best.lifted,
                     best.seam, best.share, note)


FIT_REASONS = {"panel": "your face sits above the bottom fifth with your whole head in",
               "split": "your whole head fits in your half"}
FIT_NOTES = {"head": "your shot is too close for your forehead and chin both to fit",
             "low": "your face reaches into the bottom fifth, where the app's caption sits (corner=top or "
                    "layout=split keeps it higher)",
             "hair": "the top of your hair may touch the panel's edge"}


# ---- where the layout changes ---------------------------------------------------------------------

@dataclass
class Item:
    kind: str  # video · picture · screen · text · line (a line of what is said)
    path: str = ""
    text: str = ""
    at: Optional[float] = None  # where in a clip it starts
    fit: str = "auto"  # auto · fill · whole
    layout: str = ""
    start: Optional[float] = None  # given times (output seconds)
    end: Optional[float] = None
    span: Optional[tuple[int, int]] = None  # a spoken line: its first and last word
    note: str = ""

    def label(self) -> str:
        if self.kind in ("text", "line"):
            t = self.text.replace("*", "")
            t = t if len(t) <= 40 else t[:38] + "…"
            return f"“{t}”"
        return Path(self.path).name


@dataclass
class Section:
    start: float
    end: float
    layout: str  # full, or one of LAYOUTS
    item: Optional[int] = None
    local: float = 0.0  # seconds of the item already shown before this section (a clip carries on)
    share: float = 0.0  # your share of the frame in this section (0: the layout's own)

    def to_plan(self) -> dict:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "layout": self.layout,
                "item": self.item, "local": round(self.local, 3), "share": round(self.share, 3)}


def sentence_breaks(words: list[Word], quiet, duration: float) -> tuple[list[float], list[float]]:
    """Where a new sentence starts (and, as second choices, a new clause after a real pause): the
    first word's start, a hair early so the move begins with the voice."""
    strong, weak = [], []
    for i in range(len(words) - 1):
        w, nxt = words[i], words[i + 1]
        if w.key in HESITATIONS:
            continue
        t = max(w.end, nxt.start - 0.04)
        if not 0.8 <= t <= duration - 0.8:
            continue
        if w.ends_sentence:
            strong.append(t)
        elif w.ends_clause and pause_after(words, i, quiet) >= 0.2:
            weak.append(t)
    return strong, weak


def choose_breaks(strong: list[float], weak: list[float], duration: float, least: float,
                  want: float = 4.2) -> list[float]:
    """The breaks that make sections closest to `want` seconds, none shorter than `least`, a sentence
    break preferred to a clause break."""
    cands = sorted({(round(t, 3), 0.0) for t in strong} | {(round(t, 3), 1.2) for t in weak if
                                                            all(abs(t - s) > 0.3 for s in strong)})
    pts = [(0.0, 0.0)] + cands + [(duration, 0.0)]
    n = len(pts)
    best = [math.inf] * n
    back = [-1] * n
    best[0] = 0.0
    for j in range(1, n):
        for i in range(j):
            length = pts[j][0] - pts[i][0]
            if length < least - 1e-6 or best[i] == math.inf:
                continue
            cost = best[i] + ((length - want) / 2.0) ** 2 + pts[j][1]
            if cost < best[j]:
                best[j], back[j] = cost, i
    if best[-1] == math.inf:
        return []
    out, j = [], n - 1
    while j > 0:
        j = back[j]
        if j > 0:
            out.append(pts[j][0])
    return sorted(out)


def merge_short(times: list[float], duration: float, least: float) -> list[float]:
    """Switch points with any section shorter than `least` joined to the one before it."""
    out: list[float] = []
    last = 0.0
    for t in sorted(times):
        if t - last >= least and duration - t >= least * 0.75:
            out.append(t)
            last = t
    return out


def parse_seconds(text: str) -> Optional[list[float]]:
    parts = [p.strip().rstrip("s") for p in re.split(r"[,;\s]+", text.strip()) if p.strip()]
    try:
        return sorted(float(p) for p in parts)
    except ValueError:
        return None


# ---- what shares the frame -------------------------------------------------------------------------

def _natural(name: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _flatness(frame: np.ndarray) -> float:
    """The share of the picture with no grain at all: large for a screen (flat colours drawn by a
    computer), small for anything a camera filmed, even a clear sky."""
    import cv2

    h, w = frame.shape[:2]
    s = 480 / max(h, w)
    g = cv2.cvtColor(cv2.resize(frame, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = cv2.blur(g, (5, 5))
    sq = cv2.blur(g * g, (5, 5))
    return float((np.sqrt(np.maximum(sq - mean * mean, 0)) < 1.0).mean())


def _bars(frames: list[np.ndarray]) -> tuple[int, int, int, int]:
    """The picture inside any black bars (x, y, w, h): rows and columns at the edges that stay
    near black in every frame looked at."""
    g = np.max(np.stack([f.astype(np.int16).max(axis=2) for f in frames]), axis=0)
    h, w = g.shape
    rows = np.flatnonzero(g.max(axis=1) > 24)
    cols = np.flatnonzero(g.max(axis=0) > 24)
    if not len(rows) or not len(cols):
        return 0, 0, w, h
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    # only real bars: at least 2% of the side, else keep the picture whole
    if y0 < 0.02 * h:
        y0 = 0
    if h - y1 < 0.02 * h:
        y1 = h
    if x0 < 0.02 * w:
        x0 = 0
    if w - x1 < 0.02 * w:
        x1 = w
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def _load_picture(path: str) -> np.ndarray:
    from PIL import Image, ImageOps

    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        raise O.OptionError(f"content: {path} could not be opened as a picture") from None
    return np.asarray(img)


def _item_from(entry, base: Path) -> Item:
    if isinstance(entry, str):
        entry = {"path": entry}
    if not isinstance(entry, dict):
        raise O.OptionError("content: each item is a path, a line of text, or {\"path\": …, \"from\": …, \"to\": …}")
    allowed = {"path", "text", "from", "to", "at", "fit", "kind", "layout"}
    unknown = set(entry) - allowed
    if unknown:
        raise O.OptionError(f"content: an item has {', '.join(sorted(unknown))}; an item takes {', '.join(sorted(allowed))}")
    it = Item(kind="text" if entry.get("text") else "")
    it.text = str(entry.get("text", "")).strip()
    if not it.text:
        p = str(entry.get("path", "")).strip()
        if not p:
            raise O.OptionError("content: an item has neither a path nor a text")
        q = Path(os.path.expanduser(p))
        if not q.is_absolute():
            q = base / q
        it.path = str(q)
    for k in ("from", "to"):
        v = entry.get(k)
        if v is None or v == "":
            continue
        if isinstance(v, (int, float)):
            setattr(it, "start" if k == "from" else "end", float(v))
        else:
            secs = parse_seconds(str(v))
            if secs is not None and len(secs) == 1:
                setattr(it, "start" if k == "from" else "end", secs[0])
            else:
                setattr(it, "note", (it.note + f" {k}={v}").strip())  # words: found once the words are read
                setattr(it, "start" if k == "from" else "end", None)
                it.__dict__.setdefault("said", {})[k] = str(v)
    if entry.get("at") not in (None, ""):
        try:
            it.at = float(str(entry["at"]).rstrip("s"))
        except ValueError:
            raise O.OptionError(f"content: at={entry['at']!r} is not a number of seconds") from None
    it.fit = str(entry.get("fit", "auto")).lower()
    if it.fit not in ("auto", "fill", "whole", "fit"):
        raise O.OptionError(f"content: fit is auto, fill or whole (got {it.fit!r})")
    it.fit = "whole" if it.fit == "fit" else it.fit
    kind = str(entry.get("kind", "")).lower()
    if kind and kind not in ("video", "picture", "screen", "text"):
        raise O.OptionError(f"content: kind is video, picture, screen or text (got {kind!r})")
    it.kind = kind or it.kind
    lay = str(entry.get("layout", "")).lower()
    if lay and lay not in LAYOUTS + ("full",):
        raise O.OptionError(f"content: layout is {', '.join(LAYOUTS)} (got {lay!r})")
    it.layout = lay
    return it


def read_items(spec: str) -> list[Item]:
    """The content as typed: a folder, a .json list, a JSON list itself, or items separated by |."""
    spec = spec.strip()
    if not spec:
        return []
    base = Path.cwd()
    data = None
    if spec[:1] in "[{":
        try:
            data = json.loads(spec)
        except json.JSONDecodeError as e:
            raise O.OptionError(f"content: the list could not be read ({e.msg} at {e.pos})") from None
    else:
        whole = Path(os.path.expanduser(spec.strip("\"'")))
        if whole.is_dir():
            files = [p for p in whole.iterdir() if p.is_file() and not p.name.startswith(".")
                     and p.suffix.lower() in VIDEO_EXT | PICTURE_EXT]
            if not files:
                raise O.OptionError(f"content: the folder {whole} has no videos or pictures in it")
            return [_item_from(str(p), whole) for p in sorted(files, key=lambda p: _natural(p.name))]
        if whole.suffix.lower() == ".json" and whole.is_file():
            try:
                data = json.loads(whole.read_text())
            except json.JSONDecodeError as e:
                raise O.OptionError(f"content: {whole} is not a JSON list ({e.msg})") from None
            base = whole.parent
    if data is not None:
        if isinstance(data, dict):
            data = data.get("content", [data])
        if not isinstance(data, list) or not data:
            raise O.OptionError("content: give a list of items")
        return [_item_from(e, base) for e in data]
    out = []
    for piece in layers.split(spec):
        if not piece:
            continue
        m = re.match(r"(?is)^\s*text\s*:\s*(.+)$", piece)
        if m:
            out.append(_item_from({"text": m.group(1).strip().strip("\"'")}, base))
            continue
        t = re.match(r"^(.*?)@\s*(\d+(?:\.\d+)?)s?(?:\s*-\s*(\d+(?:\.\d+)?)s?)?\s*$", piece)
        entry: dict = {"path": piece.strip().strip("\"'")}
        if t and not Path(os.path.expanduser(piece.strip())).exists():
            entry = {"path": t.group(1).strip().strip("\"'"), "from": float(t.group(2))}
            if t.group(3):
                entry["to"] = float(t.group(3))
        p = Path(os.path.expanduser(entry["path"]))
        if p.is_dir():
            out.extend(read_items(str(p)))
            continue
        out.append(_item_from(entry, base))
    return out


def classify(it: Item) -> None:
    """What kind of thing each item is, and how it fills its panel."""
    if it.kind == "text":
        return
    p = Path(it.path)
    if not p.exists():
        raise O.OptionError(f"content: there is no file at {p}")
    ext = p.suffix.lower()
    if ext in PICTURE_EXT or it.kind in ("picture",) and ext not in VIDEO_EXT:
        frames = [_load_picture(str(p))]
        guess = "picture"
    elif ext in VIDEO_EXT or it.kind == "video":
        try:
            info = video.probe(p)
        except video.MediaError as e:
            raise O.OptionError(f"content: {e}") from None
        frames = [video.read_frame(info, int(info.frames * k)) for k in (0.2, 0.5, 0.8)]
        guess = "video"
        it.__dict__["info"] = info
    else:
        raise O.OptionError(f"content: {p.name} is neither a video nor a picture the pack can read "
                            f"(videos: {', '.join(sorted(VIDEO_EXT))}; pictures: {', '.join(sorted(PICTURE_EXT))})")
    flat = float(np.median([_flatness(f) for f in frames]))
    screen = it.kind == "screen" or (not it.kind and (flat >= SCREEN_FLAT or bool(SCREEN_NAMES.search(p.stem))))
    it.__dict__["flat"] = flat
    it.__dict__["bars"] = _bars(frames)
    it.__dict__["moving"] = guess == "video"
    if screen:
        it.kind, it.fit = "screen", "whole"
    else:
        it.kind = guess if it.kind not in ("video", "picture") else it.kind


# ---- reading what shares the frame, frame by frame ------------------------------------------------

def screen_pad(pw: float, ph: float, unit: float) -> int:
    """The matte round a screen in its panel."""
    return int(round(max(10 * unit, 0.035 * min(pw, ph))))


def _fps_text(fps) -> str:
    return f"{fps.numerator}/{fps.denominator}" if hasattr(fps, "numerator") else f"{float(fps):.6f}"


class Footage:
    """A clip of the person's own, read at the size its panel needs, frame by frame at the film's
    rate. It keeps reading forward while frames are asked for in order, and starts again from the
    right second when one is asked for out of order (a still)."""

    def __init__(self, it: Item, size: tuple[int, int], fps, mode: str):
        """`mode`: fill (cover `size`, cropped about the middle), fit (whole, inside `size`), or exact
        (scaled to exactly `size`, which already has the picture's shape)."""
        self.it, self.size, self.fps = it, size, fps
        self.rate = float(fps)
        info = it.__dict__["info"]
        self.info = info
        bx, by, bw, bh = it.__dict__["bars"]
        pw, ph = size
        crop = ""
        if mode == "fill":
            k = max(pw / bw, ph / bh)
            sw, sh = max(pw, int(math.ceil(bw * k))), max(ph, int(math.ceil(bh * k)))
            self.out = (pw, ph)
            crop = f",crop={pw}:{ph}:{(sw - pw) // 2}:{(sh - ph) // 2}"
        elif mode == "fit":
            k = min(pw / bw, ph / bh)
            sw, sh = min(pw, max(2, int(round(bw * k)))), min(ph, max(2, int(round(bh * k))))
            self.out = (sw, sh)
        else:
            sw, sh = pw, ph
            k = pw / bw
            self.out = (pw, ph)
        self.scale = k
        cut = f"crop={bw}:{bh}:{bx}:{by}," if (bw, bh) != (info.width, info.height) else ""
        flags = "lanczos+accurate_rnd+full_chroma_int" if k < 1 else "bicubic+accurate_rnd+full_chroma_int"
        self.vf = (f"fps={_fps_text(fps)},{cut}scale={sw}:{sh}:flags={flags}:in_color_matrix={info.color_matrix}"
                   f"{crop},format=rgb24")
        self.proc = None
        self.next = 0
        self.last: Optional[np.ndarray] = None
        self.ended = False

    def _start(self, index: int) -> None:
        self.close()
        t = index / self.rate
        cmd = ["ffmpeg", "-v", "error", "-nostdin"]
        if t > 0:
            cmd += ["-ss", f"{t:.6f}"]
        cmd += ["-i", str(Path(self.info.path).resolve()), "-map", "0:v:0", "-vf", self.vf, "-f", "rawvideo",
                "-pix_fmt", "rgb24", "-"]
        w, h = self.out
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=w * h * 6)
        self.next, self.ended = index, False

    def _read(self) -> Optional[np.ndarray]:
        w, h = self.out
        buf = self.proc.stdout.read(w * h * 3) if self.proc else b""
        if len(buf) < w * h * 3:
            self.ended = True
            return None
        self.next += 1
        return np.frombuffer(bytearray(buf), np.uint8).reshape(h, w, 3)

    def frame(self, t: float) -> np.ndarray:
        """The picture `t` seconds into the clip (its last frame once it has ended)."""
        index = max(0, int(round(t * self.rate)))
        if self.last is not None and (index == self.next - 1 or (self.ended and index >= self.next - 1)):
            return self.last  # the frame just read (or the last there is)
        if self.proc is None or index < self.next or index > self.next + 2 * self.rate:
            if not (self.ended and index >= self.next and self.last is not None):
                self._start(index)
        while not self.ended and self.next <= index:
            f = self._read()
            if f is not None:
                self.last = f
        if self.last is None:  # nothing at all from there: the clip is shorter than that
            self._start(max(0, int((self.info.duration - 0.5) * self.rate)))
            while not self.ended:
                f = self._read()
                if f is not None:
                    self.last = f
            if self.last is None:
                raise video.MediaError(f"could not read the picture of {self.info.path}")
        return self.last

    def close(self) -> None:
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait()
            self.proc = None


class Content:
    """One item drawn into its panel, at the panel's size: `frame(t)` is the panel's picture `t`
    seconds into the time the item has been on screen."""

    def __init__(self, it: Item, size: tuple[int, int], fps, look: Look, unit: float, total: float,
                 card: Optional["Card"] = None, band: Optional[Rect] = None):
        self.it, self.size, self.fps, self.look, self.unit = it, size, fps, look, unit
        self.band = band or Rect(0.0, 0.0, float(size[0]), float(size[1]))  # the part the app leaves clear
        self.total = max(0.5, total)  # seconds it is on screen in all
        self.card = card
        self.footage: Optional[Footage] = None
        self.still: Optional[np.ndarray] = None
        self.base: Optional[np.ndarray] = None
        pw, ph = size
        self.notes: list[str] = []
        if it.kind in ("text", "line"):
            return
        if it.kind == "screen":
            self._screen()
            return
        bx, by, bw, bh = it.__dict__["bars"]
        kept = min(bw / bh, pw / ph) / max(bw / bh, pw / ph)
        fill = it.fit == "fill" or (it.fit == "auto" and kept >= FILL_KEEPS)
        self.fill = fill
        if it.kind == "video":
            self.footage = Footage(it, size, fps, "fill" if fill else "fit")
            at = it.at if it.at is not None else (min(1.0, 0.1 * it.__dict__["info"].duration)
                                                   if it.__dict__["info"].duration >= 1.5 * self.total else 0.0)
            self.at = max(0.0, min(at, it.__dict__["info"].duration - 0.1))
            how = (f"fills its panel (keeps {kept:.0%} of the picture)" if fill
                   else f"shown whole over a blurred copy (filling would keep only {kept:.0%})")
            self.notes.append(f"{it.label()}: your clip, {how}, from {self.at:.1f} s")
            # a clip shorter than its time on screen plays again from where it began, through a dissolve
            self.run = max(0.5, it.__dict__["info"].duration - self.at - 1.5 / float(fps))
            self.again: Optional[Footage] = None
            if self.total > self.run + 0.05:
                self.again = Footage(it, size, fps, "fill" if fill else "fit")
                self.notes.append(f"{it.label()} has {self.run:.1f} s from there but is on screen for "
                                  f"{self.total:.1f} s: it plays again through a dissolve")
        else:
            self._picture(fill, kept)

    # -- a picture: cropped to fill (or whole over its blur), moving slowly closer
    def _picture(self, fill: bool, kept: float) -> None:
        import cv2

        pw, ph = self.size
        img = _load_picture(self.it.path)
        bx, by, bw, bh = self.it.__dict__["bars"]
        img = img[by:by + bh, bx:bx + bw]
        self.push = float(np.clip(0.012 * self.total, 0.03, 0.08))
        room = 1.0 + self.push + 0.02
        if fill:
            k = max(pw / bw, ph / bh) * room
            self.base = cv2.resize(img, (max(pw, int(math.ceil(bw * k))), max(ph, int(math.ceil(bh * k)))),
                                   interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
            self.notes.append(f"{self.it.label()}: your picture, fills its panel (keeps {kept:.0%}) and moves "
                              f"{self.push:.0%} closer over {self.total:.1f} s")
        else:
            k = min(pw / bw, ph / bh) * room
            self.base = cv2.resize(img, (max(2, int(round(bw * k))), max(2, int(round(bh * k)))),
                                   interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
            self.notes.append(f"{self.it.label()}: your picture, shown whole over a blurred copy, moving "
                              f"{self.push:.0%} closer")
        self.fill = fill

    # -- a screen: scaled once, sharp, a window on a card
    def _screen(self) -> None:
        import cv2

        pw, ph = self.size
        it = self.it
        bx, by, bw, bh = it.__dict__["bars"]
        inset = screen_pad(pw, ph, self.unit)
        band = self.band
        room_h = max(2 * inset + 8, band.h)
        k = min((pw - 2 * inset) / bw, (room_h - 2 * inset) / bh)
        sw, sh = _even(bw * k), _even(bh * k)
        y = int(round(band.y0 + (band.h - sh) / 2)) if band.h >= sh + 2 * inset else (ph - sh) // 2
        self.window = (sw, sh, (pw - sw) // 2, min(max(0, y), ph - sh))
        self.scale = k
        self.radius = max(4.0, 10 * self.unit)
        if it.__dict__.get("moving"):
            self.footage = Footage(it, (sw, sh), self.fps, "exact")
            self.at = it.at if it.at is not None else 0.0
        else:
            img = _load_picture(it.path)[by:by + bh, bx:bx + bw]
            from PIL import Image

            self.still = np.asarray(Image.fromarray(img).resize((sw, sh), Image.LANCZOS))
        self.notes.append(f"{it.label()}: a screen, shown whole and never moved, scaled once to {sw}x{sh} "
                          f"({k:.2f} of its size)")
        if k < 0.5:
            self.notes.append(f"{it.label()} is wide for this panel, so its words are small: crop it to the part "
                              f"that matters, or use layout=side on a wide film")
        self._screen_ground = None

    def _screen_frame(self, t: float) -> np.ndarray:
        sw, sh, x, y = self.window
        if self._screen_ground is None:
            pw, ph = self.size
            g = np.empty((ph, pw, 3), np.uint8)
            g[:] = self.look.matte
            shadow = _rounded_shadow(sw, sh, self.radius, 14 * self.unit, 0.45)
            over(g, shadow, x - (shadow.w - sw) / 2, y - (shadow.h - sh) / 2 + 5 * self.unit)
            self._screen_ground = g
            self._screen_mask = _mask(sw, sh, self.radius)
        out = self._screen_ground.copy()
        pic = self.still if self.still is not None else self.footage.frame(self.at + t)
        _paste(out, pic, x, y, self._screen_mask, 1.0)
        return out

    def _blurred(self, pic: np.ndarray) -> np.ndarray:
        """`pic` spread over the whole panel, blurred and dimmed: the ground for a picture shown whole."""
        import cv2

        pw, ph = self.size
        h, w = pic.shape[:2]
        k = max(pw / w, ph / h)
        small = cv2.resize(pic, (max(4, int(w * k / 12)), max(4, int(h * k / 12))), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 2.2)
        big = cv2.resize(small, (int(math.ceil(w * k)), int(math.ceil(h * k))), interpolation=cv2.INTER_LINEAR)
        y0, x0 = (big.shape[0] - ph) // 2, (big.shape[1] - pw) // 2
        return (big[y0:y0 + ph, x0:x0 + pw].astype(np.float32) * 0.62).astype(np.uint8)

    def frame(self, t: float, t_film: float) -> np.ndarray:
        import cv2

        pw, ph = self.size
        if self.card is not None:
            return self.card.frame(t_film)
        if self.it.kind == "screen":
            return self._screen_frame(t)
        if self.footage is not None:
            pic = self._clip_frame(t)
            if self.fill:
                return pic.copy()
            ground = self._blurred(pic)
            h, w = pic.shape[:2]
            ground[(ph - h) // 2:(ph - h) // 2 + h, (pw - w) // 2:(pw - w) // 2 + w] = pic
            return ground
        # a picture moving closer at one steady rate
        z = 1.0 + self.push * min(1.0, max(0.0, t / self.total))
        bh, bw = self.base.shape[:2]
        room = 1.0 + self.push + 0.02
        if self.fill:
            k = z / room  # base pixels -> panel pixels
            cx, cy = bw / 2, bh / 2 - 0.012 * bh * min(1.0, t / self.total)
            m = np.float32([[k, 0, pw / 2 - cx * k], [0, k, ph / 2 - cy * k]])
            return cv2.warpAffine(self.base, m, (pw, ph), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        k = z / room
        ground = self._blurred(self.base)
        m = np.float32([[k, 0, pw / 2 - bw / 2 * k], [0, k, ph / 2 - bh / 2 * k]])
        a = (cv2.warpAffine(np.full((bh, bw), 255, np.uint8), m, (pw, ph), flags=cv2.INTER_LINEAR)
             .astype(np.float32) / 255.0)[..., None]
        pic = cv2.warpAffine(self.base, m, (pw, ph), flags=cv2.INTER_LINEAR)
        return (ground * (1 - a) + pic * a + 0.5).astype(np.uint8)

    def _clip_frame(self, t: float) -> np.ndarray:
        """Your clip `t` seconds into its time on screen: played through, and again from where it began
        as often as it must, each time through a dissolve from where it ended (the two plays read by
        two readers in turn, so neither has to seek back)."""
        if self.again is None or t < self.run:
            return self.footage.frame(self.at + min(t, self.run))
        n = int(t // self.run)
        tt = t - n * self.run
        reader, before = (self.again, self.footage) if n % 2 else (self.footage, self.again)
        pic = reader.frame(self.at + tt)
        if tt >= LOOP_DISSOLVE:
            return pic
        ended = before.frame(self.at + self.run)  # where the last play ended, held under the dissolve
        a = _smooth(tt / LOOP_DISSOLVE)
        return (ended.astype(np.float32) * (1 - a) + pic.astype(np.float32) * a + 0.5).astype(np.uint8)

    def close(self) -> None:
        if self.footage is not None:
            self.footage.close()
        if getattr(self, "again", None) is not None:
            self.again.close()


# ---- a line of text on a card ----------------------------------------------------------------------

class Card:
    """A line set on a card, as large as its safe part allows (at most three lines), with its key word
    on a marked block. A line that is said is on the card from the section's first frame, faint, and
    each word darkens as it is said, the mark landing on the key word when it is: the card is never
    empty and reads with the voice. A line that is given rises in as its section starts."""

    FACE = "poppins-extrabold"

    def __init__(self, tokens: list[str], key: Optional[int], times: Optional[list[float]], start: float,
                 size: tuple[int, int], room: Rect, look: Look, unit: float):
        self.size, self.look, self.unit = size, look, unit
        self.tokens = [t for t in tokens if t]
        self.key = key if key is not None and 0 <= key < len(self.tokens) else None
        self.start = start
        self.spoken = bool(times)
        self.times = times or [start + 0.06 * k for k in range(len(self.tokens))]
        pw, ph = size
        self.ground = np.empty((ph, pw, 3), np.uint8)
        self.ground[:] = look.card
        self.room = room
        self._layout()

    def _wrap(self, px: int) -> list[list[int]]:
        space = textmod.advance(" ", self.FACE, px)
        widths = [textmod.advance(t, self.FACE, px) for t in self.tokens]
        limit = self.room.w
        best = None
        n = len(self.tokens)
        for lines in (1, 2, 3):
            if lines > n:
                break
            # the most even split into `lines` lines
            target = (sum(widths) + space * (n - 1)) / lines
            rows, cur, cur_w = [], [], 0.0
            for i, w in enumerate(widths):
                add = w if not cur else cur_w + space + w
                if cur and add > target * 1.08 and len(rows) < lines - 1:
                    rows.append(cur)
                    cur, cur_w = [i], w
                else:
                    cur.append(i)
                    cur_w = add
            rows.append(cur)
            widest = max(sum(widths[i] for i in r) + space * (len(r) - 1) for r in rows)
            if widest <= limit:
                return rows
            best = rows
        return best if best is not None else [list(range(n))]

    def _layout(self) -> None:
        room = self.room
        lo, hi = 12, int(min(room.h * 0.36, 150 * self.unit))
        chosen = None
        while lo <= hi:
            px = (lo + hi) // 2
            rows = self._wrap(px)
            cap = textmod.cap_height(self.FACE, px)
            lead = cap * 1.55
            height = cap + lead * (len(rows) - 1) + cap * 0.35
            space = textmod.advance(" ", self.FACE, px)
            widest = max(sum(textmod.advance(self.tokens[i], self.FACE, px) for i in r) + space * (len(r) - 1)
                         for r in rows)
            if widest <= room.w and height <= room.h and len(rows) <= 3:
                chosen = (px, rows)
                lo = px + 1
            else:
                hi = px - 1
        if chosen is None:
            chosen = (12, self._wrap(12))
        px, rows = chosen
        self.px = px
        cap = textmod.cap_height(self.FACE, px)
        lead = cap * 1.55
        space = textmod.advance(" ", self.FACE, px)
        block_h = cap + lead * (len(rows) - 1)
        y = room.cy - block_h / 2 + cap  # the first baseline
        self.pieces = []  # (token index, sprite, x, y top-left)
        self.mark_rect: Optional[Rect] = None
        for r in rows:
            widths = [textmod.advance(self.tokens[i], self.FACE, px) for i in r]
            line_w = sum(widths) + space * (len(r) - 1)
            x = room.cx - line_w / 2
            for i, w in zip(r, widths):
                is_key = i == self.key
                fill = self.look.mark_ink if is_key else self.look.ink
                word = textmod.draw_word(self.tokens[i], self.FACE, px, fill=fill)
                self.pieces.append((i, word.sprite, x - word.ox, y - word.oy))
                if is_key:
                    pad_x, pad_y = 0.16 * cap, 0.22 * cap
                    self.mark_rect = Rect(x - pad_x, y - cap - pad_y, x + w + pad_x, y + pad_y + 0.18 * cap)
                x += w + space
            y += lead
        self.text_rect = Rect.around(Rect(px_x, px_y, px_x + s.w, px_y + s.h) for _, s, px_x, px_y in self.pieces)
        if self.mark_rect is not None:
            m = self.mark_rect
            self.mark = textmod.rounded_box(m.w, m.h, 0.18 * cap, fill=self.look.mark)

    def frame(self, t: float) -> np.ndarray:
        out = self.ground.copy()
        if self.mark_rect is not None and self.key is not None:
            on = self.times[self.key]
            q = _out((t - on + 1 / 60) / 0.16)
            if q > 0:
                m = self.mark_rect
                w = max(1, int(round(m.w * q)))
                part = Sprite(rgb=self.mark.rgb[:, :w], a=self.mark.a[:, :w])
                over(out, part, m.x0, m.y0)
        rise = 0.10 * textmod.cap_height(self.FACE, self.px)
        for i, sprite, x, y in self.pieces:
            if self.spoken:  # faint from the start, full as it is said
                first = _out((t - self.start + 1 / 60) / 0.2)
                q = _out((t - self.times[i] + 1 / 60) / 0.15)
                over(out, sprite, x, y, opacity=first * (0.26 + 0.74 * q))
                continue
            q = _out((t - self.times[i] + 1 / 60) / 0.18)
            if q <= 0:
                continue
            over(out, sprite, x, y + rise * (1 - q), opacity=q)
        return out


# ---- drawing panels ----------------------------------------------------------------------------------

_MASKS: "OrderedDict[tuple, tuple]" = OrderedDict()


def _sdf(w: int, h: int, r: float, pad: int = 0) -> np.ndarray:
    """Distance from the edge of a w x h box with round corners of radius r (negative inside)."""
    ys = np.arange(-pad, h + pad, dtype=np.float32)[:, None] + 0.5
    xs = np.arange(-pad, w + pad, dtype=np.float32)[None, :] + 0.5
    r = float(max(0.0, min(r, w / 2, h / 2)))
    qx = np.abs(xs - w / 2) - (w / 2 - r)
    qy = np.abs(ys - h / 2) - (h / 2 - r)
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r


def _mask(w: int, h: int, r: float):
    """The panel's coverage (0..1), and the rows from `top` to `bot` it covers across its whole width,
    with the coverage of the rows above and below them (the round corners): kept for reuse."""
    key = (int(w), int(h), round(float(r), 1))
    got = _MASKS.get(key)
    if got is not None:
        _MASKS.move_to_end(key)
        return got
    a = np.clip(0.5 - _sdf(int(w), int(h), r), 0.0, 1.0).astype(np.float32)
    rows = np.flatnonzero((a >= 0.999).all(axis=1))
    top, bot = (int(rows[0]), int(rows[-1]) + 1) if len(rows) else (int(h), int(h))
    got = (a, top, bot, a[:top, :, None].copy(), a[bot:, :, None].copy())
    _MASKS[key] = got
    if len(_MASKS) > 96:
        _MASKS.popitem(last=False)
    return got


_SHADOWS: "OrderedDict[tuple, object]" = OrderedDict()


def _keep(key, make):
    got = _SHADOWS.get(key)
    if got is None:
        got = make()
        _SHADOWS[key] = got
        if len(_SHADOWS) > 64:
            _SHADOWS.popitem(last=False)
    return got


def _rounded_shadow(w: int, h: int, r: float, blur: float, strength: float) -> Sprite:
    import cv2

    def make() -> Sprite:
        pad = int(math.ceil(blur * 3))
        a = np.clip(0.5 - _sdf(int(w), int(h), r, pad), 0.0, 1.0).astype(np.float32)
        k = 4 if min(a.shape) > 64 else 1
        small = cv2.resize(a, (a.shape[1] // k, a.shape[0] // k), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), max(0.5, blur / k))
        soft = cv2.resize(small, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
        return Sprite(rgb=np.zeros(a.shape + (3,), np.float32), a=(soft * strength).astype(np.float32))

    return _keep(("shadow", int(w), int(h), round(float(r), 1), round(blur, 1), round(strength, 3)), make)


def _ring(w: int, h: int, r: float, width: float, colour: tuple) -> Sprite:
    def make() -> Sprite:
        d = _sdf(int(w), int(h), r)
        a = np.clip(1.0 - np.abs(d + width / 2) / (width / 2 + 0.5), 0.0, 1.0) * (d < 0.5) * colour[3]
        rgb = np.empty(d.shape + (3,), np.float32)
        rgb[:] = np.asarray(colour[:3], np.float32)
        return Sprite(rgb=rgb * a[..., None], a=a.astype(np.float32))

    return _keep(("ring", int(w), int(h), round(float(r), 1), round(width, 2), colour), make)


def _edges(sprite: Sprite, r0: int, r1: int, c0: int, c1: int) -> list[tuple[Sprite, int, int]]:
    """`sprite` without the block rows r0..r1 x columns c0..c1 (where a panel will lie over it, or
    where it has nothing): the strips round that block, each with its offset in the sprite."""
    H, W = sprite.a.shape
    r0, r1 = max(0, min(H, r0)), max(0, min(H, r1))
    c0, c1 = max(0, min(W, c0)), max(0, min(W, c1))
    if r1 <= r0 or c1 <= c0:
        return [(sprite, 0, 0)]
    out = []
    for (a, b, c, d) in ((0, r0, 0, W), (r1, H, 0, W), (r0, r1, 0, c0), (r0, r1, c1, W)):
        if b > a and d > c:
            out.append((Sprite(rgb=sprite.rgb[a:b, c:d], a=sprite.a[a:b, c:d]), c, a))
    return out


def _paste(canvas: np.ndarray, img: np.ndarray, x: float, y: float, mask, alpha: float) -> None:
    """Lay `img` on `canvas` with its top-left at (x, y), through a panel mask from `_mask`: the rows
    it covers whole are copied straight across, and only the round corners are blended."""
    a, top, bot, a_top, a_bot = mask
    H, W = canvas.shape[:2]
    h, w = a.shape
    x0, y0 = int(round(x)), int(round(y))
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if cx1 <= cx0 or cy1 <= cy0 or alpha <= 0.002:
        return
    if (cx0, cy0, cx1, cy1) == (x0, y0, x0 + w, y0 + h) and alpha >= 0.998:
        region = canvas[cy0:cy1, cx0:cx1]
        region[top:bot] = img[top:bot, :w]
        if top:
            under = region[:top].astype(np.float32)
            region[:top] = (under + (img[:top, :w] - under) * a_top + 0.5).astype(np.uint8)
        if bot < h:
            under = region[bot:].astype(np.float32)
            region[bot:] = (under + (img[bot:h, :w] - under) * a_bot + 0.5).astype(np.uint8)
        return
    sy0, sx0 = cy0 - y0, cx0 - x0
    sub = a[sy0:sy0 + cy1 - cy0, sx0:sx0 + cx1 - cx0][..., None] * alpha
    region = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
    pic = img[sy0:sy0 + cy1 - cy0, sx0:sx0 + cx1 - cx0].astype(np.float32)
    canvas[cy0:cy1, cx0:cx1] = (region * (1.0 - sub) + pic * sub + 0.5).astype(np.uint8)


# ---- your panel: the window of your picture it shows ------------------------------------------------

def follow(target: np.ndarray, hold: float, fps: float, starts: list[int]) -> np.ndarray:
    """A calm operator's path after `target`: it stays put while the target is within `hold`, eases
    after it when it drifts further, and starts afresh at each of `starts` (the clip's own cuts),
    where the picture jumps anyway."""
    out = np.empty(len(target), np.float64)
    bounds = sorted({0, len(target), *[s for s in starts if 0 < s < len(target)]})
    rate = min(1.0, 3.2 / fps)
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = target[a:b]
        pos = float(np.median(seg[:max(1, int(0.4 * fps))]))
        path = np.empty(len(seg))
        for k, v in enumerate(seg):
            off = v - pos
            if abs(off) > hold:
                pos += (off - math.copysign(hold, off)) * rate
            path[k] = pos
        sigma = 0.18 * fps
        if len(path) > 3 and sigma > 0.5:
            r = int(3 * sigma)
            ker = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
            ker /= ker.sum()
            path = np.convolve(np.pad(path, (r, r), mode="edge"), ker, mode="valid")
        out[a:b] = path
    return out


class Framer:
    """Where your face is in every frame of your own picture, and the window each panel shows."""

    # the part of the head that is never cut: the face box grown to the forehead and under the chin
    KEEP = {"left": 0.10, "top": 0.34, "right": 0.10, "bottom": 0.14}

    def __init__(self, clip, cuts: list[int]):
        info = clip.info
        self.W0, self.H0 = info.width, info.height
        self.fps = info.rate
        self.n = info.frames
        face = clip.face
        self.found = face.any() and face.found_share >= 0.3
        if self.found:
            b = face.boxes.astype(np.float64)
        else:  # nobody found: a face-sized box in the upper middle, where a speaker usually is
            w = 0.36 * self.W0
            b = np.tile([self.W0 / 2 - w / 2, 0.30 * self.H0 - w * 0.55, w, w * 1.1], (self.n, 1))
        self.bx, self.by, self.bw, self.bh = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        self.cx = follow(self.bx + self.bw / 2, 0.035 * self.W0, self.fps, cuts)
        self.cy = follow(self.by + self.bh / 2, 0.025 * self.H0, self.fps, cuts)
        self.cuts = cuts

    def keep(self, i: int) -> Rect:
        k = self.KEEP
        x, y, w, h = self.bx[i], self.by[i], self.bw[i], self.bh[i]
        return Rect(x - k["left"] * w, y - k["top"] * h, x + w * (1 + k["right"]), y + h * (1 + k["bottom"]))

    def head(self, i: int) -> Rect:
        x, y, w, h = self.bx[i], self.by[i], self.bw[i], self.bh[i]
        return Rect(x - HEAD["left"] * w, y - HEAD["top"] * h, x + w * (1 + HEAD["right"]), y + h * (1 + HEAD["bottom"]))

    def needs(self, frames: np.ndarray) -> tuple[float, float]:
        """The widest panel shape (width over height) that holds forehead to chin in every frame
        given, and the widest that holds the whole head with a little room."""
        if not len(frames):
            frames = np.arange(self.n)
        keep_h = np.percentile([self.keep(i).h for i in frames], 95)
        head_h = np.percentile([self.head(i).h for i in frames], 90)
        widest = self.W0 / min(self.H0, keep_h * 1.04)
        roomy = self.W0 / min(self.H0, head_h * 1.08)
        return widest, roomy

    def height_for(self, frames: np.ndarray, aspect: float, fill: float) -> float:
        """How tall a window (in your picture's pixels) a panel of `aspect` shows through a section:
        your head about `fill` of it, never less than forehead to chin with room, never larger than
        your picture, and never enlarged more than twice."""
        if not len(frames):
            frames = np.arange(self.n)
        head_h = float(np.percentile([self.head(i).h for i in frames], 80))
        keep_h = float(np.percentile([self.keep(i).h for i in frames], 95))
        keep_w = float(np.percentile([self.keep(i).w for i in frames], 95))
        most = min(self.H0, self.W0 / aspect)
        least = max(keep_h / 0.9, keep_w / (0.86 * aspect))
        return float(min(most, max(least, head_h / fill)))

    def window(self, i: int, aspect: float, h: float, eye: float, room: float = 0.07) -> tuple[float, float, float, float]:
        """The window (x, y, w, h) of your picture at frame i for a panel of `aspect`, `h` tall, your
        face's middle `eye` of the way down it, at least `room` of it over your hair where it can be,
        kept inside your picture."""
        w = h * aspect
        if w > self.W0:
            w = float(self.W0)
            h = w / aspect
        if h > self.H0:
            h = float(self.H0)
            w = h * aspect
        x = self.cx[i] - w / 2
        y = self.cy[i] - eye * h
        head, keep = self.head(i), self.keep(i)
        if room > 0:  # room over the hair (a cap stands taller than the face says); 0: the forehead is enough
            y = min(y, head.y0 - room * h)
        y = max(y, keep.y1 + 0.02 * h - h)  # the chin inside
        y = min(y, keep.y0 - 0.015 * h)  # the forehead inside, above all
        if keep.w + 0.04 * w <= w:  # the calm path may lag a quick move: never so far that the face is cut
            x = max(min(x, keep.x0 - 0.02 * w), keep.x1 + 0.02 * w - w)
        x = min(max(x, 0.0), self.W0 - w)
        y = min(max(y, 0.0), self.H0 - h)
        return x, y, w, h

    def sample(self, frame: np.ndarray, window, size: tuple[int, int]) -> np.ndarray:
        """The window of `frame` at `size`, without stepping a whole pixel as it moves."""
        import cv2

        x, y, w, h = window
        ow, oh = size
        s = ow / w
        src = frame
        f = 1.0
        if s < 0.66:  # a strong reduction: take most of it with an area average first, so nothing shimmers
            f = min(1.0, s * 1.5)
            src = cv2.resize(frame, (max(2, int(round(frame.shape[1] * f))), max(2, int(round(frame.shape[0] * f)))),
                             interpolation=cv2.INTER_AREA)
        k = s / f
        m = np.float32([[k, 0, -x * f * k], [0, k, -y * f * k]])
        return cv2.warpAffine(src, m, (ow, oh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ---- the frame, composed -----------------------------------------------------------------------------

@dataclass
class Moment:
    """What frame i shows: the section, how far into its arrival, your panel and its window."""
    k: int
    u: float  # 0..1 into the section's arrival; 1 once settled
    speaker: Rect
    radius: float
    lifted: float
    window: tuple


FILL = {"panel": 0.62, "split": 0.62, "side": 0.56, "pip": 0.62}  # your head's share of your panel's height
EYE = {"panel": 0.42, "split": 0.42, "side": 0.40, "pip": 0.40, "full": 0.42}  # your face's middle, down it


def section_framing(framer: "Framer", sh: Shape, frames: np.ndarray, H: int,
                    every_frame: bool = True) -> tuple[float, float, float]:
    """How a section frames you in `sh`: the window's height in your picture, where your face's middle
    sits down it, and the room kept over your hair. A panel that reaches into the bottom fifth puts
    your face as high as keeps it above on every frame (`every_frame`), and gives up the room over
    your hair for that (your forehead always stays in). Choosing the panel's size (`face_bottoms`)
    reads the typical face instead, so the panel is the size it always was and only the framing
    inside it moves."""
    aspect = sh.speaker.w / sh.speaker.h
    if sh.layout == "full":
        return float(min(framer.H0, framer.W0 / aspect)), EYE["full"], 0.07
    h = framer.height_for(frames, aspect, FILL[sh.layout])
    eye = EYE[sh.layout]
    p = sh.speaker
    if p.y1 > 0.8 * H + 1:
        s = p.h / h
        # How far the chin falls below the middle the window follows, at its lowest in the section: the
        # followed middle is calm while the face box is not (a mouth opened wide drops the chin), so
        # half the median face would let the chin into the bottom fifth on the frames it is lowest.
        at = frames if len(frames) else np.arange(framer.n)
        if every_frame:
            below = float(np.max(framer.by[at] + framer.bh[at] - framer.cy[at]))
        else:
            below = 0.5 * float(np.median(framer.bh[at]))
        lim = ((0.79 * H - p.y0) / s - below) / h
        return h, max(0.30, min(eye, lim)), 0.0
    return h, eye, 0.07


def face_bottoms(framer: "Framer", sh: Shape, frames: np.ndarray, H: int, step: int = 6) -> np.ndarray:
    """Where your chin lands on the frame through a section in `sh` (every `step`-th frame)."""
    h, eye, room = section_framing(framer, sh, frames, H, every_frame=False)
    aspect = sh.speaker.w / sh.speaker.h
    out = []
    for i in frames[::step]:
        wx, wy, ww, wh = framer.window(int(i), aspect, h, eye, room)
        out.append(sh.speaker.y0 + (framer.by[i] + framer.bh[i] - wy) * sh.speaker.h / wh)
    return np.asarray(out)


class Composer:
    """Every frame of the film: the ground, the content panel, your panel, their shadows and borders."""

    def __init__(self, clip, W: int, H: int, grid: Grid, look: Look, border: bool, transition: str,
                 sections: list[Section], shapes: list[Shape], items: list[Item],
                 framer: Framer, unit: float):
        self.clip, self.W, self.H = clip, W, H
        self.grid, self.look, self.border, self.transition = grid, look, border, transition
        self.sections, self.shapes, self.items = sections, shapes, items
        self.framer = framer
        self.unit = unit
        self.fps = clip.info.rate
        self.n = clip.info.frames
        self.d = TRANSITION_S[transition]
        self.contents: dict[tuple[int, int, int], Content] = {}
        self.cards: dict[int, Card] = {}
        self.inside: dict[tuple[int, int], list[layers.Layer]] = {}
        self._ground_cache: Optional[np.ndarray] = None
        self._grounds: dict[int, np.ndarray] = {}
        # what each section's panel shows of your picture: its height and where your face sits
        self.sec_h: list[float] = []
        self.sec_eye: list[float] = []
        self.sec_room: list[float] = []
        self.zone = SafeZone(W, H).rect
        for k, s in enumerate(sections):
            frames = np.arange(int(s.start * self.fps), max(int(s.start * self.fps) + 1, int(s.end * self.fps)))
            frames = frames[frames < self.n]
            sh = shapes[k]
            h, eye, room = section_framing(framer, sh, frames, H)
            self.sec_h.append(h)
            self.sec_eye.append(eye)
            self.sec_room.append(room)
        self.at_frame = np.zeros(self.n, np.int32)
        for k, s in enumerate(sections):
            a, b = int(round(s.start * self.fps)), int(round(s.end * self.fps))
            self.at_frame[max(0, a):max(0, min(self.n, b))] = k
        self.moments = [self._moment(i) for i in range(self.n)]

    def _moment(self, i: int) -> Moment:
        t = i / self.fps
        k = int(self.at_frame[i])
        s = self.sections[k]
        sh = self.shapes[k]
        u = 1.0
        if k > 0 and self.d > 0 and t < s.start + self.d:
            u = max(0.0, (t - s.start) / self.d)
        if u >= 1.0:
            rect, radius, lifted = sh.speaker, sh.r_speaker, sh.lifted
            h, eye, room = self.sec_h[k], self.sec_eye[k], self.sec_room[k]
        else:
            prev = self.shapes[k - 1]
            e = (_out if self.transition == "slide" else _smooth)(u)
            rect = _lerp_rect(prev.speaker, sh.speaker, e)
            radius = _lerp(prev.r_speaker, sh.r_speaker, e)
            lifted = _lerp(prev.lifted, sh.lifted, e)
            h = _lerp(self.sec_h[k - 1], self.sec_h[k], e)
            eye = _lerp(self.sec_eye[k - 1], self.sec_eye[k], e)
            room = _lerp(self.sec_room[k - 1], self.sec_room[k], e)
        aspect = max(0.05, rect.w / max(1.0, rect.h))
        return Moment(k, u, rect, radius, lifted, self.framer.window(i, aspect, h, eye, room))

    # -- where your picture is, for the effects laid over it (core/view.py)
    def place(self, i: int):
        m = self.moments[int(np.clip(i, 0, self.n - 1))]
        return m.window, m.speaker

    # -- the content of a section, at its panel's size
    def content_for(self, k: int, size: tuple[int, int]) -> Optional[Content]:
        s = self.sections[k]
        if s.item is None:
            return None
        key = (s.item, size[0], size[1])
        c = self.contents.get(key)
        if c is None:
            it = self.items[s.item]
            total = sum(x.end - x.start for x in self.sections if x.item == s.item)
            ct = self.shapes[k].content
            band = Rect(0.0, max(0.0, self.zone.y0 - ct.y0), float(size[0]),
                        float(size[1]) - max(0.0, ct.y1 - self.zone.y1))
            c = Content(it, size, self.clip.info.fps, self.look, self.unit, total, card=self.cards.get(k),
                        band=band)
            self.contents[key] = c
        if it_card := self.cards.get(k):
            c.card = it_card
        return c

    def panel_picture(self, k: int, size: tuple[int, int], t: float) -> np.ndarray:
        s = self.sections[k]
        c = self.content_for(k, size)
        return c.frame(s.local + (t - s.start), t)

    def _ground(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        W, H = self.W, self.H
        if not self.look.blur or frame is None:
            if self._ground_cache is None:
                g = np.empty((H, W, 3), np.uint8)
                g[:] = self.look.ground
                self._ground_cache = g
            return self._ground_cache.copy()
        # your own picture, cropped to the frame's shape, blurred and dimmed small, then brought up once
        h, w = frame.shape[:2]
        if w / h > W / H:
            cw, ch = int(round(h * W / H)), h
        else:
            cw, ch = w, int(round(w * H / W))
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        small = cv2.resize(frame[y0:y0 + ch, x0:x0 + cw], (max(4, W // 16), max(4, H // 16)),
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 2.5).astype(np.float32)
        small = (small * 0.42 + np.asarray(self.look.ground, np.float32) * 0.29 + 0.5).astype(np.uint8)
        return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)

    def _shadow(self, canvas: np.ndarray, rect: Rect, radius: float, opacity: float) -> None:
        """A panel's soft shadow, a little below it: only the strips round the panel are drawn, since
        the panel itself lies over the rest."""
        x0, y0 = int(round(rect.x0)), int(round(rect.y0))
        w, h = int(round(rect.x1)) - x0, int(round(rect.y1)) - y0
        if w < 2 or h < 2 or opacity <= 0.01 or self.look.shadow <= 0:
            return
        if w >= self.W - 1 and h >= self.H - 1:
            return
        r = min(radius, w / 2, h / 2)
        sh = _rounded_shadow(w, h, r, 20 * self.unit, self.look.shadow)
        pad = (sh.w - w) // 2
        dy = int(round(9 * self.unit))
        ri = int(math.ceil(r)) + 1
        for part, ox, oy in _edges(sh, pad - dy + ri, pad - dy + h - ri, pad, pad + w):
            over(canvas, part, x0 - pad + ox, y0 - pad + dy + oy, opacity=opacity)

    def _panel(self, canvas: np.ndarray, pic: np.ndarray, rect: Rect, radius: float, lifted: float,
               alpha: float = 1.0, shadow: bool = True) -> None:
        """Lay a panel's picture at `rect` (its size), with its shadow and border as lifted as `lifted`."""
        import cv2

        x0, y0 = int(round(rect.x0)), int(round(rect.y0))
        w, h = int(round(rect.x1)) - x0, int(round(rect.y1)) - y0
        if w < 2 or h < 2:
            return
        if pic.shape[1] != w or pic.shape[0] != h:
            pic = cv2.resize(pic, (w, h), interpolation=cv2.INTER_AREA if pic.shape[1] > w else cv2.INTER_LINEAR)
        r = min(radius, w / 2, h / 2)
        if shadow and lifted > 0.01:
            self._shadow(canvas, Rect(x0, y0, x0 + w, y0 + h), r, lifted * alpha)
        _paste(canvas, pic, x0, y0, _mask(w, h, r), alpha)
        if self.border and lifted > 0.01:
            bw = max(1.0, 2.0 * self.unit)
            ring = _ring(w, h, r, bw, self.look.edge)
            ri, rb = int(math.ceil(r)) + 2, int(math.ceil(bw)) + 2
            for part, ox, oy in _edges(ring, ri, h - ri, rb, w - rb):
                over(canvas, part, x0 + ox, y0 + oy, opacity=lifted * alpha)

    def _settled_ground(self, k: int) -> np.ndarray:
        """The ground of a settled section with its panels' shadows on it, made once (a plain ground
        does not change from frame to frame)."""
        got = self._grounds.get(k)
        if got is None:
            got = self._ground(None)
            sh = self.shapes[k]
            if sh.lifted > 0.01:
                if sh.content is not None and self.sections[k].item is not None:
                    self._shadow(got, sh.content, sh.r_content, sh.lifted)
                self._shadow(got, sh.speaker, sh.r_speaker, sh.lifted)
            self._grounds[k] = got
            if len(self._grounds) > 3:
                self._grounds.pop(next(iter(self._grounds)))
        return got.copy()

    def compose(self, frame: np.ndarray, i: int) -> np.ndarray:
        """Frame i of the film from frame i of your clip."""
        i = int(np.clip(i, 0, self.n - 1))
        t = i / self.fps
        mo = self.moments[i]
        k = mo.k
        s = self.sections[k]
        sh = self.shapes[k]
        e_raw = mo.u
        settled = e_raw >= 1 and not self.look.blur
        canvas = self._settled_ground(k) if settled else self._ground(frame)
        lift = not settled  # a settled plain ground already has the shadows on it
        ease = _out if self.transition == "slide" else _smooth
        e = ease(e_raw) if e_raw < 1 else 1.0
        prev = self.sections[k - 1] if (k > 0 and e_raw < 1) else None
        psh = self.shapes[k - 1] if prev is not None else None
        # the content arriving or settled, and the content leaving
        if sh.content is not None and s.item is not None:
            size = (int(sh.content.w), int(sh.content.h))
            pic = self.panel_picture(k, size, t)
            for L in self.inside.get((k, 0), []):
                L.draw(pic, i, t)
            rect, alpha = sh.content, 1.0
            if prev is not None:
                if psh.content is None or prev.item is None:  # arriving from you alone
                    rect, alpha = self._arrive(sh, mo, e)
                elif prev.item != s.item or psh.layout != sh.layout:
                    old_size = (int(psh.content.w), int(psh.content.h))
                    old = self.panel_picture(k - 1, old_size, t)
                    if psh.layout == sh.layout:
                        pic = self._swap(old, pic, e)
                    else:
                        rect = _lerp_rect(psh.content, sh.content, e)
                        self._panel(canvas, old, rect, _lerp(psh.r_content, sh.r_content, e),
                                    _lerp(psh.lifted, sh.lifted, e), 1.0 - e)
                        alpha = e
            self._panel(canvas, pic, rect, sh.r_content, sh.lifted if prev is None else _lerp(psh.lifted, sh.lifted, e),
                        alpha, shadow=lift and not (sh.layout == "pip"))
        elif prev is not None and psh.content is not None and prev.item is not None:  # leaving: you fill it again
            size = (int(psh.content.w), int(psh.content.h))
            old = self.panel_picture(k - 1, size, t)
            rect, alpha = self._leave(psh, mo, e)
            self._panel(canvas, old, rect, psh.r_content, psh.lifted * (1 - e), alpha)
        # you
        rect = mo.speaker
        w, h = int(round(rect.x1)) - int(round(rect.x0)), int(round(rect.y1)) - int(round(rect.y0))
        pic = self.framer.sample(frame, mo.window, (max(2, w), max(2, h)))
        self._panel(canvas, pic, rect, mo.radius, mo.lifted, shadow=lift)
        return canvas

    def _offset(self, sh: Shape, speaker_now: Rect) -> tuple[float, float]:
        """How far the content panel stands from its place while your panel is still moving to its own:
        it travels with your panel's inner edge, so it is pushed in rather than laid over."""
        sp, ct = sh.speaker, sh.content
        if sh.layout == "pip":
            return 0.0, 0.0
        if ct.y1 <= sp.y0 + 1:  # content above you
            return 0.0, speaker_now.y0 - sp.y0
        if ct.y0 >= sp.y1 - 1:  # content below you
            return 0.0, speaker_now.y1 - sp.y1
        if ct.x1 <= sp.x0 + 1:  # content to your left
            return speaker_now.x0 - sp.x0, 0.0
        return speaker_now.x1 - sp.x1, 0.0

    def _arrive(self, sh: Shape, mo: Moment, e: float) -> tuple[Rect, float]:
        if self.transition == "slide":
            if sh.layout == "pip":
                return sh.content.shift((1 - e) * 0.25 * self.W * (1 if sh.speaker.cx < self.W / 2 else -1), 0), e
            dx, dy = self._offset(sh, mo.speaker)
            return sh.content.shift(dx, dy), 1.0
        return _scaled(sh.content, 0.86 + 0.14 * e), min(1.0, 1.4 * e)

    def _leave(self, psh: Shape, mo: Moment, e: float) -> tuple[Rect, float]:
        if self.transition == "slide":
            if psh.layout == "pip":
                return psh.content, 1 - e
            dx, dy = self._offset(psh, mo.speaker)
            return psh.content.shift(dx, dy), 1.0
        return _scaled(psh.content, 1.0 - 0.12 * e), 1.0 - e

    def _swap(self, old: np.ndarray, new: np.ndarray, e: float) -> np.ndarray:
        """One item giving way to the next in the same panel: pushed across, or grown in over it."""
        import cv2

        h, w = new.shape[:2]
        if old.shape[:2] != (h, w):
            old = cv2.resize(old, (w, h), interpolation=cv2.INTER_AREA)
        if self.transition == "slide":
            out = np.empty_like(new)
            x = int(round(w * e))
            out[:, :w - x] = old[:, x:]
            out[:, w - x:] = new[:, :x]
            return out
        k = 1.06 - 0.06 * e
        m = np.float32([[k, 0, w / 2 * (1 - k)], [0, k, h / 2 * (1 - k)]])
        grown = cv2.warpAffine(new, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        a = min(1.0, 1.3 * e)
        return (old.astype(np.float32) * (1 - a) + grown.astype(np.float32) * a + 0.5).astype(np.uint8)

    def close(self) -> None:
        for c in self.contents.values():
            c.close()


class Composed(Framing):
    """The layout as the render's framing: every frame of the film is composed from the same frame
    of your clip, at its full size. A box of your picture maps to where your panel shows it."""

    def __init__(self, composer: Composer, note: str):
        n = composer.n
        windows = np.array([m.window for m in composer.moments], np.float64).reshape(n, 4)
        super().__init__(composer.framer.W0, composer.framer.H0, composer.W, composer.H, windows, note)
        self.composer = composer

    def apply(self, frame: np.ndarray, i: int) -> np.ndarray:
        return self.composer.compose(frame, i)

    def to_output(self, r: Rect, i: int) -> Rect:
        window, rect = self.composer.place(i)
        wx, wy, ww, _ = window
        s = rect.w / ww
        return Rect((r.x0 - wx) * s + rect.x0, (r.y0 - wy) * s + rect.y0, (r.x1 - wx) * s + rect.x0,
                    (r.y1 - wy) * s + rect.y0)


# ---- a soft swish under each move ----------------------------------------------------------------------

def _swish(rate: int, seconds: float, rng: np.random.Generator) -> np.ndarray:
    """Air moving past: noise through a band that rises as it swells and falls away (a steep top, so it
    is a rush of air around two kilohertz rather than a hiss), peak 1. Made here, so the pack carries
    no sound files."""
    n = int(seconds * rate)
    t = np.arange(n, dtype=np.float64) / n
    noise = rng.standard_normal(n)
    out = np.empty(n)
    lp = [0.0, 0.0, 0.0]
    hp = [0.0, 0.0]
    for j in range(n):
        f = 350.0 + 1800.0 * math.sin(math.pi * min(1.0, t[j] * 1.15)) ** 2
        a = math.exp(-2 * math.pi * f / rate)
        b = math.exp(-2 * math.pi * f * 0.3 / rate)
        x = y = noise[j]
        for k in range(3):
            lp[k] = (1 - a) * x + a * lp[k]
            x = lp[k]
        for k in range(2):
            hp[k] = (1 - b) * y + b * hp[k]
            y = hp[k]
        out[j] = x - y
    env = np.sin(np.pi * np.clip(t / 0.42, 0, 1) / 2) ** 2 * np.exp(-np.maximum(0, t - 0.42) * 5.5)
    out *= env
    peak = float(np.abs(out).max()) or 1.0
    return (out / peak).astype(np.float32)


# ---- the effect --------------------------------------------------------------------------------------------

def _layouts(text: str) -> list[str]:
    names = [x.strip().lower() for x in re.split(r"[,|/\s]+", text) if x.strip()]
    bad = [x for x in names if x not in LAYOUTS]
    if bad or not names:
        raise O.OptionError(f"layout: choose from {', '.join(LAYOUTS)}, or several separated by commas "
                            f"(got {text!r})")
    return names


def _canvas(info, frame: str, size) -> tuple[int, int]:
    if size:
        return size
    long_side = min(1920, max(info.width, info.height))
    if frame == "9:16":
        return _even(long_side * 9 / 16), _even(long_side)
    if frame == "16:9":
        return _even(long_side), _even(long_side * 9 / 16)
    return output_size(info.width, info.height)


def _key_line(words: list[Word], quiet, a: float, b: float,
              back: Optional[moments.TakenBack] = None) -> Optional[tuple[int, int]]:
    """The strongest short line said between a and b seconds, by the shared picker
    (`moments.pick_line`, the headline's): whole indices into `words`. `back`: what the speaker
    took back, read from the clip's whole words, which no line holds."""
    back = moments.taken_back(words, quiet) if back is None else back
    idx = [i for i, w in enumerate(words) if a - 0.05 <= w.start < b - 0.2]
    if not idx:
        return None
    sub = [words[i] for i in idx]
    got = moments.pick_line(sub, quiet, back=back)
    if got is None:
        return None
    return idx[got[0]], idx[got[1]]


def _tokens_given(text: str) -> tuple[list[str], Optional[int]]:
    """A typed line's words, and its key word: the one marked *like this*, else the most specific."""
    from .captions import key_word

    raw = text.split()
    key = None
    out = []
    for i, w in enumerate(raw):
        if w.startswith("*") and w.rstrip(".,!?;:").endswith("*") and len(w.strip("*.,!?;:")) > 0:
            key = i
            w = w.replace("*", "")
        out.append(w)
    if key is None and out:
        key = key_word([Word(w, 0.0, 0.0) for w in out])
    return out, key


def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from ..core import framing as framing_mod
    from ..core import shots
    from ..core.timeline import Timeline
    from ..core.view import View
    from . import Job, PlanError
    from .captions import key_word, shown

    info = clip.info
    fps = info.rate
    duration = info.frames / fps
    W, H = _canvas(info, opts["frame"], O.size_of(opts["size"]))
    unit = min(W, H) / 1080.0
    look = LOOKS[opts["style"]]
    grid = Grid(W, H, opts["corner"], float(opts["ratio"]))
    warnings: list[str] = []
    notes: list[str] = []
    order = _layouts(opts["layout"])
    words = clip.words
    back = moments.taken_back(words, clip.quiet)  # never a line's or a card's key word

    # -- the clip's own cuts: the operator starts afresh there, and they can be the switches
    try:
        cut_times = shots.cuts(clip)
    except Exception as e:  # never stop for this: the panel simply follows across the whole clip
        cut_times = []
        notes.append(f"the clip's cuts could not be read ({e})")
    cut_frames = [int(round(t * fps)) for t in cut_times]
    framer = Framer(clip, cut_frames)
    if not framer.found:
        warnings.append("no face was found in enough of the clip: your panel shows the upper middle of the picture")

    # -- what shares the frame
    if plan is not None:
        try:
            items = [Item(**{k: v for k, v in d.items() if k in Item.__dataclass_fields__}) for d in plan["items"]]
            for it in items:
                if it.span is not None:
                    it.span = tuple(it.span)
            sections = [Section(float(s["start"]), float(s["end"]), str(s["layout"]),
                                None if s.get("item") is None else int(s["item"]), float(s.get("local", 0.0)),
                                float(s.get("share", 0.0)))
                        for s in plan["sections"]]
        except (KeyError, TypeError, ValueError) as e:
            raise PlanError(f"the plan's sections or items could not be read ({e})") from None
        for s in sections:
            if s.layout not in LAYOUTS + ("full",):
                raise PlanError(f"a section's layout is {s.layout!r}; it must be full or one of {', '.join(LAYOUTS)}")
            if not (0 <= s.start < s.end <= duration + 0.05):
                raise PlanError(f"a section runs {s.start:.2f}-{s.end:.2f} s, outside the clip's {duration:.2f} s")
            if s.item is not None and not 0 <= s.item < len(items):
                raise PlanError(f"a section shows item {s.item}, but the plan has {len(items)} items")
            if s.layout == "full" and s.item is not None:
                raise PlanError("a full section shows you alone, so it has no item")
        for a, b in zip(sections, sections[1:]):
            if abs(a.end - b.start) > 1e-3:
                raise PlanError(f"the sections must follow on: one ends at {a.end:.2f} s, the next starts at {b.start:.2f} s")
        if not sections or sections[0].start > 1e-3 or abs(sections[-1].end - duration) > 0.06:
            raise PlanError(f"the sections must run from 0 to the end of the clip ({duration:.2f} s)")
        for it in items:
            if it.kind not in ("text", "line"):
                classify(it)
        switch_note = "from the plan"
    else:
        items = read_items(opts["content"])
        for it in items:
            classify(it)
        sections, switch_note = _sections(clip, opts, items, words, cut_times, duration, notes, warnings)
    # every section's layout cycles through `layout`, unless its item names one
    if plan is None:
        turn = 0
        for s in sections:
            if s.item is None:
                s.layout = "full"
                continue
            it = items[s.item]
            if it.layout and it.layout != "full":
                s.layout = it.layout
            else:
                s.layout = order[turn % len(order)]
                turn += 1
        # an item carries on where it left off when it comes back
        shown_for: dict[int, float] = {}
        for s in sections:
            if s.item is not None:
                s.local = shown_for.get(s.item, 0.0)
                shown_for[s.item] = s.local + (s.end - s.start)

    # -- the layouts used, each fitted to your shot; then each section's panel shaped to what it shows
    zone = SafeZone(W, H).rect
    shapes = _shapes(grid, framer, sections, items, fps, info.frames, zone, unit, notes, plan is not None)

    # -- the cards: typed lines, and the strongest line said in each section when nothing is given
    composer = Composer(clip, W, H, grid, look, opts["border"], opts["transition"], sections, shapes, items,
                        framer, unit)
    for k, s in enumerate(sections):
        if s.item is None:
            continue
        it = items[s.item]
        if it.kind not in ("text", "line"):
            continue
        sh = shapes[k]
        ct = sh.content
        room = _card_room(sh, zone, unit)
        room = room.shift(-ct.x0, -ct.y0)
        size = (int(ct.w), int(ct.h))
        if it.kind == "line" and it.span is not None:
            a, b = it.span
            ws = [w for w in words[a:b + 1] if w.key not in HESITATIONS]
            toks = [shown(w) for w in ws]
            key = key_word(ws, clip.quiet, back)
            times = [max(s.start, w.start) for w in ws]
            composer.cards[k] = Card(toks, key, times, s.start, size, room, look, unit)
        else:
            toks, key = _tokens_given(it.text)
            composer.cards[k] = Card(toks, key, None, s.start + 0.05, size, room, look, unit)

    framing = Composed(composer, f"{W}x{H}, laid out in {len(sections)} sections")
    tl = Timeline.whole(duration)

    # -- other effects laid over the whole frame, and inside the content panel
    over_layers: list[layers.Layer] = []
    hosted: list[dict] = []

    def still(i: int) -> np.ndarray:
        return composer.compose(video.read_frame(info, i), i)

    whole = View(clip, (W, H), place=composer.place, picture=still)
    whole_fr = framing_mod.plan(W, H, fps, info.frames, None, shape="keep")
    specs = [layers.parse(x) for x in layers.split(opts["over"]) if x.strip()] if opts["over"].strip() else []
    for spec in specs:
        if layers.kind(spec.name) != "over":
            raise O.OptionError(f"over: {spec.name} repaints the whole picture; run it on your clip first, then "
                                f"split-layout on the result")
        L = _lay(whole, spec, whole_fr, opts["fix"], "over", notes)
        over_layers.append(L)
        hosted.append({"where": "over", "effect": str(spec), "checks": _short(L.job.checks)})
    if opts["captions"] != "off":
        over_layers += _captions(opts, clip, composer, sections, shapes, W, H, whole, whole_fr, hosted, notes)
    inside_specs = [layers.parse(x) for x in layers.split(opts["inside"]) if x.strip()] if opts["inside"].strip() else []
    for spec in inside_specs:
        if layers.kind(spec.name) != "over":
            raise O.OptionError(f"inside: {spec.name} repaints the whole picture, so it cannot sit in a panel")
        _inside(spec, opts, clip, composer, sections, shapes, zone, hosted, warnings)

    def draw(frame: np.ndarray, j: int, t: float) -> None:
        for L in over_layers:
            L.draw(frame, j, t)

    # -- a soft swish under each move
    moves = []
    if opts["sound"] and opts["transition"] != "cut":
        for k in range(1, len(sections)):
            a, b = sections[k - 1], sections[k]
            if a.layout != b.layout:
                moves.append((b.start, 1.0))
            elif a.item != b.item:
                moves.append((b.start, 0.6))

    def mix(sound: np.ndarray, rate: int) -> np.ndarray:
        if not moves:
            return sound
        out = sound.copy()
        voice = np.abs(sound).max(axis=1) if len(sound) else np.zeros(1)
        level = float(np.percentile(voice, 99.5)) if len(voice) else 0.0
        peak = float(np.clip(0.16 * level, 0.012, 0.07))
        rng = np.random.default_rng(29)
        for at, weight in moves:
            s = _swish(rate, 0.5, rng) * peak * weight
            i0 = int(round(max(0.0, at - 0.06) * rate))
            i1 = min(len(out), i0 + len(s))
            if i1 > i0:
                out[i0:i1, 0] += s[:i1 - i0]
                out[i0:i1, 1] += s[:i1 - i0]
        return out

    checks = _report(composer, framer, sections, shapes, items, W, H, hosted)
    checks["sections"] = " · ".join(_describe(s, items) for s in sections)
    checks["switch"] = switch_note
    contents = []
    for c in composer.contents.values():
        contents += c.notes
    for k, s in enumerate(sections):  # make every content once now, so its notes are here
        if s.item is not None and shapes[k].content is not None:
            ct = shapes[k].content
            c = composer.content_for(k, (int(ct.w), int(ct.h)))
            for line in c.notes:
                if line not in contents:
                    contents.append(line)
    if contents:
        checks["content"] = " | ".join(contents)
    if notes:
        checks["notes"] = " | ".join(notes)
    checks["warnings"] = warnings + [w for h in hosted for w in h.get("warnings", [])]
    sheet = _moments(sections, composer, duration, fps, 6)
    out_plan = {"effect": NAME, "clip": clip.path, "options": opts,
                "output": {"width": W, "height": H, "fps": round(fps, 3), "seconds": round(duration, 3),
                           "framing": framing.note},
                "sections": [s.to_plan() for s in sections],
                "items": [_item_plan(it) for it in items],
                "panels": [{"speaker": v.speaker.as_list(), "content": v.content.as_list() if v.content else None}
                           for v in shapes],
                "cuts": cut_times, "hosted": hosted, "sheet_times": sheet}

    def still_times() -> list[float]:
        return _moments(sections, composer, duration, fps, 5)

    def after(result: dict, out: Path) -> dict:
        composer.close()
        return {}

    return Job(NAME, clip, tl, framing, draw if over_layers else None, out_plan, checks=checks,
               still_times=still_times, after=after, sound=mix if moves else None)


def _aspect(it: Item) -> Optional[float]:
    if it.kind in ("text", "line") or "bars" not in it.__dict__:
        return None
    _, _, bw, bh = it.__dict__["bars"]
    return bw / bh


def _shapes(grid: Grid, framer: Framer, sections: list[Section], items: list[Item], fps: float, n: int,
            zone: Rect, unit: float, notes: list[str], from_plan: bool) -> list[Shape]:
    """Each section's panels: its layout fitted to your shot (your share grows when your head needs
    it), then its content panel shaped to what it shows whole (a screen, a wide clip in a tall
    panel), your panel taking the rest. A plan's own shares are kept."""
    base: dict[str, Shape] = {"full": grid.shape("full")}
    for lay in sorted({s.layout for s in sections if s.layout != "full"}):
        frames = np.concatenate([np.arange(int(s.start * fps), int(s.end * fps)) for s in sections
                                 if s.layout == lay])
        frames = frames[(frames >= 0) & (frames < n)]
        widest, roomy = framer.needs(frames)

        def verdict(sh: Shape, frames=frames, widest=widest, roomy=roomy) -> str:
            aspect = sh.speaker.w / sh.speaker.h
            if aspect > widest + 1e-6:
                return "head"
            if sh.speaker.y1 > 0.8 * grid.H + 1:  # low: your face above the bottom fifth
                chins = face_bottoms(framer, sh, frames, grid.H)
                return "good" if not len(chins) or np.percentile(chins, 97) <= 0.8 * grid.H else "low"
            return "good" if aspect <= roomy else "hair"

        base[lay] = grid.fitted(lay, verdict)
        if base[lay].note:
            notes.append(f"{lay}: {base[lay].note}")
    out = []
    for s in sections:
        sh = base[s.layout]
        if from_plan and s.share > 0 and s.layout != "full":
            sh = grid.shape(s.layout, s.share)
        elif s.layout != "full" and s.item is not None and sh.content is not None:
            it = items[s.item]
            a = _aspect(it)
            if a is not None:
                ct = sh.content
                kept = min(a, ct.w / ct.h) / max(a, ct.w / ct.h)
                if it.kind == "screen" or it.fit == "whole" or (it.fit == "auto" and kept < FILL_KEEPS):
                    pad = screen_pad(ct.w, ct.h, unit) if it.kind == "screen" else 0.0
                    covered = (max(0.0, zone.y0 - ct.y0) + max(0.0, ct.y1 - zone.y1)) if it.kind == "screen" else 0.0
                    sh = grid.adapt(sh, a, pad, covered)
        s.share = round(sh.share, 3) if s.layout != "full" else 0.0
        out.append(sh)
    return out


def _moments(sections: list[Section], composer: "Composer", duration: float, fps: float, n: int) -> list[float]:
    """The moments that show the layout best, for stills and the sheet: each section with content once
    it has settled (a card once its line has been said), one move between layouts, and one moment of
    you alone; spread out, never two within a quarter second."""
    last = duration - 1.0 / fps
    shown, alone = [], []
    for k, s in enumerate(sections):
        settled = s.start + (composer.d + 0.1 if k else 0.0)
        if s.item is None:
            alone.append(min(last, (s.start + s.end) / 2))
            continue
        card = composer.cards.get(k)
        t = max(c for c in card.times) + 0.35 if card is not None else (s.start + s.end) / 2
        shown.append(min(last, s.end - 0.12, max(settled, t)))
    moves = [s.start + 0.5 * composer.d for k, s in enumerate(sections) if k and composer.d > 0]

    def spread(ts: list[float], m: int) -> list[float]:
        if len(ts) <= m:
            return list(ts)
        return [ts[int(i)] for i in np.linspace(0, len(ts) - 1, m).round()]

    want = spread(shown, max(1, n - 2)) + moves[:1] + alone[:1]
    out: list[float] = []
    for t in sorted(want):
        if all(abs(t - u) >= 0.25 for u in out):
            out.append(round(float(max(0.0, t)), 3))
    # room left: more moments across the sections with content, then across the rest
    spare = []
    for k, s in enumerate(sections):
        a, b = s.start + (composer.d + 0.1 if k else 0.0), min(last, s.end - 0.12)
        if s.item is not None and b > a:
            spare += [a + (b - a) * f for f in (0.2, 0.5, 0.8)]
    spare += [duration * (k + 0.5) / n for k in range(n)]
    for t in spare:
        if len(out) >= n:
            break
        if all(abs(t - u) >= 0.25 for u in out):
            out.append(round(float(t), 3))
    return sorted(out)[:n]


def _lay(view, spec, base, fix: str, where: str, notes: list[str]) -> layers.Layer:
    """Another effect of the pack laid on a view of the film, keeping its own moment. An effect whose
    ending holds the last frame on screen (a tier list's finished board) would make the film longer,
    so inside a layout that hold is left off unless the person set it."""
    from . import load

    try:
        return layers.prepare(view, spec, None, None, base, fix=fix)
    except ValueError as e:
        module = load(spec.name)
        if "changes the clip's length" in str(e) and "hold" in module.OPTIONS and "hold" not in spec.given:
            try:
                got = layers.prepare(view, layers.Spec(spec.name, {**spec.given, "hold": "0"}), None, None, base,
                                     fix=fix)
            except ValueError as e2:
                raise O.OptionError(f"{where}: {spec.name}: {e2}") from None
            notes.append(f"{where}: {spec.name} keeps no held ending inside a layout (hold=0); the film keeps its length")
            return got
        raise O.OptionError(f"{where}: {spec.name}: {e}") from None


def _short(checks: dict) -> dict:
    keep = ("frames_over_face", "frames_outside_safe_zone", "placed", "lands", "on_screen", "warnings")
    return {k: v for k, v in checks.items() if k in keep}


def _item_plan(it: Item) -> dict:
    d = {"kind": it.kind, "path": it.path, "text": it.text, "at": it.at, "fit": it.fit, "layout": it.layout,
         "start": it.start, "end": it.end}
    if it.span is not None:
        d["span"] = list(it.span)
    return d


def _describe(s: Section, items: list[Item]) -> str:
    what = "you" if s.item is None else items[s.item].label()
    return f"{s.start:.2f}-{s.end:.2f} {s.layout} ({what})"


def _card_room(sh: Shape, zone: Rect, unit: float) -> Rect:
    """Where a card's words may go: inside the content panel, inset, inside the app's safe zone, and
    in `pip` clear of your panel."""
    ct = sh.content
    inset = max(0.07 * min(ct.w, ct.h), 24 * unit)
    room = Rect(ct.x0 + inset, ct.y0 + inset, ct.x1 - inset, ct.y1 - inset)
    room = _inter(room, zone) or room
    if sh.layout == "pip":
        sp = sh.speaker
        gap = 24 * unit
        if sp.y0 > room.cy:
            room = Rect(room.x0, room.y0, room.x1, min(room.y1, sp.y0 - gap))
        else:
            room = Rect(room.x0, max(room.y0, sp.y1 + gap), room.x1, room.y1)
    return room


def _count(n: int, thing: str) -> str:
    return f"{n} {thing}" + ("" if n == 1 else "s")


def _sections(clip, opts, items, words, cut_times, duration, notes, warnings):
    """The sections, from the switch the person asked for, and which of them show content."""
    fps = clip.info.rate
    sw = opts["switch"].strip().lower() or "auto"
    given = parse_seconds(sw) if sw not in ("auto", "cuts", "words", "times") else None
    if sw not in ("auto", "cuts", "words", "times") and given is None:
        raise O.OptionError(f"switch: auto, cuts, words, times, or seconds such as 4.2, 9, 15 (got {opts['switch']!r})")
    # times said as words
    for it in items:
        said = it.__dict__.get("said", {})
        for k, phrase in said.items():
            from .headline import find_said

            got = find_said(words, phrase)
            if got is None:
                raise O.OptionError(f"content: the words \"{phrase}\" are not said in the clip; give seconds instead")
            t = words[got[0]].start if k == "from" else words[got[1]].end
            setattr(it, "start" if k == "from" else "end", t)
    timed = [it for it in items if it.start is not None]
    if timed and len(timed) != len(items):
        raise O.OptionError("content: give every item a from (or none of them, and the switch decides)")
    if sw == "times" and not timed:
        raise O.OptionError("switch=times needs a from (and a to) on each item of content")
    if timed:
        return _sections_from_times(items, duration, warnings), f"the times given ({len(items)} items)"
    # where the layout may change
    if sw == "auto":
        sw = "cuts" if len(cut_times) >= 2 else "words"
    if given is not None:
        breaks = [t for t in given if 0.3 < t < duration - 0.3]
        note = f"the seconds given ({', '.join(f'{t:g}' for t in breaks)})"
    elif sw == "cuts":
        if not cut_times:
            notes.append("the clip has no cuts of its own, so the layout changes between sentences instead")
            sw = "words"
        else:
            breaks = merge_short(cut_times, duration, MIN_SECTION["cuts"])
            note = (f"the clip's own cuts: {len(cut_times)} found, {len(breaks)} used"
                    + (f" (sections of at least {MIN_SECTION['cuts']:g} s)" if len(breaks) < len(cut_times) else ""))
    if given is None and sw == "words":
        strong, weak = sentence_breaks(words, clip.quiet, duration)
        breaks = choose_breaks(strong, weak, duration, MIN_SECTION["words"])
        note = (f"the breaks between sentences: {_count(len(strong), 'sentence end')} and "
                f"{_count(len(weak), 'pause')} found, {len(breaks)} used (sections of about 4 s)")
    edges = [0.0] + sorted(breaks) + [duration]
    raw = [Section(a, b, "full") for a, b in zip(edges[:-1], edges[1:]) if b - a > 1e-3]
    # which are you alone
    rule = opts["full"]
    n = len(raw)
    alone = set()
    if rule in ("first", "ends") and n >= 2:
        alone.add(0)
    if rule == "ends" and n >= 3:
        alone.add(n - 1)
    if rule == "between":
        alone = {k for k in range(n) if k % 2 == 0} if n >= 2 else set()
    content_secs = [k for k in range(n) if k not in alone]
    if not content_secs:
        content_secs = [n - 1]
        alone.discard(n - 1)
    # the items across the content sections, in order; nothing given: each section's strongest line
    back = moments.taken_back(words, clip.quiet)
    if not items:
        for k in content_secs:
            s = raw[k]
            span = _key_line(words, clip.quiet, s.start, s.end, back)
            if span is None:
                alone.add(k)
                continue
            text = " ".join(w.clean for w in words[span[0]:span[1] + 1] if w.key not in HESITATIONS)
            items.append(Item(kind="line", text=text, span=span))
            s.item = len(items) - 1
        if not items:
            warnings.append("nothing was said clearly enough to set as a line: give content to show")
    else:
        K, M = len(content_secs), len(items)
        if K < M:
            warnings.append(f"{M - K} of your {M} items had no section to show in (there are {K}); use "
                            f"switch=words, more sections with full=none, or fewer items")
        for j, k in enumerate(content_secs):
            raw[k].item = min(M - 1, int(j * M / K)) if K >= M else j
    # neighbours showing the same item are one section; so are neighbours of you alone
    merged: list[Section] = []
    for s in raw:
        if merged and merged[-1].item == s.item:
            merged[-1].end = s.end
        else:
            merged.append(Section(s.start, s.end, "full", s.item))
    return merged, note


def _sections_from_times(items: list[Item], duration: float, warnings: list[str]) -> list[Section]:
    order = sorted(range(len(items)), key=lambda j: items[j].start)
    out: list[Section] = []
    t = 0.0
    for pos, j in enumerate(order):
        it = items[j]
        a = max(0.0, min(duration, it.start))
        nxt = items[order[pos + 1]].start if pos + 1 < len(order) else duration
        b = it.end if it.end is not None else nxt
        b = max(a + 0.2, min(duration, b, nxt))
        if a > t + 0.05:
            out.append(Section(t, a, "full"))
        elif a < t:
            warnings.append(f"{it.label()} starts at {it.start:.2f} s, before the item before it ends; it starts at {t:.2f} s")
            a = t
        if b - a < MIN_SECTION["times"]:
            warnings.append(f"{it.label()} is on screen for only {b - a:.2f} s")
        out.append(Section(a, b, "full", j))
        t = b
    if t < duration - 0.05:
        out.append(Section(t, duration, "full"))
    elif out:
        out[-1].end = duration
    return out


def _captions(opts, clip, composer, sections, shapes, W, H, whole, whole_fr, hosted, notes) -> list:
    """The pack's captions laid over the film: anywhere clear of your face (frame), or kept to the
    seam between you and your content, or to your panel, in the sections that have one."""
    from ..core.view import View

    want = opts["captions"]
    spec = layers.parse("captions")
    zone = SafeZone(W, H).rect
    if want == "frame":
        L = layers.prepare(whole, spec, None, None, whole_fr, fix=opts["fix"])
        hosted.append({"where": "over", "effect": "captions", "checks": _short(L.job.checks)})
        return [L]
    out = []
    by_zone: dict = {}
    face = whole.face

    def seam_band(sh: Shape) -> Optional[Rect]:
        if sh.seam is None or sh.layout not in ("panel", "split") or not composer.grid.tall:
            return None
        mid = sh.seam.cy
        return Rect(zone.x0, max(zone.y0, mid - 0.19 * H), zone.x1, min(zone.y1, mid + 0.06 * H))

    for k, s in enumerate(sections):
        sh = shapes[k]
        room = None
        if s.layout != "full" and want == "seam":
            room = seam_band(sh)
        elif s.layout != "full" and want == "panel":
            sp = sh.speaker
            inset = 0.05 * sp.w
            room = _inter(Rect(sp.x0 + inset, sp.y0 + inset, sp.x1 - inset, sp.y1 - inset), zone)
            # the room under your face in your panel, through the section: captions need a tenth of the frame
            frames = range(int(s.start * composer.fps), min(composer.n, int(s.end * composer.fps)), 3)
            chins = [h.y1 for j in frames if (h := face.head(j)) is not None]
            free = (room.y1 - float(np.percentile(chins, 90))) if (room is not None and chins) else 0.0
            if room is None or free < 0.1 * H:
                room = seam_band(sh)
                notes.append(f"captions: your {s.layout} panel has no room under your face at "
                             f"{s.start:.1f}-{s.end:.1f} s, so they sit on the seam there"
                             if room is not None else
                             f"captions: your {s.layout} panel has no room under your face at "
                             f"{s.start:.1f}-{s.end:.1f} s, so they sit anywhere clear of it there")
            else:
                room = Rect(room.x0, max(room.y0, room.y1 - free), room.x1, room.y1)
        key = None if room is None else tuple(round(v) for v in room.as_list())
        if key not in by_zone:
            view = whole if room is None else View(clip, (W, H), place=composer.place, picture=whole.info.picture,
                                                   zone=room)
            by_zone[key] = view
        view = by_zone[key]
        try:
            L = layers.prepare(view, spec, s.start, s.end, whole_fr, fix=opts["fix"])
        except ValueError as e:
            raise O.OptionError(f"captions: {e}") from None
        out.append(L)
    hosted.append({"where": f"captions ({want})", "effect": "captions",
                   "checks": _shown_cards(out, whole.face, SafeZone(W, H).rect, composer.fps)})
    return out


def _shown_cards(laid: list, face, zone: Rect, fps: float) -> dict:
    """What the captions put on screen, checked frame by frame where each layer shows them: a layer
    kept to one section plans the whole clip's phrases, but only that section's are seen."""
    over_face = outside = 0
    where: dict[str, int] = {}
    for L in laid:
        a0, b0 = L.start, (L.end if L.end is not None else math.inf)
        for c in L.job.plan.get("cards") or []:
            a, b = max(a0, float(c["start"])), min(b0, float(c["end"]))
            if b <= a:
                continue
            r = Rect.of(c["rect"])
            told = c.get("where", "?")
            if told == "outside the safe zone" and r.inside(zone, slack=1.0):
                told = "out of its band, clear of your face"  # the captions' own last resort, still safe
            where[told] = where.get(told, 0) + 1
            for j in range(int(math.ceil(a * fps)), int(math.ceil(b * fps))):
                head = face.head(j)
                if head is not None and r.intersects(head):
                    over_face += 1
                if not r.inside(zone, slack=1.0):
                    outside += 1
    return {"frames_over_face": over_face, "frames_outside_safe_zone": outside, "placed": where}


def _inside(spec, opts, clip, composer, sections, shapes, zone, hosted, warnings) -> None:
    """An effect drawn inside the content panel of every section that has one, prepared on a view of
    that panel: no face there, and the panel's own safe part."""
    from ..core import framing as framing_mod
    from ..core.view import View

    by_size: dict = {}
    for k, s in enumerate(sections):
        sh = shapes[k]
        if s.item is None or sh.content is None:
            continue
        ct = sh.content
        size = (int(ct.w), int(ct.h))
        if size not in by_size:
            room = _inter(Rect(ct.x0, ct.y0, ct.x1, ct.y1), zone)
            if sh.layout == "pip":
                room = _card_room(sh, zone, composer.unit)
            room = (room or ct).shift(-ct.x0, -ct.y0)

            def picture(i: int, _size=size) -> np.ndarray:
                kk = int(composer.at_frame[int(np.clip(i, 0, composer.n - 1))])
                sec = composer.sections[kk]
                same = [j for j, x in enumerate(composer.sections) if x.item is not None
                        and composer.shapes[j].content is not None
                        and (int(composer.shapes[j].content.w), int(composer.shapes[j].content.h)) == _size]
                if kk not in same:  # this panel is not up then: the nearest moment it is
                    kk = min(same, key=lambda j: abs(j - kk))
                    sec = composer.sections[kk]
                    t = (sec.start + sec.end) / 2
                else:
                    t = i / composer.fps
                return composer.panel_picture(kk, _size, t)

            view = View(clip, size, place=None, picture=picture, zone=room)
            fr = framing_mod.plan(size[0], size[1], composer.fps, composer.n, None, shape="keep")
            L = _lay(view, spec, fr, opts["fix"], "inside", warnings)
            by_size[size] = L
            hosted.append({"where": f"inside the {s.layout} content panel", "effect": str(spec),
                           "checks": _short(L.job.checks)})
            # does it show while a content panel is up?
            shown_frames = [j for j in range(0, composer.n, max(1, int(composer.fps / 10)))
                            if layers.change_on(L, size, j, composer.fps).share > 0.0004]
            if shown_frames:
                up = [j for j in shown_frames if composer.sections[int(composer.at_frame[j])].item is not None
                      and composer.shapes[int(composer.at_frame[j])].content is not None
                      and (int(composer.shapes[int(composer.at_frame[j])].content.w),
                           int(composer.shapes[int(composer.at_frame[j])].content.h)) == size]
                if len(up) < 0.6 * len(shown_frames):
                    warnings.append(f"inside: {spec.name} shows mostly while you fill the frame alone, where there is "
                                    f"no content panel; give it a time inside a content section, or use full=none")
        composer.inside.setdefault((k, 0), []).append(by_size[size])


def _report(composer: Composer, framer: Framer, sections, shapes, items, W, H, hosted) -> dict:
    """Frame by frame: your forehead and chin inside your panel, your face above the bottom fifth
    when the layout puts you low, nothing of ours over your face, no bars, words inside the safe zone."""
    n = composer.n
    cut = low = face_hidden = 0
    for i in range(n):
        m = composer.moments[i]
        wx, wy, ww, wh = m.window
        keep = framer.keep(i)
        inside_src = keep.x0 >= 0 and keep.y0 >= 0 and keep.x1 <= framer.W0 and keep.y1 <= framer.H0
        if inside_src and not (keep.x0 >= wx - 1 and keep.y0 >= wy - 1 and keep.x1 <= wx + ww + 1
                               and keep.y1 <= wy + wh + 1):
            cut += 1
        if sections[m.k].layout != "full" and m.u >= 1:
            s = m.speaker.w / ww
            face_bottom = (framer.by[i] + framer.bh[i] - wy) * s + m.speaker.y0
            if face_bottom > 0.8 * H + 1:
                low += 1
        sec = sections[m.k]
        sh = shapes[m.k]
        if sh.layout == "pip" and sec.item is not None and composer.cards.get(m.k) is not None:
            card = composer.cards[m.k]
            tr = card.text_rect.shift(sh.content.x0, sh.content.y0)
            if tr.intersects(m.speaker):
                face_hidden += 1
    outside = 0
    zone = SafeZone(W, H).rect
    for k, c in composer.cards.items():
        sh = shapes[k]
        tr = c.text_rect.shift(sh.content.x0, sh.content.y0)
        if not tr.inside(zone, slack=1.0):
            outside += int(round((sections[k].end - sections[k].start) * composer.fps))
    over_face = sum(int(h["checks"].get("frames_over_face", 0) or 0) for h in hosted
                    if str(h["checks"].get("frames_over_face", "0")).isdigit())
    sp: dict = {}
    for sec, v in zip(sections, shapes):
        if sec.layout != "full":
            sp.setdefault(f"{sec.layout} you {int(v.speaker.w)}x{int(v.speaker.h)}"
                          + (f", content {int(v.content.w)}x{int(v.content.h)}" if v.content else ""), None)
    return {"frames_over_face": over_face, "frames_outside_safe_zone": outside,
            "frames_head_cut": cut, "frames_face_in_bottom_fifth": low, "frames_words_under_your_panel": face_hidden,
            "panels": " · ".join(sp),
            "hosted": " | ".join(f"{h['effect']} {h['where']}: {h['checks']}" for h in hosted) if hosted else "none"}
