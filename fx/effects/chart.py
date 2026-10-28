"""chart: a frosted glass card beside or above the head, where a line climbs (or bars rise) while the
big number counts up and lands as each value is said.

How a chart is made:

1. **The numbers.** `values` gives them, oldest first (`2000,15000,197000`, `2K,15K,197K`,
   `$2K,$197K`, `12%,3.5%`, and `Jan:2K` to name a point), or, left on auto, they are taken from what
   is said: "from 2K to 197K" is two points, "$2,000, then $15,000, then $197,000" three, "1.5M, up
   from 200K" two drawn oldest first. Years and lengths of time are never taken for amounts, and a
   small number said in passing ("two things") is not a chart. Nothing is rounded or worked out: each
   value lands written exactly as said or typed, shortened ("197,000" to "197K") only when the short
   form is the same number.
2. **When.** Each value lands on the word that says it, timed from when the voice begins (the
   transcriber marks a word late; the core pulls it back and checks it against the sound). Values
   that are given and not said land across the words `when` names, the last on its last word; with
   a time in seconds the card comes in at that moment (as every effect told a moment does, so it
   can be laid inside another) and its values land after it. The line draws, or the bar rises, in
   the moment before its value lands, and holds while the speaker talks on.
3. **Where.** The room above the head, beside it, or below the chin, measured over every frame the
   card is up and kept inside the safe zone. The card takes the shape that fills its room best (tall
   with the chart under the number, or wide with the chart beside it) at the largest size that
   fits. It never covers the face; when only a small card would clear the hair too, it may cross the
   top of the hair, never the face itself.
4. **The glass.** What is behind the card is blurred on every frame and bent at its rim, smoked just
   enough that white type on it reads at about 4.5 : 1 however bright the room, lit along its edge,
   with a soft shadow under it. `style=solid` is a plain dark card for a background too busy to see
   through.
5. **The motion and the sound.** The card frosts in; the number counts with the line's tip (or the
   rising bar) and settles on each value as it is said; when the last lands, an arrow shows which
   way it went and the last point pulses. A very soft tick runs under the count, well under the
   voice; `sound=off` leaves the clip's sound exactly as it was.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core import glass as glassmod
from ..core import moments
from ..core import amounts as num
from ..core import counting
from ..core import text as textmod
from ..core import words as wordsmod
from ..core.compose import Sprite, over
from ..core.layout import Rect, SafeZone, halo_room
from ..core.words import HESITATIONS, unheard
from . import options as O
from .captions import head_in_output
from .headline import _parse_time, find_said, keepout_in_output

NAME = "chart"
TYPES = ("line", "bars", "counter")
POSITIONS = ("auto", "above", "beside", "left", "right", "below")

OPTIONS = {
    "type": O.Choice(TYPES, "line: a line that climbs from point to point (the default) · bars: a bar rises "
                            "for each value · counter: the number and its label alone"),
    "values": O.Text("the numbers: auto takes the ones you say (\"from 2K to 197K\"); or give them oldest first, "
                     "such as 2000,15000,197000 or $2K,$15K,$197K or 12%,3.5%, and Jan:2K names a point"),
    "label": O.Text("the small line over the number (Followers, Monthly revenue): auto takes the words you say "
                    "with the number; empty for none"),
    "when": O.Text("auto: each value lands as it is said · words that are said: given values land across them, "
                   "the last on the last word · a time in seconds: the card comes in then · start · end"),
    "position": O.Choice(POSITIONS, "auto: the roomiest place clear of your face (above the head, beside it, "
                                    "or below the chin) · above · beside · left · right · below"),
    "accent": O.Colour("the line, the dots, the bars and the arrow: auto (green when it rises, red when it "
                       "falls), a name or a hex colour"),
    "prefix": O.Text("what stands before the number, such as $ or £: auto keeps what you said or typed"),
    "suffix": O.Text("what stands after it: auto keeps what you said or typed (K, M, %, x); K, M or B also "
                     "shortens the number, so 197000 shows as 197K"),
    "style": O.Choice(("glass", "solid"), "glass: frosted glass that blurs what is behind it · solid: a plain dark "
                                          "card, for a background too busy to see through"),
    "duration": O.Number(0, 30, "seconds on screen; 0 keeps it up for a couple of seconds after the last value "
                                "lands"),
    "sound": O.Switch("on lays a very soft tick under the count; off keeps the clip's sound exactly"),
    "fix": O.Text("misheard words as wrong=right, separated by ;  e.g. \"Jon=John; ask editor=AskEditor\""),
}

# The runs `tests/matrix.py chart` makes. None of the test clips says a number, so each run gives its
# values and lands them across words that are said there; finding numbers in speech is checked on
# fixed transcripts by `tests/amounts.py`.
MATRIX = [
    # the plain wall with room above the head: the line
    ("D", "D", {"values": "2K,15K,197K", "label": "Followers", "when": "the lively city come to life"}),
    # room above the head: bars, with a name under each
    ("B", "B", {"type": "bars", "values": "Mon:1.2K,Tue:3.4K,Wed:8.9K,Thu:21K", "label": "Views",
                "when": "improve your production quality"}),
    # a close shot with no room above: the counter goes below the chin
    ("A", "A", {"type": "counter", "values": "197K", "label": "Views", "when": "what draws people in"}),
    # trees and sky behind the glass, a currency sign
    ("C", "C", {"values": "$2K,$15K,$197K", "label": "Revenue", "when": "edit your videos professionally"}),
    # the raw take: the plain card, per cent
    ("33", "33", {"style": "solid", "values": "12%,31%,58%", "label": "Engagement",
                  "when": "Distribution is a real big issue"}),
]

ENTER = 0.42  # the card frosts in over this long
EXIT = 0.3
HOLD = 2.2  # it stays this long after the last value lands
SEG_SHARE = 0.8  # a stretch of line draws over this share of the time between two values
SEG_MIN, SEG_MAX = 0.32, 1.05
COUNT = 1.1  # a lone value counts up from nought over this long before it is said
STEP = 0.62  # between values that are not said
BAR_FIRST = 0.45  # the first bar rises over this long, once the card is in
TICK_UNDER = 20.0  # dB: how far under the voice's loud syllables a tick is heard
K_MAX = {"tall": 1.0, "wide": 1.0, "counter": 1.3}  # the largest each card is drawn (1: its design size)
K_MIN = 0.5  # smaller than this and the type is too small to read on a phone
K_COMFORT = 0.66  # clear of the whole head at this size or more, the hair is never crossed
GAP = 0.022  # between the card and the head, as a share of the frame's height
PREF = {"above": 1.0, "left": 1.0, "right": 1.0, "below": 0.86}  # how much each room is wanted
LAYOUT_PREF = {"tall": 1.0, "wide": 0.97, "counter": 1.0}
RISE_ACCENT = "#36e39b"
FALL_ACCENT = "#ff6b6b"
SOLID = (20, 21, 27)
FACE_NUM = "poppins-bold"
FACE_LABEL = "poppins-semibold"

# The card's design, in pixels of a frame 1080 wide at size 1 (`s` is that scale on the real frame).
PAD = 40
LABEL_PX = {"tall": 36, "wide": 34, "counter": 36}  # about 13 points on a phone: the smallest that reads
NUM_PX = {"tall": 132, "wide": 108, "counter": 140}
CHART_H = 172  # the tall card's chart
CHART_W = {"line": 380, "bars": 440}  # the wide card's chart
TALL_W = 520  # the tall card's inside, at least
AXIS_PX = 27  # the values on the bars and the names under the chart
RADIUS = 34


@dataclass
class Point:
    value: float
    at: float  # when it lands, in seconds
    label: str = ""  # the name given for it, shown under the chart
    said: str = ""  # the words that say it, when they are said


# ---- small drawing helpers --------------------------------------------------------------------------

def _canvas(w: float, h: float, k: int = 4):
    from PIL import Image

    return Image.new("RGBA", (max(1, int(round(w * k))), max(1, int(round(h * k)))), (0, 0, 0, 0))


def _down(img, w: float, h: float) -> Sprite:
    from PIL import Image

    img = img.resize((max(1, int(round(w))), max(1, int(round(h)))), Image.LANCZOS)
    return Sprite.from_rgba(np.asarray(img))


def _circle(d: float, colour, alpha: float = 1.0) -> Sprite:
    from PIL import ImageDraw

    img = _canvas(d, d)
    ImageDraw.Draw(img).ellipse((0, 0, img.width - 1, img.height - 1), fill=tuple(colour) + (int(255 * alpha),))
    return _down(img, d, d)


def _ring(d: float, width: float, colour, alpha: float = 1.0) -> Sprite:
    from PIL import ImageDraw

    img = _canvas(d, d)
    ImageDraw.Draw(img).ellipse((0, 0, img.width - 1, img.height - 1), outline=tuple(colour) + (int(255 * alpha),),
                                width=max(1, int(round(width * 4))))
    return _down(img, d, d)


def _glow(d: float, colour, alpha: float) -> Sprite:
    """A soft round glow, brightest in the middle."""
    n = max(3, int(round(d)))
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    r = np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2) / (n / 2)
    a = (np.clip(1.0 - r, 0, 1) ** 2.2 * alpha).astype(np.float32)
    rgb = np.empty((n, n, 3), np.float32)
    rgb[:] = np.asarray(colour, np.float32)
    return Sprite(rgb=rgb * a[..., None], a=a)


def _dot(d: float, accent) -> Sprite:
    """A point on the line: a white middle in a ring of the accent, with a little shadow."""
    from PIL import ImageDraw, ImageFilter

    k = 4
    D = d * 1.6
    img = _canvas(D, D, k)
    S = img.width
    c = S / 2
    shadow = _canvas(D, D, k)
    ImageDraw.Draw(shadow).ellipse((c - d * k / 2, c - d * k / 2 + k, c + d * k / 2, c + d * k / 2 + k), fill=(0, 0, 0, 90))
    shadow = shadow.filter(ImageFilter.GaussianBlur(d * k * 0.12))
    img.alpha_composite(shadow)
    d0 = ImageDraw.Draw(img)
    d0.ellipse((c - d * k / 2, c - d * k / 2, c + d * k / 2, c + d * k / 2), fill=tuple(accent) + (255,))
    inner = d * 0.5 * k / 2
    d0.ellipse((c - inner, c - inner, c + inner, c + inner), fill=(255, 255, 255, 255))
    return _down(img, D, D)


def _arrow_chip(d: float, accent, up: bool) -> Sprite:
    """A round chip in the accent with an arrow pointing the way the values went."""
    from PIL import ImageDraw

    k = 4
    img = _canvas(d, d, k)
    S = img.width
    dr = ImageDraw.Draw(img)
    dr.ellipse((0, 0, S - 1, S - 1), fill=tuple(accent) + (245,))
    lum = 0.2126 * accent[0] + 0.7152 * accent[1] + 0.0722 * accent[2]
    ink = (12, 22, 16, 255) if lum > 150 else (255, 255, 255, 255)
    lw = max(2, int(S * 0.11))
    top, bot = S * 0.24, S * 0.76
    if not up:
        top, bot = bot, top
    dr.line([(S * 0.5, bot), (S * 0.5, top)], fill=ink, width=lw)
    head = S * 0.2 * (1 if up else -1)
    dr.line([(S * 0.5 - S * 0.2, top + head), (S * 0.5, top), (S * 0.5 + S * 0.2, top + head)], fill=ink, width=lw,
            joint="curve")
    return _down(img, d, d)


def _ease_in_out(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * x * (x * (6 * x - 15) + 10)


def _ease_out(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 1.0 - (1.0 - x) ** 3


def _back_out(x: float, c: float = 1.6) -> float:
    x = min(1.0, max(0.0, x))
    return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2


# ---- the line's shape ---------------------------------------------------------------------------------

def _monotone(xs: np.ndarray, ys: np.ndarray, at: np.ndarray) -> np.ndarray:
    """A smooth curve through every point that never overshoots between them (so a line between
    two values never dips below the lower or rises past the higher)."""
    n = len(xs)
    h = np.diff(xs)
    d = np.diff(ys) / h
    m = np.zeros(n)
    m[0], m[-1] = d[0], d[-1]
    for i in range(1, n - 1):
        if d[i - 1] * d[i] <= 0:
            m[i] = 0.0
        else:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    out = np.empty_like(at)
    for j, x in enumerate(at):
        i = int(np.clip(np.searchsorted(xs, x) - 1, 0, n - 2))
        t = (x - xs[i]) / h[i]
        h00, h10 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t
        h01, h11 = -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
        out[j] = h00 * ys[i] + h10 * h[i] * m[i] + h01 * ys[i + 1] + h11 * h[i] * m[i + 1]
    return out


def _two_point(v0: float, v1: float, u: np.ndarray) -> np.ndarray:
    """The shape between two values and nothing else known: a rise that gathers pace, or a fall
    that eases off, as growth and decline usually go. Only the two ends are values."""
    k = 2.1
    if v1 >= v0:
        f = (np.exp(k * u) - 1) / (math.exp(k) - 1)
    else:
        f = (1 - np.exp(-k * u)) / (1 - math.exp(-k))
    return v0 + (v1 - v0) * f


# ---- the number --------------------------------------------------------------------------------------

# The big number is drawn by the core's `counting.Figures` (shared with number-counter).
Figures = counting.Figures


# ---- the chart -----------------------------------------------------------------------------------------

class Chart:
    def __init__(self, clip, opts: dict, size: tuple[int, int], fps: float, n_frames: int, head_at, face_at,
                 frame_at, plan: Optional[dict] = None):
        self.W, self.H = size
        self.U = self.W / 1080.0
        self.fps = fps
        self.n = n_frames
        self.seconds = n_frames / fps
        self.opts = opts
        self.zone = SafeZone(self.W, self.H)
        self.heads = [head_at(j) for j in range(n_frames)]
        self.faces = [face_at(j) for j in range(n_frames)]
        self.frame_at = frame_at
        self.warnings: list[str] = []
        self.kind = opts["type"]
        if plan is not None:
            self._from_plan(plan)
        else:
            self._numbers(clip)
        self._shape()
        self._schedule()
        self._colours()
        self._place(forced=plan.get("chart", {}).get("where") if plan else None)
        self._build()

    # -- what the numbers are --------------------------------------------------------------------------
    def _first_listen(self, clip) -> tuple[list, dict[int, float]]:
        """The transcriber's own words and their corrected starts (`counting.first_listen`): amounts
        are read from the first listen, never from words a second listen changed."""
        return counting.first_listen(clip)

    def _start_of(self, a: num.Amount, heard, corrected: dict[int, float]) -> float:
        return counting.start_of(a, heard, corrected)

    def _numbers(self, clip) -> None:
        opts = self.opts
        when = opts["when"].strip()
        wl = when.lower()
        words = [w for w in clip.words if w.key not in HESITATIONS]
        heard, corrected = self._first_listen(clip)
        # a figure the speaker took back ("ten, sorry, twelve dollars"), said in a first attempt or in
        # an aside is never charted: the shared picker's filter (`moments.taken_back`)
        back = moments.taken_back(heard, clip.quiet)
        said = [a for a in num.spoken(heard) if not back.holds(heard, a.first, a.last)]
        self.span: Optional[tuple[int, int]] = None
        self.at_time: Optional[float] = None
        near = None
        if wl not in ("", "auto", "start", "end"):
            t = _parse_time(when)
            if t is not None:
                if t >= self.seconds:
                    raise O.OptionError(f"when: {t:g} s is after the end of the video ({self.seconds:.2f} s)")
                self.at_time = near = t
            else:
                hit = find_said(words, when)
                if not hit:
                    raise O.OptionError(f"when: the words \"{when}\" are not said in this video. Read the words with "
                                        f"`python -m fx.run words <video>` and use them as heard, or give a time in "
                                        f"seconds")
                self.span = (hit[0], hit[1])
                near = words[hit[0]].start
        self.words = words
        given = opts["values"].strip()
        label = None
        if given.lower() in ("", "auto"):
            run = num.series(said, heard, near=near)
            if run is None:
                raise O.OptionError("no amounts are said in this video that a chart could show (years, lengths of time "
                                    "and small counts are passed over). Give them, oldest first: "
                                    "values=2000,15000,197000, and when= to say where they land")
            amounts = run.amounts
            times = [self._start_of(a, heard, corrected) for a in amounts]
            if run.reversed:  # "1.5M, up from 200K": the whole line is drawn as the first is said
                land = min(times)
                times = [land - (len(amounts) - 1 - k) * STEP for k in range(len(amounts))]
            self.points = [Point(a.value, t, said=a.said) for a, t in zip(amounts, times)]
            self.amounts = amounts
            self._settle_kind()
            amounts = self.amounts
            label = num.label_for(run, heard)
            said_text = " … ".join(f"“{a.said}”" for a in sorted(amounts, key=lambda a: a.first))
            self.found = f"said: {said_text}" + (" (said newest first; drawn oldest first)" if run.reversed else "")
            self.said_all = True
        else:
            try:
                amounts = num.parse_list(given)
            except ValueError as e:
                raise O.OptionError(f"values: {e}") from None
            self.points = [Point(a.value, 0.0, label=a.label) for a in amounts]
            self.amounts = amounts
            self._settle_kind()
            amounts = self.amounts
            self._land_given(said, heard, corrected)
            label = self.spoken_label if self.said_all else None
        self.amounts = amounts
        lab = opts["label"].strip()
        if lab.lower() == "auto":  # the words said with the numbers, with a capital to start the card
            lab = label or ""
            lab = lab[:1].upper() + lab[1:]
        self.label = lab  # a label given is kept exactly as written ("iPhone sales")

    def _land_given(self, said: list[num.Amount], heard, corrected) -> None:
        """When given values land: on the words that say them, if they are said; otherwise across the
        words `when` names, at a time, or at the start."""
        pts = self.points
        n = len(pts)
        # are they said? each value matched to an amount said with the same value, in order
        cands = [a for a in said if a.kind == "amount"]
        matched: list[Optional[float]] = [None] * n
        heard_as: list[num.Amount] = []
        after = -1.0
        for k, p in enumerate(pts):
            for a in cands:
                t = self._start_of(a, heard, corrected)
                if t > after and abs(a.value - p.value) <= 1e-6 * max(1.0, abs(p.value)):
                    matched[k] = t
                    p.said = a.said
                    heard_as.append(a)
                    after = t
                    break
        self.spoken_label = num.label_for(num.Series(amounts=heard_as), heard) if heard_as else ""
        if matched[-1] is not None and (self.span is None and self.at_time is None):
            last = matched[-1]
            prev = last
            for k in range(n - 1, -1, -1):
                if matched[k] is None or matched[k] > prev:
                    matched[k] = prev - STEP
                prev = matched[k]
            for p, t in zip(pts, matched):
                p.at = t
            self.found = "where each value is said" if all(p.said for p in pts) else \
                f"landing on “{pts[-1].said}”, where the last value is said"
            self.said_all = all(p.said for p in pts)
            return
        self.said_all = False
        step = STEP if self.kind != "counter" else STEP * 0.9
        if self.span is not None:
            i0, i1 = self.span
            starts = [w.start for w in self.words[i0:i1 + 1]]
            if n == 1:
                times = [starts[-1]]
            elif len(starts) >= n:
                times = [starts[round(k * (len(starts) - 1) / (n - 1))] for k in range(n)]
            else:
                times = [starts[0] + (starts[-1] - starts[0]) * k / (n - 1) for k in range(n)]
            for k in range(n - 2, -1, -1):  # never closer together than a stretch of line can draw
                times[k] = min(times[k], times[k + 1] - SEG_MIN * 1.25)
            said = " ".join(w.clean for w in self.words[i0:i1 + 1])
            self.found = f"across the words “{said}”, the last value on “{self.words[i1].clean}”"
        elif self.at_time is not None:  # the card comes in at that moment, and its values land after it
            times = [self.at_time + self._lead() + k * step for k in range(n)]
            self.found = f"coming in at {self.at_time:.2f} s"
        else:
            wl = self.opts["when"].strip().lower()
            if wl == "end":
                last = self.seconds - HOLD - 0.2
                times = [last - (n - 1 - k) * step for k in range(n)]
                self.found = "near the end"
            else:
                first = self.words[0].start if self.words else 0.0
                times = [first + self._lead() + k * step for k in range(n)]
                self.found = "at the start"
                if wl in ("", "auto"):
                    self.warnings.append("the values are not said in this video, so the chart lands at the start; "
                                         "when=\"…\" or when=<seconds> moves it")
        for p, t in zip(pts, times):
            p.at = t

    def _from_plan(self, plan: dict) -> None:
        c = plan.get("chart") or {}
        try:
            self.points = [Point(float(p["value"]), float(p["at"]), str(p.get("label", "")), str(p.get("said", "")))
                           for p in c["points"]]
        except (KeyError, TypeError, ValueError):
            raise ValueError("the plan has no chart points; run once without --plan to get one to edit") from None
        if not self.points:
            raise ValueError("the plan's chart has no points")
        self.points.sort(key=lambda p: p.at)
        self.label = str(c.get("label", ""))
        self.kind = str(c.get("type", self.kind))
        if self.kind not in TYPES:
            raise ValueError(f"the plan's chart type must be one of {', '.join(TYPES)}")
        self.amounts = [num.Amount(p.value) for p in self.points]
        self._settle_kind()
        unit = c.get("unit") or {}
        self.plan_unit = num.Unit(prefix=str(unit.get("prefix", "")), suffix=str(unit.get("suffix", "")),
                                  scale=str(unit.get("scale", "")), decimals=int(unit.get("decimals", 0)))
        self.found = "from the plan"
        self.said_all = all(p.said for p in self.points)
        self.span = None
        self.at_time = None
        self.plan_times = (c.get("start"), c.get("end"))

    # -- the chart's kind and unit ------------------------------------------------------------------------
    def _settle_kind(self) -> None:
        """At most eight values, and one value is a counter: a line or bars need two."""
        n = len(self.points)
        if n > 8:
            self.warnings.append(f"{n} values is more than a card can show clearly; the first {n - 8} were left out")
            self.points = self.points[-8:]
            self.amounts = self.amounts[-8:]
            n = 8
        if self.kind in ("line", "bars") and n == 1:
            self.warnings.append(f"one value is not a {'line' if self.kind == 'line' else 'set of bars'}: it is shown "
                                 f"as a counter (give two or more values for a {self.kind})")
            self.kind = "counter"

    def _lead(self) -> float:
        """How long before its first value lands the card comes in: it frosts in first, and a lone
        count or the first bar runs up from nought before its value."""
        n = len(self.points)
        if self.kind == "counter" and n == 1:
            return COUNT + ENTER * 0.6
        if self.kind == "bars":
            return BAR_FIRST + ENTER * 0.45
        return ENTER

    def _shape(self) -> None:
        n = len(self.points)
        if self.kind == "bars" and any(p.value < 0 for p in self.points):
            raise O.OptionError("bars start at nought, so they cannot show a value below it; use type=line")
        if hasattr(self, "plan_unit"):
            self.unit = self.plan_unit
        else:
            pre = self.opts["prefix"].strip()
            suf = self.opts["suffix"]
            pre = None if pre.lower() == "auto" else pre
            suf = None if suf.strip().lower() == "auto" else suf
            if pre is not None and len(pre) > 3:
                raise O.OptionError("prefix: keep it to a sign of up to three characters, such as $ or £")
            if suf is not None and len(suf.strip()) > 8:
                raise O.OptionError("suffix: keep it short (K, %, x, pts); say what is counted in label=")
            amounts = self.amounts
            if suf is not None and suf.strip().upper() in num.SCALE_LETTERS:
                mult = num.SCALE_LETTERS[suf.strip().upper()]
                if max(abs(p.value) for p in self.points) < mult:  # given already in that scale: 2,15,197 with K
                    for p in self.points:
                        p.value *= mult
                    amounts = [num.Amount(p.value) for p in self.points]
            self.unit = num.Unit.of(amounts, prefix=pre, suffix=suf)
        # how each value is written where it lands
        self.texts = [self.unit.exact(p.value) for p in self.points]
        self.start_value = 0.0 if (self.kind == "bars" or (self.kind == "counter" and n == 1)) else self.points[0].value

    # -- when everything happens ----------------------------------------------------------------------------
    def _schedule(self) -> None:
        pts = self.points
        n = len(pts)
        for k in range(1, n):  # in order, never two on one instant
            pts[k].at = max(pts[k].at, pts[k - 1].at + 0.2)
        # Values asked to land in the first moments of the video: the earlier ones are spaced into the
        # time there is, or, when there is too little, the build starts a little later than asked.
        soonest = 0.6 if n == 1 else 0.3
        if pts[0].at < soonest:
            last = pts[-1].at
            if n > 1 and last - soonest >= (n - 1) * SEG_MIN:
                for k in range(n - 1):
                    pts[k].at = soonest + (last - soonest) * k / (n - 1)
            else:
                shift = soonest - pts[0].at
                for p in pts:
                    p.at += shift
                if shift > 0.1:
                    self.warnings.append(f"the video starts too soon for the build to come first; the last value "
                                         f"lands {shift:.1f} s later than asked")
        lead = self._lead()
        if self.kind == "counter" and n == 1:
            count = min(COUNT, max(0.45, pts[0].at - 0.2))
            self.segments = [(pts[0].at - count, pts[0].at, 0.0, pts[0].value, -1, 0)]
            lead = count + ENTER * 0.6
        else:
            self.segments = []
            if self.kind == "bars":
                rise = min(BAR_FIRST, max(0.25, pts[0].at - 0.25))
                self.segments.append((pts[0].at - rise, pts[0].at, 0.0, pts[0].value, -1, 0))
                lead = rise + ENTER * 0.45  # the first bar starts up as the card is half in
            for k in range(1, n):
                gap = pts[k].at - pts[k - 1].at
                d = gap if gap < SEG_MIN else min(SEG_MAX, max(SEG_MIN, SEG_SHARE * gap))
                self.segments.append((pts[k].at - d, pts[k].at, pts[k - 1].value, pts[k].value, k - 1, k))
        self.start = max(0.0, pts[0].at - lead)
        self.enter = max(0.2, min(ENTER, pts[0].at - self.start - 0.05)) if self.start > 0 else min(ENTER, 0.25)
        self.land = pts[-1].at
        dur = float(self.opts["duration"])
        end = self.start + dur if dur > 0 else self.land + HOLD
        if dur > 0 and end < self.land + 0.6:
            self.warnings.append(f"duration={dur:g} would end the card before its last value lands; it stays until "
                                 f"{self.land + 0.6 - self.start:.1f} s")
            end = self.land + 0.6
        if getattr(self, "plan_times", None) and self.plan_times[1] is not None:
            end = float(self.plan_times[1])
        self.end = min(self.seconds, end)
        if self.seconds - self.end < 0.4:
            self.end = self.seconds
        if self.land > self.seconds - 0.05:
            raise O.OptionError(f"the last value would land at {self.land:.2f} s, after the video ends "
                                f"({self.seconds:.2f} s); give an earlier when=")
        if self.land - self.start < 0.3:
            self.warnings.append("the first value is said in the first moment of the video, so the card arrives "
                                 "with it, without frosting in")

    def _colours(self) -> None:
        a = self.opts["accent"]
        rising = self.points[-1].value >= self.points[0].value if len(self.points) > 1 else True
        self.rising = rising
        self.accent = textmod.colour((RISE_ACCENT if rising else FALL_ACCENT) if a == "auto" else a)

    # -- where it goes, and how big -------------------------------------------------------------------------
    def _span_box(self, boxes: list) -> Optional[Rect]:
        a = int(max(0, math.floor(self.start * self.fps)))
        b = int(min(self.n - 1, math.ceil(self.end * self.fps)))
        return Rect.around(r for r in boxes[a:b + 1] if r is not None)

    def _layouts(self) -> list[str]:
        return ["counter"] if self.kind == "counter" else ["tall", "wide"]

    def _rooms(self, box: Optional[Rect]) -> dict[str, Rect]:
        z = self.zone.rect
        gap = GAP * self.H
        if box is None:  # no face found: the band above where a head would be
            return {"above": Rect(z.x0, z.y0, z.x1, z.y0 + 0.36 * self.H)}
        rooms = {"above": Rect(z.x0, z.y0, z.x1, min(z.y1, box.y0 - gap)),
                 "below": Rect(z.x0, max(z.y0, box.y1 + gap), z.x1, z.y1),
                 "left": Rect(z.x0, z.y0, max(z.x0, box.x0 - gap), z.y1),
                 "right": Rect(min(z.x1, box.x1 + gap), z.y0, z.x1, z.y1)}
        return {k: r for k, r in rooms.items() if r.w > 8 and r.h > 8}

    def _fits(self, rooms: dict[str, Rect], allowed: list[str]) -> list[tuple[float, float, str, str, Rect]]:
        """Every (score, size, room, layout, room box) that fits, best first."""
        out = []
        for name, room in rooms.items():
            if name not in allowed:
                continue
            for lay in self._layouts():
                w1, h1 = self._size(lay, 1.0)
                # the card and its shadow, kept inside the safe zone (at full size: the size found
                # is then made to fit, below)
                inner = halo_room(room, self.zone.rect, glassmod.Pane.reach(self.U))
                k = float(min(inner.w / w1, inner.h / h1, K_MAX[lay]))
                out.append((k * PREF[name] * LAYOUT_PREF[lay], k, name, lay, room))
        out.sort(key=lambda f: -f[0])
        return out

    def _place(self, forced: Optional[str] = None) -> None:
        pos = forced or self.opts["position"]
        allowed = {"auto": ["above", "left", "right", "below"], "beside": ["left", "right"]}.get(pos, [pos])
        heads, faces = self._span_box(self.heads), self._span_box(self.faces)
        best = None
        for attempt in (allowed, ["above", "left", "right", "below"]):
            whole = self._fits(self._rooms(heads), attempt)
            hair = self._fits(self._rooms(faces), attempt) if faces is not None else []
            pick, clearance = (whole[0], "head") if whole else (None, "head")
            # clear of the whole head unless that leaves only a small card and crossing the top of
            # the hair (never the face) gives a much larger one
            if hair and (pick is None or (pick[1] < K_COMFORT and hair[0][1] > 1.25 * pick[1])):
                pick, clearance = hair[0], "face"
            if pick is not None and pick[1] >= K_MIN:
                best = (pick, clearance)
                break
            if attempt is allowed and pos != "auto":
                self.warnings.append(f"there is no room for the card {pos} clear of the face; it went where there is")
        self.outside = False
        if best is None:  # no room clear of the face inside the safe zone: the roomiest margin, smaller
            fr = Rect(0.03 * self.W, 0.015 * self.H, 0.97 * self.W, 0.985 * self.H)
            box = faces or heads
            rooms = {}
            if box is not None:
                rooms = {"above": Rect(fr.x0, fr.y0, fr.x1, box.y0 - 0.01 * self.H),
                         "below": Rect(fr.x0, box.y1 + 0.01 * self.H, fr.x1, fr.y1)}
            fits = self._fits({k: r for k, r in rooms.items() if r.h > 8}, ["above", "below"]) if rooms else []
            if not fits:
                raise O.OptionError("the face fills the frame, so there is nowhere to put the card without covering it")
            pick = fits[0]
            best = (pick, "face")
            self.outside = True
            self.warnings.append("there was no room for the card inside the safe zone clear of the face; it sits in "
                                 "the margin, smaller")
        (_, k, name, lay, room), clearance = best
        self.where, self.layout, self.clearance = name, lay, clearance
        self.k = float(k)
        self.room = room
        self.s = float(k * self.U)
        box = (faces if clearance == "face" else heads)
        self._measure(lay, self.s)
        # the real size, set in whole pixels, can come out a little larger than the estimate; the card
        # goes where its shadow stays inside the room as well
        zone = self.zone.rect if not self.outside else room
        inner = halo_room(room, zone, glassmod.Pane.reach(self.s))
        for _ in range(12):
            if self.cw <= inner.w + 0.5 and self.ch <= inner.h + 0.5:
                break
            self.k *= 0.97
            self.s = self.k * self.U
            self._measure(lay, self.s)
            inner = halo_room(room, zone, glassmod.Pane.reach(self.s))
        if self.k < K_MIN and not self.outside:
            self.warnings.append(f"the card is small ({self.k:.0%} of its size): there is little room clear of the face")
        self._position(name, inner, box)

    def _position(self, name: str, room: Rect, box: Optional[Rect]) -> None:
        w, h = self.cw, self.ch
        cx = box.cx if box is not None else room.cx
        if name in ("above", "below"):
            x = min(max(cx - w / 2, room.x0), room.x1 - w)
            slack = room.h - h
            y = room.y1 - h - slack * 0.38 if name == "above" else room.y0 + slack * 0.3
        else:
            x = room.x0 + (room.w - w) / 2
            cy = box.y0 + 0.42 * box.h if box is not None else room.cy
            y = min(max(cy - h / 2, room.y0), room.y1 - h)
        # whole pixels, never rounded out of the room
        self.x = int(min(max(round(x), math.ceil(room.x0)), max(math.ceil(room.x0), math.floor(room.x1 - w))))
        self.y = int(min(max(round(y), math.ceil(room.y0)), max(math.ceil(room.y0), math.floor(room.y1 - h))))

    @property
    def rect(self) -> Rect:
        return Rect(self.x, self.y, self.x + self.cw, self.y + self.ch)

    # -- the card's measurements ------------------------------------------------------------------------
    def _count_texts(self) -> list[tuple[str, str, str, str]]:
        """Every way the number is written while it counts and lands (the card is as wide as the widest)."""
        out = list(self.texts) + [self.unit.exact(self.start_value)]
        for (a, b, v0, v1, _, _) in self.segments:
            for q in np.linspace(0, 1, 24):
                out.append(self.unit.during(v0 + (v1 - v0) * q, v1))
        return out

    def _size(self, lay: str, k: float) -> tuple[float, float]:
        self._measure(lay, k * self.U, keep=False)
        return self._m_w, self._m_h

    def _measure(self, lay: str, s: float, keep: bool = True) -> None:
        """The card's size and where its parts sit, for layout `lay` at scale `s`."""
        s = float(s)
        num_px = max(10, int(round(NUM_PX[lay] * s)))
        label_px = max(8, int(round(LABEL_PX[lay] * s)))
        figs = Figures(num_px, s)
        num_w = max(figs.width(p) for p in self._count_texts())
        chip_d = (0.5 * figs.cap) if (len(self.points) > 1 and self.points[-1].value != self.points[0].value) else 0.0
        chip_gap = 0.22 * figs.cap if chip_d else 0.0
        label_w = textmod.advance(self.label, FACE_LABEL, label_px) if self.label else 0.0
        label_cap = textmod.cap_height(FACE_LABEL, label_px)
        pad = PAD * s
        head_w = max(label_w, num_w + chip_gap + chip_d)
        label_block = (label_cap + 0.62 * label_px) if self.label else 0.0
        axis = any(p.label for p in self.points) and lay != "counter"
        axis_h = (AXIS_PX * 1.55 * s) if axis else 0.0
        m = {"num_px": num_px, "label_px": label_px, "chip_d": chip_d, "chip_gap": chip_gap, "label_cap": label_cap,
             "pad": pad, "axis_h": axis_h, "num_w": num_w}
        if lay == "tall":
            w = max(TALL_W * s, head_w) + 2 * pad
            y_label = pad + label_cap
            y_num = (y_label + 0.62 * label_px if self.label else pad) + figs.cap
            chart_top = y_num + 30 * s
            h = chart_top + CHART_H * s + axis_h + 0.85 * pad
            m.update(chart=(pad, chart_top, w - pad, chart_top + CHART_H * s), y_label=y_label, y_num=y_num,
                     x_text=pad)
        elif lay == "wide":
            text_h = label_block + figs.cap
            h = max(2 * pad + 158 * s + axis_h, text_h + 2 * pad)
            y0 = (h - axis_h - text_h) / 2
            y_label = y0 + label_cap
            y_num = y0 + label_block + figs.cap
            chart_x0 = pad + head_w + 46 * s
            w = chart_x0 + CHART_W.get(self.kind, CHART_W["line"]) * s + pad
            m.update(chart=(chart_x0, pad * 0.95, w - pad, h - pad * 0.95 - axis_h), y_label=y_label, y_num=y_num,
                     x_text=pad)
        else:
            w = head_w + 2 * pad
            y_label = pad + label_cap
            y_num = (y_label + 0.62 * label_px if self.label else pad) + figs.cap
            h = y_num + 1.02 * pad
            m.update(chart=None, y_label=y_label, y_num=y_num, x_text=pad)
        self._m_w, self._m_h = math.ceil(w), math.ceil(h)
        if keep:
            self.cw, self.ch = int(math.ceil(w)), int(math.ceil(h))
            self.m = m
            self.figs = figs

    # -- the pieces, drawn once --------------------------------------------------------------------------
    def _build(self) -> None:
        s = self.s
        m = self.m
        style = self.opts["style"] if "style" in self.opts else "glass"
        self.pane = glassmod.Pane(self.cw, self.ch, radius=RADIUS * s, unit=s, frosted=(style == "glass"),
                                  fill=SOLID)
        frames = []
        if style == "glass":  # the picture behind the card at its start, middle and end
            for t in (self.start + 0.1, (self.start + self.end) / 2, self.end - 0.1):
                frames.append(self.frame_at(int(min(self.n - 1, max(0, round(t * self.fps))))))
        brightest = self.pane.tone_over(frames, self.x, self.y)
        self.contrast = glassmod.contrast_with_white(brightest)
        shadow = (0.0, 2.0 * s, 4.0 * s, 0.32)
        self.label_sprite = None
        if self.label:
            wd = textmod.draw_word(self.label, FACE_LABEL, m["label_px"], fill=(232, 234, 240), shadow=shadow)
            self.label_sprite = wd
        self.chip = _arrow_chip(m["chip_d"], self.accent, self.rising) if m["chip_d"] else None
        if self.kind == "line":
            self._build_line()
        elif self.kind == "bars":
            self._build_bars()
        self._build_axis()

    def _xs(self) -> list[float]:
        x0, y0, x1, y1 = self.m["chart"]
        n = len(self.points)
        inset = 16 * self.s
        if n == 1:
            return [(x0 + x1) / 2]
        return [x0 + inset + k * (x1 - x0 - 2 * inset) / (n - 1) for k in range(n)]

    def _build_line(self) -> None:
        from PIL import Image, ImageDraw, ImageFilter

        s = self.s
        x0, y0, x1, y1 = self.m["chart"]
        vals = np.array([p.value for p in self.points], np.float64)
        lo, hi = float(vals.min()), float(vals.max())
        if hi - lo < 1e-9:
            lo, hi = hi - 1.0, hi + 1.0
        top, bottom = y0 + 20 * s, y1 - 12 * s
        self.y_of = lambda v: bottom - (v - lo) / (hi - lo) * (bottom - top)
        xs = np.array(self._xs(), np.float64)
        self.px_x = xs
        # the curve, sampled finely, as values along x
        samples = np.linspace(xs[0], xs[-1], 220)
        if len(xs) == 2:
            curve = _two_point(vals[0], vals[1], (samples - xs[0]) / (xs[-1] - xs[0]))
        else:
            curve = _monotone(xs, vals, samples)
        self.curve_x, self.curve_v = samples, curve
        cy = np.array([self.y_of(v) for v in curve])
        self.curve_y = cy
        # the whole drawn line, once: its glow, the fill under it, the line itself
        ox, oy = int(math.floor(x0 - 14 * s)), int(math.floor(y0 - 14 * s))
        W, H = int(math.ceil(x1 + 14 * s)) - ox, int(math.ceil(y1 + 14 * s)) - oy
        k = 3
        pts = [((x - ox) * k, (y - oy) * k) for x, y in zip(samples, cy)]
        acc = tuple(self.accent)
        # the fill: the accent, strongest just under the line and gone by the floor
        poly = Image.new("L", (W * k, H * k), 0)
        ImageDraw.Draw(poly).polygon(pts + [(pts[-1][0], (y1 - oy) * k), (pts[0][0], (y1 - oy) * k)], fill=255)
        poly = np.asarray(poly.resize((W, H), Image.LANCZOS), np.float32) / 255.0
        col_y = np.interp(np.arange(W) + ox + 0.5, samples, cy)
        rows = (np.arange(H, dtype=np.float32)[:, None] + oy + 0.5)
        depth = np.clip((rows - col_y[None, :]) / np.maximum(1.0, (y1 - col_y[None, :])), 0, 1)
        fill_a = poly * 0.34 * (1.0 - depth) ** 1.5
        # the glow: the line drawn wide and blurred
        glow = Image.new("L", (W * k, H * k), 0)
        ImageDraw.Draw(glow).line(pts, fill=255, width=int(round(12 * s * k)), joint="curve")
        glow = glow.resize((W, H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(7 * s))
        glow_a = np.asarray(glow, np.float32) / 255.0 * 0.5
        # the line
        line = Image.new("L", (W * k, H * k), 0)
        lw = max(2, int(round(6.2 * s * k)))
        ImageDraw.Draw(line).line(pts, fill=255, width=lw, joint="curve")
        r = lw / 2
        for px, py in (pts[0], pts[-1]):  # round ends
            ImageDraw.Draw(line).ellipse((px - r, py - r, px + r, py + r), fill=255)
        line_a = np.asarray(line.resize((W, H), Image.LANCZOS), np.float32) / 255.0
        a = 1.0 - (1.0 - fill_a) * (1.0 - glow_a) * (1.0 - line_a)
        light = np.asarray(acc, np.float32)
        # the line itself a touch lighter than its glow and fill, so it reads as the edge
        lit = light + (255.0 - light) * 0.18
        rgb = light[None, None, :] * (a - line_a)[..., None] + lit[None, None, :] * line_a[..., None]
        self.line_sprite = Sprite(rgb=rgb.astype(np.float32), a=a.astype(np.float32))
        self.line_at = (ox, oy)
        self.dot = _dot(17 * s, acc)
        self.tip = _glow(56 * s, acc, 0.75)
        self.tip_core = _circle(10 * s, (255, 255, 255))
        self.pulse_d = 17 * s
        # the faint rules behind it
        rules = _canvas(x1 - x0, y1 - y0, 2)
        dr = ImageDraw.Draw(rules)
        for q in (0.0, 1 / 3, 2 / 3):
            yy = (top + (bottom - top) * q - y0) * 2
            dr.line([(0, yy), (rules.width, yy)], fill=(255, 255, 255, 22), width=max(1, int(round(1.2 * s * 2))))
        yy = (bottom + 6 * s - y0) * 2
        dr.line([(0, yy), (rules.width, yy)], fill=(255, 255, 255, 40), width=max(1, int(round(1.4 * s * 2))))
        self.rules = _down(rules, x1 - x0, y1 - y0)

    def _build_bars(self) -> None:
        from PIL import ImageDraw

        s = self.s
        x0, y0, x1, y1 = self.m["chart"]
        n = len(self.points)
        slot = (x1 - x0) / n
        bw = min(slot * 0.64, 92 * s)
        self.bar_w = bw
        self.bar_x = [x0 + slot * (k + 0.5) - bw / 2 for k in range(n)]
        vmax = max(p.value for p in self.points) or 1.0
        # a tag over each bar, when the tags fit their slots
        tag_px = max(8, int(round(AXIS_PX * 1.05 * s)))
        tags = [num.Unit.join(t) for t in self.texts]
        widths = [textmod.advance(t, FACE_LABEL, tag_px) for t in tags]
        self.tags_fit = max(widths) <= slot * 0.98
        room = (y1 - y0) - (tag_px * 1.5 if self.tags_fit else 6 * s)
        self.base = y1 - 4 * s
        self.bar_h = [max(2 * s, p.value / vmax * room) for p in self.points]
        r = bw * 0.2
        self.bar_r = r
        hmax = max(self.bar_h) * 1.12 + r

        def bar(colour, alpha: float, top_light: float) -> Sprite:
            img = _canvas(bw, hmax, 4)
            dr = ImageDraw.Draw(img)
            dr.rounded_rectangle((0, 0, img.width - 1, img.height + r * 8), radius=r * 4,
                                 fill=tuple(colour) + (int(255 * alpha),))
            sp = _down(img, bw, hmax)
            if top_light > 0:
                yy = np.arange(sp.h, dtype=np.float32)[:, None]
                lift = top_light * np.clip(1.0 - yy / max(1.0, r), 0, 1) ** 1.5  # within the rounded top only
                sp = Sprite(rgb=sp.rgb + (255.0 * sp.a[..., None] - sp.rgb) * lift[..., None], a=sp.a)
            return sp

        self.bar_on = bar(self.accent, 1.0, 0.22)
        self.bar_off = bar((255, 255, 255), 0.3, 0.12)
        self.tags = []
        if self.tags_fit:
            for t in tags:
                self.tags.append(textmod.draw_word(t, FACE_LABEL, tag_px, fill=(236, 238, 244),
                                                   shadow=(0.0, 1.5 * s, 3.0 * s, 0.3)))
        from PIL import ImageDraw as _D

        rules = _canvas(x1 - x0, y1 - y0, 2)
        _D.Draw(rules).line([(0, (self.base + 2 * s - y0) * 2), (rules.width, (self.base + 2 * s - y0) * 2)],
                            fill=(255, 255, 255, 46), width=max(1, int(round(1.4 * s * 2))))
        self.rules = _down(rules, x1 - x0, y1 - y0)

    def _build_axis(self) -> None:
        self.axis = []
        if self.kind == "counter" or not any(p.label for p in self.points):
            return
        s = self.s
        px = max(8, int(round(AXIS_PX * s)))
        xs = self.bar_x if self.kind == "bars" else self._xs()
        if self.kind == "bars":
            xs = [x + self.bar_w / 2 for x in xs]
        words = [textmod.draw_word(p.label or " ", FACE_LABEL, px, fill=(205, 208, 216)) for p in self.points]
        x0, y0, x1, y1 = self.m["chart"]
        widths = [w.width for w in words]
        crowded = any(xs[k + 1] - xs[k] < (widths[k] + widths[k + 1]) / 2 + 8 * s for k in range(len(xs) - 1))
        keep = [0, len(xs) - 1] if crowded else range(len(xs))
        y = y1 + self.m["axis_h"] * 0.78
        for k in keep:
            if not self.points[k].label:
                continue
            x = min(max(xs[k] - widths[k] / 2, x0), x1 - widths[k])
            self.axis.append((words[k], x, y))

    # -- the moment ---------------------------------------------------------------------------------------
    def _ease(self):
        """How a count moves: a line's tip speeds up and settles on each point; a bar shoots up and
        settles, and a lone count runs fast and slows into its value."""
        return _ease_in_out if self.kind == "line" else _ease_out

    def _progress(self, t: float):
        """Where the build is at `t`: (the value shown, its parts, the index of the stretch counting or
        None, that stretch's progress, how many points have landed)."""
        landed = sum(1 for p in self.points if t >= p.at)
        for (a, b, v0, v1, i0, i1) in self.segments:
            if a <= t < b:
                q = (t - a) / (b - a)
                if self.kind == "line":
                    e = _ease_in_out(q)
                    x = self.px_x[i0] + (self.px_x[i1] - self.px_x[i0]) * e
                    v = float(np.interp(x, self.curve_x, self.curve_v))
                    return v, self.unit.during(v, v1), (i0, i1), e, landed
                e = self._ease()(q)
                v = v0 + (v1 - v0) * e
                return v, self.unit.during(v, v1), (i0, i1), e, landed
        if landed == 0:
            return self.start_value, self.unit.exact(self.start_value), None, 0.0, 0
        return self.points[landed - 1].value, self.texts[landed - 1], None, 1.0, landed

    def draw(self, frame: np.ndarray, j: int, t: float) -> None:
        if not (self.start <= t < self.end):
            return
        s = self.s
        p_in = (t - self.start) / self.enter if self.enter > 0 else 1.0
        a_in = _ease_out(p_in)
        a_out = 1.0
        if self.end < self.seconds and t > self.end - EXIT:
            a_out = _ease_in_out((self.end - t) / EXIT)
        frost = min(1.0, max(0.0, (t - self.start) / (self.enter * 1.25))) * a_out if self.enter > 0 else a_out
        self.pane.draw(frame, self.x, self.y, opacity=a_in * a_out, frost=max(frost, 0.0))
        # the type comes in a beat after the glass starts to form, and leaves a beat before it clears,
        # so it is never seen floating on clear glass
        alpha = _ease_out((p_in - 0.3) / 0.7) * a_out ** 1.7
        X, Y = self.x, self.y
        m = self.m
        rise = (1.0 - _ease_out(p_in)) * 10 * s  # the type settles up into place as the card arrives
        v, parts, seg, e, landed = self._progress(t)
        if self.label_sprite is not None:
            w = self.label_sprite
            over(frame, w.sprite, X + m["x_text"] - w.ox, Y + m["y_label"] + rise - w.oy, alpha * 0.92)
        self.figs.draw(frame, parts, X + m["x_text"], Y + m["y_num"] + rise, alpha)
        if m["chart"] is not None:
            cx0, cy0, _, _ = m["chart"]
            over(frame, self.rules, X + cx0, Y + cy0, alpha)
        if self.kind == "line":
            self._draw_line(frame, t, seg, e, landed, alpha)
        elif self.kind == "bars":
            self._draw_bars(frame, t, seg, e, landed, alpha)
        for wd, x, y in self.axis:
            over(frame, wd.sprite, X + x - wd.ox, Y + y - wd.oy, alpha * 0.8)
        # the arrow, as the last value lands
        if self.chip is not None and t >= self.land:
            q = (t - self.land) / 0.28
            sc = 0.35 + 0.65 * _back_out(q)
            ch = self.chip.scaled(sc)
            fw = self.figs.width(self.texts[-1])
            cx = X + m["x_text"] + fw + m["chip_gap"] + m["chip_d"] / 2
            cy = Y + m["y_num"] - self.figs.cap / 2
            over(frame, ch, cx - ch.w / 2, cy - ch.h / 2, alpha * min(1.0, q * 3))

    def _tip_x(self, t: float, seg, e: float, landed: int) -> float:
        if seg is not None:
            i0, i1 = seg
            return self.px_x[i0] + (self.px_x[i1] - self.px_x[i0]) * e
        return self.px_x[max(0, landed - 1)]

    def _draw_line(self, frame, t: float, seg, e: float, landed: int, alpha: float) -> None:
        s = self.s
        X, Y = self.x, self.y
        tip = self._tip_x(t, seg, e, landed)
        ox, oy = self.line_at
        sp = self.line_sprite
        cut = tip - ox
        if cut > 0:
            w = int(min(sp.w, math.ceil(cut + 3 * s)))
            xs = np.arange(w, dtype=np.float32)
            keep = np.clip((cut + 1.5 * s - xs) / (3.0 * s), 0, 1)
            part = Sprite(rgb=sp.rgb[:, :w] * keep[None, :, None], a=sp.a[:, :w] * keep[None, :])
            over(frame, part, X + ox, Y + oy, alpha)
        # the dots, each popping as the tip reaches it
        for k, p in enumerate(self.points):
            if k == 0:
                on = self.start + self.enter * 0.6
            else:
                on = p.at
            if t < on:
                continue
            q = (t - on) / 0.24
            sc = 0.3 + 0.7 * _back_out(q) if q < 1 else 1.0
            d = self.dot.scaled(sc)
            px, py = self.px_x[k], self.y_of(p.value)
            over(frame, d, X + px - d.w / 2, Y + py - d.h / 2, alpha)
        # the tip, glowing while it draws
        if seg is not None:
            v = float(np.interp(tip, self.curve_x, self.curve_v))
            ty = self.y_of(v)
            g, c = self.tip, self.tip_core
            over(frame, g, X + tip - g.w / 2, Y + ty - g.h / 2, alpha * 0.9)
            over(frame, c, X + tip - c.w / 2, Y + ty - c.h / 2, alpha)
        # the last point pulses once as it lands
        if t >= self.land and len(self.points) > 1:
            q = (t - self.land) / 0.75
            if q < 1:
                d = self.pulse_d * (1.0 + 2.2 * _ease_out(q))
                ring = _ring(d, max(1.5, 2.6 * s), self.accent)
                px, py = self.px_x[-1], self.y_of(self.points[-1].value)
                over(frame, ring, X + px - ring.w / 2, Y + py - ring.h / 2, alpha * (1 - q) * 0.85)

    def _draw_bars(self, frame, t: float, seg, e: float, landed: int, alpha: float) -> None:
        s = self.s
        X, Y = self.x, self.y
        r = self.bar_r
        for k, p in enumerate(self.points):
            a, b = self._bar_window(k)
            if t < a:
                continue
            q = (t - a) / max(1e-3, b - a)
            grow = _back_out(q, 1.2) if q < 1 else 1.0
            h = max(1.0, self.bar_h[k] * grow)
            # the bar being counted is in the accent; once the next one rises, it settles to glass
            nxt = self._bar_window(k + 1)[0] if k + 1 < len(self.points) else math.inf
            fade = min(1.0, max(0.0, (t - nxt) / 0.28))
            x = X + self.bar_x[k]
            top = Y + self.base - h
            for sprite, op in ((self.bar_on, 1.0 - fade), (self.bar_off, fade)):
                if op <= 0:
                    continue
                rows = max(1, min(int(math.ceil(r)) + 1, int(round(h))))
                cap = Sprite(rgb=sprite.rgb[:rows], a=sprite.a[:rows])
                body_h = int(max(0, round(h - cap.h)))
                over(frame, cap, x, top, alpha * op)
                if body_h > 0:
                    full = sprite.a.shape[0]
                    body = Sprite(rgb=sprite.rgb[full - body_h:], a=sprite.a[full - body_h:])
                    over(frame, body, x, top + cap.h, alpha * op)
            if self.tags and t >= b:
                tq = min(1.0, (t - b) / 0.18)
                wd = self.tags[k]
                tx = x + self.bar_w / 2 - wd.width / 2
                over(frame, wd.sprite, tx - wd.ox, top - 8 * s - wd.oy + (1 - tq) * 6 * s, alpha * tq * 0.92)

    def _bar_window(self, k: int) -> tuple[float, float]:
        for (a, b, _, _, i0, i1) in self.segments:
            if i1 == k:
                return a, b
        return self.points[k].at - 0.4, self.points[k].at

    # -- the sound --------------------------------------------------------------------------------------
    def tick_times(self) -> list[tuple[float, bool]]:
        """Each tick (seconds, is it the landing one): spaced by how fast the count is moving, so they
        bunch while it runs and slow as it settles (`counting.tick_times`)."""
        return counting.tick_times(self.segments, self._ease(), self.start, self.end)

    def mix(self, sound: np.ndarray, rate: int) -> np.ndarray:
        if not self.opts["sound"]:
            return sound
        ticks = self.tick_times()
        if not ticks:
            return sound
        # each tick heard TICK_UNDER dB under the voice's loud syllables, the landing one 2 dB nearer
        out, self.tick_db = counting.mix_ticks(sound, rate, ticks, TICK_UNDER)
        return out

    # -- checks, plan and the moments to look at ---------------------------------------------------------
    def report(self) -> dict:
        over_face = over_hair = outside = 0
        r = self.rect
        for j in range(self.n):
            t = j / self.fps
            if not (self.start <= t < self.end):
                continue
            f, h = self.faces[j], self.heads[j]
            if f is not None and r.intersects(f):
                over_face += 1
            elif h is not None and r.intersects(h):
                over_hair += 1
            if not self.zone.contains(r.grow(*glassmod.Pane.reach(self.s))):  # its shadow counted
                outside += 1
        values = " → ".join(num.Unit.join(t) for t in self.texts)
        land = num.Unit.join(self.texts[-1])
        said = f" on “{self.points[-1].said}”" if self.points[-1].said else ""
        return {"frames_over_face": over_face, "frames_over_hair": over_hair, "frames_outside_safe_zone": outside,
                "placed": f"{self.where}, {self.layout} card at {self.k:.0%} ({self.cw}x{self.ch})",
                "values": values, "lands": f"{land} at {self.land:.2f} s{said}", "found": self.found,
                "label": self.label or "(none)", "on_screen": f"{self.start:.2f}-{self.end:.2f} s",
                "type_contrast": f"white type on the card at {self.contrast:.1f} : 1 where it is brightest",
                "ticks": len(self.tick_times()) if self.opts["sound"] else 0,
                "warnings": list(self.warnings)}

    def plan(self) -> dict:
        return {"type": self.kind, "style": self.opts["style"], "layout": self.layout, "where": self.where,
                "clearance": self.clearance, "rect": self.rect.as_list(), "size": round(self.k, 3),
                "start": round(self.start, 3), "end": round(self.end, 3), "label": self.label,
                "accent": "#%02x%02x%02x" % tuple(self.accent),
                "unit": {"prefix": self.unit.prefix, "suffix": self.unit.suffix, "scale": self.unit.scale,
                         "decimals": self.unit.decimals},
                "points": [{"value": p.value, "text": num.Unit.join(t), "at": round(p.at, 3), "label": p.label,
                            "said": p.said} for p, t in zip(self.points, self.texts)],
                "found": self.found}

    def sheet_times(self) -> list[float]:
        """Six moments: the card arriving, the first value, the build under way, the last value just
        before and just after it lands, and the hold."""
        s, e, land = self.start, self.end, self.land
        first = self.points[0].at
        if self.segments:
            a, b = self.segments[-1][0], self.segments[-1][1]
            mid = a + 0.5 * (b - a)
        else:
            mid = (first + land) / 2
        picks = [s + min(0.2, self.enter * 0.5), max(first + 0.05, s + self.enter), mid, land - 0.06,
                 land + 0.4, (land + 0.4 + e) / 2]
        return sorted(min(e - 0.5 / self.fps, max(s, x)) for x in picks)


# ---- the effect --------------------------------------------------------------------------------------

def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from ..core import framing as framing_mod
    from ..core import video
    from ..core.timeline import Timeline
    from . import Job, PlanError

    info = clip.info
    fr = framing_mod.plan(info.width, info.height, info.rate, info.frames, clip.face, shape="keep")
    tl = Timeline.whole(info.frames / info.rate)

    def frame_at(j: int) -> np.ndarray:
        k = tl.source_frame(j, info.rate, info.frames)
        return fr.apply(video.read_frame(info, k), k)

    try:
        chart = Chart(clip, opts, (fr.out_w, fr.out_h), info.rate, tl.output_frames(info.rate),
                      head_in_output(clip, tl, fr), keepout_in_output(clip, tl, fr), frame_at, plan=plan)
    except ValueError as e:
        if plan is not None and not isinstance(e, O.OptionError):
            raise PlanError(str(e)) from None
        raise
    out_plan = {"effect": NAME, "clip": clip.path, "options": opts,
                "output": {"width": fr.out_w, "height": fr.out_h, "fps": round(info.rate, 3),
                           "seconds": round(tl.duration, 3), "framing": fr.note},
                "chart": chart.plan(), "sheet_times": [round(t, 3) for t in chart.sheet_times()]}

    def after(result: dict, out) -> dict:
        if opts["sound"] and getattr(chart, "tick_db", None) is not None:
            return {"sound": f"{len(chart.tick_times())} ticks under the count, their loudest "
                             f"{-chart.tick_db:.0f} dB under the voice"}
        return {}

    times = chart.sheet_times()
    return Job(NAME, clip, tl, fr, chart.draw, out_plan, checks=chart.report(),
               still_times=lambda: [times[1], times[2], times[4], times[5]], after=after, sound=chart.mix)
