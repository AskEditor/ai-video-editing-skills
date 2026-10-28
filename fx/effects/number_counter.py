"""number-counter: every amount you say counts up on screen and lands on its word, or a 3-2-1 flip
clock counts down above your head.

How it works:

1. **The numbers.** Left on auto, every amount said is taken, read the way `chart` reads them
   (`fx/core/amounts.py`): "2,000", "197K", "$3 million", "19.5%", "2.5 million". Years, lengths of
   time and a small number said in passing ("two things") are passed over, and so is a figure the
   speaker took back or said in an aside (the shared picker's filter). Amounts said together in one
   train of thought ("from 2,000 to 15,000") share one counter, which runs on from one to the next.
   Each lands written exactly as said: "197,000" stays 197,000 and "197K" stays 197K.
   Most talks say no amounts at all; then the run stops and says so, unless `values=` gives them.
2. **When.** Each value is whole and still on the first frame its first word is said (timed from
   where the voice begins, as `chart` times it): the count runs in the moment before. Given values
   land on their own words when they are said; else across the words `when=` names, the last on
   its last word; else at the times `when=` gives; else on the strongest words of the talk (the
   shared picker), and the run says which.
3. **The looks.** `roll`: the figures roll like a car's odometer on frosted glass, each place
   turning over as the one after it passes nine, blurred while it spins. `flip`: split-flap tiles of
   our own, a tile flipping each time its figure changes. `countdown`: one big flip tile counting
   3, 2, 1 above your head; when you count down aloud ("three, two, one") each number flips on its
   word, else the 1 lands on the word or second `when=` names.
4. **Where.** Above the head, beside it or under the chin, wherever the room is largest over every
   frame the counter is up, inside the safe zone with its shadow. Never over the face.
5. **Sound.** A soft tick under a roll (the landing tick rounder), a small clack for each flip, all
   made in code and set well under the voice. `sound=off` keeps the clip's sound exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core import amounts as num
from ..core import counting
from ..core import glass as glassmod
from ..core import moments
from ..core import text as textmod
from ..core.compose import Sprite, over
from ..core.items import snap
from ..core.layout import Rect, SafeZone
from ..core.parts import parse_time
from ..core.words import HESITATIONS, normalize
from . import options as O
from .captions import head_in_output
from .headline import find_said, keepout_in_output
from .tier_list import bands

NAME = "number-counter"
STYLES = ("roll", "flip", "countdown")
POSITIONS = ("auto", "above", "beside", "below")
ROOM_OF = {"above": "top", "beside": "side", "below": "below"}

OPTIONS = {
    "style": O.Choice(STYLES, "roll: the figures roll up like an odometer on frosted glass · flip: split-flap "
                              "tiles flip to each figure · countdown: one big tile flips 3, 2, 1 over your head"),
    "values": O.Text("auto: every amount you say (\"2,000\", \"$3 million\", \"19.5%\") · or give them, such as "
                     "197K or $2K,$15K,12% (each lands where it is said, else where when= says)"),
    "when": O.Text("auto · words that are said: given values land across them, the last on the last word (the "
                   "countdown's 1 lands there) · times in seconds such as 2.1,5.4 · empty"),
    "label": O.Text("the small line under the number: auto takes the words you say with it (followers, views); "
                    "empty for none; or your own words"),
    "from": O.Whole(2, 9, "the countdown starts at this number (3 counts 3, 2, 1)"),
    "position": O.Choice(POSITIONS, "auto: the roomiest place clear of your face · above · beside · below. Never "
                                    "over your face, whichever you pick"),
    "hold": O.Number(0.6, 10.0, "seconds each number stays on screen after it lands"),
    "sound": O.Switch("on: a soft tick under the roll, a clack for each flip, under your voice · off keeps the "
                      "clip's sound exactly"),
    "fix": O.Text("misheard words as wrong=right, separated by ;  e.g. \"ask editor=AskEditor\""),
}

# None of the test clips says an amount, so the talking clips are given values, landed across words
# each really says; the two clips that say amounts (read by a computer voice over a grey frame, made
# for `chart`) run on auto.
MATRIX = [
    ("numbers", "numbers", {}),
    ("numbers-flip", "numbers-2", {"style": "flip"}),
    ("D", "D", {"values": "197,000", "label": "Followers", "when": "the lively city come to life"}),
    ("B", "B", {"style": "flip", "values": "$2K,$15K", "label": "Revenue", "when": "improve your production quality"}),
    ("A", "A", {"values": "40%", "when": "what draws people in"}),
    ("D-countdown", "D", {"style": "countdown", "when": "my hotel"}),
    ("B-countdown", "B", {"style": "countdown", "when": "an editor that edits for you"}),
    ("33", "33", {"values": "500,000", "label": "Ads made", "when": "Distribution is a real big issue"}),
    ("C", "C", {"values": "19.5%", "label": "Growth"}),
]

# Timing, in seconds.
COUNT = 0.95  # a number counts up from nought over this long before it is said
COUNT_MIN = 0.3
NEXT_SHARE = 0.75  # a number that runs on from the one before counts over this share of the time between
ENTER = 0.26  # the counter arrives this long before its count starts
EXIT = 0.28
GROUP_GAP = 3.5  # amounts said this close, of one kind, share a counter
FLIP = 0.16  # one tile's flip
STEP_MIN = 0.075  # the tiles flip no faster than this
TICK_UNDER = 20.0  # dB under the voice's loud syllables
CLACK_UNDER = 16.0
COUNTDOWN_STEP = 1.0  # between the countdown's numbers when they are not said
COUNTDOWN_HOLD = 0.7  # the 1 stays this long, then the clock goes

# Design, in pixels of a frame 1080 wide at size 1.
ROLL_PX = 168
FLIP_PX = 132
DOWN_PX = 250
LABEL_PX = 38
PAD_X, PAD_TOP, PAD_BOTTOM = 54, 40, 40
LABEL_GAP = 20
K_MAX = {"roll": 1.15, "flip": 1.45, "countdown": 1.2}
SIDE_PAD = 0.22  # the sign's and the unit's small tile, past the type, as a share of their size
K_MIN = 0.45  # smaller than this and the figures are too small to read on a phone
PREF = {"top": 1.0, "side": 0.92, "below": 0.84}
TILE = (31, 33, 40)  # the flip tiles: a warm charcoal of our own
TILE_LIGHT = 16  # the tile's top is this much lighter than its foot
INK = (246, 246, 242)


def _clamp(x: float) -> float:
    return min(1.0, max(0.0, x))


def _ease_out(x: float) -> float:
    x = _clamp(x)
    return 1.0 - (1.0 - x) ** 3


def _ease_in_out(x: float) -> float:
    x = _clamp(x)
    return x * x * x * (x * (6 * x - 15) + 10)


def _back_out(x: float, c: float = 1.5) -> float:
    x = _clamp(x)
    return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2


# ---- what is counted ---------------------------------------------------------------------------------

@dataclass
class Beat:
    """One number: where it lands and how it is written."""
    value: float
    parts: tuple[str, str, str, str]  # (sign, figures, scale, unit), exactly as said or typed
    at: float  # the first frame it is whole: where its first word begins
    said: str = ""  # the words that said it, or "" when given
    how: str = "said"  # "said", "words", "time", "picked", "countdown"
    word: str = ""  # the word it lands on, when not its own
    runs_on: bool = False  # it counts on from the number before it on the same counter
    count_from: float = 0.0  # the value the count starts from
    count_at: float = 0.0  # when the count starts


@dataclass
class Show:
    """One counter on screen: one or more beats in a row."""
    beats: list[Beat]
    label: str = ""
    start: float = 0.0
    end: float = 0.0
    where: str = ""
    rect: Optional[Rect] = None
    k: float = 1.0
    notes: list[str] = field(default_factory=list)


def _unit_of(a: num.Amount) -> num.Unit:
    """How one amount is written, as it was said or typed: "197,000" plain, "197K" with its letter."""
    return num.Unit(prefix=a.prefix, suffix=a.suffix, scale=a.scale, decimals=a.decimals)


def _parts(a: num.Amount) -> tuple[str, str, str, str]:
    u = _unit_of(a)
    if a.scale:
        return u.exact(a.value)
    pre, _, _, suf = u.exact(a.value)
    sign = "-" if a.value < 0 else ""
    return sign + a.prefix, f"{abs(a.value):,.{a.decimals}f}", "", suf


def _said_runs(clip) -> tuple[list[list[num.Amount]], list, dict]:
    """The amounts said, grouped into the runs that share a counter (the core's grammar of runs:
    one kind of amount, said in one train of thought), in the order said."""
    heard, corrected = counting.first_listen(clip)
    back = moments.taken_back(heard, clip.quiet)
    said = [a for a in num.spoken(heard) if a.kind == "amount" and not back.holds(heard, a.first, a.last)]
    runs = []
    left = list(said)
    while left:
        run = num.series(left, heard)
        if run is None:
            break
        ordered = sorted(run.amounts, key=lambda a: a.first)
        runs.append((ordered, run))
        ids = {id(a) for a in run.amounts}
        left = [a for a in left if id(a) not in ids]
    runs.sort(key=lambda r: r[0][0].first)
    return runs, heard, corrected


def _schedule(show: Show, prev_end: float, seconds: float) -> None:
    """When each beat's count starts, and when the counter comes and goes."""
    b0 = show.beats[0]
    room = b0.at - max(prev_end, 0.0) - ENTER
    count = min(COUNT, max(0.0, room))
    if count < COUNT_MIN:
        count = min(COUNT_MIN, max(0.0, b0.at - prev_end))
    b0.count_at, b0.count_from = b0.at - count, 0.0
    for prev, b in zip(show.beats, show.beats[1:]):
        gap = b.at - prev.at
        b.count_from = prev.value if b.runs_on else 0.0
        b.count_at = b.at - max(min(COUNT, NEXT_SHARE * gap), min(gap, COUNT_MIN))
    show.start = max(prev_end, b0.count_at - ENTER, 0.0)


def _countdown_said(words) -> Optional[list[tuple[int, int]]]:
    """A countdown said aloud ("three, two, one", "3 2 1", "five… one"): [(number, word)], or None."""
    small = {k: v for k, v in num.SMALL.items() if 1 <= v <= 10}
    vals = []
    for i, w in enumerate(words):
        k = normalize(w.text)
        v = small.get(k) if k in small else (int(k) if k.isdigit() and 1 <= int(k) <= 10 else None)
        vals.append(v)
    best = None
    for i in range(len(words)):
        if vals[i] is None:
            continue
        seq = [(vals[i], i)]
        j = i
        while j + 1 < len(words) and vals[j + 1] == seq[-1][0] - 1 and words[j + 1].start - words[j].start < 1.8:
            seq.append((vals[j + 1], j + 1))
            j += 1
        if seq[-1][0] == 1 and len(seq) >= 3 and (best is None or len(seq) > len(best)):
            best = seq
    return best


# ---- drawing the figures -------------------------------------------------------------------------------

def _paste(rgb: np.ndarray, a: np.ndarray, sp: Sprite, x: float, y: float, opacity: float = 1.0) -> None:
    """Lay a sprite onto a small premultiplied buffer (rgb, a), changed in place."""
    H, W = a.shape
    x0, y0 = int(round(x)), int(round(y))
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x0 + sp.w), min(H, y0 + sp.h)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    sa = sp.a[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] * opacity
    srgb = sp.rgb[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] * opacity
    rgb[cy0:cy1, cx0:cx1] = srgb + rgb[cy0:cy1, cx0:cx1] * (1 - sa[..., None])
    a[cy0:cy1, cx0:cx1] = sa + a[cy0:cy1, cx0:cx1] * (1 - sa)


@dataclass
class Slot:
    ch: str  # the character in the final figures
    x: float  # its left edge, from the number's start
    w: float
    place: Optional[int]  # for a figure: its place counted from the last (0 the last figure); None for , or .


class Number:
    """One value's figures laid out once, so they can roll or flip without moving: the slots of its
    final form (sign, figures with their , and ., scale and unit)."""

    def __init__(self, figs: counting.Figures, parts: tuple[str, str, str, str]):
        self.figs, self.parts = figs, parts
        pre, digits, scale, suf = parts
        self.pre, self.digits, self.tail = pre, digits, scale + suf
        items, self.width = figs.layout(parts)
        self.slots: list[Slot] = []
        k = len(pre)
        figures = [c for c in digits if c.isdigit()]
        place = len(figures)
        x_after = items[k + len(digits)][1] if k + len(digits) < len(items) else self.width
        for n, ch in enumerate(digits):
            _, x, _ = items[k + n]
            g = figs.glyph(ch)
            if ch.isdigit():
                place -= 1
                cell = figs.cell_one if ch == "1" else figs.cell
                self.slots.append(Slot(ch, x - (cell - g.width) / 2, cell, place))
            else:
                self.slots.append(Slot(ch, x, g.width, None))
        self.digits_x0 = self.slots[0].x if self.slots else 0.0
        self.digits_x1 = x_after
        self.decimals = len(digits.split(".")[1]) if "." in digits else 0
        self.ticks = int(digits.replace(",", "").replace(".", "") or "0")
        self.mult = num.SCALE_LETTERS.get(scale, 1.0)

    def ticks_of(self, v: float) -> float:
        """A value in this number's smallest step ("19.5%" counts in tenths), as a float."""
        return abs(v) / self.mult * 10 ** self.decimals


class Roller:
    """The roll: each place turns like an odometer's wheel, the last one fastest, a place turning
    over only while the one after it passes from 9 to 0, blurred while it spins."""

    def __init__(self, figs: counting.Figures):
        import cv2  # noqa: F401  (the blur)

        self.figs = figs
        cap = figs.cap
        self.pitch = cap * 1.32
        self.win_top = cap * 1.18  # the window's top, above the baseline
        self.win_h = int(math.ceil(cap * 1.18 + cap * 0.30 + 2))
        h = self.win_h
        ys = np.arange(h, dtype=np.float32)
        fe = max(2.0, 0.16 * h)
        self.feather = np.clip(np.minimum(ys / fe, (h - 1 - ys) / fe), 0, 1).astype(np.float32)
        self.feather = (self.feather * self.feather * (3 - 2 * self.feather))[:, None]

    @staticmethod
    def wheel(x: float, place: int) -> float:
        """Where the wheel of `place` stands (a figure and how far it has turned to the next) when the
        whole count stands at `x` steps."""
        if place == 0:
            return x
        size = 10 ** place
        base = math.floor(x / size)
        rest = x - base * size
        return base + max(0.0, rest - (size - 1))

    def draw_slot(self, frame, slot: Slot, c: float, speed: float, x: float, baseline: float, alpha: float,
                  blank_below: bool) -> None:
        import cv2

        figs = self.figs
        w = int(math.ceil(slot.w + 2 * figs.px * 0.12))
        pad = (w - slot.w) / 2
        h = self.win_h
        rgb = np.zeros((h, w, 3), np.float32)
        a = np.zeros((h, w), np.float32)
        lo = math.floor(c + 1e-9)
        f = c - lo
        for k, dy in ((lo, -f * self.pitch), (lo + 1, (1 - f) * self.pitch)):
            if dy <= -self.pitch or dy >= self.pitch:
                continue
            d = k % 10
            if blank_below and k <= 0:
                continue  # no leading nought: the wheel shows nothing until the count reaches it
            g = figs.glyph(str(d))
            gx = pad + (slot.w - g.width) / 2 - g.ox
            gy = self.win_top + dy - g.oy
            _paste(rgb, a, g.sprite, gx, gy)
        if not a.any():
            return
        blur = int(round(min(self.pitch * 0.85, speed * self.pitch * 0.55)))
        if blur >= 2:
            rgb = cv2.blur(rgb, (1, blur))
            a = cv2.blur(a, (1, blur))
        m = self.feather
        over(frame, Sprite(rgb=rgb * m[..., None], a=a * m), x + slot.x - pad, baseline - self.win_top, alpha)


class Tiles:
    """Split-flap tiles of our own: a warm charcoal tile, a thin seam across its middle, the figure
    in white; when its figure changes the top half folds down over the hinge and the new bottom half
    falls into place."""

    def __init__(self, px: int, s: float, chars: str):
        from PIL import Image, ImageDraw, ImageFilter

        s = float(s)
        self.px, self.s = px, s
        face = counting.FACE
        self.cap = textmod.cap_height(face, px)
        self.tw = int(round(max(textmod.advance(c, face, px) for c in "0123456789") * 1.22 + 0.18 * px))
        self.th = int(round(self.cap * 1.62))
        self.r = 0.13 * self.th
        self.m = int(math.ceil(18 * s))  # the shadow's margin round each tile
        self.hinge = self.th // 2
        self.seam = max(1.0, 2.2 * s)
        k = 4
        W, H = self.tw * k, self.th * k
        base = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        grad = np.zeros((H, W, 4), np.uint8)
        t = np.linspace(0, 1, H)[:, None]
        for c in range(3):
            grad[..., c] = np.clip(TILE[c] + TILE_LIGHT * (1 - t) - 6 * t, 0, 255).astype(np.uint8)
        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, W - 1, H - 1), radius=self.r * k, fill=255)
        grad[..., 3] = np.asarray(mask)
        base = Image.fromarray(grad, "RGBA")
        # a hairline of light along the top edge
        hl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(hl).rounded_rectangle((0, 0, W - 1, H - 1), radius=self.r * k, outline=(255, 255, 255, 34),
                                             width=max(1, int(1.2 * s * k)))
        base = Image.alpha_composite(base, hl)
        self.base = base.resize((self.tw, self.th), Image.LANCZOS)
        f = textmod.font(face, px)
        self.faces: dict[str, np.ndarray] = {}
        for ch in set(chars) | {" "}:
            img = self.base.copy()
            if ch.strip():
                l, tp, rr, bb = f.getbbox(ch, anchor="ls")
                adv = textmod.advance(ch, face, px)
                x = (self.tw - adv) / 2
                y = (self.th + self.cap) / 2
                ImageDraw.Draw(img).text((x, y), ch, font=f, fill=INK + (255,), anchor="ls")
            arr = np.asarray(img).astype(np.float32)
            # the seam: a dark line across the middle, where the halves part
            y0 = int(round(self.hinge - self.seam / 2))
            y1 = int(round(self.hinge + self.seam / 2)) + 1
            arr[y0:y1, :, :3] *= 0.35
            self.faces[ch] = arr
        # the shadow under a tile
        sh = Image.new("L", (self.tw + 2 * self.m, self.th + 2 * self.m), 0)
        ImageDraw.Draw(sh).rounded_rectangle((self.m, self.m + 4 * s, self.m + self.tw, self.m + self.th + 4 * s),
                                             radius=self.r, fill=255)
        sh = sh.filter(ImageFilter.GaussianBlur(7 * s))
        a = np.asarray(sh, np.float32) / 255 * 0.42
        self.shadow = Sprite(rgb=np.zeros(a.shape + (3,), np.float32), a=a)

    def _sprite(self, arr: np.ndarray, shade: float = 1.0) -> Sprite:
        a = arr[..., 3] / 255.0
        return Sprite(rgb=arr[..., :3] * shade * a[..., None], a=a)

    def draw(self, frame, x: float, y: float, old: str, new: str, q: float, alpha: float) -> None:
        """A tile at (x, y) showing `new`, folding over from `old` at `q` (0..1; 1 is still)."""
        import cv2

        over(frame, self.shadow, x - self.m, y - self.m, alpha)
        hz = self.hinge
        if q >= 1.0 or old == new:
            over(frame, self._sprite(self.faces[new]), x, y, alpha)
            return
        top_new, bot_old = self.faces[new][:hz], self.faces[old][hz:]
        over(frame, self._sprite(top_new, 0.82 + 0.18 * q), x, y, alpha)
        over(frame, self._sprite(bot_old, 1.0 - 0.35 * min(1.0, q * 2)), x, y + hz, alpha)
        ang = q * math.pi
        if q < 0.5:  # the old top half folding down to the hinge
            hh = max(1, int(round(hz * math.cos(ang))))
            flap = cv2.resize(self.faces[old][:hz], (self.tw, hh), interpolation=cv2.INTER_AREA)
            over(frame, self._sprite(flap, 1.0 - 0.45 * (1 - math.cos(ang))), x, y + hz - hh, alpha)
        else:  # the new bottom half falling from the hinge into place
            hh = max(1, int(round((self.th - hz) * -math.cos(ang))))
            flap = cv2.resize(self.faces[new][hz:], (self.tw, hh), interpolation=cv2.INTER_AREA)
            over(frame, self._sprite(flap, 0.62 + 0.38 * (-math.cos(ang))), x, y + hz, alpha)


# ---- the effect -------------------------------------------------------------------------------------

class Counter:
    def __init__(self, clip, opts: dict, size: tuple[int, int], fps: float, n_frames: int, head_at, face_at,
                 frame_at, plan: Optional[dict] = None):
        self.W, self.H = size
        self.U = self.W / 1080.0
        self.fps, self.n = fps, n_frames
        self.seconds = n_frames / fps
        self.opts = opts
        self.style = opts["style"]
        self.zone = SafeZone(self.W, self.H)
        self.heads = [head_at(j) for j in range(n_frames)]
        self.faces = [face_at(j) for j in range(n_frames)]
        self.frame_at = frame_at
        self.warnings: list[str] = []
        self.found = ""
        self.words = [w for w in clip.words if w.key not in HESITATIONS]
        self.quiet = clip.quiet
        if plan is not None:
            self._from_plan(plan)
        elif self.style == "countdown":
            self._countdown(clip)
        else:
            self._numbers(clip)
        self._times()
        for show in self.shows:
            self._place(show)
        self._build()

    # -- the numbers ----------------------------------------------------------------------------------
    def _numbers(self, clip) -> None:
        opts = self.opts
        given = opts["values"].strip()
        lab = opts["label"].strip()
        runs, heard, corrected = _said_runs(clip)
        shows: list[Show] = []
        if given.lower() in ("", "auto"):
            if not runs:
                raise O.OptionError("no amounts are said in this video (years, lengths of time and small numbers "
                                    "said in passing are passed over), so there is nothing to count. Give the "
                                    "number: values=197K, and when= for the words it lands on")
            for amounts, run in runs:
                beats = [Beat(a.value, _parts(a), counting.start_of(a, heard, corrected), said=a.said)
                         for a in amounts]
                # a run said newest first ("3 million, up from 500,000") is not one count: each from nought
                for b in beats[1:]:
                    b.runs_on = not run.reversed
                label = num.label_for(run, heard) if lab.lower() == "auto" else lab
                shows.append(Show(beats, label=label[:1].upper() + label[1:] if lab.lower() == "auto" else label))
            self.found = "said: " + " … ".join(f"“{b.said}”" for s in shows for b in s.beats)
        else:
            try:
                typed = num.parse_list(given)
            except ValueError as e:
                raise O.OptionError(f"values: {e}") from None
            if len(typed) > 12:
                raise O.OptionError("values: twelve numbers at most")
            shows = self._land_given(typed, runs, heard, corrected)
        self.shows = shows

    def _land_given(self, typed: list[num.Amount], runs, heard, corrected) -> list[Show]:
        """Given values land on their own words when said; else across the words `when` names; else
        at the times it gives; else on the talk's strongest words (the shared picker)."""
        opts = self.opts
        lab = opts["label"].strip()
        when = opts["when"].strip()
        said = [a for amounts, _ in runs for a in amounts]
        beats: list[Beat] = []
        notes = []
        if said and all(any(abs(s.value - t.value) < 1e-9 for s in said) for t in typed):
            for t in typed:
                s = next(s for s in said if abs(s.value - t.value) < 1e-9)
                beats.append(Beat(t.value, _parts(t), counting.start_of(s, heard, corrected), said=s.said))
            self.found = "given, and said: " + " … ".join(f"“{b.said}”" for b in beats)
        elif when and when.lower() != "auto":
            times = [parse_time(x) for x in when.replace(";", ",").split(",")]
            if all(t is not None for t in times):
                if len(times) != len(typed):
                    raise O.OptionError(f"when: give one time for each number ({len(typed)})")
                for a, t in zip(typed, times):
                    if t >= self.seconds:
                        raise O.OptionError(f"when: {t:g} s is after the end of the video ({self.seconds:.2f} s)")
                    at, word = snap(self.words, t)
                    beats.append(Beat(a.value, _parts(a), at, how="time", word=word))
                self.found = "given, at the times given" + (" (each moved onto the word starting there)"
                                                             if any(b.word for b in beats) else "")
            else:
                hit = find_said(self.words, when)
                if not hit:
                    raise O.OptionError(f"when: the words \"{when}\" are not said in this video. Read the words with "
                                        f"`python -m fx.run words <video>` and use them as heard, or give times in "
                                        f"seconds")
                i0, i1 = hit[0], hit[1]
                n = len(typed)
                for k, a in enumerate(typed):
                    i = i1 if n == 1 else i0 + round(k * (i1 - i0) / (n - 1))
                    beats.append(Beat(a.value, _parts(a), self.words[i].start, how="words", word=self.words[i].clean))
                self.found = f"given, landed across “{when}”, the last on “{self.words[i1].clean}”"
        else:
            beats = self._picked(typed)
            notes.append("not said in the video: landed on the strongest words of the talk (say when= to choose)")
            self.found = "given, not said: landed on " + ", ".join(f"“{b.word}” ({b.at:.2f} s)" for b in beats) + \
                         ", the talk's strongest words"
        self.warnings += notes
        label = "" if lab.lower() == "auto" else lab
        # given values close together share a counter, as said ones do
        shows, cur = [], [beats[0]]
        for b in beats[1:]:
            if b.at - cur[-1].at <= GROUP_GAP and b.parts[0] == cur[-1].parts[0] and b.parts[3] == cur[-1].parts[3]:
                b.runs_on = True
                cur.append(b)
            else:
                shows.append(Show(cur, label=label))
                cur = [b]
        shows.append(Show(cur, label=label))
        return shows

    def _picked(self, typed: list[num.Amount]) -> list[Beat]:
        """The strongest words of the talk, one for each value, at least two and a half seconds apart."""
        words = self.words
        quiet = self.quiet
        back = moments.taken_back(words, quiet)
        cands = []
        for line in moments.lines(words, quiet):
            c = moments.best_in(words, line, quiet, back)
            if c is not None:
                cands.append(c)
        cands.sort(key=lambda c: -c.score)
        chosen = []
        for c in cands:
            if c.word.start < COUNT_MIN + ENTER:
                continue
            if all(abs(c.word.start - o.word.start) >= 2.5 for o in chosen):
                chosen.append(c)
            if len(chosen) == len(typed):
                break
        if len(chosen) < len(typed):
            raise O.OptionError(f"values: {len(typed)} numbers will not fit in this video without a word for each; "
                                f"give when= with times")
        chosen.sort(key=lambda c: c.word.start)
        return [Beat(a.value, _parts(a), c.word.start, how="picked", word=c.word.clean) for a, c in zip(typed, chosen)]

    def _countdown(self, clip) -> None:
        opts = self.opts
        top = int(opts["from"])
        when = opts["when"].strip()
        said = _countdown_said(self.words)
        beats = []
        if said and (not when or when.lower() == "auto"):
            for v, i in said:
                beats.append(Beat(float(v), ("", str(v), "", ""), self.words[i].start, how="countdown",
                                  word=self.words[i].clean))
            self.found = "said aloud: " + ", ".join(f"“{self.words[i].clean}”" for _, i in said)
        else:
            end = None
            how = ""
            if when and when.lower() != "auto":
                t = parse_time(when)
                if t is not None:
                    if t >= self.seconds:
                        raise O.OptionError(f"when: {t:g} s is after the end of the video ({self.seconds:.2f} s)")
                    end, word = snap(self.words, t)
                    how = f"the 1 lands at {end:.2f} s" + (f" on “{word}”" if word else "")
                else:
                    hit = find_said(self.words, when)
                    if not hit:
                        raise O.OptionError(f"when: the words \"{when}\" are not said in this video; give a time in "
                                            f"seconds")
                    end = self.words[hit[0]].start
                    how = f"the 1 lands on “{self.words[hit[0]].clean}” ({end:.2f} s)"
            if end is None:  # the first line that starts once there is time for the count before it
                starts = [s for s in moments.strong_starts(self.words, self.quiet) if s.at >= (top - 1) * COUNTDOWN_STEP + 0.5]
                end = starts[0].at if starts else min(self.seconds - COUNTDOWN_HOLD, (top - 1) * COUNTDOWN_STEP + 0.6)
                how = f"the 1 lands at {end:.2f} s, where a line starts (say when= to choose)"
            first = end - (top - 1) * COUNTDOWN_STEP
            if first < 0.25:
                raise O.OptionError(f"from: a countdown from {top} needs {top - 1:g} s before {end:.2f} s; start lower "
                                    f"or land it later")
            for k, v in enumerate(range(top, 0, -1)):
                beats.append(Beat(float(v), ("", str(v), "", ""), first + k * COUNTDOWN_STEP, how="countdown"))
            self.found = f"counted {top} to 1, a second each: {how}"
        lab = opts["label"].strip()
        self.shows = [Show(beats, label="" if lab.lower() == "auto" else lab)]

    def _from_plan(self, plan: dict) -> None:
        try:
            shows = []
            for s in plan["counter"]["shows"]:
                beats = []
                for b in s["beats"]:
                    a = num.parse(b["text"]) if self.style != "countdown" else None
                    parts = _parts(a) if a is not None else ("", b["text"], "", "")
                    value = a.value if a is not None else float(b["text"])
                    beats.append(Beat(value, parts, float(b["at"]), said=b.get("said", ""), how=b.get("how", "plan")))
                for x in beats[1:]:
                    x.runs_on = bool(s.get("runs_on", True))
                show = Show(beats, label=s.get("label", ""))
                if s.get("rect"):
                    show.rect, show.k, show.where = Rect.of(s["rect"]), float(s["size"]), s.get("where", "plan")
                shows.append(show)
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"the plan's counter could not be read ({e})") from None
        self.shows = shows
        self.found = "from the plan"

    # -- when ---------------------------------------------------------------------------------------------
    def _times(self) -> None:
        shows = sorted(self.shows, key=lambda s: s.beats[0].at)
        hold = float(self.opts["hold"])
        prev_end = 0.0
        for n, show in enumerate(shows):
            if self.style == "countdown":
                b0 = show.beats[0]
                for b in show.beats:
                    b.count_at = b.at - FLIP
                show.start = max(0.0, b0.at - FLIP - ENTER)
                show.end = min(self.seconds, show.beats[-1].at + COUNTDOWN_HOLD + EXIT)
                continue
            _schedule(show, prev_end, self.seconds)
            show.end = min(self.seconds, show.beats[-1].at + hold)
            if n + 1 < len(shows):
                nxt = shows[n + 1].beats[0].at - COUNT_MIN - ENTER
                if show.end > nxt:
                    show.end = max(show.beats[-1].at + 0.5, nxt)
                    if show.end > nxt:
                        show.notes.append("the next number comes soon after: this one is cut short")
            prev_end = show.end
        self.shows = shows

    # -- where ---------------------------------------------------------------------------------------------
    def _design(self, show: Show) -> tuple[float, float]:
        """The counter's size at size 1, in design pixels: (width, height)."""
        figs = counting.Figures(self._px(), 1.0)
        if self.style == "roll":
            w = max(figs.width(b.parts) for b in show.beats) + 2 * PAD_X
            lw = textmod.advance(show.label, counting.FACE, LABEL_PX) + 2 * PAD_X if show.label else 0
            h = PAD_TOP + figs.cap * 1.18 + figs.cap * 0.30 + PAD_BOTTOM
            if show.label:
                h += LABEL_GAP + textmod.cap_height(counting.FACE, LABEL_PX) * 1.4
            return max(w, lw), h
        tiles = Tiles(self._px(), 1.0, "0123456789")
        small = counting.Figures(max(8, int(round(self._px() * 0.62))), 1.0)
        least = max(sum(c.isdigit() for c in b.parts[1]) for b in show.beats)
        w = max(self._tile_width(b.parts, tiles, small, figs, least) for b in show.beats) + 2 * tiles.m
        h = tiles.th + 2 * tiles.m
        if show.label:
            h += LABEL_GAP + textmod.cap_height(counting.FACE, LABEL_PX) * 1.9
            w = max(w, textmod.advance(show.label, counting.FACE, LABEL_PX) + 60)
        return w, h

    @staticmethod
    def _tile_width(parts, tiles: "Tiles", small: counting.Figures, figs: counting.Figures, least: int = 0) -> float:
        """How wide a row of tiles is: a tile for each figure, the , and . between them, the sign
        before and the scale and unit after, smaller."""
        pre, digs, scale, suf = parts
        gap = 0.07 * tiles.px
        n = max(least, sum(c.isdigit() for c in digs))
        w = n * tiles.tw + max(0, n - 1) * gap + sum(figs.glyph(c).width for c in digs if not c.isdigit())
        if pre:
            w += small.width(("", "", "", pre)) + SIDE_PAD * small.px * 2 + 0.07 * tiles.px
        if scale + suf:
            w += small.width(("", "", "", scale + suf)) + SIDE_PAD * small.px * 2 + 0.07 * tiles.px
        return w

    def _px(self) -> int:
        return {"roll": ROLL_PX, "flip": FLIP_PX, "countdown": DOWN_PX}[self.style]

    def _halo(self, k: float) -> tuple[float, float, float, float]:
        """How far the drawing reaches past the counter's box: the glass's shadow (the tiles carry
        their shadow inside their box)."""
        if self.style != "roll":
            return (0.0, 0.0, 0.0, 0.0)
        u = k * self.U
        sp = math.ceil(22 * u)
        return float(sp), float(max(0.0, sp - 7 * u)), float(sp), float(sp + 7 * u)

    def _place(self, show: Show) -> None:
        if show.rect is not None:
            return
        a = int(max(0, math.floor(show.start * self.fps)))
        b = int(min(self.n - 1, math.ceil(show.end * self.fps)))
        keep = self.heads[a:b + 1]
        faces = self.faces[a:b + 1]
        w1, h1 = self._design(show)
        w1, h1 = w1 * self.U, h1 * self.U
        halo = self._halo(K_MAX[self.style])
        z = self.zone.rect.grow(-halo[0], -halo[1], -halo[2], -halo[3])
        rooms = bands(z, keep, faces, self.W, self.H, behind=False)
        pos = self.opts["position"]
        names = [ROOM_OF[pos]] if pos != "auto" else ["top", "side", "below"]
        best = None
        for name in names:
            room = rooms.get(name)
            if room is None:
                continue
            k = min(room.w / w1, room.h / h1, K_MAX[self.style])
            score = k * PREF[name] if k >= K_MIN else k * 0.5
            if best is None or score > best[0]:
                best = (score, k, name, room)
        if best is None and pos != "auto":
            self.warnings.append(f"position={pos}: there is no room {pos} the head here, so it goes where there is")
            for name in ("top", "side", "below"):
                room = rooms.get(name)
                if room is not None:
                    k = min(room.w / w1, room.h / h1, K_MAX[self.style])
                    if best is None or k > best[1]:
                        best = (k, k, name, room)
        if best is None:
            raise O.OptionError("there is no room clear of the face for a counter here (the face fills the frame)")
        _, k, name, room = best
        if k < K_MIN:
            self.warnings.append(f"the counter is small ({k:.0%} of its size): there is little room clear of the face")
        w, h = w1 * k, h1 * k
        boxes = [f for f in faces if f is not None]
        fx = Rect.around(boxes).cx if boxes else room.cx
        fy = Rect.around(boxes).cy if boxes else room.cy
        if name == "top":
            x0 = min(max(fx - w / 2, room.x0), room.x1 - w)
            y0 = room.y1 - h
        elif name == "below":
            x0 = min(max(fx - w / 2, room.x0), room.x1 - w)
            y0 = room.y0
        else:
            x0 = room.x1 - w if room.cx < fx else room.x0
            y0 = min(max(fy - h / 2, room.y0), room.y1 - h)
        show.rect = Rect(x0, y0, x0 + w, y0 + h)
        show.k, show.where = float(k), {"top": "above the head", "side": "beside the head", "below": "under the chin"}[name]

    # -- drawing ------------------------------------------------------------------------------------------
    def _build(self) -> None:
        self.drawn = []
        for show in self.shows:
            s = show.k * self.U
            px = max(10, int(round(self._px() * s)))
            d: dict = {"s": s}
            figs = counting.Figures(px, s)
            d["figs"] = figs
            r = show.rect
            if show.label:
                lp = max(8, int(round(LABEL_PX * s)))
                d["label"] = textmod.draw_word(show.label, counting.FACE, lp, fill=(236, 237, 242) if self.style == "roll"
                                               else INK, shadow=(0, 1.5 * s, 3 * s, 0.4))
                d["label_cap"] = textmod.cap_height(counting.FACE, lp)
            if self.style == "roll":
                d["numbers"] = [Number(figs, b.parts) for b in show.beats]
                d["roller"] = Roller(figs)
                pane = glassmod.Pane(int(round(r.w)), int(round(r.h)), 0.30 * min(r.w, r.h), unit=s)
                fr = [self.frame_at(int(min(self.n - 1, max(0, round(t * self.fps)))))
                      for t in np.linspace(show.start, show.end, 5)]
                bright = pane.tone_over(fr, r.x0, r.y0)
                d["pane"], d["contrast"] = pane, glassmod.contrast_with_white(bright)
                d["baseline"] = r.y0 + PAD_TOP * s + figs.cap * 1.18
            else:
                chars = "0123456789" + "".join(set("".join(b.parts[1] for b in show.beats)))
                tiles = Tiles(px, s, "".join(c for c in chars if c.isdigit()))
                d["tiles"] = tiles
                d["small"] = counting.Figures(max(8, int(round(px * 0.62))), s)
            self.drawn.append(d)

    def _value_at(self, show: Show, t: float) -> tuple[int, float, float]:
        """(which beat is counting or shown, the value in its steps, how fast the last place turns in
        steps a frame)."""
        beats = show.beats
        k = 0
        for i, b in enumerate(beats):
            if t >= b.count_at:
                k = i
        b = beats[k]
        if t >= b.at:  # landed: whole and still, never blurred
            return k, self._x(show, k, t), 0.0
        return k, self._x(show, k, t), abs(self._x(show, k, t) - self._x(show, k, t - 1 / self.fps))

    def _x(self, show: Show, k: int, t: float) -> float:
        b = show.beats[k]
        d = self.drawn[self.shows.index(show)] if self.style == "roll" else None
        nmb = d["numbers"][k] if d else None
        x1 = nmb.ticks if nmb else 0
        x0 = nmb.ticks_of(b.count_from or 0.0) if nmb else 0
        if t >= b.at:
            return float(x1)
        q = (t - b.count_at) / max(1e-6, b.at - b.count_at)
        return x0 + (x1 - x0) * _ease_out(q)

    def draw(self, frame: np.ndarray, j: int, t: float) -> None:
        for show, d in zip(self.shows, self.drawn):
            if show.start <= t < show.end:
                if self.style == "roll":
                    self._draw_roll(frame, show, d, t)
                else:
                    self._draw_tiles(frame, show, d, t)

    def _fade(self, show: Show, t: float) -> tuple[float, float]:
        """(opacity, how far it has risen into place: 0 still)."""
        p_in = (t - show.start) / ENTER
        a = _ease_out(p_in)
        if show.end < self.seconds and t > show.end - EXIT:
            a *= _ease_in_out((show.end - t) / EXIT)
        return a, 1.0 - _ease_out(p_in)

    def _draw_roll(self, frame, show: Show, d: dict, t: float) -> None:
        s, r = d["s"], show.rect
        alpha, rise = self._fade(show, t)
        pane = d["pane"]
        pane.draw(frame, r.x0, r.y0, opacity=alpha, frost=alpha, scale=0.94 + 0.06 * _ease_out((t - show.start) / ENTER))
        ta = alpha ** 1.6
        k, x, speed = self._value_at(show, t)
        nmb: Number = d["numbers"][k]
        roller: Roller = d["roller"]
        figs: counting.Figures = d["figs"]
        baseline = d["baseline"] + rise * 12 * s
        # the places not yet reached are not drawn: the number grows from the right and stays centred
        vis = []
        for sl in nmb.slots:
            if sl.place is None:
                vis.append(1.0)
            else:
                need = 10 ** sl.place
                lead = sl.place > nmb.decimals  # a place left of the units
                vis.append(1.0 if not lead else _clamp((x - need) / max(1.0, 0.04 * need) + 1.0))
        for i, sl in enumerate(nmb.slots):  # a comma shows with the figure before it
            if sl.place is None and i > 0:
                vis[i] = vis[i - 1]
        hidden = sum((1 - v) * sl.w for v, sl in zip(vis, nmb.slots))
        width = nmb.width - hidden
        x_start = r.x0 + (r.w - width) / 2 - hidden
        shown_first = next((i for i, v in enumerate(vis) if v > 0.01), len(vis) - 1)
        # the sign before the first figure shown, the scale and the unit after the last
        items, _ = figs.layout(nmb.parts)
        pre_n = len(nmb.pre)
        lead_x = nmb.slots[shown_first].x + (1 - vis[shown_first]) * nmb.slots[shown_first].w
        for g, dx, up in items[:pre_n]:
            over(frame, g.sprite, x_start + lead_x - nmb.digits_x0 + dx - g.ox, baseline - up - g.oy, ta)
        for g, dx, up in items[pre_n + len(nmb.digits):]:
            over(frame, g.sprite, x_start + dx - g.ox, baseline - up - g.oy, ta)
        for v, sl in zip(vis, nmb.slots):
            if v <= 0.01:
                continue
            if sl.place is None:
                g = figs.glyph(sl.ch)
                over(frame, g.sprite, x_start + sl.x - g.ox, baseline - g.oy, ta * v)
                continue
            c = roller.wheel(x, sl.place)
            sp = speed / 10 ** sl.place if sl.place else speed
            lead = sl.place > nmb.decimals
            roller.draw_slot(frame, sl, c, sp, x_start, baseline, ta * v, blank_below=lead)
        if "label" in d:
            w = d["label"]
            ly = r.y1 - PAD_BOTTOM * s - 0.2 * d["label_cap"] + rise * 12 * s
            over(frame, w.sprite, r.cx - w.width / 2 - w.ox, ly - w.oy, ta * 0.9)

    def _steps(self, show: Show, k: int) -> tuple[int, list[tuple[float, int]]]:
        """Beat k's flips: the value the tiles start from, and (when each flip starts, the value shown
        after it), in the beat's own smallest steps. The last flip ends on the frame the number is said."""
        b = show.beats[k]
        dec = len(b.parts[1].split(".")[1]) if "." in b.parts[1] else 0
        mult = num.SCALE_LETTERS.get(b.parts[2], 1.0)
        n1 = int(round(abs(b.value) / mult * 10 ** dec))
        n0 = int(round(abs(b.count_from) / mult * 10 ** dec))
        if self.style == "countdown":
            return n0, [(b.at - FLIP, n1)]
        span = max(0.0, b.at - FLIP - b.count_at)
        steps = int(np.clip(math.floor(span / STEP_MIN), 1, 11))
        out, last = [], n0
        for i in range(1, steps + 1):
            v = n1 if i == steps else int(math.floor(n0 + (n1 - n0) * _ease_out(i / steps)))
            if v != last:
                out.append((b.count_at + span * (i / steps), v))
                last = v
        if not out or out[-1][1] != n1:
            out.append((b.at - FLIP, n1))
        return n0, out

    @staticmethod
    def _figures(b: Beat, v: int) -> str:
        """Beat b's final figures with the value v (in its smallest steps) written into them: the same
        , and . in the same places, a place not yet reached blank (the units place always shows)."""
        digs = b.parts[1]
        n = sum(c.isdigit() for c in digs)
        dec = len(digs.split(".")[1]) if "." in digs else 0
        s = str(max(0, v))[-n:].rjust(n)
        shown = n - dec - 1  # the places left of the units place may be blank
        s = "".join(" " if (i < shown and ch == " ") else ("0" if ch == " " else ch) for i, ch in enumerate(s))
        out, i = [], 0
        for c in digs:
            if c.isdigit():
                out.append(s[i])
                i += 1
            else:
                out.append(c)
        return "".join(out)

    def _tile_state(self, show: Show, t: float) -> tuple[Beat, str, str, float]:
        """At `t`: the beat shown, what its tiles showed, what they show, and how far the flip between
        has gone (1 still)."""
        k = 0
        for i, b in enumerate(show.beats):
            if t >= b.count_at:
                k = i
        b = show.beats[k]
        if self.style == "countdown":
            prev = show.beats[k - 1] if k > 0 else None
            old = prev.parts[1] if prev else " " * len(b.parts[1])
            return b, old.rjust(len(b.parts[1]))[-len(b.parts[1]):], b.parts[1], _clamp((t - (b.at - FLIP)) / FLIP)
        if t >= b.at:
            return b, b.parts[1], b.parts[1], 1.0
        n0, steps = self._steps(show, k)
        old = new = self._figures(b, n0) if (b.runs_on or n0) else self._figures(b, 0)
        q = 1.0
        for at, v in steps:
            if t >= at:
                old, new = new, self._figures(b, v)
                q = _clamp((t - at) / FLIP)
        return b, old, new, q

    @staticmethod
    def _side(frame, d: dict, parts, x: float, y: float, alpha: float) -> float:
        """The sign before the tiles or the scale and unit after them, smaller, on a small tile of
        their own (so they read over a bright room too), level with the tiles' middle. Returns its width."""
        tiles: Tiles = d["tiles"]
        small: counting.Figures = d["small"]
        tw = small.width(parts)
        w = tw + SIDE_PAD * small.px * 2
        h = small.cap * 1.9
        key = ("side", round(w), round(h))
        if key not in d:
            d[key] = textmod.rounded_box(w, h, 0.22 * h, fill=TILE, opacity=1.0)
        box = d[key]
        top = y + (tiles.th - h) / 2
        over(frame, box, x, top, alpha)
        small.draw(frame, parts, x + SIDE_PAD * small.px, top + (h + small.cap) / 2, alpha)
        return w

    def _draw_tiles(self, frame, show: Show, d: dict, t: float) -> None:
        s, r = d["s"], show.rect
        alpha, rise = self._fade(show, t)
        tiles: Tiles = d["tiles"]
        small: counting.Figures = d["small"]
        figs: counting.Figures = d["figs"]
        b, old, new, q = self._tile_state(show, t)
        pre, digs, tail = b.parts[0], b.parts[1], b.parts[2] + b.parts[3]
        gap = 0.07 * tiles.px
        # every number on one counter shares its row of tiles: the widest number's, unreached tiles blank
        least = max(sum(c.isdigit() for c in x.parts[1]) for x in show.beats)
        blanks = least - sum(c.isdigit() for c in digs)
        width = self._tile_width(b.parts, tiles, small, figs, least)
        x = r.x0 + (r.w - width) / 2
        y = r.y0 + tiles.m + rise * 14 * s
        mid = y + (tiles.th + small.cap) / 2  # the small sign and unit sit level with the tiles' middle
        if pre:
            x += self._side(frame, d, ("", "", "", pre), x, y, alpha) + 0.07 * tiles.px
        first = True
        for _ in range(blanks):
            if not first:
                x += gap
            first = False
            tiles.draw(frame, x, y, " ", " ", 1.0, alpha)
            x += tiles.tw
        for c_final, c_old, c_new in zip(digs, old.ljust(len(digs)), new.ljust(len(digs))):
            if not c_final.isdigit():
                g = figs.glyph(c_final)
                over(frame, g.sprite, x - g.ox, y + (tiles.th + tiles.cap) / 2 - g.oy, alpha)
                x += g.width
                continue
            if not first:
                x += gap
            first = False
            co = c_old if c_old in tiles.faces else " "
            cn = c_new if c_new in tiles.faces else " "
            tiles.draw(frame, x, y, co, cn, q if co != cn else 1.0, alpha)
            x += tiles.tw
        if tail:
            self._side(frame, d, ("", "", "", tail), x + 0.07 * tiles.px, y, alpha)
        if "label" in d:
            w = d["label"]
            box_h = d["label_cap"] * 1.9
            by = r.y0 + tiles.th + 2 * tiles.m + LABEL_GAP * s * 0.5 + rise * 14 * s
            if "pill" not in d:
                d["pill"] = textmod.rounded_box(w.width + 1.2 * box_h, box_h, box_h / 2, fill=TILE, opacity=0.92)
            pill = d["pill"]
            over(frame, pill, r.cx - pill.w / 2, by, alpha)
            over(frame, w.sprite, r.cx - w.width / 2 - w.ox, by + (box_h + d["label_cap"]) / 2 - w.oy, alpha)

    # -- sound --------------------------------------------------------------------------------------------
    def hits(self) -> list[tuple[float, object, float]]:
        out = []
        for show in self.shows:
            if self.style == "roll":
                segs = [(b.count_at, b.at, b.count_from or 0.0, b.value) for b in show.beats]
                for tm, is_land in counting.tick_times(segs, _ease_out, show.start, show.end):
                    out.append((tm, counting.land if is_land else counting.tick, 2.0 if is_land else 0.0))
            else:
                for k, b in enumerate(show.beats):
                    for at, _ in self._steps(show, k)[1]:
                        if show.start <= at < show.end:
                            last = at >= b.at - FLIP - 1e-6
                            out.append((at + FLIP * 0.55, _clack, 3.0 if last else 0.0))
        return sorted(out, key=lambda h: h[0])

    def mix(self, sound: np.ndarray, rate: int) -> np.ndarray:
        if not self.opts["sound"]:
            return sound
        hits = self.hits()
        if not hits:
            return sound
        under = TICK_UNDER if self.style == "roll" else CLACK_UNDER
        out, self.hit_db = counting.mix_under(sound, rate, hits, under)
        return out

    # -- checks and plan ----------------------------------------------------------------------------------
    def report(self) -> dict:
        over_face = outside = 0
        for j in range(self.n):
            t = j / self.fps
            f = self.faces[j]
            for show in self.shows:
                if not (show.start <= t < show.end):
                    continue
                if f is not None and show.rect.intersects(f):
                    over_face += 1
                if not self.zone.contains(show.rect.grow(*self._halo(show.k))):
                    outside += 1
        lands = []
        for show in self.shows:
            for b in show.beats:
                on = f" on “{b.word or b.said}”" if (b.word or b.said) else ""
                lands.append(f"{num.Unit.join(b.parts)} at {b.at:.2f} s (frame {math.ceil(b.at * self.fps - 1e-6)}){on}")
        out = {"frames_over_face": over_face, "frames_outside_safe_zone": outside, "style": self.style,
               "found": self.found, "lands": lands,
               "placed": [f"{s.where} at {s.k:.0%}, {s.start:.2f}-{s.end:.2f} s" for s in self.shows],
               "warnings": list(self.warnings) + [n for s in self.shows for n in s.notes]}
        if self.style == "roll":
            out["type_contrast"] = ", ".join(f"{d['contrast']:.1f} : 1" for d in self.drawn) + " (white on the glass, brightest)"
        return out

    def plan(self) -> dict:
        return {"style": self.style, "found": self.found, "shows": [
            {"label": s.label, "where": s.where, "rect": s.rect.as_list(), "size": round(s.k, 3),
             "start": round(s.start, 3), "end": round(s.end, 3),
             "runs_on": all(b.runs_on for b in s.beats[1:]) if len(s.beats) > 1 else True,
             "beats": [{"text": num.Unit.join(b.parts), "at": round(b.at, 3), "said": b.said, "how": b.how,
                        "count_from": round(b.count_from or 0.0, 4), "count_at": round(b.count_at, 3)}
                       for b in s.beats]} for s in self.shows]}

    def sheet_times(self) -> list[float]:
        """Six moments: the counter arriving, mid-count, the land frame, and the hold, over the first
        shows (the countdown: each number just after it lands)."""
        out = []
        for show in self.shows[:3]:
            b0, bl = show.beats[0], show.beats[-1]
            if self.style == "countdown":
                out += [b.at + 0.05 for b in show.beats] + [bl.at - FLIP * 0.5]
            else:
                out += [(b0.count_at + b0.at) / 2, b0.at + 0.5 / self.fps]
                if len(show.beats) > 1:
                    out += [(bl.count_at + bl.at) / 2, bl.at + 0.5 / self.fps]
                out.append(min(show.end - 0.2, bl.at + 0.8))
        out = sorted(set(round(min(self.seconds - 0.5 / self.fps, max(0.0, x)), 3) for x in out))
        while len(out) > 6:
            out.pop(len(out) // 2)
        return out


def _clack(rate: int, rng: np.random.Generator, level: float) -> np.ndarray:
    """One flap falling: a dry tap of card on card with a little low knock behind it, made here."""
    n = int(0.05 * rate)
    t = np.arange(n) / rate
    tap = rng.standard_normal(n) * np.exp(-t / 0.0022)
    # brighten the tap: a first difference takes the low end out of the noise
    tap = np.concatenate([[0.0], np.diff(tap)]) * 0.6
    knock = np.sin(2 * np.pi * rng.uniform(310, 360) * t) * np.exp(-t / 0.009) * 0.55
    s = tap + knock
    s *= np.minimum(1.0, t / 0.0003)
    return (s / max(1e-9, np.abs(s).max()) * level).astype(np.float32)


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
        counter = Counter(clip, opts, (fr.out_w, fr.out_h), info.rate, tl.output_frames(info.rate),
                          head_in_output(clip, tl, fr), keepout_in_output(clip, tl, fr), frame_at, plan=plan)
    except ValueError as e:
        if plan is not None and not isinstance(e, O.OptionError):
            raise PlanError(str(e)) from None
        raise
    out_plan = {"effect": NAME, "clip": clip.path, "options": opts,
                "output": {"width": fr.out_w, "height": fr.out_h, "fps": round(info.rate, 3),
                           "seconds": round(tl.duration, 3), "framing": fr.note},
                "counter": counter.plan(), "sheet_times": counter.sheet_times()}

    def after(result: dict, out) -> dict:
        if opts["sound"] and getattr(counter, "hit_db", None) is not None:
            what = "ticks under the roll" if counter.style == "roll" else "clacks, one for each flip"
            return {"sound": f"{len(counter.hits())} {what}, their loudest {-counter.hit_db:.0f} dB under the voice"}
        return {}

    times = counter.sheet_times()
    return Job(NAME, clip, tl, fr, counter.draw, out_plan, checks=counter.report(),
               still_times=lambda: times[:5], after=after, sound=counter.mix)
