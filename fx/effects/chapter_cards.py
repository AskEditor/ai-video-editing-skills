"""chapter-cards: the parts of a talk on screen. A title lands in the quiet before each part, and a
thin rail along the top fills as the video plays, one segment a part, so a viewer always sees how far
in they are and where the next part begins.

How it works:

1. **The parts** come from the core (`fx/core/parts.py`): where the speaker moves on to a new idea,
   read from the pauses, the signposts ("first", "next", "now let's", "finally") and the words
   changing. Each part is titled by its own key words, or by `titles=`, which can also say how many
   parts there are and when each begins.
2. **A card lands in the pause** before each part from the second on; the first part's opening line
   belongs to the hook. The card arrives after the voice stops and is gone before it starts again,
   so it is never on screen over speech. In a long pause it comes at the end, leading straight into
   the part. Where the pause is too short to read a title (under a second), the part's title hangs
   small from the rail, at the mark where the part begins, for a few seconds instead; `cards=corner`
   always does that.
3. **Nothing covers the face.** A card goes in the band above the head or below the chin, whichever
   holds it larger, clear of the head over every frame it is on screen. On a tight shot it may
   cross the top of the hair, never the face from the hairline to the chin. A small tag moves along
   the rail, or to the bottom of the safe zone, until it is clear of the face. Everything stays
   inside the safe zone.
4. **The rail** sits just under the app's own top bar, across the safe width: thin enough to stay
   out of the way, with a shadow so it reads on a bright wall. When the face rises into its band (a
   camera lifted mid-take), the rail fades out and comes back after.

Two looks of our own, in the pack's own fonts: `bold` (heavy capitals, the number on a coloured tab,
a segmented rail) and `serif` (an editorial serif beside a large number, a hairline rail with a dot
where each part begins).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from ..core import parts as partsmod
from ..core import text as textmod
from ..core.compose import Sprite, over
from ..core.layout import Rect, SafeZone
from ..core.words import Word
from . import options as O
from .captions import head_in_output
from .headline import keepout_in_output

NAME = "chapter-cards"
LOOKS = ("bold", "serif")


@dataclass(frozen=True)
class MinPart:
    """auto, or the shortest a part may be in seconds."""
    help: str

    def parse(self, raw: Any) -> Any:
        v = str(raw).strip().lower().rstrip("s")
        if v in ("auto", ""):
            return "auto"
        try:
            x = float(v)
        except ValueError:
            raise O.OptionError(f"say auto or give seconds from 4 to 600 (got {raw!r})") from None
        if not 4 <= x <= 600:
            raise O.OptionError(f"give seconds from 4 to 600 (got {raw!r})")
        return x


OPTIONS = {
    "titles": O.Text("auto: each part named by its own key words · a number such as 4: that many parts · or your own "
                     "titles between | marks, each may begin with the time its part starts: "
                     "\"Intro | 0:31 The train | 1:05 Food\""),
    "cards": O.Choice(("full", "corner", "off"), "full: a big title in the pause before each part, or small under the "
                                                 "rail where the pause is too short · corner: always the small one · "
                                                 "off: no titles, the rail only"),
    "rail": O.Switch("on: a thin line under the app's top bar fills as the video plays, one segment a part · off"),
    "look": O.Choice(LOOKS, "bold: heavy capitals, the number on a coloured tab, a segmented rail · serif: an "
                            "editorial serif beside a large number, a hairline rail with a dot at each part"),
    "min_part": MinPart("the shortest a part may be: auto (a tenth of the video, 6 to 15 s) or seconds, such as 12"),
    "color": O.Colour("the accent (the number's tab, the rail's fill): auto, a name such as yellow or pink, or a hex "
                      "colour such as #ffd400"),
}

# The clips the matrix runs, with the misheard name put right as a person would.
MATRIX = [
    ("E", "E", {}),  # close, long pauses: every card in a pause
    ("33", "33", {}),  # quick talker, short pauses: the small tag under the rail
    # the longest whole talk: 107 s, one trip told in five parts; the names put right as a person would
    ("31", "31", {"fix": "Hawaiian=Hua Hin; pat tie=patty; pat ties=patties"}),
    ("E-serif", "E", {"look": "serif"}),
    ("31-serif", "31", {"look": "serif", "fix": "Hawaiian=Hua Hin; pat tie=patty; pat ties=patties",
                            "titles": "Hua Hin | The train | The hotel | Who comes here | Food and cameras"}),
    ("A", "A", {}),  # 8 s: too short for parts, said plainly
    ("B", "B", {}),
    ("C", "C", {}),
]

# ---- the numbers that place and time things ------------------------------------------------------

RAIL_TOP = 0.104  # the rail's top, as a share of the height: just under the app's own top bar (the top tenth)
CARD_MIN = 0.9  # a pause shorter than this cannot show a card long enough to read: the small tag instead
CARD_MAX = 2.4  # the longest a card stays, at the end of a long pause
CLEAR = 0.06  # seconds between the voice and a card, on both sides
CARD_IN, CARD_OUT = 0.26, 0.18
TAG_ON, TAG_IN, TAG_OUT = 2.8, 0.2, 0.25
RAIL_FADE = 0.2  # the rail fades out and back over this, when the face rises into its band
SMALLEST_CARD = 0.5  # a card that would come out smaller than this share of its size becomes a tag


@dataclass(frozen=True)
class Look:
    name: str
    accent: str  # when color=auto
    title_face: str
    title_cap: float  # the title's cap height, as a share of the frame's height, at full size
    caps: bool
    rail_h: float  # the rail's thickness, as a share of the height
    tag_face: str
    tag_cap: float


LOOK = {
    "bold": Look("bold", "#ffd400", "anton", 0.052, True, 0.0046, "poppins-bold", 0.0165),
    "serif": Look("serif", "#f2c46d", "dm-serif", 0.044, False, 0.0030, "dm-serif", 0.0185),
}
INK = (255, 255, 255)
EDGE = (14, 14, 16)
DARK = (20, 20, 22)


# ---- drawing pieces -----------------------------------------------------------------------------------

def _shadow(px: float, k: float = 1.0) -> tuple[float, float, float, float]:
    return (0.0, 0.04 * px * k, 0.06 * px * k, 0.5)


def _tracked(text: str, face: str, px: int, fill, track: float) -> tuple[Sprite, float, float]:
    """`text` with `track` pixels between its letters: (sprite, its width, the baseline's depth in it)."""
    from PIL import Image, ImageDraw

    f = textmod.font(face, px)
    widths = [f.getlength(ch) for ch in text]
    total = sum(widths) + track * (len(text) - 1)
    top = -textmod.ink("HPRT0123456789", face, px)[1]
    pad = 2
    w, h = int(math.ceil(total)) + 2 * pad, int(math.ceil(top)) + 2 * pad + 2
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x = pad
    for ch, cw in zip(text, widths):
        d.text((x, pad + top), ch, font=f, fill=tuple(fill) + (255,), anchor="ls")
        x += cw + track
    return Sprite.from_rgba(np.asarray(img)), total, pad + top


def _bar(w: float, h: float, fill, opacity: float = 1.0, radius: Optional[float] = None) -> Sprite:
    return textmod.rounded_box(max(1.0, w), max(1.0, h), radius=h / 2 if radius is None else radius, fill=fill,
                               opacity=opacity)


def _soft_shadow(w: int, h: int, blur: float, opacity: float) -> Sprite:
    """A dark blurred box, `blur` pixels past each edge: what keeps a thin line readable on a bright wall."""
    from PIL import Image, ImageDraw, ImageFilter

    pad = int(math.ceil(blur * 2.5))
    img = Image.new("L", (w + 2 * pad, h + 2 * pad), 0)
    ImageDraw.Draw(img).rounded_rectangle((pad, pad, pad + w - 1, pad + h - 1), radius=h / 2, fill=255)
    img = img.filter(ImageFilter.GaussianBlur(blur))
    a = np.asarray(img, np.float32) / 255.0 * opacity
    rgba = np.zeros((a.shape[0], a.shape[1], 4), np.uint8)
    rgba[..., 3] = np.clip(a * 255, 0, 255).astype(np.uint8)
    return Sprite.from_rgba(rgba)


def _crop(s: Sprite, x0: int, x1: int) -> Sprite:
    x0, x1 = max(0, x0), min(s.w, x1)
    return Sprite(rgb=s.rgb[:, x0:x1], a=s.a[:, x0:x1])


def _ease_out(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 1.0 - (1.0 - x) ** 3


def _ease_in_out(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


@dataclass
class Element:
    """One piece of a card: where it rests (relative to the card's top-left) and how it arrives."""
    sprite: Sprite
    x: float
    y: float
    delay: float = 0.0  # seconds after the card begins
    dx: float = 0.0  # where it starts, relative to where it rests
    dy: float = 0.0
    grow: bool = False  # grows from its middle, top to bottom (a rule)
    pop: float = 1.0  # the scale it starts at


@dataclass
class Card:
    """A part's title on screen: a full card in the pause, or a small tag under the rail."""
    part: int  # the part's index
    kind: str  # "full" or "tag"
    on: float
    off: float
    elements: list[Element]
    w: float
    h: float
    x: float = 0.0
    y: float = 0.0
    scale: float = 1.0
    where: str = ""

    @property
    def rect(self) -> Rect:
        return Rect(self.x, self.y, self.x + self.w, self.y + self.h)


def _lines(words: list[str], widths: list[float], space: float, limit: float) -> list[list[int]]:
    """One line, or two as even as possible; a word of one or two letters never stands alone."""
    n = len(words)
    if n <= 1 or sum(widths) + space * (n - 1) <= limit:
        return [list(range(n))]
    best, cost = [list(range(n))], math.inf
    for cut in range(1, n):
        a, b = list(range(cut)), list(range(cut, n))
        wa = sum(widths[:cut]) + space * (cut - 1)
        wb = sum(widths[cut:]) + space * (n - cut - 1)
        c = abs(wa - wb) + (10 * limit if max(wa, wb) > limit else 0)
        c += sum(3 * limit for part in (a, b) if len(part) == 1 and len(words[part[0]]) <= 2)
        if c < cost:
            best, cost = [a, b], c
    return best


def _px_for_cap(face: str, cap: float) -> int:
    per = textmod.cap_height(face, 200) / 200.0
    return max(6, int(round(cap / per)))


DARK_INK = (26, 24, 30)  # the serif's letters on a bright wall
DARK_ACCENT = "#a8741c"  # its numeral there


def build_full(look: Look, number: int, total: int, title: str, H: int, width: float, accent, scale: float,
               bright: bool = False) -> Card:
    """A full card at `scale`, laid out to fit `width`: its elements, relative to its top-left. `bright`
    says the wall behind it is bright: the serif's letters then turn dark (the bold's outline reads on
    anything)."""
    els: list[Element] = []
    ink = DARK_INK if (bright and look.name == "serif") else INK
    if bright and look.name == "serif" and tuple(accent) == textmod.colour(look.accent):
        accent = textmod.colour(DARK_ACCENT)
    if look.name == "bold":
        label = title.upper()
        cap = look.title_cap * H * scale
        px = _px_for_cap(look.title_face, cap)
        words = label.split()
        for _ in range(40):
            widths = [textmod.advance(w, look.title_face, px) for w in words]
            space = 0.26 * px
            lines = _lines(words, widths, space, width)
            if max(sum(widths[i] for i in ln) + space * (len(ln) - 1) for ln in lines) <= width or px <= 8:
                break
            px = int(px * 0.94)
        cap = textmod.cap_height(look.title_face, px)
        # the tab: PART 2 in dark capitals on the accent
        tab_cap = 0.34 * cap
        tab_px = _px_for_cap("poppins-black", tab_cap)
        tab_text, tab_w, tab_base = _tracked(f"PART {number}", "poppins-black", tab_px, DARK, 0.16 * tab_px)
        pad_x, pad_y = 0.62 * tab_cap, 0.48 * tab_cap
        tab_box = _bar(tab_w + 2 * pad_x, tab_cap + 2 * pad_y, accent, radius=0.3 * (tab_cap + 2 * pad_y))
        y = 0.0
        els.append(Element(tab_box, -tab_box.w / 2, y, delay=0.0, pop=0.82))
        els.append(Element(tab_text, -tab_text.w / 2 + 1, y + pad_y + tab_cap - tab_base + 1, delay=0.0, pop=0.82))
        y += tab_box.h + 0.34 * cap
        line_gap = 0.22 * cap
        k = 0
        for li, ln in enumerate(lines):
            lw = sum(widths[i] for i in ln) + space * (len(ln) - 1)
            pen = -lw / 2
            base = y + cap
            for i in ln:
                d = textmod.draw_word(words[i], look.title_face, px, fill=INK, outline=0.045 * px, outline_fill=EDGE,
                                      shadow=_shadow(px))
                els.append(Element(d.sprite, pen - d.ox, base - d.oy, delay=0.07 + 0.055 * k, dy=0.42 * cap))
                pen += widths[i] + space
                k += 1
            y = base + line_gap
    else:
        cap = look.title_cap * H * scale
        px = _px_for_cap(look.title_face, cap)
        words = title.split()
        num_room = 0.0
        for _ in range(40):
            cap = textmod.cap_height(look.title_face, px)
            num_room = 1.9 * cap + 1.0 * cap  # the numeral and the rule beside it
            widths = [textmod.advance(w, look.title_face, px) for w in words]
            space = 0.27 * px
            lines = _lines(words, widths, space, width - num_room)
            if max(sum(widths[i] for i in ln) + space * (len(ln) - 1) for ln in lines) <= width - num_room or px <= 8:
                break
            px = int(px * 0.94)
        cap = textmod.cap_height(look.title_face, px)
        top = -textmod.ink("Hdlk", look.title_face, px)[1]
        foot = textmod.ink("gjpqy", look.title_face, px)[3]
        pitch = (top + foot) * 1.04
        block_h = top + pitch * (len(lines) - 1) + foot
        # the numeral, as tall as the block of lines, in the accent
        num_cap = max(1.55 * cap, top + pitch * (len(lines) - 1))
        num_px = _px_for_cap("dm-serif", num_cap)
        glow = None if bright else _shadow(num_px, 0.6)
        num = textmod.draw_word(str(number), "dm-serif", num_px, fill=accent, shadow=glow)
        num_cap = textmod.cap_height("dm-serif", num_px)
        rule_w = max(2.0, 0.07 * cap)
        gap_a, gap_b = 0.42 * cap, 0.5 * cap
        line_ws = [sum(widths[i] for i in ln) + space * (len(ln) - 1) for ln in lines]
        total_w = num.width + gap_a + rule_w + gap_b + max(line_ws)
        x0 = -total_w / 2
        rule_h = max(block_h, num_cap) + 0.3 * cap
        y_mid = rule_h / 2
        # numeral, centred on the rule's height
        els.append(Element(num.sprite, x0 - num.ox, y_mid + num_cap / 2 - num.oy, delay=0.0, dx=-0.35 * cap))
        rule = _bar(rule_w, rule_h, ink, opacity=0.85, radius=rule_w / 2)
        rx = x0 + num.width + gap_a
        els.append(Element(rule, rx, 0.0, delay=0.04, grow=True))
        tx = rx + rule_w + gap_b
        y_top = y_mid - block_h / 2
        for li, ln in enumerate(lines):
            line = " ".join(words[i] for i in ln)
            d = textmod.draw_word(line, look.title_face, px, fill=ink, shadow=None if bright else _shadow(px, 0.8))
            base = y_top + top + li * pitch
            els.append(Element(d.sprite, tx - d.ox, base - d.oy, delay=0.1 + 0.08 * li, dx=0.3 * cap))
    x0 = min(e.x for e in els)
    y0 = min(e.y for e in els)
    x1 = max(e.x + e.sprite.w for e in els)
    y1 = max(e.y + e.sprite.h for e in els)
    for e in els:
        e.x -= x0
        e.y -= y0
    return Card(part=number - 1, kind="full", on=0.0, off=0.0, elements=els, w=x1 - x0, h=y1 - y0, scale=scale)


def build_tag(look: Look, number: int, title: str, H: int, width: float, accent) -> Card:
    """The small one: the number and the title on a dark chip."""
    cap = look.tag_cap * H
    face = look.tag_face
    px = _px_for_cap(face, cap)
    num_face = "poppins-black" if look.name == "bold" else "dm-serif"
    num_px = _px_for_cap(num_face, cap)
    for _ in range(30):
        num = textmod.draw_word(str(number), num_face, num_px, fill=accent)
        txt = textmod.draw_word(title, face, px, fill=INK)
        pad_x = 0.75 * cap
        inner = num.width + 0.55 * cap + txt.width
        if inner + 2 * pad_x <= width or px <= 8:
            break
        px, num_px = int(px * 0.93), int(num_px * 0.93)
        cap = textmod.cap_height(face, px)
    pad_y = 0.62 * cap
    h = cap + 2 * pad_y
    w = inner + 2 * pad_x
    chip = textmod.rounded_box(w, h, radius=(h / 2 if look.name == "bold" else 0.28 * h), fill=(14, 14, 16),
                               opacity=0.8 if look.name == "bold" else 0.62)
    base = pad_y + cap
    els = [Element(chip, 0.0, 0.0),
           Element(num.sprite, pad_x - num.ox, base - num.oy),
           Element(txt.sprite, pad_x + num.width + 0.55 * cap - txt.ox, base - txt.oy)]
    return Card(part=number - 1, kind="tag", on=0.0, off=0.0, elements=els, w=w, h=h)


# ---- the rail ------------------------------------------------------------------------------------------

class Rail:
    """The thin line under the top bar: one segment a part, filled up to the moment playing."""

    def __init__(self, look: Look, zone: SafeZone, W: int, H: int, duration: float, starts: list[float], accent):
        self.look = look
        self.duration = max(1e-6, duration)
        self.x0 = zone.rect.x0
        self.x1 = zone.rect.x1
        self.h = max(2.0, look.rail_h * H)
        self.dot_d = max(6.0, 4.2 * self.h) if look.name == "serif" else 0.0
        # the line's top, low enough that a dot on it stays below the app's top bar too
        self.y = max(RAIL_TOP * H, zone.rect.y0 + max(0.0, (self.dot_d - self.h) / 2) + 1.0)
        width = self.x1 - self.x0
        self.width = width
        self.marks = [self.x_at(t) for t in starts[1:]]
        n_w, n_h = int(round(width)), int(math.ceil(self.h))
        track = np.zeros((n_h, n_w, 4), np.float32)
        fill = np.zeros((n_h, n_w, 4), np.float32)
        if look.name == "bold":
            gap = max(3.0, 0.0075 * W)
            edges = [0.0] + [m - self.x0 for m in self.marks] + [width]
            self.segments = []
            for a, b in zip(edges[:-1], edges[1:]):
                a2 = a + (gap / 2 if a > 0 else 0)
                b2 = b - (gap / 2 if b < width else 0)
                self.segments.append((a2, b2))
                seg_track = _bar(b2 - a2, self.h, (255, 255, 255), opacity=0.34)
                seg_fill = _bar(b2 - a2, self.h, accent)
                for src, dst in ((seg_track, track), (seg_fill, fill)):
                    xa = int(round(a2))
                    xb = min(n_w, xa + src.w)
                    hh = min(n_h, src.h)
                    dst[:hh, xa:xb, :3] += src.rgb[:hh, :xb - xa]
                    dst[:hh, xa:xb, 3] = np.maximum(dst[:hh, xa:xb, 3], src.a[:hh, :xb - xa])
        else:
            line_track = _bar(width, self.h, (255, 255, 255), opacity=0.45)
            line_fill = _bar(width, self.h, (255, 255, 255))
            track[:line_track.h, :line_track.w, :3] = line_track.rgb[:n_h, :n_w]
            track[:line_track.h, :line_track.w, 3] = line_track.a[:n_h, :n_w]
            fill[:line_fill.h, :line_fill.w, :3] = line_fill.rgb[:n_h, :n_w]
            fill[:line_fill.h, :line_fill.w, 3] = line_fill.a[:n_h, :n_w]
            self.segments = [(0.0, width)]
        self.track = Sprite(rgb=track[..., :3], a=track[..., 3])
        self.fill = Sprite(rgb=fill[..., :3], a=fill[..., 3])
        # a dark halo under the line, so a thin white rail still reads on a white wall
        blur = max(2.0, (1.4 if look.name == "bold" else 2.2) * self.h)
        self.shadow = _soft_shadow(n_w, n_h, blur, 0.42 if look.name == "bold" else 0.6)
        self.shadow_pad = (self.shadow.w - n_w) / 2
        # the serif rail's dots, where each part begins: hollow ahead, filled once passed
        if self.dot_d:
            d = self.dot_d
            self.dot_ahead = _ring(d, max(1.5, 0.22 * d), (255, 255, 255))
            self.dot_passed = textmod.rounded_box(d, d, radius=d / 2, fill=accent)
            self.dot_shadow = _soft_shadow(int(round(d)), int(round(d)), max(2.0, 0.35 * d), 0.4)

    def x_at(self, t: float) -> float:
        return self.x0 + (self.x1 - self.x0) * min(1.0, max(0.0, t / self.duration))

    @property
    def rect(self) -> Rect:
        r = Rect(self.x0, self.y, self.x1, self.y + self.h)
        if self.dot_d:
            r = r.grow(0, (self.dot_d - self.h) / 2, 0, (self.dot_d - self.h) / 2)
        return r

    def draw(self, frame: np.ndarray, t: float, alpha: float) -> None:
        if alpha <= 0.01:
            return
        over(frame, self.shadow, self.x0 - self.shadow_pad, self.y - self.shadow_pad, alpha)
        over(frame, self.track, self.x0, self.y, alpha)
        n = int(round(self.x_at(t) - self.x0))
        if n > 0:
            over(frame, _crop(self.fill, 0, n), self.x0, self.y, alpha)
        if self.dot_d:
            here = self.x_at(t)
            for m in self.marks:
                d = self.dot_d
                cy = self.y + self.h / 2
                over(frame, self.dot_shadow, m - self.dot_shadow.w / 2, cy - self.dot_shadow.h / 2 + 0.1 * d, alpha)
                over(frame, self.dot_passed if here >= m else self.dot_ahead, m - d / 2, cy - d / 2, alpha)


def _ring(d: float, stroke: float, colour) -> Sprite:
    from PIL import Image, ImageDraw

    k = 4
    n = max(2, int(round(d * k)))
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    dr.ellipse((0, 0, n - 1, n - 1), fill=(18, 18, 20, 150), outline=tuple(colour) + (255,), width=int(round(stroke * k)))
    img = img.resize((max(1, int(round(d))), max(1, int(round(d)))), Image.LANCZOS)
    return Sprite.from_rgba(np.asarray(img))


# ---- the whole effect ------------------------------------------------------------------------------------

class Chapters:
    def __init__(self, clip, opts: dict, size: tuple[int, int], fps: float, n_frames: int,
                 head_at: Callable[[int], Optional[Rect]], face_at: Callable[[int], Optional[Rect]],
                 found: list[partsmod.Part], frame_at: Optional[Callable[[int], np.ndarray]] = None):
        self.W, self.H = size
        self.fps = fps
        self.n = n_frames
        self.duration = n_frames / fps
        self.look = LOOK[opts["look"]]
        self.accent = textmod.colour(self.look.accent if opts["color"] in ("auto", "", None) else opts["color"])
        self.zone = SafeZone(self.W, self.H)
        self.frame_at = frame_at
        self.parts = found
        self.heads = [head_at(j) for j in range(n_frames)]
        self.faces = [face_at(j) for j in range(n_frames)]
        self.speech = [(w.start, w.end) for w in clip.words]
        self.notes: list[str] = []
        self.rail = Rail(self.look, self.zone, self.W, self.H, self.duration, [p.start for p in found], self.accent) \
            if opts["rail"] else None
        self.rail_alpha = self._rail_alpha() if self.rail else np.zeros(n_frames, np.float32)
        self.cards: list[Card] = []
        if opts["cards"] != "off":
            for p in found[1:]:
                card = None
                if opts["cards"] == "full":
                    card = self._full(p)
                if card is None:
                    card = self._tag(p)
                if card is not None:
                    self.cards.append(card)

    # -- timing
    def _frames(self, a: float, b: float) -> tuple[int, int]:
        return (int(max(0, math.floor(a * self.fps))), int(min(self.n - 1, math.ceil(b * self.fps))))

    def _over(self, boxes: list[Optional[Rect]], a: float, b: float) -> Optional[Rect]:
        f0, f1 = self._frames(a, b)
        return Rect.around(r for r in boxes[f0:f1 + 1] if r is not None)

    def _rail_alpha(self) -> np.ndarray:
        """1 where the rail shows, easing to 0 around every frame where the face reaches its band."""
        band = self.rail.rect.grow(0, 0.004 * self.H, 0, 0.004 * self.H)
        blocked = np.array([f is not None and f.intersects(band) for f in self.faces], bool)
        alpha = np.ones(self.n, np.float32)
        if blocked.any():
            idx = np.flatnonzero(blocked)
            j = np.arange(self.n)
            # distance in seconds to the nearest frame where the face is in the band
            pos = np.searchsorted(idx, j)
            left = np.where(pos > 0, j - idx[np.maximum(pos - 1, 0)], 10 ** 9)
            right = np.where(pos < len(idx), idx[np.minimum(pos, len(idx) - 1)] - j, 10 ** 9)
            dist = np.minimum(left, right) / self.fps
            alpha = np.clip((dist - 0.04) / RAIL_FADE, 0.0, 1.0).astype(np.float32)
            self.notes.append(f"the rail steps aside for {blocked.sum() / self.fps:.1f} s where the face rises into its "
                              f"band (first at {idx[0] / self.fps:.2f} s)")
        return alpha

    # -- a full card in the pause
    def _full(self, p: partsmod.Part) -> Optional[Card]:
        room = p.room
        if room is None:
            return None
        a, b = room[0] + CLEAR, room[1] - CLEAR
        if b - a < CARD_MIN:
            self.notes.append(f"part {p.index + 1} (“{p.title}”) follows a pause of {max(0.0, room[1] - room[0]):.2f} s, "
                              f"too short for a card: its title hangs from the rail instead")
            return None
        on = max(a, b - CARD_MAX)
        off = b
        number, total = p.index + 1, len(self.parts)
        width = 0.9 * self.zone.rect.w
        probe = build_full(self.look, number, total, p.title, self.H, width, self.accent, 1.0)
        top = (self.rail.rect.y1 + 0.022 * self.H) if self.rail else self.zone.rect.y0
        bottom = self.zone.rect.y1
        for avoid, label in ((self.heads, "clear of the head"), (self.faces, "over the top of the hair")):
            keep = self._over(avoid, on, off)
            best = None
            if keep is None:
                spots = [("middle", top, bottom)]
            else:
                spots = [("above the head", top, keep.y0 - 0.02 * self.H),
                         ("below the chin", keep.y1 + 0.02 * self.H, bottom)]
            for where, y0, y1 in spots:
                room_h = y1 - y0
                if room_h <= 0:
                    continue
                s = min(1.0, room_h / probe.h, self.zone.rect.w / probe.w)
                if best is None or s > best[0] + 0.05 or (abs(s - best[0]) <= 0.05 and where == "above the head"):
                    best = (s, where, y0, y1)
            if best and best[0] >= SMALLEST_CARD:
                s, where, y0, y1 = best
                card = build_full(self.look, number, total, p.title, self.H, width, self.accent, s) if s < 0.999 else probe
                # shrink a hair more if the rebuilt card came out taller than its band
                for _ in range(6):
                    if card.h <= (y1 - y0) + 0.5:
                        break
                    s *= 0.95
                    card = build_full(self.look, number, total, p.title, self.H, width, self.accent, s)
                card.x = self.zone.rect.cx - card.w / 2
                card.y = y0 + ((y1 - y0) - card.h) / 2
                if self.look.name == "serif" and self._bright(card.rect, on, off):
                    dark = build_full(self.look, number, total, p.title, self.H, width, self.accent, card.scale,
                                      bright=True)
                    dark.x, dark.y = self.zone.rect.cx - dark.w / 2, card.y + (card.h - dark.h) / 2
                    card = dark
                    self.notes.append(f"part {p.index + 1}'s card sits on a bright wall: its letters are dark")
                card.on, card.off, card.where = on, off, where + ("" if label == "clear of the head" else
                                                                  ", over the top of the hair")
                return card
        self.notes.append(f"part {p.index + 1}: no room for a card clear of the face; its title hangs from the rail")
        return None

    def _bright(self, r: Rect, a: float, b: float) -> bool:
        """Whether the picture behind `r` is bright while the card is on (three frames looked at)."""
        if self.frame_at is None:
            return False
        vals = []
        for t in (a + 0.1, (a + b) / 2, b - 0.1):
            f = self.frame_at(int(min(self.n - 1, max(0, round(t * self.fps)))))
            x0, y0 = int(max(0, r.x0)), int(max(0, r.y0))
            x1, y1 = int(min(self.W, r.x1)), int(min(self.H, r.y1))
            if x1 > x0 and y1 > y0:
                patch = f[y0:y1, x0:x1].astype(np.float32)
                vals.append(float((0.2126 * patch[..., 0] + 0.7152 * patch[..., 1] + 0.0722 * patch[..., 2]).mean()) / 255)
        return bool(vals) and sum(vals) / len(vals) > 0.6

    # -- the small tag under the rail
    def _tag(self, p: partsmod.Part) -> Optional[Card]:
        card = build_tag(self.look, p.index + 1, p.title, self.H, 0.62 * self.zone.rect.w, self.accent)
        on = max(0.0, p.start - 0.12)
        nxt = self.parts[p.index + 1].start if p.index + 1 < len(self.parts) else self.duration
        off = min(on + TAG_ON, nxt - 0.3, self.duration)
        if off - on < 1.0:
            off = min(self.duration, on + 1.0)
        z = self.zone.rect
        y_top = (self.rail.rect.y1 + 0.016 * self.H) if self.rail else z.y0
        spots = [("top left", z.x0, y_top), ("top right", z.x1 - card.w, y_top),
                 ("bottom left", z.x0, z.y1 - card.h - 0.01 * self.H)]
        if self.rail:  # first choice: hanging from the rail, at the mark where its part begins
            mark = self.rail.x_at(p.start)
            spots.insert(0, ("under the rail", min(max(mark - 0.006 * self.H, z.x0), z.x1 - card.w), y_top))
        # Each place in turn: clear of the whole head, or else of the face from the hairline down (a small
        # tag may cross the top of the hair; the rail is where it belongs, not down by the hands).
        heads, faces = self._over(self.heads, on, off), self._over(self.faces, on, off)
        for where, x, y in spots:
            r = Rect(x, y, x + card.w, y + card.h)
            if not self.zone.contains(r):
                continue
            for keep, label in ((heads, ""), (faces, ", over the top of the hair")):
                if keep is None or not r.intersects(keep, margin=0.01 * self.H):
                    card.x, card.y, card.on, card.off, card.where = x, y, on, off, where + label
                    return card
        self.notes.append(f"part {p.index + 1} (“{p.title}”): no room for its title clear of the face; it "
                          f"is only on the rail")
        return None

    # -- drawing
    def _card_alpha(self, c: Card, t: float) -> float:
        step = 1.0 / self.fps
        if t < c.on or t >= c.off:
            return 0.0
        fade_in, fade_out = (CARD_IN, CARD_OUT) if c.kind == "full" else (TAG_IN, TAG_OUT)
        a_in = _ease_out((t - c.on + step) / fade_in)
        a_out = _ease_out((c.off - t) / fade_out)
        return min(a_in, a_out)

    def draw(self, frame: np.ndarray, j: int, t: float) -> None:
        if self.rail:
            self.rail.draw(frame, t, float(self.rail_alpha[min(j, self.n - 1)]))
        for c in self.cards:
            if not (c.on <= t < c.off):
                continue
            leaving = _ease_out((c.off - t) / (CARD_OUT if c.kind == "full" else TAG_OUT))
            lift = (1.0 - leaving) * (-0.06 * c.h)
            for e in c.elements:
                k = _ease_out((t - c.on - e.delay + 1.0 / self.fps) / (CARD_IN if c.kind == "full" else TAG_IN))
                if k <= 0.0:
                    continue
                alpha = min(k, leaving)
                x = c.x + e.x + (1 - k) * e.dx
                y = c.y + e.y + (1 - k) * e.dy + lift
                if c.kind == "tag":
                    y += (1 - k) * (-0.35 * c.h)
                sprite = e.sprite
                if e.grow and k < 0.999:
                    g = _ease_in_out(k)
                    hh = max(1, int(round(sprite.h * g)))
                    top = (sprite.h - hh) // 2
                    sprite = Sprite(rgb=sprite.rgb[top:top + hh], a=sprite.a[top:top + hh])
                    y += top
                if e.pop < 0.999 and k < 0.999:
                    s = e.pop + (1 - e.pop) * _ease_out(k * 1.3)
                    scaled = sprite.scaled(s)
                    x += (sprite.w - scaled.w) / 2
                    y += (sprite.h - scaled.h) / 2
                    sprite = scaled
                over(frame, sprite, x, y, alpha)

    # -- what a person (or the matrix) checks
    def report(self) -> dict:
        over_face = outside = over_speech = rail_over_face = 0
        words = sorted(self.speech)
        for j in range(self.n):
            t = j / self.fps
            face = self.faces[j]
            if self.rail and self.rail_alpha[j] > 0.02:
                if face is not None and face.intersects(self.rail.rect):
                    rail_over_face += 1
                if not self.zone.contains(self.rail.rect):
                    outside += 1
            for c in self.cards:
                if not c.on <= t < c.off:
                    continue
                if face is not None and c.rect.intersects(face):
                    over_face += 1
                if not self.zone.contains(c.rect):
                    outside += 1
                if c.kind == "full" and self._card_alpha(c, t) > 0.02:
                    if any(s - 0.02 <= t <= e + 0.02 for s, e in words if s <= t + 0.5 and e >= t - 0.5):
                        over_speech += 1
        full = [c for c in self.cards if c.kind == "full"]
        tags = [c for c in self.cards if c.kind == "tag"]
        return {
            "parts": " · ".join(f"{p.index + 1} {partsmod.stamp(p.start)} {p.title}" for p in self.parts),
            "cards": f"{len(full)} in the pause, {len(tags)} small under the rail or in a corner",
            "frames_over_face": over_face + rail_over_face,
            "frames_outside_safe_zone": outside,
            "card_frames_over_speech": over_speech,
            "placed": {c.where: sum(1 for d in self.cards if d.where == c.where) for c in self.cards},
            "warnings": list(self.notes),
        }

    def plan(self) -> list[dict]:
        out = []
        for p in self.parts:
            entry = p.to_json()
            c = next((c for c in self.cards if c.part == p.index), None)
            if c is not None:
                entry["card"] = {"kind": c.kind, "on": round(c.on, 3), "off": round(c.off, 3), "where": c.where,
                                 "rect": c.rect.as_list(), "scale": round(c.scale, 3)}
            out.append(entry)
        return out

    def sheet_times(self) -> list[float]:
        """Six moments: each card at its fullest, and the rail between them."""
        picks = []
        for c in self.cards:
            if c.kind == "full":
                picks.append(min(c.off - 0.1, c.on + max(CARD_IN + 0.25, 0.6 * (c.off - c.on))))
            else:
                picks.append(min(c.off - 0.2, c.on + 0.9))
        picks = sorted(picks)[:5]
        fill = [self.duration * f for f in (0.08, 0.5, 0.92, 0.3, 0.7, 0.2, 0.8)]
        for t in fill:
            if len(picks) >= 6:
                break
            if all(abs(t - p) > 1.0 for p in picks):
                picks.append(t)
        return sorted(min(self.duration - 0.5 / self.fps, max(0.0, t)) for t in picks)[:6]


def _parts_from_plan(plan: dict, words: list[Word], quiet, duration: float) -> list[partsmod.Part]:
    """The parts exactly as an edited plan gives them: each start and title kept as written."""
    from . import PlanError

    raw = plan.get("parts")
    if not isinstance(raw, list) or len(raw) < 2:
        raise PlanError("the plan needs two or more parts, each with a start and a title")
    starts, titles = [], []
    for k, p in enumerate(raw):
        try:
            start = 0.0 if k == 0 else float(p["start"])
        except (KeyError, TypeError, ValueError):
            raise PlanError(f"part {k + 1} of the plan has no start time") from None
        if not 0 <= start < duration:
            raise PlanError(f"part {k + 1} starts at {start} s, outside the video ({duration:.1f} s)")
        starts.append(start)
        titles.append(str(p.get("title", "")).strip() or f"Part {k + 1}")
    if any(b <= a for a, b in zip(starts, starts[1:])):
        raise PlanError("the parts' start times must go up, one after another")
    found = []
    for k, (s0, title) in enumerate(zip(starts, titles)):
        s1 = starts[k + 1] if k + 1 < len(starts) else duration
        first = next((i for i, w in enumerate(words) if w.start >= s0 - 0.05), len(words))
        last = next((i - 1 for i, w in enumerate(words) if w.start >= s1 - 0.05), len(words) - 1)
        boundary = None
        if k:
            # the quiet that ends where this part begins, if the voice really stops there
            near = [(max(q0, s0 - CARD_MAX - 1.0), min(q1, s0)) for q0, q1 in quiet if q0 <= s0 + 0.05 and q1 >= s0 - 0.3]
            room = max(near, key=lambda r: r[1] - r[0], default=(s0, s0))
            boundary = partsmod.Boundary(after=max(0, first - 1), at=s0, room=room, pause=max(0.0, room[1] - room[0]),
                                         sentence=False, signpost="", strength=0.0, question=False)
        found.append(partsmod.Part(index=k, start=s0, end=s1, first=first, last=max(first - 1, last), title=title,
                                   given=True, boundary=boundary))
    return found


def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from ..core import framing as framing_mod
    from ..core import video
    from ..core.timeline import Timeline
    from . import Job, PlanError

    info = clip.info
    fr = framing_mod.plan(info.width, info.height, info.rate, info.frames, clip.face, shape="keep")
    tl = Timeline.whole(info.frames / info.rate)
    duration = tl.duration
    words, quiet = clip.words, clip.quiet
    if plan is not None:
        found = _parts_from_plan(plan, words, quiet, duration)
    else:
        count, given = partsmod.parse_titles(opts["titles"])
        min_part = None if opts["min_part"] == "auto" else float(opts["min_part"])
        if count == 1:
            raise O.OptionError("titles: give two or more titles (or a number of parts from 2), one for each part")
        found = partsmod.find(words, quiet, duration, min_part=min_part, count=count, titles=given)
        if count and len(found) < count:
            ask = f"give {len(found)} titles or fewer" if given else f"ask for {len(found)} parts or fewer"
            raise PlanError(f"this video holds only {len(found)} parts of at least {partsmod.SHORTEST:g} s, each "
                            f"beginning where the speaker moves on; {ask}")
    if len(found) < 2:
        if not words:
            raise PlanError("no speech was heard in this video, so it has no parts to mark")
        least = partsmod.auto_min_part(duration) if opts["min_part"] == "auto" else float(opts["min_part"])
        raise PlanError(f"this video ({duration:.1f} s) is one part: nowhere does a new idea begin with at least "
                        f"{least:g} s on either side of it. Chapter cards need a talk that moves on at least once. "
                        f"To mark parts anyway, give them: titles=\"Intro | {partsmod.stamp(duration / 2)} Second part\"")
    n_out = tl.output_frames(info.rate)

    def frame_at(j: int) -> np.ndarray:
        k = tl.source_frame(j, info.rate, info.frames)
        return fr.apply(video.read_frame(info, k), k)

    ch = Chapters(clip, opts, (fr.out_w, fr.out_h), info.rate, n_out, head_in_output(clip, tl, fr),
                  keepout_in_output(clip, tl, fr), found, frame_at)
    out_plan = {"effect": NAME, "clip": clip.path, "options": opts,
                "output": {"width": fr.out_w, "height": fr.out_h, "fps": round(info.rate, 3),
                           "seconds": round(duration, 3), "framing": fr.note},
                "parts": ch.plan(), "chapters": partsmod.chapter_lines(found, duration),
                "chapters_for_youtube": partsmod.check_chapters(partsmod.chapter_lines(found, duration), duration)
                or "these lines pass YouTube's chapter rules",
                "sheet_times": [round(t, 3) for t in ch.sheet_times()]}
    from .captions import unsure_words

    out_plan["unsure"] = unsure_words(clip)
    return Job(NAME, clip, tl, fr, ch.draw, out_plan, checks=ch.report(), still_times=lambda: ch.sheet_times()[:5])
