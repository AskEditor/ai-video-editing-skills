"""headline: big type over the clip on the words that matter.

How a line becomes a headline:

1. **The words.** `when` names the moment: words that are said ("bigger captions"), a time in
   seconds, `start`, `end`, or `auto`, which picks the line that makes the point: a short clause
   said whole, with the most specific words in it, that can be set large where it will sit; never
   a line the speaker takes back or a first attempt said again, and a scrap of a sentence the clip
   came in on halfway only when nothing else stands (the core's picker, `fx/core/moments.py`). `text`
   is what the headline says: by default the words said there, as said; or your own words, which
   land on the words said at `when`, and otherwise open the video. Every time comes from the
   corrected word times (the transcriber marks a word's start late; the core pulls it back and
   checks it against the sound).
2. **The lines.** One to three lines, at most six words a line and twelve in all, set large: two
   to three times the height of a caption, as wide as most of the safe width. The break keeps the
   type as large as the room allows, the lines close in width, a comma at a line's end rather than
   "the", and never a short word ("of", "in") alone on a line. `|` in the text sets the lines
   yourself. One key word takes the accent: the thing the line names (the measure words-behind
   sets its word by), or the word you put between asterisks (`*captions*`).
3. **The look, laid out once.** `block` (heavy capitals with an outline), `serif` (editorial, the
   key word in italic), `highlight` (each line on a box that hugs it, the key word on a coloured
   one) or `sticker` (die-cut capitals, each word a little tilted). The whole headline is laid out
   at its final size and place before anything is drawn, and each word appears in that place as
   it is said, by the look's own move (`slam`, `rise`, `type` or `wipe`; `animation` picks
   another). Nothing else moves: a highlight's box grows with its words and never shows empty.
4. **The place.** Above the head or below it, never over the face: what it keeps clear of is taken
   over every frame the headline is on screen, so a turn of the head never brings the face under
   the words, and a slam that starts large starts smaller where its first frames would reach the
   face. It keeps clear of the whole head, hair included, where that leaves room for large type;
   on a close shot where it does not, it may cross the top of the hair, but never the face from
   the hairline to the chin. Always inside the safe zone. A headline that still cannot fit is made
   smaller, and one that cannot fit at all is put in the margin and the run says so.
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core import moments
from ..core import text as textmod
from ..core.compose import Sprite, over
from ..core.layout import Rect, SafeZone
from ..core.words import HESITATIONS, Word, normalize, pause_after
from . import options as O
from .captions import _LEADS_ON, _NOT_KEY, head_in_output

NAME = "headline"
STYLES = ("block", "serif", "highlight", "sticker")
MOVES = ("slam", "rise", "type", "wipe")

OPTIONS = {
    "text": O.Text("what the headline says. Empty: the words said at `when`, as said. Your own words land on the "
                   "words said at `when`; | breaks a line, *stars* pick the word in the accent colour"),
    "when": O.Text("when it lands: auto (the line that makes the point), words that are said (\"bigger captions\"), "
                   "a time in seconds such as 3.2, start or end"),
    "style": O.Choice(STYLES, "block: heavy capitals with an outline · serif: editorial, the key word in italic · "
                              "highlight: each line on a box, the key word on a coloured one · sticker: die-cut "
                              "capitals, each word a little tilted"),
    "animation": O.Choice(("auto",) + MOVES, "how each word lands as it is said: auto (the style's own: block and "
                                             "sticker slam, serif rises, highlight wipes) · slam · rise · type · wipe"),
    "accent": O.Colour("the key word's colour: auto (the style's own), a name such as yellow, green or pink, or a "
                       "hex colour such as #ffd400"),
    "size": O.Number(0.5, 1.6, "how big the type is: 1 is the style's own; 1.3 bigger, 0.8 smaller (it is made "
                               "smaller by itself when there is no room)"),
    "duration": O.Number(0, 30, "seconds on screen; 0 keeps it up until its last word is said, plus a beat"),
    "position": O.Choice(("auto", "top", "bottom"), "auto: above the head when there is room, else below it · "
                                                    "top · bottom. Never over the face, whichever you pick"),
    "fix": O.Text("misheard words as wrong=right, separated by ;  e.g. \"Jon=John; ask editor=AskEditor\""),
}

MAX_WORDS = 12  # in all
MAX_PER_LINE = 6
MAX_LINES = 3
HOLD = 0.7  # how long it stays after its last word is said
MIN_ON = 1.6  # never on screen for less than this
OUT = 0.16  # the fade at the end
STAGGER = 0.12  # between words when they are not said
TYPE_RATE = 24.0  # letters a second when the words are not said
MIN_REACH = 0.3  # smaller than this share of its size, and the headline goes to the margin instead
ROOMY = 0.6  # above the head, less than this share of its size clear of the whole head lets it cross the hair
TOP_BIAS = 0.7  # above the head wins when it gets at least this share of the size below the chin
HAIRLINE = 0.3  # the face from the hairline: how far above the face box's top it reaches, in face heights


# ---- the looks ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Look:
    face: str
    key_face: str
    caps: bool
    cap: float  # cap height at size 1, as a share of the frame's width
    move: str
    accent: str
    pitch: float  # baseline to baseline, as a multiple of the cap height


LOOKS = {
    "block": Look("anton", "anton", True, 0.15, "slam", "#ffd400", 1.2),
    "serif": Look("dm-serif", "dm-serif-italic", False, 0.11, "rise", "auto", 1.42),
    "highlight": Look("poppins-extrabold", "poppins-extrabold", False, 0.085, "wipe", "#ffd400", 2.02),
    "sticker": Look("poppins-black", "poppins-black", True, 0.11, "slam", "#ff3d68", 1.55),
}
INK = (255, 255, 255)
EDGE = (14, 14, 16)
DARK_INK = (20, 20, 22)
PAPER = (255, 255, 255)
SERIF_LIGHT_ACCENT = "#ffd27a"  # on a dark background
SERIF_DARK_ACCENT = "#c2361f"  # on a bright one


# ---- finding the words ----------------------------------------------------------------------------

_NUMBERS = {str(i): w for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}


def _same(a: str, b: str) -> bool:
    """Two words that are the same word as far as finding a phrase goes: case, apostrophes, a plural
    s, a number written either way, or one or two letters the transcriber got wrong in a long word."""
    if a == b:
        return True
    a, b = _NUMBERS.get(a, a).replace("'", ""), _NUMBERS.get(b, b).replace("'", "")
    if a == b:
        return True
    if min(len(a), len(b)) >= 4 and a.rstrip("s") == b.rstrip("s"):
        return True
    return min(len(a), len(b)) >= 5 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.84


def _keys(text: str) -> list[str]:
    return [k for k in (normalize(x) for x in re.split(r"[\s|*]+", text)) if k]


def find_said(words: list[Word], text: str, near: Optional[float] = None) -> Optional[tuple[int, int, int]]:
    """Where `text` is said: (first word, last word, how many of its words were heard), or None.

    The words must come in order and close together (at most a second and a bit a word, so a "the"
    said four seconds earlier is never taken for the start of the phrase). An "um" inside is
    passed over, as is one word the transcriber added or one it got wrong in a phrase of four or
    more. The closest match wins; between equally close ones, the one nearest `near`, else the first.
    """
    want = _keys(text)
    n = len(want)
    if not n or not words:
        return None
    keys = [w.key for w in words]
    allowed_miss = 1 if n >= 4 else 0
    found = []
    for i in range(len(words)):
        for k0 in ((0, 1) if allowed_miss else (0,)):  # the phrase's first word may be the one misheard
            if not _same(keys[i], want[k0]):
                continue
            k, j, skips, missed, last = k0 + 1, i, 0, k0, i
            while k < n and j + 1 < len(words):
                nxt = keys[j + 1]
                if nxt in HESITATIONS:
                    j += 1
                    continue
                if _same(nxt, want[k]):
                    j, last, k = j + 1, j + 1, k + 1
                    continue
                if skips < 1 and j + 2 < len(words) and _same(keys[j + 2], want[k]):
                    j, skips = j + 1, skips + 1  # a word heard that the phrase does not have
                    continue
                if missed < allowed_miss and k + 1 < n and _same(nxt, want[k + 1]):
                    k, missed = k + 1, missed + 1  # a word of the phrase heard as something else
                    continue
                if missed < allowed_miss and k == n - 1:
                    k, missed = k + 1, missed + 1  # the last word misheard
                    continue
                break
            if k < n:
                continue
            span = words[last].end - words[i].start
            if span > 1.2 * n + 1.5:
                continue
            gap = abs(words[i].start - near) if near is not None else 0.0
            found.append(((missed + skips, gap, words[i].start, span), (i, last, n - missed)))
    if not found:
        return None
    return min(found)[1]


def _runs(words: list[Word], quiet) -> list[list[int]]:
    """The words split where a sentence ends, a clause ends, or the voice really stops."""
    runs, cur = [], []
    for i, w in enumerate(words):
        if w.key in HESITATIONS:
            if cur:
                runs.append(cur)
                cur = []
            continue
        cur.append(i)
        if w.ends_clause or pause_after(words, i, quiet) >= 0.3:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def _parse_time(when: str) -> Optional[float]:
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*s?\s*", when)
    return float(m.group(1)) if m else None


# ---- the words to draw --------------------------------------------------------------------------

@dataclass
class Token:
    text: str  # as drawn: the style sets the case
    key: bool = False
    word: Optional[Word] = None  # the spoken word it shows, when there is one


def _edge_clean(t: str) -> str:
    t = t.strip().replace("’", "'").replace("‘", "'")
    t = t.lstrip("\"“(").rstrip("\"”)")
    return t.rstrip(",;:…-—").rstrip(".") if not t.endswith(("?", "!")) else t


def tokens_from_words(ws: list[Word]) -> list[Token]:
    toks = []
    for k, w in enumerate(ws):
        t = w.clean.strip()
        if k < len(ws) - 1:
            t = t.rstrip(".")  # a full stop inside a line reads as a typo on a headline
        toks.append(Token(t, word=w))
    toks[-1].text = _edge_clean(toks[-1].text)
    toks[0].text = toks[0].text.lstrip("\"“(")
    return [t for t in toks if t.text]


def tokens_from_text(text: str) -> tuple[list[list[Token]], bool]:
    """The person's own words: lines where they put `|`, the accent where they put *stars*.
    Returns the lines (one line when no `|`) and whether the lines were set by hand."""
    parts = [p for p in (s.strip() for s in text.split("|")) if p]
    lines = []
    for p in parts:
        toks = []
        for raw in p.split():
            key = raw.startswith("*") or raw.endswith("*") or "*" in raw
            word = raw.replace("*", "")
            if word:
                toks.append(Token(word, key=key))
        if toks:
            lines.append(toks)
    return lines, len(parts) > 1


def _bind_words(toks: list[Token], ws: list[Word]) -> None:
    """Lay the spoken words onto the person's tokens, in order, by matching what was said."""
    j = 0
    for t in toks:
        k = normalize(t.text)
        for jj in range(j, min(len(ws), j + 3)):
            if _same(ws[jj].key, k):
                t.word, j = ws[jj], jj + 1
                break


# ---- drawing one word ---------------------------------------------------------------------------

@dataclass
class Ink:
    """Words drawn once: their picture, where their baseline starts, and what the moves need."""
    rgba: np.ndarray
    ox: float  # the start of the baseline inside the picture
    oy: float
    advance: float  # the pen's travel
    cuts: list[float]  # after each letter, the x of its right edge (from the baseline's start)
    spans: list[tuple[float, float]]  # each token's pen start and end (from the baseline's start)


def draw_line(toks: list[Token], faces: list[str], fills: list[tuple], px: int, space: float,
              stroke: float = 0.0, stroke_fill=(0, 0, 0), shadow: Optional[tuple] = None,
              glow: Optional[tuple] = None) -> Ink:
    """`shadow` is (dx, dy, blur, opacity) in pixels, dark; `glow` is (blur, opacity, rgb), a soft
    halo behind the letters that keeps them readable on a busy background."""
    from PIL import Image, ImageDraw, ImageFilter

    fonts = [textmod.font(f, px) for f in faces]
    sw = int(round(stroke))
    pens, x = [], 0.0
    for t, f in zip(toks, fonts):
        pens.append(x)
        x += f.getlength(t.text) + space
    advance = x - space
    boxes = [f.getbbox(t.text, anchor="ls", stroke_width=sw) for t, f in zip(toks, fonts)]
    l = min(p + b[0] for p, b in zip(pens, boxes))
    t_ = min(b[1] for b in boxes)
    r = max(p + b[2] for p, b in zip(pens, boxes))
    b_ = max(b[3] for b in boxes)
    pad = 2 + sw
    if shadow:
        pad += int(math.ceil(shadow[2] * 3 + max(abs(shadow[0]), abs(shadow[1]))))
    if glow:
        pad += int(math.ceil(glow[0] * 3))
    w, h = int(math.ceil(r - l)) + 2 * pad, int(math.ceil(b_ - t_)) + 2 * pad
    ox, oy = pad - l, pad - t_
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    def mask_of(dx: float, dy: float) -> "Image.Image":
        m = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(m)
        for tok, f, p in zip(toks, fonts, pens):
            d.text((ox + p + dx, oy + dy), tok.text, font=f, fill=255, anchor="ls", stroke_width=sw, stroke_fill=255)
        return m

    for spec, colour in ((glow, None), (shadow, (0, 0, 0))):
        if not spec:
            continue
        if spec is glow:
            blur, opacity, colour = glow
            m = mask_of(0, 0).filter(ImageFilter.GaussianBlur(blur))
        else:
            dx, dy, blur, opacity = shadow
            m = mask_of(dx, dy)
            if blur > 0:
                m = m.filter(ImageFilter.GaussianBlur(blur))
        layer = np.zeros((h, w, 4), np.uint8)
        layer[..., :3] = colour
        layer[..., 3] = (np.asarray(m, np.float32) * opacity).clip(0, 255).astype(np.uint8)
        img = Image.alpha_composite(img, Image.fromarray(layer, "RGBA"))
    top = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(top)
    for tok, f, p, fill in zip(toks, fonts, pens, fills):
        d.text((ox + p, oy), tok.text, font=f, fill=tuple(fill) + (255,), anchor="ls", stroke_width=sw,
               stroke_fill=tuple(stroke_fill) + (255,))
    img = Image.alpha_composite(img, top)
    cuts, spans = [], []
    for tok, f, p in zip(toks, fonts, pens):
        for k in range(1, len(tok.text) + 1):
            cuts.append(p + f.getbbox(tok.text[:k], anchor="ls", stroke_width=sw)[2])
        spans.append((p, p + f.getlength(tok.text)))
    return Ink(np.asarray(img), ox, oy, advance, cuts, spans)


def _rotate(rgba: np.ndarray, degrees: float) -> np.ndarray:
    from PIL import Image

    if abs(degrees) < 0.05:
        return rgba
    return np.asarray(Image.fromarray(rgba, "RGBA").rotate(degrees, resample=Image.BICUBIC, expand=True))


def _ink_box(sprite: Sprite) -> tuple[float, float, float, float]:
    ys, xs = np.nonzero(sprite.a > 0.08)
    if not len(xs):
        return 0.0, 0.0, float(sprite.w), float(sprite.h)
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


# ---- the headline, laid out once ------------------------------------------------------------------

@dataclass
class Piece:
    """One word, drawn once, in its final place inside the block."""
    tok: Token
    sprite: Sprite
    x: float  # the sprite's top-left inside the block
    y: float
    ink: tuple[float, float, float, float]  # the visible part of the sprite
    cuts: list[float]  # type: the right edge after each letter, in sprite pixels
    line: int
    floor: float  # rise: the bottom of the word's slot, inside the block
    cap: float
    base: float  # the line's baseline, inside the block
    caret_rgb: tuple = INK
    on: float = 0.0  # when it appears
    until: float = 0.0  # when its saying ends

    def rect(self, s: float = 1.0) -> Rect:
        """Its visible part inside the block, scaled by `s` about the sprite's centre."""
        cx, cy = self.x + self.sprite.w / 2, self.y + self.sprite.h / 2
        x0, y0, x1, y1 = (self.x + self.ink[0], self.y + self.ink[1], self.x + self.ink[2], self.y + self.ink[3])
        return Rect(cx + (x0 - cx) * s, cy + (y0 - cy) * s, cx + (x1 - cx) * s, cy + (y1 - cy) * s)


@dataclass
class Box:
    """highlight: a line's box, or the coloured box under its key word."""
    sprite: Sprite
    x: float
    y: float
    line: int
    key: bool
    pad: float  # how far the box reaches past the words it holds


@dataclass
class Block:
    pieces: list[Piece]
    boxes: list[Box]
    w: float
    h: float
    px: int
    ink: Rect  # what is visible once every word is in, inside the block


def _cased(look: Look, toks: list[Token], first: bool = True) -> list[Token]:
    """The words in the look's case: capitals, or as written with a capital to open the headline."""
    out = []
    for i, t in enumerate(toks):
        s = t.text.upper() if look.caps else t.text
        if not look.caps and first and i == 0 and s[:1].islower():
            s = s[:1].upper() + s[1:]
        out.append(Token(s, t.key, t.word))
    return out


def _space(look: Look, style: str, px: int) -> float:
    k = 0.92 if look.caps else 1.0
    if style == "sticker":
        k = 1.3  # room for each word's white edge
    return textmod.advance(" ", look.face, px) * k


def build(style: str, lines: list[list[Token]], px: int, accent: tuple, ink_dark: bool) -> Block:
    """Every word of the headline drawn at size `px` in its final place: lines stacked and centred."""
    look = LOOKS[style]
    cap = textmod.cap_height(look.face, px)
    pitch = look.pitch * cap
    space = _space(look, style, px)
    pieces: list[Piece] = []
    boxes: list[Box] = []
    n_word = 0
    for li, raw in enumerate(lines):
        toks = _cased(look, raw, first=li == 0)
        faces = [look.key_face if t.key else look.face for t in toks]
        widths = [textmod.advance(t.text, f, px) for t, f in zip(toks, faces)]
        adv = sum(widths) + space * (len(toks) - 1)
        baseline = li * pitch
        pen = -adv / 2
        spans = []
        for (tok, face, wd), orig in zip(zip(toks, faces, widths), raw):
            if style == "block":
                ink = draw_line([tok], [face], [accent if tok.key else INK], px, 0.0, stroke=0.075 * px,
                                stroke_fill=EDGE, shadow=(0.0, 0.05 * px, 0.04 * px, 0.55))
                caret = INK
            elif style == "serif":
                base = DARK_INK if ink_dark else (255, 252, 245)
                glow = (0.16 * px, 0.75, (255, 255, 255)) if ink_dark else (0.16 * px, 0.62, (0, 0, 0))
                ink = draw_line([tok], [face], [accent if tok.key else base], px, 0.0, glow=glow)
                caret = base
            elif style == "highlight":
                key_ink = (17, 17, 17) if sum(accent) > 380 else (255, 255, 255)
                ink = draw_line([tok], [face], [key_ink if tok.key else (17, 17, 17)], px, 0.0)
                caret = (17, 17, 17)
            else:  # sticker: dark capitals cut out with a thick white edge, a soft shadow under the cut
                ink = draw_line([tok], [face], [accent if tok.key else DARK_INK], px, 0.0, stroke=0.15 * px,
                                stroke_fill=PAPER, shadow=(0.02 * px, 0.06 * px, 0.05 * px, 0.45))
                caret = DARK_INK
            rgba = ink.rgba
            x, y = pen - ink.ox, baseline - ink.oy
            if style == "sticker":
                tilt = (-2.6, 1.8, -1.2, 2.4, -2.0, 1.4)[n_word % 6]
                turned = _rotate(rgba, tilt)
                x -= (turned.shape[1] - rgba.shape[1]) / 2
                y -= (turned.shape[0] - rgba.shape[0]) / 2
                rgba = turned
            sprite = Sprite.from_rgba(rgba)
            piece = Piece(Token(tok.text, tok.key, orig.word), sprite, x, y, _ink_box(sprite),
                          [ink.ox + c for c in ink.cuts], li, 0.0, cap, baseline, caret)
            piece.floor = y + piece.ink[3] + 1
            pieces.append(piece)
            spans.append((pen, pen + wd, tok.key))
            pen += wd + space
            n_word += 1
        if style == "highlight":
            pad_x, top, foot = 0.36 * px, 0.98 * cap + 0.24 * px, 0.27 * px + 0.13 * px
            bw, bh = adv + 2 * pad_x, top + foot
            whole_key = all(k for _, _, k in spans)
            boxes.append(Box(textmod.rounded_box(bw, bh, radius=0.16 * px, fill=accent if whole_key else PAPER),
                             -bw / 2, baseline - top, li, False, pad_x))
            keys = [(a, b) for a, b, k in spans if k]
            if keys and not whole_key:
                k0, k1 = min(a for a, _ in keys), max(b for _, b in keys)
                kw, kh = (k1 - k0) + 0.3 * px, bh - 0.16 * px
                boxes.append(Box(textmod.rounded_box(kw, kh, radius=0.12 * px, fill=accent),
                                 k0 - 0.15 * px, baseline - top + 0.08 * px, li, True, 0.15 * px))
    xs0 = [p.x for p in pieces] + [b.x for b in boxes]
    ys0 = [p.y for p in pieces] + [b.y for b in boxes]
    xs1 = [p.x + p.sprite.w for p in pieces] + [b.x + b.sprite.w for b in boxes]
    ys1 = [p.y + p.sprite.h for p in pieces] + [b.y + b.sprite.h for b in boxes]
    x0, y0 = min(xs0), min(ys0)
    for p in pieces:
        p.x, p.y, p.floor, p.base = p.x - x0, p.y - y0, p.floor - y0, p.base - y0
    for b in boxes:
        b.x, b.y = b.x - x0, b.y - y0
    ink = Rect.around([p.rect() for p in pieces] + [Rect(b.x, b.y, b.x + b.sprite.w, b.y + b.sprite.h) for b in boxes])
    return Block(pieces, boxes, max(xs1) - x0, max(ys1) - y0, px, ink)


# ---- choosing the lines and the size ---------------------------------------------------------------

def _splits(n: int, parts: int):
    if parts == 1:
        yield [n]
        return
    for first in range(1, n - parts + 2):
        for rest in _splits(n - first, parts - 1):
            yield [first] + rest


def _measure(look: Look, style: str, toks: list[Token], px: int) -> float:
    toks = _cased(look, toks)
    return sum(textmod.advance(t.text, look.key_face if t.key else look.face, px) for t in toks) + \
        _space(look, style, px) * (len(toks) - 1)


EXTRA = {"block": 0.32, "serif": 0.5, "highlight": 0.72, "sticker": 0.6}  # outline, descenders, box: in px


def _block_height(look: Look, px: int, n_lines: int, style: str) -> float:
    cap = textmod.cap_height(look.face, px)
    return cap * (look.pitch * (n_lines - 1) + 1.0) + EXTRA[style] * px


def arrange(style: str, toks: list[Token], width: float, px0: int, room_h: float,
            fixed: Optional[list[list[Token]]] = None) -> tuple[list[list[Token]], int, float]:
    """The lines and the size: the break that keeps the type largest, with lines close in width,
    never a short word alone, and a line ending on a comma rather than on "the". Returns the
    lines, the size, and the share of the wanted size it reached (1 when nothing had to shrink)."""
    look = LOOKS[style]
    probe = 100
    options = []
    groups = [fixed] if fixed else []
    if not fixed:
        n = len(toks)
        for parts in range(1, min(MAX_LINES, n) + 1):
            for sizes in _splits(n, parts):
                if max(sizes) > MAX_PER_LINE:
                    continue
                out, i = [], 0
                for s in sizes:
                    out.append(toks[i:i + s])
                    i += s
                groups.append(out)
    for lines in groups:
        widths = [_measure(look, style, ln, probe) for ln in lines]
        px_w = min(px0, int(probe * width / max(widths)))
        unit = _block_height(look, probe, len(lines), style) / probe
        px_h = int(room_h / unit) if room_h > 0 else 0
        px = max(4, min(px_w, px_h))
        score = px / px0
        # size on a log scale: from half size to two thirds is worth as much as from three quarters
        # to all of it, so a better break is not traded away for a few pixels at the top of the range
        value = math.log(max(1e-3, score))
        cost = 0.0
        if not fixed:
            for k, ln in enumerate(lines):
                last = ln[-1]
                if len(ln) == 1 and (len(normalize(last.text)) <= 3 or normalize(last.text) in _LEADS_ON):
                    cost += 0.6  # a short word alone on a line towers, and reads as a mistake
                elif len(ln) == 1 and last.key and look.caps:
                    cost -= 0.04  # the key word on a line of its own: the loudest place for it
                if k < len(lines) - 1:
                    nxt = lines[k + 1][0]
                    if last.text.rstrip().endswith((",", ";", ":", ".", "?", "!", "—")):
                        cost -= 0.08
                    elif normalize(last.text) in _LEADS_ON:
                        cost += 0.06
                    elif normalize(last.text) not in _NOT_KEY and normalize(nxt.text) not in _NOT_KEY:
                        cost += 0.12  # between two words that belong together ("production | quality")
                if k == len(lines) - 1 and len(lines) > 1 and len(ln) == 1 and not last.key and len(toks) > 3:
                    cost += 0.05  # the lone last word: the eye stops on it
            cost += 0.25 * (max(widths) - min(widths)) / max(widths) if len(lines) > 1 else 0.0
            cost += 0.06 * (len(lines) - 1)
        options.append((value - cost, -len(lines), lines, px, score))
    best = max(options, key=lambda o: (o[0], o[1]))
    return best[2], best[3], best[4]


# ---- the moves ------------------------------------------------------------------------------------

def _ease_out(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 1.0 - (1.0 - x) ** 3


def _ease_in_out(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


SLAM = 0.22  # seconds: a word drops in from larger, dips under its size and settles
RISE = 0.26  # a word rises out of its slot
WIPE = 0.2  # a word is uncovered from the left (or over its saying, if that is shorter)


def _slam_scale(q: float, s0: float) -> float:
    if q < 0.62:
        return s0 - (s0 - 0.95) * (q / 0.62) ** 2
    if q < 1.0:
        return 0.95 + 0.05 * _ease_out((q - 0.62) / 0.38)
    return 1.0


def _cut_right(sprite: Sprite, width: float) -> Sprite:
    w = int(max(0, min(sprite.w, round(width))))
    return Sprite(sprite.rgb[:, :w], sprite.a[:, :w])


def _soft_reveal(sprite: Sprite, x0: float, x1: float, q: float) -> Sprite:
    """The sprite uncovered from `x0` to a soft edge `q` of the way to `x1` (anything left of `x0`,
    a shadow or a halo, comes with the first letter)."""
    if q >= 1.0:
        return sprite
    feather = max(2.0, 0.18 * (x1 - x0))
    edge = x0 + q * (x1 - x0 + feather)
    xs = np.arange(sprite.w, dtype=np.float32)
    ramp = np.clip((edge - xs) / feather, 0.0, 1.0)
    return Sprite(sprite.rgb * ramp[None, :, None], sprite.a * ramp[None, :])


# ---- the headline -----------------------------------------------------------------------------------

def keepout_in_output(clip, timeline, fr, top: float = HAIRLINE):
    """The face from the hairline to the chin and from ear to ear, at each output frame: the part a
    headline never crosses, even when it may cross the top of the hair."""
    face = clip.face
    fps, n_src = clip.info.rate, clip.info.frames

    def grow(r: Rect) -> Rect:
        return r.grow(0.2 * r.w, top * r.h, 0.2 * r.w, 0.16 * r.h)

    def at(j: int) -> Optional[Rect]:
        k = timeline.source_frame(j, fps, n_src)
        b = face.box(k)
        if b is None:
            return None
        r = grow(b)
        x, y, w, h, _ = face.trusted[int(np.clip(k, 0, face.n - 1))]
        if not np.isnan(x):
            r = r.union(grow(Rect(x, y, x + w, y + h)))
        return fr.to_output(r, k)

    return at


class Headline:
    def __init__(self, clip, opts: dict, size: tuple[int, int], fps: float, n_frames: int, head_at, face_at,
                 frame_at, plan: Optional[dict] = None):
        self.W, self.H = size
        self.fps = fps
        self.n = n_frames
        self.opts = opts
        self.style = opts["style"]
        self.look = LOOKS[self.style]
        self.move = self.look.move if opts["animation"] == "auto" else opts["animation"]
        self.zone = SafeZone(self.W, self.H)
        self.heads = [head_at(j) for j in range(n_frames)]  # the whole head, hair included
        self.faces = [face_at(j) for j in range(n_frames)]  # the face, from the hairline down
        self.frame_at = frame_at
        self.warnings: list[str] = []
        self.seconds = n_frames / fps
        self.px0 = int(round(self.look.cap * self.W * float(opts["size"]) / (textmod.cap_height(self.look.face, 100) / 100)))
        if plan is not None:
            self._from_plan(plan)
        else:
            self._decide(clip)
        self._accent_and_ink()
        self._place()
        self._time()

    # -- what it says, and when
    def _span_reach(self, words: list[Word]):
        """How large a line said from word `a` to word `b` could be set where it would sit."""
        def reach(a: int, b: int) -> float:
            toks = tokens_from_words(words[a:b + 1])
            if not toks:
                return 0.0
            t0, t1 = words[a].start, max(words[b].end, words[a].start + MIN_ON) + HOLD
            return self._best_place(toks, None, t0, min(self.seconds, t1))[0]
        return reach

    def _decide(self, clip) -> None:
        words = [w for w in clip.words if w.key not in HESITATIONS]
        # What the speaker took back, read with the fillers (which break a first attempt off): no
        # line picked by itself holds it, and no key word is one.
        back = moments.taken_back(clip.words, clip.quiet)
        runs = _runs(words, clip.quiet)
        text, when = self.opts["text"].strip(), self.opts["when"].strip()
        wl = when.lower()
        span: Optional[tuple[int, int]] = None  # the words said where it lands (first, last)
        at: Optional[float] = None  # where it lands when no words are said there
        if not words and not text:
            raise O.OptionError("nothing is said in this video, so there is nothing to set: give `text`")
        if wl in ("", "auto"):
            if text:
                hit = find_said(words, text)
                if hit:
                    span, found = hit[:2], "where the text is said"
                else:
                    at = words[0].start if words else 0.0
                    found = "the text is not said, so it opens the video (when= moves it)"
            else:
                span = moments.pick_line(words, clip.quiet, self._span_reach(words), back=back)
                found = "on the line picked as the one that makes the point"
                if span is None:
                    raise O.OptionError("no line stands out to set on its own: say which with when=\"…\" or give text")
        elif wl in ("start", "end"):
            if not runs:
                raise O.OptionError(f"when={wl} needs words that are said; give a time in seconds instead")
            if wl == "start":
                run = runs[0]
            else:  # the last line that is more than a word and leaves time to read it before the end
                fit = [r for r in runs if len(r) >= 2 and self.seconds - words[r[0]].start >= MIN_ON]
                run = fit[-1] if fit else runs[-1]
            found = "on the first line said" if wl == "start" else "on the last line said"
            if text:
                at = words[run[0]].start
            else:
                span = (run[0], run[-1])
        elif (t := _parse_time(when)) is not None:
            if t >= self.seconds:
                raise O.OptionError(f"when: {t:g} s is after the end of the video ({self.seconds:.2f} s)")
            found = f"at {t:.2f} s"
            if text:
                hit = find_said(words, text, near=t)
                if hit and abs(words[hit[0]].start - t) < 1.0:
                    span = hit[:2]
                else:
                    at = t
            else:
                later = [i for i, w in enumerate(words) if w.start >= t - 0.15]
                if not later:
                    raise O.OptionError(f"when: nothing is said after {t:g} s; give text to set there")
                run = next(r for r in runs if later[0] in r)
                span = (later[0], run[-1])
        else:
            hit = find_said(words, when)
            if not hit:
                raise O.OptionError(f"when: the words \"{when}\" are not said in this video. Read the words with "
                                    f"`python -m fx.run words <video>` and use them as heard, or give a time in seconds")
            span = hit[:2]
            found = f"on \"{' '.join(w.clean for w in words[hit[0]:hit[1] + 1])}\""
        spoken = words[span[0]:span[1] + 1] if span else []
        fixed = None
        if text:
            lines, by_hand = tokens_from_text(text)
            if not lines:
                raise O.OptionError("text: give the words to set")
            toks = [t for ln in lines for t in ln]
            fixed = lines if by_hand else None
            if spoken:
                _bind_words(toks, spoken)
        else:
            if len(spoken) > MAX_WORDS:
                spoken = spoken[:MAX_WORDS]
                self.warnings.append(f"the line said there is long; the headline keeps its first {MAX_WORDS} words")
            toks = tokens_from_words(spoken)
        if len(toks) > MAX_WORDS:
            raise O.OptionError(f"text: a headline holds at most {MAX_WORDS} words (this has {len(toks)}); "
                                f"keep the words that make the point, and leave the rest to captions")
        if fixed and (len(fixed) > MAX_LINES or any(len(ln) > MAX_PER_LINE for ln in fixed)):
            raise O.OptionError(f"text: at most {MAX_LINES} lines of at most {MAX_PER_LINE} words each")
        if not any(t.key for t in toks) and toks:
            ws = [t.word or Word(t.text, 0.0, 0.0) for t in toks]
            k = moments.key_word(ws, clip.quiet, back)  # the thing named: the same measure words-behind sets by
            if k is not None:
                toks[k].key = True
        self.toks, self.fixed, self.spoken, self.found = toks, fixed, spoken, found
        self.said = bool(spoken)
        self.planned_on = None
        if spoken:
            self.start = spoken[0].start
            self.said_end = spoken[-1].end
        else:
            self.start = float(at or 0.0)
            self.said_end = self.start + STAGGER * (len(toks) - 1) + 0.6

    def _from_plan(self, plan: dict) -> None:
        h = plan.get("headline")
        if not isinstance(h, dict) or not h.get("lines"):
            raise ValueError("the plan has no headline; run once without --plan to get one to edit")
        lines, ons = [], []
        for ln in h["lines"]:
            toks = [Token(t["text"], bool(t.get("key"))) for t in ln.get("tokens", [])] if isinstance(ln, dict) else []
            if not toks:
                raise ValueError("every line of the headline needs its words")
            lines.append(toks)
            ons.extend(float(t["on"]) if "on" in t else None for t in ln["tokens"])
        self.fixed = lines
        self.toks = [t for ln in lines for t in ln]
        if len(self.toks) > MAX_WORDS:
            raise ValueError(f"a headline holds at most {MAX_WORDS} words")
        self.start = float(h["start"])
        self.said_end = float(h.get("said_end", self.start + 0.9))
        self.planned_on = ons if all(o is not None for o in ons) else None
        self.planned_end = float(h["end"])
        self.said = bool(h.get("said"))
        self.spoken = []
        self.found = "from the plan"
        if not (0 <= self.start < self.seconds):
            raise ValueError(f"the headline starts at {self.start:.2f} s, outside the video")

    # -- colours
    def _accent_and_ink(self) -> None:
        a = self.opts["accent"]
        self.ink_dark = False
        if self.style == "serif":
            # The serif has no outline and no box, so its ink follows the background: dark letters on
            # a bright room, light ones on a dark room, each with a soft halo of the other.
            self.ink_dark = self._background_luma() > 0.62
            default = SERIF_DARK_ACCENT if self.ink_dark else SERIF_LIGHT_ACCENT
            self.accent = textmod.colour(default if a == "auto" else a)
        else:
            self.accent = textmod.colour(self.look.accent if a == "auto" else a)

    def _background_luma(self) -> float:
        """How bright the picture is where a headline usually sits (the band above the head, or the
        band below it), over the headline's seconds."""
        lumas = []
        for t in (self.start, (self.start + self.said_end) / 2, self.said_end):
            j = int(min(self.n - 1, max(0, round(t * self.fps))))
            frame = self.frame_at(j)
            head = self.heads[j]
            z = self.zone.rect
            if head is not None and head.y0 - z.y0 > 0.12 * self.H:
                band = frame[int(z.y0):int(head.y0), int(z.x0):int(z.x1)]
            elif head is not None:
                band = frame[int(head.y1):int(z.y1), int(z.x0):int(z.x1)]
            else:
                band = frame[int(z.y0):int(z.y0 + 0.25 * self.H), int(z.x0):int(z.x1)]
            if band.size:
                lumas.append(float((band[..., 0] * 0.2126 + band[..., 1] * 0.7152 + band[..., 2] * 0.0722).mean()) / 255)
        return float(np.mean(lumas)) if lumas else 0.5

    # -- where it goes
    def _over(self, boxes: list, t0: float, t1: float) -> Optional[Rect]:
        a = int(max(0, math.floor(t0 * self.fps)))
        b = int(min(self.n - 1, math.ceil(t1 * self.fps)))
        return Rect.around(h for h in boxes[a:b + 1] if h is not None)

    def _rooms(self, keep: Optional[Rect]) -> dict:
        z = self.zone.rect
        gap = 0.02 * self.H
        if keep is None:  # no face found: the band at the top, the usual place
            return {"top": (z.y0, z.y0 + 0.3 * self.H), "bottom": (z.y1 - 0.3 * self.H, z.y1)}
        return {"top": (z.y0, min(z.y1, keep.y0 - gap)), "bottom": (max(z.y0, keep.y1 + gap), z.y1)}

    def _best_place(self, toks, fixed, t0: float, t1: float) -> tuple[float, Optional[tuple]]:
        """The best side and size for these words over [t0, t1]. Above the head it keeps clear of
        the whole head, hair included, when that leaves room for type at least `ROOMY` of its size;
        on a close shot where it does not, it measures the room clear of the face from the hairline
        down and may cross the top of the hair. Below the chin the edge is the same either way.
        Above the head wins unless it gets less than `TOP_BIAS` of the size below the chin, where
        captions and hands are. Returns the size reached and (lines, px, reach, side, room, clearance)."""
        width = 0.96 * self.zone.rect.w
        tries = {}
        for clearance, boxes in (("head", self.heads), ("face", self.faces)):
            r0, r1 = self._rooms(self._over(boxes, t0, t1))["top"]
            if r1 - r0 >= 0.04 * self.H:
                lines, px, reach = arrange(self.style, toks, width, self.px0, r1 - r0, fixed)
                if "top" not in tries or reach > tries["top"][2]:
                    tries["top"] = (lines, px, reach, "top", (r0, r1), clearance)
            if "top" in tries and tries["top"][2] >= ROOMY:
                break
        r0, r1 = self._rooms(self._over(self.heads, t0, t1))["bottom"]
        if r1 - r0 >= 0.04 * self.H:
            lines, px, reach = arrange(self.style, toks, width, self.px0, r1 - r0, fixed)
            tries["bottom"] = (lines, px, reach, "bottom", (r0, r1), "head")
        if not tries:
            return 0.0, None
        pos = self.opts["position"]
        if pos in tries and tries[pos][2] >= MIN_REACH:
            found = tries[pos]
        else:
            t, b = tries.get("top"), tries.get("bottom")
            found = t if t and (not b or t[2] >= TOP_BIAS * b[2]) else b
        self.sides = {k: [round(v[2], 2), v[5]] for k, v in tries.items()}
        return found[2], found

    def _place(self) -> None:
        probe_end = max(self.said_end, self.start + MIN_ON) + HOLD
        if float(self.opts["duration"]) > 0:
            probe_end = self.start + float(self.opts["duration"])
        if getattr(self, "planned_end", None) is not None:
            probe_end = self.planned_end
        self.probe_end = min(self.seconds, probe_end)
        reach, found = self._best_place(self.toks, self.fixed, self.start, self.probe_end)
        pos = self.opts["position"]
        if found and pos in ("top", "bottom") and found[3] != pos:
            self.warnings.append(f"there is no room {'above' if pos == 'top' else 'below'} the face; it went "
                                 f"{'below' if pos == 'top' else 'above'} instead")
        if not found or reach < MIN_REACH:
            self._last_resort()
            return
        lines, px, reach, side, (r0, r1), clearance = found
        self.clearance = clearance
        px = self._fit(lines, px, r1 - r0)
        self._set(lines, px, side, r0, r1, 0.62 if side == "top" else 0.38)
        if clearance == "face":
            self.warnings.append("a close shot: the headline crosses the top of the hair to stay large, and keeps "
                                 "clear of the face from the hairline down")

    def _fit(self, lines: list[list[Token]], px: int, room_h: float) -> int:
        """Build at `px`, then smaller until what is visible fits the room and the zone's width."""
        z = self.zone.rect
        self.block = build(self.style, lines, px, self.accent, self.ink_dark)
        for _ in range(40):
            ink = self.block.ink
            k = min(1.0, (room_h - 0.5) / max(1.0, ink.h), (z.w - 0.5) / max(1.0, ink.w))
            if k >= 1.0 or px <= 6:
                break
            px = max(6, int(px * min(0.97, k)))
            self.block = build(self.style, lines, px, self.accent, self.ink_dark)
        return px

    def _set(self, lines, px: int, where: str, r0: float, r1: float, lean: float) -> None:
        """Put the block's visible part in the room: across, centred on the safe zone; down, `lean`
        of the spare room from its top (the room above the head leans towards the head)."""
        ink = self.block.ink
        self.x = self.zone.rect.cx - (ink.x0 + ink.x1) / 2
        self.y = r0 + max(0.0, (r1 - r0) - ink.h) * lean - ink.y0
        self.where, self.px, self.lines = where, px, lines
        self.reach = min(1.0, px / max(1.0, self.px0))  # the share of the size asked for that it got

    def _last_resort(self) -> None:
        """No room inside the safe zone clear of the face. Covering the face is the one thing never
        done, so the headline goes small into the margin on the roomier side, and the run says so."""
        keep = self._over(self.faces, self.start, self.probe_end)
        room_above = keep.y0 if keep is not None else self.H
        room_below = self.H - keep.y1 if keep is not None else self.H
        r0, r1 = (0.015 * self.H, keep.y0 - 0.01 * self.H) if room_above >= room_below else \
                 (keep.y1 + 0.01 * self.H, 0.985 * self.H)
        lines, px, reach = arrange(self.style, self.toks, 0.96 * self.zone.rect.w, self.px0, max(10.0, r1 - r0), self.fixed)
        self.clearance = "face"
        px = self._fit(lines, px, max(10.0, r1 - r0))
        self._set(lines, px, "outside the safe zone", r0, r1, 0.5)
        self.warnings.append("there was no room for the headline inside the safe zone clear of the face; it sits "
                             "in the margin, smaller. Try fewer words, size=0.8, or another moment")

    @property
    def rect(self) -> Rect:
        """What is visible, in the frame, once every word is in."""
        ink = self.block.ink
        return Rect(self.x + ink.x0, self.y + ink.y0, self.x + ink.x1, self.y + ink.y1)

    # -- when each word appears
    def _time(self) -> None:
        """Each word appears as it is said. Your own words over a spoken line are spread across its
        saying; words that are not said at all follow one another a beat apart."""
        pieces = self.block.pieces
        n = len(pieces)
        if self.planned_on and len(self.planned_on) == n:
            for p, on in zip(pieces, self.planned_on):
                p.on = on
        else:
            s0, s1 = (self.spoken[0].start, self.spoken[-1].start) if self.spoken else (self.start, self.start)
            for k, p in enumerate(pieces):
                if p.tok.word is not None:
                    p.on = p.tok.word.start
                elif self.spoken:
                    p.on = s0 + (s1 - s0) * k / max(1, n - 1)
                else:
                    p.on = self.start + STAGGER * k
        prev = -math.inf
        for k, p in enumerate(pieces):  # in reading order, and never two in the same frame
            p.on = max(p.on, self.start if k == 0 else prev + 1.0 / self.fps)
            prev = p.on
        for k, p in enumerate(pieces):
            nxt = pieces[k + 1].on if k + 1 < n else None
            if p.tok.word is not None:
                p.until = max(p.on + 0.05, p.tok.word.end)
            else:
                p.until = p.on + len(p.tok.text) / TYPE_RATE
            if nxt is not None:
                p.until = min(p.until, max(p.on + 0.05, nxt))
        self.start = pieces[0].on
        dur = float(self.opts["duration"])
        if getattr(self, "planned_end", None) is not None:
            end = self.planned_end
        elif dur > 0:
            end = self.start + dur
        else:
            end = max(self.said_end, pieces[-1].until, pieces[-1].on + 0.3) + HOLD
            end = max(end, self.start + MIN_ON)
        self.end = min(self.seconds, end)
        if self.end - self.start < 0.5:
            self.warnings.append("the headline lands less than half a second before the video ends")

    # -- drawing
    def _slam_start(self, p: Piece) -> float:
        """How large a word may start its slam: never so large that its first frames reach the face
        or leave the frame."""
        keep = self._over(self.faces, p.on, p.on + SLAM)
        s = 1.4
        while s > 1.02:
            r = p.rect(s).shift(self.x, self.y)
            if (keep is None or not r.intersects(keep)) and r.x0 >= 0 and r.x1 <= self.W and r.y0 >= 0 and r.y1 <= self.H:
                break
            s -= 0.05
        return s

    def _shown(self, p: Piece, t: float) -> float:
        """How far across its ink a word is uncovered at `t`, 0 to 1 (the wipe and the type)."""
        if self.move == "wipe":
            return _ease_in_out((t - p.on) / max(0.08, min(WIPE, p.until - p.on + 0.05)))
        if self.move == "type":
            n = self._letters(p, t)
            return 1.0 if n >= len(p.cuts) else (p.cuts[n - 1] - p.ink[0]) / max(1.0, p.ink[2] - p.ink[0])
        return 1.0

    def _letters(self, p: Piece, t: float) -> int:
        per = min(0.08, max(0.03, (p.until - p.on) / max(1, len(p.cuts))))
        return min(len(p.cuts), int((t - p.on) / per) + 1)

    def draw(self, frame: np.ndarray, j: int, t: float) -> None:
        if not (self.start <= t < self.end):
            return
        if not hasattr(self, "slam_from"):
            self.slam_from = [self._slam_start(p) for p in self.block.pieces]
        fade = min(1.0, (self.end - t) / OUT)
        X, Y = self.x, self.y
        # highlight: a line's box reaches only as far as its words have come, so it never shows empty
        for b in self.block.boxes:
            edge = self._edge(b, t)
            if edge is None or edge <= b.x:
                continue
            spr = b.sprite if edge >= b.x + b.sprite.w else _cut_right(b.sprite, edge - b.x)
            over(frame, spr, X + b.x, Y + b.y, fade)
        latest = None
        for k, p in enumerate(self.block.pieces):
            if t < p.on:
                continue
            latest = p
            q = t - p.on
            if self.move == "slam":
                s = _slam_scale(q / SLAM, self.slam_from[k])
                alpha = min(1.0, q / (0.3 * SLAM)) * fade
                spr = p.sprite if abs(s - 1.0) < 1e-3 else p.sprite.scaled(s)
                cx, cy = X + p.x + p.sprite.w / 2, Y + p.y + p.sprite.h / 2
                over(frame, spr, cx - spr.w / 2, cy - spr.h / 2, alpha)
            elif self.move == "rise":
                drop = (1.0 - _ease_out(q / RISE)) * 0.55 * p.cap
                alpha = min(1.0, q / 0.08) * fade
                if drop < 0.5:  # in its place: the whole word, its halo included
                    over(frame, p.sprite, X + p.x, Y + p.y, alpha)
                else:  # coming up out of a slot whose floor is just under the letters
                    keep = int(max(0, min(p.sprite.h, p.floor - (p.y + drop))))
                    if keep > 0:
                        over(frame, Sprite(p.sprite.rgb[:keep], p.sprite.a[:keep]), X + p.x, Y + p.y + drop, alpha)
            elif self.move == "wipe":
                f = self._shown(p, t)
                if f > 0:
                    over(frame, _soft_reveal(p.sprite, p.ink[0], p.ink[2], f), X + p.x, Y + p.y, fade)
            else:  # type
                n = self._letters(p, t)
                spr = p.sprite if n >= len(p.cuts) else _cut_right(p.sprite, p.cuts[n - 1] + 2)
                over(frame, spr, X + p.x, Y + p.y, fade)
        if self.move == "type" and latest is not None:
            n = self._letters(latest, t)
            typing = n < len(latest.cuts)
            if typing or (t - latest.until < 0.6 and int(t * 2.5) % 2 == 0):
                x = latest.x + (latest.cuts[n - 1] if latest.cuts else latest.ink[2]) + 0.05 * self.block.px
                h = latest.cap * 1.1
                y = latest.base - latest.cap * 1.05
                bar = Sprite(np.full((max(1, int(h)), max(2, int(0.07 * self.block.px)), 3), latest.caret_rgb, np.float32),
                             np.ones((max(1, int(h)), max(2, int(0.07 * self.block.px))), np.float32))
                over(frame, bar, X + x, Y + y, fade)

    def _edge(self, b: Box, t: float) -> Optional[float]:
        """How far right a highlight box reaches at `t`, inside the block: past the words of its line
        that have appeared, as far as each has been uncovered (None when none has)."""
        right = None
        for p in self.block.pieces:
            if p.line != b.line or t < p.on:
                continue
            if b.key and not p.tok.key:
                continue
            f = self._shown(p, t)
            r = p.x + p.ink[0] + f * (p.ink[2] - p.ink[0])
            right = r if right is None else max(right, r)
        if right is None:
            return None
        if b.key:
            first = min(p.x + p.ink[0] for p in self.block.pieces if p.line == b.line and p.tok.key)
            return b.x + (right - first) + 2 * b.pad if right > first else None
        return right + b.pad

    # -- what a person (or the matrix) checks
    def drawn_rect(self, t: float) -> Optional[Rect]:
        """The box of what is visible at `t`, moves included."""
        if not (self.start <= t < self.end):
            return None
        if not hasattr(self, "slam_from"):
            self.slam_from = [self._slam_start(p) for p in self.block.pieces]
        out = None
        for k, p in enumerate(self.block.pieces):
            if t < p.on:
                continue
            s = max(1.0, _slam_scale((t - p.on) / SLAM, self.slam_from[k])) if self.move == "slam" else 1.0
            r = p.rect(s)
            out = r if out is None else out.union(r)
        for b in self.block.boxes:
            edge = self._edge(b, t)
            if edge is not None and edge > b.x:
                r = Rect(b.x, b.y, min(b.x + b.sprite.w, edge), b.y + b.sprite.h)
                out = r if out is None else out.union(r)
        return None if out is None else out.shift(self.x, self.y)

    def report(self) -> dict:
        over_face = over_head = outside = 0
        for j in range(self.n):
            r = self.drawn_rect(j / self.fps)
            if r is None:
                continue
            if self.faces[j] is not None and r.intersects(self.faces[j]):
                over_face += 1
            elif self.heads[j] is not None and r.intersects(self.heads[j]):
                over_head += 1
            if not self.zone.contains(self.rect):
                outside += 1
        return {"frames_over_face": over_face, "frames_over_hair": over_head, "frames_outside_safe_zone": outside,
                "placed": self.where, "lands": f"{self.start:.2f} s ({self.found})",
                "on_screen": f"{self.start:.2f}-{self.end:.2f} s",
                "size": f"{self.reach:.0%} of the size asked for, cap height {self._cap_share():.1%} of the frame",
                "warnings": list(self.warnings)}

    def _cap_share(self) -> float:
        return textmod.cap_height(self.look.face, self.block.px) / self.H

    def plan(self) -> dict:
        pieces = iter(self.block.pieces)
        lines = []
        for ln in self.lines:
            toks = []
            for t in ln:
                p = next(pieces)
                tok = {"text": t.text, "key": t.key, "on": round(p.on, 3)}
                if t.word is not None:
                    tok["said_at"] = round(t.word.start, 3)
                toks.append(tok)
            lines.append({"tokens": toks})
        return {"text": " | ".join(" ".join(t.text for t in ln) for ln in self.lines),
                "style": self.style, "animation": self.move, "where": self.where, "clearance": self.clearance,
                "room": getattr(self, "sides", {}),
                "rect": self.rect.as_list(), "start": round(self.start, 3), "end": round(self.end, 3),
                "said_end": round(self.said_end, 3), "said": self.said, "px": self.px, "lines": lines}

    def sheet_times(self) -> list[float]:
        """Six moments across the headline's seconds: the first word arriving, the words coming in,
        the last one landing, the whole headline, and the way out."""
        ps = self.block.pieces
        s, e = self.start, self.end
        mid = ps[len(ps) // 2]
        last = ps[-1]
        picks = [ps[0].on + 0.06, mid.on + 0.05, last.on + 0.04, last.on + 0.35, (last.on + e) / 2, e - 0.05]
        return sorted(min(e - 0.5 / self.fps, max(s, x)) for x in picks)


# ---- the effect --------------------------------------------------------------------------------

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
        hl = Headline(clip, opts, (fr.out_w, fr.out_h), info.rate, tl.output_frames(info.rate),
                      head_in_output(clip, tl, fr), keepout_in_output(clip, tl, fr), frame_at, plan=plan)
    except ValueError as e:
        if plan is not None and not isinstance(e, O.OptionError):
            raise PlanError(str(e)) from None
        raise
    out_plan = {"effect": NAME, "clip": clip.path, "options": opts,
                "output": {"width": fr.out_w, "height": fr.out_h, "fps": round(info.rate, 3),
                           "seconds": round(tl.duration, 3), "framing": fr.note},
                "headline": hl.plan(), "sheet_times": [round(t, 3) for t in hl.sheet_times()]}
    return Job(NAME, clip, tl, fr, hl.draw, out_plan, checks=hl.report(), still_times=lambda: hl.sheet_times()[1:5])
