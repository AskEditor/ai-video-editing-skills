"""captions: word-timed captions in four looks, placed where they never cover the face.

How a transcript becomes captions:

1. **Phrases are cut where a reader expects** (`fx/core/phrasing.py`, which every caption in the
   pack shares): at the end of a sentence or a long silence, then after a comma or a real pause in
   the voice, before a word that opens a clause, before one that opens a phrase. A card never ends
   on a word that leans forward ("the", "of", "is") in the middle of a sentence, never splits what
   is said as one ("right now", a name, a number and its unit), and holds at most `max_words`; the
   look's own number may be passed by one word to keep a phrase whole. A word is alone on a card
   only when it was said alone or is the point of its line.
2. **One key word, or none.** The key word is the most specific word in the phrase: a long word
   that is not grammar or filler, a number, a name. A phrase with nothing specific enough gets no
   key word rather than a highlighted "basically".
3. **Each look lays its phrase out** (`one-word`, `stacked`, `serif-chip`, `fitted`), bounded so
   a two-letter word is never huge and a long word never overflows.
4. **Placement keeps clear of the head** over every frame the phrase is on screen, not just its
   first: below the face at chest level, else in the lower third, else above the head, always
   inside the safe zone. The clip gets one steady place; a phrase moves only if it must.
5. **A short band is not a reason to squeeze.** When the head fills the frame (a close selfie, a
   tight wide-to-tall reframe) the band clear of it can be too short for a stacked card of two or
   three lines, which used to be drawn at half size or pushed out of the safe zone. Such a card is
   set flatter instead: on one line, else on two even lines. Where an effect gives the face itself (hairline to chin) and even one
   line cannot clear the whole head, the line may cross the top of the hair, never the face, and
   the report counts those frames as over the hair.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..core import layout, moments, phrasing, text
from ..core.compose import Sprite, over
from ..core.layout import Rect, SafeZone
from ..core.moments import LEADS_ON as _LEADS_ON
from ..core.moments import NOT_KEY as _NOT_KEY
from ..core.moments import OPENS as _OPENS
from ..core.moments import bare, key_score, shown  # noqa: F401 (key_score is read by other effects)
from ..core.words import HESITATIONS, Word, pause_after
from . import options as O

NAME = "captions"
LOOKS = ("stacked", "one-word", "serif-chip", "fitted")

OPTIONS = {
    "look": O.Choice(LOOKS, "stacked: two or three lines, the key word larger · one-word: one big word at a time · "
                            "serif-chip: serif words in a dark rounded box, the spoken word brighter · "
                            "fitted: one big accent word with the rest fitted around it"),
    "color": O.Colour("the accent colour (the key word; the spoken word in serif-chip): auto, a name such as "
                      "yellow, green or pink, or a hex colour such as #ffd400"),
    "position": O.Choice(("auto", "bottom", "top"), "auto: below the face, else the lower third, else above the "
                                                    "head · bottom · top. Never over the face, whichever you pick"),
    "max_words": O.Whole(0, 12, "the most words on screen at once; 0 lets the look decide"),
    "frame": O.Choice(("keep", "9:16"), "keep the clip's own shape, or make it 9:16 following the face"),
    "size": O.Size("auto (1080x1920 at most, never enlarged) or a size such as 1080x1920"),
    "fix": O.Text("misheard words as wrong=right, separated by ;  e.g. \"Jon=John; ask editor=AskEditor\""),
}

# ---- the words no caption ever makes its key word -------------------------------------------
# Grammar, filler and words too vague to be the point of a line (`_NOT_KEY`), the words that lead
# into the next (`_LEADS_ON`) or open a clause (`_OPENS`), and how specific a word is (`key_score`)
# live in the core (`fx/core/moments.py`), where every picker reads them; they keep their names here.


def key_word(words: list[Word], quiet=None, back: Optional[moments.TakenBack] = None) -> Optional[int]:
    """The word of a phrase set in the accent, or None when nothing stands out enough: the shared
    picker's (`moments.key_word`), the one headline and words-behind set: the thing named, at the end
    of its phrase, a tie going to the later word. Never a word the speaker took back (`back`, from
    the clip's whole words: "town" of "the lively town, or the lively city"); the word said instead
    counts as what its line says new. `quiet`: the clip's quiet stretches (a pause ends a phrase)."""
    return moments.key_word(words, [] if quiet is None else quiet, back)


def _is_scrap(w: Word) -> bool:
    """A word that must never stand alone on screen or on a line."""
    return w.key in _NOT_KEY or len(w.key) <= 2


# ---- phrasing --------------------------------------------------------------------------------
# Which words share a card is the core's (`fx/core/phrasing.py`), so every caption in the pack is
# cut the same way. What stays here is how long a card is on screen.

HOLD = 0.30  # how long a phrase stays after its last word when nothing follows straight away
MIN_ON = 0.55  # no phrase flashes by quicker than this
SHRINK, SHRINK_STEPS = 0.88, 6  # a card that does not fit is drawn this much smaller, up to this many times
# A stacked card drawn smaller than this for want of height (not width) is set flatter instead,
# and one drawn smaller than OVER_HAIR_BELOW (or not placed at all) may cross the top of the hair.
FLAT_BELOW = SHRINK ** 2 - 1e-9
OVER_HAIR_BELOW = SHRINK ** 3 - 1e-9
FLAT_STEPS = 2  # a one-line card is drawn at most this many steps smaller


def phrases(words: list[Word], quiet: list[tuple[float, float]], most: int, over: int = 1) -> list[list[int]]:
    """Word indexes grouped into the phrases shown together: at most `most` words a card, or one
    more where that keeps a phrase whole (`over=0` when the person chose `most`)."""
    return phrasing.phrases(words, quiet, most, over)


# ---- the looks -------------------------------------------------------------------------------

@dataclass
class Look:
    name: str
    most: int  # words on screen at once, unless max_words says otherwise
    accent: str  # the colour when color=auto
    reveal: bool  # words appear one by one as they are said


LOOK = {
    "stacked": Look("stacked", 4, "#ffd400", True),
    "one-word": Look("one-word", 1, "#ffd400", True),
    "serif-chip": Look("serif-chip", 6, "#ffffff", False),
    "fitted": Look("fitted", 5, "#ffd400", True),
}

INK = (255, 255, 255)
EDGE = (12, 12, 12)
DIM = (150, 150, 152)  # the words of a serif chip that are not being said


@dataclass
class Piece:
    sprite: Sprite
    x: float  # top-left, relative to the card
    y: float
    on: float  # when it appears (seconds)
    word: Optional[int] = None  # the phrase's word it shows
    lit: Optional[Sprite] = None  # how it looks while it is being said (serif-chip)
    pop: bool = True


@dataclass
class Card:
    start: float
    end: float
    words: list[Word]
    key: Optional[int]
    w: float = 0.0
    h: float = 0.0
    pieces: list[Piece] = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0
    where: str = ""
    scale: float = 1.0
    flat: int = 0  # a stacked card set flatter, for a band too short for it: 1 on one line, 2 on two even lines

    @property
    def rect(self) -> Rect:
        return Rect(self.x, self.y, self.x + self.w, self.y + self.h)


def _shadow(px: float) -> tuple[float, float, float, float]:
    return (0.0, 0.045 * px, 0.055 * px, 0.55)


def _bounds(pieces: list[Piece]) -> tuple[float, float, float, float]:
    x0 = min(p.x for p in pieces)
    y0 = min(p.y for p in pieces)
    x1 = max(p.x + p.sprite.w for p in pieces)
    y1 = max(p.y + p.sprite.h for p in pieces)
    return x0, y0, x1, y1


def _finish(card: Card, pieces: list[Piece]) -> None:
    x0, y0, x1, y1 = _bounds(pieces)
    for p in pieces:
        p.x -= x0
        p.y -= y0
    card.pieces = pieces
    card.w, card.h = x1 - x0, y1 - y0


def _even_lines(idx: list[int], widths: list[float], space: float, limit: float,
                scrap: Callable[[int], bool], fit_one: int = 3) -> list[list[int]]:
    """Break word indexes into one or two lines, as even in width as possible, never leaving a
    short word alone on a line, and within `limit` if it can be. Up to `fit_one` words that fit
    stay on one line."""
    n = len(idx)
    total = sum(widths) + space * (n - 1)
    if n <= 1 or (total <= limit and n <= fit_one):
        return [idx]
    best, best_cost = [idx], math.inf
    for cut in range(1, n):
        a, b = idx[:cut], idx[cut:]
        wa = sum(widths[:cut]) + space * (cut - 1)
        wb = sum(widths[cut:]) + space * (n - cut - 1)
        cost = abs(wa - wb) + (10 * limit if max(wa, wb) > limit else 0)
        for part in (a, b):
            if len(part) == 1 and scrap(part[0]):
                cost += 3 * limit
        if cost < best_cost:
            best, best_cost = [a, b], cost
    return best


def _layout_stacked(card: Card, W: int, H: int, zone_w: float, accent, scale: float, flat: int = 0) -> None:
    # The key word is larger, but the words around it stay at about 70% of its height so the line
    # still reads at phone size, and they are never taken below that to make the key word fit.
    base = max(8, int(round(0.046 * H * scale)))
    big = max(base + 2, int(round(base * 1.42)))
    limit = 0.94 * zone_w
    words = card.words
    k = card.key
    labels = [shown(w).upper() if i != k else bare(w).upper() for i, w in enumerate(words)]

    def style(i: int) -> tuple[str, int, tuple]:
        return ("poppins-black", big, accent) if i == k else ("poppins-extrabold", base, INK)

    # Lines: the words before the key word, the key word, the words after it. A short word that
    # would sit alone on a line rides with the key word instead.
    if flat == 1:  # every word on one line, the key word still larger
        lines = [list(range(len(words)))]
    elif flat == 2:  # two lines as even as can be, the key word larger inside its line
        idx = list(range(len(words)))
        widths = [text.advance(labels[i], *style(i)[:2]) for i in idx]
        lines = _even_lines(idx, widths, text.advance(" ", "poppins-extrabold", base), limit,
                            lambda i: _is_scrap(words[i]), fit_one=0)
    elif k is not None:
        before, after = list(range(0, k)), list(range(k + 1, len(words)))
        middle = [k]
        if len(before) == 1 and _is_scrap(words[before[0]]):
            middle, before = before + middle, []
        if len(after) == 1 and _is_scrap(words[after[0]]):
            middle, after = middle + after, []
        lines = [ln for ln in (before, middle, after) if ln]
    else:
        idx = list(range(len(words)))
        widths = [text.advance(labels[i], "poppins-extrabold", base) for i in idx]
        lines = _even_lines(idx, widths, text.advance(" ", "poppins-extrabold", base), limit,
                            lambda i: _is_scrap(words[i]))
    # Shrink the key word (and only it) if its line would run past the width.
    for ln in lines:
        if k in ln:
            while big > base + 2:
                width = sum(text.advance(labels[i], *style(i)[:2]) for i in ln) + text.advance(" ", "poppins-extrabold", base) * (len(ln) - 1)
                if width <= limit:
                    break
                big -= 2
    pieces: list[Piece] = []
    y_base = 0.0
    gap = 0.26 * text.cap_height("poppins-extrabold", base)
    for li, ln in enumerate(lines):
        caps = [text.cap_height(style(i)[0], style(i)[1]) for i in ln]
        y_base += (gap if li else 0) + max(caps)
        space = text.advance(" ", "poppins-extrabold", base)
        widths = [text.advance(labels[i], style(i)[0], style(i)[1]) for i in ln]
        line_w = sum(widths) + space * (len(ln) - 1)
        pen = -line_w / 2
        for i, wdt in zip(ln, widths):
            face, px, fill = style(i)
            d = text.draw_word(labels[i], face, px, fill=fill, outline=0.075 * px, outline_fill=EDGE, shadow=_shadow(px))
            pieces.append(Piece(d.sprite, pen - d.ox, y_base - d.oy, words[i].start, word=i))
            pen += wdt + space
    _finish(card, pieces)


def _layout_one_word(card: Card, W: int, H: int, zone_w: float, accent, scale: float) -> None:
    w = card.words[0]
    label = (bare(w) if card.key == 0 else shown(w)).upper()
    px = int(round(0.098 * H * scale))
    px = min(px, text.fit_px(label, "anton", 0.92 * zone_w))
    fill = accent if card.key == 0 else INK
    d = text.draw_word(label, "anton", px, fill=fill, outline=0.06 * px, outline_fill=EDGE, shadow=_shadow(px))
    _finish(card, [Piece(d.sprite, -d.ox, -d.oy, w.start, word=0)])


def _layout_serif_chip(card: Card, W: int, H: int, zone_w: float, accent, scale: float) -> None:
    face = "dm-serif"
    px = max(8, int(round(0.037 * H * scale)))
    labels = [shown(w) for w in card.words]
    space = text.advance(" ", face, px)
    pad_x, pad_y = 0.42 * px, 0.24 * px
    limit = 0.94 * zone_w - 2 * pad_x
    widths = [text.advance(s, face, px) for s in labels]
    while sum(widths) + space * (len(widths) - 1) > 2 * limit and px > 8:  # two full lines at most
        px -= 1
        space = text.advance(" ", face, px)
        widths = [text.advance(s, face, px) for s in labels]
        pad_x, pad_y = 0.42 * px, 0.24 * px
    idx_lines = _even_lines(list(range(len(card.words))), widths, space, limit,
                            lambda i: _is_scrap(card.words[i]), fit_one=4)
    # One line's height from the face's own letters: the top of a tall letter to the foot of a
    # letter that hangs below the line.
    top = -text.ink("Hdlk", face, px)[1]
    foot = text.ink("gjpqy", face, px)[3]
    pitch = (top + foot) * 1.06
    dim = DIM
    block_w = max(sum(widths[i] for i in ln) + space * (len(ln) - 1) for ln in idx_lines)
    box_w = block_w + 2 * pad_x
    box_h = pitch * (len(idx_lines) - 1) + top + foot + 2 * pad_y
    box = text.rounded_box(box_w, box_h, radius=0.34 * px, fill=(16, 16, 18), opacity=0.84)
    pieces = [Piece(box, -box_w / 2, 0.0, card.start, pop=False)]
    for li, ln in enumerate(idx_lines):
        y_base = pad_y + top + li * pitch
        line_w = sum(widths[i] for i in ln) + space * (len(ln) - 1)
        pen = -line_w / 2
        for i in ln:
            plain = text.draw_word(labels[i], face, px, fill=dim)
            lit = text.draw_word(labels[i], face, px, fill=accent)
            pieces.append(Piece(plain.sprite, pen - plain.ox, y_base - plain.oy, card.start, word=i,
                                lit=lit.sprite, pop=False))
            pen += widths[i] + space
    _finish(card, pieces)


def _balanced(idx: list[int], widths: list[float], parts: int, scrap: Callable[[int], bool]) -> list[list[int]]:
    """Split word indexes into `parts` lines as even in width as possible, no short word alone."""
    n = len(idx)
    if parts <= 1 or n <= 1:
        return [idx]
    parts = min(parts, n)
    best: Optional[list[list[int]]] = None
    best_cost = math.inf

    def walk(start: int, left: int, acc: list[list[int]]) -> None:
        nonlocal best, best_cost
        if left == 1:
            lines = acc + [idx[start:]]
            ws = [sum(widths[idx.index(i)] for i in ln) for ln in lines]
            cost = max(ws) - min(ws) + sum(10 ** 6 for ln in lines if len(ln) == 1 and scrap(ln[0]))
            if cost < best_cost:
                best, best_cost = lines, cost
            return
        for cut in range(start + 1, n - left + 2):
            walk(cut, left - 1, acc + [idx[start:cut]])

    walk(0, parts, [])
    return best or [idx]


def _layout_fitted(card: Card, W: int, H: int, zone_w: float, accent, scale: float) -> None:
    """The key word big in the accent, the words before it above and after it below, each line
    set as wide as the key word where it can be, so the phrase reads as one block. The words
    around the key word stay between 58% and 72% of its height: a line too long for that is split
    in two rather than shrunk. With no key word the phrase is one or two lines of one size."""
    face = "anton"
    words, k = card.words, card.key
    per_px = text.cap_height(face, 200) / 200.0  # cap height per pixel of size
    wmax = 0.9 * zone_w
    labels = [(bare(w) if i == k else shown(w)).upper() for i, w in enumerate(words)]
    gap_k = 0.28  # the space between words, as a share of the size

    def joined(ln: list[int]) -> str:
        return " ".join(labels[i] for i in ln)

    def scrap(i: int) -> bool:
        return _is_scrap(words[i])

    if k is not None:
        before, middle, after = list(range(0, k)), [k], list(range(k + 1, len(words)))
        # A short word that would stand alone on a line rides on the key word's line instead.
        if len(before) == 1 and scrap(before[0]):
            middle, before = before + middle, []
        if len(after) == 1 and scrap(after[0]):
            middle, after = middle + after, []
        key_px = int(0.084 * H * scale / per_px)
        if text.advance(joined(middle), face, key_px) > wmax:
            key_px = text.fit_px(joined(middle), face, wmax)
        block = max(text.advance(joined(middle), face, key_px), 0.55 * wmax)
        lo, hi = 0.58 * key_px, 0.72 * key_px
        groups: list = [before, None, after]
    else:
        key_px, middle = 0, []
        hi = 0.062 * H * scale / per_px
        lo = 0.8 * hi
        block = 0.8 * wmax
        groups = [list(range(len(words)))]
    lines: list[tuple[list[int], int]] = []
    for g in groups:
        if g is None:
            lines.append((middle, key_px))
            continue
        if not g:
            continue
        widths = [text.advance(labels[i], face, 100) for i in g]
        done = None
        for parts in range(1, min(3, len(g)) + 1):
            sized = []
            for part in _balanced(g, widths, parts, scrap):
                px = min(hi, text.fit_px(joined(part), face, block))
                if px < lo:
                    if text.advance(joined(part), face, int(lo)) > wmax:
                        sized = None
                        break
                    px = lo  # wider than the key word, but still inside the safe zone
                sized.append((part, int(px)))
            if sized:
                done = sized
                break
        if done is None:  # three lines at the smallest size still do not fit: fit each line
            done = [(part, text.fit_px(joined(part), face, wmax)) for part in _balanced(g, widths, 3, scrap)]
        lines.extend(done)
    pieces: list[Piece] = []
    y_base = 0.0
    prev_cap = None
    for ln, px in lines:
        cap = px * per_px
        y_base += (0.17 * min(cap, prev_cap) if prev_cap else 0.0) + cap
        prev_cap = cap
        space = px * gap_k
        ws = [text.advance(labels[i], face, px) for i in ln]
        pen = -(sum(ws) + space * (len(ln) - 1)) / 2
        for i, wdt in zip(ln, ws):
            fill = accent if i == k else INK
            d = text.draw_word(labels[i], face, px, fill=fill, outline=0.05 * px, outline_fill=EDGE, shadow=_shadow(px))
            pieces.append(Piece(d.sprite, pen - d.ox, y_base - d.oy, words[i].start, word=i))
            pen += wdt + space
    _finish(card, pieces)


LAYOUTS: dict[str, Callable] = {
    "stacked": _layout_stacked,
    "one-word": _layout_one_word,
    "serif-chip": _layout_serif_chip,
    "fitted": _layout_fitted,
}


# ---- the captions of one output --------------------------------------------------------------

def _ease(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 1.0 - (1.0 - x) ** 3


class Captions:
    """Every card of one output, laid out and placed, ready to draw on its frames."""

    def __init__(self, words: list[Word], quiet: list[tuple[float, float]], size: tuple[int, int], fps: float,
                 n_frames: int, head_at: Callable[[int], Optional[Rect]], look: str = "stacked",
                 color: str = "auto", position: str = "auto", max_words: int = 0,
                 cards: Optional[list[dict]] = None, face_at: Optional[Callable[[int], Optional[Rect]]] = None):
        self.W, self.H = size
        self.fps = fps
        self.n = n_frames
        self.look = LOOK[look]
        self.accent = text.colour(self.look.accent if color in ("auto", "", None) else color)
        self.position = position
        self.zone = SafeZone(self.W, self.H)
        self.heads = [head_at(j) for j in range(n_frames)]
        # The face alone (hairline to chin), when the effect gives it: what a line may never cross even
        # where the head leaves no room (see `_flatter`).
        self.faces = [face_at(j) for j in range(n_frames)] if face_at is not None else None
        self.warnings: list[str] = []
        # What the speaker took back is never the accent: read from the words given, fillers and all.
        self.back = moments.taken_back(words, quiet)
        self.quiet = quiet
        # An "um" is heard, never shown; nor is a word the second listen could not be sure of.
        words = [w for w in words if w.key not in HESITATIONS and not w.unsure]
        most = max_words or self.look.most
        if cards is not None:
            self.cards = [self._card_from_plan(c) for c in cards]
        else:  # a number the person chose is never passed; the look's own may be, by one word
            self.cards = self._cards(words, quiet, most, over=0 if max_words else 1)
        self._time()
        for c in self.cards:
            self._layout(c, 1.0)
        self._place()

    # -- which words go together, and for how long
    def _cards(self, words: list[Word], quiet, most: int, over: int = 1) -> list[Card]:
        if self.look.name == "one-word":
            # Key words are chosen among the phrases a reader would hear, then shown one at a time.
            keys = set()
            for ph in phrases(words, quiet, 4):
                k = key_word([words[i] for i in ph], quiet, self.back)
                if k is not None:
                    keys.add(ph[k])
            return [Card(w.start, w.end, [w], 0 if i in keys else None) for i, w in enumerate(words)]
        out = []
        for ph in phrases(words, quiet, most, over):
            for part in self._at_full_size(ph, words, quiet, most):
                ws = [words[i] for i in part]
                out.append(Card(ws[0].start, ws[-1].end, ws, key_word(ws, quiet, self.back)))
        return out

    def _at_full_size(self, ph: list[int], words: list[Word], quiet, most: int) -> list[list[int]]:
        """A phrase given one word more than the look's number keeps it, unless that costs its type
        its size. The stacked look sets the words either side of its key word on one line each, and
        four there can run so far past the safe zone that the card is drawn at half size. When the
        phrase's halves could each be drawn at least two steps larger, and its best place to cut
        breaks nothing, it is cut there instead ("The marketing platform | of UGC")."""
        if len(ph) <= most:
            return [ph]
        whole = self._fitting_scale([words[i] for i in ph])
        if whole >= 1.0:
            return [ph]
        parts = phrasing.split_run(words, ph, quiet, most, over=0)
        if any(len(p) < 2 for p in parts) or any(
                phrasing.cut_cost(words, p[-1], quiet) > phrasing.INSIDE_A_PHRASE for p in parts[:-1]):
            return [ph]
        halves = min(self._fitting_scale([words[i] for i in p]) for p in parts)
        return parts if halves >= whole / SHRINK ** 2 - 1e-9 else [ph]

    def _fitting_scale(self, ws: list[Word]) -> float:
        """The size these words would be drawn at, as one card, to stay inside the safe zone's width:
        full size (1), or one of the steps `_place` shrinks a card by."""
        card = Card(ws[0].start, ws[-1].end, ws, key_word(ws, self.quiet, self.back))
        scale = 1.0
        for step in range(SHRINK_STEPS + 1):
            if step:
                scale *= SHRINK
            self._layout(card, scale)
            if card.w <= self.zone.rect.w:
                break
        return scale

    def _card_from_plan(self, c: dict) -> Card:
        ws = [Word(str(x["w"]), float(x["s"]), float(x["e"])) for x in c["words"]]
        key = c.get("key")
        if key is not None and not (0 <= int(key) < len(ws)):
            raise ValueError(f"a card's key {key} is not one of its {len(ws)} words")
        return Card(ws[0].start, ws[-1].end, ws, None if key is None else int(key))

    def _time(self) -> None:
        cards = self.cards
        for i, c in enumerate(cards):
            nxt = cards[i + 1].start if i + 1 < len(cards) else math.inf
            last = c.words[-1]
            if self.look.name == "one-word":
                end = nxt if nxt - last.end < 0.45 else last.end + 0.25
                end = max(end, min(nxt, c.start + 0.2))
            else:
                end = min(nxt, last.end + HOLD)
                if end - c.start < MIN_ON:
                    end = min(nxt, c.start + MIN_ON)
            c.end = max(c.start + 1.0 / self.fps, end)

    def _layout(self, c: Card, scale: float) -> None:
        c.scale = scale
        if c.flat:
            _layout_stacked(c, self.W, self.H, self.zone.rect.w, self.accent, scale, flat=c.flat)
            return
        LAYOUTS[self.look.name](c, self.W, self.H, self.zone.rect.w, self.accent, scale)

    def _width_scale(self, c: Card) -> float:
        """The size the card must be drawn at to fit the safe zone's width (1, or a shrink step); the
        card is left laid out at full size."""
        scale = 1.0
        for step in range(SHRINK_STEPS + 1):
            if step:
                scale *= SHRINK
            self._layout(c, scale)
            if c.w <= self.zone.rect.w:
                break
        self._layout(c, 1.0)
        return scale

    def _frames(self, c: Card) -> tuple[int, int]:
        first = int(max(0, math.floor(c.start * self.fps)))
        last = int(min(self.n - 1, math.ceil(c.end * self.fps)))
        return first, max(first, last)

    def _head_over(self, c: Card, boxes: Optional[list] = None) -> Optional[Rect]:
        a, b = self._frames(c)
        boxes = self.heads if boxes is None else boxes
        return Rect.around(h for h in boxes[a:b + 1] if h is not None)

    @staticmethod
    def _every(boxes: list) -> Optional[Rect]:
        found = [h for h in boxes if h is not None]
        if not found:
            return None
        arr = np.array([[h.x0, h.y0, h.x1, h.y1] for h in found])
        return Rect(float(np.percentile(arr[:, 0], 2)), float(np.percentile(arr[:, 1], 2)),
                    float(np.percentile(arr[:, 2], 98)), float(np.percentile(arr[:, 3], 98)))

    # -- where each card goes
    def _place(self) -> None:
        zone = self.zone
        gap = 0.024 * self.H
        every = self._every(self.heads)
        order = layout.ORDERS[self.position]
        tallest = max((c.h for c in self.cards), default=0.0)
        widest = max((c.w for c in self.cards), default=0.0)
        anchor = None
        if every is not None and self.cards:
            for where in order:
                if layout.place(widest, tallest, zone, every, order=(where,), gap=gap):
                    anchor = where
                    break
        self.anchor = anchor or ("lower-third" if every is None else order[0])
        for c in self.cards:
            stacked = self.look.name == "stacked" and len(c.words) > 1
            wide = self._width_scale(c) if stacked else 1.0
            placed = False
            for attempt in range(SHRINK_STEPS + 1):
                if attempt:
                    self._layout(c, c.scale * SHRINK)
                if self._try_place(c, every, gap, order):
                    placed = True
                    break
            if placed and (not stacked or c.scale >= FLAT_BELOW or c.scale >= wide - 1e-9):
                continue  # drawn at its size, or as small as its width (not the band's height) asks
            was = (c.scale, c.x, c.y, c.where)
            hair = not placed or c.scale < OVER_HAIR_BELOW
            if stacked and self._flatter(c, every, gap, order, hair):
                continue
            c.flat = 0
            self._layout(c, was[0])
            if placed:
                c.x, c.y, c.where = was[1], was[2], was[3]
            else:
                self._last_resort(c)

    def _flatter(self, c: Card, every: Optional[Rect], gap: float, order, hair: bool) -> bool:
        """A stacked card the band clear of the head is too short for, set flatter (on one line, else on
        two even lines): clear of the whole head if it fits there, else (when `hair`, and the effect gave
        the face) clear of the face alone, over the top of the hair. False, with the card as it was,
        when none of these fits."""
        tries = [(None, every, "")]
        if hair and self.faces is not None:
            tries.append((self.faces, self._every(self.faces), " over the hair"))
        for boxes, ev, said in tries:
            for flat in (1, 2):
                c.flat = flat
                for step in range(FLAT_STEPS + 1):
                    self._layout(c, SHRINK ** step)
                    if self._try_place(c, ev, gap, order, boxes):
                        c.where += said
                        return True
        c.flat = 0
        return False
    def _last_resort(self, c: Card) -> None:
        """Nothing fits inside the safe zone clear of the head. Covering the face is the one thing
        never done, so the words go into the margin the app's own buttons use, the side away from
        the head, and the report says so."""
        head = self._head_over(c)
        c.x = self.zone.rect.cx - c.w / 2
        room_above = head.y0 if head is not None else 0.0
        room_below = self.H - head.y1 if head is not None else self.H
        if room_below >= room_above:
            c.y = min(self.H - c.h - 0.01 * self.H, max(head.y1 + 0.01 * self.H if head else 0.0, self.zone.rect.y1 - c.h))
        else:
            c.y = max(0.01 * self.H, min(head.y0 - 0.01 * self.H - c.h, self.zone.rect.y0))
        c.where = "outside the safe zone"
        clear = head is None or not c.rect.intersects(head)
        self.warnings.append(f"the words at {c.start:.2f} s did not fit inside the safe zone clear of the face; "
                             + ("they sit outside the safe zone" if clear else "they could not be kept off the face"))

    def _try_place(self, c: Card, every: Optional[Rect], gap: float, order, boxes: Optional[list] = None) -> bool:
        zone = self.zone
        head = self._head_over(c, boxes)
        x0 = zone.rect.cx - c.w / 2
        # The clip's steady place first.
        if every is not None:
            if self.anchor == "chest":
                y0 = max(every.y1 + gap, zone.rect.y0)
            elif self.anchor == "lower-third":
                y0 = zone.rect.y1 - 0.01 * self.H - c.h
            else:
                y0 = min(every.y0 - gap, zone.rect.y1) - c.h
            r = Rect(x0, y0, x0 + c.w, y0 + c.h)
            if zone.contains(r) and (head is None or not r.intersects(head, margin=gap * 0.5)):
                c.x, c.y, c.where = r.x0, r.y0, self.anchor
                return True
        if head is None:
            y0 = zone.rect.y1 - 0.01 * self.H - c.h
            r = Rect(x0, y0, x0 + c.w, y0 + c.h)
            if zone.contains(r):
                c.x, c.y, c.where = r.x0, r.y0, "lower-third"
                return True
            return False
        rest = (self.anchor,) + tuple(o for o in order if o != self.anchor)
        p = layout.place(c.w, c.h, zone, head, order=rest, gap=gap)
        if p is None:
            return False
        c.x, c.y, c.where = p.rect.x0, p.rect.y0, p.where
        return True

    # -- drawing
    def active(self, t: float) -> list[Card]:
        return [c for c in self.cards if c.start <= t < c.end]

    def _look_of(self, c: Card, p: Piece, t: float) -> tuple[float, float]:
        """A piece's opacity and scale at time t. A frame shows the moment it starts, so the frame a
        word appears on already counts as one frame into its pop: starting from nothing would leave
        a blank frame wherever one phrase follows straight on from another."""
        step = 1.0 / self.fps
        if p.pop and self.look.reveal:
            k = t - p.on + step
            return _ease(k / 0.06), 0.9 + 0.1 * _ease(k / 0.1)
        return _ease((t - c.start + step) / 0.08), 1.0

    def draw(self, frame: np.ndarray, j: int, t: float) -> None:
        for c in self.active(t):
            current = None
            if self.look.name == "serif-chip":
                said = [i for i, w in enumerate(c.words) if w.start <= t]
                current = said[-1] if said else None
            for p in c.pieces:
                if self.look.reveal and t < p.on:
                    continue
                alpha, scale = self._look_of(c, p, t)
                sprite = p.lit if (p.lit is not None and p.word == current) else p.sprite
                if scale < 0.999:
                    s = sprite.scaled(scale)
                    cx = c.x + p.x + sprite.w / 2
                    cy = c.y + p.y + sprite.h / 2
                    over(frame, s, cx - s.w / 2, cy - s.h / 2, alpha)
                else:
                    over(frame, sprite, c.x + p.x, c.y + p.y, alpha)

    # -- what a person (or the matrix) checks
    def report(self) -> dict:
        over_face = 0
        over_hair = 0
        outside = 0
        for j in range(self.n):
            t = j / self.fps
            act = self.active(t)
            if not act:
                continue
            h = self.heads[j]
            f = self.faces[j] if self.faces is not None else h
            for c in act:
                if f is not None and c.rect.intersects(f):
                    over_face += 1
                elif h is not None and c.rect.intersects(h):
                    over_hair += 1
                if not self.zone.contains(c.rect):
                    outside += 1
        # How visible each phrase is on the first frame it is on screen: a phrase that starts
        # invisible right after another one ends is a blank frame, a flicker.
        first_frame = []
        for c in self.cards:
            t0 = math.ceil(c.start * self.fps - 1e-6) / self.fps
            shown = [self._look_of(c, p, t0)[0] for p in c.pieces if not (self.look.reveal and t0 < p.on)]
            first_frame.append(max(shown, default=0.0))
        where = {}
        for c in self.cards:
            where[c.where] = where.get(c.where, 0) + 1
        out = {"cards": len(self.cards), "with_key_word": sum(c.key is not None for c in self.cards),
               "frames_over_face": over_face, "frames_outside_safe_zone": outside, "placed": where,
               "least_visible_first_frame": round(min(first_frame, default=1.0), 2),
               "shrunk": sum(c.scale < 1 for c in self.cards), "warnings": list(self.warnings)}
        if self.faces is not None:
            out["frames_over_hair"] = over_hair
        if any(c.flat for c in self.cards):
            out["set_flatter"] = {"one line": sum(c.flat == 1 for c in self.cards),
                                  "two lines": sum(c.flat == 2 for c in self.cards)}
        return out

    def plan(self) -> list[dict]:
        return [{"start": round(c.start, 3), "end": round(c.end, 3), "where": c.where,
                 "rect": c.rect.as_list(), "key": c.key, **({"lines": c.flat} if c.flat else {}),
                 "text": " ".join(shown(w) for w in c.words),
                 "words": [{"w": w.text, "s": round(w.start, 3), "e": round(w.end, 3)} for w in c.words]}
                for c in self.cards]

    def busy_times(self, count: int = 6) -> list[float]:
        """`count` moments spread across the output, each in the middle of a card with all its
        words showing: what a contact sheet should show."""
        if not self.cards:
            return []
        picks = np.linspace(0, len(self.cards) - 1, min(count, len(self.cards))).round().astype(int)
        out = []
        for i in picks:
            c = self.cards[int(i)]
            t = max(c.words[-1].start + 0.12, (c.start + c.end) / 2)
            out.append(min(c.end - 0.5 / self.fps, t))
        return out


# ---- the effect --------------------------------------------------------------------------------

def unsure_words(clip) -> list[dict]:
    """The words heard but never shown, with why: what a plan records so nothing heard is lost."""
    from ..core.words import unsure_stretches

    return unsure_stretches(clip.unsure)


def head_in_output(clip, timeline, fr):
    """The head's box at each output frame, in output pixels (None where no face is known)."""
    face = clip.face
    fps, n_src = clip.info.rate, clip.info.frames

    def at(j: int) -> Optional[Rect]:
        k = timeline.source_frame(j, fps, n_src)
        h = face.head(k)
        return fr.to_output(h, k) if h is not None else None

    return at


def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from ..core import framing as framing_mod
    from ..core.timeline import Timeline
    from . import Job, PlanError

    info = clip.info
    fr = framing_mod.plan(info.width, info.height, info.rate, info.frames, clip.face, shape=opts["frame"],
                          size=O.size_of(opts["size"]))
    tl = Timeline.whole(info.frames / info.rate)
    cards = None
    if plan is not None:
        cards = plan.get("cards")
        if not isinstance(cards, list) or not cards:
            raise PlanError("the plan has no cards; run once without --plan to get one to edit")
        for c in cards:
            ws = c.get("words") or []
            if not ws:
                raise PlanError("every card needs at least one word")
            for w in ws:
                if not (0 <= float(w["s"]) <= float(w["e"]) <= tl.duration + 0.5):
                    raise PlanError(f"the word {w.get('w')!r} has times outside the clip")
    caps = Captions(clip.words, clip.quiet, (fr.out_w, fr.out_h), info.rate, tl.output_frames(info.rate),
                    head_in_output(clip, tl, fr), look=opts["look"], color=opts["color"],
                    position=opts["position"], max_words=opts["max_words"], cards=cards)
    out_plan = {"effect": NAME, "clip": clip.path, "options": opts,
                "output": {"width": fr.out_w, "height": fr.out_h, "fps": round(info.rate, 3),
                           "seconds": round(tl.duration, 3), "framing": fr.note},
                "anchor": caps.anchor, "cards": caps.plan(), "unsure": unsure_words(clip)}
    return Job(NAME, clip, tl, fr, caps.draw, out_plan, checks=caps.report(),
               still_times=lambda: caps.busy_times(4))
