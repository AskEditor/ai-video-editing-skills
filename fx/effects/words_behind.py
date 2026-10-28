"""words-behind: a giant word behind the speaker's head, landing on the line it belongs to.

How it decides, in order:

1. **The word.** Unless `text` is given, it comes from the speech: the opening line that is a whole
   thought (a clip that starts in the middle of a sentence skips that scrap), and in it the word
   that names the thing being talked about: a noun at the end of its phrase, a name, a number. Not
   the longest word, and never a word of grammar, a filler or a word the transcriber was unsure of,
   a word the speaker takes back ("the lively town, or the lively city" sets CITY) or a word of a
   first attempt said again: the core's picker (`fx/core/moments.py`), which headline and
   prompt-box pick by too. The plan lists the runners-up, so another can be asked for by name, and
   the words passed over because they were taken back.
2. **The moment.** The word is in place on the frame its sound starts. The transcriber marks words
   late, so the core's corrected times are used (`Clip.words`), and the entrance is drawn before
   that frame rather than after it.
3. **The size and the place, measured on the person mask** over every frame the word is on screen
   (the head moves): as large as the safe zone allows, and low enough that the top of the head
   hides a little of the letters' foot, which is what makes it read as behind. Never so low that
   a letter is lost: the share of each letter the person hides is measured, not guessed.
4. **The colour,** light or dark, whichever stands out from the wall behind the letters. When the
   wall is busy or close in tone, the wall right around the letters is quietly shaded until the
   letters reach a readable contrast.
5. **The cut-out.** The person is drawn in front, the edge rebuilt from the frame itself. A pixel
   on the edge is part person and part wall; only the wall's part is swapped for the word, so no
   rim of the old wall shows round the head where it crosses the letters. Inside the person the
   mask is exactly opaque, so nothing shows through.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np

from ..core import moments
from ..core import text as textmod
from ..core.compose import Sprite
from ..core.layout import Rect, SafeZone
from ..core.words import HESITATIONS, Word, normalize, pause_after
from . import options as O
from .captions import bare

NAME = "words-behind"

FONTS = {"condensed": "anton", "bold": "poppins-black", "serif": "dm-serif"}
ANIMATIONS = ("rise", "drop", "fade", "scale")
STYLES = ("solid", "outline", "glow")

# The two inks `color=auto` chooses between, and the glow's own colour. Chosen for this pack.
LIGHT = (250, 245, 234)
DARK = (24, 22, 27)
GLOW = (255, 77, 109)
GLOW_DIM = 0.5  # a glow dims the room behind the person by this much while it is up: light needs dark
GLOW_SHADE = 0.45  # and pools more dark right round the letters


# ---- the options -------------------------------------------------------------------------------

@dataclass(frozen=True)
class _Words:
    help: str

    def parse(self, raw) -> str:
        v = str(raw).strip().strip("\"“”")
        if v.lower() in ("", "auto"):
            return "auto"
        if len(v.split()) > 4 or len(v) > 30:
            raise O.OptionError("give a word or a short phrase, four words at most: a long line set behind a head "
                                f"is too small to read (got {raw!r})")
        if not any(c.isalnum() for c in v):
            raise O.OptionError(f"give a word to show (got {raw!r})")
        return v


@dataclass(frozen=True)
class _When:
    help: str

    def parse(self, raw) -> str:
        v = str(raw).strip()
        if v.lower() in ("", "auto"):
            return "auto"
        try:
            t = float(v.rstrip("s"))
        except ValueError:
            said = v.strip("\"“”")
            if not [normalize(x) for x in said.split() if normalize(x)]:
                raise O.OptionError("give auto, a time in seconds such as 2.4, or words said in the clip "
                                    f"(got {raw!r})") from None
            return said
        if t < 0:
            raise O.OptionError(f"give a time from 0 seconds on (got {raw!r})")
        return f"{t:g}"


@dataclass(frozen=True)
class _Hold:
    help: str

    def parse(self, raw) -> str:
        v = str(raw).strip().lower()
        if v in ("", "auto", "end"):
            return v or "auto"
        try:
            s = float(v.rstrip("s"))
        except ValueError:
            raise O.OptionError(f"give auto, end, or seconds such as 3 (got {raw!r})") from None
        if not 0.5 <= s <= 60:
            raise O.OptionError(f"give between 0.5 and 60 seconds (got {raw!r})")
        return f"{s:g}"


@dataclass(frozen=True)
class _Share:
    help: str

    def parse(self, raw) -> str:
        v = str(raw).strip().lower()
        if v in ("", "auto"):
            return "auto"
        try:
            s = float(v.rstrip("%")) / (100.0 if v.endswith("%") else 1.0)
        except ValueError:
            raise O.OptionError(f"give auto, or a share from 0.3 to 1 such as 0.7 (got {raw!r})") from None
        if not 0.3 <= s <= 1.0:
            raise O.OptionError(f"give a share from 0.3 to 1, such as 0.7 (got {raw!r})")
        return f"{s:g}"


OPTIONS = {
    "text": _Words("auto: the word that names the point of the line, taken from the speech · or the word or short "
                   "phrase to show, up to four words"),
    "when": _When("auto: the opening line · a time in seconds such as 2.4 · or words said in the clip, such as "
                  "\"what draws people in\": the word lands as they are said"),
    "font": O.Choice(tuple(FONTS), "condensed: tall heavy capitals, the largest for a long word · bold: a wide heavy "
                                   "sans · serif: a display serif"),
    "color": O.Colour("auto: light or dark, whichever stands out from the wall behind the letters · a name such as "
                      "yellow or red · or a hex colour such as #ffd400"),
    "size": _Share("auto: as large as the room above and behind the head allows · or a share of that from 0.3 to 1, "
                   "such as 0.7"),
    "animation": O.Choice(ANIMATIONS, "rise: comes up from behind the head · drop: falls into place · fade: fades "
                                      "in · scale: lands from larger"),
    "hold": _Hold("auto: to the end of its line, two seconds at least · seconds such as 3 · or end: to the end of "
                  "the clip"),
    "style": O.Choice(STYLES, "solid letters · outline: hollow letters · glow: lit letters with a soft glow"),
    "frame": O.Choice(("keep", "9:16"), "keep the clip's own shape, or make it 9:16 following the face"),
    "fix": O.Text("misheard words as wrong=right, separated by ;  e.g. \"Jon=John\""),
}

# The runs `tests/matrix.py words-behind` makes: the plain wall and the clips in the brief's order,
# the train with a word given (it has no speech to take one from), then the other looks and a phrase
# that lands word by word.
MATRIX = [
    ("D", "D", {}),
    ("B", "B", {}),
    ("A", "A", {}),
    ("C", "C", {}),
    ("F", "F", {"text": "NEXT STOP", "when": "1.5"}),
    ("D-serif-outline", "D", {"font": "serif", "style": "outline", "animation": "drop"}),
    ("B-glow-scale", "B", {"style": "glow", "font": "bold", "animation": "scale"}),
    ("D-phrase", "D", {"text": "the lively city", "animation": "fade"}),
]


# ---- choosing the word ---------------------------------------------------------------------------
#
# The word is chosen by the core's picker (`fx/core/moments.py`), the same one headline and
# prompt-box choose by: the thing named, at the end of its phrase, never a word the speaker took
# back ("the lively town, or the lively city" sets CITY) or a word of a first attempt said again.
# The names below are kept for the effects that read them from here.

hero_score = moments.word_strength  # how well one word stands alone, giant, for its line
lines_of = moments.lines
_scrap_lines = moments.scrap_lines
best_in = moments.best_in
Candidate = moments.Candidate
MIN_HERO = moments.MIN_HERO


def choose(words: list[Word], quiet, start_after: float = 0.0, within: Optional[tuple[float, float]] = None,
           back: Optional[moments.TakenBack] = None) -> tuple[Optional[Candidate], list[Candidate]]:
    """The word for the line that lands, and the runners-up. `within` limits it to a stretch the
    person named (the best word there, however plain); `start_after` to the lines from a time on.
    `back`: what the speaker took back, from the clip's whole words (`moments.taken_back`)."""
    return moments.pick_word(words, quiet, back, start_after=start_after, within=within)


def find_said(words: list[Word], said: str) -> Optional[tuple[int, int]]:
    """Where the words `said` are spoken: (first, last) word indexes, most of them matching in order."""
    want = [normalize(x) for x in said.split() if normalize(x)]
    keys = [w.key for w in words]
    if not want or not keys:
        return None
    best, best_hits = None, 0
    for a in range(len(keys)):
        if keys[a] != want[0] and (len(want) < 2 or a + 1 >= len(keys) or keys[a] != want[1]):
            continue
        hits, j, last = 0, a, a
        for wk in want:
            for step in (0, 1):  # one word misheard or added
                if j + step < len(keys) and keys[j + step] == wk:
                    hits += 1
                    last = j + step
                    j += step + 1
                    break
            else:
                j += 1
        if hits > best_hits:
            best, best_hits = (a, last), hits
    if best is None or best_hits < max(1, math.ceil(0.6 * len(want))):
        return None
    return best


# ---- the block of letters ------------------------------------------------------------------------

@dataclass(frozen=True)
class Block:
    lines: tuple[str, ...]
    face: str
    px: int
    cap: float
    gap: float
    w: float
    h: float
    line_tops: tuple[float, ...]  # each line's cap top, from the block's top
    line_x: tuple[float, ...]  # each line's left, from the block's left (lines are centred)


@lru_cache(maxsize=256)
def make_block(lines: tuple[str, ...], face: str, px: int) -> Block:
    cap = textmod.cap_height(face, px)
    gap = 0.16 * cap
    widths = [textmod.advance(ln, face, px) for ln in lines]
    w = max(widths)
    tops = tuple(k * (cap + gap) for k in range(len(lines)))
    xs = tuple((w - lw) / 2 for lw in widths)
    return Block(lines, face, px, cap, gap, w, cap * len(lines) + gap * (len(lines) - 1), tops, xs)


MARGIN = 2  # pixels round the block in its ink mask


def word_slots(b: Block) -> list[tuple[int, str, float, float]]:
    """Every word of the block: (line, word, pen x, baseline y), in block pixels."""
    f = textmod.font(b.face, b.px)
    out = []
    for k, ln in enumerate(b.lines):
        base = b.line_tops[k] + b.cap
        pos = 0
        for wtext in ln.split(" "):
            at = ln.index(wtext, pos)
            out.append((k, wtext, b.line_x[k] + f.getlength(ln[:at]), base))
            pos = at + len(wtext)
    return out


@lru_cache(maxsize=64)
def block_ink(b: Block, only: Optional[tuple[int, ...]] = None) -> tuple[np.ndarray, tuple[tuple[float, ...], ...]]:
    """The block's ink (0..1) with MARGIN pixels round it and room below for anything that hangs
    under the last line, and each letter's box (top, bottom, left, right) in that mask with the
    share of it the person may hide. `only` draws just those words (their indexes in `word_slots`)."""
    from PIL import Image, ImageDraw

    f = textmod.font(b.face, b.px)
    hang = int(math.ceil(0.35 * b.cap))
    W, H = int(math.ceil(b.w)) + 2 * MARGIN, int(math.ceil(b.h)) + 2 * MARGIN + hang
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    letters = []
    for n, (k, wtext, pen, base) in enumerate(word_slots(b)):
        if only is not None and n not in only:
            continue
        draw.text((MARGIN + pen, MARGIN + base), wtext, font=f, fill=255, anchor="ls")
        for c in range(len(wtext)):
            if wtext[c].isspace():
                continue
            left = pen + f.getlength(wtext[:c])
            cl, _, cr, _ = f.getbbox(wtext[c], anchor="ls")
            letters.append((MARGIN + b.line_tops[k], MARGIN + base, MARGIN + left + cl, MARGIN + left + cr,
                            FOOT_LIMIT if wtext[c].upper() in FOOT_LETTERS else LETTER_MAX))
    return np.asarray(img, np.float32) / 255.0, tuple(letters)


# ---- placing it ----------------------------------------------------------------------------------

Q = 4  # placement is measured at a quarter of the output size
# The share of the letters' ink the person hides: enough to read as behind them, never so much that a
# letter turns into another. Measured on every frame the word is on screen (the head moves).
HIDE_MIN, HIDE_TARGET, HIDE_MAX = 0.025, 0.09, 0.20
LETTER_MAX = 0.30  # of any one letter
# Letters told apart by their foot: hide the foot of an E and it reads F, of a B and it reads R, of an
# L and it reads I. These may lose only a sliver.
FOOT_LETTERS = set("EBLZDJQSG235689")
FOOT_LIMIT = 0.12
TOUCH = 0.004  # 'just touching the head': the least overlap that still reads as behind it
BIG = 0.82  # sizes down to this share of the largest come before any smaller size
REACH = 0.012  # the mask's soft edge kept this close (a share of the width) to where the person is sure
EDGE = 0.5  # where the person's edge sits (the core's `lean`): the model's own, then decided by colour
CAP_MIN, CAP_MAX = 0.035, 0.24  # the letters' height, as a share of the frame's height


class MaskStore:
    """The person mask of chosen output frames, in output space at 1/Q size, read once in order."""

    def __init__(self, clip, timeline, framing, out_size: tuple[int, int], frames: list[int]):
        import cv2

        self.w, self.h = max(2, out_size[0] // Q), max(2, out_size[1] // Q)
        self.by_frame: dict[int, np.ndarray] = {}
        masks = clip.person()
        fps, n_src = clip.info.rate, clip.info.frames
        want: dict[int, list[int]] = {}
        for j in frames:
            want.setdefault(timeline.source_frame(j, fps, n_src), []).append(j)
        if not want:
            return
        last = max(want)
        sx, sy = masks.width / clip.info.width, masks.height / clip.info.height
        for k, m in enumerate(masks.frames()):
            if k > last:
                break
            if k not in want:
                continue
            x, y, w, h = framing.window(k)
            x0, y0 = int(np.floor(x * sx)), int(np.floor(y * sy))
            x1, y1 = int(np.ceil((x + w) * sx)), int(np.ceil((y + h) * sy))
            crop = m[max(0, y0):min(masks.height, y1), max(0, x0):min(masks.width, x1)]
            small = cv2.resize(crop, (self.w, self.h), interpolation=cv2.INTER_AREA)
            for j in want[k]:
                self.by_frame[j] = small

    def stack(self) -> np.ndarray:
        if not self.by_frame:
            return np.zeros((0, self.h, self.w), np.float32)
        return np.stack([self.by_frame[j] for j in sorted(self.by_frame)])

    def crowns(self, level: float = 0.6) -> np.ndarray:
        """For each frame and column, the first row the person reaches (the height where nobody is)."""
        M = self.stack()
        if not len(M):
            return np.zeros((0, self.w), np.int32)
        hit = M >= level
        return np.where(hit.any(axis=1), hit.argmax(axis=1), M.shape[1]).astype(np.int32)


@dataclass
class Placement:
    block: Block
    x: float  # the block's top-left (its first cap top), output pixels
    y: float
    hidden_median: float
    hidden_p90: float
    letter_p90: float
    where: str
    warn: str = ""

    @property
    def rect(self) -> Rect:
        return Rect(self.x, self.y, self.x + self.block.w, self.y + self.block.h)


def _scan(b: Block, tops: np.ndarray, x: float, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For the block with its top-left at (x, each of ys) in output pixels: the median and 90th
    percentile share of its ink the person hides across the frames, and the 90th percentile of the
    most-hidden letter, as a share of what that letter may lose (over 1 is too much). Measured on
    each column's crown: the person is taken to fill the column from there down, which holds above
    the shoulders, where the word goes."""
    import cv2

    ink_full, letters_full = block_ink(b)
    s = 1.0 / Q
    ih = max(1, int(round(ink_full.shape[0] * s)))
    iw = max(1, int(round(ink_full.shape[1] * s)))
    ink = cv2.resize(ink_full, (iw, ih), interpolation=cv2.INTER_AREA)
    total = float(ink.sum()) or 1.0
    below = np.zeros((ih + 1, iw), np.float64)
    below[:ih] = np.cumsum(ink[::-1], axis=0)[::-1]  # ink at or below each row, per column
    n, Wm = tops.shape
    ox = int(round((x - MARGIN) * s))
    cols = np.arange(ox, ox + iw)
    inside = (cols >= 0) & (cols < Wm)
    T = np.full((n, iw), 10 ** 6, np.int64)
    T[:, inside] = tops[:, cols[inside]]
    oys = np.round((ys - MARGIN) * s).astype(np.int64)
    K = np.clip(T[:, None, :] - oys[None, :, None], 0, ih)  # (n, ny, iw): first hidden row per column
    hidden = below[K, np.arange(iw)[None, None, :]].sum(axis=-1) / total  # (n, ny)
    worst = np.zeros_like(hidden)  # the most-hidden letter, as a share of what that letter may lose
    for (r0, r1, c0, c1, limit) in letters_full:
        a, bb = int(math.floor(r0 * s)), int(math.ceil(r1 * s))
        ca, cb = max(0, int(math.floor(c0 * s))), min(iw, int(math.ceil(c1 * s)))
        if cb <= ca or bb <= a:
            continue
        a, bb = max(0, a), min(ih, bb)
        tot_l = float(below[a, ca:cb].sum() - below[bb, ca:cb].sum())
        if tot_l <= 1e-6:
            continue
        Kl = np.clip(K[..., ca:cb], a, bb)
        part = below[Kl, np.arange(ca, cb)[None, None, :]] - below[bb, ca:cb][None, None, :]
        worst = np.maximum(worst, part.sum(axis=-1) / tot_l / limit)
    return np.median(hidden, axis=0), np.percentile(hidden, 90, axis=0), np.percentile(worst, 90, axis=0)


def _position(b: Block, tops: np.ndarray, zone: Rect, W: int, H: int, strict: bool,
              least: float = HIDE_MIN) -> Optional[Placement]:
    """Where the block goes at this size: centred if it can be, else to one side of the head.
    `strict` keeps every letter readable and at least `least` of the ink behind the person."""
    y_lo, y_hi = zone.y0, zone.y1 - b.h
    if y_hi < y_lo or b.w > zone.w + 0.5:
        return None
    ys = np.arange(y_lo, y_hi + 0.01, max(1.0, Q / 2.0))
    best = None
    for off in (0.0, -0.12, 0.12, -0.24, 0.24):
        cx = zone.cx + off * W
        x = cx - b.w / 2
        if x < zone.x0 - 0.5 or x + b.w > zone.x1 + 0.5:
            continue
        if not len(tops):
            med = p90 = lp = np.zeros(len(ys))
        else:
            med, p90, lp = _scan(b, tops, x, ys)
        if strict:
            ok = (med >= least) & (p90 <= HIDE_MAX) & (lp <= 1.0)
            if not ok.any():
                continue
            cost = np.where(ok, np.abs(med - HIDE_TARGET), np.inf)
        else:
            cost = p90 + lp
        i = int(np.argmin(cost))
        c = float(cost[i]) + 1.5 * abs(off)
        if best is None or c < best[0]:
            if not strict:
                where = "as clear of the head as the shot allows"
            elif med[i] < TOUCH:
                where = "above the head: no letter could go behind it and stay readable"
            elif med[i] < HIDE_MIN:
                where = "just touching the head" + ("" if off == 0 else ", to one side")
            else:
                where = "behind the head" if off == 0 else "behind the head, to one side"
            warn = "" if strict else ("the shot has little room round the head: "
                                      f"{p90[i]:.0%} of the words are hidden at worst, and some letters "
                                      "may not read; a shorter phrase or one word fits better")
            best = (c, Placement(b, x, float(ys[i]), float(med[i]), float(p90[i]), float(lp[i]), where, warn))
        if strict and off == 0.0:
            break  # centred fits: the sides are not needed
    return best[1] if best else None


def _sizes(lines: tuple[str, ...], face: str, zone: Rect, H: int) -> list[int]:
    """The sizes to try, largest first: as wide as the safe zone, never taller than CAP_MAX."""
    cap_per_px = textmod.cap_height(face, 400) / 400.0
    widest = max(lines, key=lambda s: textmod.advance(s, face, 400))
    px = min(textmod.fit_px(widest, face, 0.97 * zone.w), int(CAP_MAX * H / cap_per_px))
    lo = max(8, int(CAP_MIN * H / cap_per_px))
    out = []
    while px >= lo:
        out.append(px)
        px = int(px * 0.95) if px > 60 else px - 2
    return out or [max(8, lo)]


def place(arrangements: list[tuple[str, ...]], face: str, store: MaskStore, size: tuple[int, int],
          share: Optional[float]) -> Placement:
    """The largest block that sits in the safe zone with its foot just behind the head, and where.

    Size comes first, since the word is the point: a large word only just touching the head beats
    a small one sunk behind it. So, in order: the large sizes (down to BIG of the largest) with
    the foot behind the head; the same sizes just touching it; the same sizes wholly above it
    (readable beats behind); then the smaller sizes the same three ways; and last, when nothing
    keeps every letter readable, the least-hidden place, with a warning."""
    W, H = size
    zone = SafeZone(W, H).rect
    tops = store.crowns()
    best: Optional[tuple[tuple, Placement]] = None
    for lines in arrangements:
        sizes = _sizes(lines, face, zone, H)
        big = [px for px in sizes if px >= BIG * sizes[0]]
        small = [px for px in sizes if px < BIG * sizes[0]]
        found = None
        for group, least in ((big, HIDE_MIN), (big, TOUCH), (big, 0.0), (small, HIDE_MIN), (small, TOUCH),
                             (small, 0.0)):
            for px in group:
                found = _position(make_block(lines, face, px), tops, zone, W, H, strict=True, least=least)
                if found is not None:
                    break
            if found is not None:
                break
        if found is None:
            found = _position(make_block(lines, face, sizes[-1]), tops, zone, W, H, strict=False)
        if found is None:
            continue
        if share is not None and share < 0.999:
            smaller = make_block(lines, face, max(8, int(found.block.px * share)))
            found = (_position(smaller, tops, zone, W, H, strict=True)
                     or _position(smaller, tops, zone, W, H, strict=True, least=TOUCH)
                     or _position(smaller, tops, zone, W, H, strict=False) or found)
        rank = (found.warn == "", round(found.block.cap, 1), -len(lines))
        if best is None or rank > best[0]:
            best = (rank, found)
    if best is None:
        raise ValueError("the word does not fit in the safe zone at a readable size; give a shorter word")
    return best[1]


# ---- colour and contrast -------------------------------------------------------------------------

def _lum(rgb) -> np.ndarray:
    """Relative luminance (0..1) of sRGB values 0..255."""
    c = np.asarray(rgb, np.float32) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]


def _ratio(a: float, b: float) -> float:
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


def wall_behind(clip, timeline, framing, store: MaskStore, rect: Rect, frames: list[int]) -> dict:
    """The wall right behind and round the letters, across the frames they are on, the person left
    out: its luminance (median, and its darker and brighter parts) and how busy it is. And the
    person's own crown under the letters' foot (hair, a cap), which the letters must also stand
    off: a head the same tone as the letters merges into them, and a letter whose foot is hidden
    by something of its own colour reads as another letter (an E as an F, a B as an R)."""
    import cv2

    from ..core import video

    W, H = framing.out_w, framing.out_h
    pad_x, pad_y = 0.04 * W, 0.25 * rect.h
    x0, y0 = int(max(0, rect.x0 - pad_x)), int(max(0, rect.y0 - pad_y))
    x1, y1 = int(min(W, rect.x1 + pad_x)), int(min(H, rect.y1 + pad_y))
    # the person inside the letters' box, and a sliver under it: what covers the letters' foot
    c0, c1 = int(max(0, rect.y0)), int(min(H, rect.y1 + 0.06 * rect.h))
    lums, busy, crowns = [], [], []
    have = [j for j in frames if j in store.by_frame]
    pick = have[:: max(1, len(have) // 6)][:6]
    fps, n_src = clip.info.rate, clip.info.frames
    for j in pick:
        k = timeline.source_frame(j, fps, n_src)
        f = framing.apply(video.read_frame(clip.info, k), k)
        full = cv2.resize(store.by_frame[j], (W, H), interpolation=cv2.INTER_LINEAR)
        region = f[y0:y1, x0:x1]
        wall = full[y0:y1, x0:x1] < 0.05
        head = full[c0:c1, int(max(0, rect.x0)):int(min(W, rect.x1))] > 0.9
        if head.sum() >= 20:
            crowns.append(_lum(f[c0:c1, int(max(0, rect.x0)):int(min(W, rect.x1))])[head])
        if wall.sum() < 50:
            continue
        lums.append(_lum(region)[wall])
        g = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        size = (max(8, g.shape[1] // 4), max(8, g.shape[0] // 4))
        g = cv2.resize(g, size, interpolation=cv2.INTER_AREA)
        lap = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))
        ws = cv2.resize(wall.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0.9
        if ws.any():
            busy.append(float(lap[ws].mean()))
    out = {"median": 0.5, "low": 0.3, "high": 0.7, "busy": 0.0}
    if lums:
        every = np.concatenate(lums)
        out = {"median": float(np.median(every)), "low": float(np.percentile(every, 15)),
               "high": float(np.percentile(every, 85)), "busy": float(np.mean(busy)) if busy else 0.0}
    if crowns:
        out["crown"] = float(np.median(np.concatenate(crowns)))
    return out


ACCENT = (232, 69, 44)  # the third ink: a red that stands off both a bright wall and dark hair
TARGET_CONTRAST = 4.5  # against the part of the wall that fights the letters most
MODERATE = 0.35  # the most shade the colour choice counts on: past that the shading is the look


def _darken_round(ink, wall: dict, style: str) -> bool:
    """Whether the shade darkens the wall round the letters (else it lightens it)."""
    return style == "glow" or float(_lum(ink)) > wall["median"]


def contrast_with(wall: dict, ink, style: str, s: float) -> float:
    """The letters' contrast with the part of the wall that fights them most, after shading `s`."""
    li = float(_lum(ink))
    if _darken_round(ink, wall, style):
        return _ratio(li, max(wall["high"], wall["median"]) * (1 - s))
    worst = min(wall["low"], wall["median"])
    return _ratio(li, worst + s * (1 - worst))


def shade_needed(wall: dict, ink, style: str) -> float:
    """How much to shade the wall right round the letters (0 to 0.7), away from the ink's tone, for
    them to reach the target contrast. A busy wall asks for more, and a glow always has some dark."""
    li = float(_lum(ink))
    target = TARGET_CONTRAST + (1.5 if wall["busy"] > 0.05 else 0.0)
    floor = 0.0
    if style == "glow":
        target, floor = max(target, 7.0), 0.3
    # A plain wall takes less shade than a busy one: there a dark cloud round the letters is the look.
    most = 0.7 if wall["busy"] > 0.05 or style == "glow" else 0.5
    if _darken_round(ink, wall, style):
        worst = max(wall["high"], wall["median"])
        want = (li + 0.05) / target - 0.05
        s = 0.0 if worst <= want else 1.0 - max(want, 0.0) / max(worst, 1e-4)
    else:
        worst = min(wall["low"], wall["median"])
        want = target * (li + 0.05) - 0.05
        s = 0.0 if worst >= want else (want - worst) / max(1e-4, 1.0 - worst)
        most = min(most, 0.6)
    s = float(np.clip(s, floor, most))
    return 0.0 if s < 0.08 else s


def choose_ink(wall: dict, color: str, style: str, covered: float = HIDE_TARGET) -> tuple[tuple[int, int, int], str]:
    """The ink, and why. Light, dark or the red, whichever stands off both the wall (allowing a
    moderate shade) and the person's crown. The wall counts for more, since it is where the letters
    are read; the crown counts as much as the head covers them (`covered`, the share hidden), so a
    word that only touches a dark cap is not made light for the cap's sake. Light or dark unless
    the red is clearly better."""
    if style == "glow":
        return (GLOW if color == "auto" else textmod.colour(color)), "the glow's colour"
    if color != "auto":
        return textmod.colour(color), "as asked"
    crown = wall.get("crown")

    def score(ink) -> tuple[float, float, float]:
        cw = contrast_with(wall, ink, style, min(shade_needed(wall, ink, style), MODERATE))
        cc = _ratio(float(_lum(ink)), crown) if crown is not None else 21.0
        wc = 0.35 * float(np.clip(covered / HIDE_TARGET, 0.2, 1.0))
        return cw ** (1 - wc) * cc ** wc, cw, cc

    scored = {name: (ink, score(ink)) for name, ink in (("light", LIGHT), ("dark", DARK), ("red", ACCENT))}
    neutral = max(("light", "dark"), key=lambda n: scored[n][1][0])
    name = "red" if scored["red"][1][0] > 1.15 * scored[neutral][1][0] else neutral
    ink, (_, cw, cc) = scored[name]
    return ink, f"{name}: {cw:.1f}:1 with the wall, {cc:.1f}:1 with the head"


# ---- the look ------------------------------------------------------------------------------------

def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    import cv2

    return a if sigma <= 0.3 else cv2.GaussianBlur(a, (0, 0), sigma)


def _morph(a: np.ndarray, r: float, grow: bool) -> np.ndarray:
    import cv2

    r = int(round(r))
    if r < 1:
        return a
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(a, k) if grow else cv2.erode(a, k)


def style_letters(m: np.ndarray, cap: float, style: str, ink) -> tuple[np.ndarray, np.ndarray, int]:
    """The letters (a 0..1 mask) styled: premultiplied RGB and alpha, padded far enough for the
    glow, the shadow and the shade to fade out inside the sprite, and the padding."""
    pad = int(math.ceil((1.25 if style == "glow" else 0.5) * cap))
    m = np.pad(m, pad)
    col = np.array(ink, np.float32)
    if style == "outline":
        w = max(2.0, 0.06 * cap)
        ring = np.clip(_morph(m, w / 2, True) - _morph(m, w / 2, False), 0, 1)
        a = _blur(ring, 0.5)
        return (a[..., None] * col).astype(np.float32), a.astype(np.float32), pad
    if style == "glow":
        # Light, not paint: the halo adds its colour to whatever is behind (no alpha of its own),
        # so it is vivid on the dimmed room and never a pastel smudge; the letters' core is opaque.
        core = 0.35 * col + 0.65 * 255.0
        halo = np.clip(_blur(m, 0.30 * cap) * 1.1 + _blur(m, 0.07 * cap) * 0.9, 0, 1.6)
        rgb = (halo * (1 - m))[..., None] * col * 0.85 + m[..., None] * core
        return rgb.astype(np.float32), m.astype(np.float32), pad
    rgb = m[..., None] * col
    a = m
    if float(_lum(ink)) > 0.3:  # a soft shadow under light letters lifts them off a busy wall: dark, not colour
        shadow = np.roll(_blur(m, 0.05 * cap), int(round(0.03 * cap)), axis=0) * 0.38
        a = m + shadow * (1 - m)
    return rgb.astype(np.float32), a.astype(np.float32), pad


def shade_of(letters: np.ndarray, cap: float, s: float, wide: bool = False) -> np.ndarray:
    """The shade behind the letters: strongest at their edges, gone a third of their height away
    (a glow's is wider: a pool of dark the light sits in)."""
    if s <= 0:
        return np.zeros_like(letters)
    grow, soft = (0.15, 0.32) if wide else (0.05, 0.12)
    sh = _blur(_morph(letters, grow * cap, True), soft * cap)
    return (np.clip(sh / max(1e-6, float(sh.max())), 0, 1) * s).astype(np.float32)


@dataclass
class Piece:
    """What lands at one moment (a word, or the whole block), drawn once: its letters and the shade."""
    text: str
    land: float
    x: float  # the sprite's top-left when landed, output pixels
    y: float
    ink: Sprite
    letters: np.ndarray  # the letters' own alpha (no shadow or glow), the sprite's size: for measuring
    shade: np.ndarray  # 0..1, the sprite's size: how much the wall round the letters is shaded
    foot: float  # the letters' baseline when landed (output y): a rise comes up out of it


def build_pieces(b: Block, pl: Placement, spec: list[dict], style: str, ink, shade_s: float) -> list[Piece]:
    """One piece per word when the words land one by one, else one for the block."""
    slots = word_slots(b)
    lands = [float(p["land"]) for p in spec]
    if len(spec) == len(slots) and len(set(round(x, 3) for x in lands)) > 1:
        groups = [((n,), spec[n]["text"], lands[n]) for n in range(len(slots))]
    else:
        groups = [(None, " ".join(b.lines), min(lands))]
    pieces = []
    for only, label, land in groups:
        m, _ = block_ink(b, only)
        rgb, a, pad = style_letters(m, b.cap, style, ink)
        letters = np.pad(m, pad).astype(np.float32)
        bases = [base for n, (_, _, _, base) in enumerate(slots) if only is None or n in only]
        pieces.append(Piece(label, land, pl.x - MARGIN - pad, pl.y - MARGIN - pad, Sprite(rgb=rgb, a=a), letters,
                            shade_of(letters, b.cap, shade_s, wide=style == "glow"), pl.y + max(bases) + 0.02 * b.cap))
    return pieces


# ---- the motion ----------------------------------------------------------------------------------

def _ease_out(u: float, p: float = 3.0) -> float:
    u = min(1.0, max(0.0, u))
    return 1.0 - (1.0 - u) ** p


def _ease_back(u: float, k: float = 1.4) -> float:
    u = min(1.0, max(0.0, u))
    return 1.0 + (k + 1.0) * (u - 1) ** 3 + k * (u - 1) ** 2


def _smooth(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


# Each entrance: how long it runs, and how long before the word's sound it starts, so the letters
# are in place on the frame the word is heard.
ENTRANCE = {"rise": (0.34, 0.30), "drop": (0.40, 0.27), "fade": (0.36, 0.24), "scale": (0.26, 0.22)}
EXIT = 0.28
BREATHE = 0.02  # how much it grows while it holds, so it never sits dead still


@dataclass
class State:
    opacity: float
    dy: float  # as a share of the block's height
    scale: float
    clip_foot: bool  # nothing drawn below the landed baseline: a rise comes up out of a line


def entrance(kind: str, u: float) -> State:
    if kind == "rise":
        return State(min(1.0, u / 0.3), (1.0 - _ease_out(u, 3.0)) * 1.05, 1.0, True)
    if kind == "drop":
        return State(min(1.0, u / 0.22), -(1.0 - _ease_back(u, 1.4)) * 0.9, 1.0, False)
    if kind == "fade":
        return State(_smooth(u), 0.0, 1.0 + 0.035 * (1.0 - _ease_out(u)), False)
    return State(min(1.0, u / 0.35), 0.0, 1.0 + 0.32 * (1.0 - _ease_out(u, 4.0)), False)


def leaving(kind: str, u: float) -> State:
    if kind == "rise":
        return State(1.0 - _smooth(u), _ease_out(u, 2.0) * 0.8, 1.0, True)
    return State(1.0 - _smooth(u), 0.0, 1.0 - 0.03 * u, False)


# ---- drawing -------------------------------------------------------------------------------------

def _wall_estimate(I: np.ndarray, alpha: np.ndarray, sigma: float) -> np.ndarray:
    """The wall's colour under each pixel the person partly covers, from the wall nearby."""
    import cv2

    h, w = alpha.shape
    sw, sh = max(2, w // 2), max(2, h // 2)
    a_s = cv2.resize(alpha, (sw, sh), interpolation=cv2.INTER_AREA)
    I_s = cv2.resize(I, (sw, sh), interpolation=cv2.INTER_AREA)
    wt = (a_s < 0.06).astype(np.float32)
    num = cv2.GaussianBlur(I_s * wt[..., None], (0, 0), sigma / 2)
    den = cv2.GaussianBlur(wt, (0, 0), sigma / 2)
    est = np.where((den > 0.02)[..., None], num / np.maximum(den, 1e-3)[..., None], I_s)
    return cv2.resize(est, (w, h), interpolation=cv2.INTER_LINEAR)


def refine_edge(frame: np.ndarray, alpha: np.ndarray, sigma: float) -> np.ndarray:
    """The person's edge decided by colour where the mask is unsure. Each unsure pixel is placed
    between the colour of the person just inside it and the wall just outside it: a pixel the
    colour of the wall (a vase the model half-took for hair) becomes wall, a real strand stays
    person, and a gap between strands lets the letters through. Where the person and the wall are
    too close in colour to tell apart, the mask is kept as it was."""
    import cv2

    fg = (alpha >= 0.97).astype(np.float32)
    bg = (alpha <= 0.03).astype(np.float32)
    unsure = (fg == 0) & (bg == 0)
    if not unsure.any() or not fg.any() or not bg.any():
        return alpha
    I = frame.astype(np.float32)

    def local(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        num = cv2.GaussianBlur(I * weight[..., None], (0, 0), sigma)
        den = cv2.GaussianBlur(weight, (0, 0), sigma)
        return num / np.maximum(den, 1e-4)[..., None], den

    F, dF = local(fg)
    B, dB = local(bg)
    d = F - B
    apart = (d * d).sum(axis=-1)
    by_colour = np.clip(((I - B) * d).sum(axis=-1) / np.maximum(apart, 1.0), 0.0, 1.0)
    trust = (np.clip((np.sqrt(apart) - 20.0) / 50.0, 0.0, 1.0)
             * np.clip(dF * 5.0, 0.0, 1.0) * np.clip(dB * 5.0, 0.0, 1.0))
    out = np.where(unsure, trust * by_colour + (1.0 - trust) * alpha, alpha)
    return out.astype(np.float32)


def composite_behind(frame: np.ndarray, rgb: np.ndarray, a: np.ndarray, alpha: np.ndarray, sigma: float) -> np.ndarray:
    """A premultiplied layer (rgb, a) laid behind the person (alpha 0..1). On the person's edge a
    pixel is part person, part wall: only the wall's part takes the layer, using the wall's colour
    from nearby, so the old wall leaves no rim. Where the pixel is all wall it is the plain 'over'."""
    I = frame.astype(np.float32)
    edge = (alpha > 0.02) & (alpha < 0.98) & (a > 0.004)
    under = I
    if edge.any():
        beta = np.clip((alpha - 0.04) / 0.5, 0.0, 1.0)
        beta = (beta * beta * (3 - 2 * beta))[..., None]
        under = I * (1.0 - beta) + _wall_estimate(I, alpha, sigma) * beta
    out = I + (1.0 - alpha[..., None]) * (rgb - a[..., None] * under)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


class Painter:
    """Draws the word on each output frame, behind the person."""

    def __init__(self, clip, timeline, framing, pieces: list[Piece], kind: str, t_out: float, end_hold: bool,
                 shade_rgb: tuple, block_h: float, lean: Optional[float] = None, wall_swap: bool = True,
                 by_colour: bool = True, dim: float = 0.0):
        self.clip, self.timeline, self.framing = clip, timeline, framing
        self.pieces = pieces
        self.kind = kind
        self.t_out = t_out
        self.end_hold = end_hold  # held to the end of the clip: no exit
        self.shade_rgb = np.array(shade_rgb, np.float32)
        self.dim = dim  # how far the whole room behind the person dims while the word is up (glow)
        self.block_h = block_h
        self.W, self.H = framing.out_w, framing.out_h
        self.masks = clip.person()
        self.lean = lean
        self.wall_swap = wall_swap
        self.by_colour = by_colour
        self._iter = None
        self._k = -1
        self._last: Optional[np.ndarray] = None
        self.dur, self.lead = ENTRANCE[kind]
        self.t_in = max(0.0, min(p.land for p in pieces) - self.lead)
        self.sigma = max(4.0, 0.012 * self.W)
        self.measured: list[tuple[float, float]] = []  # (t, share of the letters hidden), once all have landed

    def _mask(self, k: int) -> np.ndarray:
        if self._iter is None or k < self._k:
            self._iter, self._k = iter(self.masks.frames()), -1
        while self._k < k:
            nxt = next(self._iter, None)
            if nxt is None:
                break
            self._last, self._k = nxt, self._k + 1
        assert self._last is not None
        return self._last

    def state(self, p: Piece, t: float) -> Optional[State]:
        start = p.land - self.lead
        if t < start or (not self.end_hold and t >= self.t_out):
            return None
        s = entrance(self.kind, (t - start) / self.dur) if t < start + self.dur else State(1.0, 0.0, 1.0, False)
        if not self.end_hold and t > self.t_out - EXIT:
            lv = leaving(self.kind, (t - (self.t_out - EXIT)) / EXIT)
            s = State(s.opacity * lv.opacity, s.dy + lv.dy, s.scale * lv.scale, s.clip_foot or lv.clip_foot)
        s.scale *= 1.0 + BREATHE * min(1.0, max(0.0, t - p.land) / 4.0)
        return s

    def active(self, t: float) -> bool:
        return t >= self.t_in and (self.end_hold or t < self.t_out)

    def layer(self, t: float) -> Optional[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]]:
        """The rows the word covers at time t, and the layer for them: premultiplied colour, alpha,
        and the letters' own alpha. The letters move as the animation says; the shade behind them
        stays where they land and only fades, so the wall seems to darken as they arrive."""
        import cv2

        states = [(p, self.state(p, t)) for p in self.pieces]
        states = [(p, s) for p, s in states if s is not None and s.opacity > 0.002]
        if not states:
            return None
        W, H = self.W, self.H
        placed = []
        for p, s in states:
            ink, letters = p.ink, p.letters
            if abs(s.scale - 1.0) > 1e-3:
                ink = ink.scaled(s.scale)
                letters = cv2.resize(p.letters, (ink.w, ink.h), interpolation=cv2.INTER_LINEAR)
            x = p.x + p.ink.w / 2 - ink.w / 2  # grown about the foot's middle
            y = p.y + p.ink.h - ink.h + s.dy * self.block_h
            placed.append((p, s, ink, letters, x, y))
        tops = [y for *_, y in placed] + [p.y for p, _ in states]
        bottoms = [y + ink.h for _, _, ink, _, _, y in placed] + [p.y + p.ink.h for p, _ in states]
        y0, y1 = int(max(0, math.floor(min(tops)))), int(min(H, math.ceil(max(bottoms))))
        if self.dim > 0:  # the whole room dims, so every row is drawn
            y0, y1 = 0, H
        if y1 <= y0:
            return None
        bh = y1 - y0
        rgb = np.zeros((bh, W, 3), np.float32)
        a = np.zeros((bh, W), np.float32)
        lets = np.zeros((bh, W), np.float32)
        shade_a = np.zeros((bh, W), np.float32)

        def window(x: float, y: float, w: int, h: int):
            xi, yi = int(round(x)), int(round(y)) - y0
            dx0, dy0, dx1, dy1 = max(0, xi), max(0, yi), min(W, xi + w), min(bh, yi + h)
            if dx1 <= dx0 or dy1 <= dy0:
                return None
            src = (slice(dy0 - yi, dy1 - yi), slice(dx0 - xi, dx1 - xi))
            return src, (slice(dy0, dy1), slice(dx0, dx1)), dy0

        for p, s in states:  # the shade, where the letters land
            got = window(p.x, p.y, p.ink.w, p.ink.h)
            if got is not None and float(p.shade.max()) > 0:
                src, dst, _ = got
                shade_a[dst] = np.maximum(shade_a[dst], p.shade[src] * s.opacity)
        for p, s, ink, letters, x, y in placed:  # the letters, where they are now
            got = window(x, y, ink.w, ink.h)
            if got is None:
                continue
            src, dst, dy0 = got
            ia, irgb, la = ink.a[src] * s.opacity, ink.rgb[src] * s.opacity, letters[src] * s.opacity
            if s.clip_foot:
                cut = int(round(p.foot)) - y0 - dy0
                if cut < ia.shape[0]:
                    cut = max(0, cut)
                    ia, irgb, la = ia.copy(), irgb.copy(), la.copy()
                    ia[cut:], irgb[cut:], la[cut:] = 0, 0, 0
            rgb[dst] = rgb[dst] * (1 - ia[..., None]) + irgb
            a[dst] = a[dst] * (1 - ia) + ia
            lets[dst] = np.maximum(lets[dst], la)
        if self.dim > 0:
            shade_a = 1.0 - (1.0 - shade_a) * (1.0 - self.dim * max(s.opacity for _, s in states))
        layer_a = shade_a + a * (1 - shade_a)
        layer_rgb = (shade_a * (1 - a))[..., None] * self.shade_rgb + rgb
        return y0, y1, layer_rgb, layer_a, lets

    def person_alpha(self, frame: np.ndarray, j: int) -> np.ndarray:
        k = self.timeline.source_frame(j, self.clip.info.rate, self.clip.info.frames)
        crop = self.framing.window(k) if self.framing.reframed else None
        lean = EDGE if self.lean is None else self.lean
        return self.masks.full(self._mask(k), (self.W, self.H), guide=frame, crop=crop, lean=lean,
                               reach=REACH * self.W)

    def draw(self, frame: np.ndarray, j: int, t: float) -> None:
        from ..core.compose import behind_person

        if not self.active(t):
            return
        got = self.layer(t)
        if got is None:
            return
        y0, y1, rgb, a, lets = got
        if float(a.max()) <= 0.002:
            return
        alpha = self.person_alpha(frame, j)[y0:y1]
        if self.by_colour:
            alpha = refine_edge(frame[y0:y1], alpha, self.sigma * 0.75)
        if self.wall_swap:
            frame[y0:y1] = composite_behind(frame[y0:y1], rgb, a, alpha, self.sigma)
        else:
            frame[y0:y1] = behind_person(frame[y0:y1], rgb, a, alpha)
        if all(t >= p.land + 0.05 for p in self.pieces) and (self.end_hold or t < self.t_out - EXIT):
            tot = float(lets.sum())
            if tot > 0:
                self.measured.append((t, float((lets * alpha).sum()) / tot))


# ---- deciding what and when ----------------------------------------------------------------------

def _is_time(v: str) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False


def _line_end(words: list[Word], quiet, i: int) -> float:
    for n in range(i, len(words)):
        if words[n].ends_clause or pause_after(words, n, quiet) >= 0.3:
            return words[n].end
    return words[-1].end


def decide(clip, opts: dict, seconds: float) -> tuple[list[dict], float, bool, dict]:
    """What is shown and when: each word with the moment it lands, when it leaves, whether it holds
    to the end of the clip, and what was chosen and why."""
    text, when, hold = opts["text"], opts["when"], opts["hold"]
    info: dict = {}
    words: list[Word] = []
    quiet: list = []
    back = moments.TakenBack()
    if not (text != "auto" and _is_time(when)):
        words = [w for w in clip.words if w.key not in HESITATIONS]
        quiet = clip.quiet
        back = moments.taken_back(clip.words, clip.quiet)  # read with the fillers, which break a first attempt off
    line_end: Optional[float] = None
    if text == "auto":
        if _is_time(when):
            cand, alts = choose(words, quiet, start_after=float(when), back=back)
        elif when != "auto":
            span = find_said(words, when)
            if span is None:
                raise ValueError(f"could not find \"{when}\" in what is said; `python -m fx.run words <video>` "
                                 "lists the words, or give a time such as when=2.4")
            cand, alts = choose(words, quiet, within=(words[span[0]].start, words[span[1]].end), back=back)
        else:
            cand, alts = choose(words, quiet, back=back)
        if cand is None:
            raise ValueError("no spoken word stands out enough to put behind you (or nothing clear is said); "
                             "give one: text=\"YOUR WORD\"")
        shown = [bare(cand.word)]
        lands = [cand.word.start]
        line_end = words[cand.line[-1]].end
        info = {"word": bare(cand.word), "said_at": round(float(cand.word.start), 3), "score": round(float(cand.score), 1),
                "line": " ".join(words[i].text for i in cand.line),
                "runners_up": [{"word": bare(c.word), "said_at": round(float(c.word.start), 2), "score": round(float(c.score), 1)}
                               for c in alts]}
        place = {w: i for i, w in enumerate(words)}
        passed = [{"word": bare(w), "said_at": round(float(w.start), 2), "why": why}
                  for w, why in back.why.items() if w in place and moments.key_score(words, place[w]) > 0]
        if passed:
            info["passed_over"] = passed[:6]  # words that could have been chosen, but the speaker took them back
    else:
        shown = text.split()
        said_at = None
        if _is_time(when):
            lands = [float(when)] * len(shown)
        else:
            inner = find_said(words, text)
            if inner is not None and inner[1] - inner[0] + 1 == len(shown) and (
                    when == "auto" or (find_said(words, when) or (-1, -1))[0] <= inner[0] <= (find_said(words, when) or (-1, -1))[1]):
                lands = [words[inner[0] + k].start for k in range(len(shown))]  # each word as it is said
                said_at = words[inner[0]].start
                line_end = _line_end(words, quiet, inner[1])
            elif when != "auto":
                span = find_said(words, when)
                if span is None:
                    raise ValueError(f"could not find \"{when}\" in what is said; give a time such as when=2.4")
                lands = [words[span[0]].start] * len(shown)
                line_end = words[span[1]].end
            else:
                cand, _ = choose(words, quiet, back=back) if words else (None, [])
                if cand is not None:
                    lands = [words[cand.line[0]].start] * len(shown)
                    line_end = words[cand.line[-1]].end
                else:
                    lands = [min(0.5, seconds / 4)] * len(shown)
                info["warn"] = f"\"{text}\" is not said in the clip, so it lands with the opening line"
        info.update({"word": text, "said_at": round(said_at, 3) if said_at is not None else None})
    lands = [max(0.0, x) for x in lands]
    if min(lands) >= seconds - 0.3:
        raise ValueError(f"the word would land at {min(lands):.2f} s, too close to the end of the clip ({seconds:.2f} s)")
    last = max(lands)
    end_hold = False
    if hold == "end":
        t_out, end_hold = seconds, True
    elif hold == "auto":
        t_out = min(max((line_end if line_end is not None else last) + 0.5, last + 2.0), last + 3.6)
    else:
        t_out = last + float(hold)
    if t_out >= seconds - 0.4:
        t_out, end_hold = seconds, True
    return [{"text": s, "land": round(x, 3)} for s, x in zip(shown, lands)], t_out, end_hold, info


def _display(s: str) -> str:
    return s.replace("'", "’").upper()


def arrangements(words: list[str], most: int = 3) -> list[tuple[str, ...]]:
    """The ways to set a phrase on one to `most` lines, each split as evenly as it can be."""
    out = []
    n = len(words)
    for k in range(1, min(most, n) + 1):
        best, best_cost = None, math.inf

        def walk(start: int, left: int, acc: list[list[str]]) -> None:
            nonlocal best, best_cost
            if left == 1:
                lines = acc + [words[start:]]
                lens = [len(" ".join(x)) for x in lines]
                cost = max(lens) - min(lens)
                if cost < best_cost:
                    best, best_cost = tuple(" ".join(x) for x in lines), cost
                return
            for cut in range(start + 1, n - left + 2):
                walk(cut, left - 1, acc + [words[start:cut]])

        walk(0, k, [])
        if best:
            out.append(best)
    return out


def _plan_values(plan: dict, seconds: float, PlanError):
    """The decisions a plan holds, checked."""
    try:
        lines = tuple(str(s) for s in plan["lines"])
        pieces = [{"text": str(p["text"]), "land": float(p["land"])} for p in plan["pieces"]]
        px = int(plan["px"])
        cx, by = float(plan["centre_x"]), float(plan["bottom_y"])
        out = float(plan["out"])
        color = str(plan["color"])
        shade = float(plan.get("shade", 0.0))
        face = str(plan["font_face"])
    except (KeyError, TypeError, ValueError):
        raise PlanError("the plan needs lines, pieces, px, centre_x, bottom_y, out, color and font_face; "
                        "run once without --plan to get one to edit") from None
    if not lines or not all(s.strip() for s in lines):
        raise PlanError("lines must be the lines of text to show")
    if not pieces:
        raise PlanError("pieces must list each word with the second it lands")
    for p in pieces:
        if not 0 <= p["land"] < seconds:
            raise PlanError(f"a word lands at {p['land']} s, outside the clip (0–{seconds:.2f} s)")
    if face not in textmod.FACES:
        raise PlanError(f"font_face must be one of the pack's fonts: {', '.join(textmod.FACES)}")
    if not 8 <= px <= 2400:
        raise PlanError(f"px {px} is not a usable size")
    if not 0.0 <= shade <= 0.9:
        raise PlanError("shade is a share from 0 to 0.9")
    try:
        textmod.hex_rgb(color)
    except ValueError as e:
        raise PlanError(str(e)) from None
    if out <= min(p["land"] for p in pieces):
        raise PlanError("out must come after the word lands")
    end_hold = bool(plan.get("hold_to_end", False)) or out >= seconds - 1e-3
    return lines, pieces, px, cx, by, min(out, seconds), end_hold, color, shade, face


# ---- the effect ----------------------------------------------------------------------------------

def prepare(clip, opts: dict, plan: Optional[dict] = None, lean: Optional[float] = None, wall_swap: bool = True,
            by_colour: bool = True):
    from ..core import framing as framing_mod
    from ..core.timeline import Timeline
    from . import Job, PlanError

    if plan is not None:  # the plan is the record: its own look comes back with it
        kept = plan.get("options") or {}
        opts = dict(opts, **{k: kept[k] for k in ("style", "animation", "frame", "font") if k in kept})
        if opts["style"] not in STYLES or opts["animation"] not in ANIMATIONS or opts["frame"] not in ("keep", "9:16"):
            raise PlanError("the plan's options name a style, animation or frame this effect does not have")
    info = clip.info
    fps = info.rate
    fr = framing_mod.plan(info.width, info.height, fps, info.frames, clip.face, shape=opts["frame"])
    tl = Timeline.whole(info.frames / fps)
    W, H = fr.out_w, fr.out_h
    seconds = tl.duration
    n_out = tl.output_frames(fps)
    style, kind = opts["style"], opts["animation"]

    if plan is not None:
        lines, spec, px, cx, by, t_out, end_hold, color_hex, shade_s, face = _plan_values(plan, seconds, PlanError)
        chosen = plan.get("chosen", {})
    else:
        spec, t_out, end_hold, chosen = decide(clip, opts, seconds)
        face = FONTS[opts["font"]]

    t_in = max(0.0, min(p["land"] for p in spec) - ENTRANCE[kind][1])
    t_end = seconds if end_hold else t_out
    span = list(range(int(math.floor(t_in * fps)), int(min(n_out - 1, math.ceil(t_end * fps))) + 1))
    sample = span[:: max(1, len(span) // 48)]
    store = MaskStore(clip, tl, fr, (W, H), sample)
    if not store.by_frame or max(float(m.max()) for m in store.by_frame.values()) < 0.5:
        raise ValueError("nobody is in the shot while the word is on screen, so there is nobody to put it behind")

    if plan is not None:
        b = make_block(lines, face, px)
        tops = store.crowns()
        x, y = cx - b.w / 2, by - b.h
        med, p90, lp = _scan(b, tops, x, np.array([y]))
        pl = Placement(b, x, y, float(med[0]), float(p90[0]), float(lp[0]), "where the plan puts it")
    else:
        shown = [_display(p["text"]) for p in spec]
        share = None if opts["size"] == "auto" else float(opts["size"])
        pl = place(arrangements(shown), face, store, (W, H), share)
    b = pl.block
    rect = pl.rect

    wall = wall_behind(clip, tl, fr, store, rect, sample)
    if plan is not None:
        ink, why = textmod.hex_rgb(color_hex), "as the plan says"
    else:
        ink, why = choose_ink(wall, opts["color"], style, covered=pl.hidden_median)
        shade_s = GLOW_SHADE if style == "glow" else shade_needed(wall, ink, style)
    dim = GLOW_DIM if style == "glow" else 0.0
    if style == "glow":  # the letters' bright core against the dimmed, shaded room
        core = tuple(0.35 * c + 0.65 * 255.0 for c in ink)
        contrast = _ratio(float(_lum(core)), max(wall["high"], wall["median"]) * (1 - dim) * (1 - shade_s))
    else:
        contrast = contrast_with(wall, ink, style, shade_s)

    pieces = build_pieces(b, pl, spec, style, ink, shade_s)
    painter = Painter(clip, tl, fr, pieces, kind, t_out, end_hold,
                      (0, 0, 0) if _darken_round(ink, wall, style) else (255, 255, 255),
                      b.h, lean=lean, wall_swap=wall_swap, by_colour=by_colour, dim=dim)

    zone = SafeZone(W, H)
    last_land = max(p.land for p in pieces)
    landed = [j for j in range(n_out) if last_land <= j / fps < (t_end if not end_hold else seconds + 1)]
    land_at = min(p.land for p in pieces)
    said = chosen.get("said_at")
    checks = {
        "word": " ".join(b.lines) + (" (as the plan says)" if plan is not None else
                                     f" (from \"{chosen['line']}\")" if chosen.get("line") else ""),
        "lands_at": f"{land_at:.2f} s" + (f", as it is said ({said:.2f} s)" if said is not None else ""),
        "on_screen": f"{painter.t_in:.2f}–{t_end:.2f} s",
        "frames_over_face": _over_face(clip, tl, fr, store, b, pl, landed),
        "frames_outside_safe_zone": 0 if zone.contains(rect) else len(landed),
        "placed": pl.where,
        "hidden": f"{pl.hidden_median:.0%} of the letters behind the person (at worst {pl.hidden_p90:.0%}; "
                  f"the most-hidden letter at {pl.letter_p90:.0%} of what it may lose)",
        "colour": why,
        "contrast": f"{contrast:.1f}:1 against the wall"
                    + (f", the wall round the letters shaded {shade_s:.0%}" if shade_s else "")
                    + (f", the room behind you dimmed {dim:.0%}" if dim else ""),
        "warnings": [w for w in (pl.warn, chosen.get("warn", "")) if w],
    }
    out_plan = {
        "effect": NAME, "clip": clip.path, "options": opts,
        "output": {"width": W, "height": H, "fps": round(fps, 3), "seconds": round(seconds, 3), "framing": fr.note},
        "text": " ".join(b.lines), "lines": list(b.lines),
        "pieces": [{"text": p["text"], "land": round(float(p["land"]), 3)} for p in spec],
        "chosen": chosen,
        "in": round(painter.t_in, 3), "land": round(land_at, 3), "out": round(t_end, 3), "hold_to_end": end_hold,
        "font_face": face, "px": b.px, "color": "#%02x%02x%02x" % tuple(int(v) for v in ink),
        "shade": round(shade_s, 3),
        "block": rect.as_list(), "centre_x": round(rect.cx, 1), "bottom_y": round(rect.y1, 1),
        "measured": {"hidden_median": round(pl.hidden_median, 3), "hidden_p90": round(pl.hidden_p90, 3),
                     "letter_p90": round(pl.letter_p90, 3), "contrast": round(contrast, 2),
                     "wall": {k: round(v, 3) for k, v in wall.items()}},
        "sheet_times": sheet_times(painter, pieces, t_end, fps, n_out),
    }

    def still_times() -> list[float]:
        hold_mid = last_land + 0.5 * max(0.2, t_end - last_land)
        return [round(v, 3) for v in (land_at - 0.5 * painter.lead, last_land + 0.1, hold_mid,
                                      max(last_land + 0.2, t_end - 0.45))]

    def after(result: dict, out_path) -> dict:
        if not painter.measured:
            return {}
        h = np.array([v for _, v in painter.measured])
        return {"hidden on the render": f"{np.median(h):.0%} of the letters behind the person "
                                        f"(at worst {h.max():.0%}) over {len(h)} frames"}

    return Job(NAME, clip, tl, fr, painter.draw, out_plan, checks=checks, still_times=still_times, after=after)


def _over_face(clip, tl, fr, store: MaskStore, b: Block, pl: Placement, landed: list[int]) -> int:
    """Frames where a letter would show inside the face's own box: only where the person mask
    missed part of the face. Measured on the sampled frames, counted over every landed frame."""
    import cv2

    face = clip.face
    if not face.any() or not store.by_frame:
        return 0
    ink_full, _ = block_ink(b)
    s = 1.0 / Q
    ink = cv2.resize(ink_full, (max(1, int(round(ink_full.shape[1] * s))), max(1, int(round(ink_full.shape[0] * s)))),
                     interpolation=cv2.INTER_AREA)
    ox, oy = int(round((pl.x - MARGIN) * s)), int(round((pl.y - MARGIN) * s))
    landed_set = set(landed)
    checked = [j for j in sorted(store.by_frame) if j in landed_set]
    count = 0
    fps, n_src = clip.info.rate, clip.info.frames
    for j in checked:
        k = tl.source_frame(j, fps, n_src)
        box = face.box(k)
        if box is None:
            continue
        r = fr.to_output(box, k).scale(s, s)
        m = store.by_frame[j]
        canvas = np.zeros_like(m)
        ih, iw = ink.shape
        cy0, cx0, cy1, cx1 = max(0, oy), max(0, ox), min(m.shape[0], oy + ih), min(m.shape[1], ox + iw)
        if cy1 > cy0 and cx1 > cx0:
            canvas[cy0:cy1, cx0:cx1] = ink[cy0 - oy:cy1 - oy, cx0 - ox:cx1 - ox]
        x0, y0, x1, y1 = int(max(0, r.x0)), int(max(0, r.y0)), int(min(m.shape[1], r.x1)), int(min(m.shape[0], r.y1))
        if x1 <= x0 or y1 <= y0:
            continue
        shows = float((canvas[y0:y1, x0:x1] * (1.0 - m[y0:y1, x0:x1])).sum()) / ((x1 - x0) * (y1 - y0))
        count += shows > 0.004
    return int(round(count * len(landed) / len(checked))) if checked else 0


def sheet_times(painter: Painter, pieces: list[Piece], t_end: float, fps: float, n_out: int) -> list[float]:
    """Six moments across the word's seconds: on its way in, landed, three while it holds, leaving."""
    first = min(p.land for p in pieces)
    land = max(p.land for p in pieces)
    hold = max(0.3, t_end - land)
    leave = t_end - 1.0 / fps if painter.end_hold else t_end - 0.55 * EXIT
    ts = [first - 0.5 * painter.lead, land + 0.04, land + 0.3 * hold, land + 0.55 * hold, land + 0.8 * hold, leave]
    return [round(min((n_out - 1) / fps, max(0.0, t)), 3) for t in ts]
