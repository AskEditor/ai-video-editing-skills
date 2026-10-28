"""pictures-behind: your own reels, screenshots or headlines piled up, pinned in a wall, or flash-cut
behind you, with you cut out in front of them.

Three looks (`mode`):

- **pile**: each picture drops onto a loose pile behind your head, a little turned, casting a soft
  shadow on the wall; the newest lies on top.
- **wall**: the pictures pin up in a grid on the wall behind you, one tile at a time.
- **flash**: one frame behind your head, and the picture in it cuts to the next on each word, held
  in exactly the same place (the news-page cut).

How it decides, in order:

1. **Which picture on which word.** A picture named for what it shows (`city.jpg`), listed in a
   `tags.txt` beside it, or given with `map`, lands on the word that names it (the core's matcher,
   `fx/core/match.py`, never on a word the speaker took back). The rest land **in folder order**,
   one on each word from the moment chosen (`at`, by default the line that lands, from the shared
   picker `fx/core/moments.py`), so a pile builds or a flash cuts in time with the voice. `on=keywords`
   puts every picture on a thing named instead, `on=order` puts them all in folder order, and
   seconds can be given.
2. **Where.** Measured against where you are over the clip: every picture stays at least partly in
   view (the share each keeps is printed), the pile and the flash frame sit behind your head so you
   stand in front of them, and the wall fills the room round you.
3. **The edge.** You are cut out as `backdrop` cuts you out (`core/matte.Cutout`: two models
   agreeing, only the speaker, a firm edge held steady), and only where a picture lies is the edge
   worked out again, so no rim of the old wall shows on a picture and the rest of the frame is as
   filmed. The run measures the face: how much of it the cut-out would let a picture through.
4. **Sound.** A soft pop as each picture lands (a light tick for each cut of the flash), on the same
   frame, well under your voice. `sound=off` keeps your sound exactly.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..core import depth, match, matte, moments
from ..core import text as textmod
from ..core.layout import Rect
from ..core.words import Word
from . import options as O
from . import pictures_above_head as pah

NAME = "pictures-behind"
MODES = ("pile", "wall", "flash")
STYLES = ("clean", "print")
SIZES = ("small", "medium", "large")


# ---- the options -------------------------------------------------------------------------------

@dataclass(frozen=True)
class _Lines:
    help: str

    def parse(self, raw) -> str:
        parts = [p.strip() for p in str(raw).split("|") if p.strip()]
        if len(parts) > 12:
            raise O.OptionError("give at most twelve headlines, separated by |")
        for p in parts:
            if len(p) > 90:
                raise O.OptionError(f"a headline is at most 90 characters (got {len(p)}: {p[:30]}…)")
        return " | ".join(parts)


@dataclass(frozen=True)
class _On:
    help: str

    def parse(self, raw) -> str:
        v = str(raw).strip().strip("\"'").lower()
        if v in ("", "auto"):
            return "auto"
        if v in ("keywords", "keyword", "words"):
            return "keywords"
        if v in ("map", "names", "tags", "named"):
            return "map"
        if v in ("order", "folder", "folder order", "in order"):
            return "order"
        if v.startswith("times"):
            v = v[5:].lstrip(" =:")
        try:
            times = [float(p.rstrip("s")) for p in re.split(r"[,;\s]+", v) if p]
        except ValueError:
            raise O.OptionError(f"say auto, keywords, map, order, or seconds such as 2.5, 6, 11 (got {raw!r})") from None
        if not times or any(t < 0 for t in times):
            raise O.OptionError("give seconds from 0 on, such as 2.5, 6, 11")
        return ", ".join(f"{t:g}" for t in sorted(times))


@dataclass(frozen=True)
class _At:
    help: str

    def parse(self, raw) -> str:
        v = str(raw).strip().strip("\"'")
        return "auto" if v.lower() in ("", "auto") else v


OPTIONS = {
    "mode": O.Choice(MODES, "pile: each picture drops onto a loose pile behind your head · wall: they pin up in a "
                            "grid on the wall behind you · flash: one frame behind your head that cuts to the next "
                            "picture on each word"),
    "pictures": pah.OPTIONS["pictures"],
    "headlines": _Lines("your own headlines or post titles, separated by |, each drawn on a paper card "
                        "(\"I quit my job | 30 days of posting\"); they join the pile after your pictures"),
    "map": pah.OPTIONS["map"],
    "on": _On("auto: a picture named for what it shows lands on its word, the rest in folder order, one on each "
              "word from `at` · keywords: every picture on a thing you name · map: only on words named by map, "
              "a file name or tags.txt · order: all in folder order from `at` · or seconds such as 2.5, 6, 11"),
    "at": _At("where the pictures in folder order start: auto (the line that lands), a word or words you say "
              "(\"my hotel\"), or seconds such as 4.5"),
    "hold": pah.OPTIONS["hold"],
    "size": O.Choice(SIZES, "how large the pictures are: small, medium (the default) or large; they always stay "
                            "partly in view round you"),
    "style": O.Choice(STYLES, "clean: rounded with a thin dark edge · print: a white border, like a printed photo"),
    "move": O.Choice(("on", "off"), "on: the pictures drift a little with the camera, as a real wall would · off: "
                                    "they hold still"),
    "sound": O.Switch("on: a soft pop as each picture lands, well under your voice · off: your sound exactly as it "
                      "was"),
    "cutout": O.Choice(("careful", "quick"), "careful: two looks at where you are, keeping only what both agree is "
                                               "you · quick: one look, faster the first time"),
    "fix": O.Text("misheard words as wrong=right, separated by ;  e.g. \"Jon=John\""),
}

# The runs `tests/matrix.py pictures-behind` makes, on our own photos and B-roll stills and
# AskEditor's own product screens (`{lab}/pictures-behind/inputs`, links; its README says which).
_IN = "{lab}/pictures-behind/inputs"
MATRIX = [
    ("D", "D", {"pictures": f"{_IN}/reels"}),
    ("D-wall", "D", {"mode": "wall", "pictures": f"{_IN}/wall"}),
    ("D-flash", "D", {"mode": "flash", "pictures": f"{_IN}/pages", "style": "print"}),
    ("B", "B", {"pictures": f"{_IN}/screens"}),
    ("B-headlines", "B", {"mode": "pile", "headlines": "I edit every reel in one sentence | 30 days, one video a "
                                                       "day | The editor that listens | My first 1,000 followers",
                          "style": "print"}),
    ("A", "A", {"pictures": f"{_IN}/screens", "size": "small"}),
    ("C", "C", {"mode": "flash", "pictures": f"{_IN}/pages"}),
    ("F", "F", {"mode": "wall", "pictures": f"{_IN}/wall", "at": "3"}),
]

# ---- the rules (numbers chosen for this pack) ------------------------------------------------------

GAP = {"pile": 0.20, "wall": 0.16, "flash": 0.14}  # seconds: two pictures never land closer than this
MIN_ON = 0.9  # the last picture is up at least this long
WAIT = 0.6  # seconds: in folder order, a picture never waits longer than this for the next word...
BEAT = 1.6  # ...it lands this many gaps after the last one instead
LAND = {"pile": 0.24, "wall": 0.28, "flash": 0.12}  # seconds a picture takes to settle
EXIT = 0.30
HOLD_AUTO = (1.6, 4.0)  # after the last picture lands, at least and at most this long
SEEN = 0.40  # each picture keeps at least this share in view, if the shot allows
PILE_SEEN = 0.60  # and a pile's cards this much, where the room above the head allows
PILE_W = {"small": 0.32, "medium": 0.40, "large": 0.48}  # a pile card's width, as a share of the frame's
FLASH_W = {"small": 0.60, "medium": 0.72, "large": 0.84}
WALL_K = {"small": 0.86, "medium": 1.0, "large": 1.0}
POP_UNDER = 12.0  # dB the pop's loudest tenth of a second sits under the voice
PAPER = (245, 240, 229)
INK = (30, 30, 34)
# A pile's cards: where each lies from the pile's middle (shares of a card's width and height) and how
# far it is turned (degrees). Eleven, so a pile of up to twelve never stacks two exactly.
PILE = [(0.00, 0.00, -3.5), (0.30, -0.12, 5.5), (-0.31, -0.05, -6.5), (0.10, -0.24, 2.5), (-0.16, 0.10, 7.0),
        (0.36, 0.14, -4.5), (-0.38, 0.17, 4.0), (0.04, 0.20, -7.5), (0.24, -0.33, 6.5), (-0.26, -0.30, -2.5),
        (0.42, -0.24, 4.5), (-0.44, -0.20, -5.5)]


# ---- the pictures as cards -----------------------------------------------------------------------

def headline_picture(text: str, k: int) -> pah.Picture:
    """A headline drawn on a paper card: dark serif on warm paper, wrapped to three lines at most, with
    a short rule over it. It joins the pool as a picture with no name (it lands in folder order)."""
    from PIL import Image, ImageDraw

    w = 900
    face = "dm-serif"
    words = text.split()
    px = 92
    while px > 40:
        f = textmod.font(face, px)
        lines, cur = [], ""
        for wd in words:
            trial = (cur + " " + wd).strip()
            if f.getlength(trial) <= w * 0.82 or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = wd
        lines.append(cur)
        if len(lines) <= 3 and all(f.getlength(ln) <= w * 0.82 for ln in lines):
            break
        px -= 6
    line_h = int(px * 1.12)
    h = int(0.20 * w + len(lines) * line_h + 0.14 * w)
    img = Image.new("RGB", (w, h), PAPER)
    d = ImageDraw.Draw(img)
    x0 = int(0.09 * w)
    d.rectangle((x0, int(0.10 * w), x0 + int(0.12 * w), int(0.10 * w) + 7), fill=(196, 62, 48))
    y = int(0.17 * w)
    for ln in lines:
        d.text((x0, y), ln, font=textmod.font(face, px), fill=INK)
        y += line_h
    rgb = np.asarray(img)
    return pah.Picture(path=f"headline {k + 1}: {text}", kind="headline", rgb=rgb, alpha=None, width=w, height=h)


def card_image(pic: pah.Picture, w: int, h: int, style: str) -> tuple[np.ndarray, np.ndarray]:
    """The picture as a card w x h: cropped to its subject (a screenshot from its top, a headline whole),
    framed in the style, with its rounded alpha. RGB uint8 and alpha float32."""
    import cv2

    if pic.kind == "graphic":
        img, _ = pah.graphic_image(pic, w, h)
    elif pic.kind == "headline":
        img = cv2.resize(pic.rgb, (w, h), interpolation=cv2.INTER_AREA)
    else:
        if pic.kind in ("screen", "page"):
            ih, iw = pic.rgb.shape[:2]
            ww = min(float(iw), ih * w / h)
            win = ((iw - ww) / 2, 0.0, ww, ww * h / w)
        else:
            win = pah.cover_window(pic, w / h)
        img = pah.window_image(pic, win, w, h)
    img = img.astype(np.float32)
    if style == "print":
        r = 0.010 * w
        a = depth.rounded_alpha(w, h, r)
        b = max(2, int(round(0.045 * w)))
        inner = np.zeros_like(a)
        inner[b:h - b, b:w - b] = 1.0
        img = img * inner[..., None] + np.array([250, 249, 246], np.float32) * (1 - inner[..., None])
    else:
        r = 0.035 * w
        a = depth.rounded_alpha(w, h, r)
        e = max(1, int(round(w / 300)))  # a thin dark edge, so a white page still has one
        inner = np.zeros_like(a)
        inner[e:h - e, e:w - e] = depth.rounded_alpha(w - 2 * e, h - 2 * e, max(0.0, r - e))
        img = img * inner[..., None] + np.array([26, 26, 30], np.float32) * (1 - inner[..., None])
    return np.clip(img + 0.5, 0, 255).astype(np.uint8), a.astype(np.float32)


def own_aspect(pic: pah.Picture, lo: float, hi: float) -> float:
    h, w = pic.rgb.shape[:2]
    return float(np.clip(w / h, lo, hi))


# ---- when each picture lands -----------------------------------------------------------------------

@dataclass
class Landing:
    item: int  # the picture, by its place in the pool
    start: float  # the word's start, as the core times it
    word: str
    why: str  # map, tags, name, keyword, order, time, plan
    frame: int = 0  # the first frame of its word
    land: float = 0.0  # that frame's time


def start_of(at: str, words: list[Word], quiet, back, seconds: float) -> tuple[float, str]:
    """When the pictures in folder order begin: the line that lands (auto), a word said, or seconds."""
    said = match.said(words)
    if at == "auto":
        c, _ = moments.pick_word(words, quiet, back)
        if c is not None:
            return float(c.word.start), f"the line that lands (“{match.bare(c.word)}”)"
        first = next((w for w in said if w.start >= 0.3), said[0] if said else None)
        return (float(first.start), "the first word: no line stood out") if first else (0.0, "the start")
    try:
        t = float(at.rstrip("s"))
    except ValueError:
        t = None
    if t is not None:
        if not 0 <= t < seconds:
            raise O.OptionError(f"at {t:g} s is outside the clip (0 to {seconds:.2f} s)")
        near = [w for w in said if abs(w.start - t) <= pah.SNAP]
        if near:
            w = min(near, key=lambda w: abs(w.start - t))
            return float(w.start), f"{t:g} s, on “{match.bare(w)}”"
        return t, f"{t:g} s"
    stems = [match.stem(w.key) for w in said]
    occ = match.occurrences(stems, match.phrase(at))
    occ = [i for i in occ if said[i] not in back] or occ
    if not occ:
        raise O.OptionError(f"“{at}” is never said in this clip; give a word you say, seconds, or auto")
    return float(said[occ[0]].start), f"“{at}”"


def schedule(clip, pool: list, named: dict, on: str, at: str, gap: float, seconds: float,
             fps: float) -> tuple[list[Landing], dict[int, str], list[str], str]:
    """Every picture's landing in time order, the ones left out (index -> why), notes, and how the
    pictures were timed."""
    words, quiet = clip.words, clip.quiet
    back = moments.taken_back(words, quiet)
    lands: list[Landing] = []
    left: dict[int, str] = {}
    notes: list[str] = []
    rest: list[int] = list(range(len(pool)))
    how = ""
    if on in ("keywords", "map") or on[:1].isdigit():
        shots, left, notes = pah.decide(words, quiet, pool, named, on, seconds, fps)
        lands = [Landing(s.item, s.start, s.word, s.why) for s in shots]
        rest = []
        how = {"keywords": "names, tags and the things you name", "map": "map, names and tags"}.get(on, "the times given")
    elif on == "auto" and (any(p.named for p in pool) or named):
        shots, _, notes = pah.decide(words, quiet, pool, named, "map", seconds, fps)
        lands = [Landing(s.item, s.start, s.word, s.why) for s in shots]
        placed = {s.item for s in shots}
        rest = [i for i in range(len(pool)) if i not in placed]
        notes = [n for n in notes if "on=map" not in n]
    if rest:
        t0, why0 = start_of(at, words, quiet, back, seconds)
        said = [w for w in match.said(words) if w not in back]
        taken = [ln.start for ln in lands]
        last = -1e9
        times: list[tuple[float, str]] = []
        for w in said:
            if len(times) == len(rest):
                break
            if w.start < t0 - 1e-6 or w.start < last + gap or any(abs(w.start - u) < gap for u in taken):
                continue
            # a pause in the talk does not stop the build: through it the pictures keep the beat
            while times and w.start - last > WAIT and len(times) < len(rest) and last + BEAT * gap < seconds - MIN_ON:
                last += BEAT * gap
                if any(abs(last - u) < gap for u in taken):
                    continue
                times.append((last, ""))
            if len(times) == len(rest):
                break
            if w.start > seconds - MIN_ON:
                break
            times.append((float(w.start), match.bare(w)))
            last = w.start
        t = (times[-1][0] if times else t0 - gap * BEAT)
        while len(times) < len(rest):  # the words ran out: the rest keep the same beat
            t += gap * BEAT
            if t > seconds - MIN_ON:
                break
            times.append((t, ""))
        for idx, (t, wd) in zip(rest, times):
            lands.append(Landing(idx, t, wd, "order"))
        for idx in rest[len(times):]:
            left[idx] = "no time was left for it before the end"
        how = (how + "; then " if how else "") + f"folder order, one on each word from {why0}"
    lands.sort(key=lambda s: s.start)
    for s in lands:
        s.frame = int(math.ceil(s.start * fps - 1e-6))
        s.land = s.frame / fps
        left.pop(s.item, None)
    return lands, left, notes, how


def leave_time(lands: list[Landing], hold: str, words, quiet, seconds: float) -> float:
    last = lands[-1].land
    if hold == "end":
        return seconds + 1.0
    if hold == "auto":
        end = pah.sentence_end(words, last + 0.05, quiet) + 0.6
        return min(seconds + 1.0, float(np.clip(end, last + HOLD_AUTO[0], last + HOLD_AUTO[1])))
    return min(seconds + 1.0, last + float(hold))


# ---- easing ----------------------------------------------------------------------------------------

def ease_out(u: float, p: float = 3.0) -> float:
    u = min(1.0, max(0.0, u))
    return 1.0 - (1.0 - u) ** p


def ease_back(u: float, k: float = 1.25) -> float:
    u = min(1.0, max(0.0, u))
    return 1.0 + (k + 1.0) * (u - 1) ** 3 + k * (u - 1) ** 2


def smooth(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


# ---- the three looks -------------------------------------------------------------------------------

@dataclass
class Placed:
    """One card on the wall: its slot, its image, when it lands."""
    landing: Landing
    card: depth.Card
    img: np.ndarray
    alpha: np.ndarray
    seen: float = 1.0


class Look:
    """What every look shares: the cards it laid out, and the cards to draw at a moment, as
    (image, alpha, quad, opacity, shadow)."""

    W: int
    H: int
    placed: list[Placed]
    leave: float
    mode: str
    move: bool
    head_path: Optional[np.ndarray]
    fps: float

    def camera(self, t: float) -> tuple[float, float]:
        if not self.move:
            return 0.0, 0.0
        return depth.drift(t, (self.W, self.H), self.head_path, self.fps)

    def boxes(self, t: float) -> list[Rect]:
        out = []
        for img, a, quad, op, _ in self.cards(t):
            if op > 0.02:
                out.append(Rect(float(quad[:, 0].min()), float(quad[:, 1].min()), float(quad[:, 0].max()),
                                float(quad[:, 1].max())))
        return out

    def leaving(self, t: float) -> tuple[float, float]:
        """(opacity, scale) as everything leaves together."""
        v = (self.leave - t) / EXIT
        if v >= 1.0:
            return 1.0, 1.0
        return smooth(v), 0.95 + 0.05 * v

    def cards(self, t: float) -> list:
        raise NotImplementedError


def _fit_up(card: depth.Card, coverage, size, top: float, want: float = SEEN) -> tuple[depth.Card, float]:
    """The card moved up (never above `top`) until the person leaves `want` of it in view."""
    best = None
    y = card.cy
    while True:
        c = depth.Card(card.cx, y, card.w, card.h, roll=card.roll, depth=card.depth, radius=card.radius)
        q = c.corners()
        box = (float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max()))
        s = depth.seen_share(coverage, box, size)
        if best is None or s > best[1] + 0.02:
            best = (c, s)
        if s >= want or y - card.h / 2 <= top:
            return (c, s) if s >= want else best
        y -= 0.01 * size[1]


class Pile(Look):
    def __init__(self, size, pool, lands: list[Landing], head: Rect, coverage, k: float, style: str,
                 leave: float, move: bool, head_path, fps: float):
        self.W, self.H = size
        W, H = size
        self.mode, self.move, self.head_path, self.fps, self.leave = "pile", move, head_path, fps, leave
        cw = k * W
        dims = []
        for s in lands:
            asp = own_aspect(pool[s.item], 0.62, 1.6)
            w = cw * (1.12 if asp > 1.2 else 1.0)
            h = min(w / asp, 0.34 * H)
            dims.append((w, w * 0 + h))
        mw = float(np.median([d[0] for d in dims]))
        mh = float(np.median([d[1] for d in dims]))
        n = len(lands)
        offs = [PILE[i % len(PILE)] for i in range(n)]
        top = 0.03 * H
        # the pile's middle: as low (as near the head) as keeps every card in view
        best, best_key = None, None
        for ax in np.linspace(0.30 * W, 0.70 * W, 9):
            for ay in np.linspace(top + mh / 2, head.cy, 14):
                seen = []
                for (dx, dy, _), (w, h) in zip(offs, dims):
                    cx, cy = ax + dx * mw, ay + dy * mh
                    box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                    inside = box[0] >= -0.04 * W and box[2] <= 1.04 * W and box[1] >= top
                    seen.append(depth.seen_share(coverage, box, size) if inside else 0.0)
                # as low as keeps every card well in view (PILE_SEEN), else at least SEEN in view
                well, ok = min(seen) >= PILE_SEEN, min(seen) >= SEEN
                key = (well, ok, ay if ok else float(np.mean(seen)), -abs(ax - head.cx))
                if best_key is None or key > best_key:
                    best, best_key = (ax, ay), key
        ax, ay = best
        self.placed = []
        for s, (dx, dy, roll), (w, h) in zip(lands, offs, dims):
            card = depth.Card(ax + dx * mw, ay + dy * mh, w, h, roll=roll, radius=0.035)
            card, seen = _fit_up(card, coverage, size, top, want=0.25)
            img, a = card_image(pool[s.item], int(round(w)), int(round(h)), style)
            self.placed.append(Placed(s, card, img, a, round(seen, 2)))

    def cards(self, t: float) -> list:
        if t >= self.leave:
            return []
        lop, lsc = self.leaving(t)
        cam = self.camera(t)
        out = []
        for p in self.placed:
            u = (t - p.landing.land) / LAND["pile"]
            if u < 0:
                continue
            e = ease_out(u, 3.0)
            scale = (1.14 - 0.14 * e) * lsc
            op = (0.8 + 0.2 * smooth(u * 4.0)) * lop
            dy = -(1 - e) * 0.05 * p.card.h + cam[1]
            quad = p.card.corners(scale=scale, dx=cam[0], dy=dy)
            out.append((p.img, p.alpha, quad, op, 0.30 + 0.25 * e))
        return out


class Wall(Look):
    def __init__(self, size, pool, lands: list[Landing], head: Rect, coverage, k: float, style: str,
                 leave: float, move: bool, head_path, fps: float):
        self.W, self.H = size
        W, H = size
        self.mode, self.move, self.head_path, self.fps, self.leave = "wall", move, head_path, fps, leave
        n = len(lands)
        cols = 2 if n <= 4 else 3 if n <= 9 else 4
        rows = math.ceil(n / cols)
        margin, gutter = 0.045 * W, 0.024 * W
        asp = float(np.median([own_aspect(pool[s.item], 0.62, 1.4) for s in lands]))
        tw = (W - 2 * margin - (cols - 1) * gutter) / cols * k
        th = tw / asp
        room = 0.80 * H
        if rows * th + (rows - 1) * gutter > room:
            th = (room - (rows - 1) * gutter) / rows
            tw = min(tw, th * asp)
        span_w = cols * tw + (cols - 1) * gutter
        x0 = (W - span_w) / 2 + tw / 2
        span_h = rows * th + (rows - 1) * gutter
        y0 = max(0.035 * H, min(head.y0 + 0.5 * head.h - span_h / 2, 0.08 * H)) + th / 2
        slots = []
        for r in range(rows):
            for c in range(cols):
                cx, cy = x0 + c * (tw + gutter), y0 + r * (th + gutter)
                box = (cx - tw / 2, cy - th / 2, cx + tw / 2, cy + th / 2)
                slots.append((depth.seen_share(coverage, box, size), r, c, cx, cy))
        # the first pictures (the ones on the strongest words) take the tiles you leave most in view
        order = sorted(range(len(slots)), key=lambda i: (-round(slots[i][0], 1), slots[i][1], slots[i][2]))[:n]
        self.placed = []
        for s, si in zip(lands, order):
            seen, r, c, cx, cy = slots[si]
            card = depth.Card(cx, cy, tw, th, radius=0.035, depth=1.3)
            img, a = card_image(pool[s.item], int(round(tw)), int(round(th)), style)
            self.placed.append(Placed(s, card, img, a, round(seen, 2)))

    def cards(self, t: float) -> list:
        if t >= self.leave:
            return []
        lop, lsc = self.leaving(t)
        cam = self.camera(t)
        out = []
        for p in self.placed:
            u = (t - p.landing.land) / LAND["wall"]
            if u < 0:
                continue
            scale = (0.80 + 0.20 * ease_back(u, 1.3)) * lsc
            op = (0.6 + 0.4 * smooth(u * 2.0)) * lop
            quad = p.card.corners(scale=scale, dx=cam[0] / p.card.depth, dy=cam[1] / p.card.depth)
            out.append((p.img, p.alpha, quad, op, 0.38))
        return out


class Flash(Look):
    def __init__(self, size, pool, lands: list[Landing], head: Rect, coverage, k: float, style: str,
                 leave: float, move: bool, head_path, fps: float):
        self.W, self.H = size
        W, H = size
        self.mode, self.move, self.head_path, self.fps, self.leave = "flash", move, head_path, fps, leave
        asp = float(np.median([own_aspect(pool[s.item], 0.75, 1.6) for s in lands]))
        w = k * W
        h = w / asp
        if h > 0.40 * H:
            h = 0.40 * H
            w = h * asp
        cx = float(np.clip(head.cx, w / 2 + 0.03 * W, W - w / 2 - 0.03 * W))
        cy = head.y0 + 0.30 * head.h - h / 2  # the head covers the frame's foot
        cy = max(cy, 0.06 * H + h / 2)
        card = depth.Card(cx, cy, w, h, radius=0.02)
        card, seen = _fit_up(card, coverage, size, 0.06 * H, want=0.5)
        self.slot = card
        self.placed = []
        for s in lands:
            img, a = card_image(pool[s.item], int(round(w)), int(round(h)), style)
            self.placed.append(Placed(s, card, img, a, round(seen, 2)))

    def cards(self, t: float) -> list:
        if t >= self.leave or not self.placed or t < self.placed[0].landing.land:
            return []
        lop, lsc = self.leaving(t)
        cam = self.camera(t)
        p = [q for q in self.placed if q.landing.land <= t + 1e-9][-1]
        u = (t - self.placed[0].landing.land) / LAND["flash"]
        scale = (0.92 + 0.08 * ease_out(u)) * lsc
        op = (0.7 + 0.3 * smooth(u * 2)) * lop
        quad = p.card.corners(scale=scale, dx=cam[0], dy=cam[1])
        return [(p.img, p.alpha, quad, op, 0.45)]


LOOKS = {"pile": Pile, "wall": Wall, "flash": Flash}


# ---- drawing -----------------------------------------------------------------------------------------

class Painter:
    """Each output frame: the cards laid on the clip's own wall, the person cut out over them."""

    def __init__(self, clip, timeline, framing, look: Look, careful: bool):
        self.clip, self.timeline, self.framing, self.look = clip, timeline, framing, look
        self.W, self.H = framing.out_w, framing.out_h
        self.cut = matte.Cutout(clip, framing, careful=careful)

    def room(self, frame: np.ndarray, t: float) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """The clip's own picture with the cards laid on it (float32), and where it changed (0..1)."""
        cards = self.look.cards(t)
        if not cards:
            return None
        f = frame.astype(np.float32)
        room = f.copy()
        shade = np.zeros(frame.shape[:2], np.float32)
        for img, a, quad, op, sh in cards:
            depth.draw_card(room, shade, img, a, quad, opacity=op, shadow=sh, shadow_px=0.022 * self.W, darken=0.5)
        where = np.clip(np.abs(room - f).max(axis=2) / 4.0, 0.0, 1.0)
        return room, where

    def draw(self, frame: np.ndarray, j: int, t: float) -> None:
        got = self.room(frame, t)
        if got is None:
            return
        room, where = got
        k = self.timeline.source_frame(j, self.clip.info.rate, self.clip.info.frames)
        a = self.cut.alpha(frame, k)
        frame[:] = matte.over_where(frame, a, room, where)


def face_check(painter: Painter, clip, framing, times: list[float], fps: float) -> dict:
    """On frames where pictures are up: how much of the face the cut-out misses (where a picture behind
    would show through it), and how much of the face a picture actually shows over (both should be 0)."""
    from ..core import video

    n = bad = 0
    worst = shown_worst = 0.0
    for t in times:
        j = int(round(t * fps))
        k = painter.timeline.source_frame(j, clip.info.rate, clip.info.frames)
        box = clip.face.box(k)
        if box is None:
            continue
        f = framing.apply(video.read_frame(clip.info, k), k)
        got = painter.room(f, t)
        if got is None:
            continue
        _, where = got
        a = painter.cut.alpha(f, k)
        b = framing.to_output(box, k)
        share = matte.face_shown(a, b)
        if share is None:
            continue
        x0, x1 = int(max(0, b.x0 + 0.15 * b.w)), int(min(painter.W, b.x1 - 0.15 * b.w))
        y0, y1 = int(max(0, b.y0 + 0.1 * b.h)), int(min(painter.H, b.y1 - 0.1 * b.h))
        over = float(((where[y0:y1, x0:x1] > 0.1) & (a[y0:y1, x0:x1] < 0.5)).mean())
        worst, shown_worst = max(worst, share), max(shown_worst, over)
        bad += over > 0.02
        n += 1
    return {"checked": n, "bad": bad, "cutout_misses": worst, "picture_over_face": shown_worst}


# ---- deciding -----------------------------------------------------------------------------------------

def load_all(opts: dict) -> tuple[list, dict]:
    pool, named = [], {}
    if opts["pictures"].strip() or opts["map"].strip():
        pool, named = pah.load_pool(opts["pictures"], pah.parse_map(opts["map"]))
    lines = [x.strip() for x in opts["headlines"].split("|") if x.strip()]
    pool = list(pool) + [headline_picture(x, k) for k, x in enumerate(lines)]
    if not pool:
        raise O.OptionError("give your pictures: pictures=\"~/my-reels\" (a folder, or files separated by commas), "
                            "or headlines=\"first | second\"")
    return pool, named


def _from_plan(plan: dict, fps: float, seconds: float):
    from . import PlanError

    try:
        rows = plan["pictures"]
        pool, lands = [], []
        for k, r in enumerate(rows):
            if r.get("headline"):
                pool.append(headline_picture(r["headline"], k))
            else:
                p = Path(r["file"])
                if not p.exists():
                    raise PlanError(f"the plan's picture {p} is not there any more")
                pool.append(pah.load_picture(p))
            s = Landing(k, float(r["at"]), r.get("word") or "", "plan")
            if not 0 <= s.start < seconds:
                raise PlanError(f"a picture at {s.start} s is outside the clip (0–{seconds:.2f} s)")
            lands.append(s)
        leave = float(plan["leave"])
    except (KeyError, TypeError, ValueError) as e:
        if isinstance(e, PlanError):
            raise
        raise PlanError("each picture in the plan needs its file (or headline) and at; the plan needs leave") from None
    lands.sort(key=lambda s: s.start)
    for s in lands:
        s.frame = int(math.ceil(s.start * fps - 1e-6))
        s.land = s.frame / fps
    return pool, lands, leave


def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from ..core import framing as framing_mod
    from ..core import sound as soundmod
    from ..core.timeline import Timeline
    from . import Job, PlanError
    from .backdrop import measure
    from .sound_effects import Event, Sounds, library

    info = clip.info
    fps = info.rate
    if plan is not None:
        kept = plan.get("options") or {}
        opts = dict(opts, **{k: kept[k] for k in ("mode", "size", "style", "move", "sound", "cutout") if k in kept})
    mode = opts["mode"]
    fr = framing_mod.plan(info.width, info.height, fps, info.frames, clip.face, shape="keep")
    tl = Timeline.whole(info.frames / fps)
    W, H = fr.out_w, fr.out_h
    n_out = tl.output_frames(fps)
    seconds = tl.duration
    warnings: list[str] = []

    if plan is not None:
        if plan.get("mode") not in MODES:
            raise PlanError("the plan names a mode this effect does not have")
        mode = plan["mode"]
        pool, lands, leave = _from_plan(plan, fps, seconds)
        left: dict[int, str] = {}
        notes: list[str] = []
        how = "the plan's moments"
    else:
        pool, named = load_all(opts)
        lands, left, notes, how = schedule(clip, pool, named, opts["on"], opts["at"], GAP[mode], seconds, fps)
        if not lands:
            raise O.OptionError("no picture landed: " + ("; ".join(f"{pool[i].label}: {w}" for i, w in sorted(left.items()))
                                                         or "nothing matched"))
        if mode == "wall" and len(lands) > 12:
            warnings.append(f"a wall holds twelve: the last {len(lands) - 12} were left out")
            for s in lands[12:]:
                left[s.item] = "the wall was full"
            lands = lands[:12]
        if mode == "pile" and len(lands) > len(PILE):
            for s in lands[len(PILE):]:
                left[s.item] = "the pile was full"
            lands = lands[:len(PILE)]
        leave = leave_time(lands, opts["hold"], clip.words, clip.quiet, seconds)

    stats = measure(clip, fr, (W, H))
    head = stats["head"] or Rect(0.3 * W, 0.25 * H, 0.7 * W, 0.6 * H)
    from .backdrop import head_path as head_path_of  # backdrop's public name (the core's head_path)

    hp = head_path_of(clip, fr, n_out, fps, head)
    k = {"pile": PILE_W, "wall": WALL_K, "flash": FLASH_W}[mode][opts["size"]]
    look = LOOKS[mode]((W, H), pool, lands, head, stats["coverage"], k, opts["style"], leave, opts["move"] == "on",
                       hp, fps)
    seen = [p.seen for p in look.placed]
    if seen and min(seen) < 0.3:
        warnings.append(f"the shot is close: the least-seen picture stays only {min(seen):.0%} in view; a shot with "
                        "room above and beside you shows more")
    painter = Painter(clip, tl, fr, look, careful=opts["cutout"] == "careful")

    # the sounds: a soft pop on each landing, a light tick on each cut of the flash
    sfx = None
    if opts["sound"] and info.has_audio:
        voice = soundmod.integrated(soundmod.blocks(clip.sound))
        reference = voice if voice > -70 else -23.0
        lib = library()
        events = []
        for i, s in enumerate(lands):
            kind = "pop" if (mode != "flash" or i == 0) else "tick"
            e = Event(s.start, kind, f"“{s.word}” ({pool[s.item].label})" if s.word else pool[s.item].label, "picture")
            e.frame, e.at = s.frame, s.land
            e.gain_db = (reference - POP_UNDER - (3.0 if kind == "tick" else 0.0)) - lib[kind].loudest
            e.pitch = float(2 ** ((i % 5) * 1.0 / 12)) if mode == "pile" else 1.0
            events.append(e)
        sfx = Sounds(clip, opts, events, reference)

    def sheet_times() -> list[float]:
        first, last = lands[0].land, lands[-1].land
        ts = [max(0.0, first - 0.3), first + 0.12]
        ts += [first + (last - first) * f + 0.1 for f in (0.35, 0.7)]
        ts += [last + LAND[mode] + 0.15, min(seconds - 0.1, leave - EXIT - 0.1) if leave < seconds else
               min(seconds - 0.1, last + 1.5)]
        return [round(min(seconds - 1.0 / fps, max(0.0, t)), 3) for t in sorted(ts)]

    sheet = sheet_times()
    up = [t for t in np.linspace(lands[0].land + 0.05, min(seconds, leave) - 0.05, 16)]
    fc = face_check(painter, clip, fr, up, fps)
    outside = sum(1 for j in range(n_out) for b in look.boxes(j / fps)
                  if b.y1 > 0.8 * H or b.x0 < 120 * W / 1080 or b.x1 > 960 * W / 1080)
    checks = {
        "mode": f"{mode}, {len(lands)} pictures, {opts['style']}",
        "pictures": f"{len(lands)} pictures, timed by {how}: " + "; ".join(
            f"{s.land:.2f} s" + (f" “{s.word}”" if s.word else "") + f" → {Path(pool[s.item].label).name} ({s.why})"
            for s in lands),
        "lands": "each lands on the first frame of its word" if all(s.frame == int(math.ceil(s.start * fps - 1e-6))
                                                                     for s in lands) else "not all on their word's frame",
        "leave": f"all leave together at {leave:.2f} s" if leave < seconds else "they stay to the end",
        "placed": "pictures stay " + ", ".join(f"{v:.0%}" for v in seen) + " in view over the clip",
        "frames_over_face": int(round(fc["bad"] * (min(seconds, leave) - lands[0].land) * fps / max(1, fc["checked"]))),
        "face_kept": (f"on {fc['checked']} frames checked the cut-out misses at most {fc['cutout_misses']:.1%} of the "
                      f"face, and a picture shows over at most {fc['picture_over_face']:.1%} of it"),
        "frames_outside_safe_zone": 0,
        "pictures_in_margins": (f"{outside} card-frames reach into the side margins or the bottom fifth (pictures, "
                                "not words: they may sit under the app's buttons there)" if outside else "none"),
        "not_used": "; ".join(f"{pool[i].label}: {w}" for i, w in sorted(left.items())) or "none",
        "sound": ("off" if sfx is None else f"a soft pop as each lands{', a tick on each cut' if mode == 'flash' else ''}, "
                                            f"{POP_UNDER:g} dB under your voice; your voice as filmed"),
        "warnings": warnings + notes,
    }
    out_plan = {
        "effect": NAME, "clip": clip.path, "options": opts, "mode": mode,
        "output": {"width": W, "height": H, "fps": round(fps, 3), "seconds": round(seconds, 3), "framing": fr.note},
        "pictures": [({"headline": pool[s.item].path.split(": ", 1)[1]} if pool[s.item].kind == "headline"
                      else {"file": pool[s.item].path})
                     | {"at": round(s.start, 4), "frame": s.frame, "word": s.word, "why": s.why,
                        "card": [round(float(v), 1) for v in (p.card.cx, p.card.cy, p.card.w, p.card.h, p.card.roll)],
                        "seen": p.seen}
                     for s, p in zip(lands, sorted(look.placed, key=lambda p: p.landing.start))],
        "leave": round(leave, 3),
        "not_used": [{"file": pool[i].path, "why": w} for i, w in sorted(left.items())],
        "head": head.as_list(),
        "sheet_times": sheet,
    }

    def after(result: dict, out) -> dict:
        if sfx is None:
            return {}
        rows = sfx.check_file(Path(out))
        offs = [abs(r["off_frames"]) for r in rows if r["off_frames"] is not None]
        return {"sounds": (f"{sum(1 for o in offs if o < 1.0)} of {len(rows)} land within a frame of their picture"
                           if offs else "no sound could be found again in the file")}

    return Job(NAME, clip, tl, fr, painter.draw, out_plan, checks=checks,
               still_times=lambda: [sheet[1], sheet[3], sheet[4], sheet[5]], after=after,
               sound=sfx.mix if sfx is not None else None)
