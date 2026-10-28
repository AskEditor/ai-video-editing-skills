"""freeze-point: the picture stops on a moment, and a hand-drawn arrow and a label point at you.

How it decides, in order:

1. **The moment.** By default, just after the word that makes the point of your opening line (the
   word `words-behind` would choose), or after words you name, or at a time. The picture stops in
   the quiet right after the word, never inside it, so no word is cut in half; within that quiet it
   stops on the sharpest frame of the face, so the frame held is not a blur.
2. **The sound.** The usual way: your voice stops with the picture, a record scratch (or a hit)
   lands on the freeze, the room's own sound holds under it (looped from the clip's quietest stretch,
   so it never plays as a dropout), and the clip carries on from exactly where it stopped.
   Nothing said is lost and nothing drifts, so the film is longer by the hold. `sound=voice` keeps
   the film's length instead: your voice carries on under the still picture, which then jumps
   ahead to meet it, and your lips are still while your words go on.
3. **The label** goes in the largest free space beside your head (measured on the person mask of
   the frame that is held), inside the part of the screen the apps leave clear, and never on you.
4. **The arrow** draws itself in one hand-made stroke, from the label to just outside your head,
   aimed at your face: the shaft bows a little as a hand does, the head is flicked on at the end
   of the same stroke, and no part of it touches your face. `mark=circle` loops round your head
   instead; `mark=panel` turns the held picture into a comic panel, inked, with a caption box.
5. **Everything is checked** before rendering: the frames where any mark would be on your face
   (0), the label inside the safe zone, the frame held and why.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core import moments, synth
from ..core import sound as soundmod
from ..core import text as textmod
from ..core.compose import Sprite, over
from ..core.layout import Rect, SafeZone
from ..core.words import HESITATIONS, Word, normalize
from . import captions as cap
from . import options as O
from . import words_behind as wb

NAME = "freeze-point"
MARKS = ("arrow", "circle", "panel")
SOUNDS = ("scratch", "hit", "none", "voice")
FACE = "caveat-brush"  # handwriting in marker
DEFAULT_LABEL = "that's me"


# ---- the options -------------------------------------------------------------------------------------

@dataclass(frozen=True)
class _Moment:
    help: str

    def parse(self, raw) -> str:
        v = str(raw).strip().strip("\"“”")
        low = v.lower()
        if low in ("", "auto"):
            return "auto"
        if low in ("start", "begin", "beginning"):
            return "start"
        try:
            t = float(low.rstrip("s"))
        except ValueError:
            if not [normalize(x) for x in v.split() if normalize(x)]:
                raise O.OptionError(f"give auto, start, a time such as 2.4, or words said in the clip (got {raw!r})") from None
            return v
        if t < 0:
            raise O.OptionError(f"give a time from 0 seconds on (got {raw!r})")
        return f"{t:g}"


@dataclass(frozen=True)
class _Label:
    help: str

    def parse(self, raw) -> str:
        v = str(raw).strip().strip("\"“”")
        if v.lower() in ("", "auto"):
            return "auto"
        if v.lower() in ("none", "off"):
            return "none"
        if len(v) > 40 or len(v.split()) > 7:
            raise O.OptionError(f"give a short label, seven words and 40 letters at most: it has to be read in a "
                                f"second or two (got {raw!r})")
        return v


OPTIONS = {
    "at": _Moment("auto: just after the word that makes the point of your opening line · start · a time such as 2.4 · "
                  "or words said in the clip, such as \"this is me\": the picture stops as they end"),
    "hold": O.Number(0.5, 6, "how many seconds the picture holds still, 1.6 by default"),
    "label": _Label("the words that point at you, up to seven (auto: that's me) · none for no label"),
    "mark": O.Choice(MARKS, "arrow: a hand-drawn arrow from the label to you · circle: a loop drawn round your head · "
                            "panel: the held picture becomes a comic panel with the label in a caption box"),
    "color": O.Colour("auto: the marker colour that stands out most where it is drawn · a name such as yellow, white "
                      "or red · or a hex colour such as #ffd400"),
    "sound": O.Choice(SOUNDS, "scratch: your voice stops with the picture, a record scratch lands on it, and the "
                              "film is longer by the hold (the room's own sound holds under it) · hit: the same with a hit · "
                              "none: the same with neither · "
                              "voice: your voice carries on under the still picture and the film keeps its length"),
    "captions": O.Choice(("off",) + cap.LOOKS, "off · or the pack's captions, in a look: stacked, one-word, "
                                                "serif-chip or fitted; they wait while the picture holds"),
    "frame": O.Choice(("keep", "9:16"), "keep the clip's own shape, or make it 9:16 following the face"),
    "fix": O.Text("misheard words as wrong=right, separated by ;  e.g. \"Jon=John\""),
}

# The runs `tests/matrix.py freeze-point` makes: the four clips the master list names, then the
# other marks and sounds on the clips they suit, and a label of the speaker's own words.
MATRIX = [
    ("A", "A", {}),
    ("D", "D", {}),
    ("B", "B", {}),
    ("C", "C", {}),
    ("B-circle", "B", {"mark": "circle", "label": "the editor", "sound": "hit"}),
    ("A-panel", "A", {"mark": "panel", "label": "me, three hooks in"}),
    ("D-voice", "D", {"sound": "voice", "label": "jet-lagged", "at": "my hotel", "captions": "stacked"}),
]

FLASH = 0.12  # seconds the white of the freeze takes to fade
OUT = 0.18  # seconds the marks take to fade before the picture moves again


# ---- the moment ------------------------------------------------------------------------------------

@dataclass
class Moment:
    t: float  # source seconds: the picture stops here (a frame's own start)
    k: int  # the source frame held
    why: str
    word: Optional[str] = None
    said: Optional[tuple[float, float]] = None
    sharp: str = ""


def _words(clip) -> list[Word]:
    return [w for w in clip.words if w.key not in HESITATIONS]


def _sharpness(img: np.ndarray, box: Optional[Rect]) -> float:
    import cv2

    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if box is not None:
        h, w = g.shape
        x0, y0 = int(max(0, box.x0)), int(max(0, box.y0))
        x1, y1 = int(min(w, box.x1)), int(min(h, box.y1))
        if x1 - x0 > 8 and y1 - y0 > 8:
            g = g[y0:y1, x0:x1]
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def decide_moment(clip, opts: dict, duration: float) -> Moment:
    """Where the picture stops: in the quiet straight after the word (or as near to the time given as
    the voice allows), on the sharpest frame of the face in that quiet."""
    from ..core import video

    fps = clip.info.rate
    at = opts["at"]
    words = _words(clip)
    env = clip.envelope
    after: Optional[float] = None
    nxt: Optional[float] = None
    word = None
    said = None
    if at == "start":
        return Moment(0.0, 0, "on the first frame")
    if at == "auto" or not wb._is_time(at):
        if at == "auto":
            # the core's picker, as words-behind uses it: never a word the speaker took back
            back = moments.taken_back(clip.words, clip.quiet)
            cand, _ = moments.pick_word(words, clip.quiet, back) if words else (None, [])
            if cand is None:
                raise O.OptionError("no spoken word stands out to freeze on (or nothing clear is said); give a moment: "
                                    "at=2.4, or words you say, such as at=\"this is me\"")
            i = cand.index
            why = f"just after “{moments.bare(cand.word)}”, the word that makes the point of " \
                  f"“{' '.join(words[n].text for n in cand.line)}”"
        else:
            span = wb.find_said(words, at)
            if span is None:
                raise O.OptionError(f"could not find \"{at}\" in what is said; `python -m fx.run words <video>` lists "
                                    "the words, or give a time such as at=2.4")
            i = span[1]
            why = f"as “{' '.join(w.text for w in words[span[0]:span[1] + 1])}” ends"
        w = words[i]
        word, said = moments.bare(w), (w.start, w.end)
        after = w.end
        nxt = words[i + 1].start if i + 1 < len(words) else duration
    else:
        t0 = min(float(at), duration - 0.1)
        why = "at the time given"
        # a time inside a word moves to the end of that word, so the word is never cut
        inside = [w for w in words if w.start + 0.03 < t0 < w.end - 0.03]
        if inside:
            w = inside[0]
            after = w.end
            k = words.index(w)
            nxt = words[k + 1].start if k + 1 < len(words) else duration
            why = f"at the end of “{moments.bare(w)}”, the word the time given falls in"
        else:
            after, nxt = t0, t0 + 0.001
    # the window it may stop in: the quiet after the word, up to a third of a second of it
    quiet = [(a, b) for a, b in clip.quiet if b > after - 0.02 and a < after + 0.3]
    if nxt is not None and nxt - after > 0.06 and quiet:
        q0, q1 = quiet[0]
        lo, hi = max(after, q0) + 0.02, min(q1 - 0.03, nxt - 0.04, max(after, q0) + 0.32)
    else:
        lo = hi = env.quietest(max(0.0, after - 0.03), min(duration, (nxt if nxt is not None else after) + 0.02)) \
            if nxt is not None and nxt > after else after
    lo, hi = max(0.0, min(lo, duration - 0.05)), max(0.0, min(max(hi, lo), duration - 0.05))
    k_lo, k_hi = int(math.ceil(lo * fps - 1e-6)), int(math.floor(hi * fps + 1e-6))
    ks = list(range(k_lo, max(k_lo, k_hi) + 1))[:12]
    best, best_s = ks[0], -1.0
    for k in ks:
        b = clip.face.box(k)
        s = _sharpness(video.read_frame(clip.info, k), b)
        if s > best_s:
            best, best_s = k, s
    sharp = f"the sharpest of {len(ks)} frames in the quiet" if len(ks) > 1 else "the frame at that moment"
    return Moment(best / fps, best, why, word, said, sharp)


# ---- where the label goes ----------------------------------------------------------------------------

def _person_at(clip, fr, k: int, size: tuple[int, int]) -> np.ndarray:
    m = clip.person()
    crop = fr.window(k) if fr.reframed else None
    return m.full(m.at(k), size, crop=crop)


@dataclass
class Label:
    lines: list[str]
    px: int
    rect: Rect  # where the label's ink box sits (output pixels)
    where: str


def label_lines(text: str) -> list[str]:
    words = text.split()
    if len(text) <= 14 or len(words) == 1:
        return [text]
    best, cost = [text], math.inf
    for k in range(1, len(words)):
        a, b = " ".join(words[:k]), " ".join(words[k:])
        c = max(len(a), len(b)) + (3 if len(words[k:]) == 1 and len(b) <= 3 else 0)
        if c < cost:
            best, cost = [a, b], c
    return best


def label_size(lines: list[str], px: int) -> tuple[float, float, float]:
    """The label's width, height and line step at `px`, measured on its ink."""
    f = textmod.font(FACE, px)
    step = 1.12 * textmod.cap_height(FACE, px) + 0.42 * px
    widths, tops, bottoms = [], [], []
    for n, ln in enumerate(lines):
        l, t, r, b = f.getbbox(ln, anchor="ls")
        widths.append(r - l)
        tops.append(n * step + t)
        bottoms.append(n * step + b)
    return max(widths), max(bottoms) - min(tops), step


def face_zone(face: Rect) -> Rect:
    """What counts as the face for an arrow's tip: the detector's box (brows to chin) with the forehead
    above it and a little either side. Hair, ears and shoulders are fair to point at."""
    return face.grow(0.08 * face.w, 0.38 * face.h, 0.08 * face.w, 0.06 * face.h)


def arrow_ends(label: Rect, face: Rect, size: tuple[int, int]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Where the arrow starts (just off the label, towards the face) and where its tip lands (just
    outside the face, on the line from the face towards the label)."""
    W, _ = size
    zone = face_zone(face)
    dx, dy = label.cx - zone.cx, label.cy - zone.cy
    L = max(1e-6, math.hypot(dx, dy))
    ux, uy = dx / L, dy / L
    gap = 0.025 * W
    t = _exit(zone.cx, zone.cy, ux, uy, zone) + gap
    tip = (zone.cx + ux * t, zone.cy + uy * t)
    vx, vy = tip[0] - label.cx, tip[1] - label.cy
    Lv = max(1e-6, math.hypot(vx, vy))
    vx, vy = vx / Lv, vy / Lv
    s0 = _exit(label.cx, label.cy, vx, vy, label) + 0.018 * W
    return (label.cx + vx * s0, label.cy + vy * s0), tip


def _crosses(points: np.ndarray, rects: list[Rect], margin: float = 0.0) -> bool:
    """Whether any of `points` (n, 2) falls inside any of `rects`, grown by `margin`."""
    for r in rects:
        g = r.grow(margin, margin, margin, margin)
        if (((points[:, 0] > g.x0) & (points[:, 0] < g.x1) & (points[:, 1] > g.y0) & (points[:, 1] < g.y1))).any():
            return True
    return False


def place_label(lines: list[str], person: np.ndarray, head: Rect, face: Rect, size: tuple[int, int],
                avoid: list[Rect], ring: Optional[Rect] = None, arrow: bool = True) -> Optional[Label]:
    """The label in the free space beside the head: inside the safe zone, on nobody, clear of the
    face, the head and anything in `avoid`; as large as reads well, and placed for the mark it goes
    with: for an arrow, where the arrow from it to the face is a good length (about a quarter of the
    frame) and crosses as little of the person as it can; for a loop, beside the loop. None when
    there is no such place at any size."""
    import cv2

    W, H = size
    zone = SafeZone(W, H).rect
    taken = cv2.dilate((person > 0.4).astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (int(0.03 * W) | 1, int(0.03 * W) | 1)))
    integral = cv2.integral(taken)
    fz = face_zone(face)
    obstacles = [fz.grow(0.02 * W, 0.02 * W, 0.02 * W, 0.02 * W), head] + list(avoid)
    if ring is not None:
        obstacles.append(ring)
    want = 0.25 * W
    for px in range(int(0.075 * H), int(0.032 * H), -int(max(2, 0.004 * H))):
        w, h, _ = label_size(lines, px)
        if w > 0.9 * zone.w:
            continue
        best = None
        for y in np.arange(zone.y0, zone.y1 - h, max(4.0, 0.012 * H)):
            for x in np.arange(zone.x0, zone.x1 - w, max(4.0, 0.012 * W)):
                r = Rect(float(x), float(y), float(x + w), float(y + h))
                x0, y0, x1, y1 = int(r.x0), int(r.y0), int(r.x1), int(r.y1)
                covered = (integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]) / max(1, (x1 - x0) * (y1 - y0))
                if covered > 0.02 or any(r.intersects(o, margin=0.01 * W) for o in obstacles):
                    continue
                cost = 0.0
                if arrow:
                    (sx, sy), (tx, ty) = arrow_ends(r, face, size)
                    length = math.hypot(tx - sx, ty - sy)
                    if length < 0.1 * W:
                        continue
                    if avoid and _crosses(np.linspace((sx, sy), (tx, ty), 24), avoid, 0.02 * W):
                        continue  # an arrow through the captions
                    cost += abs(length - want) / W * 3.0
                    n = 24  # how much of the person the shaft would cross
                    xs = np.clip(np.linspace(sx, tx, n), 0, W - 1).astype(int)
                    ys = np.clip(np.linspace(sy, ty, n), 0, H - 1).astype(int)
                    cost += 1.5 * float((person[ys, xs] > 0.5).mean())
                else:
                    ref = ring if ring is not None else head
                    gap = math.hypot(max(0.0, max(ref.x0 - r.x1, r.x0 - ref.x1)), max(0.0, max(ref.y0 - r.y1, r.y0 - ref.y1)))
                    cost += abs(gap - 0.03 * W) / W * 4.0
                cost += max(0.0, r.cy - face.y1) / H * 2.0  # under the chin reads worse than beside or above
                if best is None or cost < best[0]:
                    best = (cost, r)
        if best is not None:
            r = best[1]
            side = "above" if r.y1 <= head.y0 + 0.2 * head.h else ("left of" if r.cx < head.cx else "right of")
            return Label(lines, px, r, f"{side} your head, on nobody")
    return None


# ---- the strokes ---------------------------------------------------------------------------------------

def _bezier(p0, p1, p2, n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * np.array(p0) + 2 * (1 - t) * t * np.array(p1) + t ** 2 * np.array(p2)


def _wobble(pts: np.ndarray, amount: float, seed: int) -> np.ndarray:
    """A hand's slight unsteadiness: a smooth sideways drift along the line, none at its ends."""
    rng = np.random.default_rng(seed)
    n = len(pts)
    d = np.gradient(pts, axis=0)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)
    noise = np.convolve(rng.standard_normal(n + 40), np.hanning(41) / np.hanning(41).sum(), mode="same")[20:20 + n]
    noise *= np.sin(np.linspace(0, np.pi, n))
    noise /= max(1e-6, float(np.abs(noise).max()))
    return pts + nrm * (noise * amount)[:, None]


def arrow_path(start: tuple[float, float], tip: tuple[float, float], bow: float, size: float, seed: int) -> np.ndarray:
    """One stroke: the shaft from `start` to `tip`, bowed sideways by `bow` (a share of its length),
    then the head flicked on in the same stroke: out along one barb, back to the tip, out along the
    other."""
    sx, sy = start
    ex, ey = tip
    L = math.hypot(ex - sx, ey - sy)
    mx, my = (sx + ex) / 2, (sy + ey) / 2
    nx, ny = -(ey - sy) / max(L, 1e-6), (ex - sx) / max(L, 1e-6)
    ctrl = (mx + nx * bow * L, my + ny * bow * L)
    shaft = _bezier(start, ctrl, tip, max(24, int(L / 4)))
    shaft = _wobble(shaft, 0.012 * L, seed)
    d = shaft[-1] - shaft[-4]
    ang = math.atan2(d[1], d[0])
    pts = [shaft]
    for k, turn in enumerate((2.62, -2.62)):  # about 150 degrees back from the way it was going
        a = ang + turn + (0.05 if k else -0.05)
        barb_end = (ex + math.cos(a) * size, ey + math.sin(a) * size)
        out = np.linspace((ex, ey), barb_end, 10)
        pts += [out, out[::-1]] if k == 0 else [out]
    return np.concatenate(pts)


def loop_path(around: Rect, size: tuple[int, int], seed: int) -> np.ndarray:
    """A loop drawn round `around` by hand: it starts at the upper right, goes round once and a bit
    more, and its end passes the start a little wide, as a quick circle does."""
    rng = np.random.default_rng(seed)
    cx, cy = around.cx, around.cy
    rx, ry = around.w / 2, around.h / 2
    a0 = -0.95
    turn = 2 * math.pi + 0.45
    n = 220
    t = np.linspace(0, 1, n)
    a = a0 - turn * t  # counter-clockwise on screen, as most hands draw it
    grow = 1.0 + 0.07 * t + 0.015 * np.sin(2 * np.pi * (2 * t + rng.uniform()))
    x = cx + rx * grow * np.cos(a)
    y = cy + ry * grow * np.sin(a)
    return np.stack([x, y], 1)


class Stroke:
    """A marker stroke along a path, drawn once with the moment each pixel was reached, so any part
    of it can be shown at once: `alpha(f)` is the stroke up to `f` of its length, its tip soft."""

    def __init__(self, path: np.ndarray, width: float, size: tuple[int, int]):
        from PIL import Image, ImageDraw

        W, H = size
        pad = int(math.ceil(width * 2 + 4))
        x0 = int(max(0, math.floor(path[:, 0].min()) - pad))
        y0 = int(max(0, math.floor(path[:, 1].min()) - pad))
        x1 = int(min(W, math.ceil(path[:, 0].max()) + pad))
        y1 = int(min(H, math.ceil(path[:, 1].max()) + pad))
        self.x0, self.y0 = x0, y0
        w, h = max(1, x1 - x0), max(1, y1 - y0)
        seg = np.hypot(*np.diff(path, axis=0).T)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(arc[-1]) if len(arc) else 1.0
        order = np.full((h, w), np.inf, np.float32)
        r = width / 2
        for i in range(len(path) - 1):
            img = Image.new("L", (w, h), 0)
            d = ImageDraw.Draw(img)
            (ax, ay), (bx, by) = path[i] - (x0, y0), path[i + 1] - (x0, y0)
            d.line([(ax, ay), (bx, by)], fill=255, width=max(1, int(round(width))))
            d.ellipse([bx - r, by - r, bx + r, by + r], fill=255)
            if i == 0:
                d.ellipse([ax - r, ay - r, ax + r, ay + r], fill=255)
            m = np.asarray(img) > 127
            order[m] = np.minimum(order[m], arc[i])
        self.order = order
        ink = np.isfinite(order).astype(np.float32)
        import cv2

        self.ink = cv2.GaussianBlur(ink, (0, 0), 0.7)
        self.order_filled = np.where(np.isfinite(order), order, self.length * 2).astype(np.float32)
        self.order_filled = cv2.dilate(self.order_filled * -1, np.ones((3, 3), np.uint8)) * -1  # the soft rim too
        self.width = width

    def alpha(self, f: float) -> np.ndarray:
        if f >= 1.0:
            return self.ink
        reach = f * self.length
        tip = max(2.0, self.width)
        shown = np.clip((reach - self.order_filled) / tip + 0.5, 0.0, 1.0)
        return self.ink * shown


# ---- colour ------------------------------------------------------------------------------------------

MARKERS = {"yellow": (255, 214, 10), "white": (250, 250, 247), "red": (255, 59, 48)}


def choose_marker(frame: np.ndarray, rects: list[Rect]) -> tuple[tuple[int, int, int], str]:
    """The marker colour that stands out most from the picture under the marks: by brightness first
    (a dark shadow under the marks does the rest), then by colour."""
    from .grade import oklab, to_linear

    H, W = frame.shape[:2]
    px = []
    for r in rects:
        x0, y0, x1, y1 = int(max(0, r.x0)), int(max(0, r.y0)), int(min(W, r.x1)), int(min(H, r.y1))
        if x1 > x0 and y1 > y0:
            px.append(frame[y0:y1:4, x0:x1:4].reshape(-1, 3))
    if not px:
        return MARKERS["yellow"], "yellow"
    under = np.concatenate(px).astype(np.float64) / 255.0
    lab_u = oklab(to_linear(under))
    best = None
    for rank, (name, rgb) in enumerate(MARKERS.items()):
        lab = oklab(to_linear(np.array(rgb, np.float64)[None] / 255.0))[0]
        d = np.sqrt(((lab_u - lab) ** 2) * np.array([2.0, 1.0, 1.0])).sum(axis=1)
        score = float(np.percentile(d, 10)) - 0.05 * rank  # the part of the picture it stands off least
        if best is None or score > best[0]:
            best = (score, name, rgb)
    return best[2], f"{best[1]}: it stands out most from the picture under the marks"


# ---- drawing the marks ---------------------------------------------------------------------------------

def _shadowed(rgb, a: np.ndarray, colour) -> tuple[np.ndarray, np.ndarray]:
    """The marks in their colour with a soft dark shadow under them, premultiplied, so a light
    marker reads on a bright wall and a dark one on a dark wall."""
    import cv2

    sh = np.roll(cv2.GaussianBlur(a, (0, 0), 2.2), (3, 2), axis=(0, 1)) * 0.55
    alpha = a + sh * (1 - a)
    col = np.array(colour, np.float32)
    out = a[..., None] * col
    return out.astype(np.float32), alpha.astype(np.float32)


class LabelInk:
    """The label written on in marker: revealed from left to right, line by line, as a hand writes."""

    def __init__(self, label: Label, colour):
        from PIL import Image, ImageDraw

        f = textmod.font(FACE, label.px)
        w, h, step = label_size(label.lines, label.px)
        pad = int(0.25 * label.px)
        W, H = int(math.ceil(label.rect.w)) + 2 * pad, int(math.ceil(label.rect.h)) + 2 * pad
        img = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(img)
        _, t0, _, _ = f.getbbox(label.lines[0], anchor="ls")
        self.lines = []
        for n, ln in enumerate(label.lines):
            l, t, r, b = f.getbbox(ln, anchor="ls")
            x = pad + (label.rect.w - (r - l)) / 2 - l
            base = pad - t0 + n * step
            d.text((x, base), ln, font=f, fill=255, anchor="ls")
            self.lines.append((x + l, x + r, base + t, base + b))
        a = np.asarray(img, np.float32) / 255.0
        self.rgb, self.a = _shadowed(colour, a, colour)
        self.x0, self.y0 = int(round(label.rect.x0)) - pad, int(round(label.rect.y0)) - pad
        self.chars = sum(len(s) for s in label.lines)

    def draw(self, frame: np.ndarray, f: float, opacity: float) -> None:
        if f <= 0 or opacity <= 0:
            return
        a = self.a
        if f < 1.0:
            # line by line, each in proportion to its letters
            total = sum(r - l for l, r, _, _ in self.lines)
            reach = f * total
            mask = np.zeros_like(a)
            xs = np.arange(a.shape[1], dtype=np.float32)
            for (l, r, t, b) in self.lines:
                span = r - l
                upto = l + min(span, max(0.0, reach))
                reach -= span
                edge = np.clip((upto - xs) / 6.0 + 0.5, 0, 1)
                y0, y1 = int(max(0, t - 0.3 * (b - t))), int(min(a.shape[0], b + 0.5 * (b - t)))
                mask[y0:y1] = np.maximum(mask[y0:y1], edge[None, :])
                if reach <= 0:
                    break
            a = a * mask
            rgb = self.rgb * mask[..., None]
        else:
            rgb = self.rgb
        over(frame, Sprite(rgb=rgb * opacity, a=a * opacity), self.x0, self.y0)


def _flash(frame: np.ndarray, k: float) -> None:
    if k <= 0.002:
        return
    import cv2

    frame[:] = cv2.addWeighted(frame, 1.0 - k, np.full_like(frame, 255), k, 0.0)


def _vignette(size: tuple[int, int], centre: tuple[float, float]) -> np.ndarray:
    W, H = size
    yy, xx = np.mgrid[0:H:4, 0:W:4].astype(np.float32)
    d = np.hypot((xx - centre[0]) / (0.62 * W), (yy - centre[1]) / (0.62 * H))
    v = 1.0 - 0.32 * np.clip((d - 0.55) / 0.75, 0, 1) ** 1.5
    import cv2

    return cv2.resize(v, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)


# ---- the comic panel -------------------------------------------------------------------------------------

def comic(img: np.ndarray, seed: int = 5) -> np.ndarray:
    """The held picture as a printed comic panel: flat colour in a few tones, black ink on its edges,
    halftone dots in its shadows."""
    import cv2

    f = cv2.bilateralFilter(img, 9, 60, 7)
    lab = cv2.cvtColor(f, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0]
    steps = 5
    Lq = np.round(L / 255.0 * (steps - 1)) / (steps - 1) * 255.0
    lab[..., 0] = 0.55 * Lq + 0.45 * L
    lab[..., 1:] = 128 + (lab[..., 1:] - 128) * 1.35
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # ink: the difference of two blurs, thresholded softly (strong lines, no speckle)
    s = max(1.0, img.shape[1] / 900.0)
    dog = cv2.GaussianBlur(g, (0, 0), 1.0 * s) - cv2.GaussianBlur(g, (0, 0), 1.8 * s)
    ink = np.clip((-dog - 2.5) / 3.0, 0, 1)
    ink = cv2.GaussianBlur(ink, (0, 0), 0.6)
    # halftone: dots on a grid, larger where it is darker
    H, W = g.shape
    cell = max(6, int(round(W / 120)))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u, v = (xx + yy) / (cell * 1.414), (xx - yy) / (cell * 1.414)  # the grid at 45 degrees
    du, dv = u - np.floor(u) - 0.5, v - np.floor(v) - 0.5
    dist = np.hypot(du, dv)
    shade = np.clip((150.0 - cv2.GaussianBlur(g, (0, 0), 2)) / 150.0, 0, 1)
    dots = np.clip((np.sqrt(shade) * 0.55 - dist) * cell, 0, 1)
    out *= (1.0 - 0.45 * dots)[..., None]
    out *= (1.0 - ink)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


class Panel:
    """The held frame inked into a comic panel: it shrinks a little into a black frame on a page,
    tilted a touch, and a caption box with the label pops on in a corner clear of the face."""

    PAGE = (246, 240, 226)
    BOX = (255, 238, 150)

    def __init__(self, frozen: np.ndarray, face: Rect, label_text: Optional[str], size: tuple[int, int], seed: int):
        import cv2
        from PIL import Image, ImageDraw

        W, H = size
        self.W, self.H = W, H
        self.inked = comic(frozen, seed)
        self.scale, self.tilt = 0.9, -1.6
        cx, cy = W / 2, H / 2
        self.M = cv2.getRotationMatrix2D((cx, cy), self.tilt, self.scale)
        self.border = max(4, int(round(0.011 * W)))
        rng = np.random.default_rng(seed)
        page = np.full((H, W, 3), self.PAGE, np.float32)
        page += rng.normal(0, 3.0, (H, W, 1)).astype(np.float32)
        self.page = np.clip(page, 0, 255).astype(np.uint8)
        # where the face lands on the page, to keep the caption box off it
        corners = np.array([[face.x0, face.y0], [face.x1, face.y0], [face.x0, face.y1], [face.x1, face.y1]], np.float32)
        moved = cv2.transform(corners[None], self.M)[0]
        self.face = Rect(float(moved[:, 0].min()), float(moved[:, 1].min()), float(moved[:, 0].max()),
                         float(moved[:, 1].max()))
        self.box = None
        self.where = ""
        if label_text:
            zone = SafeZone(W, H).rect
            text = label_text.upper()
            lines = label_lines(text) if len(text) > 18 else [text]
            px = int(0.042 * H)
            while px > 18:
                w, h, step = label_size(lines, px)
                if w <= 0.62 * W:
                    break
                px -= 2
            w, h, step = label_size(lines, px)
            padx, pady = 0.6 * px, 0.45 * px
            bw, bh = w + 2 * padx, h + 2 * pady
            # the panel's corners, inside the safe zone, the one furthest from the face first
            spots = {"top left": (zone.x0 + 0.01 * W, zone.y0 + 0.01 * H), "top right": (zone.x1 - bw - 0.01 * W, zone.y0 + 0.01 * H),
                     "bottom left": (zone.x0 + 0.01 * W, zone.y1 - bh - 0.01 * H),
                     "bottom right": (zone.x1 - bw - 0.01 * W, zone.y1 - bh - 0.01 * H)}
            order = sorted(spots, key=lambda k: -math.hypot(spots[k][0] + bw / 2 - self.face.cx, spots[k][1] + bh / 2 - self.face.cy))
            for name in order:
                x, y = spots[name]
                r = Rect(x, y, x + bw, y + bh)
                if not r.intersects(self.face, margin=0.015 * W):
                    self.box, self.where = r, f"a caption box at the {name}, clear of your face"
                    break
            if self.box is not None:
                r = self.box
                k = 3
                img = Image.new("RGBA", (int(bw) * k, int(bh) * k), (0, 0, 0, 0))
                d = ImageDraw.Draw(img)
                lw = max(2, int(round(0.004 * W))) * k
                d.rectangle([0, 0, img.width - 1, img.height - 1], fill=self.BOX + (255,), outline=(17, 17, 17, 255), width=lw)
                img = img.resize((int(bw), int(bh)), Image.LANCZOS)
                d = ImageDraw.Draw(img)
                f = textmod.font(FACE, px)
                _, t0, _, _ = f.getbbox(lines[0], anchor="ls")
                for n, ln in enumerate(lines):
                    l, t, rr, b = f.getbbox(ln, anchor="ls")
                    d.text(((bw - (rr - l)) / 2 - l, pady - t0 + n * step), ln, font=f, fill=(20, 20, 20, 255), anchor="ls")
                self.box_sprite = Sprite.from_rgba(np.asarray(img))

    def draw(self, frame: np.ndarray, u: float) -> None:
        import cv2

        W, H = self.W, self.H
        grow = min(1.0, max(0.0, u / 0.2))
        ease = 1 - (1 - grow) ** 3
        s = 1.0 + (self.scale - 1.0) * ease
        tilt = self.tilt * ease
        M = cv2.getRotationMatrix2D((W / 2, H / 2), tilt, s)
        ink = min(1.0, u / 0.12)
        pic = self.inked if ink >= 1.0 else cv2.addWeighted(frame, 1 - ink, self.inked, ink, 0)
        warped = cv2.warpAffine(pic, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mask = cv2.warpAffine(np.ones((H, W), np.float32), M, (W, H), flags=cv2.INTER_LINEAR)
        b = self.border
        edge = cv2.dilate((mask > 0.5).astype(np.uint8), np.ones((2 * b + 1, 2 * b + 1), np.uint8)).astype(np.float32)
        edge = cv2.GaussianBlur(edge, (0, 0), 0.8)
        out = self.page.astype(np.float32) * (1 - edge[..., None]) + 17.0 * edge[..., None]
        out = out * (1 - mask[..., None]) + warped.astype(np.float32) * mask[..., None]
        frame[:] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        if self.box is not None and u >= 0.16:
            k = min(1.0, (u - 0.16) / 0.14)
            pop = 0.85 + 0.15 * (1 - (1 - k) ** 3)
            sp = self.box_sprite.scaled(pop)
            over(frame, sp, self.box.cx - sp.w / 2, self.box.cy - sp.h / 2, min(1.0, 1.6 * k))


# ---- the arrow's ends ----------------------------------------------------------------------------------

def _exit(cx: float, cy: float, ux: float, uy: float, r: Rect) -> float:
    """How far from (cx, cy), inside `r`, a ray in direction (ux, uy) goes before it leaves `r`."""
    ts = []
    for v, lo, hi, c in ((ux, r.x0, r.x1, cx), (uy, r.y0, r.y1, cy)):
        if abs(v) > 1e-9:
            ts.append(max((lo - c) / v, (hi - c) / v))
    return max(0.0, min(ts)) if ts else 0.0


def arrow_for(label: Rect, face: Rect, size: tuple[int, int], seed: int) -> tuple[np.ndarray, dict]:
    """The arrow's stroke from beside the label to just outside the face. It bows to the side away
    from the face, and is drawn straighter if the bow would bring it onto the face."""
    W = size[0]
    start, tip = arrow_ends(label, face, size)
    shaft = math.hypot(tip[0] - start[0], tip[1] - start[1])
    barb = float(np.clip(0.2 * shaft, 0.03 * W, 0.06 * W))
    zone = face_zone(face)
    nx, ny = -(tip[1] - start[1]) / max(shaft, 1e-6), (tip[0] - start[0]) / max(shaft, 1e-6)
    mx, my = (start[0] + tip[0]) / 2, (start[1] + tip[1]) / 2
    away = 1.0 if (mx + nx - zone.cx) ** 2 + (my + ny - zone.cy) ** 2 > (mx - nx - zone.cx) ** 2 + (my - ny - zone.cy) ** 2 else -1.0
    clear = zone.grow(0.01 * W, 0.01 * W, 0.01 * W, 0.01 * W)
    for bow in (0.17 * away, 0.09 * away, 0.0, -0.09 * away):
        path = arrow_path(start, tip, bow, barb, seed)
        inside = ((path[:, 0] > clear.x0) & (path[:, 0] < clear.x1) & (path[:, 1] > clear.y0) & (path[:, 1] < clear.y1))
        if not inside.any():
            break
    gap = _exit(zone.cx, zone.cy, *(np.subtract(tip, (zone.cx, zone.cy)) / max(1e-6, math.hypot(tip[0] - zone.cx, tip[1] - zone.cy))), zone)
    dist = math.hypot(tip[0] - zone.cx, tip[1] - zone.cy) - gap
    info = {"start": [round(start[0], 1), round(start[1], 1)], "tip": [round(tip[0], 1), round(tip[1], 1)],
            "bow": round(bow, 3), "barb": round(barb, 1), "from_face": round(dist, 1), "length": round(shaft, 1)}
    return path, info


# ---- drawing ---------------------------------------------------------------------------------------------

class Painter:
    """Draws the freeze: the flash, the held picture a touch darker at its edges, the marks drawing
    themselves on, then fading as the picture moves again; captions outside the hold."""

    def __init__(self, t0: float, hold: float, mark: str, size: tuple[int, int], colour,
                 stroke: Optional[Stroke], label: Optional[LabelInk], panel: Optional[Panel],
                 vignette: Optional[np.ndarray], frozen: Optional[np.ndarray], captions=None,
                 captions_in_hold: bool = False):
        self.t0, self.hold, self.mark = t0, hold, mark
        self.W, self.H = size
        self.colour = colour
        self.stroke, self.label, self.panel = stroke, label, panel
        self.vignette, self.frozen = vignette, frozen
        self.captions, self.captions_in_hold = captions, captions_in_hold
        # the marks' timing inside the hold, squeezed into a short one
        k = float(np.clip((hold - OUT - 0.1) / 0.75, 0.45, 1.0))
        if mark == "circle":
            self.draw_mark, self.write = (0.06 * k, 0.5 * k), (0.42 * k, 0.78 * k)
        else:
            self.write, self.draw_mark = (0.06 * k, 0.4 * k), (0.32 * k, 0.66 * k)

    @staticmethod
    def _p(u: float, span: tuple[float, float]) -> float:
        a, b = span
        x = min(1.0, max(0.0, (u - a) / max(1e-6, b - a)))
        return x * x * (3 - 2 * x)

    def in_hold(self, t: float) -> bool:
        return self.t0 - 1e-9 <= t < self.t0 + self.hold

    def draw(self, frame: np.ndarray, j: int, t: float) -> None:
        if not self.in_hold(t):
            if self.captions is not None:
                self.captions.draw(frame, j, t)
            return
        u = t - self.t0
        if self.frozen is not None:  # the voice carries on: the picture is held here
            frame[:] = self.frozen
        fade = 1.0 - self._p(u, (self.hold - OUT, self.hold))
        if self.panel is not None:
            held = frame.copy()
            self.panel.draw(frame, u)
            if fade < 1.0:
                frame[:] = np.clip(held.astype(np.float32) + (frame.astype(np.float32) - held) * fade + 0.5,
                                   0, 255).astype(np.uint8)
        else:
            if self.vignette is not None:
                import cv2

                v = 1.0 - (1.0 - self.vignette) * fade
                frame[:] = cv2.multiply(frame.astype(np.float32), cv2.merge([v, v, v])).clip(0, 255).astype(np.uint8)
            if self.stroke is not None:
                f = self._p(u, self.draw_mark)
                if f > 0:
                    a = self.stroke.alpha(f)
                    rgb, alpha = _shadowed(self.colour, a, self.colour)
                    over(frame, Sprite(rgb=rgb * fade, a=alpha * fade), self.stroke.x0, self.stroke.y0)
            if self.label is not None:
                self.label.draw(frame, self._p(u, self.write), fade)
        if self.captions is not None and self.captions_in_hold:
            self.captions.draw(frame, j, t)
        _flash(frame, 0.55 * max(0.0, 1.0 - u / FLASH))


# ---- the sound ---------------------------------------------------------------------------------------------

class Sound:
    """What is heard while the picture holds: the clip's own room, looped from its quietest stretch
    with nobody speaking and laid at the room's level, so the hold plays as the room holding still and
    never as the sound dropping out; and over it the scratch or the hit (none for `sound=none`), set
    against the voice's loudness. The voice itself is the clip's, stopped for the hold by the
    timeline and untouched either side of it."""

    def __init__(self, clip, kind: str, at: float, hold: float):
        self.kind, self.at, self.hold = kind, at, hold
        self.clip = clip
        self.voice = soundmod.integrated(soundmod.blocks(clip.sound)) if clip.info.has_audio else soundmod.SILENCE_DB
        self.ref = self.voice if self.voice > -70 else -23.0
        self.limited_db = 0.0
        self.laid = ""

    def _room(self, out: np.ndarray, rate: int) -> str:
        """The room under the hold, crossfaded with the voice as it fades out into the freeze and
        back in after it (the same fades, so the sum keeps its power and nothing clicks)."""
        from ..core import audio
        from ..core.render import FREEZE_FADE

        if not self.clip.info.has_audio:
            return ""
        n = max(1, int(round(FREEZE_FADE * rate)))
        cut = int(round(self.at * rate))
        span = int(round(self.hold * rate))
        a = max(0, cut - n)
        room = audio.room_tone(self.clip.sound, rate, (cut + span + n - a) / rate, self.clip.quiet)
        x = room.samples.copy()
        if not len(x):
            return ""
        m = min(n, cut - a)
        if m:
            x[:m] *= np.sin(np.linspace(0, np.pi / 2, m, dtype=np.float32))[:, None]
        x[len(x) - n:] *= np.cos(np.linspace(0, np.pi / 2, n, dtype=np.float32))[:, None]
        b = min(len(out), a + len(x))
        out[a:b] += x[:b - a]
        return (f"the room's own sound under the hold, looped from {room.taken[0]:.2f}–{room.taken[1]:.2f} s "
                f"at {room.level_db:.0f} dBFS")

    def mix(self, voice: np.ndarray, rate: int) -> np.ndarray:
        from ..core import loudness

        out = np.array(voice, np.float32)
        said = [self._room(out, rate)]
        if self.kind in ("scratch", "hit"):
            if self.kind == "scratch":
                x, anchor = synth.scratch()
            else:
                from .sound_effects import library

                s = library()["hit"]
                x, anchor = s.samples, s.anchor
            stem = np.zeros_like(out)
            soundmod.lay(stem, synth.at_level(x, self.ref - 3.0), self.at, rate, anchor=anchor)
            out = out + stem
            said.insert(0, f"a {'record scratch' if self.kind == 'scratch' else 'hit'} on the freeze at "
                           f"{self.at:.2f} s, 3 dB under your voice")
        ceiling = max(-1.0, loudness.true_peak_db(voice) + 0.05)
        if loudness.true_peak_db(out) > ceiling:
            out, self.limited_db = loudness.limit(out, ceiling)
        self.laid = "; ".join(x for x in said if x)
        return out.astype(np.float32)


# ---- the effect ----------------------------------------------------------------------------------------------

def _plan_moment(plan: dict, clip, duration: float, PlanError) -> Moment:
    try:
        fz = plan["freeze"]
        t = float(fz["at"])
    except (KeyError, TypeError, ValueError):
        raise PlanError("the plan needs freeze.at (seconds); run once without --plan to get one to edit") from None
    if not 0 <= t < duration:
        raise PlanError(f"the plan freezes at {t} s, outside the clip (0–{duration:.2f} s)")
    k = int(round(t * clip.info.rate))
    return Moment(k / clip.info.rate, k, "as the plan says", fz.get("word"))


def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from ..core import framing as framing_mod
    from ..core import video
    from ..core.timeline import Timeline
    from . import Job, PlanError

    if plan is not None:
        kept = plan.get("options") or {}
        opts = dict(opts, **{k: kept[k] for k in ("mark", "sound", "hold", "frame", "label", "captions") if k in kept})
        if opts["mark"] not in MARKS or opts["sound"] not in SOUNDS:
            raise PlanError("the plan names a mark or a sound this effect does not have")
    info = clip.info
    fps = info.rate
    duration = info.frames / fps
    fr = framing_mod.plan(info.width, info.height, fps, info.frames, clip.face, shape=opts["frame"])
    W, H = fr.out_w, fr.out_h
    moment = _plan_moment(plan, clip, duration, PlanError) if plan is not None else decide_moment(clip, opts, duration)
    hold = round(opts["hold"] * fps) / fps  # a whole number of frames, so picture and sound stop and start together
    voice = opts["sound"] == "voice"
    to_end = voice and moment.t + hold > duration - 1.0 / fps
    if to_end:  # the voice runs on and the film keeps its length: a hold can last only as long as the clip does
        hold = max(1.0 / fps, math.floor((duration - moment.t) * fps - 1) / fps)
    whole = Timeline.whole(duration)
    tl = whole if voice else whole.frozen(moment.t, hold)
    t0 = moment.t if voice else tl.to_output(moment.t)
    n_out = tl.output_frames(fps)
    warnings: list[str] = []

    k = moment.k
    held = fr.apply(video.read_frame(info, k), k)
    hb, fb = clip.face.head(k), clip.face.box(k)
    if hb is None or fb is None:
        raise O.OptionError("no face is found on the frame that would be held, so there is nobody to point at; try "
                            "another moment with at=")
    head, face = fr.to_output(hb, k), fr.to_output(fb, k)
    person = _person_at(clip, fr, k, (W, H))
    seed = 7 + k

    # captions: in the output's own time; while the picture holds they wait (or carry on, with the voice)
    caps = None
    avoid: list[Rect] = []
    if opts["captions"] != "off":
        words = [Word(w.text, tl.to_output_nearest(w.start), max(tl.to_output_nearest(w.end), tl.to_output_nearest(w.start) + 0.02),
                      w.prob) for w in clip.words]
        quiet = tl.map_spans(clip.quiet) + tl.still_spans()
        caps = cap.Captions(words, sorted(quiet), (W, H), fps, n_out, cap.head_in_output(clip, tl, fr),
                            look=opts["captions"])
        if voice:
            avoid = [c.rect for c in caps.cards if c.start < t0 + hold and c.end > t0]

    label_text = DEFAULT_LABEL if opts["label"] == "auto" else ("" if opts["label"] == "none" else opts["label"])
    label = stroke = ink = panel = None
    avoid_failed = False
    vignette = None
    mark_info: dict = {}
    if opts["mark"] == "panel":
        panel = Panel(held, face, label_text or None, (W, H), seed)
        placed = panel.where or "the whole frame, inked as a comic panel"
        colour, colour_why = Panel.BOX, "the caption box's own yellow"
        mark_rects = [panel.box] if panel.box is not None else []
        face_after = panel.face
        if label_text and panel.box is None:
            warnings.append("no corner of the panel is clear of your face, so the caption box is left out")
    else:
        lines = label_lines(label_text or DEFAULT_LABEL)
        ring = None
        if opts["mark"] == "circle":
            ring = head.grow(0.2 * head.w, 0.12 * head.h, 0.2 * head.w, 0.1 * head.h)
        label = place_label(lines, person, head, face, (W, H), avoid, ring, arrow=opts["mark"] == "arrow")
        if label is None and avoid:  # no room clear of the captions as well: the captions wait instead
            label = place_label(lines, person, head, face, (W, H), [], ring, arrow=opts["mark"] == "arrow")
            avoid_failed = label is not None
        if label is None:
            raise O.OptionError("there is no free space beside your head on the held frame for a label: try "
                                "another moment (at=), a shorter label, or mark=panel")
        if opts["mark"] == "arrow":
            path, mark_info = arrow_for(label.rect, face, (W, H), seed)
        else:
            path = loop_path(ring, (W, H), seed)
            mark_info = {"round": ring.as_list()}
        stroke = Stroke(path, max(4.0, 0.012 * W), (W, H))
        bbox = Rect(stroke.x0, stroke.y0, stroke.x0 + stroke.ink.shape[1], stroke.y0 + stroke.ink.shape[0])
        if opts["color"] != "auto":
            colour, colour_why = textmod.hex_rgb(opts["color"]), "as asked"
        else:
            colour, colour_why = choose_marker(held, [label.rect, bbox])
        ink = LabelInk(label, colour) if label_text else None
        vignette = _vignette((W, H), (face.cx, face.cy))
        placed = (label.where if label_text else "no label") + (", an arrow to you" if opts["mark"] == "arrow"
                                                               else ", a loop round your head")
        mark_rects = [label.rect] if label_text else []
        face_after = face

    # nothing over the face: every pixel of the marks against the face's own box on the held frame
    over_face = 0
    hold_frames = int(round(hold * fps))
    fx0, fy0, fx1, fy1 = int(max(0, face_after.x0)), int(max(0, face_after.y0)), int(min(W, face_after.x1)), int(min(H, face_after.y1))
    if stroke is not None:
        sx0, sy0 = stroke.x0, stroke.y0
        sub = stroke.ink[max(0, fy0 - sy0):max(0, fy1 - sy0), max(0, fx0 - sx0):max(0, fx1 - sx0)]
        if sub.size and float(sub.max()) > 0.05:
            over_face = hold_frames
    for r in mark_rects:
        if r.intersects(face_after):
            over_face = hold_frames
    zone = SafeZone(W, H)
    outside = hold_frames if any(not zone.contains(r) for r in mark_rects) else 0

    in_hold = voice
    if caps is not None and voice and (avoid_failed or (stroke is not None and _crosses(path, avoid, 0.01 * W))):
        in_hold = False
        warnings.append("the captions wait while the picture holds: the label and its arrow had no room round them")
    painter = Painter(t0, hold, opts["mark"], (W, H), colour, stroke, ink, panel, vignette,
                      held if voice else None, caps, captions_in_hold=in_hold)
    sounds = Sound(clip, opts["sound"], t0, hold) if not voice else None

    said = f" (said {moment.said[0]:.2f}–{moment.said[1]:.2f} s)" if moment.said else ""
    length = (f"the film keeps its length ({tl.duration:.2f} s): your voice carries on under the still picture, "
              f"which then jumps ahead to meet it" if voice else
              f"your voice stops with the picture" + (", a record scratch lands on it" if opts["sound"] == "scratch" else
                                                     ", a hit lands on it" if opts["sound"] == "hit" else "")
              + ", the room's own sound holds under the still picture"
              + f"; the film is {hold:.2f} s longer ({duration:.2f} → {tl.duration:.2f} s)")
    checks = {
        "freeze": f"at {moment.t:.2f} s, {moment.why}{said}; frame {k} held {hold:.2f} s"
                  + (f", {moment.sharp}" if moment.sharp else "") + (", to the end of the clip" if to_end else ""),
        "sound": length,
        "mark": opts["mark"] + (f": {mark_info.get('length', 0):.0f} px long, from the label to "
                                f"{mark_info.get('from_face', 0):.0f} px outside your face (the forehead counted)"
                                if opts["mark"] == "arrow" else ""),
        "label": f"“{label_text}”" if label_text else "none",
        "colour": colour_why,
        "frames_over_face": over_face + int(caps.report()["frames_over_face"]) if caps else over_face,
        "frames_outside_safe_zone": outside + (int(caps.report()["frames_outside_safe_zone"]) if caps else 0),
        "placed": placed,
        "warnings": warnings + (list(caps.report()["warnings"]) if caps else []),
    }
    sheet = [round(min((n_out - 2) / fps, max(0.0, x)), 3) for x in
             (t0 - 0.45, t0 + 0.04, t0 + painter.write[1] * 0.8 if opts["mark"] != "circle" else t0 + painter.draw_mark[1] * 0.6,
              t0 + max(painter.write[1], painter.draw_mark[1]) + 0.05, t0 + hold - OUT - 0.08, t0 + hold + 0.45)]
    out_plan = {
        "effect": NAME, "clip": clip.path, "options": opts,
        "output": {"width": W, "height": H, "fps": round(fps, 3), "seconds": round(tl.duration, 3), "framing": fr.note},
        "freeze": {"at": round(moment.t, 4), "frame": k, "output_at": round(t0, 4), "hold": round(hold, 4),
                   "why": moment.why, "word": moment.word,
                   "said": [round(x, 3) for x in moment.said] if moment.said else None},
        "mark": opts["mark"], "mark_geometry": mark_info,
        "label": {"text": label_text, "lines": label.lines if label else None, "px": label.px if label else None,
                  "rect": label.rect.as_list() if label else (panel.box.as_list() if panel and panel.box else None)},
        "color": "#%02x%02x%02x" % tuple(int(v) for v in colour),
        "head": head.as_list(), "face": face.as_list(),
        "sheet_times": sheet,
    }

    def still_times() -> list[float]:
        return sheet[1:5]

    def after(result: dict, out_path) -> dict:
        return {"sound": sounds.laid + (f"; peaks held {sounds.limited_db:.1f} dB" if sounds.limited_db else "")} \
            if sounds is not None and sounds.laid else {}

    return Job(NAME, clip, tl, fr, painter.draw, out_plan, checks=checks, still_times=still_times, after=after,
               sound=sounds.mix if sounds is not None else None)
