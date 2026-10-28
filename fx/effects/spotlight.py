"""spotlight: the lights go down. The room goes dark and loses its colour, a soft light falls on you
from above, and you stay lit as you were filmed. Five looks:

- `dark`: the room alone, dark;
- `neon`: a neon sign of your word on the wall behind your head, striking on like a real tube;
- `led`: a dot-matrix sign behind your head that types your word out;
- `rays`: golden rays from behind your head, sparkles in the dark;
- `haze`: a beam of light falling on you through haze, dust drifting in it.

How it decides, in order:

1. **When.** The lights go down on a word (`at`): by default the word that names the point of your
   opening line (the same word `words-behind` would choose), so the beat lands on it; for a sign,
   half a second before the word, so the sign strikes on as it is said. `at=start` puts them down
   from the first frame. They stay down to the end, or come back up on `until`.
2. **The room.** Darkened and drained of colour by `dark`, with a faint, broad pool of light left on
   the wall behind your head and shoulders, so the room keeps some depth. Broad on purpose: light
   kept close round your outline reads as a glow stuck to you, a halo by another name.
3. **You.** As filmed, from your head to your chest: the light on your face is never changed, so
   skin keeps its own colour (measured: before against after). Lower down you fall off a little,
   as a light from above does.
4. **The edge.** The person is cut out with the core's matte (`core/matte.py`): two looks at each
   frame and only what both call you, the edge firmed, steadied where nothing moves, and only the
   old room's share of each edge pixel darkened. A bright rim of the lit room round a dark room is
   what gives a cut-out away; the run measures it on the render, and how much the edge flickers
   against the clip's own.
5. **The sign.** Placed as `words-behind` places its word (as large as the room above you allows,
   a little hidden behind your head, every frame measured), drawn as tubes or dots, and lit: a white
   core, its colour round it and its light thrown on the wall, and a touch of that light on the
   edge of your hair and shoulders. Its colour, unless given, is chosen to stand off the dark room
   and what you wear near it.
6. **The sound.** The switch as the lights go, the neon's crackle and hum, the sign's ticks, the
   light's shimmer: all made in code (`core/synth.py`), set under your voice, which is kept as it is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core import matte, moments, synth
from ..core import sound as soundmod
from ..core import text as textmod
from ..core.layout import Rect, SafeZone
from ..core.words import HESITATIONS, normalize
from . import captions as cap
from . import options as O
from . import words_behind as wb

NAME = "spotlight"
LOOKS = ("dark", "neon", "led", "rays", "haze")
NEON_FACE = "poppins-bold"  # its skeleton makes a sign's tubes: one even line, round at the ends


# ---- the options -------------------------------------------------------------------------------------

@dataclass(frozen=True)
class _Moment:
    help: str
    first: str = "auto"

    def parse(self, raw) -> str:
        v = str(raw).strip().strip("\"“”")
        low = v.lower()
        if low in ("", "auto"):
            return self.first
        if low in ("start", "begin", "beginning", "end"):
            return low if low in ("start", "end") else "start"
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
class _Sign:
    help: str

    def parse(self, raw) -> str:
        v = str(raw).strip().strip("\"“”")
        if v.lower() in ("", "auto"):
            return "auto"
        if len(v.split()) > 4 or len(v) > 24:
            raise O.OptionError(f"give a word or a short phrase, four words and 24 letters at most: a sign behind a "
                                f"head has room for little more (got {raw!r})")
        if not any(c.isalnum() for c in v):
            raise O.OptionError(f"give a word to show (got {raw!r})")
        return v


OPTIONS = {
    "at": _Moment("auto: on the word that makes the point (a sign: half a second before it) · start: from the first "
                  "frame · a time such as 2.4 · or words said in the clip, such as \"the marketing campaign\""),
    "until": _Moment("end: the lights stay down · a time such as 6.5 · or words said in the clip: the lights come "
                     "back up as they are said", first="end"),
    "dark": O.Number(0, 1, "how dark the room goes: 0 not at all, 1 black, 0.8 by default"),
    "look": O.Choice(LOOKS, "dark: the room alone · neon: a neon sign of your word behind your head · led: a "
                            "dot-matrix sign that types your word out · rays: golden rays and sparkles behind you · "
                            "haze: a beam of light through haze"),
    "word": _Sign("auto: the word that makes the point of your line, as you say it · or the word or short phrase "
                  "for the neon or LED sign"),
    "color": O.Colour("auto: chosen against the dark room · a name such as pink, blue or yellow · or a hex colour "
                      "such as #ff3d9a. The sign's colour, the rays', the beam's, or the light on the wall"),
    "sound": O.Switch("on: the switch as the lights go, and the sign's or the light's own sound, under your voice · "
                      "off: your sound as it was"),
    "captions": O.Choice(("off",) + cap.LOOKS, "off · or the pack's captions on top, in a look: stacked, one-word, "
                                                "serif-chip or fitted"),
    "frame": O.Choice(("keep", "9:16"), "keep the clip's own shape, or make it 9:16 following the face"),
    "fix": O.Text("misheard words as wrong=right, separated by ;  e.g. \"Jon=John\""),
}

# The runs `tests/matrix.py spotlight` makes: the four clips the master list names, each on the
# default (the room alone), then each look on the clip it suits, and the train with a sign given
# (nothing clear is said on it), the hard test for a steady edge.
MATRIX = [
    ("D", "D", {}),
    ("A", "A", {}),
    ("B", "B", {}),
    ("C", "C", {}),
    ("D-neon", "D", {"look": "neon"}),
    ("A-neon", "A", {"look": "neon"}),
    ("B-led", "B", {"look": "led"}),
    ("C-rays", "C", {"look": "rays"}),
    ("B-haze", "B", {"look": "haze"}),
    ("F-neon", "F", {"look": "neon", "word": "NEXT STOP", "at": "0.6"}),
]

DOWN = 0.32  # seconds the lights take to go down: most of it in the first tenth, then the filament's tail
UP = 0.55  # and to come back up
LEAD = 0.04  # the lights start down this long before the word, so its first frame is already darker
SIGN_BEFORE = 0.5  # a sign's lights go down this long before its word, so it strikes on as it is said
STRIKE_LEAD = 0.14  # the tube starts to strike this long before its word: lit as it is heard


def _ease_down(u: float) -> float:
    """The lights going out: a breaker drops most of the light at once, a filament fades the rest."""
    u = min(1.0, max(0.0, u))
    return (1.0 - math.exp(-4.5 * u)) / (1.0 - math.exp(-4.5))


def _smooth(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    u = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _lum(rgb) -> np.ndarray:
    return wb._lum(rgb)


# ---- when ------------------------------------------------------------------------------------------

@dataclass
class Timing:
    down: float  # output seconds: the lights start to go down
    up: Optional[float]  # and come back up; None: they stay down
    why: str
    word: Optional[str] = None  # the word the lights went down on
    said_at: Optional[float] = None


def _words(clip) -> list:
    return [w for w in clip.words if w.key not in HESITATIONS]


def _said(clip, phrase: str, what: str) -> tuple[float, str, float]:
    words = _words(clip)
    span = wb.find_said(words, phrase)
    if span is None:
        raise O.OptionError(f"{what}: could not find \"{phrase}\" in what is said; `python -m fx.run words <video>` "
                            "lists the words, or give a time such as 2.4")
    first = words[span[0]]
    return first.start, " ".join(w.text for w in words[span[0]:span[1] + 1]), words[span[1]].end


def decide_timing(clip, opts: dict, seconds: float, sign_land: Optional[float]) -> Timing:
    at, until = opts["at"], opts["until"]
    if at == "start":
        t = Timing(0.0, None, "from the first frame")
    elif at == "auto":
        if sign_land is not None:
            down = max(0.0, sign_land - SIGN_BEFORE)
            t = Timing(down, None, "half a second before the sign's word" if down > 0 else "from the first frame: "
                       "the sign's word comes straight away")
        else:
            words = _words(clip)
            # the core's picker, as words-behind uses it: never a word the speaker took back
            back = moments.taken_back(clip.words, clip.quiet)
            cand, _ = moments.pick_word(words, clip.quiet, back) if words else (None, [])
            if cand is None:
                t = Timing(0.0, None, "from the first frame: no spoken word stands out to go down on")
            else:
                t = Timing(max(0.0, cand.word.start - LEAD), None, f"on “{moments.bare(cand.word)}”, the word that "
                           f"makes the point of “{' '.join(words[i].text for i in cand.line)}”",
                           moments.bare(cand.word), cand.word.start)
    elif wb._is_time(at):
        t = Timing(float(at), None, "at the time given")
    else:
        start, said, _ = _said(clip, at, "at")
        t = Timing(max(0.0, start - LEAD), None, f"on “{said}”", said, start)
    if t.down >= seconds - 0.4:
        raise O.OptionError(f"the lights would go down at {t.down:.2f} s, too close to the end of the clip "
                            f"({seconds:.2f} s)")
    if until not in ("end", "auto"):
        if wb._is_time(until):
            up = float(until)
        else:
            up, _, _ = _said(clip, until, "until")
            up = max(0.0, up - LEAD)
        if up <= t.down + DOWN + 0.2:
            raise O.OptionError(f"the lights would come back up at {up:.2f} s, before they are down "
                                f"({t.down:.2f} s); give a later time or later words")
        t.up = up if up < seconds - 0.05 else None
    return t


def weight(t: float, timing: Timing) -> float:
    """How far down the lights are at output time t: 0 as filmed, 1 all the way down."""
    if t < timing.down:
        return 0.0
    w = _ease_down((t - timing.down) / DOWN)
    if timing.up is not None and t >= timing.up:
        w *= 1.0 - _smooth((t - timing.up) / UP)
    return w


# ---- where the person is ---------------------------------------------------------------------------

def head_path(clip, fr, n_out: int, fps: float, tl) -> list[Optional[Rect]]:
    """The head's box at every output frame, in output pixels, smoothed so the light follows calmly;
    None where no face is known anywhere."""
    face = clip.face
    if not face.any():
        return [None] * n_out
    rows = []
    for j in range(n_out):
        k = tl.source_frame(j, fps, clip.info.frames)
        h = face.head(k)
        r = fr.to_output(h, k) if h is not None else None
        rows.append([r.cx, r.cy, r.w, r.h] if r is not None else [np.nan] * 4)
    a = np.array(rows, np.float64)
    ok = ~np.isnan(a[:, 0])
    idx = np.flatnonzero(ok)
    out = np.zeros_like(a)
    sigma = max(1.0, 0.25 * fps)
    rad = int(3 * sigma)
    kern = np.exp(-0.5 * (np.arange(-rad, rad + 1) / sigma) ** 2)
    kern /= kern.sum()
    for c in range(4):
        filled = np.interp(np.arange(n_out), idx, a[idx, c])
        out[:, c] = np.convolve(np.pad(filled, rad, mode="edge"), kern, mode="valid")
    return [Rect(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2) for cx, cy, w, h in out]


# ---- the light -------------------------------------------------------------------------------------

class Light:
    """The room darkened and drained of colour, with a faint, broad pool of light on the wall behind
    the head and shoulders; and the person lit from above, the face exactly as filmed.

    The pool is broad on purpose: a light kept close round the person's outline reads as a glow
    hugging them, which is a halo by another name (the first try on clip D, 28 October 2026)."""

    def __init__(self, size: tuple[int, int], dark: float, tint: Optional[tuple[int, int, int]] = None):
        self.W, self.H = size
        self.dark = float(dark)
        self.far = (1.0 - self.dark) ** 1.3 * (1.0 - 0.35 * self.dark)  # the room away from the person
        self.pool = 0.16 * self.dark  # the wall behind the head and shoulders gains this much light
        self.near = self.far + self.pool
        self.keep = 1.0 - 0.8 * self.dark  # how much of the room's colour stays
        self.fall = 0.3 * self.dark  # how much darker the person is at the bottom of the frame
        self.tint = None if tint is None else np.array(tint, np.float32) / max(1.0, float(max(tint)))
        q = (max(8, self.W // 8), max(8, self.H // 8))
        self.q = q
        self.yy = np.linspace(0.0, self.H, q[1], dtype=np.float32)[:, None]
        self.xx = np.linspace(0.0, self.W, q[0], dtype=np.float32)[None, :]

    def maps(self, a_small: np.ndarray, head: Optional[Rect]) -> tuple[np.ndarray, np.ndarray]:
        """The room's light (0..1) and the person's (0..1) at an eighth of the frame."""
        W, H = self.W, self.H
        if head is not None:
            cx, cy = head.cx, head.cy + 0.55 * head.h
            rx, ry = 0.75 * head.w + 0.16 * W, 1.25 * head.h + 0.12 * H
            body = 1.0 - self.fall * _smoothstep(head.y1 + 0.15 * head.h, head.y1 + 2.4 * head.h, self.yy)
            side = np.abs(self.xx - head.cx) / max(1.0, 1.8 * head.w + 0.15 * W)
            body = body * (1.0 - 0.5 * self.fall * _smoothstep(0.6, 1.4, side))
        else:
            cx, cy, rx, ry = W / 2, 0.45 * H, 0.45 * W, 0.4 * H
            body = 1.0 - self.fall * _smoothstep(0.5 * H, H, self.yy)
        d2 = ((self.xx - cx) / rx) ** 2 + ((self.yy - cy) / ry) ** 2
        room = self.far + self.pool * np.exp(-1.6 * d2)
        person = np.broadcast_to(body, a_small.shape).astype(np.float32)
        return room.astype(np.float32), person

    def room(self, F: np.ndarray, room_small: np.ndarray) -> np.ndarray:
        """The room as it is with the lights down: darker, greyer, deeper in the shadows."""
        import cv2

        g = cv2.resize(room_small, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        k = self.keep
        # each channel: keep of its own, the rest the grey of the pixel, then the room's light
        mix = np.full((3, 3), (1.0 - k), np.float32) * np.array([0.2126, 0.7152, 0.0722], np.float32)[None, :]
        mix += np.eye(3, dtype=np.float32) * k
        out = cv2.transform(F, mix)
        out = cv2.multiply(out, cv2.merge([g, g, g]))
        if self.tint is not None:  # a coloured light on the wall: only where the pool falls
            kk = np.clip((g - self.far) / max(1e-3, self.pool), 0.0, 1.0)[..., None]
            out *= 1.0 + 0.8 * kk * (self.tint - 1.0)
        return out

    def person(self, F: np.ndarray, person_small: np.ndarray) -> np.ndarray:
        import cv2

        L = cv2.resize(person_small, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return cv2.multiply(F, cv2.merge([L, L, L]))


# ---- colour ----------------------------------------------------------------------------------------

# The tube colours `color=auto` chooses between, in the order it prefers them. Chosen for this pack.
NEON = {"pink": (255, 52, 154), "blue": (54, 132, 255), "violet": (164, 84, 255), "red": (255, 46, 64),
        "aqua": (34, 224, 206), "lime": (154, 255, 64), "amber": (255, 168, 36)}
LED = {"amber": (255, 152, 32), "red": (255, 52, 40), "white": (236, 240, 255), "green": (84, 255, 120),
       "blue": (64, 150, 255)}
GOLD = (255, 196, 92)
HAZE = (255, 238, 214)


def _oklab_hue(rgb) -> tuple[float, float]:
    """Hue (degrees) and chroma of an sRGB colour, in Oklab."""
    from .grade import oklab, to_linear

    lab = oklab(to_linear(np.asarray(rgb, np.float64).reshape(1, 3) / 255.0))[0]
    return float(math.degrees(math.atan2(lab[2], lab[1])) % 360), float(math.hypot(lab[1], lab[2]))


def _hue_name(h: float) -> str:
    for upto, name in ((15, "red"), (45, "orange"), (75, "skin and wood"), (115, "yellow"), (165, "green"),
                       (215, "teal"), (285, "blue"), (335, "violet"), (360, "red")):
        if h < upto:
            return name
    return "red"


def choose_colour(pool: dict, near: list[np.ndarray]) -> tuple[str, tuple[int, int, int], str]:
    """The colour from `pool` that stands off most from the colours `near` it (the dark room round the
    sign and the person's own colours beside it), each weighed by how colourful it is; among those
    that stand off as well, the pool's own order (the classic colour of the look first)."""
    seen = []
    for px in near:
        px = np.asarray(px, np.float64).reshape(-1, 3)
        if not len(px):
            continue
        for c in px[:: max(1, len(px) // 400)]:
            h, ch = _oklab_hue(c)
            if ch > 0.02:
                seen.append((h, min(1.0, ch / 0.12)))
    total = max(1.0, len(seen) / 10.0)
    best = None
    for rank, name in enumerate(pool):
        h, ch = _oklab_hue(pool[name])
        clash = 0.0
        if ch > 0.04:  # a white has no hue to clash with
            clash = sum(w * math.exp(-0.5 * (((h - sh + 180) % 360 - 180) / 28.0) ** 2) for sh, w in seen) / total
        score = clash + 0.08 * rank
        if best is None or score < best[0]:
            best = (score, name, pool[name], clash)
    _, name, rgb, clash = best
    if not seen:
        return name, rgb, f"{name}: nothing near it has a colour of its own to clash with"
    hues: dict = {}
    for h, w in seen:
        hues[_hue_name(h)] = hues.get(_hue_name(h), 0.0) + w
    main = max(hues, key=hues.get)
    return name, rgb, f"{name}: it stands off the {main} near it, in the dark room and on what you wear"


# ---- the neon sign -------------------------------------------------------------------------------

@dataclass
class Tube:
    """One letter's tube, drawn once: its glass (0..1) and the glow round it, in a box of the frame."""
    x0: int
    y0: int
    core: np.ndarray  # the tube itself, 0..1
    halo: np.ndarray  # the light round it, 0..~1.6
    wall: np.ndarray  # its light thrown on the wall, at a quarter size, for the frame's quarter grid
    wx0: int
    wy0: int


def _skeleton(m: np.ndarray) -> np.ndarray:
    import cv2

    sk = cv2.ximgproc.thinning((m > 0.5).astype(np.uint8) * 255, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    return (sk > 0).astype(np.float32)


def _strike(n: int, seed: int) -> list[dict]:
    """How each of `n` tubes strikes on, as offsets from the moment the sign starts: a few quick
    flashes, then steady; one tube catches late and stutters once more after it has settled."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        t = float(rng.uniform(0.0, 0.07)) + 0.012 * i
        flashes = []
        for _ in range(int(rng.integers(1, 4))):
            on = float(rng.uniform(0.03, 0.07))
            flashes.append((t, t + on, float(rng.uniform(0.55, 1.0))))
            t += on + float(rng.uniform(0.04, 0.1))
        out.append({"flashes": flashes, "steady": t, "drop": None})
    if n:
        k = int(rng.integers(0, n))
        s = out[k]["steady"] + float(rng.uniform(0.35, 0.6))
        out[k]["drop"] = (s, s + float(rng.uniform(0.05, 0.09)))
    return out


class NeonSign:
    """The word as neon tubes on the wall behind the head: each letter its own tube, striking on by
    itself, a white-hot core in its colour's glow, its light thrown on the wall and on the edge of
    the person in front of it."""

    def __init__(self, pl: "wb.Placement", colour, size: tuple[int, int], strike: float, off: Optional[float],
                 seed: int):
        import cv2
        from PIL import Image, ImageDraw

        self.W, self.H = size
        self.colour = np.array(colour, np.float32)
        b = pl.block
        self.block, self.rect = b, pl.rect
        self.cap = b.cap
        self.strike, self.off = strike, off
        f = textmod.font(b.face, b.px)
        width = max(2, int(round(0.042 * b.cap)))
        k_tube = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
        pad = int(math.ceil(0.55 * b.cap))
        self.tubes: list[Tube] = []
        Q = 4
        self.qsize = (max(2, self.W // Q), max(2, self.H // Q))
        for k_line, line in enumerate(b.lines):
            base = pl.y + b.line_tops[k_line] + b.cap
            for i, ch in enumerate(line):
                if ch.isspace():
                    continue
                pen = pl.x + b.line_x[k_line] + f.getlength(line[:i])
                l, t, r, bt = f.getbbox(ch, anchor="ls")
                x0, y0 = int(math.floor(pen + l)) - pad, int(math.floor(base + t)) - pad
                w, h = int(math.ceil(r - l)) + 2 * pad, int(math.ceil(bt - t)) + 2 * pad
                img = Image.new("L", (w, h), 0)
                ImageDraw.Draw(img).text((pen - x0, base - y0), ch, font=f, fill=255, anchor="ls")
                m = np.asarray(img, np.float32) / 255.0
                tube = cv2.dilate(_skeleton(m), k_tube)
                core = cv2.GaussianBlur(tube, (0, 0), 0.7)
                halo = (cv2.GaussianBlur(tube, (0, 0), 0.07 * b.cap) * 1.25
                        + cv2.GaussianBlur(tube, (0, 0), 0.2 * b.cap) * 0.7)
                # the light on the wall, at a quarter of the frame: wide and faint
                wpad = int(math.ceil(1.6 * b.cap / Q))
                small = cv2.resize(tube, (max(1, w // Q), max(1, h // Q)), interpolation=cv2.INTER_AREA)
                small = np.pad(small, wpad)
                wall = cv2.GaussianBlur(small, (0, 0), 0.55 * b.cap / Q) * 3.2
                self.tubes.append(Tube(x0, y0, core.astype(np.float32), halo.astype(np.float32),
                                       wall.astype(np.float32), x0 // Q - wpad, y0 // Q - wpad))
        self.plan = _strike(len(self.tubes), seed)
        # Once every tube has settled the sign is one layer, drawn once and only dimmed or brightened
        # as a whole from then on: the tubes one by one only while they strike.
        self.settled = self.strike + (max((p["steady"] for p in self.plan), default=0.0) + 0.9)
        bx0 = min(tb.x0 for tb in self.tubes) if self.tubes else 0
        by0 = min(tb.y0 for tb in self.tubes) if self.tubes else 0
        bx1 = max(tb.x0 + tb.core.shape[1] for tb in self.tubes) if self.tubes else 1
        by1 = max(tb.y0 + tb.core.shape[0] for tb in self.tubes) if self.tubes else 1
        self.box = (bx0, by0)
        self.all_halo = np.zeros((by1 - by0, bx1 - bx0), np.float32)
        self.all_core = np.zeros((by1 - by0, bx1 - bx0), np.float32)
        for tb in self.tubes:
            sl = (slice(tb.y0 - by0, tb.y0 - by0 + tb.core.shape[0]), slice(tb.x0 - bx0, tb.x0 - bx0 + tb.core.shape[1]))
            self.all_halo[sl] += tb.halo
            self.all_core[sl] = np.maximum(self.all_core[sl], tb.core)
        qw, qh = self.qsize
        self.all_wall = np.zeros((qh, qw), np.float32)
        for tb in self.tubes:
            self._add_q(self.all_wall, tb, 1.0)

    def steady(self, t: float) -> Optional[float]:
        """The whole sign's light once every tube has settled (None before, or once it is off): a faint
        flutter, as a lit tube has."""
        if t < self.settled or (self.off is not None and t >= self.off):
            return None
        u = t - self.strike
        return 1.0 + 0.012 * math.sin(2 * math.pi * 7.3 * u) + 0.008 * math.sin(2 * math.pi * 13.1 * u)

    # -- how lit each tube is
    def _level(self, i: int, t: float) -> float:
        if t < self.strike:
            return 0.0
        if self.off is not None and t >= self.off:
            u = t - self.off  # a sign switched off dies with a last flicker
            return 0.0 if u > 0.12 or 0.03 < u < 0.07 else 0.6
        s = self.plan[i]
        u = t - self.strike
        if u >= s["steady"]:
            if s["drop"] and s["drop"][0] <= u < s["drop"][1]:
                return 0.08
            warm = 0.86 + 0.14 * (1.0 - math.exp(-(u - s["steady"]) / 0.3))
            shimmer = 1.0 + 0.012 * math.sin(2 * math.pi * 7.3 * u + i) + 0.008 * math.sin(2 * math.pi * 13.1 * u)
            return warm * shimmer
        for a, b, lvl in s["flashes"]:
            if a <= u < b:
                return lvl
        return 0.0

    def levels(self, t: float, fps: float) -> np.ndarray:
        """Each tube's light over the frame that starts at t: the average of the moment, as a camera
        sees a flash shorter than its frame."""
        steps = [t + (s + 0.5) / 8.0 / fps for s in range(8)]
        return np.array([np.mean([self._level(i, u) for u in steps]) for i in range(len(self.tubes))], np.float32)

    def lit_signal(self, seconds: float, rate: int) -> tuple[np.ndarray, list[int]]:
        """For the sound: how lit the sign is at each sample, and the samples where a tube strikes."""
        n = int(round(seconds * rate))
        t = np.arange(n) / rate
        lit = np.zeros(n, np.float32)
        strikes = []
        step = 1.0 / 400.0  # the flicker is worked out every 2.5 ms
        grid = np.arange(0.0, seconds, step)
        for i in range(len(self.tubes)):
            v = np.array([self._level(i, u) for u in grid], np.float32)
            on = v > 0.3
            rises = np.flatnonzero(np.diff(np.concatenate([[False], on]).astype(np.int8)) > 0)
            strikes += [int(round(grid[r] * rate)) for r in rises]
            lit += np.interp(t, grid, v).astype(np.float32)
        return lit / max(1, len(self.tubes)), sorted(strikes)

    # -- drawing
    def unlit(self, room: np.ndarray, glass: float) -> None:
        """The tubes dark, as glass catches what light there is: barely there."""
        if glass <= 0:
            return
        tint = 0.25 * self.colour + 0.75 * 70.0
        for tb in self.tubes:
            self._lay(room, tb.x0, tb.y0, tb.core * glass, tint, opaque=True)

    def add(self, room: np.ndarray, lv: Optional[np.ndarray], wall_q: np.ndarray, g: Optional[float] = None) -> None:
        """The lit tubes and their glow onto the room (float32, changed in place); the wall light is
        summed into `wall_q` (the frame's quarter grid, one channel) for the caller to lay once.
        With `g` (the settled sign's light), the whole sign at once."""
        hot = 0.18 * self.colour + 0.82 * 255.0
        if g is not None:
            x0, y0 = self.box
            self._lay(room, x0, y0, self.all_halo * g, self.colour * 0.92, opaque=False)
            self._lay(room, x0, y0, np.clip(self.all_core * min(1.0, g * 1.15), 0, 1), hot, opaque=True)
            wall_q += self.all_wall * g
            return
        for tb, b in zip(self.tubes, lv):
            if b <= 0.003:
                continue
            self._lay(room, tb.x0, tb.y0, tb.halo * b, self.colour * 0.92, opaque=False)
            self._lay(room, tb.x0, tb.y0, np.clip(tb.core * min(1.0, b * 1.15), 0, 1), hot, opaque=True)
            self._add_q(wall_q, tb, b)

    def _lay(self, room: np.ndarray, x0: int, y0: int, a: np.ndarray, colour: np.ndarray, opaque: bool) -> None:
        H, W = room.shape[:2]
        h, w = a.shape
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        sub = a[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0][..., None]
        region = room[cy0:cy1, cx0:cx1]
        if opaque:
            region *= 1.0 - np.clip(sub, 0, 1)
            region += np.clip(sub, 0, 1) * colour
        else:
            region += sub * colour

    def _add_q(self, wall_q: np.ndarray, tb: Tube, b: float) -> None:
        H, W = wall_q.shape
        h, w = tb.wall.shape
        x0, y0 = tb.wx0, tb.wy0
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
        if cx1 > cx0 and cy1 > cy0:
            wall_q[cy0:cy1, cx0:cx1] += tb.wall[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] * b


# ---- the LED sign --------------------------------------------------------------------------------

# A 5 by 7 dot font, written for this pack: each glyph is seven rows of five dots, # lit.
_ROWS = {
    "A": (".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "B": ("####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."),
    "C": (".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."),
    "D": ("####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."),
    "E": ("#####", "#....", "#....", "####.", "#....", "#....", "#####"),
    "F": ("#####", "#....", "#....", "####.", "#....", "#....", "#...."),
    "G": (".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".####"),
    "H": ("#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "I": (".###.", "..#..", "..#..", "..#..", "..#..", "..#..", ".###."),
    "J": ("..###", "...#.", "...#.", "...#.", "...#.", "#..#.", ".##.."),
    "K": ("#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"),
    "L": ("#....", "#....", "#....", "#....", "#....", "#....", "#####"),
    "M": ("#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"),
    "N": ("#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"),
    "O": (".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "P": ("####.", "#...#", "#...#", "####.", "#....", "#....", "#...."),
    "Q": (".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"),
    "R": ("####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"),
    "S": (".####", "#....", "#....", ".###.", "....#", "....#", "####."),
    "T": ("#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."),
    "U": ("#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "V": ("#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."),
    "W": ("#...#", "#...#", "#...#", "#.#.#", "#.#.#", "#.#.#", ".#.#."),
    "X": ("#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"),
    "Y": ("#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."),
    "Z": ("#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"),
    "0": (".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."),
    "1": ("..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."),
    "2": (".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"),
    "3": ("#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."),
    "4": ("...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."),
    "5": ("#####", "#....", "####.", "....#", "....#", "#...#", ".###."),
    "6": ("..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."),
    "7": ("#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."),
    "8": (".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."),
    "9": (".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."),
    "!": ("..#..", "..#..", "..#..", "..#..", "..#..", ".....", "..#.."),
    "?": (".###.", "#...#", "....#", "...#.", "..#..", ".....", "..#.."),
    ".": (".....", ".....", ".....", ".....", ".....", ".....", "..#.."),
    ",": (".....", ".....", ".....", ".....", "..#..", "..#..", ".#..."),
    "'": ("..#..", "..#..", ".#...", ".....", ".....", ".....", "....."),
    "-": (".....", ".....", ".....", ".###.", ".....", ".....", "....."),
    ":": (".....", "..#..", ".....", ".....", ".....", "..#..", "....."),
    "&": (".##..", "#..#.", "#.#..", ".#...", "#.#.#", "#..#.", ".##.#"),
    "%": ("##...", "##..#", "...#.", "..#..", ".#...", "#..##", "...##"),
    "+": (".....", "..#..", "..#..", "#####", "..#..", "..#..", "....."),
    "/": (".....", "....#", "...#.", "..#..", ".#...", "#....", "....."),
    "$": ("..#..", ".####", "#.#..", ".###.", "..#.#", "####.", "..#.."),
    "#": (".#.#.", ".#.#.", "#####", ".#.#.", "#####", ".#.#.", ".#.#."),
    "@": (".###.", "#...#", "#.###", "#.#.#", "#.###", "#....", ".###."),
    " ": (".....",) * 7,
}
assert all(len(r) == 7 and all(len(x) == 5 for x in r) for r in _ROWS.values()), "a dot glyph is not 5 by 7"
_GLYPHS = {k: "".join(r) for k, r in _ROWS.items()}


def led_lines(text: str, most: int = 12) -> list[str]:
    """The sign's text in capitals on one or two lines of at most `most` characters, split evenly."""
    words = [w for w in text.upper().split() if w]
    words = ["".join(c for c in w if c in _GLYPHS) for w in words]
    words = [w for w in words if w]
    one = " ".join(words)
    if len(one) <= most or len(words) == 1:
        return [one[: most + 4]]
    best, cost = None, math.inf
    for k in range(1, len(words)):
        a, b = " ".join(words[:k]), " ".join(words[k:])
        c = max(len(a), len(b))
        if c < cost:
            best, cost = [a, b], c
    return best


class LedSign:
    """A dot-matrix board on the wall behind the head: dark, its unlit dots barely there, and the
    word typing itself out in lit dots, a cursor blinking ahead of it."""

    PAD = 2  # dots of board round the text

    def __init__(self, lines: list[str], rect: Rect, pitch: float, colour, size: tuple[int, int],
                 types: list[tuple[int, int, float]], done: float, off: Optional[float]):
        import cv2

        self.W, self.H = size
        self.lines, self.rect, self.p = lines, rect, float(pitch)
        self.colour = np.array(colour, np.float32)
        self.types, self.done, self.off = types, done, off
        cols = max(len(s) for s in lines) * 6 - 1 + 2 * self.PAD
        rows = len(lines) * 8 - 1 + 2 * self.PAD
        self.cols, self.rows = cols, rows
        p = self.p
        bw, bh = int(math.ceil(cols * p)), int(math.ceil(rows * p))
        self.x0, self.y0 = int(round(rect.x0)), int(round(rect.y0))
        pad = int(math.ceil(2.2 * p))
        self.pad = pad
        W2, H2 = bw + 2 * pad, bh + 2 * pad
        yy, xx = np.mgrid[0:H2, 0:W2].astype(np.float32)
        # one dot's light: a round core and a soft glow, stamped where each lit dot is
        r = 0.36 * p

        def dot_field(cells: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
            core = np.zeros((H2, W2), np.float32)
            for (cx, cy) in cells:
                px, py = pad + (cx + 0.5) * p, pad + (cy + 0.5) * p
                x0, x1 = int(max(0, px - 2 * p)), int(min(W2, px + 2 * p + 1))
                y0, y1 = int(max(0, py - 2 * p)), int(min(H2, py + 2 * p + 1))
                d = np.hypot(xx[y0:y1, x0:x1] - px, yy[y0:y1, x0:x1] - py)
                core[y0:y1, x0:x1] = np.maximum(core[y0:y1, x0:x1], np.clip((r - d) / 1.2 + 0.5, 0, 1))
            glow = cv2.GaussianBlur(core, (0, 0), 0.55 * p) * 1.3 + cv2.GaussianBlur(core, (0, 0), 1.6 * p) * 0.6
            return core, glow

        every = [(cx, cy) for cy in range(rows) for cx in range(cols)]
        self.grid, _ = dot_field(every)
        board = np.zeros((H2, W2), np.float32)
        board[pad:pad + bh, pad:pad + bw] = 1.0
        self.board = cv2.GaussianBlur(board, (0, 0), 0.6)
        frame_ring = cv2.GaussianBlur(board, (0, 0), 0.25 * p) * (1 - cv2.GaussianBlur(board, (0, 0), 0.08 * p))
        self.ring = np.clip(frame_ring * 3, 0, 1)
        # each character's lit dots (and the cursor's at each place)
        self.chars: list[tuple[np.ndarray, np.ndarray]] = []
        self.cursor: list[tuple[np.ndarray, np.ndarray]] = []
        self.slots = []
        for li, line in enumerate(lines):
            offset = (max(len(s) for s in lines) - len(line)) * 3
            for ci, ch in enumerate(line):
                gx, gy = self.PAD + offset + ci * 6, self.PAD + li * 8
                bits = _GLYPHS.get(ch, _GLYPHS[" "])
                cells = [(gx + c, gy + r_) for r_ in range(7) for c in range(5) if bits[r_ * 5 + c] == "#"]
                self.chars.append(dot_field(cells))
                self.cursor.append(dot_field([(gx + c, gy + r_) for r_ in range(7) for c in range(5)]))
                self.slots.append((li, ci))

    @property
    def bounds(self) -> Rect:
        return Rect(self.x0 - self.pad, self.y0 - self.pad, self.x0 - self.pad + self.grid.shape[1],
                    self.y0 - self.pad + self.grid.shape[0])

    def typed(self, t: float) -> int:
        return sum(1 for _, _, at in self.types if at <= t)

    def draw(self, room: np.ndarray, t: float, w: float) -> float:
        """The board, its dots and the typed letters onto the room; returns how much of the sign is
        lit (0..1), for its light on the wall and on the person."""
        if w <= 0:
            return 0.0
        x0, y0 = self.x0 - self.pad, self.y0 - self.pad
        H, W = room.shape[:2]
        h, wd = self.grid.shape
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x0 + wd), min(H, y0 + h)
        if cx1 <= cx0 or cy1 <= cy0:
            return 0.0
        sl = (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0))
        region = room[cy0:cy1, cx0:cx1]
        # the board: near black, a faint edge, every dot a little visible
        b = (self.board[sl] * w)[..., None]
        region *= 1.0 - 0.82 * b
        region += (self.ring[sl] * 0.10 * w)[..., None] * np.array([60.0, 60.0, 64.0], np.float32)
        region += (self.grid[sl] * 0.075 * w)[..., None] * (0.5 * self.colour + 0.5 * 90.0)
        off = 1.0
        if self.off is not None and t >= self.off:
            off = max(0.0, 1.0 - (t - self.off) / 0.12)
        n = self.typed(t)
        lit_core = np.zeros(region.shape[:2], np.float32)
        lit_glow = np.zeros(region.shape[:2], np.float32)
        for i in range(min(n, len(self.chars))):
            age = t - self.types[i][2]
            flash = 1.0 + 0.35 * max(0.0, 1.0 - age / 0.08)
            core, glow = self.chars[i]
            lit_core = np.maximum(lit_core, core[sl] * flash)
            lit_glow += glow[sl] * flash
        if n < len(self.chars) or t < self.done + 0.9:  # the cursor, blinking at the next place
            k = min(n, len(self.chars) - 1) if n < len(self.chars) else None
            if k is not None and (n > 0 or t >= self.types[0][2] - 0.6) and (int(t * 5.0) % 2 == 0):
                core, glow = self.cursor[k]
                lit_core = np.maximum(lit_core, core[sl] * 0.9)
                lit_glow += glow[sl] * 0.9
        lit_core *= off * w
        lit_glow *= off * w
        hot = 0.3 * self.colour + 0.7 * 255.0
        region += lit_glow[..., None] * self.colour * 0.75
        c = np.clip(lit_core, 0, 1.3)[..., None]
        region *= 1.0 - np.clip(c, 0, 1)
        region += c * hot
        total = float(lit_core.sum()) / max(1.0, float(sum(cc.sum() for cc, _ in self.chars)))
        return min(1.0, total)


# ---- rays and sparkles, haze ---------------------------------------------------------------------

class Rays:
    """Golden rays from behind the head, turning slowly, and sparkles twinkling in the dark room."""

    def __init__(self, size: tuple[int, int], colour, start: float, seed: int, n_sparkles: int = 34):
        self.W, self.H = size
        self.colour = np.array(colour, np.float32)
        self.start = start
        rng = np.random.default_rng(seed)
        n = 15
        base = np.sort(rng.uniform(0, 2 * np.pi, n))
        self.angles = base
        self.widths = rng.uniform(math.radians(1.6), math.radians(4.8), n)
        self.strength = rng.uniform(0.45, 1.0, n)
        self.freq = rng.uniform(0.12, 0.35, n)
        self.phase = rng.uniform(0, 2 * np.pi, n)
        self.spin = math.radians(2.2) * (1 if rng.random() < 0.5 else -1)
        q = 4
        self.q = q
        qw, qh = max(8, self.W // q), max(8, self.H // q)
        self.yy, self.xx = np.mgrid[0:qh, 0:qw].astype(np.float32)
        self.yy *= q
        self.xx *= q
        # the sparkles: where, how big, how fast they twinkle and drift
        self.sp = {"x": rng.uniform(0.04, 0.96, n_sparkles) * self.W, "y": rng.uniform(0.05, 0.85, n_sparkles) * self.H,
                   "size": rng.uniform(0.006, 0.02, n_sparkles) * self.W, "f": rng.uniform(0.6, 1.6, n_sparkles),
                   "ph": rng.uniform(0, 2 * np.pi, n_sparkles), "rise": rng.uniform(0.006, 0.02, n_sparkles) * self.H}
        self._stars: dict[int, np.ndarray] = {}

    def _star(self, px: int) -> np.ndarray:
        import cv2

        if px not in self._stars:
            s = 2 * px + 1
            yy, xx = np.mgrid[0:s, 0:s].astype(np.float32) - px
            arm = np.exp(-(xx / (0.09 * px + 0.6)) ** 2) * np.exp(-(np.abs(yy) / px) * 2.2)
            arm = np.maximum(arm, arm.T)
            core = np.exp(-(xx * xx + yy * yy) / (2 * (0.18 * px + 0.5) ** 2))
            self._stars[px] = cv2.GaussianBlur(np.maximum(arm, core), (0, 0), 0.6)
        return self._stars[px]

    def field(self, t: float, origin: tuple[float, float]) -> np.ndarray:
        """The rays' light at a quarter of the frame (0..~1)."""
        u = t - self.start
        grow = _ease(min(1.0, max(0.0, u / 0.45)))
        if grow <= 0:
            return np.zeros(self.xx.shape, np.float32)
        ox, oy = origin
        dx, dy = self.xx - ox, self.yy - oy
        ang = np.arctan2(dy, dx)
        r = np.hypot(dx, dy)
        out = np.zeros(self.xx.shape, np.float32)
        reach = grow * 1.15 * max(self.W, self.H)
        fade = (1.0 - _smoothstep(0.0, reach, r)) * _smoothstep(0.0, 0.08 * self.W, r)
        for a0, wd, s, f, ph in zip(self.angles, self.widths, self.strength, self.freq, self.phase):
            a = a0 + self.spin * u
            d = (ang - a + np.pi) % (2 * np.pi) - np.pi
            breathe = 0.7 + 0.3 * math.sin(2 * math.pi * f * u + ph)
            out += (s * breathe) * np.exp(-0.5 * (d / wd) ** 2)
        glow = np.exp(-(r / (0.22 * self.W)) ** 2) * 0.45
        return ((out * 0.3 * fade + glow * grow)).astype(np.float32)

    def sparkles(self, room: np.ndarray, t: float, w: float) -> None:
        u = t - self.start
        if u <= 0:
            return
        H, W = room.shape[:2]
        for i in range(len(self.sp["x"])):
            tw = math.sin(2 * math.pi * self.sp["f"][i] * u + self.sp["ph"][i])
            b = max(0.0, tw) ** 3 * w * min(1.0, u / 0.6)
            if b < 0.02:
                continue
            px = max(2, int(self.sp["size"][i]))
            star = self._star(px)
            x = int(self.sp["x"][i])
            y = int((self.sp["y"][i] - self.sp["rise"][i] * u) % H)
            x0, y0 = x - px, y - px
            cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x0 + star.shape[1]), min(H, y0 + star.shape[0])
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            a = star[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0][..., None] * b
            room[cy0:cy1, cx0:cx1] += a * (0.55 * self.colour + 0.45 * 255.0)


class Haze:
    """A beam of light falling on the person from above through haze: a soft-edged cone, brightest
    where it comes in, with fine streaks of haze drifting slowly along it, and dust turning in it."""

    def __init__(self, size: tuple[int, int], colour, start: float, seed: int):
        import cv2

        self.W, self.H = size
        self.colour = np.array(colour, np.float32)
        self.start = start
        rng = np.random.default_rng(seed)
        q = 4
        self.q = q
        qw, qh = max(8, self.W // q), max(8, self.H // q)
        self.qw, self.qh = qw, qh
        self.yy, self.xx = np.mgrid[0:qh, 0:qw].astype(np.float32)
        self.yy *= q
        self.xx *= q
        # haze: fine noise stretched along the beam (streaks), two layers drifting at their own pace,
        # each taller than the frame so it can drift down through it
        self.layers = []
        for sx, sy, amp in ((0.012, 0.09, 0.6), (0.025, 0.05, 0.4)):
            n = rng.standard_normal((int(qh * 2.2) + 4, qw + 4)).astype(np.float32)
            n = cv2.GaussianBlur(n, (0, 0), sigmaX=max(0.6, sx * qw), sigmaY=max(0.6, sy * qh))
            n = (n - n.mean()) / (n.std() + 1e-6)
            self.layers.append((n, amp))
        self.fall = [rng.uniform(0.03, 0.05) * self.H, rng.uniform(0.06, 0.09) * self.H]  # pixels a second
        k = 40
        self.dust = {"x": rng.uniform(0, 1, k), "y": rng.uniform(0, 1, k), "vx": rng.uniform(-0.008, 0.008, k),
                     "vy": rng.uniform(-0.006, 0.01, k), "r": rng.uniform(0.8, 2.4, k), "ph": rng.uniform(0, 6.3, k)}

    def beam(self, head: Optional[Rect]) -> np.ndarray:
        cx = head.cx if head is not None else self.W / 2
        hy = head.cy if head is not None else 0.4 * self.H
        src_y = -0.35 * self.H
        top_half = 0.04 * self.W  # where it comes into the frame's top, more or less
        at_head = (0.95 * head.w if head is not None else 0.5 * self.W)
        grow = np.clip((self.yy - src_y) / max(1.0, hy - src_y), 0.0, 2.5)
        half = top_half + (at_head - top_half) * grow
        d = np.abs(self.xx - cx) / np.maximum(half, 1.0)
        cone = 1.0 - _smoothstep(0.55, 1.05, d)
        along = 1.0 - 0.6 * _smoothstep(0.0, 0.9 * self.H, self.yy)
        return (cone * along).astype(np.float32)

    def field(self, t: float, head: Optional[Rect]) -> np.ndarray:
        u = t - self.start
        grow = _smooth(u / 0.6)
        if grow <= 0:
            return np.zeros((self.qh, self.qw), np.float32)
        b = self.beam(head)
        tex = np.zeros_like(b)
        for (n, amp), fall in zip(self.layers, self.fall):
            span = n.shape[0] - self.qh
            oy = span - int(round(fall * u / self.q)) % span  # drifting down the beam
            tex += amp * n[oy:oy + self.qh, 2:2 + self.qw]
        haze = np.clip(0.8 + 0.2 * tex, 0.4, 1.3)
        return (b * haze * grow).astype(np.float32)

    def dust_on(self, room: np.ndarray, beam_q: np.ndarray, t: float, w: float) -> None:
        u = t - self.start
        if u <= 0:
            return
        H, W = room.shape[:2]
        d = self.dust
        for i in range(len(d["x"])):
            x = ((d["x"][i] + d["vx"][i] * u + 0.004 * math.sin(0.7 * u + d["ph"][i])) % 1.0) * W
            y = ((d["y"][i] + d["vy"][i] * u) % 1.0) * H
            xi, yi = int(x), int(y)
            if not (0 <= xi < W and 0 <= yi < H):
                continue
            b = float(beam_q[min(beam_q.shape[0] - 1, yi // self.q), min(beam_q.shape[1] - 1, xi // self.q)])
            b *= w * (0.6 + 0.4 * math.sin(3.1 * u + d["ph"][i]))
            if b < 0.05:
                continue
            r = d["r"][i] * W / 1080.0
            rr = int(math.ceil(3 * r))
            x0, y0 = xi - rr, yi - rr
            yy, xx = np.mgrid[-rr:rr + 1, -rr:rr + 1].astype(np.float32)
            spot = np.exp(-(xx * xx + yy * yy) / (2 * r * r)) * b * 0.9
            cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x0 + spot.shape[1]), min(H, y0 + spot.shape[0])
            if cx1 > cx0 and cy1 > cy0:
                room[cy0:cy1, cx0:cx1] += spot[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0][..., None] * self.colour


def _ease(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return 1.0 - (1.0 - u) ** 3


# ---- placing the neon sign --------------------------------------------------------------------------

SIGN_WIDTH = 0.84  # a sign, not a wall of letters: at most this share of the safe zone's width
SIGN_HIDDEN = 0.06  # the most of its letters the head may hide, on any frame but the worst tenth


def place_neon(arrangements: list[tuple[str, ...]], face: str, store: "wb.MaskStore", size: tuple[int, int],
               heads: list[Optional[Rect]]) -> "wb.Placement":
    """Where the sign hangs: as large as fits (one to three lines), over the head, as low on the wall
    as it can go while the head hides almost none of it on any frame. A sign is read, not half
    hidden: unlike a giant word, its foot does not sink behind the head (its tubes would break
    into pieces); it sits just above, the head brushing its lowest letters at most."""
    W, H = size
    zone = SafeZone(W, H).rect
    tops = store.crowns()
    known = [h for h in heads if h is not None]
    head_cx = float(np.median([h.cx for h in known])) if known else zone.cx
    best = None
    for lines in arrangements:
        sizes = [px for px in wb._sizes(lines, face, zone, H)
                 if max(textmod.advance(ln, face, px) for ln in lines) <= SIGN_WIDTH * zone.w]
        for px in sizes:
            b = wb.make_block(tuple(lines), face, px)
            found = None
            for cx in (head_cx, zone.cx):
                x = float(np.clip(cx - b.w / 2, zone.x0, zone.x1 - b.w))
                ys = np.arange(zone.y1 - b.h, zone.y0 - 0.01, -2.0)
                if not len(ys):
                    continue
                med, p90, lp = wb._scan(b, tops, x, ys) if len(tops) else (np.zeros(len(ys)),) * 3
                ok = np.flatnonzero((p90 <= SIGN_HIDDEN) & (lp <= 0.5))
                if len(ok):
                    i = int(ok[0])  # the lowest that keeps every letter whole
                    found = wb.Placement(b, x, float(ys[i]), float(med[i]), float(p90[i]), float(lp[i]),
                                         "just above your head" if med[i] < wb.TOUCH else "behind your head, its "
                                         "lowest letters brushed by it")
                    break
            if found is not None:
                rank = (round(b.cap, 1), -len(lines))
                if best is None or rank > best[0]:
                    best = (rank, found)
                break
    if best is None:  # no room above the head at any size: the smallest, at the top, said plainly
        lines = arrangements[-1]
        b = wb.make_block(tuple(lines), face, wb._sizes(lines, face, zone, H)[-1])
        x = float(np.clip(head_cx - b.w / 2, zone.x0, zone.x1 - b.w))
        med, p90, lp = wb._scan(b, tops, x, np.array([zone.y0])) if len(tops) else (np.zeros(1),) * 3
        return wb.Placement(b, x, zone.y0, float(med[0]), float(p90[0]), float(lp[0]), "at the top of the frame",
                            "the shot has no room above your head: the sign sits at the top, and your head may "
                            "cover some of it; a shot with more room above you suits a sign better")
    return best[1]


# ---- placing the LED board -------------------------------------------------------------------------

def place_board(lines: list[str], store: "wb.MaskStore", size: tuple[int, int],
                heads: list[Optional[Rect]]) -> tuple[Rect, float, str, float]:
    """Where the board goes and its dot pitch: as wide as the safe zone allows (and never taller than
    a fifth of the frame), centred over the head, as low as it can sit with the head hiding only its
    border, never a dot of a letter (a hidden foot turns an L into an I). Returns (rect, pitch,
    where, hidden)."""
    W, H = size
    zone = SafeZone(W, H).rect
    cols = max(len(s) for s in lines) * 6 - 1 + 2 * LedSign.PAD
    rows = len(lines) * 8 - 1 + 2 * LedSign.PAD
    tops = store.crowns()  # (frames, W/Q): the first row each column of the person reaches
    Q = wb.Q
    known = [h for h in heads if h is not None]
    cx = float(np.median([h.cx for h in known])) if known else zone.cx
    p = min(0.9 * zone.w / cols, 0.2 * H / rows)
    best = None
    while p >= 6.0:
        bw, bh = cols * p, rows * p
        x0 = float(np.clip(cx - bw / 2, zone.x0, zone.x1 - bw))
        text_top, text_bot = LedSign.PAD * p, bh - LedSign.PAD * p
        c0, c1 = int(max(0, x0 // Q)), int(min(tops.shape[1] if len(tops) else 0, (x0 + bw) // Q + 1))
        for y in np.arange(zone.y1 - bh, zone.y0 - 0.5, -2.0):
            if len(tops) and c1 > c0:
                crown = tops[:, c0:c1].astype(np.float32) * Q  # output rows
                hidden_text = np.clip((y + text_bot) - np.maximum(crown, y + text_top), 0, text_bot - text_top)
                share = hidden_text.mean(axis=1) / max(1.0, text_bot - text_top)  # per frame
                med, p90 = float(np.median(share)), float(np.percentile(share, 90))
                body = np.clip((y + bh) - crown, 0, bh).mean(axis=1) / bh
                touch = float(np.median(body))
            else:
                med = p90 = touch = 0.0
            if p90 <= 0.012 and med <= 0.004:  # the letters themselves are never hidden: an L would read I
                where = ("behind your head" if touch >= 0.04 else "just above your head")
                cand = (Rect(x0, y, x0 + bw, y + bh), p, where, p90)
                best = cand
                break
        if best is not None:
            break
        p *= 0.92
    if best is None:
        p = max(6.0, p)
        bw, bh = cols * p, rows * p
        x0 = float(np.clip(cx - bw / 2, zone.x0, zone.x1 - bw))
        best = (Rect(x0, zone.y0, x0 + bw, zone.y0 + bh), p, "at the top: the shot has no room above your head", 1.0)
    return best


# ---- the edge, decided by colour where the cut-out is unsure ----------------------------------------

def drop_room(frame: np.ndarray, a: np.ndarray, sigma: float) -> np.ndarray:
    """`a` with the old room taken back out of the person's edge where the cut-out half-took it for
    the person: each unsure edge pixel is placed between the person's colour just inside it and the
    room's just outside it, and may only lose alpha, never gain it. A strap or a white wall the
    cut-out let into the edge (clip F) goes dark with the room; a real strand of hair, the person's
    colour, keeps what it had. Where the person and the room are too close in colour to tell
    apart, nothing changes. The colours either side are worked out at a quarter of the size, and
    only the edge's own pixels one by one, as the core's matte does."""
    edge = (a > 0.03) & (a < 0.97)
    ys, xs = np.nonzero(edge)
    if not len(ys):
        return a
    H, W = a.shape
    pad = int(3 * sigma)
    y0, y1 = max(0, int(ys.min()) - pad), min(H, int(ys.max()) + pad + 1)
    x0, x1 = max(0, int(xs.min()) - pad), min(W, int(xs.max()) + pad + 1)
    y0, x0 = y0 - y0 % 4, x0 - x0 % 4
    I = frame[y0:y1, x0:x1].astype(np.float32)
    room, body = matte.colours_near_edge(I, a[y0:y1, x0:x1], sigma)
    ly, lx = ys - y0, xs - x0
    sy, sx = room.shape[0] / max(1, y1 - y0), room.shape[1] / max(1, x1 - x0)
    B = matte.sample(room, ly, lx, sy, sx)
    F = matte.sample(body, ly, lx, sy, sx)
    px = frame[ys, xs].astype(np.float32)
    d = F - B
    apart = (d * d).sum(axis=1)
    side = np.clip(((px - B) * d).sum(axis=1) / np.maximum(apart, 1.0), 0.0, 1.0)
    trust = np.clip((np.sqrt(apart) - 25.0) / 50.0, 0.0, 1.0)
    old = a[ys, xs]
    out = a.copy()
    out[ys, xs] = np.minimum(old, trust * side + (1.0 - trust) * old)
    return out


# ---- drawing -----------------------------------------------------------------------------------------

class Painter:
    """Draws each output frame with the lights down: the room, the sign or the light in it, the person
    in front with a clean edge, the sign's light on their edge, and the captions over it all."""

    def __init__(self, clip, timeline, framing, light: Light, heads: list[Optional[Rect]], timing: Timing,
                 neon: Optional[NeonSign] = None, led: Optional[LedSign] = None, rays: Optional[Rays] = None,
                 haze: Optional[Haze] = None, glow_colour=None, captions=None):
        from ..core import person as personmod

        self.clip, self.timeline, self.framing = clip, timeline, framing
        self.light, self.heads, self.timing = light, heads, timing
        self.neon, self.led, self.rays, self.haze = neon, led, rays, haze
        self.glow_colour = None if glow_colour is None else np.array(glow_colour, np.float32)
        self.captions = captions
        self.by_colour = True  # see drop_room
        self.W, self.H = framing.out_w, framing.out_h
        self.fps = clip.info.rate
        self.masks = clip.person()
        self._sx = self.masks.width / clip.info.width
        self.steady = None
        self.models = 1
        if type(self.masks) is personmod.PersonMasks:  # a clip read from its file: both looks, steadied
            try:
                self.steady = matte.Steady([self.masks, clip.person("person-parts")], clip.info)
                self.models = 2
            except Exception:  # the detailed model is optional: one look, still steadied
                self.steady = matte.Steady(self.masks, clip.info)
        self._iter, self._k, self._last = None, -1, None
        self.seen: dict = {"halo": [], "edge": [], "source": [], "rim": 0.0}
        self._prev = None

    def _mask(self, k: int) -> np.ndarray:
        if self.steady is not None:
            return self.steady.get(k)
        if self._iter is None or k < self._k:
            self._iter, self._k = iter(self.masks.frames()), -1
        while self._k < k:
            nxt = next(self._iter, None)
            if nxt is None:
                break
            self._last, self._k = nxt, self._k + 1
        return self._last

    def person_alpha(self, frame: np.ndarray, k: int) -> np.ndarray:
        m = self._mask(k)
        head = self.clip.face.head(k)
        if head is not None:
            s = self._sx
            m = matte.main_only(m, (head.x0 * s, head.y0 * s, head.x1 * s, head.y1 * s))
        crop = self.framing.window(k) if self.framing.reframed else None
        a = matte.alpha(self.masks, m, (self.W, self.H), guide=frame, crop=crop, reach=0.012 * self.W)
        return drop_room(frame, a, max(4.0, 0.012 * self.W)) if self.by_colour else a

    def face_box(self, k: int) -> Optional[Rect]:
        b = self.clip.face.box(k)
        return self.framing.to_output(b, k) if b is not None else None

    def compose(self, frame: np.ndarray, j: int, t: float, w: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The frame with the lights all the way down (w only fades what fades with the room), the
        person's alpha, and the room as it is behind them (for measuring the edge)."""
        import cv2

        k = self.timeline.source_frame(j, self.fps, self.clip.info.frames)
        a = self.person_alpha(frame, k)
        F = frame.astype(np.float32)
        head = self.heads[j] if j < len(self.heads) else None
        a_small = cv2.resize(a, self.light.q, interpolation=cv2.INTER_AREA)
        room_l, person_l = self.light.maps(a_small, head)
        room = self.light.room(F, room_l)
        glow_q = None
        if self.neon is not None:
            glow_q = np.zeros((self.neon.qsize[1], self.neon.qsize[0]), np.float32)
            g = self.neon.steady(t)
            if g is not None:
                self.neon.add(room, None, glow_q, g)
            else:
                lv = self.neon.levels(t, self.fps)
                if float(lv.min()) < 0.5:
                    self.neon.unlit(room, 0.035 * w * w)
                self.neon.add(room, lv, glow_q)
            self._wall(room, glow_q, self.neon.colour * 0.55)
        if self.led is not None:
            lit = self.led.draw(room, t, w)
            if lit > 0:
                glow_q = self._board_glow(lit)
                self._wall(room, glow_q, self.led.colour * 0.3)
        if self.rays is not None:
            origin = (head.cx, head.y0 + 0.32 * head.h) if head is not None else (self.W / 2, 0.3 * self.H)
            f = self.rays.field(t, origin)
            self._wall(room, f, self.rays.colour * 0.75 * w)
            self.rays.sparkles(room, t, w)
            glow_q = f * 0.35
        if self.haze is not None:
            f = self.haze.field(t, head)
            self._wall(room, f, self.haze.colour * 0.3 * w)
            self.haze.dust_on(room, f, t, w)
        P = self.light.person(F, person_l)
        out = matte.over(P, a, room)
        if glow_q is not None:
            self._rim(out, a, glow_q, k)
        return out, a, room

    def _board_glow(self, lit: float) -> np.ndarray:
        import cv2

        q = 4
        qw, qh = max(2, self.W // q), max(2, self.H // q)
        g = np.zeros((qh, qw), np.float32)
        r = self.led.rect
        x0, y0, x1, y1 = int(r.x0 // q), int(r.y0 // q), int(math.ceil(r.x1 / q)), int(math.ceil(r.y1 / q))
        g[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 1.0
        return cv2.GaussianBlur(g, (0, 0), max(2.0, 0.6 * (y1 - y0))) * 0.8 * lit

    def _wall(self, room: np.ndarray, field_q: np.ndarray, colour: np.ndarray) -> None:
        """A light at a quarter of the frame added onto the room where it is not faint."""
        import cv2

        ys, xs = np.nonzero(field_q > 0.004)
        if not len(ys):
            return
        q = self.W / field_q.shape[1]
        y0, y1 = int(ys.min()), int(ys.max()) + 2
        x0, x1 = int(xs.min()), int(xs.max()) + 2
        sub = field_q[y0:y1, x0:x1]
        Y0, Y1 = int(y0 * q), min(self.H, int(y1 * q))
        X0, X1 = int(x0 * q), min(self.W, int(x1 * q))
        big = cv2.resize(sub, (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_LINEAR)
        room[Y0:Y1, X0:X1] += big[..., None] * colour

    def _rim(self, out: np.ndarray, a: np.ndarray, glow_q: np.ndarray, k: int) -> None:
        """The sign's light catching the edge of the person in front of it (hair, shoulders), never
        the face: a thin band just inside their outline, as strong as the sign's light there."""
        import cv2

        qh, qw = glow_q.shape
        aq = cv2.resize(a, (qw, qh), interpolation=cv2.INTER_AREA)
        inner = np.clip(aq * (1.0 - cv2.GaussianBlur(aq, (0, 0), 1.4)) * 3.0, 0.0, 1.0)
        rim = inner * np.clip(glow_q, 0.0, 1.0)
        face = self.face_box(k)
        if face is not None:
            s = qw / self.W
            yy, xx = np.mgrid[0:qh, 0:qw].astype(np.float32)
            fx0, fx1 = (face.x0 - 0.12 * face.w) * s, (face.x1 + 0.12 * face.w) * s
            fy0, fy1 = (face.y0 - 0.05 * face.h) * s, (face.y1 + 0.15 * face.h) * s
            inside = (_smoothstep(fx0 - 3, fx0 + 3, xx) * (1 - _smoothstep(fx1 - 3, fx1 + 3, xx))
                      * _smoothstep(fy0 - 3, fy0 + 3, yy) * (1 - _smoothstep(fy1 - 3, fy1 + 3, yy)))
            rim *= 1.0 - inside
        ys, xs = np.nonzero(rim > 0.01)
        if not len(ys):
            return
        q = self.W / qw
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 2, int(xs.min()), int(xs.max()) + 2
        Y0, Y1, X0, X1 = int(y0 * q), min(self.H, int(y1 * q)), int(x0 * q), min(self.W, int(x1 * q))
        big = cv2.resize(rim[y0:y1, x0:x1], (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_LINEAR)
        colour = self.glow_colour if self.glow_colour is not None else np.array([255, 255, 255], np.float32)
        region = out[Y0:Y1, X0:X1]
        region += (0.42 * big)[..., None] * (colour - 0.5 * region)
        self.seen["rim"] = max(self.seen["rim"], float(big.max()))

    def draw(self, frame: np.ndarray, j: int, t: float) -> None:
        import cv2

        w = weight(t, self.timing)
        if w > 0.0005:
            out, a, room = self.compose(frame, j, t, w)
            if j % 8 == 0 and w >= 0.999:
                self._measure_halo(out, a, room)
            if w < 0.9995:
                F = frame.astype(np.float32)
                out = F + w * (out - F)
            done = cv2.convertScaleAbs(out)
            if w >= 0.999:
                self._measure_flicker(frame, a, j)
            frame[:] = done
        if self.captions is not None:
            self.captions.draw(frame, j, t)

    # -- the edge, measured on the render itself
    def _measure_halo(self, out: np.ndarray, a: np.ndarray, room: np.ndarray) -> None:
        """How much brighter the room is just outside the person's edge (2 to 10 pixels at 1080
        across) than the dark room drawn there: a rim of the lit room left round them shows here."""
        import cv2

        s = self.W / 1080.0
        hard = (a > 0.5).astype(np.uint8)
        near = cv2.dilate(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(21 * s) | 1, int(21 * s) | 1)))
        skin = cv2.dilate(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(5 * s) | 1, int(5 * s) | 1)))
        band = (near > 0) & (skin == 0)
        if band.sum() < 200:
            return
        lum = np.array([0.2126, 0.7152, 0.0722], np.float32)
        o = out[band] @ lum
        r = room[band] @ lum
        self.seen["halo"].append((float(np.mean(o - r)), float(np.percentile(o - r, 95)), float(np.mean(r))))

    def _measure_flicker(self, src: np.ndarray, a: np.ndarray, j: int) -> None:
        """How far the person's edge moves from one frame to the next where the picture itself did not
        change (pixels at the output's size): the cut-out's own wobble, which the dark room behind
        would show as a shimmer. Measured on pairs of frames, at half size."""
        import cv2

        if j % 4 not in (0, 1):
            return
        small = (max(8, self.W // 2), max(8, self.H // 2))
        g_src = cv2.cvtColor(cv2.resize(src, small, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32)
        a_s = cv2.resize(a, small, interpolation=cv2.INTER_AREA)
        if self._prev is not None and self._prev[0] == j - 1:
            _, ps, pa = self._prev
            edge = ((a_s > 0.05) & (a_s < 0.95)) | ((pa > 0.05) & (pa < 0.95))
            still = edge & (cv2.GaussianBlur(np.abs(g_src - ps), (0, 0), 2) < 3.0)
            if still.sum() > 100:
                gy, gx = np.gradient(0.5 * (a_s + pa))
                grad = np.hypot(gx, gy)[still]
                moved = np.abs(a_s - pa)[still]
                self.seen["edge"].append(2.0 * float(moved.sum()) / max(1e-6, float(grad.sum())))
                self.seen["source"].append(float(still.sum()))
        self._prev = (j, g_src, a_s)


# ---- deciding ----------------------------------------------------------------------------------------

def _sign_text(clip, opts: dict, seconds: float) -> tuple[list[dict], dict]:
    """The sign's words and when each is said: the word words-behind would choose (its pick, as it
    is), or the words given."""
    when = opts["at"] if opts["at"] not in ("start",) else "auto"
    spec, _, _, info = wb.decide(clip, {"text": opts["word"], "when": when, "hold": "end"}, seconds)
    return spec, info


def _near_colours(clip, tl, fr, store: "wb.MaskStore", rect: Rect, light: Light, frames: list[int]) -> list[np.ndarray]:
    """The colours round where the sign goes, with the lights down (the dark room), and what the person
    wears just under it (a cap, a collar; their skin is left out, being always warm): what the sign's
    colour must stand off."""
    import cv2

    from ..core import video

    W, H = fr.out_w, fr.out_h
    out = []
    pad = 0.06 * W
    x0, y0 = int(max(0, rect.x0 - pad)), int(max(0, rect.y0 - pad))
    x1, y1 = int(min(W, rect.x1 + pad)), int(min(H, rect.y1 + 0.5 * rect.h))
    have = [j for j in frames if j in store.by_frame]
    for j in have[:: max(1, len(have) // 4)][:4]:
        k = tl.source_frame(j, clip.info.rate, clip.info.frames)
        f = fr.apply(video.read_frame(clip.info, k), k).astype(np.float32)
        m = cv2.resize(store.by_frame[j], (W, H), interpolation=cv2.INTER_LINEAR)
        region = f[y0:y1, x0:x1]
        mm = m[y0:y1, x0:x1]
        y = region @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        dark_room = (y[..., None] + light.keep * (region - y[..., None])) * light.far
        out.append(dark_room[mm < 0.05].reshape(-1, 3) * 2.5)  # the room's colour, brightened to judge its hue
        worn = region[mm > 0.9].reshape(-1, 3)
        if len(worn):  # what the person wears near it (a cap, a collar), not their skin: skin is always warm
            from .grade import oklab, skin_weight, to_linear

            worn = worn[:: max(1, len(worn) // 3000)]
            keep = skin_weight(oklab(to_linear(worn.astype(np.float64) / 255.0))) < 0.35
            out.append(worn[keep])
    return out


def _skin(clip, tl, fr, painter: Painter, times: list[float]) -> str:
    """The face's skin before and after, on frames with the lights down: hue, colour and lightness."""
    from ..core import video
    from .grade import _describe, _skin_pixels, oklab, to_linear

    before, after = [], []
    fps = clip.info.rate
    for t in times:
        j = int(round(t * fps))
        k = tl.source_frame(j, fps, clip.info.frames)
        box = clip.face.box(k)
        if box is None:
            continue
        f = fr.apply(video.read_frame(clip.info, k), k)
        out = np.ascontiguousarray(f.copy())
        w = weight(j / fps, painter.timing)
        if w <= 0:
            continue
        o, _, _ = painter.compose(out, j, j / fps, w)
        b = fr.to_output(box, k)
        px0 = _skin_pixels(f, b)
        px1 = _skin_pixels(np.clip(o + 0.5, 0, 255).astype(np.uint8), b)
        if px0 is None or px1 is None:
            continue
        before.append(oklab(to_linear(px0)))
        after.append(oklab(to_linear(px1)))
    if not before:
        return "no face found to check the skin on"
    b, a = _describe(np.concatenate(before)), _describe(np.concatenate(after))
    turn = (a["hue"] - b["hue"] + 180) % 360 - 180
    ratio = a["saturation"] / max(b["saturation"], 1e-6)
    verdict = "as filmed" if abs(turn) < 3 and 0.93 < ratio < 1.07 and abs(a["lightness"] - b["lightness"]) < 0.03 \
        else "changed"
    return (f"{verdict}: hue {b['hue']}° → {a['hue']}°, saturation ×{ratio:.2f}, lightness {b['lightness']} → "
            f"{a['lightness']} ({len(before)} frames with the lights down)")


def _face_check(painter: Painter, clip, tl, fr, n_out: int, count: int = 12) -> tuple[int, int, float]:
    """On frames with the lights down, the share of the face (its box, a little inside it) where the
    dark room would show: the cut-out missing part of the face. Frames over 2% count."""
    from ..core import video

    fps = clip.info.rate
    js = [j for j in np.linspace(0, n_out - 1, count * 3).astype(int) if weight(j / fps, painter.timing) > 0.5]
    js = js[:: max(1, len(js) // count)][:count]
    bad, n, worst = 0, 0, 0.0
    for j in js:
        k = tl.source_frame(int(j), fps, clip.info.frames)
        box = clip.face.box(k)
        if box is None:
            continue
        f = fr.apply(video.read_frame(clip.info, k), k)
        a = painter.person_alpha(f, k)
        b = fr.to_output(box, k)
        x0, x1 = int(max(0, b.x0 + 0.15 * b.w)), int(min(painter.W, b.x1 - 0.15 * b.w))
        y0, y1 = int(max(0, b.y0 + 0.1 * b.h)), int(min(painter.H, b.y1 - 0.1 * b.h))
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        share = float((a[y0:y1, x0:x1] < 0.5).mean())
        worst = max(worst, share)
        bad += share > 0.02
        n += 1
    return int(round(bad * n_out / max(1, n))) if n else 0, n, worst


# ---- the sound -----------------------------------------------------------------------------------------

class Sound:
    """The switch, and the look's own sound, laid under the clip's: each set against the voice's
    loudness, the mix held under the voice's own peak."""

    def __init__(self, clip, timing: Timing, look: str, seconds: float, neon: Optional[NeonSign],
                 led: Optional[LedSign], rays_at: Optional[float]):
        self.clip, self.timing, self.look, self.seconds = clip, timing, look, seconds
        self.neon, self.led, self.rays_at = neon, led, rays_at
        self.voice = soundmod.integrated(soundmod.blocks(clip.sound)) if clip.info.has_audio else soundmod.SILENCE_DB
        self.ref = self.voice if self.voice > -70 else -23.0
        self.laid: list[str] = []
        self.limited_db = 0.0

    def mix(self, voice: np.ndarray, rate: int) -> np.ndarray:
        from ..core import loudness

        stem = np.zeros_like(voice)
        laid = []
        x, anchor = synth.clunk()
        soundmod.lay(stem, synth.at_level(x, self.ref - 5.0), self.timing.down, rate, anchor=anchor)
        laid.append(f"the switch at {self.timing.down:.2f} s, 5 dB under your voice")
        if self.timing.up is not None:
            soundmod.lay(stem, synth.at_level(x, self.ref - 9.0), self.timing.up, rate, anchor=anchor)
            laid.append(f"the switch again at {self.timing.up:.2f} s")
        if self.neon is not None:
            lit, strikes = self.neon.lit_signal(len(voice) / rate, rate)
            hum, crackle = synth.tube(lit, strikes)
            steady = hum[np.abs(hum) > 0]
            if len(steady):
                stem += synth.at_level(np.stack([hum, hum], 1), self.ref - 25.0)
            if strikes:
                stem += synth.at_level(np.stack([crackle, crackle], 1), self.ref - 8.0)
            laid.append(f"the tubes striking on ({len(strikes)} crackles) and their hum, 25 dB under your voice")
        if self.led is not None:
            tk, ta = synth.tick()
            tk = synth.at_level(tk, self.ref - 17.0)
            for _, _, at in self.led.types:
                soundmod.lay(stem, tk, at, rate, anchor=ta)
            laid.append(f"a tick for each of the {len(self.led.types)} letters")
        if self.rays_at is not None:
            sh, sa = synth.shimmer()
            soundmod.lay(stem, synth.at_level(sh, self.ref - 11.0), self.rays_at + sa, rate, anchor=sa)
            laid.append(f"a shimmer as the light arrives at {self.rays_at:.2f} s")
        out = voice + stem
        if stem.any():
            ceiling = max(-1.0, loudness.true_peak_db(voice) + 0.05)
            if loudness.true_peak_db(out) > ceiling:
                out, self.limited_db = loudness.limit(out, ceiling)
        self.laid = laid
        return out.astype(np.float32)


# ---- the effect ----------------------------------------------------------------------------------------

def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from ..core import framing as framing_mod
    from ..core.timeline import Timeline
    from . import Job, PlanError, layers

    if plan is not None:  # the plan is the record: its own look and lights come back with it
        kept = plan.get("options") or {}
        opts = dict(opts, **{k: kept[k] for k in ("look", "dark", "frame", "sound", "captions") if k in kept})
        if opts["look"] not in LOOKS:
            raise PlanError("the plan names a look this effect does not have")
    info = clip.info
    fps = info.rate
    fr = framing_mod.plan(info.width, info.height, fps, info.frames, clip.face, shape=opts["frame"])
    tl = Timeline.whole(info.frames / fps)
    W, H = fr.out_w, fr.out_h
    seconds = tl.duration
    n_out = tl.output_frames(fps)
    look = opts["look"]
    warnings: list[str] = []

    # -- the sign's words, and when the lights go down
    spec, sign_info = None, {}
    if look in ("neon", "led"):
        if plan is not None and plan.get("sign"):
            sg = plan["sign"]
            try:
                spec = [{"text": str(p["text"]), "land": float(p["land"])} for p in sg["pieces"]]
            except (KeyError, TypeError, ValueError):
                raise PlanError("the plan's sign needs its pieces, each with text and land") from None
            sign_info = {"word": sg.get("text"), "said_at": sg.get("said_at")}
        else:
            try:
                spec, sign_info = _sign_text(clip, opts, seconds)
            except ValueError as e:  # words-behind's pick speaks of its own option, text=; ours is word=
                msg = str(e).replace("text=", "word=").replace("give one: word", "give the sign's word: word")
                raise O.OptionError(msg if "word=" in msg else f"{msg}; or give the sign's word: word=\"YOUR WORD\"") from None
        if sign_info.get("warn"):
            warnings.append(sign_info["warn"])
    elif opts["word"] != "auto":
        warnings.append(f"word is for the neon and LED signs; look={look} shows no word, so “{opts['word']}” is not used")
    land = min(p["land"] for p in spec) if spec else None
    if plan is not None and plan.get("lights"):
        try:
            down = float(plan["lights"]["down"])
            up = plan["lights"].get("up")
            timing = Timing(down, None if up is None else float(up), "as the plan says")
        except (KeyError, TypeError, ValueError):
            raise PlanError("the plan's lights need down (and up, or null)") from None
        if not 0 <= timing.down < seconds:
            raise PlanError(f"the plan's lights go down at {timing.down} s, outside the clip (0–{seconds:.2f} s)")
    else:
        timing = decide_timing(clip, opts, seconds, land)
    if spec:  # a sign never lands before the room is dark enough to see it strike
        spec = [dict(p, land=max(p["land"], timing.down + 0.3)) for p in spec]
        land = min(p["land"] for p in spec)

    # -- the light
    heads = head_path(clip, fr, n_out, fps, tl)
    tint = None
    if look == "dark" and opts["color"] != "auto":
        tint = textmod.hex_rgb(opts["color"])
    light = Light((W, H), opts["dark"], tint)

    # -- the sign or the light in the room
    neon = led = rays = haze = None
    glow_colour = None
    colour_why = ""
    sign_rect: Optional[Rect] = None
    on_from = timing.down
    t_end = timing.up if timing.up is not None else seconds
    span = list(range(int(math.floor(on_from * fps)), int(min(n_out - 1, math.ceil(t_end * fps))) + 1))
    sample = span[:: max(1, len(span) // 40)]
    placed = "the room behind you, everywhere"
    hidden = ""
    contrast = None
    seed = sum(ord(c) * (i + 1) for i, c in enumerate(" ".join(p["text"] for p in spec) if spec else look)) % (2 ** 31)
    if look in ("neon", "led"):
        store = wb.MaskStore(clip, tl, fr, (W, H), sample)
        if not store.by_frame or max(float(m.max()) for m in store.by_frame.values()) < 0.5:
            raise O.OptionError("nobody is in the shot while the lights are down, so there is nobody to put the sign "
                                "behind")
    if look == "neon":
        shown = [p["text"] if opts["word"] != "auto" else wb._display(p["text"]) for p in spec]
        if plan is not None and plan.get("sign", {}).get("px"):
            sg = plan["sign"]
            b = wb.make_block(tuple(sg["lines"]), NEON_FACE, int(sg["px"]))
            x, y = float(sg["centre_x"]) - b.w / 2, float(sg["bottom_y"]) - b.h
            med, p90, lp = wb._scan(b, store.crowns(), x, np.array([y]))
            pl = wb.Placement(b, x, y, float(med[0]), float(p90[0]), float(lp[0]), "where the plan puts it")
        else:
            pl = place_neon(wb.arrangements(shown), NEON_FACE, store, (W, H), heads)
        sign_rect = pl.rect
        if plan is not None and plan.get("sign", {}).get("color"):
            rgb, colour_why = textmod.hex_rgb(plan["sign"]["color"]), "as the plan says"
        elif opts["color"] != "auto":
            rgb, colour_why = textmod.hex_rgb(opts["color"]), "as asked"
        else:
            _, rgb, colour_why = choose_colour(NEON, _near_colours(clip, tl, fr, store, sign_rect, light, sample))
        strike = max(timing.down + 0.12, land - STRIKE_LEAD)
        neon = NeonSign(pl, rgb, (W, H), strike, timing.up, seed)
        glow_colour = rgb
        placed = pl.where
        hidden = (f"{pl.hidden_median:.0%} of the letters behind you (at worst {pl.hidden_p90:.0%}; the most-hidden "
                  f"letter at {pl.letter_p90:.0%} of what it may lose)")
        if pl.warn:
            warnings.append(pl.warn)
        wall = wb.wall_behind(clip, tl, fr, store, sign_rect, sample)
        core = 0.18 * np.array(rgb, np.float32) + 0.82 * 255.0
        wall_dark = max(wall["high"], wall["median"]) * light.far ** 2.2 + 0.006
        contrast = wb._ratio(float(_lum(core)), wall_dark)
    elif look == "led":
        text = " ".join(p["text"] for p in spec)
        lines = led_lines(text)
        if plan is not None and plan.get("sign", {}).get("pitch"):
            sg = plan["sign"]
            r = sg["rect"]
            rect, pitch, placed, p90 = Rect(*map(float, r)), float(sg["pitch"]), "where the plan puts it", 0.0
        else:
            rect, pitch, placed, p90 = place_board(lines, store, (W, H), heads)
        sign_rect = rect
        if plan is not None and plan.get("sign", {}).get("color"):
            rgb, colour_why = textmod.hex_rgb(plan["sign"]["color"]), "as the plan says"
        elif opts["color"] != "auto":
            rgb, colour_why = textmod.hex_rgb(opts["color"]), "as asked"
        else:
            _, rgb, colour_why = choose_colour(LED, _near_colours(clip, tl, fr, store, rect, light, sample))
        # each letter types as its word is said; letters of one word follow quickly
        chars = [(li, ci) for li, ln in enumerate(lines) for ci in range(len(ln))]
        word_lands = [p["land"] for p in spec]
        types = []
        wi, step = 0, 0.055
        t_word = word_lands[0] - 0.02
        for n_ch, (li, ci) in enumerate(chars):
            ch = lines[li][ci]
            if ch == " " or (ci == 0 and li > 0 and n_ch > 0):
                if ch == " " or (ci == 0 and li > 0):
                    wi = min(wi + 1, len(word_lands) - 1)
                    t_word = max(word_lands[wi] - 0.02, (types[-1][2] + step) if types else 0.0)
                    if ch == " ":
                        types.append((li, ci, t_word))
                        continue
            types.append((li, ci, t_word))
            t_word += step
        types = [(li, ci, max(timing.down + 0.25, at)) for li, ci, at in types]
        led = LedSign(lines, rect, pitch, rgb, (W, H), types, types[-1][2], timing.up)
        glow_colour = rgb
        hidden = f"the letters hidden {p90:.0%} at worst" if p90 < 1.0 else "the board is above you"
        wall = wb.wall_behind(clip, tl, fr, wb.MaskStore(clip, tl, fr, (W, H), sample[:8]), rect, sample[:8])
        core = 0.3 * np.array(rgb, np.float32) + 0.7 * 255.0
        contrast = wb._ratio(float(_lum(core)), 0.18 * max(wall["high"], wall["median"]) * light.far + 0.004)
    elif look == "rays":
        rgb = textmod.hex_rgb(opts["color"]) if opts["color"] != "auto" else GOLD
        colour_why = "as asked" if opts["color"] != "auto" else "gold"
        rays = Rays((W, H), rgb, timing.down + 0.05, seed)
        glow_colour = rgb
        placed = "rays from behind your head, sparkles in the room round you"
    elif look == "haze":
        rgb = textmod.hex_rgb(opts["color"]) if opts["color"] != "auto" else HAZE
        colour_why = "as asked" if opts["color"] != "auto" else "warm white"
        haze = Haze((W, H), rgb, timing.down + 0.05, seed)
        placed = "a beam of light from above, behind you"
    else:
        colour_why = "a coloured light on the wall, as asked" if tint else "no colour of its own"

    zone = SafeZone(W, H)
    caption_layer = None
    caption_report: dict = {}
    if opts["captions"] != "off":
        given = {"look": opts["captions"], "frame": opts["frame"]}
        caption_layer = layers.prepare(clip, layers.Spec("captions", given), None, None, fr, fix=opts["fix"])
        if sign_rect is not None:
            cards = caption_layer.job.plan.get("cards", [])
            if any(Rect.of(c["rect"]).intersects(sign_rect, margin=0.01 * H) and c["end"] > on_from for c in cards):
                given["position"] = "bottom"
                caption_layer = layers.prepare(clip, layers.Spec("captions", given), None, None, fr, fix=opts["fix"])
        caption_report = caption_layer.job.checks

    painter = Painter(clip, tl, fr, light, heads, timing, neon=neon, led=led, rays=rays, haze=haze,
                      glow_colour=glow_colour, captions=caption_layer)
    sounds = Sound(clip, timing, look, seconds, neon, led, rays.start if rays is not None else None) \
        if opts["sound"] else None

    # -- the checks
    over_face, sampled, worst = _face_check(painter, clip, tl, fr, n_out)
    t_lit = (land + 0.5) if land is not None else timing.down + DOWN + 0.4
    t_lit = min(t_end - 0.05, t_lit)
    skin_times = [min(t_end - 0.05, x) for x in np.linspace(timing.down + DOWN + 0.05, t_end - 0.05, 4)]
    skin = _skin(clip, tl, fr, painter, skin_times)
    keep_pct = int(round(100 * light.keep))
    lights = (f"down at {timing.down:.2f} s, {timing.why}" + (f" (said at {timing.said_at:.2f} s)" if timing.said_at else "")
              + f", over {DOWN:.2f} s; " + (f"back up at {timing.up:.2f} s" if timing.up is not None else "down to the end"))
    checks = {
        "lights": lights,
        "room": f"dark {opts['dark']:g}: the room at {light.far:.0%} of its light and {keep_pct}% of its colour, "
                f"the wall behind your head and shoulders at up to {light.near:.0%}",
        "look": look + (f" “{' '.join(p['text'] for p in spec)}”" if spec else "")
                + (f" ({sign_info.get('line') and 'from “' + sign_info['line'] + '”' or ''})" if sign_info.get("line") else ""),
        "colour": colour_why,
        "frames_over_face": over_face + int(caption_report.get("frames_over_face", 0) or 0),
        "frames_outside_safe_zone": (0 if sign_rect is None or zone.contains(sign_rect) else n_out)
                                    + int(caption_report.get("frames_outside_safe_zone", 0) or 0),
        "placed": placed,
        "face_kept": f"the dark room shows through at most {worst:.1%} of the face on {sampled} frames checked",
        "skin": skin,
        "cut_out": "two looks, only what both call you, steadied" if painter.models == 2 else "one look, steadied",
        "warnings": warnings + list(caption_report.get("warnings", []) or []),
    }
    if hidden:
        checks["hidden"] = hidden
    if contrast is not None:
        checks["contrast"] = f"{contrast:.1f}:1, the sign's core against the dark wall"
    if neon is not None:
        checks["sign"] = f"strikes on from {neon.strike:.2f} s, lit by {neon.strike + max(s['steady'] for s in neon.plan):.2f} s"
    if led is not None:
        checks["sign"] = f"types from {led.types[0][2]:.2f} s to {led.done:.2f} s, {pitch:.0f} px between dots"
    if caption_layer is not None:
        checks["captions"] = f"{opts['captions']}, " + str(caption_report.get("placed", ""))

    sheet = sheet_times(timing, land, neon, led, seconds, fps, n_out)
    out_plan = {
        "effect": NAME, "clip": clip.path, "options": opts,
        "output": {"width": W, "height": H, "fps": round(fps, 3), "seconds": round(seconds, 3), "framing": fr.note},
        "lights": {"down": round(timing.down, 3), "up": None if timing.up is None else round(timing.up, 3),
                   "why": timing.why},
        "look": look,
        "room": {"far": round(light.far, 3), "near": round(light.near, 3), "colour_kept": round(light.keep, 3)},
        "sheet_times": sheet,
    }
    if neon is not None:
        b = neon.block
        out_plan["sign"] = {"kind": "neon", "text": " ".join(b.lines), "lines": list(b.lines), "px": b.px,
                            "centre_x": round(sign_rect.cx, 1), "bottom_y": round(sign_rect.y1, 1),
                            "color": "#%02x%02x%02x" % tuple(int(v) for v in neon.colour),
                            "pieces": [{"text": p["text"], "land": round(p["land"], 3)} for p in spec],
                            "strike": round(neon.strike, 3), "said_at": sign_info.get("said_at"),
                            "chosen": {k: v for k, v in sign_info.items() if k in ("line", "runners_up")}}
    if led is not None:
        out_plan["sign"] = {"kind": "led", "text": " ".join(led.lines), "lines": led.lines, "pitch": round(led.p, 2),
                            "rect": sign_rect.as_list(), "color": "#%02x%02x%02x" % tuple(int(v) for v in led.colour),
                            "pieces": [{"text": p["text"], "land": round(p["land"], 3)} for p in spec],
                            "types": [round(at, 3) for _, _, at in led.types], "said_at": sign_info.get("said_at")}

    def still_times() -> list[float]:
        return [round(x, 3) for x in (max(0.0, timing.down - 0.2), timing.down + 0.1, sheet[3], sheet[5])]

    def after(result: dict, out_path) -> dict:
        lines = {}
        s = painter.seen
        if s["halo"]:
            h = np.array(s["halo"])
            lines["edge halo"] = (f"just outside your edge the room is {np.mean(h[:, 0]):+.1f} levels (of 255) from the "
                                  f"dark room drawn there, {np.max(h[:, 1]):+.1f} at the brightest 5% "
                                  f"(the room there averages {np.mean(h[:, 2]):.0f}), over {len(h)} frames")
        if s["edge"]:
            e = np.array(s["edge"])
            lines["edge steadiness"] = (f"where the picture is still, your edge moves {np.median(e):.2f} px a frame "
                                        f"(at worst {np.percentile(e, 90):.2f}), over {len(e)} pairs of frames")
        if sounds is not None and sounds.laid:
            lines["sound"] = "; ".join(sounds.laid) + (f"; peaks held {sounds.limited_db:.1f} dB" if sounds.limited_db else "")
        return lines

    return Job(NAME, clip, tl, fr, painter.draw, out_plan, checks=checks, still_times=still_times, after=after,
               sound=sounds.mix if sounds is not None else None)


def sheet_times(timing: Timing, land: Optional[float], neon: Optional[NeonSign], led: Optional[LedSign],
                seconds: float, fps: float, n_out: int) -> list[float]:
    """Six moments: just before the lights go, as they go, the sign striking or the room settling,
    lit, and two later."""
    end = timing.up if timing.up is not None else seconds
    d = timing.down
    if neon is not None:
        mid = neon.strike + 0.06
        lit = neon.strike + max(s["steady"] for s in neon.plan) + 0.15
    elif led is not None:
        mid = led.types[len(led.types) // 2][2] + 0.02
        lit = led.done + 0.25
    else:
        mid = d + DOWN + 0.25
        lit = d + DOWN + 0.8
    late = lit + 0.5 * max(0.2, end - lit)
    last = (timing.up + UP + 0.3) if timing.up is not None else end - 0.4
    ts = [d - 0.25, d + 0.08, mid, lit, late, last]
    return [round(min((n_out - 1) / fps, max(0.0, t)), 3) for t in ts]
