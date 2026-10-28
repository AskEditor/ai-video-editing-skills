"""transitions: the join between two shots, in one of eight looks, each with a sound of its own.

Two ways in:

- **The joins inside one video.** The cuts `clean-cut` made, read from the plan it writes beside
  its video and checked against the picture, or the cuts in a person's own edit, found in the
  picture (`core/shots.py`). A file has nothing spare either side of a cut: what came before it and
  what came after are all there is. So nothing overlaps, and the film keeps its length and its sound
  to the sample. A look that shows both pictures at once (a crossfade, a wipe) holds the last frame
  of the shot going out and the first frame of the shot coming in, each for its half; every other
  look (a punch, a whip's blur, a flash, a glitch, a light leak, dots) is drawn over each side's own
  frames and switches on the cut.
- **Several clips joined into one** (`clips`). Each side has frames of its own, so a crossfade or a
  wipe overlaps the two (the film is shorter by the overlap) and their sounds cross over the same
  frames. When a voice is heard in an overlap, the sound switches at the overlap's quietest moment
  instead, so two voices never play at once.

Which joins: a talking video never gets one on every cut. `auto` takes the joins where a sentence
ends (`between-ideas`), at least three seconds apart; `every` takes them all; or seconds. Between
clips, every join. A talking video with no cut at all has the moments between its sentences, for
the looks that work inside one shot.

Where: the middle of each transition (the most blur, the brightest leak, the white of a flash, the
switch of a glitch) is on the first frame of the new shot, and its sound lands there with it. The
sounds are made here from numbers, the pack's own whoosh and hit among them, and sit under the voice.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..core import audio, loudness, moments, shots, sound, video
from ..core import glitch as glitchmod
from ..core import framing as framing_mod
from ..core.framing import Framing
from ..core.layout import Rect
from ..core.paths import PACK
from ..core.timeline import Timeline
from ..core.zoom import CHIN_LINE, CROWN, punch_progress, punch_window  # noqa: F401 (the shared framing)
from ..core.zoom import between as _between
from ..core.zoom import crop as _crop
from ..core.zoom import ease as _ease
from ..core.zoom import place_window as _place_window  # noqa: F401
from . import options as O

NAME = "transitions"
MOVES_PERSON = True  # a punch or a whip moves the whole picture: never laid inside another effect
RATE = video.AUDIO_RATE

LOOKS = ("crossfade", "wipe", "whip", "glitch", "light-leak", "dot-matrix", "punch", "flash")
ALIASES = {"fade": "crossfade", "dissolve": "crossfade", "cross-fade": "crossfade", "mix": "crossfade",
           "whip-pan": "whip", "swish": "whip", "pan": "whip", "zoom-punch": "punch", "punch-in": "punch",
           "zoom": "punch", "leak": "light-leak", "lightleak": "light-leak", "light": "light-leak",
           "dots": "dot-matrix", "dot": "dot-matrix", "dot-reveal": "dot-matrix", "matrix": "dot-matrix",
           "white-flash": "flash", "dip": "flash", "glitches": "glitch", "wipes": "wipe"}
# A sensible length for each look, in seconds: (inside one video, between clips). Between clips a
# crossfade or a wipe can take its time; on a talking video a long one blurs two moments of a face.
AUTO = {"crossfade": (0.40, 0.80), "wipe": (0.45, 0.60), "whip": (0.30, 0.34), "glitch": (0.30, 0.40),
        "light-leak": (0.80, 1.00), "dot-matrix": (0.90, 1.10), "punch": (0.25, 0.25), "flash": (0.30, 0.40)}
# The share of each look's length that comes before the join: a flash flares fast and fades slowly,
# dots break up quickly and take longer to build the new shot, a punch lands almost on the cut.
BEFORE = {"crossfade": 0.5, "wipe": 0.5, "whip": 0.5, "glitch": 0.5, "light-leak": 0.45, "dot-matrix": 0.35,
          "punch": 0.2, "flash": 0.3}
SOUND_OF = {"crossfade": "air", "wipe": "swish", "whip": "whoosh", "glitch": "crackle", "light-leak": "shimmer",
            "dot-matrix": "blips", "punch": "hit", "flash": "snap"}
APART = 3.0  # seconds: between ideas, transitions at least this far apart
EDGE = 0.4  # seconds: no transition this close to the start or the end of the film
UNDER_VOICE = 10.0  # dB: each sound's loudest tenth of a second under the voice
PUNCH_ZOOM = 1.15  # how far a punch goes in, at the least, when the head leaves room
PUNCH_MOST = 1.30  # and at the most: past this a phone's picture goes soft
PUNCH_GAIN = 1.12  # a punch in leaves the face this much larger than it was just before the cut
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}


# ---- options -----------------------------------------------------------------------------------

def _clock(text: str) -> float:
    """'12.5', '12.5s' or '0:12.5' as seconds."""
    t = text.strip().lower().rstrip("s")
    try:
        if ":" in t:
            m, s = t.split(":", 1)
            return int(m) * 60 + float(s)
        return float(t)
    except ValueError:
        raise O.OptionError(f"give seconds such as 4.5 or 0:04.5 (got {text!r})") from None


@dataclass(frozen=True)
class LookList:
    help: str

    def parse(self, raw) -> str:
        v = str(raw).strip().lower()
        if v in ("", "auto"):
            return "auto"
        out = []
        for part in re.split(r"[,;|/]+|\s+then\s+|\s+and\s+", v):
            p = re.sub(r"[\s_]+", "-", part.strip())
            if not p:
                continue
            p = ALIASES.get(p, p)
            if p not in LOOKS:
                raise O.OptionError(f"no look called {part.strip()!r}; the looks are {', '.join(LOOKS)} "
                                    "(several, separated by commas, take turns, one per join)")
            out.append(p)
        return ",".join(out) or "auto"


@dataclass(frozen=True)
class JoinChoice:
    help: str

    def parse(self, raw) -> str:
        v = str(raw).strip().lower()
        if v in ("", "auto"):
            return "auto"
        key = re.sub(r"[\s_]+", "-", v)
        if key in ("every", "all", "every-cut", "each", "every-join", "all-cuts"):
            return "every"
        if key in ("between-ideas", "ideas", "sentences", "between-sentences", "ideas-only"):
            return "between-ideas"
        times = [_clock(p) for p in re.split(r"[,;\s]+", v) if p.strip()]
        if not times or any(t < 0 for t in times):
            raise O.OptionError(f"say auto, every, between-ideas, or seconds such as 4.5, 12 (got {raw!r})")
        return ", ".join(f"{t:g}" for t in sorted(times))


@dataclass(frozen=True)
class Seconds:
    lo: float
    hi: float
    help: str

    def parse(self, raw):
        v = str(raw).strip().lower()
        if v in ("", "auto"):
            return "auto"
        x = _clock(v)
        if not self.lo <= x <= self.hi:
            raise O.OptionError(f"give auto or seconds between {self.lo:g} and {self.hi:g} (got {raw!r})")
        return x


OPTIONS = {
    "look": LookList("crossfade, wipe, whip, glitch, light-leak, dot-matrix, punch or flash; several separated by "
                     "commas take turns, one per join; auto: a punch on a talking video, a crossfade otherwise"),
    "joins": JoinChoice("which joins get one: auto (between ideas on a talking video, every join otherwise), every, "
                        "between-ideas (where a sentence ends, three seconds apart at least), or seconds such as 4.5, 12"),
    "duration": Seconds(0.1, 3.0, "how long each one lasts, in seconds; auto gives each look a length of its own"),
    "clips": O.Text("more clips to join after this one, in order: a folder (its videos by name) or paths separated "
                    "by ;, each with the seconds to use if not all of it, such as lake.mp4 2-5"),
    "sound": O.Switch("on lays each look's own sound under the voice; off keeps the video's sound only"),
    "color": O.Colour("the colour of the look where it has one (a flash, a light leak, a wipe's edge, the dots, a "
                      "crossfade dipping through a colour); auto chooses"),
    "events": O.Text("the plan of the edit that made the cuts (empty: the one beside the video)"),
}

LAB = "{lab}"
# The runs tests/matrix.py makes on the shared clips. The films of joined B-roll and the chain on E
# (clean-cut without captions, then transitions, then captions) are made by the lab's own script,
# since their clips are not among the matrix's.
MATRIX = [
    ("33", "33", {}),  # its own cuts, a punch between ideas
    ("33-every-whip", "33", {"joins": "every", "look": "whip"}),  # what every cut looks like
    ("E-cut", "E-cut", {"look": "whip, flash", "events": LAB + "/clean-cut/E.plan.json"}),  # clean-cut's cuts
    ("A", "A", {}),  # no cut: a punch between its two sentences
    ("B", "B", {"look": "light-leak", "joins": "2.4"}),  # a leak on a bright scene, at a time given
    ("A-then-B", "A", {"clips": LAB + "/test-clips/B-airport-headroom-1080x1920.mp4", "look": "crossfade"}),
]


# ---- small pieces --------------------------------------------------------------------------------

_TO_LINEAR = ((np.arange(256, dtype=np.float32) / 255.0) ** 2.2).astype(np.float32)
_FROM_LINEAR = np.clip(np.round(255.0 * (np.linspace(0, 1, 4096, dtype=np.float32) ** (1 / 2.2))), 0, 255).astype(np.uint8)


def linear(img: np.ndarray) -> np.ndarray:
    """A picture in light (0..1), so that blending two of them looks like light, not paint."""
    return _TO_LINEAR[img]


def from_linear(x: np.ndarray) -> np.ndarray:
    return _FROM_LINEAR[np.clip((x * 4095.0 + 0.5).astype(np.int32), 0, 4095)]


def _rgb(hex_colour: str) -> np.ndarray:
    from ..core.text import colour

    return np.array(colour(hex_colour), np.float32) / 255.0


def _seed(*parts) -> int:
    h = 2166136261
    for p in parts:
        for ch in str(p):
            h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h


# ---- the looks: each draws one frame of its transition ------------------------------------------

@dataclass
class Moment:
    """One frame of a transition, as a look draws it."""
    u: float  # 0 to 1 through the transition; the first frame of the new shot is just past 0.5
    a: np.ndarray  # the shot going out at this frame (playing, or its last frame held)
    b: np.ndarray  # the shot coming in (playing, or its first frame held)
    incoming: bool  # this frame is on the new shot's side of the join
    i: int  # this frame's number inside the transition
    n: int  # frames in the transition


@dataclass
class Transition:
    n: int  # its number, from 1
    look: str
    at: float  # seconds: where the new shot starts (the join)
    frame: int  # the first frame of the new shot
    duration: float
    first: int  # its first frame
    last: int  # its last frame
    source: str  # where its join came from
    colour: str = "auto"
    overlap: int = 0  # frames the two clips overlap (between clips), 0 inside one video
    direction: int = 1  # which way it travels, for the looks that move
    zoom_from: float = 1.0  # a punch: the framing before and after
    zoom_to: float = 1.0
    note: str = ""
    sound: Optional[dict] = None
    settle: int = -1  # the first frame that really belongs to the new shot (past any stray frame)
    punch_note: str = ""  # a punch that went in less than asked, and why

    @property
    def frames(self) -> int:
        return self.last - self.first + 1

    def u(self, j: int) -> float:
        """How far through it frame j is: the frames before the join share 0..0.5 and the frames
        from the join on share 0.5..1, each frame at its middle, so the join sits exactly at 0.5."""
        before, after = self.frame - self.first, self.last - self.frame + 1
        if j < self.frame:
            return 0.5 * (j - self.first + 0.5) / max(1, before)
        return 0.5 + 0.5 * (j - self.frame + 0.5) / max(1, after)

    def as_dict(self, fps: float) -> dict:
        d = {"n": self.n, "look": self.look, "at": round(self.at, 4), "frame": self.frame,
             "duration": round(self.duration, 3), "first_frame": self.first, "last_frame": self.last,
             "from": round(self.first / fps, 4), "to": round((self.last + 1) / fps, 4), "join_from": self.source}
        if self.colour != "auto":
            d["color"] = self.colour
        if self.overlap:
            d["overlap_frames"] = self.overlap
        if self.look in ("whip", "wipe", "light-leak"):
            d["direction"] = "left" if self.direction > 0 else "right"
        if self.look == "punch":
            d["zoom"] = [round(self.zoom_from, 3), round(self.zoom_to, 3)]
        if self.sound:
            d["sound"] = self.sound
        if self.note:
            d["note"] = self.note
        return d


def look_crossfade(m: Moment, tr: Transition) -> np.ndarray:
    if tr.colour != "auto":  # a dip through a colour: out to it, then in from it
        c = _rgb(tr.colour)
        if not m.incoming:
            w = float(_ease(m.u / 0.5))
            return from_linear(linear(m.a) * (1 - w) + c * w)
        w = float(_ease((m.u - 0.5) / 0.5))
        return from_linear(c * (1 - w) + linear(m.b) * w)
    w = float(_ease(m.u))
    return from_linear(linear(m.a) * (1 - w) + linear(m.b) * w)


_GRIDS: dict = {}


def _grid(h: int, w: int):
    key = (h, w)
    if key not in _GRIDS:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        _GRIDS[key] = (xx, yy)
    return _GRIDS[key]


def look_wipe(m: Moment, tr: Transition) -> np.ndarray:
    """A soft edge, leaning a little, crossing the frame; it reaches the middle on the join. A thin
    line of light rides the edge, and the new shot casts a faint shadow ahead of it."""
    h, w = m.a.shape[:2]
    xx, yy = _grid(h, w)
    lean = math.tan(math.radians(9.0))
    feather = 0.022 * w
    reach = (h / 2) * lean + 3 * feather
    p = float(_ease(m.u))
    edge = -reach + (w + 2 * reach) * p
    x = xx if tr.direction > 0 else (w - 1 - xx)
    s = (x - edge - (yy - h / 2) * lean) * math.cos(math.atan(lean))  # signed distance from the edge
    b_share = np.clip(0.5 - s / (2 * feather), 0, 1)
    b_share = b_share * b_share * (3 - 2 * b_share)
    la, lb = linear(m.a), linear(m.b)
    shade = 1.0 - 0.28 * np.exp(-np.clip(s, 0, None) / (2.5 * feather)) * (s > 0)
    out = lb * b_share[..., None] + la * (shade * (1 - b_share))[..., None]
    line = np.exp(-(s / (0.0035 * w + 0.6)) ** 2)
    col = _rgb(tr.colour) if tr.colour != "auto" else np.array([1.0, 1.0, 1.0], np.float32)
    strength = 0.55 if tr.colour == "auto" else 0.9
    out = out + (line * strength)[..., None] * (col - out) * 0.9
    return from_linear(np.clip(out, 0, 1))


def _whip_offset(u: float, steep: float = 2.6) -> float:
    """How far the shot has travelled (0 to 1) at u: slow, then a rush through the join, then slow."""
    return 0.5 + 0.5 * math.tanh(steep * (2 * u - 1)) / math.tanh(steep)


def look_whip(m: Moment, tr: Transition) -> np.ndarray:
    """The camera flung sideways: the two shots side by side race across, blurred along the move by
    as far as they travel in a frame, the seam between them crossing the middle on the join."""
    import cv2

    h, w = m.a.shape[:2]
    du = 0.5 / max(1, m.n)
    x0, x1 = _whip_offset(max(0.0, m.u - du)), _whip_offset(min(1.0, m.u + du))
    off = int(round(w * _whip_offset(m.u)))
    off = min(w, max(0, off))
    if tr.direction > 0:  # the picture moves left: the new shot comes in from the right
        strip = np.concatenate([m.a[:, off:], m.b[:, :off]], axis=1)
    else:
        strip = np.concatenate([m.b[:, w - off:], m.a[:, :w - off]], axis=1)
    travel = abs(x1 - x0) * w  # pixels in one frame
    blur = int(round(0.9 * travel))
    if blur >= 3:
        strip = cv2.blur(strip, (blur, 1), borderType=cv2.BORDER_REFLECT)
        # a fast pan also lifts a little light into the smear
        lift = min(0.10, blur / w * 0.25)
        strip = from_linear(np.clip(linear(strip) * (1 + lift), 0, 1))
    return np.ascontiguousarray(strip)


def look_glitch(m: Moment, tr: Transition) -> np.ndarray:
    """A burst of broken signal: bands of the picture torn sideways, the colours pulled apart, blocks
    gone coarse and scan lines, strongest on the join, with bands of the other shot flashing through
    (drawn by `core/glitch.py`, shared with the `glitch` effect)."""
    rng = np.random.default_rng(_seed("glitch", tr.n, m.i))
    peak = max(0.0, 1.0 - abs(m.u - 0.5) / 0.5) ** 0.9  # strongest on the join, gone by its ends
    stutter = (0.35, 1.0, 0.7, 1.0, 0.5)[m.i % 5]  # the signal catches and drops, frame to frame
    e = peak * stutter if 0 < m.i < m.n - 1 else 0.0  # its first and last frames are clean
    base = (m.b if m.incoming else m.a).copy()
    other = m.a if m.incoming else m.b
    return glitchmod.tear(base, other, e, rng, tr.colour)


# The colours light turns as it gets hotter, from where it barely shows to its white core: a burn
# ramp, (intensity, colour). With `color` given, the ramp runs through that colour instead.
BURN = ((0.00, (0.55, 0.05, 0.02)), (0.30, (1.00, 0.22, 0.05)), (0.60, (1.00, 0.52, 0.12)),
        (0.95, (1.00, 0.82, 0.42)), (1.30, (1.00, 0.97, 0.88)))


def _ramp(hex_colour: str) -> tuple[np.ndarray, np.ndarray]:
    if hex_colour == "auto":
        stops = BURN
    else:
        c = _rgb(hex_colour)
        stops = ((0.0, tuple(0.45 * c)), (0.30, tuple(c)), (0.65, tuple(np.clip(c * 0.7 + 0.3, 0, 1))),
                 (1.00, tuple(np.clip(c * 0.35 + 0.65, 0, 1))), (1.30, (1.0, 1.0, 0.97)))
    xs = np.array([s[0] for s in stops], np.float32)
    cs = np.array([s[1] for s in stops], np.float32)
    return xs, cs


def look_light_leak(m: Moment, tr: Transition) -> np.ndarray:
    """Light spilling into the lens: a long pool of it sweeps in from one side, deep red where it
    barely reaches, orange, then a yellow-white core, swelling until the core covers the frame on the
    join (the shot changes inside it) and drifting off the other side as it fades. Drawn here, frame
    by frame; no footage of a real leak is used."""
    import cv2

    h, w = m.a.shape[:2]
    pic = m.b if m.incoming else m.a
    sw, sh = max(8, w // 8), max(8, h // 8)
    xx, yy = np.meshgrid(np.linspace(0, 1, sw, dtype=np.float32), np.linspace(0, 1, sh, dtype=np.float32))
    aspect = h / w
    u = m.u
    rise = float(_ease(u / 0.5)) ** 1.3 if u < 0.5 else 1.0
    fall = 1.0 if u < 0.5 else float(1.0 - _ease((u - 0.5) / 0.5))
    strength = rise * fall
    side = 1.0 if tr.direction > 0 else -1.0
    t = float(_ease(u))
    # Three pools travelling across: (x from, x to, y from, y to, width, height, weight), in shares of
    # the frame; the first is the long main sweep, the others give it an uneven, organic edge.
    pools = ((1.25, -0.25, 0.30, 0.55, 0.42, 0.75, 1.15), (1.45, 0.05, 0.75, 0.40, 0.30, 0.38, 0.65),
             (1.05, -0.45, 0.05, 0.20, 0.28, 0.30, 0.55))
    heat = np.zeros((sh, sw), np.float32)
    for x0, x1, y0, y1, rw, rh, g in pools:
        cx = 0.5 + side * ((x0 + (x1 - x0) * t) - 0.5)
        cy = y0 + (y1 - y0) * t
        d2 = ((xx - cx) / rw) ** 2 + ((yy - cy) * aspect / (rh * aspect)) ** 2
        heat += g * np.exp(-d2)
    heat = heat * (1.55 * strength) + 0.25 * strength  # and a glow over all of it, so the join is hidden
    heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)
    xs, cs = _ramp(tr.colour)
    hc = np.clip(heat, 0, xs[-1])
    light = np.stack([np.interp(hc, xs, cs[:, c]) for c in range(3)], axis=2).astype(np.float32)
    light *= np.clip(heat / 0.25, 0, 1)[..., None]  # nothing where there is no heat
    lp = linear(pic)
    lit = 1.0 - (1.0 - lp) * (1.0 - np.clip(light, 0, 1))  # screen: light added
    # On a white wall screening adds nothing, so the light also colours what it falls on.
    tint = (0.55 * np.clip(heat / 0.6, 0, 1) * (1 - np.clip((heat - 0.9) / 0.5, 0, 1)))[..., None]
    out = lit * (1 - tint) + light * tint
    return from_linear(np.clip(out, 0, 1))


_DOTS: dict = {}


def _dot_distance(h: int, w: int, cell: int) -> np.ndarray:
    key = (h, w, cell)
    if key not in _DOTS:
        xx, yy = _grid(h, w)
        cx = (np.floor(xx / cell) + 0.5) * cell
        cy = (np.floor(yy / cell) + 0.5) * cell
        _DOTS[key] = (np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (cell / 2)).astype(np.float32)
    return _DOTS[key]


def _dots(img: np.ndarray, radius: np.ndarray, glow: np.ndarray, cell: int, colour: str) -> np.ndarray:
    """`img` as a board of round lights, one per cell: each the colour of its cell (or `colour`, as
    bright as its cell), `radius` (a share of the cell, per cell) wide and `glow` times as bright."""
    import cv2

    h, w = img.shape[:2]
    gh, gw = math.ceil(h / cell), math.ceil(w / cell)
    small = cv2.resize(img, (gw, gh), interpolation=cv2.INTER_AREA)
    lin = linear(small)
    if colour != "auto":
        lum = (lin * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(axis=2, keepdims=True)
        lin = lum * _rgb(colour) * 1.4
    lin = np.clip(lin * glow[..., None], 0, 1.5)
    big_r = np.repeat(np.repeat(radius, cell, 0), cell, 1)[:h, :w]
    big_c = np.repeat(np.repeat(lin, cell, 0), cell, 1)[:h, :w]
    d = _dot_distance(h, w, cell)
    soft = 2.4 / cell
    mask = np.clip((big_r - d) / soft + 0.5, 0, 1)
    out = big_c * mask[..., None]
    haze = cv2.GaussianBlur(cv2.resize(out, (w // 4, h // 4), interpolation=cv2.INTER_AREA), (0, 0), cell / 5)
    haze = cv2.resize(haze, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(1.0 - (1.0 - np.clip(out, 0, 1)) * (1.0 - 0.55 * np.clip(haze, 0, 1)), 0, 1)


def look_dot_matrix(m: Moment, tr: Transition) -> np.ndarray:
    """The shot going out breaks into a board of lights and goes dark; on the join the board shows
    the new shot, its lights coming on in a wave from the middle, brighter as they arrive; then the
    full picture snaps back into focus."""
    h, w = m.a.shape[:2]
    cell = max(10, int(round(w / 34)))
    gh, gw = math.ceil(h / cell), math.ceil(w / cell)
    u = m.u
    if not m.incoming:
        k = float(_ease(u / 0.38))  # into dots
        dim = float(_ease((u - 0.34) / 0.16))  # then the lights shrink and fade
        radius = np.full((gh, gw), 0.92 * (1 - 0.45 * dim), np.float32)
        glow = np.full((gh, gw), 1.15 * (1 - 0.65 * dim), np.float32)
        board = _dots(m.a, radius, glow, cell, tr.colour)
        return from_linear(linear(m.a) * (1 - k) + board * k)
    v = (u - 0.5) / 0.5  # 0..1 across the new shot's half
    gy, gx = np.mgrid[0:gh, 0:gw].astype(np.float32)
    dist = np.sqrt(((gx + 0.5) / gw - 0.5) ** 2 + (((gy + 0.5) / gh - 0.45) * gh / gw) ** 2)
    dist /= max(1e-6, float(dist.max()))
    jitter = np.random.default_rng(_seed("dots", tr.n)).uniform(0, 0.08, (gh, gw)).astype(np.float32)
    lit_at = -0.12 + 0.55 * dist + jitter  # when each light comes on, in v: the middle already on the join
    grow = np.clip((v - lit_at) / 0.16, 0, 1)
    radius = (0.92 * _ease(grow)).astype(np.float32)
    glow = (1.0 + 0.9 * np.exp(-((v - lit_at - 0.08) / 0.1) ** 2) * (grow > 0)).astype(np.float32)
    board = _dots(m.b, radius, glow, cell, tr.colour)
    snap = float(_ease((v - 0.74) / 0.14))  # into focus
    flare = 0.25 * math.exp(-((v - 0.82) / 0.06) ** 2)
    out = board * (1 - snap) + linear(m.b) * snap
    return from_linear(np.clip(out * (1 + flare), 0, 1))


def look_flash(m: Moment, tr: Transition) -> np.ndarray:
    """A burst of light: the shot going out flares to white (or the colour) in a few frames, the cut
    happens inside the white, and the new shot comes out of it more slowly."""
    c = _rgb(tr.colour) if tr.colour != "auto" else np.array([1.0, 1.0, 1.0], np.float32)
    if not m.incoming:
        f = (m.u / 0.5) ** 2.2 * 0.97
        pic = m.a
    else:
        f = 0.97 * (1.0 - (m.u - 0.5) / 0.5) ** 2.4
        pic = m.b
    return from_linear(linear(pic) * (1 - f) + c * f)


LOOK_DRAW: dict[str, Callable[[Moment, Transition], np.ndarray]] = {
    "crossfade": look_crossfade, "wipe": look_wipe, "whip": look_whip, "glitch": look_glitch,
    "light-leak": look_light_leak, "dot-matrix": look_dot_matrix, "flash": look_flash,
}


def needs_both(look: str, colour: str) -> bool:
    """Does the look show both shots at once, so that between clips they overlap? A crossfade
    through a colour does not: it goes out to the colour and comes in from it."""
    return look == "wipe" or (look == "crossfade" and colour == "auto")


# ---- the punch: a closer framing on the join, centred on the face (the framing is the core's: core/zoom.py)

# ---- the sounds, made here ----------------------------------------------------------------------

def _band_noise(n: int, centre: np.ndarray, width: float, seed: int, size: int = 1024, hop: int = 256) -> np.ndarray:
    """Noise whose pitch follows `centre` (Hz, one value per sample) in a band `width` octaves wide:
    air moving, rising or falling. Made frame by frame in the frequency domain and joined smoothly."""
    rng = np.random.default_rng(seed)
    pad = size
    x = rng.standard_normal(n + 2 * pad).astype(np.float64)
    win = np.hanning(size)
    freqs = np.fft.rfftfreq(size, 1.0 / RATE)
    lf = np.log2(np.maximum(freqs, 1.0))
    out = np.zeros(n + 2 * pad)
    norm = np.zeros(n + 2 * pad)
    sigma = max(0.05, width) / 2.355
    for s in range(0, n + pad, hop):
        k = int(min(n - 1, max(0, s - pad + size // 2)))
        mask = np.exp(-0.5 * ((lf - math.log2(max(20.0, float(centre[k])))) / sigma) ** 2)
        seg = np.fft.irfft(np.fft.rfft(x[s:s + size] * win) * mask, size) * win
        out[s:s + size] += seg
        norm[s:s + size] += win * win
    y = (out / np.maximum(norm, 1e-6))[pad:pad + n]
    return (y / (np.abs(y).max() + 1e-12)).astype(np.float32)


def _stereo(mono: np.ndarray, pan) -> np.ndarray:
    theta = (np.clip(pan, -1, 1) + 1) * np.pi / 4
    return (np.stack([mono * np.cos(theta), mono * np.sin(theta)], axis=1) * np.sqrt(2)).astype(np.float32)


def _swell(t: np.ndarray, peak: float, rise: float, fall: float) -> np.ndarray:
    return np.where(t < peak, np.exp(-((t - peak) / rise) ** 2), np.exp(-((t - peak) / fall) ** 2))


def sound_air(d: float, seed: int) -> tuple[np.ndarray, float]:
    """A breath of air through a crossfade: a soft, low swell that peaks on the join."""
    T = d + 0.5
    t = np.arange(int(T * RATE)) / RATE
    peak = T / 2
    env = _swell(t, peak, 0.45 * T / 2, 0.5 * T / 2)
    centre = 260 + 520 * env
    a = _band_noise(len(t), centre, 1.4, seed)
    b = _band_noise(len(t), centre * 1.1, 1.4, seed + 1)
    mono = 0.5 * (a + b) * env
    return np.stack([0.75 * mono + 0.25 * a * env, 0.75 * mono + 0.25 * b * env], axis=1).astype(np.float32), peak


def sound_swish(d: float, direction: int, seed: int) -> tuple[np.ndarray, float]:
    """A wipe's edge going by: a brighter rush that climbs to the join and drops away, travelling
    across with the edge."""
    T = d + 0.35
    t = np.arange(int(T * RATE)) / RATE
    peak = 0.6 * d + 0.12
    env = _swell(t, peak, 0.55 * d, 0.16)
    centre = 700 * (5200 / 700) ** (env ** 0.8)
    mono = _band_noise(len(t), centre, 1.1, seed) * env
    pan = np.clip((t / max(1e-6, peak) - 0.5) * 1.4, -0.8, 0.8) * direction
    return _stereo(mono, pan), peak


def sound_crackle(d: float, seed: int) -> tuple[np.ndarray, float]:
    """A digital crackle: short bursts of crushed noise and buzzing tones, stuttering, thickest on
    the join, each burst somewhere else across the stereo field (`core/glitch.py`)."""
    return glitchmod.crackle(d, seed)


def sound_shimmer(d: float, seed: int) -> tuple[np.ndarray, float]:
    """Light flooding in: an airy swell that brightens to the join, with a soft ringing chord on top
    that swims slowly, then lets go."""
    T = d + 0.5
    t = np.arange(int(T * RATE)) / RATE
    peak = 0.45 * d + 0.18
    env = _swell(t, peak, 0.32 * d + 0.07, 0.4 * d + 0.1)
    air = _band_noise(len(t), 1800 + 4200 * env, 1.6, seed)
    ring = np.zeros(len(t))
    for k, f in enumerate((1318.5, 1975.5, 2637.0, 3951.1)):
        vib = 1 + 0.004 * np.sin(2 * np.pi * (4.5 + k) * t + k)
        ring += np.sin(2 * np.pi * f * np.cumsum(vib) / RATE) / (1 + k)
    ring /= np.abs(ring).max() + 1e-9
    mono = (0.75 * air + 0.35 * ring) * env
    left = _band_noise(len(t), 2600 + 3000 * env, 1.8, seed + 7) * env
    return np.stack([mono + 0.25 * left, mono - 0.25 * left], axis=1).astype(np.float32) * 0.8, peak


def sound_blips(d: float, n_in: float, seed: int) -> tuple[np.ndarray, float]:
    """Lights coming on: a few falling blips as the old shot breaks up, a quickening rising run as
    the new one builds, and a click as it snaps into focus. `n_in` is where the join is (seconds in)."""
    rng = np.random.default_rng(seed)
    T = d + 0.25
    n = int(T * RATE)
    out = np.zeros((n, 2), np.float32)
    scale = (523.3, 587.3, 659.3, 784.0, 880.0, 1046.5, 1174.7, 1318.5, 1568.0, 1760.0, 2093.0)

    def blip(at: float, f: float, level: float, pan: float, length: float = 0.022):
        m = int(length * RATE)
        tt = np.arange(m) / RATE
        tone = np.sin(2 * np.pi * f * tt) + 0.25 * np.sin(4 * np.pi * f * tt)
        env = np.sin(np.pi * np.clip(tt / length, 0, 1)) ** 2
        a = int(at * RATE)
        if 0 <= a < n:
            b = min(n, a + m)
            out[a:b] += _stereo((tone * env * level)[:b - a].astype(np.float32), pan)

    before = n_in
    for k in range(3):  # falling, as the old shot goes out
        blip(before * (0.15 + 0.25 * k), scale[6 - 2 * k], 0.45, rng.uniform(-0.5, 0.5))
    build = 0.75 * (d - before)
    count = 9
    for k in range(count):  # rising, closer together, as the new shot builds
        frac = (k / (count - 1)) ** 0.8
        blip(before + 0.02 + build * frac, scale[min(len(scale) - 1, 1 + k)], 0.5 + 0.4 * frac, rng.uniform(-0.7, 0.7))
    focus = before + 0.86 * (d - before)
    m = int(0.03 * RATE)  # the click of focus
    tt = np.arange(m) / RATE
    click = (np.sin(2 * np.pi * 3200 * tt) * np.exp(-tt / 0.003) + 0.6 * np.sin(2 * np.pi * 1100 * tt) * np.exp(-tt / 0.006))
    a = int(focus * RATE)
    if a < n:
        b = min(n, a + m)
        out[a:b] += _stereo(click[:b - a].astype(np.float32) * 0.9, 0.0)
    return out, n_in


def sound_snap(seed: int) -> tuple[np.ndarray, float]:
    """A flash going off: a bright crack and a short hiss of light."""
    rng = np.random.default_rng(seed)
    T = 0.45
    t = np.arange(int(T * RATE)) / RATE
    crack = np.diff(rng.standard_normal(len(t) + 1)) * np.exp(-t / 0.004)
    hiss = _band_noise(len(t), np.full(len(t), 7200.0), 1.2, seed + 3) * np.exp(-t / 0.09)
    thump = np.sin(2 * np.pi * (140 + 260 * np.exp(-t / 0.01)) * t) * np.exp(-t / 0.04)
    mono = (0.5 * crack + 0.8 * hiss + 0.5 * thump) * np.clip(t / 0.0008, 0, 1)
    return _stereo(mono.astype(np.float32), 0.0), 0.0


library = sound.library  # the pack's own sounds, read once (core/sound.py)


def make_sound(tr: Transition, fps: float) -> tuple[str, np.ndarray, float]:
    """The sound for a transition: (its name, the samples, where in them it lands on the join)."""
    seed = _seed("sound", tr.look, tr.n)
    d = tr.frames / fps
    before = (tr.frame - tr.first) / fps
    kind = SOUND_OF[tr.look]
    if kind == "whoosh":
        x, anchor = library("whoosh")
        speed = float(np.clip(0.32 / max(0.12, d), 0.85, 1.5))
        x = sound.resample(x, speed)
        anchor /= speed
        if tr.direction < 0:
            x = x[:, ::-1]
        return kind, np.ascontiguousarray(x), anchor
    if kind == "hit":
        x, anchor = library("hit")
        if tr.zoom_to < tr.zoom_from:  # going back out: a little lower
            x = sound.resample(x, 0.89)
        return kind, x, anchor
    if kind == "air":
        x, anchor = sound_air(d, seed)
    elif kind == "swish":
        x, anchor = sound_swish(d, tr.direction, seed)
    elif kind == "crackle":
        x, anchor = sound_crackle(d, seed)
    elif kind == "shimmer":
        x, anchor = sound_shimmer(d, seed)
    elif kind == "blips":
        x, anchor = sound_blips(d, before, seed)
    else:
        x, anchor = sound_snap(seed)
    return kind, x, anchor


# ---- joins inside one video -----------------------------------------------------------------------

@dataclass
class Join:
    at: float  # seconds: where the new shot starts
    frame: int  # the first frame of the new shot
    source: str  # the plan, the picture, given, between sentences, between clips
    cut: bool = True  # a change of shot (False: a moment between two sentences in one shot)
    removed: float = 0.0  # seconds the edit took out here
    before: str = ""  # the last words before it
    after: str = ""  # the first words after it
    sentence: bool = False  # the words before it end a sentence
    pause: float = 0.0  # seconds of quiet around it
    chosen: bool = False
    why: str = ""

    def as_dict(self) -> dict:
        d = {"at": round(self.at, 4), "frame": self.frame, "from": self.source, "cut": self.cut,
             "chosen": self.chosen}
        for k in ("before", "after", "why"):
            if getattr(self, k):
                d[k] = getattr(self, k)
        if self.removed:
            d["removed_s"] = round(self.removed, 3)
        if self.cut and self.before:
            d["ends_sentence"] = self.sentence
        return d


def _edit_plan(clip, given: str, notes: list[str]) -> tuple[Optional[dict], Optional[Path]]:
    """The plan of the edit that made this video: `events` when given, else the one beside it."""
    from . import PlanError

    path = Path(given).expanduser() if given.strip() else Path(clip.path).with_suffix(".plan.json")
    if not path.exists():
        if given.strip():
            raise PlanError(f"there is no plan at {given}")
        return None, None
    try:
        plan = json.loads(path.read_text())
    except ValueError:
        raise PlanError(f"{path} is not a plan this pack wrote") from None
    seconds = float((plan.get("output") or {}).get("seconds", -1))
    mine = clip.info.frames / clip.info.rate
    if seconds > 0 and abs(seconds - mine) > 1.5 / clip.info.rate:
        if given.strip():
            raise PlanError(f"that plan is for a video {seconds:.2f} s long, and this one is {mine:.2f} s: give the "
                            f"video that {plan.get('effect', 'the edit')} made")
        notes.append(f"the plan beside the video is for one {seconds:.2f} s long, so it was not used")
        return None, None
    return plan, path


def _plan_joins(plan: dict) -> list[tuple[float, float]]:
    """(where, seconds removed) for each join a clean-cut plan made."""
    if plan.get("effect") != "clean-cut" or not isinstance(plan.get("removals"), list):
        return []
    seconds_in = float((plan.get("summary") or {}).get("seconds_in", 0)) or max(
        (float(r["end"]) for r in plan["removals"]), default=0.0)
    removals = [(float(r["start"]), float(r["end"])) for r in plan["removals"]]
    tl = Timeline.keeping(seconds_in + 1e-3, removals)
    out = []
    for (a1, b1), (a2, b2), j in zip(tl.segments, tl.segments[1:], tl.joins):
        out.append((j, a2 - b1))
    return out


def _on_the_picture(changes: np.ndarray, frame: int) -> tuple[int, str]:
    """A planned join moved onto the frame where the picture really changes, when that is within two
    frames of it; a jump cut that barely changes the picture stays where it was planned."""
    n = len(changes)
    lo, hi = max(1, frame - 2), min(n - 1, frame + 2)
    if hi < lo:
        return frame, ""
    best = lo + int(np.argmax(changes[lo:hi + 1]))
    around = np.r_[changes[max(1, frame - 12):lo], changes[hi + 1:frame + 13]]
    usual = float(np.median(around)) if len(around) else 0.0
    if changes[best] >= max(3.0, 2.5 * (usual + 1.0)):
        return best, ("on the planned frame, where the picture changes" if best == frame
                      else f"moved {best - frame:+d} frame(s) to where the picture changes")
    return frame, "a jump cut that barely changes the picture"


def settled_frame(changes: np.ndarray, frame: int, fps: float, most: float = 0.12) -> int:
    """The first frame from `frame` on that belongs to the new shot. A clip joined by hand sometimes
    keeps a frame or two of what was cut out just after the cut: such a frame differs from the one
    after it about as much as the cut itself does (on take 33, 145 against the cut's 145, where
    somebody moving changes a frame by 2 to 6). Those are skipped, and hidden under the transition."""
    n = len(changes)
    lim = min(n - 2, frame + max(1, int(round(most * fps))))
    cut = float(changes[frame]) if 0 <= frame < n else 0.0
    k = frame
    while k < lim:
        after = changes[k + 2:k + 10]
        moving = float(np.median(after)) if len(after) else 0.0
        jump = float(changes[k + 1])
        if jump >= max(5.0, 0.45 * cut, 3.0 * (moving + 1.0)):
            k += 1
        else:
            break
    return k


def _words_around(words, at: float) -> tuple[Optional[object], Optional[object]]:
    before = [w for w in words if w.end <= at + 0.08]
    after = [w for w in words if w.start >= at - 0.08]
    return (before[-1] if before else None), (after[0] if after else None)


def _quiet_around(quiet, at: float) -> float:
    best = 0.0
    for q0, q1 in quiet:
        if q0 - 0.15 <= at <= q1 + 0.15:
            best = max(best, q1 - q0)
    return best


def find_joins(clip, opts: dict, notes: list[str], talking: bool) -> tuple[list[Join], str]:
    """Every join in the video, before any is chosen: from the edit's plan, else the picture."""
    info = clip.info
    fps = info.rate
    plan, path = _edit_plan(clip, opts["events"], notes)
    found: list[Join] = []
    source = ""
    if plan is not None and plan.get("effect") == "clean-cut":
        changes = shots.changes(info)
        for at, removed in _plan_joins(plan):
            frame = int(math.ceil(at * fps - 1e-6))
            frame, how = _on_the_picture(changes, frame)
            found.append(Join(frame / fps, frame, "the clean-cut plan", removed=removed, why=how))
        source = f"the cuts clean-cut made ({path.name}), each checked against the picture"
        if plan.get("options", {}).get("captions") and plan.get("cards"):
            notes.append("this video has captions drawn into it by clean-cut: a punch or a whip moves them with the "
                         "picture. For clean captions, run clean-cut with captions=off, then transitions, then captions")
    elif plan is not None and plan.get("effect") not in (None, NAME):
        notes.append(f"the plan beside the video is a {plan.get('effect')} plan, which makes no cuts; the cuts were "
                     "looked for in the picture")
    if not found:
        speaker = clip.face.found_share >= 0.5
        for t in shots.cuts(clip, speaker=speaker):
            frame = int(round(t * fps))
            found.append(Join(frame / fps, frame, "the picture"))
        source = "the cuts in the picture" + ("" if speaker else " (nobody talking in it, so found by the jump alone)")
    if talking:
        words, quiet = clip.words, clip.quiet
        for jn in found:
            last, nxt = _words_around(words, jn.at)
            if last is not None and nxt is not None:
                jn.before = " ".join(w.text for w in words if last.end - 1.2 <= w.end <= last.end + 1e-6)[-60:]
                jn.after = " ".join(w.text for w in words if nxt.start - 1e-6 <= w.start <= nxt.start + 1.0)[:60]
                jn.sentence = last.ends_sentence
            jn.pause = _quiet_around(quiet, jn.at) + jn.removed
    return found, source


def sentence_moments(clip) -> list[Join]:
    """In a talking video with no cut, the moments between its sentences: just before each new
    sentence begins, inside the pause. Never beside what the speaker took back (the shared picker's
    `moments.taken_back`: a first attempt said again, an aside about the recording), which is not
    an idea of its own."""
    fps = clip.info.rate
    words = clip.words
    back = moments.taken_back(words, clip.quiet)
    out = []
    for i, w in enumerate(words[:-1]):
        if not w.ends_sentence:
            continue
        nxt = words[i + 1]
        if w in back or nxt in back:
            continue
        at = max(w.end, nxt.start - 0.1)
        frame = int(math.ceil(at * fps - 1e-6))
        jn = Join(frame / fps, frame, "between sentences", cut=False,
                  before=" ".join(x.text for x in words[max(0, i - 4):i + 1]),
                  after=" ".join(x.text for x in words[i + 1:i + 5]), sentence=True,
                  pause=max(0.0, nxt.start - w.end))
        out.append(jn)
    return out


def choose(joins: list[Join], how: str, duration: float, notes: list[str]) -> list[Join]:
    """The joins that get a transition: every one, the ones between ideas, or the times given."""
    inside = [j for j in joins if EDGE <= j.at <= duration - EDGE]
    if how == "every":
        for j in inside:
            j.chosen, j.why = True, j.why or "every join"
        return inside
    if how == "between-ideas":
        cands = [j for j in inside if j.sentence]
        for j in inside:
            if not j.sentence and j.before:
                j.why = "mid-sentence: " + (j.why or "the words run on across it")
        picked: list[Join] = []
        for j in sorted(cands, key=lambda j: (-j.pause, j.at)):  # the clearest breaks first
            if all(abs(j.at - p.at) >= APART for p in picked):
                picked.append(j)
                j.chosen, j.why = True, "a sentence ends here"
            else:
                j.why = f"a sentence ends here, but within {APART:g} s of a clearer break"
        return sorted(picked, key=lambda j: j.at)
    # seconds given: each on the nearest join within a quarter second, else on that moment
    out = []
    for t in [float(x) for x in how.split(",")]:
        near = [j for j in joins if abs(j.at - t) <= 0.25]
        if near:
            j = min(near, key=lambda j: abs(j.at - t))
            j.chosen, j.why = True, f"given ({t:g} s)"
        else:
            j = Join(t, 0, "given", cut=False, chosen=True, why=f"given ({t:g} s): no cut there, so on that moment")
        if all(abs(j.at - o.at) > 1e-3 for o in out):
            out.append(j)
    return sorted(out, key=lambda j: j.at)


# ---- several clips joined ---------------------------------------------------------------------------

@dataclass
class Piece:
    path: str
    info: video.VideoInfo
    start: float
    end: float

    @property
    def seconds(self) -> float:
        return self.end - self.start


_RANGE = re.compile(r"\s+(\d+(?:\.\d+)?|\d+:\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?|\d+:\d+(?:\.\d+)?)\s*s?\s*$")


def parse_clips(text: str, first: str) -> list[Piece]:
    """The clips to join, in order: the video given first, then `text`, a folder (its videos by
    name) or paths separated by ; or new lines, each optionally followed by the seconds to use."""
    from . import PlanError

    text = text.strip()
    items: list[tuple[str, Optional[tuple[float, float]]]] = []
    folder = Path(text).expanduser()
    if folder.is_dir():
        items = [(str(p), None) for p in sorted(folder.iterdir()) if p.suffix.lower() in VIDEO_EXTS
                 and not p.name.startswith(".")]
        if not items:
            raise PlanError(f"there are no videos in {folder}")
    else:
        parts = [p.strip().strip("\"'") for p in re.split(r"[;\n]", text) if p.strip()]
        if len(parts) == 1 and "," in parts[0]:
            commas = [p.strip().strip("\"'") for p in parts[0].split(",") if p.strip()]
            if all(Path(_RANGE.sub("", c)).expanduser().exists() for c in commas):
                parts = commas
        for p in parts:
            m = _RANGE.search(" " + p)
            if m:
                path = (" " + p)[:m.start()].strip()
                items.append((path, (_clock(m.group(1)), _clock(m.group(2)))))
            else:
                items.append((p, None))
    first_path = str(Path(first).expanduser().resolve())
    if all(str(Path(p).expanduser().resolve()) != first_path for p, _ in items):
        items.insert(0, (first_path, None))
    pieces = []
    for p, rng in items:
        path = Path(p).expanduser()
        if not path.exists():
            raise PlanError(f"there is no clip at {p}")
        info = video.probe(path)
        a, b = (0.0, info.duration) if rng is None else rng
        b = min(b, info.duration)
        if not 0 <= a < b:
            raise PlanError(f"{path.name}: {a:g}-{b:g} s is not inside the clip (0-{info.duration:.2f} s)")
        if b - a < 0.5:
            raise PlanError(f"{path.name}: {b - a:.2f} s is too short to join (half a second at least)")
        pieces.append(Piece(str(path.resolve()), info, a, b))
    if len(pieces) < 2:
        raise PlanError("joining needs two clips at least")
    return pieces


def film_shape(pieces: list[Piece]) -> tuple[tuple[int, int], Fraction]:
    """The film's size and frame rate: the first clip's shape, as large as the largest clip can fill
    (never past 1080 by 1920), at the lowest frame rate among them, so no clip shows a frame twice."""
    first = pieces[0].info
    aspect = first.width / first.height
    best_w = 0.0
    for p in pieces:
        w = min(p.info.width, p.info.height * aspect)
        best_w = max(best_w, w)
    size = framing_mod.output_size(best_w, best_w / aspect)
    rate = min((p.info.fps for p in pieces), key=float)
    return size, rate


class Source:
    """One clip's frames at the film's rate, filling the film's frame (cropped, never stretched),
    from `start` seconds: `at(i)` is its i-th frame, read in order, and found again by seeking when
    asked for out of order."""

    def __init__(self, piece: Piece, size: tuple[int, int], rate: Fraction):
        self.piece = piece
        self.rate = rate
        self.info = replace(piece.info, fps=rate, frames=max(1, int(round(piece.info.duration * float(rate)))))
        W, H = size
        k = max(W / piece.info.width, H / piece.info.height)
        self.read = (max(W, int(math.ceil(piece.info.width * k / 2)) * 2), max(H, int(math.ceil(piece.info.height * k / 2)) * 2))
        self.x0 = (self.read[0] - W) // 2
        self.y0 = int(round((self.read[1] - H) * 0.45))
        self.size = size
        self.first = int(round(piece.start * float(rate)))
        self.count = max(1, int(round(piece.seconds * float(rate))))
        self._reader = None
        self._iter = None
        self._n = -1
        self._frame: Optional[np.ndarray] = None

    def _open(self, g: int) -> None:
        self.close()
        self._reader = video.FrameReader(self.info, size=self.read, start=max(0.0, (g - 0.25) / float(self.rate)))
        self._iter = iter(self._reader)
        self._n, self._frame = g - 1, None

    def at(self, i: int) -> np.ndarray:
        i = int(min(self.count - 1, max(0, i)))
        g = min(self.info.frames - 1, self.first + i)
        if g == self._n and self._frame is not None:
            return self._frame
        if self._iter is None or g < self._n or g > self._n + int(float(self.rate)):
            self._open(g)
        while self._n < g:
            nxt = next(self._iter, None)
            if nxt is None:
                break
            W, H = self.size
            self._frame = np.ascontiguousarray(nxt[self.y0:self.y0 + H, self.x0:self.x0 + W])
            self._n += 1
        if self._frame is None:
            raise video.MediaError(f"could not read the picture of {self.piece.path}")
        self._n = g
        return self._frame

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
        self._reader, self._iter = None, None


@dataclass
class Film:
    """Several clips laid end to end, overlapping where a transition shows both at once."""
    pieces: list[Piece]
    sources: list[Source]
    starts: list[int]  # each clip's first frame in the film
    counts: list[int]  # its frames
    overlaps: list[int]  # frames each join overlaps (one fewer than the clips)
    size: tuple[int, int]
    rate: Fraction

    @property
    def frames(self) -> int:
        return self.starts[-1] + self.counts[-1]

    def owner(self, j: int) -> int:
        """The clip on screen at frame j (in an overlap, the one going out)."""
        for k in range(len(self.pieces)):
            if j < self.starts[k] + self.counts[k]:
                return k
        return len(self.pieces) - 1

    def plain(self, j: int) -> np.ndarray:
        k = self.owner(j)
        return self.sources[k].at(j - self.starts[k])


def even_frames(seconds: float, fps: float) -> int:
    """An overlap as a whole, even number of frames (so its middle falls on a frame boundary), never
    longer than asked: an overlap is capped by the quiet between two voices, and rounding up would
    let a word under it."""
    return 2 * int(math.floor(seconds * fps / 2 + 1e-6))


def lay_film(pieces: list[Piece], overlaps_s: list[float], size, rate: Fraction) -> Film:
    fps = float(rate)
    counts = [max(1, int(round(p.seconds * fps))) for p in pieces]
    overlaps = [even_frames(o, fps) for o in overlaps_s]
    starts = [0]
    for k in range(1, len(pieces)):
        starts.append(starts[-1] + counts[k - 1] - overlaps[k - 1])
    sources = [Source(p, size, rate) for p in pieces]
    return Film(pieces, sources, starts, counts, overlaps, size, rate)


@dataclass
class Voice:
    """What a clip's sound says about speech: whether it holds a voice, and the quiet before its first
    word and after its last (seconds). Wind, traffic and silence hold no voice."""
    voiced: bool
    lead: float
    trail: float


def hear(samples: np.ndarray) -> Voice:
    if not len(samples) or float(np.abs(samples).max()) < 1e-4:
        return Voice(False, math.inf, math.inf)
    env = audio.envelope(video.mono(samples, RATE, 16000), 16000)
    if env.speech - env.floor < 20.0:  # a voice rises well clear of the clip's own floor
        return Voice(False, math.inf, math.inf)
    length = len(env.db) * env.hop
    runs = audio.quiet_runs(env)
    lead = runs[0][1] if runs and runs[0][0] <= env.hop else 0.0
    trail = length - runs[-1][0] if runs and runs[-1][1] >= length - env.hop else 0.0
    return Voice(True, max(0.0, lead - 0.03), max(0.0, trail - 0.03))  # a breath of margin on each word


def quiet_room(a: Voice, b: Voice) -> float:
    """How long two clips may overlap with no word under the overlap: all of it when neither talks,
    the quiet after the last word going out and before the first word coming in when they do."""
    if a.voiced and b.voiced:
        return a.trail + b.lead
    if a.voiced:
        return a.trail
    if b.voiced:
        return b.lead
    return math.inf


def clip_sounds(pieces: list[Piece]) -> list[np.ndarray]:
    out = []
    for p in pieces:
        s = video.read_audio(p.info)
        out.append(s[int(round(p.start * RATE)):int(round(p.end * RATE))])
    return out


def film_sound(film: Film, parts: list[np.ndarray], voices: list[Voice]) -> tuple[np.ndarray, list[dict]]:
    """The clips' own sounds laid as their pictures are. Where two pictures overlap, the sounds
    cross over the same frames; when both clips talk, the sound switches between the last word going
    out and the first word coming in (the overlap is never longer than the quiet between them), so a
    voice is never doubled or cut. A plain cut fades over 10 ms either side so it never clicks."""
    fps = float(film.rate)
    total = int(round(film.frames / fps * RATE))
    out = np.zeros((total, 2), np.float32)
    crossings = []
    gains = [np.ones(len(x), np.float32) for x in parts]
    for k, o in enumerate(film.overlaps):
        a, b = parts[k], parts[k + 1]
        n_o = min(len(a), len(b), int(round(o / fps * RATE)))
        if n_o > 0:
            va, vb = voices[k], voices[k + 1]
            if va.voiced and vb.voiced:
                # A's last word ends (o - trail) into the overlap; B's first word starts at its lead.
                lo_t = max(0.0, n_o / RATE - va.trail)
                hi_t = min(n_o / RATE, vb.lead)
                both = audio.envelope(video.mono(a[len(a) - n_o:] + b[:n_o], RATE, 16000), 16000)
                at = both.quietest(lo_t, hi_t) if hi_t > lo_t + 0.02 else 0.5 * (lo_t + hi_t)
                at = min(max(at, lo_t), hi_t)
                c = int(round(at * RATE))
                # the crossing stays inside the quiet: 60 ms when there is room, 10 ms at the least
                half = int(max(0.005, min(0.03, at - lo_t, hi_t - at)) * RATE)
                lo, hi = max(0, c - half), min(n_o, c + half)
                ga = np.ones(n_o, np.float32)
                ga[hi:] = 0.0
                if hi > lo:
                    ga[lo:hi] = np.cos(np.linspace(0, np.pi / 2, hi - lo, dtype=np.float32))
                gb = np.sqrt(np.clip(1.0 - ga * ga, 0, 1)).astype(np.float32)
                gains[k][len(a) - n_o:] *= ga
                gains[k + 1][:n_o] *= gb
                crossings.append({"join": k + 1, "how": f"both clips talk: the sound switches {at:.2f} s into the "
                                  f"{n_o / RATE:.2f} s overlap, between the last word out and the first word in"})
            else:
                ramp = np.linspace(0, np.pi / 2, n_o, dtype=np.float32)
                gains[k][len(a) - n_o:] *= np.cos(ramp)
                gains[k + 1][:n_o] *= np.sin(ramp)
                said = " (no word under it)" if va.voiced or vb.voiced else ""
                crossings.append({"join": k + 1, "how": f"crossed over the {n_o / RATE:.2f} s the pictures overlap{said}"})
        else:
            m = int(0.01 * RATE)
            if len(a) > m and len(b) > m:
                gains[k][len(a) - m:] *= np.cos(np.linspace(0, np.pi / 2, m, dtype=np.float32))
                gains[k + 1][:m] *= np.sin(np.linspace(0, np.pi / 2, m, dtype=np.float32))
            crossings.append({"join": k + 1, "how": "a cut in the sound too, faded over 10 ms either side"})
    for k, x in enumerate(parts):
        at = int(round(film.starts[k] / fps * RATE))
        y = x * gains[k][:, None]
        n = min(len(y), total - at)
        if n > 0:
            out[at:at + n] += y[:n]
    return out, crossings


class FilmClip:
    """The joined film as the renderer sees a clip: its picture drawn by `Film.plain`, its sound the
    clips' sounds crossed. Words, faces and the person are not looked for in clips joined this way."""

    def __init__(self, first, film: Film, sound_: np.ndarray):
        self.path = first.path
        self.base = first
        fps = float(film.rate)
        self.info = video.VideoInfo(path=first.path, width=film.size[0], height=film.size[1], fps=film.rate,
                                    duration=film.frames / fps, frames=film.frames, has_audio=True, audio_offset=0.0,
                                    color_matrix="bt709", picture=film.plain)
        self.sound = sound_
        self.words: list = []
        self.unsure: list = []
        self.quiet: list = []


# ---- the plan: transitions on frames ---------------------------------------------------------------

def _looks(opts: dict, talking: bool, clips: bool) -> list[str]:
    if opts["look"] != "auto":
        return opts["look"].split(",")
    return ["punch"] if talking and not clips else ["crossfade"]


def place(chosen: list[tuple[float, int, str, str]], looks_dur: list[tuple[str, float]], fps: float, n: int,
          colour: str, notes: list[str]) -> list[Transition]:
    """Each chosen join (at, frame, source, note) with its look and length, as frames: the share
    before the join and the rest after it, kept inside the film and clear of its neighbours."""
    out: list[Transition] = []
    for k, ((at, frame, source, why), (look, d)) in enumerate(zip(chosen, looks_dur)):
        total = max(2, int(round(d * fps)))
        before = max(1, int(round(total * BEFORE[look])))
        after = max(1, total - before)
        out.append(Transition(k + 1, look, at, frame, d, frame - before, frame + after - 1, source, colour=colour,
                              direction=1 if k % 2 == 0 else -1, note=why))
    for k, tr in enumerate(out):  # keep inside the film
        if tr.first < 0 or tr.last > n - 1:
            tr.first, tr.last = max(0, tr.first), min(n - 1, tr.last)
            tr.note = (tr.note + "; " if tr.note else "") + "shortened to fit the film"
    for a, b in zip(out, out[1:]):  # and clear of each other
        if a.last >= b.first:
            gap = b.frame - a.frame
            room_a = max(1, int((a.last - a.frame + 1) * gap / max(1, (a.last - a.frame + 1) + (b.frame - b.first))))
            a.last = min(a.last, a.frame + room_a - 1)
            b.first = max(b.first, a.last + 1)
            for t in (a, b):
                t.note = (t.note + "; " if t.note else "") + "shortened to keep clear of its neighbour"
    for tr in out:
        tr.duration = tr.frames / fps
    return out


# ---- drawing: one frame at a time -----------------------------------------------------------------

class Painter:
    """Draws every frame of the output: the plain picture where nothing happens, a look inside a
    transition's frames, and the punch's framing from its first join on. Inside one video the other
    side of a join is its held frame; between clips it is the other clip's own frame."""

    def __init__(self, transitions: list[Transition], fps: float, size: tuple[int, int],
                 raw: Callable[[int], np.ndarray], incoming: Optional[Callable[[Transition, int], np.ndarray]],
                 face_at: Optional[Callable[[int], Optional[Rect]]], n: int, notes: list[str],
                 shots: Optional[list[int]] = None):
        self.shots = shots or []  # between clips: the first frame of each clip after the first
        self.trs = transitions
        self.fps = fps
        self.size = size
        self.raw = raw  # the plain picture at frame j (before any punch)
        self.incoming = incoming  # between clips: the clip coming in, playing, at frame j of an overlap
        self.face_at = face_at
        self.n = n
        self.notes = notes
        self._held: dict = {}
        self.stretches, self.moves = self._punch_windows()

    # -- the punch: one framing per stretch between punches
    def _punch_windows(self) -> tuple[list[tuple[int, tuple]], list[tuple[Transition, tuple, tuple]]]:
        """The framing of each stretch, as (first frame, window), and each punch as (transition, the
        window it leaves, the window it lands on). Punches take turns: in, then back out, then in.
        Between clips each clip is its own shot, so a closer framing ends with its clip: the next
        clip starts on the whole frame (the change hidden by the join it starts on)."""
        W, H = self.size
        full = (0.0, 0.0, float(W), float(H))
        stretches: list[tuple[int, tuple]] = [(0, full)]
        moves: list[tuple[Transition, tuple, tuple]] = []
        punches = {t.frame: t for t in self.trs if t.look == "punch"}
        events = sorted(set(punches) | set(self.shots))
        zoom = 1.0
        for i, frame in enumerate(events):
            tr = punches.get(frame)
            if tr is None:  # a new clip, not punched: back to its whole frame
                if zoom > 1.0:
                    stretches.append((frame, full))
                    zoom = 1.0
                continue
            end = events[i + 1] - 1 if i + 1 < len(events) else self.n - 1
            if zoom > 1.0 and frame not in self.shots:  # inside one shot: back out to the whole frame
                tr.zoom_from, tr.zoom_to = zoom, 1.0
                moves.append((tr, stretches[-1][1], full))
                stretches.append((tr.frame, full))
                zoom = 1.0
                continue
            head, centre = self._head_over(tr.frame, end)
            want = PUNCH_ZOOM
            if frame not in self.shots:
                # Judged against what was on screen just before: when the new shot already shows the
                # face smaller (the speaker sat back between takes), a fixed zoom would not read as
                # going in at all.
                shown = self._head_width(max(0, tr.frame - int(round(0.5 * self.fps))), tr.frame - 1) * zoom
                raw = self._head_width(tr.frame, end)
                if shown > 0 and raw > 0:
                    want = min(PUNCH_MOST, max(PUNCH_ZOOM, PUNCH_GAIN * shown / raw))
            win, z, note = punch_window(self.size, want, head, centre)
            if z < 1.04:
                tr.punch_note = (note or "no room round the head") + ": no punch in, the sound only"
                tr.note = (tr.note + "; " if tr.note else "") + tr.punch_note
                tr.zoom_from = tr.zoom_to = zoom
                continue
            if note:
                tr.punch_note = note
                tr.note = (tr.note + "; " if tr.note else "") + note
            # a new clip punched: it lands closer from its own whole frame, not from the last clip's framing
            leaving = full if frame in self.shots else stretches[-1][1]
            tr.zoom_from, tr.zoom_to = (1.0 if frame in self.shots else zoom), z
            moves.append((tr, leaving, win))
            stretches.append((tr.frame, win))
            zoom = z
        return stretches, moves

    def _head_width(self, first: int, last: int) -> float:
        """The head's usual width over frames first..last (0 when no face is known)."""
        if self.face_at is None or last < first:
            return 0.0
        ws = [h.w for j in range(first, last + 1, max(1, (last - first) // 30 or 1)) if (h := self.face_at(j)) is not None]
        return float(np.median(ws)) if ws else 0.0

    def _head_over(self, first: int, last: int) -> tuple[Optional[Rect], Optional[tuple[float, float]]]:
        if self.face_at is None:
            return None, None
        heads = [h for j in range(first, last + 1, max(1, (last - first) // 40 or 1)) if (h := self.face_at(j)) is not None]
        if not heads:
            return None, None
        union = Rect.around(heads)
        cx = float(np.median([h.cx for h in heads]))
        cy = float(np.median([h.y0 + 0.55 * h.h for h in heads]))
        return union, (cx, cy)

    def _window(self, j: int) -> tuple[tuple, list[tuple]]:
        """The framing at frame j, and the framings it sweeps through during the frame (for the blur)."""
        for tr, w0, w1 in self.moves:
            if tr.first <= j <= tr.last and not (j < tr.frame and tr.frame in self.shots):
                # (a clip coming in lands closer on its own first frame; the clip going out is left alone)
                d = tr.frames / self.fps

                def at(jj: int):
                    return _between(w0, w1, punch_progress((jj - tr.frame) / self.fps, d), self.size)

                here, prev = at(j), at(j - 1)
                moving = abs(here[2] - prev[2]) > 0.006 * self.size[0]
                # a half-open shutter: the frame is smeared over the second half of its own movement
                return here, ([_between(prev, here, s) for s in np.linspace(0.5, 0.95, 7)] if moving else [])
        window = self.stretches[0][1]
        for start, w in self.stretches:
            if start <= j:
                window = w
        return window, []

    def framed(self, img: np.ndarray, j: int) -> np.ndarray:
        if not self.moves:
            return img
        win, sweep = self._window(j)
        out = _crop(img, win)
        if sweep:  # a zoom blur across the frame's own movement
            acc = out.astype(np.float32)
            for w in sweep:
                acc += _crop(img, w).astype(np.float32)
            out = (acc / (len(sweep) + 1)).astype(np.uint8)
        return out

    # -- the held frames of a join inside one video
    def held(self, tr: Transition, which: str) -> np.ndarray:
        key = (tr.n, which)
        if key not in self._held:
            j = tr.frame - 1 if which == "out" else max(tr.frame, tr.settle)
            self._held[key] = self.framed(self.raw(j), j)
            if len(self._held) > 8:
                for old in list(self._held)[:2]:
                    del self._held[old]
        return self._held[key]

    def transition_at(self, j: int) -> Optional[Transition]:
        for tr in self.trs:
            if tr.first <= j <= tr.last and tr.look != "punch":
                return tr
        return None

    def draw(self, frame: np.ndarray, j: int, t: float) -> None:
        tr = self.transition_at(j)
        for t_ in self.trs:  # a stray frame of what was cut out, just after a cut: the new shot instead
            if t_.frame <= j < t_.settle:
                frame[:] = self.raw(t_.settle)
                break
        base = self.framed(frame, j)
        if tr is None:
            if base is not frame:
                frame[:] = base
            return
        if tr.overlap:  # between clips, both playing
            a, b = base, self.framed(self.incoming(tr, j), j)
            inc = j >= tr.frame
        elif j < tr.frame:
            a, b, inc = base, self.held(tr, "in"), False
            if j == tr.frame - 1:  # the last frame of the shot going out: kept to hold after the join
                self._held[(tr.n, "out")] = base.copy()
        else:
            a, b, inc = self.held(tr, "out"), base, True
        m = Moment(tr.u(j), a, b, inc, j - tr.first, tr.frames)
        frame[:] = LOOK_DRAW[tr.look](m, tr)


# ---- the sound laid under the picture ---------------------------------------------------------------

@dataclass
class Sounds:
    events: list[dict]  # {tr, kind, samples, anchor, gain_db, at}
    reference: float
    limited_db: float = 0.0
    stem: Optional[np.ndarray] = None

    def mix(self, base: np.ndarray, rate: int) -> np.ndarray:
        stem = np.zeros_like(base)
        for e in self.events:
            sound.lay(stem, e["samples"], e["at"], rate, anchor=e["anchor"], level=sound.gain(e["gain_db"]))
        out = base + stem
        if stem.any():
            ceiling = max(-1.0, loudness.true_peak_db(base) + 0.05)
            if loudness.true_peak_db(out) > ceiling:
                out, self.limited_db = loudness.limit(out, ceiling)
        self.stem = stem
        return out.astype(np.float32)


NO_VOICE = -16.0  # LUFS: what the sounds are set against in a film nobody talks in


def lay_sounds(trs: list[Transition], fps: float, base: np.ndarray, talking: bool) -> Sounds:
    """Each transition's sound, its landing on the join, its loudest tenth of a second set under the
    voice. In a film nobody talks in (scenery, wind) the sounds are what is heard at the joins, so
    they are set against a fixed level instead of under the wind."""
    voice = sound.integrated(sound.blocks(base)) if len(base) and np.abs(base).max() > 1e-5 else sound.SILENCE_DB
    reference = voice if talking and voice > -60 else NO_VOICE
    events = []
    for tr in trs:
        kind, x, anchor = make_sound(tr, fps)
        loud = sound.loudest(x) if len(x) else sound.SILENCE_DB
        under = UNDER_VOICE + (3.0 if kind == "hit" and tr.zoom_to < tr.zoom_from else 0.0)  # going back out: softer
        gain_db = (reference - under) - loud
        at = tr.frame / fps  # on the start of the new shot's first frame
        events.append({"tr": tr, "kind": kind, "samples": x, "anchor": anchor, "gain_db": gain_db, "at": at})
        tr.sound = {"kind": kind, "lands": round(at, 4), "under_voice_db": under}
    return Sounds(events, reference)


def _lag(a: np.ndarray, b: np.ndarray, most: int = 4800) -> int:
    """How many samples `b` runs late against `a`."""
    n = min(len(a), len(b), 10 * RATE)
    if n < 2 * most:
        return 0
    size = 1 << int(np.ceil(np.log2(2 * n)))
    xc = np.fft.irfft(np.fft.rfft(b[:n], size) * np.conj(np.fft.rfft(a[:n], size)), size)
    lags = np.concatenate([xc[:most + 1], xc[-most:]])
    k = int(np.argmax(lags))
    return k if k <= most else k - len(lags)


def check_sounds(out: Path, base: np.ndarray, sfx: Sounds, fps: float) -> tuple[list[dict], float]:
    """Each sound found again in the encoded file (the film's own sound taken away) and how far its
    landing is from its join, and how far the file's sound runs from the sound it was given (ms)."""
    y = sound.read(out)
    n = min(len(base), len(y))
    lag = _lag(base[:n].mean(axis=1), y[:n].mean(axis=1))
    shifted = np.zeros((n, base.shape[1]), np.float32)
    if lag >= 0:
        shifted[lag:] = base[:n - lag]
    else:
        shifted[:n + lag] = base[-lag:n]
    res = (y[:n] - shifted).mean(axis=1)
    rows = []
    for e in sfx.events:
        tmpl = e["samples"].mean(axis=1) * sound.gain(e["gain_db"])
        start = int(round((e["at"] - e["anchor"]) * RATE))
        a, b = max(0, start - int(0.04 * RATE)), min(len(res), start + len(tmpl) + int(0.04 * RATE))
        found = None
        if b - a > len(tmpl) // 3 and len(tmpl) > 16:
            seg = res[a:b]
            t = tmpl[: max(1, min(len(tmpl), b - a - 1))]
            xc = np.correlate(seg, t, mode="valid")
            if len(xc):
                found = (a + int(np.argmax(xc))) / RATE + e["anchor"]
        tr = e["tr"]
        rows.append({"n": tr.n, "look": tr.look, "sound": e["kind"], "join_at": round(e["at"], 4),
                     "lands_in_file": None if found is None else round(found, 4),
                     "off_ms": None if found is None else round((found - e["at"]) * 1000, 1)})
    return rows, lag * 1000.0 / RATE


# ---- the sheet: each transition frame by frame, over its sound ------------------------------------

def strip_frames(tr: Transition) -> list[int]:
    """Six frames across a transition: its start, on the way, the last frame of the old shot, the
    first of the new, on the way out, its end."""
    a, j, z = tr.first, tr.frame, tr.last
    if tr.look == "punch":
        a, z = max(0, j - 3), j + max(4, tr.last - j)
    picks = [a, (a + j - 1) // 2, j - 1, j, (j + z) // 2, z]
    out = []
    for p in picks:
        if not out or p > out[-1]:
            out.append(p)
    return out


def draw_sheet(rows: list[tuple[Transition, list[tuple[int, np.ndarray]]]], fps: float, own: Optional[np.ndarray],
               added: Optional[np.ndarray], title: str, path: Path, lines: list[str]) -> Path:
    """Each transition as a row of frames, over a lane of its sound in decibels: the film's own sound
    in blue and the transition's sound in orange, with the join in yellow."""
    from PIL import Image, ImageDraw, ImageFont

    def font(px: int, bold: bool = False):
        name = "Poppins-SemiBold.ttf" if bold else "Poppins-Regular.ttf"
        try:
            return ImageFont.truetype(str(PACK / "fonts/Poppins" / name), px)
        except OSError:
            return ImageFont.load_default()

    tile_h = 300
    pad = 10
    rows_img = []
    width = 0
    for tr, frames in rows:
        tiles = []
        for j, img in frames:
            im = Image.fromarray(img)
            im = im.resize((max(1, int(round(im.width * tile_h / im.height))), tile_h), Image.LANCZOS)
            cell = Image.new("RGB", (im.width, tile_h + 34), (22, 22, 22))
            cell.paste(im, (0, 0))
            d = ImageDraw.Draw(cell)
            mark = "  JOIN" if j == tr.frame else ""
            d.text((6, tile_h + 6), f"{j / fps:.2f} s{mark}", fill=(255, 214, 90) if mark else (220, 220, 220), font=font(17, bool(mark)))
            if j == tr.frame:
                d.rectangle([0, 0, cell.width - 1, tile_h - 1], outline=(255, 214, 90), width=3)
            tiles.append(cell)
        strip_w = sum(c.width for c in tiles) + 6 * (len(tiles) - 1)
        wave_h = 64 if own is not None else 0
        head_h = 34
        row = Image.new("RGB", (strip_w, head_h + tile_h + 34 + wave_h), (40, 40, 40))
        d = ImageDraw.Draw(row)
        label = f"{tr.n} · {tr.look} · join {tr.at:.2f} s · {tr.duration:.2f} s · {tr.source}"
        if tr.sound:
            label += f" · {tr.sound['kind']}"
        d.text((4, 6), label, fill=(240, 240, 240), font=font(17, True))
        x = 0
        for c in tiles:
            row.paste(c, (x, head_h))
            x += c.width + 6
        if own is not None and frames:
            t0, t1 = frames[0][0] / fps - 0.2, (frames[-1][0] + 1) / fps + 0.2
            a, b = max(0, int(t0 * RATE)), min(len(own), int(t1 * RATE))
            top = head_h + tile_h + 34
            base_y = top + wave_h - 4
            cols = strip_w

            def lane(x: np.ndarray) -> np.ndarray:
                seg = np.abs(x[a:b]).max(axis=1)
                idx = np.linspace(0, len(seg), cols + 1).astype(int)
                pk = np.array([seg[idx[i]:max(idx[i] + 1, idx[i + 1])].max() for i in range(cols)])
                return np.clip((20 * np.log10(np.maximum(pk, 1e-6)) + 54.0) / 54.0, 0, 1)  # -54..0 dBFS

            if b > a:
                for arr, col in ((own, (110, 175, 245)), (added, (255, 150, 60))):
                    if arr is None:
                        continue
                    hts = lane(arr) * (wave_h - 8)
                    for cx in range(cols):
                        if hts[cx] >= 1:
                            d.line([(cx, base_y - int(hts[cx])), (cx, base_y)], fill=col)
                jx = int((tr.at - t0) / (t1 - t0) * cols)
                d.line([(jx, top), (jx, top + wave_h)], fill=(255, 214, 90), width=2)
        rows_img.append(row)
        width = max(width, row.width)
    text_h = 30 * (len(lines) + 1) + 10
    sheet = Image.new("RGB", (width + 2 * pad, text_h + sum(r.height + pad for r in rows_img) + pad), (30, 30, 30))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 8), title, fill=(255, 255, 255), font=font(20, True))
    for i, ln in enumerate(lines):
        d.text((pad, 38 + 28 * i), ln, fill=(200, 200, 200), font=font(16))
    y = text_h
    for r in rows_img:
        sheet.paste(r, (pad, y))
        y += r.height + pad
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=86)
    return path


def frames_from_file(path: Path, wanted: list[int]) -> dict[int, np.ndarray]:
    """The frames `wanted` (by number) of a rendered file, each run of them read in one pass."""
    info = video.probe(path)
    out: dict[int, np.ndarray] = {}
    wanted = sorted(set(wanted))
    groups: list[list[int]] = []
    for j in wanted:
        if groups and j - groups[-1][-1] <= 12:
            groups[-1].append(j)
        else:
            groups.append([j])
    for g in groups:
        start = g[0]
        reader = video.FrameReader(info, start=max(0.0, (start - 0.25) / info.rate),
                                   duration=(g[-1] - start + 2) / info.rate)
        for k, f in enumerate(reader):
            j = start + k
            if j in g:
                out[j] = f.copy()
            if j >= g[-1]:
                break
        reader.close()
    return out


# ---- the effect ------------------------------------------------------------------------------------

def _from_plan(plan: dict) -> list[dict]:
    from . import PlanError

    items = plan.get("transitions")
    if not isinstance(items, list) or not items:
        raise PlanError("the plan has no transitions; run once without --plan to get one to edit")
    out = []
    for it in items:
        try:
            at, look = float(it["at"]), str(it["look"])
        except (KeyError, TypeError, ValueError):
            raise PlanError(f"each transition needs an at (seconds) and a look: {it!r}") from None
        look = ALIASES.get(look, look)
        if look not in LOOKS:
            raise PlanError(f"no look called {look!r}; the looks are {', '.join(LOOKS)}")
        d = float(it.get("duration", AUTO[look][0]))
        if not 0.08 <= d <= 3.0:
            raise PlanError(f"a transition lasts 0.1 to 3 seconds (got {d:g})")
        out.append({"at": at, "look": look, "duration": d, "color": str(it.get("color", "auto"))})
    return sorted(out, key=lambda x: x["at"])


def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from . import Job, PlanError

    notes: list[str] = []
    joining = bool(opts["clips"].strip()) or (plan is not None and plan.get("mode") == "clips")
    edited = _from_plan(plan) if plan is not None else None
    colour = opts["color"]

    if joining:
        clip_text = opts["clips"]
        if plan is not None and isinstance(plan.get("clips"), list):
            clip_text = "; ".join(f"{c['path']} {c['from']}-{c['to']}" for c in plan["clips"])
        pieces = parse_clips(clip_text, clip.path)
        size, rate = film_shape(pieces)
        fps = float(rate)
        n_joins = len(pieces) - 1
        looks = _looks(opts, False, True)
        if edited is not None:
            per = [(e["look"], e["duration"]) for e in edited][:n_joins]
            per += [(looks[k % len(looks)], AUTO[looks[k % len(looks)]][1]) for k in range(len(per), n_joins)]
            chosen_k = list(range(n_joins))
        else:
            chosen_k = list(range(n_joins))
            if opts["joins"] not in ("auto", "every", "between-ideas"):
                # seconds given between clips: the joins nearest them, as if the clips were simply laid end to end
                ends = np.cumsum([p.seconds for p in pieces])[:-1]
                chosen_k = sorted({int(np.argmin(np.abs(ends - float(t)))) for t in opts["joins"].split(",")})
            elif opts["joins"] == "between-ideas":
                notes.append("between clips every join is a change of shot, so every join has one")
            per = []
            for i, k in enumerate(chosen_k):
                look = looks[i % len(looks)]
                d = opts["duration"] if opts["duration"] != "auto" else AUTO[look][1]
                per.append((look, float(d)))
        parts = clip_sounds(pieces)
        voices = [hear(x) for x in parts]
        overl = [0.0] * n_joins
        for i, ((look, d), k) in enumerate(zip(per, chosen_k)):
            if needs_both(look, colour):
                most = 0.45 * min(pieces[k].seconds, pieces[k + 1].seconds)
                if d > most:
                    notes.append(f"join {k + 1}: the {look} is shortened to {most:.2f} s, the most these two clips allow")
                    d = most
                room = quiet_room(voices[k], voices[k + 1])
                if d > room:
                    if room >= 4 / fps:
                        notes.append(f"join {k + 1}: the voices leave {room:.2f} s of quiet between the last word out and "
                                     f"the first word in, so the {look} overlaps that much and no more")
                        d = room
                    else:
                        notes.append(f"join {k + 1}: a word runs right up to the cut on both sides, so the {look} holds "
                                     "a frame either side instead of overlapping, and the sound cuts between the words")
                        d = 0.0
                    if d > 0:  # with no overlap the held frames keep the length asked for
                        per[i] = (look, d)
                overl[k] = d
        film = lay_film(pieces, overl, size, rate)
        base, crossings = film_sound(film, parts, voices)
        fclip = FilmClip(clip, film, base)
        n = film.frames
        chosen = []
        looks_dur = []
        for (look, d), k in zip(per, chosen_k):
            if film.overlaps[k]:
                frame = film.starts[k + 1] + film.overlaps[k] // 2
                d = film.overlaps[k] / fps
            else:
                frame = film.starts[k + 1]
            chosen.append((frame / fps, frame, f"between clips {k + 1} and {k + 2}", ""))
            looks_dur.append((look, d))
        trs = place(chosen, looks_dur, fps, n, colour, notes)
        for tr, k in zip(trs, chosen_k):
            if film.overlaps[k]:
                tr.overlap = film.overlaps[k]
                tr.first, tr.last = film.starts[k + 1], film.starts[k + 1] + film.overlaps[k] - 1
                tr.duration = tr.frames / fps

        def incoming(tr: Transition, j: int) -> np.ndarray:
            k = [i for i in range(len(film.pieces) - 1) if film.starts[i + 1] <= tr.frame <= film.starts[i + 1] + film.overlaps[i]][0]
            return film.sources[k + 1].at(j - film.starts[k + 1])

        job_clip = fclip
        fr = Framing(size[0], size[1], size[0], size[1], None, "the first clip's shape, every clip filling it")
        painter = Painter(trs, fps, size, film.plain, incoming, None, n, notes, shots=film.starts[1:])
        joins_info: list[dict] = []
        source = f"{len(pieces)} clips joined"
        talking = any(v.voiced for v in voices)  # sounds under the voice when the clips talk
        clips_info = [{"path": p.path, "from": round(p.start, 3), "to": round(p.end, 3), "seconds": round(p.seconds, 3),
                       "size": f"{p.info.width}x{p.info.height}", "fps": round(p.info.rate, 3),
                       "starts_at": round(film.starts[k] / fps, 3)} for k, p in enumerate(pieces)]
        seconds_in = sum(p.seconds for p in pieces)
    else:
        info = clip.info
        fps = info.rate
        n = info.frames
        L = n / fps
        talking = len(clip.words) >= 4
        fr = framing_mod.plan(info.width, info.height, fps, n, None, shape="keep")
        size = (fr.out_w, fr.out_h)
        crossings = []
        clips_info = []
        seconds_in = L
        if edited is not None:
            chosen = [(e["at"], int(math.ceil(e["at"] * fps - 0.01)), "the plan given", "") for e in edited]  # times kept to 0.1 ms
            looks_dur = [(e["look"], e["duration"]) for e in edited]
            joins_info = [{"at": e["at"], "from": "the plan given", "chosen": True} for e in edited]
            source = "the plan given"
            colour = edited[0]["color"] if edited and edited[0]["color"] != "auto" else colour
        else:
            joins, source = find_joins(clip, opts, notes, talking)
            how = opts["joins"]
            if how == "auto":
                how = "between-ideas" if talking else "every"
            if how == "between-ideas" and not talking:
                notes.append("no words were heard, so there are no sentences to go by: every join has one")
                how = "every"
            if not any(j.cut for j in joins) and how in ("every", "between-ideas") and talking:
                joins = sentence_moments(clip)
                source = "no cut in the picture: the moments between sentences"
                how = "between-ideas"
            if not joins and how in ("every", "between-ideas"):
                said = (" and it is one sentence, so nothing falls between two ideas" if talking else "")
                raise PlanError(f"no cut was found in this video{said}. Give the seconds with joins=4.5,9 (a punch, "
                                "whip, glitch, flash, light leak or dots work inside one shot), join clips with "
                                "clips=…, or run clean-cut first")
            picked = choose(joins, how, L, notes)
            if how not in ("every", "between-ideas"):
                on_cuts = sum(1 for j in picked if j.cut)
                if on_cuts == len(picked):
                    source = "the seconds given, each on a cut"
                elif on_cuts:
                    source = f"the seconds given, {on_cuts} on a cut and {len(picked) - on_cuts} on moments inside a shot"
                else:
                    source = "the seconds given, each a moment inside one shot (no cut there)"
            if not picked:
                raise PlanError("none of the joins falls between two sentences. Say joins=every to use them all, or give "
                                "the seconds with joins=…")
            looks = _looks(opts, talking, False)
            chosen, looks_dur = [], []
            for i, jn in enumerate(picked):
                look = looks[i % len(looks)]
                if not jn.cut and needs_both(look, colour):
                    raise PlanError(f"at {jn.at:.2f} s there is no cut, so a {look} would cross the shot into itself. "
                                    "Use a punch, whip, glitch, flash, light-leak or dot-matrix there, or give a time with "
                                    "a cut")
                frame = jn.frame if jn.frame else int(math.ceil(jn.at * fps - 1e-6))
                d = opts["duration"] if opts["duration"] != "auto" else AUTO[look][0]
                chosen.append((frame / fps, frame, jn.source, jn.why))
                looks_dur.append((look, float(d)))
            joins_info = [j.as_dict() for j in joins]
            for j in picked:
                if all(abs(j.at - x["at"]) > 1e-3 for x in joins_info):
                    joins_info.append(j.as_dict())
        trs = place(chosen, looks_dur, fps, n, colour, notes)
        changes = shots.changes(info)
        for tr in trs:
            tr.settle = settled_frame(changes, tr.frame, fps)
            if tr.settle > tr.frame:
                gone = tr.settle - tr.frame
                tr.note = (tr.note + "; " if tr.note else "") + (
                    f"{gone} stray frame{'s' if gone > 1 else ''} of what was cut out, just after the cut, hidden "
                    f"under the {tr.look}")
        held_frames = {}

        def raw(j: int) -> np.ndarray:
            j = int(min(n - 1, max(0, j)))
            if j not in held_frames:
                if len(held_frames) > 6:
                    held_frames.pop(next(iter(held_frames)))
                held_frames[j] = fr.apply(video.read_frame(info, j), j)
            return held_frames[j]

        face_at = None
        if any(t.look == "punch" for t in trs):
            face = clip.face

            def face_at(j: int) -> Optional[Rect]:
                h = face.head(j)
                return fr.to_output(h, j) if h is not None else None

        job_clip = clip
        painter = Painter(trs, fps, size, raw, None, face_at, n, notes)

    tl = Timeline.whole(n / fps)
    sfx = lay_sounds(trs, fps, job_clip.sound, talking) if opts["sound"] else None
    for tr in trs:
        if not opts["sound"]:
            tr.sound = None
    out_seconds = n / fps
    by_look: dict[str, int] = {}
    for tr in trs:
        by_look[tr.look] = by_look.get(tr.look, 0) + 1
    sheet_times = []
    for tr in trs[:2]:
        for j in strip_frames(tr)[:3] if len(trs) > 1 else strip_frames(tr):
            sheet_times.append(round((j + 0.5) / fps, 4))
    sheet_times = sorted(set(sheet_times))[:6]
    out_plan = {"effect": NAME, "clip": clip.path, "options": opts, "mode": "clips" if joining else "one video",
                "output": {"width": size[0], "height": size[1], "fps": round(fps, 3), "seconds": round(out_seconds, 3),
                           "framing": fr.note},
                "length": {"seconds_in": round(seconds_in, 3), "seconds_out": round(out_seconds, 3)},
                "joins_from": source, "joins": joins_info,
                "transitions": [t.as_dict(fps) for t in trs],
                "notes": notes, "sheet_times": sheet_times}
    if joining:
        out_plan["clips"] = clips_info
        out_plan["sound_crossings"] = crossings
    if not joining and talking:
        from .captions import unsure_words

        out_plan["unsure"] = unsure_words(clip)

    held = [t for t in trs if not t.overlap and needs_both(t.look, t.colour)]
    if joining:
        overlapped = sum(film.overlaps) / fps
        length = (f"{seconds_in:.2f} s of clips -> {out_seconds:.2f} s"
                  + (f" ({overlapped:.2f} s shorter: the pictures overlap where both show)" if overlapped else ""))
    else:
        length = f"{out_seconds:.2f} s, unchanged: inside one video nothing overlaps, so the length and the sound are kept"
    if held:
        length += f"; {len(held)} hold the last frame out and the first frame in, each for its half"
    checks = {"joins": f"{len(trs)} of {len(joins_info) if not joining else len(pieces) - 1} from {source}",
              "looks": ", ".join(f"{v} {k}" for k, v in by_look.items()),
              "where": "; ".join(f"{t.n}: {t.look} on frame {t.frame} ({t.at:.2f} s), frames {t.first}-{t.last}"
                                 for t in trs[:8]) + (" …" if len(trs) > 8 else ""),
              "length": length}
    if joining and crossings:
        checks["sound"] = "; ".join(f"join {c['join']}: {c['how']}" for c in crossings)
    if sfx is not None:
        checks["sounds"] = (", ".join(sorted({SOUND_OF[t.look] for t in trs})) + ", each landing on its join, "
                            + (f"{UNDER_VOICE:g} dB under the voice ({sfx.reference:.1f} LUFS)" if talking else
                               f"nobody talks, so each at {NO_VOICE - UNDER_VOICE:g} LUFS at its loudest"))
    punched = [t for t in trs if t.look == "punch"]
    if punched:
        checks["punch"] = "; ".join(f"{t.n}: {t.zoom_from:.2f}x -> {t.zoom_to:.2f}x"
                                    + (f" ({t.punch_note})" if t.punch_note else "") for t in punched)
    strays = [t for t in trs if t.settle > t.frame]
    if strays:
        checks["stray frames"] = "; ".join(f"{t.n}: frame {t.frame}" + (f"-{t.settle - 1}" if t.settle - t.frame > 1 else "")
                                           + f" (left over from what was cut out), hidden under the {t.look}" for t in strays)
    if notes:
        checks["notes"] = "; ".join(notes)

    def stills() -> list[float]:
        out = []
        for tr in trs[:4]:
            j = tr.frame + (1 if tr.look == "punch" else 0)
            if tr.look == "dot-matrix":
                j = tr.frame + int(0.45 * (tr.last - tr.frame))
            out.append(round((j + 0.01) / fps, 4))
        return out or [out_seconds / 2]

    def strip_rows(get) -> list:
        rows = []
        for tr in trs[:6]:
            rows.append((tr, [(j, get(j)) for j in strip_frames(tr) if 0 <= j < n]))
        return rows

    def sheet_lines(measured: Optional[str] = None) -> list[str]:
        lines = [f"{len(trs)} transition(s): {checks['looks']}; {checks['length']}",
                 f"joins from {source}"]
        if measured:
            lines.append(measured)
        lines.append("yellow: the join, the first frame of the new shot; under each row, the film's own sound "
                     "(blue) and the transition's (orange), in dB")
        return lines

    def preview(out: Path) -> dict:
        from ..core import render

        wanted = sorted({j for tr in trs[:6] for j in strip_frames(tr) if 0 <= j < n})
        got = {int(round(t * fps)): f for t, f in
               render.still_frames(job_clip, tl, fr, painter.draw, [(j + 0.01) / fps for j in wanted])}
        if sfx is not None:
            sfx.mix(job_clip.sound, RATE)
        path = draw_sheet(strip_rows(lambda j: got.get(j, np.zeros((size[1], size[0], 3), np.uint8))), fps,
                          job_clip.sound, sfx.stem if sfx is not None else None,
                          f"transitions · {Path(clip.path).name} (stills, nothing rendered)", out.with_suffix(".transitions.jpg"),
                          sheet_lines())
        return {"sound sheet": str(path)}

    def after(result: dict, out: Path) -> dict:
        lines = {}
        written = int(result.get("frames", 0))
        lines["frames"] = f"{written} written, {n} planned" + ("" if written == n else " (MISMATCH)")
        measured = None
        if sfx is not None:
            rows, lag_ms = check_sounds(out, job_clip.sound, sfx, fps)
            offs = [abs(r["off_ms"]) for r in rows if r["off_ms"] is not None]
            frame_ms = 1000.0 / fps
            if offs:
                lines["landing"] = (f"{sum(1 for o in offs if o < frame_ms)} of {len(rows)} sounds land within a frame of "
                                    f"their join in the encoded file (largest {max(offs):.1f} ms)")
            lines["sound in step"] = ("the film's own sound is in step in the file" if abs(lag_ms) < 0.05
                                      else f"the file's sound runs {lag_ms:+.2f} ms against the film's")
            measured = lines.get("landing", "")
        got = frames_from_file(out, [j for tr in trs[:6] for j in strip_frames(tr) if 0 <= j < n])
        sheet = draw_sheet(strip_rows(lambda j: got.get(j, np.zeros((size[1], size[0], 3), np.uint8))), fps,
                           job_clip.sound, sfx.stem if sfx is not None else None,
                           f"transitions · {Path(out).name}", out.with_suffix(".transitions.jpg"), sheet_lines(measured))
        lines["sound sheet"] = str(sheet)
        return lines

    return Job(NAME, job_clip, tl, fr, painter.draw, out_plan, checks=checks, still_times=stills, after=after,
               sound=sfx.mix if sfx is not None else None, preview=preview)
