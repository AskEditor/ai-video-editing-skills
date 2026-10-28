"""grade: a colour look in one word, on the whole clip or a span of it, with skin kept natural.

How a look is made and laid:

1. **Each look is a 3D LUT built here, by code.** `apply()` is the look as arithmetic on colours;
   `cube()` runs it over a 33×33×33 grid and writes a standard `.cube` file, which ffmpeg's lut3d
   applies while the picture is read (`python -m fx.effects.grade cubes <folder>` writes all six,
   for any other editor). Nothing is copied from a preset pack.
2. **Correct the tone, then colour it.** A look works in a perceptual colour space (Oklab): the
   white balance moves in linear light, the tone curve bends lightness only, and colour is added
   to the shadows, middle and highlights by how bright each pixel is, so a look never turns the
   whole frame one flat colour.
3. **Skin is held.** Every look also decides how skin-coloured each input colour is (by its hue,
   how colourful it is and how bright) and gives those colours back most of their own hue and
   colourfulness. Warm looks then never turn faces orange, and cool or moody ones never turn them
   grey or green. The run measures it: the face's skin before and after, in numbers, on every clip.
4. **The finish is drawn per frame:** a vignette centred on where the person stands (not the
   middle of the frame), and fine grain, monochrome and strongest in the mid-tones as film's is,
   the same every time for the same frame.
5. **A span eases in and out** over a quarter of a second, and can start or end on a word: the
   look turns as "moody" is said.
"""

from __future__ import annotations

import atexit
import hashlib
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import options as O

NAME = "grade"
LOOKS = ("moody", "black-and-white", "warm", "cool", "pop", "film")
LUT_SIZE = 33
LOOK_VERSION = 3  # raise whenever the maths of a look changes, so no cached .cube is reused
EASE = 0.25  # seconds a span takes to turn on, and off


# The test matrix's runs (tests/matrix.py): the controlled clips on the default look, then each other
# look on a clip that suits it, and one span that turns on a spoken word.
MATRIX = [
    ("A", "A", {}),
    ("B", "B", {}),
    ("C", "C", {}),
    ("A-film", "A", {"look": "film"}),
    ("A-on-hooks", "A", {"look": "black-and-white", "start": "hooks"}),
    ("B-warm", "B", {"look": "warm"}),
    ("C-pop", "C", {"look": "pop"}),
    ("D-black-and-white", "D", {"look": "black-and-white"}),
    ("F-cool", "F", {"look": "cool"}),
]


# ---- options of this effect's own kinds ------------------------------------------------------

@dataclass(frozen=True)
class Amount:
    """auto (the look decides), off, on, or an amount from 0 to 1."""
    help: str

    def parse(self, raw: Any) -> Any:
        v = str(raw).strip().lower()
        if v in ("auto", ""):
            return "auto"
        if v in ("off", "none", "no"):
            return 0.0
        if v in ("on", "yes"):
            return "on"
        try:
            x = float(v)
        except ValueError:
            raise O.OptionError(f"say auto, off, on or an amount from 0 to 1 (got {raw!r})") from None
        if not 0.0 <= x <= 1.0:
            raise O.OptionError(f"give an amount from 0 to 1 (got {raw!r})")
        return x


@dataclass(frozen=True)
class Moment:
    """A moment in the clip: seconds, or words that are said there."""
    help: str

    def parse(self, raw: Any) -> Any:
        v = str(raw).strip().strip("\"'“”‘’").strip()
        low = v.lower()
        if low in ("", "start", "begin", "beginning"):
            return 0.0
        if low in ("end", "the end", "auto"):
            return "end"
        try:
            x = float(low.rstrip("s"))
        except ValueError:
            if not re.search(r"\w", v):
                raise O.OptionError(f"give seconds (2.5) or words that are said (got {raw!r})") from None
            return v  # words to find in the transcript
        if x < 0:
            raise O.OptionError(f"give seconds from the start, 0 or more (got {raw!r})")
        return x


@dataclass(frozen=True)
class LookName:
    """One of the looks, by its name or the plain words people use for it."""
    help: str
    names: tuple[str, ...] = LOOKS
    aliases: tuple[tuple[str, str], ...] = (
        ("dark", "moody"), ("moodier", "moody"), ("bw", "black-and-white"), ("b&w", "black-and-white"),
        ("black and white", "black-and-white"), ("mono", "black-and-white"), ("monochrome", "black-and-white"),
        ("greyscale", "black-and-white"), ("grayscale", "black-and-white"), ("golden", "warm"),
        ("golden hour", "warm"), ("warmer", "warm"), ("cold", "cool"), ("blue", "cool"), ("cooler", "cool"),
        ("vivid", "pop"), ("punchy", "pop"), ("vibrant", "pop"), ("bright", "pop"), ("cinematic", "film"),
        ("filmic", "film"), ("vintage", "film"), ("analog", "film"), ("analogue", "film"))

    def parse(self, raw: Any) -> str:
        v = re.sub(r"\s+", " ", str(raw).strip().lower().replace("_", " "))
        if v.replace(" ", "-") in self.names:
            return v.replace(" ", "-")
        for word, look in self.aliases:
            if v == word:
                return look
        raise O.OptionError(f"choose one of {', '.join(self.names)} (got {raw!r})")


OPTIONS = {
    "look": LookName("moody: darker, quieter colour, cool shadows · black-and-white · warm: golden light · "
                     "cool: crisp blue light · pop: punchier contrast and colour · film: soft contrast, "
                     "lifted blacks and fine grain. Skin stays natural in all six"),
    "intensity": O.Number(0, 1, "how much of the look: 1 is the look as made, 0.5 half of it"),
    "vignette": Amount("darker corners round you: auto (the look decides), off, or an amount from 0 to 1"),
    "grain": Amount("fine film grain: auto (film has it), off, or an amount from 0 to 1"),
    "start": Moment("where the look begins: seconds (2.5), or words that are said there (\"make it moody\"); "
                    "0 is the start"),
    "end": Moment("where it ends: end, seconds, or words that are said there (the look goes as they "
                  "are said: look=black-and-white end=\"hooks\" turns to colour on the word)"),
    "fix": O.Text("misheard words as wrong=right, separated by ; (only matters when start or end are words)"),
}


# ---- colour: encoded RGB <-> linear light <-> Oklab -------------------------------------------
# Oklab is the perceptual colour space Björn Ottosson published in 2020; the two matrices below
# are its definition. L is lightness; a/b's angle is the hue, and its length the chroma.

_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])
_M1i = np.linalg.inv(_M1)
_M2i = np.linalg.inv(_M2)


def to_linear(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def to_encoded(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1 / 2.4) - 0.055)


def oklab(lin: np.ndarray) -> np.ndarray:
    return np.cbrt(lin @ _M1.T) @ _M2.T


def from_oklab(lab: np.ndarray) -> np.ndarray:
    return ((lab @ _M2i.T) ** 3) @ _M1i.T


def _smooth(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _hue_deg(lab: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360


def _wrap(d: np.ndarray) -> np.ndarray:
    return (d + 180.0) % 360.0 - 180.0


# ---- the looks ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class Recipe:
    """One look, as numbers. Hues are Oklab angles in degrees (about: red 30, orange 60, yellow 110,
    green 145, cyan 195, blue 265, magenta 330); tint amounts are Oklab a/b units (0.01 is subtle)."""
    name: str
    mono: Optional[tuple[float, float, float]] = None  # black and white: how much red, green, blue
    temp: float = 0.0  # white balance: + warmer (amber), - cooler (blue)
    tint: float = 0.0  # + magenta, - green
    gamma: float = 1.0  # above 1 darkens the middle
    contrast: float = 1.0  # the S-curve's strength about `pivot`
    pivot: float = 0.5
    black: float = 0.0  # where black lands (a lift above 0 is a matte, faded black)
    white: float = 1.0  # where white lands
    sat: float = 1.0  # colourfulness, all colours
    vib: float = 0.0  # extra colourfulness for the dull colours only
    shadows: tuple[float, float] = (0.0, 0.0)  # (hue, amount) laid into the dark end
    mids: tuple[float, float] = (0.0, 0.0)
    highs: tuple[float, float] = (0.0, 0.0)
    bend: tuple[tuple[float, float, float], ...] = ()  # (hue, width, turn): one range of hues turned
    skin_hold: float = 0.8  # how much of skin's own hue and colour is given back (0..1)
    skin_sat: float = 1.0  # skin's colourfulness against as shot
    skin_light: float = 0.0  # how much of the look's darkening skin is spared (0..1): the face stays lit
    vignette: float = 0.0  # the finish the look asks for when the options say auto
    grain: float = 0.0


RECIPES = {
    "moody": Recipe("moody", temp=-0.035, gamma=1.18, contrast=1.22, pivot=0.48, black=0.028, white=0.9,
                    sat=0.7, shadows=(200.0, 0.030), mids=(175.0, 0.010), highs=(75.0, 0.006),
                    bend=((140.0, 45.0, 14.0),), skin_hold=0.9, skin_sat=0.95, skin_light=0.4, vignette=0.5),
    "black-and-white": Recipe("black-and-white", mono=(0.42, 0.46, 0.12), contrast=1.28, pivot=0.45, black=0.012,
                              white=0.985, vignette=0.25, grain=0.12),
    "warm": Recipe("warm", temp=0.085, tint=0.012, gamma=0.98, contrast=1.05, black=0.012, white=0.985, sat=1.04,
                   vib=0.12, shadows=(55.0, 0.004), highs=(80.0, 0.016), skin_hold=0.95, skin_sat=1.0),
    "cool": Recipe("cool", temp=-0.125, tint=0.006, contrast=1.08, black=0.02, white=0.975, sat=0.9,
                   shadows=(240.0, 0.022), mids=(225.0, 0.006), highs=(215.0, 0.014), skin_hold=0.92,
                   skin_sat=1.0),
    "pop": Recipe("pop", gamma=0.99, contrast=1.28, pivot=0.46, black=0.0, white=0.99, sat=1.05, vib=0.5,
                  skin_hold=0.94, skin_sat=1.05),
    "film": Recipe("film", temp=0.012, contrast=1.06, pivot=0.48, black=0.06, white=0.935, sat=0.86,
                   shadows=(185.0, 0.016), highs=(78.0, 0.016), bend=((140.0, 50.0, -16.0),), skin_hold=0.85,
                   skin_sat=0.97, vignette=0.25, grain=0.35),
}


def _tone(L: np.ndarray, r: Recipe) -> np.ndarray:
    """The tone curve on lightness: a middle power, an S about the pivot, then the black and white
    levels. Smooth and rising everywhere, so nothing bands or reverses."""
    x = np.clip(L, 0.0, 1.0)
    if r.gamma != 1.0:
        x = x ** r.gamma
    if r.contrast != 1.0:
        a = math.log(0.5) / math.log(r.pivot)
        u = np.clip(x, 1e-6, 1 - 1e-6) ** a
        c = r.contrast
        s = u ** c / (u ** c + (1 - u) ** c)
        x = s ** (1 / a)
    return r.black + (r.white - r.black) * x


def skin_weight(lab: np.ndarray) -> np.ndarray:
    """How skin-coloured each colour is, 0..1: a warm hue between red and orange, some colour but
    not loud colour, neither near black nor near white. Darker and lighter skin both fall inside;
    so do wood and sand, which then keep their own colour too."""
    L = lab[..., 0]
    C = np.hypot(lab[..., 1], lab[..., 2])
    # A plateau from red-brown to golden (20° to 62°): the skin of every clip measured, light or dark,
    # under daylight, office light or a phone's own processing, falls inside it.
    h = _hue_deg(lab)
    d = np.maximum(0.0, np.maximum(_wrap(20.0 - h), _wrap(h - 62.0)))
    hue = np.exp(-0.5 * (d / 16.0) ** 2)
    chroma = _smooth(0.006, 0.022, C) * (1.0 - _smooth(0.15, 0.24, C))
    light = _smooth(0.06, 0.16, L) * (1.0 - _smooth(0.86, 0.96, L))
    return hue * chroma * light


GAMUT_KNEE = 0.85  # colours using more than this share of the screen's room are eased, not cut


def _gamut(lab: np.ndarray) -> np.ndarray:
    """Linear RGB for `lab`, easing colour (never lightness) as it nears what a screen can show.

    For each colour, the room is how far its chroma could grow at its lightness and hue before a
    channel leaves 0..1. Colours past `GAMUT_KNEE` of the room roll off smoothly towards its edge,
    and colours beyond it come back inside, so a bright yellow lamp or a sunset grades without the
    hard corner that a straight clip leaves (and that a LUT cannot follow)."""
    L = np.clip(lab[:, 0], 0.0, 1.0)
    ab = lab[:, 1:]
    lo, hi = np.zeros(len(lab)), np.full(len(lab), 4.0)
    for _ in range(18):
        mid = (lo + hi) / 2
        test = from_oklab(np.concatenate([L[:, None], ab * mid[:, None]], axis=1))
        ok = ((test >= -1e-5) & (test <= 1 + 1e-5)).all(axis=-1)
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    used = 1.0 / np.maximum(lo, 1e-6)  # the share of the room this colour uses; above 1 is outside
    k = GAMUT_KNEE
    eased = np.where(used <= k, used, k + (1 - k) * np.tanh((used - k) / (1 - k)))
    scale = eased / np.maximum(used, 1e-9)
    return np.clip(from_oklab(np.concatenate([L[:, None], ab * scale[:, None]], axis=1)), 0.0, 1.0)


def _band_weights(L: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shadows = np.clip(1.0 - L / 0.55, 0.0, 1.0) ** 1.5
    highs = np.clip((L - 0.45) / 0.55, 0.0, 1.0) ** 1.5
    mids = np.clip(1.0 - shadows - highs, 0.0, 1.0)
    holds = _smooth(0.0, 0.12, L) * (1.0 - _smooth(0.93, 1.0, L))  # pure black and white hold no colour
    return shadows * holds, mids * holds, highs * holds


def _offset(hue_amount: tuple[float, float]) -> np.ndarray:
    h, amount = hue_amount
    return amount * np.array([math.cos(math.radians(h)), math.sin(math.radians(h))])


def apply(rgb: np.ndarray, look: str | Recipe, intensity: float = 1.0) -> np.ndarray:
    """The look on colours: `rgb` is (..., 3) encoded RGB from 0 to 1, as a video frame is read.
    Returns the same shape. `intensity` mixes the look with the colours as shot."""
    r = RECIPES[look] if isinstance(look, str) else look
    shape = rgb.shape
    x = np.asarray(rgb, np.float64).reshape(-1, 3)
    lin = to_linear(x)
    if r.mono is not None:
        y = lin @ np.asarray(r.mono, np.float64) / sum(r.mono)
        L = _tone(np.cbrt(np.clip(y, 0.0, 1.0)), r)
        out = to_encoded(np.repeat(np.clip(L, 0.0, 1.0)[:, None] ** 3, 3, axis=1))
    else:
        lab_in = oklab(lin)
        if r.temp or r.tint:
            gains = np.exp(np.array([r.temp, -r.tint, -r.temp]))
            gains /= gains @ np.array([0.2126, 0.7152, 0.0722])
            lab = oklab(lin * gains)
        else:
            lab = lab_in.copy()
        L0 = lab[:, 0]
        L1 = _tone(L0, r)
        ab = lab[:, 1:]
        C = np.hypot(ab[:, 0], ab[:, 1])
        keep = np.clip(L1 / np.maximum(L0, 1e-4), 0.6, 1.5)  # the same saturation at the new lightness
        boost = r.sat * (1.0 + r.vib * np.clip(1.0 - C / 0.12, 0.0, 1.0))
        ab = ab * (keep * boost)[:, None]
        for hue, width, turn in r.bend:
            h = _hue_deg(np.concatenate([L1[:, None], ab], axis=1))
            w = np.exp(-0.5 * (_wrap(h - hue) / (width / 2.0)) ** 2)
            ang = np.radians(turn * w)
            ca, sa = np.cos(ang), np.sin(ang)
            ab = np.stack([ab[:, 0] * ca - ab[:, 1] * sa, ab[:, 0] * sa + ab[:, 1] * ca], axis=1)
        ws, wm, wh = _band_weights(L1)
        ab = ab + ws[:, None] * _offset(r.shadows) + wm[:, None] * _offset(r.mids) + wh[:, None] * _offset(r.highs)
        # Skin: its own hue and colourfulness, at the look's new lightness, mostly given back.
        sw = skin_weight(lab_in)
        w = sw * r.skin_hold
        if r.skin_light > 0:  # a darker look keeps the face lit: part of skin's darkening is undone
            L1 = L1 + sw * r.skin_light * np.maximum(0.0, L0 - L1)
        if w.any():
            C_in = np.hypot(lab_in[:, 1], lab_in[:, 2])
            # Half the look's change of lightness carries into skin's colour, so a darker look
            # keeps faces rich rather than ashen, and a brighter one never makes them louder.
            scale = np.clip(L1 / np.maximum(lab_in[:, 0], 1e-4), 0.6, 1.5) ** 0.5 * r.skin_sat
            own = lab_in[:, 1:] / np.maximum(C_in, 1e-9)[:, None] * (C_in * scale)[:, None]
            ab = ab + w[:, None] * (own - ab)
        out = to_encoded(_gamut(np.concatenate([L1[:, None], ab], axis=1)))
    if intensity < 1.0:
        out = x + (out - x) * float(intensity)
    return out.reshape(shape)


# ---- the .cube file -----------------------------------------------------------------------------

def cube(look: str | Recipe, intensity: float = 1.0, size: int = LUT_SIZE) -> np.ndarray:
    """The look as a table of size³ colours, indexed [blue][green][red] as a .cube file lists them."""
    g = np.linspace(0.0, 1.0, size)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")
    grid = np.stack([r, gg, b], axis=-1)
    return apply(grid, look, intensity)


def write_cube(path: str | os.PathLike, table: np.ndarray, title: str) -> Path:
    path = Path(path)
    size = table.shape[0]
    lines = [f'TITLE "{title}"', f"LUT_3D_SIZE {size}", "DOMAIN_MIN 0.0 0.0 0.0", "DOMAIN_MAX 1.0 1.0 1.0"]
    flat = np.clip(table.reshape(-1, 3), 0.0, 1.0)
    lines += [f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in flat]
    tmp = path.with_name(path.name + f".{os.getpid()}.part")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(path)
    return path


def _lut_folder() -> Path:
    from ..core.paths import CACHE

    for folder in (CACHE / "looks", Path(tempfile.gettempdir()) / "video-editing-looks"):
        try:
            folder.mkdir(parents=True, exist_ok=True)
            if os.access(folder, os.W_OK):
                return folder
        except OSError:
            continue
    folder = Path(tempfile.mkdtemp(prefix="looks-"))
    atexit.register(shutil.rmtree, folder, ignore_errors=True)
    return folder


def lut_file(look: str, intensity: float) -> Path:
    """The look's .cube, made once and kept in the pack's cache (named by everything that shapes it)."""
    recipe = RECIPES[look]
    key = hashlib.sha1(repr((asdict(recipe), round(intensity, 4), LUT_SIZE, LOOK_VERSION)).encode()).hexdigest()[:10]
    path = _lut_folder() / f"{look}-{key}.cube"
    if not path.exists():
        write_cube(path, cube(recipe, intensity), f"{look} ({intensity:g})")
    return path


# ---- the finish: vignette and grain -----------------------------------------------------------

class Finish:
    """Darker corners round the person, and grain, laid on an output frame. Both are worked out
    from the frame's size and number alone, so a still and the render match exactly."""

    def __init__(self, width: int, height: int, vignette: float, grain: float,
                 centre: tuple[float, float]):
        self.W, self.H = width, height
        self.vignette, self.grain = float(vignette), float(grain)
        self._mask = None
        self._full = None
        if self.vignette > 0:
            import cv2

            cx, cy = centre
            yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
            # An ellipse the frame's own shape, so a tall frame darkens top and bottom as much as the sides.
            dx = (xx - cx) / (0.5 * width)
            dy = (yy - cy) / (0.5 * height)
            r = np.sqrt(dx * dx + dy * dy)
            self._mask = (_smooth(0.55, 1.45, r) ** 1.3).astype(np.float32)
            self._full = cv2.merge([1.0 - 0.55 * self.vignette * self._mask] * 3)
        # Grain the size of a fine film's: about a pixel and a half on a 1920-tall frame. Each frame's
        # grain is drawn from a generator seeded by the frame's number: the same every run.
        self._sigma = max(0.6, 0.95 * min(width, height) / 1080.0)
        if self.grain > 0:
            import cv2

            cv2.setRNGSeed(7)
            probe = np.empty((256, 256), np.float32)
            cv2.randn(probe, 0.0, 1.0)
            self._unit = 1.0 / float(cv2.GaussianBlur(probe, (0, 0), self._sigma)[16:-16, 16:-16].std())
            # How much grain each brightness takes: most in the mid-tones, as film's; a little in the shadows.
            y = np.arange(256, dtype=np.float32) / 255.0
            self._by_light = (0.3 + 0.7 * 4.0 * y * (1.0 - y)).astype(np.float32)

    def apply(self, frame: np.ndarray, j: int, strength: float = 1.0) -> None:
        if strength <= 0 or (self._mask is None and self.grain <= 0):
            return
        import cv2

        out = frame
        if self._mask is not None:
            if strength >= 0.999:
                scale = self._full
            else:
                scale = cv2.merge([1.0 - (0.55 * self.vignette * strength) * self._mask] * 3)
            out = cv2.multiply(out, scale, dtype=cv2.CV_8U)
        if self.grain > 0:
            cv2.setRNGSeed(1_000_003 * (j + 1) % 2_147_483_647)
            n = np.empty((self.H, self.W), np.float32)
            cv2.randn(n, 0.0, 1.0)
            n = cv2.GaussianBlur(n, (0, 0), self._sigma)
            amount = self._by_light * (13.0 * self.grain * strength * self._unit)
            n = cv2.multiply(n, cv2.LUT(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY), amount))
            out = cv2.add(out, cv2.merge([n, n, n]), dtype=cv2.CV_8U)
        frame[:] = out


# ---- the graded picture, read beside the render -------------------------------------------------

class GradedFrames:
    """The clip's frames through the look's LUT (ffmpeg's lut3d), shaped like the output. It keeps
    pace with the render (frame after frame from one reader) and jumps when asked for another."""

    def __init__(self, info, fr, lut: Path):
        self.info, self.fr = info, fr
        self.filter = f"lut3d=file={lut.name}:interp=tetrahedral"
        self.cwd = str(lut.parent)
        self._iter = None
        self._reader = None
        self._k = -1
        self._frame: Optional[np.ndarray] = None
        self._shown: tuple[int, Optional[np.ndarray]] = (-1, None)
        atexit.register(self.close)

    def _open(self, k: int) -> None:
        from ..core import video

        self.close()
        start = max(0.0, (k - 0.25) / self.info.rate)
        self._reader = video.FrameReader(self.info, start=start, extra=self.filter, cwd=self.cwd)
        self._iter = iter(self._reader)
        self._k = k - 1
        self._frame = None

    def at(self, k: int) -> np.ndarray:
        if self._shown[0] == k and self._shown[1] is not None:
            return self._shown[1]
        if self._iter is None or k < self._k or k > self._k + 3:
            self._open(k)
        while self._k < k:
            nxt = next(self._iter, None)
            if nxt is None:  # the clip ran short of its stated length: hold the last frame
                break
            self._frame, self._k = nxt, self._k + 1
        if self._frame is None:
            from ..core.video import MediaError

            raise MediaError(f"could not read the picture of {self.info.path} through the look")
        self._k = k
        self._shown = (k, self.fr.apply(self._frame, k))
        return self._shown[1]

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
        self._reader, self._iter = None, None


# ---- the span ---------------------------------------------------------------------------------

def _find_words(words, phrase: str) -> tuple[float, str]:
    """When `phrase` begins to be said (its first time), and the words as heard. The look turns there,
    in either direction: on as "moody" is said, or off as the word that ends it is said."""
    from ..core.words import normalize

    want = [normalize(w) for w in phrase.split() if normalize(w)]
    keys = [w.key for w in words]
    for i in range(len(keys) - len(want) + 1):
        if keys[i:i + len(want)] == want:
            hit = words[i:i + len(want)]
            return hit[0].start, " ".join(w.text for w in hit)
    raise O.OptionError(f"the words {phrase!r} are not said in this clip; run `fx.run words` to read what is")


def resolve_span(clip, start: Any, end: Any, duration: float) -> dict:
    said = {}
    if isinstance(start, str) and start != "end":
        start, said["start"] = _find_words(clip.words, start)
    if isinstance(end, str) and end != "end":
        end, said["end"] = _find_words(clip.words, end)
    s = 0.0 if start == "end" else float(start)
    e = duration if end == "end" else float(end)
    s, e = max(0.0, min(s, duration)), max(0.0, min(e, duration))
    if e <= s:
        raise O.OptionError(f"the look would end ({e:.2f} s) before it begins ({s:.2f} s)")
    return {"start": round(s, 3), "end": round(e, 3), "said": said}


def ramp(t: float, span: dict, duration: float, ease: float = EASE) -> float:
    """How much of the look shows at output time `t`. Each turn begins at its moment and takes `ease`
    seconds: on from `start`, off from `end`. A look that starts the clip or runs to its end has no
    turn there."""
    s, e = span["start"], span["end"]
    on = 1.0 if s <= 1e-3 else float(_smooth(s, s + ease, np.array(t)))
    off = 1.0 if e >= duration - 1e-3 else 1.0 - float(_smooth(e, e + ease, np.array(t)))
    return min(on, off) if t >= s else 0.0


# ---- the checks: skin and tone, measured -----------------------------------------------------

def _skin_pixels(frame: np.ndarray, box) -> Optional[np.ndarray]:
    """The skin of the face in `box` (the detector's brows-to-mouth box): its cheeks, nose and
    brow, without the eyes' glasses, the beard's edge or the background beside the head."""
    H, W = frame.shape[:2]
    x0, x1 = int(max(0, box.x0 + 0.22 * box.w)), int(min(W, box.x1 - 0.22 * box.w))
    y0, y1 = int(max(0, box.y0 + 0.08 * box.h)), int(min(H, box.y0 + 0.70 * box.h))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    px = frame[y0:y1, x0:x1].reshape(-1, 3).astype(np.float64) / 255.0
    lab = oklab(to_linear(px))
    C = np.hypot(lab[:, 1], lab[:, 2])
    h = _hue_deg(lab)
    keep = (lab[:, 0] > 0.15) & (lab[:, 0] < 0.88) & (C > 0.012) & (C < 0.2) & ((h < 95) | (h > 345))
    return px[keep] if keep.sum() >= 30 else None


def _describe(lab: np.ndarray) -> dict:
    """Skin as three numbers: lightness, chroma (how much colour), and hue (which colour). Also its
    saturation, chroma for its lightness: what reads as orange when it rises and ashen when it falls."""
    m = lab.mean(axis=0)
    chroma = float(np.hypot(m[1], m[2]))
    return {"lightness": round(float(m[0]), 3), "chroma": round(chroma, 4),
            "hue": round(float(math.degrees(math.atan2(m[2], m[1])) % 360), 1),
            "saturation": round(chroma / max(float(m[0]), 1e-6), 4)}


def skin_check(clip, recipe: Recipe, intensity: float, span: dict, samples: int = 10) -> dict:
    """The face's skin before and after the look, on frames across the span: its hue (in degrees,
    where 30 is red and 60 orange), its colourfulness and its lightness, and a plain verdict."""
    from ..core import video

    info, face = clip.info, clip.face
    first, last = info.frame_at(span["start"]), info.frame_at(max(span["start"], span["end"] - 1.0 / info.rate))
    picks = [int(i) for i in np.linspace(first, last, samples * 3).astype(int) if face.found[int(i)]]
    picks = picks[:: max(1, len(picks) // samples)][:samples]
    before, after = [], []
    for i in picks:
        px = _skin_pixels(video.read_frame(info, i), face.box(i))
        if px is None:
            continue
        before.append(oklab(to_linear(px)))
        after.append(oklab(to_linear(apply(px, recipe, intensity))))
    if not before:
        return {"checked": False, "verdict": "no face found to check the skin on"}
    b, a = _describe(np.concatenate(before)), _describe(np.concatenate(after))
    turn = float(_wrap(np.array(a["hue"] - b["hue"])))
    ratio = a["saturation"] / max(b["saturation"], 1e-6)
    if recipe.mono is not None:
        verdict = ("black and white: skin is grey by design" if intensity >= 0.999 else
                   f"black and white at {intensity:g}: skin keeps {ratio:.0%} of its colour")
    elif ratio > 1.15 or (turn > 7 and ratio > 1.03):
        verdict = "warning: skin turns more orange"
    elif turn > 10:
        verdict = "warning: skin turns yellow or green"
    elif turn < -10:
        verdict = "warning: skin turns red or pink"
    elif ratio < 0.72:
        verdict = "warning: skin turns grey"
    else:
        verdict = "natural"
    return {"checked": True, "frames": len(before), "before": b, "after": a, "hue_turn": round(turn, 1),
            "saturation_ratio": round(ratio, 3), "verdict": verdict}


def _skin_line(skin: dict) -> str:
    if not skin.get("checked"):
        return skin["verdict"]
    b, a = skin["before"], skin["after"]
    if skin["verdict"].startswith("black and white"):
        return f"{skin['verdict']}; lightness {b['lightness']} → {a['lightness']} ({skin['frames']} frames)"
    return (f"{skin['verdict']}: hue {b['hue']}° → {a['hue']}°, saturation ×{skin['saturation_ratio']}, "
            f"lightness {b['lightness']} → {a['lightness']} ({skin['frames']} frames)")


def tone_check(clip, recipe: Recipe, intensity: float, span: dict) -> dict:
    """Where the darks, middle and lights sit before and after (0 black, 100 white), on one frame
    in the middle of the span: blacks crushed or whites clipped show here."""
    from ..core import video

    info = clip.info
    frame = video.read_frame(info, info.frame_at((span["start"] + span["end"]) / 2))
    small = frame[::4, ::4].reshape(-1, 3).astype(np.float64) / 255.0
    luma = np.array([0.2126, 0.7152, 0.0722])
    y0 = small @ luma
    y1 = apply(small, recipe, intensity) @ luma
    p = lambda y: [round(float(v) * 100, 1) for v in np.percentile(y, [2, 50, 98])]  # noqa: E731
    out = {"before": p(y0), "after": p(y1),
           "crushed": round(float(((y1 < 0.02) & (y0 >= 0.02)).mean()) * 100, 2),
           "clipped": round(float(((y1 > 0.985) & (y0 <= 0.985)).mean()) * 100, 2)}
    return out


# ---- the effect ---------------------------------------------------------------------------------

def finish_amounts(recipe: Recipe, opts: dict) -> tuple[float, float]:
    def pick(value, own):
        if value == "auto":
            return own * opts["intensity"]
        if value == "on":
            return own if own > 0 else 0.5
        return float(value)

    return pick(opts["vignette"], recipe.vignette), pick(opts["grain"], recipe.grain)


def subject_centre(clip, width: int, height: int) -> tuple[float, float]:
    """Where the vignette centres: halfway between the frame's middle and where the face usually is."""
    face = clip.face
    if not face.any():
        return width / 2, height / 2
    boxes = [face.box(i) for i in range(0, face.n, max(1, face.n // 40)) if face.found[i]]
    if not boxes:
        return width / 2, height / 2
    cx = float(np.median([b.cx for b in boxes])) * width / clip.info.width
    cy = float(np.median([b.cy for b in boxes])) * height / clip.info.height
    return (0.5 * width + 0.5 * cx, 0.5 * height + 0.5 * cy)


def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from ..core import framing as framing_mod
    from ..core.timeline import Timeline
    from . import Job, PlanError

    info = clip.info
    fr = framing_mod.plan(info.width, info.height, info.rate, info.frames, None, shape="keep")
    duration = info.frames / info.rate
    tl = Timeline.whole(duration)
    recipe = RECIPES[opts["look"]]
    if plan is not None:
        sp = plan.get("span") or {}
        try:
            span = {"start": float(sp["start"]), "end": float(sp["end"]), "said": sp.get("said", {})}
        except (KeyError, TypeError, ValueError):
            raise PlanError("the plan needs a span with a start and an end in seconds") from None
        if not 0 <= span["start"] < span["end"] <= duration + 0.5:
            raise PlanError(f"the plan's span must sit inside the clip's {duration:.2f} s")
    else:
        span = resolve_span(clip, opts["start"], opts["end"], duration)
    vignette, grain = finish_amounts(recipe, opts)
    lut = lut_file(recipe.name, opts["intensity"])
    graded = GradedFrames(info, fr, lut)
    finish = Finish(fr.out_w, fr.out_h, vignette, grain, subject_centre(clip, fr.out_w, fr.out_h))
    identity = opts["intensity"] <= 0

    def draw(frame: np.ndarray, j: int, t: float) -> None:
        w = ramp(t, span, duration)
        if w <= 0:
            return
        k = tl.source_frame(j, info.rate, info.frames)
        if not identity:
            g = graded.at(k)
            if w >= 0.999:
                frame[:] = g
            else:
                frame[:] = np.clip(frame + (g.astype(np.float32) - frame) * w + 0.5, 0, 255).astype(np.uint8)
        finish.apply(frame, j, w)

    skin = skin_check(clip, recipe, opts["intensity"], span)
    tone = tone_check(clip, recipe, opts["intensity"], span)
    warnings = []
    if skin.get("verdict", "").startswith("warning"):
        warnings.append(f"{skin['verdict']} (hue {skin['hue_turn']:+.1f}°, saturation ×{skin['saturation_ratio']:.2f})")
    if tone["crushed"] > 3:
        warnings.append(f"{tone['crushed']:.1f}% of the frame goes to black that was not black")
    if tone["clipped"] > 3:
        warnings.append(f"{tone['clipped']:.1f}% of the frame goes to white that was not white")
    s, e = span["start"], span["end"]
    if e - s >= duration - 1e-3:
        sheet = [duration * (k + 0.5) / 6 for k in range(6)]
    else:
        inside = np.linspace(s + min(EASE + 0.15, (e - s) / 3), e - min(EASE + 0.15, (e - s) / 3), 3)
        sheet = [max(0.0, s - 0.5), s + EASE / 2] + [float(x) for x in inside] + [min(duration - 0.05, e + 0.4)]
    out_plan = {"effect": NAME, "clip": clip.path, "options": opts,
                "output": {"width": fr.out_w, "height": fr.out_h, "fps": round(info.rate, 3),
                           "seconds": round(tl.duration, 3), "framing": fr.note},
                "span": span, "finish": {"vignette": round(vignette, 3), "grain": round(grain, 3)},
                "lut": str(lut), "skin": skin, "tone": tone, "sheet_times": [round(x, 3) for x in sheet]}
    checks = {"look": f"{recipe.name} at {opts['intensity']:g}",
              "span": f"{s:.2f}–{e:.2f} s" + "".join(f", {k}s on “{v.strip(' ,.;:!?')}”"
                                                     for k, v in span.get("said", {}).items()),
              "skin": _skin_line(skin),
              "tone": f"darks/middle/lights {tone['before']} → {tone['after']}",
              "warnings": warnings}
    still = [x for x in sheet[1:5]] if e - s < duration - 1e-3 else [duration * x for x in (0.15, 0.5, 0.85)]

    def after(result: dict, out: Path) -> dict:
        graded.close()
        return {}

    return Job(NAME, clip, tl, fr, draw, out_plan, checks=checks, still_times=lambda: still, after=after)


# ---- writing the looks as .cube files for any other editor --------------------------------------

def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m fx.effects.grade",
                                 description="Write the looks as .cube files, for any editor that takes a LUT.")
    ap.add_argument("what", choices=["cubes"])
    ap.add_argument("folder")
    ap.add_argument("--size", type=int, default=LUT_SIZE)
    ap.add_argument("--intensity", type=float, default=1.0)
    args = ap.parse_args(argv)
    folder = Path(args.folder).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    for look in LOOKS:
        p = write_cube(folder / f"{look}.cube", cube(look, args.intensity, args.size), look)
        print(p)
    print("The vignette and the grain are drawn per frame, so they are not in a .cube.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
