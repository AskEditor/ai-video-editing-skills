"""Fonts and drawing words.

Every face comes from the pack's own `fonts/` folder (OFL licensed, licences beside each). A face
that is missing stops the render with a plain message: it is never swapped for whatever the
computer has, because a fallback face looks right only on the machine that happens to have the
real one installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from .compose import Sprite
from .paths import FONTS, NeedSetup

FACES = {
    "anton": "Anton/Anton-Regular.ttf",
    "poppins-semibold": "Poppins/Poppins-SemiBold.ttf",
    "poppins-bold": "Poppins/Poppins-Bold.ttf",
    "poppins-extrabold": "Poppins/Poppins-ExtraBold.ttf",
    "poppins-black": "Poppins/Poppins-Black.ttf",
    "dm-serif": "DMSerifDisplay/DMSerifDisplay-Regular.ttf",
    "dm-serif-italic": "DMSerifDisplay/DMSerifDisplay-Italic.ttf",
    "instrument-serif": "InstrumentSerif/InstrumentSerif-Regular.ttf",
    "instrument-serif-italic": "InstrumentSerif/InstrumentSerif-Italic.ttf",
    "caveat-brush": "CaveatBrush/CaveatBrush-Regular.ttf",  # handwriting in marker: labels drawn on a picture
    "vt323": "VT323/VT323-Regular.ttf",  # a screen's own lettering: a camcorder's or a tape deck's readout
    "plex-mono": "IBMPlexMono/IBMPlexMono-Medium.ttf",  # one width for every letter: a receipt, a count
    "plex-mono-bold": "IBMPlexMono/IBMPlexMono-Bold.ttf",
}


def face_path(name: str) -> Path:
    if name not in FACES:
        raise ValueError(f"unknown face {name!r}")
    path = FONTS / FACES[name]
    if not path.exists():
        raise NeedSetup(f"the font {FACES[name]} is missing from the pack's fonts folder; get the pack again")
    return path


@lru_cache(maxsize=256)
def font(name: str, px: int):
    from PIL import ImageFont

    return ImageFont.truetype(str(face_path(name)), max(4, int(px)))


@lru_cache(maxsize=64)
def cap_height(name: str, px: int) -> float:
    """How tall a capital H stands in this face at this size, in pixels."""
    l, t, r, b = font(name, px).getbbox("H", anchor="ls")
    return float(-t)


def advance(text: str, name: str, px: int) -> float:
    """How far the pen moves for `text`: the width to lay the next word after."""
    return float(font(name, px).getlength(text))


def ink(text: str, name: str, px: int) -> tuple[float, float, float, float]:
    """The ink's box (left, top, right, bottom) relative to the start of the baseline."""
    l, t, r, b = font(name, px).getbbox(text, anchor="ls")
    return float(l), float(t), float(r), float(b)


def fit_px(text: str, name: str, width: float, lo: int = 6, hi: int = 1200) -> int:
    """The largest size at which `text` is at most `width` pixels across."""
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if advance(text, name, mid) <= width:
            lo = mid
        else:
            hi = mid - 1
    return lo


@dataclass
class Word:
    """A drawn word: its sprite, and where the start of its baseline sits inside the sprite."""
    sprite: Sprite
    ox: float
    oy: float
    width: float  # the pen's advance
    ascent: float  # the face's cap height at this size: what lines are spaced by


def draw_word(text: str, name: str, px: int, fill=(255, 255, 255), outline: float = 0.0,
              outline_fill=(0, 0, 0), shadow: Optional[tuple[float, float, float, float]] = None) -> Word:
    """One word as a sprite. `outline` is the stroke in pixels; `shadow` is (dx, dy, blur, opacity)."""
    from PIL import Image, ImageDraw, ImageFilter

    f = font(name, px)
    stroke = int(round(outline))
    l, t, r, b = f.getbbox(text, anchor="ls", stroke_width=stroke)
    pad = 2 + stroke
    if shadow:
        dx, dy, blur, _ = shadow
        pad += int(np.ceil(blur * 3 + max(abs(dx), abs(dy))))
    w, h = int(np.ceil(r - l)) + 2 * pad, int(np.ceil(b - t)) + 2 * pad
    ox, oy = pad - l, pad - t
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if shadow:
        dx, dy, blur, opacity = shadow
        m = Image.new("L", (w, h), 0)
        ImageDraw.Draw(m).text((ox + dx, oy + dy), text, font=f, fill=255, anchor="ls",
                               stroke_width=stroke, stroke_fill=255)
        if blur > 0:
            m = m.filter(ImageFilter.GaussianBlur(blur))
        alpha = (np.asarray(m, np.float32) * opacity).astype(np.uint8)
        dark = np.zeros((h, w, 4), np.uint8)
        dark[..., 3] = alpha
        img = Image.alpha_composite(img, Image.fromarray(dark, "RGBA"))
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((ox, oy), text, font=f, fill=tuple(fill) + (255,), anchor="ls",
                               stroke_width=stroke, stroke_fill=tuple(outline_fill) + (255,))
    img = Image.alpha_composite(img, layer)
    return Word(Sprite.from_rgba(np.asarray(img)), ox, oy, advance(text, name, px), cap_height(name, px))


def rounded_box(w: float, h: float, radius: float, fill=(0, 0, 0), opacity: float = 1.0) -> Sprite:
    """A filled box with round corners, drawn four times larger and brought down so its edge is smooth."""
    from PIL import Image, ImageDraw

    k = 4
    W, H = max(1, int(round(w))), max(1, int(round(h)))
    img = Image.new("RGBA", (W * k, H * k), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle((0, 0, W * k - 1, H * k - 1), radius=max(0, radius * k),
                                          fill=tuple(fill) + (int(round(255 * opacity)),))
    img = img.resize((W, H), Image.LANCZOS)
    return Sprite.from_rgba(np.asarray(img))


def hex_rgb(value: str) -> tuple[int, int, int]:
    """'#ffd400', 'ffd400' or '#fd0' -> (255, 212, 0)."""
    v = value.strip().lstrip("#")
    if len(v) == 3:
        v = "".join(c * 2 for c in v)
    if len(v) != 6 or any(c not in "0123456789abcdefABCDEF" for c in v):
        raise ValueError(f"{value!r} is not a colour; use a hex colour like #ffd400")
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


COLOURS = {
    "white": "#ffffff", "yellow": "#ffd400", "green": "#3ddc84", "red": "#ff4d4d", "blue": "#4da3ff",
    "pink": "#ff6fb5", "orange": "#ff9a1f", "purple": "#a678ff", "cyan": "#33e0e0", "black": "#111111",
}


def colour(value: str) -> tuple[int, int, int]:
    """A colour by name ('yellow') or hex ('#ffd400')."""
    return hex_rgb(COLOURS.get(value.strip().lower(), value))
