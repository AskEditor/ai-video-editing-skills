"""Frosted glass: a pane that blurs the picture behind it, with light along its edge and a soft shadow
under it, so a card reads as a sheet of glass held over the video rather than a sticker on it.

    pane = glass.Pane(w, h, radius, unit=s)              # its shape, its light, its shadow: made once
    pane.tone_over(frames, x, y)                         # smoked just enough for white type over these
    pane.draw(frame, x, y)                               # each frame: what is behind, blurred, bent, lit
    pane.draw(frame, x, y, opacity=a, frost=f, scale=k)  # arriving: fading in, frosting over, growing

`unit` is the size of one design pixel on the real frame (the frame's width over 1080, times the
pane's own scale); the blur, the edge and the shadow follow it, so a pane looks the same at any size.

What the glass does to what is behind it is its `Tone`:

    Tone(smoke=0.55)              # dark smoked glass, for white type: what `tone_over` chooses
    Tone(smoke=0.0, lift=0.35)    # light frosted glass, for dark type: what `light_tone_for` chooses
    Tone(smoke=0.5, tint=(48, 12, 24))   # smoked toward a colour of your own

What makes it glass, each part made once from the pane's shape (a rounded rectangle measured as a
distance from its edge, so every part follows the same curve):

- **the blur** of whatever is behind it, worked out on every frame, because the picture moves;
- **the bend at the rim:** near the edge the pane shows the picture just outside it, pulled in,
  the way a thick pane bends light at its edge;
- **the tone:** smoked toward a cool slate just enough that white type on it stays readable however
  bright the room is (`tone_over` measures the brightest of the moments it is given), graded a
  little lighter at the top; over a dark room, a faint white frost instead, so it still reads as glass;
- **the light:** a thin rim lit from the upper left and faintly from below, a second line inside
  it, a sheen across the top, and a little shade inside the lower edge, which gives it thickness;
- **the shadow:** soft and a little below, with a hairline at the edge that parts it from a bright room.

Only the picture under the pane (and a margin) is blurred, a quarter size and brought back up, so a
pane costs a few milliseconds a frame however large the video is. `frosted=False` is a plain pane of
`fill` with the same edge and shadow, for when the picture behind should not show through.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .compose import Sprite, over

LIGHT = (-0.55, -0.83)  # where the light comes from: the upper left
SLATE = (16.0, 19.0, 28.0)  # what the glass is smoked toward unless told otherwise
TARGET = 116.0  # the brightest the glass may be (0..255) for white type on it to read at about 4.5 : 1
GRADE = 0.06  # the top of a pane is this much less smoked than its middle, the bottom this much more


@dataclass(frozen=True)
class Tone:
    """What the glass does to the picture behind it, after the blur."""
    smoke: float = 0.0  # how far it is taken toward `tint` (0..1), at the pane's middle
    lift: float = 0.0  # then how much white is laid over it (0..1): a frost
    vivid: float = 1.15  # colour kept a little stronger, as it is through real glass
    tint: tuple[float, float, float] = SLATE

    def brightest(self, behind: float) -> float:
        """The pane's brightness (0..255) at its lightest, the top, over a picture of `behind`."""
        k = max(0.0, self.smoke - GRADE)
        return (behind * (1 - k) + brightness(self.tint) * k) * (1 - self.lift) + 255 * self.lift

    def darkest(self, behind: float) -> float:
        """The pane's brightness (0..255) at its darkest, the bottom, over a picture of `behind`."""
        k = min(0.95, self.smoke + GRADE)
        return (behind * (1 - k) + brightness(self.tint) * k) * (1 - self.lift) + 255 * self.lift


def brightness(colour: Sequence[float]) -> float:
    """How bright a colour is (0..255), as the eye weighs red, green and blue."""
    return float(np.dot(np.asarray(colour, np.float32)[:3], (0.2126, 0.7152, 0.0722)))


def luma(img: np.ndarray) -> np.ndarray:
    """The brightness of every pixel of an RGB picture (0..255)."""
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


def _linear(v: float) -> float:
    c = max(0.0, min(1.0, v / 255.0))
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def contrast_with_white(ground: float) -> float:
    """How well white type stands out on a ground of brightness `ground` (0..255): 21 on black,
    1 on white; 4.5 reads comfortably."""
    return 1.05 / (_linear(ground) + 0.05)


def contrast(a: float, b: float) -> float:
    """How well two brightnesses (0..255) stand apart, type on a ground: 1 for the same, 21 for black
    on white; 4.5 reads comfortably."""
    hi, lo = max(_linear(a), _linear(b)), min(_linear(a), _linear(b))
    return (hi + 0.05) / (lo + 0.05)


def _sdf(w: int, h: int, r: float, pad: int = 0) -> np.ndarray:
    """The signed distance from the edge of a w x h rounded rectangle (negative inside), on a grid
    `pad` pixels larger on every side."""
    ys, xs = np.mgrid[-pad:h + pad, -pad:w + pad].astype(np.float32)
    xs += 0.5
    ys += 0.5
    qx = np.abs(xs - w / 2.0) - (w / 2.0 - r)
    qy = np.abs(ys - h / 2.0) - (h / 2.0 - r)
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    inside = np.minimum(np.maximum(qx, qy), 0)
    return outside + inside - r


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """A wide blur, worked out on a quarter-size copy and brought back up (much faster, and a blur
    this wide loses nothing by it)."""
    import cv2

    h, w = img.shape[:2]
    k = 4 if min(h, w) >= 64 else 1
    small = cv2.resize(img, (max(1, w // k), max(1, h // k)), interpolation=cv2.INTER_AREA) if k > 1 else img
    small = cv2.GaussianBlur(small, (0, 0), max(0.5, sigma / k))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR) if k > 1 else small


SHADOW_PAD = 22.0  # the shadow's canvas past the pane on each side, in units
SHADOW_DY = 7.0  # how far below the pane its shadow falls, in units


class Pane:
    """One pane of glass, w x h pixels, with corners of `radius` (half the shorter side makes a pill
    or a circle). `shadow` is how dark its shadow is (0 for none); `light` where its light comes from."""

    @staticmethod
    def reach(unit: float = 1.0) -> tuple[float, float, float, float]:
        """How far a pane's drawing (its shadow and edge) reaches past the pane itself, in pixels:
        (left, top, right, bottom). What is kept inside the safe zone is the pane with this round it."""
        sp = math.ceil(SHADOW_PAD * unit)
        return float(sp), float(max(0.0, sp - SHADOW_DY * unit)), float(sp), float(sp + SHADOW_DY * unit)

    def __init__(self, w: int, h: int, radius: float, unit: float = 1.0, frosted: bool = True,
                 fill: Sequence[int] = (20, 20, 25), shadow: float = 0.34,
                 light: tuple[float, float] = LIGHT):
        import cv2

        self.w, self.h = max(1, int(w)), max(1, int(h))
        self.r = float(max(0.0, min(radius, self.w / 2, self.h / 2)))
        self.u = float(unit)
        self.frosted = frosted
        self.fill = np.array(fill, np.float32)[:3]
        self.shadow_strength = float(shadow)
        self.light_from = light
        self.sigma = 15.0 * self.u  # the blur
        self.bend = 11.0 * self.u  # how wide the rim that bends the picture is
        self.pad = int(math.ceil(self.bend * 1.2 + 4))
        self._sizes: dict[float, Pane] = {}
        d = _sdf(self.w, self.h, self.r)
        self.mask = np.clip(0.5 - d, 0.0, 1.0).astype(np.float32)
        inside = np.maximum(0.0, -d)
        gy, gx = np.gradient(d)  # the outward normal, from the distance's slope
        n = np.hypot(gx, gy) + 1e-6
        nx, ny = gx / n, gy / n
        facing = nx * light[0] + ny * light[1]  # 1 on the edge the light falls on, -1 opposite
        # the light: a thin rim (bright where the light falls, fainter below), a second line inside
        # it where the glass's thickness catches the light, a glow, a sheen across the top, a glare
        rim_w = max(1.0, 1.5 * self.u)
        rim = np.clip(1.0 - np.abs(inside - rim_w * 0.55) / rim_w, 0.0, 1.0)
        rim *= 0.22 + 0.66 * np.maximum(0, facing) ** 1.3 + 0.34 * np.maximum(0, -facing) ** 2
        inner = 0.10 * np.clip(1.0 - np.abs(inside - 3.2 * self.u) / (1.1 * self.u), 0, 1) * np.maximum(0, facing)
        glow = 0.06 * np.exp(-inside / (7.0 * self.u)) * (0.4 + 0.6 * np.maximum(0, facing))
        ys = np.arange(self.h, dtype=np.float32)[:, None] / self.h
        xs = np.arange(self.w, dtype=np.float32)[None, :] / self.w
        sheen = 0.07 * np.clip(1.0 - ys / 0.45, 0, 1) ** 2
        glare = 0.035 * np.clip(1.0 - ((xs * self.w + ys * self.h) / (0.9 * (self.w + self.h))) ** 2 * 2.2, 0, 1)
        self.light = np.clip(rim + inner + glow + sheen + glare, 0, 0.92).astype(np.float32)[..., None]
        # the smoke grades from the top (lighter) to the bottom (darker), as light falls through it
        self.grade = ((ys - 0.5) * 2.0 * GRADE).astype(np.float32)[..., None]
        # thickness: a little shade inside the lower edge
        self.shade = (0.13 * np.exp(-inside / (7.0 * self.u)) * np.maximum(0, ny) ** 1.5).astype(np.float32)[..., None]
        # the bend: near the rim the pane shows the picture just outside it, pulled in
        t = np.clip(1.0 - inside / self.bend, 0.0, 1.0) ** 2
        gx_, gy_ = np.meshgrid(np.arange(self.w, dtype=np.float32), np.arange(self.h, dtype=np.float32))
        self.map_x = (gx_ + self.pad + nx * t * self.bend * 1.05).astype(np.float32)
        self.map_y = (gy_ + self.pad + ny * t * self.bend * 1.05).astype(np.float32)
        # a fine grain, the same on every frame, so a smooth blur never shows bands
        self.grain = (np.random.default_rng(11).standard_normal((self.h, self.w, 1)) * 1.3).astype(np.float32)
        # the shadow, soft and a little below, and the hairline at the edge
        sp = int(math.ceil(SHADOW_PAD * self.u))
        self.shadow_pad = sp
        sd = _sdf(self.w, self.h, self.r, pad=sp)
        soft = cv2.GaussianBlur(np.clip(0.5 - sd, 0.0, 1.0).astype(np.float32), (0, 0), max(1.0, 9.0 * self.u))
        black = np.zeros(sd.shape + (3,), np.float32)
        self.shadow = Sprite(rgb=black, a=(soft * self.shadow_strength).astype(np.float32))
        self.shadow_dy = SHADOW_DY * self.u
        hairline = np.clip(1.0 - np.abs(sd - 0.6) / 0.9, 0.0, 1.0) * (sd > 0)
        self.edge = Sprite(rgb=black, a=(hairline * 0.16 * min(1.0, self.shadow_strength / 0.34)).astype(np.float32))
        self.tone = Tone()

    # -- the tone -----------------------------------------------------------------------------------
    @property
    def tone(self) -> Tone:
        return self._tone

    @tone.setter
    def tone(self, tone: Tone) -> None:
        """Setting the tone works out, once, everything about the pane's colour that does not change
        from frame to frame; per frame it is then one multiply and one add."""
        self._tone = tone
        smoke = np.clip(tone.smoke + self.grade, 0.0, 0.95)
        lit = 1.0 - self.light - self.shade
        self._mul = ((1.0 - smoke) * (1.0 - tone.lift) * lit).astype(np.float32)
        add = np.asarray(tone.tint, np.float32) * smoke * (1.0 - tone.lift) + 255.0 * tone.lift + self.grain
        self._add = (add * lit + 255.0 * self.light).astype(np.float32)
        for p in self._sizes.values():
            p.tone = tone

    def tone_over(self, frames: Sequence[np.ndarray], x: float, y: float, target: float = TARGET) -> float:
        """Choose the tone for the pane at (x, y) over `frames` (the moments it is up, uint8 RGB): see
        `tone_for`. Sets it, and returns the pane's brightest brightness under it (0..255), which
        `contrast_with_white` turns into how well white type reads on it."""
        x, y = int(round(x)), int(round(y))
        regions = []
        for f in frames:
            H, W = f.shape[:2]
            regions.append(f[max(0, y - self.pad):min(H, y + self.h + self.pad),
                             max(0, x - self.pad):min(W, x + self.w + self.pad)])
        if not self.frosted:
            return brightness(self.fill)
        tone, bright = tone_for(regions, self.sigma, target)
        self.tone = tone
        return bright

    # -- drawing ------------------------------------------------------------------------------------
    def behind(self, frame: np.ndarray, x: int, y: int) -> np.ndarray:
        """What is behind the pane at (x, y), with the margin round it, as float32; the frame's edge
        is mirrored where the margin runs off it."""
        import cv2

        H, W = frame.shape[:2]
        p = self.pad
        x0, y0, x1, y1 = x - p, y - p, x + self.w + p, y + self.h + p
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            return np.zeros((y1 - y0, x1 - x0, 3), np.float32)
        region = frame[cy0:cy1, cx0:cx1]
        if (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1):
            region = cv2.copyMakeBorder(region, cy0 - y0, y1 - cy1, cx0 - x0, x1 - cx1, cv2.BORDER_REFLECT)
        return region.astype(np.float32)

    def colours(self, frame: np.ndarray, x: int, y: int, frost: float = 1.0) -> np.ndarray:
        """The pane's own colours at (x, y), before it is laid on the frame: what is behind it
        blurred, bent at the rim, toned and lit (float32, h x w x 3). With `frost` under 1 the pane
        is part way from clear to frosted."""
        import cv2

        if not self.frosted:
            base = np.broadcast_to(self.fill, (self.h, self.w, 3)).astype(np.float32)
            return np.clip(base * (1.0 - self.light - self.shade) + 255.0 * self.light, 0, 255)
        region = self.behind(frame, x, y)
        soft = _blur(region, self.sigma)
        if frost < 1.0:
            soft = region * (1.0 - frost) + soft * frost
        base = cv2.remap(soft, self.map_x, self.map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        tone = self.tone
        vivid = 1.0 + (tone.vivid - 1.0) * frost
        if abs(vivid - 1.0) > 1e-3:
            g = luma(base)[..., None]
            base = g + (base - g) * vivid
        if frost >= 1.0:
            return np.clip(base * self._mul + self._add, 0, 255)
        smoke = np.clip((tone.smoke + self.grade) * frost, 0.0, 0.95)
        lift = tone.lift * frost
        base = base * (1.0 - smoke) + np.asarray(tone.tint, np.float32) * smoke
        base = base * (1.0 - lift) + 255.0 * lift + self.grain * frost
        return np.clip(base * (1.0 - self.light - self.shade) + 255.0 * self.light, 0, 255)

    def scaled(self, k: float) -> "Pane":
        """The same pane drawn `k` times its size (made once for each size and kept), for a pane
        that grows into place."""
        k = round(float(k) / 0.02) * 0.02
        if abs(k - 1.0) < 0.01:
            return self
        k = max(0.1, k)
        if k not in self._sizes:
            p = Pane(round(self.w * k), round(self.h * k), self.r * k, self.u * k, self.frosted, tuple(self.fill),
                     self.shadow_strength, self.light_from)
            p.tone = self.tone
            self._sizes[k] = p
        return self._sizes[k]

    def draw(self, frame: np.ndarray, x: float, y: float, opacity: float = 1.0, frost: float = 1.0,
             scale: float = 1.0) -> None:
        """Lay the pane on `frame` (uint8 RGB, changed in place) with its top-left at (x, y).
        `opacity` fades it, `frost` takes it from clear (0) to frosted (1), and `scale` grows it
        about its middle, which stays where it is at full size."""
        if opacity <= 0.0:
            return
        if abs(scale - 1.0) >= 0.01:
            p = self.scaled(scale)
            cx, cy = x + self.w / 2, y + self.h / 2
            if p is not self:
                p.draw(frame, cx - p.w / 2, cy - p.h / 2, opacity, frost)
                return
        x, y = int(round(x)), int(round(y))
        over(frame, self.shadow, x - self.shadow_pad, y - self.shadow_pad + self.shadow_dy, opacity)
        over(frame, self.edge, x - self.shadow_pad, y - self.shadow_pad, opacity)
        H, W = frame.shape[:2]
        cx0, cy0, cx1, cy1 = max(0, x), max(0, y), min(W, x + self.w), min(H, y + self.h)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        g = self.colours(frame, x, y, frost)
        sx0, sy0 = cx0 - x, cy0 - y
        sx1, sy1 = sx0 + (cx1 - cx0), sy0 + (cy1 - cy0)
        a = self.mask[sy0:sy1, sx0:sx1, None] * float(opacity)
        region = frame[cy0:cy1, cx0:cx1].astype(np.float32)
        region = region * (1.0 - a) + g[sy0:sy1, sx0:sx1] * a
        frame[cy0:cy1, cx0:cx1] = np.clip(region + 0.5, 0, 255).astype(np.uint8)


def tone_for(regions: Sequence[Optional[np.ndarray]], sigma: float, target: float = TARGET,
             tint: tuple[float, float, float] = SLATE) -> tuple[Tone, float]:
    """The tone for a pane over `regions` (what is behind it at a few moments, uint8 RGB), blurred
    by `sigma` as the pane blurs them: smoked toward `tint` just enough that the brightest of them
    comes to `target` at the pane's top, where it is lightest; over a dark room hardly smoked, with a
    faint white frost so it still reads as glass. Returns the tone and the pane's brightest
    brightness under it (0..255)."""
    lows = []
    for reg in regions:
        if reg is None or not reg.size:
            continue
        soft = _blur(reg.astype(np.float32), sigma)
        lows.append(float(np.percentile(luma(soft), 92)))
    behind = max(lows) if lows else 128.0
    lift = 0.03
    if behind <= target:
        tone = Tone(smoke=0.18, lift=lift + 0.05 * (1.0 - behind / target), tint=tint)
    else:
        # (behind (1 - k) + tint k)(1 - lift) + 255 lift = target, for the smoke k at the pane's top
        want = (target - 255 * lift) / (1 - lift)
        k = (behind - want) / max(1e-3, behind - brightness(tint))
        tone = Tone(smoke=min(0.9, max(0.18, k + GRADE)), lift=lift, tint=tint)
    return tone, tone.brightest(behind)


LIGHT_TARGET = 178.0  # the darkest light glass may be (0..255): dark type on it reads at about 7 : 1
LIGHT_FROST = 0.24  # light glass is lifted at least this much, so over a bright room it still reads as glass


def light_tone_for(regions: Sequence[Optional[np.ndarray]], sigma: float, target: float = LIGHT_TARGET,
                   tint: tuple[float, float, float] = SLATE) -> tuple[Tone, float]:
    """The tone for light frosted glass, for dark type: lifted toward white just enough that the
    darkest of `regions` (what is behind the pane at a few moments, uint8 RGB, blurred as the pane
    blurs them) comes to `target` at the pane's bottom, where it is darkest; over a bright room a
    little frost still, so the pane reads as glass. Returns the tone and the pane's darkest
    brightness under it (0..255), which `contrast` sets against the type's."""
    lows = []
    for reg in regions:
        if reg is None or not reg.size:
            continue
        soft = _blur(reg.astype(np.float32), sigma)
        lows.append(float(np.percentile(luma(soft), 8)))
    behind = min(lows) if lows else 128.0
    # (behind (1 - g) + tint g)(1 - lift) + 255 lift = target, with the bottom's own smoke g
    base = behind * (1 - GRADE) + brightness(tint) * GRADE
    lift = (target - base) / max(1e-3, 255.0 - base)
    tone = Tone(smoke=0.0, lift=min(0.9, max(LIGHT_FROST, lift)), vivid=1.05, tint=tint)
    return tone, tone.darkest(behind)
