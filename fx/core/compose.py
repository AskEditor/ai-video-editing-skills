"""Laying small images over frames, and laying a whole layer behind the person.

A `Sprite` is a small premultiplied RGBA image. Effects build their sprites once (a word, a box, a
card) and lay them on each frame with `over()`, so nothing is drawn again per frame.

Something that takes its look from what is behind it (frosted glass) is drawn behind the person
another way: on the picture `without_person()`, then laid back with `behind()`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Sprite:
    rgb: np.ndarray  # float32 (h, w, 3), 0..255, already multiplied by alpha
    a: np.ndarray  # float32 (h, w), 0..1

    @property
    def w(self) -> int:
        return int(self.a.shape[1])

    @property
    def h(self) -> int:
        return int(self.a.shape[0])

    @staticmethod
    def from_rgba(rgba: np.ndarray) -> "Sprite":
        rgba = np.asarray(rgba)
        a = rgba[..., 3].astype(np.float32) / 255.0
        rgb = rgba[..., :3].astype(np.float32) * a[..., None]
        return Sprite(rgb=rgb, a=a)

    def scaled(self, s: float) -> "Sprite":
        """The sprite scaled by `s` (about its centre when laid with `over(..., centre=True)`)."""
        if abs(s - 1.0) < 1e-3:
            return self
        import cv2

        w, h = max(1, int(round(self.w * s))), max(1, int(round(self.h * s)))
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
        return Sprite(rgb=cv2.resize(self.rgb, (w, h), interpolation=interp),
                      a=cv2.resize(self.a, (w, h), interpolation=interp))


def rotated(sprite: Sprite, degrees: float) -> tuple[Sprite, float, float]:
    """`sprite` turned by `degrees` (positive turns it anticlockwise, as a protractor reads) about its
    centre, on a canvas grown to hold all of it, and how far that canvas's top-left moved: lay the
    result at (x - dx, y - dy) to keep the centre where the unturned sprite's was. Turned once, when
    a look is built, never per frame."""
    if abs(degrees) < 1e-3:
        return sprite, 0.0, 0.0
    import cv2

    h, w = sprite.h, sprite.w
    m = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    c, s = abs(m[0, 0]), abs(m[0, 1])
    W, H = int(np.ceil(w * c + h * s)) + 2, int(np.ceil(w * s + h * c)) + 2
    m[0, 2] += W / 2 - w / 2
    m[1, 2] += H / 2 - h / 2
    rgb = cv2.warpAffine(sprite.rgb, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    a = cv2.warpAffine(sprite.a, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return Sprite(rgb=rgb.astype(np.float32), a=np.clip(a, 0, 1).astype(np.float32)), (W - w) / 2, (H - h) / 2


def over(frame: np.ndarray, sprite: Sprite, x: float, y: float, opacity: float = 1.0) -> None:
    """Lay `sprite` on `frame` (uint8 RGB, changed in place) with its top-left at (x, y)."""
    if opacity <= 0.0:
        return
    H, W = frame.shape[:2]
    x0, y0 = int(round(x)), int(round(y))
    x1, y1 = x0 + sprite.w, y0 + sprite.h
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    sx0, sy0 = cx0 - x0, cy0 - y0
    sx1, sy1 = sx0 + (cx1 - cx0), sy0 + (cy1 - cy0)
    a = sprite.a[sy0:sy1, sx0:sx1]
    rgb = sprite.rgb[sy0:sy1, sx0:sx1]
    if opacity < 1.0:
        a = a * opacity
        rgb = rgb * opacity
    region = frame[cy0:cy1, cx0:cx1].astype(np.float32)
    region *= 1.0 - a[..., None]
    region += rgb
    frame[cy0:cy1, cx0:cx1] = np.clip(region + 0.5, 0, 255).astype(np.uint8)


def over_centred(frame: np.ndarray, sprite: Sprite, cx: float, cy: float, scale: float = 1.0,
                 opacity: float = 1.0) -> None:
    """Lay `sprite` with its centre at (cx, cy), scaled about that centre."""
    s = sprite.scaled(scale)
    over(frame, s, cx - s.w / 2, cy - s.h / 2, opacity)


def behind_person(frame: np.ndarray, layer_rgb: np.ndarray, layer_a: np.ndarray, person_a: np.ndarray) -> np.ndarray:
    """A full-frame layer laid behind the person: it covers the background and the person stays
    on top. `layer_rgb` is premultiplied (float32 0..255), `layer_a` and `person_a` are 0..1."""
    keep = person_a[..., None]
    a = layer_a[..., None] * (1.0 - keep)
    out = frame.astype(np.float32) * (1.0 - a) + layer_rgb * (1.0 - keep)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _fill_from(img: np.ndarray, known: np.ndarray) -> np.ndarray:
    """`img` (float32, h x w x 3) with every pixel that is not `known` (0..1) filled from the known
    pixels round it: each level of a pyramid, half the size of the one above, fills that one's holes,
    so a small gap takes its colour from close by and a large one from further out."""
    import cv2

    h, w = known.shape
    if known.min() >= 0.999:
        return img
    if h <= 3 or w <= 3:
        total = float(known.sum())
        mean = (img * known[..., None]).reshape(-1, 3).sum(axis=0) / total if total > 1e-6 else np.full(3, 128.0)
        return img * known[..., None] + mean.astype(np.float32) * (1.0 - known[..., None])
    half = (max(1, w // 2), max(1, h // 2))
    k_small = cv2.resize(known, half, interpolation=cv2.INTER_AREA)
    i_small = cv2.resize(img * known[..., None], half, interpolation=cv2.INTER_AREA)
    i_small = i_small / np.maximum(k_small, 1e-6)[..., None]
    # a coarse pixel with some of the known picture in it counts as known one level down
    filled = _fill_from(i_small.astype(np.float32), np.clip(k_small * 4.0, 0.0, 1.0).astype(np.float32))
    up = cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR)
    return img * known[..., None] + up * (1.0 - known[..., None])


def without_person(frame: np.ndarray, person_a: np.ndarray, sure: float = 0.04) -> np.ndarray:
    """The picture with the person taken out: what is behind them continued in from the wall round
    them, and every pixel of plain wall (`person_a` at most `sure`) left exactly as it was. float32
    RGB, the frame's size. On the person's edge a pixel is part person and part wall; it leans to
    the wall the more of it is the person, as `behind()` expects.

    What it is for: something drawn behind the person that takes its look from what is behind it (a
    pane of glass blurs the room through it) must be drawn on this, or the person's own hair and
    shoulders show through it as a dark smear round them."""
    import cv2

    picture = frame.astype(np.float32)
    h, w = person_a.shape
    known = (person_a <= sure).astype(np.float32)
    # the fill is only ever seen blurred, or under the person's soft edge: a quarter size is plenty
    q = (max(1, w // 4), max(1, h // 4))
    small = cv2.resize(picture * known[..., None], q, interpolation=cv2.INTER_AREA)
    k_small = cv2.resize(known, q, interpolation=cv2.INTER_AREA)
    small = small / np.maximum(k_small, 1e-6)[..., None]
    wall = _fill_from(small.astype(np.float32), np.clip(k_small * 2.0, 0.0, 1.0).astype(np.float32))
    wall = cv2.resize(wall, (w, h), interpolation=cv2.INTER_LINEAR)
    lean = _smoothstep((person_a - sure) / 0.5)[..., None]
    return picture * (1.0 - lean) + wall * lean


def behind(frame: np.ndarray, drawn: np.ndarray, clean: np.ndarray, person_a: np.ndarray) -> np.ndarray:
    """What was drawn on `clean` (the frame `without_person`, float32) laid behind the person: `drawn`
    is `clean` with things drawn on it, and only the wall's share of each pixel takes the change, so
    the person stays in front and where nothing was drawn the frame comes back exactly. uint8."""
    out = frame.astype(np.float32) + (1.0 - person_a[..., None]) * (drawn.astype(np.float32) - clean)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def blank_layer(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """An empty full-frame layer to lay sprites on before putting it behind the person."""
    return np.zeros((height, width, 3), np.float32), np.zeros((height, width), np.float32)


def onto_layer(layer: tuple[np.ndarray, np.ndarray], sprite: Sprite, x: float, y: float, opacity: float = 1.0) -> None:
    """Lay a sprite onto a full-frame layer (premultiplied 'over' in float)."""
    rgb, a = layer
    H, W = a.shape
    x0, y0 = int(round(x)), int(round(y))
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x0 + sprite.w), min(H, y0 + sprite.h)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    sx0, sy0 = cx0 - x0, cy0 - y0
    sa = sprite.a[sy0:sy0 + cy1 - cy0, sx0:sx0 + cx1 - cx0] * opacity
    srgb = sprite.rgb[sy0:sy0 + cy1 - cy0, sx0:sx0 + cx1 - cx0] * opacity
    rgb[cy0:cy1, cx0:cx1] = rgb[cy0:cy1, cx0:cx1] * (1.0 - sa[..., None]) + srgb
    a[cy0:cy1, cx0:cx1] = a[cy0:cy1, cx0:cx1] * (1.0 - sa) + sa
