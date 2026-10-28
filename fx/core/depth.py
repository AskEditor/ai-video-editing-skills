"""Pictures laid in depth in the room behind the person: cards that sit at a distance, turn, cast a
soft shadow on the wall and move a little with the camera.

What is here was first built for `backdrop`'s screens and is shared so every effect that lays
pictures in the room draws them the same way:

- `Card`: a rectangle in the room (its middle, size, turn about its upright axis, roll in its own
  plane, and depth), and `Card.corners()`: where its four corners fall on the screen, in
  perspective.
- `draw_card()`: a picture laid into such a quad on a canvas, with its soft shadow put into a
  shade map below it, or (`darken`) cast at once on what is already drawn, so a card on a pile
  shades the cards under it and never itself.
- `rounded_alpha()`: a rounded rectangle's alpha.
- `seen_share()`: how much of a box the person leaves in view over the clip, from a coverage map
  (the share of frames the person covers each spot).
- `head_path()`: per output frame, how far the head is from where it usually is, smoothed.
- `drift()`: the room's sideways and upward drift at a moment: a slow sway plus part of the head's
  own movement. A card at depth d moves by the drift divided by d (parallax).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .layout import Rect  # noqa: F401  (head_path takes the usual head as a Rect)


def seen_share(coverage: Optional[np.ndarray], box: tuple[float, float, float, float], size: tuple[int, int]) -> float:
    """How much of `box` (x0, y0, x1, y1, output pixels) the person leaves in view over the clip,
    from the coverage map (the share of frames the person covers each spot), counting only the
    part inside the frame."""
    W, H = size
    x0, y0, x1, y1 = max(0.0, box[0]), max(0.0, box[1]), min(float(W), box[2]), min(float(H), box[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    if coverage is None:
        return 1.0
    ch, cw = coverage.shape
    a, b = int(y0 * ch / H), max(int(y0 * ch / H) + 1, int(math.ceil(y1 * ch / H)))
    c, d = int(x0 * cw / W), max(int(x0 * cw / W) + 1, int(math.ceil(x1 * cw / W)))
    inside = (x1 - x0) * (y1 - y0) / max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    return float((1.0 - coverage[a:b, c:d].mean()) * inside)



def rounded_alpha(w: int, h: int, radius: float) -> np.ndarray:
    """A rounded rectangle's alpha (0..1), smooth at the corners."""
    from PIL import Image, ImageDraw

    k = 4
    img = Image.new("L", (w * k, h * k), 0)
    ImageDraw.Draw(img).rounded_rectangle((0, 0, w * k - 1, h * k - 1), radius=max(0.0, radius * k), fill=255)
    return np.asarray(img.resize((w, h), Image.LANCZOS), np.float32) / 255.0



@dataclass
class Card:
    """A rectangle laid in the room: where it sits, its tilt and depth, and its picture."""
    cx: float
    cy: float
    w: float
    h: float
    tilt: float = 0.0  # degrees about its upright axis (the far side turned away)
    roll: float = 0.0  # degrees in the picture's plane
    depth: float = 1.0  # 1 is nearest; larger moves less with the camera
    radius: float = 0.02  # corner radius as a share of its width

    def corners(self, scale: float = 1.0, dx: float = 0.0, dy: float = 0.0, focal: float = 2000.0) -> np.ndarray:
        """Its four corners on the screen (top-left, top-right, bottom-right, bottom-left)."""
        w, h = self.w * scale / 2, self.h * scale / 2
        pts = np.array([[-w, -h], [w, -h], [w, h], [-w, h]], np.float32)
        a = math.radians(self.tilt)
        x3, z3 = pts[:, 0] * math.cos(a), pts[:, 0] * math.sin(a)
        px = focal * x3 / (focal + z3)
        py = focal * pts[:, 1] / (focal + z3)
        r = math.radians(self.roll)
        rx = px * math.cos(r) - py * math.sin(r)
        ry = px * math.sin(r) + py * math.cos(r)
        return np.stack([rx + self.cx + dx, ry + self.cy + dy], axis=1).astype(np.float32)


def draw_card(canvas: np.ndarray, shade: np.ndarray, img: np.ndarray, alpha: np.ndarray, quad: np.ndarray,
              opacity: float = 1.0, shadow: float = 0.45, shadow_px: float = 18.0, dim: float = 0.0,
              darken: float = 0.0) -> None:
    """Lay `img` (with its rounded `alpha`) onto `canvas` (float32, changed in place) inside `quad`,
    and its soft shadow into `shade` (0..1, deepened in place) below it."""
    import cv2

    H, W = canvas.shape[:2]
    pad = int(3 * shadow_px) + 4
    x0, y0 = np.floor(quad.min(axis=0)).astype(int) - pad
    x1, y1 = np.ceil(quad.max(axis=0)).astype(int) + pad
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0 or opacity <= 0.002:
        return
    ih, iw = img.shape[:2]
    src = np.array([[0, 0], [iw, 0], [iw, ih], [0, ih]], np.float32)
    M = cv2.getPerspectiveTransform(src, quad - np.array([x0, y0], np.float32))
    size = (x1 - x0, y1 - y0)
    col = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    a = cv2.warpPerspective(alpha, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) * opacity
    if shadow > 0:  # soft, so it is worked out at a quarter size
        q = 4
        qs = (max(2, size[0] // q), max(2, size[1] // q))
        s = cv2.resize(a, qs, interpolation=cv2.INTER_AREA)
        s = cv2.warpAffine(s, np.float32([[1, 0, 0], [0, 1, shadow_px * 0.6 / q]]), qs,
                           borderMode=cv2.BORDER_CONSTANT)
        s = cv2.GaussianBlur(s, (0, 0), max(0.8, shadow_px / q))
        s = cv2.resize(s, size, interpolation=cv2.INTER_LINEAR)
        if darken > 0:  # the shadow falls now, on what is already drawn, and under this card only
            canvas[y0:y1, x0:x1] *= (1 - darken * shadow * s)[..., None]
        else:
            shade[y0:y1, x0:x1] = 1 - (1 - shade[y0:y1, x0:x1]) * (1 - shadow * s)
    region = canvas[y0:y1, x0:x1]
    region *= 1 - a[..., None]
    c = col.astype(np.float32)
    if dim > 0:
        c *= 1 - dim
    region += c * a[..., None]



def head_path(clip, framing, n_out: int, fps: float, head: Optional[Rect]) -> Optional[np.ndarray]:
    """Per output frame, how far the head is from where it usually is (output pixels), smoothed:
    what the screens' parallax follows."""
    if head is None or not clip.face.any():
        return None
    pts = []
    for j in range(n_out):
        k = min(clip.info.frames - 1, j)
        b = clip.face.box(k)
        if b is None:
            pts.append((0.0, 0.0))
            continue
        r = framing.to_output(b, k)
        pts.append((r.cx - head.cx, r.cy - (head.y0 + head.y1) / 2))
    p = np.array(pts, np.float64)
    sigma = 0.35 * fps
    rad = max(1, int(3 * sigma))
    kern = np.exp(-0.5 * (np.arange(-rad, rad + 1) / sigma) ** 2)
    kern /= kern.sum()
    for c in (0, 1):
        p[:, c] = np.convolve(np.pad(p[:, c], rad, mode="edge"), kern, mode="valid")
    return p



def drift(t: float, size: tuple[int, int], head: Optional[np.ndarray], fps: float) -> tuple[float, float]:
    """The room's sideways and upward drift at t (output pixels): a slow sway, plus a part of the
    head's own movement (`head` is `head_path()`, or None), so cards move a little as a real room
    would behind someone filmed by hand."""
    W, H = size
    dx = 0.010 * W * math.sin(2 * math.pi * t / 9.0)
    dy = 0.004 * H * math.sin(2 * math.pi * t / 7.0 + 1.0)
    if head is not None and len(head):
        j = int(np.clip(round(t * fps), 0, len(head) - 1))
        dx += 0.22 * float(head[j, 0])
        dy += 0.12 * float(head[j, 1])
    return dx, dy
