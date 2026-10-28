"""A closer framing on the face: where the window goes, how it moves, and the picture inside it.

Shared by every effect that moves the camera (`transitions`' punch, `punch-in`, `hook`'s snap):

- `punch_window(size, zoom, head, centre)`: the window for a framing `zoom` times closer, about the
  face where it already is, with the whole head in, a tenth of the window above the crown and the
  band under the chin kept free for captions. A crown the recording already cut is left as shot.
  It answers with the zoom it could take, and says why when that is less than asked.
- `place_window(...)`: the same window at a zoom already decided (no search).
- `between(w0, w1, p, size)`: a window part of the way from one framing to another, never wider
  than the picture.
- `punch_progress(tau, d)`: how far a punch has gone, `tau` seconds after it lands: a short wind-up,
  past the new framing on the moment, and settled back onto it.
- `crop(img, window)`: the window filling the frame (an exact affine warp, so a window between
  pixels never steps).
- `ease(x)`: smooth in and out, 0 to 1.
- `CameraPath(size, moves)`: a camera over time: the whole frame, then each `Move` (a snap, a slow
  push, a cut, a smooth return) in turn, holding between them; `frame(img, t, fps)` is the picture
  it shows, smeared along the move while it moves fast.
- `aim(face, framing, first, last)`: the head over a stretch of frames, in output pixels, and where
  to aim at it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .layout import Rect

CROWN = 0.10  # the room kept above the top of the head in a punched frame, as a share of its height
CHIN_LINE = 0.70  # a punch keeps the bottom of the head above this share of the frame's height


def ease(x):
    """Smooth in and out, 0 to 1 (numbers or arrays)."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def place_window(size: tuple[int, int], z: float, head: Optional[Rect], cx: float, cy: float,
                 line: Optional[float] = None) -> tuple[float, float, float, float]:
    W, H = size
    w, h = W / z, H / z
    # Zoom about the face where it already is: it grows on the spot instead of being moved to the
    # middle of the frame (which would drop the chin towards the captions' band and the bottom fifth).
    x, y = cx - (cx / W) * w, cy - (cy / H) * h
    if head is not None:
        top = head.y0 - CROWN * h  # the window may start no lower than this (the crown's room)...
        chin = head.y1 + 0.04 * h - h  # ...and no higher than this (the chin in)
        if line is not None:  # nor so high that the chin drops past the line: the face rises instead
            chin = max(chin, head.y1 - line / z)
        y = min(y, top)
        if chin <= top:
            y = max(y, chin)
        x = min(x, head.x0 - 0.04 * w)
        x = max(x, head.x1 + 0.04 * w - w)
    x = min(max(0.0, x), W - w)
    y = min(max(0.0, y), H - h)
    return x, y, w, h


def punch_window(size: tuple[int, int], zoom: float, head: Optional[Rect], centre: Optional[tuple[float, float]]
                 ) -> tuple[tuple[float, float, float, float], float, str]:
    """The window for a framing `zoom` times closer, about the face (it grows where it is), keeping
    the whole head in with a tenth of the window above the crown, and the band under the chin free
    for the captions that come after (the bottom of the head above 70% of the frame, or no lower
    than it already was). A crown the recording itself already cut is left as it was shot (the
    window keeps to the top edge) rather than holding the punch back. Returns the window (x, y, w,
    h), the zoom it could take, and a note when that is less than asked."""
    W, H = size
    note = ""
    z = max(1.0, zoom)
    cx, cy = centre if centre is not None else (W / 2, H * 0.45)
    crown_cut = head is not None and head.y0 <= 0.01 * H
    if head is not None and z > 1.0:
        seen = Rect(head.x0, max(0.0, head.y0), head.x1, head.y1)
        room = 0.94 if crown_cut else (1 - CROWN - 0.06)  # the crown's room, and a little under the chin
        most = max(1.0, min(0.86 * W / max(1.0, seen.w), room * H / max(1.0, seen.h)))
        if most < z - 1e-3:
            note = f"the head leaves room for {most:.2f}x, not {z:.2f}x"
            z = most
        line = max(CHIN_LINE * H, head.y1)
        roomy = z
        while z > 1.0:
            x, y, w, h = place_window(size, z, head, cx, cy, line)
            if (head.y1 - y) * (H / h) <= line + 0.5:
                break
            z = max(1.0, z - 0.01)
        if z < roomy - 1e-3:
            note = f"{z:.2f}x, to keep the band under the chin free for captions"
        return place_window(size, z, head, cx, cy, line), z, note
    return place_window(size, z, head, cx, cy), z, note


def crop(img: np.ndarray, win: tuple[float, float, float, float]) -> np.ndarray:
    """The window of `img`, filling the frame: an affine warp, so a window between pixels is exact."""
    import cv2

    H, W = img.shape[:2]
    x, y, w, h = win
    if w >= W - 0.5 and h >= H - 0.5 and abs(x) < 0.5 and abs(y) < 0.5:
        return img
    sx, sy = W / w, H / h
    mat = np.float32([[sx, 0, -x * sx], [0, sy, -y * sy]])
    return cv2.warpAffine(img, mat, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def between(w0, w1, p: float, size: Optional[tuple[int, int]] = None):
    """The window `p` of the way from w0 to w1 (past 1 overshoots), kept inside the picture: a
    framing can go closer than the shot, never wider than it."""
    x, y, w, h = (a + (b - a) * p for a, b in zip(w0, w1))
    if size is not None:
        W, H = size
        k = min(1.0, W / max(1e-6, w), H / max(1e-6, h))
        cx, cy = x + w / 2, y + h / 2
        w, h = w * k, h * k
        x = min(max(0.0, cx - w / 2), W - w)
        y = min(max(0.0, cy - h / 2), H - h)
    return x, y, w, h


def punch_progress(tau: float, d: float) -> float:
    """How far a punch has gone (0 to 1, with a little overshoot), `tau` seconds after its join: it
    starts just before, lands past the new framing on the join, and settles back onto it."""
    pre = 0.2 * d
    if tau < -pre:
        return 0.0
    if tau < 0:
        return 0.45 * float(ease((tau + pre) / pre)) ** 1.5
    settle = max(0.05, 0.8 * d)
    return 1.0 + 0.09 * math.exp(-tau / (0.35 * settle)) * math.cos(math.pi * tau / settle)


# ---- a camera that moves over time: a path of framings ------------------------------------------

Window = tuple[float, float, float, float]

CURVES = ("smooth", "snap", "slow", "cut")


def curve(kind: str, u: float) -> float:
    """How far (0 to 1) a move of this kind has gone `u` of the way through its time.

    - smooth: eased in and out, for going back to the whole frame;
    - snap: fast off the mark and landing softly, for a quick punch onto a word;
    - slow: a gentle start and finish with a steady middle, for a push or a drift across a line;
    - cut: all at once, at its start."""
    u = min(1.0, max(0.0, u))
    if kind == "cut":
        return 1.0
    if kind == "snap":
        return 1.0 - (1.0 - u) ** 3
    if kind == "slow":  # smootherstep: zero speed and zero acceleration at both ends
        return u * u * u * (u * (6.0 * u - 15.0) + 10.0)
    return float(ease(u))


@dataclass
class Move:
    """The camera going from wherever it is at `start` to the framing `to`, arriving at `end`
    (output seconds). `settle` (seconds) lets a snap land a touch past `to` and settle back onto it,
    as a hand-held punch does."""
    start: float
    end: float
    to: Window
    kind: str = "smooth"
    settle: float = 0.0


class CameraPath:
    """The framing at every moment, from the whole frame through each move in turn. Between moves
    the framing holds. A move that starts before the last one arrived leaves from where the camera
    is at that moment, so moves never jump."""

    def __init__(self, size: tuple[int, int], moves: list[Move]):
        self.size = (int(size[0]), int(size[1]))
        self.full: Window = (0.0, 0.0, float(self.size[0]), float(self.size[1]))
        self.moves = sorted(moves, key=lambda m: m.start)
        self._from: list[Window] = []
        for k, m in enumerate(self.moves):
            self._from.append(self._chain(k, m.start))

    def _chain(self, upto: int, t: float) -> Window:
        w = self.full
        for k in range(upto):
            m = self.moves[k]
            if t < m.start:
                break
            w = self._through(k, t)
        return w

    def _through(self, k: int, t: float) -> Window:
        m, w0 = self.moves[k], self._from[k]
        if m.kind == "cut" or t >= m.end:
            if m.settle > 0 and m.kind != "cut" and t < m.end + m.settle:
                tau = (t - m.end) / m.settle  # a small overshoot past `to`, then back onto it
                return between(w0, m.to, 1.0 + 0.06 * math.sin(math.pi * tau) * (1.0 - tau), self.size)
            return m.to
        u = (t - m.start) / max(1e-6, m.end - m.start)
        return between(w0, m.to, curve(m.kind, u), self.size)

    def at(self, t: float) -> Window:
        """The framing at output second `t`."""
        return self._chain(len(self.moves), t)

    def zoom(self, t: float) -> float:
        return self.size[0] / max(1e-6, self.at(t)[2])

    def moving(self, t: float, dt: float) -> bool:
        a, b = self.at(t - dt), self.at(t)
        return max(abs(p - q) for p, q in zip(a, b)) > 0.004 * self.size[0]

    def frame(self, img: np.ndarray, t: float, fps: float, blur: bool = True) -> np.ndarray:
        """`img` (a whole output frame) as the camera frames it at `t`. While the framing moves fast
        enough to show, the frame is smeared over the second half of its own movement, as a camera's
        shutter would: a zoom blur that follows the move, never a fixed amount."""
        here = self.at(t)
        out = crop(img, here)
        dt = 1.0 / fps
        if not blur or t < dt or not self.moving(t, dt):
            return out
        if any(m.kind == "cut" and t - dt < m.start <= t + 1e-9 for m in self.moves):
            return out  # a cut is a new framing, not a movement: nothing to smear
        prev = self.at(t - dt)
        acc = out.astype(np.float32)
        travel = max(abs(p - q) for p, q in zip(prev, here)) / max(1.0, self.size[0])
        steps = np.linspace(0.5, 0.95, int(np.clip(round(travel * 160), 6, 18)))  # enough steps that the smear never bands
        for s in steps:
            acc += crop(img, between(prev, here, float(s))).astype(np.float32)
        return (acc / (len(steps) + 1)).astype(np.uint8)


def aim(face, framing, first: int, last: int) -> tuple[Optional[Rect], Optional[tuple[float, float]]]:
    """The head to frame over source frames first..last, in output pixels: the box round it in every
    frame (so a movement inside a held framing never takes the crown out), and where to aim (the
    middle of the face, a little below the eyes). (None, None) when no face is found there."""
    if face is None or not face.any():
        return None, None
    step = max(1, (last - first) // 40 or 1)
    heads = [framing.to_output(h, j) for j in range(first, last + 1, step) if (h := face.head(j)) is not None]
    if not heads:
        return None, None
    union = Rect.around(heads)
    cx = float(np.median([h.cx for h in heads]))
    cy = float(np.median([h.y0 + 0.55 * h.h for h in heads]))
    return union, (cx, cy)
