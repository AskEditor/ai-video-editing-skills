"""The person over a whole new room: an edge with no rim of the old room, solid inside, and steady.

`person.full()` gives the edge for a layer that covers part of the frame (a word behind a head).
A new room behind the whole person shows every pixel of the edge, against a colour that may be
nothing like the old wall, so it needs more. Each point below was measured on the test clips
(28 October 2026):

- **Firm.** Across a dark cap on clip B's bright ceiling, the guided edge keeps a 10–20% tail of
  the old wall for eight pixels outside (a pale halo on the new room) and lets the new room 10–35%
  into the cap for six pixels inside. `alpha()` firms the edge round its middle, so the tail goes
  and the inside is solid, without the jagged edge a pixel-by-pixel colour decision gave on hair.
- **No old wall in the edge.** A pixel on the edge is part person, part old wall. `over()` takes
  the old wall's share out, using the wall's colour from nearby, and puts the new room's in. Where
  a pixel is mostly wall its own colour says little about the person, so it leans on the person's
  colour just inside instead (otherwise the old wall's texture rides along in the halo).
- **Steady.** With nothing moving, the model's edge on clip D changed by 10% a frame on average.
  `Steady` averages each mask with its neighbours wherever the picture itself did not change, so a
  still edge holds and a moving one does not trail behind.
- **Only the speaker.** A passer-by or a stray patch that is not joined to the speaker is dropped
  (`main_only`); the speaker's own arms and a held microphone stay.
- **Two models agreeing.** `Steady` takes several masks of the clip and keeps only what all of them
  call the person: the core's two models take different parts of a room for a person.

    steady = matte.Steady(clip.person(), clip.info)
    m = matte.main_only(steady.get(k), head_box_in_mask_pixels)
    a = matte.alpha(masks, m, (w, h), guide=frame)
    out = matte.over(frame, a, room)   # room: float32 (h, w, 3), 0..255
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from .person import PersonMasks, _read_gray
from .video import FrameReader, VideoInfo

# The edge is firmed between these levels of the guided mask: below LOW is wall, above HIGH is
# the person. Half way stays where the guided edge put it.
LOW, HIGH = 0.25, 0.75
# Steady: how far each side neighbours are averaged (seconds; at least 2 and at most 6 frames),
# and how much a pixel may change (grey levels, 0..255, at the mask's size, softened) before it
# counts as movement rather than noise. Measured on A, B and D: two frames each side cut the
# still edge's wobble by a fifth, four by about a half; further changes little.
REACH_S = 0.1
STILL = 8.0


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    u = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def main_only(m: np.ndarray, head: Optional[tuple[float, float, float, float]]) -> np.ndarray:
    """The small mask `m` with only the speaker kept: the part under the head box (x0, y0, x1, y1
    in the mask's own pixels), else the largest part, and any part at least a sixth of its size
    (an arm the frame's edge has cut off, say). A smaller patch apart from them goes."""
    import cv2

    hard = (m >= 0.5).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hard, connectivity=8)
    if n <= 2:
        return m
    keep: set[int] = set()
    if head is not None:
        x0, y0, x1, y1 = (int(round(v)) for v in head)
        x0, y0 = max(0, x0), max(0, y0)
        sub = lab[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)]
        ids, counts = np.unique(sub[sub > 0], return_counts=True)
        if len(ids):
            keep.add(int(ids[np.argmax(counts)]))
    if not keep:
        keep.add(int(1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])))
    main = max(int(stats[i, cv2.CC_STAT_AREA]) for i in keep)
    keep |= {i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= main / 6}
    if len(keep) == n - 1:
        return m
    sel = np.isin(lab, sorted(keep)).astype(np.float32)
    sel = cv2.dilate(sel, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    sel = cv2.GaussianBlur(sel, (0, 0), 1.0)
    return (m * np.clip(sel * 1.5, 0.0, 1.0)).astype(np.float32)


class Steady:
    """The clip's small person masks, each averaged with its neighbours either side wherever the
    picture did not change between them. Read in order (`get(k)` with k rising one at a time), it
    keeps a short window; asked for any other frame it reads that frame's window afresh, so a still
    and the same frame of the render are the same.

    Given several masks of the same clip (from different models, the same size), a pixel is the
    person only as far as every one of them says so. The two models the core has fail in different
    places: on the test clips the plain one took a vase by the hair (A), a ceiling sprinkler (D) and
    a train's window panel (F) for the person, the detailed one did not, and what both agree on
    kept every hand, microphone and strand of hair the plain one had."""

    def __init__(self, masks, info: VideoInfo, reach: Optional[int] = None, still: float = STILL):
        self.all = list(masks) if isinstance(masks, (list, tuple)) else [masks]
        self.masks, self.info = self.all[0], info
        if any((m.width, m.height) != (self.masks.width, self.masks.height) for m in self.all):
            raise ValueError("masks to combine must be the same size")
        self.reach = int(reach) if reach is not None else int(np.clip(round(REACH_S * info.rate), 2, 6))
        self.still = float(still)
        self.n = self.masks.n
        d = np.arange(-self.reach, self.reach + 1, dtype=np.float32)
        self.time_w = np.exp(-0.5 * (d / max(0.5, self.reach / 1.5)) ** 2)
        self._win: deque = deque()  # (k, mask, grey) for consecutive frames
        self._next = 0  # the next frame the readers give
        self._mreaders: list = []
        self._freader = None

    # ---- reading ----
    def _open(self, start: int) -> None:
        self._close()
        start = int(np.clip(start, 0, self.n - 1))
        t = max(0.0, (start - 0.25) / self.info.rate)
        size = (self.masks.width, self.masks.height)
        self._mreaders = [iter(_read_gray(m.path, size[0], size[1], start=t)) for m in self.all]
        self._freader = iter(FrameReader(self.info, size=size, start=t, gray=True))
        self._next = start
        self._win.clear()

    def _close(self) -> None:
        for r in self._mreaders + [self._freader]:
            if r is not None and hasattr(r, "close"):
                r.close()
        self._mreaders, self._freader = [], None

    def _pull(self) -> None:
        """Read the next frame into the window (holding the last one if the clip runs short)."""
        import cv2

        ms = [next(r, None) for r in self._mreaders]
        g = next(self._freader, None)
        m = None if any(x is None for x in ms) else (ms[0] if len(ms) == 1 else np.minimum.reduce(ms))
        if m is None or g is None:
            if self._win:
                _, m0, g0 = self._win[-1]
                self._win.append((self._next, m0, g0))
            else:
                z = np.zeros((self.masks.height, self.masks.width), np.float32)
                self._win.append((self._next, z, z))
        else:  # the grey picture is softened once, here: only its slow changes count as movement
            self._win.append((self._next, m.astype(np.float32) / 255.0,
                              cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.5)))
        self._next += 1

    def get(self, k: int) -> np.ndarray:
        """Frame k's steadied mask (float32 0..1, the masks' small size)."""
        k = int(np.clip(k, 0, self.n - 1))
        lo, hi = max(0, k - self.reach), min(self.n - 1, k + self.reach)
        first = self._win[0][0] if self._win else None
        if (not self._mreaders or (first is not None and lo < first)
                or (first is None and self._next != lo) or lo > self._next + 4 * self.reach + 8):
            self._open(lo)  # back before the window, or far ahead: start reading there
        while self._win and self._win[0][0] < lo:
            self._win.popleft()
        while self._next <= hi:
            self._pull()
        frames = {w[0]: w for w in self._win}
        _, mk, gk = frames[k]
        if self.reach <= 0:
            return mk
        num = mk * self.time_w[self.reach]
        den = np.full_like(mk, self.time_w[self.reach])
        for i, d in enumerate(range(-self.reach, self.reach + 1)):
            j = k + d
            if d == 0 or j not in frames:  # the clip's first and last frames have fewer neighbours
                continue
            _, mj, gj = frames[j]
            w = self.time_w[i] * np.exp(-((np.abs(gj - gk) / self.still) ** 2))
            num += w * mj
            den += w
        return (num / den).astype(np.float32)

    def close(self) -> None:
        self._close()


def alpha(masks: PersonMasks, m: np.ndarray, size: tuple[int, int], guide: Optional[np.ndarray] = None,
          crop: Optional[tuple[float, float, float, float]] = None, reach: Optional[float] = None,
          low: float = LOW, high: float = HIGH) -> np.ndarray:
    """The person at `size` (w, h), float32 0..1: the guided edge (`PersonMasks.full`, the soft edge
    kept within `reach` output pixels of where the model is sure), firmed between `low` and `high`."""
    a = masks.full(m, size, guide=guide, crop=crop, lean=0.5, reach=reach)
    lut = _firm_table(low, high)
    return lut[np.clip(a * 1023.0 + 0.5, 0, 1023).astype(np.int16)]



_TABLES: dict = {}


def _firm_table(low: float, high: float) -> np.ndarray:
    key = (round(low, 4), round(high, 4))
    if key not in _TABLES:
        _TABLES[key] = _smoothstep(low, high, np.linspace(0.0, 1.0, 1024, dtype=np.float32)).astype(np.float32)
    return _TABLES[key]


def _norm_blur(img: np.ndarray, w: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    num = cv2.GaussianBlur(img * w[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    return num / np.maximum(den, 1e-4)[..., None], den


def colours_near_edge(frame: np.ndarray, a: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """At a quarter of the frame's size: under each spot, the old wall's colour from the nearby
    pixels that are all wall, and the person's colour from the nearby pixels that are all person
    (float32). Both change slowly across the frame, so a quarter size loses nothing."""
    import cv2

    h, w = a.shape
    sw, sh = max(2, w // 4), max(2, h // 4)
    a_s = cv2.resize(a, (sw, sh), interpolation=cv2.INTER_AREA)
    I_s = cv2.resize(np.ascontiguousarray(frame, np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    s = max(1.0, sigma / 4)
    wall, dw = _norm_blur(I_s, (a_s < 0.04).astype(np.float32), s)
    body, db = _norm_blur(I_s, (a_s > 0.96).astype(np.float32), s)
    wall = np.where((dw > 0.02)[..., None], wall, I_s)
    body = np.where((db > 0.02)[..., None], body, I_s)
    return wall, body


def sample(img: np.ndarray, ys: np.ndarray, xs: np.ndarray, sy: float, sx: float) -> np.ndarray:
    """`img` (a smaller map: `sy`, `sx` times the frame's size) read at the frame's pixels (ys, xs),
    blended between its four nearest points."""
    h, w = img.shape[:2]
    fy = np.clip((ys + 0.5) * sy - 0.5, 0, max(0.0, h - 1.001))
    fx = np.clip((xs + 0.5) * sx - 0.5, 0, max(0.0, w - 1.001))
    y0, x0 = fy.astype(np.int32), fx.astype(np.int32)
    wy, wx = (fy - y0).astype(np.float32), (fx - x0).astype(np.float32)
    if img.ndim == 3:
        wy, wx = wy[:, None], wx[:, None]
    top = img[y0, x0] * (1 - wx) + img[y0, x0 + 1] * wx
    bot = img[y0 + 1, x0] * (1 - wx) + img[y0 + 1, x0 + 1] * wx
    return top * (1 - wy) + bot * wy


def over(frame: np.ndarray, a: np.ndarray, room: np.ndarray, sigma: Optional[float] = None,
         gains: Optional[np.ndarray] = None) -> np.ndarray:
    """The person (from `frame`, RGB, times `gains` per channel when given) over `room` (float32
    0..255, the same size), with `a` the person (0..1). Returns float32 0..255. On the edge only the
    old wall's share of each pixel is swapped for the room; a pixel that is mostly wall takes the
    person's colour from just inside. Only the edge's own pixels are worked out one by one."""
    import cv2

    h, w = a.shape
    sigma = sigma if sigma is not None else max(4.0, 0.012 * w)
    I = frame.astype(np.float32)
    if gains is not None:
        I *= np.asarray(gains, np.float32)
    out = cv2.blendLinear(I, room, a, 1.0 - a)
    edge = (a > 0.004) & (a < 0.996)
    ys, xs = np.nonzero(edge)
    if not len(ys):
        return out
    pad = int(3 * sigma)
    y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad + 1)
    x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad + 1)
    y0, x0 = y0 - y0 % 4, x0 - x0 % 4
    wall, body = colours_near_edge(I[y0:y1, x0:x1], a[y0:y1, x0:x1], sigma)
    ly, lx = ys - y0, xs - x0
    sy, sx = wall.shape[0] / max(1, y1 - y0), wall.shape[1] / max(1, x1 - x0)
    wall_e = sample(wall, ly, lx, sy, sx)
    body_e = sample(body, ly, lx, sy, sx)
    a_e = a[ys, xs][:, None]
    I_e = I[ys, xs]
    own = np.clip((I_e - (1.0 - a_e) * wall_e) / np.maximum(a_e, 1e-3), 0.0, 255.0)
    lean = _smoothstep(0.08, 0.45, a_e)
    fg = own * lean + body_e * (1.0 - lean)
    out[ys, xs] = fg * a_e + room[ys, xs] * (1.0 - a_e)
    return out


class Cutout:
    """The speaker cut out for a frame of a render, the way `backdrop` cuts them out for a whole new
    room: the steadied mask (both models agreeing when `careful`), only the speaker kept (the part
    under the head), then the guided edge firmed. Any effect that lays something behind the person
    over much of the frame uses this, so its edge is measured and made the same way.

        cut = matte.Cutout(clip, framing, careful=True)
        a = cut.alpha(frame, k)              # frame: the output frame; k: the source frame
        shown, worst = matte.face_shown(a, face_box_in_output)
    """

    def __init__(self, clip, framing, careful: bool = True):
        self.clip, self.framing = clip, framing
        self.W, self.H = framing.out_w, framing.out_h
        self.masks = clip.person()
        self.steady = Steady([self.masks, clip.person("person-parts")] if careful else self.masks, clip.info)
        self._sx = self.masks.width / clip.info.width

    def alpha(self, frame: np.ndarray, k: int) -> np.ndarray:
        """The person at the output's size (float32 0..1) for source frame `k`, on `frame`."""
        m = self.steady.get(k)
        head = self.clip.face.head(k)
        if head is not None:
            s = self._sx
            m = main_only(m, (head.x0 * s, head.y0 * s, head.x1 * s, head.y1 * s))
        crop = self.framing.window(k) if self.framing.reframed else None
        return alpha(self.masks, m, (self.W, self.H), guide=frame, crop=crop, reach=0.012 * self.W)


def face_shown(a: np.ndarray, box) -> Optional[float]:
    """How much of the face (its box in output pixels, a little inside it) the cut-out misses: the
    share of those pixels under half sure, where whatever is behind the person would show through
    the face. Over 2% counts as a frame over the face. None when the box is too small to judge."""
    H, W = a.shape
    x0, x1 = int(max(0, box.x0 + 0.15 * box.w)), int(min(W, box.x1 - 0.15 * box.w))
    y0, y1 = int(max(0, box.y0 + 0.1 * box.h)), int(min(H, box.y1 - 0.1 * box.h))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return float((a[y0:y1, x0:x1] < 0.5).mean())


def over_where(frame: np.ndarray, a: np.ndarray, room: np.ndarray, where: np.ndarray,
               sigma: Optional[float] = None) -> np.ndarray:
    """The person (alpha `a`) over `room` (float32 0..255: the clip's own picture with things laid
    in it behind the person), worked out only where `where` (0..1, the share of each pixel that
    something was laid on) says the room changed, and a little round it. Everywhere else the frame
    comes back exactly as filmed, so an effect laid behind part of the picture leaves the rest of it
    untouched; where something was laid, the edge has no rim of the old wall (`over`). uint8."""
    import cv2

    h, w = a.shape
    sigma = sigma if sigma is not None else max(4.0, 0.012 * w)
    ys, xs = np.nonzero(where > 0.002)
    if not len(ys):
        return frame
    pad = int(4 * sigma)
    y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad + 1)
    x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad + 1)
    sub = (slice(y0, y1), slice(x0, x1))
    done = over(frame[sub], a[sub], room[sub], sigma=sigma)
    # the edge is worked out against the room only near where the room changed: further off, the
    # frame as filmed (a soft step, so no seam shows where the two meet)
    near = cv2.GaussianBlur((where[sub] > 0.002).astype(np.float32), (0, 0), sigma)
    near = np.clip(near * 3.0, 0.0, 1.0)[..., None]
    out = frame.copy()
    f = frame[sub].astype(np.float32)
    out[sub] = np.clip(f + near * (done - f) + 0.5, 0, 255).astype(np.uint8)
    return out
