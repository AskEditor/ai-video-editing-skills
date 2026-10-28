"""The output's shape and size, and a window into the clip that keeps the face in frame.

- The output is 1080 wide at most (1080×1920 for 9:16, 1080×1350 for 4:5, 1080×1080 square,
  1920×1080 for 16:9), and the clip's own pixels when that is smaller: a picture is never enlarged
  unless `size` asks for it.
- A clip wider than the shape asked for keeps its full height, and the window slides across. A
  clip taller than it keeps its full width, and the window slides up and down, keeping the top of
  the head inside with a tenth of the frame to spare above it (a crown a few pixels from the edge
  reads as cut).
- The window moves like a calm camera operator (`follow`): it holds still while the face stays
  near where it is (a dead zone, so a nod or a small lean moves nothing), and when the face settles
  somewhere new it eases there, starting and stopping gently, a little before the head arrives
  rather than after. Whatever the moves, the face never leaves the window: where a quick lean
  would take it out, the window is pushed along with it. Where the clip itself cuts, so does the
  window, instead of gliding across the cut.
- When the face is found in too few frames to follow, the window stays centred and says so.
- `fill` keeps the whole picture: it is fitted inside the shape, over a blurred, darkened copy of
  itself that fills the rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .face import FaceTrack
from .layout import Rect

PORTRAIT = (1080, 1920)
SHAPES = {"9:16": 9 / 16, "4:5": 4 / 5, "1:1": 1.0, "16:9": 16 / 9}
CROWN_ROOM = 0.10  # the room kept above the top of the head, as a share of the frame's height
MIN_FOUND = 0.5  # the face must be found in at least this share of frames to be followed

# How the window follows (shares of the window's own size, and seconds).
DEAD_ZONE = 0.10  # the face may wander this far from where the window sits before it moves
PATIENCE = 0.40  # a wander shorter than this (a gesture, a quick look away) is ignored
SETTLE = 1.00  # where the face settles: the middle of where it is over this long,
AHEAD = 0.25  # starting this long after it left the dead zone
MOVE_MIN, MOVE_MAX = 0.40, 1.20  # a move lasts this long at least and at most
MOVE_SPEED = 0.9  # windows a second, before the limits above
LEAD = 0.35  # a move starts this share of its length before the face leaves the dead zone
MERGE = 1.5  # a glide the same way within this long of the last one becomes one longer glide
EDGE = 0.04  # the face is kept at least this far inside the window's edges
CUT_JUMP = 0.35  # the face jumping this far in a moment, and staying, is a cut in the clip
CUT_HOLD = 0.5  # seconds the face must stay put on each side of a jump for it to be a cut


def _even(v: float) -> int:
    return max(2, int(round(v / 2)) * 2)


def _smooth(series: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or len(series) < 3:
        return series.copy()
    radius = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(series, (radius, radius), mode="edge"), k, mode="valid")


def ease(u):
    """0 to 1 with no jolt at either end: zero speed and zero acceleration where it starts and stops."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * u * (u * (u * 6 - 15) + 10)


@dataclass
class Move:
    start: int  # the first frame of the move
    end: int  # the frame it arrives on
    a: float
    b: float
    cut: bool = False  # the clip itself cuts here, so the window cuts instead of gliding

    def as_dict(self, fps: float) -> dict:
        return {"from_s": round(self.start / fps, 3), "to_s": round(self.end / fps, 3), "from": round(self.a, 1),
                "to": round(self.b, 1), "kind": "cut" if self.cut else "glide"}


@dataclass
class Path:
    """Where the window's centre sits on one axis, frame by frame, and why."""

    pos: np.ndarray  # the window's centre (source pixels)
    target: np.ndarray  # where the face would put it
    lo: np.ndarray  # the range the centre must stay inside for the face to stay in the window
    hi: np.ndarray
    moves: list[Move] = field(default_factory=list)
    pushed: int = 0  # frames where the face would have left the window, so the window was pushed

    def stats(self, fps: float, size: float) -> dict:
        """How calm it is: the share of frames the window is still, its fastest move, and how many
        times it turns back while moving (jitter shows as turns)."""
        v = np.diff(self.pos) if len(self.pos) > 1 else np.zeros(1)
        moving = np.abs(v) > 0.05
        both = moving[1:] & moving[:-1]
        turns = int(np.sum(np.sign(v[1:][both]) != np.sign(v[:-1][both])))
        cut_frames = {m.start for m in self.moves if m.cut}
        speed = np.abs(v).copy()
        for c in cut_frames:  # a cut is not a speed
            if 0 < c <= len(speed):
                speed[c - 1] = 0.0
        return {"still_share": round(1.0 - float(moving.mean()), 3),
                "glides": sum(1 for m in self.moves if not m.cut), "cuts": len(cut_frames),
                "fastest_per_s": round(float(speed.max() * fps / size), 3) if len(speed) else 0.0,
                "turns_while_moving": turns, "pushed_frames": self.pushed}


def follow(target: np.ndarray, lo: np.ndarray, hi: np.ndarray, size: float, fps: float,
           cuts: Sequence[int] = ()) -> Path:
    """A calm path for the window's centre on one axis.

    `target` is where the face would put the centre in each frame; `lo` and `hi` bound where the
    centre may be for the face to stay inside the window (and the window inside the clip); `size`
    is the window's length on this axis. The window holds still while the target stays within the
    dead zone of it; when the target leaves it for longer than `PATIENCE`, the window eases to where
    the target settles. A frame listed in `cuts` (a cut in the clip itself) makes the window cut
    there too. Last, the face is kept inside: a frame where the path would leave it out is pushed
    in, and the push is spread over the frames around it, so it reads as a move and not a jolt.
    """
    n = len(target)
    target = np.asarray(target, np.float64)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    dz = DEAD_ZONE * size
    patience = max(1, int(round(PATIENCE * fps)))
    settle = max(1, int(round(SETTLE * fps)))
    ahead = int(round(AHEAD * fps))
    merge = int(round(MERGE * fps))
    cut_set = sorted(set(int(c) for c in cuts if 0 < c < n))

    def settle_at(i: int, until: int) -> float:
        # Where the face ends up, not where it is half-way: the middle of a stretch that starts a
        # moment after it left.
        a0 = min(until - 1, i + ahead)
        j = max(a0 + 1, min(until, a0 + settle))
        v = float(np.median(target[a0:j]))
        a, b = float(np.max(lo[a0:j])), float(np.min(hi[a0:j]))
        return float(np.clip(v, a, b)) if a <= b else v

    def ok(i: int, p: float) -> bool:
        return abs(target[i] - p) <= dz and lo[i] - 0.5 <= p <= hi[i] + 0.5

    pos = np.empty(n)
    moves: list[Move] = []
    bounds = [0] + cut_set + [n]
    for s0, s1 in zip(bounds, bounds[1:]):  # each stretch between the clip's own cuts on its own
        p = settle_at(s0, s1)
        if s0 > 0:
            moves.append(Move(s0, s0, float(pos[s0 - 1]), p, cut=True))
        decided = s0  # frames before this are fixed (a glide already written)
        i = s0
        while i < s1:
            if ok(i, p):
                pos[i] = p
                i += 1
                continue
            # The face has left the dead zone. A wander that comes back within the patience is
            # ignored, unless the window would lose the face meanwhile.
            limit = min(s1, i + patience)
            j = i
            while j < limit and not ok(j, p):
                j += 1
            lost = not (lo[i] - 0.5 <= p <= hi[i] + 0.5)
            if j < limit and not lost:
                pos[i:j] = p
                i = j
                continue
            new = settle_at(i, s1)
            dist = abs(new - p)
            if dist < 1.0:
                pos[i] = p
                i += 1
                continue
            last = moves[-1] if moves else None
            if (last is not None and not last.cut and last.start >= s0 and (new - p) * (last.b - last.a) > 0
                    and i - last.end <= merge):
                # The face moved on the same way soon after the last glide: make the two one glide.
                moves.pop()
                a, start = last.a, last.start
                dist = abs(new - a)
            else:
                a, start = None, None
            seconds = float(np.clip(MOVE_MIN + dist / (MOVE_SPEED * size), MOVE_MIN, MOVE_MAX))
            length = max(2, int(round(fps * seconds)))
            merged = start is not None
            if start is None:
                start = max(decided, i - int(round(LEAD * length)))
                a = float(pos[start - 1]) if start > s0 else p
            # The glide must keep the face in all the way: where the face moves faster than it, the
            # glide goes quicker (down to a quarter of a second), so the window never has to be
            # yanked after it.
            quickest = max(2, int(round(0.25 * fps)))
            while True:
                end = min(s1 - 1, max(start + length, i))
                u = (np.arange(start, end + 1) - start) / max(1, end - start)
                glide = a + (new - a) * ease(u)
                lost = (glide < lo[start:end + 1] - 0.5) | (glide > hi[start:end + 1] + 0.5)
                if not lost.any() or length <= quickest:
                    break
                length = max(quickest, int(length * 0.75))
                if not merged:
                    start = max(decided, i - int(round(LEAD * length)))
                    a = float(pos[start - 1]) if start > s0 else p
            pos[start:end + 1] = glide
            moves.append(Move(start, end, a, new))
            p = new
            decided = end + 1
            i = end + 1
    # Keep the face in: whatever a glide could not hold (a face moving faster than a quarter-second
    # glide) is pushed inside [lo, hi] frame by frame. The bounds follow the smoothed face, so the
    # push moves with it, without the turn-backs a smoothed correction makes.
    out = np.clip(pos, lo, hi)
    pushed = int(np.sum(np.abs(out - pos) > 0.5))
    return Path(out, target, lo, hi, moves, pushed)


@dataclass
class Framing:
    src_w: int
    src_h: int
    out_w: int
    out_h: int
    windows: Optional[np.ndarray]  # (n, 4) x, y, w, h per source frame; None = the whole frame
    note: str = ""
    fill: Optional[tuple[float, float, float]] = None  # fill mode: (scale, x, y) of the fitted picture
    path: Optional[Path] = None  # how the window moved, for the path sheet and the checks
    axis: str = ""  # "x" or "y": the axis the window moves on
    face_out: float = 0.0  # the share of frames where the face is not wholly inside the window
    head_cut: float = 0.0  # the share of frames where the top of the head is above the window
    # The part of the clip that is picture (x0, y0, x1, y1), when black bars were filmed or
    # exported into it: everything above (windows, the fill) is measured inside this part.
    content: Optional[tuple[int, int, int, int]] = None

    @property
    def reframed(self) -> bool:
        return self.windows is not None or self.fill is not None

    def window(self, i: int) -> tuple[float, float, float, float]:
        if self.windows is None:
            return 0.0, 0.0, float(self.src_w), float(self.src_h)
        x, y, w, h = self.windows[int(np.clip(i, 0, len(self.windows) - 1))]
        return float(x), float(y), float(w), float(h)

    def apply(self, frame: np.ndarray, i: int) -> np.ndarray:
        """Source frame `i` as an output frame."""
        import cv2

        if self.content is not None:
            x0, y0, x1, y1 = self.content
            frame = frame[y0:y1, x0:x1]
        if self.fill is not None:
            return self._fill(frame)
        x, y, w, h = self.window(i)
        if self.windows is not None:
            scale = self.out_w / w
            if scale > 1.001:  # enlarged: place the window to the sub-pixel, or a slow glide steps
                m = np.array([[scale, 0.0, -x * scale], [0.0, scale, -y * scale]], np.float64)
                return cv2.warpAffine(frame, m, (self.out_w, self.out_h), flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REPLICATE)
            x0, y0 = int(round(x)), int(round(y))
            frame = frame[y0:y0 + int(round(h)), x0:x0 + int(round(w))]
        fh, fw = frame.shape[:2]
        if (fw, fh) == (self.out_w, self.out_h):
            return np.ascontiguousarray(frame)
        interp = cv2.INTER_AREA if fw > self.out_w else cv2.INTER_CUBIC
        return cv2.resize(frame, (self.out_w, self.out_h), interpolation=interp)

    def _fill(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        s, fx, fy = self.fill
        W, H = self.out_w, self.out_h
        # The ground: the picture scaled to cover, blurred at an eighth of the size (fast), and
        # darkened, so the fitted picture stays what the eye goes to.
        cover = max(W / self.src_w, H / self.src_h)
        cw, ch = int(np.ceil(self.src_w * cover)), int(np.ceil(self.src_h * cover))
        small = cv2.resize(frame, (max(8, cw // 8), max(8, ch // 8)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 5)
        ground = cv2.resize(small, (cw, ch), interpolation=cv2.INTER_LINEAR)
        gx, gy = (cw - W) // 2, (ch - H) // 2
        out = (ground[gy:gy + H, gx:gx + W].astype(np.float32) * 0.55).astype(np.uint8)
        fw, fh = int(round(self.src_w * s)), int(round(self.src_h * s))
        pic = cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        x0, y0 = int(round(fx)), int(round(fy))
        out[y0:y0 + fh, x0:x0 + fw] = pic[:H - y0, :W - x0]
        return np.ascontiguousarray(out)

    def to_output(self, r: Rect, i: int) -> Rect:
        """A box in source frame `i`'s pixels, in output pixels."""
        if self.content is not None:
            r = r.shift(-self.content[0], -self.content[1])
        if self.fill is not None:
            s, fx, fy = self.fill
            return r.scale(s, s, fx, fy)
        x, y, w, h = self.window(i)
        sx, sy = self.out_w / w, self.out_h / h
        return Rect((r.x0 - x) * sx, (r.y0 - y) * sy, (r.x1 - x) * sx, (r.y1 - y) * sy)


def output_size(w: float, h: float, size: Optional[tuple[int, int]] = None) -> tuple[int, int]:
    """The output size for a picture of w×h: `size` when given, else no larger than 1080×1920
    (1920×1080 on its side), keeping the shape."""
    if size:
        return size
    long_side, short_side = (PORTRAIT[1], PORTRAIT[0])
    if h >= w:
        k = min(1.0, long_side / h, short_side / w)
    else:
        k = min(1.0, long_side / w, short_side / h)
    return _even(w * k), _even(h * k)


def shape_size(aspect: float, size: Optional[tuple[int, int]] = None) -> tuple[int, int]:
    """A shape's full size: 1080 wide when it is upright or square, 1920 wide on its side."""
    if size:
        return size
    if aspect > 1.0 + 1e-6:
        return 1920, _even(1920 / aspect)
    return 1080, _even(1080 / aspect)


def find_cuts(face: FaceTrack, size: float, axis: int) -> list[int]:
    """Frames where the clip itself cuts: the face jumps more than `CUT_JUMP` of the window across
    the frame between two detections a moment apart, and stays near each place for `CUT_HOLD`
    seconds on either side (a blip of a false face does not)."""
    raw = face.raw
    good = ~np.isnan(raw[:, 0]) & (raw[:, 4] >= 0.6)
    found = np.flatnonzero(good)
    if len(found) < 4:
        return []
    c = raw[:, axis] + raw[:, axis + 2] / 2
    hold = max(2, int(round(CUT_HOLD * face.fps)))
    out = []
    for k in range(1, len(found)):
        a, b = found[k - 1], found[k]
        if b - a > max(2, int(0.1 * face.fps)) or abs(c[b] - c[a]) <= CUT_JUMP * size:
            continue
        before = found[(found <= a) & (found > a - hold)]
        after = found[(found >= b) & (found < b + hold)]
        if len(before) < hold // 2 or len(after) < hold // 2:
            continue
        if (np.abs(c[before] - c[a]).max() < CUT_JUMP * size / 2
                and np.abs(c[after] - c[b]).max() < CUT_JUMP * size / 2):
            out.append(int(b))
    return out


def faces_with_cuts(face: FaceTrack, n: int, info_w: int, info_h: int, aspect: float,
                    cuts: Optional[Sequence[int]] = None) -> tuple[list[Optional[Rect]], list[int]]:
    """The face's box in every frame, and the frames where the clip itself cuts (found when not
    given). Each stretch between cuts is followed from its own detections, so the face is never
    glided from one shot into the next."""
    wide = info_w / info_h > aspect
    size_axis = _even(min(info_w, info_h * aspect)) if wide else _even(min(info_h, info_w / aspect))
    cuts = list(cuts) if cuts is not None else find_cuts(face, size_axis, 0 if wide else 1)
    faces = [face.box(i) for i in range(n)]
    if cuts:
        segments = [0] + cuts + [n]
        for s0, s1 in zip(segments, segments[1:]):
            boxes = face.boxes_between(s0, s1)
            for k in range(s1 - s0):
                x, y, w, h = boxes[k]
                faces[s0 + k] = Rect(x, y, x + w, y + h) if w > 0 else None
    return faces, cuts


def plan(info_w: int, info_h: int, fps: float, n: int, face: Optional[FaceTrack], shape: str = "9:16",
         size: Optional[tuple[int, int]] = None, mode: str = "track",
         faces: Optional[Sequence[Optional[Rect]]] = None, cuts: Optional[Sequence[int]] = None) -> Framing:
    """The framing for a clip of info_w×info_h.

    `shape` is "9:16", "4:5", "1:1" or "16:9" (reframed when the clip is another shape), or "keep"
    (the clip's own shape). `mode` is "track" (the window follows the face), "center" (it stays
    put in the middle) or "fill" (the whole picture, fitted over a blurred copy of itself).
    `faces` (a face box per frame, or None where there is none) replaces the face track, for a
    caller that decides whom to follow (the speaker, when there are two people); `cuts` lists the
    frames where the window cuts instead of gliding (found in the face track when not given).
    """
    if shape == "keep" or shape not in SHAPES:
        ow, oh = output_size(info_w, info_h, size)
        return Framing(info_w, info_h, ow, oh, None, "the clip's own shape")
    aspect = SHAPES[shape]
    if abs(info_w / info_h - aspect) < 0.01 * aspect:
        ow, oh = output_size(info_w, info_h, size)
        return Framing(info_w, info_h, ow, oh, None, f"already {shape}")
    if mode == "fill":
        ow, oh = shape_size(aspect, size)
        s = min(ow / info_w, oh / info_h)
        fw, fh = info_w * s, info_h * s
        return Framing(info_w, info_h, ow, oh, None, f"fitted inside {shape} over a blurred copy of itself",
                       fill=(s, (ow - fw) / 2, (oh - fh) / 2))

    if faces is None and face is not None and face.any() and face.found_share >= MIN_FOUND:
        faces, found = faces_with_cuts(face, n, info_w, info_h, aspect, cuts)
        cuts = found if cuts is None else cuts
    following = mode == "track" and faces is not None and any(f is not None for f in faces)
    cuts = list(cuts or [])

    windows = np.zeros((n, 4), np.float64)
    path = None
    rule = ""
    if info_w / info_h > aspect:  # wider: full height, slide across
        axis = "x"
        h = float(info_h)
        w = float(_even(min(info_w, info_h * aspect)))
        half = w / 2
        if following:
            boxes = _held(faces, n)
            target = np.clip(np.array([b.cx for b in boxes]), half, info_w - half)
            # The centres that keep the face inside, never narrower than the dead zone (so the
            # window can still hold still when a big face nearly fills it).
            lo, hi = _band([b.x1 + EDGE * w - half for b in boxes], [b.x0 - EDGE * w + half for b in boxes],
                           DEAD_ZONE * w, half, info_w - half)
            path = follow(target, lo, hi, w, fps, cuts)
            cx = path.pos
        elif mode == "center":
            cx = np.full(n, info_w / 2.0)
        else:
            cx = np.full(n, info_w / 2.0)
        x = np.clip(cx - half, 0, info_w - w)
        windows[:] = np.stack([x, np.zeros(n), np.full(n, w), np.full(n, h)], axis=1)
    else:  # taller: full width, keep the crown in with room above it
        axis = "y"
        w = float(info_w)
        h = float(_even(min(info_h, info_w / aspect)))
        half = h / 2
        if following:
            boxes = _held(faces, n)
            heads = [FaceTrack._grow(b) for b in boxes]
            room = CROWN_ROOM * h
            # One rule for the whole clip, so the frame never hops between two compositions: the
            # whole head with room above it when it fits in most frames; else brows to chin, the
            # brows a little under the top, and the top of the hair let go (the note says so).
            fits = np.mean([(b.y1 + EDGE * h) - (hd.y0 - 0.5 * room) <= h for b, hd in zip(boxes, heads)])
            rule = "head" if fits >= 0.7 else "face"
            if rule == "head":
                target = np.array([hd.y0 - room + half for hd in heads])
                tops = [min(b.y0 - EDGE * h, hd.y0 - 0.5 * room) + half for b, hd in zip(boxes, heads)]
            else:
                target = np.array([b.y0 - 0.12 * h + half for b in boxes])
                tops = [b.y0 - EDGE * h + half for b in boxes]
            lo, hi = _band([b.y1 + EDGE * h - half for b in boxes], tops, DEAD_ZONE * h, half, info_h - half)
            target = np.clip(target, half, info_h - half)
            path = follow(target, lo, hi, h, fps, cuts)
            cy = path.pos
        elif mode == "center":
            cy = np.full(n, info_h / 2.0)
        else:  # no face to follow: the upper part, where a head usually is
            cy = np.full(n, (info_h - h) * 0.3 + half)
        y = np.clip(cy - half, 0, info_h - h)
        windows[:] = np.stack([np.zeros(n), y, np.full(n, w), np.full(n, h)], axis=1)
    ow, oh = output_size(w, h, size)
    fr = Framing(info_w, info_h, ow, oh, windows, "", path=path, axis=axis)
    if following:
        fr.face_out, fr.head_cut = _measure(fr, _held(faces, n))
        note = f"reframed to {shape}, following the face"
        if rule == "face":
            note += (f"; the head is taller than a {shape} frame, so the face is kept and the top of the "
                     f"hair is cut")
    elif mode == "center":
        note = f"reframed to {shape}, centred (as asked)"
    elif face is not None and face.any():
        note = f"reframed to {shape}, centred: the face was found in only {face.found_share:.0%} of frames"
    else:
        note = f"reframed to {shape}, centred: no face was found"
    fr.note = note
    return fr


def _band(lo: Sequence[float], hi: Sequence[float], least: float, floor: float, ceil: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame bounds for the window's centre, each at least `least` wide (widened evenly about its
    middle where the face leaves less room than that) and inside [floor, ceil]."""
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    short = hi - lo < least
    mid = (lo + hi) / 2
    lo = np.where(short, mid - least / 2, lo)
    hi = np.where(short, mid + least / 2, hi)
    lo, hi = np.clip(lo, floor, ceil), np.clip(hi, floor, ceil)
    return np.minimum(lo, hi), np.maximum(lo, hi)


def _measure(fr: "Framing", boxes: Sequence[Rect]) -> tuple[float, float]:
    """The share of frames where the face box is not wholly inside the window, and where the top of
    the head (the face box grown to take in the hair) is above the window's top."""
    out = cut = 0
    for i, b in enumerate(boxes):
        x, y, w, h = fr.window(i)
        if b.x0 < x - 0.5 or b.x1 > x + w + 0.5 or b.y0 < y - 0.5 or b.y1 > y + h + 0.5:
            out += 1
        if fr.axis == "y" and FaceTrack._grow(b).y0 < y - 0.5:
            cut += 1
    n = max(1, len(boxes))
    return round(out / n, 3), round(cut / n, 3)


def _held(faces: Sequence[Optional[Rect]], n: int) -> list[Rect]:
    """A box for every frame: where there is none, the nearest one before (or after, at the start)."""
    out: list[Optional[Rect]] = list(faces[:n]) + [None] * max(0, n - len(faces))
    last = next((f for f in out if f is not None), None)
    for i in range(n):
        if out[i] is None:
            out[i] = last
        else:
            last = out[i]
    return out  # type: ignore[return-value]
