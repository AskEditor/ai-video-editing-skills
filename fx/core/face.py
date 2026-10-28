"""A face box for every frame, smoothed, cached next to the clip as `<clip>.face.json`.

The detector's box is tight: brows to mouth, cheek to cheek. Text has to keep clear of the whole
head, so `head()` grows it to take in the hair or a cap above, the chin and beard below, and the
ears at the sides. Effects place text against `heads(first, last)`: the union of the head over
every frame the text is on screen, from the raw detections and the smoothed track together, so a
quick movement never brings the face under the words.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import paths
from .layout import Rect
from .video import FrameReader, VideoInfo

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

CACHE_VERSION = 1
DETECT_WIDTH = 480
# How far the head reaches past the detector's box, as fractions of the box's own size. The box
# runs from the brows to about the chin; drawn on clips A, B and C (28 October 2026) the hair, a cap
# and braids reach up to 0.6 of its height above it, and a beard about 0.1 below it.
HEAD = {"left": 0.20, "top": 0.60, "right": 0.20, "bottom": 0.16}


@contextlib.contextmanager
def quiet_native():
    """Silence the C++ start-up chatter of the vision library (it writes straight to stderr)."""
    try:
        saved = os.dup(2)
    except OSError:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


@dataclass
class FaceTrack:
    width: int
    height: int
    fps: float
    raw: np.ndarray  # (n, 5): x, y, w, h, score, as detected; NaN where no face was found
    boxes: np.ndarray  # (n, 4): x, y, w, h, smoothed and filled across gaps
    seconds: float = 0.0  # how long finding the faces took (0 when read from the cache)
    trusted: Optional[np.ndarray] = None  # `raw` without the detections that leap away from the rest

    def __post_init__(self) -> None:
        if self.trusted is None:
            self.trusted = trust(self.raw, self.fps)

    @property
    def n(self) -> int:
        return len(self.boxes)

    @property
    def found(self) -> np.ndarray:
        return ~np.isnan(self.trusted[:, 0])

    @property
    def found_share(self) -> float:
        return float(self.found.mean()) if self.n else 0.0

    def any(self) -> bool:
        return bool(self.found.any())

    def box(self, i: int) -> Optional[Rect]:
        if not self.any():
            return None
        x, y, w, h = self.boxes[int(np.clip(i, 0, self.n - 1))]
        return Rect(x, y, x + w, y + h)

    @staticmethod
    def _grow(r: Rect) -> Rect:
        return r.grow(HEAD["left"] * r.w, HEAD["top"] * r.h, HEAD["right"] * r.w, HEAD["bottom"] * r.h)

    def head(self, i: int) -> Optional[Rect]:
        b = self.box(i)
        if b is None:
            return None
        head = self._grow(b)
        x, y, w, h, _ = self.trusted[int(np.clip(i, 0, self.n - 1))]
        if not np.isnan(x):
            head = head.union(self._grow(Rect(x, y, x + w, y + h)))
        return head

    def heads(self, first: int, last: int) -> Optional[Rect]:
        """The head's box over frames first..last (inclusive): what text on screen then must avoid."""
        first, last = int(np.clip(first, 0, self.n - 1)), int(np.clip(last, 0, self.n - 1))
        return Rect.around(r for i in range(first, last + 1) if (r := self.head(i)) is not None)

    def center(self, i: int) -> Optional[tuple[float, float]]:
        b = self.box(i)
        return (b.cx, b.cy) if b else None

    def boxes_between(self, first: int, last: int) -> np.ndarray:
        """Smoothed boxes (x, y, w, h) for frames first..last-1, found from those frames' own
        detections only: across a cut in the clip, the face is not glided from one shot to the next.
        Rows of zeros where the stretch has no face at all."""
        raw = self.raw[first:last]
        trusted = trust(raw, self.fps)
        return _fill_and_smooth(trusted, self.fps)

    def to_json(self, signature: dict) -> dict:
        rows = [None if np.isnan(r[0]) else [round(float(v), 1) for v in r[:4]] + [round(float(r[4]), 3)]
                for r in self.raw]
        return {"version": CACHE_VERSION, "source": signature, "width": self.width, "height": self.height,
                "fps": self.fps, "frames": self.n, "boxes": rows}


def _smooth(series: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or len(series) < 3:
        return series.copy()
    radius = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    padded = np.pad(series, (radius, radius), mode="edge")
    return np.convolve(padded, k, mode="valid")


def trust(raw: np.ndarray, fps: float, window: float = 0.5, size_jump: float = 0.3, move: float = 0.45,
          min_score: float = 0.6) -> np.ndarray:
    """`raw` with the detections that cannot be the same face set to NaN.

    A finger over the lens or a blur of skin can make the detector report a "face" twice the size,
    somewhere else, for a frame or two (clip C, 9.2 s). Each detection is compared with the median
    of the detections within `window` seconds around it: one whose size differs by more than
    `size_jump`, or whose centre is more than `move` face-widths away, or whose score is under
    `min_score`, is dropped, and the track is bridged across it from the faces either side.
    """
    out = raw.copy()
    found = np.flatnonzero(~np.isnan(raw[:, 0]))
    if len(found) < 3:
        return out
    r = max(1, int(round(window * fps)))
    w = raw[:, 2]
    cx = raw[:, 0] + raw[:, 2] / 2
    cy = raw[:, 1] + raw[:, 3] / 2
    for i in found:
        near = found[(found >= i - r) & (found <= i + r)]
        mw, mx, my = np.median(w[near]), np.median(cx[near]), np.median(cy[near])
        if (raw[i, 4] < min_score or abs(w[i] - mw) > size_jump * mw
                or np.hypot(cx[i] - mx, cy[i] - my) > move * mw):
            out[i] = np.nan
    return out


def _fill_and_smooth(raw: np.ndarray, fps: float, max_gap: float = 1.0, sigma_s: float = 0.08) -> np.ndarray:
    n = len(raw)
    found = ~np.isnan(raw[:, 0])
    boxes = np.zeros((n, 4), np.float64)
    if not found.any():
        return boxes
    idx = np.flatnonzero(found)
    cx = raw[:, 0] + raw[:, 2] / 2
    cy = raw[:, 1] + raw[:, 3] / 2
    series = [cx, cy, raw[:, 2], raw[:, 3]]
    frames = np.arange(n)
    out = []
    for s in series:
        filled = np.interp(frames, idx, s[idx])  # holds the nearest value past either end
        out.append(_smooth(filled, sigma_s * fps))
    cx, cy, w, h = out
    # Across a long gap the box holds still rather than gliding between far-apart detections.
    gap_start = None
    for i in range(n + 1):
        missing = i < n and not found[i]
        if missing and gap_start is None:
            gap_start = i
        elif not missing and gap_start is not None:
            if (i - gap_start) / fps > max_gap:
                before = gap_start - 1 if gap_start > 0 else i
                hold = before if before < n else gap_start
                for v in (cx, cy, w, h):
                    v[gap_start:i] = v[hold]
            gap_start = None
    boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] = cx - w / 2, cy - h / 2, w, h
    return boxes


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0.0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def detect(info: VideoInfo, min_score: float = 0.5) -> np.ndarray:
    """Runs the face model on every frame. Returns (n, 5) rows of x, y, w, h, score in the clip's
    own pixels, NaN where no face was found. The main face is the largest one, followed from frame
    to frame so a passer-by never takes its place."""
    from .models import vision_path

    with quiet_native():
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

    dw = min(DETECT_WIDTH, info.width)
    dw -= dw % 2
    dh = int(round(info.height * dw / info.width / 2)) * 2
    scale = info.width / dw
    rows = []
    previous = None
    with quiet_native():
        options = vision.FaceDetectorOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(vision_path("face"))),
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=min_score,
        )
        with vision.FaceDetector.create_from_options(options) as detector:
            for i, frame in enumerate(FrameReader(info, size=(dw, dh))):
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
                result = detector.detect_for_video(image, int(round(i * 1000 / info.rate)))
                cands = []
                for d in result.detections:
                    b = d.bounding_box
                    cands.append(np.array([b.origin_x * scale, b.origin_y * scale, b.width * scale,
                                           b.height * scale, d.categories[0].score if d.categories else 0.0]))
                choice = None
                if cands:
                    largest = max(cands, key=lambda c: c[2] * c[3])
                    choice = largest
                    if previous is not None:
                        linked = [c for c in cands if _iou(c, previous) > 0.3]
                        if linked and _iou(largest, previous) < 0.1:
                            choice = max(linked, key=lambda c: _iou(c, previous))
                if choice is None:
                    rows.append([np.nan] * 5)
                else:
                    rows.append(list(choice))
                    previous = choice
    raw = np.array(rows, np.float64) if rows else np.zeros((0, 5))
    # The reader can come up a frame short or long of the estimate; line it up with the clip.
    if len(raw) < info.frames:
        pad = np.full((info.frames - len(raw), 5), np.nan)
        raw = np.vstack([raw, pad]) if len(raw) else pad
    return raw[: info.frames]


def track(info: VideoInfo, refresh: bool = False) -> FaceTrack:
    """The clip's face track, from the cache beside the clip or found now."""
    cache = paths.beside(info.path, "face.json")
    signature = paths.source_signature(info.path)
    if not refresh and cache.exists():
        try:
            data = json.loads(cache.read_text())
            if (data.get("version") == CACHE_VERSION and data.get("source", {}).get("bytes") == signature["bytes"]
                    and data.get("frames") == info.frames):
                raw = np.array([r if r else [np.nan] * 5 for r in data["boxes"]], np.float64)
                trusted = trust(raw, info.rate)
                return FaceTrack(info.width, info.height, info.rate, raw, _fill_and_smooth(trusted, info.rate),
                                 trusted=trusted)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    started = time.time()
    raw = detect(info)
    trusted = trust(raw, info.rate)
    t = FaceTrack(info.width, info.height, info.rate, raw, _fill_and_smooth(trusted, info.rate),
                  seconds=time.time() - started, trusted=trusted)
    tmp = Path(str(cache) + ".part")
    tmp.write_text(json.dumps(t.to_json(signature), separators=(",", ":")) + "\n")
    tmp.replace(cache)
    return t
