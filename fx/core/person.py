"""The person in every frame as a soft mask, cached next to the clip as `<clip>.person.mkv`.

What words-behind and every later "behind you" effect stands on:

    masks = person.masks(info)              # found once, then read from the cache
    for i, m in enumerate(masks.frames()):  # one small mask per frame, 0..1, in order
        alpha = masks.full(m, (w, h), guide=frame)   # at the output size, edges following the frame
        out = compose.behind_person(frame, layer, alpha)

**The range is the whole range.** A mask that never quite reaches 1 inside the person lets
whatever is drawn "behind" show through them: a mask stored as 16..235 instead of 0..255 once put a
faint copy of a word over a speaker's face in AskEditor. So the model's confidence is stretched and
clamped before it is stored (inside the person is exactly 1, the background exactly 0), the cache is
written lossless in a grey format no video player re-ranges, and the first frame is read back and
compared before the cache is trusted.

The masks are kept small (`WORK_WIDTH` across) and cheap to store; `full()` brings one up to the
output size with a guided filter, so the edge follows the hair and the shoulders in the real frame
rather than the blocky small mask.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from . import paths
from .face import quiet_native
from .video import FrameReader, MediaError, VideoInfo, need_ffmpeg

CACHE_VERSION = 1
WORK_WIDTH = 360
MODEL = "person"  # the plain person model: reaches a clean 0..1 and is several times faster
# The model's confidence is stretched from LOW..HIGH to 0..1 and clamped: below LOW is background,
# above HIGH is certainly the person. The detailed model never goes above about 0.94.
STRETCH = {"person": (0.10, 0.85), "person-parts": (0.10, 0.80)}
# Where the edge sits after the upsample: the level of the guided mask that becomes half opaque.
# Below 0.5 leans the edge outward, onto the background, by a pixel or two at 1080 across.
EDGE_LEAN = 0.36
# How sure a pixel of the stored mask (0..1) must be to count as surely the person, for `reach`.
SURE = 0.8


@dataclass
class PersonMasks:
    info: VideoInfo
    path: Path
    width: int
    height: int
    model: str
    seconds: float = 0.0  # how long finding the masks took (0 when read from the cache)

    @property
    def n(self) -> int:
        return self.info.frames

    def frames(self) -> Iterator[np.ndarray]:
        """Every frame's mask in order, float32 0..1 at the work size. Always exactly `n` of them."""
        last = np.zeros((self.height, self.width), np.float32)
        count = 0
        for m in _read_gray(self.path, self.width, self.height):
            if count >= self.n:
                break
            last = m.astype(np.float32) / 255.0
            count += 1
            yield last
        while count < self.n:  # the cache is a frame short of the picture: hold the last one
            count += 1
            yield last

    def at(self, i: int) -> np.ndarray:
        """One frame's mask (float32 0..1, work size)."""
        i = int(np.clip(i, 0, self.n - 1))
        start = max(0.0, (i - 0.25) / self.info.rate)
        for m in _read_gray(self.path, self.width, self.height, start=start, count=1):
            return m.astype(np.float32) / 255.0
        return np.zeros((self.height, self.width), np.float32)

    def full(self, mask: np.ndarray, size: tuple[int, int], guide: Optional[np.ndarray] = None,
             crop: Optional[tuple[float, float, float, float]] = None, lean: float = EDGE_LEAN,
             reach: Optional[float] = None) -> np.ndarray:
        """The mask at `size` (w, h), float32 0..1.

        `crop` (x, y, w, h, in the clip's own pixels) takes the part of the mask under a reframed
        picture first. `guide` is the output frame itself (RGB, at `size`): with it the edge is
        rebuilt from the real picture, so hair and collars keep their shape. `lean` moves the edge
        toward the background: for anything laid behind the person a sliver of the real background
        at the edge is invisible, and a sliver of the layer on the person is not.

        `reach` (output pixels) keeps the soft edge only that close to where the model is sure of
        the person; further out, a faint guess is dropped. The model half-takes a vase or a plant
        behind a head for hair, in a flame that flickers from frame to frame (clip A, 28 October
        2026: up to 35 rows of the small mask above the head, never more than 0.6 sure), and
        anything laid behind the person goes see-through there. A real hair edge sits within a few
        pixels of the sure part.
        """
        import cv2

        m = mask
        if crop is not None:
            sx, sy = self.width / self.info.width, self.height / self.info.height
            x, y, w, h = crop
            x0, y0 = int(np.floor(x * sx)), int(np.floor(y * sy))
            x1, y1 = int(np.ceil((x + w) * sx)), int(np.ceil((y + h) * sy))
            m = m[max(0, y0):min(self.height, y1), max(0, x0):min(self.width, x1)]
        out = cv2.resize(m, size, interpolation=cv2.INTER_LINEAR)
        if guide is not None:
            g = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            radius = max(2, int(round(size[0] / 220)))
            out = cv2.ximgproc.guidedFilter(guide=g, src=out, radius=radius, eps=2e-3)
        if reach is not None and reach > 0:
            r = max(1, int(round(reach * m.shape[1] / size[0])))  # in the small mask's pixels
            sure = (m >= SURE).astype(np.uint8)
            near = cv2.dilate(sure, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
            near = cv2.GaussianBlur(near.astype(np.float32), (0, 0), max(0.7, r / 2))
            out = out * np.clip(cv2.resize(near, size, interpolation=cv2.INTER_LINEAR) * 2.0, 0.0, 1.0)
        # The filter can ring a little past 0 and 1: clamp, keep the inside exactly opaque, and
        # put the half-way point where `lean` says (0.5 is the model's own edge).
        lo = 0.02
        hi = max(lo + 0.05, 2 * lean - lo)
        return np.clip((out - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    def crown(self, mask: np.ndarray, x0: float, x1: float, level: float = 0.5) -> Optional[float]:
        """The top of the person between x0 and x1 (the clip's own pixels): where the head ends,
        hair and hat included. None when nobody is in that band."""
        sx = self.width / self.info.width
        a, b = int(max(0, x0 * sx)), int(min(self.width, np.ceil(x1 * sx)))
        if b <= a:
            return None
        rows = np.flatnonzero((mask[:, a:b] > level).mean(axis=1) > 0.15)
        return float(rows[0] * self.info.height / self.height) if len(rows) else None


def _work_size(info: VideoInfo) -> tuple[int, int]:
    w = min(WORK_WIDTH, info.width)
    w -= w % 2
    h = int(round(info.height * w / info.width / 2)) * 2
    return w, h


def _read_gray(path: Path, width: int, height: int, start: float = 0.0, count: Optional[int] = None) -> Iterator[np.ndarray]:
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start > 0:
        cmd += ["-ss", f"{start:.6f}"]
    cmd += ["-i", str(path)]
    if count is not None:
        cmd += ["-frames:v", str(count)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "gray", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = width * height
    try:
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(height, width)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def _segment(info: VideoInfo, model: str, size: tuple[int, int]) -> Iterator[np.ndarray]:
    """The model's person confidence per frame (float32 0..1, work size), unprocessed."""
    import cv2

    from .models import vision_path

    with quiet_native():
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        options = vision.ImageSegmenterOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(vision_path(model))),
            running_mode=vision.RunningMode.VIDEO,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        segmenter = vision.ImageSegmenter.create_from_options(options)
    try:
        for i, frame in enumerate(FrameReader(info, size=size)):
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            result = segmenter.segment_for_video(image, int(round(i * 1000 / info.rate)))
            masks = result.confidence_masks
            if len(masks) == 1:
                p = masks[0].numpy_view()
            else:  # the detailed model: everything that is not background is the person
                p = 1.0 - masks[0].numpy_view()
            p = np.asarray(p, np.float32).reshape(size[1], size[0])
            if p.shape != (size[1], size[0]):
                p = cv2.resize(p, size, interpolation=cv2.INTER_LINEAR)
            yield p
    finally:
        segmenter.close()


def _clean(p: np.ndarray, model: str) -> np.ndarray:
    """Stretch to the full range, clamp, and soften the stair-steps a little."""
    import cv2

    lo, hi = STRETCH[model]
    m = np.clip((p - lo) / (hi - lo), 0.0, 1.0)
    return cv2.GaussianBlur(m, (0, 0), 0.8)


def _find(info: VideoInfo, model: str, size: tuple[int, int], target: Path, meta: dict) -> None:
    """Segment every frame, steady each pixel across three frames (a median, which removes
    one-frame flicker without dragging a moving edge), and write the masks lossless."""
    need_ffmpeg()
    w, h = size
    tmpdir = tempfile.mkdtemp(prefix="fx-person-")
    partial = Path(tmpdir) / "person.mkv"
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
           "-s", f"{w}x{h}", "-framerate", f"{info.fps.numerator}/{info.fps.denominator}", "-i", "-",
           "-c:v", "ffv1", "-level", "3", "-g", "1", "-pix_fmt", "gray",
           "-metadata", "comment=" + json.dumps(meta, separators=(",", ":")), str(partial)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    first: Optional[np.ndarray] = None
    written = 0
    try:
        window: list[np.ndarray] = []

        def emit(m: np.ndarray) -> None:
            nonlocal first, written
            q = np.round(m * 255.0).astype(np.uint8)
            if first is None:
                first = q
            proc.stdin.write(q.tobytes())  # type: ignore[union-attr]
            written += 1

        for p in _segment(info, model, size):
            window.append(_clean(p, model))
            if len(window) == 2 and written == 0:
                emit(window[0])  # the first frame has no frame before it
            if len(window) == 3:
                emit(np.median(np.stack(window), axis=0))
                window.pop(0)
        if window:
            emit(window[-1])  # the last frame has no frame after it
        while written < info.frames:  # the reader can come up a frame short
            emit(window[-1] if window else np.zeros((h, w), np.float32))
        proc.stdin.close()
        if proc.wait() != 0:
            err = proc.stderr.read().decode(errors="replace").strip() if proc.stderr else ""
            raise MediaError("the person masks could not be written" + (f": {err.splitlines()[-1]}" if err else ""))
        # Read the first frame back: the cache is trusted only if it is bit for bit what was written.
        back = next(_read_gray(partial, w, h, count=1), None)
        if first is None or back is None or not np.array_equal(back, first):
            raise MediaError("the person masks did not read back exactly; they were not kept")
        shutil.move(str(partial), str(target))
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        shutil.rmtree(tmpdir, ignore_errors=True)


def _cached_meta(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=comment", "-of", "json", str(path)],
                         capture_output=True, text=True)
    try:
        tags = json.loads(out.stdout).get("format", {}).get("tags", {})
        return json.loads(tags.get("comment") or tags.get("COMMENT") or "{}")
    except (ValueError, AttributeError):
        return {}


def masks(info: VideoInfo, model: str = MODEL, refresh: bool = False) -> PersonMasks:
    """The clip's person masks, from the cache beside the clip or found now (once)."""
    if model not in STRETCH:
        raise ValueError(f"unknown person model {model!r}")
    size = _work_size(info)
    cache = paths.beside(info.path, "person.mkv" if model == MODEL else f"person-{model}.mkv")
    meta = {"version": CACHE_VERSION, "model": model, "source": paths.source_signature(info.path),
            "frames": info.frames, "size": list(size)}
    if not refresh and cache.exists() and _cached_meta(cache) == meta:
        return PersonMasks(info, cache, size[0], size[1], model)
    started = time.time()
    _find(info, model, size, cache, meta)
    return PersonMasks(info, cache, size[0], size[1], model, seconds=time.time() - started)


def check_range(m: PersonMasks, sample: int = 12) -> dict:
    """What the stored masks span across a few frames: the inside must be exactly 1 and the far
    background exactly 0 (the ghost check)."""
    lows, highs, full = [], [], []
    for i in np.linspace(0, m.n - 1, min(sample, m.n)).astype(int):
        a = m.at(int(i))
        lows.append(float(a.min()))
        highs.append(float(a.max()))
        full.append(float((a >= 1.0).mean()))
    return {"min": min(lows), "max": max(highs), "opaque_share": round(float(np.mean(full)), 3)}


def _env_quiet() -> None:
    os.environ.setdefault("GLOG_minloglevel", "2")


_env_quiet()
