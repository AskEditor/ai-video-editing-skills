"""A clip as a layout shows it: what an effect laid inside a layout is prepared on.

A layout moves the speaker into a panel and fills the rest of the frame with other pictures. An
effect laid inside it (captions over the whole frame, a chart in the top panel) has to be prepared
for the picture it will be drawn on, not for the clip as it was shot: its words must keep clear of
the face where the face now is, and stay inside the part of the panel where text is safe.

    view = View(clip, (w, h), place=where_each_frame_goes, picture=draw_frame, zone=safe_rect)
    layer = layers.prepare(view, spec, start, end, view_framing)    # applies the zone too

A `View` is the clip with everything about its picture replaced and everything else kept:

- `info`: the picture's own size (the whole canvas, or one panel of it). Frame rate, length, sound
  and file stay the clip's. Its `picture` draws frame i, so an effect that looks at what it sits on
  (to choose a tone, to tint glass) sees the composed picture; one that streams the clip's own
  picture from its file cannot be laid on a view, and says so.
- `face`: the clip's face track moved to where each frame shows it, or no face at all.
- `person()`: the person masks moved the same way, or empty.
- `zone`: where text is safe in this picture, in its own pixels (None: the app's fractions, which
  is right for a whole frame).
- `words`, `quiet`, `sound` and everything else: the clip's own.

`place(i)` says where frame i of the clip is shown: `(window, rect)`, the window of the clip's own
picture (x, y, w, h in the clip's pixels) shown in `rect` (a Rect in the view's pixels) at one
scale, or None when the clip is not in this view at that frame. Only what lands inside `rect` is
in the view: a face whose middle falls outside it is not there.
"""

from __future__ import annotations

import dataclasses
import os
from functools import cached_property
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

from . import person as personmod
from .face import FaceTrack
from .layout import Rect

Window = tuple[float, float, float, float]
Place = Callable[[int], Optional[tuple[Window, Rect]]]


def _move(box: np.ndarray, window: Window, rect: Rect) -> np.ndarray:
    """x, y, w, h in the clip's pixels -> the same box where `rect` shows `window`."""
    wx, wy, ww, _ = window
    s = rect.w / ww
    return np.array([(box[0] - wx) * s + rect.x0, (box[1] - wy) * s + rect.y0, box[2] * s, box[3] * s])


def moved_faces(face: FaceTrack, place: Optional[Place], size: tuple[int, int]) -> FaceTrack:
    """The face track as the view shows it: every box moved to where its frame is placed. Frames
    where the face is not in the view hold the nearest frame where it is (as the clip's own track
    holds across a gap), and count as not found."""
    n = face.n
    raw = np.full((n, 5), np.nan)
    trusted = np.full((n, 5), np.nan)
    boxes = np.zeros((n, 4))
    shown = np.zeros(n, bool)
    if place is not None and face.any():
        for i in range(n):
            p = place(i)
            if p is None:
                continue
            window, rect = p
            b = _move(face.boxes[i], window, rect)
            if not (rect.x0 <= b[0] + b[2] / 2 <= rect.x1 and rect.y0 <= b[1] + b[3] / 2 <= rect.y1):
                continue
            boxes[i], shown[i] = b, True
            for src, dst in ((face.raw, raw), (face.trusted, trusted)):
                if not np.isnan(src[i, 0]):
                    dst[i, :4] = _move(src[i, :4], window, rect)
                    dst[i, 4] = src[i, 4]
    if shown.any() and not shown.all():
        idx = np.flatnonzero(shown)
        for c in range(4):
            boxes[:, c] = np.interp(np.arange(n), idx, boxes[idx, c])
    return FaceTrack(int(size[0]), int(size[1]), face.fps, raw, boxes, trusted=trusted)


class MovedMasks(personmod.PersonMasks):
    """The clip's person masks as the view shows them: each frame's mask moved to where its frame is
    placed, and nothing anywhere else. With no masks to move (a panel the person is not in), every
    frame is empty."""

    def __init__(self, base: Optional[personmod.PersonMasks], info, place: Optional[Place]):
        w = personmod.WORK_WIDTH
        h = max(2, int(round(w * info.height / info.width)))
        super().__init__(info=info, path=base.path if base is not None else Path(os.devnull), width=w, height=h,
                         model=base.model if base is not None else personmod.MODEL)
        self.base, self.place = base, place

    def _moved(self, m: np.ndarray, i: int) -> np.ndarray:
        import cv2

        p = self.place(i) if self.place is not None else None
        if p is None:
            return np.zeros((self.height, self.width), np.float32)
        (wx, wy, ww, _), rect = p
        b = self.base
        bx, by = b.width / b.info.width, b.height / b.info.height  # the clip's pixels -> its small mask
        k = self.width / self.info.width  # the view's pixels -> its small mask
        s = rect.w / ww
        m2 = np.float32([[s * k / bx, 0, (rect.x0 - wx * s) * k], [0, s * k / by, (rect.y0 - wy * s) * k]])
        out = cv2.warpAffine(m, m2, (self.width, self.height), flags=cv2.INTER_LINEAR, borderValue=0)
        inside = np.zeros_like(out)
        x0, y0 = int(max(0, np.floor(rect.x0 * k))), int(max(0, np.floor(rect.y0 * k)))
        x1, y1 = int(min(self.width, np.ceil(rect.x1 * k))), int(min(self.height, np.ceil(rect.y1 * k)))
        inside[y0:y1, x0:x1] = out[y0:y1, x0:x1]
        return inside

    def frames(self) -> Iterator[np.ndarray]:
        n = self.n
        if self.base is None or self.place is None:
            empty = np.zeros((self.height, self.width), np.float32)
            for _ in range(n):
                yield empty
            return
        count = 0
        for i, m in enumerate(self.base.frames()):
            if count >= n:
                break
            count += 1
            yield self._moved(m, i)

    def at(self, i: int) -> np.ndarray:
        i = int(np.clip(i, 0, self.n - 1))
        if self.base is None or self.place is None:
            return np.zeros((self.height, self.width), np.float32)
        return self._moved(self.base.at(i), i)


class View:
    """The clip as a picture of `size` (w, h) shows it. See the module's notes."""

    def __init__(self, clip, size: tuple[int, int], place: Optional[Place] = None,
                 picture: Optional[Callable[[int], np.ndarray]] = None, zone: Optional[Rect] = None):
        self.base = clip
        self.size = (int(size[0]), int(size[1]))
        self.info = dataclasses.replace(clip.info, width=self.size[0], height=self.size[1], picture=picture)
        self.place = place
        self.zone = zone

    @cached_property
    def face(self) -> FaceTrack:
        return moved_faces(self.base.face, self.place, self.size)

    def person(self, model: str = personmod.MODEL) -> MovedMasks:
        return MovedMasks(self.base.person(model) if self.place is not None else None, self.info, self.place)

    def __getattr__(self, name: str):
        # Everything that is not about the picture (words, sound, the file, the fixes) is the clip's.
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)
