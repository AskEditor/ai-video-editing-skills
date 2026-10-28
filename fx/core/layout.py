"""The safe zone, and placing text where it never covers the face.

Instagram's safe zone, the rule every effect keeps: important text sits between x = 120 and
x = 960 on a 1080-wide frame and above the bottom fifth, where the caption and buttons sit. The top
tenth holds the app's own bar, so text stays below it too. On any other size the same fractions
apply.

**A panel's own safe part.** When an effect is drawn inside one panel of a layout (a chart in the
top half of a split screen), its frame is the panel, and the app's fractions of the panel are not
where text is safe on the screen. The layout says where it is with `zone_for(w, h, rect)`; inside
that block every `SafeZone(w, h)` of that size is `rect`, so the effects prepared there keep to it
without knowing they are inside anything.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    def intersects(self, other: "Rect", margin: float = 0.0) -> bool:
        return (self.x0 - margin < other.x1 and other.x0 < self.x1 + margin
                and self.y0 - margin < other.y1 and other.y0 < self.y1 + margin)

    def overlap(self, other: "Rect") -> float:
        w = min(self.x1, other.x1) - max(self.x0, other.x0)
        h = min(self.y1, other.y1) - max(self.y0, other.y0)
        return max(0.0, w) * max(0.0, h)

    def inside(self, other: "Rect", slack: float = 0.5) -> bool:
        return (self.x0 >= other.x0 - slack and self.x1 <= other.x1 + slack
                and self.y0 >= other.y0 - slack and self.y1 <= other.y1 + slack)

    def union(self, other: "Rect") -> "Rect":
        return Rect(min(self.x0, other.x0), min(self.y0, other.y0), max(self.x1, other.x1), max(self.y1, other.y1))

    def grow(self, left: float, top: float, right: float, bottom: float) -> "Rect":
        return Rect(self.x0 - left, self.y0 - top, self.x1 + right, self.y1 + bottom)

    def shift(self, dx: float, dy: float) -> "Rect":
        return Rect(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)

    def scale(self, sx: float, sy: float, dx: float = 0.0, dy: float = 0.0) -> "Rect":
        """Map into another frame: x' = x * sx + dx."""
        return Rect(self.x0 * sx + dx, self.y0 * sy + dy, self.x1 * sx + dx, self.y1 * sy + dy)

    def as_list(self) -> list[float]:
        return [round(self.x0, 1), round(self.y0, 1), round(self.x1, 1), round(self.y1, 1)]

    @staticmethod
    def of(values: Sequence[float]) -> "Rect":
        return Rect(*map(float, values[:4]))

    @staticmethod
    def around(rects: Iterable["Rect"]) -> Optional["Rect"]:
        out = None
        for r in rects:
            out = r if out is None else out.union(r)
        return out


# The safe parts given for frames that are panels of a layout: ((width, height), rect), newest last.
_GIVEN: ContextVar[tuple] = ContextVar("given_safe_zones", default=())


@contextlib.contextmanager
def zone_for(width: int, height: int, rect: Rect):
    """Inside this block, `SafeZone(width, height)` is `rect` (in that frame's own pixels) rather
    than the app's fractions of it. Frames of any other size are not touched."""
    size = (int(round(width)), int(round(height)))
    token = _GIVEN.set(_GIVEN.get() + ((size, rect),))
    try:
        yield
    finally:
        _GIVEN.reset(token)


class SafeZone:
    LEFT = 120 / 1080
    RIGHT = 960 / 1080
    TOP = 0.10
    BOTTOM = 0.80

    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.rect = Rect(width * self.LEFT, height * self.TOP, width * self.RIGHT, height * self.BOTTOM)
        size = (int(round(width)), int(round(height)))
        for given, rect in reversed(_GIVEN.get()):
            if given == size:
                self.rect = rect
                break

    def contains(self, r: Rect) -> bool:
        return r.inside(self.rect)


def halo_room(room: Rect, zone: Rect, halo: tuple[float, float, float, float]) -> Rect:
    """Where a thing whose drawing (its shadow, say) reaches `halo` (left, top, right, bottom) past its
    box may go in `room` so that the drawing stays inside the safe zone `zone`: the room, kept `halo`
    in from the zone's edges. A side of the room inside the zone (towards the head, a neighbour) is
    left as it is, since a shadow may fall towards it."""
    left, top, right, bottom = halo
    x0, y0 = max(room.x0, zone.x0 + left), max(room.y0, zone.y0 + top)
    x1, y1 = min(room.x1, zone.x1 - right), min(room.y1, zone.y1 - bottom)
    return Rect(x0, y0, max(x0, x1), max(y0, y1))


@dataclass(frozen=True)
class Placement:
    rect: Rect
    where: str  # "chest", "lower-third", "top" or "fallback"


ORDERS = {
    "auto": ("chest", "lower-third", "top"),
    "bottom": ("lower-third", "chest", "top"),
    "top": ("top", "chest", "lower-third"),
}


def place(block_w: float, block_h: float, zone: SafeZone, head: Optional[Rect],
          order: Sequence[str] = ORDERS["auto"], gap: Optional[float] = None) -> Optional[Placement]:
    """Where a block of `block_w` x `block_h` goes: centred across the safe zone, at the first spot
    in `order` that stays inside the zone and clear of `head` (the head's box over every frame the
    block is on screen). None when no spot fits; the caller then makes the block smaller."""
    z = zone.rect
    gap = zone.height * 0.02 if gap is None else gap
    if block_w > z.w + 0.5 or block_h > z.h + 0.5:
        return None
    x0 = z.cx - block_w / 2
    for where in order:
        if head is None:
            # No face found: the lower third is the usual place.
            y1 = z.y1 - zone.height * 0.01
            return Placement(Rect(x0, y1 - block_h, x0 + block_w, y1), "lower-third")
        if where == "chest":
            y0 = max(head.y1 + gap, z.y0)
            r = Rect(x0, y0, x0 + block_w, y0 + block_h)
        elif where == "lower-third":
            y1 = z.y1 - zone.height * 0.01
            r = Rect(x0, y1 - block_h, x0 + block_w, y1)
            if r.y0 < head.y1 + gap:
                continue
        elif where == "top":
            y1 = min(head.y0 - gap, z.y1)
            r = Rect(x0, y1 - block_h, x0 + block_w, y1)
        else:
            continue
        if zone.contains(r) and not r.intersects(head):
            return Placement(r, where)
    return None
