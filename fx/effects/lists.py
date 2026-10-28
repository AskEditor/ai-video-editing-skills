"""The engine `ranked-list` and `checklist` share: rows that land on their words, on a card where the
head is not, with a soft sound per row under the voice.

- **Rows.** Given (`items=`: names, `name@4.2`, `5: name`, or a .json file) or heard in the words
  (`items.in_order`: "number five", "step two", "first", "then", "and finally", a run with a last
  "and"). Each row lands on the first frame of its word, the word's start corrected for the
  transcriber's lateness by the core; its entrance runs in the few frames before, so on that frame
  it is already still and whole. A row given a time lands on the word starting nearest it.
- **Steps done** (the checklist). A step is ticked where it is said done ("Done.", "Check.", "and
  that's it"), else where the next step starts (its "then", "next", "step three"), the last where
  its sentence ends. A tick never falls on the same frame as the next row: it moves a beat earlier.
- **Room.** The head is followed over every frame the card is up (tier-list's `head_track` and
  `bands`, the crown read from the person mask), and the card is made as large as the room above,
  beside or below it allows. Where the hair rises into the card for a moment, the person is laid
  back over it (`PersonFront`); the face itself is never under it.
- **Words that do not fit** are made shorter by whole words: first without "just", "really" and
  "the" (`items.shorter`), then cut between two words with an ellipsis (`items.cut_to`). The run
  says which rows were shortened, and the plan keeps what was said.
- **A look** draws the rows: a numbered board, a countdown, a tick, a spinner, a switch, a strike
  or a bar. The engine times, places, sounds and checks them, the same for every look.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..core import items as found_items
from ..core import moments
from ..core import text as textmod
from ..core.compose import Sprite, over
from ..core.layout import Rect, SafeZone, halo_room
from ..core.words import Word

STYLES = ("solid", "glass", "paper")
POSITIONS = ("auto", "top", "side", "below")

# ---- the look's colours ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Palette:
    card: tuple[int, int, int]
    card_alpha: float
    ink: tuple[int, int, int]  # the words
    line: tuple[int, int, int]  # hairlines, empty slots, a switch's track
    line_alpha: float
    on_accent: tuple[int, int, int]  # words and marks laid on the accent
    face_text: str
    face_bold: str  # numbers and titles


PALETTES = {
    "solid": Palette((15, 16, 20), 0.90, (255, 255, 255), (255, 255, 255), 0.14, (15, 16, 20), "poppins-bold",
                     "poppins-black"),
    "glass": Palette((15, 16, 20), 0.0, (255, 255, 255), (255, 255, 255), 0.20, (15, 16, 20), "poppins-bold",
                     "poppins-black"),
    "paper": Palette((248, 244, 235), 1.0, (35, 30, 25), (35, 30, 25), 0.16, (255, 255, 255), "dm-serif",
                     "dm-serif"),
}

# Sizes, as shares of u (the height of one row).
PAD = 0.22  # the card's own margin
GAP = 0.08  # between rows
TEXT = 0.50  # the words of a row
TITLE_H = 0.72
U_MAX = 0.064  # the largest row, as a share of the frame's height (any larger looks shouted)
READABLE = 0.0135  # the smallest words that read on a phone, as a share of the frame's height
ROOM_GAP = 0.012  # kept between the card and the head, as a share of the frame's height
MIN_SHARE = 0.46  # a card is at least this share of the safe zone's width, so it reads as a card
SHADOW = 0.34  # a solid or paper card's shadow reaches this far past it on every side, as a share of u

# Motion, in seconds.
ENTER = 0.16  # a row's entrance: it is still and whole on the first frame of its word
CARD_IN = 0.30  # the card's own entrance
CARD_LEAD = 0.45  # the card is up this long before the first row lands, its empty rows a promise
CARD_OUT = 0.22
TICK_DRAW = 0.18  # a tick drawing itself
PULSE = 0.30
APART = 0.20  # a tick is never closer than this to the next row's landing


def frame_of(t: float, fps: float) -> int:
    """The first frame that shows a moment: the first one starting at or after it (the pack's rule)."""
    return int(math.ceil(t * fps - 1e-6))


def ease_out(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 1.0 - (1.0 - x) ** 3


def ease_in_out(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 3 * x * x - 2 * x * x * x


def back_out(x: float, s: float = 1.7) -> float:
    x = min(1.0, max(0.0, x)) - 1.0
    return 1.0 + (s + 1.0) * x ** 3 + s * x ** 2


# ---- the rows -------------------------------------------------------------------------------------------


@dataclass
class Given:
    """An item as it was given: words, a time if given, a picture, a number, and (checklist) when done."""
    text: str
    at: Optional[float] = None
    image: Optional[str] = None
    number: int = 0
    done: Optional[float] = None


@dataclass
class Row:
    said: str  # the item's words, as said or given
    at: float  # when its word starts (seconds)
    frame: int  # the first frame of its word: where it lands
    land: float  # that frame's start (seconds): when it is still and whole, and its sound
    word: str  # the word it lands on, as heard
    how: str  # heard, said, given, placed or plan
    number: int = 0  # the number it shows
    image: Optional[str] = None
    marker_at: Optional[float] = None  # when its marker ("then", "step two") starts
    ends: Optional[float] = None  # when its own words end
    words: Optional[tuple[int, int]] = None
    done: Optional[float] = None  # the checklist's tick: a frame's start
    done_frame: Optional[int] = None
    done_how: str = ""
    done_word: str = ""
    asked: Optional[float] = None
    text: str = ""  # as shown (made shorter when it must be)
    cut: int = 0  # 0 as said, 1 shorter words, 2 cut with an ellipsis
    px: int = 0
    # drawing
    slot: Optional[Rect] = None  # the row's box inside the card
    sprite: Optional[Sprite] = None
    dim_sprite: Optional[Sprite] = None


def parse_items(spec: str, clip_dir: Path, most: int) -> list[Given]:
    """The items given: `name; name@4.2; 5: name` (or one per line, or commas when no ";" is
    used), JSON text or a .json file: a list of names or {"text", "at", "image", "number", "done"},
    or {"items": [...]}."""
    data, base = found_items.load(spec, clip_dir)
    raw: list = []
    if data is None:
        s = spec.strip()
        if not s:
            return []
        parts = re.split(r"[;\n|]", s) if re.search(r"[;\n|]", s) else s.split(",")
        raw = [p.strip() for p in parts if p.strip()]
    elif isinstance(data, dict):
        raw = data.get("items") or data.get("steps") or data.get("rows") or []
        if not isinstance(raw, list):
            raise found_items.ItemsError("the items file needs a list: {\"items\": [\"…\", \"…\"]}")
    elif isinstance(data, list):
        raw = data
    else:
        raise found_items.ItemsError("the items must be a list of names, such as [\"Lighting\", \"Sound\"]")
    out = []
    for r in raw:
        number = 0
        done = None
        if isinstance(r, str):
            m = re.fullmatch(r"\s*#?([0-9]{1,2})\s*[:.)]\s*(.+)", r)
            if m:
                number, r = int(m.group(1)), m.group(2)
        elif isinstance(r, dict):
            try:
                number = int(r.get("number") or 0)
                done = float(r["done"]) if r.get("done") is not None else None
            except (TypeError, ValueError):
                raise found_items.ItemsError(f"a number or a done time that is not a number: {r!r}") from None
        name, at, pic = found_items.entry(r, base, clip_dir)
        if not name and not pic:
            raise found_items.ItemsError("an item has no words and no picture")
        if len(name) > 60:
            raise found_items.ItemsError(f"{name[:40]!r}… is too long for one row; keep it under 60 letters")
        out.append(Given(name, at, pic, number, done))
    if len(out) > most:
        raise found_items.ItemsError(f"{len(out)} items is more than this look shows readably; keep it to {most}")
    return out


def _span_of(words: list[Word], name: str, used: set[int]) -> Optional[tuple[int, int]]:
    for i, j, _ in found_items.name_spans(words, name):
        if not any(k in used for k in range(i, j + 1)):
            return i, j
    return None


def rows_of(words: list[Word], given: list[Given], fps: float, duration: float,
            notes: list[str], quiet=None) -> tuple[list[Row], list[found_items.Step], str]:
    """The rows in the order they are shown, each timed, and the list heard (for steps said done).
    Given items keep their order; heard ones come in the order said. Returns (rows, heard, source).
    A given item lands where its name was not taken back (`moments.taken_back` of `words`, the
    clip's whole words, with `quiet` its quiet stretches), unless it is said nowhere else."""
    heard = found_items.in_order(words)
    if not given:
        if not heard:
            return [], [], "none"
        rows = []
        for s in heard:
            rows.append(Row(s.text, s.at, frame_of(s.at, fps), frame_of(s.at, fps) / fps,
                            words[s.anchor].clean, "heard", s.group, None, s.marker_at, s.ends, s.words))
        kinds = {s.kind for s in heard}
        source = ("a run said with commas and a last \"and\"" if kinds == {"series"} else
                  "the markers said (" + ", ".join(dict.fromkeys(s.marker for s in heard if s.marker)) + ")")
        return rows, heard, source
    timed, more = found_items.time_items(given, words, heard, duration, heard_how="heard",
                                         back=moments.taken_back(words, [] if quiet is None else quiet))
    notes.extend(more)
    by_n = {t.n: t for t in timed}
    used: set[int] = set()
    rows = []
    for n, g in enumerate(given):
        t = by_n[n]
        at = t.at
        span = None
        marker_at = None
        ends = None
        if t.heard is not None:
            span = t.heard.words
            marker_at = getattr(t.heard, "marker_at", None)
            ends = getattr(t.heard, "ends", None)
        elif t.how == "said":
            span = _span_of(words, g.text, used)
        if span is not None:
            used.update(range(span[0], span[1] + 1))
            ends = ends if ends is not None else words[span[1]].end
        f = frame_of(at, fps)
        rows.append(Row(g.text, at, f, f / fps, t.word, t.how, g.number, g.image, marker_at, ends, span,
                        asked=t.asked))
        if g.done is not None:
            rows[-1].done = g.done
            rows[-1].done_how = "given"
    return rows, heard, "the items given"


def number_rows(rows: list[Row], countdown: bool, notes: list[str]) -> None:
    """The number each row shows. Numbers said or given win; otherwise the order: 1, 2, 3 for a
    list, and in a countdown the first said is the highest number, the last is number one."""
    n = len(rows)
    said = [r.number for r in rows]
    if all(said) and len(set(said)) == n and max(said) <= max(n, 10):
        return
    if any(said) and any(r.how != "heard" for r in rows if r.number):
        notes.append("only some items were given a number, so they are numbered in order")
    for k, r in enumerate(rows):
        r.number = n - k if countdown else k + 1


def done_times(rows: list[Row], words: list[Word], fps: float, duration: float, end: float) -> None:
    """When each step is ticked: given; said done; where the next step starts; the last where its
    sentence ends. Never on the frame the next row lands, nor before its own row is still."""
    steps = []
    for r in rows:
        if r.words is not None:
            steps.append(found_items.Step(r.said, 0, r.at, r.words, r.words[0], r.said,
                                          marker_at=r.marker_at if r.marker_at is not None else r.at,
                                          ends=r.ends if r.ends is not None else r.at))
        else:
            steps.append(None)
    said_done: list[Optional[tuple[float, int]]] = [None] * len(rows)
    real = [s for s in steps if s is not None]
    if real:
        found = found_items.done_after(words, real)
        it = iter(found)
        said_done = [next(it) if s is not None else None for s in steps]
    for k, r in enumerate(rows):
        if r.done is not None and r.done_how == "given":
            t, how, word = r.done, "given", ""
        elif said_done[k] is not None:
            t, wi = said_done[k]
            how, word = "said done", words[wi].clean
        elif k + 1 < len(rows):
            nxt = rows[k + 1]
            t = nxt.marker_at if nxt.marker_at is not None else nxt.at
            how, word = "the next step starts", ""
            if nxt.marker_at is not None and nxt.marker_at < nxt.at - 0.05:
                word = "the next step's marker"
        else:
            t = (r.ends if r.ends is not None else r.at + 1.0) + 0.35
            how, word = "its sentence ends", ""
            t = min(t, end - 0.15)
        nxt_land = rows[k + 1].land if k + 1 < len(rows) else None
        if nxt_land is not None and abs(nxt_land - t) < APART:
            t = nxt_land - APART
        t = max(t, r.land + 0.45)
        t = min(t, end - 1.0 / fps)
        f = frame_of(t, fps)
        if nxt_land is not None and f >= frame_of(nxt_land, fps):
            f = max(r.frame + 1, frame_of(nxt_land, fps) - max(1, int(round(APART * fps))))
        r.done_frame = f
        r.done = f / fps
        r.done_how = how
        r.done_word = word


# ---- fitting words ----------------------------------------------------------------------------------


def fit_words(said: str, room: float, px: int, face: str, allow: int) -> Optional[tuple[str, int, int]]:
    """The row's words in at most `room` pixels: (text, px, cut). `allow` 0 keeps them as said, 1
    lets them lose softeners and articles and come down to 0.88 of the size, 2 lets them be cut
    between words with an ellipsis. None when nothing allowed fits."""
    def fits(s: str, p: int) -> bool:
        return textmod.advance(s, face, p) <= room

    if fits(said, px):
        return said, px, 0
    if allow < 1:
        return None
    forms = found_items.shorter(said)
    small = max(4, int(px * 0.88))
    for p in (px, small):
        for k, f in enumerate(forms):
            if fits(f, p):
                return f, p, 0 if k == 0 else 1
    if allow < 2:
        return None
    lean = forms[-1]
    cut = found_items.cut_to(lean, lambda s: fits(s, small))
    if fits(cut, small):
        return cut, small, 2
    return None


# ---- drawing --------------------------------------------------------------------------------------------


def rgba(w: float, h: float):
    from PIL import Image

    return Image.new("RGBA", (max(1, int(math.ceil(w))), max(1, int(math.ceil(h)))), (0, 0, 0, 0))


def round_rect(img, box: tuple[float, float, float, float], radius: float, fill, k: int = 4) -> None:
    """A smooth rounded rectangle laid onto `img` (PIL RGBA), drawn four times larger and brought down."""
    from PIL import Image, ImageDraw

    x0, y0, x1, y1 = box
    w, h = int(math.ceil(x1 - x0)) + 2, int(math.ceil(y1 - y0)) + 2
    big = Image.new("RGBA", (w * k, h * k), (0, 0, 0, 0))
    fx, fy = x0 - math.floor(x0), y0 - math.floor(y0)
    ImageDraw.Draw(big).rounded_rectangle(
        (fx * k, fy * k, (fx + x1 - x0) * k - 1, (fy + y1 - y0) * k - 1), radius=max(0.0, radius * k), fill=fill)
    img.alpha_composite(big.resize((w, h), Image.LANCZOS), (int(math.floor(x0)), int(math.floor(y0))))


def shadowed(img, blur: float, dy: float, opacity: float, margin: int):
    """`img` with a soft shadow of its own shape under it, on a canvas `margin` larger each side."""
    from PIL import Image, ImageFilter

    W, H = img.width + 2 * margin, img.height + 2 * margin
    a = Image.new("L", (W, H), 0)
    a.paste(img.getchannel("A"), (margin, margin + int(round(dy))))
    if blur > 0:
        a = a.filter(ImageFilter.GaussianBlur(blur))
    a = a.point(lambda v: int(v * opacity))
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out.putalpha(a)
    out.alpha_composite(img, (margin, margin))
    return out


def words_sprite(s: str, face: str, px: int, colour, alpha: float = 1.0) -> tuple[Sprite, float, float]:
    """`s` as a sprite: (sprite, the baseline's x and y inside it). Measured on its ink, so a row's
    words sit where they are drawn."""
    from PIL import ImageDraw

    f = textmod.font(face, px)
    l, t, r, b = f.getbbox(s, anchor="ls")
    pad = 3
    img = rgba(r - l + 2 * pad, b - t + 2 * pad)
    ox, oy = pad - l, pad - t
    ImageDraw.Draw(img).text((ox, oy), s, font=f, fill=tuple(colour) + (int(round(255 * alpha)),), anchor="ls")
    return Sprite.from_rgba(np.asarray(img)), ox, oy


def tick_points(d: float) -> list[tuple[float, float]]:
    """A tick inside a d x d square: a short stroke down, a long one up."""
    return [(0.27 * d, 0.53 * d), (0.43 * d, 0.68 * d), (0.74 * d, 0.34 * d)]


def stroke_part(points: list[tuple[float, float]], share: float) -> list[tuple[float, float]]:
    """The first `share` of a polyline, by length."""
    segs = [math.dist(points[i], points[i + 1]) for i in range(len(points) - 1)]
    want = sum(segs) * min(1.0, max(0.0, share))
    out = [points[0]]
    for i, s in enumerate(segs):
        if want >= s:
            out.append(points[i + 1])
            want -= s
        else:
            a, b = points[i], points[i + 1]
            q = want / s if s else 0
            out.append((a[0] + (b[0] - a[0]) * q, a[1] + (b[1] - a[1]) * q))
            break
    return out


def mark_sprite(d: float, kind: str, accent, palette: Palette, share: float = 1.0, angle: float = 0.0,
                k: int = 4) -> Sprite:
    """A row's mark, d pixels across: `ring` (an empty circle), `done` (a filled circle with a tick
    drawn `share` of the way), `spin` (a ring with a turning arc at `angle` degrees)."""
    from PIL import Image, ImageDraw

    D = int(math.ceil(d)) + 2
    big = Image.new("RGBA", (D * k, D * k), (0, 0, 0, 0))
    dr = ImageDraw.Draw(big)
    c = (1 + d / 2) * k
    r = d / 2 * k
    line = max(1.5, 0.085 * d) * k
    ring = tuple(palette.line) + (int(255 * min(1.0, palette.line_alpha * 3.2)),)
    if kind == "ring":
        dr.ellipse((c - r + line / 2, c - r + line / 2, c + r - line / 2, c + r - line / 2), outline=ring,
                   width=int(round(line)))
    elif kind == "spin":
        dr.ellipse((c - r + line / 2, c - r + line / 2, c + r - line / 2, c + r - line / 2),
                   outline=tuple(palette.line) + (int(255 * palette.line_alpha * 1.6),), width=int(round(line)))
        dr.arc((c - r + line / 2, c - r + line / 2, c + r - line / 2, c + r - line / 2), start=angle,
               end=angle + 100, fill=tuple(accent) + (255,), width=int(round(line * 1.15)))
    else:  # done
        dr.ellipse((c - r, c - r, c + r, c + r), fill=tuple(accent) + (255,))
        pts = [((1 + x) * k, (1 + y) * k) for x, y in tick_points(d)]
        part = stroke_part(pts, share)
        if len(part) >= 2:
            w = int(round(max(2.0, 0.12 * d) * k))
            dr.line(part, fill=tuple(palette.on_accent) + (255,), width=w, joint="curve")
            for p in (part[0], part[-1]):
                dr.ellipse((p[0] - w / 2, p[1] - w / 2, p[0] + w / 2, p[1] + w / 2), fill=tuple(palette.on_accent) + (255,))
    small = big.resize((D, D), Image.LANCZOS)
    return Sprite.from_rgba(np.asarray(small))


# ---- the room off the face --------------------------------------------------------------------------------


@dataclass
class Place:
    where: str
    band: Rect
    u: float
    w: float
    h: float
    x: float
    y: float
    cut: int  # the most any row had to be shortened at this size


def choose(room: dict[str, Rect], measure: Callable[[float, Rect, int], Optional[tuple[float, float, int]]],
           H: float, want: str, head: Optional[Rect], notes: list[str],
           room_at: Optional[Callable[[Rect, float], Rect]] = None) -> Optional[Place]:
    """The largest card each room holds, then the room: the one asked for when it has any, else the
    roomiest, above the head on a near tie. Rows are first kept as said, then made shorter, then cut,
    each at sizes down to an easy reading size before the next is tried; below that size only when
    nothing else fits. `room_at(band, u)` is where in a band the card itself may go at row height u
    (so that its shadow stays inside the safe zone); by default, the whole band."""
    room_at = room_at or (lambda band, u: band)
    floor_read = READABLE * H / TEXT
    floor_any = 0.008 * H / TEXT
    for allow, floor in ((0, floor_read), (1, floor_read), (2, floor_read), (2, floor_any)):
        fits: dict[str, Place] = {}
        for where, band in room.items():
            u = min(U_MAX * H, band.h)
            while u >= floor:
                got = measure(u, room_at(band, u), allow)
                if got is not None:
                    w, h, cut = got
                    fits[where] = Place(where, band, u, w, h, 0.0, 0.0, cut)
                    break
                u -= max(0.5, u * 0.012)
        if not fits:
            continue
        pick = None
        if want != "auto":
            pick = fits.get(want)
            if pick is None:
                notes.append(f"there is no room {'beside' if want == 'side' else want} the head for the list, so it "
                             f"goes where there is most room")
        if pick is None:
            best = max(p.u for p in fits.values())
            for where in ("top", "side", "below"):
                if where in fits and fits[where].u >= 0.9 * best:
                    pick = fits[where]
                    break
        band = room_at(pick.band, pick.u)
        cx = head.cx if (head is not None and pick.where != "side") else band.cx
        if pick.where == "side" and head is not None:
            cx = band.x0 + pick.w / 2 if band.x0 >= head.cx else band.x1 - pick.w / 2
        pick.x = min(max(cx - pick.w / 2, band.x0), band.x1 - pick.w)
        pick.y = band.y0 + (band.h - pick.h) / 2
        if pick.where == "top":  # nearer the head than the top of the frame: it belongs to them
            pick.y = band.y1 - pick.h - min(band.h - pick.h, 0.35 * (band.h - pick.h))
        return pick
    return None


# ---- the vertical card: a board, a countdown, or a checklist ---------------------------------------------


@dataclass
class CardSpec:
    """What a look puts in each row of a vertical card."""
    look: str
    lead: float  # the mark before the words, as a share of u (0 for none)
    trail: float  # a switch after the words (0 for none)
    header: float  # a bar over the rows (0 for none): it never moves as rows are added
    all_rows_first: bool  # every row shown from the start (a board's empty slots, a settings card);
    # otherwise the card grows a row at a time as each is said, the rows above never moving
    number: bool  # the mark is a number
    countdown: bool = False


class Card:
    """A vertical card of rows, laid out once, drawn on each frame."""

    def __init__(self, spec: CardSpec, rows: list[Row], title: str, style: str, accent, fps: float,
                 size: tuple[int, int], zone: SafeZone):
        self.spec, self.rows, self.title, self.style = spec, rows, title, style
        self.p = PALETTES[style]
        self.accent = tuple(accent)
        self.fps = fps
        self.W, self.H = size
        self.zone = zone
        self.u = 0.0
        self.x = self.y = 0.0
        self.w = self.h = 0.0
        self.t_in = 0.0
        self.t_out: Optional[float] = None
        self.keep: list[Optional[Rect]] = []
        self.front = None
        self.pane = None
        self._marks: dict = {}
        self.cut = 0
        self.card_in = CARD_IN  # a card that grows arrives with its first row instead

    # -- layout
    def place(self, room: dict[str, Rect], H: float, want: str, head: Optional[Rect], notes: list[str],
              keep: list[Optional[Rect]], faces: list[Optional[Rect]]) -> Optional[Place]:
        """Where the card goes: the largest it can be in the room above, beside or below the head. A
        card that grows a row at a time above the head is checked frame by frame instead: each height
        only has to be clear of the head while the card is that tall, so a head that rises early, while
        the card holds one row, does not make every row small."""
        base = choose(room, self.measure, H, want, head, notes,
                      lambda band, u: halo_room(band, self.zone.rect, self.halo(u)))
        if not self.grows() or want not in ("auto", "top") or head is None:
            return base
        grown = self._grown_place(H, head, keep, faces, base)
        return grown if grown is not None else base

    def _grown_place(self, H: float, head: Rect, keep: list[Optional[Rect]], faces: list[Optional[Rect]],
                     base: Optional[Place]) -> Optional[Place]:
        zone = self.zone.rect
        j0 = frame_of(self.t_in, self.fps)
        lands = sorted(r.land for r in self.rows)
        shown = []
        for i in range(len(keep)):
            t = (j0 + i) / self.fps
            k = max(1, sum(1 for land in lands if t >= land - ENTER))
            shown.append(k)
        floor = max(READABLE * H / TEXT, base.u * 1.05 if base is not None and base.where == "top" else 0.0)
        gap = ROOM_GAP * H
        tolerance = 0.04 if self.front is not None else 0.0
        # every size with the words as said first; shorter words only if the room had needed them anyway
        for allow in ((0,) if base is not None and base.cut == 0 else (0, 1)):
            u = U_MAX * H
            while u >= floor:
                room = halo_room(zone, zone, self.halo(u))
                got = self.measure(u, room, allow)
                if got is None:
                    u -= max(0.5, u * 0.012)
                    continue
                w, h, cut = got
                x = min(max(head.cx - w / 2, room.x0), room.x1 - w)
                y = room.y0
                heights = {k: self.height(u, k) for k in set(shown)}
                into_head = 0
                ok = True
                for i, k in enumerate(shown):
                    r = Rect(x, y, x + w, y + heights[k])
                    f = faces[i] if i < len(faces) else None
                    hd = keep[i] if i < len(keep) else None
                    if f is not None and r.intersects(f.grow(gap, 2 * gap, gap, gap)):
                        ok = False
                        break
                    if hd is not None and r.intersects(hd.grow(0, gap, 0, 0)):
                        into_head += 1
                if ok and into_head <= tolerance * len(shown):
                    return Place("top", Rect(zone.x0, zone.y0, zone.x1, y + h), u, w, h, x, y, cut)
                u -= max(0.5, u * 0.012)
        return None

    def halo(self, u: float) -> tuple[float, float, float, float]:
        """How far the card's drawing reaches past its box at row height u, in pixels (left, top,
        right, bottom): the whole canvas of its shadow, or of a glass pane's. The card is placed so
        that this, and not only the box, stays inside the safe zone."""
        if self.style == "glass":
            from ..core import glass as glassmod

            return glassmod.Pane.reach(max(0.5, self.W / 1080))
        if self.style in ("solid", "paper"):
            m = float(math.ceil(SHADOW * u))
            return m, m, m, m
        return 0.0, 0.0, 0.0, 0.0

    def measure(self, u: float, band: Rect, allow: int) -> Optional[tuple[float, float, int]]:
        """The card at row height u inside `band`: (width, height, cut), or None when it does not fit.
        Rows are fitted to the room left for words; nothing is drawn."""
        s = self.spec
        pad, gap = PAD * u, GAP * u
        lead = s.lead * u
        lead_gap = 0.26 * u if lead else 0.0
        trail = (s.trail * u + 0.34 * u) if s.trail else 0.0
        n = len(self.rows)
        h = self.height(u, n)
        if h > band.h + 0.5:
            return None
        room = band.w - 2 * pad - lead - lead_gap - trail
        if room < 1.6 * u:
            return None
        px = max(4, int(TEXT * u))
        widths, worst = [], 0
        for r in self.rows:
            got = fit_words(r.said, room, px, self.p.face_text, allow)
            if got is None:
                return None
            text, p, cut = got
            widths.append(textmod.advance(text, self.p.face_text, p))
            worst = max(worst, cut)
        if self.title:
            tp = int(0.44 * u)
            if textmod.advance(self.title, self.p.face_bold, tp) > band.w - 2 * pad:
                return None
        inner = max(widths + [0.0])
        if self.title:
            inner = max(inner, textmod.advance(self.title, self.p.face_bold, int(0.44 * u)) - lead - lead_gap - trail)
        w = 2 * pad + lead + lead_gap + inner + trail + 0.30 * u
        w = max(w, min(band.w, MIN_SHARE * self.zone.rect.w), 4.6 * u)
        w = min(w, band.w)
        return w, h, worst

    def settle(self, place: Place, allow_cut: int) -> None:
        """Fix the layout at the place chosen: each row's words, size and box."""
        u = place.u
        self.u, self.x, self.y, self.w, self.h = u, float(round(place.x)), float(round(place.y)), place.w, place.h
        s = self.spec
        pad, gap = PAD * u, GAP * u
        lead = s.lead * u
        lead_gap = 0.26 * u if lead else 0.0
        trail = (s.trail * u + 0.34 * u) if s.trail else 0.0
        room = self.w - 2 * pad - lead - lead_gap - trail
        px = max(4, int(TEXT * u))
        y = pad + (TITLE_H * u + gap if self.title else 0.0)
        self.header_y = y
        y += (s.header * u + gap) if s.header else 0.0
        self.text_x = pad + lead + lead_gap
        self.lead_x = pad
        self.trail_x = self.w - pad - s.trail * u
        for r in self.rows:
            got = fit_words(r.said, room, px, self.p.face_text, 2) or (found_items.cut_to(r.said, lambda q: False), px, 2)
            r.text, r.px, r.cut = got
            r.slot = Rect(pad, y, self.w - pad, y + u)
            y += u + gap
        self.cut = max([r.cut for r in self.rows] + [0])

    def height(self, u: float, rows: int) -> float:
        """The card's height at row height u with `rows` rows showing."""
        s = self.spec
        top = (TITLE_H * u + GAP * u if self.title else 0.0) + (s.header * u + GAP * u if s.header else 0.0)
        return 2 * PAD * u + top + rows * u + max(0, rows - 1) * GAP * u

    @property
    def rect(self) -> Rect:
        return Rect(self.x, self.y, self.x + self.w, self.y + self.h)

    def grows(self) -> bool:
        return not self.spec.all_rows_first and len(self.rows) > 1

    def shown_h(self, t: float) -> float:
        """The card's height at t: whole for a board, else as far as the rows said so far."""
        if not self.grows():
            return self.h
        order = sorted(r.land for r in self.rows)
        k = sum(1 for land in order if t >= land)
        h = self.height(self.u, max(1, k))
        if k < len(order) and t >= order[k] - ENTER and k >= 1:  # the next row coming: the card opens for it
            q = ease_out((t - (order[k] - ENTER)) / ENTER)
            h += (self.height(self.u, k + 1) - h) * q
        return min(self.h, h)

    # -- the pieces, drawn once
    def build(self, frames_behind: list[np.ndarray]) -> None:
        u, p = self.u, self.p
        self._panels: dict[int, tuple[Sprite, int]] = {}
        self._panes: dict = {}
        self.pane = None
        if self.style == "glass":
            from ..core import glass as glassmod

            self.pane = glassmod.Pane(int(math.ceil(self.w)), int(math.ceil(self.h)), radius=0.30 * u,
                                      unit=max(0.5, self.W / 1080))
            self.pane.tone_over(frames_behind, self.x, self.y)
            self._panes[int(math.ceil(self.h))] = self.pane
        self.panel, self.panel_m = self.panel_at(self.h)
        for r in self.rows:
            r.sprite = words_sprite(r.text, p.face_text, r.px, p.ink)
            r.dim_sprite = words_sprite(r.text, p.face_text, r.px, p.ink, 0.42)

    def panel_at(self, h: float) -> tuple[Sprite, int]:
        """The card's own face at height h (kept for each height): its fill, edge, shadow and title."""
        from PIL import Image, ImageDraw

        key = int(round(h / 2.0)) * 2
        if key in self._panels:
            return self._panels[key]
        u, p = self.u, self.p
        h = float(max(key, 2))
        radius = 0.30 * u
        W, H = int(math.ceil(self.w)), int(math.ceil(h))
        panel = rgba(W, H)
        if self.style == "paper":
            edge = max(1.0, 0.03 * u)
            round_rect(panel, (0, 0, self.w, h), 0.16 * u, p.ink + (60,))
            round_rect(panel, (edge, edge, self.w - edge, h - edge), 0.16 * u - edge, p.card + (255,))
            a = np.asarray(panel).astype(np.float32)
            grain = np.random.default_rng(5).normal(0.0, 2.2, a.shape[:2])[..., None]
            a[..., :3] = np.clip(a[..., :3] + grain * (a[..., 3:] > 0), 0, 255)
            panel = Image.fromarray(a.astype(np.uint8), "RGBA")
        elif self.style == "solid":
            round_rect(panel, (0, 0, self.w, h), radius, p.card + (int(255 * p.card_alpha),))
        draw = ImageDraw.Draw(panel)
        if self.title:
            tp = int(0.44 * u)
            cap = textmod.cap_height(p.face_bold, tp)
            ty = PAD * u + (TITLE_H * u + cap) / 2 - 0.04 * u
            draw.text((PAD * u, ty), self.title, font=textmod.font(p.face_bold, tp), fill=p.ink + (255,), anchor="ls")
            ly = PAD * u + TITLE_H * u + GAP * u * 0.5
            draw.line((PAD * u, ly, self.w - PAD * u, ly), fill=p.line + (int(255 * p.line_alpha),),
                      width=max(1, int(round(0.03 * u))))
        m = 0
        if self.style in ("solid", "paper"):
            m = int(math.ceil(SHADOW * u))
            panel = shadowed(panel, 0.22 * u, 0.07 * u, 0.40 if self.style == "solid" else 0.24, m)
        out = (Sprite.from_rgba(np.asarray(panel)), m)
        self._panels[key] = out
        return out

    def pane_at(self, h: float):
        key = int(round(h / 2.0)) * 2
        if key not in self._panes:
            from ..core import glass as glassmod

            pane = glassmod.Pane(int(math.ceil(self.w)), max(2, key), radius=0.30 * self.u, unit=max(0.5, self.W / 1080))
            pane.tone = self.pane.tone
            self._panes[key] = pane
        return self._panes[key]

    # -- each frame
    def visible(self, t: float) -> bool:
        return t >= self.t_in and (self.t_out is None or t < self.t_out + CARD_OUT)

    def card_alpha(self, t: float) -> tuple[float, float]:
        """(opacity, scale) of the whole card at t."""
        a = ease_out((t - self.t_in + 1.0 / self.fps) / self.card_in)
        s = 0.95 + 0.05 * a
        if self.t_out is not None and t >= self.t_out:
            q = ease_in_out((t - self.t_out) / CARD_OUT)
            a *= 1.0 - q
            s *= 1.0 - 0.04 * q
        return a, s

    def row_state(self, r: Row, t: float) -> tuple[float, float]:
        """(opacity, how far it still has to come, 0 at rest) of a row's words at t: still and whole
        from the first frame of its word."""
        if t >= r.land:
            return 1.0, 0.0
        q = (t - (r.land - ENTER)) / ENTER
        if q <= 0:
            return 0.0, 1.0
        e = ease_out(q)
        return e, 1.0 - e

    def draw(self, frame: np.ndarray, j: int, t: float) -> None:
        if not self.visible(t):
            return
        head = self._head_in(j)
        original = frame.copy() if head is not None else None
        self._draw(frame, t)
        if head is not None and original is not None:
            self._person_back(frame, original, j, head)

    def _draw(self, frame: np.ndarray, t: float) -> None:
        a, s = self.card_alpha(t)
        if a <= 0:
            return
        h = self.shown_h(t)
        if self.pane is not None:
            pane = self.pane if h >= self.h - 1 else self.pane_at(h)
            if abs(s - 1.0) > 1e-3:
                pane.draw(frame, self.x + self.w * (1 - s) / 2, self.y + h * (1 - s) / 2, opacity=a,
                          frost=min(1.0, a * 1.2), scale=s)
            else:
                pane.draw(frame, self.x, self.y, opacity=a, frost=min(1.0, a * 1.2))
        panel, m = (self.panel, self.panel_m) if h >= self.h - 1 else self.panel_at(h)
        if abs(s - 1.0) > 1e-3 or a < 0.999:
            sp = panel.scaled(s)
            cx, cy = self.x + self.w / 2, self.y + h / 2
            over(frame, sp, cx - sp.w / 2, cy - sp.h / 2, a)
        else:
            over(frame, panel, self.x - m, self.y - m)
        X, Y = self.x, self.y
        if self.spec.header:
            self._draw_header(frame, t, X, Y, a)
        for k, r in enumerate(self.rows):
            self._draw_row(frame, k, r, t, X, Y, a)

    # the look's own parts: overridden by ranked-list and checklist
    def _draw_row(self, frame, k: int, r: Row, t: float, X: float, Y: float, card_a: float) -> None:
        raise NotImplementedError

    def _draw_header(self, frame, t: float, X: float, Y: float, card_a: float) -> None:
        return None

    def _words(self, frame, r: Row, t: float, X: float, Y: float, alpha: float, dim: bool = False,
               nudge: float = 0.0) -> None:
        sp, ox, oy = r.dim_sprite if dim else r.sprite
        cap = textmod.cap_height(self.p.face_text, r.px)
        bx = X + self.text_x + nudge
        by = Y + r.slot.y0 + (r.slot.h + cap) / 2
        over(frame, sp, bx - ox, by - oy, alpha)

    def mark(self, kind: str, d: float, share: float = 1.0, angle: float = 0.0, accent=None) -> Sprite:
        key = (kind, round(d, 1), round(share, 2), int(angle) % 360, accent)
        if key not in self._marks:
            self._marks[key] = mark_sprite(d, kind, accent or self.accent, self.p, share, angle)
        return self._marks[key]

    # -- the person in front where the hair reaches into the card
    def _head_in(self, j: int) -> Optional[Rect]:
        if self.front is None or j >= len(self.keep):
            return None
        head = self.keep[j]
        return head if head is not None and head.intersects(self.rect.grow(*(4 * [0.3 * self.u]))) else None

    def _person_back(self, frame: np.ndarray, original: np.ndarray, j: int, head: Rect) -> None:
        u = self.u
        H, W = frame.shape[:2]
        r = self.rect.grow(*(4 * [0.6 * u]))
        x0, y0, x1, y1 = max(0, int(r.x0)), max(0, int(r.y0)), min(W, int(r.x1) + 1), min(H, int(r.y1) + 1)
        if x1 <= x0 or y1 <= y0:
            return
        pm = self.front.alpha(j, original)[y0:y1, x0:x1]
        hb = head.grow(0.12 * head.w, 0.12 * head.h, 0.12 * head.w, 0.04 * head.h)
        feather = max(2.0, 0.25 * u)
        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        wx = np.clip(1.0 - np.maximum(hb.x0 - xs, xs - hb.x1) / feather, 0.0, 1.0)
        wy = np.clip(1.0 - np.maximum(hb.y0 - ys, ys - hb.y1) / feather, 0.0, 1.0)
        a = (pm * wy[:, None] * wx[None, :])[..., None]
        drawn = frame[y0:y1, x0:x1].astype(np.float32)
        frame[y0:y1, x0:x1] = np.clip(drawn * (1.0 - a) + original[y0:y1, x0:x1].astype(np.float32) * a + 0.5,
                                      0, 255).astype(np.uint8)

    # -- what is checked
    def rects(self, t: float) -> list[Rect]:
        if not self.visible(t):
            return []
        a, s = self.card_alpha(t)
        if a <= 0.02:
            return []
        h = self.shown_h(t)
        if s >= 1.0:
            return [Rect(self.x, self.y, self.x + self.w, self.y + h)]
        cx, cy = self.x + self.w / 2, self.y + h / 2
        return [Rect(cx - self.w * s / 2, cy - h * s / 2, cx + self.w * s / 2, cy + h * s / 2)]

    def drawn(self, t: float) -> list[Rect]:
        """What is drawn at t, its shadow included: what the safe zone is checked against."""
        left, top, right, bottom = self.halo(self.u)
        return [r.grow(left, top, right, bottom) for r in self.rects(t)]

    def cues(self) -> list[tuple[float, str, float, float]]:
        """The sounds: (when, which, pitch, dB under the voice)."""
        return []


# ---- the sound ---------------------------------------------------------------------------------------------


def mix_cues(cues: list[tuple[float, str, float, float]], clip, sound_in: np.ndarray, rate: int) -> np.ndarray:
    """Each cue's sound laid so that it lands on its moment, to the sample, `level` dB under the voice's
    own loudness (every sound measured the same way), its pitch stepped as asked. The pack's own sounds."""
    from ..core import sound as soundmod
    from . import sound_effects as sfx

    if not cues:
        return sound_in
    lib = sfx.library()
    voice = soundmod.integrated(soundmod.blocks(clip.sound)) if clip.info.has_audio else soundmod.SILENCE_DB
    reference = voice if voice > -70 else -23.0
    out = sound_in.copy()
    stem = np.zeros_like(out)
    for at, kind, pitch, under in cues:
        s = lib[kind]
        x, anchor = s.samples, s.anchor
        if abs(pitch - 1.0) > 1e-4:
            x = soundmod.resample(x, pitch)
            anchor = anchor / pitch
        gain = soundmod.gain((reference - under) - s.loudest)
        soundmod.lay(stem, x, at, rate, anchor=anchor, level=gain)
    out = out + stem
    peak = float(np.abs(out).max()) if len(out) else 0.0
    if peak > 0.98:
        out *= 0.98 / peak
    return out.astype(np.float32)


# ---- the frames each check counts --------------------------------------------------------------------------


def check(rects_at: Callable[[float], list[Rect]], keep: list[Optional[Rect]], faces: list[Optional[Rect]],
          zone: Rect, fps: float, n: int, front: bool,
          drawn_at: Optional[Callable[[float], list[Rect]]] = None) -> dict:
    """Frame by frame: anything drawn over the face box, or over the head with nobody laid back over
    it (both must be 0); how often the head passes in front; anything outside the safe zone (0), the
    shadows counted (`drawn_at`, the drawing with its shadow, when it reaches past `rects_at`)."""
    over_face = behind = outside = 0
    for j in range(n):
        rects = rects_at(j / fps)
        if not rects:
            continue
        drawn = drawn_at(j / fps) if drawn_at is not None else rects
        head, face = keep[j] if j < len(keep) else None, faces[j] if j < len(faces) else None
        on_face = face is not None and any(r.intersects(face) for r in rects)
        on_head = head is not None and any(r.intersects(head) for r in rects)
        if on_face or (on_head and not front):
            over_face += 1
        elif on_head:
            behind += 1
        if any(not r.inside(zone) for r in drawn):
            outside += 1
    return {"frames_over_face": over_face, "frames_head_in_front": behind, "frames_outside_safe_zone": outside}


def landing_report(rows: list[Row], words: list[Word], fps: float) -> str:
    """How many rows land on the first frame of their word (a given time with no word near counts as given)."""
    on = sum(1 for r in rows if r.frame == frame_of(r.at, fps))
    return f"{on} of {len(rows)} rows still and whole on the first frame of their word"


def stills(rows: list[Row], t_in: float, end: float, hold: float, kept: float, fps: float,
           extra: list[float] = (), empty_first: bool = True) -> list[float]:
    """Six moments that show the list: empty (or its first row, for a list that grows), rows at rest
    across the run, the done marks (`extra`), and the finished list held."""
    first = min(end, t_in + CARD_IN + 0.15)
    if rows and empty_first:  # the list before anything arrives, never a row half way in
        first = max(t_in + 1.0 / fps, min(first, min(r.land for r in rows) - ENTER - 1.0 / fps))
    elif rows:
        first = min(end, min(r.land for r in rows) + 0.10)
    rest = [min(end, r.land + 0.30) for r in rows]
    picks = [first]
    if rest:
        picks += [rest[round(k * (len(rest) - 1) / 3)] for k in range(4)] if len(rest) >= 4 else rest
    picks += list(extra)
    held = kept + hold / 2 if hold > 0 else end
    picks.append(min(end, held))
    out = sorted(set(round(min(end, max(0.0, p)), 3) for p in picks))
    while len(out) > 6:  # keep the first and last, thin the middle evenly
        mid = out[1:-1]
        drop = min(range(len(mid)), key=lambda i: (mid[i + 1] - mid[i - 1]) if 0 < i < len(mid) - 1 else 99)
        out = [out[0]] + [m for i, m in enumerate(mid) if i != drop] + [out[-1]]
    return out


# ---- one run, from the words to a job ------------------------------------------------------------------------


def accent_of(opts: dict, style: str, default: dict) -> tuple[int, int, int]:
    value = opts.get("accent", "auto")
    if value and value != "auto":
        return textmod.colour(value)
    return textmod.hex_rgb(default[style])


def rows_from_plan(plan: dict, words: list[Word], fps: float, duration: float) -> tuple[list[Row], str]:
    from . import PlanError

    raw = plan.get("rows")
    if not isinstance(raw, list) or not raw:
        raise PlanError("the plan needs its rows; run once without --plan to get one to edit")
    rows = []
    for k, r in enumerate(raw):
        said = str(r.get("said") or r.get("text") or "").strip()
        if not said:
            raise PlanError(f"row {k + 1} has no words")
        try:
            at = float(r["at"])
        except (KeyError, TypeError, ValueError):
            raise PlanError(f"row {k + 1} ({said!r}) needs a time in seconds") from None
        if not 0 <= at <= duration:
            raise PlanError(f"row {k + 1} ({said!r}) has a time outside the clip ({at})")
        near = min(words, key=lambda w: abs(w.start - at), default=None)
        word = near.clean if near is not None and abs(near.start - at) <= 0.25 else ""
        f = frame_of(at, fps)
        row = Row(said, at, f, f / fps, word, "plan", int(r.get("number") or 0), r.get("image") or None,
                  r.get("marker_at"), r.get("ends"))
        if r.get("text") and r.get("text") != said:
            row.said = str(r["text"])  # words changed in the plan are shown as written
        if r.get("done") is not None:
            row.done = float(r["done"])
            row.done_how = "given"
        rows.append(row)
    return rows, "the plan given"


def prepare_list(clip, opts: dict, plan: Optional[dict], *, name: str, card: Card, most: int,
                 countdown: bool = False, steps: bool = False, stills_extra: Callable[[list[Row]], list[float]] = None,
                 measure_place=None):
    """Everything decided before a frame is drawn, for a vertical card (`card`, with its rows still
    empty): the rows, their numbers and ticks, the room, the size, the checks and the plan."""
    from ..core import framing as framing_mod
    from ..core import video
    from ..core.timeline import Timeline
    from . import Job, PlanError
    from .tier_list import PersonFront, bands, head_track

    info = clip.info
    fps = info.rate
    fr = framing_mod.plan(info.width, info.height, fps, info.frames, clip.face, shape=opts["frame"],
                          size=_size_of(opts["size"]))
    W, H = fr.out_w, fr.out_h
    duration = info.frames / fps
    words = clip.words
    notes: list[str] = []
    try:
        if plan is not None:
            rows, source = rows_from_plan(plan, words, fps, duration)
            heard = []
        else:
            given = parse_items(opts["items"], Path(clip.path).parent, most)
            rows, heard, source = rows_of(words, given, fps, duration, notes, clip.quiet)
    except found_items.ItemsError as e:
        raise PlanError(str(e)) from None
    if not rows:
        raise PlanError("no list was heard in the words (such as \"number one …\", \"first …, then …\", \"step two …\" "
                        "or \"A, B and C\"). Say it that way, or give the items: items=\"Lighting; Sound; Framing\"")
    if len(rows) > most:
        notes.append(f"{len(rows)} items were heard; the first {most} are shown")
        rows = rows[:most]
    number_rows(rows, countdown, notes)
    tl = Timeline.whole(duration).held(opts["hold"])
    n_out = tl.output_frames(fps)
    if steps:
        done_times(rows, words, fps, duration, duration)
    card.rows = rows
    card.fps = fps
    card.W, card.H = W, H
    zone = SafeZone(W, H)
    card.zone = zone
    first = min(r.land for r in rows)
    if card.grows():  # the card and its first row arrive together, both whole on the first frame of the word
        card.card_in = ENTER
        card.t_in = max(0.0, first - ENTER)
    else:
        lead = 0.9 if card.spec.all_rows_first and not card.spec.number else CARD_LEAD
        card.t_in = max(0.0, first - ENTER - lead)
    last = max([r.land for r in rows] + [r.done or 0.0 for r in rows])
    leave = float(opts.get("leave", 0) or 0)
    card.t_out = last + leave if leave > 0 and last + leave + CARD_OUT < tl.kept else None
    keep, faces = head_track(clip, tl, fr, notes)
    j_in = frame_of(card.t_in, fps)
    j_out = n_out if card.t_out is None else min(n_out, frame_of(card.t_out + CARD_OUT, fps) + 1)
    head = Rect.around(r for r in keep[j_in:j_out] if r is not None)
    front = None
    if head is not None:
        try:
            front = PersonFront(clip, tl, fr, (W, H))
        except Exception as e:  # no mask: the card keeps clear of the head in every frame instead
            notes.append(f"the person could not be cut out ({e}); the list keeps clear of the head in every frame")
    room = bands(zone.rect, keep[j_in:j_out], faces[j_in:j_out], W, H, behind=front is not None)
    card.keep, card.front = keep, front
    place = card.place(room, H, opts["position"], head, notes, keep[j_in:j_out], faces[j_in:j_out])
    if place is None:
        raise PlanError("there is no room anywhere off the face for the list on this clip; a shot with more room "
                        "above the head, fewer items or shorter ones make room for it")
    card.settle(place, 2)
    frames_behind = []
    if card.style == "glass":
        for t in (card.t_in + 0.2, (card.t_in + tl.kept) / 2, tl.kept - 0.2):
            k = int(min(info.frames - 1, max(0, round(t * fps))))
            frames_behind.append(fr.apply(video.read_frame(info, k), k))
    card.build(frames_behind)
    card.keep, card.front = keep, front
    checks = check(card.rects, keep, faces, zone.rect, fps, n_out, front is not None, drawn_at=card.drawn)
    text_px = max(r.px for r in rows)
    if text_px < READABLE * H:
        notes.append(f"the room off the face is small on this clip, so the words are {text_px} px "
                     f"({text_px / H:.1%} of the height, under the {READABLE:.2%} that reads easily on a phone); "
                     "a shot with more room above the head, or fewer items, makes them clearer")
    shortened = [r for r in rows if r.cut]
    if shortened:
        notes.append("made shorter to fit the room off the face: " + "; ".join(
            f"{r.said!r} as {r.text!r}" for r in shortened) + " (shorter items, or position=top, keep them whole)")
    hows = {}
    for r in rows:
        hows[r.how] = hows.get(r.how, 0) + 1
    report = {
        "list": f"{len(rows)} rows from {source}: " + ", ".join(f"{n} {k}" for k, n in hows.items()),
        "card": f"{place.where}, {card.w:.0f}x{card.h:.0f} px ({card.h / H:.0%} of the height), rows {card.u:.0f} px, "
                f"words {text_px} px",
        "lands": " | ".join(f"{r.number}. {r.text} {r.land:.2f} s (frame {r.frame}" + (f", '{r.word}'" if r.word else "")
                            + ")" for r in rows),
        "on_the_word": landing_report(rows, words, fps),
        **({"ticks": " | ".join(f"{r.number}. {r.done:.2f} s ({r.done_how}" + (f", '{r.done_word}'" if r.done_word
                                                                                  and r.done_how == 'said done' else "")
                                  + ")" for r in rows)} if steps else {}),
        **checks,
        "placed": place.where,
        "warnings": notes,
    }
    sheet = stills(rows, card.t_in, tl.duration - 0.5 / fps, tl.hold, tl.kept, fps,
                   stills_extra(rows) if stills_extra else [], empty_first=not card.grows())
    out_plan = {
        "effect": name, "clip": clip.path, "options": opts,
        "output": {"width": W, "height": H, "fps": round(fps, 3), "seconds": round(tl.duration, 3),
                   "framing": fr.note, "hold": tl.hold},
        "list_from": source,
        "board": {"in": round(card.t_in, 3), "out": None if card.t_out is None else round(card.t_out, 3),
                 "position": place.where, "rect": card.rect.as_list(), "row_px": round(card.u, 1), "words_px": text_px},
        "rows": [{"number": r.number, "text": r.text, "said": r.said, "at": round(r.at, 3), "frame": r.frame,
                  "lands": round(r.land, 4), "word": r.word, "how": r.how}
                 | ({"image": r.image} if r.image else {})
                 | ({"asked": r.asked} if r.asked is not None and abs(r.asked - r.at) > 1e-3 else {})
                 | ({"done": round(r.done, 4), "done_frame": r.done_frame, "done_how": r.done_how}
                    if steps and r.done is not None else {})
                 | ({"marker_at": round(r.marker_at, 3)} if r.marker_at is not None else {})
                 | ({"ends": round(r.ends, 3)} if r.ends is not None else {})
                 for r in rows],
        "sheet_times": sheet,
    }
    picks = [sheet[0], sheet[len(sheet) // 2], sheet[-1]]
    cues = card.cues()
    sound = (lambda s, rate: mix_cues(cues, clip, s, rate)) if opts["sound"] and cues else None
    out_plan["sounds"] = [{"at": round(a, 4), "sound": k, "pitch": round(p, 4), "under_voice_db": u} for a, k, p, u in cues] \
        if opts["sound"] else []

    def after(result: dict, out: Path) -> dict:
        try:
            return measure_file(out, clip, card, cues if opts["sound"] else [])
        except Exception as e:  # a measurement that cannot be made never fails a finished render
            return {"measured": f"not measured ({e})"}

    return Job(name, clip, tl, fr, card.draw, out_plan, checks=report, still_times=lambda: picks, sound=sound,
               after=after)


def measure_file(out: Path, clip, card: Card, cues: list[tuple[float, str, float, float]]) -> dict:
    """What the finished file shows: on which frame each row stops moving (read from the picture in
    its box), and where each sound lands and how far under the voice it sits (the file's sound
    against the clip's own, the difference found and timed)."""
    from ..core import sound as soundmod
    from ..core import video
    from . import sound_effects as sfx

    lines: dict = {}
    info = video.probe(str(out))
    fps = info.rate
    # the rows: still from the first frame of their word (a vertical card's words, where they stand)
    still_on, checked = 0, 0
    flat = card.style != "glass" and hasattr(card, "text_x") and card.__class__.__name__ not in ("Rapid", "Arc")
    for r in card.rows if flat else []:
        if r.slot is None or r.frame < 3 or r.frame + 3 >= info.frames:
            continue
        x0 = int(card.x + card.text_x)
        x1 = int(min(info.width, x0 + max(8.0, textmod.advance(r.text, card.p.face_text, r.px))))
        y0 = int(card.y + r.slot.y0)
        y1 = int(min(info.height, y0 + r.slot.h))
        crops = []
        for f in video.FrameReader(info, start=max(0.0, (r.frame - 2 - 0.25) / fps), duration=5.5 / fps):
            crops.append(f[y0:y1, x0:x1].astype(np.float32))
            if len(crops) == 5:
                break
        if len(crops) < 5:
            continue
        moving = [float(np.abs(crops[k + 1] - crops[k]).mean()) for k in range(4)]
        checked += 1
        # moving into its frame, then nothing changes there but what shows through the card
        if moving[1] > 2.0 * max(0.5, moving[2]) and moving[2] <= 1.5 and moving[3] <= 1.5:
            still_on += 1
    if checked:
        lines["rows in the file"] = f"{still_on} of {checked} rows stop moving on the first frame of their word"
    # the sounds: each found again by its first 0.15 s, within 80 ms of its frame
    if cues:
        rate = soundmod.AUDIO_RATE
        y = soundmod.read(out)
        x = clip.sound
        if len(x) < len(y):
            x = np.concatenate([x, np.zeros((len(y) - len(x), x.shape[1]), np.float32)])
        n = min(len(x), len(y))
        res = (y[:n] - x[:n]).mean(axis=1)
        voice = soundmod.integrated(soundmod.blocks(clip.sound)) if clip.info.has_audio else soundmod.SILENCE_DB
        lib = sfx.library()
        offs, unders = [], []
        for at, kind, pitch, under in cues:
            smp = lib[kind]
            full = smp.samples.mean(axis=1)
            anchor = smp.anchor
            if abs(pitch - 1.0) > 1e-4:
                full = soundmod.resample(full[:, None], pitch)[:, 0]
                anchor /= pitch
            k0 = int(round(anchor * rate))
            tmpl = full[max(0, k0 - int(0.02 * rate)):k0 + int(0.13 * rate)]
            lead = k0 - max(0, k0 - int(0.02 * rate))
            a = int(round((at - 0.08) * rate)) - lead
            b = int(round((at + 0.08) * rate)) - lead + len(tmpl)
            if a < 0 or b > len(res) or len(tmpl) < 16:
                continue
            xc = np.correlate(res[a:b], tmpl, mode="valid")
            if not len(xc):
                continue
            k = int(np.argmax(np.abs(xc)))
            found = (a + k + lead) / rate
            offs.append((found - at) * 1000.0)
            piece = res[a + k:a + k + len(tmpl)]
            unders.append(voice - soundmod.loudest(np.stack([piece, piece], axis=1)))
        if offs:
            lines["sounds in the file"] = (f"{len(offs)} of {len(cues)} found; each lands {min(offs):+.1f} to "
                                           f"{max(offs):+.1f} ms from its frame (one frame is {1000 / fps:.1f} ms)")
        if unders:
            lines["under the voice"] = f"{min(unders):.1f} to {max(unders):.1f} dB"
    return lines


def _size_of(value: str):
    return None if value == "auto" else tuple(int(x) for x in value.split("x"))
