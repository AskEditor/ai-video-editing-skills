"""before-after: one wipe across the playing clip, the clip as shot on one side and the after on the other.

How it is laid:

1. **The after** is either one of `grade`'s looks on the same clip (with that look's own vignette and
   grain), or a second video: an edit of the same take. A second video is read at the clip's frame
   rate, filled to the clip's frame without stretching, and lined up by its sound when its sound
   says how (an export that starts a few frames late is shifted back into step); the run reports it.
2. **The wipe:** at `at`, a divider travels in `direction` over `duration` seconds, eased like a hand
   dragging a slider. `hold` stops it halfway and keeps the frame split; `sweep` swings it back and
   forth across the person; `reveal` carries it all the way, so the after takes the frame.
3. **The divider is crisp on any background:** a white line with a soft dark edge either side, drawn
   at sub-pixel positions so it glides rather than steps.
4. **Each label belongs to its picture.** BEFORE is drawn on the clip as shot and AFTER on the after,
   before the two are joined, so the divider uncovers and covers them with their pictures. Both sit
   inside their own half, inside the safe zone with a margin, and clear of the head over every frame
   of the clip: never on the face.
5. The sound is the clip's own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import grade
from . import options as O
from ..core.compose import Sprite, over
from ..core.layout import Rect, SafeZone

NAME = "before-after"
MODES = ("auto", "hold", "sweep", "reveal")
DIRECTIONS = ("left", "right", "up", "down")
DEFAULT_DURATION = {"hold": 1.0, "reveal": 1.0, "sweep": 2.0}
SWING = 0.35  # a sweep swings this far either side of the middle (0.15 to 0.85 of the frame)

# The test matrix's runs (tests/matrix.py): the default on A, B and C; the raw take against its
# captioned edit; a sweep; a reveal upwards.
MATRIX = [
    ("A", "A", {}),
    ("B", "B", {}),
    ("C", "C", {}),
    ("A-edit", "A", {"after": "{lab}/captions/A.mp4", "labels": "RAW|EDIT"}),
    ("B-sweep", "B", {"mode": "sweep", "after": "black-and-white"}),
    ("C-reveal-up", "C", {"mode": "reveal", "direction": "up", "after": "warm"}),
]


# ---- options of this effect's own kinds ------------------------------------------------------

@dataclass(frozen=True)
class After:
    """A look by name, or the path of a video."""
    help: str

    def parse(self, raw: Any) -> str:
        v = str(raw).strip().strip("\"'")
        try:
            return grade.LookName("").parse(v)
        except O.OptionError:
            pass
        p = Path(v).expanduser()
        if p.is_file():
            return str(p.resolve())
        raise O.OptionError(f"give a look ({', '.join(grade.LOOKS)}) or the path of a video that exists (got {raw!r})")


@dataclass(frozen=True)
class Labels:
    """on, off, or the two labels' words as BEFORE|AFTER."""
    help: str

    def parse(self, raw: Any) -> Any:
        v = str(raw).strip().strip("\"'")
        if v.lower() in ("on", "yes", "auto", ""):
            return "BEFORE|AFTER"
        if v.lower() in ("off", "no", "none"):
            return "off"
        parts = [x.strip() for x in v.split("|" if "|" in v else "/")]
        if len(parts) != 2 or not all(parts):
            raise O.OptionError(f"say on, off, or two labels with | or / between them, like RAW|EDIT (got {raw!r})")
        if any(len(x) > 14 for x in parts):
            raise O.OptionError(f"keep each label to 14 letters or fewer, so it reads at a glance (got {raw!r})")
        return f"{parts[0]}|{parts[1]}"


@dataclass(frozen=True)
class Seconds:
    """auto, or seconds."""
    help: str
    lo: float = 0.2
    hi: float = 10.0

    def parse(self, raw: Any) -> Any:
        v = str(raw).strip().lower().rstrip("s")
        if v in ("auto", ""):
            return "auto"
        try:
            x = float(v)
        except ValueError:
            raise O.OptionError(f"say auto or give seconds from {self.lo:g} to {self.hi:g} (got {raw!r})") from None
        if not self.lo <= x <= self.hi:
            raise O.OptionError(f"give seconds from {self.lo:g} to {self.hi:g} (got {raw!r})")
        return x


OPTIONS = {
    "after": After("what the after side shows: a look (moody, black-and-white, warm, cool, pop, film), or the "
                   "path of a video of the same take, such as your edit"),
    "mode": O.Choice(MODES, "auto: hold for a look, sweep for a video (a split held still would cut an edit's own "
                            "captions in half) · hold: the divider stops halfway and the frame stays split · sweep: "
                            "it swings back and forth across you · reveal: it goes all the way and the after takes "
                            "the frame"),
    "direction": O.Choice(DIRECTIONS, "which way the divider travels: left (the after comes in from the right, "
                                      "before on the left once split), right, up or down"),
    "at": grade.Moment("when the wipe starts: seconds (1.5), or words that are said there"),
    "duration": Seconds("how long the divider takes to travel: auto (1 s; 2 s a swing for sweep) or seconds"),
    "labels": Labels("the two labels: on (BEFORE and AFTER), off, or your own words as RAW|EDIT or RAW/EDIT"),
    "fix": O.Text("misheard words as wrong=right, separated by ; (only matters when at is words)"),
}


# ---- the after side -----------------------------------------------------------------------------

class LookAfter:
    """The clip through one of grade's looks, with that look's own finish."""

    def __init__(self, clip, fr, look: str):
        self.look = look
        recipe = grade.RECIPES[look]
        self.graded = grade.GradedFrames(clip.info, fr, grade.lut_file(look, 1.0))
        self.finish = grade.Finish(fr.out_w, fr.out_h, recipe.vignette, recipe.grain,
                                   grade.subject_centre(clip, fr.out_w, fr.out_h))

    def at(self, k: int, j: int) -> np.ndarray:
        g = self.graded.at(k).copy()
        self.finish.apply(g, j)
        return g

    def close(self) -> None:
        self.graded.close()


def _cover(src_w: int, src_h: int, out_w: int, out_h: int) -> tuple[int, int, int, int]:
    """The size to read a picture at so it fills out_w×out_h without stretching, and where to cut it."""
    k = max(out_w / src_w, out_h / src_h)
    w, h = max(out_w, int(math.ceil(src_w * k / 2) * 2)), max(out_h, int(math.ceil(src_h * k / 2) * 2))
    return w, h, (w - out_w) // 2, (h - out_h) // 2


class VideoAfter:
    """A second video, frame for frame with the clip: at the clip's rate, filled to its frame, and
    `shift` seconds later in its own time (a positive shift skips the after video's first seconds)."""

    def __init__(self, path: str, clip, fr, shift: float):
        from ..core import video

        info = video.probe(path)
        rate = clip.info.fps
        self.info = replace(info, fps=rate, frames=max(1, int(round(info.duration * float(rate)))))
        self.w, self.h = fr.out_w, fr.out_h
        self.read_w, self.read_h, self.cx, self.cy = _cover(info.width, info.height, self.w, self.h)
        self.shift = float(shift)
        self.rate = clip.info.rate
        self._iter, self._reader = None, None
        self._n = -1  # the after video's own frame number last read
        self._frame: Optional[np.ndarray] = None

    def _open(self, n: int) -> None:
        from ..core import video

        self.close()
        start = max(0.0, (n - 0.25) / self.rate)
        self._reader = video.FrameReader(self.info, size=(self.read_w, self.read_h), start=start)
        self._iter = iter(self._reader)
        self._n, self._frame = n - 1, None

    def at(self, k: int, j: int) -> np.ndarray:
        n = int(min(self.info.frames - 1, max(0, round(k + self.shift * self.rate))))
        if self._iter is None or n < self._n or n > self._n + 3 or (n == self._n and self._frame is None):
            self._open(n)
        while self._n < n:
            nxt = next(self._iter, None)
            if nxt is None:  # the after video is shorter: hold its last frame
                break
            self._frame, self._n = nxt, self._n + 1
        if self._frame is None:
            from ..core.video import MediaError

            raise MediaError(f"could not read the picture of {self.info.path}")
        self._n = n
        f = self._frame
        return np.ascontiguousarray(f[self.cy:self.cy + self.h, self.cx:self.cx + self.w]).copy()

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
        self._reader, self._iter = None, None


def line_up(clip, path: str) -> dict:
    """How far the second video's sound runs behind the clip's (seconds), found by matching the two
    speech envelopes; and whether that can be trusted."""
    from ..core import audio, video

    info = video.probe(path)
    if not (clip.info.has_audio and info.has_audio):
        return {"shift": 0.0, "trusted": False, "how": "no sound on one of them to line them up by; played as given"}
    hop = audio.HOP

    def env(samples):
        e = audio.envelope(video.mono(samples, video.AUDIO_RATE, 16000), 16000).db
        e = e - np.median(e)
        return np.clip(e, -30.0, None)

    a = env(clip.sound)
    b = env(video.read_audio(info))
    most = int(round(3.0 / hop))
    best, best_lag, scores = -1.0, 0, {}
    for lag in range(-most, most + 1):
        # b shifted by `lag` steps: b[i + lag] against a[i]
        i0, i1 = max(0, -lag), min(len(a), len(b) - lag)
        if i1 - i0 < int(2.0 / hop):
            continue
        x, y = a[i0:i1], b[i0 + lag:i1 + lag]
        sx, sy = x.std(), y.std()
        if sx < 1e-6 or sy < 1e-6:
            continue
        r = float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))
        scores[lag] = r
        if r > best:
            best, best_lag = r, lag
    others = [r for lag, r in scores.items() if abs(lag - best_lag) > int(0.25 / hop)]
    runner_up = max(others) if others else -1.0
    trusted = best >= 0.6 and best - runner_up >= 0.12
    shift = best_lag * hop if trusted else 0.0
    if not trusted:
        how = f"their sound does not match clearly enough to line them up (best {best:.2f}); played as given"
    elif abs(shift) < 0.5 / clip.info.rate:
        how = f"in step: their sound matches with no shift (match {best:.2f})"
    else:
        how = (f"the after video runs {abs(shift):.2f} s {'late' if shift > 0 else 'early'} against the clip; "
               f"shifted into step by its sound (match {best:.2f})")
    return {"shift": round(shift, 3), "trusted": trusted, "match": round(best, 3), "how": how,
            "after_seconds": round(info.duration, 3)}


def edit_marks(clip, fr, after: "VideoAfter", samples: int = 16) -> tuple[list[Rect], str]:
    """Where the after video carries words or graphics of its own (burned-in captions, a title): the
    places it differs sharply from the clip, on frames across it. A grade or a different exposure
    moves every pixel a little and is ignored; an edit that moves the picture itself (a zoom, a
    reframe) changes too much to tell, and says so."""
    import cv2

    from ..core import video

    W, H = fr.out_w, fr.out_h
    q = 4
    w, h = max(8, W // q), max(8, H // q)
    hit = np.zeros((h, w), np.uint8)
    info = clip.info
    for k in np.linspace(0, info.frames - 1, samples).astype(int):
        b = fr.apply(video.read_frame(info, int(k)), int(k))
        a = after.at(int(k), int(k))
        gb = cv2.cvtColor(cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.int16)
        ga = cv2.cvtColor(cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.int16)
        d = np.abs(ga - gb)
        sharp = (d > float(np.median(d)) + 45).astype(np.uint8)
        hit |= cv2.morphologyEx(sharp, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    after.close()
    if hit.mean() > 0.35:
        return [], "the edit changes too much of the frame to tell where its own words are"
    hit = cv2.dilate(hit, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(hit)
    marks = []
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area >= 0.0008 * hit.size:
            marks.append(Rect(x * q, y * q, (x + bw) * q, (y + bh) * q))
    return marks, (f"the edit's own words or graphics found in {len(marks)} place(s); the labels keep off them"
                   if marks else "no words or graphics of the edit's own found")


# ---- the wipe --------------------------------------------------------------------------------

def _smoother(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * x * (x * (6 * x - 15) + 10)


def share(t: float, mode: str, at: float, duration: float) -> float:
    """How much of the frame the after holds at output time `t`, from 0 to 1, from its own side."""
    if t < at:
        return 0.0
    u = (t - at) / duration
    if mode == "hold":
        return 0.5 * _smoother(u)
    if mode == "reveal":
        return _smoother(u)
    top = 0.5 + SWING
    if u <= 1.0:
        return top * _smoother(u)
    return 0.5 + SWING * math.cos(math.pi * (u - 1.0))


@dataclass
class Wipe:
    W: int
    H: int
    direction: str

    @property
    def across(self) -> bool:  # the divider is a vertical line moving across
        return self.direction in ("left", "right")

    def edge(self, p: float) -> float:
        """Where the divider is, in pixels along the axis it travels, for a share `p` of after."""
        size = self.W if self.across else self.H
        return size * (1.0 - p) if self.direction in ("left", "up") else size * p

    def sides(self, p: float) -> tuple[slice, slice]:
        """The (before, after) slices along the travel axis."""
        e = int(round(self.edge(p)))
        if self.direction in ("left", "up"):
            return slice(0, e), slice(e, None)
        return slice(e, None), slice(0, e)

    def region(self, which: str, p: float = 0.5) -> Rect:
        """The before or after side as a box, at share `p` (the split a held wipe rests at)."""
        b, a = self.sides(p)
        s = b if which == "before" else a
        size = self.W if self.across else self.H
        lo, hi = s.start or 0, size if s.stop is None else s.stop
        return Rect(lo, 0, hi, self.H) if self.across else Rect(0, lo, self.W, hi)

    def join(self, frame: np.ndarray, after: np.ndarray, p: float) -> None:
        _, a = self.sides(p)
        if self.across:
            frame[:, a] = after[:, a]
        else:
            frame[a, :] = after[a, :]

    def divider(self, frame: np.ndarray, p: float) -> None:
        """A white line with a soft dark edge either side, at a sub-pixel position."""
        fade = min(1.0, p / 0.03, (1.0 - p) / 0.03)
        if fade <= 0:
            return
        size = self.W if self.across else self.H
        short = min(self.W, self.H)
        e = self.edge(p)
        width = max(2.0, 0.0045 * short)
        sigma = max(1.5, 0.0045 * short)
        reach = int(math.ceil(width / 2 + 3 * sigma)) + 1
        lo, hi = max(0, int(math.floor(e)) - reach), min(size, int(math.ceil(e)) + reach + 1)
        if hi <= lo:
            return
        centre = np.arange(lo, hi, dtype=np.float32) + 0.5 - e
        line = np.clip(width / 2 + 0.5 - np.abs(centre), 0.0, 1.0) * fade
        shade = 0.42 * np.exp(-0.5 * (np.maximum(0.0, np.abs(centre) - width / 2) / sigma) ** 2) * fade
        if self.across:
            strip = frame[:, lo:hi].astype(np.float32)
            strip *= (1.0 - shade)[None, :, None]
            strip = strip * (1.0 - line)[None, :, None] + 255.0 * line[None, :, None]
            frame[:, lo:hi] = np.clip(strip + 0.5, 0, 255).astype(np.uint8)
        else:
            strip = frame[lo:hi, :].astype(np.float32)
            strip *= (1.0 - shade)[:, None, None]
            strip = strip * (1.0 - line)[:, None, None] + 255.0 * line[:, None, None]
            frame[lo:hi, :] = np.clip(strip + 0.5, 0, 255).astype(np.uint8)


# ---- the labels --------------------------------------------------------------------------------

FACE = "poppins-semibold"


def pill(word: str, height_px: float, dark: bool) -> Sprite:
    """One label: capitals spaced a little apart in a rounded pill with a soft shadow under it.
    `dark` is white words on dark glass; otherwise dark words on white."""
    from PIL import Image, ImageDraw, ImageFilter

    from ..core import text

    k = 3  # drawn three times larger, then brought down, so every edge is smooth
    word = word.upper()
    px = int(round(height_px * 0.44 * k))
    f = text.font(FACE, px)
    track = 0.11 * px
    widths = [f.getlength(c) for c in word]
    text_w = sum(widths) + track * (len(word) - 1)
    cap = text.cap_height(FACE, px)
    h = int(round(height_px * k))
    pad = 0.62 * h
    w = int(round(text_w + 2 * pad))
    blur = int(round(0.12 * h))
    margin = 2 * blur + int(0.06 * h)
    W, H = w + 2 * margin, h + 2 * margin
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow = Image.new("L", (W, H), 0)
    ImageDraw.Draw(shadow).rounded_rectangle((margin, margin + int(0.05 * h), margin + w, margin + h + int(0.05 * h)),
                                             radius=h // 2, fill=int(255 * (0.34 if dark else 0.30)))
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
    img.putalpha(shadow)
    body = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(body)
    fill = (14, 14, 16, int(255 * 0.62)) if dark else (250, 250, 248, 245)
    d.rounded_rectangle((margin, margin, margin + w, margin + h), radius=h // 2, fill=fill,
                        outline=(255, 255, 255, int(255 * 0.30)) if dark else None, width=max(1, k))
    ink = (255, 255, 255, 255) if dark else (17, 17, 19, 255)
    x = margin + pad
    base = margin + h / 2 + cap / 2
    for c, cw in zip(word, widths):
        d.text((x, base), c, font=f, fill=ink, anchor="ls")
        x += cw + track
    img = Image.alpha_composite(Image.new("RGBA", (W, H), (0, 0, 0, 0)), img)
    img = Image.alpha_composite(img, body)
    img = img.resize((W // k, H // k), Image.LANCZOS)
    s = Sprite.from_rgba(np.asarray(img))
    s.inset = margin / k  # how far the pill itself sits inside the sprite, every side
    return s


@dataclass
class Label:
    sprite: Sprite
    which: str  # before or after
    x: float = 0.0  # the pill's own top-left (not the sprite's, which has room for the shadow)
    y: float = 0.0
    where: str = ""

    @property
    def w(self) -> float:
        return self.sprite.w - 2 * self.sprite.inset

    @property
    def h(self) -> float:
        return self.sprite.h - 2 * self.sprite.inset

    @property
    def rect(self) -> Rect:
        return Rect(self.x, self.y, self.x + self.w, self.y + self.h)

    def draw(self, frame: np.ndarray) -> None:
        over(frame, self.sprite, self.x - self.sprite.inset, self.y - self.sprite.inset)


def place_labels(labels: list[Label], wipe: Wipe, head: Optional[Rect], zone: SafeZone,
                 marks: Optional[list[Rect]] = None, mode: str = "hold") -> list[str]:
    """Each label inside the part of the frame its picture holds, inside the safe zone with a margin,
    clear of `head` (the head over the whole clip) and of `marks` (an edit's own words and graphics).

    A held or swept split keeps each label in its own half (at the split it rests on). A reveal ends
    with the after over the whole frame, so both labels may sit anywhere: the after's waits where
    the wipe uncovers it. Side by side, both share a line: the top of the zone first, then its foot,
    then the first line down the frame where both fit. Returns warnings."""
    z = zone.rect
    m = 0.022 * wipe.W
    gap = 0.02 * wipe.H
    marks = marks or []

    v = 0.5 * m  # top and foot: the zone's own edge already sits clear of the app's bars
    clear = 0.004 * wipe.H  # the head's box already takes in the hair and the beard

    def fits(r: Rect, side: Rect) -> bool:
        inside = z.grow(-m, -v, -m, -v)
        return (r.inside(inside) and r.inside(side.grow(-m, -v, -m, -v))
                and (head is None or not r.intersects(head, margin=clear))
                and not any(r.intersects(k, margin=gap * 0.5) for k in marks))

    sides = {lb.which: wipe.region(lb.which) for lb in labels}
    on_left = {lb.which: sides[lb.which].cx < wipe.W / 2 for lb in labels}
    if mode == "reveal":
        whole = Rect(0, 0, wipe.W, wipe.H)
        sides = {lb.which: whole for lb in labels}
        if not wipe.across:  # stacked halves have no left and right: before reads first
            on_left = {"before": True, "after": False}
    if wipe.across or mode == "reveal":  # one line for both, each label at the far edge of its half
        def at_line(lb: Label, line: str, y_line: float) -> Rect:
            side = sides[lb.which]
            if on_left[lb.which]:  # against the zone's left edge
                x0 = max(side.x0, z.x0) + m
            else:  # against its right edge
                x0 = min(side.x1, z.x1) - m - lb.w
            if line == "top":
                y0 = z.y0 + v
            elif line == "foot":
                y0 = z.y1 - v - lb.h
            else:
                y0 = y_line
            return Rect(x0, y0, x0 + lb.w, y0 + lb.h)

        tallest = max(lb.h for lb in labels)
        scan = [float(y) for y in np.arange(z.y0 + v, z.y1 - v - tallest, 0.01 * wipe.H)]
        for line, y_line in [("top", 0.0), ("foot", 0.0)] + [(f"{y / wipe.H:.0%} down", y) for y in scan]:
            rects = [at_line(lb, line, y_line) for lb in labels]
            if all(fits(r, sides[lb.which]) for r, lb in zip(rects, labels)):
                for r, lb in zip(rects, labels):
                    lb.x, lb.y, lb.where = r.x0, r.y0, line
                return []
    else:  # stacked halves: the before label high in its half, the after label low in its, same column
        def at_column(lb: Label, column: str) -> Rect:
            side = sides[lb.which]
            if column == "left":
                x0 = z.x0 + m
            elif column == "right":
                x0 = z.x1 - m - lb.w
            else:
                x0 = z.cx - lb.w / 2
            y0 = max(side.y0, z.y0) + v if side.y0 <= wipe.H / 4 else min(side.y1, z.y1) - v - lb.h
            return Rect(x0, y0, x0 + lb.w, y0 + lb.h)

        for column in ("left", "right", "centre"):
            rects = [at_column(lb, column) for lb in labels]
            if all(fits(r, sides[lb.which]) for r, lb in zip(rects, labels)):
                for r, lb in zip(rects, labels):
                    lb.x, lb.y, lb.where = r.x0, r.y0, f"{column}, {'top' if r.cy < wipe.H / 2 else 'foot'}"
                return []
    return ["no place for the labels both clear of the face and inside the safe zone"]


# ---- the effect ---------------------------------------------------------------------------------

def head_union(clip, fr, first: int, last: int) -> Optional[Rect]:
    """The head's box over frames first..last, in output pixels."""
    face = clip.face
    if not face.any():
        return None
    boxes = []
    for i in range(max(0, first), min(face.n - 1, last) + 1):
        h = face.head(i)
        if h is not None:
            boxes.append(fr.to_output(h, i))
    return Rect.around(boxes)


def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from ..core import framing as framing_mod
    from ..core.timeline import Timeline
    from . import Job, PlanError

    info = clip.info
    fr = framing_mod.plan(info.width, info.height, info.rate, info.frames, None, shape="keep")
    duration = info.frames / info.rate
    tl = Timeline.whole(duration)
    W, H = fr.out_w, fr.out_h
    after_is_video = opts["after"] not in grade.LOOKS
    mode, direction = opts["mode"], opts["direction"]
    if mode == "auto":  # a look is compared side by side; an edit's own words read whole as it swings past
        mode = "sweep" if after_is_video else "hold"
    travel = DEFAULT_DURATION[mode] if opts["duration"] == "auto" else float(opts["duration"])
    said = {}
    if isinstance(opts["at"], str) and opts["at"] != "end":
        at, said["at"] = grade._find_words(clip.words, opts["at"])
    else:
        at = 0.0 if opts["at"] == "end" else float(opts["at"])
    warnings: list[str] = []
    room = duration - travel - 0.4
    if at > room:
        new = max(0.0, room)
        if new < at - 1e-3:
            warnings.append(f"the wipe would not finish before the clip ends; it starts at {new:.2f} s instead")
        at = new
    if plan is not None:
        w = plan.get("wipe") or {}
        try:
            at, travel = float(w["at"]), float(w["duration"])
        except (KeyError, TypeError, ValueError):
            raise PlanError("the plan needs a wipe with at and duration in seconds") from None
        if not (0 <= at < duration and 0.2 <= travel <= 10):
            raise PlanError(f"the plan's wipe must start inside the clip's {duration:.2f} s and take 0.2 to 10 s")

    sync, marks, marks_note = None, [], ""
    if after_is_video:
        sync = line_up(clip, opts["after"])
        after = VideoAfter(opts["after"], clip, fr, sync["shift"])
        if opts["labels"] != "off":
            marks, marks_note = edit_marks(clip, fr, after)
        # At clip time t the after shows its own time t + shift: it must reach from `shift` to `duration + shift`.
        short = duration + sync["shift"] - sync.get("after_seconds", duration + sync["shift"])
        if short > 1.0 / info.rate:
            warnings.append(f"the after video ends {short:.2f} s before the clip; its last frame holds")
        if sync["shift"] < 0 and at < -sync["shift"]:
            warnings.append(f"the after video starts {-sync['shift']:.2f} s into the take; its first frame holds "
                            f"until then")
    else:
        after = LookAfter(clip, fr, opts["after"])

    wipe = Wipe(W, H, direction)
    zone = SafeZone(W, H)
    labels: list[Label] = []
    if opts["labels"] != "off":
        words = opts["labels"].split("|")
        pill_h = 0.046 * H
        head = head_union(clip, fr, 0, info.frames - 1)
        for scale in (1.0, 0.85, 0.72):
            labels = [Label(pill(words[0], pill_h * scale, dark=True), "before"),
                      Label(pill(words[1], pill_h * scale, dark=False), "after")]
            problems = place_labels(labels, wipe, head, zone, marks, mode)
            if not problems:
                break
        else:
            warnings += problems
            labels = []  # never on the face: no labels rather than labels over it
        if plan is not None and plan.get("labels"):
            for lb, saved in zip(labels, plan["labels"]):
                lb.x, lb.y = float(saved["rect"][0]), float(saved["rect"][1])

    def draw(frame: np.ndarray, j: int, t: float) -> None:
        p = share(t, mode, at, travel)
        for lb in labels:
            if lb.which == "before":
                lb.draw(frame)
        if p <= 0:
            return
        k = tl.source_frame(j, info.rate, info.frames)
        a = after.at(k, j)
        for lb in labels:
            if lb.which == "after":
                lb.draw(a)
        wipe.join(frame, a, p)
        wipe.divider(frame, p)

    # The report: frames where a label touches the head, or leaves the safe zone, while it shows.
    n = tl.output_frames(info.rate)
    over_face = outside = 0
    has_face = clip.face.any()
    for j in range(n):
        t = j / info.rate
        p = share(t, mode, at, travel)
        k = tl.source_frame(j, info.rate, info.frames)
        h = clip.face.head(k) if has_face else None
        h = fr.to_output(h, k) if h is not None else None
        b_side, a_side = wipe.region("before", p), wipe.region("after", p)
        for lb in labels:
            side = b_side if lb.which == "before" else a_side
            if p <= 0 and lb.which == "after":
                continue
            if lb.rect.overlap(side) <= 0:
                continue
            if h is not None and lb.rect.intersects(h):
                over_face += 1
            if not zone.contains(lb.rect):
                outside += 1
    if after_is_video:
        what = f"the video {Path(opts['after']).name}"
    else:
        what = f"the look {opts['after']}"
    if mode == "sweep":
        u = [at - 0.3, at + 0.5 * travel, at + travel, at + 1.5 * travel, at + 2 * travel, at + 2.5 * travel]
    else:
        u = [at - 0.3, at + 0.3 * travel, at + 0.5 * travel, at + 0.75 * travel, at + travel + 0.6, duration - 0.3]
    sheet = [round(min(duration - 0.05, max(0.0, x)), 3) for x in u]
    out_plan = {"effect": NAME, "clip": clip.path, "options": opts,
                "output": {"width": W, "height": H, "fps": round(info.rate, 3), "seconds": round(tl.duration, 3),
                           "framing": fr.note},
                "after": {"kind": "video" if after_is_video else "look", "what": opts["after"], "sync": sync},
                "wipe": {"mode": mode, "direction": direction, "at": round(at, 3), "duration": round(travel, 3),
                         "said": said},
                "labels": [{"which": lb.which, "rect": lb.rect.as_list(), "where": lb.where} for lb in labels],
                "edit_marks": [r.as_list() for r in marks],
                "sheet_times": sheet}
    checks = {"after": what,
              "wipe": f"{mode} {direction} from {at:.2f} s over {travel:.2f} s"
                      + (f", starting on “{said['at'].strip(' ,.;:!?')}”" if said else ""),
              "labels": (" | ".join(f"{lb.which} at {lb.where}" for lb in labels) if labels else
                         ("off" if opts["labels"] == "off" else "none: no room clear of the face")),
              "frames_over_face": over_face, "frames_outside_safe_zone": outside,
              "warnings": warnings}
    if sync is not None:
        checks["sync"] = sync["how"]
        if marks_note:
            checks["edit_marks"] = marks_note
    else:
        skin = grade.skin_check(clip, grade.RECIPES[opts["after"]], 1.0, {"start": 0.0, "end": duration})
        checks["skin"] = grade._skin_line(skin)
    still = [x for x in sheet[1:5]]

    def finished(result: dict, out: Path) -> dict:
        after.close()
        return {}

    return Job(NAME, clip, tl, fr, draw, out_plan, checks=checks, still_times=lambda: still, after=finished)
