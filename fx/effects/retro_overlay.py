"""retro-overlay: the clip as if seen through something older: a camcorder's viewfinder, an editing
timeline, a tape played back, or film with its burns. All of it drawn here, in code.

The four looks:

- **camcorder:** the viewfinder's corner marks, a red dot that blinks beside REC, a battery, the
  timecode running with the clip (hours, minutes, seconds, frames) and the date the file was
  recorded (or the one typed). The picture is softened and lifted a little, as a small sensor's is.
- **timeline:** a frosted panel under the chin with the clip laid out as an editor shows it: a ruler,
  a video track whose clips are the video's own shots (each cut where the video cuts, filled with
  its own frames), the sound's own waveform under it, and a playhead that moves with the real time.
- **vhs:** colour that bleeds sideways and splits a little at the edges, soft scanlines, a tracking
  band that drifts up the picture, a tape's tone, and PLAY with a tape counter.
- **film:** a warm film tone, grain and a slow gate weave, specks of dust and a passing scratch, the
  rounded corners of a projected frame, and a film burn flaring at the start and the end.

What it keeps:

- **Never over the face.** The readouts sit in the top and bottom bands of the safe zone, each moved
  to the other band when the head would reach it; the timeline panel goes below the chin, else above
  the head; the burns bloom from the edges and fade where the face is; dust and scratches land off it.
- **The safe zone.** Words and readouts stay inside it; only the viewfinder's corner marks and the
  film's corners reach the margins, as pictures do.
- **The colour is the grade's own maths** (`grade.py`, run on recipes of this effect's own), so
  skin keeps its colour under the tape's and the film's tone.
- **The sound is kept,** with a quiet sound of the machine under it (a camcorder's beep as REC
  starts, a tape deck's whirr, a projector's clatter, a slide as the timeline rises): `sound=off`
  leaves it untouched.
- **Captions** (`captions=on`) are drawn over the overlay, and the readouts and the panel make room
  for them.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from ..core import sound as soundmod
from ..core import synth
from ..core import text as textmod
from ..core.compose import Sprite, over
from ..core.layout import Rect, SafeZone
from . import options as O

NAME = "retro-overlay"
LOOKS = ("camcorder", "timeline", "vhs", "film")
CAPTIONS = ("off", "on", "stacked", "one-word", "serif-chip", "fitted")
EASE = 0.25  # seconds the look takes to come on, and to go
UNDER_VOICE = 12.0  # a one-shot's loudest tenth of a second, this far under the voice (LU)
BED_UNDER = 25.0  # a machine's bed, this far under the voice (LU)
OSD = (246, 246, 240)
REC_RED = (236, 46, 40)

OPTIONS = {
    "look": O.Choice(LOOKS, "camcorder: a viewfinder with REC, a timecode, battery and date · timeline: an "
                            "editing timeline of your own shots and sound, its playhead moving · vhs: a tape played "
                            "back · film: film grain, dust and a burn"),
    "date": O.Text("the date on the camcorder and the tape: auto (the date the file was recorded, when it says), "
                   "a date as you want it shown (\"OCT 28 2026\"), or off"),
    "strength": O.Number(0.0, 1.0, "how strong the look is, from 0 (only the readouts) to 1"),
    "start": O.Text("when the look comes on: a time in seconds, or words you say; 0 for the start"),
    "end": O.Text("when it goes: a time in seconds, words you say, or end"),
    "captions": O.Choice(CAPTIONS, "off, or the pack's captions over the look, clear of your face: on (stacked), or "
                                   "a look: one-word, serif-chip, fitted"),
    "sound": O.Switch("the machine's own quiet sound under your voice (a beep, a tape's whirr, a projector): on or off"),
    "fix": O.Text("misheard words as wrong=right, separated by ;  e.g. \"ask editor=AskEditor\""),
}

MATRIX = [
    ("A", "A", {}),
    ("B", "B", {"look": "timeline"}),
    ("C", "C", {"look": "vhs"}),
    ("D", "D", {"look": "film"}),
    ("B-vhs", "B", {"look": "vhs", "captions": "on"}),
    ("A-film", "A", {"look": "film"}),
    ("C-camcorder", "C", {"date": "JUL 14 2026"}),
    ("33-timeline", "33", {"look": "timeline"}),
]


# ---- this effect's own colour, made by the grade's maths --------------------------------------------

def recipes():
    from .grade import Recipe

    return {
        "camcorder": Recipe("retro-camcorder", temp=0.018, contrast=1.08, pivot=0.5, black=0.04, white=0.95, sat=1.06,
                            highs=(95.0, 0.008), skin_hold=0.85, skin_sat=1.0),
        "vhs": Recipe("retro-vhs", temp=0.028, tint=0.018, gamma=0.96, contrast=0.9, pivot=0.5, black=0.075,
                      white=0.9, sat=0.8, shadows=(300.0, 0.012), highs=(72.0, 0.012), skin_hold=0.78, skin_sat=0.98),
        "film": Recipe("retro-film", temp=0.04, contrast=1.1, pivot=0.47, black=0.05, white=0.93, sat=0.88,
                       shadows=(205.0, 0.014), highs=(66.0, 0.02), bend=((145.0, 50.0, -10.0),), skin_hold=0.86,
                       skin_sat=0.98),
    }


def lut_for(look: str, intensity: float) -> Optional[Path]:
    """The look's colour as a .cube, made once and kept in the temporary folder, named by what makes it."""
    from dataclasses import asdict

    from . import grade

    rs = recipes()
    if look not in rs or intensity <= 0:
        return None
    r = rs[look]
    key = hashlib.sha1(repr((asdict(r), round(intensity, 4), grade.LUT_SIZE, 1)).encode()).hexdigest()[:10]
    folder = Path(tempfile.gettempdir()) / "fx-retro-overlay"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{look}-{key}.cube"
    if not path.exists():
        with np.errstate(all="ignore"):
            table = np.nan_to_num(grade.cube(r, intensity))
        grade.write_cube(path, table, f"{r.name} ({intensity:g})")
    return path


# ---- when --------------------------------------------------------------------------------------------

def _moment(clip, value: str, duration: float, default: float, what: str) -> tuple[float, str]:
    v = str(value).strip()
    if v.lower() in ("", "end") and default >= duration:
        return duration, "to the end"
    if v.lower() in ("", "0", "start"):
        return default, "from the start" if default == 0 else ""
    try:
        t = float(v.rstrip("s"))
    except ValueError:
        from .headline import find_said

        found = find_said(clip.words, v)
        if found is None:
            raise O.OptionError(f"{what}: “{v}” is not said in this clip; give a time in seconds") from None
        w = clip.words[found[0]] if what == "start" else clip.words[found[1]]
        return (w.start if what == "start" else w.end), f"on “{v}”"
    if not 0 <= t <= duration:
        raise O.OptionError(f"{what}: {t:g} s is outside the clip's {duration:.2f} s")
    return t, f"at {t:.2f} s"


def weight(t: float, s: float, e: float, duration: float) -> float:
    """How much of the look is on at `t`: eased in after `s`, out before `e` (no ease at the clip's ends)."""
    if t < s or t > e:
        return 0.0
    a = 1.0 if s <= 1e-6 else min(1.0, (t - s) / EASE)
    b = 1.0 if e >= duration - 1e-6 else min(1.0, (e - t) / EASE)
    x = max(0.0, min(a, b))
    return x * x * (3 - 2 * x)


# ---- the date ----------------------------------------------------------------------------------------

def file_date(path: str) -> Optional[dt.date]:
    """The date the file says it was recorded, when it says: the phone's own tag first, else the
    container's creation time."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json", path],
                             capture_output=True, text=True, timeout=20).stdout
        tags = {k.lower(): v for k, v in (json.loads(out).get("format", {}).get("tags") or {}).items()}
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    for key in ("com.apple.quicktime.creationdate", "creation_time", "date"):
        v = tags.get(key)
        if v:
            try:
                return dt.date.fromisoformat(v[:10])
            except ValueError:
                continue
    return None


def date_text(clip, value: str) -> tuple[str, str]:
    v = value.strip()
    if v.lower() in ("off", "none", ""):
        return "", "no date"
    if v.lower() == "auto":
        d = file_date(clip.path)
        if d is None:
            return "", "no date (the file does not say when it was recorded; give date=)"
        return d.strftime("%b %d %Y").upper(), "the date the file says it was recorded"
    return v.upper(), "as typed"


# ---- readouts ------------------------------------------------------------------------------------------

def readout(text: str, px: int, colour=OSD) -> textmod.Word:
    """A line of a screen's own lettering, with the soft dark edge a viewfinder draws behind it."""
    return textmod.draw_word(text, "vt323", px, fill=colour, shadow=(0.0, 0.0, max(1.0, px / 18), 0.75))


def _box_of(word: textmod.Word, x: float, base: float) -> Rect:
    return Rect(x, base - word.ascent * 1.15, x + word.width, base + word.ascent * 0.25)


def corner_marks(W: int, H: int, u: float, inset_x: float, top: float, bottom: float) -> Sprite:
    """The viewfinder's four corner marks, as one sprite over the whole frame."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    arm, th = 64 * u, max(2, int(round(4 * u)))
    x0, x1 = inset_x, W - inset_x
    for (cx, cy, sx, sy) in ((x0, top, 1, 1), (x1, top, -1, 1), (x0, bottom, 1, -1), (x1, bottom, -1, -1)):
        d.line([(cx, cy + sy * arm), (cx, cy), (cx + sx * arm, cy)], fill=OSD + (235,), width=th, joint="curve")
    arr = np.asarray(img).copy()
    # a soft dark edge so the marks read on a white wall
    import cv2

    a = arr[..., 3].astype(np.float32) / 255.0
    halo = np.clip(cv2.GaussianBlur(a, (0, 0), 2.2 * u) * 1.8, 0, 1) * 0.55
    out_a = np.maximum(a, halo)
    rgb = arr[..., :3].astype(np.float32) * a[..., None]
    return Sprite(rgb=rgb, a=out_a)


def battery(u: float, fill: float = 0.75) -> Sprite:
    from PIL import Image, ImageDraw

    k = 4
    w, h = int(58 * u * k), int(28 * u * k)
    img = Image.new("RGBA", (w + 8 * k, h + 8 * k), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    o = 4 * k
    lw = max(k, int(3 * u * k))
    d.rounded_rectangle((o, o, o + w - 6 * u * k, o + h), radius=4 * u * k, outline=OSD + (255,), width=lw)
    d.rectangle((o + w - 6 * u * k, o + h * 0.3, o + w, o + h * 0.7), fill=OSD + (255,))
    inner = (o + 2 * lw, o + 2 * lw, o + (w - 6 * u * k) - 2 * lw, o + h - 2 * lw)
    bars = 3
    gap = 3 * u * k
    bw = (inner[2] - inner[0] - gap * (bars - 1)) / bars
    for b in range(int(round(fill * bars))):
        x = inner[0] + b * (bw + gap)
        d.rectangle((x, inner[1], x + bw, inner[3]), fill=OSD + (255,))
    img = img.resize((img.width // k, img.height // k), Image.LANCZOS)
    arr = np.asarray(img)
    import cv2

    a = arr[..., 3].astype(np.float32) / 255.0
    halo = np.clip(cv2.GaussianBlur(a, (0, 0), 2.0 * u) * 1.8, 0, 1) * 0.55
    return Sprite(rgb=arr[..., :3].astype(np.float32) * a[..., None], a=np.maximum(a, halo))


def triangle(h: float, colour) -> Sprite:
    """A play mark: a triangle pointing right, h tall, with the readouts' soft dark edge."""
    import cv2
    from PIL import Image, ImageDraw

    k = 4
    w = int(math.ceil(h * 0.9)) + 6
    H = int(math.ceil(h)) + 6
    img = Image.new("L", (w * k, H * k), 0)
    ImageDraw.Draw(img).polygon([(3 * k, 3 * k), (3 * k + h * 0.9 * k, 3 * k + h * k / 2), (3 * k, 3 * k + h * k)], fill=255)
    a = np.asarray(img.resize((w, H), Image.LANCZOS), np.float32) / 255.0
    halo = np.clip(cv2.GaussianBlur(a, (0, 0), max(1.0, h / 14)) * 1.8, 0, 1) * 0.55
    return Sprite(rgb=a[..., None] * np.asarray(colour, np.float32), a=np.maximum(a, halo))


def dot(d: float, colour) -> Sprite:
    return textmod.rounded_box(d, d, d / 2, fill=colour)


def timecode(t: float, fps: float) -> str:
    f = int(math.floor(t * fps + 1e-6))
    r = max(1, int(round(fps)))
    s, ff = divmod(f, r)
    m, ss = divmod(s, 60)
    h, mm = divmod(m, 60)
    return f"{h:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def counter(t: float) -> str:
    s = int(t)
    return f"{s // 3600}:{(s // 60) % 60:02d}:{s % 60:02d}"


class Readouts:
    """The words on screen for camcorder and vhs, in two bands of the safe zone (top and bottom),
    each moved to the other band, or left out, when the head or the captions reach it."""

    def __init__(self, look: str, W: int, H: int, fps: float, head: Optional[Rect], avoid: list[Rect], date: str):
        self.look, self.W, self.H, self.fps = look, W, H, fps
        u = W / 1080.0
        self.u = u
        z = SafeZone(W, H).rect
        self.zone = z
        self.px = int(round(58 * u))
        cap = textmod.cap_height("vt323", self.px)
        top, bottom = z.y0 + 12 * u + cap, z.y1 - 14 * u
        self.rows = {"top": top, "top2": top + 1.6 * cap + 18 * u, "bottom": bottom, "bottom2": bottom - 1.6 * cap - 18 * u}
        self.notes: list[str] = []
        self.items: list[tuple[str, str, str]] = []  # (what, row, side)
        if look == "camcorder":
            want = [("rec", "top", "left"), ("battery", "top", "right"), ("date", "bottom", "left"),
                    ("tc", "bottom", "right")]
        else:  # vhs
            want = [("play", "top", "left"), ("counter", "top", "right"), ("date", "bottom", "left")]
        if not date:
            want = [w for w in want if w[0] != "date"]
        self.date = date
        self.rec = readout("REC", self.px)
        self.play = readout("PLAY", self.px)
        self.arrow = triangle(0.62 * textmod.cap_height("vt323", self.px) * 1.3, OSD)
        self.date_w = readout(date, self.px) if date else None
        self.batt = battery(u)
        self.red = dot(26 * u, REC_RED)
        self.cache: dict[str, textmod.Word] = {}
        tc_w = textmod.advance("00:00:00:00", "vt323", self.px)
        widths = {"rec": 40 * u + self.rec.width, "battery": self.batt.w, "date": self.date_w.width if self.date_w else 0,
                  "tc": tc_w, "play": self.play.width + 0.9 * textmod.cap_height("vt323", self.px) * 1.3 + 16 * u, "counter": textmod.advance("SP 0:00:00", "vt323", self.px)}
        self.boxes: dict[str, Rect] = {}
        taken: list[Rect] = []
        obstacles = ([head.grow(10 * u, 10 * u, 10 * u, 10 * u)] if head is not None else []) + avoid
        for what, row, side in want:
            tries = [row, "top2" if row == "top" else "bottom2", "bottom" if row.startswith("top") else "top",
                     "bottom2" if row.startswith("top") else "top2"]
            placed = None
            for r in tries:
                base = self.rows[r]
                w = widths[what]
                x = z.x0 + 4 * u if side == "left" else z.x1 - 4 * u - w
                box = Rect(x, base - cap * 1.2, x + w, base + cap * 0.3)
                if any(box.intersects(o, 4 * u) for o in obstacles + taken):
                    continue
                placed = (r, box)
                break
            if placed is None:
                self.notes.append(f"{what} left out: no room clear of the face")
                continue
            if placed[0] != row:
                self.notes.append(f"{what} moved to the {placed[0].rstrip('2')} to keep clear of the face")
            self.items.append((what, placed[0], side))
            self.boxes[what] = placed[1]
            taken.append(placed[1])

    def _word(self, text: str) -> textmod.Word:
        if text not in self.cache:
            if len(self.cache) > 400:
                self.cache.clear()
            self.cache[text] = readout(text, self.px)
        return self.cache[text]

    def draw(self, frame: np.ndarray, t: float, since: float, a: float) -> None:
        u = self.u
        for what, row, side in self.items:
            box = self.boxes[what]
            base = self.rows[row]
            if what == "rec":
                if ((t - since) % 1.0) < 0.62:
                    over(frame, self.red, box.x0, base - self.rec.ascent / 2 - 13 * u - 1, a)
                w = self.rec
                over(frame, w.sprite, box.x0 + 40 * u - w.ox, base - w.oy, a)
            elif what == "play":
                w = self.play
                over(frame, w.sprite, box.x0 - w.ox, base - w.oy, a)
                over(frame, self.arrow, box.x0 + w.width + 16 * u, base - w.ascent - (self.arrow.h - w.ascent) / 2, a)
            elif what == "battery":
                over(frame, self.batt, box.x0, base - textmod.cap_height("vt323", self.px) / 2 - self.batt.h / 2, a)
            elif what == "date" and self.date_w is not None:
                w = self.date_w
                over(frame, w.sprite, box.x0 - w.ox, base - w.oy, a)
            elif what == "tc":
                w = self._word(timecode(t, self.fps))
                over(frame, w.sprite, box.x1 - w.width - w.ox, base - w.oy, a)
            elif what == "counter":
                w = self._word("SP " + counter(t))
                over(frame, w.sprite, box.x1 - w.width - w.ox, base - w.oy, a)


def head_reach(clip, fr, first: int, last: int) -> Optional[Rect]:
    """Where the head keeps to over frames first..last, in output pixels: each edge where it stays
    in 95% of the frames, so a moment's lean on a long take does not claim the whole frame (what
    is drawn there still keeps clear of the face itself, which sits well inside this)."""
    face = clip.face
    if not face.any():
        return None
    step = max(1, (last - first) // 400)
    boxes = [fr.to_output(h, i) for i in range(first, last + 1, step) if (h := face.head(i)) is not None]
    if not boxes:
        return None
    a = np.array([[b.x0, b.y0, b.x1, b.y1] for b in boxes])
    return Rect(float(np.percentile(a[:, 0], 2.5)), float(np.percentile(a[:, 1], 2.5)),
                float(np.percentile(a[:, 2], 97.5)), float(np.percentile(a[:, 3], 97.5)))


# ---- the tape: colour that bleeds, scanlines, a tracking band ------------------------------------------

class Tape:
    def __init__(self, W: int, H: int, strength: float):
        self.W, self.H, self.s = W, H, strength
        self.u = W / 1080.0
        rows = np.arange(H, dtype=np.float32)
        period = max(3.0, 4.0 * self.u)
        self.lines = (1.0 - 0.17 * strength * (0.5 + 0.5 * np.cos(2 * np.pi * rows / period)))[:, None, None]
        self.lines = self.lines.astype(np.float32)

    def apply(self, frame: np.ndarray, j: int, t: float, w: float) -> None:
        import cv2

        s, u, H, W = self.s * w, self.u, self.H, self.W
        if s <= 0:
            return
        ycc = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycc)
        k = max(3, int(round(27 * u * s)) | 1)
        cr = cv2.blur(cr, (k, 1))
        cb = cv2.blur(cb, (k, 1))
        shift = 9 * u * s
        m1 = np.float32([[1, 0, shift], [0, 1, 0]])
        m2 = np.float32([[1, 0, -0.6 * shift], [0, 1, 0]])
        cr = cv2.warpAffine(cr, m1, (W, H), borderMode=cv2.BORDER_REPLICATE)
        cb = cv2.warpAffine(cb, m2, (W, H), borderMode=cv2.BORDER_REPLICATE)
        y = cv2.blur(y, (max(1, int(round(4 * u * s))) | 1, 1))
        cr = cv2.addWeighted(cr, 0.85, np.full_like(cr, 128), 0.15, 0)  # tape keeps less colour than it was given
        cb = cv2.addWeighted(cb, 0.85, np.full_like(cb, 128), 0.15, 0)
        out = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2RGB).astype(np.float32)
        out *= self.lines if abs(s - self.s) < 1e-6 else (1.0 - (1.0 - self.lines) * (s / max(1e-6, self.s)))
        rng = np.random.default_rng(10007 + j)
        noise = cv2.resize(rng.standard_normal((H // 2, W // 2)).astype(np.float32), (W, H))
        out += noise[..., None] * 7.0 * s
        # the tracking band: drifts up the picture every seven seconds, rows pulled sideways in it
        cyc = 7.0
        pos = 1.15 - ((t % cyc) / cyc) * 1.35
        band_h = 0.045 * H
        yc = pos * H
        y0, y1 = int(max(0, yc - band_h)), int(min(H, yc + band_h))
        if y1 > y0:
            rows = np.arange(y0, y1)
            fall = np.exp(-((rows - yc) / (0.5 * band_h)) ** 2)
            offs = (rng.standard_normal(len(rows)) * 3.0 + 5.0) * u * s * fall
            seg = out[y0:y1]
            for i, (o, f) in enumerate(zip(offs, fall)):
                if abs(o) >= 0.5:
                    seg[i] = np.roll(seg[i], int(round(o)), axis=0)
                seg[i] += f * (rng.standard_normal((W, 1)).astype(np.float32) * 18.0 * s + 10.0 * s)
        # the head-switching noise along the bottom edge
        hb = int(0.012 * H)
        if hb > 0:
            tail = out[H - hb:]
            tail[:] = np.roll(tail, int(round(9 * u * s)), axis=1)
            tail += rng.standard_normal((hb, W, 1)).astype(np.float32) * 22.0 * s
        frame[:] = np.clip(out + 0.5, 0, 255).astype(np.uint8)


# ---- film: weave, flicker, dust, a scratch, burns ----------------------------------------------------

class Film:
    def __init__(self, clip, fr, W: int, H: int, strength: float, spans: tuple[float, float], duration: float):
        from . import grade

        self.W, self.H, self.s = W, H, strength
        self.u = W / 1080.0
        self.clip, self.fr = clip, fr
        self.s0, self.s1, self.duration = spans[0], spans[1], duration
        self.finish = grade.Finish(W, H, 0.35 * strength, 0.55 * strength, grade.subject_centre(clip, W, H))
        u = self.u
        # the projected frame's rounded corners
        r = 46 * u
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        qx = np.abs(xx + 0.5 - W / 2) - (W / 2 - r)
        qy = np.abs(yy + 0.5 - H / 2) - (H / 2 - r)
        d = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) - r
        self.gate = np.clip(0.5 - d / (1.5 * u), 0, 1).astype(np.float32)[..., None]
        self.small = (max(8, W // 8), max(8, H // 8))
        sy, sx = np.mgrid[0:self.small[1], 0:self.small[0]].astype(np.float32)
        self.sx, self.sy = sx / self.small[0], sy / self.small[1]
        # the burns: one as it starts, one as it ends, and a soft leak every so often in between
        self.events = []
        if self.s0 <= 0.05:
            self.events.append((self.s0, 1.1, "burn", 0))
        else:
            self.events.append((self.s0 - 0.1, 1.1, "burn", 0))
        t = self.s0 + 4.5
        k = 1
        while t < self.s1 - 3.0:
            self.events.append((t, 1.6, "leak", k))
            t += 5.5
            k += 1
        if self.s1 - self.s0 > 3.0:
            self.events.append((self.s1 - 1.0, 1.0, "burn", k))

    def _face_mask(self, j: int) -> np.ndarray:
        """1 away from the face, falling towards 0.25 on it: what keeps a burn off the face."""
        b = self.clip.face.box(j) if self.clip.face.any() else None
        if b is None:
            return np.ones_like(self.sx)
        b = self.fr.to_output(b, j)
        cx, cy = b.cx / self.W, (b.cy - 0.1 * b.h) / self.H
        rx, ry = 1.1 * b.w / self.W, 1.0 * b.h / self.H
        d = ((self.sx - cx) / rx) ** 2 + ((self.sy - cy) / ry) ** 2
        return (0.25 + 0.75 * np.clip(d - 0.6, 0, 1)).astype(np.float32)

    def glow(self, j: int, t: float) -> Optional[np.ndarray]:
        """The burns' light at `t`, small, as float RGB 0..1: None when there is none."""
        total = None
        for start, length, kind, k in self.events:
            q = (t - start) / length
            if not 0 <= q <= 1:
                continue
            rng = np.random.default_rng(500 + k)
            corner = rng.integers(0, 4)
            ex = (corner % 2) * 1.0
            ey = (corner // 2) * 1.0
            if kind == "burn":
                grow = 0.35 + 0.95 * math.sin(math.pi * min(1.0, q * 1.15)) ** 0.8
                amp = math.sin(math.pi * q) ** 0.7
                dx, dy = (self.sx - ex) * 1.0, (self.sy - ey) * 0.62
                rad = np.sqrt(dx * dx + dy * dy) / max(0.05, 0.55 * grow)
                wob = 0.12 * np.sin(9.0 * self.sx + 5.0 * self.sy + 6.0 * q + k)
                g = np.clip(1.0 - rad + wob, 0, 1) ** 1.6 * amp
                core = np.clip(g * 1.6 - 0.6, 0, 1)
                rgb = np.stack([np.clip(g * 1.25, 0, 1), g * 0.52 + core * 0.4, g * 0.12 + core * 0.45], axis=-1)
            else:
                cx = -0.3 + 1.6 * q if k % 2 else 1.3 - 1.6 * q
                d = np.sqrt(((self.sx - cx) / 0.45) ** 2 + ((self.sy - (0.2 + 0.6 * ey)) / 0.7) ** 2)
                g = np.clip(1.0 - d, 0, 1) ** 2 * math.sin(math.pi * q) * 0.55
                rgb = np.stack([g * 1.0, g * 0.45, g * 0.32], axis=-1)
            total = rgb if total is None else np.maximum(total, rgb)
        if total is None:
            return None
        return (total * self._face_mask(j)[..., None]).astype(np.float32)

    def apply(self, frame: np.ndarray, j: int, t: float, w: float) -> None:
        import cv2

        s, u = self.s * w, self.u
        if s <= 0:
            return
        W, H = self.W, self.H
        rng = np.random.default_rng(20011 + j)
        # the gate weave: a slow, small drift, mostly up and down
        dx = (math.sin(t * 2.1) * 0.6 + math.sin(t * 5.3 + 1.0) * 0.4) * 1.2 * u * s
        dy = (math.sin(t * 1.7 + 0.5) * 0.6 + math.sin(t * 6.1) * 0.4) * 2.2 * u * s
        out = cv2.warpAffine(frame, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H), borderMode=cv2.BORDER_REFLECT)
        # the flicker: a slow breath of a couple of percent, never a flash
        fl = 1.0 + 0.022 * s * (math.sin(t * 13.0) * 0.5 + math.sin(t * 29.0 + 2.0) * 0.3 + rng.standard_normal() * 0.2)
        f = out.astype(np.float32) * fl
        g = self.glow(j, t)
        if g is not None:
            big = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR) * (255.0 * s)
            f = 255.0 - (255.0 - f) * (255.0 - big) / 255.0  # screen: light added, never darkening
            # and burnt: the light is coloured, so it shows on a white wall as well as a dark room
            heat = np.clip(big[..., :1] / 255.0, 0, 1)
            f *= 1.0 - heat * np.asarray([0.0, 0.28, 0.62], np.float32) * s
        # dust and a scratch, off the face
        b = self.clip.face.box(j) if self.clip.face.any() else None
        face = self.fr.to_output(b, j).grow(0.3 * b.w, 0.5 * b.h, 0.3 * b.w, 0.3 * b.h) if b is not None else None
        for _ in range(int(rng.integers(0, 4))):
            x, y = rng.random() * W, rng.random() * H
            if face is not None and face.x0 <= x <= face.x1 and face.y0 <= y <= face.y1:
                continue
            rad = (1.2 + rng.random() * 3.2) * u
            shade = -110.0 if rng.random() < 0.7 else 90.0
            x0, y0 = int(max(0, x - rad * 2)), int(max(0, y - rad * 2))
            x1, y1 = int(min(W, x + rad * 2 + 1)), int(min(H, y + rad * 2 + 1))
            if x1 > x0 and y1 > y0:
                yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
                m = np.clip(1.0 - np.hypot(xx - x, yy - y) / rad, 0, 1)[..., None] * 0.8 * s
                f[y0:y1, x0:x1] = f[y0:y1, x0:x1] * (1 - m) + (12.0 if shade < 0 else 235.0) * m
        seg = int(t / 1.3)
        srng = np.random.default_rng(31000 + seg)
        if srng.random() < 0.55:
            x = srng.random() * W
            if face is None or not (face.x0 - 20 * u <= x <= face.x1 + 20 * u):
                x += math.sin(t * 40.0) * 1.5 * u
                xi = int(x)
                if 1 <= xi < W - 2:
                    f[:, xi:xi + max(1, int(round(1.5 * u)))] += 38.0 * s
        f = np.clip(f + 0.5, 0, 255).astype(np.uint8)
        self.finish.apply(f, j, w)
        frame[:] = (f.astype(np.float32) * (1.0 - (1.0 - self.gate) * min(1.0, s * 1.2))).astype(np.uint8)


# ---- the timeline --------------------------------------------------------------------------------------

class TimelinePanel:
    """A frosted panel with the clip laid out as an editor shows it: its own shots, its own sound,
    and a playhead that moves with the real time."""

    def __init__(self, clip, fr, W: int, H: int, fps: float, duration: float, cuts: list[float],
                 head: Optional[Rect], avoid: list[Rect]):
        from ..core import glass, video

        self.W, self.H, self.fps, self.duration = W, H, fps, duration
        u = W / 1080.0
        self.u = u
        z = SafeZone(W, H).rect
        self.notes: list[str] = []
        scale = 1.0
        halo = glass.Pane.reach(u)
        placed = None
        while scale >= 0.7 and placed is None:
            pw, ph = (z.w - halo[0] - halo[2]) * scale, 330 * u * scale
            x = (W - pw) / 2
            spots = []
            if head is not None:
                spots.append(("below the chin", max(head.y1 + 24 * u, z.y1 - halo[3] - ph - 4 * u)))
                spots.append(("over the chest", head.y1 + 24 * u))
                spots.append(("above the head", head.y0 - 24 * u - ph))
            else:
                spots.append(("in the lower third", z.y1 - halo[3] - ph - 4 * u))
            for where, y in spots:
                r = Rect(x, y, x + pw, y + ph)
                if not r.grow(*halo).inside(z, slack=1.0):
                    continue
                if head is not None and r.intersects(head, 8 * u):
                    continue
                if any(r.intersects(a, 6 * u) for a in avoid):
                    continue
                placed = (where, r)
                break
            if placed is None:
                scale -= 0.1
        if placed is None:  # a head that fills the frame: the lowest place the safe zone allows, at full size
            scale = 1.0
            pw, ph = z.w - halo[0] - halo[2], 330 * u
            r = Rect((W - pw) / 2, z.y1 - halo[3] - ph - 4 * u, (W + pw) / 2, z.y1 - halo[3] - 4 * u)
            placed = ("at the bottom of the safe zone, where the head reaches at times", r)
            self.notes.append("no room always clear of the head: the panel sits as low as the safe zone allows")
        self.where, self.rect = placed
        if scale < 0.999:
            self.notes.append(f"made {scale:.0%} of its size to fit")
        k = self.rect.h / (330 * u)
        self.k = k
        self.pane = glass.Pane(int(round(self.rect.w)), int(round(self.rect.h)), 26 * u * k, unit=u * k)
        sample = [fr.apply(video.read_frame(clip.info, clip.info.frame_at(duration * q)), clip.info.frame_at(duration * q))
                  for q in (0.2, 0.5, 0.8)]
        self.pane.tone_over(sample, self.rect.x0, self.rect.y0)
        # what is inside the panel, drawn once
        self.pad = 26 * u * k
        self.label_w = 54 * u * k
        self.track_x0 = self.pad + self.label_w
        self.track_x1 = self.rect.w - self.pad
        self.y_ruler = self.pad + 62 * u * k
        self.y_video = (self.y_ruler + 26 * u * k, self.y_ruler + 26 * u * k + 118 * u * k)
        self.y_audio = (self.y_video[1] + 14 * u * k, self.rect.h - self.pad)
        self.content = self._content(clip, fr, cuts)
        self.tc_px = int(round(40 * u * k))
        self.total = timecode(duration, fps)[3:]
        self.cache: dict[str, textmod.Word] = {}
        self.head_sprite = self._playhead()

    def x_of(self, t: float) -> float:
        return self.track_x0 + (self.track_x1 - self.track_x0) * min(1.0, max(0.0, t / max(1e-6, self.duration)))

    def _content(self, clip, fr, cuts: list[float]) -> Sprite:
        import cv2
        from PIL import Image, ImageDraw

        from ..core import video

        u, k = self.u, self.k
        w, h = int(round(self.rect.w)), int(round(self.rect.h))
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        lab = textmod.font("poppins-semibold", int(round(22 * u * k)))
        small = textmod.font("vt323", int(round(28 * u * k)))
        # track labels
        for label, (a, b) in (("V1", self.y_video), ("A1", self.y_audio)):
            d.text((self.pad, (a + b) / 2), label, font=lab, fill=(230, 232, 238, 200), anchor="lm")
        # the ruler: a tick a second (or every two, five, ten on a long clip), a number on the big ones
        dur = self.duration
        span = self.track_x1 - self.track_x0
        step = next(s for s in (1, 2, 5, 10, 15, 30, 60) if span / max(1.0, dur / s) >= 54 * u * k)
        n = int(dur // step) + 1
        for i in range(n * 2 + 1):
            t = i * step / 2
            if t > dur + 1e-6:
                break
            x = self.x_of(t)
            big = i % 2 == 0
            d.line([(x, self.y_ruler - (14 if big else 7) * u * k), (x, self.y_ruler)], fill=(225, 228, 235, 190 if big else 120),
                   width=max(1, int(round(2 * u * k))))
            if big and x < self.track_x1 - 30 * u * k:
                s = int(t)
                d.text((x + 5 * u * k, self.y_ruler - 16 * u * k), f"{s // 60}:{s % 60:02d}", font=small,
                       fill=(225, 228, 235, 200), anchor="ls")
        # the video track: one clip for each of the video's own shots, filled with its own frames
        bounds = [0.0] + [c for c in cuts if 0 < c < dur] + [dur]
        a_y, b_y = self.y_video
        th = int(round(b_y - a_y))
        tw = max(8, int(round(th * clip.info.width / clip.info.height)))
        frames: list[np.ndarray] = []
        for s0, s1 in zip(bounds, bounds[1:]):
            x0, x1 = self.x_of(s0) + 2 * u * k, self.x_of(s1) - 2 * u * k
            if x1 - x0 < 3:
                continue
            bw = int(round(x1 - x0))
            strip = np.zeros((th, bw, 3), np.uint8)
            count = max(1, int(math.ceil(bw / tw)))
            for q in range(count):
                tm = s0 + (s1 - s0) * (q + 0.5) / count
                i = clip.info.frame_at(tm)
                small_f = cv2.resize(fr.apply(video.read_frame(clip.info, i), i), (tw, th), interpolation=cv2.INTER_AREA)
                frames.append(small_f)
                xa = q * tw
                xb = min(bw, xa + tw)
                strip[:, xa:xb] = small_f[:, : xb - xa]
            box = Image.new("L", (bw, th), 0)
            ImageDraw.Draw(box).rounded_rectangle((0, 0, bw - 1, th - 1), radius=10 * u * k, fill=255)
            pic = Image.fromarray(strip).convert("RGBA")
            pic.putalpha(box)
            img.alpha_composite(pic, (int(round(x0)), int(round(a_y))))
            d.rounded_rectangle((x0, a_y, x0 + bw - 1, a_y + th - 1), radius=10 * u * k, outline=(255, 255, 255, 150),
                                width=max(1, int(round(2 * u * k))))
        self.shots = len(bounds) - 1
        # the audio track: the clip's own sound, as a waveform
        a_y, b_y = self.y_audio
        mid, half = (a_y + b_y) / 2, (b_y - a_y) / 2
        d.rounded_rectangle((self.track_x0, a_y, self.track_x1, b_y), radius=10 * u * k, fill=(46, 160, 120, 70))
        env = clip.envelope
        db = env.db
        hop = env.hop
        cols = int((self.track_x1 - self.track_x0) / (4 * u * k))
        for c in range(cols):
            t0 = dur * c / cols
            t1 = dur * (c + 1) / cols
            seg = db[int(t0 / hop):max(int(t0 / hop) + 1, int(t1 / hop))]
            v = float(seg.max()) if len(seg) else -90.0
            amp = np.clip((v - env.floor) / max(6.0, (env.speech + 6.0) - env.floor), 0.04, 1.0)
            x = self.track_x0 + (c + 0.5) * (self.track_x1 - self.track_x0) / cols
            d.line([(x, mid - amp * half * 0.9), (x, mid + amp * half * 0.9)], fill=(120, 236, 186, 235),
                   width=max(1, int(round(2.4 * u * k))))
        return Sprite.from_rgba(np.asarray(img))

    def _playhead(self) -> Sprite:
        from PIL import Image, ImageDraw

        u, k = self.u, self.k
        top = self.y_ruler - 30 * u * k
        h = int(round(self.y_audio[1] + 6 * u * k - top))
        w = int(round(26 * u * k))
        s = 4
        img = Image.new("RGBA", (w * s, h * s), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        lw = max(s, int(round(3 * u * k * s)))
        d.rectangle(((w * s - lw) / 2, 10 * u * k * s, (w * s + lw) / 2, h * s), fill=REC_RED + (255,))
        d.polygon([(0, 0), (w * s, 0), (w * s, 12 * u * k * s), (w * s / 2, 24 * u * k * s), (0, 12 * u * k * s)],
                  fill=REC_RED + (255,))
        self.head_top = top
        return Sprite.from_rgba(np.asarray(img.resize((w, h), Image.LANCZOS)))

    def _word(self, text: str, px: int) -> textmod.Word:
        key = f"{px}:{text}"
        if key not in self.cache:
            if len(self.cache) > 300:
                self.cache.clear()
            self.cache[key] = textmod.draw_word(text, "vt323", px, fill=(246, 246, 240))
        return self.cache[key]

    def draw(self, frame: np.ndarray, t: float, since: float) -> None:
        u, k = self.u, self.k
        q = min(1.0, max(0.0, (t - since) / 0.4))
        e = 1 - (1 - q) ** 3
        if e <= 0:
            return
        dy = (1 - e) * 60 * u
        X, Y = self.rect.x0, self.rect.y0 + dy
        self.pane.draw(frame, X, Y, opacity=e)
        over(frame, self.content, X, Y, e)
        w = self._word(timecode(t, self.fps)[3:], self.tc_px)
        over(frame, w.sprite, X + self.pad - w.ox, Y + self.pad + w.ascent - w.oy, e)
        tot = self._word("/ " + self.total, int(self.tc_px * 0.8))
        over(frame, tot.sprite, X + self.rect.w - self.pad - tot.width - tot.ox, Y + self.pad + w.ascent - tot.oy, e * 0.75)
        x = self.x_of(t)
        over(frame, self.head_sprite, X + x - self.head_sprite.w / 2, Y + self.head_top, e)


# ---- sound ----------------------------------------------------------------------------------------------

class Sounds:
    def __init__(self, clip, look: str, s: float, e: float):
        voice = soundmod.integrated(soundmod.blocks(clip.sound)) if clip.info.has_audio else soundmod.SILENCE_DB
        self.ref = voice if voice > -70 else -20.0
        self.look, self.s, self.e = look, s, e
        self.made: list[str] = []

    def mix(self, base: np.ndarray, rate: int) -> np.ndarray:
        from ..core import loudness

        stem = np.zeros_like(base)
        length = max(0.3, self.e - self.s)
        if self.look == "camcorder":
            x, anchor = synth.beep()
            soundmod.lay(stem, synth.at_level(x, self.ref - UNDER_VOICE - 4), self.s + 0.02, rate, anchor=anchor)
            self.made.append("a beep as REC starts")
        elif self.look == "timeline":
            x, anchor = soundmod.library("swoosh-in")
            soundmod.lay(stem, synth.at_level(x, self.ref - UNDER_VOICE - 2), self.s + 0.1, rate, anchor=anchor)
            self.made.append("a soft slide as the panel rises")
        elif self.look in ("vhs", "film"):
            bed = synth.whirr(length) if self.look == "vhs" else synth.projector(length)
            lufs = soundmod.integrated(soundmod.blocks(bed))
            bed = (bed * soundmod.gain((self.ref - BED_UNDER) - lufs)).astype(np.float32)
            soundmod.lay(stem, bed, self.s, rate)
            self.made.append("a tape deck's whirr" if self.look == "vhs" else "a projector's clatter")
        out = base + stem
        ceiling = max(-1.0, loudness.true_peak_db(base) + 0.05)
        if loudness.true_peak_db(out) > ceiling:
            out, _ = loudness.limit(out, ceiling)
        return out.astype(np.float32)


# ---- the effect -------------------------------------------------------------------------------------------

def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from ..core import framing as framing_mod
    from ..core import shots
    from ..core.timeline import Timeline
    from . import Job, PlanError, grade, layers

    info = clip.info
    fps = info.rate
    fr = framing_mod.plan(info.width, info.height, fps, info.frames, None, shape="keep")
    W, H = fr.out_w, fr.out_h
    duration = info.frames / fps
    tl = Timeline.whole(duration)
    look = opts["look"]
    st = opts["strength"]
    if plan is not None:
        try:
            s, e = float(plan["span"]["start"]), float(plan["span"]["end"])
        except (KeyError, TypeError, ValueError):
            raise PlanError("the plan needs a span with a start and an end in seconds") from None
        if not 0 <= s < e <= duration + 0.5:
            raise PlanError(f"the plan's span must sit inside the clip's {duration:.2f} s")
        why_s = why_e = "from the plan"
    else:
        s, why_s = _moment(clip, opts["start"], duration, 0.0, "start")
        e, why_e = _moment(clip, opts["end"], duration, duration, "end")
        if e <= s + 0.3:
            raise O.OptionError(f"the look would be on for {max(0.0, e - s):.2f} s; give an end after the start")
    head = head_reach(clip, fr, info.frame_at(s), info.frame_at(min(duration, e) - 1e-3))
    cap_layer = None
    avoid: list[Rect] = []
    if opts["captions"] != "off":
        given = {} if opts["captions"] == "on" else {"look": opts["captions"]}
        try:
            cap_layer = layers.prepare(clip, layers.Spec("captions", given), None, None, fr, fix=opts["fix"])
        except ValueError as err:
            raise O.OptionError(f"captions: {err}") from None
        avoid = [Rect.of(c["rect"]) for c in cap_layer.job.plan.get("cards") or [] if c.get("rect")]
    lut = lut_for(look, st)
    graded = grade.GradedFrames(info, fr, lut) if lut is not None else None
    marks = readouts = tape = film = panel = None
    date, date_why = ("", "")
    cuts: list[float] = []
    u = W / 1080.0
    z = SafeZone(W, H).rect
    if look in ("camcorder", "vhs"):
        date, date_why = date_text(clip, opts["date"])
        readouts = Readouts(look, W, H, fps, head, avoid, date)
    if look == "camcorder":
        marks = corner_marks(W, H, u, 70 * u, z.y0 - 0.02 * H, z.y1 + 0.03 * H)
    if look == "vhs":
        tape = Tape(W, H, st)
    if look == "film":
        film = Film(clip, fr, W, H, st, (s, e), duration)
    if look == "timeline":
        cuts = shots.cuts(clip)
        panel = TimelinePanel(clip, fr, W, H, fps, duration, cuts, head, avoid)

    def draw(frame: np.ndarray, j: int, t: float) -> None:
        w = weight(t, s, e, duration)
        if w > 0:
            k = tl.source_frame(j, fps, info.frames)
            if graded is not None:
                g = graded.at(k)
                if w >= 0.999:
                    frame[:] = g
                else:
                    frame[:] = np.clip(frame + (g.astype(np.float32) - frame) * w + 0.5, 0, 255).astype(np.uint8)
            if look == "camcorder" and st > 0:
                import cv2

                soft = cv2.GaussianBlur(frame, (0, 0), 1.1 * u)
                sharp = cv2.addWeighted(frame, 1.0 + 0.6 * st * w, soft, -0.6 * st * w, 0)  # a small camera's edge halo
                frame[:] = sharp
            if tape is not None:
                tape.apply(frame, j, t, w)
            if film is not None:
                film.apply(frame, j, t, w)
            if marks is not None:
                over(frame, marks, 0, 0, w)
            if readouts is not None:
                readouts.draw(frame, t, s, w)
            if panel is not None:
                panel.draw(frame, t, s)
        if cap_layer is not None:
            cap_layer.draw(frame, j, t)

    mixer = Sounds(clip, look, s, e) if opts["sound"] and info.has_audio else None
    words = []
    if readouts is not None:
        words = [b for b in readouts.boxes.values()]
    if panel is not None:
        words.append(panel.rect)
    outside = [r.as_list() for r in words if not r.inside(z, slack=1.0)]
    over_face = head is not None and any(r.intersects(head) for r in words)
    span_len = e - s
    sheet = [s + min(0.4, span_len / 4)] + [s + span_len * q for q in (0.2, 0.4, 0.6, 0.8)] + [max(s, e - 0.3)]
    out_plan = {"effect": NAME, "clip": clip.path, "options": opts,
                "output": {"width": W, "height": H, "fps": round(fps, 3), "seconds": round(duration, 3), "framing": fr.note},
                "span": {"start": round(s, 3), "end": round(e, 3)}, "date": date,
                "lut": str(lut) if lut else None, "words": [r.as_list() for r in words],
                "sheet_times": [round(x, 3) for x in sheet]}
    checks = {"look": f"{look} at {st:g}", "span": f"{s:.2f}–{e:.2f} s" + (f" ({why_s})" if why_s else "")}
    if readouts is not None:
        checks["date"] = f"{date or '(none)'}: {date_why}"
        checks["readouts"] = ", ".join(f"{w} {r.rstrip('2')}" for w, r, _ in readouts.items)
        if readouts.notes:
            checks["moved"] = "; ".join(readouts.notes)
    if panel is not None:
        out_plan["timeline"] = {"rect": panel.rect.as_list(), "where": panel.where, "shots": panel.shots,
                                "cuts": cuts}
        checks["timeline"] = f"{panel.shots} shot{'s' if panel.shots != 1 else ''} from the video's own cuts, {panel.where}"
        if panel.notes:
            checks["moved"] = "; ".join(panel.notes)
    if film is not None:
        checks["burns"] = ", ".join(f"{k} at {a:.1f} s" for a, _, k, _ in film.events)
    if cap_layer is not None:
        checks["captions"] = f"{len(avoid)} cards over the look; readouts and panel kept clear of them"
    warnings = []
    if outside:
        warnings.append(f"{len(outside)} readouts reach outside the safe zone")
    if over_face:
        warnings.append("a readout or the panel reaches the head's box")
    checks["warnings"] = warnings
    still = [sheet[0], sheet[2], sheet[4]]

    def after(result: dict, out: Path) -> dict:
        if graded is not None:
            graded.close()
        return {"sounds": ", ".join(mixer.made)} if mixer is not None and mixer.made else {}

    return Job(NAME, clip, tl, fr, draw, out_plan, checks=checks, still_times=lambda: still, after=after,
               sound=mixer.mix if mixer is not None else None)
