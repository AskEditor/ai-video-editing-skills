"""hook: the first three seconds, built on the speaker's first line.

The line is the speaker's own: the first whole line they say (a scrap of a sentence the clip came in
on halfway is skipped), the words a person would type into the app (fillers and words taken back
left out), at most a dozen or so words, ending where a clause ends. Or, with `open=strongest`, the
strongest line of the whole clip (the pack's shared picker, `core/moments.py`) is played first as a
teaser, and the film then starts from the beginning.

The line lands **on the first frame of speech**, set in a box like the app's own text (or dark, or
plain), placed where it never covers the face over every frame it is up (above the head first, then
under the chin, always inside the safe zone), and clear of captions already drawn into the video.

The looks, each with the line:

- `text-box`: the line alone, as if typed in the app, with a soft pop;
- `snap-zoom`: the camera snaps in on the face on the line's strongest word, holds, and eases back
  in the next pause (the core's camera, `core/zoom.py`, shared with punch-in);
- `whoosh-in`: the film starts rushed in on the face and settles back, smeared along the move, with
  a whoosh;
- `iris-open`: the film opens out of black through a circle widening from the face;
- `shake-flash`: held a little in from the first frame, the frame jolts and flashes white on the
  strongest word, then settles.

Every zoom keeps the whole head in the picture, a tenth of the frame above the crown.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..core import moments, sound, synth, text, zoom
from ..core.compose import Sprite, over
from ..core.layout import Rect, SafeZone
from ..core.words import HESITATIONS, Word
from . import options as O

NAME = "hook"
MOVES_PERSON = True  # most looks move the whole picture

LOOKS = ("text-box", "snap-zoom", "whoosh-in", "iris-open", "shake-flash")
STYLES = ("box", "dark", "plain")
MOST_WORDS = 14  # the most words of the line set on screen
EASY_WORDS = 10  # past this, the line ends where a second thought joins on ("and", "but"), when it can
JOINERS = ("and", "but", "so", "or", "because", "then")
REACH = 3.2  # seconds: a line takes in the next part only if it starts this soon after the first word
SHOWN = (1.8, 3.6)  # seconds the line stays up: at least, at most (the first three seconds)
AFTER = 0.45  # seconds it stays after its last word
FACE = "poppins-semibold"
OWN_SOUND = {"text-box": "pop", "snap-zoom": "hit", "whoosh-in": "whoosh", "iris-open": "swoosh-in",
             "shake-flash": "hit"}
UNDER_VOICE = 8.0
TEASER = (1.0, 4.0)  # seconds: the shortest and the longest strongest line played first

OPTIONS = {
    "look": O.Choice(LOOKS, "text-box: the line as the app's own text · snap-zoom: a snap in on your face on its "
                            "strongest word · whoosh-in: the film starts rushed in and settles · iris-open: it opens "
                            "out of black from your face · shake-flash: a jolt and a white flash on the strongest word"),
    "open": O.Choice(("first", "strongest"), "first: the film as it is, the hook on your first line · strongest: "
                                             "your strongest line played first, then the film from the start"),
    "text": O.Text("the words to set: empty for the line you say (the first, or the strongest with open=strongest), "
                   "off for none, or your own words"),
    "style": O.Choice(STYLES, "box: black words on white boxes, like the app's own text · dark: white on dark boxes · "
                              "plain: white words with a soft shadow"),
    "sound": O.Choice(("auto", "off"), "auto: the look's own sound under your voice · off: the clip's sound only"),
    "frame": O.Choice(("keep", "9:16"), "keep the clip's own shape, or make it 9:16 following the face"),
    "fix": O.Text("misheard words as wrong=right, separated by ;  e.g. \"Jon=John; ask editor=AskEditor\""),
}

MATRIX = [
    ("A", "A", {}),
    ("B", "B", {}),
    ("C", "C", {}),
    ("D", "D", {}),
    ("A-snap-zoom", "A", {"look": "snap-zoom"}),
    ("B-whoosh-in", "B", {"look": "whoosh-in"}),
    ("C-iris-open", "C", {"look": "iris-open"}),
    ("D-shake-flash", "D", {"look": "shake-flash", "style": "dark"}),
    ("A-strongest", "A", {"open": "strongest", "look": "snap-zoom"}),
    ("B-strongest", "B", {"open": "strongest"}),
    ("33-strongest", "33", {"open": "strongest", "look": "whoosh-in"}),
]


# ---- the line ----------------------------------------------------------------------------------------

@dataclass
class Line:
    first: int  # word indexes, in the (possibly re-ordered) clip's words
    last: int
    text: str
    start: float  # the first word begins: the line lands on this frame
    end: float  # the last word ends
    key: Optional[int] = None  # its strongest word
    why: str = ""


def _typed(words: list[Word], idx: list[int], back) -> str:
    """The words as a person would type them: fillers and words taken back left out, the last
    stop or comma dropped."""
    kept = [words[i] for i in idx if words[i].key not in HESITATIONS and words[i] not in back]
    out = " ".join(w.text.strip() for w in kept).strip()
    while out and out[-1] in ".,;:—-":
        out = out[:-1].rstrip()
    return out[:1].upper() + out[1:]  # set on its own, a line starts with a capital


def first_line(words: list[Word], quiet) -> Optional[Line]:
    """The first whole line said: the first line that is not a scrap of a sentence begun before the
    clip and holds a word worth saying, joined with the lines after it in the same sentence while
    they start soon enough and the words stay few."""
    if not words:
        return None
    back = moments.taken_back(words, quiet)
    runs = moments.lines(words, quiet)
    scraps = moments.scrap_lines(words, runs)
    start = None
    for n, ln in enumerate(runs):
        if n in scraps:
            continue
        real = [i for i in ln if words[i].key not in HESITATIONS and words[i] not in back]
        if real:
            start = n
            break
    if start is None:
        return None
    idx: list[int] = []
    t0 = words[runs[start][0]].start
    for ln in runs[start:]:
        real = [i for i in ln if words[i].key not in HESITATIONS and words[i] not in back]
        if idx and (words[ln[0]].start - t0 > REACH or len(idx) + len(real) > MOST_WORDS):
            break
        idx += ln
        if words[ln[-1]].ends_sentence:
            break
    real = [i for i in idx if words[i].key not in HESITATIONS and words[i] not in back]
    if len(real) > EASY_WORDS:  # long: end it where the speaker joins on a second thought, if that leaves enough
        joins = [k for k, i in enumerate(real) if 6 <= k <= EASY_WORDS and (words[i].key in JOINERS or words[real[k - 1]].ends_clause)]
        if joins:
            real = real[:joins[-1]]
            idx = [i for i in idx if i <= real[-1]]
    if len(real) > MOST_WORDS:  # one long line with no break: up to the last clause end that fits
        cut = real[:MOST_WORDS]
        ends = [k for k, i in enumerate(cut) if words[i].ends_clause and k >= 3]
        cut = cut[:ends[-1] + 1] if ends else cut
        idx = [i for i in idx if i <= cut[-1]]
    c = moments.best_in(words, real, quiet, back)
    first = next(i for i in idx if words[i].key not in HESITATIONS)
    last = idx[-1]
    return Line(first, last, _typed(words, idx, back), words[first].start, words[last].end,
                c.index if c else None, "the first line you say")


def strongest_line(words: list[Word], quiet) -> Optional[tuple[int, int]]:
    """The strongest line in the clip, by the shared picker, widened along its own line to last a
    second at the least."""
    back = moments.taken_back(words, quiet)
    span = moments.pick_line(words, quiet, back=back)
    if span is None:
        return None
    a, b = span
    runs = moments.lines(words, quiet)
    run = next((r for r in runs if a in r), [a, b])
    while words[b].end - words[a].start < TEASER[0]:
        grew = False
        if b + 1 in run:
            b, grew = b + 1, True
        elif a - 1 in run:
            a, grew = a - 1, True
        if not grew:
            break
    while words[b].end - words[a].start > TEASER[1] and b > a + 1:
        b -= 1
    return a, b


# ---- a film that plays a stretch first, then the whole clip ---------------------------------------------

class Teaser:
    """The clip with a stretch of it played first: `info`, `sound`, `words`, `quiet` and `face` as
    that film has them, its picture read from the clip's file (seeking back once, where the film
    starts again from the top)."""

    def __init__(self, clip, a: float, b: float):
        from ..core import audio, video

        self.base = clip
        self.path = clip.path
        self.fix = clip.fix
        info = clip.info
        fps = info.rate
        self.fps = fps
        fa, fb = int(round(a * fps)), int(round(b * fps))
        self.a, self.b = fa / fps, fb / fps
        self.lead = fb - fa  # frames of the teaser
        self.n = self.lead + info.frames
        self.index = np.concatenate([np.arange(fa, fb), np.arange(info.frames)]).astype(int)
        self._reader = self._iter = None
        self._at = -1
        self._frame = None
        self.info = dataclasses.replace(info, frames=self.n, duration=self.n / fps, picture=self._picture)
        rate = video.AUDIO_RATE
        joined = audio.join(clip.sound, rate, [(int(round(self.a * rate)), int(round(self.b * rate))),
                                                (0, len(clip.sound))], fade=0.03)
        want = int(round(self.n / fps * rate))
        if len(joined) < want:
            joined = np.concatenate([joined, np.zeros((want - len(joined), joined.shape[1]), np.float32)])
        self.sound = np.ascontiguousarray(joined[:want])
        lead_s = self.lead / fps
        self.words = ([dataclasses.replace(w, start=w.start - self.a, end=w.end - self.a) for w in clip.words
                       if w.start >= self.a - 1e-6 and w.end <= self.b + 1e-6]
                      + [dataclasses.replace(w, start=w.start + lead_s, end=w.end + lead_s) for w in clip.words])
        q = [(max(0.0, s - self.a), min(lead_s, e - self.a)) for s, e in clip.quiet if e > self.a and s < self.b]
        self.quiet = [x for x in q if x[1] > x[0]] + [(s + lead_s, e + lead_s) for s, e in clip.quiet]
        f = clip.face
        from ..core.face import FaceTrack

        self.face = FaceTrack(f.width, f.height, f.fps, f.raw[self.index], f.boxes[self.index],
                              trusted=f.trusted[self.index])
        self.join = lead_s  # where the film starts again

    def _picture(self, i: int) -> np.ndarray:
        from ..core import video

        k = int(self.index[int(np.clip(i, 0, self.n - 1))])
        if k == self._at and self._frame is not None:
            return self._frame
        if self._iter is None or k < self._at or k > self._at + 30:
            if self._reader is not None:
                self._reader.close()
            base = dataclasses.replace(self.base.info, picture=None)
            self._reader = video.FrameReader(base, start=max(0.0, (k - 0.25) / self.fps))
            self._iter = iter(self._reader)
            self._at = k - 1
        while self._at < k:
            nxt = next(self._iter, None)
            if nxt is None:
                break
            self._frame, self._at = nxt, self._at + 1
        self._at = k
        return self._frame


# ---- the text ------------------------------------------------------------------------------------------

@dataclass
class Block:
    sprite: Sprite
    rect: Rect  # where it sits
    lines: list[str]
    px: int


def _wrap(words: list[str], px: int, width: float) -> list[str]:
    lines, cur = [], []
    for w in words:
        trial = " ".join(cur + [w])
        if cur and text.advance(trial, FACE, px) > width:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    if len(lines) > 1 and len(lines[-1].split()) == 1:  # a last line of one word joins the line before
        prev = lines[-2].split()
        if len(prev) > 1:
            lines[-2], lines[-1] = " ".join(prev[:-1]), prev[-1] + " " + lines[-1]
    return lines


def _balance(words: list[str], px: int, width: float) -> list[str]:
    """Lines of even length, as a person would break them: the fewest lines that fit, then the
    narrowest width that keeps that many."""
    lines = _wrap(words, px, width)
    n = len(lines)
    lo, hi = width * 0.45, width
    for _ in range(14):
        mid = (lo + hi) / 2
        if len(_wrap(words, px, mid)) <= n:
            hi = mid
        else:
            lo = mid
    return _wrap(words, px, hi)


def build_block(line: str, style: str, W: int, H: int, scale: float = 1.0) -> Block:
    """The line set as the app sets typed text: each line of words in its own rounded box, the boxes
    touching, centred; or plain words with a soft shadow."""
    from PIL import Image, ImageDraw

    px = max(14, int(round(0.054 * W * scale)))
    words = line.split()
    lines = _balance(words, px, 0.70 * W)
    f = text.font(FACE, px)
    asc, desc = f.getmetrics()
    lh = int(round((asc + desc) * 1.02))
    padx, pady = int(round(0.42 * px)), int(round(0.14 * px))
    widths = [int(math.ceil(text.advance(ln, FACE, px))) for ln in lines]
    bw = max(widths) + 2 * padx
    bh = len(lines) * (lh + 2 * pady)
    k = 3  # drawn larger, brought down: smooth edges
    pad = int(round(0.3 * px))
    img = Image.new("RGBA", ((bw + 2 * pad) * k, (bh + 2 * pad) * k), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    fill, ink = {"box": ((255, 255, 255, 255), (12, 12, 12, 255)),
                 "dark": ((14, 14, 16, 232), (255, 255, 255, 255)),
                 "plain": ((0, 0, 0, 0), (255, 255, 255, 255))}[style]
    big = text.font(FACE, px * k)
    r = 0.36 * (lh + 2 * pady)
    y = pad
    for ln, w in zip(lines, widths):
        x0 = pad + (bw - (w + 2 * padx)) / 2
        if style != "plain":
            d.rounded_rectangle(((x0) * k, y * k, (x0 + w + 2 * padx) * k, (y + lh + 2 * pady + 1) * k),
                                radius=r * k, fill=fill)
        y += lh + 2 * pady
    if style == "plain":  # a soft shadow under plain words
        from PIL import ImageFilter

        sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ds = ImageDraw.Draw(sh)
        y = pad
        for ln, w in zip(lines, widths):
            ds.text(((pad + (bw - w) / 2) * k, (y + pady + 0.04 * px) * k), ln, font=big, fill=(0, 0, 0, 150))
            y += lh + 2 * pady
        img = Image.alpha_composite(img, sh.filter(ImageFilter.GaussianBlur(0.12 * px * k)))
        d = ImageDraw.Draw(img)
    y = pad
    for ln, w in zip(lines, widths):
        d.text(((pad + (bw - w) / 2) * k, (y + pady) * k), ln, font=big, fill=ink)
        y += lh + 2 * pady
    img = img.resize((bw + 2 * pad, bh + 2 * pad), Image.LANCZOS)
    spr = Sprite.from_rgba(np.asarray(img))
    return Block(spr, Rect(0, 0, spr.w, spr.h), lines, px)


def place_block(line: str, style: str, W: int, H: int, head: Optional[Rect], avoid: list[Rect]) -> tuple[Block, str]:
    """The block where it is clear of the head and of anything to avoid, inside the safe zone: above
    the head first (where typed text usually sits), then under the chin, then the lower third; made
    smaller until it fits."""
    zone = SafeZone(W, H)
    z = zone.rect
    gap = 0.025 * H
    for scale in (1.0, 0.9, 0.8, 0.7, 0.6):
        b = build_block(line, style, W, H, scale)
        bw, bh = b.sprite.w, b.sprite.h
        x0 = z.cx - bw / 2
        spots = []
        if head is not None:
            spots.append(("above your head", min(head.y0 - gap, z.y0 + 0.5 * (head.y0 - z.y0) + bh / 2) - bh))
            spots.append(("above your head", head.y0 - gap - bh))
            spots.append(("under your chin", head.y1 + gap))
        spots.append(("in the lower third", z.y1 - 0.01 * H - bh))
        spots.append(("at the top", z.y0 + 0.01 * H))
        for where, y0 in spots:
            r = Rect(x0, y0, x0 + bw, y0 + bh)
            if not r.inside(z, slack=1.0):
                continue
            if head is not None and r.intersects(head):
                continue
            if any(r.intersects(a) for a in avoid):
                continue
            b.rect = r
            return b, where
    raise O.OptionError("there is no room clear of your face for the line, even small: try fewer words (text=\"…\")")


# ---- the looks ------------------------------------------------------------------------------------------

def gaps_after(words, t: float, duration: float) -> list[tuple[float, float]]:
    out = []
    spoken = [w for w in words if w.key not in HESITATIONS]
    for a, b in zip(spoken, spoken[1:]):
        if b.start > a.end and a.end >= t - 1e-6:
            out.append((a.end, b.start))
    if spoken and spoken[-1].end >= t:
        out.append((spoken[-1].end, duration))
    return out


def back_out(words, after: float, duration: float) -> tuple[float, float, str]:
    """A return to the whole frame at `after` or later, in a pause, never under a word."""
    for a, b in gaps_after(words, after, duration):
        a = max(a, after)
        room = b - a
        if room >= 0.22:
            return a + 0.03, a + 0.03 + min(0.45, room - 0.06), "smooth"
        if room >= 0.08:
            return a + room / 2, a + room / 2, "cut"
    return duration, duration, "cut"


@dataclass
class Look:
    path: zoom.CameraPath
    iris: Optional[tuple[float, float, tuple[float, float]]] = None  # start, end, centre
    flash: Optional[float] = None  # seconds: a white flash peaks here
    shake: Optional[float] = None  # seconds: the jolt starts here
    sound_at: Optional[float] = None
    notes: str = ""


def make_look(look: str, line: Line, words, clip, fr, fps: float, t_start: float, duration: float) -> Look:
    """The camera, iris and flash for a look. `t_start` is where the film (or the teaser) begins."""
    W, H = fr.out_w, fr.out_h
    full = (0.0, 0.0, float(W), float(H))
    n = clip.info.frames
    key_at = words[line.key].start if line.key is not None else line.start
    end_hold = max(line.end, key_at + 1.0)
    first = int(t_start * fps)
    if look == "text-box":
        return Look(zoom.CameraPath((W, H), []), sound_at=line.start)
    if look == "snap-zoom":
        o0, o1, kind = back_out(words, end_hold, duration)
        head, centre = zoom.aim(clip.face, fr, int(max(0, key_at - 0.2) * fps), min(n - 1, int(o1 * fps) + 1))
        win, z, note = zoom.punch_window((W, H), 1.25, head, centre)
        moves = [zoom.Move(max(0.0, key_at - 0.15), key_at, win, "snap", 0.14)]
        if o0 < duration - 0.15:
            moves.append(zoom.Move(o0, o1, full, kind))
        return Look(zoom.CameraPath((W, H), moves), sound_at=key_at, notes=f"{z:.2f}x on “{moments.shown(words[line.key])}”"
                    if line.key is not None else f"{z:.2f}x")
    if look == "whoosh-in":
        head, centre = zoom.aim(clip.face, fr, first, min(n - 1, first + int(0.7 * fps)))
        win, z, note = zoom.punch_window((W, H), 1.35, head, centre)
        moves = [zoom.Move(t_start, t_start, win, "cut"), zoom.Move(t_start + 1e-3, t_start + 0.55, full, "snap")]
        return Look(zoom.CameraPath((W, H), moves), sound_at=t_start + 0.06, notes=f"from {z:.2f}x")
    if look == "iris-open":
        head, centre = zoom.aim(clip.face, fr, first, min(n - 1, first + int(0.8 * fps)))
        win, z, _ = zoom.punch_window((W, H), 1.08, head, centre)
        moves = [zoom.Move(t_start, t_start, win, "cut"), zoom.Move(t_start + 1e-3, t_start + 1.1, full, "slow")]
        c = (centre if centre is not None else (W / 2, H * 0.42))
        # the circle is drawn on the framed picture: where the face is in the first framing
        x, y, w, h = win
        cc = ((c[0] - x) * W / w, (c[1] - y) * H / h)
        return Look(zoom.CameraPath((W, H), moves), iris=(t_start, t_start + 0.8, cc), sound_at=t_start + 0.1)
    # shake-flash: a little in from the first frame, a jolt and a flash on the key word, back out later
    o0, o1, kind = back_out(words, end_hold, duration)
    head, centre = zoom.aim(clip.face, fr, first, min(n - 1, int(o1 * fps) + 1))
    win, z, _ = zoom.punch_window((W, H), 1.12, head, centre)
    if z < 1.07:  # a jolt needs a little room to move in: a touch closer, the crown still kept
        z = 1.07
        c = centre if centre is not None else (W / 2, H * 0.45)
        win = zoom.place_window((W, H), z, head, c[0], c[1])
    moves = [zoom.Move(t_start, t_start, win, "cut")]
    if o0 < duration - 0.15:
        moves.append(zoom.Move(o0, o1, full, kind))
    return Look(zoom.CameraPath((W, H), moves), flash=key_at, shake=key_at, sound_at=key_at,
                notes=f"held at {z:.2f}x, the jolt on “{moments.shown(words[line.key])}”" if line.key is not None else "")


JOLT = ((0.030, -0.018), (-0.026, 0.014), (0.016, 0.020), (-0.008, -0.010))  # four hard steps, as a share of the frame
JOLT_STEP = 0.05  # seconds each step holds: a shake that eases reads as a wobble


def jolt(t: float, at: Optional[float], W: int) -> tuple[float, float]:
    if at is None or not (at <= t < at + JOLT_STEP * len(JOLT)):
        return 0.0, 0.0
    dx, dy = JOLT[int((t - at) / JOLT_STEP)]
    return dx * W, dy * W


# ---- sound ------------------------------------------------------------------------------------------------

class Sounds:
    def __init__(self, base_sound: np.ndarray, kind: str, at: float, has_audio: bool):
        voice = sound.integrated(sound.blocks(base_sound)) if has_audio else sound.SILENCE_DB
        self.ref = voice if voice > -70 else -20.0
        self.kind, self.at = kind, at

    def mix(self, base: np.ndarray, rate: int) -> np.ndarray:
        from ..core import loudness

        x, anchor = sound.library(self.kind)
        stem = np.zeros_like(base)
        sound.lay(stem, synth.at_level(x, self.ref - UNDER_VOICE), self.at, rate, anchor=anchor)
        out = base + stem
        ceiling = max(-1.0, loudness.true_peak_db(base) + 0.05)
        if loudness.true_peak_db(out) > ceiling:
            out, _ = loudness.limit(out, ceiling)
        return out.astype(np.float32)


# ---- the effect -------------------------------------------------------------------------------------------

def _caption_cards(clip) -> list[tuple[float, float, Rect]]:
    """Captions already drawn into the video, from the plan beside it: kept clear of."""
    p = Path(clip.path).with_suffix(".plan.json")
    try:
        plan = json.loads(p.read_text())
    except (OSError, ValueError):
        return []
    out = []
    for c in plan.get("cards") or []:
        try:
            out.append((float(c["start"]), float(c["end"]), Rect.of(c["rect"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from ..core import framing as framing_mod
    from ..core.timeline import Timeline
    from . import Job, PlanError

    notes: list[str] = []
    look_name = opts["look"]
    if plan is not None:
        kept = plan.get("options") or {}
        opts = dict(opts, **{k: kept[k] for k in ("look", "open", "style", "sound") if k in kept})
        look_name = opts["look"]
    src = clip
    teaser = None
    if opts["open"] == "strongest":
        span = None
        if plan is not None and plan.get("teaser"):
            try:
                span = (float(plan["teaser"]["from"]), float(plan["teaser"]["to"]))
            except (KeyError, TypeError, ValueError):
                raise PlanError("the plan's teaser needs from and to (seconds)") from None
        else:
            got = strongest_line(clip.words, clip.quiet)
            if got is None:
                raise O.OptionError("no line in this clip stands out to play first; use open=first")
            a, b = got
            ws = clip.words
            # cut in the quiet either side, never into a word
            lo = ws[a - 1].end if a > 0 else 0.0
            hi = ws[b + 1].start if b + 1 < len(ws) else clip.info.frames / clip.info.rate
            span = (max(lo, ws[a].start - 0.12), min(hi, ws[b].end + 0.18))
            first_spoken = next((w.start for w in ws if w.key not in HESITATIONS), 0.0)
            if ws[a].start - first_spoken < 0.5:
                notes.append("your strongest line is already your first, so nothing is played twice")
                span = None
        if span is not None:
            teaser = Teaser(clip, *span)
            clip = teaser
    info = clip.info
    fps = info.rate
    n = info.frames
    duration = n / fps
    fr = framing_mod.plan(info.width, info.height, fps, n, clip.face, shape=opts["frame"])
    W, H = fr.out_w, fr.out_h
    words, quiet = clip.words, clip.quiet

    if teaser is not None:
        tw = [i for i, w in enumerate(words) if w.end <= teaser.join + 1e-6]
        back = moments.taken_back(src.words, src.quiet)
        tline = [w for w in words[:len(tw)]]
        key = moments.best_in(words, tw, quiet)
        line = Line(tw[0], tw[-1], _typed(words, tw, moments.TakenBack()) if tw else "", words[tw[0]].start,
                    words[tw[-1]].end, key.index if key else None, "your strongest line, played first")
        del back, tline
    else:
        line = first_line(words, quiet)
        if line is None:
            raise O.OptionError("no words were heard to build the hook on; give the words with text=\"…\"")
    shown = line.text if opts["text"] == "" else ("" if opts["text"].strip().lower() == "off" else opts["text"].strip())
    t_land = math.floor(line.start * fps + 1e-6) / fps  # the frame the first word starts on
    t_off = min(max(line.end + AFTER, t_land + SHOWN[0]), t_land + SHOWN[1])
    if teaser is not None:
        t_off = teaser.join  # the line goes with the teaser
    t_off = min(t_off, duration)

    film_start = 0.0
    lk = make_look(look_name, line, words, clip, fr, fps, film_start, duration)
    path = lk.path

    # the head as the camera shows it, over every frame the line is up
    heads = []
    for j in range(int(t_land * fps), int(math.ceil(t_off * fps))):
        h = clip.face.head(min(n - 1, j))
        if h is None:
            continue
        r = fr.to_output(h, j)
        x, y, w, hh = path.at(j / fps)
        s = W / w
        dx, dy = jolt(j / fps, lk.shake, W)
        heads.append(Rect((r.x0 - x) * s + dx, (r.y0 - y) * s + dy, (r.x1 - x) * s + dx, (r.y1 - y) * s + dy))
    head = Rect.around(heads)
    if head is not None:
        head = head.grow(0.02 * W, 0.01 * H, 0.02 * W, 0.01 * H)
    lead = teaser.join if teaser is not None else 0.0
    avoid = [r for a, b, r in _caption_cards(src) if a + lead < t_off and b + lead > t_land]
    block, where = place_block(shown, opts["style"], W, H, head, avoid) if shown else (None, "no words")

    def draw(frame: np.ndarray, j: int, t: float) -> None:
        img = frame
        dx, dy = jolt(t, lk.shake, W)
        if path.moves or dx or dy:
            x, y, w, h = path.at(t)
            if dx or dy:
                s = w / W
                x = min(max(0.0, x - dx * s), W - w)
                y = min(max(0.0, y - dy * s), H - h)
                out = zoom.crop(img, (x, y, w, h))
            else:
                out = path.frame(img, t, fps)
            frame[:] = out
        if lk.iris is not None:
            a, b, (cx, cy) = lk.iris
            if t < b:
                u = float(zoom.curve("slow", (t - a) / (b - a)))
                far = math.hypot(max(cx, W - cx), max(cy, H - cy))
                r = far * u
                yy, xx = np.ogrid[0:H, 0:W]
                d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                soft = max(2.0, 0.012 * W)
                m = np.clip((r - d) / soft + 0.5, 0, 1).astype(np.float32)
                frame[:] = (frame.astype(np.float32) * m[..., None]).astype(np.uint8)
        if lk.flash is not None:
            k = t - lk.flash
            if -0.02 <= k < 0.22:
                v = 0.85 if k < 0.03 else 0.85 * (1 - (k - 0.03) / 0.19) ** 2
                frame[:] = (frame.astype(np.float32) * (1 - v) + 255 * v).astype(np.uint8)
        if block is not None and t_land - 1e-6 <= t < t_off:
            k = t - t_land
            s = 1.0 + 0.05 * max(0.0, 1 - k / 0.1) ** 2  # a small settle as it lands, whole from the first frame
            fade = 1.0 if teaser is not None else min(1.0, max(0.0, (t_off - t) / 0.12))
            spr = block.sprite.scaled(s) if s > 1.001 else block.sprite
            over(frame, spr, block.rect.cx - spr.w / 2, block.rect.cy - spr.h / 2, fade)

    # checks: over the face, outside the zone, the crown
    over_face = 0
    if block is not None:
        for j in range(int(t_land * fps), int(math.ceil(t_off * fps))):
            h = clip.face.head(min(n - 1, j))
            if h is None:
                continue
            r = fr.to_output(h, j)
            x, y, w, hh = path.at(j / fps)
            s = W / w
            hr = Rect((r.x0 - x) * s, (r.y0 - y) * s, (r.x1 - x) * s, (r.y1 - y) * s)
            if hr.intersects(block.rect):
                over_face += 1
    crown_cut = 0
    for j in range(n):
        x, y, w, hh = path.at(j / fps)
        if w >= W - 0.5:
            continue
        h = clip.face.head(j)
        if h is None:
            continue
        r = fr.to_output(h, j)
        if r.y0 > 0.01 * H and (r.y0 - y) * (H / hh) < 0:
            crown_cut += 1
    zone = SafeZone(W, H).rect
    outside = 0 if block is None or block.rect.inside(zone, slack=1.0) else int((t_off - t_land) * fps)

    sounds = None
    if opts["sound"] == "auto" and lk.sound_at is not None:
        sounds = Sounds(clip.sound, OWN_SOUND[look_name], lk.sound_at, src.info.has_audio)

    checks = {
        "line": f"“{line.text}”, {line.why}, {line.start:.2f}–{line.end:.2f} s",
        "shown": f"“{shown}”" if shown else "no words",
        "lands": f"{t_land:.3f} s (the frame its first word starts on), up until {t_off:.2f} s",
        "look": look_name + (f": {lk.notes}" if lk.notes else ""),
        "placed": where + (f", {len(block.lines)} line(s) at {block.px} px" if block else ""),
        "frames_over_face": over_face,
        "frames_outside_safe_zone": outside,
        "frames_crown_cut": crown_cut,
        "sound": f"{OWN_SOUND[look_name]} at {lk.sound_at:.2f} s, {UNDER_VOICE:g} dB under your voice" if sounds else "the clip's own only",
        "notes": notes,
    }
    if teaser is not None:
        checks["open"] = (f"your strongest line ({teaser.a:.2f}–{teaser.b:.2f} s) plays first, then the film from the "
                          f"start at {teaser.join:.2f} s; the film is {teaser.join:.2f} s longer")

    sheet = sorted({round(min(duration - 1.5 / fps, max(0.0, t)), 3) for t in
                    (0.0, t_land, t_land + 0.12, (t_land + t_off) / 2,
                     (lk.flash + 0.04) if lk.flash is not None else (t_land + 0.6), t_off + 0.3)})
    while len(sheet) < 6:
        sheet.append(round(min(duration - 0.1, sheet[-1] + 0.5), 3))
    out_plan = {
        "effect": NAME, "clip": src.path, "options": opts,
        "output": {"width": W, "height": H, "fps": round(fps, 3), "seconds": round(duration, 3), "framing": fr.note},
        "line": {"text": line.text, "shown": shown, "from": round(line.start, 3), "to": round(line.end, 3),
                 "lands": round(t_land, 3), "off": round(t_off, 3),
                 "key": moments.shown(words[line.key]) if line.key is not None else None},
        "block": {"rect": block.rect.as_list(), "lines": block.lines, "px": block.px, "where": where} if block else None,
        "sheet_times": sorted(sheet)[:6],
    }
    if teaser is not None:
        out_plan["teaser"] = {"from": round(teaser.a, 4), "to": round(teaser.b, 4), "join": round(teaser.join, 4)}

    def stills() -> list[float]:
        return out_plan["sheet_times"][:5]

    return Job(effect=NAME, clip=clip, timeline=Timeline.whole(duration), framing=fr, draw=draw, plan=out_plan,
               checks=checks, still_times=stills, sound=sounds.mix if sounds is not None else None)
