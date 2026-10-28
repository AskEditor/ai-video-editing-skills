"""glitch: the frame breaks apart for a split second on a word or a cut (the colours pulled apart,
slices of the picture shoved sideways, blocks gone to mush, lines of static), then is whole again.
An optional bar counts up first ("Uninstalling doubt…", or a progress bar) and breaks with it.

What it does, in order:

1. **Where.** `at` names it: the strongest word in the clip by the shared picker (`core/moments`,
   so never a word taken back, a filler or an aside), a word or words said, a second, or `cuts`
   for the clip's own changes of shot. A long clip gets one for about every twelve seconds, at
   least six seconds apart, three at most.
2. **When.** The worst frame is the first frame of the word (its start corrected for the
   transcriber's lateness by the core), or the first frame of the new shot. The break builds for a
   few frames before it and dies away after it, `strength` deciding how long: 0.24 s, 0.36 s or
   0.52 s, never longer than 0.6 s, so it never sits on the voice for long. The frames before and
   after it are untouched, and over the face the break is a third as strong, so the face reads all
   the way through.
3. **The break** is drawn by `core/glitch.py` (shared with `transitions`): bands torn sideways, some
   from a moment earlier in the clip (the stale picture a broken stream drags along), blocks gone
   coarse, the colours split, scan lines. On top of it here: the whole frame cut into slices that
   slide against each other, a block of the stale picture dragged as a broken stream drags it,
   and lines of static.
4. **The bar** (`bar=uninstall` or `progress`), on frosted glass above the head (else under the chin),
   inside the safe zone: it arrives about 1.2 s before, fills in uneven steps (as a real one does)
   and reaches 100% on the worst frame. It is torn with the frame; `uninstall` is gone when the
   frame comes back, `progress` stays a moment to say it is done.
5. **The sound.** A digital crackle made in code, thickest on the worst frame, under the voice;
   with the bar, a soft tick as it appears. `sound=off` keeps the clip's sound exactly.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core import glitch as glitchmod
from ..core import items, moments, shots
from ..core import sound as soundmod
from ..core import synth
from ..core import text as textmod
from ..core.compose import Sprite, over
from ..core.layout import Rect, SafeZone
from . import options as O

NAME = "glitch"
STRENGTHS = {"light": (0.24, 0.62), "medium": (0.36, 0.84), "heavy": (0.52, 1.0)}  # seconds, how broken
BARS = ("off", "uninstall", "progress")
LONGEST = 0.6  # seconds: a glitch never lasts longer
APART = 6.0  # seconds between two glitches the pack chooses
EVERY = 12.0  # one glitch the pack chooses for about this many seconds of clip
MOST = 3
FACE_SHARE = 0.33  # how strong the break is over the face, against the rest of the frame
STALE = 0.20  # seconds before the break: the picture a broken stream drags along
LEAD = 1.2  # seconds the bar is up before the break
UNDER = 9.0  # dB: the crackle under the voice's loudness
TICK_UNDER = 22.0

OPTIONS = {
    "at": O.Text("where it breaks. Empty or auto: the strongest word. Or a word or words you say "
                 "(\"production\"), a time in seconds, or cuts (each change of shot); several separated by ;"),
    "strength": O.Choice(tuple(STRENGTHS), "light: a quick flicker (0.24 s) · medium: a clear break (0.36 s) · "
                                           "heavy: the frame falls apart (0.52 s). Never longer than 0.6 s"),
    "bar": O.Choice(BARS, "off: the break alone · uninstall: \"Uninstalling …\" fills to 100% and breaks · "
                          "progress: a loading bar fills to 100% and breaks"),
    "sound": O.Switch("a digital crackle under your voice on the break (and a tick as the bar appears): on or off"),
}

MATRIX = [
    ("A", "A", {}),
    ("B", "B", {}),
    ("C", "C", {}),
    ("D", "D", {}),
    ("B-uninstall", "B", {"bar": "uninstall", "at": "production"}),
    ("D-progress", "D", {"bar": "progress", "strength": "heavy", "at": "city"}),
    ("A-light", "A", {"strength": "light", "at": "campaign"}),
    ("33", "33", {}),
    ("33-cuts", "33", {"at": "cuts", "strength": "light"}),
]


def frame_of(t: float, fps: float) -> int:
    """The first frame that shows a moment: the first one starting at or after it."""
    return int(math.ceil(t * fps - 1e-6))


def _seed(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def _smooth(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


# ---- where it breaks -------------------------------------------------------------------------------

@dataclass
class Hit:
    t: float  # seconds: the word's start, or the cut
    word: str  # the word it breaks on, as heard ("" on a cut with nothing said)
    why: str
    frame: int = 0  # the worst frame
    first: int = 0  # the burst's first and last frames (both clean)
    last: int = 0


def _bare(s: str) -> str:
    return s.strip().strip(moments.EDGE_PUNCT).strip()


def _introducing(words, i: int) -> bool:
    """The speaker saying who they are ("it's Eben"): a name there is not the point of the line."""
    return i > 0 and words[i - 1].key in ("it's", "i'm", "called") and words[i].clean[:1].isupper()


def auto_hits(clip, seconds: float) -> list[Hit]:
    words, quiet = clip.words, clip.quiet
    back = moments.taken_back(words, quiet)
    hero, runners = moments.pick_word(words, quiet, back)
    cands = ([hero] if hero is not None else []) + list(runners)
    want = max(1, min(MOST, int(round(seconds / EVERY))))
    out: list[Hit] = []
    for c in cands:
        w = c.word
        if w.start < 0.30 or w.start > seconds - 0.40 or _introducing(words, c.index):
            continue
        if any(abs(w.start - h.t) < APART for h in out):
            continue
        out.append(Hit(w.start, w.clean, "the strongest word" if c is hero else "a strong word, apart from the first"))
        if len(out) >= want:
            break
    if not out:
        for t in shots.cuts(clip)[:1]:
            out.append(Hit(t, _word_after(words, t), "the clip's own cut (no word stands out)"))
    return sorted(out, key=lambda h: h.t)


def _word_after(words, t: float) -> str:
    w = next((x for x in words if x.start >= t - 0.05 and x.start - t < 0.8), None)
    return w.clean if w is not None else ""


def asked_hits(clip, at: str, seconds: float) -> list[Hit]:
    words = clip.words
    back = moments.taken_back(words, clip.quiet)
    out: list[Hit] = []
    for p in [x.strip() for x in at.replace("|", ";").split(";") if x.strip()]:
        if p.lower() in ("cuts", "cut", "every cut", "the cuts"):
            found = shots.cuts(clip)
            if not found:
                raise O.OptionError("at: this clip has no cut of its own; name a word or a time instead")
            for t in found[:8]:
                if not any(abs(t - h.t) < 1.5 for h in out):
                    out.append(Hit(t, _word_after(words, t), "the clip's own cut"))
            continue
        try:
            t = float(p.rstrip("s"))
        except ValueError:
            t = None
        if t is not None:
            if not 0.0 <= t <= seconds:
                raise O.OptionError(f"at: {t:g} s is outside the clip (it is {seconds:.1f} s long)")
            when, said = items.snap(words, t)
            out.append(Hit(when, said, f"at {t:g} s" + (f", on “{said}”" if said else "")))
            continue
        spans = items.name_spans(words, p)
        if not spans:
            raise O.OptionError(f"at: “{p}” is not said in this video. Read the words with "
                                f"`python -m fx.run words <video>`, or give a time in seconds")
        kept = [s for s in spans if not back.holds(words, s[0], s[1])] or spans
        i, j, _ = min(kept[:3], key=lambda s: words[s[0]].start) if len(kept) > 1 and kept[0][2] == kept[-1][2] \
            else kept[0]
        out.append(Hit(words[i].start, " ".join(w.clean for w in words[i:j + 1]), f"on “{p}”"))
    return sorted(out, key=lambda h: h.t)


def time_hits(hits: list[Hit], fps: float, n_frames: int, length: float) -> list[Hit]:
    """Each burst's frames: the worst on the first frame of its moment, 40% of it before."""
    n = max(5, int(round(min(LONGEST, length) * fps)))
    for h in hits:
        h.frame = min(n_frames - 2, max(1, frame_of(h.t, fps)))
        h.first = max(0, h.frame - int(round(0.4 * (n - 1))))
        h.last = min(n_frames - 1, h.first + n - 1)
    keep: list[Hit] = []
    for h in hits:  # two bursts never run into each other
        if keep and h.first <= keep[-1].last + 2:
            continue
        keep.append(h)
    return keep


# ---- how broken each frame is ------------------------------------------------------------------------

STUTTER = (0.55, 1.0, 0.45, 0.9, 0.7, 1.0, 0.35, 0.8)  # the signal catching and dropping, frame to frame


def broken(h: Hit, j: int, scale: float) -> float:
    """How broken frame j is for burst h (0 clean, 1 at its worst): rising fast to the worst frame,
    dying away after it, stuttering; its first and last frames clean."""
    if j <= h.first or j >= h.last:
        return 0.0
    if j == h.frame:
        return scale
    if j < h.frame:
        q = (j - h.first) / max(1, h.frame - h.first)
        env = q ** 1.3
    else:
        q = (h.last - j) / max(1, h.last - h.frame)
        env = q ** 0.8
    return scale * env * STUTTER[(j - h.first) % len(STUTTER)]


# ---- the break, frame by frame ---------------------------------------------------------------------

def slices(img: np.ndarray, e: float, rng: np.random.Generator) -> np.ndarray:
    """The whole frame cut into slices of uneven height, each shoved sideways against the next."""
    h, w = img.shape[:2]
    out = img.copy()
    y = 0
    while y < h:
        sh = int(h * rng.uniform(0.03, 0.16))
        dx = int(round(rng.choice((-1, 1)) * rng.uniform(0.01, 0.075) * w * e))
        out[y:y + sh] = np.roll(img[y:y + sh], dx, axis=1)
        y += max(2, sh)
    return out


def drag(img: np.ndarray, stale: np.ndarray, e: float, rng: np.random.Generator) -> np.ndarray:
    """A block of the stale picture dragged on in steps of whole cells, as a broken stream drags it,
    its colours coarse."""
    h, w = img.shape[:2]
    cell = max(8, int(round(w / 54)))
    bw = int(w * rng.uniform(0.30, 0.65))
    bh = int(h * rng.uniform(0.08, 0.20) * (0.6 + 0.6 * e))
    bw, bh = max(cell, bw - bw % cell), max(cell, bh - bh % cell)
    x0 = int(rng.integers(0, max(1, w - bw)))
    y0 = int(rng.integers(0, max(1, h - bh)))
    dx = int(rng.integers(-3, 4)) * cell
    dy = int(rng.integers(-2, 3)) * cell
    sx0, sy0 = min(max(0, x0 + dx), w - bw), min(max(0, y0 + dy), h - bh)
    block = stale[sy0:sy0 + bh, sx0:sx0 + bw]
    levels = 6 if e > 0.7 else 10
    block = (np.round(block.astype(np.float32) / 255 * levels) / levels * 255).astype(np.uint8)
    out = img.copy()
    out[y0:y0 + bh, x0:x0 + bw] = block
    return out


def static(img: np.ndarray, e: float, rng: np.random.Generator) -> np.ndarray:
    """Lines of static across the frame, and one band where the picture rolls brighter."""
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    u = max(1, int(round(h / 960)))
    for _ in range(3 + int(round(9 * e))):
        th = int(rng.integers(1, 4)) * u
        y0 = int(rng.integers(0, h - th))
        noise = rng.integers(0, 256, (th, w, 1)).astype(np.float32)
        a = rng.uniform(0.35, 0.8)
        out[y0:y0 + th] = out[y0:y0 + th] * (1 - a) + noise * a
    bh = int(h * 0.035)
    y0 = int(rng.integers(0, h - bh))
    ramp = np.sin(np.linspace(0, np.pi, bh))[:, None, None].astype(np.float32)
    out[y0:y0 + bh] = out[y0:y0 + bh] * (1 + 0.45 * e * ramp) + 18 * e * ramp
    return np.clip(out, 0, 255).astype(np.uint8)


def break_frame(img: np.ndarray, stale: np.ndarray, e: float, seed: int) -> np.ndarray:
    """One frame broken at strength e: the shared tear, then slices, a dragged block and static."""
    rng = np.random.default_rng(seed)
    out = glitchmod.tear(img.copy(), stale, e, rng)
    if e > 0.35:
        out = slices(out, e, rng)
    if e > 0.5:
        out = drag(out, stale, e, rng)
    out = static(out, e, rng)
    return out


def face_weight(shape: tuple[int, int], face: Optional[Rect]) -> Optional[np.ndarray]:
    """1 over the face (a soft oval a little larger than its box), 0 away from it."""
    if face is None:
        return None
    import cv2

    h, w = shape
    k = 4  # worked out at a quarter size, then brought up
    m = np.zeros((h // k + 1, w // k + 1), np.float32)
    cx, cy = face.cx / k, face.cy / k
    ax, ay = face.w * 0.75 / k, face.h * 0.80 / k
    cv2.ellipse(m, (int(cx), int(cy)), (max(1, int(ax)), max(1, int(ay))), 0, 0, 360, 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), max(1.0, face.w * 0.12 / k))
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)[..., None]


# ---- the bar ---------------------------------------------------------------------------------------

FILL = ((0.0, 0.0), (0.10, 0.0), (0.28, 0.36), (0.42, 0.39), (0.60, 0.71), (0.76, 0.74), (1.0, 1.0))
ACCENT = {"uninstall": (255, 92, 84), "progress": (88, 206, 255)}


def fill_at(q: float) -> float:
    """How full the bar is at q (0 when it appears, 1 on the break): uneven steps, as a real one fills."""
    q = min(1.0, max(0.0, q))
    for (a, va), (b, vb) in zip(FILL, FILL[1:]):
        if q <= b:
            return va + (vb - va) * _smooth((q - a) / max(1e-6, b - a))
    return 1.0


class Bar:
    def __init__(self, kind: str, hit: Hit, W: int, H: int, fps: float):
        self.kind, self.hit, self.W, self.H, self.fps = kind, hit, W, H, fps
        self.s = W / 1080.0
        self.t_in = max(0.0, hit.frame / fps - LEAD)
        self.t_break = hit.frame / fps
        self.t_gone = (hit.last + 1) / fps if kind == "uninstall" else (hit.last + 1) / fps + 0.6
        self.t_out = self.t_gone + (0.0 if kind == "uninstall" else 0.18)
        self.k = 1.0
        self.x = self.y = 0.0
        self.pane = None

    def size(self, k: float) -> tuple[float, float]:
        return 700 * self.s * k, 138 * self.s * k

    def labels(self) -> tuple[str, str]:
        word = _bare(self.hit.word)
        if self.kind == "uninstall":
            return (f"Uninstalling {word}…" if word else "Uninstalling…"), "Uninstalled"
        return (f"Loading {word}…" if word else "Loading…"), "Done"

    def reach(self) -> tuple[float, float, float, float]:
        from ..core import glass

        return glass.Pane.reach(max(0.5, self.s * self.k))

    def rect(self) -> Rect:
        w, h = self.size(self.k)
        return Rect(self.x, self.y, self.x + w, self.y + h)

    def drawn(self) -> Rect:
        return self.rect().grow(*self.reach())

    def place(self, zone: SafeZone, head: Optional[Rect], face: Optional[Rect]) -> Optional[str]:
        """Above the head, else under the chin, else low in the safe zone: the largest that fits,
        down to 70%, clear of the head and inside the safe zone with its shadow."""
        z = zone.rect
        gap = 0.016 * self.H
        for k in [1.0 - 0.05 * i for i in range(7)]:
            self.k = k
            w, h = self.size(k)
            left, top, right, bottom = self.reach()
            room = Rect(z.x0 + left, z.y0 + top, z.x1 - right, z.y1 - bottom)
            if w > room.w:
                continue
            cx = head.cx if head is not None else room.cx
            x = min(max(cx - w / 2, room.x0), room.x1 - w)
            spots = []
            if head is None:
                spots.append(("top", room.y0 + 0.08 * self.H))
            else:
                spots.append(("top", head.y0 - gap - h))
                spots.append(("below", max(head.y1, face.y1 if face is not None else head.y1) + gap))
                spots.append(("low", room.y1 - h))
            for where, y in spots:
                r = Rect(x, y, x + w, y + h)
                if r.y0 < room.y0 - 0.5 or r.y1 > room.y1 + 0.5:
                    continue
                if head is not None and r.intersects(head):
                    continue
                self.x, self.y = float(round(x)), float(round(y))
                return where
        return None

    def build(self, behind: list[np.ndarray]) -> None:
        from ..core import glass

        w, h = self.size(self.k)
        u = self.s * self.k
        self.pane = glass.Pane(int(math.ceil(w)), int(math.ceil(h)), radius=26 * u, unit=max(0.5, u))
        self.pane.tone_over(behind, self.x, self.y)
        self.u = u
        self.w, self.h = w, h
        pad = 34 * u
        self.pad = pad
        self.label_px = int(round(33 * u))
        self.pct_px = int(round(31 * u))
        self.track_h = max(4.0, 13 * u)
        say, done = self.labels()
        pct_w = textmod.advance("100%", "plex-mono-bold", self.pct_px)
        room = w - 2 * pad - pct_w - 22 * u
        while textmod.advance(say, "poppins-semibold", self.label_px) > room and " " in say:
            say = say.rsplit(" ", 1)[0].rstrip("…") + "…"
        self.say = _words(say, "poppins-semibold", self.label_px, (255, 255, 255))
        self.done = _words(done, "poppins-semibold", self.label_px, (255, 255, 255))
        self.pcts = {}
        self.pct_w = pct_w
        self.track_y = h - pad - self.track_h
        self.text_y = pad + 0.5 * (self.track_y - pad) + textmod.cap_height("poppins-semibold", self.label_px) * 0.5 - 6 * u
        tw = w - 2 * pad
        self.track = _pill(tw, self.track_h, (255, 255, 255, 60))
        self.fills: dict[int, Sprite] = {}
        self.track_w = tw

    def _pct(self, v: int):
        if v not in self.pcts:
            self.pcts[v] = _words(f"{v}%", "plex-mono-bold", self.pct_px, (255, 255, 255))
        return self.pcts[v]

    def _fill(self, share: float) -> Optional[Sprite]:
        w = int(round(self.track_w * share))
        if w < 2:
            return None
        w = max(int(math.ceil(self.track_h)), w)
        if w not in self.fills:
            self.fills[w] = _pill(w, self.track_h, ACCENT[self.kind] + (255,))
        return self.fills[w]

    def alpha(self, t: float) -> float:
        if t < self.t_in or t >= self.t_out + 1.0 / self.fps:
            return 0.0
        a = _smooth((t - self.t_in) / 0.18)
        if self.kind == "progress" and t >= self.t_gone:
            a *= 1.0 - _smooth((t - self.t_gone) / 0.18)
        if self.kind == "uninstall" and t >= self.t_gone:
            return 0.0
        return a

    def draw(self, frame: np.ndarray, t: float) -> None:
        a = self.alpha(t)
        if a <= 0.0:
            return
        rise = (1.0 - _smooth((t - self.t_in) / 0.18)) * 10 * self.u
        x, y = self.x, self.y + rise
        self.pane.draw(frame, x, y, opacity=a, frost=min(1.0, a * 1.2))
        q = (t - self.t_in) / max(1e-6, self.t_break - self.t_in)
        share = fill_at(q)
        after = t >= (self.hit.last + 1) / self.fps
        sp, ox, oy = (self.done if after else self.say)
        over(frame, sp, x + self.pad - ox, y + self.text_y - oy, a)
        v = int(math.floor(share * 100 + 1e-6))
        ps, pox, poy = self._pct(v)
        right = x + self.w - self.pad - textmod.advance(f"{v}%", "plex-mono-bold", self.pct_px)
        over(frame, ps, right - pox, y + self.text_y - poy, a)
        over(frame, self.track, x + self.pad - 1, y + self.track_y - 1, a)
        f = self._fill(share)
        if f is not None:
            over(frame, f, x + self.pad - 1, y + self.track_y - 1, a)


def _words(s: str, face: str, px: int, colour) -> tuple[Sprite, float, float]:
    from PIL import Image, ImageDraw

    f = textmod.font(face, px)
    l, t, r, b = f.getbbox(s, anchor="ls")
    pad = 3
    img = Image.new("RGBA", (int(r - l + 2 * pad), int(b - t + 2 * pad)), (0, 0, 0, 0))
    ox, oy = pad - l, pad - t
    ImageDraw.Draw(img).text((ox, oy), s, font=f, fill=tuple(colour) + (255,), anchor="ls")
    return Sprite.from_rgba(np.asarray(img)), ox, oy


def _pill(w: float, h: float, fill) -> Sprite:
    from PIL import Image, ImageDraw

    k = 4
    W, H = int(math.ceil(w)) + 2, int(math.ceil(h)) + 2
    big = Image.new("RGBA", (W * k, H * k), (0, 0, 0, 0))
    ImageDraw.Draw(big).rounded_rectangle((k, k, (1 + w) * k - 1, (1 + h) * k - 1), radius=h / 2 * k, fill=fill)
    return Sprite.from_rgba(np.asarray(big.resize((W, H), Image.LANCZOS)))


# ---- one run ---------------------------------------------------------------------------------------

def hits_from_plan(plan: dict, seconds: float) -> list[Hit]:
    from . import PlanError

    raw = plan.get("glitches")
    if not isinstance(raw, list) or not raw:
        raise PlanError("the plan has no glitches; run once without --plan to get one to edit")
    out = []
    for k, g in enumerate(raw):
        try:
            t = float(g["t"])
        except (KeyError, TypeError, ValueError):
            raise PlanError(f"glitch {k + 1} needs a time in seconds (t)") from None
        if not 0 <= t <= seconds:
            raise PlanError(f"glitch {k + 1} is outside the clip ({t})")
        out.append(Hit(t, str(g.get("word") or ""), "from the plan"))
    return sorted(out, key=lambda h: h.t)


def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from ..core import framing as framing_mod
    from ..core import video
    from ..core.timeline import Timeline
    from . import Job
    from .tier_list import head_track

    info = clip.info
    fps = info.rate
    fr = framing_mod.plan(info.width, info.height, fps, info.frames, clip.face, shape="keep")
    tl = Timeline.whole(info.frames / fps)
    W, H = fr.out_w, fr.out_h
    n = tl.output_frames(fps)
    seconds = n / fps
    notes: list[str] = []
    length, scale = STRENGTHS[opts["strength"]]
    if plan is not None:
        hits = hits_from_plan(plan, seconds)
    else:
        at = opts["at"].strip()
        hits = auto_hits(clip, seconds) if at.lower() in ("", "auto") else asked_hits(clip, at, seconds)
    if not hits:
        raise O.OptionError("no word stands out for a glitch and the clip has no cut; name the word with at=\"…\" "
                            "or give a time in seconds")
    before = len(hits)
    hits = time_hits(hits, fps, n, length)
    if len(hits) < before:
        notes.append(f"{before - len(hits)} glitch(es) left out: too close to the one before")
    keep, faces = head_track(clip, tl, fr, notes)
    zone = SafeZone(W, H)

    # the stale picture each burst drags along: a moment before it starts
    stale = {}
    for h in hits:
        k = int(min(info.frames - 1, max(0, h.first - round(STALE * fps))))
        stale[h.frame] = fr.apply(video.read_frame(info, k), k)

    bars: list[Bar] = []
    if opts["bar"] != "off":
        for h in hits:
            b = Bar(opts["bar"], h, W, H, fps)
            j0, j1 = frame_of(b.t_in, fps), min(n, frame_of(b.t_out, fps) + 1)
            head = Rect.around(r for r in keep[j0:j1] if r is not None)
            face = Rect.around(r for r in faces[j0:j1] if r is not None)
            where = b.place(zone, head, face)
            if where is None:
                notes.append(f"no room off the face for the bar before “{h.word}”; the glitch runs without it")
                continue
            if b.k < 1.0:
                notes.append(f"the bar is drawn at {b.k:.0%}: the room off the face is tight")
            behind = []
            for t in (b.t_in + 0.1, (b.t_in + b.t_break) / 2, b.t_break):
                k = int(min(info.frames - 1, max(0, round(t * fps))))
                behind.append(fr.apply(video.read_frame(info, k), k))
            b.build(behind)
            b.where = where
            bars.append(b)

    def draw(frame: np.ndarray, j: int, t: float) -> None:
        for b in bars:
            b.draw(frame, t)
        for h in hits:
            if not (h.first < j < h.last):
                continue
            e = broken(h, j, scale)
            if e < 0.03:
                continue
            seed = _seed(NAME, h.frame, j)
            strong = break_frame(frame, stale[h.frame], e, seed)
            fw = face_weight(frame.shape[:2], faces[j] if j < len(faces) else None)
            if fw is not None:
                weak = break_frame(frame, stale[h.frame], e * FACE_SHARE, seed)
                strong = (strong.astype(np.float32) * (1 - fw) + weak.astype(np.float32) * fw + 0.5).astype(np.uint8)
            frame[:] = strong

    # the checks
    over_face = outside = 0
    for j in range(n):
        t = j / fps
        for b in bars:
            if b.alpha(t) <= 0:
                continue
            r = b.rect()
            hd = keep[j] if j < len(keep) else None
            fc = faces[j] if j < len(faces) else None
            if (hd is not None and r.intersects(hd)) or (fc is not None and r.intersects(fc)):
                over_face += 1
            if not zone.contains(b.drawn()):
                outside += 1
    spans = [(w.start, w.end) for w in clip.words]
    checks: dict = {"glitches": len(hits)}
    for i, h in enumerate(hits, 1):
        a, z = (h.first + 1) / fps, h.last / fps
        talk = sum(max(0.0, min(z, e) - max(a, s)) for s, e in spans)
        checks[f"glitch {i}"] = (f"worst on frame {h.frame} ({h.frame / fps:.2f} s)" + (f" on “{h.word}”" if h.word else "")
                                 + f", {h.why}; broken from {a:.2f} to {z:.2f} s ({z - a:.2f} s, {talk:.2f} s of it over "
                                 f"speech); frames {h.first} and {h.last} clean")
    for b in bars:
        checks[f"bar before “{b.hit.word}”"] = (f"{b.kind}, {b.where}, up {b.t_in:.2f} s, 100% on {b.t_break:.2f} s, "
                                                f"{b.rect().w:.0f}x{b.rect().h:.0f} px")
    checks["frames_over_face"] = over_face
    checks["frames_outside_safe_zone"] = outside
    checks["face"] = f"a third as broken over the face as elsewhere; untouched before and after each glitch"
    checks["sound"] = "a crackle under your voice on each break" if opts["sound"] else "your sound, untouched"
    checks["warnings"] = notes

    voice = soundmod.integrated(soundmod.blocks(clip.sound)) if info.has_audio else soundmod.SILENCE_DB
    reference = voice if voice > -70 else -23.0
    laid: list[dict] = []

    def sound(s: np.ndarray, rate: int) -> np.ndarray:
        out = s.copy()
        stem = np.zeros_like(out)
        for h in hits:
            d = (h.last - h.first) / fps
            x, peak = glitchmod.crackle(d, _seed("crackle", h.frame))
            gain = soundmod.gain((reference - UNDER) - soundmod.loudest(x))
            soundmod.lay(stem, x, h.frame / fps, rate, anchor=peak, level=gain)
        for b in bars:
            x, anchor = synth.tick(_seed("tick", b.hit.frame))
            gain = soundmod.gain((reference - TICK_UNDER) - soundmod.loudest(x))
            soundmod.lay(stem, x, b.t_in, rate, anchor=anchor, level=gain)
        out = out + stem
        peak = float(np.abs(out).max()) if len(out) else 0.0
        if peak > 0.98:
            out *= 0.98 / peak
        return out.astype(np.float32)

    if opts["sound"]:
        laid = [{"at": round(h.frame / fps, 4), "sound": "crackle", "under_voice_db": UNDER} for h in hits] + \
               [{"at": round(b.t_in, 4), "sound": "tick", "under_voice_db": TICK_UNDER} for b in bars]

    out_plan = {
        "effect": NAME, "clip": clip.path, "options": opts,
        "output": {"width": W, "height": H, "fps": round(fps, 3), "seconds": round(tl.duration, 3), "framing": fr.note},
        "glitches": [{"t": round(h.t, 4), "word": h.word, "why": h.why, "frame": h.frame, "first": h.first,
                      "last": h.last, "seconds": round((h.last - h.first) / fps, 3)} for h in hits],
        "bars": [{"kind": b.kind, "word": b.hit.word, "in": round(b.t_in, 3), "full": round(b.t_break, 3),
                  "out": round(b.t_out, 3), "rect": b.rect().as_list(), "where": b.where} for b in bars],
        "sounds": laid,
    }
    sheet: list[float] = []
    for h in hits[:2]:
        pre = [(h.first - 1) / fps] if not bars else [(bars[0].t_in + bars[0].t_break) / 2]
        sheet += pre + [(h.frame - 1) / fps, h.frame / fps, (h.frame + 1) / fps, (h.last + 2) / fps]
    if len(hits) > 2:
        sheet = sheet[:3] + [hits[-1].frame / fps] + sheet[3:5]
    out_plan["sheet_times"] = sorted(round(min(seconds - 0.5 / fps, max(0.0, t)), 4) for t in sheet[:6])
    picks = out_plan["sheet_times"]
    return Job(NAME, clip, tl, fr, draw, out_plan, checks=checks, still_times=lambda: picks,
               sound=sound if opts["sound"] else None)
