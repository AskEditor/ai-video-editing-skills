"""punch-in: the camera moves closer on the word that matters: a quick punch, a slow push, a drift
or a pull-back, always with the whole head in the picture.

How a move is placed:

- **Which words** (`at=auto`): each spoken line's strongest word, weighed by the pack's shared
  picker (`core/moments.py`), so a word the speaker took back, a filler or a first attempt is never
  one. The strongest go first, spaced apart, about one for every few seconds of the clip. Or the
  words a person names (`at="money, launch"`), or seconds (`at=2.4, 6`).
- **When**: a punch lands on the first frame of its word, never part-way through it. It holds to the
  end of the phrase and goes back out in the next pause between words (a smooth return when the pause
  has room, a cut in the gap when it is short), so the picture never moves under the middle of a word.
  A push arrives on its word, having crept in from the start of the line; a pull starts close at the
  start of the line and opens back out; a drift creeps sideways and in over the whole line.
- **How close**: `amount` times closer (about 1.2 by default), aimed at the face where it already
  is. The framing is the core's (`core/zoom.py`, the one `transitions` punches with): the whole head
  is kept, with a tenth of the frame above the crown, and the band under the chin stays free for
  captions. A head too big for the amount gets less, and the run says so.
- **Captions** (`captions=on`) are laid over the moving picture, not inside it: they are prepared on
  the picture as the camera frames it (`core/view.py`, through `layers.py`), so they keep clear of
  the face where the punch puts it, stay inside the safe zone, and never grow or slide with the zoom.
- **Sound**: a soft hit on a punch, a riser into a push, air on a pull; under the voice.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..core import moments, sound, synth, zoom
from ..core.layout import Rect, SafeZone
from ..core.words import HESITATIONS
from . import options as O

NAME = "punch-in"
MOVES_PERSON = True  # the camera moves the whole picture: other effects are laid inside it, never it inside them

STYLES = ("punch", "push", "drift", "pull")
AMOUNT = {"punch": 1.22, "push": 1.2, "drift": 1.15, "pull": 1.22}  # how much closer, by default
LEAST = 1.04  # a move the head leaves less room for than this is left out (it would not read)
SOUNDS = ("auto", "off", "hit", "whoosh", "swoosh-in", "pop", "click", "tick", "riser", "ding")
OWN_SOUND = {"punch": "hit", "push": "riser", "pull": "whoosh", "drift": None}
UNDER_VOICE = 9.0  # dB: each sound's loudest tenth of a second under the voice
CAPTIONS = ("off", "on", "stacked", "one-word", "serif-chip", "fitted")

APART = 2.6  # seconds: auto moves land at least this far apart
EVERY = 3.6  # seconds of clip for each auto move, at the most
SNAP = 0.16  # seconds a punch takes to arrive (it lands on the word's first frame)
SETTLE = 0.14  # seconds it settles after landing
HOLD = 0.9  # seconds a punch holds at the least before it may go back out
GAP = 0.10  # seconds: the shortest pause a return may use (shorter, and it waits for the next)
RETURN = 0.42  # seconds a smooth return takes, at the most
CLEAR = 0.35  # seconds between one move's return and the next move's start, at the least
EDGE = 0.25  # seconds: nothing lands this close to the start or the end of the clip


# ---- options -----------------------------------------------------------------------------------

def _clock(text: str) -> Optional[float]:
    t = text.strip().lower().rstrip("s")
    try:
        if ":" in t:
            m, s = t.split(":", 1)
            return int(m) * 60 + float(s)
        return float(t)
    except ValueError:
        return None


@dataclass(frozen=True)
class At:
    help: str

    def parse(self, raw) -> str:
        v = str(raw).strip()
        if v.lower() in ("", "auto"):
            return "auto"
        parts = [p.strip().strip("\"'“”") for p in re.split(r"[,;|]+", v) if p.strip().strip("\"'“”")]
        times = [_clock(p) for p in parts]
        if parts and all(t is not None for t in times):
            if any(t < 0 for t in times):
                raise O.OptionError(f"seconds must be 0 or more (got {raw!r})")
            return ", ".join(f"{t:g}" for t in sorted(times))
        if any(t is not None for t in times):
            raise O.OptionError(f"give either words or seconds, not both (got {raw!r})")
        return " | ".join(parts)


@dataclass(frozen=True)
class Amount:
    help: str

    def parse(self, raw):
        v = str(raw).strip().lower().rstrip("x×")
        if v in ("", "auto"):
            return "auto"
        try:
            x = float(v)
        except ValueError:
            raise O.OptionError(f"give auto or how much closer, such as 1.2 (got {raw!r})") from None
        if x > 2.0 and x <= 50:  # "20" meaning 20% closer
            x = 1 + x / 100
        if not 1.05 <= x <= 1.5:
            raise O.OptionError(f"give an amount between 1.05 and 1.5 (1.15 to 1.3 reads best; got {raw!r})")
        return round(x, 3)


OPTIONS = {
    "at": At("auto (each line's strongest word, spaced out), the words to land on separated by commas "
             "(\"money, launch\"), or seconds such as 2.4, 6"),
    "amount": Amount("how much closer: auto (about 1.2x; 1.15x for a drift), or 1.05 to 1.5 (1.15 to 1.3 reads best). "
                     "Less when the head leaves no room"),
    "style": O.Choice(STYLES, "punch: a quick zoom landing on the word · push: a slow push arriving on it · "
                              "drift: a slow sideways creep over the line · pull: starts close and pulls back"),
    "sound": O.Choice(SOUNDS, "auto (a soft hit on a punch, a riser into a push, air on a pull, none on a drift), "
                              "off, or one of the pack's sounds: hit, whoosh, swoosh-in, pop, click, tick, riser, ding"),
    "captions": O.Choice(CAPTIONS, "off, or the pack's captions laid over the moving picture, clear of your face: "
                                   "on (their own look), stacked, one-word, serif-chip or fitted"),
    "frame": O.Choice(("keep", "9:16"), "keep the clip's own shape, or make it 9:16 following the face"),
    "fix": O.Text("misheard words as wrong=right, separated by ;  e.g. \"Jon=John; ask editor=AskEditor\""),
}

LAB = "{lab}"
MATRIX = [
    ("A", "A", {}),
    ("B", "B", {}),
    ("C", "C", {}),
    ("D", "D", {}),
    ("F-times", "F", {"at": "3.0, 6.0"}),  # nobody says a word worth landing on: the seconds given
    ("A-push", "A", {"style": "push"}),
    ("B-pull", "B", {"style": "pull"}),
    ("D-drift", "D", {"style": "drift"}),
    ("B-captions", "B", {"captions": "on"}),
    ("33", "33", {}),
    ("33-captions", "33", {"captions": "on"}),
]


# ---- the words to land on ------------------------------------------------------------------------

@dataclass
class Beat:
    """One moment the camera moves on: a word, with its line."""
    index: int  # the word, in the clip's words (-1: a second given, no word there)
    at: float  # seconds: when the word begins
    end: float  # when it ends
    line: list[int]  # its line's words
    score: float = 0.0
    why: str = ""
    word: str = ""


def _line_of(lines: list[list[int]], i: int) -> list[int]:
    for ln in lines:
        if i in ln:
            return ln
    return [i]


def auto_beats(clip, duration: float, fits=None) -> tuple[list[Beat], list[str]]:
    """Each line's strongest word (the shared picker's measure), strongest first, kept when it lands
    far enough from the ones already kept and, when `fits(beat)` is given, when the head leaves room
    to move closer there (a word whose moment has no room gives way to the next strongest)."""
    words, quiet = clip.words, clip.quiet
    notes: list[str] = []
    if not words:
        return [], ["no words were heard in this clip"]
    back = moments.taken_back(words, quiet)
    runs = moments.lines(words, quiet)
    scraps = moments.scrap_lines(words, runs)
    cands: list[Beat] = []
    for n, ln in enumerate(runs):
        c = moments.best_in(words, ln, quiet, back)
        if c is None:
            continue
        score = c.score - (3.0 if n in scraps else 0.0)
        w = words[c.index]
        cands.append(Beat(c.index, w.start, w.end, ln, score, "its line's strongest word", moments.shown(w)))
    strong = [b for b in cands if b.score >= moments.MIN_HERO]
    if not strong:
        fair = sorted((b for b in cands if b.score >= moments.SPECIFIC), key=lambda b: -b.score)[:1]
        if fair:
            notes.append("no word stands out strongly, so only the clearest one gets a move")
        strong = fair
    most = max(1, int(round(duration / EVERY)))
    kept: list[Beat] = []
    for b in sorted(strong, key=lambda b: -b.score):
        if b.at < EDGE or b.at > duration - EDGE:
            continue
        if all(abs(b.at - k.at) >= APART for k in kept):
            if fits is not None and not fits(b):
                notes.append(f"“{b.word}” at {b.at:.2f} s leaves no room round the head, so the next strongest word is used")
                continue
            kept.append(b)
        if len(kept) >= most:
            break
    return sorted(kept, key=lambda b: b.at), notes


def _norm(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", text.lower().replace("’", "'")) if w]


def named_beats(clip, wanted: list[str]) -> tuple[list[Beat], list[str]]:
    """The words a person named, each where it is first said (a phrase: on its first word); a word
    named twice lands on its second saying. A word said and then taken back is skipped for the one
    said instead."""
    words, quiet = clip.words, clip.quiet
    back = moments.taken_back(words, quiet)
    runs = moments.lines(words, quiet)
    keys = [w.key for w in words]
    out, notes, used = [], [], set()
    for phrase in wanted:
        want = _norm(phrase)
        if not want:
            continue
        hits = [i for i in range(len(words) - len(want) + 1)
                if [moments.bare(words[i + k]) for k in range(len(want))] == want or keys[i:i + len(want)] == want]
        hits = [i for i in hits if words[i] not in back and i not in used] or [i for i in hits if i not in used]
        if not hits:
            notes.append(f"“{phrase}” is not said in this clip, so nothing lands there")
            continue
        i = hits[0]
        used.add(i)
        w = words[i]
        out.append(Beat(i, w.start, words[i + len(want) - 1].end, _line_of(runs, i), 0.0, "named", phrase))
    return sorted(out, key=lambda b: b.at), notes


def timed_beats(clip, times: list[float], duration: float) -> tuple[list[Beat], list[str]]:
    """The seconds given, each moved onto the start of the word nearest it, so a move never lands
    part-way through a word. With no word near, the second itself."""
    words, quiet = clip.words, clip.quiet
    runs = moments.lines(words, quiet) if words else []
    out, notes = [], []
    for t in times:
        if t >= duration:
            notes.append(f"{t:g} s is past the end of the clip ({duration:.2f} s)")
            continue
        near = [i for i, w in enumerate(words) if w.key not in HESITATIONS and abs(w.start - t) <= 0.6]
        if near:
            i = min(near, key=lambda i: abs(words[i].start - t))
            w = words[i]
            out.append(Beat(i, w.start, w.end, _line_of(runs, i), 0.0, f"the word nearest {t:g} s", moments.shown(w)))
        else:
            inside = [w for w in words if w.start < t < w.end]
            if inside:  # in the middle of a word with none starting near: the start of that word
                w = inside[0]
                i = words.index(w)
                out.append(Beat(i, w.start, w.end, _line_of(runs, i), 0.0, f"the start of the word under {t:g} s",
                                moments.shown(w)))
            else:
                out.append(Beat(-1, t, t + 0.3, [], 0.0, f"{t:g} s, as given", ""))
    return sorted(out, key=lambda b: b.at), notes


# ---- when the camera moves -------------------------------------------------------------------------

def gaps(words, quiet, duration: float) -> list[tuple[float, float]]:
    """The pauses between words (from the word times, widened by the quiet the sound shows), where the
    camera may move without moving under a word."""
    out = []
    spoken = [w for w in words if w.key not in HESITATIONS] or list(words)
    if spoken and spoken[0].start > 0:
        out.append((0.0, spoken[0].start))
    for a, b in zip(spoken, spoken[1:]):
        if b.start > a.end:
            out.append((a.end, b.start))
    if spoken and spoken[-1].end < duration:
        out.append((spoken[-1].end, duration))
    if not spoken:
        out.append((0.0, duration))
    return out


def word_at(words, t: float):
    for w in words:
        if w.start - 1e-6 <= t < w.end - 1e-6:
            return w
    return None


@dataclass
class Plan:
    """One move: in on a beat, a hold, and back out."""
    beat: Beat
    style: str
    start: float  # the camera starts moving
    land: float  # it arrives (a punch: the word's first frame)
    out_start: float  # it starts back
    out_end: float  # it is back on the whole frame
    out_kind: str  # smooth or cut
    zoom: float = 1.0
    window: tuple = ()
    drift_from: tuple = ()  # a drift: where it creeps from (the framing it arrives at is `window`)
    note: str = ""
    sound: Optional[str] = None
    end_of_clip: float = math.inf

    def as_dict(self) -> dict:
        d = {"style": self.style, "word": self.beat.word, "word_index": self.beat.index,
             "word_at": round(self.beat.at, 3), "why": self.beat.why,
             "start": round(self.start, 3), "land": round(self.land, 3),
             "out_start": round(self.out_start, 3), "out_end": round(self.out_end, 3), "out": self.out_kind,
             "zoom": round(self.zoom, 3), "window": [round(v, 1) for v in self.window]}
        if self.drift_from:
            d["drift_from"] = [round(v, 1) for v in self.drift_from]
        if self.sound:
            d["sound"] = self.sound
        if self.note:
            d["note"] = self.note
        return d


def release(after: float, words, pauses, limit: float, duration: float) -> tuple[float, float, str]:
    """Where the camera goes back to the whole frame, at `after` or later and before `limit`: in the
    first pause long enough for a smooth return, else with a cut in the first gap between words
    there is. It never moves under a word."""
    for a, b in pauses:
        if b <= after + 1e-6:
            continue
        a = max(a, after)
        if a >= limit:
            break
        room = min(b, limit) - a
        if room >= 0.2:
            start = a + min(0.04, room * 0.1)
            return start, start + min(RETURN, room - 0.06), "smooth"
        if room >= GAP or b >= duration - 1e-6:
            mid = a + room / 2
            return mid, mid, "cut"
    # no pause before the limit: a cut on the start of the next word (between two words, never inside one)
    for w in words:
        if w.start >= after and w.start < limit:
            return w.start, w.start, "cut"
    end = min(limit, duration)
    return end, end, "cut"


def time_moves(beats: list[Beat], style: str, clip, duration: float) -> tuple[list[Plan], list[str]]:
    """Each beat's move in time: when it starts, lands, holds to and returns, never moving under the
    middle of a word, with room between one move's return and the next's start."""
    words, quiet = clip.words, clip.quiet
    pauses = gaps(words, quiet, duration)
    plans: list[Plan] = []
    notes: list[str] = []
    for k, b in enumerate(beats):
        nxt = beats[k + 1] if k + 1 < len(beats) else None
        line = [words[i] for i in b.line] if b.line else []
        line_start = line[0].start if line else b.at
        line_end = line[-1].end if line else b.end
        if style == "punch":
            start, land = max(0.0, b.at - SNAP), b.at
            hold = max(b.end, land + HOLD, line_end)
        elif style == "push":
            land = b.end if b.end - b.at < 0.8 else b.at  # arrives as the word is said
            start = line_start
            if land - start < 0.9:  # too short to read as a push: creep in from the pause before the line
                start = max(0.0, land - 1.2)
                inside = word_at(words, start)
                if inside is not None and inside.start < start:
                    start = inside.end if inside.end < land - 0.6 else start
            hold = max(line_end, land + 0.5)
        elif style == "pull":
            start = land = line_start if line else b.at  # close from the start of the line…
            if start < 0.45:  # the film's first line: close from its very first frame, not a cut in
                start = land = 0.0
            hold = max(b.end, line_end)  # …opening back out by its end
        else:  # drift
            start, land = line_start, max(line_end, line_start + 1.6)
            hold = land
        limit = (nxt_start(nxt, words, style) - CLEAR) if nxt is not None else duration
        if hold >= limit:
            hold = max(land, limit - 0.2)
        if style == "pull":
            out_start, out_end, kind = start, min(duration, limit, start + min(2.2, max(1.0, hold - start))), "smooth"
        else:
            out_start, out_end, kind = release(hold, words, pauses, limit, duration)
        p = Plan(b, style, start, land, out_start, out_end, kind, end_of_clip=duration)
        if plans and p.start < plans[-1].out_end + 0.05:
            notes.append(f"the move on “{b.word or f'{b.at:.2f} s'}” would start before the last one is back; left out")
            continue
        plans.append(p)
    return plans, notes


def nxt_start(b: Beat, words, style: str) -> float:
    """Roughly where the move on beat `b` will start (the room the one before it must leave)."""
    if style == "punch":
        return b.at - SNAP
    if style == "push":
        return b.at - 1.2
    return min(b.at, words[b.line[0]].start if b.line else b.at)


# ---- how close -------------------------------------------------------------------------------------

def frame_moves(plans: list[Plan], amount, clip, fr, fps: float) -> list[str]:
    """Each move's framing, from the head over every frame it is held: the whole head kept, a tenth
    of the frame above the crown, the band under the chin free (the core's `punch_window`)."""
    W, H = fr.out_w, fr.out_h
    notes = []
    n = clip.info.frames
    for p in plans:
        want = AMOUNT[p.style] if amount == "auto" else float(amount)
        first = int(max(0, math.floor(p.start * fps)))
        last = int(min(n - 1, math.ceil(max(p.out_end, p.land) * fps)))
        head, centre = zoom.aim(clip.face, fr, first, last)
        win, z, note = zoom.punch_window((W, H), want, head, centre)
        p.zoom, p.window = z, win
        if note:
            p.note = note
        if p.style == "drift" and z >= LEAST:
            p.drift_from = drift_start(win, head, (W, H))
    return notes


def drift_start(win, head: Optional[Rect], size) -> tuple:
    """Where a drift creeps from: the same framing slid sideways, as far as the head allows (it
    stays inside with room either side), at most a sixteenth of the frame."""
    W, H = size
    x, y, w, h = win
    most = W / 16
    lo, hi = max(0.0, x - most), min(W - w, x + most)
    if head is not None:
        lo = max(lo, head.x1 + 0.05 * w - w)
        hi = min(hi, head.x0 - 0.05 * w)
    left, right = x - lo, hi - x
    if max(left, right) < 2:
        return tuple(win)
    nx = lo if left >= right else hi
    return (float(nx), y, w, h)


def camera(plans: list[Plan], size) -> zoom.CameraPath:
    """The plans as a camera path: in, hold, out, for each in turn."""
    full = (0.0, 0.0, float(size[0]), float(size[1]))
    moves = []
    for p in plans:
        if p.zoom < LEAST:
            continue
        if p.style == "punch":
            moves.append(zoom.Move(p.start, p.land, p.window, "snap", SETTLE))
        elif p.style == "push":
            moves.append(zoom.Move(p.start, p.land, p.window, "slow"))
        elif p.style == "pull":
            moves.append(zoom.Move(p.start, p.start, p.window, "cut"))
            moves.append(zoom.Move(p.start + 1e-3, p.out_end, full, "slow"))
            continue
        else:  # drift: in gently onto the side, then creep across while the line is said
            ease_in = min(0.5, 0.3 * (p.land - p.start))
            moves.append(zoom.Move(p.start, p.start + ease_in, p.drift_from or p.window, "smooth"))
            moves.append(zoom.Move(p.start + ease_in, p.land, p.window, "slow"))
        if p.out_start < p.end_of_clip - 0.15:  # a return in the last frames would only flicker: it stays close
            moves.append(zoom.Move(p.out_start, p.out_end, full, "cut" if p.out_kind == "cut" else "smooth"))
    return zoom.CameraPath(size, moves)


# ---- sound ---------------------------------------------------------------------------------------------

class Sounds:
    """Each move's sound, landing with it, its loudest tenth of a second under the voice."""

    def __init__(self, clip, plans: list[Plan]):
        self.plans = plans
        voice = sound.integrated(sound.blocks(clip.sound)) if clip.info.has_audio else sound.SILENCE_DB
        self.ref = voice if voice > -70 else -20.0
        self.limited_db = 0.0

    def mix(self, base: np.ndarray, rate: int) -> np.ndarray:
        from ..core import loudness

        stem = np.zeros_like(base)
        for p in self.plans:
            if not p.sound or p.zoom < LEAST:
                continue
            x, anchor = sound.library(p.sound)
            if p.sound == "riser":
                at = p.land
            elif p.style == "pull":
                at = p.start + 0.25
            elif p.style == "drift":
                at = p.start + 0.4
            else:
                at = p.land
            sound.lay(stem, synth.at_level(x, self.ref - UNDER_VOICE), at, rate, anchor=anchor)
        out = base + stem
        if stem.any():
            ceiling = max(-1.0, loudness.true_peak_db(base) + 0.05)
            if loudness.true_peak_db(out) > ceiling:
                out, self.limited_db = loudness.limit(out, ceiling)
        return out.astype(np.float32)


# ---- captions over the moving picture ---------------------------------------------------------------------

def lay_captions(clip, opts, path: zoom.CameraPath, fr, fps: float):
    """The pack's captions, prepared on the picture as the camera frames it, and drawn over it: the
    face where the punch puts it, the safe zone of the screen, and nothing that grows with the zoom."""
    from ..core import framing as framing_mod
    from ..core.view import View
    from . import layers

    W, H = fr.out_w, fr.out_h
    sx, sy = fr.src_w, fr.src_h

    def place(i: int):
        fx, fy, fw, fh = fr.window(i)
        px, py, pw, ph = path.at(i / fps)
        s = fw / W
        return (fx + px * s, fy + py * (fh / H), pw * s, ph * (fh / H)), Rect(0, 0, W, H)

    view = View(clip, (W, H), place=place)
    base = framing_mod.plan(W, H, fps, clip.info.frames, None, shape="keep")
    given = {} if opts["captions"] == "on" else {"look": opts["captions"]}
    spec = layers.Spec("captions", given)
    try:
        return layers.prepare(view, spec, None, None, base, fix=opts["fix"]), view
    except ValueError as e:
        raise O.OptionError(f"captions: {e}") from None


# ---- checks -------------------------------------------------------------------------------------------

def head_checks(clip, fr, path: zoom.CameraPath, fps: float, n: int) -> dict:
    """Over every frame: is the top of the head inside the picture, with room (unless the recording
    cut it already), and how far down the frame the chin goes."""
    W, H = fr.out_w, fr.out_h
    cut = tight = 0
    lowest = 0.0
    for j in range(n):
        h = clip.face.head(j)
        if h is None:
            continue
        r0 = fr.to_output(h, j)
        x, y, w, hh = path.at(j / fps)
        if w >= W - 0.5:
            continue
        k = H / hh
        top, chin = (r0.y0 - y) * k, (r0.y1 - y) * k
        was_cut = r0.y0 <= 0.01 * H
        if top < 0 and not was_cut:
            cut += 1
        elif top < 0.05 * H and not was_cut:
            tight += 1
        lowest = max(lowest, chin / H)
    return {"frames_crown_cut": cut, "frames_crown_tight": tight, "chin_lowest": round(lowest, 3)}


def caption_checks(layer, view, fps: float) -> dict:
    over, outside = 0, 0
    zone = SafeZone(view.size[0], view.size[1]).rect
    for c in layer.job.plan.get("cards") or []:
        r = Rect.of(c["rect"])
        for j in range(int(math.ceil(float(c["start"]) * fps)), int(math.ceil(float(c["end"]) * fps))):
            head = view.face.head(j)
            if head is not None and r.intersects(head):
                over += 1
        if not r.inside(zone, slack=1.0):
            outside += int(round((float(c["end"]) - float(c["start"])) * fps))
    return {"caption_frames_over_face": over, "caption_frames_outside_safe_zone": outside}


# ---- the effect -------------------------------------------------------------------------------------------

def _burned_captions(clip) -> Optional[str]:
    p = Path(clip.path).with_suffix(".plan.json")
    try:
        plan = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    if plan.get("cards") and (plan.get("effect") == "captions" or plan.get("options", {}).get("captions") not in (None, "off", False)):
        return (f"this video has captions drawn into it by {plan.get('effect')}: the zoom moves them with the picture. "
                f"For captions that stay put, run it without captions, then punch-in with captions=on")
    return None


def _from_plan(plan: dict, clip, style_default: str) -> list[Plan]:
    from . import PlanError

    out = []
    try:
        for d in plan["moves"]:
            words = clip.words
            i = int(d.get("word_index", -1))
            line = []
            b = Beat(i, float(d["word_at"]), float(d["word_at"]) + 0.3, line, 0.0, d.get("why", "from the plan"),
                     d.get("word", ""))
            p = Plan(b, d.get("style", style_default), float(d["start"]), float(d["land"]), float(d["out_start"]),
                     float(d["out_end"]), d.get("out", "smooth"), float(d.get("zoom", 1.0)),
                     tuple(float(v) for v in d["window"]),
                     tuple(float(v) for v in d.get("drift_from", ())), d.get("note", ""), d.get("sound"))
            if p.style not in STYLES or len(p.window) != 4:
                raise ValueError
            p.end_of_clip = clip.info.frames / clip.info.rate
            if 0 <= i < len(words):
                b.end = words[i].end
            out.append(p)
    except (KeyError, TypeError, ValueError):
        raise PlanError("the plan's moves need style, word_at, start, land, out_start, out_end and window; "
                        "run once without --plan to get one to edit") from None
    return out


def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from ..core import framing as framing_mod
    from ..core.timeline import Timeline
    from . import Job

    info = clip.info
    fps = info.rate
    n = info.frames
    duration = n / fps
    fr = framing_mod.plan(info.width, info.height, fps, n, clip.face, shape=opts["frame"])
    W, H = fr.out_w, fr.out_h
    notes: list[str] = []
    if not clip.face.any() or clip.face.found_share < 0.3:
        notes.append("the face is found in few frames, so the camera aims at the middle of the frame")

    style = opts["style"]
    if plan is not None:
        plans = _from_plan(plan, clip, style)
        source = "the plan given"
    else:
        at = opts["at"]
        if at == "auto":
            def fits(b: Beat) -> bool:
                trial, _ = time_moves([b], style, clip, duration)
                frame_moves(trial, opts["amount"], clip, fr, fps)
                return bool(trial) and trial[0].zoom >= LEAST

            beats, more = auto_beats(clip, duration, fits)
            source = "each line's strongest word, spaced out"
        elif all(_clock(p) is not None for p in at.split(",")):
            beats, more = timed_beats(clip, [float(p) for p in at.split(",")], duration)
            source = "the seconds given, each on the start of its word"
        else:
            beats, more = named_beats(clip, [p.strip() for p in at.split("|")])
            source = "the words named"
        notes += more
        if not beats:
            raise O.OptionError("no word in this clip stands out enough to land on" if at == "auto" else
                                "none of those land in this clip"
                                + "; name the words (at=\"money, launch\") or give seconds (at=2.4, 6)")
        plans, more = time_moves(beats, style, clip, duration)
        notes += more
        frame_moves(plans, opts["amount"], clip, fr, fps)
        for p in plans:
            p.sound = (OWN_SOUND[p.style] if opts["sound"] == "auto" else (None if opts["sound"] == "off" else opts["sound"]))
    for p in plans:
        if p.zoom < LEAST:
            p.note = (p.note + "; " if p.note else "") + "no room round the head: left out"
    live = [p for p in plans if p.zoom >= LEAST]
    if not live:
        raise O.OptionError("the head fills the frame, so there is no room to move closer without cutting it; "
                            "this works on a shot with some room round the head")
    path = camera(plans, (W, H))

    layer = view = None
    if opts["captions"] != "off":
        layer, view = lay_captions(clip, opts, path, fr, fps)

    def draw(frame: np.ndarray, j: int, t: float) -> None:
        out = path.frame(frame, t, fps)
        if out is not frame:
            frame[:] = out
        if layer is not None:
            layer.draw(frame, j, t)

    burned = _burned_captions(clip)
    if burned:
        notes.append(burned)
    sounds = Sounds(clip, plans) if any(p.sound for p in live) else None

    checks = {
        "moves": "; ".join(f"{p.style} on “{p.beat.word or '—'}” at {p.land:.2f} s, {p.zoom:.2f}x, back "
                           f"{'by a cut' if p.out_kind == 'cut' else 'smoothly'} at {p.out_start:.2f} s"
                           + (f" ({p.note})" if p.note else "") for p in plans),
        "chosen_by": source,
    }
    checks.update(head_checks(clip, fr, path, fps, n))
    mid = [p for p in live if any(w.start + 0.03 < p.land < w.end - 0.03 for w in clip.words)]
    checks["moves_landing_mid_word"] = len(mid)
    if layer is not None:
        checks.update(caption_checks(layer, view, fps))
    checks["sound"] = ("; ".join(f"{p.sound} at {p.land:.2f} s" for p in live if p.sound) + f", {UNDER_VOICE:g} dB under the voice"
                       if sounds else "the clip's own only")
    checks["notes"] = notes

    def stills() -> list[float]:
        out = []
        for p in live[:2]:
            out += [max(0.0, p.start - 0.15), p.land + 0.05, (p.land + p.out_start) / 2]
        return [round(min(duration - 1 / fps, t), 3) for t in out][:5]

    sheet = []
    for p in live[:2]:
        sheet += [max(0.0, p.start - 0.2), p.land + 1.5 / fps if p.style != "pull" else p.start + 0.05,
                  (p.land + p.out_start) / 2 if p.style != "pull" else (p.start + p.out_end) / 2]
    if len(live) == 1:
        p = live[0]
        sheet += [p.out_start - 0.05, p.out_end + 0.1, min(duration - 0.1, p.out_end + 0.8)]
    sheet = sorted(round(min(duration - 1.5 / fps, max(0.0, t)), 3) for t in sheet)[:6]
    out_plan = {
        "effect": NAME, "clip": clip.path, "options": opts,
        "output": {"width": W, "height": H, "fps": round(fps, 3), "seconds": round(duration, 3), "framing": fr.note},
        "moves": [p.as_dict() for p in plans],
        "sheet_times": sheet,
    }
    if layer is not None:
        out_plan["captions"] = {"cards": len(layer.job.plan.get("cards") or [])}
    return Job(effect=NAME, clip=clip, timeline=Timeline.whole(duration), framing=fr, draw=draw, plan=out_plan,
               checks=checks, still_times=stills, sound=sounds.mix if sounds is not None else None)
