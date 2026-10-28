"""notifications: message notifications pop in one after another, each landing on a word, the
newest on top and the older ones sliding down under it. The cards are of our own design.

How it works:

1. **What they say.** `items=` gives them ("Can you send the link?; Booked for Friday@4.2", or a
   .json file where each can name its own sender with `from`). Left empty, they are the short
   phrases said that stand best as a line of their own, chosen by the shared picker
   (`fx/core/moments.py`): never a word taken back, a first attempt, a filler or an aside. A phrase
   too long for two lines is shortened by whole words.
2. **When.** Each card is whole and still on the first frame of its word: a phrase said, on its
   first word; a given item, at its time moved onto the word starting there, else where its words
   are said, else on the strongest line starts of the talk (and the run says so). It drops in over
   the moment before, with a soft chime made in code, under the voice.
3. **The stack.** The newest sits on top; the ones before slide down a place each time one arrives,
   and when more arrive than the room holds, the oldest fades as it slides out. They all go a
   couple of seconds after the last.
4. **The card.** Frosted dark glass (or a plain dark card), a rounded tile in the accent colour with
   a speech mark of our own, the sender and "now", and up to two lines of the message.
5. **Where.** Above the head, or under the chin, wherever the stack has room over every frame it is
   up, inside the safe zone with its shadow. Never on the face.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..core import counting
from ..core import glass as glassmod
from ..core import items as itemsmod
from ..core import moments
from ..core import text as textmod
from ..core.compose import Sprite, over
from ..core.layout import Rect, SafeZone
from ..core.words import HESITATIONS
from . import options as O
from .captions import head_in_output
from .headline import find_said, keepout_in_output
from .tier_list import bands

NAME = "notifications"
POSITIONS = ("auto", "above", "below")
ROOM_OF = {"above": "top", "below": "below"}

OPTIONS = {
    "items": O.Text("the messages, separated by ; (a time in seconds with @ is optional: Booked for Friday@4.2), "
                    "or a .json file where each can have text, at and from. Empty: the short phrases you say "
                    "that stand best as a line"),
    "count": O.Whole(1, 8, "how many phrases to take when items is empty"),
    "sender": O.Text("who they are from, on every card (a .json item's own from wins): your name, a handle, "
                     "a brand; auto writes New message"),
    "style": O.Choice(("glass", "solid"), "glass: frosted dark glass, your video blurred behind it · solid: a "
                                          "plain dark card, for a busy background"),
    "position": O.Choice(POSITIONS, "auto: wherever the stack has most room · above: above your head · below: "
                                    "under your chin. Never on your face"),
    "most": O.Whole(1, 5, "the most cards on screen at once; the oldest slides out when another arrives"),
    "accent": O.Colour("the tile on each card: a name or a hex colour (auto is our coral)"),
    "hold": O.Number(0.5, 20.0, "seconds the stack stays after the last card lands"),
    "sound": O.Switch("on: a soft chime as each card lands, under your voice · off keeps the clip's sound exactly"),
    "fix": O.Text("misheard words as wrong=right, separated by ;  e.g. \"ask editor=AskEditor\""),
}

MATRIX = [
    ("D", "D", {}),
    ("B", "B", {"sender": "Eben"}),
    ("A", "A", {"items": "Is the hook ready?; Send me the campaign@5.6", "sender": "Producer"}),
    ("C", "C", {"style": "solid", "fix": "Jon=John"}),
    ("33", "33", {"count": 4, "sender": "Team"}),
]

ARRIVE = 0.30  # a card drops in over this long, whole and still on the frame of its word
SLIDE = 0.30
FADE_ROOM = 0.12  # a card leaving moves this share of a place down as it fades
CLEAR = 5.0  # seconds between two cards after which the stack goes and a fresh one starts  # the cards before it slide down a place over the same moment
EXIT = 0.36
CHIME_UNDER = 15.0  # dB under the voice's loud syllables
APART = 1.0
FINISH = 3.0  # how much less the shared scorer may rate a phrase that finishes its clause, and still win  # seconds between two phrases chosen by the picker, at least
ACCENT = "#ff6f59"

# Design, in pixels of a frame 1080 wide at size 1.
CARD_W = 820
PAD = 28
TILE = 78
TILE_R = 22
HEAD_PX = 29
BODY_PX = 33
LINE = 1.36  # body line pitch, as a share of BODY_PX
RADIUS = 38
GAP = 18  # between two cards
DROP = 0.22  # a card drops from this share of its own height above its place
K_MAX, K_MIN = 1.0, 0.5
READABLE = 0.82  # under this size the message is under about 1.4% of the frame's height: hard to read
SOLID = (24, 25, 31)
INK = (255, 255, 255)
SOFT = (208, 211, 220)


def _clamp(x: float) -> float:
    return min(1.0, max(0.0, x))


def _ease_out(x: float) -> float:
    x = _clamp(x)
    return 1.0 - (1.0 - x) ** 3


def _ease_in_out(x: float) -> float:
    x = _clamp(x)
    return x * x * x * (x * (6 * x - 15) + 10)


@dataclass
class Note:
    text: str
    at: float  # the first frame it is whole and still
    sender: str
    how: str  # "said", "given", "time", "picked"
    word: str = ""
    lines: tuple[str, ...] = ()


# ---- what they say --------------------------------------------------------------------------------

def _line_text(words, a: int, b: int) -> str:
    t = " ".join(moments.shown(words[i]) for i in range(a, b + 1)).strip().rstrip(",;:")
    return t[:1].upper() + t[1:]


def _phrases(words, quiet, count: int) -> list[tuple[int, int]]:
    """The strongest short lines said, none overlapping, at least `APART` seconds apart, in time order."""
    ranked: list = []
    moments.pick_line(words, quiet, ranked=ranked)
    score = {span: sc for sc, span in ranked}
    chosen: list[tuple[int, int]] = []
    for _, (a, b) in ranked:
        if a > 0 and words[a - 1].key == "i" and not words[a - 1].ends_clause:
            continue  # a phrase cut off from its "I" ("I | think this audio works better") is not a message
        # a phrase that stops one word short of its clause's end ("a real big | issue") takes the word,
        # when the shared scorer rates the whole nearly as high
        while not words[b].ends_clause and (a, b + 1) in score and score[(a, b + 1)] >= score[(a, b)] - FINISH:
            b += 1
        if any(not (b < x or a > y) for x, y in chosen):
            continue
        if any(abs(words[a].start - words[x].start) < APART for x, _ in chosen):
            continue
        chosen.append((a, b))
        if len(chosen) == count:
            break
    return sorted(chosen, key=lambda s: s[0])


def _given(spec: str, clip_dir: Path) -> list[tuple[str, Optional[float], str]]:
    data, base = itemsmod.load(spec, clip_dir)
    raws = data if isinstance(data, list) else (data.get("items") if isinstance(data, dict) else None)
    if data is None:
        raws = [p for p in spec.split(";") if p.strip()]
    if not raws:
        raise O.OptionError("items: give at least one message")
    out = []
    for raw in raws:
        try:
            text, at, pic = itemsmod.entry(raw, base, clip_dir)
        except itemsmod.ItemsError as e:
            raise O.OptionError(f"items: {e}") from None
        if pic or not text:
            raise O.OptionError("items: each message is words (pictures are not shown on a notification)")
        out.append((text, at, str(raw.get("from", "")).strip() if isinstance(raw, dict) else ""))
    if len(out) > 12:
        raise O.OptionError("items: twelve messages at most")
    return out


# ---- the effect -------------------------------------------------------------------------------------

class Stack:
    def __init__(self, clip, opts: dict, size: tuple[int, int], fps: float, n_frames: int, head_at, face_at,
                 frame_at, plan: Optional[dict] = None):
        self.W, self.H = size
        self.U = self.W / 1080.0
        self.fps, self.n = fps, n_frames
        self.seconds = n_frames / fps
        self.opts = opts
        self.zone = SafeZone(self.W, self.H)
        self.heads = [head_at(j) for j in range(n_frames)]
        self.faces = [face_at(j) for j in range(n_frames)]
        self.frame_at = frame_at
        self.warnings: list[str] = []
        self.words = [w for w in clip.words if w.key not in HESITATIONS]
        self.quiet = clip.quiet
        sender = opts["sender"].strip()
        self.sender = "New message" if sender.lower() in ("", "auto") else sender
        self.accent = textmod.colour(ACCENT if opts["accent"] == "auto" else opts["accent"])
        if plan is not None:
            self._from_plan(plan)
        else:
            self._notes(clip)
        self.notes.sort(key=lambda n: n.at)
        self._space()
        self._timing()
        self._place(plan)
        self._build()

    # -- what and when -------------------------------------------------------------------------------
    def _notes(self, clip) -> None:
        spec = self.opts["items"].strip()
        words = self.words
        notes: list[Note] = []
        if not spec:
            spans = _phrases(words, self.quiet, int(self.opts["count"]))
            if not spans:
                raise O.OptionError("no short phrase said here stands as a message by itself; give the messages: "
                                    "items=\"Can you send the link?; Booked for Friday\"")
            for a, b in spans:
                notes.append(Note(_line_text(words, a, b), words[a].start, self.sender, "said", words[a].clean))
            self.found = "phrases said, chosen by the shared picker"
        else:
            given = _given(spec, Path(clip.path).parent)
            free = []
            for text, at, frm in given:
                who = frm or self.sender
                if at is not None:
                    if at >= self.seconds:
                        raise O.OptionError(f"items: {at:g} s is after the end of the video ({self.seconds:.2f} s)")
                    t, word = itemsmod.snap(words, at)
                    notes.append(Note(text, t, who, "time", word))
                    continue
                hit = find_said(words, text)
                if hit:
                    notes.append(Note(text, words[hit[0]].start, who, "said", words[hit[0]].clean))
                else:
                    free.append(Note(text, -1.0, who, "picked"))
            if free:
                taken = [n.at for n in notes]
                starts = sorted((s for s in moments.strong_starts(words, self.quiet) if s.strength is not None
                                 and s.at >= ARRIVE and all(abs(s.at - t) >= APART for t in taken)),
                                key=lambda s: -s.strength)
                chosen = []
                for s in starts:
                    if all(abs(s.at - c.at) >= APART for c in chosen):
                        chosen.append(s)
                    if len(chosen) == len(free):
                        break
                if len(chosen) < len(free):
                    raise O.OptionError("items: there are not enough places to land these messages; give each a "
                                        "time with @ (Booked for Friday@4.2)")
                chosen.sort(key=lambda s: s.at)
                for n, s in zip(free, chosen):
                    n.at, n.word = s.at, words[s.index].clean
                notes += free
                self.warnings.append(f"{len(free)} of the messages are not said: each lands where a line starts "
                                     f"(give a time with @ to choose)")
            self.found = "given"
        self.notes = notes

    def _from_plan(self, plan: dict) -> None:
        try:
            self.notes = [Note(str(n["text"]), float(n["at"]), str(n.get("from", self.sender)), n.get("how", "plan"),
                               n.get("word", "")) for n in plan["stack"]["notes"]]
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"the plan's notifications could not be read ({e})") from None
        if not self.notes:
            raise ValueError("the plan has no notifications")
        self.found = "from the plan"

    def _space(self) -> None:
        """Two cards never land closer than a drop apart: the later one waits for the earlier."""
        for a, b in zip(self.notes, self.notes[1:]):
            if b.at - a.at < ARRIVE + 0.05:
                self.warnings.append(f"“{b.text}” moved to {a.at + ARRIVE + 0.05:.2f} s, after the card before it")
                b.at = a.at + ARRIVE + 0.05
        first = self.notes[0]
        if first.at < ARRIVE:
            first.at = ARRIVE  # the first can drop in no earlier than the clip begins

    def _timing(self) -> None:
        """Cards said close together share a stack; after a long gap the stack goes and a fresh one
        comes with the next card, so a stack is never left up through a stretch of talk."""
        hold = float(self.opts["hold"])
        self.group_of, self.spans = [], []
        for i, n in enumerate(self.notes):
            if i and n.at - self.notes[i - 1].at <= CLEAR:
                self.group_of.append(self.group_of[-1])
            else:
                self.group_of.append(len(self.spans))
                self.spans.append([max(0.0, n.at - ARRIVE), 0.0])
            self.spans[self.group_of[-1]][1] = n.at
        for k, sp in enumerate(self.spans):
            nxt = self.spans[k + 1][0] if k + 1 < len(self.spans) else self.seconds
            sp[1] = min(self.seconds, sp[1] + hold, nxt)
        self.start, self.end = self.spans[0][0], self.spans[-1][1]

    # -- the card's design ----------------------------------------------------------------------------
    def _lines(self, text: str, width: float, px: int) -> tuple[str, ...]:
        """The message on at most two lines of `width`, shortened by whole words when it is longer."""
        face = "poppins-semibold"

        def wrap(t: str) -> Optional[tuple[str, ...]]:
            out, cur = [], ""
            for w in t.split():
                nxt = (cur + " " + w).strip()
                if textmod.advance(nxt, face, px) <= width:
                    cur = nxt
                else:
                    if not cur:
                        return None
                    out.append(cur)
                    cur = w
            if cur:
                out.append(cur)
            return tuple(out) if len(out) <= 2 else None

        for form in itemsmod.shorter(text):
            got = wrap(form)
            if got:
                return got
        cut = itemsmod.cut_to(text, lambda c: wrap(c) is not None)
        got = wrap(cut)
        if not got:
            got = (cut,)
        self.warnings.append(f"“{text}” was shortened to fit two lines")
        return got

    def _size(self, k: float) -> tuple[float, float]:
        """A card's width and height at size k (every card the same height, so the stack is even)."""
        u = k * self.U
        w = CARD_W * u
        body_w = w - (2 * PAD + TILE + 22) * u
        two = any(len(self._lines_quiet(n.text, body_w, max(8, int(round(BODY_PX * u))))) > 1 for n in self.notes)
        h = (2 * PAD + HEAD_PX * 1.25 + 8 + BODY_PX * LINE * (2 if two else 1)) * u
        return w, max(h, (2 * PAD + TILE) * u)

    def _lines_quiet(self, text, width, px):
        keep = list(self.warnings)
        got = self._lines(text, width, px)
        self.warnings = keep
        return got

    def _halo(self, k: float) -> tuple[float, float, float, float]:
        u = k * self.U
        sp = math.ceil(22 * u)
        return float(sp), float(max(0.0, sp - 7 * u)), float(sp), float(sp + 7 * u)

    def _place(self, plan: Optional[dict]) -> None:
        a = int(max(0, math.floor(self.start * self.fps)))
        b = int(min(self.n - 1, math.ceil(self.end * self.fps)))
        up = [j for j in range(a, b + 1) if any(s0 <= j / self.fps < s1 for s0, s1 in self.spans)]
        keep, faces = [self.heads[j] for j in up], [self.faces[j] for j in up]
        halo = self._halo(K_MAX)
        z = self.zone.rect.grow(-halo[0], -halo[1], -halo[2], -halo[3])
        rooms = bands(z, keep, faces, self.W, self.H, behind=False)
        pos = self.opts["position"]
        names = [ROOM_OF[pos]] if pos != "auto" else ["top", "below"]
        want = min(int(self.opts["most"]), len(self.notes))
        best = None
        for name in names:
            room = rooms.get(name)
            if room is None:
                continue
            w1, h1 = self._size(1.0)
            for cap in range(want, 0, -1):
                need = h1 * (cap + DROP) + GAP * self.U * (cap - 1) + FADE_ROOM * (h1 + GAP * self.U)
                k = min(K_MAX, room.w / w1, room.h / need)
                if k < K_MIN and cap > 1:
                    continue
                # more cards at once is worth a little size, but never type too small to read on a phone
                score = k * (1.0 + 0.12 * cap) * (1.0 if k >= READABLE else 0.6)
                if best is None or score > best[0]:
                    best = (score, k, cap, name, room)
        if best is None and pos != "auto":
            self.warnings.append(f"position={pos}: there is no room {pos} the head here, so the stack goes where "
                                 f"there is")
            self.opts = dict(self.opts, position="auto")
            return self._place(plan)
        if best is None:
            raise O.OptionError("there is no room clear of the face for a notification here (the face fills the "
                                "frame)")
        _, k, cap, name, room = best
        if k < K_MIN:
            self.warnings.append(f"the cards are small ({k:.0%} of their size): there is little room clear of the face")
        if cap < want:
            self.warnings.append(f"room for {cap} card{'s' if cap > 1 else ''} at once here: the older slide out")
        self.k, self.cap, self.where = float(k), cap, {"top": "above the head", "below": "under the chin"}[name]
        self.cw, self.ch = self._size(k)
        self.cw, self.ch = int(round(self.cw)), int(round(self.ch))
        self.gap = GAP * self.U * k
        boxes = [f for f in faces if f is not None]
        fx = Rect.around(boxes).cx if boxes else room.cx
        x0 = min(max(fx - self.cw / 2, room.x0), room.x1 - self.cw)
        stack_h = self.cap * self.ch + (self.cap - 1) * self.gap
        drop = DROP * self.ch
        if name == "top":
            # the stack's foot just above the head, with room for a card fading out under it; the drop
            # is in the room above
            y0 = room.y1 - stack_h - FADE_ROOM * (self.ch + self.gap)
        else:
            y0 = room.y0 + drop
        self.x0, self.y0 = float(x0), float(y0)
        self.room = room

    # -- drawing ------------------------------------------------------------------------------------------
    def _build(self) -> None:
        u = self.k * self.U
        self.u = u
        frosted = self.opts["style"] == "glass"
        self.pane = glassmod.Pane(self.cw, self.ch, RADIUS * u, unit=u, frosted=frosted, fill=SOLID)
        if frosted:
            frames = [self.frame_at(int(min(self.n - 1, max(0, round(t * self.fps)))))
                      for t in np.linspace(self.start, self.end, 5)]
            # the glass is toned for the whole height the stack may cover
            bright = max(self.pane.tone_over(frames, self.x0, self.y0 + i * (self.ch + self.gap))
                         for i in range(self.cap))
            self.contrast = glassmod.contrast_with_white(bright)
        else:
            self.contrast = glassmod.contrast_with_white(float(np.mean(SOLID)))
        head_px = max(8, int(round(HEAD_PX * u)))
        body_px = max(8, int(round(BODY_PX * u)))
        tile = TILE * u
        body_x = (PAD + TILE + 22) * u
        body_w = self.cw - body_x - PAD * u
        shadow = (0, 1.0 * u, 2.0 * u, 0.25)
        self.tile = self._tile(tile)
        now = textmod.draw_word("now", "poppins-semibold", head_px, fill=SOFT, shadow=shadow)
        self.cards = []
        for n in self.notes:
            n.lines = self._lines(n.text, body_w, body_px)
            rgb = np.zeros((self.ch, self.cw, 3), np.float32)
            a = np.zeros((self.ch, self.cw), np.float32)
            sp = Sprite(rgb, a)
            ty = (self.ch - tile) / 2 if len(n.lines) == 1 else PAD * u
            _lay(sp, self.tile, PAD * u, ty)
            head_base = PAD * u + textmod.cap_height("poppins-bold", head_px) + 2 * u
            who = n.sender
            room = body_w - now.width - 16 * u
            while who and textmod.advance(who, "poppins-bold", head_px) > room:
                who = (who[:-2] + "…") if len(who) > 2 else ""
            if who:
                wd = textmod.draw_word(who, "poppins-bold", head_px, fill=INK, shadow=shadow)
                _lay(sp, wd.sprite, body_x - wd.ox, head_base - wd.oy)
            _lay(sp, now.sprite, self.cw - PAD * u - now.width - now.ox, head_base - now.oy)
            base = head_base + 8 * u + BODY_PX * u * LINE * 0.86
            for i, line in enumerate(n.lines):
                lw = textmod.draw_word(line, "poppins-semibold", body_px, fill=(244, 245, 249), shadow=shadow)
                _lay(sp, lw.sprite, body_x - lw.ox, base + i * BODY_PX * u * LINE - lw.oy)
            self.cards.append(sp)

    def _tile(self, d: float) -> Sprite:
        """The card's tile: a rounded square in the accent with a speech mark of our own (a round
        bubble with a short tail low on its left), drawn four times larger and brought down."""
        from PIL import Image, ImageDraw

        k = 4
        n = int(math.ceil(d))
        big = Image.new("RGBA", (n * k, n * k), (0, 0, 0, 0))
        dr = ImageDraw.Draw(big)
        top = tuple(min(255, int(c * 1.06 + 10)) for c in self.accent)
        foot = tuple(int(c * 0.86) for c in self.accent)
        grad = np.zeros((n * k, n * k, 4), np.uint8)
        t = np.linspace(0, 1, n * k)[:, None]
        for c in range(3):
            grad[..., c] = (top[c] * (1 - t) + foot[c] * t).astype(np.uint8)
        mask = Image.new("L", (n * k, n * k), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, n * k - 1, n * k - 1), radius=TILE_R / TILE * d * k, fill=255)
        grad[..., 3] = np.asarray(mask)
        big = Image.fromarray(grad, "RGBA")
        dr = ImageDraw.Draw(big)
        s = n * k
        dr.ellipse((0.22 * s, 0.24 * s, 0.78 * s, 0.70 * s), fill=(255, 255, 255, 255))
        dr.polygon([(0.30 * s, 0.58 * s), (0.25 * s, 0.80 * s), (0.46 * s, 0.66 * s)], fill=(255, 255, 255, 255))
        for i, cx in enumerate((0.38, 0.50, 0.62)):  # three dots of typing, in the accent
            r = 0.04 * s
            dr.ellipse(((cx * s) - r, 0.47 * s - r, (cx * s) + r, 0.47 * s + r), fill=self.accent + (255,))
        img = big.resize((n, n), Image.LANCZOS)
        return Sprite.from_rgba(np.asarray(img))

    def _state(self, i: int, t: float) -> Optional[tuple[float, float]]:
        """Card i at `t`: (its top, its opacity), or None when it is not on screen."""
        n = self.notes[i]
        if t < n.at - ARRIVE:
            return None
        # how many places it has slid down: one for each card that has arrived after it
        g = self.group_of[i]
        g0, g1 = self.spans[g]
        if t >= g1:
            return None
        later = [m for m, gm in zip(self.notes[i + 1:], self.group_of[i + 1:]) if gm == g]
        rank = sum(_ease_in_out((t - (m.at - SLIDE)) / SLIDE) for m in later if t >= m.at - SLIDE)
        if rank >= self.cap:
            return None
        # past the last place a card does not slide out of the room: it fades where it is, a little lower
        alpha = 1.0 if rank <= self.cap - 1 else _clamp(self.cap - rank) ** 1.5
        place = min(rank, self.cap - 1 + FADE_ROOM * (rank - (self.cap - 1)))
        p = (t - (n.at - ARRIVE)) / ARRIVE
        y = self.y0 + place * (self.ch + self.gap) - (1.0 - _ease_out(p)) * DROP * self.ch
        alpha *= _ease_out(p * 1.4)
        if g1 < self.seconds and t > g1 - EXIT:
            alpha *= _ease_in_out((g1 - t) / EXIT)
        return y, alpha

    def draw(self, frame: np.ndarray, j: int, t: float) -> None:
        if not any(s0 <= t < s1 for s0, s1 in self.spans):
            return
        for i in range(len(self.notes)):  # the oldest first, so a newer card lies over it as it drops
            st = self._state(i, t)
            if st is None:
                continue
            y, a = st
            p = _clamp((t - (self.notes[i].at - ARRIVE)) / ARRIVE)
            scale = 0.96 + 0.04 * _ease_out(p)
            self.pane.draw(frame, self.x0, y, opacity=a, frost=a, scale=scale)
            card = self.cards[i] if scale > 0.995 else self.cards[i].scaled(scale)
            over(frame, card, self.x0 + (self.cw - card.w) / 2, y + (self.ch - card.h) / 2, a)

    # -- sound ----------------------------------------------------------------------------------------
    def mix(self, sound: np.ndarray, rate: int) -> np.ndarray:
        if not self.opts["sound"]:
            return sound
        hits = [(n.at - 0.02, _chime, 0.0) for n in self.notes if self.start <= n.at < self.end]
        out, self.chime_db = counting.mix_under(sound, rate, hits, CHIME_UNDER)
        return out

    # -- checks and plan ------------------------------------------------------------------------------
    def report(self) -> dict:
        over_face = outside = 0
        halo = self._halo(self.k)
        for j in range(self.n):
            t = j / self.fps
            if not (self.start <= t < self.end):
                continue
            f = self.faces[j]
            hit = out = False
            for i in range(len(self.notes)):
                st = self._state(i, t)
                if st is None or st[1] < 0.02:
                    continue
                r = Rect(self.x0, st[0], self.x0 + self.cw, st[0] + self.ch)
                hit |= f is not None and r.intersects(f)
                out |= not self.zone.contains(r.grow(*halo))
            over_face += hit
            outside += out
        return {"frames_over_face": over_face, "frames_outside_safe_zone": outside, "found": self.found,
                "lands": [f"“{n.text}” at {n.at:.2f} s (frame {math.ceil(n.at * self.fps - 1e-6)})"
                          + (f" on “{n.word}”" if n.word else "") for n in self.notes],
                "placed": f"{self.where}, {self.cap} at once, cards {self.cw}x{self.ch} at {self.k:.0%}, " +
                          ", ".join(f"{s0:.2f}-{s1:.2f} s" for s0, s1 in self.spans),
                "type_contrast": f"white on the card at {self.contrast:.1f} : 1 where it is brightest",
                "warnings": list(self.warnings)}

    def plan(self) -> dict:
        return {"where": self.where, "at_once": self.cap, "size": round(self.k, 3), "x": round(self.x0, 1),
                "y": round(self.y0, 1), "card": [self.cw, self.ch], "start": round(self.start, 3),
                "end": round(self.end, 3), "found": self.found,
                "notes": [{"text": n.text, "at": round(n.at, 3), "from": n.sender, "how": n.how, "word": n.word,
                           "lines": list(n.lines)} for n in self.notes]}

    def sheet_times(self) -> list[float]:
        """The first card dropping in, then each card just after it lands, and the full stack."""
        out = [self.notes[0].at - ARRIVE * 0.45]
        out += [n.at + 1.5 / self.fps for n in self.notes]
        out.append(min(self.end - EXIT - 0.1, self.notes[-1].at + 0.9))
        out = sorted(set(round(min(self.seconds - 0.5 / self.fps, max(0.0, x)), 3) for x in out))
        while len(out) > 6:
            out.pop(len(out) // 2)
        return out


def _lay(dst: Sprite, src: Sprite, x: float, y: float) -> None:
    """Lay `src` onto `dst` (both premultiplied), changed in place."""
    H, W = dst.a.shape
    x0, y0 = int(round(x)), int(round(y))
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x0 + src.w), min(H, y0 + src.h)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    sa = src.a[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    dst.rgb[cy0:cy1, cx0:cx1] = src.rgb[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] + dst.rgb[cy0:cy1, cx0:cx1] * (1 - sa[..., None])
    dst.a[cy0:cy1, cx0:cx1] = sa + dst.a[cy0:cy1, cx0:cx1] * (1 - sa)


def _chime(rate: int, rng: np.random.Generator, level: float) -> np.ndarray:
    """A soft two-note chime of our own, made here: a rounded bell tone, then one a fourth above, each
    with a faint high partial, gone in a third of a second."""
    n = int(0.42 * rate)
    t = np.arange(n) / rate
    out = np.zeros(n)
    for start, hz, gain in ((0.0, 1318.5, 0.8), (0.085, 1760.0, 1.0)):
        tt = t - start
        on = tt >= 0
        env = np.where(on, np.exp(-np.maximum(tt, 0) / 0.11) * np.minimum(1.0, np.maximum(tt, 0) / 0.002), 0)
        out += gain * env * (np.sin(2 * np.pi * hz * tt) + 0.18 * np.sin(2 * np.pi * hz * 2.76 * tt))
    return (out / max(1e-9, np.abs(out).max()) * level).astype(np.float32)


def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from ..core import framing as framing_mod
    from ..core import video
    from ..core.timeline import Timeline
    from . import Job, PlanError

    info = clip.info
    fr = framing_mod.plan(info.width, info.height, info.rate, info.frames, clip.face, shape="keep")
    tl = Timeline.whole(info.frames / info.rate)

    def frame_at(j: int) -> np.ndarray:
        k = tl.source_frame(j, info.rate, info.frames)
        return fr.apply(video.read_frame(info, k), k)

    try:
        stack = Stack(clip, opts, (fr.out_w, fr.out_h), info.rate, tl.output_frames(info.rate),
                      head_in_output(clip, tl, fr), keepout_in_output(clip, tl, fr), frame_at, plan=plan)
    except ValueError as e:
        if plan is not None and not isinstance(e, O.OptionError):
            raise PlanError(str(e)) from None
        raise
    out_plan = {"effect": NAME, "clip": clip.path, "options": opts,
                "output": {"width": fr.out_w, "height": fr.out_h, "fps": round(info.rate, 3),
                           "seconds": round(tl.duration, 3), "framing": fr.note},
                "stack": stack.plan(), "sheet_times": stack.sheet_times()}

    def after(result: dict, out) -> dict:
        if opts["sound"] and getattr(stack, "chime_db", None) is not None:
            return {"sound": f"{len(stack.notes)} chimes, their loudest {-stack.chime_db:.0f} dB under the voice"}
        return {}

    times = stack.sheet_times()
    return Job(NAME, clip, tl, fr, stack.draw, out_plan, checks=stack.report(),
               still_times=lambda: times[:5], after=after, sound=stack.mix)
