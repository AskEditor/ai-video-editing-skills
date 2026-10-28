"""checklist: steps tick off beside the head as they are said: a tick, a spinner that becomes a tick,
switches flipping on in a settings card of our own, a strike-through, or a bar that fills by steps.

What it does, in order (the engine is `lists.py`, shared with `ranked-list`):

1. **The steps.** Given (`items="Plug in the mic; Open the app; Press record"`, a time with `@`, a
   done time in a .json file) or heard in the words: "first … then … and then …", "step one",
   "next", "and finally", or a run with a last "and" ("it picks your best line, cuts you out, and
   pops you out"). Each step's words are the clause after its marker, without what only led into it.
2. **When a step appears.** Still and whole on the first frame of its word, the word's start
   corrected for the transcriber's lateness. In the settings card every row is there from the start,
   and a row lights as it is said.
3. **When it is ticked.** Where it is said done ("Done.", "Check.", "and that's it"), else where the
   next step starts (its "then", "next", "step three"), the last where its sentence ends. The tick
   starts on that frame, with its sound, and never on the frame the next step lands.
4. **The bar** fills by steps: it jumps a step's share on each tick and stands still between them,
   with a count beside it (2/4), never a made-up percentage.
5. **Where.** Above the head, beside it or under the chin, as large as the room allows, inside the
   safe zone; a step too long for its row is made shorter by whole words, never cut inside one.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..core import text as textmod
from ..core.compose import Sprite, over
from . import lists as L
from . import options as O

NAME = "checklist"
LOOKS = ("tick", "spinner", "switches", "strike", "progress")
ACCENTS = {"solid": "#2fd88f", "glass": "#3fe7a0", "paper": "#2b8a57"}
SPECS = {
    "tick": L.CardSpec("tick", 0.60, 0.0, 0.0, False, False),
    "spinner": L.CardSpec("spinner", 0.60, 0.0, 0.0, False, False),
    "switches": L.CardSpec("switches", 0.0, 1.12, 0.0, True, False),
    "strike": L.CardSpec("strike", 0.30, 0.0, 0.0, False, False),
    "progress": L.CardSpec("progress", 0.52, 0.0, 0.58, False, False),
}
SPIN = 400.0  # degrees a second
SWITCH = 0.18  # a switch's knob crossing
STRIKE = 0.24  # the strike drawing across
JUMP = 0.24  # the bar's jump to its next step

OPTIONS = {
    "items": O.Text("the steps. Empty: found in your words (\"first…, then…\", \"step two…\", \"A, B and C\"). "
                    "Or Plug in the mic; Open the app@3.1; Press record (a time in seconds is optional), or a .json "
                    "file, where a step can also have its own done time"),
    "look": O.Choice(LOOKS, "tick: a tick draws itself · spinner: a spinner turns, then becomes a tick · switches: a "
                            "settings card, each switch flipping on · strike: each step struck through · progress: "
                            "ticks with a bar that fills by steps"),
    "title": O.Text("a heading across the top, such as \"Before I post\"; empty for none"),
    "position": O.Choice(L.POSITIONS, "auto: wherever there is most room · top: above your head · side: beside it · "
                                      "below: under your chin. Never over your face, whichever you pick"),
    "style": O.Choice(L.STYLES, "solid: a dark card · glass: frosted, your video blurred behind it · paper: a light card"),
    "accent": O.Colour("the colour of a done step (a name or a hex colour; auto picks the look's own green)"),
    "hold": O.Number(0.0, 10.0, "seconds the finished list stays on screen at the end, on the last frame; 0 for none"),
    "leave": O.Number(0.0, 60.0, "seconds the list stays after its last tick before it goes; 0 keeps it to the end"),
    "sound": O.Switch("a soft tick as each step is done, under your voice (a click for a switch): on or off"),
    "frame": O.Choice(("keep", "9:16"), "keep the clip's own shape, or make it 9:16 following the face"),
    "size": O.Size("auto (1080x1920 at most, never enlarged) or a size such as 1080x1920"),
    "fix": O.Text("misheard words as wrong=right, separated by ;  e.g. \"ask editor=AskEditor\""),
}

# our own takes say two lists: the Hua Hin journey ("First of all, I went to the train station
# … I then boarded the train … and then eventually I got off the train … I eventually went to my hotel")
# and three actions in one breath. The controlled clips say none, so their steps are words each clip
# really says, given without times: each lands where it is said.
MATRIX = [
    ("journey", "journey", {}),
    ("close-actions", "close-actions", {"look": "spinner"}),
    ("D", "D", {"items": "Made my way; Lively city; My hotel"}),
    ("B", "B", {"look": "progress", "items": "Make a video; Improve your production quality; An editor that edits for you"}),
    ("B-switches", "B", {"look": "switches", "title": "Settings",
                         "items": "Make a video; Improve your production quality; An editor that edits for you"}),
    ("A", "A", {"look": "strike", "items": "Hooks; What draws people in; Marketing campaign"}),
    ("C", "C", {"items": "Product; Edit your videos; Get started", "fix": "Jon=John"}),
    ("33", "33", {"look": "progress", "title": "Demo day",
                  "items": "The actual product; Better audio; Distribution; Stand out"}),
]


class CheckCard(L.Card):
    def build(self, frames_behind: list) -> None:
        super().build(frames_behind)
        u, p = self.u, self.p
        self.order = sorted(self.rows, key=lambda r: r.land)
        self.n = len(self.rows)
        self.row_bg = self._box(self.w - 2 * L.PAD * u + 0.16 * u, u + 0.04 * u, 0.22 * u, p.ink + (17,))
        look = self.spec.look
        if look == "switches":
            self.track_w, self.track_h = self.spec.trail * u, 0.60 * u
            inset = 0.075 * u
            self.knob = self.track_h - 2 * inset
            self.inset = inset
            self.track_off = self._box(self.track_w, self.track_h, 0.21 * u, p.ink + (int(255 * p.line_alpha * 1.4),),
                                       edge=p.ink + (int(255 * p.line_alpha * 1.8),))
            self.track_on = self._box(self.track_w, self.track_h, 0.21 * u, self.accent + (255,))
            self.knob_off = self._knob(self.knob, False)
            self.knob_on = self._knob(self.knob, True)
        if look == "progress":
            self.bar_h = 0.20 * u
            self.counts = {}
            lp = int(0.38 * u)
            for k in range(self.n + 1):
                self.counts[k] = L.words_sprite(f"{k}/{self.n}", p.face_bold, lp, p.ink, 0.92)
            self.count_w = max(textmod.advance(f"{k}/{self.n}", p.face_bold, lp) for k in range(self.n + 1))
            self.bar_x0 = L.PAD * u
            self.bar_x1 = self.w - L.PAD * u - self.count_w - 0.30 * u
            bw = self.bar_x1 - self.bar_x0
            self.bar_track = self._box(bw, self.bar_h, self.bar_h / 2, p.ink + (int(255 * p.line_alpha * 1.2),))
            self.bar_fill = {}

    def _box(self, w: float, h: float, r: float, fill, edge=None) -> Sprite:
        img = L.rgba(w + 2, h + 2)
        if edge is not None:
            L.round_rect(img, (1, 1, w + 1, h + 1), r, edge)
            e = max(1.0, 0.03 * self.u)
            L.round_rect(img, (1 + e, 1 + e, w + 1 - e, h + 1 - e), max(0.0, r - e), fill)
        else:
            L.round_rect(img, (1, 1, w + 1, h + 1), r, fill)
        return Sprite.from_rgba(np.asarray(img))

    def _knob(self, d: float, on: bool) -> Sprite:
        from PIL import ImageDraw

        k = 4
        m = int(math.ceil(0.12 * self.u))
        img = L.rgba(d + 2, d + 2)
        L.round_rect(img, (1, 1, d + 1, d + 1), 0.30 * d, (255, 255, 255, 255))
        if on:
            from PIL import Image

            big = Image.new("RGBA", (int(math.ceil(d + 2)) * k, int(math.ceil(d + 2)) * k), (0, 0, 0, 0))
            pts = [((1 + x * 0.62 + d * 0.19) * k, (1 + y * 0.62 + d * 0.19) * k) for x, y in L.tick_points(d)]
            ImageDraw.Draw(big).line(pts, fill=self.accent + (255,), width=int(round(max(2.0, 0.11 * d) * k)),
                                     joint="curve")
            img.alpha_composite(big.resize(img.size, Image.LANCZOS))
        return Sprite.from_rgba(np.asarray(L.shadowed(img, 0.05 * self.u, 0.03 * self.u, 0.35, m))), m

    def _fill(self, share: float) -> Optional[Sprite]:
        bw = self.bar_x1 - self.bar_x0
        w = round(max(self.bar_h, bw * share), 0)
        if share <= 0:
            return None
        if w not in self.bar_fill:
            self.bar_fill[w] = self._box(w, self.bar_h, self.bar_h / 2, self.accent + (255,))
        return self.bar_fill[w]

    # -- what a row shows at t
    def _draw_row(self, frame, k: int, r: L.Row, t: float, X: float, Y: float, card_a: float) -> None:
        u = self.u
        look = self.spec.look
        s = r.slot
        alpha, still = self.row_state(r, t)
        if look == "switches":
            alpha, still = 1.0, 0.0
        elif alpha <= 0:
            return
        said = t >= r.land
        done = r.done is not None and t >= r.done
        if said and not done:
            over(frame, self.row_bg, X + s.x0 - 0.08 * u - 1, Y + s.y0 - 0.02 * u - 1, card_a * alpha)
        cy = Y + s.y0 + u / 2
        # the mark before the words
        if self.spec.lead:
            d = self.spec.lead * u
            mx, my = X + self.lead_x, cy - d / 2
            if look == "strike":
                kind = "done" if done else "ring"
                sp = self.mark(kind, d, 0.0 if kind == "done" else 1.0)
                over(frame, sp, mx - 1, my - 1, card_a * alpha)
            elif done:
                q = (t - r.done) / L.TICK_DRAW
                sp = self.mark("done", d, round(min(1.0, q) * 12) / 12)
                pop = 0.72 + 0.28 * L.back_out(min(1.0, (t - r.done) / 0.22), 2.4)
                if abs(pop - 1.0) > 1e-3:
                    ss = sp.scaled(pop)
                    over(frame, ss, mx - 1 + sp.w / 2 - ss.w / 2, my - 1 + sp.h / 2 - ss.h / 2, card_a * alpha)
                else:
                    over(frame, sp, mx - 1, my - 1, card_a * alpha)
            elif look == "spinner" and said:
                ang = (-(t - r.land) * SPIN) % 360
                over(frame, self.mark("spin", d, 1.0, round(ang / 10) * 10), mx - 1, my - 1, card_a * alpha)
            else:
                over(frame, self.mark("ring", d), mx - 1, my - 1, card_a * alpha)
        # the words
        if look == "switches":
            wa = 1.0 if said else 0.58
            self._words(frame, r, t, X, Y, card_a * wa)
        elif done:
            dim = 0.46 if look == "strike" else 0.66
            self._words(frame, r, t, X, Y, card_a * alpha * dim)
            if look == "strike":
                self._strike(frame, r, t, X, Y, card_a * alpha)
        else:
            self._words(frame, r, t, X, Y, card_a * alpha, nudge=-0.32 * u * still)
        # the switch after the words
        if look == "switches":
            self._switch(frame, r, t, X, Y, card_a)

    def _strike(self, frame, r: L.Row, t: float, X: float, Y: float, a: float) -> None:
        q = L.ease_out((t - r.done) / STRIKE)
        if q <= 0:
            return
        u = self.u
        tw = textmod.advance(r.text, self.p.face_text, r.px)
        cap = textmod.cap_height(self.p.face_text, r.px)
        base = Y + r.slot.y0 + (r.slot.h + cap) / 2
        th = max(2.0, 0.07 * u)
        w = (tw + 0.10 * u) * q
        key = ("strike", round(w))
        if key not in self._marks:
            self._marks[key] = self._box(w, th, th / 2, self.accent + (255,))
        over(frame, self._marks[key], X + self.text_x - 0.05 * u - 1, base - 0.40 * cap - th / 2 - 1, a)

    def _switch(self, frame, r: L.Row, t: float, X: float, Y: float, a: float) -> None:
        u = self.u
        q = 0.0 if r.done is None or t < r.done else L.ease_in_out((t - r.done) / SWITCH)
        tx = X + self.trail_x
        ty = Y + r.slot.y0 + (u - self.track_h) / 2
        over(frame, self.track_off, tx - 1, ty - 1, a)
        if q > 0:
            over(frame, self.track_on, tx - 1, ty - 1, a * q)
        knob, m = self.knob_on if q >= 0.5 else self.knob_off
        kx = tx + self.inset + q * (self.track_w - 2 * self.inset - self.knob)
        over(frame, knob, kx - 1 - m, ty + self.inset - 1 - m, a)

    def _draw_header(self, frame, t: float, X: float, Y: float, card_a: float) -> None:
        u = self.u
        done_times = sorted(r.done for r in self.rows if r.done is not None)
        k = sum(1 for d in done_times if t >= d)
        share = k / self.n
        if k and t - done_times[k - 1] < JUMP:  # jumping from the step before to this one, then still
            q = L.ease_out((t - done_times[k - 1]) / JUMP)
            share = ((k - 1) + q) / self.n
        fy = Y + self.header_y + (self.spec.header * u - self.bar_h) / 2
        over(frame, self.bar_track, X + self.bar_x0 - 1, fy - 1, card_a)
        fill = self._fill(share)
        if fill is not None:
            over(frame, fill, X + self.bar_x0 - 1, fy - 1, card_a)
        sp, ox, oy = self.counts[k]
        cap = textmod.cap_height(self.p.face_bold, int(0.38 * u))
        over(frame, sp, X + self.w - L.PAD * u - self.count_w - ox, fy + self.bar_h / 2 + cap / 2 - oy, card_a)

    def cues(self) -> list[tuple[float, str, float, float]]:
        out = []
        look = self.spec.look
        done = sorted((r for r in self.rows if r.done is not None), key=lambda r: r.done)
        for k, r in enumerate(done):
            if look == "switches":
                out.append((r.done, "click", 2 ** (min(k, 7) / 12), 13.0))
            elif k == len(done) - 1 and len(done) > 1:  # the list done: a soft bell instead of the last tick
                out.append((r.done, "ding", 1.0, 16.0))
            else:
                out.append((r.done, "tick", 2 ** (min(k, 7) / 12), 13.0))
        return out


def prepare(clip, opts: dict, plan: Optional[dict] = None):
    style = opts["style"]
    accent = L.accent_of(opts, style, ACCENTS)
    spec = SPECS[opts["look"]]
    card = CheckCard(spec, [], opts["title"], style, accent, 30.0, (1080, 1920), None)

    def ticks(rows: list[L.Row]) -> list[float]:
        return [r.done + 0.30 for r in rows if r.done is not None]

    return L.prepare_list(clip, opts, plan, name=NAME, card=card, most=8, steps=True, stills_extra=ticks)
