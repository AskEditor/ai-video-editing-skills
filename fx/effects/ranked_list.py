"""ranked-list: items pop into a numbered list as they are said: a top-five board, a countdown, a rapid
run, or numbered tiles in an arc above the head.

What it does, in order (the engine is `lists.py`, shared with `checklist`):

1. **The items.** Given (`items="Canva; CapCut; Premiere"`, `5: Canva`, a time with `@`, or a .json
   file, with a picture per row if wanted) or heard: "number five … number one", "step two",
   "first … second … lastly", "then", "and finally", or a run with a last "and" ("three apps:
   Notion, Figma and Canva"). Given items keep their order and land where each is said.
2. **When.** Each row is still and whole on the first frame of its word, the word's start corrected
   for the transcriber's lateness; its entrance runs in the few frames before, and its sound lands
   on that frame.
3. **The numbers.** A board counts up in the order said. A countdown reveals in the order said
   while the numbers count down: the first item said is the highest number, number one comes last,
   and lands brighter, with a bell. Numbers said or given always win.
4. **Where.** Above the head, beside it or under the chin, as large as the room allows, inside the
   safe zone; a word too long for its row is made shorter by whole words, never cut inside one.
   The arc needs room above the head; where there is none it says so and uses the board.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..core import text as textmod
from ..core.compose import Sprite, over
from ..core.layout import Rect
from . import lists as L
from . import options as O

NAME = "ranked-list"
LOOKS = ("board", "countdown", "rapid", "arc")
ACCENTS = {"solid": "#ffc93c", "glass": "#ffd24d", "paper": "#d8412f"}
TOP_GLOW = (255, 214, 92)

OPTIONS = {
    "items": O.Text("what the list holds. Empty: found in your words (\"number one…\", \"first…, then…\", \"A, B and C\"). "
                    "Or Lighting; Sound@4.2; Framing (a time in seconds is optional; 5: Canva gives a number), "
                    "or a .json file"),
    "look": O.Choice(LOOKS, "board: numbered rows that fill as you name them · countdown: numbers count down to number one · "
                            "rapid: a quick run, the newest lit · arc: numbered tiles in an arc above your head"),
    "title": O.Text("a heading across the top, such as \"My top 5 apps\"; empty for none"),
    "position": O.Choice(L.POSITIONS, "auto: wherever there is most room · top: above your head · side: beside it · "
                                      "below: under your chin. Never over your face, whichever you pick"),
    "style": O.Choice(L.STYLES, "solid: a dark card · glass: frosted, your video blurred behind it · paper: a light card"),
    "accent": O.Colour("the colour of the numbers (a name or a hex colour; auto picks the look's own)"),
    "hold": O.Number(0.0, 10.0, "seconds the finished list stays on screen at the end, on the last frame; 0 for none"),
    "leave": O.Number(0.0, 60.0, "seconds the list stays after its last item before it goes; 0 keeps it to the end"),
    "sound": O.Switch("a soft pop as each item lands, under your voice (a bell on number one in a countdown): on or off"),
    "frame": O.Choice(("keep", "9:16"), "keep the clip's own shape, or make it 9:16 following the face"),
    "size": O.Size("auto (1080x1920 at most, never enlarged) or a size such as 1080x1920"),
    "fix": O.Text("misheard words as wrong=right, separated by ;  e.g. \"explain a serif=Explainer Serif\""),
}

# Each effect names its own runs on the test clips (tests/matrix.py). None of the controlled clips
# says a list, so their items are words each clip really says, given without times: each lands where
# it is said. The close-up says one ("make them serif, text behind or explainer serif"), found on its own.
MATRIX = [
    ("D", "D", {"items": "Lively town; Lively city; My hotel"}),
    ("B", "B", {"items": "Video; Production quality; Editor", "title": "What you get"}),
    ("B-arc", "B", {"look": "arc", "items": "Video; Production quality; Editor"}),
    ("B-glass", "B", {"look": "countdown", "style": "glass", "items": "Video; Production quality; Editor"}),
    ("A", "A", {"items": "Hooks; What draws people in; Marketing campaign"}),
    ("C", "C", {"items": "Product; Edit your videos; Get started", "fix": "Jon=John"}),
    ("33", "33", {"look": "countdown", "items": "Audio; UGC; Distribution; Hooks; Marketing campaign"}),
    ("close-looks", "close-looks", {"look": "rapid", "fix": "explain a serif=Explainer Serif"}),
]


# ---- the board and the countdown ---------------------------------------------------------------------------


class RankCard(L.Card):
    """Numbered rows. The empty slots show their numbers from the start, the next one to fill a
    little brighter; each fills as its item is said."""

    def build(self, frames_behind: list) -> None:
        super().build(frames_behind)
        u, p = self.u, self.p
        d = 0.76 * u
        self.badge_d = d
        self.badges = {}
        nums = sorted({r.number for r in self.rows})
        px = int(0.46 * u)
        for n in nums:
            for state in ("empty", "next", "full", "top"):
                self.badges[(n, state)] = self._badge(n, d, px, state)
        # where each number's slot is: number one at the top
        order = sorted(self.rows, key=lambda r: r.number)
        slots = [r.slot for r in sorted(self.rows, key=lambda r: r.slot.y0)]
        for r, s in zip(order, slots):
            r.slot = s
        room = self.w - self.text_x - L.PAD * u
        self.dash = self._dash(0.42 * room, 0.075 * u)
        self.current_bg = self._row_bg(self.w - 2 * L.PAD * u + 0.16 * u, u + 0.04 * u, (*p.ink, 16))
        self.top_bg = self._row_bg(self.w - 2 * L.PAD * u + 0.16 * u, u + 0.04 * u, (*TOP_GLOW, 30),
                                   ring=(*TOP_GLOW, 235))
        self.sweeps = [self._sweep(self.w - 2 * L.PAD * u, u, q / 11) for q in range(12)]
        self.glow = self._glow(1.9 * d)
        self.reveal = sorted(self.rows, key=lambda r: r.land)

    def _badge(self, n: int, d: float, px: int, state: str) -> Sprite:
        from PIL import ImageDraw

        p = self.p
        img = L.rgba(d + 2, d + 2)
        if state == "empty":
            fill, ink, a = p.ink + (int(255 * p.line_alpha * 0.75),), p.ink, 0.40
        elif state == "next":
            fill, ink, a = p.ink + (int(255 * p.line_alpha * 1.25),), p.ink, 0.95
        elif state == "top":  # gold, with dark ink on every style: light ink on gold does not read
            fill, ink, a = tuple(TOP_GLOW) + (255,), (24, 20, 12), 1.0
        else:
            fill, ink, a = self.accent + (255,), p.on_accent, 1.0
        L.round_rect(img, (1, 1, d + 1, d + 1), 0.24 * d, fill)
        label = str(n)
        size = px if len(label) == 1 else int(px * 0.82)
        cap = textmod.cap_height(p.face_bold, size)
        ImageDraw.Draw(img).text((1 + d / 2, 1 + d / 2 + cap / 2), label, font=textmod.font(p.face_bold, size),
                                 fill=tuple(ink) + (int(255 * a),), anchor="ms")
        return Sprite.from_rgba(np.asarray(img))

    def _dash(self, w: float, h: float) -> Sprite:
        img = L.rgba(w + 2, h + 2)
        L.round_rect(img, (1, 1, w + 1, h + 1), h / 2, self.p.ink + (int(255 * self.p.line_alpha * 0.8),))
        return Sprite.from_rgba(np.asarray(img))

    def _row_bg(self, w: float, h: float, fill, ring=None) -> Sprite:
        img = L.rgba(w + 2, h + 2)
        if ring is not None:  # an outline in the colour, then the fill inside it
            e = max(1.5, 0.045 * self.u)
            L.round_rect(img, (1, 1, w + 1, h + 1), 0.22 * self.u, ring)
            inner = L.rgba(w + 2, h + 2)
            L.round_rect(inner, (1 + e, 1 + e, w + 1 - e, h + 1 - e), 0.22 * self.u - e, (255, 255, 255, 255))
            a = np.asarray(img).copy()
            hole = np.asarray(inner.getchannel("A"), np.float32) / 255.0
            a[..., 3] = (a[..., 3] * (1 - hole)).astype(np.uint8)
            from PIL import Image

            img = Image.fromarray(a, "RGBA")
            fill_img = L.rgba(w + 2, h + 2)
            L.round_rect(fill_img, (1 + e, 1 + e, w + 1 - e, h + 1 - e), 0.22 * self.u - e, fill)
            img.alpha_composite(fill_img)
        else:
            L.round_rect(img, (1, 1, w + 1, h + 1), 0.22 * self.u, fill)
        return Sprite.from_rgba(np.asarray(img))

    def _sweep(self, w: float, h: float, q: float) -> Sprite:
        """A soft band of light partway across a row, clipped to the row's rounded box."""
        from PIL import Image

        box = L.rgba(w + 2, h + 2)
        L.round_rect(box, (1, 1, w + 1, h + 1), 0.22 * self.u, (255, 255, 255, 255))
        xs = np.arange(box.width, dtype=np.float32)
        centre = -0.25 * w + q * 1.5 * w
        band = np.exp(-((xs - centre) / (0.10 * w)) ** 2)
        a = np.tile(band[None, :], (box.height, 1)) * 0.12
        a *= np.asarray(box.getchannel("A"), np.float32) / 255.0
        light = (255, 255, 255) if self.style != "paper" else self.accent
        rgba = np.zeros(a.shape + (4,), np.uint8)
        rgba[..., :3] = light
        rgba[..., 3] = np.clip(a * 255, 0, 255).astype(np.uint8)
        return Sprite.from_rgba(np.asarray(Image.fromarray(rgba, "RGBA")))

    def _glow(self, d: float) -> Sprite:
        D = int(math.ceil(d))
        ys, xs = np.mgrid[0:D, 0:D].astype(np.float32)
        r = np.hypot(xs - D / 2, ys - D / 2) / (D / 2)
        a = np.clip(1.0 - r, 0, 1) ** 2
        rgba = np.zeros((D, D, 4), np.uint8)
        rgba[..., :3] = TOP_GLOW
        rgba[..., 3] = (a * 200).astype(np.uint8)
        return Sprite.from_rgba(rgba)

    def _next_up(self, t: float) -> Optional[L.Row]:
        for r in self.reveal:
            if t < r.land - L.ENTER:
                return r
        return None

    def _current(self, t: float) -> Optional[L.Row]:
        cur = None
        for r in self.reveal:
            if t >= r.land:
                cur = r
        return cur

    def _draw_row(self, frame, k: int, r: L.Row, t: float, X: float, Y: float, card_a: float) -> None:
        u = self.u
        s = r.slot
        top = self.spec.countdown and r.number == 1
        alpha, still = self.row_state(r, t)
        cur = self._current(t)
        if t >= r.land and (top or r is cur):
            bg = self.top_bg if top else self.current_bg
            over(frame, bg, X + s.x0 - 0.08 * u, Y + s.y0 - 0.02 * u, card_a)
        if t >= r.land and t - r.land < 0.42 and not top and self.style != "paper":
            q = (t - r.land) / 0.42
            over(frame, self.sweeps[min(11, int(q * 12))], X + s.x0, Y + s.y0, card_a * (1 - q * 0.5))
        # the number
        d = self.badge_d
        bx, by = X + self.lead_x, Y + s.y0 + (u - d) / 2
        if alpha <= 0:
            nxt = self._next_up(t)
            state = "next" if (nxt is r and t >= self.t_in + L.CARD_IN) else "empty"
            over(frame, self.badges[(r.number, state)], bx - 1, by - 1, card_a)
            # an empty slot: a dash where the words will go
            over(frame, self.dash, X + self.text_x - 1, Y + s.y0 + u / 2 - self.dash.h / 2, card_a)
            return
        if top and t >= r.land and t - r.land < 0.7:
            g = self.glow
            q = (t - r.land) / 0.7
            over(frame, g, bx + d / 2 - g.w / 2, by + d / 2 - g.h / 2, card_a * (1 - q) * 0.9)
        q = 1.0 - still
        scale = 0.6 + 0.4 * L.back_out(q, 2.0) if still > 0 else 1.0
        if t >= r.land and t - r.land < L.PULSE * 0.6:
            scale *= 1.0 + 0.06 * math.sin(math.pi * (t - r.land) / (L.PULSE * 0.6))
        sp = self.badges[(r.number, "top" if top else "full")]
        if abs(scale - 1.0) > 1e-3:
            ss = sp.scaled(scale)
            over(frame, ss, bx + d / 2 + 1 - ss.w / 2 - 1, by + d / 2 + 1 - ss.h / 2 - 1, card_a)
        else:
            over(frame, sp, bx - 1, by - 1, card_a)
        self._words(frame, r, t, X, Y, card_a * alpha, nudge=-0.40 * u * still)

    def cues(self) -> list[tuple[float, str, float, float]]:
        out = []
        for k, r in enumerate(sorted(self.rows, key=lambda r: r.land)):
            if self.spec.countdown and r.number == 1:
                out.append((r.land, "ding", 1.0, 15.0))
            else:
                out.append((r.land, "pop", 2 ** (min(k, 7) / 12), 17.0))
        return out


# ---- the rapid run ------------------------------------------------------------------------------------------


RAPID_GAP = 0.16
RAPID_ENTER = 0.12
RAPID_SCROLL = 0.14


class Rapid(L.Card):
    """A run of pills, one a word, the newest lit in the colour until the next arrives. More than
    fit roll up and away, a row at a time."""

    PILL_SHADOW = 0.22  # each pill's shadow reaches this far past it, as a share of u

    def halo(self, u: float) -> tuple[float, float, float, float]:
        """Each pill's shadow on every side; above, when more than three may have to roll, the row
        that rolls up and away as it fades."""
        m = float(math.ceil(self.PILL_SHADOW * u))
        roll = (1 + RAPID_GAP) * u if len(self.rows) > 3 else 0.0
        return m, m + roll, m, m

    def measure(self, u: float, band: Rect, allow: int) -> Optional[tuple[float, float, int]]:
        px = max(4, int(L.TEXT * u))
        lead = 0.66 * u
        room = band.w - lead - 0.62 * u
        if room < 1.4 * u:
            return None
        widths, worst = [], 0
        for r in self.rows:
            got = L.fit_words(r.said, room, px, self.p.face_text, allow)
            if got is None:
                return None
            widths.append(textmod.advance(got[0], self.p.face_text, got[1]))
            worst = max(worst, got[2])
        n = len(self.rows)
        fit = int((band.h + RAPID_GAP * u) // ((1 + RAPID_GAP) * u))
        k = min(n, fit)
        if k < min(n, 3):
            return None
        h = k * u + (k - 1) * RAPID_GAP * u
        w = lead + max(widths) + 0.62 * u
        return w, h, worst

    def settle(self, place: L.Place, allow_cut: int) -> None:
        u = place.u
        self.u, self.x, self.y, self.w, self.h = u, float(round(place.x)), float(round(place.y)), place.w, place.h
        px = max(4, int(L.TEXT * u))
        room = self.w - 0.66 * u - 0.62 * u
        for r in self.rows:
            got = L.fit_words(r.said, room, px, self.p.face_text, 2) or (r.said, px, 2)
            r.text, r.px, r.cut = got
        self.window = int(round((self.h + RAPID_GAP * u) / ((1 + RAPID_GAP) * u)))
        self.cut = max([r.cut for r in self.rows] + [0])

    def build(self, frames_behind: list) -> None:
        from PIL import ImageDraw

        u, p = self.u, self.p
        self.order = sorted(self.rows, key=lambda r: r.land)
        self.pills = {}
        for k, r in enumerate(self.order):
            tw = textmod.advance(r.text, p.face_text, r.px)
            w = 0.66 * u + tw + 0.62 * u
            for lit in (False, True):
                img = L.rgba(w + 2, u + 2)
                fill = self.accent + (255,) if lit else (p.card + (int(255 * max(0.86, p.card_alpha)),)
                                                         if self.style != "glass" else (16, 17, 22, 205))
                L.round_rect(img, (1, 1, w + 1, u + 1), u / 2, fill)
                dr = ImageDraw.Draw(img)
                d = 0.50 * u
                cx, cy = 1 + 0.16 * u + d / 2, 1 + u / 2
                dr.ellipse((cx - d / 2, cy - d / 2, cx + d / 2, cy + d / 2),
                           fill=(p.on_accent if lit else self.accent) + (255,))
                npx = int(0.30 * u)
                cap = textmod.cap_height(p.face_bold, npx)
                dr.text((cx, cy + cap / 2), str(r.number), font=textmod.font(p.face_bold, npx),
                        fill=(self.accent if lit else p.on_accent) + (255,), anchor="ms")
                ink = p.on_accent if lit else (p.ink if self.style != "glass" else (255, 255, 255))
                cap = textmod.cap_height(p.face_text, r.px)
                dr.text((1 + 0.66 * u + 0.08 * u, 1 + u / 2 + cap / 2), r.text, font=textmod.font(p.face_text, r.px),
                        fill=tuple(ink) + (255,), anchor="ls")
                m = int(math.ceil(self.PILL_SHADOW * u))
                self.pills[(k, lit)] = (Sprite.from_rgba(np.asarray(L.shadowed(img, 0.14 * u, 0.05 * u, 0.35, m))), m)
        self.panel = None
        self.pane = None

    def visible(self, t: float) -> bool:
        return any(t >= r.land - RAPID_ENTER for r in self.rows) and (self.t_out is None or t < self.t_out + L.CARD_OUT)

    def _scroll(self, t: float) -> float:
        """How many rows the run has rolled up by at t (fractional while it rolls)."""
        landed = [r.land for r in self.order if r.land - RAPID_ENTER <= t]
        n = len(landed)
        if n <= self.window:
            return 0.0
        base = n - 1 - self.window
        q = L.ease_in_out((t - (landed[-1] - RAPID_ENTER - RAPID_SCROLL * 0.2)) / RAPID_SCROLL)
        return base + q

    def _draw(self, frame: np.ndarray, t: float) -> None:
        u = self.u
        out_a = 1.0
        if self.t_out is not None and t >= self.t_out:
            out_a = 1.0 - L.ease_in_out((t - self.t_out) / L.CARD_OUT)
        roll = self._scroll(t)
        step = (1 + RAPID_GAP) * u
        for k, r in enumerate(self.order):
            if t < r.land - RAPID_ENTER:
                break
            y = self.y + (k - roll) * step
            if y < self.y - step + 0.5:
                continue
            fade = 1.0 if y >= self.y - 0.5 else max(0.0, 1.0 - (self.y - y) / step)
            nxt = self.order[k + 1] if k + 1 < len(self.order) else None
            lit = t >= r.land - RAPID_ENTER and (nxt is None or t < nxt.land - RAPID_ENTER)
            sp, m = self.pills[(k, lit)]
            if t < r.land:
                q = 1.0 - (r.land - t) / RAPID_ENTER
                scale = 0.7 + 0.3 * L.back_out(q, 2.2)
                a = L.ease_out(q * 1.6)
            else:
                scale, a = 1.0, 1.0
            if abs(scale - 1.0) > 1e-3:
                ss = sp.scaled(scale)
                cx = self.x - m + sp.w / 2 - (sp.w - ss.w) * 0.5 * 0  # grows from its own middle
                over(frame, ss, self.x - m + (sp.w - ss.w) / 2, y - m + (sp.h - ss.h) / 2, a * fade * out_a)
            else:
                over(frame, sp, self.x - m, y - m, a * fade * out_a)

    def rects(self, t: float) -> list[Rect]:
        if not self.visible(t):
            return []
        u = self.u
        roll = self._scroll(t)
        step = (1 + RAPID_GAP) * u
        out = []
        for k, r in enumerate(self.order):
            if t < r.land - RAPID_ENTER:
                break
            y = self.y + (k - roll) * step
            if y < self.y - step + 0.5:
                continue
            w = self.pills[(k, False)][0].w - 2 * self.pills[(k, False)][1]
            out.append(Rect(self.x, max(y, self.y), self.x + w, y + u))
        return out

    def drawn(self, t: float) -> list[Rect]:
        """The pills as drawn, a row rolling away above the run included, each with its shadow."""
        if not self.visible(t):
            return []
        u = self.u
        roll = self._scroll(t)
        step = (1 + RAPID_GAP) * u
        out = []
        for k, r in enumerate(self.order):
            if t < r.land - RAPID_ENTER:
                break
            y = self.y + (k - roll) * step
            if y < self.y - step + 0.5:
                continue
            sp, m = self.pills[(k, False)]
            out.append(Rect(self.x - m, y - m, self.x - m + sp.w, y - m + sp.h))
        return out

    def cues(self) -> list[tuple[float, str, float, float]]:
        return [(r.land, "tick", 2 ** (min(k, 9) / 12), 15.0) for k, r in enumerate(sorted(self.rows, key=lambda r: r.land))]


# ---- the arc above the head ---------------------------------------------------------------------------------


class Arc(L.Card):
    """Numbered tiles on an arc above the head, all one size, the first on the left, each popping out
    from the head onto its place as it is said. Tiles lean with the arc, under half as much, so they
    still read. A long item takes two lines rather than a smaller size."""

    LEAN_MOST = 16.0  # the end tiles lean this many degrees; the ones between, in proportion
    TILE_SHADOW = 0.2  # each tile's shadow reaches this far past it, as a share of u
    BADGE_OUT = 0.16  # the number badge stands this far out of the tile's left edge, as a share of u

    def halo(self, u: float) -> tuple[float, float, float, float]:
        """How far a tile's drawing reaches past its box (whose top already holds the badge): its
        shadow on every side, and the badge out to the side (to the left, or either way once leant)."""
        m = float(math.ceil(self.TILE_SHADOW * u))
        return m + self.BADGE_OUT * u, m, m + self.BADGE_OUT * u, m

    @staticmethod
    def _arch(n: int, cx: float, low_y: float, chord: float, sag: float, lean_end: float) -> list[tuple[float, float, float]]:
        """n points (x, y, lean) on an arch: its ends `chord` apart at low_y, its middle `sag` higher."""
        if n == 1:
            return [(cx, low_y, 0.0)]
        if sag <= 1e-3:
            return [(cx - chord / 2 + chord * k / (n - 1), low_y, 0.0) for k in range(n)]
        R = (chord * chord / 4 + sag * sag) / (2 * sag)
        ccy = (low_y - sag) + R
        a_end = math.degrees(math.asin(min(1.0, (chord / 2) / R)))
        out = []
        for k in range(n):
            a = a_end - 2 * a_end * k / (n - 1)  # from the left end (+) to the right (-)
            rad = math.radians(90.0 + a)
            out.append((cx + R * math.cos(rad), ccy - R * math.sin(rad), lean_end * a / a_end if a_end else 0.0))
        return out

    def place(self, room, H, want, head, notes, keep, faces) -> Optional[L.Place]:
        band = room.get("top")
        if band is None or head is None:
            notes.append("there is no room above the head for an arc on this clip, so the board is used instead")
            return None
        self.head_box = head
        zone = self.zone.rect
        floor_read = L.READABLE * H / L.TEXT
        for allow in (0, 1, 2):
            u = L.U_MAX * H
            while u >= floor_read:
                got = self._fit(u, allow, band, zone, head)
                if got is not None:
                    self.layout = got
                    box = Rect.around(t[0] for t in got[0])
                    return L.Place("top", band, u, box.w, box.h, box.x0, box.y0, max(t[5] for t in got[0]))
                u -= max(0.5, u * 0.02)
        notes.append("there is no room above the head for an arc on this clip, so the board is used instead")
        return None

    def _lines(self, said: str, u: float, allow: int, cap_w: float) -> Optional[tuple[list[str], int, int]]:
        """The item in one line, or two balanced lines, inside `cap_w`: (lines, px, cut)."""
        px = max(4, int(L.TEXT * u * 0.9))
        face = self.p.face_text
        if textmod.advance(said, face, px) <= cap_w:
            return [said], px, 0
        words = said.split()
        if len(words) >= 2:
            best = None
            for k in range(1, len(words)):
                a, b = " ".join(words[:k]), " ".join(words[k:])
                w = max(textmod.advance(a, face, px), textmod.advance(b, face, px))
                if w <= cap_w and (best is None or w < best[0]):
                    best = (w, [a, b])
            if best is not None:
                return best[1], px, 0
        got = L.fit_words(said, cap_w, px, face, allow)
        if got is None:
            return None
        return [got[0]], got[1], got[2]

    def _fit(self, u: float, allow: int, band: Rect, zone: Rect, head: Rect):
        """N tiles of one size on an arch inside the room above the head: its ends low at the sides of
        that room, its middle high, centred over the head. None when they do not fit at this size."""
        n = len(self.rows)
        m = 0.30 * u  # kept from the room's edges
        cx = min(max(band.cx, head.x0), head.x1)  # over the head, as near the middle of the room as that allows
        half = min(cx - band.x0, band.x1 - cx)
        if n > 1:
            cap_w = min(5.0 * u, 2 * (half - m) / (n - 1) * 1.05)
        else:
            cap_w = min(5.0 * u, 2 * half - 2 * m)
        if cap_w < 1.8 * u:
            return None
        lines = []
        for r in self.rows:
            got = self._lines(r.said, u, allow, cap_w - 0.8 * u)
            if got is None:
                return None
            lines.append(got)
        two = any(len(l[0]) > 1 for l in lines)
        h = (1.62 if two else 1.06) * u
        w = max(max(textmod.advance(x, self.p.face_text, l[1]) for x in l[0]) for l in lines) + 0.8 * u
        w = max(w, 1.9 * u)
        lean_end = self.LEAN_MOST if n > 1 else 0.0
        phi = math.radians(lean_end)
        bh_end = abs(w * math.sin(phi)) + abs(h * math.cos(phi))
        top_y = band.y0 + m + h / 2 + 0.25 * u  # the middle tile's centre, its badge kept inside the room
        low_y = band.y1 - m - bh_end / 2  # the end tiles' centre, leaning, kept above the head
        if low_y < top_y:
            return None
        chord = 2 * (half - m - w / 2) if n > 1 else 0.0
        if n > 1 and chord <= 0:
            return None
        sag = min(low_y - top_y, 0.28 * chord) if n > 1 else 0.0
        for _ in range(6):
            pts = self._arch(n, cx, low_y, chord, sag, lean_end)
            tiles = []
            for x, y, lean in pts:
                q = math.radians(lean)
                bw = abs(w * math.cos(q)) + abs(h * math.sin(q))
                bh = abs(w * math.sin(q)) + abs(h * math.cos(q))
                tiles.append((Rect(x - bw / 2, y - bh / 2 - 0.22 * u, x + bw / 2, y + bh / 2), x, y, lean))
            reach = self.halo(u)
            if all(t[0].grow(*reach).inside(zone) and not t[0].intersects(head) for t in tiles):
                break
            sag *= 0.75  # flatter, so the ends sit higher
        else:
            return None
        for k in range(len(tiles) - 1):
            a, b = tiles[k], tiles[k + 1]
            # the tile with its number badge, which stands off its top-left corner, and a little room
            if _tiles_touch((a[1] - 0.10 * u, a[2] - 0.11 * u, a[3]), (b[1] - 0.10 * u, b[2] - 0.11 * u, b[3]),
                            w + 0.20 * u + 0.16 * u, h + 0.22 * u):
                return None
        self.tile_w, self.tile_h = w, h
        out = [(t[0], t[1], t[2], t[3], w, l[2]) for t, l in zip(tiles, lines)]
        return out, [(l[0], l[1]) for l in lines]

    def settle(self, place: L.Place, allow_cut: int) -> None:
        tiles, words = self.layout
        self.layout = tiles
        self.u = place.u
        for r, (box, x, y, lean, w, cut), (lines, p) in zip(self.rows, tiles, words):
            r.text, r.px, r.cut = " ".join(lines), p, cut
            r.lines = lines
            r.slot = box
        box = Rect.around(t[0] for t in tiles)
        self.x, self.y, self.w, self.h = box.x0, box.y0, box.w, box.h
        self.cut = max([r.cut for r in self.rows] + [0])

    def build(self, frames_behind: list) -> None:
        from PIL import Image, ImageDraw

        u, p = self.u, self.p
        w, h = self.tile_w, self.tile_h
        self.tiles = []
        for r, (box, x, y, lean, _, cut) in zip(self.rows, self.layout):
            pad = int(math.ceil(0.5 * u))
            img = L.rgba(w + 2 * pad, h + 2 * pad)
            fill = p.card + (int(255 * max(0.90, p.card_alpha)),) if self.style != "glass" else (18, 19, 24, 218)
            L.round_rect(img, (pad, pad, pad + w, pad + h), 0.26 * u, fill)
            dr = ImageDraw.Draw(img)
            cap = textmod.cap_height(p.face_text, r.px)
            ink = p.ink if self.style != "glass" else (255, 255, 255)
            lines = r.lines
            step = 1.18 * cap * 1.6 if len(lines) > 1 else 0
            top = pad + h / 2 + cap / 2 - step * (len(lines) - 1) / 2
            for k, line in enumerate(lines):
                dr.text((pad + w / 2, top + k * step), line, font=textmod.font(p.face_text, r.px),
                        fill=tuple(ink) + (255,), anchor="ms")
            d = 0.58 * u
            bx, by = pad - 0.16 * u, pad - 0.22 * u
            dr.ellipse((bx, by, bx + d, by + d), fill=self.accent + (255,))
            npx = int(0.34 * u)
            ncap = textmod.cap_height(p.face_bold, npx)
            dr.text((bx + d / 2, by + d / 2 + ncap / 2), str(r.number), font=textmod.font(p.face_bold, npx),
                    fill=p.on_accent + (255,), anchor="ms")
            big = img.resize((img.width * 2, img.height * 2), Image.LANCZOS).rotate(lean, resample=Image.BICUBIC,
                                                                                  expand=True)
            rot = big.resize((max(1, big.width // 2), max(1, big.height // 2)), Image.LANCZOS)
            m = int(math.ceil(self.TILE_SHADOW * u))
            rot = L.shadowed(rot, 0.14 * u, 0.06 * u, 0.32, m)
            self.tiles.append((Sprite.from_rgba(np.asarray(rot)), x, y))
        self.panel = None
        self.pane = None
        self.head_c = (self.head_box.cx, self.head_box.y0)

    def visible(self, t: float) -> bool:
        return any(t >= r.land - L.ENTER for r in self.rows) and (self.t_out is None or t < self.t_out + L.CARD_OUT)

    def _draw(self, frame: np.ndarray, t: float) -> None:
        out_a = 1.0
        if self.t_out is not None and t >= self.t_out:
            out_a = 1.0 - L.ease_in_out((t - self.t_out) / L.CARD_OUT)
        hx, hy = self.head_c
        for r, (sp, x, y) in zip(self.rows, self.tiles):
            if t < r.land - L.ENTER:
                continue
            if t < r.land:
                q = 1.0 - (r.land - t) / L.ENTER
                pull = 1.0 - L.ease_out(q)
                scale = 0.55 + 0.45 * L.back_out(q, 1.8)
                a = L.ease_out(q * 1.5)
                px, py = x + (hx - x) * 0.12 * pull, y + (hy - y) * 0.12 * pull
                ss = sp.scaled(scale)
                over(frame, ss, px - ss.w / 2, py - ss.h / 2, a * out_a)
            else:
                over(frame, sp, x - sp.w / 2, y - sp.h / 2, out_a)

    def rects(self, t: float) -> list[Rect]:
        if not self.visible(t):
            return []
        return [r.slot for r in self.rows if t >= r.land - L.ENTER]

    def cues(self) -> list[tuple[float, str, float, float]]:
        return [(r.land, "pop", 2 ** (min(k, 7) / 12), 17.0) for k, r in enumerate(sorted(self.rows, key=lambda r: r.land))]


def _corners(x: float, y: float, lean: float, w: float, h: float) -> list[tuple[float, float]]:
    c, s = math.cos(math.radians(lean)), math.sin(math.radians(lean))
    return [(x + dx * c + dy * s, y - dx * s + dy * c) for dx, dy in
            ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]


def _tiles_touch(a: tuple[float, float, float], b: tuple[float, float, float], w: float, h: float) -> bool:
    """Whether two leaning tiles (x, y, lean in degrees) of w x h overlap: no axis of either one
    separates them."""
    pa, pb = _corners(*a, w, h), _corners(*b, w, h)
    for poly in (pa, pb):
        for i in range(4):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % 4]
            nx, ny = y2 - y1, x1 - x2
            ra = [nx * px + ny * py for px, py in pa]
            rb = [nx * px + ny * py for px, py in pb]
            if max(ra) <= min(rb) or max(rb) <= min(ra):
                return False
    return True


# ---- the effect ---------------------------------------------------------------------------------------------


def _card(look: str, title: str, style: str, accent, countdown: bool) -> L.Card:
    if look == "rapid":
        spec = L.CardSpec("rapid", 0.0, 0.0, 0.0, False, True)
        return Rapid(spec, [], title, style, accent, 30.0, (1080, 1920), None)
    if look == "arc":
        spec = L.CardSpec("arc", 0.0, 0.0, 0.0, False, True)
        return Arc(spec, [], title, style, accent, 30.0, (1080, 1920), None)
    spec = L.CardSpec(look, 0.76, 0.0, 0.0, True, True, countdown=countdown)
    return RankCard(spec, [], title, style, accent, 30.0, (1080, 1920), None)


def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from . import PlanError

    style = opts["style"]
    accent = L.accent_of(opts, style, ACCENTS)
    look = opts["look"]
    countdown = look == "countdown"
    most = {"board": 10, "countdown": 10, "rapid": 20, "arc": 7}[look]
    card = _card(look, opts["title"], style, accent, countdown)
    try:
        return L.prepare_list(clip, opts, plan, name=NAME, card=card, most=most, countdown=countdown)
    except PlanError as e:
        if look != "arc" or "no room" not in str(e):
            raise
    # no room for an arc on this clip: the board, said plainly
    card = _card("board", opts["title"], style, accent, False)
    job = L.prepare_list(clip, opts, plan, name=NAME, card=card, most=10)
    job.checks["warnings"] = ["there is no room above the head for an arc on this clip, so the board is used "
                              "instead"] + list(job.checks.get("warnings", []))
    job.plan["look_used"] = "board"
    return job
