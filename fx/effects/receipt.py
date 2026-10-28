"""receipt: a list prints out like a till receipt above the head, a line as each item is said, with a
price beside each item when one is said and a total at the bottom.

What it does, in order (the engine is `lists.py`, shared with `ranked-list` and `checklist`):

1. **The lines.** Given (`items="Coffee $4; Bagel $3; Mic $45"`, a time with `@`) or heard in the
   words ("first … then …", "a coffee, a bagel and a mic"), each through `core/items`. Each lands
   still and whole on the first frame of its word (the word's start corrected for the
   transcriber's lateness by the core). A word taken back is never landed on (`core/moments`).
2. **The prices.** An amount said inside an item or just after it ("a coffee for four dollars") is
   read by `core/amounts` from the transcriber's own words and timed by `core/counting`, so a price
   prints on the frame its first word is said; the words of the amount leave the item's line. A
   price typed after an item is printed with it.
3. **The total.** `total=auto` prints one when a total is said ("that comes to fifty-two dollars"),
   on its word; else the sum of the prices, once at least two are said in one currency, half a
   second after the last. `total=off` never; `total="$52"` prints what is typed.
4. **The paper** is our own: warm thermal paper hanging from a printer's dark mouth, a torn edge
   of small teeth along its foot, the words in a mono face, dotted leaders to the prices, a dashed
   rule above the total. It grows downward a line at a time and the lines above never move.
5. **Room.** Above the head (else beside it, else under the chin), inside the safe zone with its
   shadow, never over the face, checked on every frame. Where there is too little room the lines
   are made smaller first (to a size that still reads on a phone); past that, the paper stops
   growing and the lines scroll up under the printer's mouth, the newest always shown.
6. **The sound.** A printer's chatter made in code as each line prints (a shorter one for a price,
   a longer one for the total), under the voice. `sound=off` keeps the clip's sound exactly.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from ..core import amounts, counting, synth
from ..core import sound as soundmod
from ..core import text as textmod
from ..core.compose import Sprite, over
from . import lists as L
from . import options as O

NAME = "receipt"
FACE, BOLD = "plex-mono", "plex-mono-bold"
PAPER = (247, 244, 236)
INK = (44, 41, 38)
MOUTH = (30, 31, 35)
TOTAL_H = 1.45  # the total's rule and its line, as a share of u
TOOTH = 0.16  # the torn foot's teeth, as a share of u
COMFORT = 0.62  # lines are made no smaller than this share of the largest before the paper scrolls instead
MOST = 12
UNDER = 19.0  # dB: the printer under the voice
TOTAL_WORDS = {"total", "altogether", "overall", "comes", "came", "all", "that's", "thats", "sum", "spent", "bill"}

OPTIONS = {
    "items": O.Text("the lines. Empty: found in your words (\"first…, then…\", \"a coffee, a bagel and a mic\"). "
                    "Or Coffee $4; Bagel $3@2.5; Mic $45 (a price and a time in seconds are optional)"),
    "title": O.Text("the heading printed at the top; empty for RECEIPT"),
    "total": O.Text("auto: a total said, else the sum of the prices said · off: none · or the total to print, "
                    "such as $52"),
    "position": O.Choice(L.POSITIONS, "auto: above your head when there is room · top · side: beside it · "
                                      "below: under your chin. Never over your face, whichever you pick"),
    "sound": O.Switch("a printer's chatter under your voice as each line prints: on or off"),
}

LAB = "{lab}"
MATRIX = [
    ("D", "D", {"items": "Made my way; Lively city; My hotel", "title": "Hua Hin"}),
    ("B", "B", {"items": "Make a video; Production quality; An editor that edits for you"}),
    ("A", "A", {"items": "Hooks; What draws people in; Marketing campaign", "title": "Demo day"}),
    ("C", "C", {"items": "A product; Edit your videos; Get started"}),
    ("journey", "journey", {"title": "Trip"}),
    ("close-actions", "close-actions", {}),
    ("prices", "prices", {"title": "Today"}),
    ("D-priced", "D", {"items": "Slow walk $0@1.12; Lively city $12@7.38; My hotel $85@16.5", "title": "Hua Hin"}),
    ("B-long", "B", {"items": "Make a video; Just like this; Improve; Production; Quality; All in an editor; "
                              "That edits; For you", "title": "To do"}),
]


@dataclass
class Price:
    text: str  # as printed: "$4"
    at: float  # when it prints (a frame's start)
    amount: Optional[amounts.Amount] = None
    how: str = ""


# ---- prices and the total -------------------------------------------------------------------------

_TRAIL = re.compile(r"^(?P<name>.*?)[\s:–—-]+(?P<price>[$£€]?\d[\d,]*(?:\.\d+)?[kKmM]?)\s*(?P<at>@\s*[\d.]+)?\s*$")


def split_prices(spec: str) -> tuple[str, dict[int, amounts.Amount]]:
    """`items` with a price typed after each name taken off ("Coffee $4@2.5" → "Coffee@2.5"), and the
    prices by item number. Only the plain `a; b; c` form carries prices."""
    s = spec.strip()
    if not s or s.startswith("[") or s.startswith("{") or s.lower().endswith(".json"):
        return spec, {}
    sep = ";" if re.search(r"[;\n|]", s) else ","
    parts = [p.strip() for p in re.split(r"[;\n|]", s)] if sep == ";" else [p.strip() for p in s.split(",")]
    out, prices = [], {}
    for k, p in enumerate(x for x in parts if x):
        m = _TRAIL.match(p)
        if m and m.group("name").strip():
            try:
                prices[k] = amounts.parse(m.group("price"))
                p = m.group("name").strip() + (m.group("at") or "").replace(" ", "")
            except ValueError:
                pass
        out.append(p)
    return "; ".join(out), prices


def written(a: amounts.Amount) -> str:
    return amounts.text(amounts.Unit.of([a]), a.value)


def _strip_said(text: str, said: str) -> str:
    """The item's words without the amount said in them, nor the word that led into it ("for")."""
    low = text.lower()
    k = low.rfind(said.lower().strip(" .,")) if said else -1
    if k < 0:
        return text
    head = text[:k].rstrip(" ,")
    head = re.sub(r"\s+(for|at|is|was|costs?|cost me|of|about|around|just)$", "", head, flags=re.I)
    rest = text[k + len(said):].strip(" ,.")
    out = (head + (" " + rest if rest else "")).strip()
    return out or text


# ---- the receipt ----------------------------------------------------------------------------------

class ReceiptCard(L.Card):
    def __init__(self, clip, total_opt: str, typed: dict[int, amounts.Amount], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = L.Palette(PAPER, 1.0, INK, INK, 0.30, PAPER, FACE, BOLD)
        self.clip = clip
        self.total_opt = total_opt.strip()
        self.typed = typed
        self.window = 0  # lines shown at once; 0 for all of them
        self.prices: list[Optional[Price]] = []
        self.total: Optional[Price] = None
        self.priced = False
        self.scroll_note = ""

    # -- what is printed, found once the rows are known
    def find_prices(self) -> None:
        if self.priced:
            return
        self.priced = True
        rows, fps = self.rows, self.fps
        self.prices = [None] * len(rows)
        for k, a in self.typed.items():
            if k < len(rows):
                self.prices[k] = Price(written(a), rows[k].land, a, "typed")
        heard, corrected = counting.first_listen(self.clip)
        said = [a for a in amounts.spoken(heard) if a.kind == "amount" and (a.marked or not a.weak)]
        timed = [(counting.start_of(a, heard, corrected), a) for a in said]
        used: set[int] = set()
        for k, r in enumerate(rows):
            if self.prices[k] is not None:
                continue
            nxt = rows[k + 1].at if k + 1 < len(rows) else math.inf
            ends = r.ends if r.ends is not None else r.at + 1.5
            for n, (t, a) in enumerate(timed):
                if n in used or not (r.at - 0.15 <= t < nxt - 0.05) or t > ends + 1.2:
                    continue
                used.add(n)
                f = L.frame_of(t, fps)
                self.prices[k] = Price(written(a), max(r.land, f / fps), a, "said")
                r.said = _strip_said(r.said, a.said)
                break
        end = self.clip.info.frames / fps
        last = max([r.land for r in rows] + [p.at for p in self.prices if p is not None])
        opt = self.total_opt.lower()
        if opt == "off":
            return
        if opt in ("", "auto"):
            for n, (t, a) in enumerate(timed):
                if n in used or t < max(r.land for r in rows) - 0.05:
                    continue
                before = {w.key for w in heard[max(0, a.first - 4):a.first]}
                if before & TOTAL_WORDS:
                    f = L.frame_of(t, fps)
                    self.total = Price(written(a), f / fps, a, "said")
                    return
            got = [p.amount for p in self.prices if p is not None and p.amount is not None]
            if len(got) >= 2 and len({a.family for a in got}) == 1 and got[0].family not in ("%", "x"):
                v = sum(a.value for a in got)
                unit = amounts.Unit.of(got)
                at = min(end - 0.3, last + 0.5)
                self.total = Price(amounts.text(unit, v), L.frame_of(at, fps) / fps, None, "the sum of the prices")
            return
        try:
            shown = written(amounts.parse(self.total_opt))
        except ValueError:
            shown = self.total_opt.upper()[:14]
        at = min(end - 0.3, last + 0.5)
        self.total = Price(shown, L.frame_of(at, fps) / fps, None, "typed")

    def _with_prices(self) -> None:
        widest = max([textmod.advance(p.text, FACE, 100) for p in self.prices if p is not None] + [0.0])
        if self.total is not None:
            widest = max(widest, textmod.advance(self.total.text, BOLD, 100))
        self.spec = replace(self.spec, trail=(widest / 100 * L.TEXT + 0.9) if widest else 0.0)

    # -- sizes
    def shown_rows(self, rows: int) -> int:
        return min(rows, self.window) if self.window else rows

    def height(self, u: float, rows: int, total: Optional[bool] = None) -> float:
        k = self.shown_rows(rows)
        h = 2 * L.PAD * u + (L.TITLE_H * u + L.GAP * u) + k * u + max(0, k - 1) * L.GAP * u + TOOTH * u
        if total is None:
            total = self.total is not None and rows >= len(self.rows)
        return h + (TOTAL_H * u if total else 0.0)

    def shown_h(self, t: float) -> float:
        order = sorted(r.land for r in self.rows)
        k = sum(1 for land in order if t >= land)
        h = self.height(self.u, max(1, k), total=False)
        if k < len(order) and t >= order[k] - L.ENTER and k >= 1:
            q = L.ease_out((t - (order[k] - L.ENTER)) / L.ENTER)
            h += (self.height(self.u, k + 1, total=False) - h) * q
        if self.total is not None and t >= self.total.at - L.ENTER:
            h += TOTAL_H * self.u * L.ease_out((t - (self.total.at - L.ENTER)) / L.ENTER)
        return min(self.h, h)

    def place(self, room, H, want, head, notes, keep, faces):
        """The largest paper that fits; when that would make the lines too small, fewer lines at once
        (the paper scrolls) at a comfortable size, but never fewer than three."""
        self.find_prices()
        self._with_prices()
        n = len(self.rows)
        self.window = 0
        full = super().place(room, H, want, head, notes, keep, faces)
        comfy = COMFORT * L.U_MAX * H
        if (full is not None and full.u >= comfy) or n <= 3:
            return full
        best = full
        for win in range(n - 1, 2, -1):
            self.window = win
            p = super().place(room, H, want, head, notes, keep, faces)
            if p is not None and (best is None or p.u > best.u * 1.04):
                best, chosen = p, win
                if p.u >= comfy:
                    break
        if best is full:
            self.window = 0
            return full
        self.window = chosen
        self.scroll_note = f"{chosen} lines at a time: the paper scrolls as the rest print"
        notes.append(f"there is room for {chosen} of the {n} lines at a size that reads, so the receipt scrolls")
        return super().place(room, H, want, head, notes, keep, faces)

    def settle(self, place, allow_cut: int) -> None:
        super().settle(place, allow_cut)
        u = self.u
        self.rows_top = self.rows[0].slot.y0 if self.rows else L.PAD * u
        k = self.shown_rows(len(self.rows))
        self.total_y = self.rows_top + k * u + max(0, k - 1) * L.GAP * u

    # -- drawn once
    def build(self, frames_behind) -> None:
        super().build(frames_behind)
        u = self.u
        px = max(4, int(L.TEXT * u))
        self.price_sp = []
        self.leaders = []
        dot = textmod.advance(".", FACE, px)
        right = self.w - L.PAD * u
        for r, p in zip(self.rows, self.prices):
            if p is None:
                self.price_sp.append(None)
                self.leaders.append(None)
                continue
            sp = L.words_sprite(p.text, FACE, px, INK)
            pw = textmod.advance(p.text, FACE, px)
            x0 = self.text_x + textmod.advance(r.text, FACE, r.px) + 0.25 * u
            x1 = right - pw - 0.25 * u
            dots = int(max(0, (x1 - x0) / dot))
            lead = L.words_sprite("." * dots, FACE, px, INK, 0.55) if dots >= 2 else None
            self.price_sp.append((sp, right - pw))
            self.leaders.append((lead, x1 - dots * dot) if lead is not None else None)
        if self.total is not None:
            tp = max(4, int(L.TEXT * u * 1.06))
            self.total_px = tp
            self.total_label = L.words_sprite("TOTAL", BOLD, tp, INK)
            self.total_value = L.words_sprite(self.total.text, BOLD, tp, INK)
            self.total_vx = right - textmod.advance(self.total.text, BOLD, tp)
            n = int((self.w - 2 * L.PAD * u) / textmod.advance("-", FACE, px))
            self.total_rule = L.words_sprite("-" * n, FACE, px, INK, 0.6)

    def panel_at(self, h: float) -> tuple[Sprite, int]:
        """The paper at height h: warm thermal paper with a faint grain, the heading and its dashed
        rule, its foot torn into small teeth; hung from a dark printer's mouth; a soft shadow."""
        from PIL import Image, ImageDraw

        key = int(round(h / 2.0)) * 2
        if key in self._panels:
            return self._panels[key]
        u = self.u
        h = float(max(key, 2))
        W, Hh = int(math.ceil(self.w)), int(math.ceil(h))
        k = 3
        big = Image.new("RGBA", (W * k, Hh * k), (0, 0, 0, 0))
        tooth = TOOTH * u
        n = max(6, int(round(self.w / (0.34 * u))))
        pts = [(0, 0), (self.w, 0), (self.w, h - tooth)]
        for i in range(n, -1, -1):
            x = self.w * i / n
            y = h if i % 2 else h - tooth
            pts.append((x, y))
        pts.append((0, h - tooth))
        ImageDraw.Draw(big).polygon([(x * k, y * k) for x, y in pts], fill=PAPER + (255,))
        paper = big.resize((W, Hh), Image.LANCZOS)
        a = np.asarray(paper).astype(np.float32)
        rng = np.random.default_rng(9)
        grain = rng.normal(0.0, 2.0, a.shape[:2])[..., None]
        ys = np.arange(Hh, dtype=np.float32)[:, None, None]
        shade = 4.0 * np.exp(-ys / max(1.0, 0.5 * u))  # the paper just out of the mouth is a shade darker
        a[..., :3] = np.clip(a[..., :3] + (grain - shade) * (a[..., 3:] > 0), 0, 255)
        paper = Image.fromarray(a.astype(np.uint8), "RGBA")
        d = ImageDraw.Draw(paper)
        tp = int(0.46 * u)
        title = (self.title or "RECEIPT").upper()
        tw = textmod.advance(title, BOLD, tp)
        cap = textmod.cap_height(BOLD, tp)
        ty = L.PAD * u + (L.TITLE_H * u + cap) / 2 - 0.04 * u
        d.text(((self.w - tw) / 2, ty), title, font=textmod.font(BOLD, tp), fill=INK + (255,), anchor="ls")
        ly = L.PAD * u + L.TITLE_H * u + L.GAP * u * 0.2
        dash, gap = 0.16 * u, 0.10 * u
        x = L.PAD * u
        while x < self.w - L.PAD * u:
            d.line((x, ly, min(self.w - L.PAD * u, x + dash), ly), fill=INK + (150,), width=max(1, int(round(0.04 * u))))
            x += dash + gap
        m = int(math.ceil(L.SHADOW * u))
        out = L.shadowed(paper, 0.20 * u, 0.08 * u, 0.30, m)
        # the printer's mouth across the top, a little wider than the paper
        mouth_h, side, up = 0.40 * u, 0.20 * u, 0.28 * u
        L.round_rect(out, (m - side, m - up, m + self.w + side, m - up + mouth_h), 0.14 * u, MOUTH + (255,))
        slit = m - up + mouth_h * 0.62
        ImageDraw.Draw(out).line((m - side + 0.3 * u, slit, m + self.w + side - 0.3 * u, slit), fill=(8, 8, 10, 255),
                                 width=max(1, int(round(0.05 * u))))
        ImageDraw.Draw(out).line((m - side + 0.3 * u, m - up + 0.06 * u, m + self.w + side - 0.3 * u, m - up + 0.06 * u),
                                 fill=(78, 80, 88, 255), width=max(1, int(round(0.03 * u))))
        res = (Sprite.from_rgba(np.asarray(out)), m)
        self._panels[key] = res
        return res

    # -- each frame
    def offset(self, t: float) -> float:
        """How far the lines have scrolled up at t (0 until the paper is full)."""
        if not self.window:
            return 0.0
        step = self.u + L.GAP * self.u
        order = sorted(r.land for r in self.rows)
        k = sum(1 for land in order if t >= land)
        off = max(0, k - self.window) * step
        if k < len(order) and k >= self.window and t >= order[k] - L.ENTER:
            off += step * L.ease_out((t - (order[k] - L.ENTER)) / L.ENTER)
        return off

    def _draw(self, frame: np.ndarray, t: float) -> None:
        super()._draw(frame, t)
        if self.total is None or t < self.total.at - L.ENTER:
            return
        a, s = self.card_alpha(t)
        q = L.ease_out((t - (self.total.at - L.ENTER)) / L.ENTER) if t < self.total.at else 1.0
        u = self.u
        y = self.y + self.total_y + 0.12 * u
        sp, ox, oy = self.total_rule
        cap = textmod.cap_height(FACE, max(4, int(L.TEXT * u)))
        over(frame, sp, self.x + L.PAD * u - ox, y + 0.25 * u + cap / 2 - oy, a * q)
        base = y + 0.45 * u + (u + textmod.cap_height(BOLD, self.total_px)) / 2
        for (sp, ox, oy), x in ((self.total_label, self.x + L.PAD * u), (self.total_value, self.x + self.total_vx)):
            over(frame, sp, x - ox, base - oy, a * q)

    def _draw_row(self, frame, k: int, r: L.Row, t: float, X: float, Y: float, card_a: float) -> None:
        alpha, still = self.row_state(r, t)
        if alpha <= 0:
            return
        u = self.u
        off = self.offset(t)
        top = Y + self.rows_top
        y = Y + r.slot.y0 - off
        if y < top - 0.5:  # gone up under the heading as the paper scrolls
            alpha *= max(0.0, 1.0 - (top - y) / (0.7 * u))
            if alpha <= 0.01:
                return
        cap = textmod.cap_height(FACE, r.px)
        base = y + (r.slot.h + cap) / 2 + 0.10 * u * still  # each line comes up out of the paper as it prints
        sp, ox, oy = r.sprite
        over(frame, sp, X + self.text_x - ox, base - oy, card_a * alpha)
        p = self.prices[k] if k < len(self.prices) else None
        if p is None or t < p.at - L.ENTER / 2:
            return
        pa = alpha if t >= p.at else alpha * L.ease_out((t - (p.at - L.ENTER / 2)) / (L.ENTER / 2))
        lead = self.leaders[k]
        if lead is not None:
            ls, lx = lead
            over(frame, ls[0], X + lx - ls[1], base - ls[2], card_a * pa)
        ps, px0 = self.price_sp[k]
        over(frame, ps[0], X + px0 - ps[1], base - ps[2], card_a * pa)

    def rects(self, t: float):
        return super().rects(t)


def _printing(rows: list[L.Row], prices: list[Optional[Price]], total: Optional[Price]) -> list[tuple[float, float, str]]:
    """When the printer chatters: (the moment the line is done, how long it chatters, what)."""
    out = [(r.land, 0.18, "line") for r in rows]
    out += [(p.at, 0.08, "price") for p in prices if p is not None and abs(p.at - 0) > 0
            and all(abs(p.at - r.land) > 0.05 for r in rows)]
    if total is not None:
        out.append((total.at, 0.32, "total"))
    return sorted(out)


def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from . import PlanError

    spec = L.CardSpec("receipt", 0.0, 0.0, 0.0, False, False)
    items, typed = split_prices(opts["items"])
    card = ReceiptCard(clip, opts["total"], typed, spec, [], opts["title"] or "RECEIPT", "paper", INK, 30.0,
                       (1080, 1920), None)
    inner = {"items": items, "position": opts["position"], "sound": opts["sound"], "frame": "keep", "size": "auto",
             "hold": 0.0, "leave": 0}
    job = L.prepare_list(clip, inner, plan, name=NAME, card=card, most=MOST)
    job.plan["options"] = opts
    fps = card.fps
    for k, (r, p) in enumerate(zip(card.rows, card.prices)):
        if p is not None:
            job.plan["rows"][k]["price"] = p.text
            job.plan["rows"][k]["price_at"] = round(p.at, 4)
            job.plan["rows"][k]["text"] = r.text
    if card.total is not None:
        job.plan["total"] = {"text": card.total.text, "at": round(card.total.at, 4), "how": card.total.how}
    job.plan["window"] = card.window or len(card.rows)
    checks = job.checks
    priced = sum(1 for p in card.prices if p is not None)
    checks["prices"] = (" | ".join(f"{r.text} {p.text} at {p.at:.2f} s ({p.how})" for r, p in zip(card.rows, card.prices)
                                   if p is not None) if priced else "none said or typed")
    checks["total"] = (f"{card.total.text} at {card.total.at:.2f} s ({card.total.how})" if card.total is not None
                       else "none")
    checks["paper"] = card.scroll_note or f"all {len(card.rows)} lines shown, growing downward"
    if card.total is not None:
        sheet = sorted(set(job.plan["sheet_times"][:5] + [round(card.total.at + 0.3, 3)]))
        job.plan["sheet_times"] = sheet[-6:] if len(sheet) > 6 else sheet
    last = job.plan["output"]["seconds"] - 2.0 / fps  # the very last frame cannot be read back from the file
    job.plan["sheet_times"] = sorted({round(min(last, t), 3) for t in job.plan["sheet_times"]})
    cues = _printing(card.rows, card.prices, card.total)
    voice = soundmod.integrated(soundmod.blocks(clip.sound)) if clip.info.has_audio else soundmod.SILENCE_DB
    reference = voice if voice > -70 else -23.0

    def sound(s: np.ndarray, rate: int) -> np.ndarray:
        out = s.copy()
        stem = np.zeros_like(out)
        for n, (at, d, _) in enumerate(cues):
            x, anchor = synth.printer(d, seed=71 + n)
            gain = soundmod.gain((reference - UNDER) - soundmod.loudest(x))
            soundmod.lay(stem, x, at, rate, anchor=anchor, level=gain)
        out = out + stem
        peak = float(np.abs(out).max()) if len(out) else 0.0
        if peak > 0.98:
            out *= 0.98 / peak
        return out.astype(np.float32)

    job.sound = sound if opts["sound"] and cues else None
    job.plan["sounds"] = ([{"at": round(a, 4), "sound": f"printer ({what})", "seconds": d, "under_voice_db": UNDER}
                           for a, d, what in cues] if opts["sound"] else [])
    checks["sound"] = "a printer's chatter under your voice as each line prints" if job.sound else "your sound, untouched"
    if not card.rows:
        raise PlanError("nothing to print")
    picks = [job.plan["sheet_times"][0], job.plan["sheet_times"][len(job.plan["sheet_times"]) // 2],
             job.plan["sheet_times"][-1]]
    job.still_times = lambda: picks
    return job
