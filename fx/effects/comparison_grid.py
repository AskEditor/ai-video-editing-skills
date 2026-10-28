"""comparison-grid: this versus that in two columns, or an if/then grid that fills row by row above the
head, each cell landing on the word that says it.

What it does, in order:

1. **The grid.** Heard in the words (`fx/core/contrasts.py`): "X versus Y" makes the two heads; a run
   of rules ("If your hook is slow, cut the first second. If nobody comments, ask a question.") makes
   the rows of an if/then grid; "Before, … Now, …" makes before/after rows. Or given:
   `items="Editing by hand | With AskEditor; Hours | Minutes; Every cut | One sentence"`, the first
   row the two heads, each cell able to carry its time (`Minutes@6.2`). `look` says which grid it is
   when the words do not.
2. **When a cell lands.** Still and whole on the first frame of its word (the word's start corrected
   for the transcriber's lateness by the core). A cell given without a time lands where its words are
   said (`items.time_items`), else between the others, and the run says so. The row being said is lit;
   the rows before it stay.
3. **Where.** On a card above the head (or beside it, or under the chin, where there is more room),
   as large as the room allows, never over the face, inside the safe zone with its shadow or its
   glass's reach; where the hair rises into it the person is laid back over it. Glass by default
   (`fx/core/glass.py`), solid or paper.
4. **Sound.** A soft pop under the voice as each cell lands, stepping up a little row by row.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..core import contrasts
from ..core import items as found_items
from ..core import moments
from ..core import text as textmod
from ..core.compose import Sprite, over
from ..core.layout import Rect, SafeZone
from . import lists as L
from . import options as O

NAME = "comparison-grid"
LOOKS = ("auto", "versus", "if-then", "before-after", "table")
ACCENTS = {"solid": "#ffcf3a", "glass": "#ffd84d", "paper": "#d0402b"}
MOST_ROWS = 4
LABELS_OF = {"if-then": ("If", "Then"), "before-after": ("Before", "Now")}

OPTIONS = {
    "items": O.Text("the grid. Empty: heard in your words (\"X versus Y\", \"if …, then …\", \"before, … now, …\"). "
                    "Or rows split by ; and cells by |, the first row the two heads: \"Phone | Camera; Cheap | Dear\". "
                    "A cell may carry its time: Dear@6.2"),
    "look": O.Choice(LOOKS, "auto: what the words say · versus: two things side by side with VS between · if-then: "
                            "rules, condition then result · before-after: then and now · table: two plain columns"),
    "title": O.Text("a heading across the top, such as \"Which one?\"; empty for none"),
    "position": O.Choice(L.POSITIONS, "auto: wherever there is most room · top: above your head · side: beside it · "
                                      "below: under your chin. Never over your face, whichever you pick"),
    "style": O.Choice(L.STYLES, "glass: frosted, your video blurred behind it · solid: a dark card · paper: a light card"),
    "accent": O.Colour("the accent: the VS, the heads' labels and the row being said (a name or a hex colour)"),
    "leave": O.Number(0.0, 60.0, "seconds the grid stays after its last cell before it goes; 0 keeps it to the end"),
    "hold": O.Number(0.0, 10.0, "seconds the finished grid stays on screen at the end, on the last frame; 0 for none"),
    "sound": O.Switch("a soft pop as each cell lands, under your voice: on or off"),
    "frame": O.Choice(("keep", "9:16"), "keep the clip's own shape, or make it 9:16 following the face"),
    "size": O.Size("auto (1080x1920 at most, never enlarged) or a size such as 1080x1920"),
    "fix": O.Text("misheard words as wrong=right, separated by ;  e.g. \"ask editor=AskEditor\""),
}

# None of our takes says "versus" or a run of rules, so D, B and 33 are given their cells,
# from words each clip really says, without times: each lands where it is said. E says its one rule
# as a question ("Want something more editorial? Then just ask for it."), heard on its own.
MATRIX = [
    ("D", "D", {"look": "before-after", "items": "Before | After; Made my way | The lively city; "
                                                 "Eventually went | My hotel"}),
    ("B", "B", {"look": "before-after",
                "items": "Before | After; Make a video | Improve your production quality; Edit it yourself | "
                         "An editor that edits for you"}),
    ("B-solid", "B", {"look": "versus", "style": "solid",
                      "items": "Production quality | An editor; Make a video | Edits for you"}),
    ("33", "33", {"look": "if-then", "items": "If | Then; Distribution | Stand out; Hooks | Draws people in; "
                                             "Marketing campaign | In-built video edit", "leave": 2.5}),
    ("E", "E", {}),
    ("33-paper", "33", {"style": "paper", "items": "Actual product | Better audio; Hooks | Marketing campaign",
                        "leave": 2.5}),
]


@dataclass
class GCell:
    text: str  # as said or given
    at: Optional[float]  # when its word starts; None for a label shown with the card
    how: str  # heard, said, given, placed, label
    word: str = ""
    land: float = 0.0  # the first frame's start
    frame: int = 0
    shown: str = ""
    lines: tuple = ()  # the shown words in one line, or two
    px: int = 0
    cut: int = 0
    row: int = 0  # 0 is the heads
    col: int = 0
    sprite: Optional[tuple] = None


# ---- the cells -------------------------------------------------------------------------------------------------


def parse_grid(spec: str, clip_dir: Path) -> list[list[tuple[str, Optional[float]]]]:
    """`A | B; c | d` (or JSON: [["A","B"],["c","d"]], or {"rows": …}) as rows of (text, time)."""
    data, _ = found_items.load(spec, clip_dir)
    if data is None:
        s = spec.strip()
        if not s:
            return []
        raw = [[c for c in re.split(r"\s*\|\s*", r.strip())] for r in re.split(r"[;\n]", s) if r.strip()]
    elif isinstance(data, dict):
        raw = data.get("rows") or []
    else:
        raw = data
    rows = []
    for r in raw:
        if not isinstance(r, list) or len(r) != 2:
            raise found_items.ItemsError(f"each row needs two cells split by |, such as \"Phone | Camera\" (got {r!r})")
        row = []
        for c in r:
            text, at, _ = found_items.entry(c, None, clip_dir)
            if not text:
                raise found_items.ItemsError("a cell has no words")
            if len(text) > 40:
                raise found_items.ItemsError(f"{text[:30]!r}… is too long for a cell; keep it under 40 letters")
            row.append((text, at))
        rows.append(row)
    if len(rows) > MOST_ROWS + 1:
        raise found_items.ItemsError(f"{len(rows) - 1} rows is more than a grid above the head shows readably; "
                                     f"keep it to {MOST_ROWS}")
    return rows


def cells_given(rows, words, quiet, duration, look: str, notes: list[str]) -> tuple[list[GCell], str]:
    """The given cells, timed: each lands where its words are said, else between the others. In an if/then
    or before/after grid a head that is only its label ("If", "Then") is shown with the card."""
    heads, body = rows[0], rows[1:]
    label_heads = look in LABELS_OF and [t.lower() for t, _ in heads] in (
        [x.lower() for x in LABELS_OF[look]], ["before", "after"], ["if", "then"])
    flat: list[GCell] = []
    timed_in = []
    for c, (t, at) in enumerate(heads):
        if label_heads and at is None:
            flat.append(GCell(t, None, "label", row=0, col=c))
        else:
            flat.append(GCell(t, at, "given", row=0, col=c))
            timed_in.append(flat[-1])
    for r, row in enumerate(body, 1):
        for c, (t, at) in enumerate(row):
            flat.append(GCell(t, at, "given", row=r, col=c))
            timed_in.append(flat[-1])

    @dataclass
    class _It:
        text: str
        at: Optional[float]
        image: Optional[str] = None

    its = [_It(g.text, g.at) for g in timed_in]
    timed, more = found_items.time_items(its, words, [], duration, back=moments.taken_back(words, quiet))
    notes.extend(more)
    for t in timed:
        g = timed_in[t.n]
        g.at, g.how, g.word = t.at, t.how, t.word
    _in_reading_order(flat, notes)
    return flat, "the cells given"


def _in_reading_order(cells: list[GCell], notes: list[str]) -> None:
    """A grid fills as it is read, row by row, left then right: a cell said before the one it follows is
    moved just after it, and the run says so."""
    last = -1.0
    moved = []
    for g in sorted(cells, key=lambda g: (g.row, g.col)):
        if g.at is None:
            continue
        if g.at < last + 0.2:
            moved.append(g.text)
            g.at = last + 0.2
            g.how = "after the cell before it"
        last = g.at
    if moved:
        notes.append("said before the cell it follows, so moved just after it: " + "; ".join(repr(m) for m in moved))


def cells_heard(words, quiet, look: str) -> tuple[list[GCell], str, str]:
    grids = contrasts.find(words, quiet)
    if look in ("versus", "if-then", "before-after"):
        grids = [g for g in grids if g.kind == look] or grids
    if not grids:
        return [], "", ""
    g = grids[0]
    out: list[GCell] = []
    if g.head_cells:
        for c, cell in enumerate(g.head_cells):
            out.append(GCell(cell.text, cell.at, "heard", words[cell.words[0]].clean, row=0, col=c))
    else:
        for c, label in enumerate(g.heads):
            out.append(GCell(label, None, "label", row=0, col=c))
    for r, (a, b) in enumerate(g.rows[:MOST_ROWS], 1):
        out.append(GCell(a.text, a.at, "heard", words[a.words[0]].clean, row=r, col=0))
        out.append(GCell(b.text, b.at, "heard", words[b.words[0]].clean, row=r, col=1))
    return out, g.kind, f"heard: \"{g.phrase}\""


# ---- the card ----------------------------------------------------------------------------------------------------


class GridCard(L.Card):
    """Two columns of cells under two heads, laid out once; each cell drawn from the frame of its word."""

    def __init__(self, cells: list[GCell], kind: str, title: str, style: str, accent, fps: float,
                 size: tuple[int, int], zone: SafeZone):
        super().__init__(L.CardSpec("grid", 0.0, 0.0, 0.0, True, False), [], title, style, accent, fps, size, zone)
        self.cells, self.kind = cells, kind
        self.n_rows = max([g.row for g in cells] + [0])
        self.body = 1.0  # a row's height in u: 1 for one line, TWO_LINES when a cell needs two
        self.modes = (False, True)  # one line first, then two (place() may hold it to one)
        if kind == "versus":  # room between the heads for the VS
            self.COL_GAP = 1.0

    TWO_LINES = 1.5
    TWO_PX = 0.80  # two lines are set this much smaller than one

    # sizes, as shares of u
    HEAD = 1.05  # the heads' row
    COL_GAP = 0.42  # between the columns, the divider in the middle
    CELL_PAD = 0.22

    def height(self, u: float, rows: int = 0) -> float:
        top = (L.TITLE_H * u + L.GAP * u) if self.title else 0.0
        return 2 * L.PAD * u + top + self._head_h(u) + self.n_rows * (self.body * u + L.GAP * u)

    def _head_h(self, u: float) -> float:
        labels = all(g.how == "label" for g in self.cells if g.row == 0)
        return self.HEAD * u * (1.0 if labels else self.body)

    def _px(self, g: GCell, u: float) -> tuple[str, int]:
        if g.row == 0:
            return ((self.p.face_bold, int(0.40 * u)) if g.how == "label" else (self.p.face_bold, int(0.50 * u)))
        return self.p.face_text, int(L.TEXT * u)

    def _shown(self, g: GCell) -> str:
        return g.text.upper() if g.how == "label" else g.text

    def fit(self, g: GCell, room: float, u: float, allow: int, two: bool) -> Optional[tuple[tuple, int, int]]:
        """A cell's words in `room`: one line as said; else (`two`, a body cell) two balanced lines as
        said; else one line made shorter (`allow`, as a list's rows are)."""
        face, px = self._px(g, u)
        text = self._shown(g)
        if textmod.advance(text, face, px) <= room:
            return (text,), px, 0
        ws = text.split()
        if two and g.how != "label" and len(ws) >= 2:
            p2 = max(4, int(px * self.TWO_PX))
            best = None
            for k in range(1, len(ws)):
                a, b = " ".join(ws[:k]), " ".join(ws[k:])
                wid = max(textmod.advance(a, face, p2), textmod.advance(b, face, p2))
                if wid <= room and (best is None or wid < best[0]):
                    best = (wid, (a, b))
            if best is not None:
                return best[1], p2, 0
        if allow < 1:
            return None
        got = L.fit_words(text, room, px, face, allow)
        return None if got is None else ((got[0],), got[1], got[2])

    def measure(self, u: float, band: Rect, allow: int) -> Optional[tuple[float, float, int]]:
        pad = L.PAD * u
        col = (band.w - 2 * pad - self.COL_GAP * u) / 2 - 2 * self.CELL_PAD * u
        if col < 1.4 * u:
            return None
        for two in self.modes:  # every cell on one line if it can be, else rows two lines tall
            self.body = self.TWO_LINES if two else 1.0
            h = self.height(u)
            if h > band.h + 0.5:
                continue
            widest, worst, ok = 0.0, 0, True
            for g in self.cells:
                got = self.fit(g, col, u, allow, two)
                if got is None:
                    ok = False
                    break
                lines, p, cut = got
                face, _ = self._px(g, u)
                widest = max([widest] + [textmod.advance(x, face, p) for x in lines])
                worst = max(worst, cut)
            if ok:
                break
        else:
            self.body = 1.0
            return None
        if self.title and textmod.advance(self.title, self.p.face_bold, int(0.44 * u)) > band.w - 2 * pad:
            return None
        w = 2 * pad + self.COL_GAP * u + 2 * (widest + 2 * self.CELL_PAD * u) + 0.12 * u
        w = min(band.w, max(w, L.MIN_SHARE * self.zone.rect.w, 5.0 * u))
        return w, h, worst

    def settle(self, place: L.Place, allow_cut: int) -> None:
        u = place.u
        self.u, self.x, self.y, self.w, self.h = u, float(round(place.x)), float(round(place.y)), place.w, place.h
        pad = L.PAD * u
        self.col_w = (self.w - 2 * pad - self.COL_GAP * u) / 2
        room = self.col_w - 2 * self.CELL_PAD * u
        self.top = pad + ((L.TITLE_H * u + L.GAP * u) if self.title else 0.0)
        self.body = 1.0
        if self.modes == (True,) or (any(self.fit(g, room, u, 0, False) is None for g in self.cells) and
                                     all(self.fit(g, room, u, 0, True) is not None for g in self.cells)):
            self.body = self.TWO_LINES
        two = self.body > 1.0
        for g in self.cells:
            got = self.fit(g, room, u, 2, two) or ((found_items.cut_to(self._shown(g), lambda q: False),),
                                                   self._px(g, u)[1], 2)
            g.lines, g.px, g.cut = got
            g.shown = " ".join(g.lines)
        self.h = self.height(u)
        self.cut = max([g.cut for g in self.cells] + [0])

    def slot(self, row: int, col: int) -> Rect:
        u = self.u
        pad = L.PAD * u
        x0 = pad + col * (self.col_w + self.COL_GAP * u)
        if row == 0:
            y0, h = self.top, self._head_h(u)
        else:
            y0, h = self.top + self._head_h(u) + L.GAP * u + (row - 1) * (self.body * u + L.GAP * u), self.body * u
        return Rect(x0, y0, x0 + self.col_w, y0 + h)

    def build(self, frames_behind: list) -> None:
        super().build(frames_behind)
        from PIL import ImageDraw

        u, p = self.u, self.p
        for g in self.cells:
            face, _ = self._px(g, u)
            colour = self.accent if g.how == "label" else p.ink
            g.sprite = [L.words_sprite(x, face, g.px, colour) for x in g.lines]
        # the lines: the divider between the columns and a hairline under the heads
        img = L.rgba(self.w, self.h)
        d = ImageDraw.Draw(img)
        line = p.line + (int(255 * p.line_alpha * 1.5),)
        lw = max(1, int(round(0.035 * u)))
        mid = L.PAD * u + self.col_w + self.COL_GAP * u / 2
        head = self.slot(0, 0)
        last = self.slot(self.n_rows, 0) if self.n_rows else head
        d.line((mid, head.y0 + 0.12 * u, mid, last.y1 - 0.08 * u), fill=line, width=lw)
        hy = head.y1 + L.GAP * u / 2
        d.line((L.PAD * u, hy, self.w - L.PAD * u, hy), fill=line, width=lw)
        # each row's empty slot, faint, so the grid promises what is coming
        for r in range(1, self.n_rows + 1):
            for c in (0, 1):
                s = self.slot(r, c)
                L.round_rect(img, (s.x0, s.y0, s.x1, s.y1), 0.18 * u, p.ink + (10,))
        self.lines = Sprite.from_rgba(np.asarray(img))
        self.lit = {}
        for r in range(1, self.n_rows + 1):
            s = self.slot(r, 0)
            box = L.rgba(self.w - 2 * L.PAD * u + 0.1 * u, s.h + 0.04 * u)
            L.round_rect(box, (0, 0, box.size[0], box.size[1]), 0.2 * u, self.accent + (30,))
            self.lit[r] = Sprite.from_rgba(np.asarray(box))
        self.badge = None
        if self.kind == "versus":
            dia = 0.86 * u
            img = L.rgba(dia + 2, dia + 2)
            d = ImageDraw.Draw(img)
            d.ellipse((1, 1, dia + 1, dia + 1), fill=self.accent + (255,))
            px = int(0.34 * u)
            f = textmod.font(p.face_bold, px)
            d.text(((dia + 2) / 2, (dia + 2) / 2), "VS", font=f, fill=tuple(p.on_accent) + (255,), anchor="mm")
            self.badge = Sprite.from_rgba(np.asarray(img))

    def place(self, room, H, want, head, notes, keep, faces):
        """One line in every cell where that reads large enough; two lines when they let the words be
        clearly larger (a grid over a lot of headroom, long cells)."""
        self.modes = (False,)
        one = super().place(room, H, want, head, notes, keep, faces)
        self.modes = (True,)
        two = super().place(room, H, want, head, [], keep, faces)
        if two is not None and (one is None or two.u * self.TWO_PX > 1.2 * one.u or one.cut):
            return two
        self.modes = (False,)
        return one

    def cell_state(self, g: GCell, t: float) -> tuple[float, float]:
        if g.at is None or t >= g.land:
            return 1.0, 0.0
        q = (t - (g.land - L.ENTER)) / L.ENTER
        if q <= 0:
            return 0.0, 1.0
        e = L.ease_out(q)
        return e, 1.0 - e

    def _draw(self, frame: np.ndarray, t: float) -> None:
        a, s = self.card_alpha(t)
        if a <= 0:
            return
        if self.pane is not None:
            if abs(s - 1.0) > 1e-3:
                self.pane.draw(frame, self.x + self.w * (1 - s) / 2, self.y + self.h * (1 - s) / 2, opacity=a,
                               frost=min(1.0, a * 1.2), scale=s)
            else:
                self.pane.draw(frame, self.x, self.y, opacity=a, frost=min(1.0, a * 1.2))
        panel, m = self.panel, self.panel_m
        if abs(s - 1.0) > 1e-3 or a < 0.999:
            sp = panel.scaled(s)
            over(frame, sp, self.x + self.w / 2 - sp.w / 2, self.y + self.h / 2 - sp.h / 2, a)
            if a < 0.98:  # the card arriving: nothing inside it yet
                return
        else:
            over(frame, panel, self.x - m, self.y - m)
        X, Y = self.x, self.y
        u = self.u
        # the row being said is lit: from its first cell's word until the next row's
        starts = {}
        for g in self.cells:
            if g.row and g.at is not None:
                starts[g.row] = min(starts.get(g.row, 1e9), g.land)
        now = [r for r, t0 in starts.items() if t >= t0]
        if now:
            r = max(now)
            sl = self.slot(r, 0)
            q = L.ease_out((t - starts[r] + L.ENTER) / L.ENTER)
            over(frame, self.lit[r], X + sl.x0 - 0.05 * u, Y + sl.y0 - 0.02 * u, a * q)
        over(frame, self.lines, X, Y, a)
        for g in self.cells:
            alpha, still = self.cell_state(g, t)
            if alpha <= 0:
                continue
            sl = self.slot(g.row, g.col)
            face, _ = self._px(g, u)
            cap = textmod.cap_height(face, g.px)
            lead = 1.18 * g.px
            n = len(g.lines)
            base = Y + sl.y0 + (sl.h + cap) / 2 - (n - 1) * lead / 2 + 0.26 * u * still
            for k, (x, (sp, ox, oy)) in enumerate(zip(g.lines, g.sprite)):
                tw = textmod.advance(x, face, g.px)
                over(frame, sp, X + sl.cx - tw / 2 - ox, base + k * lead - oy, a * alpha)
        if self.badge is not None:
            h0 = self.slot(0, 0)
            heads = [g for g in self.cells if g.row == 0]
            ba, _ = (min((self.cell_state(g, t) for g in heads), key=lambda v: v[0]) if heads else (1.0, 0.0))
            mid = L.PAD * u + self.col_w + self.COL_GAP * u / 2
            pop = 0.7 + 0.3 * L.back_out(ba, 2.2)
            b = self.badge.scaled(pop) if abs(pop - 1) > 1e-3 else self.badge
            over(frame, b, X + mid - b.w / 2, Y + h0.cy - b.h / 2, a * ba)

    def cues(self) -> list[tuple[float, str, float, float]]:
        out = []
        for k, g in enumerate(sorted((g for g in self.cells if g.at is not None), key=lambda g: g.land)):
            out.append((g.land, "pop", 2 ** (min(g.row, 5) * 2 / 12), 14.0))
        return out


# ---- the job ----------------------------------------------------------------------------------------------------


def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from ..core import framing as framing_mod
    from ..core import video
    from ..core.timeline import Timeline
    from . import Job, PlanError
    from .tier_list import PersonFront, bands, head_track

    info = clip.info
    fps = info.rate
    fr = framing_mod.plan(info.width, info.height, fps, info.frames, clip.face, shape=opts["frame"],
                          size=O.size_of(opts["size"]))
    W, H = fr.out_w, fr.out_h
    duration = info.frames / fps
    words, quiet = clip.words, clip.quiet
    notes: list[str] = []
    look = opts["look"]
    try:
        if plan is not None:
            cells, kind, source = _from_plan(plan, duration)
        elif opts["items"].strip():
            rows = parse_grid(opts["items"], Path(clip.path).parent)
            if len(rows) < 2:
                raise found_items.ItemsError("give the two heads and at least one row: \"Phone | Camera; Cheap | Dear\"")
            kind = look if look != "auto" else ("if-then" if [t.lower() for t, _ in rows[0]] == ["if", "then"] else
                                                 "before-after" if rows[0][0][0].lower() == "before" else "versus")
            cells, source = cells_given(rows, words, quiet, duration, kind, notes)
        else:
            cells, kind, source = cells_heard(words, quiet, look)
            if look not in ("auto", kind) and cells:
                notes.append(f"the words say a {kind} grid, so it is drawn as one")
    except found_items.ItemsError as e:
        raise PlanError(str(e)) from None
    if not cells:
        raise PlanError("no comparison was heard in the words (such as \"X versus Y\", \"if …, then …\", or "
                        "\"before, … now, …\"). Say it that way, or give the cells: items=\"Phone | Camera; Cheap | Dear\"")
    if look == "table":
        kind = "table"
    for g in cells:
        if g.at is not None:
            g.frame = L.frame_of(g.at, fps)
            g.land = g.frame / fps
    tl = Timeline.whole(duration).held(opts["hold"])
    n_out = tl.output_frames(fps)
    style = opts["style"]
    accent = L.accent_of(opts, style, ACCENTS)
    zone = SafeZone(W, H)
    card = GridCard(cells, kind, opts["title"], style, accent, fps, (W, H), zone)
    lands = [g.land for g in cells if g.at is not None]
    first, last = min(lands), max(lands)
    card.t_in = max(0.0, first - L.ENTER - L.CARD_LEAD)
    leave = float(opts["leave"] or 0)
    card.t_out = last + leave if leave > 0 and last + leave + L.CARD_OUT < tl.kept else None
    keep, faces = head_track(clip, tl, fr, notes)
    j_in = L.frame_of(card.t_in, fps)
    j_out = n_out if card.t_out is None else min(n_out, L.frame_of(card.t_out + L.CARD_OUT, fps) + 1)
    head = Rect.around(r for r in keep[j_in:j_out] if r is not None)
    front = None
    if head is not None:
        try:
            front = PersonFront(clip, tl, fr, (W, H))
        except Exception as e:  # no mask: the grid keeps clear of the head in every frame instead
            notes.append(f"the person could not be cut out ({e}); the grid keeps clear of the head in every frame")
    room = bands(zone.rect, keep[j_in:j_out], faces[j_in:j_out], W, H, behind=front is not None)
    card.keep, card.front = keep, front
    place = card.place(room, H, opts["position"], head, notes, keep[j_in:j_out], faces[j_in:j_out])
    if place is None:
        raise PlanError("there is no room off the face for the grid on this clip; a shot with more room above the "
                        "head, fewer rows or shorter cells make room for it")
    card.settle(place, 2)
    frames_behind = []
    if style == "glass":
        for t in (card.t_in + 0.2, 0.5 * (card.t_in + tl.kept), tl.kept - 0.2):
            k = int(min(info.frames - 1, max(0, round(t * fps))))
            frames_behind.append(fr.apply(video.read_frame(info, k), k))
    card.build(frames_behind)
    checks = L.check(card.rects, keep, faces, zone.rect, fps, n_out, front is not None, drawn_at=card.drawn)
    text_px = max(g.px for g in cells)
    if text_px < L.READABLE * H:
        notes.append(f"the room off the face is small on this clip, so the words are {text_px} px; a shot with more "
                     "room above the head, or fewer rows, makes them clearer")
    short = [g for g in cells if g.cut]
    if short:
        notes.append("made shorter to fit: " + "; ".join(f"{g.text!r} as {g.shown!r}" for g in short))
    on_word = [g for g in cells if g.at is not None]
    still_on = sum(1 for g in on_word if g.frame == L.frame_of(g.at, fps))
    report = {
        "grid": f"{kind}, {card.n_rows} row{'s' if card.n_rows != 1 else ''} under two heads, from {source}",
        "card": f"{place.where}, {card.w:.0f}x{card.h:.0f} px, rows {card.u:.0f} px, words {text_px} px",
        "lands": " | ".join(f"r{g.row}c{g.col + 1} {g.shown} " + (f"{g.land:.2f} s ({g.how}" + (f", '{g.word}'" if g.word
                                                                                                   else "") + ")"
                                                                    if g.at is not None else "with the card")
                            for g in sorted(cells, key=lambda g: (g.row, g.col))),
        "on_the_word": f"{still_on} of {len(on_word)} cells still and whole on the first frame of their word",
        **checks,
        "warnings": notes,
    }
    rests = sorted(set(round(g.land + 0.30, 3) for g in on_word))
    end = tl.duration - 0.5 / fps
    picks = [max(card.t_in + 1.0 / fps, first - L.ENTER - 1.0 / fps)] + rests
    if len(picks) > 5:
        picks = [picks[0]] + [picks[round(k * (len(picks) - 1) / 4)] for k in range(1, 5)]
    held = (card.t_out - 0.05) if card.t_out is not None else min(end, last + 1.0)
    sheet = sorted(set(round(min(end, max(0.0, x)), 3) for x in picks + [held]))[:6]
    cues = card.cues()
    sound = (lambda s, rate: L.mix_cues(cues, clip, s, rate)) if opts["sound"] and cues else None
    out_plan = {
        "effect": NAME, "clip": clip.path, "options": opts,
        "output": {"width": W, "height": H, "fps": round(fps, 3), "seconds": round(tl.duration, 3),
                   "framing": fr.note, "hold": tl.hold},
        "grid": {"kind": kind, "from": source, "in": round(card.t_in, 3),
                 "out": None if card.t_out is None else round(card.t_out, 3), "position": place.where,
                 "rect": card.rect.as_list(), "row_px": round(card.u, 1)},
        "cells": [{"row": g.row, "col": g.col + 1, "text": g.text, "shown": g.shown,
                   "at": None if g.at is None else round(g.at, 3), "how": g.how, "word": g.word}
                  for g in sorted(cells, key=lambda g: (g.row, g.col))],
        "sounds": [{"at": round(a, 4), "sound": k, "pitch": round(p, 4), "under_voice_db": u} for a, k, p, u in cues]
        if opts["sound"] else [],
        "sheet_times": sheet,
    }
    picks3 = [sheet[0], sheet[len(sheet) // 2], sheet[-1]]

    def after(result: dict, out: Path) -> dict:
        try:
            return L.measure_file(out, clip, card, cues if opts["sound"] else [])
        except Exception as e:  # a measurement that cannot be made never fails a finished render
            return {"measured": f"not measured ({e})"}

    return Job(NAME, clip, tl, fr, card.draw, out_plan, checks=report, still_times=lambda: picks3, sound=sound,
               after=after)


def _from_plan(plan: dict, duration: float) -> tuple[list[GCell], str, str]:
    from . import PlanError

    raw = plan.get("cells")
    grid = plan.get("grid") or {}
    if not isinstance(raw, list) or not raw:
        raise PlanError("the plan needs its cells; run once without --plan to get one to edit")
    out = []
    for c in raw:
        try:
            row, col = int(c["row"]), int(c["col"]) - 1
            at = None if c.get("at") is None else float(c["at"])
        except (KeyError, TypeError, ValueError):
            raise PlanError(f"a cell needs its row, its col (1 or 2) and a time or null: {c!r}") from None
        if col not in (0, 1) or not 0 <= row <= MOST_ROWS:
            raise PlanError(f"a cell sits at row 0 to {MOST_ROWS} and col 1 or 2: {c!r}")
        if at is not None and not 0 <= at <= duration:
            raise PlanError(f"the cell {c.get('text')!r} has a time outside the clip ({at})")
        text = str(c.get("text") or "").strip()
        if not text:
            raise PlanError("a cell has no words")
        out.append(GCell(text, at, "label" if at is None else "plan", str(c.get("word") or ""), row=row, col=col))
    return out, str(grid.get("kind") or "versus"), "the plan given"
