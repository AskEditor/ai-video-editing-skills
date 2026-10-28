"""A number that counts: its figures drawn so they can change on every frame, when each figure said
begins, and the soft ticks laid under a count. Shared by `chart` and `number-counter`.

    figs = counting.Figures(px, s)                    # the figures, set once at a size
    figs.draw(frame, ("$", "197", "K", ""), x, baseline, alpha)
    heard, corrected = counting.first_listen(clip)    # the transcriber's own words, and their true starts
    at = counting.start_of(amount, heard, corrected)  # when the voice begins on an amount
    ticks = counting.tick_times(segments, ease, start, end)
    sound, under = counting.mix_ticks(sound, rate, ticks)

**The figures** are drawn from single characters, so a count changes them on every frame without
setting the number again: each figure in a cell of one width (so the count does not wobble), the
sign before and the scale and unit after smaller.

**When a figure is said** is read from the transcriber's first listen, not from the words a second
listen changed: a figure read aloud takes far longer than its few characters suggest ("197" is "one
hundred and ninety-seven"), so a second listen can take it for a stretched word and bring it back
wrong ("19.5%" as "19 17 .5"). Its start is the core's corrected one (the transcriber marks a word
late; the core pulls it back and checks it against the sound).

**The ticks** run under the count, spaced by how fast it moves (they bunch while it runs and slow
as it settles), with a rounder one where it lands. Each is set by how loud it is heard over 30 ms,
as the voice's loud syllables are measured, never by its peak: a click's peak says little about how
loud it sounds. Every sound here is made in code, the same every time.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import numpy as np

from . import text as textmod
from . import words as wordsmod
from .compose import over
from .words import unheard

FACE = "poppins-bold"
SMALL = 0.6  # the sign, the scale letter and the unit, beside the figures
WHITE = (255, 255, 255)
UNDER = 20.0  # dB: how far under the voice's loud syllables a tick is heard


# ---- the figures -----------------------------------------------------------------------------------

class Figures:
    """The big number drawn from single characters, so it can count on every frame without being
    set again: figures at full size, each in a cell of one width (so the count does not wobble), the
    sign before it and the scale and unit after it smaller."""

    def __init__(self, px: int, s: float, face: str = FACE, fill=WHITE, small: float = SMALL):
        self.px = px
        self.face = face
        self.fill = tuple(fill)
        self.small_px = max(8, int(round(px * small)))
        self.shadow = (0.0, 2.2 * s, 5.0 * s, 0.38)
        self.cap = textmod.cap_height(face, px)
        self.small_cap = textmod.cap_height(face, self.small_px)
        # every figure but 1 in a cell of one width, so the count does not wobble; 1 is narrow in
        # this face, and in a full cell it would stand apart from its neighbours ("1 97K")
        widths = [textmod.advance(c, face, px) for c in "023456789"]
        self.cell = sum(widths) / len(widths)
        self.cell_one = textmod.advance("1", face, px) + 0.06 * px
        self._glyphs: dict[tuple[str, bool], textmod.Word] = {}
        self._cache: dict[str, tuple[list, float]] = {}

    def glyph(self, ch: str, small: bool = False) -> textmod.Word:
        """One character, drawn once and kept: at full size, or `small` (the sign, the scale, the unit)."""
        return self._glyph(ch, small)

    def _glyph(self, ch: str, small: bool) -> textmod.Word:
        # every character is drawn through here (glass-widgets re-inks the figures by replacing it)
        key = (ch, small)
        if key not in self._glyphs:
            self._glyphs[key] = textmod.draw_word(ch, self.face, self.small_px if small else self.px, fill=self.fill,
                                                  shadow=self.shadow)
        return self._glyphs[key]

    def layout(self, parts: tuple[str, str, str, str]) -> tuple[list, float]:
        """Where each character of (sign, figures, scale, unit) goes: [(glyph, x, raise)] and the width."""
        key = "\x00".join(parts)
        if key in self._cache:
            return self._cache[key]
        pre, digits, scale, suf = parts
        items, x = [], 0.0
        lift = self.cap - self.small_cap  # a sign before the figures sits up, its top level with theirs
        for ch in pre:
            if ch == "-":
                g = self._glyph(ch, False)
                items.append((g, x, 0.0))
                x += g.width
            else:
                g = self._glyph(ch, True)
                items.append((g, x, lift))
                x += g.width + 0.04 * self.px
        for ch in digits:
            g = self._glyph(ch, False)
            if ch.isdigit():
                cell = self.cell_one if ch == "1" else self.cell
                items.append((g, x + (cell - g.width) / 2, 0.0))
                x += cell
            else:
                items.append((g, x, 0.0))
                x += g.width
        tail = scale + suf
        if tail:
            x += 0.01 * self.px
            for ch in tail:
                g = self._glyph(ch, True)
                items.append((g, x, 0.0))
                x += g.width
        self._cache[key] = (items, x)
        return items, x

    def width(self, parts) -> float:
        return self.layout(parts)[1]

    def draw(self, frame, parts, x: float, baseline: float, alpha: float) -> None:
        items, _ = self.layout(parts)
        for g, dx, up in items:
            over(frame, g.sprite, x + dx - g.ox, baseline - up - g.oy, alpha)


# ---- when a figure is said ---------------------------------------------------------------------------

def first_listen(clip) -> tuple[list, dict[int, float]]:
    """The transcriber's own words (with the corrections given), and for each, where the voice
    begins on it once the core has corrected its timing. Amounts are read from these (see above)."""
    first = wordsmod.apply_fixes(wordsmod.load(clip.path), clip.fix)
    quiet = clip.quiet
    kept = [i for i, w in enumerate(first) if w.key and not unheard(w, quiet)]
    timed = wordsmod.timed(first, quiet)
    return first, ({i: w.start for i, w in zip(kept, timed)} if len(kept) == len(timed) else {})


def start_of(amount, heard, corrected: dict[int, float]) -> float:
    """When the voice begins on `amount` (an `amounts.Amount` read from `heard`): its first word's
    corrected start, else the transcriber's start pulled back by how late it marks a word."""
    for i in range(amount.first, amount.last + 1):
        if i in corrected:
            return corrected[i]
    return max(0.0, heard[amount.first].start - wordsmod.HEARD_LATE)


# ---- the ticks ---------------------------------------------------------------------------------------

def tick(rate: int, rng: np.random.Generator, level: float) -> np.ndarray:
    """A small, rounded tick: a short knock of a tone with a breath of noise at its start."""
    n = int(0.018 * rate)
    t = np.arange(n) / rate
    body = np.sin(2 * np.pi * rng.uniform(1750, 2050) * t) * np.exp(-t / 0.0036)
    edge = rng.standard_normal(n) * np.exp(-t / 0.0006) * 0.22
    s = body + edge
    s *= np.minimum(1.0, t / 0.0004)
    return (s / max(1e-9, np.abs(s).max()) * level).astype(np.float32)


def land(rate: int, rng: np.random.Generator, level: float) -> np.ndarray:
    """The tick the count settles on: rounder and a little longer."""
    n = int(0.06 * rate)
    t = np.arange(n) / rate
    s = (np.sin(2 * np.pi * 1560 * t) * np.exp(-t / 0.011) * 0.8
         + np.sin(2 * np.pi * 2340 * t) * np.exp(-t / 0.007) * 0.35
         + rng.standard_normal(n) * np.exp(-t / 0.0007) * 0.2)
    s *= np.minimum(1.0, t / 0.0006)
    return (s / max(1e-9, np.abs(s).max()) * level).astype(np.float32)


def tick_times(segments: Sequence[tuple], ease: Callable[[float], float], start: float = -math.inf,
               end: float = math.inf) -> list[tuple[float, bool]]:
    """Each tick of the counts in `segments` ((from s, to s, from value, to value, …) each), as
    (seconds, is it the landing one): spaced by how fast `ease` moves the count, so they bunch while
    it runs and slow as it settles. Only the ticks from `start` to before `end` are kept."""
    out = []
    for seg in segments:
        a, b, v0, v1 = seg[0], seg[1], seg[2], seg[3]
        if b - a < 0.15 or abs(v1 - v0) < 1e-12:
            continue
        m = int(np.clip(round((b - a) * 15), 4, 16))
        qs = np.linspace(0, 1, 400)
        e = np.array([ease(q) for q in qs])
        for i in range(1, m):
            q = qs[int(np.searchsorted(e, i / m))] if i / m <= e[-1] else 1.0
            out.append((a + q * (b - a), False))
        out.append((b, True))
    return [(t, is_land) for t, is_land in out if start <= t < end]


def voice_level(sound: np.ndarray, rate: int) -> float:
    """How loud the voice's loud syllables are (the 90th percentile of 30 ms windows above the room)."""
    mono = sound.mean(axis=1)
    win = int(0.03 * rate)
    n = len(mono) // win
    rms = np.sqrt((mono[:n * win].reshape(n, win) ** 2).mean(axis=1) + 1e-12) if n else np.array([1e-3])
    loud = rms[rms > 10 ** (-45 / 20)]
    return float(np.percentile(loud, 90)) if loud.size else 0.05


def mix_under(sound: np.ndarray, rate: int, hits: Sequence[tuple[float, Callable, float]], under: float = UNDER,
              seed: int = 5) -> tuple[np.ndarray, Optional[float]]:
    """`sound` with a short sound laid at each of `hits` ((seconds, make, lift dB) each, where
    `make(rate, rng, level)` makes it), each heard `under - lift` dB under the voice, and how far
    under the voice the loudest of them sit (dB, negative), measured as they are heard."""
    if not hits:
        return sound, None
    win = int(0.03 * rate)
    voice = voice_level(sound, rate)

    def heard(x: np.ndarray) -> float:
        return float(np.sqrt((x[:win] ** 2).sum() / win))

    gains: dict[tuple[Callable, float], float] = {}
    rng = np.random.default_rng(seed)
    out = sound.copy()
    added = np.zeros(len(sound), np.float32)
    for t, make, lift in hits:
        key = (make, lift)
        if key not in gains:
            gains[key] = voice * 10 ** (-(under - lift) / 20) / max(1e-9, heard(make(rate, np.random.default_rng(0), 1.0)))
        s = make(rate, rng, gains[key])
        if s.ndim > 1:
            s = s.mean(axis=1)
        i = int(round(t * rate))
        if i >= len(out):
            continue
        k = min(len(s), len(out) - i)
        added[i:i + k] += s[:k]
    out += added[:, None]
    m = len(added) // win
    hit_rms = np.sqrt((added[:m * win].reshape(m, win) ** 2).mean(axis=1) + 1e-12) if m else np.array([0.0])
    busy = hit_rms[hit_rms > 1e-6]
    db = 20 * math.log10(max(1e-9, float(np.percentile(busy, 90)) if busy.size else 1e-9) / max(1e-9, voice))
    return np.clip(out, -1.0, 1.0), db


def mix_ticks(sound: np.ndarray, rate: int, ticks: Sequence[tuple[float, bool]], under: float = UNDER
              ) -> tuple[np.ndarray, Optional[float]]:
    """`sound` with a tick at each of `ticks` (from `tick_times`), the landing ones rounder and 2 dB
    nearer the voice; and how far under the voice they sit (dB, negative)."""
    return mix_under(sound, rate, [(t, land, 2.0) if is_land else (t, tick, 0.0) for t, is_land in ticks], under)
