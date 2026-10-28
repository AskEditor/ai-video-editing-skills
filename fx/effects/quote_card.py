"""quote-card: a quote, a post or a clipping pops in as you read it out, each word lit as you say it.

What it does, in order:

1. **The quote.** Given (`text="…"`, several with `|`) and found where it is said, or found in the
   words: a line said as a quote, after "he said", "she told me", "the post says", "it reads",
   "and I quote", "they were like", up to the end of that sentence. With no quote said, the
   strongest line of the take stands as a pull quote (`moments.pick_line`, grown to its whole
   clause), and the run says so. Never a word taken back, a first attempt or an aside.
2. **Two looks of our own.** `post`: a pane of frosted glass with a round avatar, the name only
   when it is given (never made up; without one, two plain bars stand where a name would be), the
   words, and three plain marks under them. `clipping`: a cutting of newsprint, torn top and bottom
   and laid a little crooked, the source at its top only when given (otherwise a double rule), and
   a highlighter that runs along the words as they are said. No app's name, mark or layout.
3. **Each word lights on its first frame as it is said:** on the post, from dim to white (the
   quote's key word in the accent); on the clipping, the highlighter reaches the word on that
   frame. The card is still and whole by the first word: it comes in during the beat before it.
4. **Where.** Above the head or below the chin, wherever the card can be largest, inside the safe
   zone with its shadow, clear of the head over every frame it is up. A tight band gets a smaller
   card: first without the marks under the words, then without the name, then smaller type.
5. **Captions** (`captions=`) show everywhere else and step aside while the card is up, so the same
   words are never on screen twice. A soft pop (post) or paper laid down (clipping) under the voice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core import moments, synth
from ..core import sound as soundmod
from ..core import text as textmod
from ..core.compose import Sprite, over, rotated
from ..core.layout import Rect, SafeZone
from ..core.words import HESITATIONS, Word
from . import captions as cap
from . import layers
from . import options as O
from .captions import head_in_output
from .headline import find_said

NAME = "quote-card"
LOOKS = ("auto", "post", "clipping")

OPTIONS = {
    "look": O.Choice(LOOKS, "auto: a clipping when the quote is from a paper, an article or a headline, else a post "
                            "· post: frosted glass with an avatar and the words lighting up · clipping: a torn "
                            "cutting of newsprint, a highlighter running along the words"),
    "text": O.Text("the quote as you say it; several with |. Empty: the quote is found in your words (\"he said…\", "
                   "\"the post says…\"), else your strongest line as a pull quote"),
    "name": O.Text("who said or posted it, on the post card (\"Sam Lee\", \"Sam Lee @samlee\"). Never made up: "
                   "left empty, no name is shown"),
    "source": O.Text("where the clipping is from, printed small at its top (\"The Daily Ledger, 2024\"). Never "
                     "made up: left empty, a plain double rule"),
    "when": O.Text("auto, or a time in seconds: for a text you show but do not read out (its words are lit at once)"),
    "hold": O.Number(0.4, 4.0, "seconds the card stays after its last word"),
    "position": O.Choice(("auto", "top", "bottom"), "auto: wherever the card can be largest · top: above your "
                                                    "head · bottom: below your chin. Never over your face"),
    "accent": O.Colour("the key word on a post, the highlighter on a clipping: auto (yellow), a name or a hex colour"),
    "captions": O.Choice(("off",) + cap.LOOKS, "off · or the pack's captions in a look (stacked, one-word, "
                                               "serif-chip, fitted), hidden while the card is up"),
    "sound": O.Switch("on lays a soft pop (post) or paper laid down (clipping) under your voice; off keeps it exact"),
    "fix": O.Text("misheard words as wrong=right, separated by ;"),
}

ACCENT = (255, 212, 0)
ENTER = 0.30  # seconds the card takes to come in, ending on the first word
LEAVE = 0.24
MOST_WORDS = 22  # a quote longer than this is not a card, it is a transcript
MOST_CARDS = 3  # past three, quote cards are the film's wallpaper
UNDER_VOICE = 9.0  # dB the card's sound sits under the voice

# ---- finding a quote in the words ------------------------------------------------------------------

SAY = {"said", "says", "say", "told", "tells", "wrote", "writes", "posted", "posts", "commented", "comments",
       "tweeted", "texted", "messaged", "replied", "replies", "asked", "asks", "goes", "went", "reads", "read",
       "quote", "quoted", "typed", "emailed", "dmed"}
# The words a reported line comes after that are not part of it.
OBJECT = {"me", "us", "him", "her", "them"}  # "told me", "asked us": after a verb that takes one
LEAD = {"that", "like", "quote"}
# Who says it: a quote follows a say-word only after one of these (or a name, in capitals).
WHO = {"he", "she", "they", "someone", "somebody", "everyone", "everybody", "people", "it", "i", "you", "we",
       "post", "comment", "comments", "headline", "article", "review", "tweet", "message", "email", "dm", "caption",
       "sign", "title", "paper", "report", "study", "bio", "friend", "mum", "mom", "dad", "boss", "client",
       "customer", "manager", "founder", "teacher", "guy", "girl", "man", "woman", "viewer", "follower", "who"}
PRESS = {"article", "headline", "paper", "newspaper", "report", "study", "magazine", "news", "journal", "story",
         "column", "press"}
LIKE = ("was", "were", "is", "are", "'s")  # "she was like …"


@dataclass
class Quote:
    words: list[Word]  # the words said, in order (fillers left out), or none for a text not said
    text: list[str]  # the words as the card prints them, one per word
    why: str
    press: bool = False  # said to be from a paper, an article, a headline
    at: Optional[float] = None  # for a text not said: when it lands

    @property
    def start(self) -> float:
        return self.words[0].start if self.words else float(self.at or 0.0)

    @property
    def end(self) -> float:
        if self.words:
            return self.words[-1].end
        return self.start + 0.6 + 0.32 * len(self.text)

    def lit_at(self, k: int) -> float:
        """When word k lights: the start of its word, or the card's start for a text not said."""
        if self.words:
            return self.words[min(k, len(self.words) - 1)].start
        return self.start


def _printed(w: Word) -> str:
    t = w.clean.strip().strip("\"“”‘’")
    return t or w.text.strip()


def _tidy(text: list[str]) -> list[str]:
    """The words as a card prints them: the first in capitals, a trailing comma or a dangling dash
    dropped, nothing added."""
    out = list(text)
    if out:
        out[0] = out[0][:1].upper() + out[0][1:]
        out[-1] = out[-1].rstrip(",;:-—")
    return out


def _sentence_end(words: list[Word], i: int, most: int) -> int:
    j = i
    while j + 1 < len(words) and j - i + 1 < most and not words[j].ends_sentence:
        if words[j + 1].key in ("unquote", "endquote") or (words[j + 1].key == "end" and j + 2 < len(words)
                                                            and words[j + 2].key == "quote"):
            break
        j += 1
    return j


def said_quotes(words: list[Word], back: moments.TakenBack) -> list[Quote]:
    """Every line said as a quote: after a say-word that follows who said it ("he said", "the post
    says", "my client told me that"), "and I quote", or "was like", to the end of that sentence."""
    out: list[Quote] = []
    used = -1
    n = len(words)
    for i, w in enumerate(words):
        if i <= used:
            continue
        k = w.key
        prev = words[i - 1].key if i > 0 else ""
        prev2 = words[i - 2].key if i > 1 else ""
        start = None
        if k in SAY and k != "quote" and (prev in WHO or prev2 in WHO or (i > 0 and words[i - 1].clean[:1].isupper()
                                                                            and prev not in ("i",))):
            if k in ("went", "read") and prev not in ("it", "post", "sign", "headline", "he", "she", "they"):
                continue
            start = i + 1
        elif k == "quote" and prev in ("i", "and") and i + 1 < n:
            start = i + 1
        elif k == "like" and prev in LIKE and prev2 in WHO:
            start = i + 1
        if start is None or start >= n:
            continue
        while start < n and words[start].key in HESITATIONS:
            start += 1
        if start < n and words[start].key in OBJECT and k in ("told", "tells", "asked", "asks", "texted", "messaged",
                                                              "emailed", "replied", "dmed"):
            start += 1
        while start < n and (words[start].key in LEAD or words[start].key in HESITATIONS) and start - i <= 3:
            start += 1
        if start >= n:
            continue
        end = _sentence_end(words, start, MOST_WORDS)
        span = [x for x in words[start:end + 1] if x.key not in HESITATIONS]
        if len(span) < 3 or any(x in back for x in span):
            continue
        press = any(words[j].key in PRESS for j in range(max(0, i - 4), i + 1))
        who = " ".join(x.clean for x in words[max(0, i - 2):i + 1])
        out.append(Quote(span, _tidy([_printed(x) for x in span]), f"said as a quote (after “{who}”)", press))
        used = end
    return out


def pull_quote(words: list[Word], quiet, back: moments.TakenBack) -> Optional[Quote]:
    """No quote said: the take's strongest line (the shared picker), grown to the whole clause it
    sits in when that stays short, so the card reads as a sentence and not a label."""
    span = moments.pick_line(words, quiet, back=back)
    if span is None:
        return None
    a, b = span
    runs = moments.lines(words, quiet)
    for run in runs:
        if run[0] <= a <= run[-1]:
            ra, rb = run[0], run[-1]
            whole = [x for x in words[ra:rb + 1] if x.key not in HESITATIONS]
            while whole and whole[0].key in ("and", "but", "or", "so", "because", "then"):
                whole = whole[1:]  # a clause opened by "or" stands without it
                ra += 1
                while ra <= rb and words[ra].key in HESITATIONS:
                    ra += 1
            if (2 <= len(whole) <= 12 and not any(x in back for x in whole) and ra <= a
                    and (whole[-1].ends_clause or whole[-1].key not in moments.LEADS_ON)):
                a, b = ra, max(b, rb)
            break
    span_words = [x for x in words[a:b + 1] if x.key not in HESITATIONS]
    if len(span_words) < 2:
        return None
    return Quote(span_words, _tidy([_printed(x) for x in span_words]), "no quote is said: your strongest line, as a pull quote")


def given_quote(words: list[Word], text: str, when: Optional[float]) -> Quote:
    """A quote typed by the person: found where it is said, and lit word by word as said; the
    words printed are theirs (a word misheard is printed as typed)."""
    typed = [t for t in text.split() if t.strip()]
    if not typed:
        raise O.OptionError("text: give the quote's words")
    if len(typed) > MOST_WORDS:
        raise O.OptionError(f"text: {len(typed)} words is a transcript, not a quote card; keep it to {MOST_WORDS}")
    plain = [w for w in words if w.key not in HESITATIONS]
    hit = find_said(plain, text)
    if hit is not None and hit[2] >= max(2, len(typed) - 1):
        said = plain[hit[0]:hit[1] + 1]
        if len(said) == len(typed):
            times = said
        else:  # a word more or less heard: each typed word takes the said word at its share of the way
            times = [said[min(len(said) - 1, int(round(k * (len(said) - 1) / max(1, len(typed) - 1))))]
                     for k in range(len(typed))]
        return Quote(times, _tidy(typed), "as you typed it, where you say it")
    if when is None:
        raise O.OptionError(f"text: “{text}” is not said in this video. Give the words as you say them (read them "
                            f"with `python -m fx.run words <video>`), or when=12.5 to show it at a time without reading it")
    return Quote([], _tidy(typed), f"as you typed it, shown at {when:.2f} s (not read out, so lit at once)", at=when)


# ---- drawing helpers --------------------------------------------------------------------------------

def _ease_out(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def _rgba(w: float, h: float, k: int = 4):
    from PIL import Image

    return Image.new("RGBA", (max(1, int(round(w * k))), max(1, int(round(h * k)))), (0, 0, 0, 0))


def _down(img, w: float, h: float) -> Sprite:
    from PIL import Image

    return Sprite.from_rgba(np.asarray(img.resize((max(1, int(round(w))), max(1, int(round(h)))), Image.LANCZOS)))


def _circle(d: float, fill) -> Sprite:
    from PIL import ImageDraw

    img = _rgba(d, d)
    ImageDraw.Draw(img).ellipse((0, 0, img.width - 1, img.height - 1), fill=tuple(fill) + (255,))
    return _down(img, d, d)


def _silhouette(d: float, ground, ink) -> Sprite:
    """A round avatar with no one in particular in it: a head and shoulders."""
    from PIL import ImageDraw

    img = _rgba(d, d)
    S = img.width
    dr = ImageDraw.Draw(img)
    dr.ellipse((0, 0, S - 1, S - 1), fill=tuple(ground) + (255,))
    dr.ellipse((S * 0.34, S * 0.2, S * 0.66, S * 0.52), fill=tuple(ink) + (255,))
    dr.pieslice((S * 0.16, S * 0.58, S * 0.84, S * 1.22), 180, 360, fill=tuple(ink) + (255,))
    mask = _rgba(d, d)
    ImageDraw.Draw(mask).ellipse((0, 0, S - 1, S - 1), fill=(255, 255, 255, 255))
    arr = np.asarray(img).copy()
    arr[..., 3] = np.minimum(arr[..., 3], np.asarray(mask)[..., 3])
    from PIL import Image

    return _down(Image.fromarray(arr, "RGBA"), d, d)


def _bar(w: float, h: float, fill, alpha: float) -> Sprite:
    return textmod.rounded_box(w, h, h / 2, fill=fill, opacity=alpha)


def _mark(kind: str, s: float, colour, alpha: float = 1.0) -> Sprite:
    """The three plain marks under a post: a reply bubble, a loop of two arrows, a heart (outline or
    filled). Drawn here; no app's own."""
    from PIL import ImageDraw

    img = _rgba(s, s)
    S = img.width
    d = ImageDraw.Draw(img)
    c = tuple(colour) + (int(255 * alpha),)
    lw = max(2, int(S * 0.085))
    if kind == "reply":
        d.rounded_rectangle((lw, S * 0.14, S - lw, S * 0.72), radius=S * 0.22, outline=c, width=lw)
        d.polygon([(S * 0.26, S * 0.66), (S * 0.22, S * 0.92), (S * 0.48, S * 0.7)], fill=c)
    elif kind == "loop":
        d.arc((S * 0.1, S * 0.18, S * 0.9, S * 0.82), 200, 340, fill=c, width=lw)
        d.arc((S * 0.1, S * 0.18, S * 0.9, S * 0.82), 20, 160, fill=c, width=lw)
        d.polygon([(S * 0.80, S * 0.22), (S * 0.97, S * 0.40), (S * 0.70, S * 0.44)], fill=c)
        d.polygon([(S * 0.20, S * 0.78), (S * 0.03, S * 0.60), (S * 0.30, S * 0.56)], fill=c)
    else:
        pts = []
        for i in range(72):
            t = 2 * math.pi * i / 72
            x = 16 * math.sin(t) ** 3
            y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
            pts.append((S / 2 + x * S / 37, S * 0.47 - y * S / 37))
        if kind == "heart-on":
            d.polygon(pts, fill=c)
        else:
            d.line(pts + [pts[0]], fill=c, width=lw, joint="curve")
    return _down(img, s, s)


def _layer(canvas: tuple[np.ndarray, np.ndarray], sp: Sprite, x: float, y: float, alpha: float = 1.0) -> None:
    """A sprite laid into a card-sized premultiplied canvas."""
    rgb, a = canvas
    H, W = a.shape
    x0, y0 = int(round(x)), int(round(y))
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x0 + sp.w), min(H, y0 + sp.h)
    if cx1 <= cx0 or cy1 <= cy0 or alpha <= 0:
        return
    sa = sp.a[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] * alpha
    srgb = sp.rgb[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] * alpha
    rgb[cy0:cy1, cx0:cx1] = rgb[cy0:cy1, cx0:cx1] * (1 - sa[..., None]) + srgb
    a[cy0:cy1, cx0:cx1] = a[cy0:cy1, cx0:cx1] * (1 - sa) + sa


def ink_box(sp: Sprite, least: float = 0.03) -> Optional[Rect]:
    """The box of a sprite's visible pixels (alpha above `least`): what the safe zone measures,
    shadow included."""
    ys, xs = np.nonzero(sp.a > least)
    if not len(xs):
        return None
    return Rect(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


# ---- the card's layout -----------------------------------------------------------------------------

@dataclass
class Placed:
    word: int
    line: int
    x: float  # pen position in the card
    base: float  # baseline in the card
    w: float


@dataclass
class Layout:
    variant: str  # full, plain (no marks under), bare (no name either)
    px: int
    w: float
    h: float
    words: list[Placed] = field(default_factory=list)
    lines: int = 0


class Card:
    """One quote's card: built once at the size that fits, then drawn each frame it is up."""

    def __init__(self, quote: Quote, look: str, opts: dict, W: int, H: int, fps: float, quiet=None):
        self.q = quote
        self.look = look
        self.opts = opts
        self.W, self.H, self.fps = W, H, fps
        self.u = W / 1080.0
        self.accent = ACCENT if opts["accent"] == "auto" else textmod.colour(opts["accent"])
        self.face = "poppins-semibold" if look == "post" else "dm-serif"
        self.key = None
        if quote.words:
            k = moments.key_word(quote.words, quiet or [], None)
            self.key = k
        self.start = max(0.0, quote.start - ENTER)
        self.rest = quote.start  # still and whole from here
        self.end = quote.end + opts["hold"]
        self.tilt = -2.2 if look == "clipping" else 0.0
        self.warnings: list[str] = []

    # -- measuring
    def _wrap(self, px: int, width: float) -> Optional[list[list[int]]]:
        space = textmod.advance(" ", self.face, px)
        widths = [textmod.advance(t, self.face, px) for t in self.q.text]
        if max(widths) > width:
            return None
        lines, cur, cw = [], [], 0.0
        for i, wd in enumerate(widths):
            add = wd if not cur else cw + space + wd
            if cur and add > width:
                lines.append(cur)
                cur, cw = [i], wd
            else:
                cur.append(i)
                cw = add
        if cur:
            lines.append(cur)
        # balance: the narrowest width that keeps the same number of lines, so no word hangs alone
        lo, hi = max(widths), width
        for _ in range(14):
            mid = (lo + hi) / 2
            trial, cur, cw = [], [], 0.0
            for i, wd in enumerate(widths):
                add = wd if not cur else cw + space + wd
                if cur and add > mid:
                    trial.append(cur)
                    cur, cw = [i], wd
                else:
                    cur.append(i)
                    cw = add
            if cur:
                trial.append(cur)
            if len(trial) <= len(lines):
                hi = mid
                best = trial
            else:
                lo = mid
        return best if len(lines) > 1 else lines

    def _measure(self, variant: str, px: int, max_w: float) -> Optional[Layout]:
        u = self.u
        pad = (40 if self.look == "post" else 44) * u
        inner = max_w - 2 * pad
        lines = self._wrap(px, inner)
        if lines is None:
            return None
        lead = px * (1.36 if self.look == "post" else 1.22)
        space = textmod.advance(" ", self.face, px)
        cap = textmod.cap_height(self.face, px)
        top = pad
        if self.look == "post" and variant == "full" or self.look == "post" and variant == "plain":
            top += 64 * u + 22 * u  # the avatar row
        if self.look == "clipping":
            top += (30 if self.opts["source"].strip() else 16) * u + 18 * u  # the rule, and the source over it
        placed, widest = [], 0.0
        for li, ln in enumerate(lines):
            x = 0.0
            for i in ln:
                wd = textmod.advance(self.q.text[i], self.face, px)
                placed.append(Placed(i, li, x, top + cap + li * lead, wd))
                x += wd + space
            widest = max(widest, x - space)
        body_h = cap + (len(lines) - 1) * lead + px * 0.28
        h = top + body_h + pad
        if self.look == "post" and variant == "full":
            h += 30 * u + 34 * u
        w = min(max_w, widest + 2 * pad)
        if self.look == "post":
            w = max(w, min(max_w, 0.62 * self.W))  # a post reads as a post at about this width
        for p in placed:
            p.x += pad
        return Layout(variant, px, w, h, placed, len(lines))

    def fit(self, max_w: float, max_h: float, margin: tuple[float, float]) -> tuple[Optional[Layout], float]:
        """The largest layout that fits `max_w` x `max_h` with its shadow (`margin`: what the shadow
        and the tilt add across and down): first with everything, then without the marks under the
        words, then without the name; then the smallest drawn smaller, down to 70%."""
        u = self.u
        variants = ("full", "plain", "bare") if self.look == "post" else ("full",)
        n = len(self.q.text)
        # a short quote is set large; a long one smaller, so it stays a card and not a page
        hi = (76 if n <= 5 else 62 if n <= 10 else 50) if self.look == "post" else (84 if n <= 5 else 68 if n <= 10 else 56)
        lows = {"full": 36, "plain": 32, "bare": 28} if self.look == "post" else {"full": 28}
        weight = {"full": 1.4, "plain": 1.2, "bare": 1.0}  # a post reads as a post with its avatar
        cands = sorted(((px_u * weight[v], v, px_u) for v in variants for px_u in range(hi, lows[v] - 1, -2)),
                       key=lambda c: -c[0])
        mx, my = margin
        for _, v, px_u in cands:
            lay = self._measure(v, int(px_u * u), max_w - mx)
            if lay is not None and lay.h + my <= max_h and lay.lines <= 5:
                return lay, 1.0
        lay = self._measure(variants[-1], int(lows[variants[-1]] * u), max_w - mx)
        if lay is None:
            return None, 0.0
        k = min((max_h) / (lay.h + my), (max_w) / (lay.w + mx))
        return lay, k

    # -- building
    def build(self, lay: Layout, scale: float, frames_behind: list[np.ndarray], x: float, y: float) -> None:
        """The pieces, drawn once, at `scale` of the layout's size, the card's top-left at (x, y)."""
        self.lay = lay
        self.s = scale
        self.x, self.y = x, y
        k = scale
        u = self.u * k
        px = max(8, int(round(lay.px * k)))
        self.cw, self.ch = int(math.ceil(lay.w * k)), int(math.ceil(lay.h * k))
        self.placed = [Placed(p.word, p.line, p.x * k, p.base * k, p.w * k) for p in lay.words]
        if self.look == "post":
            self._build_post(px, u, frames_behind)
        else:
            self._build_clipping(px, u)
        self._cache: dict = {}

    def _build_post(self, px: int, u: float, frames_behind) -> None:
        from ..core import glass

        self.pane = glass.Pane(self.cw, self.ch, radius=34 * u, unit=max(0.5, u))
        self.pane.tone_over(frames_behind, self.x, self.y)
        white = (255, 255, 255)
        self.lit = [textmod.draw_word(t, self.face, px, self.accent if i == self.key else white)
                    for i, t in enumerate(self.q.text)]
        self.dim = [textmod.draw_word(t, self.face, px, white) for t in self.q.text]
        # the head: an avatar and the name, or two plain bars where a name would be
        self.fixed: list[tuple[Sprite, float, float, float]] = []
        pad = 40 * u
        if self.lay.variant in ("full", "plain"):
            av = 64 * u
            name = self.opts["name"].strip()
            handle = ""
            if "@" in name:
                at = name.index("@")
                name, handle = name[:at].strip(), name[at:].strip()
            if name or handle:
                initial = (name or handle.lstrip("@"))[:1].upper()
                self.fixed.append((_circle(av, (64, 70, 92)), pad, pad, 1.0))
                ini = textmod.draw_word(initial, "poppins-bold", int(av * 0.46), white)
                self.fixed.append((ini.sprite, pad + (av - ini.width) / 2 - ini.ox + 0.0,
                                   pad + av / 2 + ini.ascent / 2 - ini.oy, 1.0))
                tx = pad + av + 20 * u
                if name:
                    nw = textmod.draw_word(name, "poppins-bold", int(30 * u), white)
                    ny = pad + av / 2 + (nw.ascent / 2 if not handle else -4 * u)
                    self.fixed.append((nw.sprite, tx - nw.ox, ny - nw.oy, 1.0))
                if handle:
                    hw = textmod.draw_word(handle, "poppins-semibold", int(25 * u), (200, 204, 216))
                    hy = pad + av / 2 + (hw.ascent + 10 * u if name else hw.ascent / 2)
                    self.fixed.append((hw.sprite, tx - hw.ox, hy - hw.oy, 1.0))
            else:
                self.fixed.append((_silhouette(av, (70, 76, 98), (150, 156, 176)), pad, pad, 1.0))
                tx = pad + av + 20 * u
                self.fixed.append((_bar(min(190 * u, self.cw * 0.34), 18 * u, white, 0.55), tx, pad + av * 0.22, 1.0))
                self.fixed.append((_bar(min(120 * u, self.cw * 0.22), 14 * u, white, 0.30), tx, pad + av * 0.62, 1.0))
        self.marks = []
        if self.lay.variant == "full":
            s = 30 * u
            y = self.ch - pad - s
            muted = (205, 208, 220)
            for n, kind in enumerate(("reply", "loop", "heart")):
                self.marks.append((kind, _mark(kind, s, muted, 0.8), pad + n * 92 * u, y))
            self.heart_on = _mark("heart-on", s, (255, 92, 120))
        self.shadow_box = None

    def _build_clipping(self, px: int, u: float) -> None:
        import cv2
        from PIL import Image, ImageDraw

        cw, ch = self.cw, self.ch
        rng = np.random.default_rng(29)
        # the paper: warm newsprint, a faint uneven tone and a fine grain, the same on every frame
        tone = cv2.GaussianBlur(rng.standard_normal((ch, cw)).astype(np.float32), (0, 0), max(2.0, 18 * u)) * 60
        grain = rng.standard_normal((ch, cw)).astype(np.float32) * 3.0
        paper = np.empty((ch, cw, 3), np.float32)
        base = np.array((241, 235, 220), np.float32)
        for c in range(3):
            paper[..., c] = base[c] + tone + grain - (2.0 if c == 2 else 0.0)
        # torn along the top and the bottom, cut straight at the sides
        k = 4
        mask = Image.new("L", (cw * k, ch * k), 0)
        step = max(3.0, 7 * u) * k
        tear = 7 * u * k
        top, bot = [], []
        x = 0.0
        while x <= cw * k + step:
            top.append((min(x, cw * k), rng.uniform(0, tear)))
            bot.append((min(x, cw * k), ch * k - rng.uniform(0, tear)))
            x += step * rng.uniform(0.6, 1.4)
        ImageDraw.Draw(mask).polygon(top + bot[::-1], fill=255)
        alpha = np.asarray(mask.resize((cw, ch), Image.LANCZOS), np.float32) / 255.0
        self.paper = Sprite(rgb=np.clip(paper, 0, 255) * alpha[..., None], a=alpha)
        # the head of the cutting: the source, small and spaced, over a double rule; or the rule alone
        ink = (34, 31, 28)
        pad = 44 * u
        self.fixed = []
        src = self.opts["source"].strip()
        y = pad
        if src:
            sw = textmod.draw_word(" ".join(src.upper()), "poppins-semibold", int(19 * u), (88, 80, 70))
            if sw.width > cw - 2 * pad:
                sw = textmod.draw_word(src.upper(), "poppins-semibold", int(19 * u), (88, 80, 70))
            self.fixed.append((sw.sprite, (cw - sw.width) / 2 - sw.ox, y + sw.ascent - sw.oy, 1.0))
            y += 30 * u
        self.fixed.append((textmod.rounded_box(cw - 2 * pad, max(1.0, 3.2 * u), 0, fill=ink), pad, y, 1.0))
        self.fixed.append((textmod.rounded_box(cw - 2 * pad, max(1.0, 1.2 * u), 0, fill=ink), pad, y + 7 * u, 1.0))
        self.ink_words = [textmod.draw_word(t, self.face, px, ink) for t in self.q.text]
        self.mark_h = px * 0.86
        self.px = px
        # the shadow under the cutting: soft and a little below, turned with it
        sh_pad = int(math.ceil(24 * u))
        a = np.pad(alpha, sh_pad)
        a = cv2.GaussianBlur(a, (0, 0), max(1.0, 9 * u)) * 0.38
        self.shadow = Sprite(rgb=np.zeros(a.shape + (3,), np.float32), a=a.astype(np.float32))
        self.shadow_pad = sh_pad
        self.shadow_dy = 9 * u
        self.shadow_rot, sdx, sdy = rotated(self.shadow, self.tilt)
        self.shadow_off = (sdx, sdy)

    # -- per frame
    def _progress(self, t: float) -> list[float]:
        """How lit each word is at `t`: 0 before its first frame, then (on the clipping) the
        highlighter's reach across it, which is part way on that very frame."""
        half = 0.5 / self.fps
        out = []
        for i in range(len(self.q.text)):
            at = self.q.lit_at(i)
            if t + half < at:
                out.append(0.0)
            elif self.look == "post" or not self.q.words:
                out.append(1.0)
            else:
                w = self.q.words[min(i, len(self.q.words) - 1)]
                span = max(1.5 / self.fps, min(0.22, w.end - w.start))
                out.append(min(1.0, 0.45 + 0.55 * (t + half - at) / span))
        return out

    def content(self, t: float) -> tuple[Sprite, float, float]:
        """The card's face at `t` (the words lit so far), turned when the look is tilted, and how
        far turning moved its top-left."""
        prog = self._progress(t)
        heart = self.look == "post" and self.lay.variant == "full" and self.q.words and t >= self.q.end + 0.25
        key = (tuple(int(p * 16) for p in prog), heart)
        if key in self._cache:
            return self._cache[key]
        canvas = (np.zeros((self.ch, self.cw, 3), np.float32), np.zeros((self.ch, self.cw), np.float32))
        if self.look == "clipping":
            _layer(canvas, self.paper, 0, 0)
            for p in self.placed:
                g = prog[p.word]
                if g <= 0:
                    continue
                # the highlighter, from a little before the word to a little past it
                x0 = p.x - 0.12 * self.px
                full = p.w + 0.24 * self.px
                bar = textmod.rounded_box(max(2.0, full * g), self.mark_h, self.mark_h * 0.18, fill=self.accent,
                                          opacity=0.62)
                _layer(canvas, bar, x0, p.base - self.mark_h * 0.82)
            for sp, x, y, a in self.fixed:
                _layer(canvas, sp, x, y, a)
            for p in self.placed:
                wd = self.ink_words[p.word]
                _layer(canvas, wd.sprite, p.x - wd.ox, p.base - wd.oy)
        else:
            for sp, x, y, a in self.fixed:
                _layer(canvas, sp, x, y, a)
            for p in self.placed:
                g = prog[p.word]
                wd = self.lit[p.word] if g > 0 else self.dim[p.word]
                _layer(canvas, wd.sprite, p.x - wd.ox, p.base - wd.oy, 1.0 if g > 0 else 0.34)
            for kind, sp, x, y in self.marks:
                _layer(canvas, self.heart_on if (kind == "heart" and heart) else sp, x, y)
        got = rotated(Sprite(rgb=canvas[0], a=canvas[1]), self.tilt)
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[key] = got
        return got

    def motion(self, t: float) -> tuple[float, float, float]:
        """(opacity, scale, drop in pixels) at `t`: coming in before the first word, going after."""
        if t < self.rest:
            q = _ease_out((t - self.start) / max(1e-3, self.rest - self.start))
        else:
            q = 1.0
        out = min(1.0, max(0.0, (self.end - t) / LEAVE))
        a = min(q * 1.6, 1.0) * out
        if self.look == "post":
            s = (0.9 + 0.1 * q) * (0.97 + 0.03 * out)
            return a, s, 0.0
        drop = (1 - q) * 46 * self.u * self.s * self.side_sign + (1 - out) * 10 * self.u * self.s
        return a, 1.0, drop

    def draw(self, frame: np.ndarray, t: float) -> None:
        if not (self.start <= t < self.end):
            return
        a, s, drop = self.motion(t)
        if a <= 0.003:
            return
        sp, dx, dy = self.content(t)
        if self.look == "post":
            self.pane.draw(frame, self.x, self.y, opacity=a, frost=min(1.0, a * 1.3), scale=s)
            if abs(s - 1.0) > 0.005:
                sc = sp.scaled(s)
                over(frame, sc, self.x + self.cw * (1 - s) / 2, self.y + self.ch * (1 - s) / 2, a)
            else:
                over(frame, sp, self.x, self.y, a)
            return
        sdx, sdy = self.shadow_off
        over(frame, self.shadow_rot, self.x - self.shadow_pad - sdx, self.y - self.shadow_pad - sdy + self.shadow_dy + drop,
             a)
        over(frame, sp, self.x - dx, self.y - dy + drop, a)

    # -- what it covers
    def drawn_box(self, at_rest: bool = True) -> Rect:
        """The box of everything the card draws, its shadow included, at rest; with `at_rest`
        False, over its whole entrance too."""
        if self.look == "post":
            p = self.pane
            r = Rect(self.x - p.shadow_pad, self.y - p.shadow_pad + p.shadow_dy,
                     self.x - p.shadow_pad + p.shadow.w, self.y - p.shadow_pad + p.shadow_dy + p.shadow.h)
            b = ink_box(p.shadow)
            if b is not None:
                r = b.shift(self.x - p.shadow_pad, self.y - p.shadow_pad + p.shadow_dy)
            return r.union(Rect(self.x, self.y, self.x + self.cw, self.y + self.ch))
        sdx, sdy = self.shadow_off
        sb = ink_box(self.shadow_rot) or Rect(0, 0, self.shadow_rot.w, self.shadow_rot.h)
        r = sb.shift(self.x - self.shadow_pad - sdx, self.y - self.shadow_pad - sdy + self.shadow_dy)
        paper_rot, dx, dy = rotated(self.paper, self.tilt)
        pb = (ink_box(paper_rot) or Rect(0, 0, paper_rot.w, paper_rot.h)).shift(self.x - dx, self.y - dy)
        r = r.union(pb)
        if not at_rest:
            d = 46 * self.u * self.s * self.side_sign
            r = r.union(r.shift(0, d))
        return r


# ---- placing it -------------------------------------------------------------------------------------

def _margins(look: str, u: float) -> tuple[float, float]:
    """What the shadow (and the clipping's tilt) adds to a card's size, across and down."""
    if look == "post":
        return 2 * 14 * u, 2 * 14 * u + 7 * u
    return 2 * 22 * u, 2 * 20 * u + 9 * u + 22 * u  # the tilt lifts one corner and drops the other


def head_room(card: Card, heads: list[Optional[Rect]], zone: SafeZone) -> dict:
    """The bands above and below the head over every frame the card is up."""
    z = zone.rect
    fps = card.fps
    a = int(max(0, math.floor(card.start * fps)))
    b = int(min(len(heads) - 1, math.ceil(card.end * fps)))
    head = Rect.around(h for h in heads[a:b + 1] if h is not None)
    gap = 0.018 * card.H
    if head is None:
        return {"bottom": (z.y1 - 0.34 * card.H, z.y1), "top": (z.y0, z.y0 + 0.34 * card.H)}
    return {"top": (z.y0, head.y0 - gap), "bottom": (head.y1 + gap, z.y1)}


def place(card: Card, heads: list[Optional[Rect]], zone: SafeZone, position: str) -> tuple[Optional[Layout], float, str, float, float]:
    """Above the head or below the chin, wherever the card comes out largest (or where `position`
    asks, when the card fits there), clear of the head over every frame it is up, inside the zone."""
    z = zone.rect
    rooms = head_room(card, heads, zone)
    margin = _margins(card.look, card.u)
    tried = {}
    for side, (r0, r1) in rooms.items():
        if r1 - r0 < 40 * card.u:
            continue
        lay, k = card.fit(z.w, r1 - r0, margin)
        if lay is not None:
            tried[side] = (lay, k, r0, r1)
    if not tried:
        return None, 0.0, "nowhere", 0.0, 0.0
    def size(s):
        lay, k, _, _ = tried[s]
        return k * lay.px * (1.15 if lay.variant == "full" else 1.0 if lay.variant == "plain" else 0.9)
    order = sorted(tried, key=lambda s: -size(s))
    side = order[0]
    if position in tried and tried[position][1] >= 0.999:
        side = position
    elif position in ("top", "bottom") and position not in tried:
        card.warnings.append(f"there is no room {'above' if position == 'top' else 'below'} your head; it went "
                             f"{'below' if position == 'top' else 'above'}")
    lay, k, r0, r1 = tried[side]
    mx, my = margin
    k = min(1.0, k)
    w, h = lay.w * k, lay.h * k
    x = z.cx - w / 2
    # inside the band: the card's drawn box (shadow and tilt) sits in it, nearer the head than the edge
    slack = max(0.0, (r1 - r0) - (h + my * k))
    y = r0 + my * k * (0.32 if card.look == "post" else 0.5) + slack * (0.7 if side == "top" else 0.3)
    return lay, k, side, x, y


# ---- the effect -------------------------------------------------------------------------------------

def _when(opts: dict, seconds: float) -> Optional[float]:
    w = opts["when"].strip().lower()
    if w in ("", "auto"):
        return None
    try:
        t = float(w.rstrip("s"))
    except ValueError:
        raise O.OptionError(f"when: give auto or a time in seconds such as 12.5 (got {opts['when']!r})") from None
    if not 0 <= t < seconds:
        raise O.OptionError(f"when: {t:g} s is outside the video (0 to {seconds:.2f} s)")
    return t


def find_quotes(clip, opts: dict, seconds: float, notes: list[str]) -> list[Quote]:
    words = clip.words
    quiet = clip.quiet
    back = moments.taken_back(words, quiet)
    when = _when(opts, seconds)
    if opts["text"].strip():
        texts = [t for t in layers.split(opts["text"]) if t.strip()]
        if len(texts) > MOST_CARDS:
            raise O.OptionError(f"text: {len(texts)} quotes is more than one video carries well; keep it to {MOST_CARDS}")
        return [given_quote(words, t.strip().strip("\"“”"), when if n == 0 else None) for n, t in enumerate(texts)]
    found = said_quotes(words, back)
    if found:
        if len(found) > MOST_CARDS:
            notes.append(f"{len(found)} quotes are said; the first {MOST_CARDS} are shown")
        return found[:MOST_CARDS]
    q = pull_quote(words, quiet, back)
    if q is None:
        raise O.OptionError("no quote is said and no line stands out to quote; give the words with text=\"…\"")
    return [q]


def _plan_quotes(plan: dict) -> list[Quote]:
    out = []
    for c in plan.get("quotes") or []:
        try:
            ws = [Word(str(w["w"]), float(w["s"]), float(w["e"])) for w in c.get("words") or []]
            text = str(c["text"]).split()
        except (KeyError, TypeError, ValueError):
            raise ValueError("a card in the plan needs its text and its words' times; run once without --plan") from None
        if ws and len(ws) != len(text):
            raise ValueError("a card's text and its words must be as many")
        out.append(Quote(ws, text, "from the plan", bool(c.get("press")), at=c.get("at")))
    if not out:
        raise ValueError("the plan has no quotes; run once without --plan to get one to edit")
    return out


class Painter:
    def __init__(self, cards: list[Card], captions: Optional[layers.Layer]):
        self.cards = cards
        self.captions = captions

    def shown(self, t: float) -> float:
        """How much of a card is up at `t` (0 to 1): the captions step aside by as much."""
        best = 0.0
        for c in self.cards:
            if c.start <= t < c.end:
                best = max(best, c.motion(t)[0])
            elif c.start - 0.12 <= t < c.start:
                best = max(best, 1 - (c.start - t) / 0.12)
        return min(1.0, best)

    def draw(self, frame: np.ndarray, j: int, t: float) -> None:
        if self.captions is not None:
            self.captions.draw(frame, j, t, seen=1.0 - self.shown(t))
        for c in self.cards:
            c.draw(frame, t)


class Sounds:
    def __init__(self, base: np.ndarray, cards: list[Card]):
        voice = soundmod.integrated(soundmod.blocks(base)) if base.size else soundmod.SILENCE_DB
        self.ref = voice if voice > -70 else -20.0
        self.cards = cards

    def mix(self, base: np.ndarray, rate: int) -> np.ndarray:
        from ..core import loudness

        stem = np.zeros_like(base)
        for c in self.cards:
            if c.look == "post":
                x, anchor = soundmod.library("pop")
                at = c.start + 0.06
            else:
                x, anchor = synth.paper()
                at = c.rest
            soundmod.lay(stem, synth.at_level(x, self.ref - UNDER_VOICE), at, rate, anchor=anchor)
        out = base + stem
        ceiling = max(-1.0, loudness.true_peak_db(base) + 0.05)
        if loudness.true_peak_db(out) > ceiling:
            out, _ = loudness.limit(out, ceiling)
        return out.astype(np.float32)


def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from ..core import framing as framing_mod
    from ..core import video
    from ..core.timeline import Timeline
    from . import Job, PlanError

    info = clip.info
    fr = framing_mod.plan(info.width, info.height, info.rate, info.frames, clip.face, shape="keep")
    tl = Timeline.whole(info.frames / info.rate)
    W, H, fps = fr.out_w, fr.out_h, info.rate
    n = tl.output_frames(fps)
    seconds = n / fps
    notes: list[str] = []
    try:
        quotes = _plan_quotes(plan) if plan is not None else find_quotes(clip, opts, seconds, notes)
    except ValueError as e:
        if plan is not None and not isinstance(e, O.OptionError):
            raise PlanError(str(e)) from None
        raise
    head_at = head_in_output(clip, tl, fr)
    heads = [head_at(j) for j in range(n)]
    zone = SafeZone(W, H)
    cards: list[Card] = []
    for q in quotes:
        look = opts["look"]
        if look == "auto":
            look = "clipping" if (q.press or opts["source"].strip()) else "post"
        c = Card(q, look, opts, W, H, fps, clip.quiet)
        c.end = min(c.end, seconds)
        if cards and c.start < cards[-1].end:
            if q.start - 0.1 > cards[-1].rest + 1.0:
                cards[-1].end = c.start  # the earlier card makes way
            else:
                notes.append(f"“{' '.join(q.text)[:40]}” comes too soon after the card before; left out")
                continue
        lay, k, side, x, y = place(c, heads, zone, opts["position"])
        if lay is None:
            raise O.OptionError("there is no room for a card above or below your face in this shot")
        if k < 0.7:
            notes.append(f"the room {'above' if side == 'top' else 'below'} your head is tight: the card is drawn at "
                         f"{k:.0%}; a wider shot gives it more room")
        c.side = side
        c.side_sign = -1.0 if side == "top" else 1.0
        # what is behind the card at a few moments, so the glass is smoked enough for white words
        behind = []
        if look == "post":
            for t in np.linspace(c.start + 0.05, max(c.start + 0.05, c.end - 0.05), 4):
                kf = tl.source_frame(int(t * fps), fps, info.frames)
                behind.append(fr.apply(video.read_frame(info, kf), kf))
        c.build(lay, max(0.55, k), behind, x, y)
        # the guess at the shadow's reach is checked against what was drawn: smaller until it fits
        band = (zone.rect.y0, head_room(c, heads, zone)[side][1]) if side == "top" else (head_room(c, heads, zone)[side][0], zone.rect.y1)
        for _ in range(8):
            r = c.drawn_box(True)
            if zone.contains(r) and band[0] - 0.5 <= r.y0 and r.y1 <= band[1] + 0.5:
                break
            k2 = c.s * 0.95
            w2, h2 = lay.w * k2, lay.h * k2
            x2 = zone.rect.cx - w2 / 2
            y2 = y + (lay.h * c.s - h2) * (1.0 if side == "top" else 0.0)
            if r.y0 < band[0]:
                y2 += band[0] - r.y0
            c.build(lay, k2, behind, x2, y2)
            y = y2
        cards.append(c)
    caption_layer = None
    if opts["captions"] != "off":
        caption_layer = layers.prepare(clip, layers.Spec("captions", {"look": opts["captions"]}), None, None, fr,
                                       fix=opts["fix"])
    painter = Painter(cards, caption_layer)
    sounds = Sounds(clip.sound, cards) if opts["sound"] else None
    # the checks, frame by frame: the card's drawn box (shadow included, and the entrance) against the head
    over_face = outside = 0
    for c in cards:
        rest = c.drawn_box(True)
        moving = c.drawn_box(False)
        for j in range(int(math.floor(c.start * fps)), min(n, int(math.ceil(c.end * fps)))):
            t = j / fps
            if not (c.start <= t < c.end):
                continue
            h = heads[j]
            r = moving if t < c.rest else rest
            if h is not None and r.intersects(h):
                over_face += 1
            if not zone.contains(rest):
                outside += 1
    checks = {
        "cards": len(cards),
        "frames_over_face": over_face,
        "frames_outside_safe_zone": outside,
    }
    for i, c in enumerate(cards, 1):
        lit = (f"each word lit on its first frame, {c.q.words[0].start:.2f}-{c.q.words[-1].start:.2f} s"
               if c.q.words else "lit at once (not read out)")
        checks[f"card {i}"] = (f"{c.look} “{' '.join(c.q.text)}”, {c.q.why}; up {c.start:.2f}-{c.end:.2f} s, "
                               f"{'above' if c.side == 'top' else 'below'} your head at {c.s:.0%}"
                               f"{'' if c.lay.variant == 'full' else ' (' + c.lay.variant + ')'}; {lit}")
    if caption_layer is not None:
        checks["captions"] = f"{opts['captions']}, stepping aside while a card is up"
        checks["frames_over_face"] += int(caption_layer.job.checks.get("frames_over_face", 0) or 0)
    checks["sound"] = ("a pop as a post lands, paper laid down for a clipping, under your voice" if opts["sound"]
                       else "your sound, untouched")
    checks["warnings"] = notes + [w for c in cards for w in c.warnings]
    out_plan = {"effect": NAME, "clip": clip.path, "options": opts,
                "output": {"width": W, "height": H, "fps": round(fps, 3), "seconds": round(tl.duration, 3),
                           "framing": fr.note},
                "quotes": [{"look": c.look, "text": " ".join(c.q.text), "press": c.q.press, "at": c.q.at,
                           "words": [{"w": w.text, "s": round(w.start, 3), "e": round(w.end, 3)} for w in c.q.words],
                           "start": round(c.start, 3), "end": round(c.end, 3), "side": c.side,
                           "rect": c.drawn_box(True).as_list(), "scale": round(c.s, 3)} for c in cards]}
    span = [x for c in cards for x in (c.start + 0.1, c.rest + 0.03, (c.rest + c.q.end) / 2, c.q.end + 0.3)]
    out_plan["sheet_times"] = [round(min(seconds - 0.05, t), 3) for t in (span + [c.end - 0.1 for c in cards])[:6]]

    def stills() -> list[float]:
        out = []
        for c in cards[:2]:
            mid = (c.q.lit_at(len(c.q.text) // 2) + 0.05) if c.q.words else c.rest + 0.3
            out += [c.start + 0.12, c.rest + 0.02, mid, min(c.end - 0.1, c.q.end + 0.4)]
        return out[:5]

    return Job(NAME, clip, tl, fr, painter.draw, out_plan, checks=checks, still_times=stills,
               sound=sounds.mix if sounds else None)


# The runs `tests/matrix.py quote-card` makes: the line it finds by itself on each clip, the
# clipping on the plain wall and the long take, a quote typed by the person, and captions around it.
MATRIX = [(c, c, {}) for c in ("A", "B", "C", "D")] + [
    ("D-clipping", "D", {"look": "clipping"}),
    ("33-clipping", "33", {"look": "clipping", "source": "Demo day notes"}),
    ("B-text", "B", {"text": "an editor that edits for you", "name": "Your name @you"}),
    ("33-text", "33", {"text": "Distribution is a real big issue"}),
    ("D-captions", "D", {"captions": "stacked"}),
]
