"""stickers: stickers that slap on with the word: a stamp, a price tag, a starburst, an emoji-style badge.

What it does, in order:

1. **Which words.** `at=auto` takes the strongest words of the take (the shared picker,
   `moments`: never a word taken back, a first attempt, an aside or a filler), an amount of money or
   a percentage said, and words of feeling ("love", "wrong", "wow"), at least two seconds apart, one
   for every five seconds or so, three at most. `at="hooks|marketing campaign"` puts one on each of
   those words where they are said; `at=2.1,5.4` on the word starting nearest each time.
2. **Which sticker.** `kind=auto` chooses by the word: an amount gets a price tag, a feeling a badge,
   "new", "free", "first" or a number a starburst, a verdict ("wrong", "done", "approved") a stamp,
   anything else a stamp or a starburst in turn. `kind=` and `text=` take one value, or one per
   sticker with `|`. Every sticker is drawn here in code: no file is downloaded and no app's emoji
   art is used; a badge is a face, a heart, a flame, a tick, a cross or a star of our own.
3. **The slap.** Each lands on the first frame of its word: it drops in from a little larger over
   three frames, squashes as it hits, settles, sits at a slight angle (alternating left and right),
   and pops away. A slap made in code is laid under the voice on the frame it hits (`sound=off`
   keeps the clip's sound exact).
4. **Where.** Beside the head, never on the face: every spot in the safe zone is weighed, and the
   one kept is clear of the head over every frame the sticker is up (its drop from larger
   included), of the captions, and of the other stickers on screen, nearest the head, on the other
   side from the sticker before. With no room it is drawn smaller, down to 55%, or left out and
   the run says so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core import amounts, items, moments, synth
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

NAME = "stickers"
KINDS = ("stamp", "price", "starburst", "badge")
BADGES = ("heart", "fire", "wow", "laugh", "check", "cross", "star")

OPTIONS = {
    "at": O.Text("auto: on the strongest words, amounts and feelings you say · words you say, several with | "
                 "(\"hooks|marketing campaign\") · or times in seconds (2.1,5.4), each moved onto the word said there"),
    "kind": O.Text("auto (chosen by the word) or stamp, price, starburst, badge; one for all, or one per sticker "
                   "with |"),
    "text": O.Text("what each says, with | between stickers (\"SOLD|$49\"); empty: the word said. For a badge: "
                   "heart, fire, wow, laugh, check, cross or star"),
    "color": O.Colour("the sticker's colour: auto (each kind its own), a name or a hex colour"),
    "count": O.Whole(0, 6, "how many stickers at=auto places; 0: one for every five seconds or so, three at most"),
    "hold": O.Number(0.6, 4.0, "seconds each sticker stays"),
    "captions": O.Choice(("off",) + cap.LOOKS, "off · or the pack's captions in a look; the stickers keep clear of them"),
    "sound": O.Switch("on lays a slap under your voice as each lands; off keeps your sound exact"),
    "fix": O.Text("misheard words as wrong=right, separated by ;"),
}

COLOURS = {"stamp": (214, 38, 46), "price": (255, 198, 38), "starburst": (255, 86, 54), "badge": (255, 200, 40)}
INK = (24, 22, 26)
WHITE = (255, 255, 255)
DROP = 3  # frames the slap drops over, ending on the word's first frame
FROM = 1.32  # how much larger it starts
LEAVE = 0.14
SPACING = 2.0  # seconds between stickers chosen by themselves
UNDER_VOICE = 8.0

FEELINGS = {
    "heart": {"love", "loved", "loves", "lovely", "heart", "favourite", "favorite", "adore", "beautiful"},
    "fire": {"fire", "hot", "insane", "amazing", "huge", "massive", "crazy", "incredible", "viral", "lit", "best"},
    "wow": {"wow", "whoa", "shocked", "surprised", "unbelievable", "believe", "shock", "omg"},
    "laugh": {"funny", "laugh", "laughing", "joke", "hilarious", "lol", "haha"},
    "check": {"yes", "correct", "works", "worked", "done", "easy", "approved", "right", "perfect"},
    "cross": {"no", "wrong", "fail", "failed", "never", "mistake", "bad", "worst", "broken"},
    "star": {"star", "stars", "pro", "top", "winner", "win", "won"},
}
VERDICTS = {"wrong", "fail", "failed", "done", "approved", "sold", "rejected", "verified", "official", "fake", "real",
            "true", "false", "banned", "issue", "problem", "fixed", "deal", "proof"}
BURSTS = {"new", "free", "first", "now", "launch", "launched", "sale", "bonus", "today", "only", "live", "save"}


@dataclass
class Spot:
    t: float  # the word's start: the frame it hits
    word: str  # as said
    kind: str
    text: str  # what it says (a badge's name for a badge)
    why: str


# ---- drawing, once per sticker ---------------------------------------------------------------------

def _img(w: float, h: float, k: int = 4):
    from PIL import Image

    return Image.new("RGBA", (max(1, int(round(w * k))), max(1, int(round(h * k)))), (0, 0, 0, 0))


def _sprite(img, w: float, h: float) -> Sprite:
    from PIL import Image

    return Sprite.from_rgba(np.asarray(img.resize((max(1, int(round(w))), max(1, int(round(h)))), Image.LANCZOS)))


def _fit_text(text: str, face: str, width: float, most: int) -> int:
    return max(8, min(most, textmod.fit_px(text, face, width)))


def _text_on(img, text: str, face: str, px: int, cx: float, cy: float, fill, k: int = 4, outline: float = 0.0,
             outline_fill=INK) -> None:
    from PIL import ImageDraw

    f = textmod.font(face, px * k)
    d = ImageDraw.Draw(img)
    stroke = int(round(outline * k))
    l, t, r, b = f.getbbox(text, anchor="ls", stroke_width=stroke)
    d.text((cx * k - (l + r) / 2, cy * k - (t + b) / 2), text, font=f, anchor="ls", fill=tuple(fill) + (255,),
           stroke_width=stroke, stroke_fill=tuple(outline_fill) + (255,))


def _die_cut(rgba: np.ndarray, edge: float, shadow: float) -> np.ndarray:
    """A sticker's white edge round its shape (`edge` pixels) and a soft shadow under it."""
    import cv2

    a = rgba[..., 3].astype(np.float32) / 255.0
    r = max(1, int(round(edge)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    grown = cv2.dilate(a, kernel)
    grown = cv2.GaussianBlur(grown, (0, 0), 0.8)
    sh = cv2.GaussianBlur(grown, (0, 0), max(1.0, shadow)) * 0.42
    dy = int(round(shadow * 0.6))
    sh = np.roll(sh, dy, axis=0)
    out = np.zeros_like(rgba, np.float32)
    # shadow, then the white edge, then the sticker
    base_a = sh
    rgb = np.zeros(rgba.shape[:2] + (3,), np.float32)
    white_a = grown
    rgb = rgb * (1 - white_a[..., None]) + 255.0 * white_a[..., None]
    base_a = base_a * (1 - white_a) + white_a
    top = rgba[..., :3].astype(np.float32)
    rgb = rgb * (1 - a[..., None]) + top * a[..., None]
    base_a = base_a * (1 - a) + a
    out[..., :3] = rgb
    out[..., 3] = base_a * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _pad(img, p: int):
    from PIL import Image

    out = Image.new("RGBA", (img.width + 2 * p, img.height + 2 * p), (0, 0, 0, 0))
    out.paste(img, (p, p))
    return out


def draw_stamp(text: str, colour, s: float) -> Sprite:
    """A rubber stamp: the word in tall capitals inside a double frame, inked unevenly."""
    from PIL import Image, ImageDraw

    face = "anton"
    px = _fit_text(text, face, s * 0.86, int(s * 0.34))
    tw = textmod.advance(text, face, px)
    w, h = tw + px * 0.95, px * 1.62
    k = 4
    img = _img(w, h, k)
    d = ImageDraw.Draw(img)
    c = tuple(colour) + (255,)
    lw = max(3, int(px * 0.1 * k))
    # a pale label under the ink, so the stamp reads over leaves or a busy room as well as a wall
    d.rounded_rectangle((lw / 2, lw / 2, img.width - lw / 2, img.height - lw / 2), radius=px * 0.16 * k,
                        fill=(250, 246, 238, 150))
    d.rounded_rectangle((lw / 2, lw / 2, img.width - lw / 2, img.height - lw / 2), radius=px * 0.16 * k, outline=c, width=lw)
    inset = lw * 2.1
    d.rounded_rectangle((inset, inset, img.width - inset, img.height - inset), radius=px * 0.1 * k, outline=c,
                        width=max(2, lw // 3))
    _text_on(img, text, face, px, w / 2, h / 2, colour, k)
    arr = np.asarray(img).copy()
    # uneven ink: specks where the rubber did not touch, the same every time
    rng = np.random.default_rng(len(text) * 31 + 7)
    import cv2

    noise = cv2.GaussianBlur(rng.random(arr.shape[:2]).astype(np.float32), (0, 0), 2.2 * k / 4 + 0.6)
    noise = (noise - noise.mean()) / (noise.std() + 1e-6)
    keep = np.clip(1.25 + 0.55 * noise, 0.0, 1.0) * 0.93
    arr[..., 3] = (arr[..., 3].astype(np.float32) * keep).astype(np.uint8)
    pad = int(6 * k)
    img = _pad(Image.fromarray(arr, "RGBA"), pad)
    return _sprite(img, w + 2 * pad / k, h + 2 * pad / k)


def draw_price(text: str, colour, s: float) -> Sprite:
    """A swing tag: a card with a cut point and a hole at its left, the amount in heavy figures."""
    from PIL import Image, ImageDraw

    face = "poppins-black"
    px = _fit_text(text, face, s * 0.7, int(s * 0.3))
    tw = textmod.advance(text, face, px)
    h = px * 1.6
    point = h * 0.42
    w = tw + px * 0.9 + point
    k = 4
    img = _img(w, h, k)
    d = ImageDraw.Draw(img)
    W, H = img.width, img.height
    P = point * k
    r = H * 0.14
    poly = [(P, 0), (W - r, 0), (W, r), (W, H - r), (W - r, H), (P, H), (0, H / 2)]
    d.polygon(poly, fill=tuple(colour) + (255,))
    hole = H * 0.11
    d.ellipse((P * 0.55 - hole, H / 2 - hole, P * 0.55 + hole, H / 2 + hole), fill=(0, 0, 0, 0))
    _text_on(img, text, face, px, point + (w - point) / 2, h / 2, INK, k)
    pad = int(14 * k)
    img = _pad(img, pad)
    arr = _die_cut(np.asarray(img), 5 * k, 7 * k)
    return _sprite(Image.fromarray(arr, "RGBA"), w + 2 * pad / k, h + 2 * pad / k)


def draw_starburst(text: str, colour, s: float) -> Sprite:
    """A burst of sixteen points with the word across it, white with a dark edge."""
    from PIL import Image, ImageDraw

    face = "anton"
    k = 4
    D = s
    img = _img(D, D, k)
    d = ImageDraw.Draw(img)
    C = D * k / 2
    pts = []
    for i in range(32):
        r = C * (0.98 if i % 2 == 0 else 0.74)
        a = math.pi * i / 16 - math.pi / 2
        pts.append((C + r * math.cos(a), C + r * math.sin(a)))
    d.polygon(pts, fill=tuple(colour) + (255,))
    px = _fit_text(text, face, D * 0.7, int(D * 0.36))
    _text_on(img, text, face, px, D / 2, D / 2, WHITE, k, outline=px * 0.07)
    pad = int(14 * k)
    img = _pad(img, pad)
    arr = _die_cut(np.asarray(img), 5 * k, 7 * k)
    return _sprite(Image.fromarray(arr, "RGBA"), D + 2 * pad / k, D + 2 * pad / k)


def _heart(d, cx, cy, size, fill) -> None:
    pts = []
    for i in range(96):
        t = 2 * math.pi * i / 96
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((cx + x * size / 34, cy - y * size / 34 - size * 0.04))
    d.polygon(pts, fill=fill)


def draw_badge(which: str, colour, s: float) -> Sprite:
    """A round badge in the spirit of an emoji, drawn here: a face (wow, laugh), a heart, a flame, a
    tick, a cross or a star, on a disc with a soft light at its top."""
    from PIL import Image, ImageDraw

    k = 4
    D = s * 0.82
    img = _img(D, D, k)
    d = ImageDraw.Draw(img)
    S = D * k
    ink = (60, 36, 20, 255)
    disc = {"heart": (255, 236, 240), "fire": (255, 244, 226), "check": (226, 250, 232), "cross": (255, 232, 230),
            "star": (236, 238, 255)}.get(which)
    face_col = tuple(colour)
    if which in ("wow", "laugh"):
        d.ellipse((0, 0, S - 1, S - 1), fill=face_col + (255,))
    else:
        d.ellipse((0, 0, S - 1, S - 1), fill=disc + (255,))
    if which == "wow":
        for ex in (0.34, 0.66):
            d.ellipse((S * (ex - 0.07), S * 0.28, S * (ex + 0.07), S * 0.46), fill=ink)
        d.ellipse((S * 0.38, S * 0.56, S * 0.62, S * 0.84), fill=ink)
        for ex in (0.34, 0.66):  # brows raised
            d.arc((S * (ex - 0.11), S * 0.13, S * (ex + 0.11), S * 0.3), 200, 340, fill=ink, width=int(S * 0.035))
    elif which == "laugh":
        for ex in (0.33, 0.67):  # eyes shut tight
            d.line([(S * (ex - 0.09), S * 0.33), (S * (ex + 0.07), S * 0.38), (S * (ex - 0.09), S * 0.43)], fill=ink,
                   width=int(S * 0.045), joint="curve")
        d.chord((S * 0.2, S * 0.38, S * 0.8, S * 0.86), 0, 180, fill=ink)
        d.chord((S * 0.28, S * 0.62, S * 0.72, S * 0.84), 0, 180, fill=(232, 92, 96, 255))
    elif which == "heart":
        _heart(d, S / 2, S / 2, S * 0.78, (236, 44, 76, 255))
    elif which == "fire":
        def flame(cx, base, wdt, hgt, fill):
            """A teardrop: round at the bottom, drawn up to a point at the top."""
            pts = []
            for i in range(90):
                ang = 2 * math.pi * i / 90
                y = -math.cos(ang)
                yy = base - hgt * 0.32 + y * hgt * (0.32 if y > 0 else 0.68)
                xx = cx + math.sin(ang) * wdt / 2 * (1 - max(0.0, -y) ** 1.6 * 0.92)
                pts.append((xx, yy))
            d.polygon(pts, fill=fill)
        flame(S * 0.5, S * 0.86, S * 0.56, S * 0.7, (255, 96, 32, 255))
        flame(S * 0.5, S * 0.86, S * 0.32, S * 0.42, (255, 200, 40, 255))
    elif which == "check":
        d.line([(S * 0.24, S * 0.53), (S * 0.43, S * 0.71), (S * 0.77, S * 0.32)], fill=(30, 170, 84, 255),
               width=int(S * 0.12), joint="curve")
    elif which == "cross":
        lw = int(S * 0.12)
        d.line([(S * 0.3, S * 0.3), (S * 0.7, S * 0.7)], fill=(222, 44, 44, 255), width=lw)
        d.line([(S * 0.7, S * 0.3), (S * 0.3, S * 0.7)], fill=(222, 44, 44, 255), width=lw)
    else:  # star
        pts = []
        for i in range(10):
            r = S * (0.38 if i % 2 == 0 else 0.16)
            a = math.pi * i / 5 - math.pi / 2
            pts.append((S / 2 + r * math.cos(a), S * 0.53 + r * math.sin(a)))
        d.polygon(pts, fill=(255, 190, 20, 255))
    # a soft light across the top of the disc
    arr = np.asarray(img).astype(np.float32)
    ys = np.linspace(0, 1, arr.shape[0])[:, None]
    shine = np.clip(1 - ys / 0.45, 0, 1) ** 2 * 0.16
    arr[..., :3] = arr[..., :3] * (1 - shine[..., None]) + 255 * shine[..., None]
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")
    pad = int(14 * k)
    img = _pad(img, pad)
    out = _die_cut(np.asarray(img), 5 * k, 7 * k)
    return _sprite(Image.fromarray(out, "RGBA"), D + 2 * pad / k, D + 2 * pad / k)


def draw(kind: str, text: str, colour, s: float) -> Sprite:
    if kind == "stamp":
        return draw_stamp(text, colour, s)
    if kind == "price":
        return draw_price(text, colour, s)
    if kind == "starburst":
        return draw_starburst(text, colour, s)
    return draw_badge(text, colour, s)


# ---- choosing ---------------------------------------------------------------------------------------

def _feeling(key: str) -> Optional[str]:
    for name, keys in FEELINGS.items():
        if key in keys:
            return name
    return None


def _shown(text: str) -> str:
    t = text.strip().strip("\"“”'.,!?;:").upper()
    return t[:14]


def kind_for(word: Word, amount: Optional[amounts.Amount], turn: int) -> tuple[str, str, str]:
    """(kind, text, why) for a sticker on `word` chosen by itself."""
    if amount is not None:
        t = "".join(amounts.Unit.of([amount]).exact(amount.value))
        return "price", t, "an amount said"
    k = word.key
    f = _feeling(k)
    if f is not None and k not in VERDICTS:
        return "badge", f, f"a feeling ({f})"
    if k in VERDICTS:
        return "stamp", _shown(moments.bare(word)), "a verdict"
    if k in BURSTS or any(ch.isdigit() for ch in k) or k in moments.NUMBER_WORDS:
        return "starburst", _shown(moments.bare(word)), "news or a number"
    return ("stamp" if turn % 2 == 0 else "starburst"), _shown(moments.bare(word)), "the strongest word of its line"


def auto_spots(clip, count: int, seconds: float) -> list[Spot]:
    words = clip.words
    quiet = clip.quiet
    back = moments.taken_back(words, quiet)
    money = {}
    for a in amounts.spoken(words):
        if a.kind == "amount" and (a.prefix or a.suffix == "%") and not back.holds(words, a.first, a.last):
            money[a.first] = a
    cands: list[tuple[float, int]] = []
    for ln in moments.lines(words, quiet):
        c = moments.best_in(words, ln, quiet, back)
        if c is not None and c.score >= moments.SPECIFIC:
            cands.append((c.score, c.index))
        for i in ln:
            if words[i] in back:
                continue
            if i in money:
                cands.append((40.0, i))
            elif _feeling(words[i].key) or words[i].key in VERDICTS:
                cands.append((20.0 + (c.score if c is not None and c.index == i else 0.0), i))
    want = count or max(1, min(3, int(round(seconds / 5.0))))
    chosen: list[int] = []
    for score, i in sorted(cands, key=lambda c: (-c[0], words[c[1]].start)):
        if i in chosen or words[i].key in HESITATIONS:
            continue
        if i > 0 and words[i - 1].key in ("it's", "i'm", "am", "called") and words[i].clean[:1].isupper():
            continue  # the speaker saying who they are: a name is not a sticker
        if words[i].start < 0.25 or words[i].start > seconds - 0.8:
            continue
        if any(abs(words[i].start - words[j].start) < SPACING for j in chosen):
            continue
        chosen.append(i)
        if len(chosen) >= want:
            break
    out = []
    for turn, i in enumerate(sorted(chosen, key=lambda j: words[j].start)):
        kind, text, why = kind_for(words[i], money.get(i), turn)
        out.append(Spot(words[i].start, words[i].clean, kind, text, why))
    return out


def asked_spots(clip, at: str) -> list[Spot]:
    words = [w for w in clip.words if w.key not in HESITATIONS]
    parts = [p.strip() for p in layers.split(at.replace(",", "|") if _all_times(at) else at) if p.strip()]
    out = []
    for p in parts:
        try:
            t = float(p.rstrip("s"))
        except ValueError:
            t = None
        if t is not None:
            when, said = items.snap(words, t)
            w = next((x for x in words if abs(x.start - when) < 1e-6), None)
            out.append(Spot(when, said, "", "", f"at {t:g} s" + (f", on “{said}”" if said else "")))
            if w is not None:
                out[-1].text = _shown(moments.bare(w))
            continue
        hit = find_said(words, p)
        if hit is None:
            raise O.OptionError(f"at: “{p}” is not said in this video. Read the words with "
                                f"`python -m fx.run words <video>`, or give a time in seconds")
        a, b = hit[0], hit[1]
        span = words[a:b + 1]
        text = " ".join(moments.bare(w) for w in span) if len(span) <= 2 else moments.bare(
            span[moments.key_word(span, clip.quiet) or len(span) - 1])
        out.append(Spot(span[0].start, " ".join(w.clean for w in span), "", _shown(text), f"on “{p}”"))
    return out


def _all_times(at: str) -> bool:
    try:
        [float(x.strip().rstrip("s")) for x in at.replace("|", ",").split(",") if x.strip()]
        return True
    except ValueError:
        return False


# ---- one sticker on screen --------------------------------------------------------------------------

class Sticker:
    def __init__(self, spot: Spot, colour, size: float, tilt: float, fps: float, hold: float, seconds: float):
        self.spot = spot
        self.fps = fps
        self.hit = spot.t
        self.start = max(0.0, self.hit - DROP / fps)
        self.end = min(seconds, self.hit + hold)
        self.tilt = tilt
        self.colour = colour
        self.base = size
        self.make(1.0)

    def make(self, k: float) -> None:
        self.k = k
        flat = draw(self.spot.kind, self.spot.text, self.colour, self.base * k)
        self.sprite, _, _ = rotated(flat, self.tilt)
        self._sizes: dict = {}

    def scale_at(self, t: float) -> float:
        if t < self.hit:  # the drop: from larger, over the frames before its word
            q = (t - self.start) / max(1e-3, self.hit - self.start)
            return FROM + (1.0 - FROM) * min(1.0, max(0.0, q)) ** 1.6
        p = t - self.hit
        if p < 0.05:
            return 1.0 - 0.07 * math.sin(math.pi * p / 0.05)  # it squashes as it hits
        if p < 0.16:
            return 1.0 + 0.025 * math.sin(math.pi * (p - 0.05) / 0.11)
        if t > self.end - LEAVE:
            q = (self.end - t) / LEAVE
            return 0.6 + 0.4 * q
        return 1.0

    def alpha_at(self, t: float) -> float:
        if t < self.start or t >= self.end:
            return 0.0
        if t < self.hit:
            return min(1.0, 0.35 + 0.65 * (t - self.start) / max(1e-3, self.hit - self.start))
        return min(1.0, (self.end - t) / LEAVE)

    def box(self, s: float = 1.0) -> Rect:
        w, h = self.sprite.w * s, self.sprite.h * s
        return Rect(self.cx - w / 2, self.cy - h / 2, self.cx + w / 2, self.cy + h / 2)

    def ink(self, s: float = 1.0) -> Rect:
        """The box of what it draws (its shadow included), at scale `s`."""
        ys, xs = np.nonzero(self.sprite.a > 0.03)
        b = Rect(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
        ox, oy = self.cx - self.sprite.w * s / 2, self.cy - self.sprite.h * s / 2
        return Rect(ox + b.x0 * s, oy + b.y0 * s, ox + b.x1 * s, oy + b.y1 * s)

    def draw(self, frame: np.ndarray, t: float) -> None:
        a = self.alpha_at(t)
        if a <= 0.003:
            return
        s = round(self.scale_at(t), 3)
        sp = self.sprite
        if abs(s - 1.0) > 1e-3:
            if s not in self._sizes:
                if len(self._sizes) > 40:
                    self._sizes.clear()
                self._sizes[s] = sp.scaled(s)
            sp = self._sizes[s]
        over(frame, sp, self.cx - sp.w / 2, self.cy - sp.h / 2, a)


def place(st: Sticker, heads: list[Optional[Rect]], zone: SafeZone, busy: list[tuple[float, float, Rect]],
          side_before: Optional[str], W: int, H: int) -> Optional[str]:
    """The best spot for `st`: inside the zone at rest, clear of the head over every frame it is up
    (its drop from larger too), of the captions and stickers up with it; nearest the head, on the
    other side from the sticker before, higher rather than lower. Smaller when nothing fits."""
    fps = st.fps
    a = int(max(0, math.floor(st.start * fps)))
    b = int(min(len(heads) - 1, math.ceil(st.end * fps)))
    head = Rect.around(h for h in heads[a:b + 1] if h is not None)
    others = [r for (s0, s1, r) in busy if s0 < st.end and st.start < s1]
    z = zone.rect
    gap = 0.012 * W
    for k in (1.0, 0.85, 0.72, 0.62, 0.55):
        if k != st.k:
            st.make(k)
        best, best_score = None, -math.inf
        for cy in np.linspace(z.y0, z.y1, 41):
            for cx in np.linspace(z.x0, z.x1, 25):
                st.cx, st.cy = float(cx), float(cy)
                rest = st.ink(1.0)
                if not zone.contains(rest):
                    continue
                big = st.ink(FROM)
                if head is not None and big.intersects(head, margin=gap):
                    continue
                if any(rest.intersects(o, margin=gap) for o in others):
                    continue
                score = 0.0
                if head is not None:
                    dx = max(head.x0 - rest.x1, rest.x0 - head.x1, 0.0)
                    dy = max(head.y0 - rest.y1, rest.y0 - head.y1, 0.0)
                    score -= math.hypot(dx, dy) / W * 4.0  # near the head
                    side = "left" if cx < head.cx else "right"
                    if side_before is not None and side == side_before:
                        score -= 0.6
                    score -= max(0.0, (cy - head.cy) / H) * 1.2  # higher rather than lower
                else:
                    score -= abs(cy - (z.y0 + 0.25 * z.h)) / H
                if score > best_score:
                    best, best_score = (float(cx), float(cy)), score
        if best is not None:
            st.cx, st.cy = best
            return ("left" if head is None or best[0] < head.cx else "right")
    return None


# ---- the effect -------------------------------------------------------------------------------------

def _per(raw: str, n: int) -> list[str]:
    parts = [p.strip() for p in layers.split(raw)] if raw.strip() else []
    if not parts:
        return [""] * n
    if len(parts) == 1:
        return parts * n
    return (parts + [""] * n)[:n]


class Sounds:
    def __init__(self, base: np.ndarray, stickers: list[Sticker]):
        voice = soundmod.integrated(soundmod.blocks(base)) if base.size else soundmod.SILENCE_DB
        self.ref = voice if voice > -70 else -20.0
        self.stickers = stickers

    def mix(self, base: np.ndarray, rate: int) -> np.ndarray:
        from ..core import loudness

        stem = np.zeros_like(base)
        for n, st in enumerate(self.stickers):
            x, anchor = synth.slap(0.95 if st.spot.kind == "stamp" else 0.35, seed=61 + n)
            soundmod.lay(stem, synth.at_level(x, self.ref - UNDER_VOICE), st.hit, rate, anchor=anchor)
        out = base + stem
        ceiling = max(-1.0, loudness.true_peak_db(base) + 0.05)
        if loudness.true_peak_db(out) > ceiling:
            out, _ = loudness.limit(out, ceiling)
        return out.astype(np.float32)


def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from ..core import framing as framing_mod
    from ..core.timeline import Timeline
    from . import Job, PlanError

    info = clip.info
    fr = framing_mod.plan(info.width, info.height, info.rate, info.frames, clip.face, shape="keep")
    tl = Timeline.whole(info.frames / info.rate)
    W, H, fps = fr.out_w, fr.out_h, info.rate
    n = tl.output_frames(fps)
    seconds = n / fps
    notes: list[str] = []
    if plan is not None:
        try:
            spots = [Spot(float(s["t"]), str(s.get("word", "")), str(s["kind"]), str(s["text"]), "from the plan")
                     for s in plan.get("stickers") or []]
        except (KeyError, TypeError, ValueError):
            raise PlanError("each sticker in the plan needs t, kind and text; run once without --plan") from None
        if not spots:
            raise PlanError("the plan has no stickers; run once without --plan to get one to edit")
    else:
        at = opts["at"].strip()
        spots = auto_spots(clip, opts["count"], seconds) if at.lower() in ("", "auto") else asked_spots(clip, at)
        if not spots:
            raise O.OptionError("no word stands out for a sticker; name the words with at=\"…\"")
        kinds = _per(opts["kind"], len(spots))
        texts = _per(opts["text"], len(spots))
        for turn, (sp, k, tx) in enumerate(zip(spots, kinds, texts)):
            k = k.lower()
            if k not in ("", "auto") + KINDS:
                raise O.OptionError(f"kind: choose auto, {', '.join(KINDS)} (got {k!r})")
            if k in ("", "auto") and not sp.kind:
                w = next((x for x in clip.words if abs(x.start - sp.t) < 1e-6), None)
                if w is not None:
                    sp.kind, auto_text, why = kind_for(w, None, turn)
                    sp.why += f"; {why}"
                    if not tx:
                        sp.text = auto_text
                else:
                    sp.kind = "starburst"
            elif k not in ("", "auto"):
                sp.kind = k
            if tx:
                sp.text = tx.upper() if sp.kind != "badge" else tx.lower()
            if sp.kind == "badge":
                if sp.text.lower() not in BADGES:
                    guess = _feeling(sp.text.lower())
                    sp.text = guess or "star"
                sp.text = sp.text.lower()
            elif not sp.text:
                sp.text = _shown(sp.word) or "WOW"
    zone = SafeZone(W, H)
    head_at = head_in_output(clip, tl, fr)
    heads = [head_at(j) for j in range(n)]
    caption_layer = None
    busy: list[tuple[float, float, Rect]] = []
    if opts["captions"] != "off":
        caption_layer = layers.prepare(clip, layers.Spec("captions", {"look": opts["captions"]}), None, None, fr,
                                       fix=opts["fix"])
        for c in caption_layer.job.plan.get("cards", []):
            busy.append((float(c["start"]), float(c["end"]), Rect.of(c["rect"])))
    u = W / 1080.0
    stickers: list[Sticker] = []
    side = None
    for i, sp in enumerate(spots):
        colour = COLOURS[sp.kind] if opts["color"] == "auto" else textmod.colour(opts["color"])
        tilt = (8.0 + 3.0 * ((i * 7) % 3) / 2) * (1 if i % 2 == 0 else -1)
        base = {"stamp": 300, "price": 300, "starburst": 270, "badge": 230}[sp.kind] * u
        st = Sticker(sp, colour, base, tilt, fps, opts["hold"], seconds)
        side_now = place(st, heads, zone, busy, side, W, H)
        if side_now is None:
            notes.append(f"no room beside your head for the sticker on “{sp.word}” at {sp.t:.2f} s; left out")
            continue
        if st.k < 1.0:
            notes.append(f"the sticker on “{sp.word}” is drawn at {st.k:.0%}: the room beside your head is tight")
        side = side_now
        busy.append((st.start, st.end, st.ink(1.0)))
        stickers.append(st)

    def painter(frame: np.ndarray, j: int, t: float) -> None:
        if caption_layer is not None:
            caption_layer.draw(frame, j, t)
        for st in stickers:
            st.draw(frame, t)

    over_face = outside = 0
    for st in stickers:
        rest = st.ink(1.0)
        for j in range(int(math.floor(st.start * fps)), min(n, int(math.ceil(st.end * fps)))):
            t = j / fps
            if st.alpha_at(t) <= 0:
                continue
            r = st.ink(max(1.0, st.scale_at(t)))
            if heads[j] is not None and r.intersects(heads[j]):
                over_face += 1
            if not zone.contains(rest):
                outside += 1
    checks = {"stickers": len(stickers), "frames_over_face": over_face, "frames_outside_safe_zone": outside}
    for i, st in enumerate(stickers, 1):
        checks[f"sticker {i}"] = (f"{st.spot.kind} “{st.spot.text}” hits at {st.hit:.2f} s on “{st.spot.word}” "
                                  f"({st.spot.why}), up to {st.end:.2f} s, at {st.k:.0%}, turned {st.tilt:+.0f}°")
    if caption_layer is not None:
        checks["captions"] = f"{opts['captions']}, the stickers clear of them"
        checks["frames_over_face"] += int(caption_layer.job.checks.get("frames_over_face", 0) or 0)
    checks["sound"] = "a slap under your voice as each lands" if opts["sound"] and stickers else "your sound, untouched"
    checks["warnings"] = notes
    if not stickers:
        raise O.OptionError("there is no room for a sticker anywhere clear of your face in this shot")
    out_plan = {"effect": NAME, "clip": clip.path, "options": opts,
                "output": {"width": W, "height": H, "fps": round(fps, 3), "seconds": round(tl.duration, 3),
                           "framing": fr.note},
                "stickers": [{"t": round(st.hit, 3), "word": st.spot.word, "kind": st.spot.kind, "text": st.spot.text,
                              "end": round(st.end, 3), "rect": st.ink(1.0).as_list(), "scale": round(st.k, 3),
                              "tilt": st.tilt} for st in stickers]}
    picks = [t for st in stickers for t in (st.hit - 1.5 / fps, st.hit + 0.02, st.hit + 0.6)]
    if len(picks) > 6:  # three stickers or more: each as it drops and once it sits
        picks = [x for st in stickers[:3] for x in (st.hit - 1.5 / fps, st.hit + 0.3)]
    out_plan["sheet_times"] = sorted(round(min(seconds - 0.05, max(0.0, t)), 3) for t in picks[:6])

    def stills() -> list[float]:
        out = []
        for st in stickers[:3]:
            out += [st.hit - 1.5 / fps, st.hit + 0.02, st.hit + 0.5]
        return out[:5]

    sounds = Sounds(clip.sound, stickers) if opts["sound"] and stickers else None
    return Job(NAME, clip, tl, fr, painter, out_plan, checks=checks, still_times=stills,
               sound=sounds.mix if sounds else None)


# The runs `tests/matrix.py stickers` makes: the words it picks by itself on each clip, every kind
# on the plain wall, a price typed by the person, and captions kept clear of.
MATRIX = [(c, c, {}) for c in ("A", "B", "C", "D")] + [
    ("D-kinds", "D", {"at": "watching|lively city|my hotel", "kind": "stamp|starburst|badge", "text": "|CITY|heart"}),
    ("B-price", "B", {"at": "production quality|edits for you", "kind": "price|badge", "text": "$49|check"}),
    ("B-captions", "B", {"captions": "stacked"}),
]
