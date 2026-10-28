"""cta-card: the call to action at the end: comment a word, follow, or watch the comment being posted.

Three kinds, each drawn here in its own plain look (no app's name, mark, colours or layout):

- `comment`: a card that says COMMENT, the word in big letters typing itself out, and a pill with
  what it gets you ("for the link"). For when people are asked to comment a word.
- `follow` (and `subscribe`): a button that pops in low on the frame; a pointer comes in, presses
  it, and it turns to "Following" with a tick and a ring. With `word` set to your handle, the
  handle sits beside the button; without it, no handle is made up.
- `comment-box`: a comment bar; the word types in, the send button lights, it is pressed, and the
  comment appears posted above the bar.

Every kind shows the press being made, so the viewer has only to do the same, and each is up for
about three seconds: long enough to read twice, short enough not to litter the frame. It lands on
the last three seconds unless `when` names a time or words that are said ("comment edit"). It sits
below the chin or above the head, whichever has room, inside the safe zone and never over the face,
taking the head's box over every frame it is on screen. A soft click is laid under the voice on the
press (and quiet keys while a word types); `sound=off` leaves the clip's sound untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core import text as textmod
from ..core.compose import Sprite, over
from ..core.layout import Rect, SafeZone
from ..core.words import HESITATIONS, normalize
from . import options as O
from .captions import head_in_output
from .headline import Token, _cut_right, _ease_in_out, _ease_out, _parse_time, draw_line, find_said

NAME = "cta-card"
KINDS = ("auto", "comment", "follow", "subscribe", "comment-box")

OPTIONS = {
    "kind": O.Choice(KINDS, "auto: a comment card when a word is given, else a follow button · comment: COMMENT "
                            "and the big word · follow (or subscribe): a button pressed and turned to Following · "
                            "comment-box: the word typed into a comment bar and posted"),
    "word": O.Text("the word people comment (EDIT, GUIDE); for follow, your handle (@yourname), shown beside the "
                   "button. Never made up: left empty, follow shows the button alone"),
    "line": O.Text("the pill under the comment card's word: what commenting gets them (default: for the link)"),
    "when": O.Text("when it lands: end (the last 3 seconds), a time in seconds such as 12.5, start, or words "
                   "that are said (\"comment edit\")"),
    "position": O.Choice(("auto", "top", "bottom"), "auto: below the chin when there is room, else above the head "
                                                    "(the comment card the other way round) · top · bottom. Never "
                                                    "over the face"),
    "accent": O.Colour("the colour of the word, the button and the send button: auto (yellow), a name, or a hex "
                       "colour"),
    "sound": O.Switch("on lays a soft click under the press (and keys while the word types); off keeps the "
                      "clip's sound exactly"),
}

LENGTH = 3.2  # seconds on screen
TYPE_FROM = 0.22  # the comment card's word starts typing this long after the card lands
ACCENT = "#ffd400"
INK = (17, 17, 19)
WHITE = (255, 255, 255)
PLATE = (20, 20, 24)  # the dark card
MUTED = (150, 150, 158)
DONE = (58, 58, 64)  # a pressed button: plain, so it reads as done
FACE = "poppins-extrabold"
FACE_BOLD = "poppins-bold"
FACE_SEMI = "poppins-semibold"
FACE_BLACK = "poppins-black"


# ---- drawing helpers ------------------------------------------------------------------------------

def _rgba_sprite(img) -> Sprite:
    return Sprite.from_rgba(np.asarray(img))


def _canvas(w: float, h: float, k: int = 4):
    from PIL import Image

    return Image.new("RGBA", (max(1, int(round(w * k))), max(1, int(round(h * k)))), (0, 0, 0, 0))


def _down(img, w: float, h: float):
    from PIL import Image

    return img.resize((max(1, int(round(w))), max(1, int(round(h)))), Image.LANCZOS)


def _pill(w: float, h: float, fill, opacity: float = 1.0, outline=None, width: float = 0.0) -> Sprite:
    from PIL import ImageDraw

    k = 4
    img = _canvas(w, h, k)
    d = ImageDraw.Draw(img)
    a = int(round(255 * opacity))
    d.rounded_rectangle((0, 0, img.width - 1, img.height - 1), radius=img.height / 2, fill=tuple(fill) + (a,),
                        outline=(tuple(outline) + (255,)) if outline else None, width=int(round(width * k)))
    return _rgba_sprite(_down(img, w, h))


def _card(w: float, h: float, radius: float, fill, opacity: float) -> Sprite:
    return textmod.rounded_box(w, h, radius, fill=fill, opacity=opacity)


def _circle(d: float, fill, opacity: float = 1.0) -> Sprite:
    from PIL import ImageDraw

    k = 4
    img = _canvas(d, d, k)
    ImageDraw.Draw(img).ellipse((0, 0, img.width - 1, img.height - 1), fill=tuple(fill) + (int(255 * opacity),))
    return _rgba_sprite(_down(img, d, d))


def _ring(d: float, width: float, colour, opacity: float) -> Sprite:
    from PIL import ImageDraw

    k = 4
    img = _canvas(d, d, k)
    ImageDraw.Draw(img).ellipse((0, 0, img.width - 1, img.height - 1), outline=tuple(colour) + (int(255 * opacity),),
                                width=max(1, int(round(width * k))))
    return _rgba_sprite(_down(img, d, d))


def _bubble_icon(s: float, colour) -> Sprite:
    """A speech bubble: a rounded body and a small tail at its lower left."""
    from PIL import ImageDraw

    k = 4
    img = _canvas(s, s, k)
    d = ImageDraw.Draw(img)
    S = img.width
    lw = max(2, int(S * 0.1))
    d.rounded_rectangle((lw, lw, S - lw, S * 0.74), radius=S * 0.24, outline=tuple(colour) + (255,), width=lw)
    d.polygon([(S * 0.24, S * 0.66), (S * 0.22, S * 0.95), (S * 0.5, S * 0.7)], fill=tuple(colour) + (255,))
    return _rgba_sprite(_down(img, s, s))


def _tick_icon(s: float, colour) -> Sprite:
    from PIL import ImageDraw

    k = 4
    img = _canvas(s, s, k)
    S = img.width
    ImageDraw.Draw(img).line([(S * 0.12, S * 0.55), (S * 0.4, S * 0.82), (S * 0.9, S * 0.22)], fill=tuple(colour) + (255,),
                             width=max(2, int(S * 0.16)), joint="curve")
    return _rgba_sprite(_down(img, s, s))


def _plus_icon(s: float, colour) -> Sprite:
    from PIL import ImageDraw

    k = 4
    img = _canvas(s, s, k)
    S = img.width
    d = ImageDraw.Draw(img)
    lw = max(2, int(S * 0.16))
    d.line([(S * 0.5, S * 0.12), (S * 0.5, S * 0.88)], fill=tuple(colour) + (255,), width=lw)
    d.line([(S * 0.12, S * 0.5), (S * 0.88, S * 0.5)], fill=tuple(colour) + (255,), width=lw)
    return _rgba_sprite(_down(img, s, s))


def _send_icon(s: float, colour) -> Sprite:
    """An arrow pointing up: send."""
    from PIL import ImageDraw

    k = 4
    img = _canvas(s, s, k)
    S = img.width
    d = ImageDraw.Draw(img)
    lw = max(2, int(S * 0.15))
    d.line([(S * 0.5, S * 0.86), (S * 0.5, S * 0.2)], fill=tuple(colour) + (255,), width=lw)
    d.line([(S * 0.2, S * 0.48), (S * 0.5, S * 0.16), (S * 0.8, S * 0.48)], fill=tuple(colour) + (255,), width=lw,
           joint="curve")
    return _rgba_sprite(_down(img, s, s))


def _heart_icon(s: float, colour, filled: bool) -> Sprite:
    from PIL import ImageDraw

    k = 4
    img = _canvas(s, s, k)
    S = img.width
    d = ImageDraw.Draw(img)
    pts = []
    for i in range(64):
        t = 2 * math.pi * i / 64
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((S / 2 + x * S / 36, S * 0.46 - y * S / 36))
    if filled:
        d.polygon(pts, fill=tuple(colour) + (255,))
    else:
        d.line(pts + [pts[0]], fill=tuple(colour) + (255,), width=max(2, int(S * 0.09)), joint="curve")
    return _rgba_sprite(_down(img, s, s))


def _pointer(s: float) -> Sprite:
    """The pointer that presses: a plain arrow, white with a dark edge and a soft shadow."""
    from PIL import Image, ImageDraw, ImageFilter

    k = 4
    W = H = s * 1.25
    img = _canvas(W, H, k)
    S = s * k
    pts = [(0.08, 0.04), (0.08, 0.8), (0.27, 0.63), (0.4, 0.93), (0.53, 0.87), (0.41, 0.58), (0.66, 0.58)]
    poly = [(x * S + 0.06 * S, y * S + 0.04 * S) for x, y in pts]
    shadow = Image.new("L", img.size, 0)
    ImageDraw.Draw(shadow).polygon([(x + 0.03 * S, y + 0.05 * S) for x, y in poly], fill=110)
    shadow = shadow.filter(ImageFilter.GaussianBlur(0.04 * S))
    base = np.zeros((img.height, img.width, 4), np.uint8)
    base[..., 3] = np.asarray(shadow)
    img = Image.alpha_composite(img, Image.fromarray(base, "RGBA"))
    d = ImageDraw.Draw(img)
    d.polygon(poly, fill=(255, 255, 255, 255), outline=(20, 20, 20, 255))
    d.line(poly + [poly[0]], fill=(20, 20, 20, 255), width=max(2, int(0.05 * S)), joint="curve")
    return _rgba_sprite(_down(img, W, H))


def _text(s: str, face: str, px: int, fill, **kw) -> tuple[Sprite, float, float, float]:
    """One run of text: its sprite, the start of its baseline inside the sprite, and its advance."""
    ink = draw_line([Token(s)], [face], [fill], px, textmod.advance(" ", face, px), **kw)
    return Sprite.from_rgba(ink.rgba), ink.ox, ink.oy, ink.advance


def _scaled_about(frame, sprite: Sprite, cx: float, cy: float, x: float, y: float, s: float, alpha: float) -> None:
    """Lay `sprite` (top-left at x, y when unscaled) scaled by `s` about (cx, cy)."""
    if s <= 0.01 or alpha <= 0:
        return
    if abs(s - 1.0) < 1e-3:
        over(frame, sprite, x, y, alpha)
        return
    sp = sprite.scaled(s)
    over(frame, sp, cx + (x - cx) * s, cy + (y - cy) * s, alpha)


def _back_out(x: float) -> float:
    """Ease out with a small overshoot."""
    x = min(1.0, max(0.0, x))
    c = 1.7
    return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2


# ---- the sounds, made here (no sound files) ------------------------------------------------------

def _click(rate: int, rng: np.random.Generator, level: float = 0.11) -> np.ndarray:
    n = int(0.03 * rate)
    t = np.arange(n) / rate
    tone = np.sin(2 * np.pi * 1900 * t) * np.exp(-t / 0.004)
    body = np.sin(2 * np.pi * 420 * t) * np.exp(-t / 0.009) * 0.6
    noise = rng.standard_normal(n) * np.exp(-t / 0.002) * 0.5
    s = tone + body + noise
    return (s / np.abs(s).max() * level).astype(np.float32)


def _key(rate: int, rng: np.random.Generator, level: float = 0.04) -> np.ndarray:
    n = int(0.018 * rate)
    t = np.arange(n) / rate
    s = rng.standard_normal(n) * np.exp(-t / 0.0025) + np.sin(2 * np.pi * rng.uniform(2600, 3400) * t) * np.exp(-t / 0.003)
    s = np.diff(np.concatenate([[0.0], s]))  # brighter, like a key
    return (s / np.abs(s).max() * level * rng.uniform(0.75, 1.0)).astype(np.float32)


def _pop(rate: int, level: float = 0.08) -> np.ndarray:
    n = int(0.09 * rate)
    t = np.arange(n) / rate
    f = 520 + 900 * (t / t[-1])
    s = np.sin(2 * np.pi * np.cumsum(f) / rate) * np.exp(-t / 0.03) * np.minimum(1.0, t / 0.004)
    return (s / np.abs(s).max() * level).astype(np.float32)


# ---- the card ---------------------------------------------------------------------------------------

@dataclass
class Cue:
    at: float  # seconds into the output
    kind: str  # click, key or pop


class CTA:
    def __init__(self, clip, opts: dict, size: tuple[int, int], fps: float, n_frames: int, head_at,
                 plan: Optional[dict] = None):
        self.W, self.H = size
        self.fps = fps
        self.n = n_frames
        self.seconds = n_frames / fps
        self.opts = opts
        self.zone = SafeZone(self.W, self.H)
        self.heads = [head_at(j) for j in range(n_frames)]
        self.u = self.W / 1080.0 * 1.3  # one unit: a pixel of a frame 1080 wide, at the card's reading size
        self.warnings: list[str] = []
        self.cues: list[Cue] = []
        word = opts["word"].strip()
        kind = opts["kind"]
        if kind == "auto":
            kind = "comment" if word and not word.startswith("@") else "follow"
        if kind in ("comment", "comment-box") and not word:
            raise O.OptionError(f"kind={kind} needs the word people should comment: word=EDIT (it is never made up)")
        if kind in ("comment", "comment-box") and len(word) > 16:
            raise O.OptionError("word: a comment word reads best short; keep it to 16 letters or fewer")
        self.kind = kind
        self.word = word
        self.accent = textmod.colour(ACCENT if opts["accent"] == "auto" else opts["accent"])
        if plan is not None:
            c = plan.get("card") or {}
            try:
                self.start, self.end = float(c["start"]), float(c["end"])
            except (KeyError, TypeError, ValueError):
                raise ValueError("the plan has no card; run once without --plan to get one to edit") from None
            self.found = "from the plan"
        else:
            self._when(clip)
        self.end = min(self.end, self.seconds)
        if self.end - self.start < 1.2:
            self.warnings.append("the card has less than 1.2 seconds before the video ends; give an earlier when=")
        self._build()
        self._place()

    # -- when
    def _when(self, clip) -> None:
        when = self.opts["when"].strip()
        wl = when.lower()
        length = min(LENGTH, self.seconds)
        if wl in ("", "auto", "end"):
            self.start = max(0.0, self.seconds - length)
            self.found = f"the last {length:.1f} s"
        elif wl == "start":
            words = [w for w in clip.words if w.key not in HESITATIONS]
            self.start = words[0].start if words else 0.0
            self.found = "at the first words"
        elif (t := _parse_time(when)) is not None:
            if t >= self.seconds:
                raise O.OptionError(f"when: {t:g} s is after the end of the video ({self.seconds:.2f} s)")
            self.start = t
            self.found = f"at {t:.2f} s"
        else:
            words = [w for w in clip.words if w.key not in HESITATIONS]
            hit = find_said(words, when)
            if not hit:
                raise O.OptionError(f"when: the words \"{when}\" are not said in this video. Read the words with "
                                    f"`python -m fx.run words <video>`, or give a time in seconds")
            self.start = words[hit[0]].start
            self.found = f"on \"{' '.join(w.clean for w in words[hit[0]:hit[1] + 1])}\""
        self.end = min(self.seconds, self.start + LENGTH)
        if self.seconds - self.start < LENGTH and wl not in ("", "auto", "end"):
            self.warnings.append(f"the video ends {self.seconds - self.start:.1f} s after the card lands; it is cut short")

    # -- the pieces, drawn once
    def _build(self) -> None:
        getattr(self, f"_build_{self.kind.replace('-', '_')}", self._build_follow)()

    def _build_comment(self) -> None:
        u = self.u
        word = self.word.upper()
        label_px = int(34 * u)
        label, lox, loy, ladv = _text("COMMENT", FACE_BOLD, label_px, (215, 215, 222))
        icon = _bubble_icon(label_px * 1.05, (215, 215, 222))
        word_px = int(150 * u)
        max_w = 0.80 * self.zone.rect.w
        while textmod.advance(word, FACE_BLACK, word_px) > max_w and word_px > 20:
            word_px -= 2
        self.word_px = word_px
        wd = draw_line([Token(word)], [FACE_BLACK], [self.accent], word_px, 0.0)
        self.word_sprite = Sprite.from_rgba(wd.rgba)
        self.word_cuts = [wd.ox + c for c in wd.cuts]
        self.word_adv = wd.advance
        self.word_ox, self.word_oy = wd.ox, wd.oy
        line = self.opts["line"].strip() or "for the link"
        pill_px = int(36 * u)
        pt, pox, poy, padv = _text(line, FACE_SEMI, pill_px, INK)
        cap = textmod.cap_height(FACE_BLACK, word_px)
        pad = 44 * u
        pill_h = pill_px * 1.75
        pill_w = padv + pill_px * 1.5
        inner_w = max(wd.advance, ladv + icon.w + 14 * u, pill_w)
        card_w = inner_w + 2 * pad
        label_h = label_px * 1.0
        gap1, gap2 = 22 * u, 30 * u
        card_h = pad + label_h + gap1 + cap + gap2 + pill_h + pad * 0.9
        self.card = _card(card_w, card_h, 38 * u, PLATE, 0.9)
        self.w, self.h = card_w, card_h
        # where each piece sits inside the card
        lx = (card_w - (icon.w + 14 * u + ladv)) / 2
        self.icon_at = (lx, pad + label_h / 2 - icon.h / 2 - 2 * u)
        self.icon = icon
        self.label = label
        self.label_at = (lx + icon.w + 14 * u - lox, pad + label_h - loy)
        self.word_at = ((card_w - wd.advance) / 2 - wd.ox, pad + label_h + gap1 + cap - wd.oy)
        self.pill = _pill(pill_w, pill_h, WHITE)
        self.pill_at = ((card_w - pill_w) / 2, card_h - pad * 0.9 - pill_h)
        self.pill_text = pt
        self.pill_text_at = (self.pill_at[0] + (pill_w - padv) / 2 - pox, self.pill_at[1] + pill_h / 2 + pill_px * 0.36 - poy)
        self.caret_h = cap

    def _build_follow(self) -> None:
        u = self.u
        sub = self.kind == "subscribe"
        self.btn_words = ("Subscribe", "Subscribed") if sub else ("Follow", "Following")
        px = int(44 * u)
        self.btn_px = px
        h = px * 2.0
        t0, o0x, o0y, a0 = _text(self.btn_words[0], FACE_BOLD, px, INK)
        t1, o1x, o1y, a1 = _text(self.btn_words[1], FACE_BOLD, px, WHITE)
        icon = px * 0.62
        w0 = a0 + icon + px * 0.42 + px * 1.5
        w1 = a1 + icon + px * 0.42 + px * 1.5
        bw = max(w0, w1)
        self.btn_w, self.btn_h = bw, h
        self.btn_up = _pill(bw, h, self.accent)
        self.btn_done = _pill(bw, h, DONE, opacity=0.94, outline=(110, 110, 118), width=2.5 * u)
        self.plus = _plus_icon(icon, INK)
        self.tick = _tick_icon(icon, WHITE)
        self.t_up = (t0, (bw - (icon + px * 0.42 + a0)) / 2 + icon + px * 0.42 - o0x, h / 2 + px * 0.36 - o0y)
        self.t_done = (t1, (bw - (icon + px * 0.42 + a1)) / 2 + icon + px * 0.42 - o1x, h / 2 + px * 0.36 - o1y)
        self.icon_up_at = ((bw - (icon + px * 0.42 + a0)) / 2, (h - icon) / 2)
        self.icon_done_at = ((bw - (icon + px * 0.42 + a1)) / 2, (h - icon) / 2)
        self.ring_d = h * 1.0
        # a handle beside the button, only when it was given
        handle = self.word.strip()
        self.handle = None
        if handle:
            if not handle.startswith("@"):
                handle = "@" + handle
            hp = int(38 * u)
            ht, hox, hoy, hadv = _text(handle, FACE_SEMI, hp, WHITE)
            av = h * 0.78
            initial = normalize(handle)[:1].upper() or "@"
            it, iox, ioy, iadv = _text(initial, FACE_BLACK, int(av * 0.5), INK)
            inset = (h - av) / 2
            chip_w = inset + av + 18 * u + hadv + 34 * u
            self.handle = {"plate": _pill(chip_w, h, PLATE, opacity=0.86), "avatar": _circle(av, self.accent),
                           "initial": it, "init_at": ((av - iadv) / 2 - iox, av / 2 + av * 0.18 - ioy),
                           "text": ht, "text_at": (inset + av + 18 * u - hox, h / 2 + hp * 0.36 - hoy), "w": chip_w,
                           "av": av, "inset": inset}
        gap = 22 * u
        self.w = bw + ((self.handle["w"] + gap) if self.handle else 0.0)
        self.h = h
        self.btn_x = (self.handle["w"] + gap) if self.handle else 0.0
        self.pointer = _pointer(70 * u)
        self.pointer_tip = (0.14 * 70 * u, 0.08 * 70 * u)

    def _build_subscribe(self) -> None:
        self._build_follow()

    def _build_comment_box(self) -> None:
        u = self.u
        bar_w = 0.86 * self.zone.rect.w
        bar_h = 104 * u
        self.bar = _pill(bar_w, bar_h, WHITE, opacity=0.96)
        av = bar_h * 0.62
        self.av = _circle(av, (205, 205, 212))
        it, iox, ioy, iadv = _text("Y", FACE_BLACK, int(av * 0.5), (110, 110, 118))
        self.av_init = (it, (av - iadv) / 2 - iox, av / 2 + av * 0.18 - ioy)
        self.av_at = (bar_h * 0.19, (bar_h - av) / 2)
        px = int(40 * u)
        self.box_px = px
        ph, phox, phoy, _ = _text("Add a comment…", FACE_SEMI, px, (150, 150, 158))
        self.placeholder = (ph, self.av_at[0] + av + 22 * u - phox, bar_h / 2 + px * 0.36 - phoy)
        wd = draw_line([Token(self.word)], [FACE_BOLD], [INK], px, 0.0)
        self.typed = Sprite.from_rgba(wd.rgba)
        self.typed_cuts = [wd.ox + c for c in wd.cuts]
        self.typed_at = (self.av_at[0] + av + 22 * u - wd.ox, bar_h / 2 + px * 0.36 - wd.oy)
        sd = bar_h * 0.72
        self.send_off = _circle(sd, (222, 222, 228))
        self.send_on = _circle(sd, self.accent)
        self.send_icon = _send_icon(sd * 0.52, INK)
        self.send_at = (bar_w - bar_h * 0.14 - sd, (bar_h - sd) / 2)
        self.send_d = sd
        self.bar_w, self.bar_h = bar_w, bar_h
        # the posted comment, above the bar
        cp = int(36 * u)
        name, nox, noy, nadv = _text("you", FACE_BOLD, cp, WHITE)
        when, wox, woy, wadv = _text("just now", FACE_SEMI, int(cp * 0.85), (175, 175, 184))
        body, box_, boy, badv = _text(self.word, FACE_BOLD, int(cp * 1.25), WHITE)
        cav = cp * 2.2
        post_w = max(bar_w * 0.82, cav + 26 * u + max(nadv + 16 * u + wadv, badv) + 90 * u)
        post_h = cp * 3.6
        self.post = _card(post_w, post_h, 30 * u, PLATE, 0.88)
        self.post_av = _circle(cav, self.accent)
        pit, piox, pioy, piadv = _text("Y", FACE_BLACK, int(cav * 0.5), INK)
        self.post_parts = {
            "av": (self.post_av, 22 * u, (post_h - cav) / 2),
            "init": (pit, 22 * u + (cav - piadv) / 2 - piox, (post_h - cav) / 2 + cav / 2 + cav * 0.18 - pioy),
            "name": (name, 22 * u + cav + 22 * u - nox, post_h * 0.42 - noy),
            "when": (when, 22 * u + cav + 22 * u + nadv + 14 * u - wox, post_h * 0.42 - woy),
            "body": (body, 22 * u + cav + 22 * u - box_, post_h * 0.8 - boy),
        }
        self.heart = _heart_icon(cp * 1.1, (200, 200, 208), False)
        self.heart_on = _heart_icon(cp * 1.1, (255, 77, 106), True)
        self.heart_at = (post_w - 30 * u - cp * 1.1, (post_h - cp * 1.1) / 2)
        self.post_w, self.post_h = post_w, post_h
        gap = 20 * u
        self.w = bar_w
        self.h = post_h + gap + bar_h
        self.post_at = ((bar_w - post_w) / 2, 0.0)
        self.bar_at = (0.0, post_h + gap)
        self.pointer = _pointer(70 * u)
        self.pointer_tip = (0.14 * 70 * u, 0.08 * 70 * u)

    # -- where
    def _head_over(self, t0: float, t1: float) -> Optional[Rect]:
        a = int(max(0, math.floor(t0 * self.fps)))
        b = int(min(self.n - 1, math.ceil(t1 * self.fps)))
        return Rect.around(h for h in self.heads[a:b + 1] if h is not None)

    def _place(self) -> None:
        """Below the chin or above the head (the comment card prefers above, the buttons below, where
        a thumb would be), inside the safe zone and clear of the head over every frame it is up.
        Too big for either band, it is drawn smaller, down to 55%; smaller than that, it goes into
        the margin on the roomier side and the run says so."""
        z = self.zone.rect
        gap = 0.025 * self.H
        head = self._head_over(self.start, self.end)
        if head is None:
            rooms = {"top": (z.y0, z.y0 + 0.35 * self.H), "bottom": (z.y1 - 0.35 * self.H, z.y1)}
        else:
            rooms = {"top": (z.y0, head.y0 - gap), "bottom": (head.y1 + gap, z.y1)}
        order = ["top", "bottom"] if self.kind == "comment" else ["bottom", "top"]
        pos = self.opts["position"]
        if pos in order:
            order.remove(pos)
            order.insert(0, pos)
        side, self.scale = None, 1.0
        for s in order:
            r0, r1 = rooms[s]
            if r1 - r0 >= self.h and self.w <= z.w:
                side = s
                break
        if side is None:
            s = max(order, key=lambda k: rooms[k][1] - rooms[k][0])
            r0, r1 = rooms[s]
            k = min((r1 - r0) / self.h, z.w / self.w)
            if k >= 0.55:
                side, self.scale = s, k
        if pos in ("top", "bottom") and side is not None and side != pos:
            self.warnings.append(f"there is no room {'above' if pos == 'top' else 'below'} the head; it went "
                                 f"{'below' if pos == 'top' else 'above'} instead")
        if side is None:  # the margin, smaller, on the roomier side; never the face
            above = head.y0 if head is not None else self.H
            below = self.H - head.y1 if head is not None else self.H
            self.scale = 0.55
            if above >= below:
                r0, r1, side = 0.015 * self.H, above - 0.01 * self.H, "top"
            else:
                r0, r1, side = self.H - below + 0.01 * self.H, 0.985 * self.H, "bottom"
            self.where = "outside the safe zone"
            self.warnings.append("there was no room for the card inside the safe zone clear of the face; it sits in "
                                 "the margin, smaller")
        else:
            r0, r1 = rooms[side]
            self.where = side
        w, h = self.w * self.scale, self.h * self.scale
        self.x = z.cx - w / 2
        self.y = r0 + max(0.0, (r1 - r0) - h) * (0.62 if side == "top" else 0.45)
        self.band = (r0, r1)
        self.side = side

    @property
    def rect(self) -> Rect:
        return Rect(self.x, self.y, self.x + self.w * self.scale, self.y + self.h * self.scale)

    # -- drawing
    def draw(self, frame: np.ndarray, j: int, t: float) -> None:
        if not (self.start <= t < self.end):
            return
        p = t - self.start
        out = min(1.0, (self.end - t) / 0.22)
        getattr(self, f"_draw_{self.kind.replace('-', '_')}", self._draw_follow)(frame, p, out)

    def _lay(self, frame, sprite: Sprite, x: float, y: float, alpha: float, pop: float = 1.0,
             about: Optional[tuple[float, float]] = None) -> None:
        """A piece of the card, `x, y` inside the card, scaled with the card (and by `pop` about
        `about`, a point inside the card)."""
        s = self.scale
        X, Y = self.x + x * s, self.y + y * s
        sp = sprite if abs(s - 1.0) < 1e-3 else sprite.scaled(s)
        if about is not None and abs(pop - 1.0) > 1e-3:
            cx, cy = self.x + about[0] * s, self.y + about[1] * s
            _scaled_about(frame, sp, cx, cy, X, Y, pop, alpha)
        else:
            over(frame, sp, X, Y, alpha)

    def _draw_comment(self, frame, p: float, out: float) -> None:
        cx, cy = self.w / 2, self.h / 2
        q = p / 0.34
        pop = 0.6 + 0.4 * _back_out(q) if q < 1 else 1.0
        a = min(1.0, p / 0.12) * out
        # the card pops in with its label and a cursor already in it: it is never seen empty
        self._lay(frame, self.card, 0, 0, a, pop, (cx, cy))
        self._lay(frame, self.icon, *self.icon_at, a, pop, (cx, cy))
        self._lay(frame, self.label, *self.label_at, a, pop, (cx, cy))
        t0 = TYPE_FROM
        per = min(0.11, 0.7 / max(1, len(self.word_cuts)))
        n = 0 if p < t0 else min(len(self.word_cuts), int((p - t0) / per) + 1)
        if n:
            sp = self.word_sprite if n >= len(self.word_cuts) else _cut_right(self.word_sprite, self.word_cuts[n - 1] + 3)
            self._lay(frame, sp, *self.word_at, a)
        typing = n < len(self.word_cuts)
        if typing or (int((p - t0) * 2.4) % 2 == 0 and p < t0 + len(self.word_cuts) * per + 0.9):
            start_x = self.word_at[0] + self.word_ox  # where the word begins: the cursor waits there
            x = (self.word_at[0] + self.word_cuts[n - 1] if n else start_x) + 0.05 * self.word_px
            bar = _bar(max(3.0, 0.06 * self.word_px), self.caret_h, self.accent)
            self._lay(frame, bar, x, self.word_at[1] + self.word_oy - self.caret_h, a, pop, (cx, cy))
        # the pill rises in once the word is out
        tp = t0 + len(self.word_cuts) * per + 0.15
        if p >= tp:
            qp = _ease_out((p - tp) / 0.3)
            dy = (1 - qp) * 24 * self.u
            self._lay(frame, self.pill, self.pill_at[0], self.pill_at[1] + dy, a * qp)
            self._lay(frame, self.pill_text, self.pill_text_at[0], self.pill_text_at[1] + dy, a * qp)

    def _press_time(self) -> float:
        return 1.15

    def _pointer_at(self, p: float, target: tuple[float, float]) -> Optional[tuple[float, float, float]]:
        """Where the pointer's tip is at `p`, in the frame, and its opacity. It comes in from the
        right edge on a gentle curve that stays in the card's own band (above the head or below the
        chin), so it never crosses the face; it presses, then goes back the way it came."""
        tx, ty = self.x + target[0] * self.scale, self.y + target[1] * self.scale
        r0, r1 = self.band
        sx = self.W * 1.06
        sy = min(r1, max(r0, ty + (0.05 if self.side == "bottom" else -0.03) * self.H))
        a_in, press = 0.45, self._press_time()
        a_out = press + 0.55
        if p < a_in:
            return None
        if p < press:
            q = _ease_in_out((p - a_in) / (press - a_in))
            mx, my = (sx + tx) / 2, min(r1, max(r0, max(sy, ty) + 0.02 * self.H))
            x = (1 - q) ** 2 * sx + 2 * (1 - q) * q * mx + q ** 2 * tx
            y = (1 - q) ** 2 * sy + 2 * (1 - q) * q * my + q ** 2 * ty
            return x, y, min(1.0, (p - a_in) / 0.12)
        if p < a_out:
            return tx, ty, 1.0
        q = _ease_in_out((p - a_out) / 0.45)
        if q >= 1:
            return None
        return tx + q * (sx - tx) * 0.5, ty + q * (sy - ty) * 0.5, 1.0 - q

    def _draw_pointer(self, frame, p: float, target, out: float) -> None:
        at = self._pointer_at(p, target)
        if at is None:
            return
        x, y, a = at
        press = self._press_time()
        k = self.scale * (0.88 if press <= p < press + 0.12 else 1.0)  # it dips as it presses
        sp = self.pointer if abs(k - 1.0) < 1e-3 else self.pointer.scaled(k)
        tx, ty = self.pointer_tip
        over(frame, sp, x - tx * k, y - ty * k, a * out)

    def _pointer_rect(self, p: float, target) -> Optional[Rect]:
        at = self._pointer_at(p, target)
        if at is None:
            return None
        x, y, _ = at
        tx, ty = self.pointer_tip
        k = self.scale
        return Rect(x - tx * k, y - ty * k, x - tx * k + self.pointer.w * k, y - ty * k + self.pointer.h * k)

    def _draw_follow(self, frame, p: float, out: float) -> None:
        bx = self.btn_x
        cx, cy = bx + self.btn_w / 2, self.btn_h / 2
        q = p / 0.32
        pop = 0.5 + 0.5 * _back_out(q) if q < 1 else 1.0
        a = min(1.0, p / 0.1) * out
        if self.handle is not None:
            hq = _ease_out((p - 0.1) / 0.3)
            if hq > 0:
                hd = self.handle
                dx = (1 - hq) * 40 * self.u
                av, inset = hd["av"], hd["inset"]
                self._lay(frame, hd["plate"], dx, 0, a * hq)
                self._lay(frame, hd["avatar"], dx + inset, inset, a * hq)
                self._lay(frame, hd["initial"], dx + inset + hd["init_at"][0], inset + hd["init_at"][1], a * hq)
                self._lay(frame, hd["text"], dx + hd["text_at"][0], hd["text_at"][1], a * hq)
        press = self._press_time()
        done = p >= press + 0.06
        squash = 1.0
        if press <= p < press + 0.16:
            squash = 1.0 - 0.08 * math.sin(math.pi * (p - press) / 0.16)
        s = pop * squash
        if not done:
            self._lay(frame, self.btn_up, bx, 0, a, s, (cx, cy))
            self._lay(frame, self.plus, bx + self.icon_up_at[0], self.icon_up_at[1], a, s, (cx, cy))
            self._lay(frame, self.t_up[0], bx + self.t_up[1], self.t_up[2], a, s, (cx, cy))
        else:
            self._lay(frame, self.btn_done, bx, 0, a, s, (cx, cy))
            self._lay(frame, self.tick, bx + self.icon_done_at[0], self.icon_done_at[1], a, s, (cx, cy))
            self._lay(frame, self.t_done[0], bx + self.t_done[1], self.t_done[2], a, s, (cx, cy))
            rq = (p - press - 0.06) / 0.45
            if 0 <= rq < 1:
                d = self.btn_h * (1.0 + 1.1 * _ease_out(rq))
                ring = _ring(d * self.scale, 4 * self.u * self.scale, self.accent, 1.0)
                over(frame, ring, self.x + cx * self.scale - ring.w / 2, self.y + cy * self.scale - ring.h / 2,
                     (1 - rq) * 0.8 * out)
        self._draw_pointer(frame, p, self._target(), out)

    def _draw_comment_box(self, frame, p: float, out: float) -> None:
        bx, by = self.bar_at
        q = _ease_out(p / 0.3)
        dy = (1 - q) * 50 * self.u
        a = min(1.0, p / 0.15) * out
        t_type = 0.45
        per = min(0.09, 0.6 / max(1, len(self.typed_cuts)))
        typed_end = t_type + len(self.typed_cuts) * per
        press = max(self._press_time(), typed_end + 0.25)
        self._press = press
        posted = p >= press + 0.08
        self._lay(frame, self.bar, bx, by + dy, a)
        avs, axx, ayy = self.av, self.av_at[0], self.av_at[1]
        self._lay(frame, avs, bx + axx, by + ayy + dy, a)
        self._lay(frame, self.av_init[0], bx + axx + self.av_init[1], by + ayy + self.av_init[2] + dy, a)
        if p < t_type or posted:
            ph = self.placeholder
            self._lay(frame, ph[0], bx + ph[1], by + ph[2] + dy, a)
        else:
            n = min(len(self.typed_cuts), int((p - t_type) / per) + 1)
            sp = self.typed if n >= len(self.typed_cuts) else _cut_right(self.typed, self.typed_cuts[n - 1] + 2)
            self._lay(frame, sp, bx + self.typed_at[0], by + self.typed_at[1] + dy, a)
            if int(p * 2.6) % 2 == 0 or n < len(self.typed_cuts):
                x = self.typed_at[0] + self.typed_cuts[n - 1] + 0.08 * self.box_px
                self._lay(frame, _bar(max(2.0, 0.07 * self.box_px), self.box_px * 0.95, (40, 40, 46)), bx + x,
                          by + self.bar_h / 2 - self.box_px * 0.5 + dy, a)
        ready = (t_type + per) <= p < press + 0.08
        sx, sy = self.send_at
        scx, scy = bx + sx + self.send_d / 2, by + sy + self.send_d / 2
        squash = 1.0 - 0.12 * math.sin(math.pi * (p - press) / 0.16) if press <= p < press + 0.16 else 1.0
        self._lay(frame, self.send_on if ready else self.send_off, bx + sx, by + sy + dy, a, squash, (scx, scy))
        ic = self.send_icon
        self._lay(frame, ic, bx + sx + (self.send_d - ic.w) / 2, by + sy + (self.send_d - ic.h) / 2 + dy, a, squash,
                  (scx, scy))
        if posted:
            pq = _ease_out((p - press - 0.08) / 0.35)
            pdy = (1 - pq) * 60 * self.u
            px0, py0 = self.post_at
            self._lay(frame, self.post, px0, py0 + pdy, a * pq)
            for key in ("av", "init", "name", "when", "body"):
                sp, x, y = self.post_parts[key]
                self._lay(frame, sp, px0 + x, py0 + y + pdy, a * pq)
            heart_on = p >= press + 0.75
            hp = 1.0 + (0.25 * math.sin(math.pi * min(1.0, (p - press - 0.75) / 0.25)) if heart_on else 0.0)
            hx, hy = self.heart_at
            self._lay(frame, self.heart_on if heart_on else self.heart, px0 + hx, py0 + hy + pdy, a * pq, hp,
                      (px0 + hx + self.heart.w / 2, py0 + hy + self.heart.h / 2))
        self._draw_pointer(frame, p, self._target(), out)

    # -- the sounds
    def sound_cues(self) -> list[Cue]:
        cues = []
        if self.kind in ("follow", "subscribe"):
            cues.append(Cue(self.start + self._press_time(), "click"))
            cues.append(Cue(self.start + self._press_time() + 0.07, "pop"))
        elif self.kind == "comment":
            per = min(0.11, 0.7 / max(1, len(self.word_cuts)))
            for k in range(len(self.word_cuts)):
                cues.append(Cue(self.start + TYPE_FROM + k * per, "key"))
            cues.append(Cue(self.start + TYPE_FROM + len(self.word_cuts) * per + 0.15, "pop"))
        else:
            per = min(0.09, 0.6 / max(1, len(self.typed_cuts)))
            for k in range(len(self.typed_cuts)):
                cues.append(Cue(self.start + 0.45 + k * per, "key"))
            press = max(self._press_time(), 0.45 + len(self.typed_cuts) * per + 0.25)
            cues.append(Cue(self.start + press, "click"))
            cues.append(Cue(self.start + press + 0.08, "pop"))
        return [c for c in cues if c.at < self.end - 0.05]

    def mix(self, sound: np.ndarray, rate: int) -> np.ndarray:
        if not self.opts["sound"]:
            return sound
        rng = np.random.default_rng(7)  # the same sounds every render
        out = sound.copy()
        for c in self.sound_cues():
            s = _click(rate, rng) if c.kind == "click" else _key(rate, rng) if c.kind == "key" else _pop(rate)
            i = int(round(c.at * rate))
            if i >= len(out):
                continue
            n = min(len(s), len(out) - i)
            out[i:i + n] += s[:n, None]
        return np.clip(out, -1.0, 1.0)

    # -- checks and plan
    def _target(self) -> tuple[float, float]:
        if self.kind == "comment-box":
            return (self.bar_at[0] + self.send_at[0] + self.send_d * 0.55, self.bar_at[1] + self.send_at[1] + self.send_d * 0.6)
        return (self.btn_x + self.btn_w * 0.62, self.btn_h * 0.58)

    def drawn_rect(self, t: float) -> Optional[Rect]:
        """The card's box at `t`, and the pointer's when it is on screen."""
        if not (self.start <= t < self.end):
            return None
        r = self.rect
        if self.kind != "comment":
            pr = self._pointer_rect(t - self.start, self._target())
            if pr is not None:
                r = r.union(pr)
        return r

    def report(self) -> dict:
        over_face = outside = pointer_over_face = 0
        for j in range(self.n):
            t = j / self.fps
            r = self.drawn_rect(t)
            if r is None:
                continue
            h = self.heads[j]
            if h is not None and self.rect.intersects(h):
                over_face += 1
            elif h is not None and self.kind != "comment":
                pr = self._pointer_rect(j / self.fps - self.start, self._target())
                if pr is not None and pr.intersects(h):
                    pointer_over_face += 1
            if not self.zone.contains(self.rect):
                outside += 1
        out = {"frames_over_face": over_face, "frames_outside_safe_zone": outside, "placed": self.where,
               "kind": self.kind, "lands": f"{self.start:.2f} s ({self.found})",
               "on_screen": f"{self.start:.2f}-{self.end:.2f} s", "size": f"{self.scale:.0%}",
               "sounds": len(self.sound_cues()) if self.opts["sound"] else 0, "warnings": list(self.warnings)}
        if self.kind != "comment":
            out["pointer_frames_over_face"] = pointer_over_face
        return out

    def plan(self) -> dict:
        return {"kind": self.kind, "word": self.word, "start": round(self.start, 3), "end": round(self.end, 3),
                "where": self.where, "rect": self.rect.as_list(), "scale": round(self.scale, 3)}

    def sheet_times(self) -> list[float]:
        s, e = self.start, self.end
        if self.kind == "comment":
            picks = [s + 0.12, s + 0.5, s + 0.85, s + 1.4, s + 2.2, e - 0.1]
        elif self.kind == "comment-box":
            press = max(self._press_time(), 0.45 + len(self.typed_cuts) * min(0.09, 0.6 / max(1, len(self.typed_cuts))) + 0.25)
            picks = [s + 0.15, s + 0.6, s + press - 0.25, s + press + 0.05, s + press + 0.5, s + press + 1.0]
        else:
            press = self._press_time()
            picks = [s + 0.12, s + 0.7, s + press - 0.02, s + press + 0.15, s + press + 0.6, e - 0.1]
        return sorted(min(e - 0.5 / self.fps, max(s, x)) for x in picks)


def _bar(w: float, h: float, colour) -> Sprite:
    w, h = max(1, int(round(w))), max(1, int(round(h)))
    return Sprite(rgb=np.full((h, w, 3), colour, np.float32), a=np.ones((h, w), np.float32))


# ---- the effect --------------------------------------------------------------------------------

def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from ..core import framing as framing_mod
    from ..core.timeline import Timeline
    from . import Job, PlanError

    info = clip.info
    fr = framing_mod.plan(info.width, info.height, info.rate, info.frames, clip.face, shape="keep")
    tl = Timeline.whole(info.frames / info.rate)
    try:
        cta = CTA(clip, opts, (fr.out_w, fr.out_h), info.rate, tl.output_frames(info.rate),
                  head_in_output(clip, tl, fr), plan=plan)
    except ValueError as e:
        if plan is not None and not isinstance(e, O.OptionError):
            raise PlanError(str(e)) from None
        raise
    out_plan = {"effect": NAME, "clip": clip.path, "options": opts,
                "output": {"width": fr.out_w, "height": fr.out_h, "fps": round(info.rate, 3),
                           "seconds": round(tl.duration, 3), "framing": fr.note},
                "card": cta.plan(), "sheet_times": [round(t, 3) for t in cta.sheet_times()]}
    return Job(NAME, clip, tl, fr, cta.draw, out_plan, checks=cta.report(),
               still_times=lambda: cta.sheet_times()[1:5], sound=cta.mix)
