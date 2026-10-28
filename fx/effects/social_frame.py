"""social-frame: the clip shrinks into a frame and keeps playing there: a post card, a phone, a fan
of three phones, or the middle tile of a profile grid.

What it does, in order:

1. **The moment.** At `at` (auto: the start of the strongest spoken line in the first half, from the
   shared picker; or a time, `start`, or words you say) the picture pulls back. Until then the
   video is exactly as shot.
2. **One camera, pulled back.** The frame is laid out once at full size: a ground of the clip's own
   colour, and the card, the phone or the grid on it, with the clip's own rectangle where it plays.
   The camera starts framed on that rectangle, so the first frame of the move is the video as it
   was, and eases back to the whole layout (`core/zoom.py`'s camera path). The clip is drawn from
   its own pixels at every size, never enlarged from a small copy, so it stays sharp all the way.
3. **The frame is ours.** A light post card, a graphite phone with a camera dot, three phones
   fanned, a grid of upright tiles: drawn here in code, with no app's layout, icon, logo or colour.
4. **Nothing is made up.** The name, the caption and the counts appear only when typed; with none
   given the frame carries no words at all. The fan's side phones and the grid's other tiles show
   your own pictures (`pictures=`), or stills of this same clip, each toned differently, which the
   run says are stand-ins.
5. **The face stays in view.** The clip is shown whole inside its frame, so the face is never cut;
   the avatar beside the name is cut from your own face (the face track); words sit outside the
   picture, inside the safe zone, so they never cover it.
6. **Sound.** A soft whoosh as the picture pulls back (and the side phones or the tiles land), under
   your voice, which is kept as it was.
7. **Captions** (`captions=on`) are the pack's own, laid over the finished frame and kept clear of
   the face where the frame puts it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..core import sound as soundmod
from ..core import synth
from ..core import text as textmod
from ..core import zoom
from ..core.compose import Sprite, over, rotated
from ..core.layout import Rect, SafeZone
from . import options as O

NAME = "social-frame"
LOOKS = ("card", "phone", "fan", "grid")
CAPTIONS = ("off", "on", "stacked", "one-word", "serif-chip", "fitted")
MOVE = {"card": 0.62, "phone": 0.62, "fan": 0.70, "grid": 1.00}  # seconds the pull-back takes
KIND = {"card": "snap", "phone": "snap", "fan": "snap", "grid": "slow"}
BACK = 0.50  # seconds the return to the whole frame takes, when `duration` ends it
LEAD = 0.08  # the pull-back starts this long before its word
UNDER_VOICE = 10.0  # the whoosh's loudest tenth of a second, this far under the voice (LU)
PAPER = (246, 244, 239)  # the post card
INK = (22, 23, 26)
INK_SOFT = (92, 95, 104)
GRAPHITE = (30, 32, 37)
RIM = (86, 90, 100)

OPTIONS = {
    "look": O.Choice(LOOKS, "card: a light post card · phone: a phone of our own · fan: three phones, yours in "
                            "the middle · grid: the middle tile of a three-by-three grid"),
    "at": O.Text("when it pulls back: auto (the start of a strong line in the first half), start, a time in "
                 "seconds, or words you say"),
    "duration": O.Number(0.0, 600.0, "seconds it stays framed before the picture fills the screen again; 0 keeps "
                                     "it framed to the end"),
    "name": O.Text("the name over the card or under the phone (yours, as typed); empty for none"),
    "caption": O.Text("one line under the card or the phone (the line under your name on the grid); empty for none"),
    "likes": O.Text("a like count to show, exactly as typed (\"1,204\"); empty for none. Never made up"),
    "comments": O.Text("a comment count to show, exactly as typed; empty for none. Never made up"),
    "pictures": O.Text("for fan and grid: a folder or a ;-separated list of your own pictures or clips for the "
                       "other phones and tiles; empty uses stills of this clip"),
    "background": O.Colour("the ground behind the frame: auto (the clip's own colour, darkened) or a colour"),
    "captions": O.Choice(CAPTIONS, "off, or the pack's captions over the finished frame, clear of your face: on "
                                   "(stacked), or a look: one-word, serif-chip, fitted"),
    "sound": O.Switch("a soft whoosh as it pulls back, under your voice: on or off"),
    "fix": O.Text("misheard words as wrong=right, separated by ;  e.g. \"ask editor=AskEditor\""),
}

MATRIX = [
    ("A", "A", {"name": "askeditor", "caption": "Demo day, part two", "likes": "1,204", "comments": "86"}),
    ("B", "B", {"look": "phone", "name": "askeditor", "caption": "Airport edit"}),
    ("D", "D", {"look": "fan", "name": "askeditor"}),
    ("B-grid", "B", {"look": "grid", "name": "askeditor", "caption": "Videos worth posting"}),
    ("D-grid", "D", {"look": "grid", "name": "askeditor"}),
    ("C", "C", {"name": "askeditor", "captions": "on"}),
    ("A-phone", "A", {"look": "phone", "at": "start", "duration": 4}),
]


# ---- small drawing helpers ------------------------------------------------------------------------

def _rgba(w: int, h: int):
    from PIL import Image

    return Image.new("RGBA", (max(1, int(w)), max(1, int(h))), (0, 0, 0, 0))


def _rr_alpha(w: float, h: float, r: float, ox: float = 0.0, oy: float = 0.0,
              size: Optional[tuple[int, int]] = None) -> np.ndarray:
    """The coverage (0..1) of a rounded rectangle w x h with corners r whose top-left is at (ox, oy),
    over a grid of `size` (default: just round it): measured as a distance, so its edge is smooth at
    any size and any sub-pixel place."""
    W, H = size if size is not None else (int(math.ceil(w)), int(math.ceil(h)))
    r = max(0.0, min(r, w / 2, h / 2))
    xs = np.arange(W, dtype=np.float32) + 0.5 - ox
    ys = np.arange(H, dtype=np.float32) + 0.5 - oy
    qx = np.abs(xs - w / 2) - (w / 2 - r)
    qy = np.abs(ys - h / 2) - (h / 2 - r)
    ax, ay = np.maximum(qx, 0)[None, :], np.maximum(qy, 0)[:, None]
    d = np.sqrt(ax * ax + ay * ay) + np.minimum(np.maximum(qx[None, :], qy[:, None]), 0) - r
    return np.clip(0.5 - d, 0.0, 1.0).astype(np.float32)


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    import cv2

    return cv2.GaussianBlur(a, (0, 0), max(0.1, sigma))


def _lay_alpha(canvas: np.ndarray, alpha: np.ndarray, colour, x: int, y: int, opacity: float = 1.0) -> None:
    """Paint `colour` through `alpha` (float, its own size) onto `canvas` (float32 RGB) at (x, y)."""
    H, W = canvas.shape[:2]
    h, w = alpha.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    a = alpha[y0 - y:y1 - y, x0 - x:x1 - x, None] * opacity
    region = canvas[y0:y1, x0:x1]
    region *= 1.0 - a
    region += a * np.asarray(colour, np.float32)


def _shadow(canvas: np.ndarray, rect: Rect, r: float, u: float, strength: float = 0.5) -> None:
    """A soft shadow under a rounded shape, a little below it."""
    pad = int(40 * u)
    a = _rr_alpha(rect.w, rect.h, r, pad, pad, (int(rect.w) + 2 * pad, int(rect.h) + 2 * pad))
    a = _blur(a, 16 * u) * strength
    _lay_alpha(canvas, a, (0, 0, 0), int(rect.x0) - pad, int(rect.y0) - pad + int(10 * u))


def _text_sprite(text: str, face: str, px: int, fill) -> tuple[Sprite, float, float]:
    """`text` as a sprite, and where its baseline's start sits in it."""
    w = textmod.draw_word(text, face, px, fill=fill)
    return w.sprite, w.ox, w.oy


def wrap(text: str, face: str, px: int, width: float, most: int = 2) -> list[str]:
    """`text` in lines no wider than `width`, broken between words; past `most` lines the last is
    shortened by whole words and ends with an ellipsis."""
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if textmod.advance(trial, face, px) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    if len(lines) > most:
        last = " ".join(lines[most - 1:]).split()
        while last and textmod.advance(" ".join(last) + "…", face, px) > width:
            last.pop()
        lines = lines[:most - 1] + [" ".join(last) + "…"]
    return lines


def heart(d: float, colour) -> Sprite:
    """A plain heart, d across."""
    from PIL import Image, ImageDraw

    k = 4
    img = Image.new("RGBA", (int(d * k) + 2, int(d * k) + 2), (0, 0, 0, 0))
    pts = []
    for i in range(120):
        a = 2 * math.pi * i / 120
        x = 16 * math.sin(a) ** 3
        y = 13 * math.cos(a) - 5 * math.cos(2 * a) - 2 * math.cos(3 * a) - math.cos(4 * a)
        pts.append(((x / 34 + 0.5) * d * k, (0.47 - y / 34) * d * k))
    ImageDraw.Draw(img).polygon(pts, fill=tuple(colour) + (255,))
    return Sprite.from_rgba(np.asarray(img.resize((int(d) + 1, int(d) + 1), Image.LANCZOS)))


def bubble(d: float, colour) -> Sprite:
    """A plain speech bubble, d across: a round-cornered box with a short tail at its lower left."""
    from PIL import Image, ImageDraw

    k = 4
    D = int(d * k)
    img = Image.new("RGBA", (D + 2, D + 2), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    dr.rounded_rectangle((0.04 * D, 0.10 * D, 0.96 * D, 0.76 * D), radius=0.22 * D, fill=tuple(colour) + (255,))
    dr.polygon([(0.20 * D, 0.70 * D), (0.18 * D, 0.95 * D), (0.44 * D, 0.72 * D)], fill=tuple(colour) + (255,))
    return Sprite.from_rgba(np.asarray(img.resize((int(d) + 1, int(d) + 1), Image.LANCZOS)))


def _over_f(canvas: np.ndarray, sp: Sprite, x: float, y: float, opacity: float = 1.0) -> None:
    """`over` for a float32 canvas."""
    H, W = canvas.shape[:2]
    x0, y0 = int(round(x)), int(round(y))
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x0 + sp.w), min(H, y0 + sp.h)
    if cx1 <= cx0 or cy1 <= cy0 or opacity <= 0:
        return
    a = sp.a[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0, None] * opacity
    rgb = sp.rgb[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] * opacity
    region = canvas[cy0:cy1, cx0:cx1]
    region *= 1.0 - a
    region += rgb


def _ramp(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def _back_out(x: float, k: float = 1.7) -> float:
    x = min(1.0, max(0.0, x)) - 1.0
    return 1.0 + (k + 1) * x ** 3 + k * x ** 2


# ---- what goes in the other phones and tiles -------------------------------------------------------

PICTURE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".mp4", ".mov", ".m4v"}
TONES = ("warm", "cool", "film", "black-and-white", "pop", "moody", "warm", "cool")


def _cover(img: np.ndarray, w: int, h: int, centre: Optional[tuple[float, float]] = None,
           zoom_in: float = 1.0) -> np.ndarray:
    """`img` cut to w:h and scaled to cover w x h, about `centre` (default its middle)."""
    import cv2

    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih) * zoom_in
    cw, ch = w / s, h / s
    cx, cy = centre if centre is not None else (iw / 2, ih / 2)
    x0 = min(max(0.0, cx - cw / 2), iw - cw)
    y0 = min(max(0.0, cy - ch / 2), ih - ch)
    m = np.float32([[s, 0, -x0 * s], [0, s, -y0 * s]])
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def own_pictures(spec: str, clip_dir: Path) -> list[Path]:
    """The person's own pictures for the other phones and tiles: a folder (its files in name order)
    or a ;-separated list of files."""
    spec = spec.strip()
    if not spec:
        return []
    out: list[Path] = []
    for part in [p.strip() for p in spec.split(";") if p.strip()]:
        p = Path(part).expanduser()
        if not p.is_absolute():
            p = (clip_dir / p) if (clip_dir / p).exists() else p
        if p.is_dir():
            out += sorted(q for q in p.iterdir() if q.suffix.lower() in PICTURE_EXTS and not q.name.startswith("."))
        elif p.exists():
            out.append(p)
        else:
            raise O.OptionError(f"pictures: there is no file or folder at {part}")
    return out


def _read_picture(p: Path) -> np.ndarray:
    from ..core import video

    if p.suffix.lower() in {".mp4", ".mov", ".m4v"}:
        info = video.probe(p)
        return video.read_frame(info, info.frames // 2)
    from PIL import Image

    return np.asarray(Image.open(p).convert("RGB"))


def stand_ins(clip, fr, n: int, w: int, h: int, avoid: float, pictures: list[Path]) -> tuple[list[np.ndarray], str]:
    """`n` pictures w x h for the frames round the live one: the person's own, else stills of this
    clip at moments spread across it (away from `avoid`, the moment it pulls back), each cut round
    the face at its own closeness and toned with one of the grade's looks."""
    from ..core import video
    from . import grade

    if pictures:
        out = [_cover(_read_picture(pictures[k % len(pictures)]), w, h) for k in range(n)]
        return out, f"{len(pictures)} of your own pictures"
    info = clip.info
    dur = info.frames / info.rate
    times = [dur * (k + 0.5) / n for k in range(n)]
    times = [t if abs(t - avoid) > 0.6 else min(dur - 0.05, t + 0.7) for t in times]
    out = []
    for k, t in enumerate(times):
        i = info.frame_at(t)
        img = fr.apply(video.read_frame(info, i), i)
        b = clip.face.box(i) if clip.face.any() else None
        centre = None
        if b is not None:
            b = fr.to_output(b, i)
            centre = (b.cx, b.cy)
        pic = _cover(img, w, h, centre, zoom_in=(1.0, 1.6, 1.2, 2.0)[k % 4])
        look = TONES[k % len(TONES)]
        with np.errstate(all="ignore"):  # the maths library's matmul warns on clean input; the result is checked below
            toned = np.nan_to_num(grade.apply(pic.astype(np.float64) / 255.0, look, 0.9))
        out.append(np.clip(toned * 255.0 + 0.5, 0, 255).astype(np.uint8))
    return out, f"stills of this clip at {', '.join(f'{t:.1f}' for t in times)} s, each toned (stand-ins: give pictures= for your own)"


def avatar(clip, fr, at: float, d: int) -> Optional[Sprite]:
    """A round picture of the face, d across, cut from the clip near `at`; None when no face is found."""
    from ..core import video

    face = clip.face
    if not face.any():
        return None
    info = clip.info
    i0 = info.frame_at(at)
    found = np.flatnonzero(face.found)
    i = int(found[np.argmin(np.abs(found - i0))])
    b = fr.to_output(face.box(i), i)
    img = fr.apply(video.read_frame(info, i), i)
    side = 2.0 * max(b.w, b.h)
    cx, cy = b.cx, b.cy + 0.08 * b.h
    import cv2

    s = d / side
    m = np.float32([[s, 0, -(cx - side / 2) * s], [0, s, -(cy - side / 2) * s]])
    pic = cv2.warpAffine(img, m, (d, d), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    a = _rr_alpha(d, d, d / 2)
    return Sprite(rgb=pic.astype(np.float32) * a[..., None], a=a)


def initial(name: str, d: int, ground) -> Sprite:
    """No face to cut: a plain disc with the name's first letter."""
    from PIL import Image, ImageDraw

    k = 4
    img = Image.new("RGBA", (d * k, d * k), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    dr.ellipse((0, 0, d * k - 1, d * k - 1), fill=tuple(ground) + (255,))
    letter = (name.strip()[:1] or "*").upper()
    dr.text((d * k / 2, d * k / 2), letter, font=textmod.font("poppins-bold", int(d * k * 0.5)),
            fill=(255, 255, 255, 255), anchor="mm")
    return Sprite.from_rgba(np.asarray(img.resize((d, d), Image.LANCZOS)))


# ---- the ground ----------------------------------------------------------------------------------

def ground_colour(clip, fr, given: str) -> tuple[int, int, int]:
    """The ground: the colour asked for, or the clip's own average colour, kept in its hue and made
    deep and quiet, so the frame on it reads first."""
    import colorsys

    from ..core import video

    if given != "auto":
        return textmod.colour(given)
    info = clip.info
    picks = []
    for k in range(5):
        i = info.frame_at(info.frames / info.rate * (k + 0.5) / 5)
        picks.append(fr.apply(video.read_frame(info, i, size=(64, max(2, int(64 * info.height / info.width)))), i)
                     if not fr.reframed else video.read_frame(info, i, size=(64, max(2, int(64 * info.height / info.width)))))
    r, g, b = (np.median(np.concatenate([p.reshape(-1, 3) for p in picks]), axis=0) / 255.0).tolist()
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    rr, gg, bb = colorsys.hsv_to_rgb(h, min(0.45, s * 0.8 + 0.08), 0.16)
    return int(rr * 255), int(gg * 255), int(bb * 255)


def make_ground(W: int, H: int, base, centre: tuple[float, float], u: float) -> np.ndarray:
    """A deep ground lit softly behind the frame, with a faint dot grid, as float32 RGB."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - centre[0]) / W) ** 2 + ((yy - centre[1]) / H) ** 2 * 0.55)
    glow = np.clip(1.0 - d / 0.75, 0, 1) ** 2
    base = np.asarray(base, np.float32)
    lit = np.minimum(255.0, base * 1.9 + 14.0)
    canvas = base[None, None, :] * (1 - glow[..., None]) + lit[None, None, :] * glow[..., None]
    step = 36 * u
    gx = np.abs(((xx + step / 2) % step) - step / 2)
    gy = np.abs(((yy + step / 2) % step) - step / 2)
    dots = np.clip(1.6 * u - np.sqrt(gx * gx + gy * gy), 0, 1) * 0.10
    canvas += dots[..., None] * 255.0 * (0.4 + 0.6 * glow[..., None])
    return np.clip(canvas, 0, 255).astype(np.float32)


# ---- the layout ----------------------------------------------------------------------------------

@dataclass
class Part:
    """Something laid on the ground that moves or appears: drawn, per frame, before the clip goes on."""
    draw: Callable[[np.ndarray, float], None]
    settled: float  # from this output second on it no longer changes


@dataclass
class Layout:
    W: int
    H: int
    clip_rect: Rect  # where the clip plays, in the frame's own pixels (the frame's own shape)
    radius: float  # the clip's corners there
    ground: np.ndarray  # float32 RGB: the ground and everything still under the clip
    parts: list[Part] = field(default_factory=list)
    words: list[Rect] = field(default_factory=list)  # where the words sit, for the checks
    notes: list[str] = field(default_factory=list)
    others: str = ""


def _fit_clip(W: int, H: int, top: float, bottom: float, most_w: float) -> tuple[float, float]:
    """The clip's width and height (the frame's own shape) to fill the band top..bottom, no wider
    than most_w."""
    h = bottom - top
    w = h * W / H
    if w > most_w:
        w, h = most_w, most_w * H / W
    return w, h


def build_layout(look: str, clip, fr, opts: dict, at: float, land: float) -> Layout:
    W, H = fr.out_w, fr.out_h
    u = W / 1080.0
    zone = SafeZone(W, H).rect
    base = ground_colour(clip, fr, opts["background"])
    name, caption = opts["name"].strip(), opts["caption"].strip()
    likes, comments = opts["likes"].strip(), opts["comments"].strip()
    builders = {"card": _card, "phone": _phone, "fan": _fan, "grid": _grid}
    return builders[look](clip, fr, W, H, u, zone, base, name, caption, likes, comments, opts, at, land)


def _counts_row(likes: str, comments: str, px: int, colour, u: float) -> tuple[list[tuple[Sprite, float, float]], float, float]:
    """The counts as typed, each after its plain mark: [(sprite, x, y)] from (0, 0), the row's width
    and height."""
    items, x = [], 0.0
    d = px * 1.05
    cap = textmod.cap_height("poppins-semibold", px)
    for mark, value in ((heart, likes), (bubble, comments)):
        if not value:
            continue
        m = mark(d, colour)
        items.append((m, x, (cap - d) / 2 - 0.08 * d))
        x += d + 10 * u
        sp, ox, oy = _text_sprite(value, "poppins-semibold", px, colour)
        items.append((sp, x - ox, cap - oy))
        x += textmod.advance(value, "poppins-semibold", px) + 34 * u
    return items, max(0.0, x - 34 * u), cap


def _lay_words(canvas: np.ndarray, lines: list[str], face: str, px: int, colour, x: float, y_base: float,
               gap: float, alpha: float, centre: bool = False, width: float = 0.0) -> float:
    """Lines of words with their first baseline at y_base; returns the last baseline."""
    y = y_base
    for k, line in enumerate(lines):
        sp, ox, oy = _text_sprite(line, face, px, colour)
        lx = x + (width - textmod.advance(line, face, px)) / 2 if centre else x
        _over_f(canvas, sp, lx - ox, y - oy, alpha)
        if k < len(lines) - 1:
            y += gap
    return y


def _appear(land: float, delay: float = 0.06, length: float = 0.28) -> Callable[[float], float]:
    return lambda t: _ramp((t - land - delay) / length)


def _card(clip, fr, W, H, u, zone, base, name, caption, likes, comments, opts, at, land) -> Layout:
    pad = 26 * u
    head_h = 96 * u if name else 0.0
    count_px = int(34 * u)
    cap_px = int(33 * u)
    has_counts = bool(likes or comments)
    foot_h = (22 * u if (has_counts or caption) else 0.0) + (count_px * 1.5 if has_counts else 0.0)
    top, bottom = zone.y0 + 6 * u, zone.y1 - 8 * u
    # the caption's lines depend on the width; fit twice
    lines: list[str] = []
    for _ in range(2):
        extra = head_h + foot_h + (len(lines) * cap_px * 1.32 + (8 * u if lines else 0.0)) + 2 * pad
        cw, ch = _fit_clip(W, H, top, bottom - extra, 0.74 * W - 2 * pad)
        lines = wrap(caption, "poppins-semibold", cap_px, cw, 2) if caption else []
    extra = head_h + foot_h + (len(lines) * cap_px * 1.32 + (8 * u if lines else 0.0)) + 2 * pad
    cw, ch = _fit_clip(W, H, top, bottom - extra, 0.74 * W - 2 * pad)
    pw, ph = cw + 2 * pad, ch + extra
    px0 = (W - pw) / 2
    py0 = top + ((bottom - top) - ph) / 2
    panel = Rect(px0, py0, px0 + pw, py0 + ph)
    clip_rect = Rect(px0 + pad, py0 + pad + head_h, px0 + pad + cw, py0 + pad + head_h + ch)
    canvas = make_ground(W, H, base, (panel.cx, panel.cy), u)
    r_panel = 34 * u
    _shadow(canvas, panel, r_panel, u, 0.55)
    _lay_alpha(canvas, _rr_alpha(pw, ph, r_panel, px0 - int(px0), py0 - int(py0), (int(pw) + 2, int(ph) + 2)),
               PAPER, int(px0), int(py0))
    lay = Layout(W, H, clip_rect, 18 * u, canvas)
    appear = _appear(land)
    if name:
        d = int(60 * u)
        av = avatar(clip, fr, at, d) or initial(name, d, (60, 64, 72))
        ax, ay = clip_rect.x0, py0 + pad + (head_h - d) / 2 - 6 * u
        npx = int(36 * u)
        nm = wrap(name, "poppins-bold", npx, cw - d - 20 * u, 1)
        ny = ay + d / 2 + textmod.cap_height("poppins-bold", npx) / 2

        def draw_head(c: np.ndarray, t: float, av=av, ax=ax, ay=ay, ny=ny) -> None:
            a = appear(t)
            _over_f(c, av, ax, ay, a)
            _lay_words(c, nm, "poppins-bold", npx, INK, ax + d + 18 * u, ny, 0, a)

        lay.parts.append(Part(draw_head, land + 0.4))
        lay.words.append(Rect(ax, ay, clip_rect.x1, ay + d))
    y = clip_rect.y1 + 22 * u
    if has_counts:
        items, rw, rh = _counts_row(likes, comments, count_px, INK, u)
        rx, ry = clip_rect.x0 + 2 * u, y

        def draw_counts(c: np.ndarray, t: float, items=items, rx=rx, ry=ry) -> None:
            a = _ramp((t - land - 0.12) / 0.28)
            for sp, x, yy in items:
                _over_f(c, sp, rx + x, ry + yy, a)

        lay.parts.append(Part(draw_counts, land + 0.5))
        lay.words.append(Rect(rx, ry - 4 * u, rx + rw, ry + rh * 1.3))
        y += count_px * 1.5
    if lines:
        yb = y + 8 * u + textmod.cap_height("poppins-semibold", cap_px)

        def draw_caption(c: np.ndarray, t: float, yb=yb) -> None:
            a = _ramp((t - land - 0.18) / 0.28)
            _lay_words(c, lines, "poppins-semibold", cap_px, INK, clip_rect.x0 + 2 * u, yb, cap_px * 1.32, a)

        lay.parts.append(Part(draw_caption, land + 0.5))
        lay.words.append(Rect(clip_rect.x0, yb - cap_px, clip_rect.x1, yb + (len(lines) - 1) * cap_px * 1.32 + 0.3 * cap_px))
    return lay


def phone_body(sw: float, sh: float, u: float) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """A phone of our own, round a screen sw x sh: graphite body with a lighter rim, a black bezel,
    a camera dot in the bezel above the screen and buttons on its sides. Returns the body as
    premultiplied RGB and alpha (its own size, with room for the buttons), where the screen's
    top-left sits in it, and the screen's corner radius."""
    bez = 22 * u
    rim = 3.0 * u
    bw, bh = sw + 2 * bez, sh + 2 * bez + 26 * u
    side = 6 * u
    Wb, Hb = int(math.ceil(bw + 2 * side)) + 2, int(math.ceil(bh)) + 2
    size = (Wb, Hb)
    colour = np.empty((Hb, Wb, 3), np.float32)
    colour[:] = np.asarray(RIM, np.float32)
    a = np.zeros((Hb, Wb), np.float32)
    for bx, by, bl in ((0.0, 0.22 * bh, 0.10 * bh), (0.0, 0.36 * bh, 0.07 * bh), (bw + 2 * side - 2.4 * side, 0.28 * bh, 0.12 * bh)):
        a = np.maximum(a, _rr_alpha(2.4 * side, bl, side, bx + 0.3 * side, by, size))
    r_body = 0.13 * bw

    def paint(alpha: np.ndarray, c) -> None:
        colour[:] = colour * (1 - alpha[..., None]) + alpha[..., None] * np.asarray(c, np.float32)

    body = _rr_alpha(bw, bh, r_body, side, 0, size)
    a = np.maximum(a, body)
    paint(_rr_alpha(bw - 2 * rim, bh - 2 * rim, r_body - rim, side + rim, rim, size), GRAPHITE)
    e = rim + 5 * u
    paint(_rr_alpha(bw - 2 * e, bh - 2 * e, r_body - e, side + e, e, size), (8, 8, 10))
    sx, sy = side + bez, bez + 26 * u
    paint(_rr_alpha(12 * u, 12 * u, 6 * u, side + bw / 2 - 6 * u, (sy - 12 * u) / 2 + 2 * u, size), (40, 44, 58))
    return colour * a[..., None], a, sx, sy, 0.085 * sw


def _phone_sprite(sw: float, sh: float, u: float, screen: Optional[np.ndarray]) -> tuple[Sprite, float, float, float]:
    rgb, a, sx, sy, rs = phone_body(sw, sh, u)
    if screen is not None:
        import cv2

        pic = cv2.resize(screen, (int(round(sw)), int(round(sh))), interpolation=cv2.INTER_AREA).astype(np.float32)
        m = _rr_alpha(sw, sh, rs, 0, 0, (pic.shape[1], pic.shape[0]))
        y0, x0 = int(round(sy)), int(round(sx))
        reg = rgb[y0:y0 + pic.shape[0], x0:x0 + pic.shape[1]]
        reg[:] = reg * (1 - m[..., None]) + pic * m[..., None]
    return Sprite(rgb=rgb, a=a), sx, sy, rs


def _under_words(lay: Layout, W, u, zone, top_y: float, name: str, caption: str, likes: str, comments: str,
                 land: float, colour=(255, 255, 255)) -> float:
    """The name, the caption and the counts, centred under a phone from `top_y`, inside the safe zone."""
    y = top_y
    width = zone.w
    if name:
        px = int(38 * u)
        nm = wrap(name, "poppins-bold", px, width, 1)
        yb = y + textmod.cap_height("poppins-bold", px)

        def draw_name(c, t, yb=yb):
            _lay_words(c, nm, "poppins-bold", px, colour, zone.x0, yb, 0, _ramp((t - land - 0.06) / 0.28), True, width)

        lay.parts.append(Part(draw_name, land + 0.4))
        lay.words.append(Rect(zone.x0, y, zone.x1, yb + 0.3 * px))
        y = yb + 18 * u
    if caption:
        px = int(31 * u)
        lines = wrap(caption, "poppins-semibold", px, width, 2)
        yb = y + textmod.cap_height("poppins-semibold", px)

        def draw_cap(c, t, yb=yb, lines=lines):
            _lay_words(c, lines, "poppins-semibold", px, tuple(int(v * 0.86) for v in colour), zone.x0, yb, px * 1.3,
                       _ramp((t - land - 0.14) / 0.28), True, width)

        lay.parts.append(Part(draw_cap, land + 0.5))
        lay.words.append(Rect(zone.x0, y, zone.x1, yb + (len(lines) - 1) * px * 1.3 + 0.3 * px))
        y = yb + (len(lines) - 1) * px * 1.3 + 18 * u
    if likes or comments:
        px = int(32 * u)
        items, rw, rh = _counts_row(likes, comments, px, colour, u)
        rx, ry = (W - rw) / 2, y + 6 * u

        def draw_counts(c, t, items=items, rx=rx, ry=ry):
            a = _ramp((t - land - 0.2) / 0.28)
            for sp, x, yy in items:
                _over_f(c, sp, rx + x, ry + yy, a)

        lay.parts.append(Part(draw_counts, land + 0.55))
        lay.words.append(Rect(rx, ry - 4 * u, rx + rw, ry + rh * 1.3))
        y = ry + rh * 1.5
    return y


def _words_height(u, zone, name, caption, likes, comments) -> float:
    h = 0.0
    if name:
        h += textmod.cap_height("poppins-bold", int(38 * u)) + 18 * u
    if caption:
        px = int(31 * u)
        n = len(wrap(caption, "poppins-semibold", px, zone.w, 2))
        h += textmod.cap_height("poppins-semibold", px) + (n - 1) * px * 1.3 + 18 * u
    if likes or comments:
        h += 6 * u + textmod.cap_height("poppins-semibold", int(32 * u)) * 1.5
    return h + (22 * u if h else 0.0)


def _phone(clip, fr, W, H, u, zone, base, name, caption, likes, comments, opts, at, land) -> Layout:
    words_h = _words_height(u, zone, name, caption, likes, comments)
    top, bottom = zone.y0 + 4 * u, zone.y1 - 6 * u - words_h
    bez = 22 * u
    sw, sh = _fit_clip(W, H, top + bez + 26 * u, bottom - bez, 0.66 * W)
    rgb, a, sx, sy, rs = phone_body(sw, sh, u)
    bh_, bw_ = a.shape
    total = bh_ + words_h
    y0 = top + ((zone.y1 - 6 * u - top) - total) / 2
    x0 = (W - bw_) / 2
    clip_rect = Rect(x0 + sx, y0 + sy, x0 + sx + sw, y0 + sy + sh)
    canvas = make_ground(W, H, base, (clip_rect.cx, clip_rect.cy), u)
    _shadow(canvas, Rect(x0 + 6 * u, y0, x0 + bw_ - 6 * u, y0 + bh_), 0.13 * bw_, u, 0.6)
    _over_f(canvas, Sprite(rgb=rgb, a=a), x0, y0)
    lay = Layout(W, H, clip_rect, rs, canvas)
    _under_words(lay, W, u, zone, y0 + bh_ + 22 * u, name, caption, likes, comments, land)
    return lay


def _fan(clip, fr, W, H, u, zone, base, name, caption, likes, comments, opts, at, land) -> Layout:
    words_h = _words_height(u, zone, name, caption, likes, comments)
    top, bottom = zone.y0 + 30 * u, zone.y1 - 6 * u - words_h
    bez = 22 * u
    sw, sh = _fit_clip(W, H, top + bez + 26 * u, bottom - bez - 20 * u, 0.50 * W)
    rgb, a, sx, sy, rs = phone_body(sw, sh, u)
    bh_, bw_ = a.shape
    total = bh_ + words_h
    y0 = top + ((zone.y1 - 6 * u - top) - total) / 2
    x0 = (W - bw_) / 2
    clip_rect = Rect(x0 + sx, y0 + sy, x0 + sx + sw, y0 + sy + sh)
    canvas = make_ground(W, H, base, (clip_rect.cx, clip_rect.cy), u)
    pics, about = stand_ins(clip, fr, 2, int(round(sw)), int(round(sh)), at,
                            own_pictures(opts["pictures"], Path(clip.path).parent))
    k = 0.88
    sides = []
    for side, pic in zip((-1, 1), pics):
        sp, _, _, _ = _phone_sprite(sw * k, sh * k, u * k, pic)
        sides.append((side, sp))
    centre = (W / 2, y0 + bh_ / 2)
    spread, tilt, drop = 0.255 * W, 9.0, 0.035 * H
    cache: dict = {}

    def side_at(side: int, sp: Sprite, q: float):
        ang = round(-side * tilt * q * 4) / 4  # quarter degrees: turned once each, then kept
        key = (side, ang)
        if key not in cache:
            cache[key] = rotated(sp, ang)
        return cache[key]

    start = land - 0.30

    def draw_sides(c: np.ndarray, t: float) -> None:
        for n, (side, sp) in enumerate(sides):
            q = _back_out((t - start - 0.07 * n) / 0.45, 1.2)
            q = max(0.0, q)
            r, dx, dy = side_at(side, sp, min(q, 1.08))
            cx = centre[0] + side * spread * q
            cy = centre[1] + drop * q
            shadow_a = _blur(r.a, 12 * u) * 0.5 * min(1.0, q * 2)
            _lay_alpha(c, shadow_a, (0, 0, 0), int(cx - r.w / 2), int(cy - r.h / 2 + 12 * u))
            _over_f(c, r, cx - r.w / 2, cy - r.h / 2)

    # the side phones go under the middle one: laid on the ground first, then the middle phone's body
    body = Sprite(rgb=rgb, a=a)
    still_ground = canvas.copy()

    def draw_all(c: np.ndarray, t: float) -> None:
        draw_sides(c, t)
        _shadow(c, Rect(x0 + 6 * u, y0, x0 + bw_ - 6 * u, y0 + bh_), 0.13 * bw_, u, 0.6)
        _over_f(c, body, x0, y0)

    # until the sides move, the middle phone alone is on the ground
    _shadow(still_ground, Rect(x0 + 6 * u, y0, x0 + bw_ - 6 * u, y0 + bh_), 0.13 * bw_, u, 0.6)
    _over_f(still_ground, body, x0, y0)
    lay = Layout(W, H, clip_rect, rs, still_ground)

    def draw_fan(c: np.ndarray, t: float) -> None:
        if t <= start:
            return
        c[:] = canvas
        draw_all(c, t)

    lay.parts.append(Part(draw_fan, start + 0.07 + 0.6))
    lay.others = about
    _under_words(lay, W, u, zone, y0 + bh_ + 22 * u + drop * 0.3, name, caption, likes, comments, land)
    lay.notes.append(f"side phones {spread / W:.0%} out, turned {tilt:g}°")
    return lay


def _grid(clip, fr, W, H, u, zone, base, name, caption, likes, comments, opts, at, land) -> Layout:
    gap = 10 * u
    gw = 0.86 * W
    tw = (gw - 2 * gap) / 3
    th = tw * H / W
    head_h = 0.0
    if name or caption:
        head_h = 120 * u
    grid_h = 3 * th + 2 * gap
    # the middle tile is centred in what the header leaves, and the rows run off the bottom if they must
    cy = zone.y0 + head_h + (zone.y1 - zone.y0 - head_h) / 2
    y_mid = cy - th / 2
    gx0 = (W - gw) / 2
    gy0 = y_mid - th - gap
    if head_h:  # the header sits over the grid, never on a tile: the rows run off the bottom instead
        gy0 = max(gy0, zone.y0 + head_h + gap)
    tiles = []
    for r in range(3):
        for c in range(3):
            tiles.append(Rect(gx0 + c * (tw + gap), gy0 + r * (th + gap), gx0 + c * (tw + gap) + tw, gy0 + r * (th + gap) + th))
    clip_rect = tiles[4]
    canvas = make_ground(W, H, base, (clip_rect.cx, clip_rect.cy), u)
    pics, about = stand_ins(clip, fr, 8, int(round(tw)), int(round(th)), at,
                            own_pictures(opts["pictures"], Path(clip.path).parent))
    r_tile = 14 * u
    sprites = []
    order = [0, 1, 2, 5, 8, 7, 6, 3]  # round the middle, from the top left
    for k, idx in enumerate(order):
        rect = tiles[idx]
        m = _rr_alpha(rect.w, rect.h, r_tile, 0, 0, (int(round(rect.w)), int(round(rect.h))))
        import cv2

        pic = cv2.resize(pics[k], (m.shape[1], m.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32) * 0.74
        sprites.append((idx, Sprite(rgb=pic * m[..., None], a=m), k))
    # a quiet ring round the live tile
    ring = _rr_alpha(clip_rect.w + 10 * u, clip_rect.h + 10 * u, r_tile + 5 * u, 0, 0,
                     (int(clip_rect.w + 10 * u) + 1, int(clip_rect.h + 10 * u) + 1))
    inner = _rr_alpha(clip_rect.w + 2 * u, clip_rect.h + 2 * u, r_tile + 1 * u, 4 * u, 4 * u, ring.shape[::-1])
    ring = np.clip(ring - inner, 0, 1)
    lay = Layout(W, H, clip_rect, r_tile, canvas)
    start = at + 0.12

    def draw_tiles(c: np.ndarray, t: float) -> None:
        for idx, sp, k in sprites:
            q = (t - start - 0.045 * k) / 0.32
            if q <= 0:
                continue
            s = 0.86 + 0.14 * _back_out(q, 1.4)
            a = _ramp(q * 1.6)
            rect = tiles[idx]
            ss = sp.scaled(s) if abs(s - 1) > 2e-3 else sp
            _over_f(c, ss, rect.cx - ss.w / 2, rect.cy - ss.h / 2, a)
        ra = _ramp((t - land) / 0.3)
        if ra > 0:
            _lay_alpha(c, ring, (255, 255, 255), int(clip_rect.x0 - 5 * u), int(clip_rect.y0 - 5 * u), 0.85 * ra)

    lay.parts.append(Part(draw_tiles, max(land + 0.3, start + 0.045 * 8 + 0.4)))
    lay.others = about
    if name or caption:
        d = int(84 * u)
        av = avatar(clip, fr, at, d) or initial(name or "•", d, (60, 64, 72))
        hx, hy = zone.x0, zone.y0 + (head_h - d) / 2 - 10 * u
        tx = hx + d + 22 * u
        avail = zone.x1 - tx

        def draw_head(c, t, av=av):
            a = _ramp((t - land + 0.1) / 0.3)
            _over_f(c, av, hx, hy, a)
            y = hy + d / 2
            if name and caption:
                npx, cpx = int(38 * u), int(29 * u)
                _lay_words(c, wrap(name, "poppins-bold", npx, avail, 1), "poppins-bold", npx, (255, 255, 255), tx,
                           y - 8 * u, 0, a)
                _lay_words(c, wrap(caption, "poppins-semibold", cpx, avail, 1), "poppins-semibold", cpx,
                           (215, 218, 224), tx, y + 34 * u, 0, a)
            else:
                npx = int(38 * u)
                text = name or caption
                _lay_words(c, wrap(text, "poppins-bold", npx, avail, 1), "poppins-bold", npx, (255, 255, 255), tx,
                           y + textmod.cap_height("poppins-bold", npx) / 2, 0, a)

        lay.parts.append(Part(draw_head, land + 0.3))
        lay.words.append(Rect(hx, hy, zone.x1, hy + d))
    if likes or comments:
        lay.notes.append("the grid shows no counts (likes and comments are for card and phone)")
    return lay


# ---- the camera ----------------------------------------------------------------------------------

def _affine(win, W: int) -> tuple[float, float, float]:
    """The camera's window as a scale and an offset: a layout point p lands at (p - (x, y)) * s."""
    x, y, w, h = win
    return W / w, x, y


def _place_clip(out: np.ndarray, frame: np.ndarray, R: tuple[float, float, float, float], radius: float,
                cache: dict) -> None:
    """The clip's own frame laid into the rectangle R (x, y, w, h, on screen) with round corners:
    made smaller from its own pixels (never enlarged from a small copy)."""
    import cv2

    H, W = out.shape[:2]
    x, y, w, h = R
    k = w / frame.shape[1]
    if k <= 1.0:
        iw, ih = max(1, int(round(w))), max(1, int(round(h)))
        ix, iy = int(round(x)), int(round(y))
        pic = cv2.resize(frame, (iw, ih), interpolation=cv2.INTER_AREA)
        key = (iw, ih, round(radius, 1))
        if key not in cache:
            cache.clear() if len(cache) > 64 else None
            cache[key] = _rr_alpha(iw, ih, radius, 0, 0, (iw, ih))[..., None]
        m = cache[key]
        x0, y0, x1, y1 = max(0, ix), max(0, iy), min(W, ix + iw), min(H, iy + ih)
        if x1 <= x0 or y1 <= y0:
            return
        p = pic[y0 - iy:y1 - iy, x0 - ix:x1 - ix].astype(np.float32)
        mm = m[y0 - iy:y1 - iy, x0 - ix:x1 - ix]
        reg = out[y0:y1, x0:x1].astype(np.float32)
        out[y0:y1, x0:x1] = np.clip(reg * (1 - mm) + p * mm + 0.5, 0, 255).astype(np.uint8)
        return
    # larger than the frame: the picture is the clip, cropped, with corners mostly off screen
    mat = np.float32([[k, 0, x], [0, k, y]])
    pic = cv2.warpAffine(frame, mat, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    m = _rr_alpha(w, h, radius, x, y, (W, H))[..., None]
    if m.min() >= 0.999:
        out[:] = pic
        return
    out[:] = np.clip(out.astype(np.float32) * (1 - m) + pic.astype(np.float32) * m + 0.5, 0, 255).astype(np.uint8)


# ---- when ----------------------------------------------------------------------------------------

def resolve_at(clip, value: str, duration: float) -> tuple[float, str]:
    """When it pulls back, and why."""
    from ..core import moments

    v = str(value).strip()
    low = v.lower()
    if low in ("start", "0", "0.0"):
        return 0.0, "from the start"
    try:
        t = float(low.rstrip("s"))
        if not 0 <= t < duration:
            raise O.OptionError(f"at: {t:g} s is outside the clip's {duration:.2f} s")
        return t, f"at {t:.2f} s, as given"
    except ValueError:
        pass
    words = clip.words
    if low == "auto":
        hi = max(0.8, min(0.5 * duration, duration - 2.5))
        best = None
        for s in moments.strong_starts(words, clip.quiet):
            if 0.8 <= s.at <= hi and s.strength is not None and (best is None or s.strength > best.strength):
                best = s
        if best is not None:
            line = " ".join(moments.shown(w) for w in words[best.line[0]:best.line[1] + 1]) if best.line else ""
            return max(0.0, best.at - LEAD), f"on “{line}”" if line else f"at {best.at:.2f} s"
        t = min(1.0, duration * 0.2)
        return t, f"at {t:.2f} s (no strong line in the first half)"
    from .headline import find_said

    found = find_said(words, v)
    if found is None:
        raise O.OptionError(f"at: “{v}” is not said in this clip; give a time in seconds, start, or auto")
    i = found[0]
    return max(0.0, words[i].start - LEAD), f"on “{v}”"


# ---- sound ---------------------------------------------------------------------------------------

class Sounds:
    def __init__(self, clip, cues: list[tuple[float, str, float, float]]):
        voice = soundmod.integrated(soundmod.blocks(clip.sound)) if clip.info.has_audio else soundmod.SILENCE_DB
        self.ref = voice if voice > -70 else -20.0
        self.cues = cues  # (at, kind, pitch, how far under the voice)

    def mix(self, base: np.ndarray, rate: int) -> np.ndarray:
        from ..core import loudness

        stem = np.zeros_like(base)
        for at, kind, pitch, under in self.cues:
            x, anchor = soundmod.library(kind)
            if abs(pitch - 1.0) > 1e-3:
                x, anchor = soundmod.resample(x, pitch), anchor / pitch
            soundmod.lay(stem, synth.at_level(x, self.ref - under), at, rate, anchor=anchor)
        out = base + stem
        ceiling = max(-1.0, loudness.true_peak_db(base) + 0.05)
        if loudness.true_peak_db(out) > ceiling:
            out, _ = loudness.limit(out, ceiling)
        return out.astype(np.float32)


# ---- the effect ----------------------------------------------------------------------------------

def lay_captions(clip, opts, fr, where: Callable[[float], tuple[float, float, float, float]], fps: float):
    """The pack's captions, prepared on the picture as the frame shows it: the face where the frame
    puts it, so the words keep clear of it."""
    from ..core import framing as framing_mod
    from ..core.view import View
    from . import layers

    W, H = fr.out_w, fr.out_h

    def place(i: int):
        x, y, w, h = where(i / fps)
        return fr.window(i), Rect(x, y, x + w, y + h)

    view = View(clip, (W, H), place=place)
    base = framing_mod.plan(W, H, fps, clip.info.frames, None, shape="keep")
    given = {} if opts["captions"] == "on" else {"look": opts["captions"]}
    try:
        return layers.prepare(view, layers.Spec("captions", given), None, None, base, fix=opts["fix"])
    except ValueError as e:
        raise O.OptionError(f"captions: {e}") from None


def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from ..core import framing as framing_mod
    from ..core.timeline import Timeline
    from . import Job, PlanError

    info = clip.info
    fps = info.rate
    fr = framing_mod.plan(info.width, info.height, fps, info.frames, None, shape="keep")
    W, H = fr.out_w, fr.out_h
    duration = info.frames / fps
    tl = Timeline.whole(duration)
    look = opts["look"]
    if plan is not None:
        try:
            at = float(plan["at"])
        except (KeyError, TypeError, ValueError):
            raise PlanError("the plan needs `at`, the second it pulls back") from None
        if not 0 <= at < duration:
            raise PlanError(f"the plan's at ({at:g} s) is outside the clip's {duration:.2f} s")
        why = "from the plan"
    else:
        at, why = resolve_at(clip, opts["at"], duration)
    move = MOVE[look]
    land = at + move
    lay = build_layout(look, clip, fr, opts, at, land)
    R = lay.clip_rect
    start_win = (R.x0, R.y0, R.w, R.h)
    full = (0.0, 0.0, float(W), float(H))
    moves = [zoom.Move(0.0, 0.0, start_win, "cut"), zoom.Move(at, land, full, KIND[look])]
    back_at = None
    if opts["duration"] > 0 and at + opts["duration"] < duration - 0.1:
        back_at = max(land + 0.3, at + opts["duration"])
        moves.append(zoom.Move(back_at, min(duration, back_at + BACK), start_win, "smooth"))
    path = zoom.CameraPath((W, H), moves)

    def clip_on_screen(t: float) -> tuple[float, float, float, float]:
        s, x, y = _affine(path.at(t), W)
        return ((R.x0 - x) * s, (R.y0 - y) * s, R.w * s, R.h * s)

    settled = max([land] + [p.settled for p in lay.parts])
    cap_layer = lay_captions(clip, opts, fr, clip_on_screen, fps) if opts["captions"] != "off" else None
    still_layout: dict = {}
    mask_cache: dict = {}

    def layout_at(t: float) -> np.ndarray:
        if t >= settled and (back_at is None or t < back_at):
            if "final" not in still_layout:
                c = lay.ground.copy()
                for p in lay.parts:
                    p.draw(c, settled + 5.0)
                still_layout["final"] = np.clip(c + 0.5, 0, 255).astype(np.uint8)
            return still_layout["final"]
        c = lay.ground.copy()
        for p in lay.parts:
            p.draw(c, t)
        return np.clip(c + 0.5, 0, 255).astype(np.uint8)

    def draw(frame: np.ndarray, j: int, t: float) -> None:
        import cv2

        win = path.at(t)
        if max(abs(a - b) for a, b in zip(win, start_win)) < 0.25:
            if cap_layer is not None:
                cap_layer.draw(frame, j, t)
            return  # the clip fills the frame, as shot
        src = frame.copy()
        s, x, y = _affine(win, W)
        scene = layout_at(t)
        if abs(s - 1.0) < 1e-4 and abs(x) < 0.01 and abs(y) < 0.01:
            frame[:] = scene
        else:
            mat = np.float32([[s, 0, -x * s], [0, s, -y * s]])
            frame[:] = cv2.warpAffine(scene, mat, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        _place_clip(frame, src, clip_on_screen(t), lay.radius * s, mask_cache)
        if cap_layer is not None:
            cap_layer.draw(frame, j, t)

    cues = []
    if opts["sound"]:
        cues.append((at + 0.05, "whoosh", 1.0, UNDER_VOICE))
        if look == "fan":
            cues += [(land - 0.25, "swoosh-in", 1.1, UNDER_VOICE + 6), (land - 0.18, "swoosh-in", 1.25, UNDER_VOICE + 7)]
        if look == "grid":
            cues += [(at + 0.12 + 0.045 * k + 0.12, "tick", 2 ** (k / 12), UNDER_VOICE + 9) for k in range(8)]
        if back_at is not None:
            cues.append((back_at, "swoosh-in", 0.9, UNDER_VOICE + 3))
    mixer = Sounds(clip, cues) if cues and info.has_audio else None

    # checks: the words inside the safe zone, the clip whole on screen once it has landed
    zone = SafeZone(W, H).rect
    outside = [r.as_list() for r in lay.words if not r.inside(zone, slack=1.0)]
    sheet = [max(0.0, at - 0.3), at + move * 0.45, land + 0.35, min(duration - 0.05, settled + 0.6),
             min(duration - 0.05, (settled + duration) / 2), max(0.0, duration - 0.15)]
    if back_at is not None:
        sheet[4] = min(duration - 0.05, back_at + BACK * 0.5)
    out_plan = {"effect": NAME, "clip": clip.path, "options": opts, "at": round(at, 3), "why": why,
                "land": round(land, 3), "back": round(back_at, 3) if back_at is not None else None,
                "output": {"width": W, "height": H, "fps": round(fps, 3), "seconds": round(duration, 3), "framing": fr.note},
                "clip_rect": R.as_list(), "scale": round(R.w / W, 3), "words": [r.as_list() for r in lay.words],
                "others": lay.others, "sheet_times": [round(v, 3) for v in sheet],
                "sounds": [{"at": round(a, 3), "kind": k} for a, k, _, _ in cues]}
    checks = {"look": look, "pulls back": f"{at:.2f} s ({why}), framed by {land:.2f} s",
              "frame": f"the clip at {R.w / W:.0%} of the width",
              "words": ("none given" if not lay.words else
                        ("inside the safe zone" if not outside else f"{len(outside)} outside the safe zone"))}
    if lay.others:
        checks["others"] = lay.others
    if back_at is not None:
        checks["back"] = f"fills the screen again from {back_at:.2f} s"
    if cap_layer is not None:
        checks["captions"] = f"{len(cap_layer.job.plan.get('cards') or [])} cards over the frame"
    if lay.notes:
        checks["notes"] = "; ".join(lay.notes)
    warnings = []
    if outside:
        warnings.append("some words reach outside the safe zone")
    if at > duration - move - 0.5:
        warnings.append("it pulls back too close to the end to be seen")
    checks["warnings"] = warnings
    still = [sheet[0], sheet[1], sheet[3], sheet[4]]
    return Job(NAME, clip, tl, fr, draw, out_plan, checks=checks, still_times=lambda: still,
               sound=mixer.mix if mixer is not None else None)
