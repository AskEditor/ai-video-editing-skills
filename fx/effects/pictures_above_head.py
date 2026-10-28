"""pictures-above-head: your own pictures, logos and proof pop in above or beside your head as you name
each one.

How it decides, in order:

1. **Which picture goes on which word: by name and tag, never by looking.** `map` (a word said ->
   a picture) first. Then the words in each picture's file name (`town.jpg`) and in a `tags.txt`
   beside it, matched to the words said, a plural matching its singular. A logo lands only where
   its brand's whole name is said: `AskEditor.png` on "AskEditor", never on "ask" alone. A picture
   named only with a camera's number (`IMG_5922.JPG`) goes, in folder order, on the strongest
   things you name. A picture whose name and tags are never said is left out and listed. (The
   matching is the core's, `fx/core/match.py`, shared with `broll-cutaways`.)
2. **When.** Each picture lands on the first frame of its word, timed from when the voice begins
   (the transcriber marks words late; the core pulls each start back and checks it against the
   sound). It stays while the next ones land, and the pictures leave together `hold` seconds after
   the last (by default once the sentence has ended), or when a picture named later needs the room.
3. **Where.** Measured on the head over every frame the pictures are up, inside the safe zone:
   an arc of tiles above the head, a cluster above it, a column on the free side, or one picture at
   a time. `layout=auto` chooses by the room the shot leaves and how many pictures share a moment;
   on a close-up where nothing readable fits above or beside the head, they sit under the chin,
   and the run says so. Never over the face (hairline to chin, ear to ear); the top of the hair
   only when that buys a much larger picture.
4. **Framed, never squashed.** A photo is cropped to its subject (a face it holds, or where its
   detail is). A screenshot lands whole, then pushes in to its busiest part when its writing would
   be too small to read on a phone. A page far taller than its tile scrolls slowly down to its
   point. A logo is shown whole, trimmed to its own edges, on a plate of its own ground (or white,
   or dark, whichever it reads on), never recoloured. Every tile is rounded, with a soft shadow.
5. **Sound.** A soft pop as each picture lands, on the same frame, well under the voice
   (`sound=off` keeps the clip's sound exactly).

This pack ships no brand marks. A logo is only ever a file the person gives.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..core import match, moments
from ..core.compose import Sprite, over
from ..core.layout import Rect, SafeZone
from ..core.paths import CACHE
from ..core.words import Word
from . import options as O

NAME = "pictures-above-head"
LAYOUTS = ("auto", "arc", "cluster", "column", "rotate")
STYLES = ("clean", "print", "glass")
SIZES = ("auto", "small", "medium", "large")
PICTURE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
KIND = match.Kind(frozenset(PICTURE_EXT), "pictures", "picture", "a picture this pack reads (jpg, png, webp)",
                  "your pictures", "give your pictures: pictures=\"~/Pictures/proof\" (a folder) or files separated by "
                                   "commas")


# ---- the options -------------------------------------------------------------------------------

@dataclass(frozen=True)
class _Files:
    help: str

    def parse(self, raw) -> str:
        return str(raw).strip()


@dataclass(frozen=True)
class _Map:
    help: str

    def parse(self, raw) -> str:
        return "; ".join(f"{w}={c}" for w, c in parse_map(str(raw)))


@dataclass(frozen=True)
class _On:
    help: str

    def parse(self, raw) -> str:
        v = str(raw).strip().strip("\"'").lower()
        if v in ("", "auto", "keywords", "keyword", "words"):
            return "keywords"
        if v in ("map", "names", "tags", "named"):
            return "map"
        if v.startswith("times"):
            v = v[5:].lstrip(" =:")
        parts = [p for p in re.split(r"[,;\s]+", v) if p]
        try:
            times = [float(p.rstrip("s")) for p in parts]
        except ValueError:
            raise O.OptionError(f"say keywords, map, or seconds such as 2.5, 6, 11 (got {raw!r})") from None
        if not times:
            raise O.OptionError("say keywords, map, or seconds such as 2.5, 6, 11")
        if any(t < 0 for t in times):
            raise O.OptionError("times start at 0 seconds")
        return ", ".join(f"{t:g}" for t in sorted(times))


@dataclass(frozen=True)
class _Hold:
    help: str

    def parse(self, raw) -> str:
        v = str(raw).strip().lower().rstrip("s")
        if v in ("", "auto"):
            return "auto"
        if v in ("end", "to the end"):
            return "end"
        try:
            d = float(v)
        except ValueError:
            raise O.OptionError(f"give auto, end, or seconds from 0.8 to 12 (got {raw!r})") from None
        if not 0.8 <= d <= 12.0:
            raise O.OptionError(f"give seconds from 0.8 to 12 (got {raw!r})")
        return f"{d:g}"


def parse_map(text: str) -> list[tuple[str, str]]:
    """`town=town.jpg; my hotel=room.jpg` or `{"town": "town.jpg"}` -> [("town", "town.jpg"), …]."""
    try:
        return match.parse_map(text, noun="picture", examples=(("town", "town.jpg"), ("my hotel", "room.jpg")))
    except ValueError as e:
        raise O.OptionError(str(e)) from None


OPTIONS = {
    "pictures": _Files("your pictures, logos or screenshots: a folder, or files separated by commas. A picture "
                       "named for what it shows (town.jpg), or listed in a tags.txt beside it, lands as you say "
                       "it; a logo named for its brand (AskEditor.png) lands as you say the brand"),
    "map": _Map("which picture goes on which word you say: word=picture pairs separated by ; (town=town.jpg; "
                "my hotel=room.jpg) or JSON ({\"town\": \"town.jpg\"}); a bare name is looked for among your "
                "pictures"),
    "layout": O.Choice(LAYOUTS, "auto: chosen by the room round your head and how many pictures share a moment · "
                                "arc: a row of tiles curving over your head · cluster: a loose pile above your "
                                "head · column: stacked on the free side of you · rotate: one at a time, each "
                                "replacing the last"),
    "on": _On("keywords: pictures land on the words that name them, and pictures with only a camera's number "
              "for a name on the strongest things you say, in folder order · map: only on words named by map, "
              "a file name or tags.txt · or seconds, such as 2.5, 6, 11, for the pictures in folder order"),
    "hold": _Hold("auto: the pictures stay until the sentence that names the last of them has ended (about 2 "
                  "to 4 seconds) · seconds such as 3 · or end: they stay to the end"),
    "size": O.Choice(SIZES, "auto or medium: as large as the room allows · small · large (still never over "
                            "your face, never outside the safe zone)"),
    "style": O.Choice(STYLES, "clean: rounded, with a soft shadow · print: a white border, like a printed "
                              "photo · glass: each picture on a pane of frosted glass"),
    "sound": O.Switch("on: a soft pop as each picture lands, well under your voice · off: your sound exactly as "
                      "it was"),
    "fix": O.Text("misheard words as wrong=right, separated by ;  e.g. \"Jon=John; ask editor=AskEditor\""),
}

# The runs `tests/matrix.py pictures-above-head` makes, on our own photos, stills from his
# B-roll, and AskEditor's own screens and logo (links in the lab's inputs; its README says which).
_IN = "{lab}/pictures-above-head/inputs"
MATRIX = [
    # room above the head: three product screens on "video", "production quality" and "editor" (tags.txt)
    ("B", "B", {"pictures": f"{_IN}/B-screens"}),
    ("B-cluster", "B", {"pictures": f"{_IN}/B-screens", "layout": "cluster", "style": "print"}),
    # town, city and hotel, by file name; a lake never said is left out
    ("D", "D", {"pictures": f"{_IN}/D-places"}),
    ("D-rotate", "D", {"pictures": f"{_IN}/D-places", "layout": "rotate", "style": "glass"}),
    # the tight close-up: screens saved with a Mac's screenshot names, on the strongest things named
    ("A", "A", {"pictures": f"{_IN}/A-stills"}),
    # the hard clip: AskEditor's logo on "product", a screen on "videos"
    ("C", "C", {"pictures": f"{_IN}/C-product"}),
    # 44 s of real talk: the logo and four screens on its product words
    ("33", "33", {"pictures": f"{_IN}/33-product"}),
]

# ---- the rules (numbers chosen for this pack) ------------------------------------------------------

ENTER = 0.28  # seconds a picture takes to pop in
EXIT = 0.24  # and to leave
POP_FROM = 0.84  # the scale it pops in from
POP_OPACITY = 0.72  # how solid it is on its first frame: it is there on the frame its word starts
APART = 0.25  # two pictures never land closer together than this
MIN_ON = 0.9  # a picture is up at least this long: one named later than this before the end is left out
HOLD_AUTO = (2.0, 4.2)  # hold=auto: after the last picture lands, at least and at most this long
SNAP = 0.3  # a time given in `on` within this of a word's start moves onto it
BRIDGE = 1.0  # pictures due to leave less than this before the next lands stay until it does
GAP = 0.02  # between a picture and the head, as a share of the frame's height
GUTTER = 0.018  # between two pictures, as a share of the frame's width
READ = 0.17  # a tile smaller than a square this share of the frame's width is small on a phone
SMALLEST = 0.10  # and none is drawn narrower than this
SIZE_K = {"small": 0.75, "auto": 1.0, "medium": 1.0, "large": 1.3}
TEXT_PX = 30.0  # writing in a screenshot reads on a phone at about this many pixels on a 1080-wide frame
POP_UNDER = 12.0  # dB the pop's loudest tenth of a second sits under the voice
MAX_GROUP = {"arc": 5, "cluster": 6, "column": 4, "rotate": 99, "row": 4}

# ---- the pictures --------------------------------------------------------------------------------


@dataclass
class Picture:
    path: str
    kind: str  # "photo", "screen", "page" or "graphic"
    rgb: np.ndarray  # upright, at most about 2000 pixels on its long side (a page keeps its length)
    alpha: Optional[np.ndarray]  # 0..1 where it has its own transparency, else None
    width: int  # the file's own size
    height: int
    names: tuple = ()  # the words of its file name (a logo: its brand)
    tags: tuple = ()  # its phrases in tags.txt
    logo: bool = False  # lands only on its brand's whole name
    ground: tuple = (255, 255, 255)  # a graphic's own ground: what its edges are
    focus: tuple = (0.5, 0.45)  # a photo's subject, as shares across and down
    faces: int = 0  # faces found in a photo
    busy: Optional[tuple] = None  # a screen's busiest part: (cx, cy) as shares
    page_point: float = 0.0  # a page's point: how far down (share) its busiest part sits
    text_px: float = 15.0  # a screen's writing: about how tall, in its own pixels

    @property
    def label(self) -> str:
        return Path(self.path).name

    @property
    def named(self) -> bool:
        return bool(self.names or self.tags)

    @property
    def whole(self) -> bool:
        """A logo is matched on its brand's whole name (`match.candidates` reads this)."""
        return self.logo

    @property
    def aspect(self) -> float:
        h, w = self.rgb.shape[:2]
        return w / h


_LOGO_HINT = re.compile(r"(?i)(^|[^a-z])(logos?|logotypes?|wordmarks?)([^a-z]|$)")  # in a file's name or its folder's
MAX_SIDE = 2000


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def measure(rgb: np.ndarray, alpha: Optional[np.ndarray]) -> dict:
    """What a picture is like, from its pixels: how much of it is perfectly flat (a screen or a graphic;
    a photo is never flat for long), how uniform its edge is, how few colours it has, and how much of
    it is see-through."""
    import cv2

    h, w = rgb.shape[:2]
    k = min(1.0, 900 / max(h, w))
    small = cv2.resize(rgb, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_NEAREST)
    g = _luma(small.astype(np.float32))
    m, m2 = cv2.blur(g, (3, 3)), cv2.blur(g * g, (3, 3))
    flat = float((np.sqrt(np.maximum(0.0, m2 - m * m)) < 0.5).mean())
    s = small.astype(np.int32)
    q = (s // 8)
    colours = len(np.unique(q[..., 0] * 1024 + q[..., 1] * 32 + q[..., 2]))
    edge = np.concatenate([s[0], s[-1], s[:, 0], s[:, -1]])
    mode = np.median(edge, axis=0)
    uniform = float((np.abs(edge - mode).max(axis=1) < 10).mean())
    clear = float((alpha < 0.5).mean()) if alpha is not None else 0.0
    return {"flat": flat, "colours": colours, "edge_uniform": uniform, "edge": tuple(int(v) for v in mode),
            "clear": clear}


def kind_of(path: Path, m: dict, aspect_hw: float) -> str:
    """photo, screen, page or graphic (a logo, an icon, a cut-out: shown whole on a plate)."""
    if m["clear"] >= 0.12 or (m["flat"] >= 0.85 and m["edge_uniform"] >= 0.9 and m["colours"] <= 400):
        return "graphic"
    if m["flat"] >= 0.5:
        return "page" if aspect_hw >= 1.75 else "screen"
    return "photo"


def _faces_in(rgb: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Faces in a photo as (x, y, w, h) shares, largest first (the core's face model, on a still)."""
    from ..core.face import quiet_native
    from ..core.models import vision_path

    try:
        with quiet_native():
            import mediapipe as mp
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision

            h, w = rgb.shape[:2]
            k = min(1.0, 640 / max(h, w))
            import cv2

            small = np.ascontiguousarray(cv2.resize(rgb, (max(1, int(w * k)), max(1, int(h * k))),
                                                    interpolation=cv2.INTER_AREA))
            options = vision.FaceDetectorOptions(base_options=mp_tasks.BaseOptions(model_asset_path=str(vision_path("face"))),
                                                 running_mode=vision.RunningMode.IMAGE, min_detection_confidence=0.6)
            with vision.FaceDetector.create_from_options(options) as det:
                res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=small))
    except Exception:  # no face model, or nothing it can read: framed by its detail instead
        return []
    sh, sw = small.shape[:2]
    out = []
    for d in res.detections:
        b = d.bounding_box
        out.append((b.origin_x / sw, b.origin_y / sh, b.width / sw, b.height / sh))
    return sorted(out, key=lambda f: -f[2] * f[3])


def detail_map(rgb: np.ndarray, size: int = 256) -> np.ndarray:
    """Where a picture has something to look at: its edges and the colours that stand out from the
    rest, as a small map from 0 to 1."""
    import cv2

    h, w = rgb.shape[:2]
    k = size / max(h, w)
    small = cv2.resize(rgb, (max(2, int(w * k)), max(2, int(h * k))), interpolation=cv2.INTER_AREA).astype(np.float32)
    g = _luma(small)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.sqrt(gx * gx + gy * gy)
    lab = cv2.cvtColor(np.clip(small, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    stand = np.linalg.norm(lab - lab.reshape(-1, 3).mean(axis=0), axis=2)

    def norm(a: np.ndarray) -> np.ndarray:
        p = float(np.percentile(a, 99))
        return np.clip(a / p, 0, 1) if p > 1e-6 else np.zeros_like(a)

    m = 0.65 * norm(edges) + 0.35 * norm(stand)
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.04 * max(m.shape))


def _busiest(imap: np.ndarray, share_w: float, share_h: float) -> tuple[float, float]:
    """The middle (shares) of the window `share_w` x `share_h` of the picture holding the most of the
    map, leaning a little toward the top and the middle, where a page puts what matters."""
    h, w = imap.shape
    ww, wh = max(1, int(round(share_w * w))), max(1, int(round(share_h * h)))
    if ww >= w and wh >= h:
        return 0.5, 0.5
    ys = (np.arange(h) + 0.5) / h
    xs = (np.arange(w) + 0.5) / w
    lean = (0.75 + 0.25 * np.exp(-0.5 * ((xs[None, :] - 0.5) / 0.3) ** 2)) * (1.0 - 0.25 * ys[:, None])
    I = np.pad(np.cumsum(np.cumsum(imap * lean, axis=0), axis=1), ((1, 0), (1, 0)))
    ww, wh = min(ww, w), min(wh, h)
    sums = I[wh:, ww:] - I[:-wh, ww:] - I[wh:, :-ww] + I[:-wh, :-ww]
    y, x = np.unravel_index(int(np.argmax(sums)), sums.shape)
    return (x + ww / 2) / w, (y + wh / 2) / h


def _trim(rgb: np.ndarray, alpha: Optional[np.ndarray], ground: tuple) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """A graphic cut to its own drawn edges: what is see-through, or the colour of its edge, taken off
    all round (a hair of it kept)."""
    if alpha is not None:
        ink = alpha > 0.04
    else:
        ink = np.abs(rgb.astype(np.int32) - np.array(ground, np.int32)).max(axis=2) > 14
    if not ink.any():
        return rgb, alpha
    ys, xs = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
    h, w = ink.shape
    pad = max(1, int(0.01 * max(h, w)))
    y0, y1 = max(0, ys[0] - pad), min(h, ys[-1] + 1 + pad)
    x0, x1 = max(0, xs[0] - pad), min(w, xs[-1] + 1 + pad)
    return rgb[y0:y1, x0:x1], (alpha[y0:y1, x0:x1] if alpha is not None else None)


def load_picture(path: Path, logo_hint: bool = False) -> Picture:
    """A picture read upright, what kind it is, and what to frame it on."""
    from PIL import Image, ImageOps

    import cv2

    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            has_alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
            if has_alpha:
                im = im.convert("RGBA")
                arr = np.asarray(im)
                rgb, alpha = np.ascontiguousarray(arr[..., :3]), arr[..., 3].astype(np.float32) / 255.0
                if alpha.min() > 0.99:
                    alpha = None
            else:
                rgb, alpha = np.asarray(im.convert("RGB")), None
    except OSError:
        raise ValueError(f"{path.name} could not be read as a picture") from None
    H0, W0 = rgb.shape[:2]
    m = measure(rgb, alpha)
    kind = kind_of(path, m, H0 / W0)
    # Kept no larger than it is ever drawn; a page keeps its length, at a width a tile can use.
    k = min(1.0, (1400 / W0) if kind == "page" else MAX_SIDE / max(H0, W0))
    if k < 1.0:
        size = (max(1, int(round(W0 * k))), max(1, int(round(H0 * k))))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        if alpha is not None:
            alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
    hinted = logo_hint or bool(_LOGO_HINT.search(path.stem))
    logo = (hinted or kind == "graphic") and bool(match.brand_words(path))
    if hinted and kind != "graphic":
        kind = "graphic"  # a logo is shown whole, whatever it looks like
    pic = Picture(str(path), kind, np.ascontiguousarray(rgb), alpha, W0, H0, logo=logo,
                  tags=match.tags_for(path))
    pic.names = match.brand_words(path) if logo else match.name_words(path)
    if kind == "graphic":
        pic.ground = m["edge"] if alpha is None else (255, 255, 255)
        pic.rgb, pic.alpha = _trim(pic.rgb, pic.alpha, pic.ground)
    elif kind == "photo":
        faces = _faces_in(pic.rgb)
        pic.faces = len(faces)
        if faces:
            x, y, w, h = faces[0]
            pic.focus = (x + w / 2, y + h * 0.55)
        else:
            imap = detail_map(pic.rgb)
            hh, ww = imap.shape
            I = imap / max(1e-6, imap.sum())
            cx = float((I.sum(axis=0) * (np.arange(ww) + 0.5) / ww).sum())
            cy = float((I.sum(axis=1) * (np.arange(hh) + 0.5) / hh).sum())
            pic.focus = (0.5 + 0.6 * (cx - 0.5), 0.47 + 0.6 * (cy - 0.5))
    else:  # a screen or a page
        h, w = pic.rgb.shape[:2]
        portrait = h > w
        # writing on a phone's screen is about 45 of its pixels at 1170 wide; on a laptop's, 15 at 1440
        pic.text_px = (45.0 * W0 / 1170.0) if portrait else (15.0 * W0 / 1440.0)
        pic.text_px *= w / W0  # in the pixels kept
        imap = detail_map(pic.rgb, size=384)
        if kind == "page":
            share_h = min(1.0, (w / 0.8) / h)
            _, cy = _busiest(imap, 1.0, share_h)
            pic.page_point = float(cy)
            pic.busy = (0.5, float(cy))
        else:
            pic.busy = _busiest(imap, 0.34, 0.34)
    return pic


# ---- framing a picture into its tile ---------------------------------------------------------------

def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    import cv2

    ih, iw = img.shape[:2]
    interp = cv2.INTER_AREA if (iw > w or ih > h) else cv2.INTER_CUBIC
    return cv2.resize(img, (max(1, w), max(1, h)), interpolation=interp)


def cover_window(pic: Picture, aspect: float) -> tuple[float, float, float, float]:
    """The largest window of `aspect` (w/h) inside the picture, centred on its subject: (x, y, w, h)
    in its pixels. A face is kept whole, with room over it."""
    h, w = pic.rgb.shape[:2]
    if w / h > aspect:
        ww, wh = h * aspect, float(h)
    else:
        ww, wh = float(w), w / aspect
    fx, fy = pic.focus
    x = min(max(fx * w - ww / 2, 0.0), w - ww)
    y = min(max(fy * h - wh / 2, 0.0), h - wh)
    return x, y, ww, wh


def screen_windows(pic: Picture, aspect: float, tile_w: float, frame_w: float) -> tuple[tuple, tuple]:
    """A screenshot in a tile of `aspect`: (the whole of it, fitted) and (the part that reads, around its
    busiest part), each (x, y, w, h) in its pixels. The two are the same when the whole already reads."""
    h, w = pic.rgb.shape[:2]
    if w / h > aspect:
        whole = (0.0, (h - w / aspect) / 2, float(w), w / aspect)  # wider than the tile: bands above and below
    else:
        whole = ((w - h * aspect) / 2, 0.0, h * aspect, float(h))
    need = TEXT_PX * (frame_w / 1080.0) / max(1e-6, pic.text_px)  # the scale its writing reads at
    scale_whole = tile_w / whole[2]
    if scale_whole >= need * 0.92:
        return whole, whole
    ww = min(float(w), tile_w / need)
    ww = max(ww, whole[2] / 3.5)  # never further in than three and a half times
    wh = ww / aspect
    if wh > h:
        wh = float(h)
        ww = wh * aspect
    cx, cy = pic.busy or (0.5, 0.5)
    x = min(max(cx * w - ww / 2, 0.0), w - ww)
    y = min(max(cy * h - wh / 2, 0.0), h - wh)
    return whole, (x, y, ww, wh)


def window_image(pic: Picture, win: tuple, w: int, h: int, ground: Optional[tuple] = None) -> np.ndarray:
    """The part `win` (x, y, w, h, possibly reaching past the picture's edges) of a picture at w x h, the
    edges filled with `ground` (or the picture's own edge colour)."""
    import cv2

    x, y, ww, wh = win
    sx, sy = w / ww, h / wh
    m = np.float32([[sx, 0, -x * sx], [0, sy, -y * sy]])
    src = pic.rgb
    if ground is None:
        edge = np.concatenate([src[0], src[-1], src[:, 0], src[:, -1]])
        ground = tuple(int(v) for v in np.median(edge, axis=0))
    interp = cv2.INTER_AREA if sx < 1 else cv2.INTER_CUBIC
    if sx < 0.5:  # far down: shrink first, so nothing aliases
        k = max(1, int(1 / sx))
        small = cv2.resize(src, (max(1, src.shape[1] // k), max(1, src.shape[0] // k)), interpolation=cv2.INTER_AREA)
        m = np.float32([[sx * k, 0, -x * sx], [0, sy * k, -y * sy]])
        src, interp = small, cv2.INTER_LINEAR
    return cv2.warpAffine(src, m, (w, h), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=ground)


def graphic_image(pic: Picture, w: int, h: int, pad: float = 0.16) -> tuple[np.ndarray, tuple]:
    """A graphic whole, on its plate: fitted inside w x h with `pad` of clear space, on its own ground
    (a see-through logo: white, or dark when its ink is light). Never recoloured. Returns the RGB tile
    and the plate's colour."""
    gh, gw = pic.rgb.shape[:2]
    if pic.alpha is not None:
        a = pic.alpha
        ink = _luma(pic.rgb.astype(np.float32))
        mean_ink = float((ink * a).sum() / max(1e-6, a.sum()))
        plate = (255, 255, 255) if mean_ink < 168 else (22, 24, 30)
    else:
        plate = tuple(int(v) for v in pic.ground)
    inner_w, inner_h = w * (1 - 2 * pad), h * (1 - 2 * pad)
    s = min(inner_w / gw, inner_h / gh)
    dw, dh = max(1, int(round(gw * s))), max(1, int(round(gh * s)))
    img = _resize(pic.rgb, dw, dh).astype(np.float32)
    out = np.empty((h, w, 3), np.float32)
    out[:] = plate
    x0, y0 = (w - dw) // 2, (h - dh) // 2
    if pic.alpha is not None:
        a = _resize(pic.alpha, dw, dh)[..., None]
        out[y0:y0 + dh, x0:x0 + dw] = out[y0:y0 + dh, x0:x0 + dw] * (1 - a) + img * a
    else:
        out[y0:y0 + dh, x0:x0 + dw] = img
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), plate


# ---- the card: a framed picture, drawn once (or each frame while its picture moves) -----------------

def _round_mask(w: int, h: int, r: float) -> np.ndarray:
    """A w x h rounded rectangle's coverage (0..1), its edge smooth."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xs += 0.5
    ys += 0.5
    r = float(max(0.0, min(r, w / 2, h / 2)))
    qx = np.abs(xs - w / 2.0) - (w / 2.0 - r)
    qy = np.abs(ys - h / 2.0) - (h / 2.0 - r)
    d = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r
    return np.clip(0.5 - d, 0.0, 1.0).astype(np.float32)


def _ease_out(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return 1 - (1 - u) ** 3


def _ease_in_out(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def _pop(u: float) -> float:
    """From 0 to 1 with a small overshoot (about 4%) before it settles."""
    u = min(1.0, max(0.0, u))
    c = 1.1
    return 1 + (c + 1) * (u - 1) ** 3 + c * (u - 1) ** 2


OVERSHOOT = max(_pop(u / 200) for u in range(201))  # the largest a card grows to as it pops


@dataclass
class Tile:
    """Where a picture sits: its middle, its size (before any tilt) and its tilt, in output pixels."""
    cx: float
    cy: float
    w: float
    h: float
    angle: float = 0.0  # degrees, positive turns it clockwise

    def corners(self, scale: float = 1.0, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
        a = math.radians(self.angle)
        c, s = math.cos(a), math.sin(a)
        hw, hh = self.w * scale / 2, self.h * scale / 2
        pts = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
        rot = np.stack([pts[:, 0] * c - pts[:, 1] * s, pts[:, 0] * s + pts[:, 1] * c], axis=1)
        return rot + np.array([self.cx + dx, self.cy + dy])

    def box(self, scale: float = 1.0, dx: float = 0.0, dy: float = 0.0) -> Rect:
        p = self.corners(scale, dx, dy)
        return Rect(float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max()))

    def as_list(self) -> list:
        return [round(self.cx, 1), round(self.cy, 1), round(self.w, 1), round(self.h, 1), round(self.angle, 2)]


class Card:
    """One picture in its tile, framed in a style: rounded, edged, shadowed, tilted. A still picture is
    made once; a screen pushing in or a page scrolling is made again on each frame it moves."""

    def __init__(self, pic: Picture, tile: Tile, style: str, frame_size: tuple[int, int], up: float):
        self.pic, self.tile, self.style = pic, tile, style
        self.W, self.H = frame_size
        self.up = up  # seconds it is on screen
        self.w, self.h = max(8, int(round(tile.w))), max(8, int(round(tile.h)))
        s = min(self.w, self.h)
        self.s = s
        if style == "print" and pic.kind != "graphic":
            self.border = max(2, int(round(0.055 * s)))
            self.radius, self.inner_r = 0.03 * s, 0.012 * s
        elif style == "glass":
            self.border = max(3, int(round(0.06 * s)))
            self.radius, self.inner_r = 0.09 * s, max(2.0, 0.09 * s - self.border)
        else:
            self.border = 0
            self.radius = self.inner_r = 0.075 * s
        self.inner = (self.w - 2 * self.border, self.h - 2 * self.border)
        aspect = self.inner[0] / self.inner[1]
        self.how = ""
        self.windows: Optional[tuple] = None  # (from, to) for a screen; the page's top and its point for a page
        if pic.kind == "photo":
            self.win = cover_window(pic, aspect)
            share = self.win[2] * self.win[3] / (pic.rgb.shape[0] * pic.rgb.shape[1])
            self.how = ("cropped round the face it holds" if pic.faces else "cropped round its detail") + \
                (f", {share:.0%} of it" if share < 0.98 else ", whole")
        elif pic.kind == "screen":
            whole, part = screen_windows(pic, aspect, self.inner[0], self.W)
            self.win = part
            if part != whole:
                self.windows = (whole, part)
                self.how = (f"lands whole, then pushes in to its busiest part ({part[2] / pic.rgb.shape[1]:.0%} "
                            "of its width), where its writing reads")
            else:
                self.how = "whole, its writing readable"
        elif pic.kind == "page":
            ph, pw = pic.rgb.shape[:2]
            wh = pw / aspect
            top = (0.0, 0.0, float(pw), wh)
            point_y = min(max(pic.page_point * ph - wh / 2, 0.0), max(0.0, ph - wh))
            self.win = top
            if point_y > 0.04 * wh:
                self.windows = (top, (0.0, point_y, float(pw), wh))
                self.how = f"a page {ph / wh:.1f} tiles long, scrolling down to its point ({pic.page_point:.0%} of the way)"
            else:
                self.how = "a page, its top part"
        else:
            self.win = None
            self.how = "whole, on " + ("its own ground" if pic.alpha is None else "a plate it reads on")
        self.mask = _round_mask(self.w, self.h, self.radius)
        self.inner_mask = _round_mask(self.inner[0], self.inner[1], self.inner_r) if self.border else self.mask
        self._still: Optional[Sprite] = None
        self.pane = None
        if style == "glass":
            from ..core import glass as glassmod

            self.pane = glassmod.Pane(self.w, self.h, self.radius, unit=s / 300.0)
        self.pad = int(math.ceil(0.09 * s)) + 2  # room round the card for its shadow and its tilt

    # -- the picture inside
    def content(self, t: float) -> np.ndarray:
        """The picture as the tile shows it, `t` seconds after it lands (RGB, the inner size)."""
        iw, ih = self.inner
        pic = self.pic
        if pic.kind == "graphic":
            if not hasattr(self, "_graphic"):
                self._graphic, self.plate = graphic_image(pic, iw, ih)
            return self._graphic
        win = self.win
        if self.windows is not None:
            a, b = self.windows
            if pic.kind == "screen":  # lands whole, then pushes in
                e = _ease_in_out((t - 0.45) / 0.75)
            else:  # a page: rests on its top, then scrolls slowly to its point and rests there
                e = _ease_in_out((t - 0.5) / max(0.8, min(3.0, self.up - 1.4)))
            win = tuple(a[k] + (b[k] - a[k]) * e for k in range(4))
        return window_image(pic, win, iw, ih)

    def moving(self, t: float) -> bool:
        if self.windows is None:
            return False
        if self.pic.kind == "screen":
            return 0.4 <= t <= 1.25
        return 0.45 <= t <= 0.55 + max(0.8, min(3.0, self.up - 1.4))

    # -- the card round it
    def _card(self, t: float) -> np.ndarray:
        """The card, untilted, as premultiplied RGBA float (with the room round it for the shadow)."""
        import cv2

        p = self.pad
        W, H = self.w + 2 * p, self.h + 2 * p
        out = np.zeros((H, W, 4), np.float32)
        s = self.s
        # the shadow: soft, a little below
        sh = np.zeros((H, W), np.float32)
        sh[p:p + self.h, p:p + self.w] = self.mask
        sh = cv2.GaussianBlur(sh, (0, 0), max(1.0, 0.035 * s))
        dy = int(round(0.022 * s))
        sh = np.roll(sh, dy, axis=0)
        if dy > 0:
            sh[:dy] = 0
        out[..., 3] = sh * (0.36 if self.style != "glass" else 0.0)
        if self.style != "glass":
            outside = np.zeros((H, W), np.uint8)
            outside[p:p + self.h, p:p + self.w] = (self.mask > 0.5).astype(np.uint8)
            d = cv2.distanceTransform(1 - outside, cv2.DIST_L2, 3)
            line = np.clip(1.0 - np.abs(d - 0.9) / 0.9, 0, 1) * (outside == 0)
            out[..., 3] = np.maximum(out[..., 3], 0.2 * line)
        img = self.content(t).astype(np.float32)
        a = self.mask
        if self.style == "print" and self.pic.kind != "graphic":
            face = np.empty((self.h, self.w, 3), np.float32)
            face[:] = (250, 249, 246)
            b = self.border
            ia = self.inner_mask[..., None]
            face[b:b + self.inner[1], b:b + self.inner[0]] = (face[b:b + self.inner[1], b:b + self.inner[0]] * (1 - ia)
                                                               + img * ia)
        elif self.style == "glass":
            face = np.zeros((self.h, self.w, 3), np.float32)
            b = self.border
            ia = self.inner_mask
            face[b:b + self.inner[1], b:b + self.inner[0]] = img * ia[..., None]
            ca = np.zeros((self.h, self.w), np.float32)
            ca[b:b + self.inner[1], b:b + self.inner[0]] = ia
            a = ca  # the glass itself is drawn on the frame; this is the picture on it
        else:
            face = img
            # a hairline of light inside the edge, so a dark picture parts from a dark room
            edge = np.clip(1.0 - np.abs(cv2.distanceTransform((self.mask > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
                                        - 1.2) / 1.0, 0, 1) * (self.mask > 0.5)
            face = face * (1 - 0.22 * edge[..., None]) + 255.0 * 0.22 * edge[..., None]
        # the card over its shadow (premultiplied)
        ca = a[..., None]
        region = out[p:p + self.h, p:p + self.w]
        region[..., :3] = region[..., :3] * (1 - ca) + face * ca
        region[..., 3] = region[..., 3] * (1 - a) + a
        return out

    def _tilted(self, card: np.ndarray) -> np.ndarray:
        import cv2

        if abs(self.tile.angle) < 0.05:
            return card
        H, W = card.shape[:2]
        a = math.radians(self.tile.angle)
        nw = int(math.ceil(abs(W * math.cos(a)) + abs(H * math.sin(a)))) + 2
        nh = int(math.ceil(abs(W * math.sin(a)) + abs(H * math.cos(a)))) + 2
        m = cv2.getRotationMatrix2D((W / 2, H / 2), -self.tile.angle, 1.0)
        m[0, 2] += (nw - W) / 2
        m[1, 2] += (nh - H) / 2
        return cv2.warpAffine(card, m, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def sprite(self, t: float) -> Sprite:
        """The card as a sprite, centred where its middle goes, `t` seconds after it lands."""
        if self._still is not None and not self.moving(t):
            if self.windows is None or t > 1.3:
                return self._still
        rgba = self._tilted(self._card(t))
        sp = Sprite(rgb=np.ascontiguousarray(rgba[..., :3]), a=np.ascontiguousarray(np.clip(rgba[..., 3], 0, 1)))
        if not self.moving(t) and (self.windows is None or t > 1.3 + (0 if self.pic.kind == "screen" else self.up)):
            self._still = sp
        return sp

    def draw(self, frame: np.ndarray, t: float, scale: float, opacity: float, dy: float) -> None:
        """Lay the card on `frame` (`t` seconds after it lands), grown by `scale` about its middle."""
        from ..core.compose import over_centred

        if opacity <= 0.003:
            return
        cx, cy = self.tile.cx, self.tile.cy + dy
        if self.pane is not None:
            k = max(0.1, scale)
            self.pane.draw(frame, cx - self.w / 2, cy - self.h / 2, opacity=opacity, frost=min(1.0, 0.4 + 0.6 * opacity),
                           scale=k)
        over_centred(frame, self.sprite(t), cx, cy, scale=scale, opacity=opacity)


# ---- which picture on which word, and when ----------------------------------------------------------

@dataclass
class Shot:
    """One picture on screen: the word it lands on, the frame it lands, when it leaves, where it sits."""
    item: int  # the picture, by its place in the folder order
    start: float  # the word's start (seconds), as the core times it
    word: str
    why: str  # map, tags, name, keyword, time, plan
    priority: int = 0
    score: float = 0.0
    frame: int = 0  # the first frame of its word
    land: float = 0.0  # that frame's time
    leave: float = 0.0  # gone by then
    group: int = 0
    tile: Optional[Tile] = None
    card: Optional[Card] = None


def load_pool(spec: str, mapping: list[tuple[str, str]]) -> tuple[list[Picture], dict[str, int]]:
    def load(p: Path) -> Picture:
        return load_picture(p, logo_hint=bool(_LOGO_HINT.search(p.parent.name)))

    return match.pool(spec, mapping, KIND, load)


def decide(words: list[Word], quiet, pool: list[Picture], named: dict[str, int], on: str, seconds: float,
           fps: float) -> tuple[list[Shot], dict[int, str], list[str]]:
    """The pictures in time order, why each picture not used was left out, and notes for the person.
    `words` are the clip's whole words, fillers and all, so what the speaker took back is found
    (`moments.taken_back`) and no picture lands on it."""
    said = match.said(words)
    back = moments.taken_back(words, quiet)
    left: dict[int, str] = {}
    notes: list[str] = []
    cands: list[Shot] = []
    if on not in ("keywords", "map"):
        times = [float(x) for x in on.split(",")]
        for k, t in enumerate(times):
            if k >= len(pool):
                notes.append(f"{len(times)} times for {len(pool)} pictures: the last {len(times) - k} not used")
                break
            near = [w for w in said if abs(w.start - t) <= SNAP]
            w = min(near, key=lambda w: abs(w.start - t)) if near else None
            cands.append(Shot(k, float(w.start) if w else t, match.bare(w) if w else "", "time", 3))
        unnamed: list[int] = []
    else:
        found, unnamed = match.candidates(said, quiet, pool, named, on, left, back)
        cands = [Shot(c.item if c.item is not None else -1, c.start, c.word, c.why, c.priority, c.score)
                 for c in found]
    placed: list[Shot] = []
    used: set[int] = set()
    slots = len(unnamed)
    for c in sorted(cands, key=lambda c: (-c.priority, -c.score, c.start)):
        named_one = c.item >= 0
        if named_one and c.item in used:
            continue
        if not named_one and slots <= 0:
            continue
        if c.start > seconds - MIN_ON:
            if named_one:
                left.setdefault(c.item, f"its word “{c.word}” at {c.start:.2f} s comes too near the end to be seen")
            continue
        clash = next((p for p in placed if abs(p.start - c.start) < APART), None)
        if clash is not None:
            if named_one:
                left.setdefault(c.item, f"its word “{c.word}” at {c.start:.2f} s comes {abs(clash.start - c.start):.2f} s "
                                        f"from “{clash.word}”, too close to land apart")
            continue
        placed.append(c)
        if named_one:
            used.add(c.item)
            left.pop(c.item, None)
        else:
            slots -= 1
    picked = [c for c in placed if c.item < 0]
    for c, idx in match.folder_order(picked, unnamed):
        c.item = idx
    if on == "keywords":
        for idx in unnamed[len(picked):]:
            left.setdefault(idx, "no strong word was left for it")
    shots = sorted((c for c in placed if c.item >= 0), key=lambda c: c.start)
    for s in shots:
        s.frame = int(math.ceil(s.start * fps - 1e-6))
        s.land = s.frame / fps
    used = {s.item for s in shots}
    for idx in range(len(pool)):
        if idx not in used:
            left.setdefault(idx, "not used")
    return shots, {i: why for i, why in left.items() if i not in used}, notes


def sentence_end(words: list[Word], t: float, quiet) -> float:
    """When the sentence (or the phrase before a real pause) that is being said at `t` has ended."""
    from ..core.words import pause_after

    for i, w in enumerate(words):
        if w.end < t - 1e-6:
            continue
        if w.ends_sentence or i == len(words) - 1 or pause_after(words, i, quiet) >= 0.5:
            return float(w.end)
    return float(words[-1].end) if words else t


def hold_for(shot: Shot, hold: str, words: list[Word], quiet, seconds: float) -> float:
    """How long the pictures stay after `shot`, the last of them, lands."""
    if hold == "end":
        return max(0.0, seconds - shot.land)
    if hold != "auto":
        return float(hold)
    lo, hi = HOLD_AUTO
    return min(hi, max(lo, sentence_end(words, shot.start, quiet) + 0.35 - shot.land))


@dataclass
class Group:
    shots: list
    layout: str = ""
    where: str = ""  # in words: "above the head", "beside the head, on the left", …
    leave: float = 0.0
    k: float = 0.0  # the tiles' width, as a share of the frame's
    note: str = ""


def group_shots(shots: list[Shot], layout: str, hold: str, words, quiet, seconds: float) -> list[Group]:
    """Pictures named while others are still up share the screen; a new group begins once the last
    has gone, or when the layout holds no more."""
    cap = MAX_GROUP.get(layout, 4) if layout != "auto" else 4
    groups: list[Group] = []
    for s in shots:
        g = groups[-1] if groups else None
        if (g is not None and layout != "rotate" and len(g.shots) < cap
                and s.land < g.shots[-1].land + hold_for(g.shots[-1], hold, words, quiet, seconds)):
            g.shots.append(s)
        else:
            groups.append(Group([s]))
    for i, g in enumerate(groups):
        last = g.shots[-1]
        g.leave = min(seconds, last.land + hold_for(last, hold, words, quiet, seconds))
        if i + 1 < len(groups):
            nxt = groups[i + 1].shots[0].land
            if nxt - g.leave < BRIDGE:
                g.leave = nxt  # gone just as the next lands, rather than a moment of nothing between them
            # one picture replacing another swaps over the new one's first frames; a group makes room
            g.leave = min(g.leave, nxt + (0.6 * ENTER if layout == "rotate" else 0.0))
        g.leave = max(g.leave, last.land + min(MIN_ON, seconds - last.land))
        for s in g.shots:
            s.leave = g.leave
            s.group = i
    return groups


# ---- where: the room round the head -----------------------------------------------------------------

def _area(poly: list) -> float:
    return 0.5 * sum(poly[i - 1][0] * poly[i][1] - poly[i][0] * poly[i - 1][1] for i in range(len(poly)))


def shared_area(a: Tile, b: Tile) -> float:
    """How much two tiles cover of each other, by their turned shapes (one clipped by the other)."""
    out = [tuple(p) for p in a.corners()]
    clip = [tuple(p) for p in b.corners()]
    sign = 1.0 if _area(clip) > 0 else -1.0
    for k in range(len(clip)):
        p, q = clip[k - 1], clip[k]
        inp, out_ = out, []
        if not inp:
            return 0.0

        def side(r) -> float:
            return sign * ((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

        for m in range(len(inp)):
            cur, prev = inp[m], inp[m - 1]
            sc, sp = side(cur), side(prev)
            if sc >= 0:
                if sp < 0:
                    t = sp / (sp - sc)
                    out_.append((prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])))
                out_.append(cur)
            elif sp >= 0:
                t = sp / (sp - sc)
                out_.append((prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])))
        out = out_
    return abs(_area(out)) if len(out) >= 3 else 0.0


class Room:
    """The head and the face over a stretch of frames, and the safe zone round them."""

    def __init__(self, W: int, H: int, heads: list, faces: list):
        self.W, self.H = W, H
        self.zone = SafeZone(W, H).rect
        self.heads, self.faces = heads, faces
        self.gap = GAP * H
        self.gutter = GUTTER * W

    def boxes(self, a: int, b: int) -> tuple[Optional[Rect], Optional[Rect]]:
        a, b = max(0, a), min(len(self.heads) - 1, b)
        head = Rect.around(r for r in self.heads[a:b + 1] if r is not None)
        face = Rect.around(r for r in self.faces[a:b + 1] if r is not None)
        return head, face

    def rooms(self, box: Optional[Rect]) -> dict[str, Rect]:
        z, g = self.zone, self.gap
        if box is None:  # no face found: the band near the top, where a tall frame leaves room over a head
            return {"above": Rect(z.x0, z.y0, z.x1, z.y0 + 0.36 * self.H)}
        out = {"above": Rect(z.x0, z.y0, z.x1, min(z.y1, box.y0 - g)),
               "left": Rect(z.x0, z.y0, max(z.x0, box.x0 - g), z.y1),
               "right": Rect(min(z.x1, box.x1 + g), z.y0, z.x1, z.y1),
               "below": Rect(z.x0, max(z.y0, box.y1 + g), z.x1, z.y1)}
        return {k: r for k, r in out.items() if r.w > 8 and r.h > 8}

    def clear(self, tiles: list[Tile], box: Optional[Rect], others: bool = True, overlap: float = 0.0) -> bool:
        """Every tile, at its largest as it pops, inside the safe zone and clear of `box`; and (unless a
        cluster allows a little) clear of each other."""
        boxes = [t.box(OVERSHOOT) for t in tiles]
        keep = box.grow(self.gap, self.gap, self.gap, self.gap) if box is not None else None
        for r in boxes:
            if not r.inside(self.zone, slack=0.5):
                return False
            if keep is not None and r.intersects(keep):
                return False
        if others:  # two tiles at rest, by their real (turned) shapes: their boxes overlap long before they do
            for i in range(len(tiles)):
                for j in range(i + 1, len(tiles)):
                    a, b = tiles[i], tiles[j]
                    if shared_area(a, b) > max(overlap, 0.004) * min(a.w * a.h, b.w * b.h):
                        return False
        return True

    def sizes(self, cap: float, floor: float = SMALLEST) -> list[float]:
        """Tile widths to try, largest first, as pixels."""
        out, w = [], cap * self.W
        while w >= floor * self.W - 1e-6:
            out.append(w)
            w *= 0.965
        return out


def _span(tiles: list[Tile]) -> Rect:
    return Rect.around(t.box(OVERSHOOT) for t in tiles)


def fit_arc(room: Room, n: int, aspect: float, box: Optional[Rect], cap: float) -> Optional[tuple[float, list[Tile]]]:
    """An arc of `n` tiles over the head: the middle highest, the ends lower and turned out a little,
    as close over the head as it can sit."""
    z, gap = room.zone, room.gap
    m = (n - 1) / 2
    tilt = 0.0 if n == 1 else 6.0 if n == 2 else 10.0
    for w in room.sizes(cap):
        h = w / aspect
        step = w + room.gutter
        for sag_k in ((0.36, 0.24, 0.12, 0.0) if n > 2 else (0.18, 0.0) if n == 2 else (0.0,)):
            rel = [Tile((i - m) * step, (sag_k * h * ((i - m) / m) ** 2) if m else 0.0, w, h,
                        (tilt * (i - m) / m) if m else 0.0) for i in range(n)]
            span = _span(rel)
            if span.w > z.w:
                continue
            cx = (box.cx if box is not None else z.cx)
            cx = min(max(cx, z.x0 - span.x0), z.x1 - span.x1)
            hi, lo = math.inf, -math.inf
            for t in rel:
                r = t.box(OVERSHOOT, dx=cx)
                lo = max(lo, z.y0 - r.y0)
                over_head = box is not None and r.x1 > box.x0 - gap and r.x0 < box.x1 + gap
                hi = min(hi, (box.y0 - gap - r.y1) if over_head else (z.y1 - r.y1))
            if box is None:
                hi = min(hi, z.y0 + 0.36 * room.H - max(t.box(OVERSHOOT).y1 for t in rel))
            if lo > hi:
                continue
            tiles = [Tile(t.cx + cx, t.cy + hi, w, h, t.angle) for t in rel]
            if room.clear(tiles, box):
                return w, tiles
    return None


def fit_column(room: Room, n: int, aspect: float, box: Optional[Rect], side: str, cap: float):
    """`n` tiles stacked on one side of the head, the column level with the face."""
    if box is None:
        return None
    rooms = room.rooms(box)
    if side not in rooms:
        return None
    r = rooms[side]
    for w in room.sizes(cap):
        if w * OVERSHOOT > r.w:
            continue
        h = w / aspect
        total = n * h + (n - 1) * room.gutter
        if total * OVERSHOOT > r.h:
            continue
        cx = r.cx
        mid = box.y0 + 0.5 * box.h
        y0 = min(max(mid - total / 2, r.y0 + 0.03 * h), r.y1 - total - 0.03 * h)
        tiles = [Tile(cx, y0 + h / 2 + i * (h + room.gutter), w, h, 0.0) for i in range(n)]
        if room.clear(tiles, box):
            return w, tiles
    return None


def fit_row(room: Room, n: int, aspect: float, box: Optional[Rect], cap: float):
    """`n` tiles side by side under the chin: where a close-up leaves its room."""
    if box is None:
        return None
    rooms = room.rooms(box)
    if "below" not in rooms:
        return None
    r = rooms["below"]
    z = room.zone
    for w in room.sizes(cap):
        h = w / aspect
        if h * OVERSHOOT > r.h:
            continue
        span = n * w + (n - 1) * room.gutter
        if span * OVERSHOOT > z.w:
            continue
        cx = min(max(box.cx, z.x0 + span / 2 + 0.03 * w), z.x1 - span / 2 - 0.03 * w)
        cy = r.y0 + h * OVERSHOOT / 2 + 1
        tiles = [Tile(cx - span / 2 + w / 2 + i * (w + room.gutter), cy, w, h, 0.0) for i in range(n)]
        if room.clear(tiles, box):
            return w, tiles
    return None


def fit_single(room: Room, aspect: float, box: Optional[Rect], cap: float, prefer: tuple = ("above", "left", "right",
                                                                                            "below")):
    """One tile, as large as one room round the head allows: (width, tile, room's name)."""
    best = None
    pref = {"above": 1.0, "left": 0.93, "right": 0.93, "below": 0.82}
    for name, r in room.rooms(box).items():
        if name not in prefer:
            continue
        w0 = 0.985 * min(cap * room.W, r.w / OVERSHOOT, r.h * aspect / OVERSHOOT)
        tile = None
        for w in (w0, 0.95 * w0, 0.9 * w0):
            if w < SMALLEST * room.W:
                break
            h = w / aspect
            hw, hh = w * OVERSHOOT / 2 + 1, h * OVERSHOOT / 2 + 1
            if name == "above":
                cx = min(max(box.cx if box is not None else r.cx, r.x0 + hw), r.x1 - hw)
                cy = max(r.y1 - hh, r.y0 + hh)
            elif name == "below":
                cx = min(max(box.cx, r.x0 + hw), r.x1 - hw)
                cy = min(r.y0 + hh, r.y1 - hh)
            else:
                cx = r.cx
                mid = box.y0 + 0.5 * box.h
                cy = min(max(mid, r.y0 + hh), r.y1 - hh)
            t = Tile(cx, cy, w, h, 0.0)
            if room.clear([t], box):
                tile = t
                break
        if tile is None:
            continue
        score = w * pref[name]
        if best is None or score > best[0]:
            best = (score, w, tile, name)
    return (best[1], best[2], best[3]) if best else None


_CLUSTER_TILT = (-4.0, 5.0, -3.0, 4.0, -5.0, 3.0)


def fit_cluster(room: Room, aspects: list[float], box: Optional[Rect], cap: float):
    """A loose pile of tiles over the head: the first a little larger, each turned a touch, touching
    its neighbours, never over the face."""
    z = room.zone
    n = len(aspects)
    for w in room.sizes(cap):
        tiles: list[Tile] = []
        ok = True
        for k, a in enumerate(aspects):
            tw = w * (1.0 if k == 0 else 0.82)
            th = tw / a
            ang = _CLUSTER_TILT[k % len(_CLUSTER_TILT)]
            probe = Tile(0.0, 0.0, tw, th, ang).box(OVERSHOOT)
            hx, hy = probe.w / 2, probe.h / 2
            step = max(4.0, 0.012 * room.W)
            xs = np.arange(z.x0 + hx, z.x1 - hx + 1e-6, step)
            ys = np.arange(z.y0 + hy, z.y1 - hy + 1e-6, step)
            if not len(xs) or not len(ys):
                ok = False
                break
            X, Y = np.meshgrid(xs, ys)
            valid = np.ones(X.shape, bool)
            if box is not None:
                keep = box.grow(room.gap, room.gap, room.gap, room.gap)
                valid &= ~((X - hx < keep.x1) & (X + hx > keep.x0) & (Y - hy < keep.y1) & (Y + hy > keep.y0))
                valid &= Y < box.y0 + 0.45 * box.h  # over the head and beside its upper half, never low by the jaw
            else:
                valid &= Y + hy <= z.y0 + 0.36 * room.H
            score = np.zeros(X.shape)
            if not tiles:
                ax = box.cx if box is not None else z.cx
                ay = (box.y0 - room.gap - hy) if box is not None else z.y0 + hy
                score -= np.hypot(X - ax, (Y - ay) * 1.3) / room.W
            else:
                boxes = [t.box(OVERSHOOT) for t in tiles]
                near = np.full(X.shape, np.inf)
                for r in boxes:
                    ox = np.clip(np.minimum(X + hx, r.x1) - np.maximum(X - hx, r.x0), 0, None)
                    oy = np.clip(np.minimum(Y + hy, r.y1) - np.maximum(Y - hy, r.y0), 0, None)
                    ov = ox * oy / min(4 * hx * hy, r.w * r.h)
                    valid &= ov <= 0.16
                    valid &= ~((np.abs(X - r.cx) < 0.5 * r.w) & (np.abs(Y - r.cy) < 0.5 * r.h))
                    dx = np.maximum(np.maximum(r.x0 - (X + hx), (X - hx) - r.x1), 0)
                    dy = np.maximum(np.maximum(r.y0 - (Y + hy), (Y - hy) - r.y1), 0)
                    near = np.minimum(near, np.hypot(dx, dy) - 3.0 * (ov > 0.02))
                cxs = [r.cx for r in boxes]
                ws_ = [r.w * r.h for r in boxes]
                hcx = box.cx if box is not None else z.cx
                centroid = (sum(c * a for c, a in zip(cxs, ws_)) + X * 4 * hx * hy) / (sum(ws_) + 4 * hx * hy)
                score -= 2.2 * np.abs(centroid - hcx) / room.W
                score -= 3.0 * np.clip(near, 0, None) / room.W  # one pile, not scattered tiles
                if box is not None:
                    score -= 0.8 * np.abs(Y - (box.y0 - hy)) / room.H
            score = np.where(valid, score, -np.inf)
            i = int(np.argmax(score))
            if not np.isfinite(score.flat[i]):
                ok = False
                break
            tiles.append(Tile(float(X.flat[i]), float(Y.flat[i]), tw, th, ang))
        if ok and room.clear(tiles, box, overlap=0.17):
            return w, tiles
    return None


def group_aspects(pics: list[Picture]) -> tuple[float, ...]:
    """The tile shapes an arc, a column or a row may take, the first preferred: upright or wide for
    photos, wide for screens, square for logos; the room decides among them (`best_shape`)."""
    kinds = {p.kind for p in pics}
    if kinds <= {"photo"}:
        return (0.8, 1.0, 1.25)
    if kinds <= {"screen"}:
        return (1.25, 1.0, 1.5)
    if kinds <= {"graphic"}:
        return (1.0, 1.25)
    return (1.0, 0.85, 1.2)


def best_shape(fit, aspects: tuple[float, ...]):
    """`fit(aspect)` for each shape, keeping the one that gives the largest picture (a shape later in
    the list must win by a tenth to be taken over an earlier one)."""
    best = None
    for k, a in enumerate(aspects):
        got = fit(a)
        if not got:
            continue
        w = got[0]
        area = w * (w / a) * (1.0 if k == 0 else 0.9)
        if best is None or area > best[0]:
            best = (area, got)
    return best[1] if best else None


def own_aspect(pic: Picture) -> float:
    """A picture's tile when it stands alone: its own shape, held between upright and wide."""
    if pic.kind == "page":
        return 0.8
    if pic.kind == "graphic":
        return min(1.6, max(0.8, pic.aspect + 0.15))
    return min(1.5, max(0.75, pic.aspect))


# ---- choosing the layout for each group -------------------------------------------------------------

ARC_CAP = {1: 0.40, 2: 0.31, 3: 0.25, 4: 0.20, 5: 0.16}  # the widest an arc's tile is, by how many share it
CAP = {"column": 0.28, "row": 0.27, "cluster": 0.30, "rotate": 0.62}
LOGO_CAP = 0.30  # a logo alone: it reads long before it fills the room
WHERE = {"arc": "an arc above the head", "cluster": "a cluster above the head", "row": "a row under the chin"}


def arrange(room: Room, g: Group, pics: list[Picture], layout: str, size_k: float, fps: float) -> Optional[list[Tile]]:
    """Where each picture of the group sits. Fills in the group's layout, where it is and its tiles'
    width; None when nothing fits clear of the face inside the safe zone."""
    a = g.shots[0].frame
    b = int(math.ceil(g.leave * fps))
    head, face = room.boxes(a, b)
    n = len(g.shots)
    gp = [pics[s.item] for s in g.shots]
    options = []  # (score, layout, where, width, tiles, crosses the hair)

    def add(name: str, where: str, got, pref: float, hair: bool) -> None:
        if got:
            tiles = got[1]
            side = min(math.sqrt(t.w * t.h) for t in tiles)  # how big the smallest looks: a square of its area
            area = sum(t.w * t.h for t in tiles) / len(tiles)
            options.append((math.sqrt(area) * pref, name, where, side, tiles, hair))

    modes = [(head, False, 1.0)] + ([(face, True, 0.85)] if face is not None and head is not None else [])
    for box, hair, mp in modes:
        if n == 1 or layout == "rotate":
            cap = (LOGO_CAP if gp[0].kind == "graphic" else CAP["rotate"]) * size_k
            own = own_aspect(gp[0])
            # a photo may be cropped wider or taller to fill the room it is given; a screen, a page and
            # a logo keep their own shape
            shapes = (own, 1.0, 1.33, 0.8) if gp[0].kind == "photo" else (own,)

            def single(a: float, box=box, cap=cap):
                r = fit_single(room, a, box, cap)
                return (r[0], [r[1]], r[2]) if r else None

            got = best_shape(single, shapes)
            if got:
                w, tile, name = got[0], got[1][0], got[2]
                where = {"above": "above the head", "left": "beside the head, on the left",
                         "right": "beside the head, on the right", "below": "under the chin"}[name]
                add("rotate" if layout == "rotate" else "single", where, (w, [tile]), 1.0 * mp, hair)
            continue
        if layout in ("auto", "arc") and n <= MAX_GROUP["arc"]:
            add("arc", WHERE["arc"], best_shape(lambda a: fit_arc(room, n, a, box, ARC_CAP[n] * size_k),
                                                group_aspects(gp)), 1.0 * mp, hair)
        if layout in ("auto", "column") and n <= MAX_GROUP["column"]:
            for side in ("left", "right"):
                add("column", f"a column on the {side} of the head",
                    best_shape(lambda a: fit_column(room, n, a, box, side, CAP["column"] * size_k),
                               group_aspects(gp)), 0.96 * mp, hair)
        if layout == "cluster":
            add("cluster", WHERE["cluster"], fit_cluster(room, [own_aspect(p) for p in gp], box, CAP["cluster"] * size_k),
                1.0 * mp, hair)
        if layout in ("auto",) or not options:
            add("row", WHERE["row"], best_shape(lambda a: fit_row(room, n, a, box, CAP["row"] * size_k),
                                                group_aspects(gp)), 0.84 * mp, hair)
    if not options:
        return None
    asked = [o for o in options if o[1] == layout or (layout == "auto")]
    pool = asked or options
    readable = [o for o in pool if o[3] >= READ * room.W]
    best = max(readable or pool, key=lambda o: o[0])
    _, g.layout, g.where, w, tiles, hair = best
    g.k = w / room.W
    notes = []
    if layout not in ("auto", "rotate") and g.layout != layout and n > 1:
        notes.append(f"no room for a {layout} here; " + WHERE.get(g.layout, g.where) + " instead")
    if hair:
        notes.append("allowed over the top of the hair to be larger, never over the face")
    if g.k < READ:
        notes.append(f"small ({g.k:.0%} of the frame's width): there is little room clear of the face on this shot")
    g.note = "; ".join(notes)
    return tiles


def place_groups(room: Room, groups: list[Group], pics: list[Picture], layout: str, size_k: float, fps: float,
                 warnings: list[str]) -> list[Group]:
    """Every group placed; one that cannot fit as asked becomes one picture at a time, and a picture with
    no room anywhere clear of the face is left out."""
    out: list[Group] = []
    queue = list(groups)
    while queue:
        g = queue.pop(0)
        tiles = arrange(room, g, pics, layout, size_k, fps)
        if tiles is None and len(g.shots) > 1:
            singles = []
            for i, s in enumerate(g.shots):
                nxt = g.shots[i + 1].land + 0.6 * ENTER if i + 1 < len(g.shots) else g.leave
                singles.append(Group([s], leave=min(g.leave, nxt)))
            warnings.append(f"{len(g.shots)} pictures named together from {g.shots[0].land:.2f} s did not fit side by "
                            "side; they come one at a time")
            queue = singles + queue
            continue
        if tiles is None:
            warnings.append(f"no room for “{g.shots[0].word}” at {g.shots[0].land:.2f} s clear of the face inside the "
                            "safe zone; that picture is left out")
            continue
        for s, t in zip(g.shots, tiles):
            s.tile = t
            s.leave = g.leave
        out.append(g)
    for i, g in enumerate(out):
        for s in g.shots:
            s.group = i
    return out


# ---- drawing them ---------------------------------------------------------------------------------

def state(s: Shot, t: float) -> Optional[tuple[float, float, float]]:
    """(scale, opacity, how far below its place) for a picture at `t`, or None when it is not up."""
    if t < s.land - 1e-9 or t >= s.leave - 1e-9:
        return None
    h = s.tile.h
    u = (t - s.land) / ENTER
    if u < 1.0:
        scale = POP_FROM + (1 - POP_FROM) * _pop(u)
        opacity = POP_OPACITY + (1 - POP_OPACITY) * _ease_out(u * 2.5)
        dy = (1 - _ease_out(u)) * 0.05 * h
    else:
        scale, opacity, dy = 1.0, 1.0, 0.0
    v = (s.leave - t) / EXIT
    if v < 1.0:
        opacity *= _ease_in_out(v)
        scale *= 0.94 + 0.06 * v
        dy -= (1 - v) * 0.03 * h  # it lifts a little as it goes
    return scale, opacity, dy


class Painter:
    def __init__(self, shots: list[Shot]):
        self.shots = sorted(shots, key=lambda s: (s.group, s.land))

    def draw(self, frame: np.ndarray, j: int, t: float) -> None:
        for s in self.shots:
            st = state(s, t)
            if st is None:
                continue
            scale, opacity, dy = st
            s.card.draw(frame, t - s.land, scale, opacity, dy)

    def boxes(self, t: float) -> list[Rect]:
        out = []
        for s in self.shots:
            st = state(s, t)
            if st is not None and st[1] > 0.02:
                out.append(s.tile.box(st[0], dy=st[2]))
        return out


def report(painter: Painter, room: Room, n_frames: int, fps: float) -> dict:
    """Frame by frame: a picture over the face (hairline to chin, ear to ear) and outside the safe zone
    must both be 0; how often one crosses the top of the hair is counted too."""
    over_face = over_hair = outside = 0
    for j in range(n_frames):
        boxes = painter.boxes(j / fps)
        if not boxes:
            continue
        face, head = room.faces[j], room.heads[j]
        if face is not None and any(r.intersects(face) for r in boxes):
            over_face += 1
        elif head is not None and any(r.intersects(head) for r in boxes):
            over_hair += 1
        if any(not r.inside(room.zone, slack=0.5) for r in boxes):
            outside += 1
    return {"frames_over_face": over_face, "frames_over_hair": over_hair, "frames_outside_safe_zone": outside}


def _short(why: str) -> str:
    for part, short in (("never said", "never said"), ("no strong word", "no word left"), ("too near the end", "too late"),
                        ("too close", "too close to another"), ("brand's name", "no brand in its name"),
                        ("on=map", "on=map"), ("no room", "no room")):
        if part in why:
            return short
    return "not used"


def pictures_sheet(pool: list[Picture], shots: list[Shot], left: dict[int, str]) -> Optional[Path]:
    """Every picture given, numbered and named, with what it was taken for (photo, screen, page or logo),
    the part of it shown (the yellow box; a screen's push ends in the green one) and the word it landed
    on, or why it was left out. The pack never looks at the pictures to choose; this sheet is for the
    person, or the agent running the pack, to look at and write a `map` from. Kept in the pack's own
    cache, never beside the pictures."""
    from PIL import Image, ImageDraw

    from ..core import text as textmod

    if not pool:
        return None
    key = hashlib.sha1("|".join(f"{p.path}:{next((s.land for s in shots if s.item == i), '')}"
                                for i, p in enumerate(pool)).encode()).hexdigest()[:16]
    folder = CACHE / "pictures"
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"sheet-{key}.jpg"
    tw, th, cols = 260, 300, 5
    try:
        font = textmod.font("poppins-semibold", 15)
    except Exception:  # a missing face never stops a render for the sake of a sheet
        font = None
    tiles = []
    for i, p in enumerate(pool):
        img = Image.fromarray(p.rgb)
        if p.alpha is not None:
            bg = Image.new("RGB", img.size, (200, 200, 200))
            bg.paste(img, (0, 0), Image.fromarray((p.alpha * 255).astype(np.uint8)))
            img = bg
        img.thumbnail((tw - 12, th - 12), Image.LANCZOS)
        k = img.width / p.rgb.shape[1]
        tile = Image.new("RGB", (tw, th + 56), (24, 24, 24))
        ox, oy = (tw - img.width) // 2, (th - img.height) // 2
        tile.paste(img, (ox, oy))
        d = ImageDraw.Draw(tile)
        shot = next((s for s in shots if s.item == i), None)
        if shot is not None and shot.card is not None and shot.card.win is not None:
            wins = [(shot.card.windows[0] if shot.card.windows else shot.card.win, (255, 214, 0))]
            if shot.card.windows:
                wins.append((shot.card.windows[1], (60, 220, 120)))
            for (x, y, w, h), col in wins:
                d.rectangle([ox + x * k, oy + y * k, ox + (x + w) * k, oy + (y + h) * k], outline=col, width=3)
        line2 = (f"on “{shot.word}” at {shot.land:.1f} s" if shot else "left out: " + _short(left.get(i, "")))[:36]
        name = p.label if len(p.label) <= 26 else "…" + p.label[-25:]
        what = "logo" if p.logo else p.kind
        d.text((6, th + 4), f"{i + 1}. {name}", fill=(240, 240, 240), font=font)
        d.text((6, th + 22), what + (f" · {' '.join(p.names)}" if p.names else ""), fill=(170, 190, 230), font=font)
        d.text((6, th + 38), line2, fill=(140, 220, 140) if shot else (220, 170, 120), font=font)
        tiles.append(tile)
    rows = math.ceil(len(tiles) / cols)
    n = min(cols, len(tiles))
    sheet = Image.new("RGB", (n * tw + (n - 1) * 8, rows * (th + 56) + (rows - 1) * 8), (40, 40, 40))
    for k_, t_ in enumerate(tiles):
        sheet.paste(t_, ((k_ % cols) * (tw + 8), (k_ // cols) * (th + 56 + 8)))
    sheet.save(out, quality=84)
    return out


# ---- a plan given back --------------------------------------------------------------------------------

def _file_px(pic: Picture, win: Optional[tuple]) -> Optional[list]:
    if win is None:
        return None
    k = pic.width / pic.rgb.shape[1]
    return [round(float(v) * k, 1) for v in win]


def _from_plan(plan: dict, seconds: float, fps: float) -> tuple[list[Picture], list[Shot], list[Group]]:
    from . import PlanError

    items = plan.get("pictures")
    if not isinstance(items, list) or not items:
        raise PlanError("the plan has no pictures; run once without --plan to get one to edit")
    pool: list[Picture] = []
    index: dict[str, int] = {}
    shots: list[Shot] = []
    for it in items:
        try:
            f, start, leave = str(it["file"]), float(it["start"]), float(it["leave"])
            cx, cy, w, h, ang = (float(v) for v in it["tile"])
        except (KeyError, TypeError, ValueError):
            raise PlanError("each picture in the plan needs file, start, leave and tile [x, y, w, h, angle]") from None
        if not Path(f).exists():
            raise PlanError(f"the plan's picture {f} is not there any more")
        if not (0 <= start < leave <= seconds + 1e-3):
            raise PlanError(f"the picture at {start:.2f}-{leave:.2f} s is not inside the clip (0-{seconds:.2f} s)")
        if f not in index:
            index[f] = len(pool)
            pool.append(load_picture(Path(f), logo_hint=bool(_LOGO_HINT.search(Path(f).parent.name))))
        s = Shot(index[f], start, str(it.get("word", "")), str(it.get("why", "plan")), 3)
        s.frame = int(math.ceil(start * fps - 1e-6))
        s.land, s.leave = s.frame / fps, leave
        s.group = int(it.get("group", 0))
        s.tile = Tile(cx, cy, w, h, ang)
        s.window = it.get("window")  # type: ignore[attr-defined]
        s.window_from = it.get("from")  # type: ignore[attr-defined]
        shots.append(s)
    groups: dict[int, Group] = {}
    for s in sorted(shots, key=lambda s: s.land):
        g = groups.setdefault(s.group, Group([], leave=s.leave))
        g.shots.append(s)
        g.leave = max(g.leave, s.leave)
        g.layout = str(next((it.get("layout", "plan") for it in items if float(it["start"]) == s.start), "plan"))
        g.where = "as the plan places them"
    return pool, shots, list(groups.values())


# ---- the effect --------------------------------------------------------------------------------------

def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from ..core import framing as framing_mod
    from ..core import sound as soundmod
    from ..core.timeline import Timeline
    from . import Job, PlanError
    from .captions import head_in_output
    from .headline import keepout_in_output
    from .sound_effects import Event, Sounds, library

    info = clip.info
    fps = info.rate
    if plan is not None:
        kept = plan.get("options") or {}
        opts = dict(opts, **{k: kept[k] for k in ("style", "sound") if k in kept})
    fr = framing_mod.plan(info.width, info.height, fps, info.frames, clip.face, shape="keep")
    tl = Timeline.whole(info.frames / fps)
    W, H = fr.out_w, fr.out_h
    n_out = tl.output_frames(fps)
    seconds = tl.duration
    head_at, face_at = head_in_output(clip, tl, fr), keepout_in_output(clip, tl, fr)
    room = Room(W, H, [head_at(j) for j in range(n_out)], [face_at(j) for j in range(n_out)])
    words = clip.words
    warnings: list[str] = []
    notes: list[str] = []
    size_k = SIZE_K[opts["size"]]

    if plan is not None:
        pool, shots, groups = _from_plan(plan, seconds, fps)
        left: dict[int, str] = {}
        how = "the plan's pictures"
    else:
        pool, named = load_pool(opts["pictures"], parse_map(opts["map"]))
        shots, left, notes = decide(words, clip.quiet, pool, named, opts["on"], seconds, fps)
        how = {"keywords": "names, tags and the things you name", "map": "map, names and tags"}.get(
            opts["on"], "the times given")
        groups = group_shots(shots, opts["layout"], opts["hold"], words, clip.quiet, seconds)
        groups = place_groups(room, groups, pool, opts["layout"], size_k, fps, warnings)
        placed = {s.item for g in groups for s in g.shots}
        for s in shots:
            if s.item not in placed:
                left[s.item] = "no room clear of the face for it"
        shots = [s for g in groups for s in g.shots]
    if not shots:
        sheet = pictures_sheet(pool, [], left)
        raise O.OptionError("no picture landed: " + ("; ".join(f"{pool[i].label}: {why}" for i, why in sorted(left.items()))
                                                     or "nothing matched")
                            + (f". The pictures, with what each was taken for, are in {sheet}: look, then give a map "
                               "such as map=\"town=IMG_1.jpg\"" if sheet else ""))
    for s in shots:
        s.card = Card(pool[s.item], s.tile, opts["style"], (W, H), up=s.leave - s.land)
        win, start = getattr(s, "window", None), getattr(s, "window_from", None)
        if win and pool[s.item].kind != "graphic":
            k = pool[s.item].rgb.shape[1] / pool[s.item].width  # the plan's pixels are the file's own
            s.card.win = tuple(float(v) * k for v in win)
            s.card.windows = (tuple(float(v) * k for v in start), s.card.win) if start else None
            s.card.how = "moves as the plan says" if start else "as the plan crops it"
    painter = Painter(shots)

    # the pops, on the frames the pictures land
    sfx = None
    if opts["sound"] and info.has_audio:
        voice = soundmod.integrated(soundmod.blocks(clip.sound))
        reference = voice if voice > -70 else -23.0
        pop = library()["pop"]
        events = []
        for g in groups:
            for k, s in enumerate(sorted(g.shots, key=lambda s: s.land)):
                e = Event(s.start, "pop", f"“{s.word}” ({pool[s.item].label})", "picture")
                e.frame, e.at = s.frame, s.land
                e.gain_db = (reference - POP_UNDER) - pop.loudest
                e.pitch = float(2 ** (min(k, 4) * 1.5 / 12))  # each picture of a group a little higher
                events.append(e)
        sfx = Sounds(clip, opts, events, reference)

    # the checks
    sheet_path = pictures_sheet(pool, shots, left)
    rep = report(painter, room, n_out, fps)
    off_word = [s for s in shots if s.why not in ("time", "plan") and not any(abs(w.start - s.start) < 1e-6 for w in words)]
    lands_late = [s for s in shots if s.frame != int(math.ceil(s.start * fps - 1e-6))]
    smallest = min(math.sqrt(s.tile.w * s.tile.h) for s in shots) / W  # a square of the same area
    by_group = []
    for g in groups:
        names = ", ".join(f"“{s.word}”" for s in g.shots)
        count = (f"1 {g.where}" if len(g.shots) == 1 else f"{len(g.shots)} {g.where}" if g.where.startswith("as ")
                 else f"{len(g.shots)} in {g.where}")
        by_group.append(f"{g.shots[0].land:.2f}-{g.leave:.2f} s: {count} ({names})"
                        + (f"; {g.note}" if g.note else ""))
    checks = {
        "pictures": (f"{len(shots)} on {how}: " + "; ".join(
            f"{s.land:.2f} s “{s.word}” → {pool[s.item].label} ({'logo' if pool[s.item].logo else pool[s.item].kind}, "
            f"{s.why})" for s in shots)),
        "lands": ("each lands on the first frame of its word" if not off_word and not lands_late else
                  f"{len(off_word) + len(lands_late)} do not start on their word's first frame"),
        "placed": "; ".join(by_group),
        "size": (f"the smallest is as large as a square {smallest:.0%} of the frame's width" +
                 (" (reads on a phone)" if smallest >= READ else " (small on a phone: there is little room on this shot)")),
        "framing": "; ".join(dict.fromkeys(f"{pool[s.item].label}: {s.card.how}" for s in shots)),
        "not_used": "; ".join(f"{pool[i].label}: {why}" for i, why in sorted(left.items())) or "none",
        "sound": ("off" if sfx is None else
                  f"a soft pop as each lands, its loudest {POP_UNDER:g} dB under your voice; your voice as filmed"),
        "frames_over_face": rep["frames_over_face"],
        "frames_over_hair": rep["frames_over_hair"],
        "frames_outside_safe_zone": rep["frames_outside_safe_zone"],
        "pictures_sheet": (f"{sheet_path} (each picture, what it was taken for, the part shown and where it landed: "
                           "look, then map a picture to the word it shows)" if sheet_path else "-"),
        "warnings": warnings + notes,
    }

    def settled(sh: Shot) -> float:
        """When a picture has finished arriving: popped in, and a screen pushed in or a page scrolled."""
        t = sh.land + ENTER + 0.08
        if sh.card.windows is not None:
            t = sh.land + (1.3 if pool[sh.item].kind == "screen" else 0.6 + max(0.8, min(3.0, sh.leave - sh.land - 1.4)))
        return min(sh.leave - 0.2, t)

    def moments() -> list[tuple[int, float, int]]:
        """What shows the effect, as (how much it shows, when, which group): each picture just landed and
        settled, every group whole, the moment after it has gone, and just before the first."""
        cand = [(0, max(0.0, shots[0].land - 0.4), -1)]
        for k, g in enumerate(groups):
            for sh in g.shots:
                cand.append((2, sh.land + 0.14, k))  # just landed, still popping
                cand.append((3, settled(sh), k))
            cand.append((4, max(g.shots[-1].land + ENTER + 0.1, g.leave - 0.45), k))  # all of them, held
            if g.leave < seconds - 0.4:
                cand.append((1, g.leave + 0.25, k))  # gone
        return [(w, round(min(seconds - 2.0 / fps, max(0.0, t)), 3), k) for w, t, k in cand]

    def pick(n: int) -> list[float]:
        """`n` moments: every group whole first, then the ones that show most; none closer than a third
        of a second to another, in time order."""
        cand = moments()
        out: list[float] = []

        def take(t: float) -> None:
            if len(out) < n and all(abs(t - u) >= 0.33 for u in out):
                out.append(t)

        for w, t, k in cand:
            if w == 4:
                take(t)
        for w, t, k in sorted(cand, key=lambda c: (-c[0], c[1])):
            take(t)
        while len(out) < n:
            out.append(round(seconds * (len(out) + 0.5) / n, 3))
        return sorted(out)

    def sheet_times() -> list[float]:
        return pick(6)

    def landed_in_file(out: Path) -> str:
        """Each picture found in the finished file: none of it on the frame before its word, there on
        the word's own frame (measured in its tile, against the clip's own picture)."""
        from ..core import video

        vin = video.probe(str(out))
        ok = checked = 0
        late = []
        for sh in shots:
            j = sh.frame
            if j <= 0 or j >= min(vin.frames, info.frames):
                continue
            box = sh.tile.box()
            covered = any(o is not sh and (st := state(o, (j - 1) / fps)) is not None
                          and o.tile.box(st[0], dy=st[2]).intersects(box) for o in shots)
            if covered:  # another picture is up there the frame before (a swap): nothing to measure against
                continue
            x0, y0 = max(0, int(box.x0)), max(0, int(box.y0))
            x1, y1 = min(W, int(box.x1)), min(H, int(box.y1))

            def change(k: int) -> float:
                ours = video.read_frame(vin, k)[y0:y1, x0:x1].astype(np.float32)
                theirs = fr.apply(video.read_frame(info, k), k)[y0:y1, x0:x1].astype(np.float32)
                return float(np.abs(ours - theirs).mean())

            checked += 1
            if change(j - 1) < 6.0 and change(j) > 15.0:
                ok += 1
            else:
                late.append(f"“{sh.word}”")
        return (f"{ok} of {checked} first show on their word's own frame, none of them on the frame before"
                + (f" (not: {', '.join(late)})" if late else ""))

    out_plan = {
        "effect": NAME, "clip": clip.path, "options": opts,
        "output": {"width": W, "height": H, "fps": round(fps, 3), "seconds": round(seconds, 3), "framing": fr.note},
        "pictures": [{"file": pool[s.item].path, "kind": pool[s.item].kind, "logo": pool[s.item].logo, "word": s.word,
                      "why": s.why, "start": round(s.start, 4), "frame": s.frame, "land": round(s.land, 4),
                      "leave": round(s.leave, 3), "group": s.group, "layout": groups[s.group].layout,
                      "tile": s.tile.as_list(),
                      # the part shown, in the file's own pixels: where a screen pushes in to, where a page
                      # scrolls to; `from` is where that move starts (leave it out for a still crop)
                      "window": _file_px(pool[s.item], s.card.windows[1] if s.card.windows else s.card.win),
                      "from": _file_px(pool[s.item], s.card.windows[0]) if s.card.windows else None,
                      "how": s.card.how} for s in shots],
        "groups": [{"start": round(g.shots[0].land, 3), "leave": round(g.leave, 3), "layout": g.layout, "where": g.where,
                    "width": round(g.k, 3), "note": g.note} for g in groups],
        "not_used": [{"file": pool[i].path, "why": why} for i, why in sorted(left.items())],
        "sheet_times": sheet_times(),
    }

    def after(result: dict, out) -> dict:
        lines = {"pictures in the file": landed_in_file(Path(out))}
        if sfx is None:
            return lines
        rows = sfx.check_file(Path(out))
        offs = [abs(r["off_frames"]) for r in rows if r["off_frames"] is not None]
        unders = [r["under_voice_db"] for r in rows]
        lines["pops"] = (f"{sum(1 for o in offs if o < 1.0)} of {len(rows)} land within a frame of their picture in "
                         f"the encoded file" if offs else "no pop could be found again in the file")
        lines["pops under the voice"] = f"{min(unders):.1f} to {max(unders):.1f} dB" if unders else "-"
        return lines

    return Job(NAME, clip, tl, fr, painter.draw, out_plan, checks=checks, still_times=lambda: pick(4),
               after=after, sound=sfx.mix if sfx is not None else None)
