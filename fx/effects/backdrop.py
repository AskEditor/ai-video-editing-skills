"""backdrop: the speaker cut out and set in a new room that changes with what they say.

Four rooms (`mode`):

- **room**: a colour room, soft-lit from behind the head or flat, plain, squared or dotted;
- **pictures**: each picture fills the room behind you and gives way to the next on each idea;
- **grid**: four pictures round you, each landing on its idea;
- **screens**: screenshots floating at three depths behind your head, tilted, with a little parallax.

How it decides, in order:

1. **When.** Each picture (or each room colour, given as a list) lands on the next idea you name.
   An idea is the thing a phrase names, found from your words: "the lively town", "production
   quality". Verbs, filler and a word said again are not new ideas. A phrase lands on its first
   word, so the picture is there while the idea is heard. Times or your own words can be given
   instead (`at`).
2. **The edge.** The person is the core's cut-out made firm and steady for a whole new room
   (`core/matte.py`): no pale rim of the old room, solid inside, held still where nothing moves,
   and nobody but the speaker cut out with them.
3. **In the room, not on it.** The room's light wraps a little way round the edge (`rim`), the
   person casts a soft shadow onto the room on the side away from the light on their face
   (`shadow`), and the person's colour leans a little toward the room's (`match`), changing as the
   pictures change. A synthetic room gets the clip's own grain, so it does not look printed.
4. **Never over the face.** Everything is behind the person. The check counts the frames where
   the room would show inside the face (the cut-out missing part of it), which should be none.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..core import matte, moments
from ..core.depth import Card, draw_card, drift, head_path, rounded_alpha, seen_share  # noqa: F401  (shared)
from ..core import text as textmod
from ..core.layout import Rect
from ..core.words import HESITATIONS, Word, normalize
from . import options as O

NAME = "backdrop"
MODES = ("auto", "room", "pictures", "grid", "screens")
LAYOUTS = {"pictures": ("full", "card"), "grid": ("four",), "screens": ("trio", "pair", "single")}
TRANSITIONS = ("fade", "cut", "slide")

# The pack's room colours, chosen for it: soft studio tones and a few deep ones.
ROOM_COLOURS = {
    "lavender": (186, 172, 228), "sage": (170, 190, 160), "sand": (226, 208, 178), "sky": (166, 200, 232),
    "blush": (236, 190, 186), "mint": (176, 224, 204), "butter": (244, 224, 150), "cream": (242, 234, 218),
    "coral": (238, 136, 112), "forest": (34, 78, 60), "navy": (30, 42, 74), "plum": (86, 48, 84),
    "slate": (92, 106, 122), "charcoal": (40, 40, 44),
}
# The colours `color=auto` chooses between, in the order it prefers them when they stand off the
# person equally well.
AUTO_ROOMS = ("lavender", "sage", "sand", "sky", "blush", "forest", "charcoal")
DARK_ROOMS = ("charcoal", "slate", "navy")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}


# ---- the options -------------------------------------------------------------------------------

@dataclass(frozen=True)
class _RoomColours:
    help: str

    def parse(self, raw) -> str:
        v = str(raw).strip()
        if v.lower() in ("", "auto"):
            return "auto"
        out = []
        for part in [p.strip() for p in re.split(r"[,;]", v) if p.strip()]:
            out.append(_hex(_colour(part)))
        if len(out) > 12:
            raise O.OptionError("give at most twelve colours")
        return ", ".join(out)


def _colour(v: str) -> tuple[int, int, int]:
    key = v.strip().lower()
    if key in ROOM_COLOURS:
        return ROOM_COLOURS[key]
    try:
        return textmod.colour(key)
    except ValueError:
        raise O.OptionError(f"{v!r} is not a colour; use a name such as {', '.join(list(ROOM_COLOURS)[:6])}, or a hex "
                            "colour such as #b9a7e8") from None


def _hex(c) -> str:
    return "#%02x%02x%02x" % tuple(int(round(x)) for x in c)


@dataclass(frozen=True)
class _Gradient:
    help: str

    def parse(self, raw) -> str:
        v = str(raw).strip().lower()
        if v in ("", "soft", "auto"):
            return "soft"
        if v in ("flat", "none", "off"):
            return "flat"
        return _hex(_colour(v))


@dataclass(frozen=True)
class _Strength:
    """auto, off, or a number from 0 to 1."""
    help: str

    def parse(self, raw) -> str:
        v = str(raw).strip().lower()
        if v in ("", "auto"):
            return "auto"
        if v in ("off", "none", "no", "0"):
            return "0"
        try:
            s = float(v)
        except ValueError:
            raise O.OptionError(f"give auto, off, or a number from 0 to 1 such as 0.5 (got {raw!r})") from None
        if not 0.0 <= s <= 1.0:
            raise O.OptionError(f"give a number from 0 to 1 (got {raw!r})")
        return f"{s:g}"


@dataclass(frozen=True)
class _Pictures:
    help: str

    def parse(self, raw) -> str:
        return str(raw).strip()


@dataclass(frozen=True)
class _At:
    help: str

    def parse(self, raw) -> str:
        v = str(raw).strip().strip("\"“”")
        if v.lower() in ("", "auto"):
            return "auto"
        parts = [p.strip() for p in re.split(r"[,;]", v) if p.strip()]
        if not parts:
            return "auto"
        times = []
        for p in parts:
            try:
                times.append(float(p.rstrip("s")))
            except ValueError:
                times = None
                break
        if times is not None:
            if any(t < 0 for t in times):
                raise O.OptionError("times start at 0 seconds")
            if times != sorted(times):
                raise O.OptionError("give the times in order, such as 0, 2.4, 6")
            return ", ".join(f"{t:g}" for t in times)
        for p in parts:
            if not [normalize(x) for x in p.split() if normalize(x)]:
                raise O.OptionError(f"give seconds or words said in the clip (got {p!r})")
        return ", ".join(parts)


OPTIONS = {
    "mode": O.Choice(MODES, "auto: a colour room, or your pictures when you give some · room · pictures: each fills "
                            "the room behind you and changes on each idea · grid: four pictures round you · screens: "
                            "your screenshots floating in depth behind your head"),
    "pictures": _Pictures("your pictures: a folder, or files separated by commas (photos, screenshots or short "
                          "clips, which play muted)"),
    "at": _At("auto: each picture lands on the next idea you name · words you say, separated by commas, such as "
              "\"town, city, hotel\": one per picture · or seconds such as 0, 2.4, 6"),
    "color": _RoomColours("auto: a room colour that stands off your clothes and hair · a name such as lavender, "
                          "sage, sand, sky, blush, forest or navy · a hex colour such as #b9a7e8 · or several, "
                          "separated by commas, to change the room on each idea"),
    "gradient": _Gradient("soft: a pool of light behind you, deepening toward the edges · flat: one even colour · "
                          "or a second colour: the room runs from the first at the top to this at the bottom"),
    "pattern": O.Choice(("none", "grid-paper", "dots"), "none · grid-paper: fine squared lines · dots"),
    "layout": O.Choice(("auto", "full", "card", "four", "trio", "pair", "single"),
                       "auto, or for pictures: full (fills the room) or card (a framed picture behind your head) · "
                       "for screens: trio (three at three depths), pair (one either side) or single (one large)"),
    "transition": O.Choice(TRANSITIONS, "how one picture gives way to the next: fade, cut (on the word's own "
                                        "frame) or slide"),
    "move": O.Choice(("on", "off"), "on: pictures drift and push in slowly, screens float with a little parallax · "
                                    "off: everything behind you holds still"),
    "blur": _Strength("auto: photos a little soft, as a camera sees a room behind you; screens stay sharp · off · "
                      "or 0 to 1"),
    "rim": _Strength("auto: the room's light wraps softly round your edge · off · or 0 to 1 (from about 0.6 it "
                     "reads as a lit edge)"),
    "shadow": _Strength("auto: a soft shadow where you meet the room · off · or 0 to 1"),
    "match": O.Number(0.0, 1.0, "how far your colours lean toward the room's light: 0 leaves you as filmed, 1 is "
                                "the most; the default leans a little"),
    "cutout": O.Choice(("careful", "quick"), "careful: two looks at where you are, keeping only what both agree "
                                               "is you, so far fewer stray pieces of the room · quick: one look, "
                                               "about half the time to find you the first time"),
    "frame": O.Choice(("keep", "9:16"), "keep the clip's own shape, or make it 9:16 following the face"),
    "fix": O.Text("misheard words as wrong=right, separated by ;  e.g. \"Jon=John\""),
}

# The runs `tests/matrix.py backdrop` makes: the default room on every clip (F is the moving train,
# where a cut-out fails most), then each kind of room. Pictures come from the lab's own inputs:
# frames of our own B-roll and AskEditor's own product screens (`{lab}` is the lab).
_P = "{lab}/backdrop/inputs"
_SCREENS = "{lab}/../home-asks/product-screens"
MATRIX = [
    ("D", "D", {}),
    ("B", "B", {}),
    ("A", "A", {}),
    ("C", "C", {}),
    ("F", "F", {}),
    ("D-pictures", "D", {"pictures": f"{_P}/rooftops.jpg, {_P}/lake.jpg, {_P}/walkway.jpg"}),
    ("B-grid", "B", {"mode": "grid", "pictures": f"{_P}/plane.jpg, {_P}/rooftops.jpg, {_P}/station.jpg, "
                                                 f"{_P}/fields.jpg"}),
    ("D-screens", "D", {"mode": "screens", "pictures": _SCREENS}),
    ("A-screens", "A", {"mode": "screens", "pictures": _SCREENS}),
    ("C-pictures", "C", {"pictures": f"{_P}/lake.jpg, {_P}/fields.jpg, {_P}/rooftops.jpg", "transition": "cut"}),
    ("B-card", "B", {"layout": "card", "pictures": f"{_P}/plane.jpg, {_P}/rooftops.jpg"}),
    ("B-green-flat", "B", {"color": "forest", "gradient": "flat"}),
    ("B-colours", "B", {"color": "sky, blush, butter", "pattern": "grid-paper"}),
]


# ---- when: the ideas in the words ----------------------------------------------------------------
#
# A picture belongs to a thing named, not to a verb. The shared picker (`fx/core/moments.py`) ranks
# how well a word stands for its line, and gives nothing to a word the speaker took back, a first
# attempt said again or an aside; here a few more rules keep the doing words out, a phrase of words
# that belong together lands once on its first word, and a word said again is not a new idea.

_VERBS = set("""
improve improves improved edit edits edited help helps helped make makes build builds built create creates created
show shows grow grows learn learns use uses start starts work works run runs change changes find finds get gets
give gives take takes want wants need needs try tries love loves like likes watch watches
""".split())
_SUBJECTS = set("that which who it he she they we you i".split())
SEEN = 0.40  # a tile or a screen should stay at least this much in view over the clip
IDEA_MIN = 6.0  # a phrase's best word must score this to be an idea
GAP = 1.0  # seconds: two pictures never land closer together than this
EARLY = 0.8  # an idea this close to the start lands at 0: the first picture is there from the first frame


@dataclass
class Moment:
    at: float  # seconds into the output
    word: Optional[str] = None  # what was being said, or None
    why: str = ""  # "idea", "given", "between ideas", "spread"
    score: float = 0.0


def _idea_scores(words: list[Word], quiet, back: moments.TakenBack) -> list[float]:
    scores = [0.0] * len(words)
    for ln in moments.lines(words, quiet):
        for i in ln:
            s = moments.strength(words, i, ln, quiet, back)
            if s <= 0:
                continue
            k = words[i].key
            if k in _VERBS or (k.endswith("ing") and len(k) > 5):
                s -= 4
            if i > 0 and words[i - 1].key in _SUBJECTS and k.endswith("s"):
                s -= 3  # "an editor that edits": doing, not naming
            scores[i] = max(0.0, s)
    return scores


def ideas(words: list[Word], quiet) -> list[Moment]:
    """Every idea in the words, in order: a run of named words lands once, on its first word, with
    its best word's score. A word said again (or a phrase whose last word was) is not a new idea, and
    a word the speaker took back is none (`words`: the clip's whole words, fillers and all)."""
    back = moments.taken_back(words, quiet)
    words = [w for w in words if w.key not in HESITATIONS]
    scores = _idea_scores(words, quiet, back)
    out: list[Moment] = []
    seen: set[str] = set()
    i = 0
    while i < len(words):
        if scores[i] < 4:
            i += 1
            continue
        j = i
        while j + 1 < len(words) and scores[j + 1] >= 4 and not words[j].ends_clause:
            j += 1
        best = max(range(i, j + 1), key=lambda n: (scores[n], n))
        head = words[best].key
        phrase = " ".join(w.text.strip(",.;:!?") for w in words[i:j + 1])
        if scores[best] >= IDEA_MIN and head not in seen:
            out.append(Moment(float(words[i].start), phrase, "idea", float(scores[best])))
        seen.add(head)
        i = j + 1
    return out


def choose_moments(found: list[Moment], n: int, seconds: float) -> list[Moment]:
    """`n` moments: the strongest ideas, kept GAP apart and in order; when there are fewer ideas than
    pictures, the rest land in the middle of the longest stretches between them."""
    picked: list[Moment] = []
    for m in sorted(found, key=lambda m: (-m.score, -m.at)):
        if len(picked) >= n:
            break
        if all(abs(m.at - p.at) >= GAP for p in picked) and m.at < seconds - GAP:
            picked.append(m)
    picked.sort(key=lambda m: m.at)
    while len(picked) < n:
        marks = [0.0] + [m.at for m in picked] + [seconds - 0.6]
        gaps = [(marks[i + 1] - marks[i], i) for i in range(len(marks) - 1)]
        size, i = max(gaps)
        if size < 2 * GAP and picked:
            break
        t = marks[i] + size / 2 if (i > 0 or picked) else 0.0
        picked.append(Moment(round(t, 3), None, "between ideas" if found else "spread"))
        picked.sort(key=lambda m: m.at)
    if picked and picked[0].at <= EARLY:
        picked[0].at = 0.0
    return picked


def find_words(words: list[Word], said: list[str]) -> list[Moment]:
    """Where each of `said` is spoken, in order, each after the one before."""
    from .words_behind import find_said

    out, after = [], -1.0
    for s in said:
        rest = [w for w in words if w.start > after + 1e-3]
        span = find_said(rest, s)
        if span is None:
            raise ValueError(f"could not find \"{s}\" in what is said"
                             + (" after the words before it" if after >= 0 else "")
                             + "; `python -m fx.run words <video>` lists the words, or give seconds instead")
        w0 = rest[span[0]]
        out.append(Moment(float(w0.start), s, "given"))
        after = float(w0.start)
    return out


# ---- the pictures --------------------------------------------------------------------------------

@dataclass
class Picture:
    path: str
    kind: str  # "image" or "video"
    width: int
    height: int
    image: Optional[np.ndarray] = None  # RGB uint8, upright
    info: object = None  # a video's VideoInfo


def _natural(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def picture_paths(spec: str) -> list[Path]:
    """The files `spec` names: a folder's pictures and clips in name order, or a list of files."""
    if not spec:
        return []
    one = Path(os.path.expanduser(spec.strip().strip("\"'")))
    if one.is_dir():
        files = sorted((p for p in one.iterdir() if p.suffix.lower() in IMAGE_EXT | VIDEO_EXT
                        and not p.name.startswith(".")), key=lambda p: _natural(p.name))
        if not files:
            raise ValueError(f"there are no pictures or clips in {one}")
        return files
    out = []
    for part in [p.strip().strip("\"'") for p in re.split(r"[,;\n]", spec) if p.strip()]:
        p = Path(os.path.expanduser(part))
        if not p.exists():
            raise ValueError(f"there is no picture at {p}")
        if p.is_dir():
            out += picture_paths(str(p))
        elif p.suffix.lower() not in IMAGE_EXT | VIDEO_EXT:
            raise ValueError(f"{p.name} is not a picture or a clip this pack reads (photos: jpg, png, webp; clips: "
                             "mp4, mov). A HEIC photo needs saving as JPEG first")
        else:
            out.append(p)
    return out


def load_picture(path: Path) -> Picture:
    from PIL import Image, ImageOps

    from ..core import video

    if path.suffix.lower() in VIDEO_EXT:
        info = video.probe(path)
        return Picture(str(path), "video", info.width, info.height, info=info)
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA")
                bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                im = Image.alpha_composite(bg, im)
            rgb = np.asarray(im.convert("RGB"))
    except OSError:
        raise ValueError(f"{path.name} could not be read as a picture") from None
    return Picture(str(path), "image", rgb.shape[1], rgb.shape[0], image=rgb)


def cover(img: np.ndarray, w: int, h: int, focus: tuple[float, float] = (0.5, 0.45)) -> np.ndarray:
    """`img` scaled to cover w×h and cropped there, keeping the point `focus` (shares) in view."""
    import cv2

    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    nw, nh = max(w, int(math.ceil(iw * s))), max(h, int(math.ceil(ih * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    big = cv2.resize(img, (nw, nh), interpolation=interp)
    x0 = int(round(np.clip(focus[0] * nw - w / 2, 0, nw - w)))
    y0 = int(round(np.clip(focus[1] * nh - h / 2, 0, nh - h)))
    return np.ascontiguousarray(big[y0:y0 + h, x0:x0 + w])


class Source:
    """One picture as the backdrop draws it: a still, or a clip that plays (muted, looping) from the
    moment it lands. `frame(t)` gives it at `size`, `t` seconds after it landed."""

    def __init__(self, pic: Picture, size: tuple[int, int], blur_px: float = 0.0):
        import cv2

        self.pic, self.size = pic, size
        self.blur = blur_px
        self._still = None
        self._reader = None
        self._k_next = 0
        self._last = None
        if pic.kind == "image":
            img = cover(pic.image, size[0], size[1])
            if blur_px > 0.3:
                img = cv2.GaussianBlur(img, (0, 0), blur_px)
            self._still = img

    def frame(self, t: float) -> np.ndarray:
        if self._still is not None:
            return self._still
        return self._video_frame(max(0.0, t))

    def _video_frame(self, t: float) -> np.ndarray:
        import cv2

        from ..core import video

        info = self.pic.info
        t = t % max(0.1, info.duration - 1.0 / info.rate)
        k = int(t * info.rate)
        if self._reader is None or k < self._k_next - 1 or k > self._k_next + int(info.rate):
            if self._reader is not None:
                self._reader.close()
            w, h = info.width, info.height
            s = max(self.size[0] / w, self.size[1] / h)
            rw, rh = max(self.size[0], int(math.ceil(w * s / 2)) * 2), max(self.size[1], int(math.ceil(h * s / 2)) * 2)
            self._reader = video.FrameReader(info, size=(rw, rh), start=max(0.0, (k - 0.25) / info.rate))
            self._iter = iter(self._reader)
            self._k_next = k
            self._rs = (rw, rh)
        while self._k_next <= k:
            f = next(self._iter, None)
            if f is None:
                break
            self._last = f
            self._k_next += 1
        if self._last is None:
            self._last = np.zeros((self.size[1], self.size[0], 3), np.uint8)
        f = self._last
        if f.shape[1] != self.size[0] or f.shape[0] != self.size[1]:
            x0, y0 = (f.shape[1] - self.size[0]) // 2, int((f.shape[0] - self.size[1]) * 0.45)
            f = f[y0:y0 + self.size[1], x0:x0 + self.size[0]]
        if self.blur > 0.3:
            f = cv2.GaussianBlur(f, (0, 0), self.blur)
        return f

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None


# ---- drawing helpers -----------------------------------------------------------------------------

def _smooth(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def _ease_out(u: float, p: float = 3.0) -> float:
    u = min(1.0, max(0.0, u))
    return 1.0 - (1.0 - u) ** p


def _ease_back(u: float, k: float = 1.3) -> float:
    u = min(1.0, max(0.0, u))
    return 1.0 + (k + 1.0) * (u - 1) ** 3 + k * (u - 1) ** 2

def _lab(rgb) -> np.ndarray:
    import cv2

    a = np.asarray(rgb, np.float32).reshape(-1, 1, 3) / 255.0
    return cv2.cvtColor(a, cv2.COLOR_RGB2Lab).reshape(-1, 3)


def _luma(rgb) -> np.ndarray:
    c = np.asarray(rgb, np.float32)
    return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]


# ---- the room ------------------------------------------------------------------------------------

def room_colour_auto(edge_colours: Optional[np.ndarray], pool: tuple[str, ...] = AUTO_ROOMS) -> tuple[str, float]:
    """The room colour that stands off the person's own edge best: the one whose worst separation
    from the colours just inside the person's edge (the 10th percentile, in perceptual distance) is
    largest, preferring the earlier ones in `pool` when they are close."""
    if edge_colours is None or not len(edge_colours):
        return pool[0], 0.0
    lab = _lab(edge_colours)
    best, best_score, best_sep = pool[0], -1e9, 0.0
    for n, name in enumerate(pool):
        c = _lab([ROOM_COLOURS[name]])[0]
        sep = float(np.percentile(np.linalg.norm(lab - c, axis=1), 10))
        score = min(sep, 45.0) - 1.5 * n  # past 45 every room reads; then the order decides
        if score > best_score:
            best, best_score, best_sep = name, score, sep
    return best, best_sep


class Room:
    """The room behind everything: one colour, or one per moment, lit softly from behind the head,
    flat, or running into a second colour, with an optional pattern."""

    def __init__(self, size: tuple[int, int], colours: list[tuple[int, int, int]], gradient: str, pattern: str,
                 light_at: tuple[float, float]):
        self.W, self.H = size
        self.gradient, self.pattern = gradient, pattern
        self.light_at = light_at
        self._shade = self._light_map() if gradient != "flat" else None
        self.bases = [self._base(c) for c in colours]

    def _light_map(self) -> np.ndarray:
        """-1..1 per pixel: positive brightens toward a pool behind the head, negative deepens toward
        the edges and the floor."""
        W, H = self.W, self.H
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        lx, ly = self.light_at
        d = np.sqrt(((xs - lx) / (0.68 * W)) ** 2 + ((ys - ly) / (0.52 * H)) ** 2)
        u = np.clip(d / 1.35, 0, 1)
        m = 0.55 - 1.25 * (u * u * (3 - 2 * u))
        floor = np.clip((ys / H - 0.72) / 0.28, 0, 1)
        return np.clip(m - 0.25 * floor, -1, 1).astype(np.float32)

    def _base(self, colour) -> np.ndarray:
        W, H = self.W, self.H
        c = np.array(colour, np.float32)
        img = np.empty((H, W, 3), np.float32)
        img[:] = c
        if self.gradient not in ("soft", "flat"):  # a second colour, top to bottom
            c2 = np.array(textmod.hex_rgb(self.gradient), np.float32)
            v = np.linspace(0, 1, H, dtype=np.float32)
            v = v * v * (3 - 2 * v)
            img[:] = c[None, None, :] * (1 - v[:, None, None]) + c2[None, None, :] * v[:, None, None]
        if self._shade is not None:
            s = self._shade[..., None] * (0.55 if self.gradient == "soft" else 0.3)
            lighter = img + (255.0 - img) * 0.42
            darker = img * 0.62
            img = np.where(s >= 0, img + (lighter - img) * s, img + (img - darker) * s)
        if self.pattern != "none":
            img = self._pattern(img, c)
        return np.clip(img, 0, 255).astype(np.float32)

    def _pattern(self, img: np.ndarray, c: np.ndarray) -> np.ndarray:
        W, H = self.W, self.H
        dark_room = float(_luma(c)) < 110
        ink = (img + (255 - img) * 0.16) if dark_room else img * 0.86
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        if self.pattern == "grid-paper":
            cell = W / 13.5
            lw = max(1.0, W / 720)
            dx = np.abs(((xs - W / 2) % cell) - cell / 2)
            dy = np.abs(((ys - H * 0.3) % cell) - cell / 2)
            a = np.clip(1.0 - np.minimum(cell / 2 - dx, cell / 2 - dy) / lw + 0.5, 0, 1)
            a *= 0.85
        else:  # dots
            cell = W / 17.0
            r = max(1.2, W / 420)
            dx = ((xs - W / 2) % cell) - cell / 2
            dy = ((ys - H * 0.3) % cell) - cell / 2
            a = np.clip(r + 0.5 - np.sqrt(dx * dx + dy * dy), 0, 1) * 0.9
        return img + (ink - img) * a[..., None]

    def frame(self, mix: list[tuple[int, float]]) -> np.ndarray:
        """The room for this frame: `mix` is (base index, weight) pairs that add up to 1."""
        if len(mix) == 1 or mix[0][1] >= 0.999:
            return self.bases[mix[0][0]].copy()
        out = np.zeros_like(self.bases[0])
        for i, w in mix:
            if w > 0:
                out += self.bases[i] * w
        return out


class Grain:
    """The clip's own grain for what is drawn behind the person, so it is not cleaner than the
    picture in front. A few fields of noise are made once (seeded) and taken in turn."""

    def __init__(self, size: tuple[int, int], sigma: float, count: int = 6, seed: int = 7):
        W, H = size
        rng = np.random.default_rng(seed)
        self.sigma = float(sigma)
        h2, w2 = (H + 1) // 2, (W + 1) // 2
        # Half-size noise, brought up: softer than per-pixel noise, which the encoder throws away.
        import cv2

        self.fields = [cv2.resize(rng.standard_normal((h2, w2)).astype(np.float32), (W, H),
                                  interpolation=cv2.INTER_LINEAR) * 1.6 for _ in range(count)]

    def add(self, img: np.ndarray, j: int) -> np.ndarray:
        if self.sigma <= 0.05:
            return img
        f = self.fields[j % len(self.fields)]
        img += (f * self.sigma)[..., None]
        return img


# ---- the layouts ---------------------------------------------------------------------------------

LEAD = {"fade": 0.22, "cut": 0.0, "slide": 0.28}  # how long before its moment a change starts
SPAN = {"fade": 0.30, "cut": 0.0, "slide": 0.38}  # how long it runs


def _arrival(t: float, at: float, transition: str) -> float:
    """0 before a picture's change, 1 once it has landed."""
    if transition == "cut":
        return 1.0 if t >= at - 1e-6 else 0.0
    return _smooth((t - (at - LEAD[transition])) / SPAN[transition])


class FullPictures:
    """Each picture fills the room, from its moment to the next; a slow drift and push while it is up."""

    def __init__(self, size, sources: list[Source], moments: list[Moment], transition: str, move: bool,
                 room: Room, seconds: float, vignette: float = 0.14):
        import cv2

        self.W, self.H = size
        self.sources, self.moments = sources, moments
        self.transition, self.move = transition, move
        self.room, self.seconds = room, seconds
        ys, xs = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        d = np.sqrt(((xs - self.W / 2) / (0.75 * self.W)) ** 2 + ((ys - self.H * 0.42) / (0.75 * self.H)) ** 2)
        self.vig = (1.0 - vignette * np.clip(d - 0.35, 0, 1) ** 1.5).astype(np.float32)[..., None]
        self._cv2 = cv2

    def _view(self, k: int, t: float) -> np.ndarray:
        """Picture k at time t, drifting: alternate pictures push in and pull out, left and right."""
        cv2 = self._cv2
        src = self.sources[k]
        start = self.moments[k].at - LEAD[self.transition]
        end = self.moments[k + 1].at + 0.4 if k + 1 < len(self.moments) else self.seconds
        img = src.frame(t - self.moments[k].at)
        if not self.move:
            z, dx, dy = 1.0, 0.0, 0.0
        else:
            u = (t - start) / max(0.5, end - start)
            u = min(1.0, max(0.0, u))
            ein = u if k % 2 == 0 else 1 - u
            z = 1.0 + 0.07 * ein
            side = 1 if k % 4 in (0, 3) else -1
            dx = side * 0.022 * self.W * (u - 0.5)
            dy = -0.012 * self.H * (u - 0.5)
        sh, sw = img.shape[:2]
        cx, cy = sw / 2 + dx, sh / 2 + dy
        M = np.array([[z, 0, self.W / 2 - z * cx], [0, z, self.H / 2 - z * cy]], np.float32)
        return cv2.warpAffine(img, M, (self.W, self.H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT).astype(np.float32)

    def frame(self, t: float, j: int) -> np.ndarray:
        ms = self.moments
        shown = [k for k in range(len(ms)) if _arrival(t, ms[k].at, self.transition) > 0]
        if not shown:
            return self.room.frame([(0, 1.0)]), None
        top = shown[-1]
        w = _arrival(t, ms[top].at, self.transition)
        under = self._view(top - 1, t) if top > 0 else self.room.frame([(0, 1.0)])
        if w >= 0.999:
            out = self._view(top, t)
        elif self.transition == "slide":  # the picture arriving pushes the one before it out to the left
            e = _ease_out((t - (ms[top].at - LEAD["slide"])) / SPAN["slide"], 3.0)
            off = int(np.clip(round((1 - e) * self.W), 0, self.W))  # how much of the old picture is left
            new = self._view(top, t)
            out = np.empty_like(new)
            out[:, :off] = under[:, self.W - off:]
            out[:, off:] = new[:, :self.W - off]
        else:
            out = under * (1 - w) + self._view(top, t) * w
        out *= self.vig
        return out, None

class Grid:
    """Four pictures round the speaker's head, two by two; picture k takes tile k mod 4 on its
    moment, so the grid fills one idea at a time and then each new picture replaces the oldest."""

    def __init__(self, size, pictures: list[Picture], moments: list[Moment], move: bool, room: Room,
                 head: Rect, blur_px: float, coverage: Optional[np.ndarray] = None):
        import cv2

        W, H = size
        self.W, self.H = W, H
        self.moments, self.move, self.room = moments, move, room
        tw, th, cross_y, self.seen = self.place(size, coverage, head.cy)
        cross_x = W / 2
        gutter = 0.026 * W
        self.tiles = []
        for r in (0, 1):
            for c in (0, 1):
                cx = cross_x + (c - 0.5) * (tw + gutter)
                cy = cross_y + (r - 0.5) * (th + gutter)
                self.tiles.append(Card(cx, cy, tw, th, radius=0.035))
        self.tile_w, self.tile_h = int(round(tw)), int(round(th))
        self.alpha = rounded_alpha(self.tile_w, self.tile_h, 0.035 * tw)
        big = (int(round(tw * 1.08)), int(round(th * 1.08)))
        self.sources = [Source(p, big, blur_px) for p in pictures]
        self._cv2 = cv2

    @staticmethod
    def place(size: tuple[int, int], coverage: Optional[np.ndarray],
              head_cy: Optional[float] = None) -> tuple[float, float, float, list[float]]:
        """The tiles' size and where the grid's middle sits, measured against where the person is
        over the clip. The person stands in front of the grid, so the inner corners are meant to be
        behind them; but every tile must stay at least SEEN in view if it can. Then the largest
        tiles win, with the grid's middle as near the head's as they allow."""
        W, H = size
        gutter, margin = 0.026 * W, 0.045 * W
        tw = (W - 2 * margin - gutter) / 2
        head_y = head_cy if head_cy is not None else 0.4 * H
        best, best_key = None, None
        for aspect in (1.30, 1.15, 1.0, 0.85, 0.72, 0.6):
            th = tw * aspect
            lo, hi = 0.07 * H + th + gutter / 2, 0.84 * H - th - gutter / 2
            if hi < lo:
                continue
            for cy in np.linspace(lo, hi, 9):
                boxes = []
                for r in (0, 1):
                    for c in (0, 1):
                        x = W / 2 + (c - 0.5) * (tw + gutter)
                        y = cy + (r - 0.5) * (th + gutter)
                        boxes.append((x - tw / 2, y - th / 2, x + tw / 2, y + th / 2))
                seen = [seen_share(coverage, b, size) for b in boxes]
                ok = min(seen) >= SEEN
                key = (ok, aspect if ok else min(seen), -abs(cy - head_y))
                if best_key is None or key > best_key:
                    best, best_key = (tw, th, float(cy), [round(v, 2) for v in seen]), key
        return best

    def frame(self, t: float, j: int) -> tuple[np.ndarray, np.ndarray]:
        cv2 = self._cv2
        canvas = self.room.frame([(0, 1.0)])
        shade = np.zeros(canvas.shape[:2], np.float32)
        for slot, card in enumerate(self.tiles):
            ks = [k for k in range(slot, len(self.moments), 4) if t >= self.moments[k].at - 0.2]
            for k in ks[-2:]:  # the picture there, and the one it is replacing
                u = (t - (self.moments[k].at - 0.2)) / 0.34
                if u <= 0:
                    continue
                s = 0.86 + 0.14 * _ease_back(u, 1.2)
                op = _smooth(u * 1.6)
                life = max(0.0, t - self.moments[k].at)
                z = 1.0 + (0.045 * min(1.0, life / 8.0) if self.move else 0.0)
                img = self.sources[k].frame(life)
                sh, sw = img.shape[:2]
                cw, ch = self.tile_w, self.tile_h
                M = np.array([[z * cw / sw * 1.08, 0, 0], [0, z * ch / sh * 1.08, 0]], np.float32)
                M[0, 2] = cw / 2 - M[0, 0] * sw / 2
                M[1, 2] = ch / 2 - M[1, 1] * sh / 2
                tile = cv2.warpAffine(img, M, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                draw_card(canvas, shade, tile, self.alpha, card.corners(scale=s), opacity=op,
                          shadow=0.42, shadow_px=0.018 * self.W)
        canvas *= (1 - 0.55 * shade)[..., None]
        return canvas, shade


class Screens:
    """Screenshots floating behind the head at different depths, turned a little toward the middle.
    Screen k takes slot k mod the slots on its moment, coming forward out of the room."""

    def __init__(self, size, pictures: list[Picture], moments: list[Moment], move: bool, room: Room, head: Rect,
                 layout: str, head_path: Optional[np.ndarray], fps: float, coverage: Optional[np.ndarray] = None):
        W, H = size
        self.W, self.H = W, H
        self.moments, self.move, self.room = moments, move, room
        self.head_path, self.fps = head_path, fps
        self.focal = 1.6 * W
        hx = float(np.clip(head.cx, 0.3 * W, 0.7 * W))
        # Each slot: across (a share of the width), how high its middle sits against the head (a
        # share of the head's height from its top: below 0 is above the head), its width, tilt,
        # roll and depth. The head is meant to stand in front of part of each screen; a screen is
        # moved up only as far as it must be to stay SEEN in view.
        if layout == "single":
            plan = [(hx / W, 0.05, 0.94, -8, -1.0, 1.6)]
        elif layout == "pair":
            plan = [(0.22, 0.45, 0.66, 17, 1.4, 1.2), (0.78, 0.10, 0.66, -17, -1.4, 1.3)]
        else:  # trio: one near on each side, a little apart in height, and one far behind the head
            plan = [(0.20, 0.55, 0.62, 18, 1.6, 1.0), (0.80, 0.15, 0.62, -18, -1.6, 1.15),
                    (hx / W, -0.05, 0.70, 0, 0.0, 1.9)]
        self.slots, self.seen = [], []
        for fx_, fy, fw, tilt, roll, depth in plan:
            w = fw * W
            h = w * 0.625
            want = float(np.clip(head.y0 + fy * head.h, 0.06 * H + h / 2, 0.80 * H - h / 2))
            best = None
            for cy in np.arange(want, 0.06 * H + h / 2 - 1, -0.01 * H):
                card = Card(fx_ * W, float(cy), w, h, tilt=tilt, roll=roll, depth=depth, radius=0.018)
                q = card.corners(focal=self.focal)
                box = (float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max()))
                seen = seen_share(coverage, box, size)
                if best is None or seen > best[1] + 0.03:
                    best = (card, seen)
                if seen >= SEEN:
                    best = (card, seen)
                    break
            self.slots.append(best[0])
            self.seen.append(round(best[1], 2))
        self.cards = []  # per picture: its image and alpha, made for the slot it lands in
        for k, p in enumerate(pictures):
            slot = self.slots[k % len(self.slots)]
            self.cards.append(self._card(p, slot))

    def _card(self, p: Picture, slot: Card) -> tuple[np.ndarray, np.ndarray]:
        """The screenshot as a card: fitted to the slot's shape (the top of a tall one kept), a thin
        frame, round corners."""
        w, h = int(round(slot.w)), int(round(slot.h))
        if p.kind == "image":
            img = cover(p.image, w, h, focus=(0.5, 0.0))
        else:
            img = Source(p, (w, h)).frame(0.5)
        img = img.astype(np.float32)
        a = rounded_alpha(w, h, slot.radius * w)
        edge = max(1, int(round(w / 420)))  # a thin dark frame, so a white page still has an edge
        inner = np.zeros_like(a)
        inner[edge:-edge, edge:-edge] = rounded_alpha(w - 2 * edge, h - 2 * edge, max(0.0, slot.radius * w - edge))
        img = img * inner[..., None] + np.array([28, 28, 32], np.float32) * (1 - inner[..., None])
        return np.clip(img, 0, 255).astype(np.uint8), a.astype(np.float32)

    def _camera(self, t: float) -> tuple[float, float]:
        """The virtual camera's sideways and upward drift at t: a slow sway, plus a part of the head's
        own movement, so the screens move a little as a real room would behind someone filmed by hand."""
        if not self.move:
            return 0.0, 0.0
        return drift(t, (self.W, self.H), self.head_path, self.fps)

    def frame(self, t: float, j: int) -> tuple[np.ndarray, np.ndarray]:
        canvas = self.room.frame([(0, 1.0)])
        shade = np.zeros(canvas.shape[:2], np.float32)
        cam = self._camera(t)
        n = len(self.slots)
        # far slots first, so a nearer screen lies over a farther one
        order = sorted(range(n), key=lambda s: -self.slots[s].depth)
        for s in order:
            slot = self.slots[s]
            ks = [k for k in range(s, len(self.moments), n) if t >= self.moments[k].at - 0.42]
            for k in ks[-2:]:
                u = (t - (self.moments[k].at - 0.42)) / 0.52
                if u <= 0:
                    continue
                e = _ease_out(u, 3.0)
                scale = 0.80 + 0.20 * e
                op = _smooth(u * 1.5)
                bob = (0.005 * self.H * math.sin(2 * math.pi * t / 6.0 + 1.7 * s)) if self.move else 0.0
                dx, dy = cam[0] / slot.depth, cam[1] / slot.depth + bob
                quad = slot.corners(scale=scale, dx=dx, dy=dy + (1 - e) * 0.03 * self.H, focal=self.focal)
                img, a = self.cards[k]
                draw_card(canvas, shade, img, a, quad, opacity=op, shadow=0.5, shadow_px=0.02 * self.W,
                          dim=0.10 * (slot.depth - 1.0))
        canvas *= (1 - 0.6 * shade)[..., None]
        return canvas, shade


class CardPictures:
    """Each picture as a framed card on the room's wall above and behind the head, placed where it
    stays seen; the next lands over it."""

    def __init__(self, size, pictures: list[Picture], moments: list[Moment], move: bool, room: Room, head: Rect,
                 blur_px: float, coverage: Optional[np.ndarray] = None):
        W, H = size
        self.W, self.H = W, H
        self.moments, self.move, self.room = moments, move, room
        self.cards, self.seen = [], []
        cx = float(np.clip(head.cx, 0.40 * W, 0.60 * W))
        for k, p in enumerate(pictures):
            aspect = float(np.clip(p.width / p.height, 0.8, 1.6))
            best, best_key = None, None
            for share in (0.86, 0.78, 0.70, 0.62, 0.54):
                w = share * W
                h = w / aspect
                for cy in np.linspace(0.07 * H + h / 2, 0.80 * H - h / 2, 12):
                    seen = seen_share(coverage, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), size)
                    ok = seen >= 0.6
                    key = (ok, share if ok else seen, -abs(cy + h / 2 - head.cy))  # its foot near the head's middle
                    if best_key is None or key > best_key:
                        best, best_key = (w, h, float(cy), seen), key
            w, h, cy, seen = best
            card = Card(cx, cy, w, h, roll=(-1.2 if k % 2 else 1.0), radius=0.025)
            iw, ih = int(round(w)), int(round(h))
            border = max(2, int(round(0.022 * w)))
            frame_img = np.full((ih, iw, 3), 246, np.float32)
            inner = Source(p, (iw - 2 * border, ih - 2 * border), blur_px * 0.5).frame(0.5)
            frame_img[border:ih - border, border:iw - border] = inner
            self.cards.append((card, frame_img.astype(np.uint8), rounded_alpha(iw, ih, 0.025 * w)))
            self.seen.append(round(seen, 2))

    def frame(self, t: float, j: int) -> tuple[np.ndarray, np.ndarray]:
        canvas = self.room.frame([(0, 1.0)])
        shade = np.zeros(canvas.shape[:2], np.float32)
        shown = [k for k in range(len(self.moments)) if t >= self.moments[k].at - 0.25]
        if not shown:
            return canvas, shade
        top = shown[-1]
        u_top = (t - (self.moments[top].at - 0.25)) / 0.36
        for k in ([top - 1] if top > 0 and u_top < 1 else []) + [top]:  # the card before only while this lands
            u = (t - (self.moments[k].at - 0.25)) / 0.36
            e = _ease_out(u, 3.0)
            card, img, a = self.cards[k]
            drift = (0.01 * min(1.0, max(0.0, t - self.moments[k].at) / 6.0)) if self.move else 0.0
            quad = card.corners(scale=1.07 - 0.07 * e + drift)
            draw_card(canvas, shade, img, a, quad, opacity=_smooth(u * 1.4), shadow=0.5, shadow_px=0.022 * self.W)
        canvas *= (1 - 0.6 * shade)[..., None]
        return canvas, shade


class RoomOnly:
    """The room alone: one colour, or a colour per moment, each easing into the next."""

    def __init__(self, room: Room, moments: list[Moment], transition: str):
        self.room, self.moments, self.transition = room, moments, transition

    def frame(self, t: float, j: int) -> tuple[np.ndarray, None]:
        if not self.moments:
            return self.room.frame([(0, 1.0)]), None
        mix = [(0, 1.0)]
        for k, m in enumerate(self.moments):
            w = _arrival(t, m.at, self.transition)
            if w <= 0:
                break
            mix = [(i, v * (1 - w)) for i, v in mix] + [(k + 1, w)]
        mix = [(i, v) for i, v in mix if v > 1e-4]
        return self.room.frame(mix), None


# ---- the person in the room ----------------------------------------------------------------------

class Light:
    """How the person sits in the room: the soft shadow they cast on it, the room's light wrapping a
    little way round their edge, and their colour leaning toward the room's."""

    def __init__(self, size: tuple[int, int], rim: float, shadow: float, cast: bool, match: float,
                 side: float, person_mean: np.ndarray):
        self.W, self.H = size
        self.rim, self.shadow, self.cast, self.match = rim, shadow, cast, match
        self.side = side  # -1..1: the shadow falls this way (away from the brighter side of the face)
        self.person_mean = np.asarray(person_mean, np.float32)

    def shade(self, a: np.ndarray) -> Optional[np.ndarray]:
        """0..1: how much of the room's light the person blocks at each pixel (worked out at a
        quarter size: it is soft)."""
        import cv2

        if self.shadow <= 0.01:
            return None
        W, H = self.W, self.H
        q = cv2.resize(a, (max(2, W // 4), max(2, H // 4)), interpolation=cv2.INTER_AREA)
        qw = q.shape[1]
        near = cv2.GaussianBlur(q, (0, 0), 0.010 * qw, borderType=cv2.BORDER_REPLICATE)
        s = 0.55 * near
        if self.cast:
            dx, dy = self.side * 0.030 * qw, 0.022 * q.shape[0]
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            drop = cv2.warpAffine(q, M, (qw, q.shape[0]), borderMode=cv2.BORDER_REPLICATE)
            drop = cv2.GaussianBlur(drop, (0, 0), 0.030 * qw, borderType=cv2.BORDER_REPLICATE)
            s = 1 - (1 - s) * (1 - 0.6 * drop)
        s = np.clip(s * self.shadow, 0, 0.85)
        return cv2.resize(s, (W, H), interpolation=cv2.INTER_LINEAR)

    def gains(self, frame: np.ndarray, a: np.ndarray, room: np.ndarray) -> np.ndarray:
        """Per-channel gains for the person: their white balance and level moved part of the way
        toward the room's light, measured on the room round them. 1, 1, 1 at match 0."""
        import cv2

        if self.match <= 0.005:
            return np.ones(3, np.float32)
        qa = np.ascontiguousarray(a[4::16, 4::16])
        r = room[4::16, 4::16].reshape(-1, 3)
        q = qa.reshape(-1)
        near = cv2.GaussianBlur(qa, (0, 0), 3).reshape(-1)
        w = (1 - q) * (0.3 + near)  # the room, counted most where it is close to the person
        if w.sum() < 1:
            return np.ones(3, np.float32)
        room_mean = (r * w[:, None]).sum(0) / w.sum()
        lr = max(8.0, float(_luma(room_mean)))
        lp = max(8.0, float(_luma(self.person_mean)))
        tint = np.clip(room_mean / lr, 0.25, 4.0)
        g = tint ** (0.30 * self.match)
        # Skin carries little colour, so a tint that would barely show on a wall drains a face: a
        # deep blue sky at 0.35 once took half the colour out of one (clip D). The channels may
        # differ by at most 10% of `match` (3.5% at the default), which keeps skin's colour within
        # about a tenth of where it was.
        spread, cap = float(g.max() / g.min()), 1.0 + 0.10 * self.match
        if spread > cap:
            g = g ** (math.log(cap) / math.log(spread))
        g = g / max(1e-3, float(_luma(self.person_mean * g)) / lp)  # keep the person's level, then...
        g = g * float(np.clip((lr / lp) ** (0.12 * self.match), 0.86, 1.16))  # ...lean it toward the room's
        return np.clip(g, 0.8, 1.25).astype(np.float32)

    def wrap(self, out: np.ndarray, a: np.ndarray, room: np.ndarray) -> np.ndarray:
        """The room's light a little way round the person's edge: lighter rooms wrap more, a dark
        room barely at all, as real light would. From about 0.6 a thin lit edge comes in as well."""
        import cv2

        if self.rim <= 0.01:
            return out
        H, W = a.shape
        hw, hh = max(2, W // 2), max(2, H // 2)
        ah = cv2.resize(a, (hw, hh), interpolation=cv2.INTER_AREA)
        r1 = 0.0055 * hw
        inner = np.clip(ah * (1 - cv2.GaussianBlur(ah, (0, 0), r1, borderType=cv2.BORDER_REPLICATE)) * 2.4, 0, 1)
        thin = None
        if self.rim > 0.6:  # a lit edge: thinner, and toward white
            thin = np.clip(ah * (1 - cv2.GaussianBlur(ah, (0, 0), 0.45 * r1, borderType=cv2.BORDER_REPLICATE)) * 3, 0, 1)
        band = cv2.resize((inner > 0.01).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.nonzero(band)
        if not len(ys):
            return out
        rq = cv2.resize(room, (max(2, W // 4), max(2, H // 4)), interpolation=cv2.INTER_AREA)
        aq = cv2.resize(a, (rq.shape[1], rq.shape[0]), interpolation=cv2.INTER_AREA)
        wq = 1 - aq
        s = 0.012 * rq.shape[1]
        num = cv2.GaussianBlur(rq * wq[..., None], (0, 0), s, borderType=cv2.BORDER_REPLICATE)
        den = cv2.GaussianBlur(wq, (0, 0), s, borderType=cv2.BORDER_REPLICATE)
        light_q = num / np.maximum(den, 1e-3)[..., None]
        light = matte.sample(light_q, ys, xs, rq.shape[0] / H, rq.shape[1] / W)
        k = (min(self.rim, 0.6) * 0.85 * matte.sample(inner, ys, xs, hh / H, hw / W))[:, None]
        o = out[ys, xs]
        screen = 255.0 - (255.0 - o) * (255.0 - light) / 255.0
        o = o + k * (screen - o)
        if thin is not None:
            glow = light * 0.5 + 255.0 * 0.5
            o = o + ((self.rim - 0.6) * 1.6 * matte.sample(thin, ys, xs, hh / H, hw / W))[:, None] * (glow - o)
        out[ys, xs] = o
        return out


class Painter:
    """Draws each output frame: the room (or its pictures), the person's shadow on it, the person
    over it with a clean edge, their light matched, the room's light round their edge."""

    def __init__(self, clip, timeline, framing, layout, light: Light, grain: Grain, careful: bool = False,
                 usual_detail: float = 0.0):
        self.clip, self.timeline, self.framing = clip, timeline, framing
        self.layout, self.light, self.grain = layout, light, grain
        self.W, self.H = framing.out_w, framing.out_h
        # careful: the detailed model's view as well, and only what both see is kept (the core's cut-out)
        self.cut = matte.Cutout(clip, framing, careful=careful)
        self.masks, self.steady = self.cut.masks, self.cut.steady
        self.seen: dict[int, dict] = {}  # what the draw measured on some frames, for the checks
        self.usual_detail = usual_detail
        self.covered: set[int] = set()  # frames shown as filmed: the lens covered, nobody to cut out

    def person_alpha(self, frame: np.ndarray, k: int) -> np.ndarray:
        return self.cut.alpha(frame, k)

    def lens_covered(self, frame: np.ndarray, a: np.ndarray) -> bool:
        """Whether the lens is covered: almost nobody found, and far less detail than the clip
        usually has. On clip C a hand rushing at the lens leaves 40% of the usual detail and nobody
        found; pressed to it, 5%. A room left empty with its detail intact is not covered. All or
        nothing: half of each is a double exposure."""
        share = float(a[::8, ::8].mean())
        if self.usual_detail <= 0 or share >= 0.05:
            return False
        d = detail(frame)
        score = float(np.clip((0.6 - d / self.usual_detail) / 0.25, 0, 1)) * float(np.clip((0.05 - share) / 0.03, 0, 1))
        return score >= 0.5

    def draw(self, frame: np.ndarray, j: int, t: float) -> None:
        import cv2

        k = self.timeline.source_frame(j, self.clip.info.rate, self.clip.info.frames)
        a = self.person_alpha(frame, k)
        room, _ = self.layout.frame(t, j)
        room = self.grain.add(room, j)
        shade = self.light.shade(a)
        if shade is not None:
            room *= (1 - shade)[..., None]
        g = self.light.gains(frame, a, room)
        out = matte.over(frame, a, room, gains=g if self.light.match > 0.005 else None)
        out = self.light.wrap(out, a, room)
        # A hand over the lens leaves nobody to cut out and almost no detail: an empty room there
        # reads as the person vanishing (clip C, 9.2 s), so such a frame shows as filmed instead.
        if self.lens_covered(frame, a):
            out = frame.astype(np.float32)
            self.covered.add(j)
        frame[:] = cv2.convertScaleAbs(out)  # rounds and clamps to 0..255
        self.seen[j] = {"gains": g}


# ---- measuring the clip --------------------------------------------------------------------------

def measure(clip, framing, size: tuple[int, int], count: int = 12) -> dict:
    """A few frames read once: where the head is over the clip, how often the person covers each
    part of the frame (where pictures stay seen), the colours just inside the person's edge (what the
    room must stand off), which side of the face is lit, the grain of the old room, and the person's
    mean colour."""
    import cv2

    from ..core import video

    W, H = size
    info = clip.info
    masks = clip.person()
    heads, edge_cols, sides, noise, means, details = [], [], [], [], [], []
    cover_sum, cover_n = np.zeros((max(2, H // 8), max(2, W // 8)), np.float32), 0
    rng = np.random.default_rng(3)
    for k in np.linspace(0, info.frames - 1, count).astype(int):
        k = int(k)
        f = framing.apply(video.read_frame(info, k), k)
        details.append(detail(f))
        crop = framing.window(k) if framing.reframed else None
        a = masks.full(masks.at(k), (W, H), guide=f, crop=crop, lean=0.5)
        head = clip.face.head(k)
        if head is not None:
            heads.append(framing.to_output(head, k).as_list())
        hard = (a > 0.5).astype(np.uint8)
        cover_sum += cv2.resize(hard.astype(np.float32), (cover_sum.shape[1], cover_sum.shape[0]),
                                interpolation=cv2.INTER_AREA)
        cover_n += 1
        if hard.sum() < 100:
            continue
        rim = (hard - cv2.erode(hard, np.ones((7, 7), np.uint8))).astype(bool)
        rim[: int(0.02 * H)] = False
        px = f[rim]
        if len(px):
            edge_cols.append(px[rng.choice(len(px), min(600, len(px)), replace=False)])
        means.append(f[a > 0.9].reshape(-1, 3).mean(0) if (a > 0.9).any() else np.array([128, 128, 128]))
        box = clip.face.box(k)
        if box is not None:
            b = framing.to_output(box, k)
            x0, x1, y0, y1 = int(max(0, b.x0)), int(min(W, b.x1)), int(max(0, b.y0)), int(min(H, b.y1))
            if x1 - x0 > 8 and y1 - y0 > 8:
                face = _luma(f[y0:y1, x0:x1].astype(np.float32))
                mid = (x0 + x1) // 2 - x0
                left, right = float(np.median(face[:, :mid])), float(np.median(face[:, mid:]))
                sides.append((right - left) / max(1.0, right + left))
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32)
        hp = g - cv2.GaussianBlur(g, (0, 0), 1.5)
        flat = (a < 0.05) & (np.abs(cv2.GaussianBlur(g, (0, 0), 4) - cv2.GaussianBlur(g, (0, 0), 1.5)) < 1.5)
        if flat.sum() > 2000:
            noise.append(float(np.std(hp[flat])))
    head = None
    if heads:
        hs = np.array(heads)
        head = Rect(*np.median(hs, axis=0).tolist())
    return {"head": head, "edge_colours": np.concatenate(edge_cols) if edge_cols else None,
            "coverage": cover_sum / max(1, cover_n),
            "detail": float(np.median(details)) if details else 0.0,
            "side": float(np.median(sides)) if sides else 0.0,
            "noise": float(np.median(noise)) if noise else 1.5,
            "person_mean": np.median(np.array(means), axis=0) if means else np.array([128.0, 128.0, 128.0])}


def detail(frame: np.ndarray) -> float:
    """How much fine detail a frame holds (the mean edge strength of an eighth-size grey copy): a
    hand over the lens, or the lens pressed to something, holds almost none."""
    import cv2

    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(8, w // 8), max(8, h // 8)), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.abs(cv2.Laplacian(g, cv2.CV_32F)).mean())



# ---- deciding ------------------------------------------------------------------------------------

def _strength(v: str, auto: float) -> float:
    return auto if v == "auto" else float(v)


def resolve_mode(mode: str, layout: str, n_pictures: int) -> tuple[str, str]:
    if mode == "auto":
        mode = "pictures" if n_pictures else "room"
    if mode != "room" and not n_pictures:
        raise ValueError(f"{mode} needs your pictures: pictures=\"~/my-pictures\" (a folder) or a list of files "
                         "separated by commas")
    if mode == "room":
        if layout not in ("auto",):
            raise ValueError("layout is for pictures, grid and screens; a room has none")
        return mode, "room"
    allowed = LAYOUTS[mode]
    if layout == "auto":
        if mode == "screens":
            layout = "single" if n_pictures == 1 else "pair" if n_pictures == 2 else "trio"
        else:
            layout = allowed[0]
    if layout not in allowed:
        raise ValueError(f"layout={layout} is not one of {mode}'s: {', '.join(allowed)}")
    return mode, layout


def decide_moments(clip, at: str, needed: int, seconds: float) -> tuple[list[Moment], str]:
    """`needed` moments from `at`: times, words said, or the ideas in the words."""
    if needed <= 0:
        return [], ""
    parts = [p.strip() for p in at.split(",")] if at != "auto" else []
    if parts:
        try:
            times = [float(p) for p in parts]
        except ValueError:
            times = None
        if len(parts) != needed:
            raise ValueError(f"at gives {len(parts)} moments for {needed} "
                             f"{'pictures' if needed != 1 else 'picture'}; give one each")
        if times is not None:
            if times[-1] >= seconds:
                raise ValueError(f"at {times[-1]:g} s is past the end of the clip ({seconds:.2f} s)")
            return [Moment(t, None, "given") for t in times], "the times given"
        return find_words([w for w in clip.words if w.key not in HESITATIONS], parts), "the words given"
    words = clip.words
    found = ideas(words, clip.quiet) if words else []
    picked = choose_moments(found, needed, seconds)
    if len(picked) < needed:
        raise ValueError(f"{needed} changes do not fit in {seconds:.1f} s at least {GAP:g} s apart; give fewer "
                         "pictures, or times with at=")
    how = "the ideas in the words" if found else "spread evenly: no idea was clear enough in the words"
    return picked, how


def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from ..core import framing as framing_mod
    from ..core.timeline import Timeline
    from . import Job, PlanError

    info = clip.info
    fps = info.rate
    if plan is not None:
        kept = plan.get("options") or {}
        opts = dict(opts, **{k: kept[k] for k in ("frame", "transition", "move", "pattern", "gradient", "cutout")
                             if k in kept})
    fr = framing_mod.plan(info.width, info.height, fps, info.frames, clip.face, shape=opts["frame"])
    tl = Timeline.whole(info.frames / fps)
    W, H = fr.out_w, fr.out_h
    seconds = tl.duration
    n_out = tl.output_frames(fps)

    # what is behind: the pictures, the mode and the layout
    if plan is not None:
        try:
            paths = [Path(p["path"]) for p in plan.get("pictures", [])]
            mode, layout = str(plan["mode"]), str(plan["layout"])
        except (KeyError, TypeError):
            raise PlanError("the plan needs mode, layout and pictures; run once without --plan to get one") from None
        if mode not in MODES[1:] or (mode != "room" and layout not in LAYOUTS.get(mode, ())):
            raise PlanError("the plan names a mode or layout this effect does not have")
        for p in paths:
            if not p.exists():
                raise PlanError(f"the plan's picture {p} is not there any more")
    else:
        paths = picture_paths(opts["pictures"])
        mode, layout = resolve_mode(opts["mode"], opts["layout"], len(paths))
    warnings = []
    if mode == "room" and paths:
        warnings.append("pictures are not used in a room; say mode=pictures, grid or screens to use them")
        paths = []
    pics = [load_picture(p) for p in paths]

    stats = measure(clip, fr, (W, H))
    head = stats["head"] or Rect(0.3 * W, 0.2 * H, 0.7 * W, 0.45 * H)

    # the room's colours
    if plan is not None:
        colours = [textmod.hex_rgb(c) for c in plan["room"]["colours"]]
        colour_why = "as the plan says"
    elif opts["color"] == "auto":
        pool = DARK_ROOMS if mode == "screens" else AUTO_ROOMS  # light screens read best in a deep room
        best, best_sep = room_colour_auto(stats["edge_colours"], pool)
        colours = [ROOM_COLOURS[best]]
        colour_why = f"{best}, chosen to stand off the colours at your edge ({best_sep:.0f} apart at the closest)"
    else:
        colours = [textmod.hex_rgb(c) for c in opts["color"].split(", ")]
        colour_why = "as asked"
    if mode != "room" and len(colours) > 1:
        warnings.append("a list of colours changes a room; with pictures only the first colour is used")
        colours = colours[:1]

    # the moments: one per picture, or one per room colour after the first
    needed = len(pics) if mode != "room" else len(colours) - 1
    if plan is not None:
        try:
            if mode == "room":
                moments = [Moment(float(m["at"]), m.get("word"), "as the plan says") for m in plan.get("changes", [])]
            else:
                moments = [Moment(float(p["at"]), p.get("word"), "as the plan says") for p in plan["pictures"]]
        except (KeyError, TypeError, ValueError):
            raise PlanError("each picture in the plan needs its path and at") from None
        if len(moments) != needed:
            raise PlanError("the plan's changes do not match its pictures or colours")
        how = "the plan's moments"
    else:
        moments, how = decide_moments(clip, opts["at"], needed, seconds)
    for m in moments:
        if not 0 <= m.at < seconds:
            raise PlanError(f"a change at {m.at} s is outside the clip (0–{seconds:.2f} s)")

    # strengths
    blur_s = _strength(opts["blur"], {"pictures": 0.15, "grid": 0.12}.get(mode, 0.0))
    if plan is not None:
        lp = plan.get("light", {})
        rim, shadow, match = float(lp.get("rim", 0.35)), float(lp.get("shadow", 0.5)), float(lp.get("match", 0.35))
        cast = bool(lp.get("cast", True))
    else:
        rim = _strength(opts["rim"], 0.35)
        cast = not (mode == "pictures" and layout == "full")
        shadow = _strength(opts["shadow"], 0.5 if cast else 0.3)
        match = float(opts["match"])
    side = float(np.clip(-stats["side"] * 6.0, -1.0, 1.0))  # away from the brighter side of the face

    # the room and what stands in it
    light_at = (head.cx, max(0.05 * H, head.y0 - 0.02 * H))
    room = Room((W, H), colours, opts["gradient"], opts["pattern"], light_at)
    move = opts["move"] == "on"
    blur_px = blur_s * 12.0 * W / 1080.0
    if mode == "room":
        layout_obj = RoomOnly(room, moments, opts["transition"])
    elif mode == "pictures" and layout == "full":
        big = (int(math.ceil(W * 1.10 / 2)) * 2, int(math.ceil(H * 1.10 / 2)) * 2)
        layout_obj = FullPictures((W, H), [Source(p, big, blur_px) for p in pics], moments, opts["transition"],
                                  move, room, seconds)
    elif mode == "pictures":
        layout_obj = CardPictures((W, H), pics, moments, move, room, head, blur_px, stats["coverage"])
    elif mode == "grid":
        layout_obj = Grid((W, H), pics, moments, move, room, head, blur_px, stats["coverage"])
    else:
        layout_obj = Screens((W, H), pics, moments, move, room, head, layout,
                             head_path(clip, fr, n_out, fps, head), fps, stats["coverage"])
    seen = getattr(layout_obj, "seen", None)
    if seen and min(seen) < 0.35:
        warnings.append(f"the shot is close: the least-seen {'tile' if mode == 'grid' else 'screen'} stays only "
                        f"{min(seen):.0%} in view; a shot with room above and beside you shows more")
    grain = Grain((W, H), float(np.clip(stats["noise"], 0.6, 3.5)))
    light = Light((W, H), rim, shadow, cast, match, side, stats["person_mean"])
    painter = Painter(clip, tl, fr, layout_obj, light, grain, careful=opts.get("cutout") == "careful",
                      usual_detail=stats["detail"])

    # the checks: the person's face never shows the room through it
    over_face, sampled, worst = face_check(painter, clip, fr, n_out)
    skin = skin_check(painter, clip, fr, moments, seconds, fps)
    landed = [f"{m.at:.2f} s" + (f" (“{m.word}”)" if m.word else f" ({m.why})") for m in moments]
    checks = {
        "mode": f"{mode}" + (f", {layout}" if mode != "room" else "") + (f", {len(pics)} pictures" if pics else ""),
        "room": colour_why + (f"; {len(colours)} colours" if len(colours) > 1 else ""),
        "changes": (f"{len(moments)} on {how}: " + ", ".join(landed)) if moments else "none: one room throughout",
        "frames_over_face": int(round(over_face * n_out / max(1, sampled))),
        "frames_outside_safe_zone": 0,
        "placed": ("the room behind you, everywhere" if mode == "room" else
                   "each picture fills the room behind you" if layout == "full" else
                   f"{'tiles' if mode == 'grid' else 'screens' if mode == 'screens' else 'the card'} stay "
                   + ", ".join(f"{v:.0%}" for v in (seen or [1.0])) + " in view over the clip"),
        "face_kept": f"the room shows through at most {worst:.1%} of the face on {sampled} frames checked",
        "light": f"rim {rim:g}, shadow {shadow:g}" + (" cast away from the light" if cast else " close round you")
                 + f", match {match:g}" + (f"; {skin}" if skin else ""),
        "warnings": warnings,
    }
    out_plan = {
        "effect": NAME, "clip": clip.path, "options": opts,
        "output": {"width": W, "height": H, "fps": round(fps, 3), "seconds": round(seconds, 3), "framing": fr.note},
        "mode": mode, "layout": layout,
        "room": {"colours": [_hex(c) for c in colours], "gradient": opts["gradient"], "pattern": opts["pattern"]},
        "pictures": [{"path": p.path, "at": round(m.at, 3), "word": m.word, "why": m.why}
                     for p, m in zip(pics, moments)],
        "changes": [{"at": round(m.at, 3), "word": m.word, "why": m.why} for m in moments] if mode == "room" else [],
        "light": {"rim": rim, "shadow": shadow, "cast": cast, "match": match, "shadow_side": round(side, 2)},
        "grain": round(grain.sigma, 2), "blur_px": round(blur_px, 2),
        "head": head.as_list(),
        "sheet_times": sheet_times(moments, seconds, fps, n_out),
    }

    def still_times() -> list[float]:
        ts = [0.0] + [m.at + 0.35 for m in moments[:3]]
        if len(ts) < 4:
            ts += [seconds * f for f in (0.5, 0.85)][: 4 - len(ts)]
        return [round(min(seconds - 1.0 / fps, max(0.0, v)), 3) for v in ts]

    def after(result: dict, out_path) -> dict:
        if not painter.covered:
            return {}
        n = len(painter.covered)
        return {"covered lens": f"{n} frames ({n / fps:.2f} s) had nobody to cut out and almost no detail (a hand "
                                "over the lens): they show as filmed"}

    return Job(NAME, clip, tl, fr, painter.draw, out_plan, checks=checks, still_times=still_times, after=after)


def face_check(painter: Painter, clip, framing, n_out: int, count: int = 16) -> tuple[int, int, float]:
    """On `count` frames, the share of the face (the face's own box, a little inside it) where the
    room would show: frames over 2% count as frames over the face."""
    from ..core import video

    bad, n, worst = 0, 0, 0.0
    for j in np.linspace(0, n_out - 1, count).astype(int):
        k = painter.timeline.source_frame(int(j), clip.info.rate, clip.info.frames)
        box = clip.face.box(k)
        if box is None:
            continue
        f = framing.apply(video.read_frame(clip.info, k), k)
        a = painter.person_alpha(f, k)
        if painter.lens_covered(f, a):
            continue  # a covered lens: the frame shows as filmed, so no room can show over the face
        share = matte.face_shown(a, framing.to_output(box, k))
        if share is None:
            continue
        worst = max(worst, share)
        bad += share > 0.02
        n += 1
    return bad, n, worst


def skin_check(painter: Painter, clip, framing, moments: list[Moment], seconds: float, fps: float) -> str:
    """How far the match moves the skin's hue and colour, measured on the face at a few moments."""
    import cv2

    from ..core import video

    if painter.light.match <= 0.005:
        return ""
    times = [min(seconds - 1 / fps, m.at + 0.5) for m in moments[:4]] or [seconds / 2]
    hue, chroma = [], []
    for t in times:
        j = int(round(t * fps))
        k = painter.timeline.source_frame(j, clip.info.rate, clip.info.frames)
        box = clip.face.box(k)
        if box is None:
            continue
        f = framing.apply(video.read_frame(clip.info, k), k)
        a = painter.person_alpha(f, k)
        room, _ = painter.layout.frame(t, j)
        g = painter.light.gains(f, a, room)
        b = framing.to_output(box, k)
        x0, x1, y0, y1 = int(max(0, b.x0)), int(min(painter.W, b.x1)), int(max(0, b.y0)), int(min(painter.H, b.y1))
        face = f[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
        if not len(face):
            continue
        before = _lab([face.mean(0)])[0]
        after = _lab([np.clip(face.mean(0) * g, 0, 255)])[0]
        h0, h1 = math.degrees(math.atan2(before[2], before[1])), math.degrees(math.atan2(after[2], after[1]))
        hue.append(abs((h1 - h0 + 180) % 360 - 180))
        chroma.append(math.hypot(after[1], after[2]) / max(1e-3, math.hypot(before[1], before[2])))
    if not hue:
        return ""
    return f"skin hue moved {max(hue):.1f}° at most, its colour ×{min(chroma):.2f}–×{max(chroma):.2f}"


def sheet_times(moments: list[Moment], seconds: float, fps: float, n_out: int) -> list[float]:
    """Six moments across the clip: the start, each change once it has landed, and between them."""
    ts = [0.3]
    ts += [m.at + 0.4 for m in moments]
    while len(ts) < 6:
        marks = sorted(ts) + [seconds]
        gaps = [(marks[i + 1] - marks[i], i) for i in range(len(marks) - 1)]
        size, i = max(gaps)
        ts.append(marks[i] + size / 2)
    ts = sorted(ts)
    if len(ts) > 6:
        keep = [ts[0]] + [ts[1 + round(i * (len(ts) - 2) / 4)] for i in range(5)]
        ts = keep
    return [round(min((n_out - 1) / fps, max(0.0, t)), 3) for t in ts]
