"""reframe: a landscape or square video made vertical (or 4:5, square or 16:9), the face kept in frame.

How it decides:

1. **The picture.** Black bars filmed or exported into the clip are found first and left out, so
   no frame ever shows a strip of black.
2. **Whom to follow.** One person: their face, from the face track. Two people: whoever is talking,
   found by which mouth moves with the sound (`core/speakers.py`); each change of speaker is a cut
   to the other face when they are too far apart to glide between, and lands in a pause. When the
   mouths cannot tell, it says so and follows the larger face. Both faces fit in the frame
   together: both are kept in it.
3. **How the frame moves** (`core/framing.py`): it holds still while the face stays near where it
   is, so a nod or a lean moves nothing; when the face settles somewhere new, it eases there, a
   little before the head arrives; where the clip itself cuts, it cuts too. The face never leaves
   the frame, and when the frame is taller than the clip's (square to 16:9, say) it keeps a tenth
   of the frame above the top of the head, or, when the head cannot fit, keeps the face whole and
   says the top of the hair is cut.
4. **Every shape at once.** `format=all` makes one file for each shape the video is not already,
   each with its own path and its own checks.

`mode=center` holds a still crop from the middle, and `mode=fill` keeps the whole picture, fitted
inside the shape over a blurred, darkened copy of itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..core import framing as framing_mod
from ..core import video
from ..core.layout import Rect
from ..core.timeline import Timeline
from . import options as O

NAME = "reframe"
FORMATS = ("9:16", "4:5", "1:1", "16:9")

OPTIONS = {
    "format": O.Choice(FORMATS + ("all",), "9:16 for Reels, Shorts and TikTok · 4:5 for the feed · 1:1 square · "
                                           "16:9 for YouTube · all: every shape the video is not already, one file each"),
    "mode": O.Choice(("track", "center", "fill"), "track: the frame follows the face · center: a still crop from the "
                                                  "middle · fill: the whole picture, fitted over a blurred copy of itself"),
    "speaker": O.Choice(("auto", "largest"), "with two people in the picture: auto follows whoever is talking; "
                                             "largest follows the bigger face"),
    "size": O.Choice(("auto", "full"), "auto never enlarges the picture; full makes each shape its full size "
                                       "(1080 wide, or 1920 wide for 16:9)"),
}

# The runs `tests/matrix.py reframe` makes: our own landscape and square takes, the
# side-by-side pair for the speaker, then the upright test clips into the other shapes.
MATRIX = [
    ("wide", "wide", {}),
    ("wide-all", "wide", {"format": "all"}),
    ("wide-center", "wide", {"mode": "center"}),
    ("wide-fill", "wide", {"mode": "fill"}),
    ("square", "square", {}),
    ("square-all", "square", {"format": "all"}),
    ("two", "two", {}),
    ("two-largest", "two", {"speaker": "largest"}),
    ("B-4x5", "B", {"format": "4:5"}),
    ("C-1x1", "C", {"format": "1:1"}),
    ("F-16x9", "F", {"format": "16:9"}),
]

BAR_LEVEL = 18  # a row or column this dark (0-255) in every frame looked at is a filmed-in bar
TWO_SHARE = 0.3  # a second face in this share of frames, about as big as the first, makes two people


def label(shape: str) -> str:
    return shape.replace(":", "x")


# ---- the picture and whom to follow ----------------------------------------------------------------

def bars(info) -> tuple[int, int, int, int]:
    """The part of the frame that is picture (x0, y0, x1, y1): rows and columns at the edges that
    are black in every one of a dozen frames across the clip are left out."""
    picks = np.linspace(0, info.frames - 1, min(12, info.frames)).astype(int)
    w = min(480, info.width)
    h = max(2, int(round(info.height * w / info.width)))
    top = np.zeros(h)
    side = np.zeros(w)
    for k in picks:
        g = video.read_frame(info, int(k), size=(w, h)).mean(axis=2)
        top = np.maximum(top, g.max(axis=1))
        side = np.maximum(side, g.max(axis=0))

    def span(v: np.ndarray) -> tuple[int, int]:
        lit = np.flatnonzero(v > BAR_LEVEL)
        if not len(lit):
            return 0, len(v)
        return int(lit[0]), int(lit[-1]) + 1

    y0, y1 = span(top)
    x0, x1 = span(side)
    sx, sy = info.width / w, info.height / h
    # Rounded inwards, with a pixel or two to spare: a sliver of picture lost does not show, a line
    # of black left at the edge does.
    x0, x1 = int(np.ceil((x0 + 1) * sx / 2) * 2), int(min(info.width, np.floor((x1 - 1) * sx / 2) * 2))
    y0, y1 = int(np.ceil((y0 + 1) * sy / 2) * 2), int(min(info.height, np.floor((y1 - 1) * sy / 2) * 2))
    # Only real bars count: a few pixels of dark edge is the picture itself.
    if x0 < 0.02 * info.width:
        x0 = 0
    if info.width - x1 < 0.02 * info.width:
        x1 = info.width
    if y0 < 0.02 * info.height:
        y0 = 0
    if info.height - y1 < 0.02 * info.height:
        y1 = info.height
    return x0, y0, x1, y1


class Follow:
    """Whom the frame follows: the face per frame, the clip's own cuts and the speaker changes."""

    def __init__(self, faces, cuts, note: str, turns: Optional[list] = None, people: int = 1,
                 talk=None, both: bool = False):
        self.faces, self.cuts, self.note = faces, cuts, note
        self.turns, self.people, self.talk, self.both = turns or [], people, talk, both


def _two_people(clip) -> Optional[list]:
    """Two or more people in the picture, as the speakers module sees them; None for one person."""
    from ..core import speakers

    info = clip.info
    # Cheap first: the face track found one face per frame. Look for a second only when the clip is
    # wide enough to hold two people side by side.
    if info.width / info.height < 1.2:
        return None
    m = speakers.find(info)
    seen = [len(rows) for rows in m.frames]
    if np.mean([k >= 2 for k in seen]) < TWO_SHARE:
        return None
    ps = speakers.people(m)
    if len(ps) < 2:
        return None
    a, b = ps[0], ps[1]
    if min(a.width(), b.width()) < 0.4 * max(a.width(), b.width()):
        return None  # the second face is far smaller: somebody in the background, not a second speaker
    return ps


def whom(clip, opts: dict, content: tuple[int, int, int, int], aspect: float, turns_given=None) -> Follow:
    info = clip.info
    n = info.frames
    x0, y0, x1, y1 = content
    W, H = x1 - x0, y1 - y0

    def inside(r: Optional[Rect]) -> Optional[Rect]:
        return None if r is None else r.shift(-x0, -y0)

    face = clip.face
    ps = _two_people(clip) if opts["mode"] == "track" else None
    if ps is not None:
        from ..core import speakers

        # Both faces fit in the frame together: keep both, no speaker needed.
        win_w = min(W, H * aspect)
        both_fit = np.mean([a is not None and b is not None and max(a.x1, b.x1) - min(a.x0, b.x0) <= 0.9 * win_w
                            for a, b in zip(ps[0].boxes, ps[1].boxes)])
        if both_fit >= 0.8:
            faces = [inside(a.union(b)) if a is not None and b is not None else inside(a or b)
                     for a, b in zip(ps[0].boxes, ps[1].boxes)]
            return Follow(faces, [], "two people, both kept in the frame", people=2, both=True)
        if opts["speaker"] == "largest":
            big = max(ps[:2], key=lambda p: p.width() * p.share)
            return Follow([inside(b) for b in big.boxes], [], "two people: following the larger face, as asked",
                          people=2)
        from ..core import audio

        # The pauses a change of speaker moves to, from the sound alone (no need to transcribe).
        talk = speakers.talk(info, clip.sound, audio.quiet_runs(clip.envelope), ps[:2])
        if turns_given is not None:
            talk.who = _who_from_turns(turns_given, n, info.rate)
            talk.switches = [i for i in range(1, n) if talk.who[i] != talk.who[i - 1]]
            talk.note = "following the turns in the plan"
            talk.turns = turns_given
        faces = [inside(b) for b in talk.boxes()]
        # A change of speaker is a cut when the two faces are too far apart to glide between.
        cuts = []
        for sw in talk.switches:
            a, b = faces[max(0, sw - 1)], faces[sw]
            if a is not None and b is not None and abs(a.cx - b.cx) > 0.5 * win_w:
                cuts.append(sw)
        return Follow(faces, cuts, f"two people: {talk.note}", talk.turns, 2, talk)
    if face is None or not face.any() or face.found_share < framing_mod.MIN_FOUND:
        return Follow(None, [], "")
    boxes, cuts = framing_mod.faces_with_cuts(face, n, W, H, aspect)
    return Follow([inside(b) for b in boxes], cuts, "")


def _who_from_turns(turns: list, n: int, fps: float) -> np.ndarray:
    from . import PlanError

    who = np.zeros(n, int)
    for t in turns:
        try:
            a, b, p = float(t["from_s"]), float(t["to_s"]), int(t["person"])
        except (KeyError, TypeError, ValueError):
            raise PlanError(f"a turn needs from_s, to_s and person (1 or 2): {t!r}") from None
        if p not in (1, 2):
            raise PlanError(f"a turn's person is 1 or 2 (got {p})")
        who[int(round(a * fps)):int(round(b * fps))] = p - 1
    return who


# ---- the jobs ------------------------------------------------------------------------------------

def _shapes(info, content, fmt: str) -> list[str]:
    if fmt != "all":
        return [fmt]
    x0, y0, x1, y1 = content
    own = (x1 - x0) / (y1 - y0)
    return [s for s in FORMATS if abs(own - framing_mod.SHAPES[s]) >= 0.01 * framing_mod.SHAPES[s]]


def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from . import Job, PlanError

    info = clip.info
    fps = info.rate
    turns_given = None
    fmt = opts["format"]
    if plan is not None:
        fmt = plan.get("format", fmt)
        if fmt not in FORMATS:
            raise PlanError(f"the plan's format must be one of {', '.join(FORMATS)} (got {fmt!r})")
        if plan.get("turns"):
            turns_given = plan["turns"]
    content = bars(info)
    x0, y0, x1, y1 = content
    W, H = x1 - x0, y1 - y0
    shapes = _shapes(info, content, fmt)
    if not shapes:
        raise PlanError("the video is already every shape asked for")
    tl = Timeline.whole(info.frames / fps)
    jobs = []
    for shape in shapes:
        aspect = framing_mod.SHAPES[shape]
        size = framing_mod.shape_size(aspect) if opts["size"] == "full" else None
        fol = whom(clip, opts, content, aspect, turns_given)  # what is slow in it is kept beside the clip
        fr = framing_mod.plan(W, H, fps, info.frames, None, shape=shape, size=size, mode=opts["mode"],
                              faces=fol.faces, cuts=fol.cuts)
        if (x0, y0, x1, y1) != (0, 0, info.width, info.height):
            fr.content = content
        jobs.append(_job(clip, opts, shape, fr, fol, tl, content, Job))
    return jobs if len(jobs) > 1 else jobs[0]


def _job(clip, opts: dict, shape: str, fr, fol: Follow, tl: Timeline, content, Job):
    info = clip.info
    fps = info.rate
    n = info.frames
    path = fr.path
    note = fr.note
    if fol.note:
        note += f"; {fol.note}"
    if content != (0, 0, info.width, info.height):
        note += f"; black bars left out (the picture is {content[2] - content[0]}x{content[3] - content[1]})"
    checks: dict = {"framing": note}
    stats = {}
    if path is not None:
        size_axis = fr.windows[0, 2] if fr.axis == "x" else fr.windows[0, 3]
        stats = path.stats(fps, size_axis)
        checks["face_inside"] = f"{1 - fr.face_out:.1%} of frames (the face box wholly inside the frame)"
        if fr.axis == "y":
            checks["head_cut"] = f"{fr.head_cut:.1%} of frames (the top of the head above the frame)"
        checks["calm"] = (f"still {stats['still_share']:.0%} of the time; {stats['glides']} glide"
                          f"{'s' if stats['glides'] != 1 else ''}, {stats['cuts']} cut{'s' if stats['cuts'] != 1 else ''}; "
                          f"fastest {stats['fastest_per_s']:.2f} frames a second; "
                          f"{stats['turns_while_moving']} turn-backs while moving (jitter)")
        if stats["pushed_frames"]:
            checks["pushed"] = f"{stats['pushed_frames']} frames where the face would have left, so the frame was pushed"
    if fol.turns:
        checks["turns"] = "; ".join(f"person {t['person']} {t['from_s']:.1f}-{t['to_s']:.1f} s" for t in fol.turns)
    if fr.out_w < framing_mod.shape_size(framing_mod.SHAPES[shape])[0]:
        checks["size"] = (f"{fr.out_w}x{fr.out_h}: smaller than the shape's full size, because the picture is "
                          f"never enlarged unless size=full is asked for")
    out_plan = {"effect": NAME, "clip": clip.path, "options": opts, "format": shape,
                "output": {"width": fr.out_w, "height": fr.out_h, "fps": round(fps, 3),
                           "seconds": round(tl.duration, 3), "framing": note},
                "content": list(content),
                "moves": [m.as_dict(fps) for m in path.moves] if path is not None else [],
                "turns": fol.turns, "calm": stats,
                "face_inside": round(1 - fr.face_out, 4), "head_cut": round(fr.head_cut, 4),
                "sheet_times": _sheet_times(fr, fol, n, fps)}

    def stills() -> list[float]:
        return _sheet_times(fr, fol, n, fps)[:5]

    def after(result: dict, out: Path) -> dict:
        sheet = path_sheet(clip, fr, fol, shape, out.with_name(out.stem + ".path.jpg"), note)
        return {"path sheet": str(sheet)}

    return Job(NAME, clip, tl, fr, None, out_plan, checks=checks, still_times=stills, after=after,
               label=label(shape))


def _sheet_times(fr, fol: Follow, n: int, fps: float) -> list[float]:
    """Six moments that show the frame at work: a moment in each speaker's turn, either side of the
    first changes of speaker and the first moves, then filled in evenly across the clip."""
    dur = n / fps
    picks: list[float] = []

    def add(t: float) -> None:
        t = round(min(dur - 0.05, max(0.0, t)), 2)
        if len(picks) < 6 and all(abs(t - m) > 0.4 for m in picks):
            picks.append(t)

    turns = fol.turns or []
    for t in turns[1:3]:  # either side of a change of speaker
        add(t["from_s"] - 0.5)
        add(t["from_s"] + 0.7)
    for t in turns[:4]:  # the middle of each turn
        add((t["from_s"] + t["to_s"]) / 2)
    if fr.path is not None:
        for m in [m for m in fr.path.moves if not m.cut][:2]:  # before and after a glide
            add(m.start / fps - 0.3)
            add(m.end / fps + 0.3)
    k = 0
    while len(picks) < 6 and k < 24:
        add(dur * ((k * 7) % 24 + 0.5) / 24)
        k += 1
    return sorted(picks)


# ---- the path sheet --------------------------------------------------------------------------------

def path_sheet(clip, fr, fol: Follow, shape: str, out: Path, note: str = "") -> Path:
    """The source with the frame drawn on it at six moments, and under it the frame's path over the
    whole clip: where the face was (orange), where the frame could be with the face inside it (the
    grey band), and where the frame went (green). Glides are shaded; cuts and speaker changes marked."""
    from PIL import Image, ImageDraw

    from ..core.paths import FONTS

    info = clip.info
    fps, n = info.rate, info.frames
    times = _sheet_times(fr, fol, n, fps)
    font = _font(FONTS, 20)
    small = _font(FONTS, 16)
    tile_h = 300
    tiles = []
    for t in times:
        i = info.frame_at(t)
        src = video.read_frame(info, i)
        if fr.content is not None:
            x0, y0, x1, y1 = fr.content
            src = src[y0:y1, x0:x1]
        k = tile_h / src.shape[0]
        img = Image.fromarray(src).resize((max(1, int(round(src.shape[1] * k))), tile_h), Image.LANCZOS)
        d = ImageDraw.Draw(img, "RGBA")
        if fr.fill is None:
            x, y, w, h = fr.window(i)
            box = [x * k, y * k, (x + w) * k, (y + h) * k]
            d.rectangle([0, 0, img.width, img.height], fill=(0, 0, 0, 0))
            # dim what is cut away
            for r in ([0, 0, box[0], img.height], [box[2], 0, img.width, img.height],
                      [box[0], 0, box[2], box[1]], [box[0], box[3], box[2], img.height]):
                if r[2] > r[0] and r[3] > r[1]:
                    d.rectangle(r, fill=(0, 0, 0, 140))
            d.rectangle(box, outline=(120, 230, 160, 255), width=3)
        if fol.faces is not None and fol.faces[i] is not None:
            b = fol.faces[i]
            d.rectangle([b.x0 * k, b.y0 * k, b.x1 * k, b.y1 * k], outline=(255, 190, 70, 255), width=2)
        d.rectangle([0, tile_h - 26, img.width, tile_h], fill=(0, 0, 0, 170))
        d.text((6, tile_h - 24), f"{t:.2f} s", fill=(240, 240, 240), font=small)
        tiles.append(img)
    width = max(1600, sum(t.width for t in tiles) + 8 * (len(tiles) - 1) + 40)
    plot_h = 260 if fr.path is not None else 0
    height = 60 + tile_h + 30 + plot_h + 70
    sheet = Image.new("RGB", (width, height), (18, 18, 20))
    d = ImageDraw.Draw(sheet)
    title = f"reframe to {shape}: {note or fr.note}"
    d.text((20, 14), title[:170], fill=(236, 236, 240), font=font)
    x = 20
    for tile in tiles:
        sheet.paste(tile, (x, 56))
        x += tile.width + 8
    if fr.path is not None:
        p = fr.path
        top = 56 + tile_h + 30
        x0p, x1p = 120, width - 20
        lo_v = float(min(p.lo.min(), p.target.min())) - 10
        hi_v = float(max(p.hi.max(), p.target.max())) + 10

        def px(i: float) -> float:
            return x0p + (x1p - x0p) * i / max(1, n - 1)

        def py(v: float) -> float:
            return top + plot_h - (v - lo_v) / (hi_v - lo_v) * plot_h

        d.rectangle([x0p, top, x1p, top + plot_h], fill=(30, 30, 34))
        for m in p.moves:
            if not m.cut:
                d.rectangle([px(m.start), top, px(m.end), top + plot_h], fill=(44, 52, 48))
        step = max(1, n // 900)
        for i in range(0, n, step):
            d.line([px(i), py(p.lo[i]), px(i), py(p.hi[i])], fill=(58, 58, 74))
        d.line([(px(i), py(p.target[i])) for i in range(0, n, step)], fill=(255, 180, 60), width=2)
        d.line([(px(i), py(p.pos[i])) for i in range(0, n, step)], fill=(120, 230, 160), width=3)
        for m in p.moves:
            if m.cut:
                d.line([px(m.start), top, px(m.start), top + plot_h], fill=(255, 90, 90), width=2)
        for t in (fol.turns or [])[1:]:
            xx = px(t["from_s"] * fps)
            d.line([xx, top, xx, top + plot_h], fill=(150, 160, 255), width=2)
            d.text((xx + 4, top + 4), f"person {t['person']} talks", fill=(150, 160, 255), font=small)
        for tt in times:
            xx = px(tt * fps)
            d.polygon([(xx - 6, top + plot_h + 2), (xx + 6, top + plot_h + 2), (xx, top + plot_h + 12)], fill=(220, 220, 220))
        axis = "across" if fr.axis == "x" else "up and down"
        d.text((20, top + 6), f"frame {axis}", fill=(170, 170, 180), font=small)
        sec = 0.0
        while sec <= n / fps + 1e-6:
            xx = px(sec * fps)
            d.text((xx - 6, top + plot_h + 14), f"{sec:g}s", fill=(150, 150, 158), font=small)
            sec += 1.0 if n / fps <= 20 else 5.0
        legend = ("orange: where the face would put the frame · grey: where the frame can be with the face "
                  "inside · green: where the frame went · shaded: a glide · red: a cut · blue: the speaker changes")
        d.text((20, top + plot_h + 40), legend, fill=(170, 170, 180), font=small)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=86)
    return out


def _font(fonts: Path, px: int):
    from PIL import ImageFont

    return ImageFont.truetype(str(fonts / "Poppins" / "Poppins-SemiBold.ttf"), px)
