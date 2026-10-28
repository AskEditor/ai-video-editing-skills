"""A picture of what a sound effect did, for the eye of somebody who cannot hear it (or an agent
that cannot): six frames of the video, and under them, on one time line, a lane for each sound (the
clip's own voice, the music, the sounds laid), each drawn as its level in dB, so a duck under the
speech, a swell in a pause and every hit show as shapes. Marks cross the lanes at the moments that
matter (a cut, a card, a splice, the film's last frame), and a row of close-ups shows a few of them
at the scale of single frames: the frame before, the frame on the moment, and the waveform with a
line for every frame boundary, so a sound landing on its frame can be seen to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .paths import FONTS
from .sound import envelope_db
from .video import AUDIO_RATE

Colour = tuple[int, int, int]
BG = (18, 18, 20)
PANEL = (30, 30, 34)
GRID = (60, 60, 66)
TEXT = (232, 232, 236)
DIM = (150, 150, 158)


@dataclass
class Lane:
    name: str
    samples: np.ndarray  # (n,) or (n, channels) at 48 kHz, lined up with the video's frame 0
    colour: Colour
    curve: Optional[tuple[np.ndarray, np.ndarray]] = None  # (seconds, dB): a gain drawn over the lane
    curve_label: str = ""
    spans: Sequence[tuple[float, float]] = ()  # shaded behind the level (where the voice talks)


@dataclass
class Mark:
    t: float
    label: str = ""
    colour: Colour = (255, 206, 84)
    lanes: Optional[Sequence[str]] = None  # None: across every lane


@dataclass
class Zoom:
    t: float  # the moment: the start of the frame an event lands on
    label: str
    lanes: Sequence[str] = ()
    onset: Optional[float] = None  # where the sound was found to land, if measured
    before: float = 0.10
    after: float = 0.18
    note: str = ""
    frame: Optional[int] = None  # the frame shown as "the moment" (default: the one starting at `t`)


@dataclass
class Sheet:
    title: str
    duration: float
    fps: float
    frame_at: Callable[[float], np.ndarray]
    lanes: list[Lane]
    frames: Sequence[float] = ()
    marks: list[Mark] = field(default_factory=list)
    zooms: list[Zoom] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)


def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    name = "Poppins-Bold.ttf" if bold else "Poppins-SemiBold.ttf"
    return ImageFont.truetype(str(FONTS / "Poppins" / name), size)


def _tile(img: np.ndarray, h: int):
    from PIL import Image

    pic = Image.fromarray(img)
    return pic.resize((max(1, int(round(pic.width * h / pic.height))), h), Image.LANCZOS)


def draw(sheet: Sheet, out: Path) -> Path:
    from PIL import Image, ImageDraw

    pad, label_w, tile_h = 24, 150, 520
    frames = list(sheet.frames)[:6]
    tiles = [(t, _tile(sheet.frame_at(t), tile_h)) for t in frames]
    tiles_w = sum(p.width for _, p in tiles) + 8 * max(0, len(tiles) - 1)
    width = max(1800, tiles_w + 2 * pad)
    lane_h, ruler_h = 120, 34
    zoom_w, zoom_h = 300, 330
    zooms = sheet.zooms[:12]
    per_row = max(1, (width - 2 * pad + 12) // (zoom_w + 12))
    zoom_rows = (len(zooms) + per_row - 1) // per_row
    head_h = 46 + 30 * len(sheet.lines)
    height = (head_h + tile_h + 44 + ruler_h + len(sheet.lanes) * (lane_h + 10) + 20
              + zoom_rows * (zoom_h + 14) + pad)
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    f_title, f_text, f_small = _font(30, True), _font(21), _font(17)

    y = 14
    d.text((pad, y), sheet.title, fill=TEXT, font=f_title)
    y += 46
    for line in sheet.lines:
        d.text((pad, y), line, fill=DIM, font=f_text)
        y += 30

    # the six frames
    x = (width - tiles_w) // 2
    tile_x = []
    for t, pic in tiles:
        img.paste(pic, (x, y))
        d.rectangle([x, y + tile_h, x + pic.width, y + tile_h + 34], fill=PANEL)
        d.text((x + 10, y + tile_h + 6), f"{t:.2f} s", fill=TEXT, font=f_small)
        tile_x.append((t, x + pic.width / 2))
        x += pic.width + 8
    y += tile_h + 44

    # the time line
    x0, x1 = pad + label_w, width - pad
    span = max(sheet.duration, 1e-3)

    def tx(t: float) -> float:
        return x0 + (x1 - x0) * min(max(t, 0.0), span) / span

    ruler_y = y
    step = 1.0 if span <= 20 else 2.0 if span <= 45 else 5.0
    k = 0.0
    while k <= span + 1e-6:
        xx = tx(k)
        d.line([xx, ruler_y + 18, xx, ruler_y + ruler_h], fill=GRID, width=1)
        d.text((xx + 3, ruler_y), f"{k:g}s", fill=DIM, font=f_small)
        k += step
    for t, cx in tile_x:  # where each frame above sits on the time line
        xx = tx(t)
        d.polygon([(xx - 7, ruler_y + 20), (xx + 7, ruler_y + 20), (xx, ruler_y + 32)], fill=TEXT)
    y = ruler_y + ruler_h

    lanes_top = y
    hop = 1.0 / max(1.0, (x1 - x0) / span) / 2  # two level steps a pixel
    for lane in sheet.lanes:
        top, bot = y, y + lane_h
        d.rectangle([x0, top, x1, bot], fill=PANEL)
        d.text((pad, top + 8), lane.name, fill=lane.colour, font=f_text)
        for a, b in lane.spans:
            d.rectangle([tx(a), top, tx(b), bot], fill=(44, 44, 52))
        for db_line in (-48, -36, -24, -12):
            yy = bot - (db_line + 60) / 60 * lane_h
            d.line([x0, yy, x1, yy], fill=GRID, width=1)
        if len(lane.samples):
            env = envelope_db(lane.samples, AUDIO_RATE, hop=max(0.002, hop))
            times = (np.arange(len(env)) + 0.5) * max(0.002, hop)
            cols = np.clip(((times / span) * (x1 - x0)).astype(int), 0, int(x1 - x0) - 1)
            top_db = np.full(int(x1 - x0), -120.0)
            np.maximum.at(top_db, cols, env)
            for c in range(len(top_db)):
                if top_db[c] <= -60:
                    continue
                h = (min(0.0, top_db[c]) + 60) / 60 * lane_h
                d.line([x0 + c, bot, x0 + c, bot - h], fill=lane.colour, width=1)
        if lane.curve is not None:
            ts, gdb = lane.curve
            pts = [(tx(float(t)), top + 6 + (6.0 - min(6.0, max(-30.0, float(g)))) / 36.0 * (lane_h - 12))
                   for t, g in zip(ts, gdb)]
            if len(pts) > 1:
                d.line(pts, fill=(255, 255, 255), width=2)
            if lane.curve_label:
                d.text((pad, top + 40), lane.curve_label, fill=DIM, font=f_small)
        y = bot + 10
    lanes_bot = y - 10

    for m in sheet.marks:
        xx = tx(m.t)
        if m.lanes is None:
            d.line([xx, lanes_top, xx, lanes_bot], fill=m.colour, width=2)
        else:
            yy = lanes_top
            for lane in sheet.lanes:
                if lane.name in m.lanes:
                    d.line([xx, yy, xx, yy + lane_h], fill=m.colour, width=2)
                yy += lane_h + 10
        if m.label:
            d.text((xx + 4, lanes_bot - 22), m.label, fill=m.colour, font=f_small)
    y = lanes_bot + 20

    # close-ups at single-frame scale
    by_name = {lane.name: lane for lane in sheet.lanes}
    for n, z in enumerate(zooms):
        col, row = n % per_row, n // per_row
        zx, zy = pad + col * (zoom_w + 12), y + row * (zoom_h + 14)
        d.rectangle([zx, zy, zx + zoom_w, zy + zoom_h], fill=PANEL)
        th = 132
        j_at = int(round(z.t * sheet.fps)) if z.frame is None else int(z.frame)
        before = _tile(sheet.frame_at(max(0.0, (j_at - 1) / sheet.fps)), th)
        on = _tile(sheet.frame_at(j_at / sheet.fps), th)
        img.paste(before, (zx + 8, zy + 8))
        img.paste(on, (zx + 16 + before.width, zy + 8))
        d.rectangle([zx + 15 + before.width, zy + 7, zx + 16 + before.width + on.width, zy + 8 + th],
                    outline=(255, 206, 84), width=2)
        tx_note = zx + 26 + before.width + on.width
        d.text((tx_note, zy + 10), f"frame {j_at - 1}", fill=DIM, font=f_small)
        d.text((tx_note, zy + 32), f"frame {j_at}", fill=(255, 206, 84), font=f_small)
        d.text((tx_note, zy + 54), "(the moment)", fill=DIM, font=f_small)
        wy0, wy1 = zy + th + 22, zy + zoom_h - 52
        wx0, wx1 = zx + 8, zx + zoom_w - 8
        t0, t1 = z.t - z.before, z.t + z.after

        def zx_of(t: float) -> float:
            return wx0 + (wx1 - wx0) * (t - t0) / (t1 - t0)

        j0 = int(np.floor(t0 * sheet.fps))
        while j0 / sheet.fps <= t1:
            xx = zx_of(j0 / sheet.fps)
            if wx0 <= xx <= wx1:
                d.line([xx, wy0, xx, wy1], fill=GRID, width=1)
            j0 += 1
        mid = (wy0 + wy1) / 2
        for name in z.lanes:
            lane = by_name.get(name)
            if lane is None or not len(lane.samples):
                continue
            s = lane.samples.mean(axis=1) if lane.samples.ndim == 2 else lane.samples
            a, b = max(0, int(t0 * AUDIO_RATE)), min(len(s), int(t1 * AUDIO_RATE))
            if b <= a:
                continue
            peak = max(1e-6, float(np.abs(s[a:b]).max()))
            cols = int(wx1 - wx0)
            edges = (np.linspace(t0, t1, cols + 1) * AUDIO_RATE).astype(int)
            for c in range(cols):
                ia, ib = max(0, edges[c]), min(len(s), max(edges[c] + 1, edges[c + 1]))
                if ib <= ia:
                    continue
                part = s[ia:ib]
                lo, hi = float(part.min()) / peak, float(part.max()) / peak
                d.line([wx0 + c, mid - hi * (wy1 - wy0) / 2.1, wx0 + c, mid - lo * (wy1 - wy0) / 2.1], fill=lane.colour)
        xx = zx_of(z.t)
        d.line([xx, wy0 - 4, xx, wy1 + 4], fill=(255, 206, 84), width=2)
        if z.onset is not None:
            xo = zx_of(z.onset)
            d.line([xo, wy0, xo, wy1], fill=(120, 230, 160), width=2)
        d.text((zx + 8, wy1 + 6), z.label, fill=TEXT, font=f_small)
        if z.note:
            d.text((zx + 8, wy1 + 26), z.note, fill=DIM, font=f_small)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, quality=86)
    return out
