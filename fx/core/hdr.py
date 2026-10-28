"""HDR footage (HLG or PQ) read as ordinary SDR, so it never comes out flat and grey.

Phones film HDR by default (an iPhone's HLG, often with Dolby Vision on top). Its numbers are
coded on another curve and in wider primaries than an ordinary video's, so a tool that reads them
as if they were ordinary BT.709 shows the picture washed out and a little grey, and the pack writes
every video as BT.709. So the reader converts first, with one table (a 3D LUT, applied by ffmpeg's
`lut3d`, which every ffmpeg build has):

1. the coded value back to light: HLG's inverse curve then its display gamma (BT.2100, a 1000-nit
   display), or PQ's curve, which is light already;
2. BT.2020's primaries to BT.709's;
3. a white becomes SDR's white, and anything brighter is rolled off smoothly into the top of the
   range instead of clipping, so walls and shirts keep their detail. Phones place a room's white
   well above broadcast's reference (203 nits): on our iPhone takes (the hotel wall, the
   train, the white room; 28 October 2026) 203 nits blew the walls out and lifted skin by half a
   stop, while 300 nits kept skin where the phone shows it and the walls white, not grey;
4. coded for an ordinary screen (BT.1886, gamma 2.4).

The table is made once per kind and kept in the pack's cache, named by everything that shapes it.
"""

from __future__ import annotations

import atexit
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

VERSION = 2  # raise whenever the maths below changes, so no cached table is reused
SIZE = 33
WHITE = 300.0  # nits that become SDR's white (see 3. above)
SHOULDER = 0.6  # SDR light (0-1) where the roll-off of highlights begins

# BT.2020 to BT.709, for linear light.
TO_709 = np.array([[1.6605, -0.5876, -0.0728],
                   [-0.1246, 1.1329, -0.0083],
                   [-0.0182, -0.1006, 1.1187]])
LUMA_2020 = np.array([0.2627, 0.6780, 0.0593])

# The names ffprobe gives a stream's transfer, and the kind each is.
TRANSFERS = {"arib-std-b67": "hlg", "smpte2084": "pq"}


def kind(transfer: str | None) -> str:
    """"hlg", "pq", or "sdr" for anything else (BT.709, sRGB, unknown)."""
    return TRANSFERS.get((transfer or "").lower(), "sdr")


def _hlg_light(e: np.ndarray) -> np.ndarray:
    """HLG's coded value to display light in nits, per channel (the inverse curve, then BT.2100's
    display gamma of 1.2 on a 1000-nit display)."""
    a, b = 0.17883277, 0.28466892
    c = 0.55991073
    e = np.clip(e, 0.0, 1.0)
    scene = np.where(e <= 0.5, e * e / 3.0, (np.exp((e - c) / a) + b) / 12.0)
    y = np.maximum((scene * LUMA_2020).sum(axis=-1, keepdims=True), 1e-9)
    return 1000.0 * y ** 0.2 * scene


def _pq_light(e: np.ndarray) -> np.ndarray:
    """PQ's coded value to light in nits (SMPTE ST 2084)."""
    m1, m2 = 0.1593017578125, 78.84375
    c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
    p = np.clip(e, 0.0, 1.0) ** (1.0 / m2)
    return 10000.0 * (np.maximum(p - c1, 0.0) / (c2 - c3 * p)) ** (1.0 / m1)


def roll_off(x: np.ndarray) -> np.ndarray:
    """SDR light, linear up to SHOULDER and then easing towards 1 without ever reaching a hard clip
    (the slope is continuous at the shoulder)."""
    k = SHOULDER
    return np.where(x <= k, x, k + (1.0 - k) * (1.0 - np.exp(-(x - k) / (1.0 - k))))


def to_sdr(coded: np.ndarray, what: str) -> np.ndarray:
    """Coded HDR R'G'B' (0-1, BT.2020) to coded SDR R'G'B' (0-1, BT.709). Shape (..., 3)."""
    light = _hlg_light(coded) if what == "hlg" else _pq_light(coded)
    rel = light / WHITE
    rel = np.einsum("ij,...j->...i", TO_709, rel)
    peak = np.maximum(rel.max(axis=-1, keepdims=True), 1e-9)
    rel = rel * (roll_off(peak) / peak)  # by the brightest channel, so hues do not shift
    rel = np.clip(rel, 0.0, 1.0)
    return rel ** (1.0 / 2.4)


def _folder() -> Path:
    from .paths import CACHE

    for folder in (CACHE / "hdr", Path(tempfile.gettempdir()) / "video-editing-hdr"):
        try:
            folder.mkdir(parents=True, exist_ok=True)
            if os.access(folder, os.W_OK):
                return folder
        except OSError:
            continue
    folder = Path(tempfile.mkdtemp(prefix="hdr-"))
    atexit.register(shutil.rmtree, folder, ignore_errors=True)
    return folder


def lut_name(what: str) -> str:
    key = hashlib.sha1(repr((VERSION, SIZE, WHITE, SHOULDER)).encode()).hexdigest()[:10]
    return f"{what}-to-sdr-{key}.cube"


def write_lut(folder: Path, what: str) -> Path:
    """The table for `what` ("hlg" or "pq") in `folder`, written there if it is not there yet."""
    path = Path(folder) / lut_name(what)
    if path.exists():
        return path
    g = np.linspace(0.0, 1.0, SIZE)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")  # a .cube lists red fastest, then green, then blue
    table = to_sdr(np.stack([r, gg, b], axis=-1), what).reshape(-1, 3)
    lines = [f'TITLE "{what.upper()} to SDR"', f"LUT_3D_SIZE {SIZE}", "DOMAIN_MIN 0.0 0.0 0.0",
             "DOMAIN_MAX 1.0 1.0 1.0"]
    lines += [f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in table]
    tmp = path.with_name(path.name + f".{os.getpid()}.part")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(path)
    return path


def reader_filter(what: str, cwd: str | os.PathLike | None) -> tuple[str, str]:
    """The ffmpeg filter that turns coded HDR RGB (as `scale` makes it with the BT.2020 matrix, at
    16 bits) into SDR RGB, and the folder ffmpeg must run in for the table's bare name to resolve:
    `cwd` when the caller already runs it in a folder of its own (the table is put there too),
    else the cache."""
    folder = Path(cwd) if cwd else _folder()
    path = write_lut(folder, what)
    return f"lut3d=file={path.name}:interp=tetrahedral", str(folder)
