"""The clip's own cuts: where the picture jumps from one shot to the next.

    times = shots.cuts(clip)     # seconds where a new shot starts, in order

How much each frame differs from the one before is measured once (every frame read small and grey)
and cached beside the clip as `<clip>.shots.json`; the cuts are decided from it each time.

- **A cut is a jump, not a change.** A camera that moves or a hand that waves changes the picture
  over several frames; a cut changes it all at once. A frame is a cut when it differs from the one
  before several times more than the frames around it differ from theirs (the median over half a
  second either side), and by enough to see.
- **The speaker is there on both sides.** These are films of somebody talking, so the face is found
  in the quarter second before a cut and in the quarter second after it. A hand over the lens
  (clip C, 9.2 s) jumps like a cut, but the face is lost behind it, so it is not one. A film with
  nobody talking in it (scenery cut together) has no face to look for: `speaker=False` finds its
  cuts by the jump alone.
- **One cut per jump.** A joined clip sometimes keeps a stray frame of what was taken out, which
  makes two jumps a frame apart; they are one cut, at the first.

Jump cuts inside one camera set-up, where the person barely moved, change too little to tell from
talking: on clip 33 the cut finds 9 of its 10 joins, and the one it misses changes nothing a viewer
would see either.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import paths
from .video import FrameReader

CACHE_VERSION = 1
READ_WIDTH = 96
AROUND = 0.5  # seconds either side that say how much the picture usually changes
JUMP = 2.2  # a cut changes the picture at least this many times more than usual...
SEEN = 5.0  # ...and by at least this much (mean change of a pixel, 0..255)
FACE_SIDE = 0.25  # seconds either side of a cut in which the face must be found...
FACE_SHARE = 0.6  # ...in at least this share of the frames
APART = 0.3  # seconds: cuts closer than this are one


def changes(info, refresh: bool = False) -> np.ndarray:
    """How much each frame differs from the one before (mean of the pixels, 0..255; 0 for the
    first frame), from the cache beside the clip or measured now."""
    cache = paths.beside(info.path, "shots.json")
    signature = paths.source_signature(info.path)
    if not refresh and cache.exists():
        try:
            data = json.loads(cache.read_text())
            if (data.get("version") == CACHE_VERSION and data.get("source", {}).get("bytes") == signature["bytes"]
                    and len(data.get("change", [])) == info.frames):
                return np.asarray(data["change"], np.float32)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    started = time.time()
    w = READ_WIDTH
    h = max(2, int(round(info.height * w / info.width / 2)) * 2)
    out = np.zeros(info.frames, np.float32)
    prev = None
    for i, f in enumerate(FrameReader(info, size=(w, h), gray=True)):
        if i >= info.frames:
            break
        g = f.astype(np.float32)
        if prev is not None:
            out[i] = float(np.abs(g - prev).mean())
        prev = g
    data = {"version": CACHE_VERSION, "source": signature, "frames": info.frames,
            "seconds": round(time.time() - started, 2), "change": [round(float(v), 2) for v in out]}
    tmp = Path(str(cache) + ".part")
    tmp.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    tmp.replace(cache)
    return out


def cuts(clip, refresh: bool = False, speaker: bool = True) -> list[float]:
    """The seconds at which a new shot starts in `clip` (a Clip: its picture and its face). With
    `speaker=False` the face is not looked for, for a film with nobody talking in it."""
    info = clip.info
    d = changes(info, refresh)
    fps = info.rate
    n = len(d)
    r = max(2, int(round(AROUND * fps)))
    q = max(2, int(round(FACE_SIDE * fps)))
    found = clip.face.found if speaker and clip.face.n == n else np.ones(n, bool)
    picked: list[int] = []
    for i in range(1, n):
        if d[i] < SEEN:
            continue
        near = np.r_[d[max(1, i - r):max(1, i - 1)], d[i + 2:i + r + 1]]
        usual = float(np.median(near)) if len(near) else 0.0
        if d[i] < JUMP * (usual + 2.0):
            continue
        if found[max(0, i - q):i].mean() < FACE_SHARE or found[i:i + q].mean() < FACE_SHARE:
            continue
        if picked and (i - picked[-1]) / fps < APART:
            continue
        picked.append(i)
    return [round(i / fps, 3) for i in picked]
