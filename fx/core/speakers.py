"""Who is talking, when two people are in the picture: the one whose mouth moves with the sound.

1. **Every face, every frame.** The face landmark model finds up to four faces in each frame and
   where each mouth is. For each face it keeps a box (the same shape as `face.py`'s: cheek to
   cheek, forehead to chin), how open the jaw is, and how much the mouth itself changed since the
   last frame (the pixels of a patch fixed to the mouth, so a head turning does not count). Kept
   beside the clip as `<clip>.mouths.json`.
2. **Faces become people.** Each face is linked to the nearest face of the frame before; a person
   seen in too few frames (a passer-by) is dropped.
3. **The mouth that moves with the sound.** Over windows of six seconds, how far each person's
   mouth is open (the jaw, the gap between the lips and the area inside them) is compared with the
   loudness of the voice, and how much their mouth moves with the moments the voice starts. The
   sound and the picture of a recording are often a few frames apart, so that offset is found for
   the clip first: the one at which some mouth fits the sound best. Measured on two of
   our own takes (28 October 2026), with both people talking at once and one voice heard,
   the person whose voice it was fitted it better in 95% of six-second windows (86 to 93% with
   any one of the four signals alone, or with four-second windows). A listener whose mouth rests is
   easier than that.
4. **Turns, not flicker.** A window decides only when the voice is there and one person fits clearly
   better; turns shorter than two seconds are merged into their neighbours, and each change of
   speaker is moved to the nearest pause, so a cut lands between words.

When it cannot tell (both mouths fit the sound about as well, or nobody's does), it says so and the
caller follows the larger face.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from . import audio, paths
from .face import quiet_native
from .layout import Rect
from .video import AUDIO_RATE, FrameReader, VideoInfo, mono, read_audio

CACHE_VERSION = 2
DETECT_WIDTH = 640
HOP = 0.5  # seconds between decisions
MAX_LAG = 0.35  # the most the sound and the picture are allowed to be apart, seconds
WINDOW = 6.0  # seconds a decision looks at
MARGIN = 0.08  # how much better one person must fit the sound for a window to decide
MIN_TURN = 2.0  # a turn shorter than this is merged into its neighbours
MIN_SHARE = 0.25  # a person must be in this share of frames to count
SNAP = 0.6  # a change of speaker moves to a pause up to this far away

# The mouth's inner edge, and the points that give the box (see the module's docstring).
INNER_UPPER, INNER_LOWER = 13, 14
INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
TOP, CHIN = 151, 152


@dataclass
class Mouths:
    """Every face found in every frame: rows of [x0, y0, x1, y1, jaw, motion, gap, area], the box in
    clip pixels."""

    fps: float
    width: int
    height: int
    frames: list[list[list[float]]]
    seconds: float = 0.0


@dataclass
class Person:
    boxes: list[Optional[Rect]]  # per frame
    jaw: np.ndarray  # per frame, NaN where not seen: how open the jaw is (the model's own estimate)
    motion: np.ndarray  # how much the mouth's pixels changed since the frame before
    gap: np.ndarray  # the gap between the lips, as a share of the face's height
    area: np.ndarray  # the area inside the lips, as a share of the face's height squared

    @property
    def share(self) -> float:
        return float(np.mean([b is not None for b in self.boxes]))

    def width(self) -> float:
        ws = [b.w for b in self.boxes if b is not None]
        return float(np.median(ws)) if ws else 0.0

    def cx(self) -> float:
        xs = [b.cx for b in self.boxes if b is not None]
        return float(np.median(xs)) if xs else 0.0


@dataclass
class Talk:
    """Who talks when: `who` per frame (an index into `people`), and how sure."""

    people: list[Person]
    who: np.ndarray
    switches: list[int] = field(default_factory=list)
    lag: float = 0.0  # seconds the sound is behind the picture
    decided: float = 0.0  # the share of voiced time a window could decide
    margin: float = 0.0  # how much better the speaker fitted, on average
    note: str = ""
    turns: list[dict] = field(default_factory=list)

    def boxes(self) -> list[Optional[Rect]]:
        """The speaker's face box in each frame (the other person's when the speaker is not seen)."""
        out = []
        for i, k in enumerate(self.who):
            b = self.people[int(k)].boxes[i]
            if b is None:
                b = next((p.boxes[i] for p in self.people if p.boxes[i] is not None), None)
            out.append(b)
        return out


# ---- finding the faces and mouths ---------------------------------------------------------------

def find(info: VideoInfo) -> Mouths:
    """Run the landmark model on every frame (or read what an earlier run kept beside the clip)."""
    cache = paths.beside(info.path, "mouths.json")
    sig = paths.source_signature(info.path)
    if cache.exists():
        try:
            d = json.loads(cache.read_text())
            if d.get("version") == CACHE_VERSION and d.get("source", {}).get("bytes") == sig["bytes"] \
                    and len(d.get("frames", [])) == info.frames:
                return Mouths(info.rate, info.width, info.height, d["frames"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    import cv2

    from .models import vision_path

    with quiet_native():
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

    dw = min(DETECT_WIDTH, info.width)
    dw -= dw % 2
    dh = int(round(info.height * dw / info.width / 2)) * 2
    s = info.width / dw
    started = time.time()
    frames: list[list[list[float]]] = []
    prev: list[tuple[float, float, np.ndarray]] = []  # (cx, cy, mouth patch) of the frame before
    with quiet_native():
        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(vision_path("landmarks"))),
            running_mode=vision.RunningMode.VIDEO, num_faces=4, output_face_blendshapes=True,
            min_face_detection_confidence=0.4, min_face_presence_confidence=0.4, min_tracking_confidence=0.4)
        with vision.FaceLandmarker.create_from_options(options) as model:
            for i, f in enumerate(FrameReader(info, size=(dw, dh))):
                r = model.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(f)),
                                           int(round(i * 1000 / info.rate)))
                gray = f.mean(axis=2).astype(np.float32)
                rows, now = [], []
                for k, pts in enumerate(r.face_landmarks):
                    xs = np.array([p.x for p in pts]) * dw
                    ys = np.array([p.y for p in pts]) * dh
                    jaw = 0.0
                    if r.face_blendshapes and k < len(r.face_blendshapes):
                        jaw = next((c.score for c in r.face_blendshapes[k] if c.category_name == "jawOpen"), 0.0)
                    fh = max(1.0, float(ys[CHIN] - ys[TOP]))
                    mx = (xs[INNER_UPPER] + xs[INNER_LOWER]) / 2
                    my = (ys[INNER_UPPER] + ys[INNER_LOWER]) / 2
                    half = max(4, int(0.22 * fh))
                    y0, y1 = int(my - half), int(my + half)
                    x0, x1 = int(mx - 1.4 * half), int(mx + 1.4 * half)
                    patch = gray[max(0, y0):max(1, y1), max(0, x0):max(1, x1)]
                    if patch.size < 16:
                        patch = np.zeros((32, 48), np.float32)
                    patch = cv2.resize(patch, (48, 32), interpolation=cv2.INTER_AREA)
                    patch = (patch - patch.mean()) / (patch.std() + 1e-6)
                    near = min(prev, key=lambda q: (q[0] - mx) ** 2 + (q[1] - my) ** 2, default=None)
                    motion = -1.0
                    if near is not None and np.hypot(near[0] - mx, near[1] - my) < 0.5 * fh:
                        motion = float(np.abs(patch - near[2]).mean())
                    now.append((mx, my, patch))
                    gap = float(np.hypot(xs[INNER_UPPER] - xs[INNER_LOWER], ys[INNER_UPPER] - ys[INNER_LOWER])) / fh
                    px, py = xs[INNER], ys[INNER]
                    area = 0.5 * abs(float(np.dot(px, np.roll(py, 1)) - np.dot(py, np.roll(px, 1)))) / fh ** 2
                    rows.append([round(float(xs.min() * s), 1), round(float(ys[TOP] * s), 1),
                                 round(float(xs.max() * s), 1), round(float(ys[CHIN] * s), 1),
                                 round(float(jaw), 4), round(motion, 4), round(gap, 5), round(area, 6)])
                frames.append(rows)
                prev = now
    frames = (frames + [[] for _ in range(info.frames)])[:info.frames]
    m = Mouths(info.rate, info.width, info.height, frames, seconds=time.time() - started)
    tmp = Path(str(cache) + ".part")
    tmp.write_text(json.dumps({"version": CACHE_VERSION, "source": sig, "frames": frames}, separators=(",", ":")))
    tmp.replace(cache)
    return m


def people(m: Mouths) -> list[Person]:
    """Faces linked frame to frame into people, the most-seen first; passers-by dropped."""
    n = len(m.frames)
    tracks: list[dict] = []  # each: last box, rows by frame
    for i, rows in enumerate(m.frames):
        free = list(range(len(rows)))
        # each live track takes the nearest face that is close enough
        for t in sorted(tracks, key=lambda t: -t["seen"]):
            if i - t["last_i"] > int(1.0 * m.fps) or not free:
                continue
            lx0, ly0, lx1, ly1 = t["last"]
            cx, cy, w = (lx0 + lx1) / 2, (ly0 + ly1) / 2, lx1 - lx0
            best = min(free, key=lambda k: np.hypot((rows[k][0] + rows[k][2]) / 2 - cx, (rows[k][1] + rows[k][3]) / 2 - cy))
            r = rows[best]
            if np.hypot((r[0] + r[2]) / 2 - cx, (r[1] + r[3]) / 2 - cy) < 0.6 * max(w, 1.0):
                t["rows"][i] = r
                t["last"], t["last_i"] = r[:4], i
                t["seen"] += 1
                free.remove(best)
        for k in free:
            tracks.append({"rows": {i: rows[k]}, "last": rows[k][:4], "last_i": i, "seen": 1})
    out = []
    for t in sorted(tracks, key=lambda t: -t["seen"]):
        if t["seen"] < MIN_SHARE * n:
            continue
        boxes: list[Optional[Rect]] = [None] * n
        jaw, motion, gap, area = (np.full(n, np.nan) for _ in range(4))
        for i, r in t["rows"].items():
            boxes[i] = Rect(r[0], r[1], r[2], r[3])
            jaw[i], gap[i], area[i] = r[4], r[6], r[7]
            motion[i] = r[5] if r[5] >= 0 else np.nan
        out.append(Person(boxes, jaw, motion, gap, area))
    return out


# ---- who fits the sound -------------------------------------------------------------------------

def _corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 0.7 * len(a):
        return np.nan
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def _shift(x: np.ndarray, lag: int) -> np.ndarray:
    """x moved earlier by `lag` frames (x[t + lag] at t), NaN past the end."""
    out = np.full_like(x, np.nan)
    if lag >= 0:
        out[:len(x) - lag] = x[lag:]
    else:
        out[-lag:] = x[:len(x) + lag]
    return out


def talk(info: VideoInfo, sound: np.ndarray, quiet: list[tuple[float, float]], who: list[Person]) -> Talk:
    """Who talks in each frame, of `who` (two or more people)."""
    n, fps = info.frames, info.rate
    env = audio.envelope(mono(sound, AUDIO_RATE, 16000), 16000)
    per = 1.0 / fps
    level = np.clip(np.array([env.mean(i * per, (i + 1) * per) for i in range(n)]) - env.floor, 0, None)
    onset = np.clip(np.diff(level, prepend=level[0]), 0, None)
    voiced = level > (env.threshold - env.floor)
    w, hop = int(round(WINDOW * fps)), max(1, int(round(HOP * fps)))
    starts = list(range(0, max(1, n - w + 1), hop))
    max_lag = int(round(MAX_LAG * fps))

    open_ = ("jaw", "gap", "area")  # how open the mouth is, against how loud the voice is

    def fit(p: Person, s: int, lag_level: int, lag_onset: int) -> float:
        lv, on = shifted[("level", lag_level)], shifted[("onset", lag_onset)]
        cs = [_corr(getattr(p, f)[s:s + w], lv[s:s + w]) for f in open_]
        cs.append(_corr(p.motion[s:s + w], on[s:s + w]))
        cs = [c for c in cs if not np.isnan(c)]
        return float(np.mean(cs)) if cs else np.nan

    shifted = {(name, lag): _shift(x, lag) for name, x in (("level", level), ("onset", onset))
               for lag in range(-max_lag, max_lag + 1)}

    # The offset between sound and picture: the one at which the people differ most in how well
    # they fit the sound, window by window (at a wrong offset every mouth fits about equally badly).
    # Found for openness and for movement separately: they peak frames apart.
    def best_lag(feature: str) -> int:
        best, best_v = 0, -np.inf
        for lag in range(-max_lag, max_lag + 1):
            x = shifted[("level" if feature == "level" else "onset", lag)]
            vals = []
            for s in starts:
                if voiced[s:s + w].mean() < 0.3:
                    continue
                if feature == "level":
                    cs = [np.nanmean([_corr(getattr(p, f)[s:s + w], x[s:s + w]) for f in open_]) for p in who]
                else:
                    cs = [_corr(p.motion[s:s + w], x[s:s + w]) for p in who]
                cs = sorted((c for c in cs if not np.isnan(c)), reverse=True)
                if len(cs) >= 2:
                    vals.append(cs[0] - cs[1])
            v = float(np.mean(vals)) if vals else -np.inf
            if v > best_v:
                best, best_v = lag, v
        return best

    lag_level, lag_onset = best_lag("level"), best_lag("onset")
    # Each window of six seconds: the person who fits the sound clearly best, while the voice is
    # there; else undecided.
    pick = np.full(len(starts), -1)
    margins = []
    for j, s in enumerate(starts):
        if voiced[s:s + w].mean() < 0.3:
            continue
        col = np.array([fit(p, s, lag_level, lag_onset) for p in who])
        if np.all(np.isnan(col)):
            continue
        order = np.argsort(-np.nan_to_num(col, nan=-9))
        top = col[order[0]]
        second = col[order[1]] if len(order) > 1 and not np.isnan(col[order[1]]) else -1.0
        if top - second >= MARGIN and top > 0.05:
            pick[j] = order[0]
            margins.append(top - second)
    decided = float(np.mean(pick >= 0)) if len(pick) else 0.0
    margin = float(np.mean(margins)) if margins else 0.0
    largest = int(np.argmax([p.width() * p.share for p in who]))
    if not margins:
        return Talk(who, np.full(n, largest), [], lag_level / fps, 0.0, 0.0,
                    "could not tell who is talking from the mouths and the sound; following the larger face")
    filled = pick.copy()
    last = next((k for k in pick if k >= 0), largest)
    for j in range(len(filled)):
        if filled[j] < 0:
            filled[j] = last
        else:
            last = filled[j]
    centres = np.array([s + w / 2 for s in starts])
    frames_who = np.array([filled[int(np.argmin(np.abs(centres - i)))] for i in range(n)])
    frames_who = _merge_short(frames_who, int(round(MIN_TURN * fps)))
    # A window of six seconds knows who talked, but only roughly when the turn changed. Each change
    # is placed where the evidence changes sides: frame by frame, how well each mouth agrees with
    # the sound, summed from the turn before up to a point and from there to the turn after.
    agree = np.stack([_agreement(p, shifted[("level", lag_level)], shifted[("onset", lag_onset)], voiced, fps)
                      for p in who])
    frames_who = _place_changes(frames_who, agree, int(round(WINDOW * fps / 2)))
    switches = [i for i in range(1, n) if frames_who[i] != frames_who[i - 1]]
    snapped = []
    for sw in switches:  # each change of speaker on the nearest pause
        t = sw / fps
        best = None
        for q0, q1 in quiet:
            mid = (q0 + q1) / 2
            if abs(mid - t) <= SNAP and (best is None or abs(mid - t) < abs(best - t)):
                best = mid
        snapped.append(int(round((best if best is not None else t) * fps)))
    out = frames_who.copy()
    for old, new in zip(switches, snapped):
        if new > old:  # the previous speaker keeps the frames up to the pause
            out[old:new] = frames_who[old - 1]
        elif new < old:  # the next speaker takes them from the pause
            out[new:old] = frames_who[old]
    switches = [i for i in range(1, n) if out[i] != out[i - 1]]
    turns = []
    edges = [0] + switches + [n]
    for a, b in zip(edges, edges[1:]):
        turns.append({"from_s": round(a / fps, 2), "to_s": round(b / fps, 2), "person": int(out[a]) + 1})
    note = (f"followed whoever is talking: {len(turns)} turn{'s' if len(turns) != 1 else ''}, "
            f"decided in {decided:.0%} of voiced windows")
    return Talk(who, out, switches, lag_level / fps, decided, margin, note, turns)


def _z(x: np.ndarray, span: int) -> np.ndarray:
    """x less its local mean, over its local spread (both over `span` frames either side), NaN kept."""
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        seg = x[max(0, i - span):i + span + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg) > 5 and not np.isnan(x[i]):
            sd = seg.std()
            out[i] = (x[i] - seg.mean()) / sd if sd > 1e-9 else 0.0
    return out


def _agreement(p: Person, level: np.ndarray, onset: np.ndarray, voiced: np.ndarray, fps: float) -> np.ndarray:
    """Per frame, how well this mouth agrees with the sound: the products of their deviations, the
    same terms a correlation adds up, averaged over the four signals; 0 where nobody speaks."""
    span = int(round(3 * fps))
    zl, zo = _z(level, span), _z(onset, span)
    terms = [_z(getattr(p, f), span) * zl for f in ("jaw", "gap", "area")] + [_z(p.motion, span) * zo]
    with np.errstate(all="ignore"):
        a = np.nanmean(np.stack(terms), axis=0)
    a = np.nan_to_num(a, nan=0.0)
    return np.where(voiced, a, 0.0)


def _place_changes(who: np.ndarray, agree: np.ndarray, reach: int) -> np.ndarray:
    """Each change of speaker moved to the frame, within `reach` of where it was, that best splits
    the evidence: the frames before favouring the speaker before, the frames after the one after."""
    out = who.copy()
    changes = [i for i in range(1, len(who)) if who[i] != who[i - 1]]
    for k, c in enumerate(changes):
        a, b = int(who[c - 1]), int(who[c])
        lo = max(changes[k - 1] + 1 if k > 0 else 1, c - reach)
        hi = min(changes[k + 1] - 1 if k + 1 < len(changes) else len(who) - 1, c + reach)
        if hi <= lo:
            continue
        seg = agree[a][lo:hi] - agree[b][lo:hi]  # positive where the earlier speaker fits better
        before = np.concatenate([[0.0], np.cumsum(seg)])
        gain = before - (before[-1] - before)
        s = lo + int(np.argmax(gain))
        out[lo:s] = a
        out[s:hi] = b
    return out


def _gauss(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or len(x) < 3:
        return x.copy()
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, (r, r), mode="edge"), k, mode="valid")


def _best_order(evidence: np.ndarray, cost: float) -> np.ndarray:
    """The speaker in each frame that collects the most evidence, paying `cost` for each change."""
    k, n = evidence.shape
    score = evidence[:, 0].copy()
    back = np.zeros((k, n), int)
    for t in range(1, n):
        stay = score
        move = score.max() - cost
        src = np.where(stay >= move, np.arange(k), int(np.argmax(score)))
        score = np.maximum(stay, move) + evidence[:, t]
        back[:, t] = src
    out = np.zeros(n, int)
    out[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        out[t - 1] = back[out[t], t]
    return out


def _merge_short(who: np.ndarray, least: int) -> np.ndarray:
    """Turns shorter than `least` frames given to the longer neighbour, until none is left."""
    who = who.copy()
    for _ in range(20):
        edges = [0] + [i for i in range(1, len(who)) if who[i] != who[i - 1]] + [len(who)]
        runs = [(a, b) for a, b in zip(edges, edges[1:])]
        short = [(b - a, k) for k, (a, b) in enumerate(runs) if b - a < least and len(runs) > 1]
        if not short:
            break
        _, k = min(short)
        a, b = runs[k]
        left = runs[k - 1] if k > 0 else None
        right = runs[k + 1] if k + 1 < len(runs) else None
        if left is None or (right is not None and right[1] - right[0] > left[1] - left[0]):
            who[a:b] = who[right[0]]
        else:
            who[a:b] = who[left[0]]
    return who
