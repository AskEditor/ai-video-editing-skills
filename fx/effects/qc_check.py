"""qc-check: the checks before "done". It reads a finished video, and the clip it was made from when
it is given, says in plain words what would embarrass it once posted, and changes nothing.

What it checks:

- **The file:** its shape for the platform, its size, its frame rate (a changing one, or a change
  from the source's), its codec and its length.
- **The sound:** loudness (integrated, measured exactly as the pack sets it) and true peak against
  the platform's target; samples at full scale (clipping); silence where there should be a voice
  (with the source: the voice in the source and not in the video); long dead air.
- **The picture:** black frames; frozen frames (a picture that stops while the sound goes on);
  rapid flashing; HDR or log footage that was never converted and so looks flat and grey.
- **With the source:** the video is lined up with the source in time (by the sound, so cuts and
  trims are found) and in space (the same frame, a crop that moves, or the picture fitted inside a
  bigger frame), and then, frame by frame:
  - **the face:** inside the face box, anything in the video that was not in the source (words, a
    card, a sticker) is measured; a frame where it covers enough of the face is flagged. A colour
    grade does not count: the comparison is of shapes, not of colours;
  - **the safe zone:** anything added outside it (the bottom fifth, where the caption and buttons
    sit, the side margins, the top bar) is flagged.

It writes `<video>.qc.md` (the verdict first, then what to fix, then every check with what was
measured) and `<video>.qc.jpg` (the frames that matter, with the safe zone and the face box drawn
and anything added over the face tinted, over a strip of the whole video). It exits with 1 when a
check fails (or, with `strict=on`, warns), so a script or an agent can stop on it.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..core import audio, loudness, video
from ..core import face as facemod
from ..core.layout import Rect, SafeZone
from . import options as O

NAME = "qc-check"
REPORTS = True

PLATFORMS = {
    "reels": {"name": "Instagram Reels", "aspect": 9 / 16, "lufs": -14.0, "tp": -1.0, "longest": 180, "zone": True},
    "tiktok": {"name": "TikTok", "aspect": 9 / 16, "lufs": -14.0, "tp": -1.0, "longest": 600, "zone": True},
    "shorts": {"name": "YouTube Shorts", "aspect": 9 / 16, "lufs": -14.0, "tp": -1.0, "longest": 180, "zone": True},
    "youtube": {"name": "YouTube", "aspect": 16 / 9, "lufs": -14.0, "tp": -1.0, "longest": None, "zone": False},
    "any": {"name": "any platform", "aspect": None, "lufs": None, "tp": -1.0, "longest": None, "zone": False},
}

OPTIONS = {
    "source": O.Text("the clip the video was made from (a full path). With it the face and the safe zone are "
                     "checked frame by frame, and missing sound or a changed frame rate is found"),
    "platform": O.Choice(tuple(PLATFORMS), "where it will be posted: reels, tiktok, shorts (9:16, -14 LUFS, "
                                           "-1 dBTP), youtube (16:9), or any (no shape or loudness target)"),
    "strict": O.Switch("on: a warning fails the check too"),
}

ANALYSIS = 640  # the longer side of the picture as it is examined
LOUD_OK, LOUD_WARN = 3.0, 6.0  # LU from the target: within the first is right, past the second fails
TP_FAIL = 0.5  # dBTP: a true peak over this fails; between the target and this warns
FACE_FLAG = 0.08  # a frame is flagged when something not in the source spans this share of the face box
FACE_FAIL = 0.12  # a flagged stretch fails when it spans this much of the face at its worst, for long enough
FACE_LONG = 0.25  # seconds a stretch must last to fail (a word sliding past the chin is a warning)
BLOB = 0.0015  # an addition outside the safe zone counts from this share of the frame
REPLACED = 0.40  # a frame where this share changed is a new picture (a new room, a full card), not an overlay
ELSEWHERE = 0.15  # a frame where fewer of its textured blocks than this still match the source shows another
#                   picture (a cutaway): cutaways measured 0-1%, overlays 93-100%, a new room round you 36-43%
DEAD = -70.0  # dBFS: sound under this is dead silence (a room, however quiet, is louder)
BLACK_PIX, BLACK_SHARE = 26, 0.98  # a frame is black when this share of its pixels is under this level
FROZEN = 0.06  # a frame changing less than this (0-255, on a small grey copy) from the one before is frozen:
# a repeated frame changes by 0.03 at most, a person sitting still at 60 fps by 0.1 or more
SPECK = 6  # changed pixels in a clump smaller than this are the encoder's noise, not something added


@dataclass
class Finding:
    check: str
    level: str  # "fail", "warn", "pass", "skip" or "info"
    what: str
    fix: str = ""
    spans: list = field(default_factory=list)  # [(start_s, end_s), …]
    at: Optional[float] = None  # the moment to show on the sheet


LEVEL_ORDER = {"fail": 0, "warn": 1, "info": 2, "pass": 3, "skip": 4}


# ---- small helpers ---------------------------------------------------------------------------------

def _spans(flags: np.ndarray, fps: float, gap: int = 1, least: int = 1) -> list[tuple[int, int]]:
    """Runs of True as (first, last+1) frame pairs, joining runs `gap` frames apart, keeping those of
    at least `least` frames."""
    out: list[list[int]] = []
    idx = np.flatnonzero(flags)
    for i in idx:
        if out and i - out[-1][1] <= gap:
            out[-1][1] = i + 1
        else:
            out.append([int(i), int(i) + 1])
    return [(a, b) for a, b in out if b - a >= least]


def _secs(spans: list[tuple[int, int]], fps: float) -> list[tuple[float, float]]:
    return [(round(a / fps, 2), round(b / fps, 2)) for a, b in spans]


def _say_spans(spans: list[tuple[float, float]], most: int = 3) -> str:
    parts = [f"{a:.2f}-{b:.2f} s" for a, b in spans[:most]]
    more = len(spans) - most
    return ", ".join(parts) + (f" and {more} more" if more > 0 else "")


def _probe(path: str) -> dict:
    run = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
                         capture_output=True, text=True)
    if run.returncode != 0:
        raise video.MediaError(f"this file could not be read as a video: {path}")
    return json.loads(run.stdout)


def _rate(text: Optional[str]) -> float:
    try:
        a, _, b = (text or "0/1").partition("/")
        return float(a) / float(b or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _gray(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def _structure(g: np.ndarray) -> np.ndarray:
    """The shapes in a grey picture, without its brightness or contrast: each pixel less the local
    mean, over the local spread (plus a floor, so flat areas stay near nought). A grade changes
    colours and contrast but not shapes; words and cards change shapes."""
    import cv2

    mu = cv2.GaussianBlur(g, (0, 0), 4)
    d = g - mu
    sd = np.sqrt(cv2.GaussianBlur(d * d, (0, 0), 4))
    return d / (sd + 6.0)


def _changed(r: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Where the video adds shapes the source does not have, as a 0-1 mask. A pixel of misalignment,
    the encoder's noise and a grade are all forgiven. Thick letters change only at their edges (inside a
    white stroke there is no shape in either picture), so what is found is closed up: a word counts by
    the area it covers, not by its outline."""
    import cv2

    nr, ns = _structure(r), _structure(s)
    diff = np.abs(nr - ns)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        diff = np.minimum(diff, np.abs(nr - np.roll(np.roll(ns, dy, axis=0), dx, axis=1)))
    diff = cv2.GaussianBlur(diff, (0, 0), 1.0)
    # Only what the video adds: more shape than the source has there. A background made plain (a
    # flat colour where a room was) takes shapes away and is not something laid over the picture.
    more = cv2.GaussianBlur(np.abs(nr), (0, 0), 2.0) > cv2.GaussianBlur(np.abs(ns), (0, 0), 2.0) + 0.25
    m = ((diff > 0.9) & more).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if count > 1:
        specks = stats[:, cv2.CC_STAT_AREA] < SPECK
        specks[0] = False
        m[specks[labels]] = 0
    k = max(3, int(round(r.shape[1] / 50)) | 1)  # about a stroke's width at the size examined
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


# ---- the file, the sound and the picture -----------------------------------------------------------

class Facts:
    def __init__(self, path: str):
        self.path = path
        d = _probe(path)
        streams = d.get("streams", [])
        self.v = next((s for s in streams if s.get("codec_type") == "video"
                       and not s.get("disposition", {}).get("attached_pic")), None)
        if self.v is None:
            raise video.MediaError(f"this file has no picture: {path}")
        self.a = next((s for s in streams if s.get("codec_type") == "audio"), None)
        self.format = d.get("format", {})
        self.info = video.probe(path)
        self.rotation = 0
        for side in self.v.get("side_data_list", []) or []:
            if "rotation" in side:
                self.rotation = int(round(float(side["rotation"])))
        self.r_rate = _rate(self.v.get("r_frame_rate"))
        self.avg_rate = _rate(self.v.get("avg_frame_rate"))
        self.transfer = (self.v.get("color_transfer") or "").lower()
        self.space = (self.v.get("color_space") or "").lower()
        self.pix_fmt = self.v.get("pix_fmt") or ""
        self.codec = self.v.get("codec_name") or ""
        self.sar = self.v.get("sample_aspect_ratio") or "1:1"


def file_checks(f: Facts, platform: dict, src: Optional[Facts]) -> list[Finding]:
    out = []
    W, H = f.info.width, f.info.height
    shape = W / H
    want = platform["aspect"]
    if want is None:
        out.append(Finding("shape", "pass", f"{W}x{H} ({shape:.3g}:1); no shape is asked of this platform"))
    elif abs(shape - want) <= 0.02 * want:
        out.append(Finding("shape", "pass", f"{W}x{H}, {'9:16' if want < 1 else '16:9'} as {platform['name']} expects"))
    else:
        out.append(Finding("shape", "warn", f"{W}x{H} is not the {'9:16' if want < 1 else '16:9'} shape "
                           f"{platform['name']} fills the screen with; it will be cropped or shown with bars",
                           "make it that shape (the reframe effect does, following the face)"))
    long_side, short_side = max(W, H), min(W, H)
    if W % 2 or H % 2:
        out.append(Finding("size", "fail", f"{W}x{H} has an odd side, which some platforms and players refuse",
                           "render it again at an even size"))
    elif short_side < 720:
        out.append(Finding("size", "warn", f"{W}x{H} is small: under 720 on its short side it looks soft on a phone",
                           "render from a bigger source, or at size=1080x1920"))
    elif short_side < 1080:
        out.append(Finding("size", "info", f"{W}x{H}: the platform enlarges it to {1080 if W < H else 1920} "
                           f"wide; fine, a little softer than full size"))
    else:
        out.append(Finding("size", "pass", f"{W}x{H}"))
    if f.sar not in ("1:1", "0:1", "N/A"):
        out.append(Finding("pixels", "warn", f"its pixels are not square ({f.sar}): some apps show it stretched",
                           "export with square pixels"))
    if f.rotation:
        out.append(Finding("rotation", "warn", f"it is stored on its side and turned by a {f.rotation}° flag, which "
                           f"some apps ignore", "export it upright"))
    fps = f.info.rate
    vfr = f.avg_rate > 0 and abs(f.r_rate - f.avg_rate) > 0.01 * f.avg_rate
    if vfr:
        out.append(Finding("frame rate", "warn", f"the frame rate changes (it averages {f.avg_rate:.3g} fps): edits "
                           f"and uploads can drift out of sync", "export at a constant frame rate"))
    elif fps < 23:
        out.append(Finding("frame rate", "warn", f"{fps:.3g} fps looks choppy", "export at 24, 25 or 30 fps"))
    elif fps > 60.5:
        out.append(Finding("frame rate", "warn", f"{fps:.3g} fps is above what platforms play; they convert it",
                           "export at 60 fps or less"))
    elif src is not None and abs(src.info.rate - fps) > 0.01 * fps:
        out.append(Finding("frame rate", "warn", f"{fps:.3g} fps, but the source is {src.info.rate:.3g} fps: motion "
                           f"stutters when frames are dropped or doubled", "keep the source's frame rate"))
    elif min(abs(fps - c) for c in (23.976, 24, 25, 29.97, 30, 50, 59.94, 60)) > 0.05:
        out.append(Finding("frame rate", "info", f"{fps:.4g} fps is unusual; platforms play it, or convert it to 30"))
    else:
        out.append(Finding("frame rate", "pass", f"{fps:.4g} fps, constant"))
    if f.codec not in ("h264", "hevc"):
        out.append(Finding("codec", "warn", f"the picture is {f.codec}; platforms want H.264 or HEVC", "export as H.264"))
    elif f.pix_fmt not in ("yuv420p", "yuvj420p", "yuv420p10le"):
        out.append(Finding("codec", "warn", f"its pixel format is {f.pix_fmt}, which some phones do not play",
                           "export as yuv420p"))
    else:
        out.append(Finding("codec", "pass", f"{f.codec.upper() if f.codec == 'hevc' else 'H.264'}, {f.pix_fmt}"))
    dur = f.info.duration
    if platform["longest"] and dur > platform["longest"]:
        out.append(Finding("length", "warn", f"{dur:.1f} s is longer than {platform['name']} takes "
                           f"({platform['longest']} s)", "cut it down, or post it elsewhere"))
    else:
        out.append(Finding("length", "pass", f"{dur:.2f} s"))
    if f.a is not None:
        a_dur = float(f.a.get("duration") or dur)
        if abs(a_dur - dur) > 0.25:
            out.append(Finding("sound length", "warn", f"the sound is {a_dur:.2f} s and the picture {dur:.2f} s: one "
                               f"ends early", "render it again so they match"))
    return out


def sound_checks(f: Facts, platform: dict, src: Optional[Facts], tmap, quiet_src,
                 still: Optional[np.ndarray] = None) -> tuple[list[Finding], dict]:
    """Loudness, true peak, clipping and silence. Returns the findings and what the sheet draws."""
    out: list[Finding] = []
    drawn: dict = {}
    if f.a is None:
        if src is not None and src.a is not None:
            out.append(Finding("sound", "fail", "the video has no sound at all, and its source had a voice",
                               "render it again keeping the sound"))
        else:
            out.append(Finding("sound", "warn", "the video has no sound", "add the voice or music back"))
        return out, drawn
    m = loudness.measure_file(f.path)
    target, tp_target = platform["lufs"], platform["tp"]
    if m.i < -70:
        out.append(Finding("loudness", "fail", "the sound is silent from start to end", "render it again with its sound"))
        return out, drawn
    if target is None:
        out.append(Finding("loudness", "info", f"{m.i:.1f} LUFS integrated (no target for this platform)"))
    else:
        off = m.i - target
        say = f"{m.i:.1f} LUFS integrated, {abs(off):.1f} LU {'under' if off < 0 else 'over'} the {target:g} LUFS " \
              f"{platform['name']} plays at"
        if abs(off) <= LOUD_OK:
            out.append(Finding("loudness", "pass", say))
        elif abs(off) <= LOUD_WARN:
            out.append(Finding("loudness", "warn", say + (": it will sound quiet next to other posts" if off < 0
                                                           else ": loud, and the platform may turn it down"),
                               f"set it to {target:g} LUFS (the music-bed effect does this for a whole mix)"))
        else:
            out.append(Finding("loudness", "fail", say + (": far too quiet; people will skip it or turn it up and "
                                                           "hear noise" if off < 0 else ": far too loud"),
                               f"set it to {target:g} LUFS before posting"))
    if m.tp > TP_FAIL:
        out.append(Finding("true peak", "fail", f"the true peak is {m.tp:+.1f} dBTP, over full scale: it distorts "
                           f"when the platform encodes it again", f"limit the peaks to {tp_target:g} dBTP"))
    elif m.tp > tp_target:
        out.append(Finding("true peak", "warn", f"the true peak is {m.tp:+.1f} dBTP, over the {tp_target:g} dBTP "
                           f"platforms ask for; the loudest moments may crackle after upload",
                           f"limit the peaks to {tp_target:g} dBTP"))
    else:
        out.append(Finding("true peak", "pass", f"{m.tp:+.1f} dBTP"))
    # Clipping and silence, from the samples themselves.
    info = f.info
    s = video.read_audio(info)
    peak = np.abs(s).max(axis=1)
    clipped = peak >= 0.999
    rate = video.AUDIO_RATE
    events = _spans(clipped, rate, gap=int(0.005 * rate))
    if events:
        first = events[0][0] / rate
        say = (f"{len(events)} moment{'s' if len(events) != 1 else ''} where the sound hits full scale and clips "
               f"({clipped.sum()} samples), the first at {first:.2f} s")
        level = "fail" if len(events) >= 10 or clipped.mean() > 1e-4 else "warn"
        out.append(Finding("clipping", level, say, "turn the sound down before the peaks, or limit it",
                           spans=[(a / rate, b / rate) for a, b in events], at=first))
    else:
        out.append(Finding("clipping", "pass", "no sample reaches full scale"))
    mono = video.mono(s, rate, 16000)
    env = audio.envelope(mono, 16000)
    drawn["envelope"] = env
    # Dead silence: the whole sound under -60 dBFS for a second or more, not at the very end (an end
    # card held in silence is the pack's own way to finish).
    hop = 0.05
    n_hop = int(len(mono) / (16000 * hop))
    rms = np.array([np.sqrt(np.mean(mono[int(k * hop * 16000):int((k + 1) * hop * 16000)] ** 2) + 1e-12)
                    for k in range(n_hop)])
    dead = 20 * np.log10(rms + 1e-12) < DEAD
    dur = info.duration
    runs = [(a * hop, b * hop) for a, b in _spans(dead, 1 / hop, least=int(1.0 / hop))]
    inner = [(a, b) for a, b in runs if b < dur - 0.25 and a > 0.25]
    ends = [(a, b) for a, b in runs if (a, b) not in inner]
    drawn["dead"] = runs
    missing = []
    if src is not None and tmap is not None and quiet_src is not None and src.a is not None:
        # The voice is in the source at that moment, and not in the video.
        speech_src = _voiced_mask(quiet_src, tmap, info.frames, info.rate)
        vq = audio.quiet_runs(env)
        quiet_here = np.zeros(info.frames, bool)
        for a, b in vq:
            quiet_here[int(a * info.rate):int(np.ceil(b * info.rate))] = True
        # A hold (the source's moment stands still: a freeze, an end card past the source's last
        # frame) has no speech of the source's to lose.
        t_src = np.asarray(tmap.src, float)[:info.frames]
        held = np.zeros(info.frames, bool)
        held[1:len(t_src)] = np.abs(np.diff(t_src)) < 0.25 / src.info.rate
        held[:len(t_src)] |= t_src >= src.info.duration - 1.5 / src.info.rate
        if still is not None:  # a freeze for effect: the picture stands still while the sound is quiet
            held[:min(len(still), info.frames)] |= still[:info.frames]
        lost = speech_src & quiet_here & ~tmap.near_cut & ~held
        missing = _secs(_spans(lost, info.rate, gap=2, least=int(0.3 * info.rate)), info.rate)
    if missing:
        out.append(Finding("silence", "fail", f"the voice is missing at {_say_spans(missing)}: the source has speech "
                           f"there and the video is silent", "render it again; check the sound was not muted or cut",
                           spans=missing, at=missing[0][0]))
    elif inner:
        level = "warn"
        out.append(Finding("silence", level, f"dead silence at {_say_spans(inner)}: nothing at all is heard there",
                           "cut the silence, or put sound under it", spans=inner, at=inner[0][0]))
    else:
        note = "no dead silence" + (f" (a silent end of {ends[-1][1] - ends[-1][0]:.1f} s, as an end card has)"
                                    if ends and ends[-1][1] >= dur - 0.3 else "")
        out.append(Finding("silence", "pass", note))
    # Long dead air in a talking video: a pause of 2.5 s or more between words.
    q = audio.quiet_runs(env)
    voiced_share = 1 - sum(b - a for a, b in q) / max(dur, 1e-3)
    long_q = [(a, b) for a, b in q if b - a >= 2.5 and a > 0.3 and b < dur - 0.3]
    if voiced_share > 0.3 and long_q:
        out.append(Finding("pauses", "warn", f"long pauses at {_say_spans(_round(long_q))}: dead air loses people",
                           "cut them (the clean-cut effect shortens pauses to a breath)", spans=_round(long_q),
                           at=long_q[0][0]))
    elif voiced_share > 0.3:
        out.append(Finding("pauses", "pass", "no pause of 2.5 s or more"))
    drawn["clipped"] = [a / rate for a, _ in events[:200]]
    drawn["lufs"], drawn["tp"] = m.i, m.tp
    return out, drawn


def _round(spans):
    return [(round(a, 2), round(b, 2)) for a, b in spans]


def _voiced_mask(quiet_src, tmap, n: int, fps: float) -> np.ndarray:
    """Per video frame: does the source have speech at the moment that frame shows?"""
    out = np.zeros(n, bool)
    starts = np.array([a for a, _ in quiet_src]) if quiet_src else np.zeros(0)
    ends = np.array([b for _, b in quiet_src]) if quiet_src else np.zeros(0)
    for j in range(n):
        t = tmap.src[j]
        k = np.searchsorted(starts, t, side="right") - 1
        out[j] = not (k >= 0 and t < ends[k])
    return out


class Picture:
    """One pass over the video's frames: what the black, frozen, flash and colour checks need."""

    def __init__(self, f: Facts):
        info = f.info
        k = ANALYSIS / max(info.width, info.height)
        self.size = (max(2, int(round(info.width * k / 2)) * 2), max(2, int(round(info.height * k / 2)) * 2))
        n = info.frames
        self.mean = np.zeros(n)
        self.dark = np.zeros(n)
        self.diff = np.full(n, np.inf)
        self.hist = []  # (frame, p2, p50, p98, saturation) every so often
        step = max(1, int(round(info.rate / 2)))
        prev = None
        import cv2

        for j, fr in enumerate(video.FrameReader(info, size=self.size)):
            if j >= n:
                break
            g = _gray(fr)
            self.mean[j] = g.mean()
            self.dark[j] = (g < BLACK_PIX).mean()
            small = cv2.resize(g, (64, max(2, int(64 * g.shape[0] / g.shape[1]))), interpolation=cv2.INTER_AREA)
            if prev is not None:
                self.diff[j] = float(np.abs(small - prev).mean())
            prev = small
            if j % step == 0:
                mx = fr.max(axis=2).astype(np.float32)
                mn = fr.min(axis=2).astype(np.float32)
                sat = float(np.mean((mx - mn) / np.maximum(mx, 1.0)))
                p2, p50, p98 = np.percentile(g, (2, 50, 98))
                self.hist.append((j, float(p2), float(p50), float(p98), sat))


def fingerprint(path) -> str:
    """The video's own bytes, in short: its size and the start of their SHA-256. Two reports of "the
    same file" with different fingerprints were of two renders."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return f"{Path(str(path)).stat().st_size} bytes, sha256 {h.hexdigest()[:12]}"


def planned_source(path) -> Optional[str]:
    """The clip the plan beside the video says it was made from, when that file is there."""
    plan = Path(str(path)).with_suffix(".plan.json")
    try:
        clip = json.loads(plan.read_text()).get("clip")
    except (OSError, ValueError, AttributeError):
        return None
    if isinstance(clip, str) and clip and Path(clip).is_file() and Path(clip).resolve() != Path(str(path)).resolve():
        return clip
    return None


def _planned_holds(path) -> list[tuple[float, float]]:
    """The holds the plan beside the video says it made (`<video>.plan.json`: a freeze's `output_at`
    and `hold`, or a list of `holds` as [start, end] seconds), so a freeze made on purpose is not
    taken for an export fault."""
    import json

    plan = Path(str(path)).with_suffix(".plan.json")
    try:
        p = json.loads(plan.read_text())
    except (OSError, ValueError):
        return []
    out = []
    fr = p.get("freeze") if isinstance(p, dict) else None
    if isinstance(fr, dict) and "output_at" in fr and "hold" in fr:
        out.append((float(fr["output_at"]), float(fr["output_at"]) + float(fr["hold"])))
    for h in (p.get("holds") or []) if isinstance(p, dict) else []:
        if isinstance(h, (list, tuple)) and len(h) == 2:
            out.append((float(h[0]), float(h[1])))
    return out


def picture_checks(f: Facts, pic: Picture, src: Optional[Facts], tmap, voiced_frames: Optional[np.ndarray]) -> list[Finding]:
    out = []
    info = f.info
    fps, n = info.rate, info.frames
    # Black.
    black = (pic.dark >= BLACK_SHARE) & (pic.mean < 0.08 * 255)
    spans = _spans(black, fps)
    if spans:
        secs = _secs(spans, fps)
        inner = [s for s in spans if s[0] > 0 and s[1] < n]
        worst = max(b - a for a, b in spans) / fps
        if inner and max(b - a for a, b in inner) >= 2:
            out.append(Finding("black frames", "fail", f"black frames at {_say_spans(secs)} (the longest "
                               f"{worst:.2f} s)", "cut them, or put the picture back", spans=secs, at=secs[0][0]))
        else:
            out.append(Finding("black frames", "warn", f"black frames at {_say_spans(secs)}, at "
                               f"{'the start' if spans[0][0] == 0 else 'the end'}: fine for a fade, but the first "
                               f"frame is the cover some apps show", spans=secs, at=secs[0][0]))
    else:
        out.append(Finding("black frames", "pass", "none"))
    # Frozen: the picture stops changing (a black stretch is reported as black, not twice).
    frozen = (pic.diff < FROZEN) & ~black
    runs = _spans(frozen, fps, gap=0, least=max(2, int(round(0.25 * fps))))
    end_hold = None
    bad, soft, paused, spoken = [], [], [], False
    meant = _planned_holds(f.info.path)
    for a, b in runs:
        start, stop = a - 1, b  # the frame before the first unchanged one is part of the freeze
        if stop >= n - 1 and (stop - start) / fps <= 6.0:
            end_hold = (start / fps, stop / fps)
            continue
        talking = voiced_frames is not None and voiced_frames[a:b].mean() > 0.3
        if src is not None and tmap is not None and np.min(np.asarray(tmap.src)[a:b]) >= src.info.duration - 1.5 / src.info.rate:
            continue  # past the source's last frame: an end card held, not a freeze
        if any(h0 - 0.3 <= start / fps and stop / fps <= h1 + 0.3 for h0, h1 in meant):
            paused.append((start / fps, stop / fps))  # the plan beside the video holds the picture here
            continue
        if voiced_frames is not None and not talking:
            # Picture and sound stop together (a freeze for effect, the clip carrying on after it). An
            # export fault freezes the picture while the sound plays on.
            paused.append((start / fps, stop / fps))
            continue
        spoken |= talking
        source_moves = False
        if src is not None and tmap is not None:
            source_moves = tmap.source_moving(a, b)
            if not source_moves and not tmap.held(a, b):
                continue  # the source is as still there: nothing was lost
        if talking or source_moves or (b - a) / fps >= 1.0:
            bad.append((start / fps, stop / fps))
        elif src is None:
            soft.append((start / fps, stop / fps))
        # else: the source holds still there too, so nothing was lost
    singles = int(np.sum(frozen & ~np.convolve(frozen, [1, 0, 1], "same").astype(bool)))
    if bad:
        bad = _round(bad)
        why = "while the voice goes on" if spoken else "where the source moves on"
        out.append(Finding("frozen frames", "fail", f"the picture freezes at {_say_spans(bad)} {why}".strip(),
                           "render it again; a freeze in the middle is an export fault", spans=bad, at=bad[0][0]))
    elif paused:
        paused = _round(paused)
        out.append(Finding("frozen frames", "warn", f"the picture holds at {_say_spans(paused)}, a freeze for effect (the plan "
                           "beside it says so, or the sound stops with it)", "check it is meant to", spans=paused, at=paused[0][0]))
    elif soft:
        soft = _round(soft)
        out.append(Finding("frozen frames", "warn", f"the picture holds still at {_say_spans(soft)}",
                           "check it is meant to (a held frame for effect)", spans=soft, at=soft[0][0]))
    elif singles > 0.08 * n:
        out.append(Finding("frozen frames", "warn", f"{singles} single frames repeat ({singles / n:.0%} of them): the "
                           f"video was converted from another frame rate, and motion stutters a little",
                           "export at the source's own frame rate"))
    else:
        out.append(Finding("frozen frames", "pass", "none" + (f" (the last {end_hold[1] - end_hold[0]:.1f} s hold one "
                                                              f"frame, as an end card does)" if end_hold else "")))
    # Flashes: changes of brightness that reverse, more than three in a second.
    lum = pic.mean / 255.0
    big = np.abs(np.diff(lum)) >= 0.10
    sign = np.sign(np.diff(lum))
    flips = [i for i in range(1, len(sign)) if big[i] and sign[i] != 0 and sign[i] != sign[i - 1]]
    worst = 0
    for i in flips:
        worst = max(worst, sum(1 for k in flips if i <= k < i + fps) // 2)
    if worst > 3:
        out.append(Finding("flashes", "warn", f"{worst} flashes within one second: rapid flashing can harm people "
                           f"with photosensitive epilepsy", "slow the flashes to three a second or fewer"))
    else:
        out.append(Finding("flashes", "pass", "no rapid flashing"))
    return out


def colour_checks(f: Facts, pic: Picture, src: Optional[Facts], aligned: Optional[list] = None) -> list[Finding]:
    """HDR or log footage shown as ordinary video looks flat and grey. `aligned` is a few frames of
    the video beside the part of the source they show, read converted and raw (`_aligned_samples`)."""
    from ..core import hdr

    out = []
    kind = hdr.kind(f.transfer)
    h = np.array([r[1:] for r in pic.hist]) if pic.hist else np.zeros((0, 4))
    p2, p50, p98, sat = (np.median(h[:, k]) if len(h) else 0.0 for k in range(4))
    measured = f"darkest {p2:.0f}, middle {p50:.0f}, brightest {p98:.0f} (of 255), colour {sat:.2f}"
    if kind != "sdr":
        out.append(Finding("colour", "warn", f"the video is HDR ({kind.upper()}): HDR phones show it bright, others "
                           f"and many apps show it flat and grey ({measured})",
                           "deliver it as SDR (BT.709); the pack converts HDR clips to SDR as it reads them"))
        return out
    if src is not None and hdr.kind(src.transfer) != "sdr":
        # Was the HDR source converted, or written out as it was? Compare the video's picture with the
        # source read both ways, at the same moments.
        verdict = _hdr_verdict(aligned) if aligned else _hdr_converted(f, src, pic)
        if verdict is not None:
            conv, raw, grey = verdict
            if raw < conv:
                level = "fail" if grey else "warn"
                out.append(Finding("colour", level, f"the source is HDR ({hdr.kind(src.transfer).upper()}) and the video "
                                   f"is its picture unconverted: {'washed out and grey' if grey else 'a little flat and grey'}"
                                   f" ({measured})", "render it again with the current pack: it converts HDR as it reads"))
                return out
            out.append(Finding("colour", "pass", f"the HDR source was converted to SDR ({measured})"))
            return out
    flat = p98 - p2 < 150 and p2 > 28 and sat < 0.12
    if src is None:
        # Alone, the picture cannot say whether an HDR clip was converted: tier-list's old D, drawn
        # unconverted, reads PASS here and WARN against its source. Say which was judged.
        measured += "; judged on the picture alone: give source= to check an HDR clip was converted"
    if flat:
        out.append(Finding("colour", "warn", f"the picture looks flat and grey, like HDR or log footage that was never "
                           f"converted ({measured})", "convert it to BT.709 (or grade it) before posting"))
    else:
        out.append(Finding("colour", "pass", measured))
    return out


def _aligned_samples(f: Facts, src: Facts, tmap: "TimeMap", geo: "Geometry", size: tuple[int, int]) -> list:
    """For six moments: the video's frame, and the part of the source it shows, read converted and
    read raw (as a tool that ignores HDR would), each grey and the same size."""
    import dataclasses

    aw, ah = size
    sw, sh = geo.src_size
    raw_info = dataclasses.replace(src.info, transfer="sdr", color_matrix="bt709")
    out = []
    for j in np.linspace(0, f.info.frames - 1, 8).astype(int)[1:-1]:
        v = _gray(video.read_frame(f.info, int(j), size=(aw, ah)))
        k = tmap.frame(int(j))
        a = _gray(video.read_frame(src.info, k, size=(sw, sh)))
        b = _gray(video.read_frame(raw_info, k, size=(sw, sh)))
        if geo.kind in ("a crop that moves across", "a crop that moves up and down"):
            across = geo.kind == "a crop that moves across"
            p = _crop_pos(v, a, across, None)
            win = (slice(0, ah), slice(p, p + aw)) if across else (slice(p, p + ah), slice(0, aw))
            a, b = a[win], b[win]
        elif geo.kind == "the whole picture fitted inside":
            x, y = int(round(geo.fit[1])), int(round(geo.fit[2]))
            hh, ww = min(sh, ah - y), min(sw, aw - x)
            v, a, b = v[y:y + hh, x:x + ww], a[:hh, :ww], b[:hh, :ww]
        if a.shape == v.shape:
            out.append((v, a, b))
    return out


def _hdr_verdict(samples: list):
    """(distance to the converted source, to the raw source, washed out?) from aligned samples."""
    conv = float(np.median([np.abs(v - a).mean() for v, a, _ in samples]))
    raw = float(np.median([np.abs(v - b).mean() for v, _, b in samples]))
    # How grey the unconverted picture is depends on the footage: a white room read raw loses its
    # whites (washed out), a darker scene only a little. Measured on the source itself, so a new
    # background in the video does not change the answer.
    grey = float(np.median([np.percentile(a, 98) - np.percentile(b, 98) for _, a, b in samples])) > 15
    return conv, raw, grey


def _hdr_converted(f: Facts, src: Facts, pic: Picture):
    """(distance to the converted source, distance to the raw source, washed out?) over a few frames,
    or None when the two cannot be compared (another shape, another length)."""
    import dataclasses

    if abs(f.info.duration - src.info.duration) > 0.5:
        return None
    raw_info = dataclasses.replace(src.info, transfer="sdr", color_matrix="bt709")
    conv, raw, greys = [], [], []
    for j, *_ in pic.hist[:: max(1, len(pic.hist) // 6)][:6]:
        t = j / f.info.rate
        v = _gray(video.read_frame(f.info, j, size=(160, int(160 * f.info.height / f.info.width))))
        k = src.info.frame_at(t)
        size = (160, int(160 * src.info.height / src.info.width))
        a = _gray(video.read_frame(src.info, k, size=size))
        b = _gray(video.read_frame(raw_info, k, size=size))
        hv, ha, hb = (np.percentile(x, (2, 25, 50, 75, 98)) for x in (v, a, b))
        conv.append(float(np.abs(hv - ha).mean()))
        raw.append(float(np.abs(hv - hb).mean()))
        greys.append(float(ha[-1] - hv[-1]))  # how much darker its whites are than the converted ones
    grey = np.median(greys) > 15
    return float(np.median(conv)), float(np.median(raw)), bool(grey)


# ---- lining the video up with its source -----------------------------------------------------------

class TimeMap:
    """For every frame of the video, the moment of the source it shows."""

    def __init__(self, src_times: np.ndarray, kind: str, cuts: list[int], n_src: int, fps_src: float,
                 src_diff: Optional[np.ndarray] = None, source: Optional["Facts"] = None):
        self.src = src_times
        self.kind = kind
        self.cuts = cuts
        self.near_cut = np.zeros(len(src_times), bool)
        for c in cuts:
            self.near_cut[max(0, c - 3):c + 3] = True
        self.n_src, self.fps_src = n_src, fps_src
        self.src_diff = src_diff
        self.source = source  # read for its motion only when a freeze needs judging

    def frame(self, j: int) -> int:
        return int(min(self.n_src - 1, max(0, round(self.src[j] * self.fps_src))))

    def snap(self, diff: np.ndarray, fps: float, reach: float = 0.8) -> None:
        """The sound places a cut to within half a second; the picture shows exactly where it is (the
        biggest jump from one frame to the next nearby). Each cut moves there, and the frames between
        take the offset of their own side."""
        n = len(self.src)
        t = np.arange(n) / fps
        off = self.src - t
        moved = []
        for c in self.cuts:
            lo, hi = max(1, c - int(reach * fps)), min(n - 1, c + int(reach * fps))
            if hi <= lo:
                moved.append(c)
                continue
            d = np.nan_to_num(diff[lo:hi + 1], nan=0.0, posinf=0.0)
            new = lo + int(np.argmax(d))
            before, after = off[max(0, lo - 1)], off[min(n - 1, hi)]
            off[lo:new] = before
            off[new:hi + 1] = after
            moved.append(new)
        self.src = np.clip(t + off, 0, (self.n_src - 1) / self.fps_src)
        self.cuts = moved
        self.near_cut = np.zeros(n, bool)
        for c in moved:
            self.near_cut[max(0, c - 3):c + 3] = True

    def held(self, a: int, b: int) -> bool:
        """Do video frames a..b show one and the same source frame (a frame held on purpose)?"""
        return len({self.frame(j) for j in range(a, b)}) <= 1

    def source_moving(self, a: int, b: int) -> bool:
        """Does the source move over the moments video frames a..b show?"""
        if self.src_diff is None and self.source is not None:
            self.src_diff = _source_motion(self.source)
        if self.src_diff is None:
            return False
        ks = sorted({self.frame(j) for j in range(a, b)})
        if len(ks) <= 1:
            return False  # the video shows one source frame there: a held frame, not a freeze
        vals = self.src_diff[[k for k in ks if k < len(self.src_diff)]]
        return bool(np.nanmean(vals) > 3 * FROZEN)


def _norm_env(env) -> np.ndarray:
    """The speech envelope on a common scale: everything more than 45 dB under the voice is one
    floor (so a muted stretch cannot swamp the rest), centred and scaled by its middle half."""
    top = float(np.percentile(env.db, 90))
    x = np.maximum(env.db, top - 45.0)
    spread = float(np.percentile(x, 75) - np.percentile(x, 25))
    return ((x - np.median(x)) / (spread + 1e-6)).astype(np.float32)


def _held_before(holds: list[tuple[float, float]], t) -> np.ndarray:
    """Seconds of the holds that lie before each moment `t` of the video."""
    t = np.asarray(t, float)
    return sum((np.clip(t - a, 0.0, b - a) for a, b in holds), np.zeros_like(t))


def time_map(f: Facts, src: Facts, env_v, env_s, holds: list[tuple[float, float]] = ()) -> Optional[TimeMap]:
    """Line the video up with the source by their sound: the same moment, a trim, or cuts.

    `holds` are the stretches of the video where the plan beside it says the picture is held on one
    frame (a freeze). The sound there is not the source's (a scratch, the room's tone looped), so it
    places nothing, and after a hold the video shows the source later by the hold's length: the one
    place the offset falls instead of growing. Without them a freeze's room tone matched the source
    anywhere, and the clip after it was lined up with the source's last frame (freeze-point on D:
    a false 13% face FAIL, the outline of the person counted as something over the face)."""
    n, fps = f.info.frames, f.info.rate
    holds = sorted((float(a), float(b)) for a, b in holds if b > a)
    hop = env_v.hop
    a, b = _norm_env(env_v), _norm_env(env_s)
    dur_v, dur_s = f.info.duration, src.info.duration
    t = np.arange(n) / fps
    # Window by window, where in the source each stretch of the video's sound is.
    L, step = int(1.5 / hop), int(0.5 / hop)
    if len(b) < L or len(a) < L:
        return None
    starts = list(range(0, len(a) - L + 1, step))
    found = []  # (centre time, offset seconds, score)
    fb = np.fft.rfft(np.concatenate([b, np.zeros(L)]))
    csum = np.concatenate([[0.0], np.cumsum(b)])
    csum2 = np.concatenate([[0.0], np.cumsum(b * b)])
    raw = env_v.db
    at_zero = []  # how well each window matches the source at no offset
    for s0 in starts:
        if any(s0 * hop < h1 and (s0 + L) * hop > h0 for h0, h1 in holds):
            continue  # its sound is the hold's, not the source's
        w = a[s0:s0 + L]
        seg = raw[s0:s0 + L]
        if float(np.percentile(seg, 90) - np.percentile(seg, 10)) < 6.0 or w.std() < 1e-3:
            continue  # nothing but a level here: no shape to place
        if s0 + L <= len(b):
            z = b[s0:s0 + L]
            at_zero.append(float(np.corrcoef(w, z)[0, 1]) if z.std() > 1e-6 else 0.0)
        w = (w - w.mean()) / (w.std() + 1e-6)
        corr = np.fft.irfft(fb * np.conj(np.fft.rfft(np.concatenate([w, np.zeros(len(b))]), len(b) + L)),
                            len(b) + L)[:len(b) - L + 1]
        mu = (csum[L:] - csum[:-L]) / L
        # A flat stretch of the source (a long pause at the floor) cannot be where a voice is: its
        # spread is given a floor, or dividing by almost nothing would make it the best match.
        sd = np.sqrt(np.maximum((csum2[L:] - csum2[:-L]) / L - mu * mu, 0.0))
        ncc = corr / (L * np.maximum(sd, 0.25))
        p = int(np.argmax(ncc))
        if ncc[p] > 0.6:
            found.append(((s0 + L / 2) * hop, (p - s0) * hop, float(ncc[p])))
    same = TimeMap(np.minimum(t, dur_s - 1 / src.info.rate), "the same frames", [], src.info.frames, src.info.rate)
    # The same timing (the common case): the lengths match and most stretches match the source at no
    # offset. A stretch whose sound was changed (muted, music laid over) does not count against it.
    if abs(dur_v - dur_s) < 0.2 and (not at_zero or np.mean([r_ >= 0.7 for r_ in at_zero]) >= 0.7):
        return same
    if len(found) < 3:
        return same if abs(dur_v - dur_s) < 0.2 else None
    # The offset only grows (a cut takes source time out; nothing is played twice): keep the longest
    # run of windows whose offsets never fall, then give every frame the offset of the nearest window.
    cen = np.array([c for c, _, _ in found])
    off = np.array([o for _, o, _ in found])
    keep = _longest_rising(off + _held_before(holds, cen), tol=0.03)
    cen, off = cen[keep], off[keep]
    per_frame = off[np.clip(np.searchsorted(cen, t), 0, len(cen) - 1)]
    near = np.abs(cen[np.clip(np.searchsorted(cen, t), 0, len(cen) - 1)] - t)
    alt = off[np.clip(np.searchsorted(cen, t) - 1, 0, len(cen) - 1)]
    near_alt = np.abs(cen[np.clip(np.searchsorted(cen, t) - 1, 0, len(cen) - 1)] - t)
    per_frame = np.where(near_alt < near, alt, per_frame)
    per_frame = _median(per_frame, 5)
    src_t = np.clip(t + per_frame, 0, dur_s - 1 / src.info.rate)
    in_hold = np.zeros(n, bool)
    for h0, h1 in holds:
        a0, a1 = int(round(h0 * fps)), min(n, int(round(h1 * fps)))
        if 0 <= a0 < a1:
            # The frame held is the source's moment where the hold begins (the offset before it).
            k = max(0, a0 - 1)
            src_t[a0:a1] = min(max(0.0, t[a0] + per_frame[k]), dur_s - 1 / src.info.rate)
            in_hold[a0:a1 + 1] = True
    cuts = [j for j in range(1, n) if abs(per_frame[j] - per_frame[j - 1]) > 0.5 / fps and not in_hold[j]]
    if cuts or not in_hold.any():
        kind = "cuts found by the sound" if cuts else f"offset by {per_frame[0]:+.2f} s"
    else:
        kind = "the same frames, with the planned hold" + ("s" if len(holds) > 1 else "")
    return TimeMap(src_t, kind, cuts, src.info.frames, src.info.rate)


def _longest_rising(x: np.ndarray, tol: float) -> np.ndarray:
    """Indices of the longest subsequence of x that never falls by more than `tol`."""
    n = len(x)
    best = np.ones(n, int)
    prev = np.full(n, -1)
    for i in range(n):
        for j in range(i):
            if x[i] >= x[j] - tol and best[j] + 1 > best[i]:
                best[i], prev[i] = best[j] + 1, j
    i = int(np.argmax(best))
    out = []
    while i >= 0:
        out.append(i)
        i = prev[i]
    return np.array(out[::-1])


def _median(x: np.ndarray, k: int) -> np.ndarray:
    pad = np.pad(x, (k // 2, k // 2), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=1)


class Geometry:
    """Where the video's picture sits in the source's: the whole frame, a crop that moves across (or
    up and down), or the whole source fitted inside a bigger frame."""

    def __init__(self, kind: str, src_size: tuple[int, int], fit: Optional[tuple[float, float, float]] = None):
        self.kind = kind
        self.src_size = src_size  # the size the source is read at for the comparison
        self.fit = fit  # for "fit": scale, x, y of the source inside the video (analysis pixels)


def _grad(g: np.ndarray) -> np.ndarray:
    import cv2

    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def geometry(f: Facts, src: Facts, tmap: TimeMap, size: tuple[int, int]) -> tuple[Optional[Geometry], float]:
    """Try each way the video could sit in the source on a few frames; keep the one that fits best."""
    import cv2

    aw, ah = size
    W, H = src.info.width, src.info.height
    va = aw / ah
    sa = W / H
    picks = np.linspace(0, f.info.frames - 1, 8).astype(int)
    tries = {}
    same = abs(va - sa) < 0.01 * sa
    if same:
        tries["the same frame"] = (aw, ah)
    elif sa > va:  # the source is wider: a crop sliding across, at full height
        tries["a crop that moves across"] = (max(aw + 2, int(round(ah * sa))), ah)
    else:
        tries["a crop that moves up and down"] = (aw, max(ah + 2, int(round(aw / sa))))
    # The whole source fitted inside the video (a fill, or bars).
    s = min(aw / W, ah / H)
    tries["the whole picture fitted inside"] = (max(2, int(round(W * s))), max(2, int(round(H * s))))
    scores = {k: [] for k in tries}
    for j in picks:
        v = _grad(_gray(video.read_frame(f.info, int(j), size=(aw, ah))))
        k = tmap.frame(int(j))
        for name, (sw, sh) in tries.items():
            g = _grad(_gray(video.read_frame(src.info, k, size=(sw, sh))))
            if name == "the same frame":
                sc = float(cv2.matchTemplate(v, g, cv2.TM_CCOEFF_NORMED).max())
            elif name == "the whole picture fitted inside":
                if sw > aw or sh > ah:
                    continue
                sc = float(cv2.matchTemplate(v, g, cv2.TM_CCOEFF_NORMED).max())
            else:
                sc = float(cv2.matchTemplate(g, v, cv2.TM_CCOEFF_NORMED).max())
            scores[name].append(sc)
    best, best_v = None, -1.0
    for name, vals in scores.items():
        if vals and np.median(vals) > best_v:
            best, best_v = name, float(np.median(vals))
    if best is None or best_v < 0.3:
        return None, best_v
    sw, sh = tries[best]
    if best == "the whole picture fitted inside":
        g = _grad(_gray(video.read_frame(src.info, tmap.frame(int(picks[len(picks) // 2])), size=(sw, sh))))
        v = _grad(_gray(video.read_frame(f.info, int(picks[len(picks) // 2]), size=(aw, ah))))
        res = cv2.matchTemplate(v, g, cv2.TM_CCOEFF_NORMED)
        _, _, _, loc = cv2.minMaxLoc(res)
        return Geometry(best, (sw, sh), fit=(s, float(loc[0]), float(loc[1]))), best_v
    return Geometry(best, (sw, sh)), best_v


# ---- the face and the safe zone, frame by frame ----------------------------------------------------

class Compare:
    """Every frame of the video against the moment of the source it shows."""

    def __init__(self):
        self.face_cover = None  # per frame: the share of the face box showing something new (NaN: not judged)
        self.outside = None  # per frame: the share of the frame added outside the safe zone
        self.where = None  # per frame: where outside (bottom, top, left, right) as a short word
        self.wordy = None  # per frame: True when what is outside looks like a line of words
        self.word_where = None  # per frame: where those words are
        self.replaced = None  # per frame: True where the whole picture changed (a new room, a card)
        self.elsewhere = None  # per frame: True where the video shows another picture (a cutaway)
        self.boxes = None  # per frame: the face box in the video's pixels (Rect or None)
        self.maps = {}  # frame -> the added-shapes mask at analysis size, for the sheet
        self.judged = 0.0
        self.aligned_by = ""


def compare(f: Facts, src: Facts, src_face: facemod.FaceTrack, tmap: TimeMap, geo: Geometry,
            size: tuple[int, int], zone_on: bool) -> Compare:
    import cv2

    info, n = f.info, f.info.frames
    aw, ah = size
    sw, sh = geo.src_size
    k_src = sw / src.info.width  # source pixels to the size the source is read at
    zone = SafeZone(aw, ah).rect
    c = Compare()
    c.face_cover = np.full(n, np.nan)
    c.outside = np.zeros(n)
    c.where = [""] * n
    c.wordy = np.zeros(n, bool)
    c.word_where = [""] * n
    c.replaced = np.zeros(n, bool)
    c.elsewhere = np.zeros(n, bool)
    c.boxes = [None] * n
    crop_x = geo.kind == "a crop that moves across"
    crop_y = geo.kind == "a crop that moves up and down"
    # The source's frames are read once, in order: the timing only moves forwards.
    reader = iter(video.FrameReader(src.info, size=(sw, sh)))
    buf: dict[int, np.ndarray] = {}
    read_upto = -1

    def src_frame(k: int) -> Optional[np.ndarray]:
        nonlocal read_upto
        k = int(min(max(k, 0), src.info.frames - 1))
        while read_upto < k + 1:
            nxt = next(reader, None)
            if nxt is None:
                break
            read_upto += 1
            buf[read_upto] = _gray(nxt)
            for old in [x for x in buf if x < read_upto - 4]:
                del buf[old]
        return buf.get(k, buf.get(read_upto))

    keep_best: list[tuple[float, int, np.ndarray]] = []
    pos_prev = None
    judged = 0

    def small(img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, (max(8, img.shape[1] // 4), max(8, img.shape[0] // 4)), interpolation=cv2.INTER_AREA)

    for j, fr in enumerate(video.FrameReader(info, size=(aw, ah))):
        if j >= n:
            break
        v = _gray(fr)
        k = tmap.frame(j)
        centre = src_frame(k)
        if centre is None:
            continue
        # Where the video's frame sits in the source's (found once a frame, on the middle candidate).
        if crop_x or crop_y:
            pos = _crop_pos(v, centre, crop_x, pos_prev)
            pos_prev = pos
            dx, dy = (-pos, 0) if crop_x else (0, -pos)
        elif geo.kind == "the whole picture fitted inside":
            dx, dy = int(round(geo.fit[1])), int(round(geo.fit[2]))
        else:
            dx = dy = 0

        def place(sf: np.ndarray) -> Optional[np.ndarray]:
            if crop_x or crop_y:
                a = sf[-dy:-dy + ah, -dx:-dx + aw]
                return a if a.shape == v.shape else None
            if geo.kind == "the whole picture fitted inside":
                a = v.copy()
                hh, ww = min(sf.shape[0], ah - dy), min(sf.shape[1], aw - dx)
                a[dy:dy + hh, dx:dx + ww] = sf[:hh, :ww]
                return a
            return sf if sf.shape == v.shape else None

        # A frame early or late is forgiven: of the three, the one closest in shape.
        best = None
        sv = _structure(small(v))
        for kk in (k - 1, k, k + 1):
            sf = src_frame(kk)
            if sf is None:
                continue
            aligned = place(sf)
            if aligned is None:
                continue
            err = float(np.abs(sv - _structure(small(aligned))).mean())
            if best is None or err < best[0]:
                best = (err, kk, aligned)
        if best is None:
            continue
        _, kk, aligned = best
        mask = _changed(v, aligned)
        if geo.kind == "the whole picture fitted inside":
            s, fx, fy = geo.fit
            outside_fit = np.ones_like(mask)
            yy, xx = int(round(fy)), int(round(fx))
            outside_fit[yy + 2:yy + sh - 2, xx + 2:xx + sw - 2] = 0
            mask = mask * (1 - outside_fit)  # the fill around the fitted picture is not an overlay
        share = float(mask.mean())
        c.elsewhere[j] = _matching(v, aligned) < ELSEWHERE
        c.replaced[j] = share > REPLACED or c.elsewhere[j]
        # The face: the source's face box (brows to chin), a little narrower than cheek to cheek, so
        # the outline of the head against a new background is not taken for something over the face.
        b = src_face.box(kk)
        if b is not None and src_face.found_share > 0.3 and not c.elsewhere[j]:
            r = Rect(b.x0 + 0.05 * b.w, b.y0, b.x1 - 0.05 * b.w, b.y1 + 0.03 * b.h)
            r = r.scale(k_src, k_src, dx, dy)
            x0, y0 = int(max(0, r.x0)), int(max(0, r.y0))
            x1, y1 = int(min(aw, r.x1)), int(min(ah, r.y1))
            if x1 - x0 > 8 and y1 - y0 > 8 and not tmap.near_cut[j]:
                c.face_cover[j] = _covered(mask[y0:y1, x0:x1])
                c.boxes[j] = Rect(x0, y0, x1, y1)
                judged += 1
        # Anything added that reaches outside the safe zone (only where the picture is not wholly new).
        if zone_on and not c.replaced[j] and not tmap.near_cut[j]:
            inside = np.zeros_like(mask)
            inside[int(zone.y0):int(zone.y1), int(zone.x0):int(zone.x1)] = 1
            # The letters of a line are joined into one shape first, so a caption counts as a caption.
            line = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(0.04 * aw)) | 1, max(3, int(0.01 * ah)) | 1))
            joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, line)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
            strokes = np.abs(_structure(v)) > 1.0  # the video's own edges: letters are many strokes
            # Pieces of one picture (a screenshot's title bar, a card's edge) join up at a coarser scale:
            # a line that belongs to a big added region is part of a picture, not a caption.
            coarse = cv2.morphologyEx(joined, cv2.MORPH_CLOSE, cv2.getStructuringElement(
                cv2.MORPH_RECT, (max(3, int(0.05 * aw)) | 1, max(3, int(0.03 * ah)) | 1)))
            _, regions, region_stats, _ = cv2.connectedComponentsWithStats(coarse, connectivity=8)
            where, word_where, area = set(), set(), 0
            for i in range(1, count):
                st = stats[i]
                bx, by, bw, bh = (int(st[cv2.CC_STAT_LEFT]), int(st[cv2.CC_STAT_TOP]), int(st[cv2.CC_STAT_WIDTH]),
                                  int(st[cv2.CC_STAT_HEIGHT]))
                if _rule(st, aw, ah) or bw * bh > 0.30 * aw * ah:
                    continue  # a divider, or a whole new background: drawn on purpose, not an overlay
                blob = labels[by:by + bh, bx:bx + bw] == i
                out_px = int((blob & (inside[by:by + bh, bx:bx + bw] == 0)).sum())
                if out_px < BLOB * aw * ah:
                    continue
                area += out_px
                # A line of words placed out there (mostly outside the zone), not a picture that
                # reaches into the margin from inside it.
                ys, xs = np.nonzero(blob)
                region = regions[by + int(ys[0]), bx + int(xs[0])]
                in_picture = region_stats[region, cv2.CC_STAT_AREA] > 0.08 * aw * ah
                # A line of words crosses many edges along a row (2.9 every 10 px for a caption,
                # measured); a border or a roofline barely any (0.2).
                rows = (strokes[by:by + bh, bx:bx + bw] & blob)[bh // 4:max(bh // 4 + 1, 3 * bh // 4)]
                crossings = np.abs(np.diff(rows.astype(np.int8), axis=1)).sum(axis=1).mean() / max(1, bw) * 10
                words = (0.015 * ah <= bh <= 0.15 * ah and bw >= 1.5 * bh and not in_picture and crossings >= 1.0
                         and out_px >= 0.6 * int(st[cv2.CC_STAT_AREA]))
                here = set()
                if by + bh > zone.y1:
                    here.add("bottom")
                if by < zone.y0:
                    here.add("top")
                if bx < zone.x0:
                    here.add("left")
                if bx + bw > zone.x1:
                    here.add("right")
                where |= here
                if words:
                    word_where |= here
            if area:
                c.outside[j] = area / (aw * ah)
                c.where[j] = "+".join(sorted(where))
                c.wordy[j] = bool(word_where)
                c.word_where[j] = "+".join(sorted(word_where))
        # Keep the masks of the frames most worth showing.
        worth = max(np.nan_to_num(c.face_cover[j]), c.outside[j] * 20)
        if worth > 0.02:
            keep_best.append((worth, j, mask))
            keep_best.sort(key=lambda t_: -t_[0])
            del keep_best[12:]
    c.maps = {j: m for _, j, m in keep_best}
    c.judged = judged / max(1, n)
    c.aligned_by = geo.kind
    return c


def _rule(stat, aw: int, ah: int) -> bool:
    """A thin line running most of the frame's length (a wipe's divider, a border): drawn on purpose
    and not words, so not counted against the safe zone."""
    import cv2

    w, h = int(stat[cv2.CC_STAT_WIDTH]), int(stat[cv2.CC_STAT_HEIGHT])
    return (w <= 0.02 * aw and h >= 0.4 * ah) or (h <= 0.02 * ah and w >= 0.4 * aw)


def _matching(v: np.ndarray, s: np.ndarray, rows: int = 12) -> float:
    """The share of the frame's textured blocks whose shapes still follow the source's there (a
    correlation over 0.3), on a copy 360 high. An overlay leaves most blocks matching; a cutaway to
    another picture leaves almost none, whatever it adds or takes away."""
    import cv2

    h = 360
    w = max(8, int(round(v.shape[1] * h / v.shape[0])))
    a = _structure(cv2.resize(v, (w, h), interpolation=cv2.INTER_AREA))
    b = _structure(cv2.resize(s, (w, h), interpolation=cv2.INTER_AREA))
    cols = max(4, round(rows * w / h))
    ok = tot = 0
    for y in range(rows):
        for x in range(cols):
            pa = a[y * h // rows:(y + 1) * h // rows, x * w // cols:(x + 1) * w // cols].ravel()
            pb = b[y * h // rows:(y + 1) * h // rows, x * w // cols:(x + 1) * w // cols].ravel()
            if pa.std() < 0.15 and pb.std() < 0.15:
                continue  # flat in both: says nothing
            tot += 1
            if pa.std() > 1e-6 and pb.std() > 1e-6 and float(np.corrcoef(pa, pb)[0, 1]) > 0.3:
                ok += 1
    return ok / tot if tot else 1.0


def _covered(part: np.ndarray) -> float:
    """How much of a box something added covers: the area each added shape spans (its bounding box,
    so a word counts by the band it sits in, letters and gaps alike), as a share of the box. Shapes
    too small to be anything but noise are left out."""
    import cv2

    count, _, stats, _ = cv2.connectedComponentsWithStats(part.astype(np.uint8), connectivity=8)
    span = np.zeros(part.shape, bool)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= 20:
            x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                          int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
            span[y:y + h, x:x + w] = True
    return float(span.mean())


def _crop_pos(v: np.ndarray, sf: np.ndarray, across: bool, prev: Optional[int]) -> int:
    """Where the video's frame sits in the source's, along the one axis a crop moves on: near where
    it was in the frame before (a crop glides), or searched for across the whole frame (at a quarter
    of the size, then refined) when it is not found there."""
    import cv2

    gv = _grad(v)
    span = sf.shape[1] - v.shape[1] if across else sf.shape[0] - v.shape[0]
    if prev is not None:
        lo, hi = max(0, prev - 24), min(span, prev + 24)
        part = sf[:, lo:hi + v.shape[1]] if across else sf[lo:hi + v.shape[0], :]
        res = cv2.matchTemplate(_grad(part), gv, cv2.TM_CCOEFF_NORMED)
        line = res[0] if across else res[:, 0]
        if line.max() > 0.5:
            return lo + int(np.argmax(line))
    q = 4
    small_s = cv2.resize(sf, (max(8, sf.shape[1] // q), max(8, sf.shape[0] // q)), interpolation=cv2.INTER_AREA)
    small_v = cv2.resize(v, (max(4, v.shape[1] // q), max(4, v.shape[0] // q)), interpolation=cv2.INTER_AREA)
    res = cv2.matchTemplate(_grad(small_s), _grad(small_v), cv2.TM_CCOEFF_NORMED)
    line = res[0] if across else res[:, 0]
    guess = int(np.argmax(line)) * q
    lo, hi = max(0, guess - 2 * q), min(span, guess + 2 * q)
    part = sf[:, lo:hi + v.shape[1]] if across else sf[lo:hi + v.shape[0], :]
    res = cv2.matchTemplate(_grad(part), gv, cv2.TM_CCOEFF_NORMED)
    line = res[0] if across else res[:, 0]
    return lo + int(np.argmax(line))


def face_and_zone_checks(c: Compare, fps: float, zone_on: bool, n: int) -> list[Finding]:
    out = []
    cover = np.nan_to_num(c.face_cover, nan=0.0)
    flagged = cover >= FACE_FLAG
    spans = _spans(flagged, fps, gap=2, least=3)
    judged = f"judged in {c.judged:.0%} of frames, the video lined up with the source as {c.aligned_by}"
    away = float(c.elsewhere.mean()) if c.elsewhere is not None and len(c.elsewhere) else 0.0
    if away > 0.02:
        judged += (f"; {away:.0%} of frames show another picture (a cutaway), where the source says nothing about "
                   "the face or the margins, so they were left out")
    if spans:
        secs = _secs(spans, fps)
        worst = max(float(cover[a:b].max()) for a, b in spans)
        bad = any(float(cover[a:b].max()) >= FACE_FAIL and (b - a) / fps >= FACE_LONG for a, b in spans)
        level = "fail" if bad else "warn"
        j = max(range(n), key=lambda i: cover[i])
        out.append(Finding("face", level, f"something not in the source covers the face at {_say_spans(secs)} (up to "
                           f"{worst:.0%} of the face box); {judged}",
                           "move the words or the graphic off the face, or put them behind the person",
                           spans=secs, at=round(j / fps, 2)))
    elif c.judged > 0:
        peak = float(cover.max()) if len(cover) else 0.0
        out.append(Finding("face", "pass", f"nothing covers the face (at most {peak:.0%} of the face box changed in "
                           f"any frame); {judged}"))
    else:
        out.append(Finding("face", "skip", "the face could not be checked: no face was found in the source"))
    if zone_on:
        flagged = c.outside > 0
        spans = _spans(flagged, fps, gap=2, least=max(3, int(0.2 * fps)))
        if spans:
            secs = _secs(spans, fps)
            wheres = sorted({w for a, b in spans for w in c.where[a:b] if w})
            parts = {"bottom": "the bottom fifth, where the caption and buttons sit",
                     "top": "the top tenth, under the app's bar", "left": "the left margin",
                     "right": "the right margin, where the buttons sit"}
            said = "; ".join(sorted({parts.get(w, w) for p in wheres for w in p.split("+")}))
            words = any(c.wordy[a:b].any() for a, b in spans)
            # Words under the caption and the buttons (the bottom fifth, the right margin) for half a
            # second or more fail; words under the top bar or in the left margin, and pictures, warn.
            long_bad = any(sum(1 for w in c.word_where[a:b] if "bottom" in w or "right" in w) >= 0.5 * fps
                           for a, b in spans)
            what = "words" if words else "a picture or graphic"
            j = int(np.argmax(c.outside))
            out.append(Finding("safe zone", "fail" if long_bad else "warn",
                               f"{what} added outside the safe zone at {_say_spans(secs)} ({said})"
                               + ("" if words else ": the app's buttons and caption may cover part of it"),
                               "move it inside: x 120-960 on a 1080-wide frame, above the bottom fifth, below the top "
                               "tenth", spans=secs, at=round(j / fps, 2)))
        else:
            rep = float(c.replaced.mean())
            extra = f" ({rep:.0%} of frames are a whole new picture and were left out)" if rep > 0.05 else ""
            out.append(Finding("safe zone", "pass", "nothing is added outside it" + extra))
    return out


def face_alone(f: Facts) -> tuple[list[Finding], Optional[facemod.FaceTrack]]:
    """Without the source: the face is found in the video itself, and a face that vanishes for a
    moment between frames where it is plain may be covered."""
    raw = facemod.detect(f.info)
    trusted = facemod.trust(raw, f.info.rate)
    track = facemod.FaceTrack(f.info.width, f.info.height, f.info.rate, raw,
                              facemod._fill_and_smooth(trusted, f.info.rate), trusted=trusted)
    fps = f.info.rate
    if track.found_share < 0.3:
        return [Finding("face", "skip", "no face in most of the video, so there is no face to keep clear "
                        "(give source= to check against the clip)")], track
    lost = ~track.found
    inner = []
    for a, b in _spans(lost, fps, gap=1, least=max(3, int(0.15 * fps))):
        before = track.found[max(0, a - int(0.5 * fps)):a].mean() if a > 0 else 0
        after = track.found[b:b + int(0.5 * fps)].mean() if b < len(lost) else 0
        if before > 0.8 and after > 0.8:
            inner.append((a, b))
    if inner:
        secs = _secs(inner, fps)
        return [Finding("face", "warn", f"the face disappears for a moment at {_say_spans(secs)}: something may cover "
                        f"it (give source= to measure it against the clip)",
                        "look at those moments", spans=secs, at=secs[0][0])], track
    return [Finding("face", "pass", f"the face is found in {track.found_share:.0%} of frames and never vanishes "
                    f"between them (give source= to measure words over it against the clip)")], track


# ---- the report and the sheet ----------------------------------------------------------------------

def verdict(findings: list[Finding], strict: bool) -> tuple[str, list[Finding], list[Finding]]:
    fails = [x for x in findings if x.level == "fail"]
    warns = [x for x in findings if x.level == "warn"]
    if fails or (strict and warns):
        return "FAIL", fails + (warns if strict else []), [] if strict else warns
    return "PASS", [], warns


def write_report(path: Path, f: Facts, src: Optional[Facts], platform: dict, findings: list[Finding],
                 strict: bool, sheet: Path, took: float) -> str:
    word, fix, know = verdict(findings, strict)
    name = Path(f.path).name
    lines = [f"# Checks before done: {name}", ""]
    if word == "PASS":
        head = f"**PASS**: ready to post on {platform['name']}."
        if know:
            head += f" {len(know)} thing{'s' if len(know) != 1 else ''} worth knowing, below."
    else:
        head = f"**FAIL**: {len(fix)} thing{'s' if len(fix) != 1 else ''} to fix before posting on {platform['name']}."
    lines += [head, ""]
    if fix:
        lines += ["## Fix first", ""]
        for k, x in enumerate(fix, 1):
            lines.append(f"{k}. **{x.check.capitalize()}**: {x.what}." + (f" Fix: {x.fix}." if x.fix else ""))
        lines.append("")
    if know:
        lines += ["## Worth knowing", ""]
        for x in know:
            lines.append(f"- **{x.check.capitalize()}**: {x.what}." + (f" Fix: {x.fix}." if x.fix else ""))
        lines.append("")
    lines += ["## Every check", "", "| Check | Result | What was measured |", "|---|---|---|"]
    for x in sorted(findings, key=lambda y: LEVEL_ORDER[y.level]):
        lines.append(f"| {x.check} | {x.level.upper()} | {x.what.replace('|', '/')} |")
    lines += ["", f"The contact sheet `{sheet.name}` shows the frames that matter, with the safe zone (dashed) and "
              f"the face box (yellow) drawn; anything in the video over the face that is not in the source is "
              f"tinted red.", ""]
    src_line = f" against its source `{Path(src.path).name}`" if src else " (no source given: the face and the " \
                                                                           "safe zone were not measured against it)"
    lines.append(f"Checked{src_line} in {took:.1f} s. The video was not changed.")
    lines += ["", f"The video checked: {fingerprint(f.path)}."]
    path.write_text("\n".join(lines) + "\n")
    return word


def draw_sheet(path: Path, f: Facts, findings: list[Finding], comp: Optional[Compare], drawn: dict,
               word: str, zone_on: bool, render_face: Optional[facemod.FaceTrack]) -> Path:
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    from ..core.paths import FONTS

    info = f.info
    fps, n, dur = info.rate, info.frames, info.duration
    font = ImageFont.truetype(str(FONTS / "Poppins" / "Poppins-SemiBold.ttf"), 22)
    bold = ImageFont.truetype(str(FONTS / "Poppins" / "Poppins-Bold.ttf"), 30)
    small = ImageFont.truetype(str(FONTS / "Poppins" / "Poppins-SemiBold.ttf"), 16)
    # The moments: each problem's worst frame first, then even fill.
    moments: list[tuple[float, str, str]] = []
    for x in sorted(findings, key=lambda y: LEVEL_ORDER[y.level]):
        if x.level in ("fail", "warn") and x.at is not None and len(moments) < 5:
            if all(abs(x.at - m[0]) > 0.15 for m in moments):
                moments.append((float(x.at), x.check, x.level))
    k = 0
    while len(moments) < 6 and k < 12:
        t = dur * (k + 0.5) / 6 if k < 6 else dur * (k - 5.75) / 6
        if all(abs(t - m[0]) > 0.5 for m in moments):
            moments.append((t, "", "pass"))
        k += 1
    moments.sort()
    tile_h = 520
    tiles = []
    colours = {"fail": (255, 90, 90), "warn": (255, 196, 70), "pass": (200, 200, 205)}
    for t, what, level in moments[:6]:
        j = info.frame_at(t)
        fr = video.read_frame(info, j)
        scale = tile_h / fr.shape[0]
        tw = max(1, int(round(fr.shape[1] * scale)))
        img = cv2.resize(fr, (tw, tile_h), interpolation=cv2.INTER_AREA)
        if comp is not None and j in comp.maps:
            m = cv2.resize(comp.maps[j].astype(np.float32), (tw, tile_h), interpolation=cv2.INTER_NEAREST) > 0.5
            img = img.copy()
            img[m] = (0.45 * img[m] + 0.55 * np.array([255, 40, 40])).astype(np.uint8)
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil, "RGBA")
        if zone_on:
            z = SafeZone(tw, tile_h).rect
            _dashed(d, [z.x0, z.y0, z.x1, z.y1], (255, 255, 255, 200))
        box = None
        if comp is not None and comp.boxes is not None and comp.boxes[j] is not None:
            b = comp.boxes[j]
            ax = tw / comp_size(comp, f)[0]
            box = [b.x0 * ax, b.y0 * ax, b.x1 * ax, b.y1 * ax]
        elif render_face is not None and render_face.box(j) is not None and render_face.found[j]:
            b = render_face.box(j)
            box = [b.x0 * scale, b.y0 * scale, b.x1 * scale, b.y1 * scale]
        if box:
            d.rectangle(box, outline=(255, 210, 60, 255), width=3)
        d.rectangle([0, tile_h - 34, tw, tile_h], fill=(0, 0, 0, 180))
        label = f"{t:.2f} s" + (f"  {what}" if what else "")
        d.text((8, tile_h - 31), label, fill=colours.get(level, (220, 220, 220)), font=small)
        tiles.append(pil)
    width = max(1700, sum(t.width for t in tiles) + 10 * (len(tiles) - 1) + 40)
    lane_h, lanes = 46, ["picture", "face", "safe zone", "sound"]
    height = 70 + 40 + tile_h + 40 + len(lanes) * (lane_h + 8) + 60
    sheet = Image.new("RGB", (width, height), (18, 18, 20))
    d = ImageDraw.Draw(sheet)
    fails = sum(1 for x in findings if x.level == "fail")
    warns = sum(1 for x in findings if x.level == "warn")
    d.text((20, 14), f"{word}", fill=(255, 90, 90) if word == "FAIL" else (120, 230, 160), font=bold)
    d.text((120, 20), f"{Path(f.path).name}  ·  {info.width}x{info.height}, {fps:.3g} fps, {dur:.2f} s  ·  "
           f"{fails} to fix, {warns} worth knowing", fill=(230, 230, 236), font=font)
    if "lufs" in drawn:
        d.text((20, 60), f"{drawn['lufs']:.1f} LUFS, true peak {drawn['tp']:+.1f} dBTP", fill=(160, 160, 170), font=small)
    x = 20
    y0 = 100
    for tile in tiles:
        sheet.paste(tile, (x, y0))
        x += tile.width + 10
    top = y0 + tile_h + 30
    x0, x1 = 140, width - 20

    def tx(t: float) -> float:
        return x0 + (x1 - x0) * min(max(t, 0.0), dur) / max(dur, 1e-3)

    by = {x_.check: x_ for x_ in findings}
    lane_spans = {
        "picture": [(s, (90, 90, 100)) for k_ in ("black frames", "frozen frames") if k_ in by for s in by[k_].spans],
        "face": [(s, (255, 90, 90)) for s in (by["face"].spans if "face" in by else [])],
        "safe zone": [(s, (255, 170, 60)) for s in (by["safe zone"].spans if "safe zone" in by else [])],
    }
    for k_, name in enumerate(lanes):
        yy = top + k_ * (lane_h + 8)
        d.rectangle([x0, yy, x1, yy + lane_h], fill=(30, 30, 34))
        d.text((20, yy + 12), name, fill=(190, 190, 198), font=small)
        if name == "sound" and "envelope" in drawn:
            env = drawn["envelope"]
            db = env.db
            cols = int(x1 - x0)
            for cx in range(cols):
                a = int(cx / cols * len(db))
                b_ = max(a + 1, int((cx + 1) / cols * len(db)))
                v = float(np.max(db[a:b_])) if b_ <= len(db) else -100
                h = np.clip((v - env.floor) / max(1.0, env.speech - env.floor + 10), 0, 1) * (lane_h - 4)
                d.line([x0 + cx, yy + lane_h - 2, x0 + cx, yy + lane_h - 2 - h], fill=(110, 170, 255))
            for a, b_ in drawn.get("dead", []):
                d.rectangle([tx(a), yy, tx(b_), yy + lane_h], outline=(255, 196, 70), width=2)
            for t in drawn.get("clipped", [])[:100]:
                d.line([tx(t), yy, tx(t), yy + lane_h], fill=(255, 90, 90), width=1)
        for (a, b_), col in lane_spans.get(name, []):
            d.rectangle([tx(a), yy + 4, max(tx(b_), tx(a) + 3), yy + lane_h - 4], fill=col)
    for t, _, _ in moments[:6]:
        xx = tx(t)
        d.polygon([(xx - 7, top - 14), (xx + 7, top - 14), (xx, top - 2)], fill=(230, 230, 230))
    yy = top + len(lanes) * (lane_h + 8) + 6
    sec, stepped = 0.0, (1.0 if dur <= 20 else 5.0 if dur <= 90 else 15.0)
    while sec <= dur + 1e-6:
        d.text((tx(sec) - 6, yy), f"{sec:g}s", fill=(150, 150, 158), font=small)
        sec += stepped
    path = Path(path)
    sheet.save(path, quality=86)
    return path


def comp_size(c: Compare, f: Facts) -> tuple[int, int]:
    k = ANALYSIS / max(f.info.width, f.info.height)
    return max(2, int(round(f.info.width * k / 2)) * 2), max(2, int(round(f.info.height * k / 2)) * 2)


def _dashed(d, box, colour, dash: int = 14) -> None:
    x0, y0, x1, y1 = box
    for a, b, c_, e in ((x0, y0, x1, y0), (x0, y1, x1, y1), (x0, y0, x0, y1), (x1, y0, x1, y1)):
        length = max(abs(c_ - a), abs(e - b))
        steps = int(length // dash)
        for k in range(0, steps, 2):
            t0, t1 = k / max(1, steps), min(1.0, (k + 1) / max(1, steps))
            d.line([a + (c_ - a) * t0, b + (e - b) * t0, a + (c_ - a) * t1, b + (e - b) * t1], fill=colour, width=2)


# ---- the run -------------------------------------------------------------------------------------

def check(video_path: str | Path, source: Optional[str] = None, platform: str = "reels", strict: bool = False,
          out: Optional[Path] = None) -> dict:
    """Every check on `video_path`; writes the report and the sheet and returns what was found."""
    started = time.time()
    plat = PLATFORMS[platform]
    f = Facts(str(video_path))
    src = None
    # The same source whoever asks: given, or else the clip the plan beside the video names (as
    # review-notes takes it), so the command line and review-notes judge the same file alike.
    source = source or planned_source(video_path)
    if source:
        sp = Path(source).expanduser()
        if not sp.exists():
            raise video.MediaError(f"there is no source at {sp}")
        src = Facts(str(sp.resolve()))
    findings = file_checks(f, plat, src)
    zone_on = plat["zone"] and f.info.height > f.info.width * 1.5
    # Line the video up with its source in time, by the sound.
    tmap = None
    quiet_src = None
    comp = None
    render_face = None
    sclip = None
    if src is not None:
        from ..core.clip import Clip

        sclip = Clip(src.path)
        if f.a is not None and src.a is not None:
            env_v = audio.envelope(video.mono(video.read_audio(f.info), video.AUDIO_RATE, 16000), 16000)
            env_s = sclip.envelope
            tmap = time_map(f, src, env_v, env_s, _planned_holds(f.info.path))
            quiet_src = audio.quiet_runs(env_s)
        elif abs(f.info.duration - src.info.duration) < 0.2:
            t = np.arange(f.info.frames) / f.info.rate
            tmap = TimeMap(np.minimum(t, src.info.duration - 1 / src.info.rate), "the same frames (no sound to "
                           "line up by; the lengths match)", [], src.info.frames, src.info.rate)
    pic = Picture(f)
    still = np.zeros(f.info.frames, bool)
    for a, b in _spans(pic.diff[:f.info.frames] < FROZEN, f.info.rate, gap=0, least=max(2, int(round(0.25 * f.info.rate)))):
        still[max(0, a - 1):b] = True
    sound, drawn = sound_checks(f, plat, src, tmap, quiet_src, still)
    findings += sound
    voiced = None
    if "envelope" in drawn:
        env = drawn["envelope"]
        voiced = np.ones(f.info.frames, bool)
        for a, b in audio.quiet_runs(env):
            voiced[int(a * f.info.rate):int(np.ceil(b * f.info.rate))] = False
    if tmap is not None and src is not None:
        tmap.source = src
        if tmap.cuts:
            tmap.snap(pic.diff, f.info.rate)
    findings += picture_checks(f, pic, src, tmap, voiced)
    geo, score = (None, 0.0)
    if src is not None and tmap is not None:
        geo, score = geometry(f, src, tmap, pic.size)
    from ..core import hdr

    aligned = (_aligned_samples(f, src, tmap, geo, pic.size)
               if geo is not None and src is not None and hdr.kind(src.transfer) != "sdr" else None)
    findings += colour_checks(f, pic, src, aligned)
    if src is not None and tmap is not None:
        size = pic.size
        if geo is None:
            findings.append(Finding("face", "skip", f"the video does not line up with the source (the best fit scored "
                                    f"{score:.2f}): a layout or a new picture, so the face was not measured"))
            if zone_on:
                findings.append(Finding("safe zone", "skip", "not measured: the video does not line up with the source"))
        else:
            comp = compare(f, src, sclip.face, tmap, geo, size, zone_on)
            findings += face_and_zone_checks(comp, f.info.rate, zone_on, f.info.frames)
            if tmap.kind != "the same frames":
                findings.append(Finding("lined up", "info", f"in time: {tmap.kind}"
                                        + (f" ({len(tmap.cuts)} cuts; the frames either side of each were not judged)"
                                           if tmap.cuts else "") + f"; in space: {geo.kind}"))
    elif src is not None:
        findings.append(Finding("face", "skip", "the video could not be lined up with the source by its sound, so the "
                                "face and the safe zone were not measured"))
    else:
        found, render_face = face_alone(f)
        findings += found
        if zone_on:
            findings.append(Finding("safe zone", "skip", "not measured: give source= (the clip it was made from) and "
                                    "anything added outside the safe zone is found"))
    vid = Path(f.path)
    if out is None:
        report_path = vid.with_name(vid.stem + ".qc.md")
    else:
        out = Path(out)
        report_path = out if out.suffix == ".md" else out.with_name(out.stem + ".qc.md")
    sheet_path = report_path.with_name(report_path.name[:-len(".qc.md")] + ".qc.jpg") \
        if report_path.name.endswith(".qc.md") else report_path.with_suffix(".jpg")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    word, _, _ = verdict(findings, strict)
    sheet = draw_sheet(sheet_path, f, findings, comp, drawn, word, zone_on, render_face)
    took = time.time() - started
    word = write_report(report_path, f, src, plat, findings, strict, sheet, took)
    return {"verdict": word, "findings": findings, "report": report_path, "sheet": sheet, "seconds": took}


# ---- the same file is checked once ----------------------------------------------------------------

CACHE_VERSION = 1


def _code_key() -> str:
    """What the checks are made of: this file and the core it reads with. A changed pack checks again."""
    import hashlib

    h = hashlib.sha256()
    for f in [Path(__file__)] + sorted((Path(__file__).resolve().parents[1] / "core").glob("*.py")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _cache_key(video_path, source: Optional[str], platform: str, strict: bool) -> str:
    """The video's bytes, the plan beside it (it names the holds), the source (its path, size and
    time), the platform, strict, and the code."""
    import hashlib

    v = Path(str(video_path))
    plan = v.with_suffix(".plan.json")
    src = source or planned_source(v)
    parts = [str(CACHE_VERSION), fingerprint(v), platform, str(bool(strict)), _code_key(),
             hashlib.sha256(plan.read_bytes()).hexdigest()[:16] if plan.exists() else "no plan"]
    if src:
        sp = Path(src).expanduser().resolve()
        st = sp.stat()
        parts += [str(sp), str(st.st_size), str(st.st_mtime_ns)]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def check_once(video_path: str | Path, source: Optional[str] = None, platform: str = "reels", strict: bool = False,
               out: Optional[Path] = None) -> dict:
    """`check`, remembered by the video's bytes: a file checked before, unchanged, with the same source,
    plan, settings and pack, gives back the findings it had (and its report and sheet, which must still
    be there) without being read again. The memory is kept beside the video (`<video>.qc-cache.json`).
    The result says `cached`."""
    from ..core import paths

    store = paths.beside(video_path, "qc-cache.json")
    key = _cache_key(video_path, source, platform, strict)
    try:
        saved = json.loads(store.read_text())
    except (OSError, ValueError):
        saved = {}
    hit = saved.get(key) if isinstance(saved, dict) else None
    want = str(Path(out)) if out is not None else None
    if hit and hit.get("out") == want and Path(hit["report"]).exists() and Path(hit["sheet"]).exists():
        findings = [Finding(d["check"], d["level"], d["what"], d.get("fix", ""),
                            [tuple(x) for x in d.get("spans", [])], d.get("at")) for d in hit["findings"]]
        return {"verdict": hit["verdict"], "findings": findings, "report": Path(hit["report"]),
                "sheet": Path(hit["sheet"]), "seconds": hit["seconds"], "cached": True}
    r = check(video_path, source, platform, strict, out)
    entry = {"verdict": r["verdict"], "report": str(r["report"]), "sheet": str(r["sheet"]), "seconds": r["seconds"],
             "out": want, "findings": [{"check": x.check, "level": x.level, "what": x.what, "fix": x.fix,
                                        "spans": [list(map(float, sp)) for sp in x.spans],
                                        "at": None if x.at is None else float(x.at)} for x in r["findings"]]}
    saved = saved if isinstance(saved, dict) else {}
    saved[key] = entry
    try:
        part = store.with_name(f"{store.name}.{__import__('os').getpid()}.part")
        part.write_text(json.dumps(saved, indent=1))
        part.replace(store)
    except OSError:
        pass  # a folder that cannot be written only means checking again next time
    return {**r, "cached": False}


def _source_motion(src: Facts) -> np.ndarray:
    """How much each frame of the source changes from the one before (on a small grey copy)."""
    import cv2

    info = src.info
    out = np.full(info.frames, np.nan)
    prev = None
    w = 64
    h = max(2, int(w * info.height / info.width))
    for j, fr in enumerate(video.FrameReader(info, size=(w * 2, h * 2))):
        if j >= info.frames:
            break
        g = cv2.resize(_gray(fr), (w, h), interpolation=cv2.INTER_AREA)
        if prev is not None:
            out[j] = float(np.abs(g - prev).mean())
        prev = g
    return out


def report(video_path, opts: dict, out: Optional[Path], still=None) -> int:
    """What `python -m fx.run qc-check <video>` runs. Prints a line per check, then where the report
    and the sheet are; returns 1 when the verdict is FAIL."""
    r = check(video_path, opts.get("source") or None, opts["platform"], opts["strict"], out)
    word = r["verdict"]
    fails = [x for x in r["findings"] if x.level == "fail"]
    warns = [x for x in r["findings"] if x.level == "warn"]
    print(f"effect: qc-check · {Path(str(video_path)).name} · {opts['platform']}")
    print(f"check: verdict: {word} ({len(fails)} to fix, {len(warns)} worth knowing)")
    for x in sorted(r["findings"], key=lambda y: LEVEL_ORDER[y.level]):
        print(f"check: {x.check}: {x.level.upper()} {x.what}")
    print(f"report: {r['report']}")
    print(f"sheet: {r['sheet']}")
    print(f"checked in {r['seconds']:.1f} s; the video was not changed.")
    print("To fix what it found without rendering again, open the edit in AskEditor: https://askeditor.com, then ask "
          "\"Check my edit before I post it\".")
    return 1 if word == "FAIL" else 0
