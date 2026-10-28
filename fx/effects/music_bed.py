"""music-bed: your own music under your voice, ducked while you talk, faded at the ends, and fitted
to the film so it ends when the film does.

How it decides:

- **Where you talk** comes from the core: the word times, checked against the quiet stretches in
  the sound (`sound.speech_spans`). Phrases less than `JOIN` seconds apart are one stretch, so the
  music never bobs up in the breath between two sentences; it comes up only in a real pause.
- **The duck is drawn, not listened for.** A compressor keyed by the voice rides every syllable, and
  the music pumps. Here the gain is laid from those stretches: it starts down `LEAD` seconds before
  the first word, holds one level for the whole phrase, waits `HOLD` after the last word, and comes
  back up over `RISE`. It moves only between phrases. A pause too near the end to come up and go
  down again stays down.
- **The level is set against your voice,** measured: `level` dB under its loudness where you are
  not talking, and `duck` dB lower again where you are.
- **The fit.** The track's beats and bars are found here: an onset envelope, the tempo from its
  autocorrelation (with a gentle preference for a walking pace, so eighth notes are not taken for
  beats), the beats by dynamic programming, and the bar lines where the low end hits hardest. Then
  `trim` plays the track from its start and fades it out from a bar line so the fade ends on the
  film's last frame; `loop` jumps back a whole number of bars, from where the music before the jump
  is most like the music before where it lands, as many times as the film needs, then fades the same
  way; `end-aligned` makes the track's own ending land on the film's last frame, by starting it later
  or by taking whole bars out of the middle, or repeating some. Every join is a short crossfade
  that finishes just before a beat, the two beats lined up on their attacks to the sample. `auto`
  (the default) is `end-aligned` when the track ends by itself (it fades or rings out), else `trim`
  when it is long enough, else `loop`.
- **Loudness.** The finished mix is set to `target_lufs` (-14 by default, what the short-video apps
  play at) with ffmpeg's two-pass loudnorm, its true peak under -1 dBTP (`core/loudness.py`).

After the render the file itself is measured (loudness, true peak), heard again by the transcriber
to show the words survive the music, and drawn as a sound sheet: six frames over the voice and the
music, the duck's gain drawn on the music, and close-ups of every join and of the last frame.
"""

from __future__ import annotations

import difflib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..core import audio, loudness, sound, soundsheet, video
from ..core import framing as framing_mod
from ..core.timeline import Timeline
from . import options as O

NAME = "music-bed"
RATE = video.AUDIO_RATE
LAB_BED = "{lab}/../home-asks/music/bed.mp3"


class Fades:
    """`in,out` seconds (one number sets both)."""

    help = "seconds the music fades in at the start and out at the end: 0.5,1.5 (one number sets both)"

    def parse(self, raw) -> str:
        parts = [p.strip().rstrip("s") for p in str(raw).replace(";", ",").split(",") if p.strip()]
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            raise O.OptionError(f"give seconds, like 0.5,1.5 or 2 (got {raw!r})") from None
        if len(vals) == 1:
            vals = vals * 2
        if len(vals) != 2 or not all(0 <= v <= 10 for v in vals):
            raise O.OptionError(f"give one or two numbers of seconds from 0 to 10, like 0.5,1.5 (got {raw!r})")
        return f"{vals[0]:g},{vals[1]:g}"


class Target:
    help = "how loud the finished video is, in LUFS: -14 suits the short-video apps; off leaves the level as it is"

    def parse(self, raw) -> str:
        v = str(raw).strip().lower()
        if v in ("off", "keep", "none"):
            return "off"
        try:
            f = float(v)
        except ValueError:
            raise O.OptionError(f"give a number from -30 to -6, or off (got {raw!r})") from None
        if not -30 <= f <= -6:
            raise O.OptionError(f"give a number from -30 to -6, or off (got {raw!r})")
        return f"{f:g}"


FITS = ("auto", "trim", "loop", "end-aligned")
OPTIONS = {
    "music": O.Text("your track: the path to an mp3, wav, m4a or any file with sound (the pack ships none)"),
    "level": O.Number(-30, 0, "how far under your voice the music sits when you are not talking, in dB"),
    "duck": O.Number(0, 30, "how many dB lower again the music goes while you talk (0 keeps it level)"),
    "fade": Fades(),
    "fit": O.Choice(FITS, "how the track is made the film's length: trim (fade out on a bar), loop (repeat whole "
                          "bars), end-aligned (its own ending on the last frame), or auto"),
    "target_lufs": Target(),
}

# The duck's timing (seconds), and the shortest pause the music comes up in.
LEAD, HOLD, RISE = 0.25, 0.15, 0.5
JOIN = 1.2  # time enough to come up, sit for a moment and go down again before the next phrase
END_STAY = HOLD + RISE + 0.45  # after the last phrase, less than this to the end: stay down
START_STAY = 1.0  # talk starting sooner than this: the music starts down (no swell to dip at once)
MAX_LATE = 8.0  # end-aligned: start the track up to this late (or half the film) rather than cut it
XF_BEAT, XF_FREE = 0.06, 0.6  # crossfade at a join: on a beat, and where no steady beat was found
HOP = 0.01

# Every test of this effect plays AskEditor's own generated bed; no music from anywhere else.
MATRIX = [
    ("A", "A", {"music": LAB_BED}),
    ("B", "B", {"music": LAB_BED}),
    ("C", "C", {"music": LAB_BED}),
    ("D", "D", {"music": LAB_BED}),
    ("E-cut", "E-cut", {"music": LAB_BED}),
    ("D-loop", "D", {"music": LAB_BED, "fit": "loop"}),
    ("E-cut-trim", "E-cut", {"music": LAB_BED, "fit": "trim"}),
]


# ---- the track: its beat, bars and ending ------------------------------------------------------------

@dataclass
class Grid:
    bpm: float
    beats: np.ndarray  # seconds
    first_bar: int  # the index in `beats` of the first downbeat
    sure: float  # how periodic the onsets are, 0 to 1

    @property
    def bar(self) -> float:
        return 240.0 / self.bpm

    @property
    def bars(self) -> np.ndarray:
        return self.beats[self.first_bar::4]

    @property
    def steady(self) -> bool:
        return self.sure >= 0.3


@dataclass
class Track:
    path: str
    samples: np.ndarray
    first: float  # where its sound begins
    end: float  # where its sound ends (its own ending, faded or rung out, included)
    ends_itself: bool  # it fades or rings out, rather than stopping mid-bar
    body_end: float  # where its ending begins: loops jump back from before this
    grid: Grid
    feats: np.ndarray  # a description of the sound every FEAT_HOP seconds, for finding like moments
    loud: float  # its loudness, LUFS
    attacks: dict = field(default_factory=dict)  # beat time -> its attack, found once

    @property
    def seconds(self) -> float:
        return self.end - self.first


FEAT_HOP = 0.05


def _spectra(mono: np.ndarray, n_fft: int = 2048, hop: int = 480):
    """Band energies (40 bands, 30 Hz to 16 kHz) and pitch classes (12), every `hop` samples."""
    frames = 1 + max(0, (len(mono) - 1) // hop)
    pad = np.pad(mono.astype(np.float32), (n_fft // 2, n_fft // 2 + hop))
    window = np.hanning(n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / RATE)
    edges = np.geomspace(30.0, 16000.0, 41)
    band_of = np.searchsorted(edges, freqs) - 1
    pitch = np.full(len(freqs), -1)
    tonal = (freqs >= 65) & (freqs <= 2100)
    pitch[tonal] = np.round(12 * np.log2(freqs[tonal] / 440.0) + 69).astype(int) % 12
    view = np.lib.stride_tricks.sliding_window_view(pad, n_fft)[::hop][:frames]
    bands = np.zeros((frames, 40), np.float32)
    chroma = np.zeros((frames, 12), np.float32)
    for a in range(0, frames, 1500):
        power = np.abs(np.fft.rfft(view[a:a + 1500] * window, axis=1)) ** 2
        for k in range(40):
            sel = band_of == k
            if sel.any():
                bands[a:a + len(power), k] = power[:, sel].sum(axis=1)
        for k in range(12):
            chroma[a:a + len(power), k] = power[:, pitch == k].sum(axis=1)
    return bands, chroma, edges


def _tempo(onset: np.ndarray) -> tuple[float, float]:
    """(seconds per beat, how sure: the normalised autocorrelation there)."""
    o = onset - onset.mean()
    n = len(o)
    if n < 300:
        return 0.6, 0.0
    spec = np.fft.rfft(o, 2 * n)
    ac = np.fft.irfft(np.abs(spec) ** 2)[:n]
    if ac[0] <= 0:
        return 0.6, 0.0
    ac = ac / ac[0]
    lags = np.arange(int(0.3 / HOP), min(n - 1, int(1.5 / HOP)))  # 200 to 40 beats a minute
    bpm = 60.0 / (lags * HOP)
    prior = np.exp(-0.5 * (np.log2(bpm / 100.0) / 0.8) ** 2)
    score = np.maximum(ac[lags], 0) * prior
    k = int(np.argmax(score))
    lag = float(lags[k])
    if 0 < k < len(lags) - 1:  # between the steps
        y0, y1, y2 = ac[lags[k] - 1], ac[lags[k]], ac[lags[k] + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            lag += 0.5 * (y0 - y2) / den
    return lag * HOP, float(max(0.0, ac[lags[k]]))


def _beats(onset: np.ndarray, period: float, tight: float = 100.0) -> np.ndarray:
    """Beat steps through the onset envelope, each about a period after the last (dynamic programming:
    the beats as a whole land on the strongest onsets without the spacing wandering)."""
    p = period / HOP
    o = onset / (onset.std() + 1e-9)
    n = len(o)
    score = o.copy()
    back = np.full(n, -1)
    lo, hi = max(1, int(round(p / 2))), int(round(2 * p))
    for t in range(lo, n):
        a, b = max(0, t - hi), t - lo + 1
        prev = np.arange(a, b)
        cand = score[prev] - tight * np.log((t - prev) / p) ** 2
        k = int(np.argmax(cand))
        if cand[k] > 0:
            score[t] = o[t] + cand[k]
            back[t] = prev[k]
    tail = max(1, int(p))
    t = n - tail + int(np.argmax(score[n - tail:]))
    out = []
    while t >= 0:
        out.append(t)
        t = back[t]
    return np.array(out[::-1], float) * HOP


def analyse(path: str) -> Track:
    x = sound.read(path)
    if len(x) < RATE:
        raise ValueError(f"the music is under a second long: {path}")
    env = sound.envelope_db(x, RATE, HOP)
    heard = env[env > -90]
    if not len(heard):
        raise ValueError(f"the music is silent: {path}")
    body = float(np.percentile(heard, 70))
    audible = np.flatnonzero(env > body - 40)
    first = max(0.0, audible[0] * HOP - 0.01)
    end = min(len(x) / RATE, (audible[-1] + 1) * HOP)
    # Its own ending: the last part well under the body's level, after it was at that level. The
    # ending begins where the level last falls away from the body for good.
    # A third of a second's average; the ends padded with their own level (a zero in dB is loud).
    smooth = np.convolve(np.pad(env, (15, 14), mode="edge"), np.ones(30) / 30, mode="valid")
    last_full = np.flatnonzero(smooth[: int(end / HOP)] > body - 8)
    body_end = (last_full[-1] * HOP) if len(last_full) else end
    tail = env[max(0, int((end - 0.35) / HOP)):int(end / HOP)]
    ends_itself = bool(len(tail) and float(np.mean(tail)) < body - 12 and float(end - body_end) >= 0.4)
    if not ends_itself:
        body_end = end

    mono = x.mean(axis=1)
    bands, chroma, edges = _spectra(mono)
    log_bands = np.log10(1e-9 + bands)
    flux = np.maximum(0.0, np.diff(log_bands, axis=0, prepend=log_bands[:1])).sum(axis=1)
    low = np.maximum(0.0, np.diff(log_bands[:, edges[1:] <= 200], axis=0,
                                  prepend=log_bands[:1, edges[1:] <= 200])).sum(axis=1)
    onset = np.maximum(0.0, flux - np.convolve(flux, np.ones(50) / 50, mode="same"))
    a, b = int(first / HOP), int(end / HOP)
    period, sure = _tempo(onset[a:b])
    beats = _beats(onset[a:b], period) + a * HOP if sure > 0 else np.arange(first, end, period)
    beats = beats[(beats >= first - 1e-6) & (beats <= end)]
    # The downbeat: of the four places a bar could start, the one where the low end hits hardest and
    # the harmony changes most.
    chroma_n = chroma / (chroma.sum(axis=1, keepdims=True) + 1e-12)
    change = np.concatenate([[0.0], np.abs(np.diff(chroma_n, axis=0)).sum(axis=1)])
    best, first_bar = -1.0, 0
    for k in range(min(4, len(beats))):
        idx = np.clip((beats[k::4] / HOP).astype(int), 0, len(low) - 1)
        s = float(np.mean(low[idx] / (low.std() + 1e-9)) + 0.5 * np.mean(change[idx] / (change.std() + 1e-9)))
        if s > best:
            best, first_bar = s, k
    grid = Grid(bpm=60.0 / period, beats=beats, first_bar=first_bar, sure=sure)

    # What the music is like every FEAT_HOP: its spectrum's shape and its harmony, for comparing two
    # moments. Each band's own average is taken off so the comparison is about the music changing.
    step = int(round(FEAT_HOP / HOP))
    n = len(log_bands) // step
    lb = log_bands[: n * step].reshape(n, step, -1).mean(axis=1)
    lb = lb - lb.mean(axis=0)
    ch = chroma_n[: n * step].reshape(n, step, -1).mean(axis=1)
    feats = np.concatenate([lb, 3.0 * (ch - ch.mean(axis=0))], axis=1).astype(np.float32)
    loud = sound.integrated(sound.blocks(x[int(first * RATE):int(end * RATE)]))
    return Track(path=str(path), samples=x, first=first, end=end, ends_itself=ends_itself, body_end=body_end,
                 grid=grid, feats=feats, loud=loud)


def _alike(track: Track, a: float, b: float, span: float, after: bool = True) -> float:
    """How alike the music is just before `a` and just before `b` (and just after each, when `after`):
    0 to 1. A jump from `a` to `b` is seamless when they are."""
    f = track.feats

    def window(t0: float, t1: float) -> np.ndarray:
        i0, i1 = int(round(t0 / FEAT_HOP)), int(round(t1 / FEAT_HOP))
        if i0 < 0 or i1 > len(f) or i1 <= i0:
            return np.zeros(0)
        return f[i0:i1].ravel()

    sims = []
    pairs = [((a - span, a), (b - span, b))]
    if after:
        pairs.append(((a, a + span), (b, b + span)))
    for (x0, x1), (y0, y1) in pairs:
        u, v = window(x0, x1), window(y0, y1)
        if len(u) and len(u) == len(v):
            den = float(np.linalg.norm(u) * np.linalg.norm(v))
            sims.append(float(u @ v) / den if den > 0 else 0.0)
    return float(np.clip(np.mean(sims), 0.0, 1.0)) if sims else 0.0


def _shape(x: np.ndarray, t: float, w: float = 0.15) -> np.ndarray:
    """The spectrum's shape (24 log bands, 60 Hz to 12 kHz) over `w` seconds either side of `t`."""
    a, b = int(max(0.0, t - w) * RATE), int((t + w) * RATE)
    seg = x[a:b].mean(axis=1)
    if len(seg) < 64:
        return np.zeros(24)
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) + 1e-9
    f = np.fft.rfftfreq(len(seg), 1.0 / RATE)
    edges = np.geomspace(60.0, 12000.0, 25)
    return np.array([np.log10(spec[(f >= lo) & (f < hi)].mean()) for lo, hi in zip(edges[:-1], edges[1:])])


def _change(x: np.ndarray, t: float) -> float:
    """How much the sound's spectrum changes across `t`: the music just before against just after."""
    return float(np.linalg.norm(_shape(x, t - 0.2) - _shape(x, t + 0.2)))


def _natural_change(track: Track) -> float:
    """The change the music makes across its own bar lines (the median): a join that changes the sound
    no more than this cannot stand out by its spectrum."""
    bars = [float(b) for b in track.grid.bars if track.first + 0.5 < b < track.body_end - 0.5]
    return float(np.median([_change(track.samples, b) for b in bars])) if bars else 2.0


def _join_change(track: Track, a: float, b: float, xf: float) -> float:
    """The change across a join made from `a` to `b` (both already on their attacks), as heard."""
    ranges = [(int(round((a - 0.6) * RATE)), int(round(a * RATE))), (int(round(b * RATE)), int(round((b + 0.6) * RATE)))]
    if ranges[0][0] < 0 or ranges[1][1] > len(track.samples):
        return 99.0
    piece = audio.join(track.samples, RATE, ranges, fade=xf)
    return _change(piece, 0.6)


def _attack(track: Track, t: float, search: float = 0.03) -> float:
    """The moment a beat's attack begins near `t`: the sharpest rise of the high end's energy, to the
    millisecond. A beat whose attack cannot be told stays at `t`."""
    x = track.samples.mean(axis=1)
    a, b = int(max(0.0, t - search - 0.02) * RATE), int(min(len(x) / RATE, t + search + 0.02) * RATE)
    seg = np.diff(x[a:b], prepend=x[a:a + 1])  # a first difference: the attack's top end, not the bass
    n = int(0.001 * RATE)
    m = len(seg) // n
    if m < 10:
        return t
    e = np.sqrt((seg[: m * n].reshape(m, n) ** 2).mean(axis=1))
    rise = np.diff(np.log10(e + 1e-7))
    k = int(np.argmax(rise))
    if rise[k] < 0.15:
        return t
    return (a + (k + 1) * n) / RATE


# ---- fitting the track to the film -----------------------------------------------------------------

@dataclass
class Fit:
    mode: str
    segments: list[tuple[float, float]]  # track seconds, in the order they play
    crossfade: float
    fade_in: float
    fade_out: Optional[tuple[float, float]]  # output seconds the fade runs over, or None
    ending_at: Optional[float]  # end-aligned: the output time the track's own ending lands
    notes: list[str] = field(default_factory=list)

    @property
    def joins(self) -> list[float]:
        out, t = [], 0.0
        for a, b in self.segments[:-1]:
            t += b - a
            out.append(t)
        return out

    def to_track(self, t_out: float) -> float:
        t = 0.0
        for a, b in self.segments:
            if t_out <= t + (b - a) + 1e-9:
                return a + (t_out - t)
            t += b - a
        return self.segments[-1][1]

    def as_dict(self) -> dict:
        return {"mode": self.mode, "segments": [[round(a, 4), round(b, 4)] for a, b in self.segments],
                "joins": [round(j, 3) for j in self.joins], "crossfade": self.crossfade,
                "fade_in": self.fade_in, "fade_out": [round(v, 3) for v in self.fade_out] if self.fade_out else None,
                "ending_at": round(self.ending_at, 3) if self.ending_at is not None else None, "notes": self.notes}


def _bars_in(track: Track, lo: float, hi: float) -> list[float]:
    return [float(t) for t in track.grid.bars if lo - 1e-6 <= t <= hi + 1e-6] if track.grid.steady else \
        [float(t) for t in np.arange(math.ceil(lo / 0.25) * 0.25, hi + 1e-6, 0.25)]


def _cut(track: Track, a: float, b: float, xf: float) -> tuple[float, float]:
    """The exact pair of track times for a jump from beat `a` to beat `b`: their attacks lined up, the
    join a crossfade's half plus 5 ms before them, so each new beat's attack plays whole."""
    if not track.grid.steady:
        return a, b
    lead = xf / 2 + 0.005
    if a not in track.attacks:
        track.attacks[a] = _attack(track, a)
    ta = track.attacks[a]
    if (a, b) not in track.attacks:  # where the music at b lines up with the music at a's attack
        track.attacks[(a, b)] = _align(track, ta, b + (ta - a))
    return ta - lead, track.attacks[(a, b)] - lead


def _align(track: Track, ta: float, guess: float, reach: float = 0.045) -> float:
    """The moment near `guess` where the music lines up best with the music around `ta`: their
    1 ms energy envelopes (of the attacks' top end) correlated, 80 ms before to 150 ms after. A bar
    that opens with two hits close together lines up hit for hit, which picking each side's
    sharpest attack alone does not (it can take the first hit on one side, the second on the other)."""
    x = track.samples.mean(axis=1)
    n = int(0.001 * RATE)

    def env(t0: float, t1: float) -> np.ndarray:
        i0, i1 = int(round(t0 * RATE)), int(round(t1 * RATE))
        if i0 < 1 or i1 > len(x):
            return np.zeros(0)
        d = np.diff(x[i0 - 1:i1])
        m = len(d) // n
        return np.sqrt((d[: m * n].reshape(m, n) ** 2).mean(axis=1))

    ref = env(ta - 0.08, ta + 0.15)
    wide = env(guess - reach - 0.08, guess + reach + 0.15)
    if len(ref) < 50 or len(wide) < len(ref):
        return guess
    ref = ref - ref.mean()
    corr = np.correlate(wide - wide.mean(), ref, mode="valid")
    k = int(np.argmax(corr))
    return guess - reach + k * 0.001


def _fade_from_bar(track: Track, fit_segments, L: float, length: float) -> tuple[float, float]:
    """A fade-out that ends at `L` and begins on a bar line about `length` seconds before it, or on the
    nearest beat when no bar line is near enough to keep the fade close to the length asked."""
    if length <= 0:
        return (L, L)
    want = L - length
    if not track.grid.steady:
        return (max(0.0, want), L)

    def nearest(times, lo: float, hi: float) -> Optional[float]:
        best, t = None, 0.0
        for a, b in fit_segments:
            for x in times:
                if a - 1e-6 <= x <= b + 1e-6:
                    o = t + (x - a)
                    if lo <= o <= hi and (best is None or abs(o - want) < abs(best - want)):
                        best = o
            t += b - a
        return best

    start = nearest(track.grid.bars, L - 1.4 * length, L - 0.75 * length)
    if start is None:
        start = nearest(track.grid.beats, want - 0.4, want + 0.4)
    return (max(0.0, start if start is not None else want), L)


def fit_track(track: Track, L: float, mode: str, fades: tuple[float, float],
              talk: list[tuple[float, float]]) -> Fit:
    fin, fout = fades
    xf = XF_BEAT if track.grid.steady else XF_FREE
    notes: list[str] = []
    asked = mode
    if mode == "auto":
        mode = "end-aligned" if track.ends_itself else ("trim" if track.seconds >= L else "loop")
    if mode == "end-aligned":
        got = _end_aligned(track, L, xf, talk, notes)
        if got is not None:
            segs, ending_at = got
            return Fit("end-aligned", segs, xf, fin, None, ending_at, notes)
        notes.append("no way to land the track's own ending on the last frame was found; faded out instead")
        mode = "trim" if track.seconds >= L else "loop"
    if mode == "trim" and track.seconds < L:
        notes.append(f"the track ({track.seconds:.1f} s) is shorter than the film ({L:.1f} s), so it loops")
        mode = "loop"
    if mode == "trim":
        segs = [(track.first, track.first + L)]
    else:
        segs = _looped(track, L, xf, talk, notes)
        if asked == "loop" and len(segs) == 1:
            notes.append("the track is long enough to play once; nothing needed repeating")
    return Fit(mode, segs, xf, fin, _fade_from_bar(track, segs, L, fout), None, notes)


def _in_talk(t: float, talk) -> bool:
    return any(a <= t <= b for a, b in talk)


def _looped(track: Track, L: float, xf: float, talk, notes) -> list[tuple[float, float]]:
    p0 = track.first
    if track.end - p0 >= L:
        return [(p0, p0 + L)]
    bar = track.grid.bar if track.grid.steady else 2.0
    stop = track.body_end
    cands = _bars_in(track, p0 + 0.5 * bar, stop)
    natural = _natural_change(track)
    best = None
    for a in cands:
        for b in cands:
            if a - b < bar - 1e-6:
                continue
            alike = _alike(track, a, b, bar, after=a + bar <= stop)
            change = _join_change(track, *_cut(track, a, b, xf), xf)
            s = alike - 0.1 * max(0.0, change - natural)  # a join that changes the sound more than a bar line does
            s += 0.02 * (a - b) / bar  # a longer loop repeats less often
            if best is None or s > best[0]:
                best = (s, a, b, alike, change)
    change = None
    if best is None:  # too short to find two bar lines: loop the whole sound
        a, b = stop, p0
        score = 0.0
    else:
        _, a, b, score, change = best
    a, b = _cut(track, a, b, xf)
    segs = [(p0, a)]
    total = a - p0
    while total < L - 1e-9:
        take = min(a - b, L - total)
        segs.append((b, b + take))
        total += take
    jumps = len(segs) - 1
    notes.append(f"loops back {a - b:.2f} s ({(a - b) / bar:.0f} bars) {'once' if jumps == 1 else f'{jumps} times'}, "
                 f"where the music either side is {score:.0%} alike")
    if change is not None:
        notes.append(f"its sound changes {change:.1f} across the join, {natural:.1f} across the track's own bar lines")
    return segs


def _end_aligned(track: Track, L: float, xf: float, talk, notes):
    """Segments whose last ends on the track's own end, adding up to L; None when none fits."""
    p0, E, T = track.first, track.end, track.seconds
    late = T - L
    # Starting the track later needs no join at all, and its start is faded in (usually under the
    # voice), so it is the first choice unless it would skip most of the track.
    if 0 <= late <= max(MAX_LATE, 0.5 * L, 0.6 * T):
        notes.append(f"starts {late:.2f} s into the track so its ending lands on the last frame")
        return [(p0 + late, E)], L
    bar = track.grid.bar if track.grid.steady else 2.0
    cands = _bars_in(track, p0 + 0.5 * bar, E - 0.5 * bar)
    most_late = max(4 * bar, 0.5 * L)
    natural = _natural_change(track)
    options = []
    for a in cands:
        for b in cands:
            if abs(a - b) < bar - 1e-6:
                continue
            if b > a:  # take the bars between out
                skip = T - L - (b - a)
                k = 1
            else:  # play the bars between twice, or more
                if a > track.body_end + 1e-6:
                    continue
                k = max(1, math.ceil((L - T) / (a - b)))
                skip = T + k * (a - b) - L
            if not 0 <= skip <= most_late or a - (p0 + skip) < bar * 0.75:
                continue
            alike = _alike(track, a, b, bar, after=b > a or a + bar <= track.body_end)
            change = _join_change(track, *_cut(track, a, b, xf), xf)
            s = alike - 0.1 * max(0.0, change - natural) - 0.05 * k - 0.02 * skip / bar
            out_at = a - (p0 + skip)
            if _in_talk(out_at, talk):
                s += 0.05  # a join under the voice is heard least
            options.append((s, a, b, k, skip, alike, change))
    if not options:
        if late > 0:
            notes.append(f"starts {late:.2f} s into the track so its ending lands on the last frame")
            return [(p0 + late, E)], L
        return None
    s, a, b, k, skip, alike, change = max(options)
    a2, b2 = _cut(track, a, b, xf)
    start = p0 + skip
    if b > a:
        segs = [(start, a2), (b2, E)]
        notes.append(f"{b - a:.2f} s ({(b - a) / bar:.0f} bars) taken out of the middle, where the music either "
                     f"side is {alike:.0%} alike")
    else:
        segs = [(start, a2)] + [(b2, a2)] * (k - 1) + [(b2, E)]
        notes.append(f"jumps back {a - b:.2f} s ({(a - b) / bar:.0f} bars) {'once' if k == 1 else f'{k} times'}, "
                     f"where the music either side is {alike:.0%} alike")
    # The joins moved to the attacks: put the start where the lengths add up to L exactly.
    total = sum(e - s0 for s0, e in segs)
    first_a, first_b = segs[0]
    segs[0] = (first_a + (total - L), first_b)
    if segs[0][0] < 0:
        return None
    if skip > 0:
        notes.append(f"starts {segs[0][0]:.2f} s into the track")
    notes.append(f"its sound changes {change:.1f} across the join, {natural:.1f} across the track's own bar lines")
    return segs, L


# ---- the duck and the fades -------------------------------------------------------------------------

def _ramp(x: np.ndarray) -> np.ndarray:
    """0 to 1 along a half cosine: no corner at either end."""
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0.0, 1.0))


def duck_db(talk: list[tuple[float, float]], L: float, depth: float, t: np.ndarray) -> np.ndarray:
    """The duck's gain in dB at times `t`: 0 in the open, -depth while talking, with its ramps."""
    g = np.zeros_like(t)
    if depth <= 0:
        return g
    for a, b in talk:
        if a < START_STAY:
            a = -LEAD - 1.0  # talk from the first second: start down rather than swell and dip
        if L - b < END_STAY:
            b = L + RISE + HOLD + 1.0  # too near the end to come up and settle: stay down
        down = _ramp((t - (a - LEAD)) / LEAD)
        up = 1.0 - _ramp((t - (b + HOLD)) / RISE)
        g = np.minimum(g, -depth * np.minimum(down, up))
    return g


def fades_gain(fit: Fit, L: float, t: np.ndarray) -> np.ndarray:
    """The fades, each along a quarter sine (the equal-power shape): a fade-out shaped so stays
    audible to its last frame, where a half cosine would be gone a tenth of a second early."""
    g = np.ones_like(t)
    if fit.fade_in > 0:
        g *= np.sin(0.5 * np.pi * np.clip(t / fit.fade_in, 0.0, 1.0))
    if fit.fade_out and fit.fade_out[1] - fit.fade_out[0] > 0:
        s, e = fit.fade_out
        g *= np.cos(0.5 * np.pi * np.clip((t - s) / (e - s), 0.0, 1.0))
    guard = 0.008  # never a click at the very end
    g *= np.clip((L - t) / guard, 0.0, 1.0)
    return g


# ---- the bed -------------------------------------------------------------------------------------

@dataclass
class Bed:
    clip: object
    opts: dict
    track: Track
    fit: Fit
    talk: list[tuple[float, float]]
    L: float
    n_samples: int
    music_gain_db: float
    voice_lufs: float
    _mixed: Optional[tuple] = None

    def stem(self) -> np.ndarray:
        """The music as it plays under the film: fitted, levelled, ducked and faded (n, 2)."""
        ranges = [(int(round(a * RATE)), int(round(b * RATE))) for a, b in self.fit.segments]
        m = audio.join(self.track.samples, RATE, ranges, fade=self.fit.crossfade)
        if len(m) < self.n_samples:
            m = np.concatenate([m, np.zeros((self.n_samples - len(m), m.shape[1]), np.float32)])
        m = m[: self.n_samples]
        t = np.arange(self.n_samples) / RATE
        curve = self.gain_curve(t)
        return (m * curve[:, None]).astype(np.float32)

    def gain_curve(self, t: np.ndarray) -> np.ndarray:
        """Linear gain: the level, the duck and the fades. Worked out on a 1 ms grid, then spread."""
        grid = np.arange(0.0, self.L + 0.002, 0.001)
        g_db = self.music_gain_db + duck_db(self.talk, self.L, self.opts["duck"], grid)
        g = (10.0 ** (g_db / 20.0)) * fades_gain(self.fit, self.L, grid)
        return np.interp(t, grid, g).astype(np.float32)

    def mix(self, voice: np.ndarray, rate: int) -> np.ndarray:
        """Job.sound: the voice with the music under it, at the target loudness."""
        music = self.stem()
        out = voice + music[: len(voice)]
        report = {}
        if self.opts["target_lufs"] != "off":
            out, report = loudness.normalize(out, float(self.opts["target_lufs"]), -1.0)
        self._mixed = (voice.copy(), music, out, report)
        return out

    def mixed(self, voice: Optional[np.ndarray] = None):
        if self._mixed is None:
            self.mix(self.clip.sound if voice is None else voice, RATE)
        return self._mixed

    # -- what is measured
    def measures(self) -> dict:
        voice, music, out, report = self.mixed()
        vb, mb = sound.blocks(voice), sound.blocks(music)
        under = sound.integrated(vb, self.talk) - sound.integrated(mb, self.talk) if self.talk else None
        open_spans = self._open_spans()
        open_db = (sound.integrated(vb, self.talk) - sound.integrated(mb, open_spans)) if self.talk and open_spans else None
        end_at = self._music_end(music)
        moves = self._moves()
        return {"under_speech_db": None if under is None else round(under, 1),
                "open_db": None if open_db is None else round(open_db, 1),
                "music_end": round(end_at, 3), "film_end": round(self.L, 3),
                "end_off_ms": round((end_at - self.L) * 1000, 1),
                "end_off_frames": round((end_at - self.L) * self.clip.info.rate, 2),
                "last_30ms_db": self._last_level(music),
                "joins": [self._beat_steadiness(j) for j in self.fit.joins],
                "track_beats_ms": self._beat_steadiness(None),
                "moves": moves, "loudness": report}

    def _beat_steadiness(self, at: Optional[float], span: float = 2.0) -> Optional[float]:
        """How far the beats either side of a join stray from the track's beat (ms, the largest): a
        join that came off the beat would show as one beat late or early. With `at` None, the same
        measure on the track itself around a bar line, for comparison."""
        g = self.track.grid
        if not g.steady:
            return None
        if at is None:
            bars = [b for b in g.bars if self.track.first + span < b < self.track.end - span]
            if not bars:
                return None
            x, at = self.track.samples, float(bars[len(bars) // 2])
        else:
            ranges = [(int(round(a * RATE)), int(round(b * RATE))) for a, b in self.fit.segments]
            x = audio.join(self.track.samples, RATE, ranges, fade=self.fit.crossfade)
        a = max(0.0, at - span - 0.5)
        seg = x[int(a * RATE):int((at + span + 0.5) * RATE)]
        if len(seg) < RATE:
            return None
        bands, _, _ = _spectra(seg.mean(axis=1))
        lb = np.log10(1e-9 + bands)
        flux = np.maximum(0.0, np.diff(lb, axis=0, prepend=lb[:1])).sum(axis=1)
        onset = np.maximum(0.0, flux - np.convolve(flux, np.ones(50) / 50, mode="same"))
        period = 60.0 / g.bpm
        beats = _beats(onset, period) + a
        near = beats[(beats >= at - span) & (beats <= at + span)]
        if len(near) < 3:
            return None
        return round(float(np.max(np.abs(np.diff(near) - period))) * 1000, 1)

    def _open_spans(self) -> list[tuple[float, float]]:
        """Stretches where the music is up in full: away from talk, its ramps, the fades, and the
        track's own ending (which is quieter by design)."""
        busy = [(0.0 if a < START_STAY else a - LEAD, (self.L if self.L - b < END_STAY else b + HOLD + RISE))
                for a, b in self.talk]
        lo = self.fit.fade_in
        hi = self.fit.fade_out[0] if self.fit.fade_out else self.L
        if self.fit.ending_at is not None:
            hi = min(hi, self.fit.ending_at - (self.track.end - self.track.body_end))
        out, t = [], lo
        for a, b in sorted(busy):
            if a > t:
                out.append((t, min(a, hi)))
            t = max(t, b)
        if t < hi:
            out.append((t, hi))
        return [(a, b) for a, b in out if b - a >= 0.4]

    def _last_level(self, music: np.ndarray) -> float:
        """How far under its own usual level the music is over the film's last 30 ms: an ended track is
        far down, a cut one is not."""
        env = sound.envelope_db(music, RATE, 0.001)
        alive = env[env > -100]
        if not len(alive):
            return 0.0
        tail = music[max(0, len(music) - int(0.03 * RATE)):]
        rms = float(np.sqrt(np.mean(tail.astype(np.float64) ** 2)))
        return round(20 * math.log10(max(rms, 1e-9)) - float(np.percentile(alive, 75)), 1)

    def _music_end(self, music: np.ndarray) -> float:
        """Where the music stops being heard: the last moment it is within 45 dB of its own level."""
        env = sound.envelope_db(music, RATE, 0.001)
        alive = env[env > -100]
        if not len(alive):
            return 0.0
        ref = float(np.percentile(alive, 75))
        idx = np.flatnonzero(env > ref - 45)
        return float((idx[-1] + 1) * 0.001) if len(idx) else 0.0

    def _moves(self) -> dict:
        """How often the music goes down and comes back up, and its shortest stay up."""
        if self.opts["duck"] <= 0 or not self.talk:
            return {"down": 0, "up": 0, "shortest_up_s": None}
        downs = sum(1 for a, _ in self.talk if a >= START_STAY)
        ups = sum(1 for _, b in self.talk if self.L - b >= END_STAY)
        stays = []
        for (a0, b0), (a1, _) in zip(self.talk, self.talk[1:]):
            stays.append((a1 - LEAD) - (b0 + HOLD + RISE))
        return {"down": downs, "up": ups, "shortest_up_s": round(min(stays), 2) if stays else None}


def _voice_and_talk(clip) -> tuple[list[tuple[float, float]], float]:
    L = clip.info.frames / clip.info.rate
    talk = sound.speech_spans(clip.words, clip.quiet, L, join=JOIN) if clip.info.has_audio else []
    voice = sound.integrated(sound.blocks(clip.sound), talk) if talk else sound.SILENCE_DB
    return talk, voice


def _from_plan(plan: dict, track: Track, L: float, fades: tuple[float, float]) -> tuple[Fit, list, Optional[float]]:
    from . import PlanError

    f = plan.get("fit") or {}
    try:
        segs = [(float(a), float(b)) for a, b in f["segments"]]
    except (KeyError, TypeError, ValueError):
        raise PlanError("the plan has no fit.segments; run once without --plan to get one to edit") from None
    total = sum(b - a for a, b in segs)
    for a, b in segs:
        if not (0 <= a < b <= len(track.samples) / RATE + 1e-3):
            raise PlanError(f"the piece {a:.2f}-{b:.2f} s is not inside the track")
    if abs(total - L) > 0.5:
        raise PlanError(f"the pieces add up to {total:.2f} s but the film is {L:.2f} s")
    try:
        talk = [(float(a), float(b)) for a, b in plan.get("talk", [])]
    except (TypeError, ValueError):
        raise PlanError("talk must be a list of [start, end] seconds") from None
    fo = f.get("fade_out")
    fit = Fit(str(f.get("mode", "by hand")), segs, float(f.get("crossfade", XF_BEAT)), fades[0],
              (float(fo[0]), float(fo[1])) if fo else None, f.get("ending_at"), ["from the plan"])
    gain = (plan.get("level") or {}).get("music_gain_db")
    return fit, talk, (float(gain) if gain is not None else None)


def _find_music(raw: str, clip_path: str) -> Path:
    if not raw.strip():
        raise O.OptionError("music-bed needs your track: give its path with music=/path/to/song.mp3. The pack "
                            "ships no music, so use a track you have the rights to")
    p = Path(raw).expanduser()
    for cand in (p, Path(clip_path).parent / p):
        if cand.exists():
            return cand.resolve()
    raise O.OptionError(f"there is no music file at {raw}")


# ---- the effect ----------------------------------------------------------------------------------

def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from . import Job

    info = clip.info
    fps = info.rate
    L = info.frames / fps
    tl = Timeline.whole(L)
    fr = framing_mod.plan(info.width, info.height, fps, info.frames, None, shape="keep")
    n_samples = int(round(tl.output_frames(fps) / fps * RATE))
    music_path = _find_music(opts["music"], clip.path)
    track = analyse(str(music_path))
    fades = tuple(float(v) for v in opts["fade"].split(","))
    talk, voice = _voice_and_talk(clip)
    gain_from_plan = None
    if plan is not None:
        fit, talk, gain_from_plan = _from_plan(plan, track, L, fades)
    else:
        fit = fit_track(track, L, opts["fit"], fades, talk)

    # The level: `level` dB under the voice where the music is up. Measured on the whole of the track as
    # it will play (a short open stretch alone could fall in a quiet bar). No voice: a usual level.
    ranges = [(int(round(a * RATE)), int(round(b * RATE))) for a, b in fit.segments]
    music_lufs = sound.integrated(sound.blocks(audio.join(track.samples, RATE, ranges, fade=fit.crossfade)))
    reference = voice if voice > -70 else -20.0
    gain_db = gain_from_plan if gain_from_plan is not None else (reference + opts["level"]) - music_lufs
    bed = Bed(clip, opts, track, fit, talk, L, n_samples, gain_db, voice)

    g = track.grid
    music = {"file": str(music_path), "seconds": round(len(track.samples) / RATE, 3),
             "sound_from": round(track.first, 3), "sound_to": round(track.end, 3), "ends_itself": track.ends_itself,
             "tempo_bpm": round(g.bpm, 1), "beat_sure": round(g.sure, 2), "bar_s": round(g.bar, 3),
             "bars": [round(float(b), 3) for b in g.bars[:64]], "loudness_lufs": round(track.loud, 1)}
    sheet_times = [round(L * (k + 0.5) / 6, 3) for k in range(6)]
    out_plan = {"effect": NAME, "clip": clip.path, "options": opts,
                "output": {"width": fr.out_w, "height": fr.out_h, "fps": round(fps, 3), "seconds": round(L, 3),
                           "framing": fr.note},
                "music": music, "fit": fit.as_dict(),
                "level": {"voice_lufs": round(voice, 1), "music_gain_db": round(gain_db, 2),
                          "open_db": opts["level"], "under_speech_db": opts["level"] - opts["duck"]},
                "talk": [[a, b] for a, b in talk],
                "duck": {"db": opts["duck"], "lead_s": LEAD, "hold_s": HOLD, "rise_s": RISE, "join_s": JOIN},
                "loudness": {"target_lufs": opts["target_lufs"], "true_peak_dbtp": -1.0},
                "sheet_times": sheet_times}

    beat = (f"{g.bpm:.0f} beats a minute, bars of {g.bar:.2f} s" if g.steady else
            "no steady beat found, so joins are long crossfades")
    checks = {
        "music": f"{music_path.name}, {len(track.samples) / RATE:.1f} s, {beat}; "
                 + ("it ends by itself" if track.ends_itself else "it stops rather than ending"),
        "fit": f"{fit.mode}: " + ("; ".join(fit.notes) if fit.notes else "plays from its start")
               + (f"; fades out over {fit.fade_out[1] - fit.fade_out[0]:.2f} s from {fit.fade_out[0]:.2f} s"
                  if fit.fade_out else ""),
        "talk": (f"{len(talk)} stretch(es) of talk, {sum(b - a for a, b in talk):.1f} s; the music "
                 f"{opts['duck']:g} dB down under them" if talk else "no talk heard: the music stays level"),
        "level": f"the music {-opts['level']:g} dB under the voice in the open, "
                 f"{-(opts['level'] - opts['duck']):g} dB under it while talking (gain {gain_db:+.1f} dB)",
    }
    if opts["level"] - opts["duck"] > -12 and talk:
        checks["warning"] = ("the music sits less than 12 dB under the voice while talking; words may be hard "
                             "to follow on a phone")

    def stills() -> list[float]:
        return sheet_times[1::2]

    def frame_from(path: str):
        vinfo = video.probe(path)
        return lambda t: video.read_frame(vinfo, vinfo.frame_at(t))

    def preview(out: Path) -> dict:
        voice_s = clip.sound
        sheet = _sheet(bed, voice_s, frame_from(clip.path), out.with_suffix(".sound.jpg"), measured=None)
        return {"sound sheet": str(sheet)}

    def after(result: dict, out: Path) -> dict:
        m = loudness.measure_file(out)
        heard = _heard_again(clip, out)
        meas = bed.measures()
        sheet = _sheet(bed, None, frame_from(str(out)), out.with_suffix(".sound.jpg"), measured=(m, heard, meas))
        lines = {
            "loudness": f"{m.i:.1f} LUFS integrated, true peak {m.tp:.1f} dBTP (measured on the file with loudnorm)",
            "voice over music": (f"{meas['under_speech_db']:.1f} dB while talking, {meas['open_db']:.1f} dB in the open"
                                 if meas["under_speech_db"] is not None and meas["open_db"] is not None else
                                 f"{meas['under_speech_db']} dB while talking"),
            "ending": f"the music ends {abs(meas['end_off_ms']):.0f} ms "
                      f"{'before' if meas['end_off_ms'] <= 0 else 'after'} the film ({abs(meas['end_off_frames']):.2f} frames); "
                      f"over the last 30 ms it is {-meas['last_30ms_db']:.0f} dB under its own level (it ends, it is not cut)",
            "words heard again": heard["verdict"],
        }
        if meas["joins"] and all(v is not None for v in meas["joins"]):
            own = meas["track_beats_ms"]
            lines["joins"] = (f"the beats across each join stay within {max(meas['joins']):.0f} ms of the track's beat"
                              + (f" (the track's own beats stray by up to {own:.0f} ms)" if own is not None else ""))
        lines["sound sheet"] = str(sheet)
        return lines

    return Job(NAME, clip, tl, fr, None, out_plan, checks=checks, still_times=stills, after=after,
               sound=bed.mix, preview=preview)


def _heard_again(clip, out: Path) -> dict:
    """The film with the music under it, transcribed again and compared word by word with the words
    heard without it: music that covers speech makes the transcriber lose or change words."""
    from ..core.clip import Clip

    before = [w.key for w in clip.heard if w.key]
    after = [w.key for w in Clip(out).heard if w.key]
    sm = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    same = sum(b.size for b in sm.get_matching_blocks())
    changed = [" ".join(before[i1:i2]) + " -> " + " ".join(after[j1:j2])
               for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]
    verdict = f"{same} of {len(before)} words the same with the music under them"
    if changed:
        verdict += " (" + "; ".join(changed[:4]) + ")"
    return {"same": same, "of": len(before), "changed": changed, "verdict": verdict}


def _sheet(bed: Bed, voice: Optional[np.ndarray], frame_at, path: Path, measured) -> Path:
    v, music, out, report = bed.mixed(voice)
    # Draw the stems at the level they have in the finished mix.
    scale = 1.0
    if report.get("after"):
        a = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
        b = float(np.sqrt(np.mean((v + music[: len(v)]).astype(np.float64) ** 2)))
        scale = a / b if b > 0 else 1.0
    t = np.arange(0.0, bed.L, 0.01)
    curve_db = bed.music_gain_db + duck_db(bed.talk, bed.L, bed.opts["duck"], t) + \
        20 * np.log10(np.maximum(fades_gain(bed.fit, bed.L, t), 1e-3)) - bed.music_gain_db
    lanes = [soundsheet.Lane("voice", v * scale, (196, 196, 206), spans=bed.talk),
             soundsheet.Lane("music", music * scale, (92, 168, 255), curve=(t, curve_db),
                             curve_label="white: its gain", spans=bed.talk)]
    marks = [soundsheet.Mark(j, f"join {k + 1}", (255, 140, 90)) for k, j in enumerate(bed.fit.joins)]
    if bed.fit.fade_out:
        marks.append(soundsheet.Mark(bed.fit.fade_out[0], "fade", (150, 150, 158), lanes=("music",)))
    marks.append(soundsheet.Mark(bed.L, "", (255, 206, 84)))
    meas = bed.measures()
    zooms = [soundsheet.Zoom(j, f"join {k + 1}: a crossfade before the beat", lanes=("music",), before=0.25,
                             after=0.25, note=f"track {bed.fit.segments[k][1]:.2f} s -> {bed.fit.segments[k + 1][0]:.2f} s")
             for k, j in enumerate(bed.fit.joins)]
    last = bed.clip.info.frames - 1
    zooms.append(soundsheet.Zoom(bed.L, "the film's end (yellow), the music's (green)", lanes=("music",),
                                 onset=meas["music_end"], before=0.4, after=0.03, frame=last,
                                 note=f"music ends {meas['end_off_ms']:+.0f} ms from the film's end"))
    g = bed.track.grid
    lines = [f"{Path(bed.track.path).name}: {bed.fit.mode}; " + ("; ".join(bed.fit.notes) or "plays from its start"),
             (f"{g.bpm:.0f} BPM, bars {g.bar:.2f} s; " if g.steady else "no steady beat; ")
             + f"duck {bed.opts['duck']:g} dB, the music {-bed.opts['level']:g} dB under the voice in the open"]
    def over(mm: dict) -> str:
        said = f"voice over music {mm['under_speech_db']} dB while talking" if mm["under_speech_db"] is not None \
            else "no talk"
        return said + (f", {mm['open_db']} dB in the open" if mm["open_db"] is not None else ", never up in full")

    if measured is not None:
        m, heard, mm = measured
        lines.append(f"file: {m.i:.1f} LUFS, true peak {m.tp:.1f} dBTP; {over(mm)}; ends {mm['end_off_ms']:+.0f} ms "
                     f"from the film's end; {heard['same']}/{heard['of']} words heard the same")
    else:
        lines.append(f"planned: {over(meas)}; ends {meas['end_off_ms']:+.0f} ms from the film's end (not rendered yet)")
    sheet = soundsheet.Sheet(title=f"music-bed · {Path(bed.clip.path).name}", duration=bed.L,
                             fps=bed.clip.info.rate, frame_at=frame_at, lanes=lanes,
                             frames=[bed.L * (k + 0.5) / 6 for k in range(6)], marks=marks, zooms=zooms, lines=lines)
    return soundsheet.draw(sheet, path)
