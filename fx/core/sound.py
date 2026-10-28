"""Sound an effect adds under the clip's own: reading a track, its loudness, laying a sound on a
moment, and where somebody is talking.

- `read(path)`: any sound file, or the sound of a video, as 48 kHz stereo float32.
- `loudness`: K-weighted loudness after ITU-R BS.1770, computed here in numpy, for setting one sound
  against another while a mix is being built. The finished mix is measured by ffmpeg itself
  (`loudness.py`), so the numbers a person reads come from the tool everybody uses.
- `lay()`: a sound added so that the moment it lands (the attack of a struck sound, the peak of a
  swell) falls on a given time, to the sample.
- `speech_spans()`: the stretches where somebody is talking, from the word times and the quiet
  stretches the core already finds, joined across the short gaps between phrases.
- `library(kind)`: one of the pack's own sounds in `sounds/` (a whoosh, a hit, a pop), read once,
  with the second it lands at.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .video import AUDIO_CHANNELS, AUDIO_RATE, MediaError, need_ffmpeg

SILENCE_DB = -120.0


# ---- reading -----------------------------------------------------------------------------------

def read(path: str | Path, rate: int = AUDIO_RATE, channels: int = AUDIO_CHANNELS) -> np.ndarray:
    """A sound file (mp3, wav, m4a, flac, ogg…) or a video's sound as float32 of shape (n, channels)."""
    need_ffmpeg()
    path = Path(path)
    if not path.exists():
        raise MediaError(f"there is no file at {path}")
    run = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0", "-vn", "-ac", str(channels),
         "-ar", str(rate), "-f", "f32le", "-"],
        capture_output=True,
    )
    if run.returncode != 0 or not run.stdout:
        raise MediaError(f"could not read any sound from {path}")
    return np.frombuffer(run.stdout, np.float32).reshape(-1, channels).copy()


def db(x: float) -> float:
    return 20.0 * np.log10(max(float(x), 1e-12))


def gain(decibels: float) -> float:
    return float(10.0 ** (decibels / 20.0))


# ---- loudness (BS.1770, in numpy) --------------------------------------------------------------

# The two filters that make "K-weighting" at 48 kHz, as the standard gives them: a shelf that lifts
# what the head makes louder, then a high-pass that drops what we hardly hear.
_SHELF = ((1.53512485958697, -2.69169618940638, 1.19839281085285), (1.0, -1.69065929318241, 0.73248077421585))
_HIGHPASS = ((1.0, -2.0, 1.0), (1.0, -1.99004745483398, 0.99007225036621))


def _response(coeffs, n_fft: int) -> np.ndarray:
    (b0, b1, b2), (_, a1, a2) = coeffs
    z = np.exp(-2j * np.pi * np.arange(n_fft // 2 + 1) / n_fft)
    return (b0 + b1 * z + b2 * z * z) / (1.0 + a1 * z + a2 * z * z)


_IR_LENGTH = 1 << 13  # 0.17 s: the two filters' response has died away long before this
_CHUNK = 1 << 18


def _k_impulse() -> np.ndarray:
    n_fft = 1 << 16
    h = _response(_SHELF, n_fft) * _response(_HIGHPASS, n_fft)
    return np.fft.irfft(h, n_fft)[:_IR_LENGTH]


def k_weight(samples: np.ndarray, rate: int = AUDIO_RATE) -> np.ndarray:
    """The samples through the K-weighting filters, as float64 (n, channels). The filters' impulse
    response is applied by FFT, a chunk at a time, which gives what running them would give."""
    if rate != 48000:
        raise ValueError("k_weight works at 48 kHz, the rate every sound in the pack is read at")
    x = samples if samples.ndim == 2 else samples[:, None]
    n, ir = len(x), _k_impulse()
    n_fft = 1 << int(np.ceil(np.log2(_CHUNK + len(ir))))
    spec = np.fft.rfft(ir, n_fft)[:, None]
    y = np.zeros((n + len(ir), x.shape[1]))
    for a in range(0, n, _CHUNK):
        part = x[a:a + _CHUNK].astype(np.float64)
        conv = np.fft.irfft(np.fft.rfft(part, n_fft, axis=0) * spec, n_fft, axis=0)[:len(part) + len(ir) - 1]
        y[a:a + len(conv)] += conv
    return y[:n]


@dataclass(frozen=True)
class Blocks:
    """The mean square of the K-weighted sound in overlapping blocks (summed over the channels)."""
    power: np.ndarray
    starts: np.ndarray  # seconds
    length: float  # seconds per block

    @property
    def lufs(self) -> np.ndarray:
        return -0.691 + 10.0 * np.log10(np.maximum(self.power, 1e-15))

    def centres(self) -> np.ndarray:
        return self.starts + self.length / 2


def blocks(samples: np.ndarray, rate: int = AUDIO_RATE, length: float = 0.4, hop: float = 0.1,
           weighted: Optional[np.ndarray] = None) -> Blocks:
    """`length`-second blocks every `hop` seconds (0.4 and 0.1 are the standard's own)."""
    y = k_weight(samples, rate) if weighted is None else weighted
    n_block, n_hop = int(round(length * rate)), int(round(hop * rate))
    if len(y) < n_block:
        y = np.concatenate([y, np.zeros((n_block - len(y), y.shape[1]))])
    cs = np.concatenate([np.zeros((1, y.shape[1])), np.cumsum(y * y, axis=0)])
    idx = np.arange(0, len(y) - n_block + 1, n_hop)
    power = ((cs[idx + n_block] - cs[idx]) / n_block).sum(axis=1)
    return Blocks(power=power, starts=idx / rate, length=length)


def integrated(b: Blocks, spans: Optional[Iterable[tuple[float, float]]] = None) -> float:
    """Gated loudness in LUFS (absolute gate at -70, relative gate 10 LU under), over the blocks whose
    middle falls in `spans` when given. -120 when there is nothing to measure."""
    keep = np.ones(len(b.power), bool)
    if spans is not None:
        c = b.centres()
        keep = np.zeros(len(b.power), bool)
        for a, e in spans:
            keep |= (c >= a) & (c <= e)
    p = b.power[keep]
    lufs = -0.691 + 10.0 * np.log10(np.maximum(p, 1e-15))
    p = p[lufs > -70.0]
    if not len(p):
        return SILENCE_DB
    relative = -0.691 + 10.0 * np.log10(p.mean()) - 10.0
    p = p[-0.691 + 10.0 * np.log10(p) > relative]
    return float(-0.691 + 10.0 * np.log10(p.mean())) if len(p) else SILENCE_DB


def loudest(samples: np.ndarray, rate: int = AUDIO_RATE, window: float = 0.1) -> float:
    """The loudness of the loudest `window` seconds, in LUFS: what a short sound is levelled by, since
    the ear judges a click or a whoosh by about its loudest tenth of a second."""
    return float(blocks(samples, rate, length=window, hop=0.005).lufs.max())


# ---- laying a sound ------------------------------------------------------------------------------

def lay(out: np.ndarray, sound: np.ndarray, at: float, rate: int = AUDIO_RATE, anchor: float = 0.0,
        level: float = 1.0) -> tuple[int, int]:
    """Add `sound` into `out` (both (n, channels)) so that its sample at `anchor` seconds falls exactly
    at `at` seconds. What would fall before the start or after the end is left off; a sound cut at
    the start gets a 3 ms fade so it never clicks. Returns the samples it covers in `out`."""
    s = sound if sound.ndim == 2 else sound[:, None]
    if s.shape[1] != out.shape[1]:
        s = np.repeat(s[:, :1], out.shape[1], axis=1)
    start = int(round(at * rate)) - int(round(anchor * rate))
    a, b = max(0, start), min(len(out), start + len(s))
    if b <= a:
        return a, a
    piece = s[a - start:b - start].astype(np.float32) * level
    if a > start:
        m = min(len(piece), int(0.003 * rate))
        piece[:m] *= np.linspace(0, 1, m, dtype=np.float32)[:, None]
    out[a:b] += piece
    return a, b


def resample(sound: np.ndarray, factor: float) -> np.ndarray:
    """The sound played `factor` times faster (higher and shorter when above 1), band-limited."""
    if abs(factor - 1.0) < 1e-6:
        return sound
    n = len(sound)
    m = max(2, int(round(n / factor)))
    spec = np.fft.rfft(sound, axis=0)
    keep = min(len(spec), m // 2 + 1)
    out_spec = np.zeros((m // 2 + 1,) + sound.shape[1:], complex)
    out_spec[:keep] = spec[:keep]
    return (np.fft.irfft(out_spec, m, axis=0) * (m / n)).astype(np.float32)


# ---- where the voice is ------------------------------------------------------------------------

def speech_spans(words, quiet: list[tuple[float, float]], duration: float, join: float = 1.2,
                 edge: float = 0.05) -> list[tuple[float, float]]:
    """Where somebody is talking, as (start, end) seconds.

    A stretch counts when the sound says a voice is there (it lies between the quiet runs) and a
    heard word sits in it; it then runs from just before its first word to just after its last. A
    word the transcriber put in a stretch the sound calls quiet (a breath, a word it invented) does
    not count, and a gust or a knock with no word in it does not either. Stretches closer than
    `join` seconds are one: a short breath between phrases is part of talking.
    """
    voiced, t = [], 0.0
    for q0, q1 in sorted(quiet):
        if q0 > t:
            voiced.append((t, q0))
        t = max(t, q1)
    if t < duration:
        voiced.append((t, duration))
    spans = []
    for v0, v1 in voiced:
        inside = [w for w in words if min(v1, w.end) - max(v0, w.start) >= min(0.06, 0.5 * (w.end - w.start))]
        if not inside:
            continue
        a = max(v0, min(w.start for w in inside) - edge)
        b = min(v1, max(w.end for w in inside) + edge)
        if b > a:
            spans.append((a, b))
    spans.sort()
    merged: list[tuple[float, float]] = []
    for a, b in spans:
        if merged and a - merged[-1][1] < join:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return [(round(a, 3), round(b, 3)) for a, b in merged]


def envelope_db(samples: np.ndarray, rate: int = AUDIO_RATE, hop: float = 0.01) -> np.ndarray:
    """The level of the sound in dB (full scale) every `hop` seconds: the RMS of each step, for drawing."""
    x = samples.mean(axis=1) if samples.ndim == 2 else samples
    n = max(1, int(round(hop * rate)))
    steps = int(np.ceil(len(x) / n))
    x = np.pad(x, (0, steps * n - len(x)))
    rms = np.sqrt(np.mean(x.reshape(steps, n).astype(np.float64) ** 2, axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-9))


# ---- the pack's own sounds ------------------------------------------------------------------------

_LIBRARY: dict = {}


def library(kind: str) -> tuple[np.ndarray, float]:
    """One of the pack's own sounds in `sounds/` (made by sounds/make_sounds.py), as float32 (n, 2)
    at 48 kHz, and the second it lands at (its anchor)."""
    import json
    import wave

    from .paths import PACK

    if kind not in _LIBRARY:
        meta = json.loads((PACK / "sounds" / "sounds.json").read_text())[kind]
        with wave.open(str(PACK / "sounds" / meta["file"])) as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, w.getnchannels())
        x = pcm.astype(np.float32) / 32768.0
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        _LIBRARY[kind] = (x, float(meta["anchor_s"]))
    return _LIBRARY[kind]
