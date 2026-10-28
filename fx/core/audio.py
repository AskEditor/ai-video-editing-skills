"""Sound: the speech envelope that finds pauses, and joins that never click.

The envelope is the loudness of the speech band (100 Hz to 7.5 kHz) in 10 ms steps, in dB. Wind
rumble sits below the band and hiss is mostly above it, so a pause reads as a pause on a windy
path too, as long as the voice is louder than the wind.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HOP = 0.01  # seconds per envelope step


@dataclass(frozen=True)
class Envelope:
    db: np.ndarray  # loudness per step
    hop: float
    floor: float  # the quiet between words
    speech: float  # a typical spoken syllable
    threshold: float  # below this is a pause

    @property
    def voiced(self) -> float:
        """A sound has to reach this, somewhere, to count as speech rather than wind or a knock."""
        return self.floor + 0.5 * (self.speech - self.floor)

    def at(self, t: float) -> float:
        i = int(min(len(self.db) - 1, max(0, round(t / self.hop))))
        return float(self.db[i])

    def mean(self, t0: float, t1: float) -> float:
        i0 = int(max(0, round(t0 / self.hop)))
        i1 = int(min(len(self.db), max(i0 + 1, round(t1 / self.hop))))
        return float(self.db[i0:i1].mean()) if i1 > i0 else self.at(t0)

    def quietest(self, t0: float, t1: float) -> float:
        """The time of the quietest 30 ms between t0 and t1: the best place for a cut."""
        i0 = int(max(0, round(t0 / self.hop)))
        i1 = int(min(len(self.db), round(t1 / self.hop) + 1))
        if i1 - i0 < 3:
            return (t0 + t1) / 2
        seg = np.convolve(self.db[i0:i1], np.ones(3) / 3, mode="same")
        return (i0 + int(np.argmin(seg))) * self.hop


def envelope(mono: np.ndarray, rate: int, hop: float = HOP, win: float = 0.025,
             band: tuple[float, float] = (100.0, 7500.0)) -> Envelope:
    n_hop = max(1, int(round(rate * hop)))
    n_win = max(n_hop, int(round(rate * win)))
    steps = int(np.ceil(len(mono) / n_hop))
    x = np.pad(mono.astype(np.float32), (n_win // 2, n_win + n_hop))
    frames = np.lib.stride_tricks.sliding_window_view(x, n_win)[::n_hop][:steps]
    window = np.hanning(n_win).astype(np.float32)
    spectrum = np.fft.rfft(frames * window, axis=1)
    freqs = np.fft.rfftfreq(n_win, 1.0 / rate)
    keep = (freqs >= band[0]) & (freqs <= band[1])
    power = (np.abs(spectrum[:, keep]) ** 2).sum(axis=1) / (window ** 2).sum()
    db = 10.0 * np.log10(power + 1e-12)
    # A 30 ms median takes out clicks and lip noise without moving the edges of words.
    padded = np.pad(db, 1, mode="edge")
    db = np.median(np.stack([padded[:-2], padded[1:-1], padded[2:]]), axis=0)

    floor = float(np.percentile(db, 5))
    speech = float(np.percentile(db, 90))
    # A pause is close to the floor: a third of the way up to speech, but always at least 18 dB
    # under it, so a take with hardly any pauses never has its quiet words called silence.
    threshold = min(floor + max(6.0, 0.33 * (speech - floor)), speech - 18.0)
    return Envelope(db=db.astype(np.float32), hop=hop, floor=floor, speech=speech, threshold=threshold)


def quiet_runs(env: Envelope, min_len: float = 0.08, bridge: float = 0.05, sustained: float = 0.25,
               protect: list[tuple[float, float]] | None = None) -> list[tuple[float, float]]:
    """Stretches with no speech in them, as (start, end) seconds.

    A stretch above the pause threshold is speech when it reaches `env.voiced` somewhere, lasts
    `sustained` seconds, or overlaps one of the `protect` spans (the middles of heard words);
    anything else (a gust of wind, a click, a breath catch, a knock) is part of the quiet around
    it. A blip shorter than `bridge` never breaks a quiet stretch.
    """
    quiet = env.db < env.threshold
    loud = ~quiet
    n_bridge = int(round(bridge / env.hop))
    n_sustained = int(round(sustained / env.hop))
    guarded = np.zeros(len(quiet), bool)
    for a, b in protect or []:
        guarded[int(max(0, a / env.hop)):int(min(len(quiet), b / env.hop + 1))] = True
    edges = np.flatnonzero(np.diff(np.concatenate([[0], loud.astype(np.int8), [0]])))
    for s, e in zip(edges[::2], edges[1::2]):
        inner = s > 0 and e < len(quiet)
        is_speech = (env.db[s:e].max() >= env.voiced or (e - s) >= n_sustained or guarded[s:e].any())
        if (inner and e - s <= n_bridge) or not is_speech:
            quiet[s:e] = True
    runs = []
    edges = np.flatnonzero(np.diff(np.concatenate([[0], quiet.astype(np.int8), [0]])))
    for s, e in zip(edges[::2], edges[1::2]):
        if (e - s) * env.hop >= min_len:
            runs.append((s * env.hop, e * env.hop))
    return runs


def join(samples: np.ndarray, rate: int, ranges: list[tuple[int, int]], fade: float = 0.02) -> np.ndarray:
    """Join sample ranges [(start, end), …] end to end with short equal-power crossfades.

    The result is exactly as long as the ranges added up, so the sound stays in step with the
    picture. Each join borrows `fade / 2` from either side of the cut and blends across it; the very
    start and end get a 5 ms fade so they never click either.
    """
    ranges = [(int(s), int(e)) for s, e in ranges if e > s]
    channels = samples.shape[1]
    total = sum(e - s for s, e in ranges)
    out = np.zeros((total, channels), np.float32)
    if not ranges:
        return out
    half = max(1, int(round(fade * rate / 2)))
    lengths = [e - s for s, e in ranges]
    # Each join's half-width, never more than half of either neighbouring piece.
    joins = [min(half, lengths[k] // 2, lengths[k + 1] // 2) for k in range(len(ranges) - 1)]
    edge = max(1, int(round(0.005 * rate)))
    n = len(samples)
    pos = 0
    for k, (s, e) in enumerate(ranges):
        length = lengths[k]
        pre = joins[k - 1] if k > 0 else 0
        post = joins[k] if k < len(joins) else 0
        a, b = s - pre, e + post
        chunk = np.zeros((b - a, channels), np.float32)
        src0, src1 = max(0, a), min(n, b)
        if src1 > src0:
            chunk[src0 - a:src1 - a] = samples[src0:src1]
        gain = np.ones(b - a, np.float32)
        if pre:
            gain[: 2 * pre] = np.sin(np.linspace(0, np.pi / 2, 2 * pre, dtype=np.float32))
        elif k == 0:
            m = min(edge, len(gain))
            gain[:m] = np.linspace(0, 1, m, dtype=np.float32)
        if post:
            gain[len(gain) - 2 * post:] *= np.cos(np.linspace(0, np.pi / 2, 2 * post, dtype=np.float32))
        elif k == len(ranges) - 1:
            m = min(edge, len(gain))
            gain[len(gain) - m:] *= np.linspace(1, 0, m, dtype=np.float32)
        chunk *= gain[:, None]
        o0 = pos - pre
        out[o0:o0 + len(chunk)] += chunk
        pos += length
    return out


# ---- room tone ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RoomTone:
    samples: np.ndarray  # (n, channels): as long as was asked for
    taken: tuple[float, float]  # the stretch of the clip it was looped from, in seconds
    level_db: float  # its level (RMS, dBFS) as laid
    gain_db: float  # what was added to the stretch to bring it to the room's level


def _rms_db(x: np.ndarray) -> float:
    return float(10.0 * np.log10(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


def room_tone(samples: np.ndarray, rate: int, seconds: float, quiet: list[tuple[float, float]],
              least: float = 0.25, longest: float = 1.0, edge: float = 0.06, fade: float = 0.08,
              most_gain: float = 6.0) -> RoomTone:
    """The clip's own room sound, `seconds` long: its quietest stretch with nobody speaking (from the
    `quiet` runs, `edge` seconds in from either end so no word's tail comes with it), looped with
    equal-power crossfades of `fade` seconds and brought to the room's level (the median of the
    quiet runs, never more than `most_gain` dB up). What fills a hold in the picture so it plays as
    the room holding still, not as the sound dropping out."""
    channels = samples.shape[1]
    n_out = max(0, int(round(seconds * rate)))
    hop = max(1, int(round(0.01 * rate)))
    win = int(round(0.05 * rate))
    runs = [(int((a + edge) * rate), int((b - edge) * rate)) for a, b in quiet]
    runs = [(max(0, a), min(len(samples), b)) for a, b in runs]
    runs = [(a, b) for a, b in runs if b - a >= int(least * rate)]
    if not runs:  # nobody is ever quiet for long: the quietest stretch of the whole clip
        runs = [(0, len(samples))]
    want = int(longest * rate)
    best, best_db = None, np.inf
    levels: list[float] = []
    for a, b in runs:
        length = min(want, b - a)
        for s in range(a, b - length + 1, hop):
            d = _rms_db(samples[s:s + length])
            if -120.0 < d < best_db:  # digital zero is not a room
                best, best_db = (s, s + length), d
        levels += [_rms_db(samples[s:s + win]) for s in range(a, b - win + 1, win)]
    if best is None or n_out == 0:
        return RoomTone(np.zeros((n_out, channels), np.float32), (0.0, 0.0), -120.0, 0.0)
    room = float(np.median([v for v in levels if v > -120.0])) if levels else best_db
    gain_db = float(np.clip(room - best_db, 0.0, most_gain))
    piece = samples[best[0]:best[1]].astype(np.float32) * np.float32(10 ** (gain_db / 20))
    p = len(piece)
    f = max(1, min(int(round(fade * rate)), p // 4))
    up = np.sin(np.linspace(0, np.pi / 2, f, dtype=np.float32))[:, None]
    down = np.cos(np.linspace(0, np.pi / 2, f, dtype=np.float32))[:, None]
    out = np.zeros((n_out + p, channels), np.float32)
    pos, first = 0, True
    while pos < n_out:
        c = piece.copy()
        if not first:
            c[:f] *= up
        if pos + p - f < n_out:  # another copy follows: fade out under it (the last one runs to the end)
            c[p - f:] *= down
        out[pos:pos + p] += c
        pos += p - f
        first = False
    out = out[:n_out]
    return RoomTone(out, (best[0] / rate, best[1] / rate), _rms_db(out) if n_out else -120.0, gain_db)
