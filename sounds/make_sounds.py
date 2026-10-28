"""The pack's own sounds, made from nothing but numbers: no recording, no sample, no library.

    .venv/bin/python sounds/make_sounds.py        # writes sounds/<name>.wav and sounds/sounds.json

Each sound is built from noise and sine waves shaped in time, levelled so that its loudest tenth of a
second has the same K-weighted loudness as every other (`LEVEL`), and written as a 48 kHz, 16-bit
stereo WAV. `sounds.json` records, for each one, where it lands: a struck sound (pop, click, tick,
hit, ding) lands on its attack, a swell (whoosh, swoosh-in, riser) on its peak. The sound-effects
effect lines that point up with the first frame of its event.

The same script gives the same files every time (fixed seeds), so a changed sound shows in git as a
changed script.
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from fx.core.sound import loudest  # noqa: E402

RATE = 48000
LEVEL = -18.0  # LUFS: every sound's loudest tenth of a second, before the effect sets it under a voice
PEAK = -1.0  # dBFS: no sound's sample goes above this


# ---- the parts sounds are made of -----------------------------------------------------------------

def times(seconds: float) -> np.ndarray:
    return np.arange(int(round(seconds * RATE))) / RATE


def shaped_noise(n: int, centre_hz: np.ndarray, width_oct: np.ndarray, seed: int,
                 n_fft: int = 1024, hop: int = 128) -> np.ndarray:
    """White noise through a band whose centre and width (in octaves, at half height) follow the
    given curves, sample by sample: the sound of air or friction moving in pitch."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n + 2 * n_fft)
    window = np.hanning(n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / RATE)
    lf = np.log2(np.maximum(freqs, 1.0))
    out = np.zeros(n + 2 * n_fft)
    weight = np.zeros(n + 2 * n_fft)
    for s in range(0, n + n_fft, hop):
        k = min(n - 1, max(0, s + n_fft // 2 - n_fft))
        sigma = max(0.05, float(width_oct[k])) / 2.355
        mask = np.exp(-0.5 * ((lf - np.log2(float(centre_hz[k]))) / sigma) ** 2)
        seg = np.fft.irfft(np.fft.rfft(x[s:s + n_fft] * window) * mask, n_fft) * window
        out[s:s + n_fft] += seg
        weight[s:s + n_fft] += window ** 2
    y = (out / np.maximum(weight, 1e-3))[n_fft:n_fft + n]
    return y / (np.abs(y).max() + 1e-12)


def glide(f: np.ndarray) -> np.ndarray:
    """A sine whose frequency follows `f` (Hz per sample)."""
    return np.sin(2 * np.pi * np.cumsum(f) / RATE)


def bright_burst(n: int, seconds: float, seed: int) -> np.ndarray:
    """A very short burst of bright noise (the 'tk' at the front of a struck sound)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / RATE
    x = np.diff(rng.standard_normal(n + 1))
    return x * np.exp(-t / seconds) / 2.0


def pan(mono: np.ndarray, position: np.ndarray | float) -> np.ndarray:
    """Equal-power placement between left (-1) and right (+1)."""
    theta = (np.clip(position, -1, 1) + 1) * np.pi / 4
    return np.stack([mono * np.cos(theta), mono * np.sin(theta)], axis=1) * np.sqrt(2)


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def attack(t: np.ndarray, seconds: float) -> np.ndarray:
    return np.clip(t / seconds, 0, 1)


# ---- the sounds -------------------------------------------------------------------------------------

def whoosh():
    """Air rushing past and away: a band of noise that swells, brightens at its height and falls off
    faster than it came, travelling from left to right. For a cut."""
    t = times(0.80)
    peak = 0.46
    env = np.where(t < peak, np.exp(-((t - peak) / 0.17) ** 2), np.exp(-((t - peak) / 0.10) ** 2))
    centre = 380 * (2700 / 380) ** (env ** 0.65)
    air = shaped_noise(len(t), centre, np.full(len(t), 1.3), seed=11)
    body = shaped_noise(len(t), centre / 2.6, np.full(len(t), 1.6), seed=12)
    mono = (air + 0.45 * body) * env
    return pan(mono, -0.7 + 1.4 * smoothstep(t / t[-1])), peak, "peak"


def swoosh_in():
    """Something arriving: noise that rises in pitch and level to the moment it lands, then stops
    short, coming in from the right to the middle. For a card or a word sliding into place."""
    t = times(0.50)
    peak = 0.445
    env = np.where(t < peak, (t / peak) ** 2.6, np.exp(-(t - peak) / 0.018))
    centre = 300 * (4400 / 300) ** np.clip(t / peak, 0, 1) ** 1.4
    air = shaped_noise(len(t), centre, np.full(len(t), 1.0), seed=21)
    mono = air * env * attack(t, 0.01)
    return pan(mono, 0.6 * (1 - smoothstep(t / peak))), peak, "peak"


def pop():
    """A round, quick pop: a short tone that bends up as it starts, with a soft thump under it. For a
    card, a sticker or a word appearing."""
    t = times(0.14)
    f = 960 - (960 - 290) * np.exp(-t / 0.011)
    tone = glide(f) + 0.22 * glide(2 * f)
    thump = np.sin(2 * np.pi * 165 * t) * np.exp(-t / 0.016)
    mono = (tone * np.exp(-t / 0.030) + 0.55 * thump) * attack(t, 0.0008)
    mono += 0.18 * bright_burst(len(t), 0.0012, seed=31)
    return pan(mono, 0.0), 0.0, "attack"


def click():
    """A crisp press: a short ping with a woody body and a bright front. For a tap or a button."""
    t = times(0.06)
    ping = np.sin(2 * np.pi * 3150 * t) * np.exp(-t / 0.0024)
    body = np.sin(2 * np.pi * 1080 * t) * np.exp(-t / 0.0060)
    mono = (ping + 0.65 * body) * attack(t, 0.0003) + 0.5 * bright_burst(len(t), 0.0007, seed=41)
    return pan(np.tanh(1.4 * mono), 0.0), 0.0, "attack"


def tick():
    """A small, dry tick, higher and shorter than the click: a clock, a key, one step of a list."""
    t = times(0.05)
    ping = np.sin(2 * np.pi * 5300 * t) * np.exp(-t / 0.0014)
    wood = np.sin(2 * np.pi * 2050 * t) * np.exp(-t / 0.0032)
    mono = (ping + 0.5 * wood) * attack(t, 0.0002) + 0.35 * bright_burst(len(t), 0.0005, seed=51)
    return pan(mono, 0.0), 0.0, "attack"


def hit():
    """An impact with weight: a low boom that drops in pitch, a punch in the middle and a crack on
    top, with harmonics so it is still felt on a phone's small speaker. For a big word or a reveal."""
    t = times(0.60)
    boom = np.tanh(2.2 * glide(48 + (110 - 48) * np.exp(-t / 0.05)) * np.exp(-t / 0.13))
    body = np.tanh(1.6 * glide(120 + (210 - 120) * np.exp(-t / 0.03)) * np.exp(-t / 0.06))
    punch = shaped_noise(len(t), np.full(len(t), 900.0), np.full(len(t), 2.0), seed=61) * np.exp(-t / 0.035)
    crack = bright_burst(len(t), 0.0025, seed=62)
    tail = shaped_noise(len(t), np.full(len(t), 320.0), np.full(len(t), 2.5), seed=63) * np.exp(-t / 0.18)
    mono = (0.42 * boom + 0.8 * body + 3.2 * punch + 1.1 * crack + 0.06 * tail) * attack(t, 0.0012)
    mono *= np.clip((t[-1] - t) / 0.05, 0, 1)
    return pan(mono, 0.0), 0.0, "attack"


def riser():
    """Tension building into a moment: noise and a rising tone climbing three octaves together,
    widening as they go, cut off on the moment itself. For a reveal: it ends on the frame."""
    T = 1.8
    t = times(T)
    u = t / T
    env = u ** 2.4 * np.clip((T - t) / 0.010, 0, 1)
    centre = 260 * (7000 / 260) ** u
    width = 1.6 - 0.8 * u
    left = shaped_noise(len(t), centre, width, seed=71)
    right = shaped_noise(len(t), centre, width, seed=72)
    f0 = 110 * 8 ** (u ** 1.15)
    tone = sum(glide(k * f0) / k for k in range(1, 6)) / 2.3
    mid = 0.5 * (left + right)
    wide = smoothstep(u) * 0.6
    stereo = np.stack([(1 - wide) * mid + wide * left, (1 - wide) * mid + wide * right], axis=1)
    out = (stereo + 0.30 * (u ** 0.5)[:, None] * tone[:, None]) * env[:, None]
    peak = float(t[int(np.argmax(env))])
    return out, peak, "end"


def ding():
    """A clear bell: a few partials that ring and fade, the upper ones first, with a slow shimmer.
    For a tick on a list, a right answer or a number landing."""
    t = times(1.6)
    f0 = 1046.5
    partials = [(1.0, 1.0, 0.80), (1.0016, 0.30, 0.75), (2.0, 0.40, 0.42), (3.0, 0.17, 0.25),
                (4.16, 0.10, 0.16), (5.43, 0.06, 0.10)]
    mono = sum(a * np.sin(2 * np.pi * f0 * r * t) * np.exp(-t / tau) for r, a, tau in partials)
    mono = mono * attack(t, 0.0012) + 0.10 * bright_burst(len(t), 0.0008, seed=81)
    return pan(mono, 0.0), 0.0, "attack"


SOUNDS = {"whoosh": whoosh, "swoosh-in": swoosh_in, "pop": pop, "click": click, "tick": tick, "hit": hit,
          "riser": riser, "ding": ding}
FOR = {"whoosh": "a cut, a wipe, a fast move", "swoosh-in": "something sliding in and landing",
       "pop": "a card, sticker or word appearing", "click": "a tap or a button press",
       "tick": "a step in a list, a key, a clock", "hit": "a big word, a slam, a reveal",
       "riser": "building into a reveal (it ends on the moment)", "ding": "a tick on a list, a right answer"}


def level(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    """`x` with its loudest tenth of a second at LEVEL, unless that would take a sample over PEAK."""
    x = x / (np.abs(x).max() + 1e-12)
    gain_db = LEVEL - loudest(x.astype(np.float32))
    peak_db = 20 * np.log10(np.abs(x).max()) + gain_db
    if peak_db > PEAK:
        gain_db -= peak_db - PEAK
    y = x * 10 ** (gain_db / 20)
    return y, loudest(y.astype(np.float32)), 20 * np.log10(np.abs(y).max())


def write_wav(path: Path, x: np.ndarray, seed: int) -> None:
    rng = np.random.default_rng(seed)
    dither = (rng.random(x.shape) - rng.random(x.shape)) / 32768.0  # triangular, one bit
    pcm = np.clip(np.round((x + dither) * 32767), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())


def main() -> int:
    manifest = {}
    for k, (name, make) in enumerate(SOUNDS.items()):
        x, anchor, lands = make()
        y, loud, peak = level(np.asarray(x, np.float64))
        write_wav(HERE / f"{name}.wav", y, seed=900 + k)
        manifest[name] = {"file": f"{name}.wav", "seconds": round(len(y) / RATE, 4), "anchor_s": round(anchor, 4),
                          "lands": lands, "loudest_lufs": round(loud, 2), "peak_dbfs": round(peak, 2),
                          "for": FOR[name], "what": " ".join((make.__doc__ or "").split())}
        print(f"{name:10s} {len(y) / RATE:5.2f} s  lands at {anchor:5.3f} s ({lands})  loudest {loud:6.2f} LUFS  "
              f"peak {peak:6.2f} dBFS")
    (HERE / "sounds.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
