"""Sounds made in code at the moment an effect needs them, shaped to its own timing.

The library in `sounds/` holds fixed one-shots (a whoosh, a pop). Some sounds cannot be fixed files:
a neon tube's hum has to switch on and off with the flicker of its own letters, and a sign that
types out ticks once per letter. Those are made here, from noise and sine waves, every time the same
for the same timing (fixed seeds), with no recording or sample from anywhere.

- `clunk()`: a heavy light switch thrown, the lights going out: a lever's first click, the contact
  slamming home with a low knock in the box behind it, a short metal ring and a rattle.
- `scratch()`: a record dragged back under the needle and stopped: music played at a speed that
  falls through nothing to backwards and then dies, with the groove's noise growing as it is dragged.
- `tube(lit)`: a neon sign's sound from how lit it is at each sample (0 to 1): the mains buzz of its
  transformer and the hiss of the gas while it is on, and apart from it a crackle wherever a tube
  strikes on.
- `tick()`: one letter of a dot-matrix sign: a small, dry electronic tick.
- `shimmer()`: light arriving: a soft, high, golden chord with a sprinkle of tiny pings.
- `slap(weight)`: a sticker slapped onto glass or a rubber stamp brought down: a flat smack of
  air and skin, a low thud under it (heavier with `weight`), and a tiny paper rustle after.
- `paper()`: a card or a cutting laid down: a soft slide of paper over a desk that ends on a light tap.
- `beep()`: a camcorder starting to record: two short, clean high tones, the second a step higher.
- `whirr(seconds)`: a tape deck running, as a bed: the motor's low hum wavering with its flutter, the
  heads' soft hiss, and a light mechanical clunk at the start as it begins to play.
- `projector(seconds)`: a film projector running, as a bed: the shutter's quick soft clatter, the
  motor's hum and a faint, uneven rattle of the reel.
- `printer(seconds)`: a small receipt printer printing one line: the print head's fast, dry chatter
  over a little motor whine as the paper feeds, ending on the soft stop of the feed.

The one-shots return float32 (n, 2) at 48 kHz and the second the sound lands at (its anchor).
`at_level()` sets a sound's loudest tenth of a second against a loudness, as the library's sounds
are set (`sound.loudest`).
"""

from __future__ import annotations

import numpy as np

from .sound import gain, loudest

RATE = 48000


def times(seconds: float) -> np.ndarray:
    return np.arange(int(round(seconds * RATE))) / RATE


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def band(x: np.ndarray, lo: float, hi: float, soft: float = 0.3) -> np.ndarray:
    """`x` (mono) kept between `lo` and `hi` Hz, with edges `soft` octaves wide (no ringing)."""
    n = len(x)
    if n == 0:
        return x
    spec = np.fft.rfft(x)
    f = np.maximum(np.fft.rfftfreq(n, 1.0 / RATE), 1.0)
    lf = np.log2(f)
    up = np.clip((lf - (np.log2(lo) - soft / 2)) / soft, 0, 1)
    down = np.clip(((np.log2(hi) + soft / 2) - lf) / soft, 0, 1)
    mask = (up * up * (3 - 2 * up)) * (down * down * (3 - 2 * down))
    return np.fft.irfft(spec * mask, n)


def snap(n: int, decay: float, seed: int) -> np.ndarray:
    """A very short, bright burst (the tick at the front of anything struck), `decay` seconds long."""
    t = np.arange(n) / RATE
    x = np.diff(_rng(seed).standard_normal(n + 1))
    return x * np.exp(-t / max(decay, 1e-5)) * 0.5


def ring(t: np.ndarray, hz: float, decay: float, end_hz: float | None = None, glide: float = 0.03) -> np.ndarray:
    """A sine that dies away over `decay` seconds, bending from `hz` to `end_hz` over `glide`."""
    f = np.full(len(t), float(hz)) if end_hz is None else end_hz + (hz - end_hz) * np.exp(-t / glide)
    return np.sin(2 * np.pi * np.cumsum(f) / RATE) * np.exp(-t / decay)


def _at(out: np.ndarray, x: np.ndarray, start: float) -> None:
    a = int(round(start * RATE))
    b = min(len(out), a + len(x))
    if b > a:
        out[a:b] += x[: b - a]


def _stereo(mono: np.ndarray, width: float = 0.0, seed: int = 0) -> np.ndarray:
    """Centre, or a touch wide: each side a hair apart in time (never more than half a millisecond)."""
    if width <= 0:
        return np.stack([mono, mono], axis=1).astype(np.float32)
    d = int(round(width * 0.0005 * RATE))
    left = mono
    right = np.concatenate([np.zeros(d), mono[: len(mono) - d]]) if d else mono
    return np.stack([left, right], axis=1).astype(np.float32)


def _peak(x: np.ndarray, to: float = 0.89) -> np.ndarray:
    return x * (to / max(1e-9, float(np.abs(x).max())))


# ---- the sounds ------------------------------------------------------------------------------------

def clunk(seed: int = 11) -> tuple[np.ndarray, float]:
    """A heavy switch thrown: the lever's first click, then the contact slamming home 28 ms later
    (the anchor: the moment the lights go), with a low knock, a short ring of the metal box and a
    last small rattle."""
    t = times(0.42)
    out = np.zeros(len(t))
    hit = 0.028
    _at(out, 0.30 * snap(len(t), 0.0011, seed), 0.0)
    u = t[: len(t) - int(hit * RATE)]
    knock = 0.95 * ring(u, 210.0, 0.034, end_hz=150.0, glide=0.02) + 0.55 * ring(u, 72.0, 0.075)
    metal = (0.22 * ring(u, 1380.0, 0.045) + 0.15 * ring(u, 2930.0, 0.03) + 0.10 * ring(u, 4470.0, 0.022)
             + 0.06 * ring(u, 6120.0, 0.016))
    grit = band(_rng(seed + 1).standard_normal(len(u)), 500.0, 3200.0) * np.exp(-u / 0.018) * 0.55
    _at(out, np.tanh(1.6 * (knock + metal + grit)) + snap(len(u), 0.0018, seed + 2), hit)
    _at(out, 0.12 * snap(len(t), 0.0009, seed + 3) + 0.05 * ring(t, 1650.0, 0.02), hit + 0.047)
    out *= np.clip((t[-1] - t) / 0.04, 0, 1)
    return _stereo(_peak(out), width=0.4), hit


def _music(seconds: float, seed: int) -> np.ndarray:
    """A few seconds of something that sounds like a record playing: a chord with a beat under it and
    the groove's hiss, rich in overtones so a change of speed is heard as a change of pitch."""
    t = times(seconds)
    chord = np.zeros(len(t))
    for hz, amp in ((130.8, 1.0), (155.6, 0.8), (196.0, 0.8), (261.6, 0.55), (311.1, 0.4)):
        for k in range(1, 16):
            if hz * k > 9000:
                break
            chord += amp * np.sin(2 * np.pi * hz * k * t + 0.7 * k) / k
    pulse = np.zeros(len(t))
    for beat in np.arange(0.0, seconds, 0.25):
        u = t - beat
        on = u >= 0
        pulse[on] += np.sin(2 * np.pi * 55 * u[on] * (1 + 2.5 * np.exp(-u[on] / 0.02))) * np.exp(-u[on] / 0.09)
    hiss = band(_rng(seed).standard_normal(len(t)), 2500.0, 11000.0) * 0.18
    return 0.35 * chord / np.abs(chord).max() + 0.6 * pulse + hiss


def scratch(seed: int = 23) -> tuple[np.ndarray, float]:
    """A record dragged back and stopped: the music slows from its own speed through nothing to
    backwards and dies, the groove's noise rising with how hard it is dragged. It lands (its anchor)
    on its first instant: the moment the record is caught."""
    src = _music(1.6, seed)
    t = times(0.46)
    # The record's speed: caught (1 to 0 in 40 ms), dragged back hard (to -3.2), let go to a stop.
    v = np.interp(t, [0.0, 0.04, 0.075, 0.13, 0.22, 0.34, 0.46], [1.0, 0.0, -3.2, -2.6, -0.8, -0.05, 0.0])
    v = np.convolve(v, np.ones(96) / 96, mode="same")
    pos = 1.1 * RATE + np.cumsum(v)
    i0 = np.clip(np.floor(pos).astype(int), 0, len(src) - 2)
    frac = pos - np.floor(pos)
    played = src[i0] * (1 - frac) + src[i0 + 1] * frac
    speed = np.abs(v)
    groove = band(_rng(seed + 1).standard_normal(len(t)), 900.0, 7000.0) * np.clip(speed / 3.0, 0, 1) ** 1.5
    out = played * np.clip(speed, 0, 1.6) ** 0.8 + 0.55 * groove
    out *= np.clip((t[-1] - t) / 0.03, 0, 1)
    return _stereo(_peak(np.tanh(1.3 * out)), width=0.6), 0.0


def tube(lit: np.ndarray, strikes: list[int] | None = None, mains: float = 120.0,
         seed: int = 31) -> tuple[np.ndarray, np.ndarray]:
    """A neon sign's sound, sample by sample, from `lit` (0 to 1 at 48 kHz: how lit the sign is):
    its hum, the transformer's buzz (the mains frequency and its overtones, slightly rough) and the
    gas's hiss while it is on; and apart from it, a crackle at each sample in `strikes` (where a
    tube strikes on; by default wherever `lit` rises through a half). Both mono, so each can be set
    at its own level."""
    n = len(lit)
    t = np.arange(n) / RATE
    rng = _rng(seed)
    wobble = 1.0 + 0.002 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * mains * np.cumsum(wobble) / RATE
    buzz = np.zeros(n)
    for k in range(1, 22):
        amp = (1.0 / k ** 1.4) * (1.5 if k % 2 == 0 else 1.0)
        buzz += amp * np.sin(k * phase + 0.4 * k)
    buzz = np.tanh(1.4 * buzz / max(1e-9, float(np.abs(buzz).max())))
    hiss = band(rng.standard_normal(n), 3000.0, 12000.0) * 0.05
    level = np.convolve(np.clip(lit, 0, 1), np.ones(48) / 48, mode="same")  # 1 ms: no clicks
    hum = (0.5 * buzz + hiss) * level
    if strikes is None:
        strikes = list(np.flatnonzero(np.diff(np.concatenate([[0.0], (lit > 0.5).astype(float)])) > 0))
    crackle = np.zeros(n)
    for k, i in enumerate(strikes):
        m = int(RATE * rng.uniform(0.012, 0.03))
        u = np.arange(m) / RATE
        fizz = band(rng.standard_normal(m), 1800.0, 9000.0) * np.exp(-u / (m / RATE / 3))
        fizz = 0.5 * fizz / max(1e-9, float(np.abs(fizz).max()))
        piece = fizz + 0.7 * snap(m, 0.0012, seed + 7 * k + 1)
        crackle[i:i + m] += piece[: max(0, min(m, n - i))]
    return hum.astype(np.float32), (0.9 * np.tanh(crackle / 0.9)).astype(np.float32)


def tick(seed: int = 41) -> tuple[np.ndarray, float]:
    """One letter of a dot-matrix sign typing: a small, dry tick with a faint electronic blip."""
    t = times(0.05)
    x = 0.8 * ring(t, 3600.0, 0.0018) + 0.4 * ring(t, 1250.0, 0.004) + 0.3 * snap(len(t), 0.0006, seed)
    return _stereo(_peak(x, 0.7)), 0.0


def shimmer(seed: int = 53) -> tuple[np.ndarray, float]:
    """Light arriving: a soft high chord that swells in over a tenth of a second and rings away, with
    tiny pings scattered through it. It lands (its anchor) at the top of its swell."""
    t = times(1.8)
    rng = _rng(seed)
    swell = np.clip(t / 0.11, 0, 1) ** 1.5
    x = np.zeros(len(t))
    for hz, amp, decay in ((1567.98, 0.7, 0.9), (1975.53, 0.55, 0.8), (2349.32, 0.5, 0.7), (3135.96, 0.35, 0.5),
                           (3951.07, 0.22, 0.4), (4698.64, 0.15, 0.3)):
        shake = 1.0 + 0.003 * np.sin(2 * np.pi * rng.uniform(4, 7) * t)
        x += amp * np.sin(2 * np.pi * hz * np.cumsum(shake) / RATE) * np.exp(-t / decay)
    x *= swell
    for _ in range(22):
        at = rng.uniform(0.03, 1.0)
        u = t[: int(0.12 * RATE)]
        _at(x, 0.10 * ring(u, rng.uniform(5200, 9800), rng.uniform(0.01, 0.03)), at)
    x *= np.clip((t[-1] - t) / 0.2, 0, 1)
    return _stereo(_peak(x), width=0.8), 0.11


def slap(weight: float = 0.5, seed: int = 61) -> tuple[np.ndarray, float]:
    """A sticker slapped on, or a rubber stamp brought down (`weight` 0 to 1: a light sticker to a
    heavy stamp): a flat smack of noise in the middle of the range, a low thud whose pitch falls,
    and a tiny rustle of paper after it. It lands (its anchor) on the smack, a hair after the start."""
    t = times(0.32)
    rng = _rng(seed)
    hit = 0.004
    out = np.zeros(len(t))
    u = t[: len(t) - int(hit * RATE)]
    smack = band(rng.standard_normal(len(u)), 700.0, 5200.0) * np.exp(-u / (0.010 + 0.006 * weight))
    thud = (0.55 + 0.6 * weight) * ring(u, 150.0 - 50.0 * weight, 0.045 + 0.04 * weight, end_hz=70.0, glide=0.025)
    _at(out, np.tanh(2.2 * smack + thud) + 0.4 * snap(len(u), 0.0009, seed + 1), hit)
    rustle = band(rng.standard_normal(len(t)), 2500.0, 9000.0) * np.exp(-((t - 0.05) / 0.03) ** 2) * 0.10
    out += rustle
    out[: int(hit * RATE)] *= np.linspace(0, 1, int(hit * RATE))
    out *= np.clip((t[-1] - t) / 0.05, 0, 1)
    return _stereo(_peak(out), width=0.3), hit


def paper(seed: int = 67) -> tuple[np.ndarray, float]:
    """A card or a cutting laid down: paper sliding over a desk for a fifth of a second, its hiss
    rising and brightening, ending on a light tap (the anchor) as it comes to rest."""
    t = times(0.42)
    rng = _rng(seed)
    slide_end = 0.22
    grow = np.clip(t / slide_end, 0, 1) ** 1.6 * (t < slide_end)
    hiss = band(rng.standard_normal(len(t)), 1800.0, 7000.0) * grow * 0.55
    grain = band(rng.standard_normal(len(t)), 300.0, 1200.0) * grow * 0.25
    out = hiss + grain
    u = t[: len(t) - int(slide_end * RATE)]
    tap = 0.7 * ring(u, 420.0, 0.012, end_hz=260.0, glide=0.01) + 0.45 * snap(len(u), 0.0012, seed + 1)
    tap += band(rng.standard_normal(len(u)), 1500.0, 6000.0) * np.exp(-u / 0.02) * 0.25
    _at(out, tap, slide_end)
    out *= np.clip((t[-1] - t) / 0.05, 0, 1)
    return _stereo(_peak(out, 0.8), width=0.5), slide_end


def beep(seed: int = 71) -> tuple[np.ndarray, float]:
    """A camcorder starting to record: two short high tones (the anchor is the first one's start)."""
    t = times(0.26)
    out = np.zeros(len(t))
    for start, hz in ((0.0, 2093.0), (0.11, 2637.0)):
        u = times(0.075)
        env = np.clip(u / 0.004, 0, 1) * np.clip((u[-1] - u) / 0.008, 0, 1)
        tone = np.sin(2 * np.pi * hz * u) + 0.12 * np.sin(2 * np.pi * 2 * hz * u)
        _at(out, tone * env, start)
    return _stereo(_peak(out, 0.7)), 0.0


def whirr(seconds: float, seed: int = 73) -> np.ndarray:
    """A tape deck running, `seconds` long: a wavering motor hum, the heads' hiss and a soft clunk as
    it starts to play. A bed: it starts and ends with short fades, and is meant to sit far under a voice."""
    t = times(seconds)
    rng = _rng(seed)
    flutter = 1.0 + 0.012 * np.sin(2 * np.pi * 5.7 * t) + 0.006 * np.sin(2 * np.pi * 0.9 * t + 1.3)
    phase = 2 * np.pi * np.cumsum(np.full(len(t), 100.0) * flutter) / RATE
    hum = 0.5 * np.sin(phase) + 0.25 * np.sin(2 * phase + 0.4) + 0.12 * np.sin(3 * phase + 1.1)
    hiss = band(rng.standard_normal(len(t)), 3000.0, 12000.0) * 0.35
    rumble = band(rng.standard_normal(len(t)), 40.0, 220.0) * 0.6 * (0.8 + 0.2 * np.sin(2 * np.pi * 1.4 * t))
    out = 0.35 * hum + hiss + rumble
    knock, _ = clunk(seed + 1)
    k = knock[:, 0] * 0.6
    out[: min(len(out), len(k))] += k[: min(len(out), len(k))]
    fade = np.clip(t / 0.15, 0, 1) * np.clip((t[-1] - t) / 0.3, 0, 1) if len(t) else t
    return _stereo(_peak(out * fade, 0.6), width=0.6)


def projector(seconds: float, seed: int = 79) -> np.ndarray:
    """A film projector running, `seconds` long: the shutter's soft clatter 24 times a second, the
    motor's hum and a faint uneven rattle of the reel. A bed, faded in and out."""
    t = times(seconds)
    rng = _rng(seed)
    out = np.zeros(len(t))
    click = band(rng.standard_normal(int(0.012 * RATE)), 600.0, 5000.0) * np.exp(-np.arange(int(0.012 * RATE)) / (0.0025 * RATE))
    step = 1.0 / 24.0
    for k, at in enumerate(np.arange(0.0, seconds, step)):
        at = at + 0.0015 * rng.standard_normal()
        _at(out, click * (0.55 + 0.25 * rng.random()) * (1.0 if k % 2 else 0.7), max(0.0, at))
    hum = 0.20 * np.sin(2 * np.pi * 50.0 * t) + 0.08 * np.sin(2 * np.pi * 150.0 * t)
    rattle = band(rng.standard_normal(len(t)), 150.0, 900.0) * 0.18 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.8 * t) ** 2)
    out = out + hum + rattle
    fade = np.clip(t / 0.25, 0, 1) * np.clip((t[-1] - t) / 0.3, 0, 1) if len(t) else t
    return _stereo(_peak(out * fade, 0.6), width=0.5)


def printer(seconds: float = 0.22, seed: int = 71) -> tuple[np.ndarray, float]:
    """One line of a receipt printer: the head's chatter (dry clicks at about 190 a second, each a
    little different, rising and falling in strength as the line crosses the paper) over a faint
    feed-motor whine, then the feed's soft stop. It lands (its anchor) where the line is done."""
    rng = _rng(seed)
    T = seconds + 0.09
    t = times(T)
    out = np.zeros(len(t))
    rate = rng.uniform(175.0, 205.0)
    k = 0
    while k / rate < seconds:
        at = k / rate + rng.uniform(-0.0006, 0.0006)
        q = (k / rate) / max(1e-6, seconds)
        level = (0.55 + 0.45 * np.sin(np.pi * min(1.0, q))) * rng.uniform(0.7, 1.0)
        u = t[: int(0.006 * RATE)]
        click = 0.7 * ring(u, rng.uniform(3000.0, 3800.0), 0.0007) + 0.5 * snap(len(u), 0.0005, seed + k)
        _at(out, level * click, max(0.0, at))
        k += 1
    feed = t < seconds + 0.02
    whine = (0.05 * np.sin(2 * np.pi * 930.0 * t) + 0.03 * np.sin(2 * np.pi * 1860.0 * t)) * feed
    whine *= np.clip(t / 0.015, 0, 1) * np.clip((seconds + 0.02 - t) / 0.015, 0, 1)
    out += whine + band(rng.standard_normal(len(t)), 2000.0, 7000.0) * 0.04 * feed
    u = t[: len(t) - int(seconds * RATE)]
    _at(out, 0.35 * ring(u, 310.0, 0.010, end_hz=220.0, glide=0.008) + 0.2 * snap(len(u), 0.0008, seed + 999), seconds)
    out *= np.clip((t[-1] - t) / 0.03, 0, 1)
    return _stereo(_peak(out, 0.8), width=0.3), seconds


def at_level(x: np.ndarray, lufs: float) -> np.ndarray:
    """`x` set so its loudest tenth of a second measures `lufs`."""
    return (x * gain(lufs - loudest(np.asarray(x, np.float32)))).astype(np.float32)
