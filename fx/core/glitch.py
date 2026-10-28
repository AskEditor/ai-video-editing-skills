"""A broken signal drawn over a frame, and the crackle it makes. Shared by `transitions` (a glitch at
a join) and `glitch` (a glitch on a word or a cut).

    rng = np.random.default_rng(seed)
    out = glitch.tear(frame, other, e, rng)   # e: how broken, 0 (clean) to 1 (at its worst)
    x, peak = glitch.crackle(seconds, seed)   # its sound, and the second it is thickest

**The tear** is a burst of broken signal on one frame: bands of the picture torn sideways, some from
`other` (the other shot flashing through, or the same frame), a few with their colour channels
swapped, coarse blocks as when a picture loses its data, the red and blue pulled apart, and scan
lines. Everything scales with `e`; under 0.03 the frame comes back unchanged. The random draws are
taken from `rng` in a fixed order, so one seed always tears the same way.

`colour` other than "auto" tints what would be a channel swap or a colour split with that colour.

**The crackle** is short bursts of crushed noise and buzzing tones, stuttering, thickest at its
peak, each burst somewhere else across the stereo field. Made from noise in code, nothing recorded.
"""

from __future__ import annotations

import math

import numpy as np

from .text import colour as _colour
from .video import AUDIO_RATE

RATE = AUDIO_RATE


def rgb(hex_colour: str) -> np.ndarray:
    """A colour by name or hex as three floats from 0 to 1."""
    return np.array(_colour(hex_colour), np.float32) / 255.0


def tint(img: np.ndarray, hex_colour: str) -> np.ndarray:
    """`img` as its own brightness in one colour, a little lifted."""
    c = rgb(hex_colour)
    lum = img.astype(np.float32).mean(axis=2, keepdims=True) / 255.0
    return np.clip(lum * c * 255.0 * 1.3, 0, 255).astype(np.uint8)


def tear(base: np.ndarray, other: np.ndarray, e: float, rng: np.random.Generator,
         colour: str = "auto") -> np.ndarray:
    """`base` (uint8 RGB, changed in place and returned) torn by a broken signal of strength `e`,
    with bands of `other` flashing through."""
    import cv2

    h, w = base.shape[:2]
    if e < 0.03:
        return base
    out = base
    bands = 2 + int(round(9 * e))
    for _ in range(bands):
        bh = max(2, int(h * rng.uniform(0.01, 0.09)))
        y0 = int(rng.integers(0, max(1, h - bh)))
        src = other if rng.random() < 0.15 + 0.5 * e else out  # the other shot flashing through
        dx = int(round(rng.normal(0, 0.07 * w * e)))
        band = np.roll(src[y0:y0 + bh], dx, axis=1)
        if rng.random() < 0.18 * e:  # a band with its channels swapped
            band = band[..., ::-1] if colour == "auto" else tint(band, colour)
        out[y0:y0 + bh] = band
    for _ in range(int(round(3 * e))):  # coarse blocks, as when a picture loses its data
        bw, bh = int(w * rng.uniform(0.15, 0.45)), int(h * rng.uniform(0.04, 0.12))
        x0, y0 = int(rng.integers(0, w - bw)), int(rng.integers(0, h - bh))
        cell = max(4, int(w * 0.03))
        part = out[y0:y0 + bh, x0:x0 + bw]
        small = cv2.resize(part, (max(1, bw // cell), max(1, bh // cell)), interpolation=cv2.INTER_AREA)
        out[y0:y0 + bh, x0:x0 + bw] = cv2.resize(small, (bw, bh), interpolation=cv2.INTER_NEAREST)
    split = int(round(0.02 * w * e * (1 + rng.random())))
    if split:
        if colour == "auto":
            out[..., 0] = np.roll(out[..., 0], split, axis=1)
            out[..., 2] = np.roll(out[..., 2], -split, axis=1)
        else:
            ghost = tint(np.roll(out, split, axis=1), colour)
            out = np.maximum(out, (ghost.astype(np.float32) * 0.6).astype(np.uint8))
    lines = np.ones((h, 1, 1), np.float32)
    lines[::3] = 1.0 - 0.35 * e
    out = np.clip(out.astype(np.float32) * lines * (1 + 0.12 * e * rng.uniform(-1, 1)), 0, 255).astype(np.uint8)
    return out


def _stereo(mono: np.ndarray, pan) -> np.ndarray:
    theta = (np.clip(pan, -1, 1) + 1) * np.pi / 4
    return (np.stack([mono * np.cos(theta), mono * np.sin(theta)], axis=1) * np.sqrt(2)).astype(np.float32)


def crackle(d: float, seed: int) -> tuple[np.ndarray, float]:
    """A digital crackle about `d` seconds long: (float32 (n, 2) at 48 kHz, the second it is thickest)."""
    rng = np.random.default_rng(seed)
    T = d + 0.18
    n = int(T * RATE)
    out = np.zeros((n, 2), np.float32)
    peak = 0.5 * d + 0.06
    t_now = 0.02
    while t_now < T - 0.02:
        closeness = math.exp(-((t_now - peak) / (0.45 * d + 0.02)) ** 2)
        length = rng.uniform(0.008, 0.035 + 0.03 * closeness)
        a, b = int(t_now * RATE), min(n, int((t_now + length) * RATE))
        m = b - a
        if m > 8 and rng.random() < 0.35 + 0.65 * closeness:
            hold = int(rng.integers(3, 24))
            noise = np.repeat(rng.uniform(-1, 1, m // hold + 1), hold)[:m]
            steps = rng.choice([4, 6, 8])
            crushed = np.round(noise * steps) / steps
            if rng.random() < 0.4:  # a buzzing tone instead
                f = rng.uniform(90, 1400)
                crushed = np.sign(np.sin(2 * np.pi * f * np.arange(m) / RATE)) * 0.7
            gate = np.ones(m)
            k = min(m // 2, int(0.002 * RATE))
            if k > 0:
                gate[:k] = np.linspace(0, 1, k)
                gate[-k:] = np.linspace(1, 0, k)
            level = (0.35 + 0.65 * closeness) * rng.uniform(0.5, 1.0)
            out[a:b] += _stereo((crushed * gate * level).astype(np.float32), rng.uniform(-0.8, 0.8))
        t_now += length + rng.uniform(0.0, 0.03 * (1.2 - closeness))
    return out, peak
