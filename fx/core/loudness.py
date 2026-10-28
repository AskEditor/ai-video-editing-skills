"""The loudness of a finished mix: measured and set with ffmpeg's `loudnorm`, in two passes.

The first pass measures the mix (integrated loudness, true peak, range). The second applies one
gain to reach the target, and is asked to stay linear: the same gain everywhere, so nothing in the
mix moves. `loudnorm` only stays linear when that gain would not push the true peak over the
ceiling; otherwise it quietly switches to a dynamic mode that rides the level up and down (it
pumps). So before the second pass the few peaks that would go over are brought down here, by a
true-peak limiter that looks ahead and lets go slowly, and the mix is measured again.

True peak is the height the waveform reaches between its samples, which is what a player's
converter and a platform's encoder see. It is estimated here at four times the sample rate.

    python -m fx.core.loudness <file>     # print the integrated loudness and the true peak
"""

from __future__ import annotations

import json
import subprocess
from typing import Optional
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .video import AUDIO_RATE, MediaError, need_ffmpeg

# The video's own sound is encoded (AAC), and so is every upload again, and a coded waveform can
# overshoot the one it was given: by 0.54 dB on the airport clip at 128 kbps, and by 1.1 dB on the
# hotel clip, a quiet voice lifted 12 dB and limited hard (28 October 2026). The mix starts this far
# under the ceiling a person asks for; a trial encode then says whether that was enough.
ENCODE_ROOM = 1.0
LRA = 11.0  # loudnorm's loudness-range target; it plays no part in the linear pass


@dataclass(frozen=True)
class Measure:
    i: float  # integrated loudness, LUFS
    tp: float  # true peak, dBTP
    lra: float  # loudness range, LU
    thresh: float  # the gate's threshold, LUFS
    offset: float  # loudnorm's own offset for a second pass

    def as_dict(self) -> dict:
        return {"lufs": round(self.i, 2), "true_peak_dbtp": round(self.tp, 2), "lra": round(self.lra, 2)}


def _json_tail(text: str) -> dict:
    try:
        return json.loads(text[text.rindex("{"):text.rindex("}") + 1])
    except ValueError:
        raise MediaError("ffmpeg's loudnorm did not report a measurement") from None


def _number(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):  # "-inf" for silence
        return -99.0


def _measure_args(src: list[str]) -> Measure:
    need_ffmpeg()
    run = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", *src, "-vn", "-af",
                          f"loudnorm=I=-14:TP=-1:LRA={LRA}:print_format=json", "-f", "null", "-"],
                         capture_output=True, text=True)
    if run.returncode != 0:
        raise MediaError("ffmpeg could not measure the loudness")
    d = _json_tail(run.stderr)
    return Measure(_number(d["input_i"]), _number(d["input_tp"]), _number(d["input_lra"]),
                   _number(d["input_thresh"]), _number(d["target_offset"]))


def _raw(samples: np.ndarray, folder: str, name: str) -> list[str]:
    path = Path(folder) / name
    np.ascontiguousarray(samples, np.float32).tofile(path)
    return ["-f", "f32le", "-ar", str(AUDIO_RATE), "-ac", str(samples.shape[1]), "-i", str(path)]


def measure(samples: np.ndarray) -> Measure:
    """The first pass: loudnorm's measurement of 48 kHz float samples (n, channels)."""
    with tempfile.TemporaryDirectory(prefix="fx-loud-") as tmp:
        return _measure_args(_raw(samples, tmp, "mix.f32"))


def measure_file(path: str | Path) -> Measure:
    """loudnorm's measurement of a file's sound: what a finished video is checked with."""
    return _measure_args(["-i", str(path)])


# ---- true peak and the limiter -------------------------------------------------------------------

def _oversampler(factor: int = 4, taps: int = 12) -> np.ndarray:
    """A windowed-sinc interpolator: `factor` phases of `taps` taps each."""
    n = factor * taps
    m = np.arange(n) - (n - 1) / 2
    h = np.sinc(m / factor) * np.kaiser(n, 8.0)
    return (h.reshape(taps, factor).T * factor / h.sum()).astype(np.float64)


def peak_envelope(samples: np.ndarray, factor: int = 4) -> np.ndarray:
    """For each sample, the highest the waveform reaches between it and the next, on any channel."""
    x = samples if samples.ndim == 2 else samples[:, None]
    phases = _oversampler(factor)
    taps = phases.shape[1]
    env = np.abs(x).max(axis=1).astype(np.float64)
    for ch in range(x.shape[1]):
        col = x[:, ch].astype(np.float64)
        for p in phases:
            y = np.convolve(col, p[::-1], mode="full")[taps // 2:taps // 2 + len(col)]
            np.maximum(env, np.abs(y), out=env)
    return env


def true_peak_db(samples: np.ndarray) -> float:
    return 20.0 * float(np.log10(max(1e-12, float(peak_envelope(samples).max()))))


def limit(samples: np.ndarray, ceiling_db: float, attack: float = 0.004, release: float = 0.12,
          rate: int = AUDIO_RATE) -> tuple[np.ndarray, float]:
    """Bring every true peak under `ceiling_db` with a smooth gain: it starts going down `attack`
    seconds before a peak and comes back at a steady rate over about `release` seconds per 6 dB,
    so a peak is turned down, not chopped. The result is measured again, and a peak the waveform
    still makes between its samples is taken down by a second pass. Returns the samples and the
    most it turned down (dB)."""
    out, most = samples, 0.0
    for _ in range(3):
        out, cut = _limit_once(out, ceiling_db, attack, release, rate)
        most += cut
        if cut <= 0 or true_peak_db(out) <= ceiling_db + 0.01:
            break
    return out, most


def _limit_once(samples: np.ndarray, ceiling_db: float, attack: float, release: float,
                rate: int) -> tuple[np.ndarray, float]:
    env = peak_envelope(samples)
    need = np.maximum(0.0, 20.0 * np.log10(np.maximum(env, 1e-12)) - ceiling_db)  # dB to take off
    if need.max() <= 0:
        return samples, 0.0
    k = np.arange(len(need), dtype=np.float64)
    down = 6.0 / max(1.0, release * rate)  # dB per sample on the way back
    up = need.max() / max(1.0, attack * rate) + 1e-9  # dB per sample on the way in
    after = np.maximum.accumulate(need + down * k) - down * k  # each peak's reduction, let go slowly
    before = (np.maximum.accumulate((need - up * k)[::-1])[::-1]) + up * k  # and reached early
    cut = np.maximum(np.maximum(after, before), 0.0)
    w = max(1, int(0.001 * rate))  # round the corners off over a millisecond, never under the need
    cut = np.maximum(np.convolve(cut, np.ones(w) / w, mode="same"), need)
    g = (10.0 ** (-cut / 20.0)).astype(np.float32)
    out = samples * (g[:, None] if samples.ndim == 2 else g)
    return out.astype(np.float32), float(cut.max())


# ---- the two passes ------------------------------------------------------------------------------

def encoded_peak(samples: np.ndarray, kbps: int = 128) -> float:
    """The true peak the sound will have once encoded as AAC at `kbps` (a trial encode, measured by
    loudnorm): the lowest rate the pack writes, where the coded waveform overshoots the most. It is
    encoded exactly as `video.FrameWriter` encodes it: in bit-exact mode the encoder overshoots more
    (0.94 against 1.74 dB under full scale for the same mix, 28 October 2026)."""
    with tempfile.TemporaryDirectory(prefix="fx-loud-") as tmp:
        src = _raw(samples, tmp, "mix.f32")
        coded = Path(tmp) / "trial.mp4"
        run = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", *src, "-c:a", "aac", "-b:a", f"{kbps}k",
                              "-movflags", "+faststart", "-map_metadata", "-1", "-fflags", "+bitexact",
                              "-flags:a", "+bitexact", str(coded)], capture_output=True, text=True)
        if run.returncode != 0:
            raise MediaError("ffmpeg could not make a trial encode of the sound")
        return measure_file(coded).tp


def normalize(samples: np.ndarray, target_i: float = -14.0, target_tp: float = -1.0) -> tuple[np.ndarray, dict]:
    """The mix at `target_i` LUFS with its true peak under `target_tp` dBTP, through loudnorm's two
    passes. The result is encoded on trial; when the encoded waveform would still go over (a quiet
    voice lifted a long way and limited hard overshoots most), the ceiling comes down by the excess
    and the passes run again, so the video's own sound is encoded once, already under. Same length,
    same timing."""
    first = measure(samples)
    if first.i < -70:
        return samples, {"skipped": "the sound is silent", "first": first.as_dict()}
    ceiling = target_tp - ENCODE_ROOM
    for _ in range(3):
        y, report = _two_passes(samples, first, target_i, ceiling)
        coded = encoded_peak(y)
        report.update(target_lufs=target_i, target_true_peak=target_tp, ceiling=round(ceiling, 2),
                      encoded_true_peak=round(coded, 2))
        if coded <= target_tp - 0.1:
            break
        ceiling -= coded - (target_tp - 0.1) + 0.1
    return y, report


_SOXR: Optional[bool] = None


def _resampler() -> str:
    """soxr where this ffmpeg has it (the better resampler), else ffmpeg's own: many builds, Homebrew's
    ffmpeg 9 among them, ship without soxr, and asking for it fails the whole loudness pass."""
    global _SOXR
    if _SOXR is None:
        run = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                              "anullsrc=d=0.1", "-af", "aresample=48000:resampler=soxr", "-f", "null", "-"],
                             capture_output=True, text=True)
        _SOXR = run.returncode == 0
    return ":resampler=soxr" if _SOXR else ""


def _two_passes(samples: np.ndarray, first: Measure, target_i: float, ceiling: float) -> tuple[np.ndarray, dict]:
    x, limited, slack = samples, 0.0, 0.0
    m = first
    for _ in range(4):  # limit, measure again (limiting lowers the loudness a little), repeat
        room = ceiling - 0.2 - (target_i - m.i)  # the peak the mix may have before its gain
        if m.tp <= room:
            break
        x, cut = limit(x, room - slack)
        limited = max(limited, cut)
        m = measure(x)
        # loudnorm finds peaks at a finer rate than the estimate here; aim under by the difference
        slack = max(slack, m.tp - true_peak_db(x) + 0.05)
    af = (f"loudnorm=I={target_i}:TP={ceiling}:LRA={LRA}:measured_I={m.i}:measured_TP={m.tp}:"
          f"measured_LRA={m.lra}:measured_thresh={m.thresh}:offset={m.offset}:linear=true:print_format=json,"
          f"aresample={AUDIO_RATE}{_resampler()}")
    with tempfile.TemporaryDirectory(prefix="fx-loud-") as tmp:
        src = _raw(x, tmp, "mix.f32")
        out = Path(tmp) / "out.f32"
        run = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-y", *src, "-af", af, "-f", "f32le",
                              "-ar", str(AUDIO_RATE), "-ac", str(x.shape[1]), str(out)], capture_output=True, text=True)
        if run.returncode != 0:
            raise MediaError("ffmpeg could not set the loudness")
        second = _json_tail(run.stderr)
        y = np.fromfile(out, np.float32).reshape(-1, x.shape[1])
    if len(y) < len(x):
        y = np.concatenate([y, np.zeros((len(x) - len(y), x.shape[1]), np.float32)])
    y = np.ascontiguousarray(y[:len(x)])
    report = {"before": first.as_dict(), "limited_db": round(limited, 2),
              "second_pass": second.get("normalization_type", "?"),
              "after": {"lufs": round(_number(second.get("output_i")), 2),
                        "true_peak_dbtp": round(_number(second.get("output_tp")), 2)}}
    return y, report


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(__doc__)
        return 0
    for p in args:
        m = measure_file(p)
        print(f"{p}: {m.i:.2f} LUFS integrated, true peak {m.tp:.2f} dBTP, range {m.lra:.1f} LU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
