"""The sound effects' parts, checked on sounds made here (no clip needed).

    .venv/bin/python tests/sound.py

- loudness: a 1 kHz tone at -20 dBFS in both channels measures -20 LUFS (the standard's own check),
  and ffmpeg's loudnorm agrees within 0.3 LU;
- the limiter: after it, no true peak is over its ceiling;
- landing: a sound laid at a time has its landing sample on that time, to the sample;
- the beat: a made-up track at 96 beats a minute is found at 96, its bar lines on its downbeats;
- the fit: for many film lengths and every way of fitting, the pieces add up to the film to the
  sample, stay inside the track, and an end-aligned fit ends on the track's own end;
- the duck: one level through each stretch of talk, back up in a long pause, never up in a short one;
- the library: every sound present, levelled, under the peak, landing inside itself;
- the plan readers: a clean-cut plan's cuts and a tier list's cards where they are.

Exits 1 on any failure.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from fx.core import loudness, sound  # noqa: E402
from fx.core.words import Word  # noqa: E402
from fx.effects import music_bed as M  # noqa: E402
from fx.effects import sound_effects as S  # noqa: E402

RATE = 48000
FAILED: list[str] = []


def check(ok: bool, what: str) -> None:
    print(("ok    " if ok else "FAIL  ") + what)
    if not ok:
        FAILED.append(what)


def tone(seconds: float, hz: float, dbfs: float) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    x = (10 ** (dbfs / 20) * np.sin(2 * np.pi * hz * t)).astype(np.float32)
    return np.stack([x, x], axis=1)


def made_up_track(seconds: float, bpm: float = 96.0, ending: bool = True) -> np.ndarray:
    """A kick on every beat (louder on the first of four), a hat between, and a chord that changes each
    bar; with `ending`, a last chord that rings out for two seconds."""
    rng = np.random.default_rng(5)
    n = int(seconds * RATE)
    x = np.zeros(n)
    beat = 60.0 / bpm
    chords = [(220.0, 277.2, 329.6), (196.0, 246.9, 293.7), (174.6, 220.0, 261.6), (196.0, 246.9, 311.1)]
    k = 0
    while k * beat < seconds - (2.0 if ending else 0.0):
        i = int(k * beat * RATE)
        t = np.arange(min(int(0.25 * RATE), n - i)) / RATE
        kick = np.sin(2 * np.pi * (50 + 90 * np.exp(-t / 0.03)) * t) * np.exp(-t / 0.09) * (1.0 if k % 4 == 0 else 0.6)
        x[i:i + len(t)] += kick
        j = int((k + 0.5) * beat * RATE)
        if j < n:
            h = rng.standard_normal(min(int(0.03 * RATE), n - j)) * np.exp(-np.arange(min(int(0.03 * RATE), n - j)) / 400)
            x[j:j + len(h)] += 0.15 * np.diff(np.concatenate([[0], h]))
        if k % 4 == 0:
            bar_len = min(int(4 * beat * RATE), n - i)
            tb = np.arange(bar_len) / RATE
            ch = chords[(k // 4) % len(chords)]
            x[i:i + bar_len] += 0.08 * sum(np.sin(2 * np.pi * f * tb) for f in ch) * np.minimum(1, tb / 0.02)
        k += 1
    if ending:
        i = int(k * beat * RATE)
        t = np.arange(n - i) / RATE
        x[i:] += 0.3 * sum(np.sin(2 * np.pi * f * t) for f in chords[0]) * np.exp(-t / 0.35)
    x = x / np.abs(x).max() * 0.5
    return np.stack([x, x], axis=1).astype(np.float32)


def track_from(samples: np.ndarray, tmp: Path) -> M.Track:
    import wave

    path = tmp / "track.wav"
    pcm = np.clip(np.round(samples * 32767), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())
    return M.analyse(str(path))


def main() -> int:
    import tempfile

    # ---- loudness
    x = tone(5.0, 1000.0, -20.0)
    mine = sound.integrated(sound.blocks(x))
    check(abs(mine + 20.0) < 0.1, f"a -20 dBFS 1 kHz tone in both channels measures -20 LUFS here ({mine:.2f})")
    theirs = loudness.measure(x).i
    check(abs(theirs - mine) < 0.3, f"ffmpeg's loudnorm agrees ({theirs:.2f})")

    # ---- the limiter
    rng = np.random.default_rng(1)
    loud = (rng.standard_normal((RATE * 3, 2)) * 0.25).astype(np.float32)
    limited, cut = loudness.limit(loud, -6.0)
    tp = loudness.true_peak_db(limited)
    check(tp <= -5.95, f"after the limiter the true peak is under its ceiling of -6 dBTP ({tp:.2f}, {cut:.1f} dB taken)")
    check(len(limited) == len(loud), "the limiter keeps the length")

    # ---- landing to the sample
    out = np.zeros((RATE * 2, 2), np.float32)
    blip = np.zeros((1000, 2), np.float32)
    blip[300] = 1.0
    sound.lay(out, blip, 1.2345, RATE, anchor=300 / RATE)
    check(int(np.argmax(out[:, 0])) == int(round(1.2345 * RATE)), "a sound's landing sample falls on its time")
    out = np.zeros((RATE, 2), np.float32)
    a, b = sound.lay(out, blip, 0.001, RATE, anchor=300 / RATE)
    check(a == 0 and int(np.argmax(out[:, 0])) == int(round(0.001 * RATE)),
          "a sound that would start before the video is cut, its landing still on time")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # ---- the beat and the bars
        tr = track_from(made_up_track(24.0, 96.0, ending=True), tmp)
        g = tr.grid
        check(abs(g.bpm - 96.0) < 1.0, f"a 96 BPM track is found at {g.bpm:.1f} BPM")
        bar = 4 * 60 / 96.0
        off = [min(abs(b - k * bar) for k in range(12)) for b in g.bars]
        check(max(off) < 0.03, f"its bar lines are on its downbeats (furthest {max(off) * 1000:.0f} ms)")
        check(tr.ends_itself, "a track whose last chord rings out is found to end by itself")
        cut_off = track_from(made_up_track(24.0, 96.0, ending=False), tmp)
        check(not cut_off.ends_itself, "a track that stops mid-bar is found not to")

        # ---- the fit
        bad = []
        for L in (5.0, 7.591, 11.074, 18.3, 23.0, 31.0, 47.3):
            for mode in ("auto", "trim", "loop", "end-aligned"):
                for t in (tr, cut_off):
                    f = M.fit_track(t, L, mode, (0.5, 1.5), [])
                    total = sum(b - a for a, b in f.segments)
                    inside = all(-1e-6 <= a < b <= len(t.samples) / RATE + 1e-6 for a, b in f.segments)
                    ends = f.mode != "end-aligned" or abs(f.segments[-1][1] - t.end) < 1e-6
                    if abs(total - L) > 1.0 / RATE or not inside or not ends:
                        bad.append(f"{mode}->{f.mode} L={L} total={total:.5f} inside={inside} ends={ends}")
        check(not bad, "every fit adds up to the film, stays in the track, and end-aligned ends on the track's end"
              + (": " + "; ".join(bad[:3]) if bad else ""))

    # ---- the duck
    talk = [(0.4, 3.0), (3.5, 6.0), (8.0, 11.0)]
    spans = sound.speech_spans([Word("x", a, b) for a, b in talk], [(0.0, 0.4), (3.0, 3.5), (6.0, 8.0), (11.0, 12.0)],
                               12.0, join=M.JOIN)
    check(spans[0][0] < 0.45 and len(spans) == 2, f"a 0.5 s breath joins two phrases; a 2 s pause does not ({spans})")
    t = np.arange(0, 12.0, 0.001)
    g = M.duck_db(spans, 12.0, 10.0, t)
    inside = [g[(t >= a) & (t <= b)] for a, b in spans]
    check(all(np.ptp(v) < 1e-9 and abs(v[0] + 10) < 1e-9 for v in inside), "one level through every stretch of talk")
    gap = g[(t > spans[0][1] + M.HOLD + M.RISE) & (t < spans[1][0] - M.LEAD)]
    check(len(gap) > 0 and np.all(gap > -1e-9), "the music is fully back up in the long pause")
    check(np.max(np.abs(np.diff(g))) / 0.001 < 10.0 / M.LEAD * 1.6 + 1e-6, "no step: every move is a ramp")

    # ---- the library
    lib = S.library()
    check(set(lib) == set(S.KINDS), f"all eight sounds are there ({', '.join(sorted(lib))})")
    for k, s in lib.items():
        peak = 20 * math.log10(float(np.abs(s.samples).max()))
        loudest = sound.loudest(s.samples)
        ok = peak <= -0.9 and 0 <= s.anchor <= len(s.samples) / RATE and abs(loudest - s.loudest) < 0.3
        check(ok, f"{k}: peak {peak:.1f} dBFS, lands at {s.anchor:.3f} s ({s.lands}), loudest {loudest:.1f} LUFS")

    # ---- the plan readers
    plan = {"effect": "clean-cut", "summary": {"seconds_in": 31.01},
            "removals": [{"start": 0.0, "end": 0.61}, {"start": 5.13, "end": 8.92}, {"start": 12.31, "end": 27.85}]}
    cuts = S._plan_cuts(plan)
    check([round(c[0], 3) for c in cuts] == [4.52, 7.91], f"a clean-cut plan's cuts are where the video joins ({cuts})")
    cards = S._plan_cards({"effect": "tier-list", "cards": [{"text": "Mic", "tier": "S", "at": 2.9, "lands": 3.14}]})
    check(cards and abs(cards[0][0] - 3.14) < 1e-9, "a tier list's card is marked where it lands")

    print(f"\n{'all passed' if not FAILED else str(len(FAILED)) + ' failed'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
