"""The timeline, checked on numbers alone (no clip needed): cuts, the end hold, and freezes inside the
clip, picture and sound together.

    .venv/bin/python tests/timeline.py

A timeline with no freeze must map every moment exactly as before freezes existed (the reference
below is the old arithmetic, written out again). A freeze must hold one source frame for its seconds,
put that much silence into the sound at the same place, faded so nothing clicks, and play on from
exactly where it stopped, so the sound after it is the clip's own sound, sample for sample, shifted
by the freeze. Exits 1 on any failure.
"""

from __future__ import annotations

import bisect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from fx.core import render  # noqa: E402
from fx.core.timeline import Timeline  # noqa: E402

FAILED = []


def check(ok: bool, what: str) -> None:
    print(("ok    " if ok else "FAIL  ") + what)
    if not ok:
        FAILED.append(what)


def old_to_output(tl: Timeline, t: float):
    starts = [a for a, _ in tl.segments]
    k = bisect.bisect_right(starts, t) - 1
    if k < 0:
        return None
    a, b = tl.segments[k]
    if t > b + 1e-9:
        return None
    outs, acc = [], 0.0
    for s0, s1 in tl.segments:
        outs.append(acc)
        acc += s1 - s0
    return outs[k] + (t - a)


def old_to_source(tl: Timeline, t_out: float) -> float:
    outs, acc = [], 0.0
    for s0, s1 in tl.segments:
        outs.append(acc)
        acc += s1 - s0
    k = max(0, bisect.bisect_right(outs, t_out) - 1)
    a, b = tl.segments[k]
    if tl.hold > 0 and t_out >= tl.kept:
        return max(a, b - 1e-4)
    return min(b, a + (t_out - outs[k]))


def main() -> int:
    # ---- no freeze: exactly as before ----------------------------------------------------------
    cut = Timeline.keeping(10.0, [(1.0, 1.5), (4.2, 5.0), (8.8, 9.1)])
    for tl, name in ((cut, "cuts"), (cut.held(1.2), "cuts and an end hold"), (Timeline.whole(7.5), "whole")):
        ts = np.linspace(-0.2, 10.2, 1041)
        same = all((tl.to_output(t) is None and old_to_output(tl, t) is None)
                   or (tl.to_output(t) is not None and abs(tl.to_output(t) - old_to_output(tl, t)) < 1e-9)
                   for t in ts)
        check(same, f"{name}: every source moment lands where it did before freezes existed")
        outs = np.linspace(0, tl.duration, 997)
        check(all(abs(tl.to_source(t) - old_to_source(tl, t)) < 1e-9 for t in outs),
              f"{name}: every output moment shows what it did before")
        check(tl.stills == 0 and abs(tl.duration - (tl.kept + tl.hold)) < 1e-12, f"{name}: the same length")
    check(Timeline.whole(7.5).is_identity(7.5), "a whole clip is still the identity")

    # ---- one freeze ------------------------------------------------------------------------------
    fps, at, hold = 30.0, 61 / 30.0, 45 / 30.0
    tl = Timeline.whole(6.0).frozen(at, hold)
    check(not tl.is_identity(6.0), "a frozen clip is not the identity (it cannot be laid inside another effect)")
    check(abs(tl.duration - 7.5) < 1e-9 and tl.output_frames(fps) == 225, "the output is longer by the freeze, frame for frame")
    frames = [tl.source_frame(j, fps, 180) for j in range(tl.output_frames(fps))]
    check(frames[:61] == list(range(61)), "before the freeze every frame plays in order")
    check(frames[61:106] == [61] * 45, "the freeze holds the frozen frame for exactly its frames")
    check(frames[106:] == list(range(61, 180)), "after it the clip plays on from the frame it stopped on")
    check(abs(tl.to_output(at) - at) < 1e-9 and abs(tl.to_output(at + 0.5) - (at + 0.5 + hold)) < 1e-9,
          "a moment before or at the freeze stays; one after it moves by the freeze")
    check(tl.still_spans() == [(at, at + hold)], "the freeze's own seconds are where it is")
    check(tl.map_spans([(1.0, 3.0)]) == [(1.0, 3.0 + hold)], "a quiet stretch across the freeze takes it in")

    # ---- the sound ---------------------------------------------------------------------------------
    rate = 48000
    n = int(6.0 * rate)
    rng = np.random.default_rng(3)
    sound = (rng.standard_normal((n, 2)) * 0.1).astype(np.float32)
    clip = SimpleNamespace(sound=sound, info=SimpleNamespace(rate=fps, frames=180))
    out = render.sound_for(clip, tl)
    check(len(out) == int(round(225 / fps * rate)), "the sound is exactly as long as the picture")
    a, b = int(round(at * rate)), int(round((at + hold) * rate))
    check(float(np.abs(out[a:b]).max()) == 0.0, "the freeze is silent")
    tail = int(render.FREEZE_FADE * rate)
    check(np.allclose(out[b + tail:b + tail + rate], sound[a + tail:a + tail + rate], atol=1e-6),
          "after the freeze the sound is the clip's own, shifted by the freeze, sample for sample")
    check(float(np.abs(out[a - 3:a]).max()) < 0.01 and float(np.abs(out[b:b + 3]).max()) < 0.01,
          "it fades into and out of the freeze: no click either side")
    check(np.allclose(out[1000:a - tail], sound[1000:a - tail], atol=1e-6), "before the freeze it is the clip's own sound")

    # ---- a freeze with cuts and an end hold -----------------------------------------------------
    tl2 = cut.frozen(6.0, 1.0).held(0.5)
    check(abs(tl2.duration - (tl2.kept + 1.0 + 0.5)) < 1e-9, "cuts, a freeze and an end hold add up")
    check(abs(tl2.joins[-1] - (old_to_output(cut, 9.1) + 1.0)) < 1e-9, "a join after the freeze moves by it")
    check(abs(tl2.joins[0] - old_to_output(cut, 1.5)) < 1e-9, "a join before the freeze stays")
    check(abs(tl2.to_source(tl2.duration - 0.1) - (10.0 - 1e-4)) < 1e-9, "the end hold still shows the last frame")
    try:
        cut.frozen(4.5, 1.0)
        check(False, "a freeze inside a cut is refused")
    except ValueError:
        check(True, "a freeze inside a cut is refused")

    print(f"\n{'all passed' if not FAILED else str(len(FAILED)) + ' failed'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
