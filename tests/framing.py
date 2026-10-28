"""The calm window, checked on made-up face tracks (no video needed).

    .venv/bin/python tests/framing.py

Each case gives the face's path across a 1920-wide clip, as a 9:16 window of 608 would follow it,
and checks what a camera operator would do: hold through jitter and small sways, glide once for a
real move, cut where the clip cuts, and never let the face leave the window.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from fx.core import framing  # noqa: E402
from fx.core.layout import Rect  # noqa: E402

FPS, N, W, H = 30.0, 300, 1920, 1080
FACE_W = 300  # the face box's width in the clip


def frame_for(centres: np.ndarray, cuts=()) -> framing.Framing:
    faces = [Rect(c - FACE_W / 2, 300, c + FACE_W / 2, 650) for c in centres]
    return framing.plan(W, H, FPS, N, None, shape="9:16", faces=faces, cuts=list(cuts))


def inside(fr: framing.Framing, centres: np.ndarray) -> float:
    ok = 0
    for i, c in enumerate(centres):
        x, _, w, _ = fr.window(i)
        ok += (c - FACE_W / 2 >= x - 0.5) and (c + FACE_W / 2 <= x + w + 0.5)
    return ok / len(centres)


def main() -> int:
    rng = np.random.default_rng(7)
    failures = []

    def check(name: str, ok: bool, said: str) -> None:
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {said}")
        if not ok:
            failures.append(name)

    # 1. A face that only jitters (the detector's noise, a few pixels) never moves the window.
    jitter = 960 + rng.normal(0, 4, N)
    fr = frame_for(jitter)
    st = fr.path.stats(FPS, 608)
    check("jitter", st["glides"] == 0 and st["still_share"] == 1.0, f"{st}")
    # 2. A sway smaller than the dead zone does not either.
    sway = 960 + 40 * np.sin(np.arange(N) / FPS * 2 * np.pi * 0.5)
    fr = frame_for(sway)
    st = fr.path.stats(FPS, 608)
    check("small sway", st["glides"] == 0, f"{st}")
    # 3. A real move (the face settles 250 px to the right) is one glide, with no turning back.
    step = np.where(np.arange(N) < 120, 900.0, 1150.0) + rng.normal(0, 3, N)
    fr = frame_for(step)
    st = fr.path.stats(FPS, 608)
    check("one move", st["glides"] == 1 and st["turns_while_moving"] == 0 and inside(fr, step) == 1.0,
          f"{st}, face inside {inside(fr, step):.0%}")
    # 4. The glide starts before the face has finished moving (it leads, it does not chase).
    m = fr.path.moves[0]
    check("leads", m.start < 125, f"the glide starts at frame {m.start}; the face moves at frame 120")
    # 5. A cut in the clip is a cut in the window, not a slide across.
    jump = np.where(np.arange(N) < 150, 500.0, 1500.0)
    fr = frame_for(jump, cuts=[150])
    st = fr.path.stats(FPS, 608)
    x149, x150 = fr.window(149)[0], fr.window(150)[0]
    check("cut", st["cuts"] == 1 and st["glides"] == 0 and abs(x150 - x149) > 600 and inside(fr, jump) == 1.0,
          f"{st}, the window jumps {x150 - x149:.0f} px at the cut")
    # 6. A quick lean far to one side keeps the face inside (the window is pushed along).
    lean = 960 + np.where((np.arange(N) > 100) & (np.arange(N) < 130), 260.0, 0.0)
    fr = frame_for(lean)
    check("quick lean", inside(fr, lean) == 1.0, f"face inside {inside(fr, lean):.0%}, "
          f"pushed {fr.path.pushed} frames")
    print("framing:", "all right" if not failures else f"{len(failures)} failed: {', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
