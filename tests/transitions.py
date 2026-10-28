"""The transitions effect's parts, checked on numbers made here (no clip needed).

    .venv/bin/python tests/transitions.py

- the options: looks by their names and the names people use, joins, lengths;
- a transition's frames: the join exactly at its middle;
- stray frames: a frame left over from what was cut out is skipped, somebody moving is not;
- the voices between clips: an overlap is never longer than the quiet between the last word out and
  the first word in, and the sound switches inside it;
- the punch: the whole head kept, with a tenth of the frame above the crown; a crown already cut by
  the recording left as it was; a head too big for 1.15x gets less;
- between ideas: only where a sentence ends, three seconds apart, the clearest break first;
- the sounds: each made, inside its own length, landing inside itself, set under the voice.

Exits 1 on any failure.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from fx.core import sound  # noqa: E402
from fx.core.layout import Rect  # noqa: E402
from fx.effects import options as O  # noqa: E402
from fx.effects import transitions as T  # noqa: E402

RATE = 48000
FAILED: list[str] = []


def check(ok: bool, what: str) -> None:
    print(("ok    " if ok else "FAIL  ") + what)
    if not ok:
        FAILED.append(what)


def refuses(kind, raw) -> bool:
    try:
        kind.parse(raw)
    except O.OptionError:
        return True
    return False


def bursts(seconds: float, spans: list[tuple[float, float]], hz: float, seed: int) -> np.ndarray:
    """A made-up voice: syllables of a buzzing tone inside `spans`, a faint room hiss everywhere."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    x = 0.002 * rng.standard_normal(len(t))
    for a, b in spans:
        m = (t >= a) & (t < b)
        syll = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t[m]) ** 2
        x[m] += 0.3 * syll * np.sign(np.sin(2 * np.pi * hz * t[m])) * 0.5
    x = x.astype(np.float32)
    return np.stack([x, x], axis=1)


def main() -> int:
    # ---- the options
    look = T.OPTIONS["look"]
    check(look.parse("whip, Light Leak, fade") == "whip,light-leak,crossfade", "looks by their names and others' words")
    check(look.parse("zoom punch then dots") == "punch,dot-matrix", "'zoom punch then dots' is a punch, then dots")
    check(refuses(look, "spin"), "a look that does not exist is refused")
    joins = T.OPTIONS["joins"]
    check(joins.parse("Between ideas") == "between-ideas" and joins.parse("all") == "every", "joins by name")
    check(joins.parse("12, 4.5 0:09") == "4.5, 9, 12", "joins as seconds, in order, with minutes")
    check(refuses(joins, "sometimes"), "joins that are neither a choice nor seconds are refused")
    dur = T.OPTIONS["duration"]
    check(dur.parse("auto") == "auto" and abs(dur.parse("0.4s") - 0.4) < 1e-9 and refuses(dur, "5"),
          "a length: auto, seconds, and no more than 3 s")

    # ---- a transition's frames
    tr = T.Transition(1, "crossfade", 1.0, 30, 0.4, 24, 35, "test")
    us = [tr.u(j) for j in range(tr.first, tr.last + 1)]
    check(all(a < b for a, b in zip(us, us[1:])) and 0 < us[0] and us[-1] < 1, "u rises through the transition")
    check(tr.u(29) < 0.5 < tr.u(30) and abs((tr.u(29) + tr.u(30)) / 2 - 0.5) < 0.05,
          "the join sits at the middle: the last frame out just under 0.5, the first in just over")

    # ---- stray frames
    still = np.full(80, 0.4, np.float32)
    stray = still.copy()
    stray[40], stray[41] = 144.6, 145.6  # into a left-over frame and out of it
    moving = still.copy()
    moving[40] = 30.0
    moving[41:] = 5.4  # somebody moving after the cut
    check(T.settled_frame(stray, 40, 60.0) == 41, "a frame left over from what was cut out is skipped")
    check(T.settled_frame(moving, 40, 48.0) == 40, "somebody moving after the cut is not a stray frame")
    check(T.settled_frame(still, 40, 30.0) == 40, "a clean cut stays where it is")

    # ---- voices between clips
    a = bursts(3.0, [(0.2, 2.6)], 220.0, 1)  # talks until 2.6 s: 0.4 s of quiet at its end
    b = bursts(3.0, [(0.25, 2.8)], 330.0, 2)  # quiet for its first 0.25 s
    hiss = bursts(3.0, [], 0.0, 3)
    va, vb, vh = T.hear(a), T.hear(b), T.hear(hiss)
    check(va.voiced and vb.voiced and not vh.voiced, "a voice is heard in talk and not in hiss")
    check(abs(va.trail - 0.37) < 0.08 and abs(vb.lead - 0.22) < 0.08,
          f"the quiet after the last word ({va.trail:.2f} s) and before the first ({vb.lead:.2f} s)")
    check(abs(T.quiet_room(va, vb) - (va.trail + vb.lead)) < 1e-9 and T.quiet_room(vh, vh) == math.inf
          and T.quiet_room(va, vh) == va.trail, "the most two clips may overlap, talking or not")
    o = T.quiet_room(va, vb)
    fps = 30.0
    ov = T.even_frames(o, fps)
    check(ov / fps <= o + 1e-9, f"the overlap in frames ({ov / fps:.3f} s) is never longer than the quiet ({o:.3f} s)")
    film = T.Film(pieces=[None, None], sources=[None, None], starts=[0, int(3 * fps) - ov],
                  counts=[int(3 * fps), int(3 * fps)], overlaps=[ov], size=(8, 8),
                  rate=__import__("fractions").Fraction(30))
    out, crossings = T.film_sound(film, [a, b], [va, vb])
    n_o = int(round(film.overlaps[0] / fps * RATE))
    start = int(round(film.starts[1] / fps * RATE))
    region = out[start:start + n_o, 0]
    a_tail = a[len(a) - n_o:, 0]
    b_head = b[:n_o, 0]
    # where A still talks inside the overlap, the output is A alone; where B already talks, B alone
    a_loud = np.abs(a_tail) > 0.05
    b_loud = np.abs(b_head) > 0.05
    check(not (a_loud & b_loud).any(), "inside the overlap the two voices never sound together in the clips")
    both = (np.abs(region - a_tail) > 1e-4) & (np.abs(region - b_head) > 1e-4) & (a_loud | b_loud)
    check(not both.any(), "and the film plays one clip's sound or the other's there, never a mix of two voices")
    check("switches" in crossings[0]["how"], "the crossing says where it switched")

    # ---- the punch
    W, H = 1080, 1920
    head = Rect(380, 420, 700, 900)
    (x, y, w, h), z, note = T.punch_window((W, H), 1.15, head, (540, 700))
    check(abs(z - 1.15) < 1e-6 and not note, "a head with room takes the whole punch")
    check(x <= head.x0 and x + w >= head.x1 and y + h >= head.y1, "the whole head stays in the punched frame")
    check(y <= head.y0 - T.CROWN * h + 1e-6, "with a tenth of the frame above the crown")
    cut = Rect(200, -150, 880, 1500)
    (x, y, w, h), z, note = T.punch_window((W, H), 1.15, cut, (540, 600))
    check(y == 0.0, "a crown the recording already cut: the window keeps to the top edge")
    big = Rect(120, 300, 960, 1800)
    (_, _, w, _), z, note = T.punch_window((W, H), 1.15, big, (540, 1000))
    check(z < 1.15 and note != "", f"a head too big for 1.15x gets less ({z:.2f}x) and says so: {note}")
    low = Rect(330, 600, 750, 1300)  # a head whose chin sits at 68% of the frame: 1.3x would take it to 73%
    (x, y, w, h), z, note = T.punch_window((W, H), 1.30, low, (540, 985))
    chin = (low.y1 - y) * H / h
    crown = (low.y0 - y) * H / h
    check(chin <= T.CHIN_LINE * H + 1 and crown >= T.CROWN * H - 1,
          f"a punch keeps the band under the chin free for captions, the crown its room ({z:.2f}x, chin at "
          f"{chin / H:.0%}, crown at {crown / H:.0%})")
    face = (540.0, 860.0)
    (x, y, w, h), z, _ = T.punch_window((W, H), 1.15, head, face)
    check(abs((face[0] - x) * W / w - face[0]) < 2 and abs((face[1] - y) * H / h - face[1]) < 60,
          "a punch grows the face where it is, rather than moving it to the middle")
    full = (0.0, 0.0, float(W), float(H))
    win = T._between((100.0, 100.0, 900.0, 1600.0), full, 1.09, (W, H))
    check(win[2] <= W + 1e-6 and win[0] >= 0 and win[1] >= 0, "a punch back out never frames wider than the shot")

    # ---- between ideas
    J = T.Join
    js = [J(2.0, 60, "t", sentence=True, pause=0.3, before="a."), J(3.5, 105, "t", sentence=True, pause=0.9, before="b."),
          J(6.0, 180, "t", sentence=False, before="c"), J(9.0, 270, "t", sentence=True, pause=0.2, before="d."),
          J(11.5, 345, "t", sentence=True, pause=0.5, before="e.")]
    picked = T.choose(js, "between-ideas", 14.0, [])
    at = [j.at for j in picked]
    check(at == [3.5, 11.5], f"between ideas: the clearer of two close breaks, sentences only, 3 s apart ({at})")
    check(all(abs(p - q) >= T.APART for p, q in zip(at, at[1:])), "no two closer than three seconds")
    given = T.choose(js, "2.1, 7", 14.0, [])
    check([round(j.at, 2) for j in given] == [2.0, 7.0] and given[0].cut and not given[1].cut,
          "seconds given: onto a cut within a quarter second, else a moment of its own")

    # ---- the sounds
    voice = bursts(4.0, [(0.3, 3.7)], 180.0, 9)
    trs = []
    for k, look_ in enumerate(T.LOOKS):
        t = T.Transition(k + 1, look_, 2.0, 60, T.AUTO[look_][0], 60 - 6, 60 + 6, "test")
        if look_ == "punch":
            t.zoom_from, t.zoom_to = 1.0, 1.15
        trs.append(t)
    for t in trs:
        kind, x, anchor = T.make_sound(t, 30.0)
        ok = (np.isfinite(x).all() and np.abs(x).max() > 1e-3 and 0 <= anchor <= len(x) / RATE
              and x.ndim == 2 and x.shape[1] == 2)
        check(ok, f"{t.look}: its sound ({kind}, {len(x) / RATE:.2f} s) is made, lands {anchor:.2f} s in")
    sfx = T.lay_sounds(trs, 30.0, voice, talking=True)
    ref = sfx.reference
    louds = [sound.loudest(e["samples"] * sound.gain(e["gain_db"])) for e in sfx.events]
    wanted = [ref - (T.UNDER_VOICE + (3.0 if e["kind"] == "hit" and e["tr"].zoom_to < e["tr"].zoom_from else 0.0))
              for e in sfx.events]
    check(all(abs(lo - w) < 0.6 for lo, w in zip(louds, wanted)),
          f"each sound's loudest tenth of a second sits {T.UNDER_VOICE:g} dB under the voice")
    quiet = T.lay_sounds(trs[:1], 30.0, hiss, talking=False)
    check(abs(quiet.reference - T.NO_VOICE) < 1e-9, "with nobody talking the sounds are set against a fixed level")

    print()
    if FAILED:
        print(f"{len(FAILED)} failed")
        return 1
    print("all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
