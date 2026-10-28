"""Which seconds of the clip are kept, and where each lands in the output.

The output plays the kept stretches one after another. Sound is cut at exactly these times (with
short crossfades); each output frame shows the source frame on screen at the matching source time,
so picture and sound never drift apart however many cuts there are.

`hold` adds seconds after the last kept stretch in which its last frame stays on screen, in
silence: an end card, or a finished graphic held long enough to read or screenshot.

`freezes` stop the picture inside a kept stretch: at a source moment the frame on screen holds for
a number of seconds, in silence, and then the clip plays on from exactly where it stopped. Picture
and sound stop and start together, so nothing said is lost and nothing drifts; the output is that
much longer. A timeline without freezes behaves exactly as it did before they existed.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Timeline:
    segments: tuple[tuple[float, float], ...]  # kept source seconds [start, end), in order
    hold: float = 0.0  # seconds the last frame stays on screen after the last stretch, in silence
    # Moments where the picture stops inside a kept stretch: (source seconds, seconds held), in order.
    freezes: tuple[tuple[float, float], ...] = ()

    @staticmethod
    def whole(duration: float) -> "Timeline":
        return Timeline(((0.0, float(duration)),))

    def held(self, seconds: float) -> "Timeline":
        """The same stretches, with the last frame held for `seconds` at the end."""
        return Timeline(self.segments, max(0.0, float(seconds)), self.freezes)

    def frozen(self, at: float, seconds: float) -> "Timeline":
        """The same, with the picture stopped on the frame at source time `at` for `seconds`, in
        silence, before it plays on from there. `at` must be inside a kept stretch."""
        if seconds <= 0:
            return self
        if self._play(at) is None:
            raise ValueError(f"a freeze at {at:.3f} s falls in a stretch that was cut")
        freezes = tuple(sorted(self.freezes + ((float(at), float(seconds)),)))
        return Timeline(self.segments, self.hold, freezes)

    @property
    def kept(self) -> float:
        """Seconds of the clip that play: the output's length without the hold or the freezes."""
        return sum(b - a for a, b in self.segments)

    @property
    def stills(self) -> float:
        """Seconds the picture stands still inside the clip (the freezes, without the end hold)."""
        return sum(s for _, s in self.freezes)

    @property
    def played(self) -> float:
        """The output's length before the end hold: the kept stretches and the freezes in them."""
        return self.kept + self.stills

    @staticmethod
    def keeping(duration: float, removed: list[tuple[float, float]]) -> "Timeline":
        """Everything except the `removed` stretches."""
        cuts = sorted((max(0.0, a), min(duration, b)) for a, b in removed if b > a)
        segs, t = [], 0.0
        for a, b in cuts:
            if a > t:
                segs.append((t, a))
            t = max(t, b)
        if t < duration:
            segs.append((t, duration))
        return Timeline(tuple(segs))

    @property
    def duration(self) -> float:
        """The output's length, the freezes and the hold included."""
        return self.kept + self.stills + self.hold

    # ---- play time: where a source moment would land with no freezes ---------------------------

    @property
    def _play_starts(self) -> list[float]:
        out, t = [], 0.0
        for a, b in self.segments:
            out.append(t)
            t += b - a
        return out

    def _play(self, t: float) -> Optional[float]:
        starts = [a for a, _ in self.segments]
        k = bisect.bisect_right(starts, t) - 1
        if k < 0:
            return None
        a, b = self.segments[k]
        if t > b + 1e-9:
            return None
        return self._play_starts[k] + (t - a)

    def freezes_in_play(self) -> list[tuple[float, float, float]]:
        """Each freeze as (play time, seconds, source time), in order. Play time is where the frozen
        moment falls in the kept stretches joined end to end, before any freeze: where the sound
        made from them (`render.sound_for`) is opened up for the freeze's silence."""
        out = []
        for at, seconds in self.freezes:
            p = self._play(at)
            if p is not None:
                out.append((p, seconds, at))
        return sorted(out)

    def _shift(self, p: float) -> float:
        """The seconds of freezes that play before play time `p` (a freeze at `p` itself does not:
        a moment that is frozen lands at the start of its freeze)."""
        return sum(s for q, s, _ in self.freezes_in_play() if q < p - 1e-9)

    # ---- the output ------------------------------------------------------------------------------

    @property
    def out_starts(self) -> list[float]:
        """Where each kept stretch begins in the output."""
        return [p + self._shift(p) for p in self._play_starts]

    @property
    def joins(self) -> list[float]:
        """Output times where one kept stretch meets the next."""
        return self.out_starts[1:]

    def still_spans(self) -> list[tuple[float, float]]:
        """Each freeze as the output seconds it fills: silent, the picture standing still."""
        out, shift = [], 0.0
        for p, s, _ in self.freezes_in_play():
            out.append((p + shift, p + shift + s))
            shift += s
        return out

    def is_identity(self, duration: float) -> bool:
        return (len(self.segments) == 1 and self.segments[0][0] <= 1e-6 and abs(self.segments[0][1] - duration) < 1e-6
                and self.hold <= 0.0 and not self.freezes)

    def to_output(self, t: float) -> Optional[float]:
        """Where source time `t` plays in the output; None when it was cut. A frozen moment lands at
        the start of its freeze."""
        p = self._play(t)
        if p is None:
            return None
        return p + self._shift(p) if self.freezes else p

    def to_output_nearest(self, t: float) -> float:
        """Like `to_output`, but a time inside a cut moves to the nearest kept edge."""
        got = self.to_output(t)
        if got is not None:
            return got
        best, best_d = 0.0, math.inf
        for a, b in self.segments:
            for src in (a, b):
                if abs(src - t) < best_d:
                    best, best_d = self.to_output(src), abs(src - t)
        return best

    def to_source(self, t_out: float) -> float:
        """The source time that plays at output time `t_out`. In a freeze it is the frozen moment.
        In the hold it is a hair before the end of the last stretch, so the frame on screen there is
        the last one that was kept."""
        if self.hold > 0 and t_out >= self.played:
            a, b = self.segments[-1]
            return max(a, b - 1e-4)
        p = t_out
        if self.freezes:
            shift = 0.0
            for q, s, at in self.freezes_in_play():
                if t_out < q + shift:
                    break
                if t_out < q + shift + s:
                    return at
                shift += s
            p = t_out - shift
        starts = self._play_starts
        k = max(0, bisect.bisect_right(starts, p) - 1)
        a, b = self.segments[k]
        return min(b, a + (p - starts[k]))

    def output_frames(self, fps: float) -> int:
        return max(1, int(round(self.duration * fps)))

    def source_frame(self, j: int, fps: float, n_source: int) -> int:
        """The source frame on screen at output frame `j`."""
        t = self.to_source(j / fps)
        return int(min(n_source - 1, max(0, math.floor(t * fps + 1e-6))))

    def sample_ranges(self, rate: int) -> list[tuple[int, int]]:
        return [(int(round(a * rate)), int(round(b * rate))) for a, b in self.segments]

    def map_spans(self, spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Source stretches (quiet runs, say) as output stretches: the parts that were kept,
        where they now play. A stretch broken by a cut comes out as the pieces either side; a
        stretch that runs across a freeze takes the freeze in with it (a freeze is silent)."""
        out = []
        for a, b in spans:
            for s0, s1 in self.segments:
                lo, hi = max(a, s0), min(b, s1)
                if hi > lo:
                    out.append((self.to_output(lo), self.to_output(hi)))
        out.sort()
        merged: list[tuple[float, float]] = []
        for a, b in out:
            if merged and a <= merged[-1][1] + 1e-6:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        return merged
