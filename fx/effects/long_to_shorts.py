"""long-to-shorts: a long video becomes ranked vertical shorts, each one complete thought with a hook
and a close, cut clean, made 9:16 and captioned.

It is the pack put together, nothing new underneath:

1. **The thoughts.** Where the talk moves on to a new idea comes from the core's parts of a talk
   (`fx/core/parts.py`: the voice stopping, a signpost such as "first of all", the words changing).
   A short may begin only where a sentence begins and end only where one ends (or where the voice
   really stops between two clauses, for a speaker the transcriber gave no full stops).
2. **What is never in a short.** clean-cut's own cuts are made inside every short (`fx/effects/clean_cut`:
   dead air, retakes and false starts, fillers, and asides, the talk about the recording itself), so a
   short never opens on a first attempt or a mic check. A stretch that is mostly asides is never a
   short at all, whatever it scores.
3. **The ranking.** Every stretch of whole sentences between `min` and `max` seconds (as it will play,
   after the cuts) is weighed on six things, each with its reason written down:
   - *hook*: how well its first line works cold (the core's `moments.strong_starts`, the measure every
     picker uses), less when it opens on "and", "but" or "as you can see", which lean on what came
     before, or on a greeting; more for a question or a signpost;
   - *close*: whether it ends on a full stop with the voice stopping after it, or on "that's it",
     and not where the next sentence carries straight on ("and then…");
   - *one thought*: whether it begins and ends where a part of the talk does, and how many changes
     of subject it runs across;
   - *its best line*: the strongest line in it (`moments.pick_line`);
   - *pace*: words a second as it will play;
   - *length*: near the middle of `min` and `max`.
   `rank` weighs them differently (`best`, `hook`, `story`, `pace`). The shorts are the best stretches
   that do not overlap, best first.
4. **Each short** is made like clean-cut makes a reel: the cuts in its own seconds, reframed to 9:16
   following the face (the framing `reframe` uses: black bars left out, the speaker followed when
   there are two), captions on. It starts in the quiet before its first word and ends 0.3 s after its
   last, in the quiet, so no word is clipped; afterwards it is heard again word by word (clean-cut's
   over-cut check) and the result goes into the index.

It reads the words, not the picture: a stretch where something is shown rather than said, a laugh or
a gesture, is invisible to it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..core import framing as framing_mod
from ..core import loudness as loudness_mod
from ..core import moments, parts
from ..core.asides import asides as find_asides
from ..core.timeline import Timeline
from ..core.words import HESITATIONS, Word, pause_after
from . import captions as cap
from .headline import keepout_in_output
from . import clean_cut
from . import options as O

NAME = "long-to-shorts"

OPTIONS = {
    "count": O.Whole(1, 10, "how many shorts to make, best first (fewer when the video holds fewer whole thoughts)"),
    "min": O.Number(5.0, 120.0, "the shortest a short may be, in seconds as it plays"),
    "max": O.Number(10.0, 180.0, "the longest a short may be, in seconds as it plays"),
    "rank": O.Choice(("best", "hook", "story", "pace"), "best: everything weighed · hook: the opening line counts "
                                                         "most · story: one whole thought from start to close · "
                                                         "pace: the liveliest stretches"),
    "only": O.Whole(0, 10, "0 makes every short; a number makes only that one (2 renders the second best again)"),
    "cuts": O.Choice(("clean", "none"), "clean: dead air, retakes, fillers and asides cut inside each short · "
                                        "none: each short plays as filmed"),
    "captions": O.Switch("on or off"),
    "loudness": O.Choice(("reels", "keep"), "reels: each short set to -14 LUFS with its peaks under -1 dBTP, as "
                                            "Reels, Shorts and TikTok play it · keep: the video's own level"),
    "look": cap.OPTIONS["look"],
    "color": cap.OPTIONS["color"],
    "position": cap.OPTIONS["position"],
    "max_words": cap.OPTIONS["max_words"],
    "frame": O.Choice(("9:16", "keep"), "9:16 follows the face when the video is wider; keep leaves its shape"),
    "size": cap.OPTIONS["size"],
    "fix": cap.OPTIONS["fix"],
}

# The runs `tests/matrix.py long-to-shorts` makes: our longest whole talk (raw 31, 107 s), the
# demo-day talk with its asides (33), the long close-up with a line said three times (E), and his
# green-screen take in landscape for the reframe (16 s, so its shorts are short).
# Raw 31's transcriber hears "Hua Hin" as "Hawaiian" and "pad thai" as "pat tie": a person would say so.
FIX_31 = "Hawaiian=Hua Hin; pat ties=pad thais; pat tie=pad thai"
MATRIX = [
    ("31", "31", {"fix": FIX_31}),
    ("33", "33", {}),
    ("E", "E", {"min": 8, "count": 2}),
    ("wide", "wide", {"min": 5, "max": 15, "count": 2}),
]
# The files beside the first one: the second best and on (`<name>-2.mp4`), and the index.
LABELS = tuple(str(k) for k in range(2, 11))
BESIDE = (".shorts.md", ".shorts.json")

# Where a short may begin or end, besides a full stop: the voice stopping this long between clauses.
CLAUSE_STOP = 0.6
LEAD = 0.12  # kept before the first word
TAIL = 0.30  # kept after the last word
MOST_ASIDE = 0.3
HOOK_SECONDS = 4.5  # the first seconds a viewer hears cold: the opening sentence, or this much of a long one
TALK_EDGE = 3.0  # the start or the end of the talk itself: as clear an edge as a thought can have  # a stretch with more than this share of its words about the recording is never a short

# Openings that lean on what came before, or only say hello: they do not work cold.
_JOINS = {"and": 0.35, "but": 0.35, "or": 0.35, "because": 0.35, "then": 0.2, "so": 0.1, "also": 0.2}
_LEANS = (("as", "you", "can", "see"), ("as", "i", "said"), ("like", "i", "said"), ("that's", "why"),
          ("this", "is", "why"), ("after", "that"), ("which",), ("it", "was"), ("they", "were"))
_HELLOS = (("hi",), ("hello",), ("hey",), ("welcome",), ("what's", "up"), ("in", "this", "video"),
           ("today", "i"), ("so", "today"), ("in", "today's"))
_CLOSES = (("that's", "it"), ("that's", "pretty", "much", "it"), ("that's", "all"), ("and", "that's", "how"),
           ("that's", "how"), ("thanks", "for", "watching"), ("that's", "the", "secret"), ("that's", "why"))
_CARRIES_ON = {"and", "but", "so", "because", "then", "which", "or"}

# How each ranking weighs the six.
WEIGHTS = {
    "best": {"hook": 1.0, "close": 0.8, "thought": 1.0, "line": 0.8, "pace": 0.3, "length": 0.3},
    "hook": {"hook": 2.2, "close": 0.6, "thought": 0.6, "line": 0.8, "pace": 0.3, "length": 0.2},
    "story": {"hook": 0.6, "close": 1.4, "thought": 2.0, "line": 0.5, "pace": 0.2, "length": 0.2},
    "pace": {"hook": 0.8, "close": 0.6, "thought": 0.6, "line": 0.5, "pace": 2.0, "length": 0.2},
}


def _starts_with(keys: list[str], table) -> Optional[tuple[str, ...]]:
    for p in table:
        if tuple(keys[:len(p)]) == p:
            return p
    return None


def _ends_with(keys: list[str], table) -> Optional[tuple[str, ...]]:
    for p in table:
        if len(keys) >= len(p) and tuple(keys[-len(p):]) == p:
            return p
    return None


def _between(words: list[Word], kept_set: set, after: int, at: int, before: bool) -> bool:
    """Is the boundary after word `after` the one just before word `at` (before=True) or just after it,
    with only words that are cut between the two?"""
    if before:
        return after < at and not any(words[x] in kept_set for x in range(after + 1, at))
    return after >= at and not any(words[x] in kept_set for x in range(at + 1, after + 1))


def _signpost_only(ws: list[Word], span: tuple[int, int]) -> bool:
    """A line that is only a signpost ("First of all,"): it says a part begins, not what it says."""
    keys = [w.key for w in ws[span[0]:span[1] + 1] if w.key]
    sp, _ = parts.signpost(keys)
    return bool(sp) and len(sp.split()) >= len(keys)

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _typed(ws: list[Word], most: int = 14) -> str:
    said = [w.text.strip() for w in ws if w.key not in HESITATIONS][:most]
    out = " ".join(said)
    return out + (" …" if len([w for w in ws if w.key not in HESITATIONS]) > most else "")


# ---- the stretches a short can be --------------------------------------------------------------------------


@dataclass
class Short:
    first: int  # its first and last word kept, as indexes into the clip's words
    last: int
    start: float  # source seconds it begins and ends (in the quiet before and after)
    end: float
    seconds: float  # as it plays, after the cuts inside it
    scores: dict = field(default_factory=dict)  # each of the six, 0 to 1
    total: float = 0.0
    reasons: list = field(default_factory=list)
    opening: str = ""
    closing: str = ""
    best_line: str = ""
    title: str = ""
    part_titles: list = field(default_factory=list)
    rank: int = 0
    removals: list = field(default_factory=list)  # clean-cut's cuts inside it, source seconds

    def as_dict(self) -> dict:
        return {"rank": self.rank, "from": round(self.start, 3), "to": round(self.end, 3),
                "seconds": round(self.seconds, 2), "title": self.title, "opens": self.opening,
                "closes": self.closing, "best_line": self.best_line, "parts": self.part_titles,
                "score": round(self.total, 3), "scores": {k: round(v, 3) for k, v in self.scores.items()},
                "why": self.reasons}


def _edges(words: list[Word], i0: int, i1: int, env, duration: float) -> tuple[float, float]:
    """Where a short of words i0..i1 begins and ends: a little before its first word and after its
    last, never into a word either side (kept or cut), at the quietest moment when they are close."""
    first, last = words[i0], words[i1]
    lo = words[i0 - 1].end + 0.03 if i0 > 0 else 0.0
    s = max(lo, first.start - LEAD)
    if s > first.start - 0.02:
        s = env.quietest(min(lo, first.start - 0.02), first.start)
    hi = words[i1 + 1].start - 0.03 if i1 + 1 < len(words) else duration
    e = min(hi, last.end + TAIL)
    if e < last.end + 0.03:
        e = env.quietest(last.end, max(hi, last.end + 0.03))
    return max(0.0, s), min(duration, e)


def _inside(removals: list, s: float, e: float) -> list[tuple[float, float]]:
    out = []
    for r in removals:
        a, b = max(s, r.start), min(e, r.end)
        if b - a >= 0.04:
            out.append((a, b))
    return out


def find_shorts(clip, opts: dict) -> tuple[list[Short], list[Short], list[str], list]:
    """The shorts, best first and never overlapping; the runners-up; notes; clean-cut's removals."""
    cands, notes, removals = stretches(clip, opts)
    chosen: list[Short] = []
    for c in cands:
        if len(chosen) >= opts["count"]:
            break
        if any(c.start < o.end and o.start < c.end for o in chosen):
            continue
        chosen.append(c)
    runners = [c for c in cands if c not in chosen and not any(c.start < o.end and o.start < c.end for o in chosen)]
    for k, c in enumerate(chosen, 1):
        c.rank = k
    if cands and len(chosen) < opts["count"]:
        notes.append(f"{len(chosen)} short{'s' if len(chosen) != 1 else ''} of {opts['min']:g} to {opts['max']:g} s "
                     f"fit this video without overlapping ({opts['count']} were asked for)")
    return chosen, runners[:3], notes, removals


def stretches(clip, opts: dict) -> tuple[list[Short], list[str], list]:
    """Every stretch of whole sentences that could be a short, weighed, best first; notes; clean-cut's
    removals."""
    words = clip.words
    quiet = clip.quiet
    env = clip.envelope
    duration = clip.info.frames / clip.info.rate
    notes: list[str] = []
    if len(words) < 6:
        return [], ["there are too few words in this video to find a thought in"], []
    lo, hi = float(opts["min"]), float(opts["max"])
    if hi < lo:
        lo, hi = hi, lo
        notes.append("min was longer than max, so the two were swapped")

    if opts["cuts"] == "clean":
        removals, kept = clean_cut.plan_cuts(clip, {"pause": 0.35, "keep_retakes": False, "fillers": "cut",
                                                    "asides": "cut"})
    else:
        removals, kept = [], list(words)
    kept_set = set(kept)
    back = moments.taken_back(words, quiet)
    aside_idx = {x for a in find_asides(words, quiet) for x in range(a.first, a.last + 1)}

    # The thoughts: the parts of the talk, and every place one could begin, weighed.
    found = parts.find(words, quiet, duration)
    part_of = {}
    for p in found:
        for x in range(p.first, p.last + 1):
            part_of[x] = p
    window = max(6.0, min(20.0, parts.auto_min_part(duration)))
    bounds = {b.after: b for b in parts.weigh(words, parts.candidates(words, quiet), window)}
    # Where a short may start: the first word, or after a full stop or a real stop between clauses.
    def can_break(b) -> bool:
        return b.sentence or b.pause >= CLAUSE_STOP or b.strength >= 0.6

    cuts = sorted(i for i, b in bounds.items() if can_break(b))
    units, a = [], 0
    for c in cuts + [len(words) - 1]:
        if c >= a:
            units.append((a, c))
            a = c + 1
    # Each unit's first and last word that stays (none when the whole unit is cut: an aside, a retake).
    def kept_edge(a: int, b: int, last: bool) -> Optional[int]:
        rng = range(b, a - 1, -1) if last else range(a, b + 1)
        for x in rng:
            w = words[x]
            if w in kept_set and w.key not in HESITATIONS and x not in aside_idx and w not in back:
                return x
        return None

    starts = {x: s.strength for s in moments.strong_starts(words, quiet, back) for x in [s.index]}
    fps = clip.info.rate
    target = min(max(lo, 0.5 * (lo + hi)), max(lo, 45.0))
    weights = WEIGHTS[opts["rank"]]
    cands: list[Short] = []
    for ua in range(len(units)):
        i0 = kept_edge(*units[ua], last=False)
        if i0 is None:
            continue
        for ub in range(ua, len(units)):
            i1 = kept_edge(*units[ub], last=True)
            if i1 is None or i1 < i0:
                continue
            s, e = _edges(words, i0, i1, env, duration)
            inner = _inside(removals, s, e)
            secs = (e - s) - sum(b - a for a, b in inner)
            if secs > hi + 1e-6:
                break
            if secs < lo - 1e-6:
                continue
            span = list(range(i0, i1 + 1))
            if sum(1 for x in span if x in aside_idx) > MOST_ASIDE * len(span):
                continue
            cands.append(_weigh(words, quiet, back, span, s, e, secs, inner, kept_set, starts, bounds, part_of,
                                found, duration, target, lo, hi, weights, removals))
    if not cands:
        talk = [x for x in range(len(words)) if words[x] in kept_set and x not in aside_idx]
        if talk:
            i0, i1 = talk[0], talk[-1]
            s, e = _edges(words, i0, i1, env, duration)
            inner = _inside(removals, s, e)
            secs = (e - s) - sum(b - a for a, b in inner)
            if secs < lo and secs >= 4.0:
                notes.append(f"the whole talk plays {secs:.1f} s after the cuts, shorter than min={lo:g}, so it is "
                             f"one short of its whole length")
                cands = [_weigh(words, quiet, back, list(range(i0, i1 + 1)), s, e, secs, inner, kept_set, starts,
                                bounds, part_of, found, duration, target, lo, hi, weights, removals)]
    cands.sort(key=lambda c: -c.total)
    return cands, notes, removals


def _weigh(words, quiet, back, span, s, e, secs, inner, kept_set, starts, bounds, part_of, found, duration,
           target, lo, hi, weights, removals) -> Short:
    i0, i1 = span[0], span[-1]
    shown = [words[x] for x in span if words[x] in kept_set and words[x].key not in HESITATIONS]
    keys = [w.key for w in shown]
    reasons = []

    # hook: what the first seconds say, by the shared measures: the line the short opens on
    # (`strong_starts`) and the strongest line in its first sentence (`pick_line`), as a viewer
    # arriving cold hears them; less when it opens leaning on what came before
    first_sentence = []
    for w in shown:
        if first_sentence and w.start - first_sentence[0].start > HOOK_SECONDS:
            break
        first_sentence.append(w)
        if w.ends_sentence:
            break
    strength = starts.get(i0)
    heard: list = []
    moments.pick_line(first_sentence, quiet, back=back, ranked=heard)
    early = max((sc for sc, sp in heard if not _signpost_only(first_sentence, sp)), default=0.0)
    hook = 0.4 * _clamp(((strength if strength is not None else 0.0) + 2.0) / 16.0) + 0.6 * _clamp(early / 16.0)
    opening = _typed(first_sentence)
    why_hook = f"opens on \"{opening}\""
    q = first_sentence[-1].text.rstrip().endswith("?") if first_sentence else False
    lean = _starts_with(keys, _LEANS)
    hello = _starts_with(keys, _HELLOS)
    if keys and keys[0] in _JOINS:
        hook -= _JOINS[keys[0]]
        why_hook += f", which leans on what came before (\"{keys[0]}\")"
    elif lean:
        hook -= 0.3
        why_hook += f", which leans on what came before (\"{' '.join(lean)}\")"
    elif hello:
        hook -= 0.3
        why_hook += ", a greeting or an introduction rather than the point"
    elif q:
        hook += 0.2
        why_hook += ", a question the rest answers"
    b_in = bounds.get(i0 - 1)
    if b_in is not None and b_in.strength >= 0.6:
        hook += 0.1
        why_hook += f", announced (\"{b_in.signpost}\")"
    hook = _clamp(hook)
    reasons.append(why_hook)

    # close: a full stop and the voice stopping, or a closing phrase; never where it carries straight on
    last = words[i1]
    after = pause_after(words, i1, quiet) if i1 + 1 < len(words) else 2.0
    close = (0.35 if last.ends_sentence else 0.1) + 0.35 * _clamp(after / 1.2)
    why_close = "ends on a full stop" if last.ends_sentence else "ends where the voice stops between clauses"
    if i1 + 1 >= len(words) or not any(w in kept_set for w in words[i1 + 1:]):
        close += 0.2
        why_close += ", where the talk ends"
    elif after >= 0.3:
        why_close += f", the voice stopping {after:.1f} s after it"
    said_close = _ends_with(keys, _CLOSES)
    if said_close:
        close += 0.3
        why_close += f", on \"{' '.join(said_close)}\""
    nxt = next((x for x in range(i1 + 1, len(words)) if words[x] in kept_set and words[x].key not in HESITATIONS),
               None)
    if nxt is not None and words[nxt].key in _CARRIES_ON and after < CLAUSE_STOP:
        close -= 0.35
        why_close += f"; but the next sentence carries straight on (\"{words[nxt].key} …\")"
    close = _clamp(close)
    closing = _typed([w for w in shown[-12:]], 12)
    reasons.append(why_close)

    # one thought: its edges are where the talk moves on most, more than anywhere inside it, and it
    # begins and ends with a part of the talk
    p0, p1 = part_of.get(i0), part_of.get(i1)
    touched = []
    for x in span:
        p = part_of.get(x)
        if p is not None and p not in touched:
            touched.append(p)
    first_kept = next((x for x in range(len(words)) if words[x] in kept_set), 0)
    last_kept = max((x for x in range(len(words)) if words[x] in kept_set), default=len(words) - 1)
    edge_in = TALK_EDGE if i0 <= first_kept else max((b.score for x, b in bounds.items() if
                                                      _between(words, kept_set, x, i0, before=True)), default=0.0)
    edge_out = TALK_EDGE if i1 >= last_kept else max((b.score for x, b in bounds.items() if
                                                      _between(words, kept_set, x, i1, before=False)), default=0.0)
    most_inside = max((float(b.score) for x, b in bounds.items() if i0 <= x < i1), default=0.0)
    thought = (0.5 + 0.2 * _clamp((edge_in - most_inside) / 1.5, -1, 1)
               + 0.2 * _clamp((edge_out - most_inside) / 1.5, -1, 1))
    starts_part = p0 is not None and not any(words[x] in kept_set for x in range(p0.first, i0))
    ends_part = p1 is not None and not any(words[x] in kept_set for x in range(i1 + 1, p1.last + 1))
    thought += 0.05 * starts_part + 0.05 * ends_part
    thought = _clamp(thought)
    names = [p.title for p in touched if p.title]
    if most_inside > max(edge_in, edge_out):
        reasons.append(f"the talk moves on inside it more than at its edges (a change of subject at "
                       f"{max(bounds.values(), key=lambda b: b.score if i0 <= b.after < i1 else -1).at:.1f} s)")
    elif len(touched) == 1 and starts_part and ends_part:
        reasons.append(f"one part of the talk, whole (\"{names[0] if names else 'the talk'}\")")
    elif len(touched) == 1:
        reasons.append(f"inside one part of the talk (\"{names[0] if names else 'the talk'}\")")
    elif len(touched) > 1 and starts_part and ends_part:
        reasons.append(f"parts of the talk whole, one leading to the next ({' → '.join(names)})")
    else:
        reasons.append(f"runs across {len(touched) - 1} change{'s' if len(touched) > 2 else ''} of subject "
                       f"({' → '.join(names)})")

    # its best line, by the shared picker
    sub = [words[x] for x in span if words[x] in kept_set]
    ranked: list = []
    moments.pick_line(sub, quiet, back=back, ranked=ranked)
    ranked = [(sc, sp) for sc, sp in ranked if not _signpost_only(sub, sp)]
    line = _clamp((ranked[0][0] if ranked else 0.0) / 20.0)
    best_line = _typed(sub[ranked[0][1][0]:ranked[0][1][1] + 1]) if ranked else ""
    if best_line:
        reasons.append(f"its strongest line: \"{best_line}\"")

    # pace and length
    wps = len(shown) / max(1.0, secs)
    pace = float(_clamp((wps - 1.4) / 1.6))
    length = _clamp(1.0 - abs(secs - target) / max(5.0, hi - lo))
    reasons.append(f"{secs:.0f} s, {wps:.1f} words a second")

    scores = {"hook": hook, "close": close, "thought": thought, "line": line, "pace": pace, "length": length}
    total = sum(weights[k] * v for k, v in scores.items()) / sum(weights.values())
    title = touched[0].title if len(touched) == 1 and touched[0].title else (best_line or opening)
    title = title.rstrip(" ,.;:…")
    title = title[:1].upper() + title[1:]
    return Short(i0, i1, s, e, secs, scores, total, reasons, opening, closing, best_line, title, names,
                 removals=[(a, b) for a, b in inner])


# ---- the jobs -------------------------------------------------------------------------------------------------


def _framing(clip, opts: dict):
    """The window for the whole video, made once and shared by every short: reframe's own (black bars left
    out, the speaker followed when there are two) when the video is wider than 9:16."""
    info = clip.info
    size = O.size_of(opts["size"])
    if opts["frame"] != "9:16" or abs(info.width / info.height - framing_mod.SHAPES["9:16"]) < 0.01 * 9 / 16 \
            or info.width / info.height < 9 / 16:
        return framing_mod.plan(info.width, info.height, info.rate, info.frames, clip.face, shape=opts["frame"],
                                size=size), ""
    from . import reframe

    content = reframe.bars(info)
    x0, y0, x1, y1 = content
    fol = reframe.whom(clip, {"mode": "track", "speaker": "auto"}, content, framing_mod.SHAPES["9:16"])
    fr = framing_mod.plan(x1 - x0, y1 - y0, info.rate, info.frames, None, shape="9:16", size=size, mode="track",
                          faces=fol.faces, cuts=fol.cuts)
    if (x0, y0, x1, y1) != (0, 0, info.width, info.height):
        fr.content = content
    return fr, fol.note


def _short_from_plan(plan: dict, clip) -> Short:
    from . import PlanError

    raw = plan.get("short")
    if not isinstance(raw, dict):
        raise PlanError("the plan has no short; run once without --plan to get one to edit")
    duration = clip.info.frames / clip.info.rate
    try:
        s, e = float(raw["from"]), float(raw["to"])
        inner = [(float(a), float(b)) for a, b in raw.get("cuts", [])]
    except (KeyError, TypeError, ValueError):
        raise PlanError("the short needs from and to in seconds, and its cuts as [start, end] pairs") from None
    if not (0 <= s < e <= duration + 1e-3):
        raise PlanError(f"the short {s:.2f}-{e:.2f} s is not inside the video (0-{duration:.2f} s)")
    for a, b in inner:
        if not (s <= a < b <= e):
            raise PlanError(f"the cut {a:.2f}-{b:.2f} s is not inside the short")
    sh = Short(0, 0, s, e, (e - s) - sum(b - a for a, b in inner), rank=int(raw.get("rank") or 1),
               removals=inner, title=str(raw.get("title", "")), opening=str(raw.get("opens", "")),
               closing=str(raw.get("closes", "")), best_line=str(raw.get("best_line", "")),
               reasons=list(raw.get("why", [])), scores=dict(raw.get("scores", {})),
               total=float(raw.get("score", 0) or 0), part_titles=list(raw.get("parts", [])))
    return sh


def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from . import Job, PlanError

    info = clip.info
    fps = info.rate
    duration = info.frames / fps
    if plan is not None:
        chosen, runners, notes = [_short_from_plan(plan, clip)], [], []
    else:
        chosen, runners, notes, _ = find_shorts(clip, opts)
        if not chosen:
            raise PlanError("no stretch of whole sentences " + (notes[0] if notes else
                            f"between {opts['min']:g} and {opts['max']:g} s was found; give a smaller min or a "
                            "larger max"))
        if opts["only"]:
            pick = [c for c in chosen if c.rank == opts["only"]]
            if not pick:
                raise PlanError(f"there is no short number {opts['only']}; this video gives {len(chosen)}")
            chosen = pick
    fr, follow_note = _framing(clip, opts)
    index: dict = {"video": clip.path, "options": opts, "notes": notes, "files": {},
                   "shorts": [c.as_dict() for c in chosen], "runners_up": [c.as_dict() for c in runners],
                   "heard_again": {}, "loudness": {}}
    jobs = []
    for n, sh in enumerate(chosen):
        jobs.append(_job(clip, opts, sh, fr, follow_note, duration, index, n, Job))
    return jobs if len(jobs) > 1 else jobs[0]


def _job(clip, opts: dict, sh: Short, fr, follow_note: str, duration: float, index: dict, n: int, Job):
    fps = clip.info.rate
    tl = Timeline.keeping(duration, [(0.0, sh.start)] + list(sh.removals) + [(sh.end, duration)])
    kept = [w for w in clip.words if tl.to_output(0.5 * (w.start + w.end)) is not None]
    out_words = []
    for w in kept:
        a, b = tl.to_output_nearest(w.start), tl.to_output_nearest(w.end)
        out_words.append(Word(w.text, a, max(b, a + 0.02), w.prob))
    out_quiet = tl.map_spans(clip.quiet)
    n_out = tl.output_frames(fps)
    caps = None
    checks: dict = {}
    if opts["captions"]:
        # The face alone too (hairline to chin): a short cut from a close shot, or reframed tight from a
        # wide one, can leave no band clear of the whole head, and a line may then cross the top of the
        # hair rather than leave the safe zone (captions' `_flatter`).
        caps = cap.Captions(out_words, out_quiet, (fr.out_w, fr.out_h), fps, n_out, cap.head_in_output(clip, tl, fr),
                            look=opts["look"], color=opts["color"], position=opts["position"],
                            max_words=opts["max_words"], face_at=keepout_in_output(clip, tl, fr))
        checks = caps.report()
    note = fr.note + (f"; {follow_note}" if follow_note else "")
    head = {"short": f"#{sh.rank} \"{sh.title}\": {sh.start:.2f}-{sh.end:.2f} s of the video, {tl.duration:.1f} s "
                     f"as it plays, {len(sh.removals)} cut{'s' if len(sh.removals) != 1 else ''} inside",
            "why": "; ".join(sh.reasons), "framing": note}
    if n == 0 and index["notes"]:
        head["notes"] = "; ".join(index["notes"])
    checks = {**head, **checks}
    label = "" if n == 0 else str(sh.rank)
    out_plan = {"effect": NAME, "clip": clip.path, "options": opts,
                "output": {"width": fr.out_w, "height": fr.out_h, "fps": round(fps, 3),
                           "seconds": round(tl.duration, 3), "framing": note},
                "short": {**sh.as_dict(), "cuts": [[round(a, 3), round(b, 3)] for a, b in sh.removals]},
                "kept_words": [{"w": w.text, "s": round(w.start, 3), "e": round(w.end, 3)} for w in kept],
                "cards": caps.plan() if caps else [],
                "sheet_times": _sheet_times(caps, tl)}

    def stills() -> list[float]:
        return _sheet_times(caps, tl)[:5]

    def level(samples, rate):
        """Job.sound: the short's own sound at the level the apps play (one gain, peaks limited)."""
        out, report = loudness_mod.normalize(samples, -14.0, -1.0)
        index["loudness"][str(sh.rank)] = report
        return out

    def record(out: Path) -> None:
        index["files"][str(sh.rank)] = str(out)
        write_index(out_index_path(out, n, sh), index)

    def preview(out: Path) -> dict:
        record(out)
        return {"index": str(out_index_path(out, n, sh).with_suffix(".md"))}

    def after(result: dict, out: Path) -> dict:
        check = clean_cut.overcut_check(clip, kept, tl, out, opts.get("fix"))
        index["heard_again"][str(sh.rank)] = check["verdict"]
        record(out)
        return {"over-cut check": check["verdict"], "index": str(out_index_path(out, n, sh).with_suffix(".md"))}

    return Job(NAME, clip, tl, fr, caps.draw if caps else None, out_plan, checks=checks, still_times=stills,
               after=after, preview=preview, label=label, sound=level if opts["loudness"] == "reels" else None)


def _sheet_times(caps, tl: Timeline) -> list[float]:
    """Six moments across the short, each while words are on screen: the first line (the hook), the
    middle, and the last (the close)."""
    d = tl.duration
    if caps and caps.cards:
        cards = caps.cards
        picks = [cards[round(k * (len(cards) - 1) / 5)] for k in range(6)] if len(cards) >= 6 else cards
        out = [min(c.end - 0.02, max(c.words[-1].start + 0.1, 0.5 * (c.start + c.end))) for c in picks]
    else:
        out = [d * (k + 0.5) / 6 for k in range(6)]
    return sorted(set(round(min(d - 0.05, max(0.0, t)), 3) for t in out))


def out_index_path(out: Path, n: int, sh: Short) -> Path:
    """The index sits beside the best short's file: `<first>.shorts.json` (and `.md`). A later short's
    file is `<first>-<rank>.mp4`, so its first file's name is found by taking the label off again."""
    if n == 0:
        return out.with_suffix(".shorts.json")
    stem = out.stem[: -(len(str(sh.rank)) + 1)] if out.stem.endswith(f"-{sh.rank}") else out.stem
    return out.with_name(stem + ".shorts.json")


def write_index(path: Path, index: dict) -> None:
    path.write_text(json.dumps(index, indent=1, ensure_ascii=False) + "\n")
    lines = [f"# Shorts from {Path(index['video']).name}", "",
             "Ranked best first. Each short starts on a whole sentence and ends on one; the reasons are what "
             "the ranking read. It reads the words, not the picture: what is shown rather than said does not count.",
             ""]
    for note in index["notes"]:
        lines.append(f"- {note}")
    if index["notes"]:
        lines.append("")
    lines += ["| # | File | From | To | Plays | Title | Score |", "|---|---|---|---|---|---|---|"]
    for s in index["shorts"]:
        f = index["files"].get(str(s["rank"]))
        lines.append(f"| {s['rank']} | {Path(f).name if f else '(not made yet)'} | {s['from']:.2f} | {s['to']:.2f} | "
                     f"{s['seconds']:.1f} s | {s['title']} | {s['score']:.2f} |")
    for s in index["shorts"]:
        lines += ["", f"## {s['rank']}. {s['title']}", "",
                  f"- **Opens:** \"{s['opens']}\"", f"- **Closes:** \"… {s['closes']}\"",
                  "- **Why it ranks here:** " + "; ".join(s["why"]) + ".",
                  "- **Scores** (0 to 1): " + ", ".join(f"{k} {v:.2f}" for k, v in s["scores"].items()) + "."]
        lv = index.get("loudness", {}).get(str(s["rank"]))
        if lv:
            lines.append("- **Level:** " + ", ".join(f"{k} {v}" for k, v in lv.items() if not isinstance(v, (dict, list)))
                         + ".")
        heard = index["heard_again"].get(str(s["rank"]))
        if heard:
            lines.append(f"- **Heard again after the cut:** {heard}.")
    if index["runners_up"]:
        lines += ["", "## Runners-up, not made", "",
                  "Stretches that scored next and do not overlap the shorts above. To make one, give its seconds "
                  "in a plan (`short.from`, `short.to`) and run with `--plan`.", ""]
        for s in index["runners_up"]:
            lines.append(f"- {s['from']:.2f}-{s['to']:.2f} s ({s['seconds']:.0f} s, score {s['score']:.2f}): "
                         f"\"{s['opens']}\"; " + "; ".join(s["why"][1:3]) + ".")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")
