"""clean-cut: dead air, retakes and false starts cut out, captions on, framed 9:16.

What is cut, and how:

- **Dead air.** A pause longer than `pause` seconds (read from the sound, not from the
  transcriber, which reports almost every gap as zero) is shortened to a breath: a little after
  the last word, so it ends naturally, and a little before the next, so its first sound is whole.
  Pauses shorter than `pause` are the speaker's own rhythm and are left alone.
- **Retakes and false starts.** A phrase begun and then said again within a few seconds keeps only
  its last take: "so the product is- so the product is really fast" keeps "so the product is really
  fast", and a line said three times keeps the third. The first attempt must be mostly said again,
  must break off (a cut-off word, a comma, an "um", or a real pause), and must not run across a
  full stop unless every word of it is said again. A list ("most ads, most people")
  repeats one word only and is never taken for a retake. A word said twice in a stammer ("the
  the") keeps one; words repeated on purpose ("very very", "no no") are kept.
- **Fillers** (um, uh, erm) when `fillers=cut`.
- **Asides** when `asides=cut`: the sentences where the speaker talks about the recording rather
  than to the viewer ("Testing, testing.", "Give me a second.", "Let me start again.", "I've got
  the audio set up."), found by the core (`asides.asides`). Each is a cut of its own, listed with
  what it is, so one can be put back by deleting it from the plan and running again with `--plan`.

Every cut lands where the voice is quiet, at a word boundary, and the two sides are joined with a
20 ms crossfade, so nothing clicks and no word is clipped. After the render the output is heard
again, word by word, against the words that were meant to be kept: any word lost or chopped at a
cut is reported (the over-cut check).
"""

from __future__ import annotations

import difflib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..core import framing as framing_mod
from ..core import words as wordsmod
from ..core.asides import Aside
from ..core.asides import asides as find_asides
from ..core.moments import LONGEST_TAKE, ON_PURPOSE, find_retakes  # noqa: F401 (kept importable from here)
from ..core.moments import RETAKE_WINDOW as WINDOW  # noqa: F401
from ..core.timeline import Timeline
from ..core.words import Word, pause_after
from . import captions as cap
from . import options as O

NAME = "clean-cut"

OPTIONS = {
    "pause": O.Number(0.15, 3.0, "pauses longer than this many seconds are shortened to a breath"),
    "keep_retakes": O.Switch("on keeps every take; off keeps only the last take of a phrase said twice"),
    "fillers": O.Choice(("cut", "keep"), "cut or keep um, uh and erm"),
    "asides": O.Choice(("cut", "keep"), "cut or keep talk about the recording itself: a mic check, "
                                        "\"give me a second\", \"let me start again\""),
    "captions": O.Switch("on or off"),
    "look": cap.OPTIONS["look"],
    "color": cap.OPTIONS["color"],
    "position": cap.OPTIONS["position"],
    "max_words": cap.OPTIONS["max_words"],
    "frame": O.Choice(("9:16", "keep"), "9:16 follows the face when the clip is wider; keep leaves its shape"),
    "size": cap.OPTIONS["size"],
    "fix": cap.OPTIONS["fix"],
}

# The breath left where a pause is shortened: after the last word, and before the next.
TAIL = 0.12
LEAD = 0.08
TAIL_SENTENCE = 0.18  # a full stop earns a slightly longer breath
LEAD_SENTENCE = 0.12
MIN_CUT = 0.12  # a cut that would take out less than this is not worth a jump in the picture
EDGE_KEEP_START = 0.10  # kept before the first word
EDGE_KEEP_END = 0.35  # kept after the last word
FADE = 0.02  # the crossfade at every join

FILLERS_CUT = wordsmod.HESITATIONS


@dataclass
class Removal:
    start: float
    end: float
    reason: str
    words: str = ""
    detail: str = ""
    apart: bool = False  # one of an aside's cuts: kept apart from the cut beside it, never merged

    @property
    def seconds(self) -> float:
        return self.end - self.start

    def as_dict(self) -> dict:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "seconds": round(self.seconds, 3),
                "reason": self.reason, "words": self.words, "detail": self.detail}


# ---- finding what to cut -------------------------------------------------------------------------

# Retakes and false starts are found by the core (`moments.find_retakes`), so the words clean-cut
# cuts are the words every picker passes over.


def _quiet_between(quiet, a: float, b: float) -> Optional[tuple[float, float]]:
    """The longest quiet stretch overlapping [a, b], clipped to it."""
    best = None
    for q0, q1 in quiet:
        lo, hi = max(q0, a), min(q1, b)
        if hi > lo and (best is None or hi - lo > best[1] - best[0]):
            best = (lo, hi)
    return best


def _cut_after(word: Word, until: float, quiet, env, sentence: bool, whole_gap: bool = False) -> float:
    """Where a cut may start after `word`: a breath into the quiet that follows it, or, if the next
    sound follows straight on, the quietest moment at the boundary. With `whole_gap` (what follows
    is a sentence cut whole, an aside), the quietest moment anywhere before `until`: the word's own
    end can be marked early when nothing quiet comes after it."""
    q = _quiet_between(quiet, word.end - 0.06, until)
    tail = TAIL_SENTENCE if sentence else TAIL
    if q is not None:
        return min(q[0] + tail, q[1])
    return env.quietest(word.end - 0.03, until if whole_gap else min(until, word.end + 0.05))


def _cut_before(word: Word, since: float, quiet, env, sentence: bool, whole_gap: bool = False) -> float:
    q = _quiet_between(quiet, since, word.start + 0.06)
    lead = LEAD_SENTENCE if sentence else LEAD
    if q is not None:
        return max(q[1] - lead, q[0])
    return env.quietest(since if whole_gap else max(since, word.start - 0.05), word.start + 0.02)


def _apart(r: Removal, idx: list[int], words: list[Word], gone: dict, aside_of: dict, quiet, env,
           kept_next: Optional[Word]) -> list[Removal]:
    """A removal that holds an aside, as one cut per aside: split where one aside (or what else is
    cut beside it) meets the next, in the quiet between them, so the cut report lists each aside
    with what it is and the plan can put one back on its own. The cuts touch, so the video is the
    same as with one. A removal without an aside is returned as it is."""
    if not any(x in aside_of for x in idx):
        return [r]

    def cause(x: int) -> tuple:
        return ("aside", aside_of[x].first) if x in aside_of else ("other", gone[x])

    groups: list[list[int]] = []
    for x in idx:
        if groups and cause(groups[-1][-1]) == cause(x):
            groups[-1].append(x)
        else:
            groups.append([x])
    edges = [r.start]
    for g, h in zip(groups, groups[1:]):
        a, b = words[g[-1]].end, words[h[0]].start
        q = _quiet_between(quiet, a, b) if b > a else None
        t = (q[0] + q[1]) / 2 if q is not None else env.quietest(min(a, b) - 0.03, max(a, b) + 0.03)
        edges.append(min(max(t, edges[-1]), r.end))
    edges.append(r.end)
    out = []
    for g, s, e in zip(groups, edges, edges[1:]):
        said = " ".join(words[x].text for x in g)
        a = aside_of.get(g[0])
        if a is not None:
            out.append(Removal(s, e, "aside", said, a.why, apart=True))
            continue
        reasons = [gone[x] for x in g]
        reason = max(set(reasons), key=reasons.count)
        detail = (f"the take kept starts at {kept_next.start:.2f} s"
                  if kept_next is not None and reason in ("retake", "false start", "stammer") else "")
        out.append(Removal(s, e, reason, said, detail, apart=True))
    return out


def plan_cuts(clip, opts: dict) -> tuple[list[Removal], list[Word]]:
    """What to remove, and the words that stay."""
    words = clip.words
    quiet = clip.quiet
    env = clip.envelope
    duration = clip.info.frames / clip.info.rate
    gone: dict[int, str] = {}
    restarts: dict[int, int] = {}
    aside_of: dict[int, Aside] = {}
    if opts.get("asides", "cut") == "cut":
        for a in find_asides(words, quiet):
            for x in range(a.first, a.last + 1):
                gone[x] = "aside"
                aside_of[x] = a
    if not opts["keep_retakes"]:
        for first, last, restart, reason in find_retakes(words, quiet):
            for x in range(first, last + 1):
                gone.setdefault(x, reason)
                restarts.setdefault(x, restart)
    if opts["fillers"] == "cut":
        for x, w in enumerate(words):
            if w.key in FILLERS_CUT:
                gone.setdefault(x, "filler")
    kept_idx = [x for x in range(len(words)) if x not in gone]
    removals: list[Removal] = []
    if not kept_idx:
        return removals, []
    kept = [words[x] for x in kept_idx]

    # Before the first word and after the last.
    first, last = kept[0], kept[-1]
    head_cut = first.start - EDGE_KEEP_START
    lead_words = [words[x] for x in range(kept_idx[0])]
    if any(x in aside_of for x in range(kept_idx[0])):
        # An aside just before the first word kept: the cut ends after it, however close the two are.
        head_cut = max(head_cut, _cut_after(lead_words[-1], first.start, quiet, env, True, whole_gap=True))
    if head_cut >= MIN_CUT:
        removals += _apart(Removal(0.0, head_cut, "dead air at the start" if not lead_words else gone[0],
                                   " ".join(w.text for w in lead_words),
                                   f"the take kept starts at {first.start:.2f} s" if lead_words else ""),
                           list(range(kept_idx[0])), words, gone, aside_of, quiet, env, first)
    tail_cut = last.end + EDGE_KEEP_END
    trail_words = [words[x] for x in range(kept_idx[-1] + 1, len(words))]
    if any(x in aside_of for x in range(kept_idx[-1] + 1, len(words))):
        # An aside just after the last word kept: the cut starts before it, however close the two are.
        tail_cut = min(tail_cut, _cut_before(trail_words[0], last.end, quiet, env, True, whole_gap=True))
    if duration - tail_cut >= MIN_CUT or trail_words:
        tail_cut = min(tail_cut, duration)
        if duration - tail_cut > 1e-3:
            removals += _apart(Removal(tail_cut, duration, "dead air at the end" if not trail_words
                                       else gone[kept_idx[-1] + 1], " ".join(w.text for w in trail_words)),
                               list(range(kept_idx[-1] + 1, len(words))), words, gone, aside_of, quiet, env, None)

    # Between each pair of kept words.
    for a, b in zip(kept_idx, kept_idx[1:]):
        p, n = words[a], words[b]
        between = list(range(a + 1, b))
        sentence = p.ends_sentence
        if between:
            r0, r1 = words[between[0]], words[between[-1]]
            whole = any(x in aside_of for x in between)  # a sentence cut whole: its edges anywhere in the gap
            cut_in = _cut_after(p, r0.start + 0.03, quiet, env, sentence, whole_gap=whole)
            cut_out = _cut_before(n, r1.end - 0.03, quiet, env, sentence, whole_gap=whole)
            reasons = [gone[x] for x in between]
            reason = max(set(reasons), key=reasons.count)
            said = " ".join(words[x].text for x in between)
            detail = f"the take kept starts at {n.start:.2f} s" if reason in ("retake", "false start", "stammer") else ""
            if cut_out - cut_in >= 0.04:
                removals += _apart(Removal(cut_in, cut_out, reason, said, detail), between, words, gone, aside_of,
                                   quiet, env, n)
            continue
        gap = pause_after(words, a, quiet)
        if gap <= opts["pause"]:
            continue
        cut_in = _cut_after(p, n.start, quiet, env, sentence)
        cut_out = _cut_before(n, p.end, quiet, env, sentence)
        if cut_out - cut_in >= MIN_CUT:
            removals.append(Removal(cut_in, cut_out, "pause", "", f"a {gap:.2f} s pause made {gap - (cut_out - cut_in):.2f} s"))
    removals = _tidy(removals, kept, duration)
    return removals, kept


def _tidy(removals: list[Removal], kept: list[Word], duration: float) -> list[Removal]:
    """In order, inside the clip, never over a kept word, and never leaving a sliver between two cuts."""
    out: list[Removal] = []
    for r in sorted(removals, key=lambda r: r.start):
        r.start, r.end = max(0.0, r.start), min(duration, r.end)
        for w in kept:  # a cut never takes any part of a word that stays
            if w.start < r.end and r.start < w.end:
                if w.start - r.start >= r.end - w.end:
                    r.end = w.start
                else:
                    r.start = w.end
        if r.end - r.start < 0.04:
            continue
        if (out and r.start - out[-1].end < 0.08 and not any(out[-1].end <= w.start < r.start for w in kept)
                and not (r.apart and out[-1].apart)):  # an aside's cut stays its own, to be put back on its own
            prev = out[-1]
            prev.end = max(prev.end, r.end)
            if r.reason != prev.reason:
                prev.reason = f"{prev.reason} + {r.reason}"
            prev.words = " ".join(x for x in (prev.words, r.words) if x)
            continue
        out.append(r)
    return out


def _from_plan(plan: dict, clip, duration: float) -> tuple[list[Removal], list[Word]]:
    from . import PlanError

    raw = plan.get("removals")
    if not isinstance(raw, list):
        raise PlanError("the plan has no removals list; run once without --plan to get one to edit")
    removals = []
    for r in raw:
        try:
            s, e = float(r["start"]), float(r["end"])
        except (KeyError, TypeError, ValueError):
            raise PlanError(f"a removal needs a start and an end in seconds: {r!r}") from None
        if not (0 <= s < e <= duration + 1e-3):
            raise PlanError(f"the removal {s:.2f}-{e:.2f} s is not inside the clip (0-{duration:.2f} s)")
        removals.append(Removal(s, e, str(r.get("reason", "by hand")), str(r.get("words", "")), str(r.get("detail", ""))))
    removals.sort(key=lambda r: r.start)
    for a, b in zip(removals, removals[1:]):
        if b.start < a.end:
            raise PlanError(f"two removals overlap at {b.start:.2f} s; merge them or move one")
    kept = [w for w in clip.words if not any(r.start <= (w.start + w.end) / 2 < r.end for r in removals)]
    return removals, kept


# ---- the effect ------------------------------------------------------------------------------

def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from . import Job

    info = clip.info
    fps = info.rate
    duration = info.frames / fps
    if plan is not None:
        removals, kept = _from_plan(plan, clip, duration)
    else:
        removals, kept = plan_cuts(clip, opts)
    tl = Timeline.keeping(duration, [(r.start, r.end) for r in removals])
    fr = framing_mod.plan(info.width, info.height, fps, info.frames, clip.face, shape=opts["frame"],
                          size=O.size_of(opts["size"]))
    # The kept words where they now play.
    out_words = []
    for w in kept:
        s, e = tl.to_output_nearest(w.start), tl.to_output_nearest(w.end)
        out_words.append(Word(w.text, s, max(e, s + 0.02), w.prob))
    out_quiet = tl.map_spans(clip.quiet)
    n_out = tl.output_frames(fps)
    caps = None
    checks: dict = {}
    if opts["captions"]:
        caps = cap.Captions(out_words, out_quiet, (fr.out_w, fr.out_h), fps, n_out,
                            cap.head_in_output(clip, tl, fr), look=opts["look"], color=opts["color"],
                            position=opts["position"], max_words=opts["max_words"])
        checks = caps.report()
    # Every aside found, and whether this render cuts it (an aside put back through the plan is kept).
    found = [{"start": round(a.start, 3), "end": round(a.end, 3), "said": a.said, "what": a.why,
              "cut": all(any(r.start <= (w.start + w.end) / 2 < r.end for r in removals)
                         for w in clip.words[a.first:a.last + 1])}
             for a in find_asides(clip.words, clip.quiet)]
    removed = sum(r.seconds for r in removals)
    by_reason: dict[str, float] = {}
    for r in removals:
        by_reason[r.reason] = by_reason.get(r.reason, 0.0) + r.seconds
    summary = {"cuts": len(removals), "seconds_in": round(duration, 2), "seconds_out": round(tl.duration, 2),
               "seconds_removed": round(removed, 2),
               "by_reason": {k: round(v, 2) for k, v in sorted(by_reason.items(), key=lambda kv: -kv[1])},
               "framing": fr.note}
    out_plan = {"effect": NAME, "clip": clip.path, "options": opts,
                "output": {"width": fr.out_w, "height": fr.out_h, "fps": round(fps, 3),
                           "seconds": round(tl.duration, 3), "framing": fr.note},
                "summary": summary,
                "removals": [r.as_dict() for r in removals],
                "asides": found,
                "kept_words": [{"w": w.text, "s": round(w.start, 3), "e": round(w.end, 3)} for w in kept],
                "unsure": cap.unsure_words(clip),
                "cards": caps.plan() if caps else []}

    def stills() -> list[float]:
        """The first line, a moment just after a cut in the middle, and the last line."""
        times = []
        if caps and caps.cards:
            c0 = caps.cards[0]
            times.append(min(c0.end - 0.5 / fps, max(c0.words[-1].start + 0.1, (c0.start + c0.end) / 2)))
        joins = tl.joins
        if joins:
            mid = joins[len(joins) // 2]
            times.append(min(tl.duration - 0.05, mid + 0.15))
        if caps and len(caps.cards) > 2:
            c = caps.cards[len(caps.cards) * 2 // 3]
            times.append(min(c.end - 0.5 / fps, max(c.words[-1].start + 0.1, (c.start + c.end) / 2)))
        if caps and caps.cards:
            c1 = caps.cards[-1]
            times.append(min(c1.end - 0.5 / fps, max(c1.words[-1].start + 0.1, (c1.start + c1.end) / 2)))
        return sorted(set(round(t, 3) for t in times)) or [tl.duration / 2]

    def after(result: dict, out: Path) -> dict:
        check = overcut_check(clip, kept, tl, out, opts.get("fix"))
        report = write_report(out, clip, summary, removals, check)
        return {"cut report": str(report), "over-cut check": check["verdict"]}

    head = {"cuts": f"{len(removals)} ({removed:.1f} s of {duration:.1f} s)"}
    if found:
        n_cut = sum(a["cut"] for a in found)
        head["asides"] = (f"{len(found)} found, {n_cut} cut" + (f", {len(found) - n_cut} kept" if n_cut < len(found) else "")
                          + ": " + "; ".join(f"{a['start']:.2f} s \"{a['said']}\"" for a in found))
    checks = {**head, **checks}
    return Job(NAME, clip, tl, fr, caps.draw if caps else None, out_plan, checks=checks,
               still_times=stills, after=after)


# ---- the over-cut check ----------------------------------------------------------------------------

# Below this share of the kept words heard again, the first listen of the output is taken to have
# stopped early (a working run hears 96 % or more) and the output is heard again in pieces. It only
# decides whether to listen again: what is lost or chopped is judged as before, on what is heard.
HEARD_AGAIN = 0.8


def _matched(a: list[str], b: list[str]) -> int:
    """How many words of `a` are heard, in order, in `b`."""
    return sum(m.size for m in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks())


def _heard_in_pieces(out: Path, bounds: list[float], fix: Optional[str]) -> list[Word]:
    """The output's words heard piece by piece: each stretch between two of `bounds` (output
    seconds, at joins cut in silence) is written to a temporary file with its sound untouched (PCM),
    heard on its own (`Clip.heard`, so with the second listen, as the whole output is), and its words'
    times are moved back to the output's."""
    import dataclasses
    import subprocess
    import tempfile

    from ..core.clip import Clip

    words: list[Word] = []
    with tempfile.TemporaryDirectory(prefix="fx-overcut-") as tmp:
        for k, (t0, t1) in enumerate(zip(bounds, bounds[1:])):
            if t1 - t0 < 0.05:
                continue
            piece = Path(tmp) / f"piece-{k:02d}.mkv"
            run = subprocess.run(
                ["ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", f"{t0:.4f}", "-i", str(out),
                 "-t", f"{t1 - t0:.4f}", "-map", "0:v:0", "-map", "0:a:0", "-vf", "scale=-2:64",
                 "-c:v", "libx264", "-preset", "ultrafast", "-crf", "40", "-threads", "4",
                 "-c:a", "pcm_s16le", str(piece)],
                capture_output=True)
            if run.returncode != 0:
                continue
            for w in wordsmod.pull_starts(wordsmod.sure(Clip(piece, fix=fix).heard)):
                if w.key:
                    words.append(dataclasses.replace(w, start=w.start + t0, end=w.end + t0))
    return words


def overcut_check(clip, kept: list[Word], tl: Timeline, out: Path, fix: Optional[str]) -> dict:
    """Hear the output again and compare it, word by word, with the words meant to be kept.

    A word missing or changed within a quarter second of a join is a suspect, and the sound
    decides: if both sides of that join were cut in silence, no part of any word was taken there,
    and the difference is the transcriber hearing the same sound another way (in a new context it
    may hear "them" as "demo"). If either side was cut while the voice was sounding, the word is
    reported as lost or chopped. Differences away from every join are listed apart.
    """
    # The output's words with the same lateness correction the kept words had, so the timing
    # comparison is like with like.
    from ..core.clip import Clip

    joins = tl.joins
    near = 0.25
    quiet = clip.quiet
    # Was each join cut in silence on both sides? (the end of one kept stretch and the start of the
    # next, each inside a stretch where nobody speaks: the same quiet the cuts were placed in)
    def in_quiet(t: float) -> bool:
        return any(q0 + 0.005 <= t <= q1 - 0.005 for q0, q1 in quiet)

    silent = {}
    for (a0, a1), (b0, b1), j in zip(tl.segments, tl.segments[1:], joins):
        silent[j] = in_quiet(a1) and in_quiet(b0)

    a = [w.key for w in kept]
    heard = [w for w in wordsmod.pull_starts(wordsmod.sure(Clip(out, fix=fix).heard)) if w.key]
    pieces = 0
    first_matched = _matched(a, [w.key for w in heard])
    if kept and first_matched < HEARD_AGAIN * len(kept):
        # The first listen heard far fewer of the kept words than a working run does (every run on
        # the test clips heard 96 % or more). The transcriber can end its window early on sound that
        # is all there: on 33 with its asides cut it heard one sentence of the first 30 s, 18 words in
        # all. So the output is heard again in pieces, split where both sides of a join were cut in
        # silence (no word is split there), and the pieces' words are compared as before.
        cuts = [j for j in joins if silent[j]]
        again = _heard_in_pieces(out, [0.0] + cuts + [tl.duration], fix)
        if _matched(a, [w.key for w in again]) > first_matched:
            heard, pieces = again, len(cuts) + 1
    b = [w.key for w in heard]

    def by_join(t: float) -> Optional[float]:
        if not joins:
            return None
        j = min(joins, key=lambda x: abs(x - t))
        return j if abs(j - t) <= near else None

    lost, chopped, extra, elsewhere = [], [], [], []
    offsets = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                expect = tl.to_output_nearest(kept[i].start)
                offsets.append(heard[j].start - expect)
            continue
        for i in range(i1, i2):
            t = tl.to_output_nearest(kept[i].start)
            t_end = tl.to_output_nearest(kept[i].end)
            jn = by_join(t) if by_join(t) is not None else by_join(t_end)
            item = {"word": kept[i].text, "at": round(t, 2)}
            if jn is None or silent.get(jn):
                elsewhere.append(dict(item, heard_as=" ".join(heard[j].text for j in range(j1, j2)),
                                      **({"next_to_silent_cut": round(jn, 2)} if jn is not None else {})))
            elif op == "delete":
                lost.append(dict(item, join=round(jn, 2)))
            else:
                chopped.append(dict(item, join=round(jn, 2), heard_as=" ".join(heard[j].text for j in range(j1, j2))))
        if op == "insert":
            for j in range(j1, j2):
                jn = by_join(heard[j].start)
                item = {"word": heard[j].text, "at": round(heard[j].start, 2)}
                (extra if jn is not None and not silent.get(jn) else elsewhere).append(item)
    offsets.sort()
    median = offsets[len(offsets) // 2] if offsets else 0.0
    matched = len(offsets)
    if lost or chopped:
        verdict = (f"{len(lost)} word(s) lost and {len(chopped)} chopped at a cut: "
                   + ", ".join(f"'{x['word']}' at {x['at']} s" for x in lost + chopped))
    elif joins:
        quiet_joins = sum(1 for v in silent.values() if v)
        verdict = (f"no word lost or chopped at any of the {len(joins)} joins, {quiet_joins} of them cut in silence "
                   f"on both sides ({matched} of {len(kept)} kept words heard again)")
    else:
        verdict = f"nothing was cut inside the take ({matched} of {len(kept)} kept words heard again)"
    if extra:
        verdict += f"; {len(extra)} sound(s) of a removed word left at a cut"
    if pieces:
        verdict += (f"; heard again in {pieces} pieces split at the joins cut in silence, because the "
                    f"first listen heard only {first_matched} of {len(kept)}")
    return {"verdict": verdict, "heard_in_pieces": pieces, "first_listen_matched": first_matched, "lost": lost, "chopped": chopped, "left_at_cut": extra,
            "joins_cut_in_silence": sum(1 for v in silent.values() if v), "joins": len(joins),
            "heard_differently_away_from_cuts": elsewhere, "matched": matched, "kept": len(kept),
            "median_start_offset_s": round(median, 3)}


def write_report(out: Path, clip, summary: dict, removals: list[Removal], check: dict) -> Path:
    path = out.with_suffix(".cuts.md")
    lines = [f"# Cut report: {Path(clip.path).name}", "",
             f"{summary['seconds_in']:.2f} s in, {summary['seconds_out']:.2f} s out: "
             f"{summary['cuts']} cuts took out {summary['seconds_removed']:.2f} s. {summary['framing'].capitalize()}.", ""]
    if summary["by_reason"]:
        lines.append("By reason: " + ", ".join(f"{k} {v:.2f} s" for k, v in summary["by_reason"].items()) + ".")
        lines.append("")
    lines += ["| # | From | To | Seconds | Why | What was said |", "|---|---|---|---|---|---|"]
    for n, r in enumerate(removals, 1):
        said = (r.words or "").replace("|", "/")
        why = r.reason + (f" ({r.detail})" if r.detail else "")
        lines.append(f"| {n} | {r.start:.2f} | {r.end:.2f} | {r.seconds:.2f} | {why} | {said} |")
    if any(r.reason == "aside" for r in removals):
        lines += ["", "An aside is talk about the recording rather than to the viewer. To keep one, delete its row from "
                      "the plan's `removals` and run again with `--plan`; `asides=keep` keeps them all."]
    lines += ["", "## The over-cut check", "", f"The output was transcribed again and compared, word by word, with the words kept: {check['verdict']}.", ""]
    if check["lost"] or check["chopped"]:
        for x in check["lost"]:
            lines.append(f"- lost: '{x['word']}' at {x['at']} s (cut at {x['join']} s)")
        for x in check["chopped"]:
            lines.append(f"- chopped: '{x['word']}' at {x['at']} s, heard as '{x['heard_as']}' (cut at {x['join']} s)")
        lines.append("")
    if check["left_at_cut"]:
        lines.append("Left at a cut: " + ", ".join(f"'{x['word']}' at {x['at']} s" for x in check["left_at_cut"]) + ".")
        lines.append("")
    if check["heard_differently_away_from_cuts"]:
        def said(x: dict) -> str:
            heard_as = f", heard as '{x['heard_as']}'" if x.get("heard_as") else ""
            beside = f" (beside the cut at {x['next_to_silent_cut']} s, made in silence)" if "next_to_silent_cut" in x else ""
            return f"'{x['word']}' at {x['at']} s{heard_as}{beside}"
        lines.append("Heard differently where no sound was cut (the transcriber, not the cut): "
                     + "; ".join(said(x) for x in check["heard_differently_away_from_cuts"]) + ".")
        lines.append("")
    lines.append(f"Words heard again in place: {check['matched']} of {check['kept']}; the median start "
                 f"differs from the plan by {check['median_start_offset_s'] * 1000:.0f} ms.")
    path.write_text("\n".join(lines) + "\n")
    return path
