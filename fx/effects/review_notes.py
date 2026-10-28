"""review-notes: the review before the final render. A cheap preview to watch, notes at exact seconds,
and a final that is refused while any note is open.

    python -m fx.run review-notes <cut.mp4>                         # the preview, the grid, an empty ledger
    python -m fx.run review-notes <cut.mp4> --opt memo=<memo.m4a>   # a voice memo's notes, at the times it names
    python -m fx.run review-notes <new.mp4> --opt notes=<cut.notes.md>   # the same notes, checked on a new render
    python -m fx.run review-notes <cut.mp4> --opt final=on          # the final, only when nothing is open

What it writes beside the cut:

- `<cut>.review.mp4`: the cut at a third of its size, at most 30 frames a second, with the time under
  the picture (in a band of its own, so it never covers anything) and a mark for every note;
- `<cut>.review.jpg`: a grid of frames at round seconds, then one tile per note: its frame on this
  cut, its words, and what the check measured there;
- `<cut>.notes.md`: the ledger. Notes are added as lines under "New notes" (`- 4.2 s: the caption
  covers my chin`) and the command is run again.

What happens to a note:

1. **It is placed.** Its second, the words said around it (so the same moment is found again in a
   new render, even after cuts), and what the plan beside the cut (`<cut>.plan.json`) had on screen
   then: "caption 7, 'about the actual product' (3.94-5.10 s)". A note may name the thing instead of
   the second: "the second caption", "the headline".
2. **It is read for a fault a check can measure:** words over the face, outside the safe zone, black
   or frozen frames, flashing, silence, a pause, the loudness. The picture checks are qc-check's own
   (`qc_check.check`), run once for every note. A note the checks cannot measure ("make the headline
   bigger") is a note for the eye.
3. **It starts a sweep.** A new note about a measurable fault sends the same check over the whole
   cut, and every other place it finds becomes a note of its own ("found by the sweep for #1").
4. **It closes only on proof.** A note whose fault the check saw closes when a later render no longer
   has it there (or no longer has the moment at all). A note the checks cannot see closes when it is
   ticked, after looking (`- [x]`); a tick on a fault still measured is undone, and says so. `- [-]`
   declines a note, with the reason under it. A note closed by a check reopens if the fault comes back.

`final=on` re-renders the cut from its plan at full quality (`<cut>-final.mp4`), so the renders made
while reviewing may be drafts (`--crf 28`). It is refused, exit 1, while any note is open.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from ..core import paths, video
from ..core import words as wordsmod
from ..core.amounts import SMALL, TENS
from ..core.paths import FONTS
from . import options as O

NAME = "review-notes"
REPORTS = True
MATRIX: list = []  # it renders no effect: tests/review_notes.py and the lab runs test it

OPTIONS = {
    "notes": O.Text("the ledger to carry on (a .notes.md), when this video is a new render of a cut already "
                    "reviewed; by default the ledger beside the video"),
    "memo": O.Text("a voice memo (a sound or a video file, a full path): each thing said with a time becomes a "
                   "note at that time (\"at four seconds the caption covers my chin\")"),
    "source": O.Text("the clip the cut was made from (a full path), for the face and the safe zone; by default "
                     "the clip its plan names"),
    "pause": O.Number(0.2, 5.0, "the seconds of quiet a pause note measures as a pause"),
    "final": O.Switch("on: render the final from the cut's plan at full quality, refused while any note is open"),
}

VERSION = 1
NEAR = 0.5  # seconds either side of a note's second that its check looks at, when no item is named
SWEEP_MOST = 10  # notes one sweep adds at most; the ledger says when it found more
FLICKER = 0.15  # seconds: a found stretch shorter than this is not swept into a note (it is most often the
#                 edge of a cut, where the picture checks line up least well); the ledger counts them
PREVIEW_SIDE = 360  # the preview's shorter side, in pixels
PREVIEW_FPS = 30.0  # the preview's most frames a second
GRID_MOST = 18  # frames in the grid's time row at most

# ---- what a note is about --------------------------------------------------------------------------------
#
# Each fault a check can measure, with the words people use for it. The first that matches wins, so the
# narrow ones come first ("a quiet bit" is a pause, "too quiet" is the loudness).

KINDS = {
    "pause": ("a pause", r"\bpauses?\b|dead air|\bgaps?\b|quiet bit|long wait|\bwaiting\b|\bsilent bit\b"),
    "silence": ("silence", r"\bsilen(t|ce)\b|no sound|sound (drops|cuts) out|can'?t hear (me|anything)|\bmuted?\b"),
    "loudness": ("the loudness", r"barely hear|hear myself|too (quiet|loud|soft)|\bvolume\b|\bloudness\b|\bclipp(ing|ed)\b|distort|"
                                 r"can'?t hear|hard to hear|\blouder\b|\bquieter\b"),
    "colour": ("washed-out colour", r"washed[ -]out|\bflat\b|\bgr[ae]y(ish)?\b|\bdull\b|\bfaded\b|\bhdr\b"),
    "black": ("black frames", r"\bblack\b"),
    "frozen": ("frozen frames", r"\bfr(oze|eezes?|ozen)\b|\bstuck\b|\bstutters?\b|picture stops"),
    "flash": ("flashing", r"\bflash(es|ing)?\b|\bflicker"),
    "face": ("words over the face", r"\b(face|chin|mouth|eyes?|nose|forehead|lips|head|cheeks?)\b|covers? me\b|"
                                    r"over me\b|on top of me\b"),
    "zone": ("the safe zone", r"cut off|off (the )?(edge|screen)|\bedges?\b|\bbuttons?\b|safe zone|too low|"
                              r"too high|\bmargin|under the (caption|app)|hidden by"),
}
EYE = "for the eye"
# The checks of qc-check each kind reads, and whether it is one reading for the whole cut.
QC = {"face": (("face",), False), "zone": (("safe zone",), False), "black": (("black frames",), False),
      "frozen": (("frozen frames",), False), "flash": (("flashes",), False),
      "silence": (("silence", "sound"), False), "loudness": (("loudness", "true peak", "clipping"), True),
      "colour": (("colour",), True)}
SWEEPS = {"face", "zone", "black", "frozen", "flash", "silence", "pause"}


def kind_of(text: str) -> str:
    low = text.lower()
    for k, (_, pattern) in KINDS.items():
        if re.search(pattern, low):
            return k
    return EYE


def kind_name(kind: str) -> str:
    return KINDS[kind][0] if kind in KINDS else "for the eye (no check can measure it)"


# ---- times, typed and spoken -----------------------------------------------------------------------------

SECOND_WORDS = {"s", "sec", "secs", "second", "seconds"}
MINUTE_WORDS = {"m", "min", "mins", "minute", "minutes"}
LEADS = {"at", "around", "about", "near", "from", "by", "@"}
_DIGITS = {**{str(i): i for i in range(10)}, **{w: v for w, v in SMALL.items() if v < 10}, "oh": 0}


def _bare(token: str) -> str:
    return re.sub(r"^[^\w@]+|[^\w]+$", "", token.lower().replace("’", "'"))


def _number(toks: list[str], i: int) -> Optional[tuple[float, int, bool]]:
    """The number that starts at toks[i]: (value, index after it, whether it carried its own unit).
    Figures ("4.2", "12s", "0:12") and words ("four point two", "twenty three")."""
    if i >= len(toks):
        return None
    t = _bare(toks[i])
    m = re.fullmatch(r"(\d+):(\d{1,2}(?:\.\d+)?)", t)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2)), i + 1, True
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(s|sec|secs|seconds?)?", t)
    j = i + 1
    if m:
        v = float(m.group(1))
        if m.group(2):
            return v, j, True
    elif t in TENS:
        v = float(TENS[t])
        if j < len(toks) and _bare(toks[j]) in SMALL and 0 < SMALL[_bare(toks[j])] < 10:
            v += SMALL[_bare(toks[j])]
            j += 1
    elif t in SMALL:
        v = float(SMALL[t])
    else:
        return None
    # "four point two", "4 point 2"
    if j + 1 < len(toks) and _bare(toks[j]) == "point" and _bare(toks[j + 1]) in _DIGITS:
        digits = ""
        k = j + 1
        while k < len(toks) and _bare(toks[k]) in _DIGITS and len(digits) < 3:
            digits += str(_DIGITS[_bare(toks[k])])
            k += 1
        v = float(f"{int(v)}.{digits}")
        j = k
    return v, j, False


@dataclass
class TimeSaid:
    seconds: float
    start: int  # the token where the phrase begins
    end: int  # the token after it


def times_in(toks: list[str], typed: bool = False) -> list[TimeSaid]:
    """Every time named in `toks`: "at 4.2 s", "at four seconds", "twelve seconds in", "0:12",
    "at one minute ten". A bare number counts only after "at" (or, typed, at the start of a line)."""
    out = []
    i = 0
    while i < len(toks):
        r = _number(toks, i)
        if r is None:
            i += 1
            continue
        v, j, unit = r
        colon = ":" in _bare(toks[i])
        nxt = _bare(toks[j]) if j < len(toks) else ""
        if nxt in MINUTE_WORDS and not unit:
            j += 1
            if j < len(toks) and _bare(toks[j]) == "and":
                j += 1
            r2 = _number(toks, j)
            secs = 0.0
            if r2 is not None:
                secs, j, _ = r2
            v = v * 60 + secs
            unit = True
            if j < len(toks) and _bare(toks[j]) in SECOND_WORDS:
                j += 1
        elif nxt in SECOND_WORDS and not unit:
            unit = True
            j += 1
        lead = i > 0 and _bare(toks[i - 1]) in LEADS
        if not unit and nxt == "point":  # "at one point the caption…" is no time
            i = j
            continue
        if unit or lead or colon or (typed and i == 0):
            if j < len(toks) and _bare(toks[j]) in ("in", "mark"):
                j += 1
            out.append(TimeSaid(v, i - 1 if lead else i, j))
            i = j
        else:
            i += 1
    return out


_JOINERS = {"and", "then", "also", "so", "oh", "um", "uh", "plus", "okay", "ok", "right"}


def _tidy(text: str) -> str:
    text = re.sub(r"^[\s,;:.\-–—]+|[\s,;:.\-–—]+$", "", text)
    toks = text.split()
    while toks and _bare(toks[0]) in _JOINERS:
        toks = toks[1:]
    while toks and _bare(toks[-1]) in _JOINERS:
        toks = toks[:-1]
    text = " ".join(toks).strip(" ,;:-")
    return text[:1].upper() + text[1:] if text else ""


def _meaningful(text: str) -> bool:
    return bool(text) and (kind_of(text) != EYE or names_a_thing(text))


def notes_in(toks: list[str], typed: bool = False) -> list[tuple[Optional[float], str]]:
    """The notes in one line or one spoken sentence, each (seconds or None, what it says). A sentence
    that names two times is two notes; the words of each are the ones after its time when the time
    opens the sentence, and the ones before it otherwise ("cut the pause at twelve seconds")."""
    found = times_in(toks, typed)
    if not found:
        text = _tidy(" ".join(toks))
        return [(None, text)] if text else []
    out = []
    seg_start = 0
    for k, t in enumerate(found):
        stop = found[k + 1].start if k + 1 < len(found) else len(toks)
        before = _tidy(" ".join(toks[seg_start:t.start]))
        after = _tidy(" ".join(toks[t.end:stop]))
        # The words that say what is wrong: after the time when they name a fault or a thing (or when
        # the time opens the sentence), else before it. "Okay, a couple of notes, at 26 seconds it
        # freezes" is about the freeze, not the preamble.
        if t.start == seg_start or not before or (_meaningful(after) and not _meaningful(before)):
            text = after
            seg_start = stop
        else:
            text = before
            seg_start = t.end
        if text:
            out.append((round(t.seconds, 2), text))
    return out


_BREAKS = {"and", "then", "also", "but", "plus"}
_OPENERS = {"the", "it", "there", "at", "my", "that", "this", "around", "about"}


def clauses(toks: list[str]) -> list[list[str]]:
    """A spoken sentence cut where it moves on to another thing: "…freezes while I'm talking, and the
    last caption has a typo" is two notes."""
    out = [[]]
    for i, t in enumerate(toks):
        b = _bare(t)
        prev_comma = i > 0 and toks[i - 1].endswith(",")
        nxt = _bare(toks[i + 1]) if i + 1 < len(toks) else ""
        if b in _BREAKS and out[-1] and (prev_comma or nxt in _OPENERS):
            out.append([])
        out[-1].append(t)
    return [c for c in out if c]


def names_a_thing(text: str) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in NOUN_WORDS.values())


def memo_notes(memo: Path) -> tuple[list[tuple[Optional[float], str, float]], str]:
    """A voice memo's notes: (seconds named, what was said, when in the memo), and its words as heard.
    A memo without a picture is given a blank one in a temporary folder, so the transcriber can read
    it; its words are kept beside the memo (`<memo>.words.json`) like any clip's."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=index",
                            "-of", "csv=p=0", str(memo)], capture_output=True, text=True)
    cache = paths.beside(memo, "words.json")
    if probe.returncode == 0 and probe.stdout.strip():
        ws = wordsmod.load(memo, out=cache)
    else:
        with tempfile.TemporaryDirectory(prefix="review-memo-") as tmp:
            clip = Path(tmp) / (memo.stem + ".mp4")
            subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i", "color=black:s=64x64:r=10",
                            "-i", str(memo), "-map", "0:v", "-map", "1:a:0", "-shortest", "-c:v", "libx264",
                            "-preset", "ultrafast", "-c:a", "aac", "-b:a", "96k", "-fflags", "+bitexact",
                            "-flags:v", "+bitexact", "-flags:a", "+bitexact", "-map_metadata", "-1", str(clip)],
                           check=True)
            ws = wordsmod.load(clip, out=cache)
    sentences: list[list[wordsmod.Word]] = []
    for w in ws:
        if not sentences or (sentences[-1] and (re.search(r"[.?!]$", sentences[-1][-1].text)
                                                 or w.start - sentences[-1][-1].end > 1.0)):
            sentences.append([])
        sentences[-1].append(w)
    out: list[tuple[Optional[float], str, float]] = []
    for s in sentences:
        for clause in clauses([w.text for w in s]):
            for at, text in notes_in(clause):
                if at is None and kind_of(text) == EYE and not names_a_thing(text):
                    # "It says in dash built": more of the note before it, or a preamble ("a couple of notes")
                    if out:
                        p_at, p_text, p_when = out[-1]
                        out[-1] = (p_at, f"{p_text}: {text[:1].lower()}{text[1:]}", p_when)
                    continue
                out.append((at, text, round(s[0].start, 2)))
    return out, " ".join(w.text for w in ws)


# ---- the plan: what was on screen ------------------------------------------------------------------------

NOUNS = {"cards": "caption", "headline": "headline", "parts": "chapter", "card": "card", "chart": "chart",
         "cluster": "widgets", "line": "hook", "cutaways": "cutaway", "sections": "layout", "transitions": "transition",
         "prompts": "prompt", "": "word"}
SKIP_KEYS = {"removals", "span", "timing", "push", "moves", "clips", "teaser", "options", "output", "summary",
             "kept_words", "sheet_times"}
NOUN_WORDS = {"caption": r"captions?|subtitles?|subs\b", "headline": r"headlines?|big text|title\b",
              "chapter": r"chapters?|part titles?", "card": r"\bcards?\b|call to action|\bcta\b|follow button",
              "chart": r"charts?|graphs?", "widgets": r"widgets?", "hook": r"\bhook\b|opening line",
              "cutaway": r"cutaways?|b-?roll", "layout": r"layouts?|split", "transition": r"transitions?",
              "prompt": r"prompts?|prompt box", "word": r"the word behind|word behind|big word"}
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
            "ninth": 9, "tenth": 10, "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "last": -1}
T0 = ("start", "on", "at", "from", "in", "t0")
T1 = ("end", "off", "to", "out", "t1")
LABELS = ("text", "title", "word", "broll", "layout")
HELD = 1.5  # seconds a card with only a landing time counts as on screen, for a note
CARD_EFFECTS = {"tier-list", "ranked-list", "checklist", "pictures-above-head"}


@dataclass
class Item:
    noun: str
    n: int  # its number among things of its kind, from 1
    start: float
    end: float
    label: str
    rect: Optional[list] = None  # where it was drawn, in the cut's pixels, when the plan says

    def say(self) -> str:
        what = f"{self.noun} {self.n}" + (f", \"{self.label}\"" if self.label else "")
        return f"{what} ({self.start:.2f}-{self.end:.2f} s)"

    def to_json(self) -> dict:
        return {"noun": self.noun, "n": self.n, "start": self.start, "end": self.end, "label": self.label}


def plan_items(plan: Optional[dict]) -> list[Item]:
    """The things the plan put on screen with their times: captions, a headline, cards, chapters."""
    if not plan:
        return []
    found: dict[str, list[Item]] = {}
    effect = plan.get("effect", "")

    def walk(o, top: str):
        if isinstance(o, dict):
            a = next((o[k] for k in T0 if isinstance(o.get(k), (int, float)) and not isinstance(o.get(k), bool)), None)
            b = next((o[k] for k in T1 if isinstance(o.get(k), (int, float)) and not isinstance(o.get(k), bool)), None)
            label = next((o[k] for k in LABELS if isinstance(o.get(k), str) and o.get(k)), "")
            if a is not None and (b is None or b <= a) and label and top not in SKIP_KEYS:
                b = a + HELD  # a card that lands and stays: its first moments are what a note is about
            if a is not None and b is not None and b > a and (label or top in NOUNS):
                noun = "card" if top == "cards" and effect in CARD_EFFECTS else NOUNS.get(top, top.rstrip("s") or "item")
                rect = o.get("rect") if isinstance(o.get("rect"), list) and len(o["rect"]) == 4 else None
                lst = found.setdefault(noun, [])
                lst.append(Item(noun, len(lst) + 1, round(float(a), 2), round(float(b), 2),
                                re.sub(r"\s*\|\s*", " ", label).strip()[:60], rect))
                return
            for k, v in o.items():
                if k not in SKIP_KEYS:
                    walk(v, top)
        elif isinstance(o, list):
            for v in o:
                walk(v, top)

    for k, v in plan.items():
        if k in SKIP_KEYS or k in ("effect", "clip"):
            continue
        if k in ("in", "out", "text") and plan.get("effect") == "words-behind":
            continue
        walk(v, k)
    if plan.get("effect") == "words-behind" and isinstance(plan.get("in"), (int, float)):
        found.setdefault("word", []).append(Item("word", 1, float(plan["in"]), float(plan.get("out", plan["in"])),
                                                 str(plan.get("text", ""))))
    return [it for lst in found.values() for it in lst]


def plan_words(plan: Optional[dict]) -> list[wordsmod.Word]:
    """The words a plan's captions carry, each with its time on the cut (`cards[].words`: w, s, e)."""
    if not plan or not isinstance(plan.get("cards"), list):
        return []
    out = {}
    for card in plan["cards"]:
        for w in card.get("words") or [] if isinstance(card, dict) else []:
            if isinstance(w, dict) and isinstance(w.get("w"), str) and isinstance(w.get("s"), (int, float)):
                out[round(float(w["s"]), 3)] = wordsmod.Word(w["w"], float(w["s"]), float(w.get("e", w["s"])))
    return [out[k] for k in sorted(out)]


def item_named(text: str, at: Optional[float], items: list[Item]) -> tuple[Optional[Item], bool]:
    """The item a note points at, and whether the note named it ("the second caption") rather than
    only being said while it was on screen."""
    low = text.lower()
    for noun, pattern in NOUN_WORDS.items():
        if not re.search(pattern, low):
            continue
        mine = [it for it in items if it.noun == noun]
        if not mine:
            continue
        m = re.search(r"\b(" + "|".join(ORDINALS) + r")\b\s+(?:" + pattern + r")", low) \
            or re.search(r"(?:" + pattern + r")\s*(?:#|number |no\.? ?)?(\d+)\b", low)
        if m:
            g = m.group(1)
            k = ORDINALS.get(g) if g in ORDINALS else int(g)
            if k == -1:
                return mine[-1], True
            if 1 <= k <= len(mine):
                return mine[k - 1], True
        if at is not None:
            near = [it for it in mine if it.start - 0.25 <= at <= it.end + 0.25]
            if near:
                return min(near, key=lambda it: abs((it.start + it.end) / 2 - at)), True
        elif len(mine) == 1:
            return mine[0], True
    if at is not None:
        on = [it for it in items if it.start <= at <= it.end]
        if on:
            return min(on, key=lambda it: it.end - it.start), False
    return None, False


CHANGES = {
    ("face", "caption"): "the captions move as one style: run the step again with `position=top`, or `max_words=2` "
                         "for a smaller block",
    ("zone", "caption"): "run the step again with `position=` another place, or `max_words=2`",
    ("face", "headline"): "run headline again with `position=` another place, or fewer words in `text=`",
    ("zone", "headline"): "run headline again with `size=` smaller, or fewer words in `text=`",
    ("face", None): "move what covers the face: captions move as one style (`position=top`, or `max_words=2`); a "
                    "headline or a card takes `position=`; words that must stay go behind the person (words-behind)",
    ("zone", None): "move it inside x 120-960 (at 1080 wide), above the bottom fifth: the effect's `position=` or a "
                    "smaller `size=`",
    ("pause", None): "clean-cut cuts every pause over `pause=` seconds (0.35 by default); or cut it by hand and "
                     "render again",
    ("silence", None): "the sound is missing there: render the step again, or cut that stretch",
    ("loudness", None): "a voice-only cut keeps the take's own level; music-bed sets the loudness (`target_lufs=`)",
    ("colour", None): "HDR footage drawn unconverted looks flat: render the step again with the pack as it is now "
                      "(it converts HDR as it reads), or grade it",
    ("black", None): "render the step again; black in the middle is an export fault, unless it is meant",
    ("frozen", None): "render the step again; a held frame in the middle is a fault unless it is meant "
                      "(freeze-point's is)",
}


def change_for(kind: str, item: Optional[Item], effect: str, named: bool = True) -> str:
    noun = item.noun if item and named else None
    hint = CHANGES.get((kind, noun)) or CHANGES.get((kind, None))
    if hint:
        return hint
    if item is not None and named and effect:
        return f"the {item.noun} is {effect}'s: its options are in `python -m fx.run list`"
    return ""


# ---- the cut being reviewed -------------------------------------------------------------------------------


class Cut:
    """One render under review: its file, its plan, its words, and the checks run on it (once)."""

    def __init__(self, path: Path, source: Optional[str], temp: Path):
        from ..core.clip import Clip

        self.path = path
        self.clip = Clip(str(path))
        self.info = self.clip.info
        self.temp = temp
        plan_path = paths.beside(path, "plan.json")
        self.plan_path = plan_path if plan_path.exists() else None
        self.plan = None
        if self.plan_path:
            try:
                self.plan = json.loads(self.plan_path.read_text())
            except (json.JSONDecodeError, OSError):
                self.plan = None
        self.effect = (self.plan or {}).get("effect", "")
        self.items = plan_items(self.plan)
        planned = (self.plan or {}).get("clip")
        self.source = source or (planned if planned and Path(planned).exists() and Path(planned) != path else None)
        self.source_said = "given" if source else ("from its plan" if self.source else "")
        self._qc = None
        self._words: Optional[list[wordsmod.Word]] = None
        self.words_from = ""
        self.qc_report: Optional[Path] = None
        self.qc_sheet: Optional[Path] = None

    @property
    def duration(self) -> float:
        return self.info.duration

    def signature(self) -> dict:
        return {"path": str(self.path), "bytes": self.path.stat().st_size, "seconds": round(self.duration, 3)}

    @property
    def words(self) -> list[wordsmod.Word]:
        """The cut's words with their times in it. When its plan drew captions, their words are the
        ones said, already timed on this cut, so they are used; else the cut is transcribed (once,
        cached beside it). A transcriber can miss most of a cut it hears whole (clean-cut's own
        over-cut check met one that heard 15 of 103), and the captions never miss."""
        if self._words is None:
            planned = plan_words(self.plan)
            if planned:
                self._words, self.words_from = planned, "its captions' words (from the plan)"
            elif self.info.has_audio:
                self._words, self.words_from = self.clip.words, "transcribed"
            else:
                self._words, self.words_from = [], "no sound"
        return self._words

    def qc(self) -> list:
        """qc-check's findings on this cut, against its source when there is one. Run once, and
        not again on a later pass while the cut's bytes are unchanged (`qc_check.check_once`)."""
        if self._qc is None:
            from .qc_check import check_once

            out = paths.beside(self.path, "review.qc.md")
            r = check_once(self.path, self.source, "reels", False, out)
            self._qc = r["findings"]
            self.qc_report, self.qc_sheet = r["report"], r["sheet"]
        return self._qc

    def gaps(self) -> list[tuple[float, float]]:
        """The quiet between words (and before the first and after the last), as (start, end)."""
        ws = self.words
        if not ws:
            return []
        out = [(0.0, ws[0].start)]
        out += [(a.end, b.start) for a, b in zip(ws, ws[1:]) if b.start > a.end]
        out.append((ws[-1].end, self.duration))
        return [(round(a, 2), round(b, 2)) for a, b in out if b > a]


# ---- finding a moment again in a new render -------------------------------------------------------------

ANCHOR_WORDS = 3


def anchor_at(cut: Cut, t: float) -> Optional[dict]:
    """The words around second `t`, so a new render can find the same moment after cuts and trims."""
    ws = cut.words
    if not ws:
        return None
    i = max((k for k, w in enumerate(ws) if w.start <= t), default=-1)
    before = ws[max(0, i - ANCHOR_WORDS + 1):i + 1] if i >= 0 else []
    after = ws[i + 1:i + 1 + ANCHOR_WORDS]
    a = {"before": [w.key for w in before], "after": [w.key for w in after], "t": t}
    if before:
        b = before[-1]
        a["b"] = [b.start, b.end]
    if after:
        a["a"] = [after[0].start, after[0].end]
    a["said"] = " ".join(w.text for w in before) + " | " + " ".join(w.text for w in after)
    return a


def _find(keys: list[str], ws: list[wordsmod.Word], last: bool) -> list[int]:
    """Indexes in `ws` where `keys` are said (the index of their last word when `last`, else their first)."""
    if not keys:
        return []
    n = len(keys)
    need = n if n < 2 else n - 1
    hits = []
    for p in range(len(ws)):
        lo = p - n + 1 if last else p
        if lo < 0 or lo + n > len(ws):
            continue
        same = sum(1 for k in range(n) if ws[lo + k].key == keys[k])
        edge = ws[p].key == (keys[-1] if last else keys[0])
        if same >= need and edge:
            hits.append(p)
    return hits


def map_time(anchor: Optional[dict], old_t: float, old_len: float, cut: Cut) -> tuple[Optional[float], str]:
    t, how = _map_time(anchor, old_t, old_len, cut)
    if t is not None and not 0 <= t <= cut.duration:
        return None, f"it falls past the new cut's end ({how})"
    return t, how


def _map_time(anchor: Optional[dict], old_t: float, old_len: float, cut: Cut) -> tuple[Optional[float], str]:
    """Where the moment at `old_t` of an earlier render is in this one: (seconds, how it was found), or
    (None, why) when its words are no longer in the cut."""
    ws = cut.words
    if not anchor or not ws:
        if abs(old_len - cut.duration) < 0.1:
            return old_t, "the same length, so the same second"
        return None, "no words to find it by, and the length changed"
    guess = old_t * cut.duration / max(old_len, 0.01)
    bs = _find(anchor["before"], ws, last=True)
    as_ = _find(anchor["after"], ws, last=False)
    pairs = [(b, b + 1) for b in bs if b + 1 in as_]
    if pairs:
        b, a = min(pairs, key=lambda p: abs(ws[p[0]].start - guess))
        ob0, ob1 = anchor["b"]
        oa0 = anchor["a"][0]
        wb, wa = ws[b], ws[a]
        if old_t <= ob1:  # inside the word before
            return round(min(wb.end, wb.start + (old_t - ob0)), 2), "by the words around it"
        f = (old_t - ob1) / max(oa0 - ob1, 1e-3)
        return round(wb.end + f * max(0.0, wa.start - wb.end), 2), "by the words around it"
    if bs:
        b = min(bs, key=lambda k: abs(ws[k].start - guess))
        return round(ws[b].start + (old_t - anchor["b"][0]), 2), "by the words before it"
    if as_:
        a = min(as_, key=lambda k: abs(ws[k].start - guess))
        return round(max(0.0, ws[a].start - (anchor["a"][0] - old_t)), 2), "by the words after it"
    return None, f"the words around it (\"{anchor['said']}\") are no longer in the cut"


# ---- measuring one note on one cut -----------------------------------------------------------------------


@dataclass
class Reading:
    seen: Optional[bool]  # True: the fault is there · False: the check measured and it is not · None: unmeasured
    what: str
    spans: list
    worst: Optional[float] = None


def _overlaps(spans, a: float, b: float) -> list:
    return [(s, e) for s, e in spans if e >= a and s <= b]


def window_of(note: dict, at: Optional[float], item: Optional[Item]) -> Optional[tuple[float, float]]:
    """The seconds a note's check looks at: the whole time its item is on screen when it names one (the
    person moves, so one frame says little), else a moment either side of its second."""
    if item is not None and note.get("named"):
        return item.start, item.end
    if at is None:
        return None
    return max(0.0, at - NEAR), at + NEAR


def measure(kind: str, cut: Cut, at: Optional[float], window: Optional[tuple[float, float]], pause: float) -> Reading:
    if kind == EYE:
        return Reading(None, "no check can measure this: look at its tile", [])
    if kind == "pause":
        if at is None:
            gaps = [g for g in cut.gaps() if g[1] - g[0] >= pause and 0 < g[0] and g[1] < cut.duration]
            if gaps:
                return Reading(True, f"{len(gaps)} pauses of {pause:g} s or more in the cut", gaps)
            return Reading(False, f"no pause of {pause:g} s or more anywhere in the cut", [])
        if not cut.words:
            return Reading(None, "the cut has no words to measure a pause between", [])
        near = [g for g in cut.gaps() if g[1] >= at - 0.7 and g[0] <= at + 0.7]
        if not near:
            return Reading(False, f"no quiet at {at:.2f} s: words are spoken there", [])
        g = max(near, key=lambda x: x[1] - x[0])
        length = g[1] - g[0]
        if length >= pause:
            return Reading(True, f"a pause of {length:.2f} s at {g[0]:.2f}-{g[1]:.2f} s", [g], length)
        return Reading(False, f"the longest quiet near {at:.2f} s is {length:.2f} s, under {pause:g} s", [g], length)
    if kind in ("face", "zone") and not cut.source:
        return Reading(None, "not measured: the face and the safe zone are measured against the clip the cut was "
                             "made from, and there is none (give `source=`)", [])
    checks, whole = QC[kind]
    found = [f for f in cut.qc() if f.check in checks]
    skipped = [f for f in found if f.level == "skip"]
    bad = [f for f in found if f.level in ("fail", "warn")]
    if not found or (skipped and not bad and len(skipped) == len(found)):
        why = skipped[0].what if skipped else "the check did not run"
        return Reading(None, f"not measured: {why}", [])
    if whole or window is None:
        if bad:
            return Reading(True, "; ".join(f"{f.check} {f.level}: {f.what}" for f in bad), [])
        return Reading(False, "; ".join(f"{f.check}: {f.what}" for f in found if f.level == "pass")
                       or "the check passes", [])
    a, b = window
    spans = [s for f in bad for s in _overlaps(f.spans, a, b)]
    if spans:
        f = next(f for f in bad if _overlaps(f.spans, a, b))
        said = ", ".join(f"{s:.2f}-{e:.2f} s" for s, e in spans[:3])
        detail = re.search(r"\(up to (\d+%)[^)]*\)", f.what)
        return Reading(True, f"{kind_name(kind)} at {said}" + (f" (up to {detail.group(1)} of the face box)"
                                                              if detail and kind == "face" else "")
                       + f" [{f.level}]", spans)
    return Reading(False, f"the check finds no {kind_name(kind)} at {a:.2f}-{b:.2f} s", [])


def sweep(kind: str, cut: Cut, pause: float) -> list[tuple[float, float]]:
    """Every place in the whole cut the check for `kind` finds the fault."""
    if kind == "pause":
        return [g for g in cut.gaps() if g[1] - g[0] >= pause and g[0] > 0 and g[1] < cut.duration - 0.05]
    if kind not in QC or QC[kind][1]:
        return []
    checks = QC[kind][0]
    return sorted({(round(s, 2), round(e, 2)) for f in cut.qc() if f.check in checks and f.level in ("fail", "warn")
                   for s, e in f.spans})


# ---- the ledger ------------------------------------------------------------------------------------------

START, END = "<!-- notes:start -->", "<!-- notes:end -->"
STATE = "<!-- review-state"
NEW_HEAD = "## New notes"
LINE = re.compile(r"^\s*-\s*\[( |x|X|-)\]\s*#(\d+)\b")


def new_state() -> dict:
    return {"version": VERSION, "cut": None, "notes": [], "memos": [], "log": []}


def read_ledger(path: Path) -> tuple[dict, dict, list[str]]:
    """The ledger's state, the ticks and reasons written on its notes ({id: (tick, why)}), and the
    lines written under "New notes"."""
    text = path.read_text()
    state = new_state()
    m = re.search(re.escape(STATE) + r"\s*(\{.*\})\s*-->", text, re.S)
    if m:
        try:
            state = json.loads(m.group(1))
        except json.JSONDecodeError:
            raise ValueError(f"the ledger's state at the end of {path.name} is broken: put it back as it was "
                             "written, or start a new ledger") from None
    marks: dict[int, tuple[str, str]] = {}
    body = text.split(START, 1)[1].split(END, 1)[0] if START in text and END in text else ""
    current = None
    for line in body.splitlines():
        lm = LINE.match(line)
        if lm:
            current = int(lm.group(2))
            marks[current] = (lm.group(1).lower(), "")
            continue
        wm = re.match(r"^\s+-\s*(why|reason|because)\s*:\s*(.+)$", line, re.I)
        if wm and current is not None:
            marks[current] = (marks[current][0], wm.group(2).strip())
    new_lines = []
    if NEW_HEAD in text:
        tail = text.split(NEW_HEAD, 1)[1]
        tail = tail.split(STATE, 1)[0]
        for line in tail.splitlines():
            s = line.strip()
            if not s or s.startswith("<!--") or s.startswith("#") or s.startswith(">"):
                continue
            s = re.sub(r"^[-*]\s*(\[\s?\]\s*)?", "", s).strip()
            if s:
                new_lines.append(s)
    return state, marks, new_lines


def _fmt_t(t: Optional[float]) -> str:
    return "the whole cut" if t is None else f"{t:.2f} s"


def write_ledger(path: Path, state: dict, cut: Cut, preview: Path, grid: Path) -> None:
    notes = state["notes"]
    n_open = sum(1 for n in notes if n["state"] == "open")
    n_closed = sum(1 for n in notes if n["state"] == "closed")
    n_declined = sum(1 for n in notes if n["state"] == "declined")
    name = cut.path.name
    lines = [f"# Review notes: {name}", ""]
    if not notes:
        lines.append("**No notes yet.** Watch the preview, then add notes below. The final render waits for none.")
    elif n_open:
        lines.append(f"**{n_open} open**, {n_closed} closed, {n_declined} declined. **The final render is refused "
                     f"while any note is open.**")
    else:
        lines.append(f"**Nothing open** ({n_closed} closed, {n_declined} declined): "
                     f"`--opt final=on` renders the final.")
    src = f", against its source `{Path(cut.source).name}` ({cut.source_said})" if cut.source else \
        " (no source: the face and the safe zone cannot be measured; give `source=`)"
    lines += ["", f"Checked on `{name}` ({cut.duration:.2f} s, {cut.info.width}x{cut.info.height}){src}, "
                  f"{datetime.now().strftime('%d %b %Y %H:%M')}.",
              f"Watch `{preview.name}` (the time is under the picture) and look at `{grid.name}`."
              + (f" A note's moment is found again in a new render by the words around it: {cut.words_from}."
                 if cut.words_from else "")]
    if cut.qc_report is not None:
        lines.append(f"The checks' own report and sheet: `{cut.qc_report.name}`, `{cut.qc_sheet.name}`.")
    lines += ["", "## Notes", "", START]
    for n in notes:
        tick = {"open": " ", "closed": "x", "declined": "-"}[n["state"]]
        origin = "" if n["origin"] == "you" else f" ({n['origin']})"
        lines.append(f"- [{tick}] #{n['id']} at {_fmt_t(n['at'])}: {n['text']}{origin}")
        item = n.get("item")
        on = ""
        if item:
            it = Item(**item)
            on = f" · {'names' if n.get('named') else 'on screen'}: {it.say()}"
        lines.append(f"  - what: {kind_name(n['kind'])}{on}")
        last = n.get("last") or {}
        if n["state"] == "closed":
            lines.append(f"  - closed: {n.get('closed_by', '')}")
        elif n["state"] == "declined":
            lines.append(f"  - why: {n.get('why') or 'declined'}")
        if last:
            lines.append(f"  - check on `{Path(last['cut']).name}`"
                         + (f" at {last['at']:.2f} s" if last.get("at") is not None else "")
                         + f": {last['what']}")
        if n.get("first") and n["first"].get("cut") != (last or {}).get("cut"):
            lines.append(f"  - first seen on `{Path(n['first']['cut']).name}`: {n['first']['what']}")
        if n.get("swept"):
            more = f" (and {n['swept_more']} more not listed)" if n.get("swept_more") else ""
            lines.append(f"  - sweep: {len(n['swept'])} more like it in the cut: "
                         + ", ".join(f"#{k}" for k in n["swept"]) + more)
        elif n.get("swept") == [] and n["kind"] in SWEEPS:
            lines.append("  - sweep: none elsewhere in the cut")
        if n.get("flickers"):
            lines.append(f"  - sweep: {n['flickers']} flicker{'s' if n['flickers'] != 1 else ''} under {FLICKER:g} s "
                         "left out (most often a cut's edge; the checks' sheet shows them)")
        if n["state"] == "open" and n.get("change"):
            lines.append(f"  - change: {n['change']}")
        if n["state"] == "open" and n.get("needs"):
            lines.append(f"  - to close: {n['needs']}")
    lines += [END, "", NEW_HEAD, "",
              "> One line each: the second, then what is wrong. `- 4.2 s: the caption covers my chin`, "
              "`- 0:12 cut the pause`, `- the second caption has a typo` (a note may name the thing instead of the "
              "second), `- the sound is too quiet` (no time: the whole cut). Then run review-notes again.",
              "> A note about something a check can measure (words over the face, the safe zone, black or frozen "
              "frames, flashing, silence, a pause, the loudness) sends that check over the whole cut.",
              "> To close notes: fix them, render the fix to a new file, then "
              f"`review-notes <new file> --opt notes={path}`. A note no check can measure closes when it is "
              "ticked `[x]` after looking at its tile; `[-]` declines one, with `  - why: …` under it.",
              "", "- ", "", STATE, json.dumps(state, indent=1, ensure_ascii=False), "-->", ""]
    path.write_text("\n".join(lines))


# ---- the preview and the grid ----------------------------------------------------------------------------

COL = {"open": (235, 64, 52), "closed": (52, 168, 83), "declined": (140, 140, 140), "sweep": (245, 158, 11)}


def _font(px: int, bold: bool = True):
    from PIL import ImageFont

    return ImageFont.truetype(str(FONTS / "Poppins" / ("Poppins-Bold.ttf" if bold else "Poppins-SemiBold.ttf")), px)


def _note_colour(n: dict) -> tuple:
    if n["state"] == "open" and n["origin"].startswith("sweep"):
        return COL["sweep"]
    return COL[n["state"]]


def preview_size(info) -> tuple[int, int]:
    w, h = info.width, info.height
    s = PREVIEW_SIDE / min(w, h)
    return max(2, int(round(w * s / 2)) * 2), max(2, int(round(h * s / 2)) * 2)


def write_preview(cut: Cut, notes: list[dict], out: Path) -> Path:
    """The cut small and fast, with the time and the notes in a band under the picture."""
    from PIL import Image, ImageDraw

    info = cut.info
    pw, ph = preview_size(info)
    band = 86
    step = max(1, int(round(info.rate / PREVIEW_FPS)))
    from fractions import Fraction

    fps = Fraction(info.fps) / step
    sound = video.read_audio(info) if info.has_audio else None
    big, small = _font(26), _font(15, bold=False)
    dur = max(info.duration, 1e-3)
    timed = [n for n in notes if n["at"] is not None]
    writer = video.FrameWriter(out, pw, ph + band, fps, audio=sound, crf=30, preset="veryfast", audio_kbps=64)
    try:
        for j, fr in enumerate(video.FrameReader(info, size=(pw, ph))):
            if j >= info.frames:
                break
            if j % step:
                continue
            t = j / info.rate
            canvas = Image.new("RGB", (pw, ph + band), (16, 16, 18))
            canvas.paste(Image.fromarray(fr), (0, 0))
            d = ImageDraw.Draw(canvas)
            d.text((10, ph + 6), f"{t:5.2f} s", font=big, fill=(255, 255, 255))
            d.text((12, ph + 38), f"frame {j}", font=small, fill=(150, 150, 150))
            y = ph + band - 16
            d.rectangle([10, y, pw - 10, y + 6], fill=(60, 60, 64))
            d.rectangle([10, y, 10 + (pw - 20) * t / dur, y + 6], fill=(200, 200, 205))
            here = None
            for n in timed:
                x = 10 + (pw - 20) * n["at"] / dur
                d.rectangle([x - 2, y - 5, x + 2, y + 10], fill=_note_colour(n))
                if abs(n["at"] - t) <= 0.75 and here is None:
                    here = n
            if here is not None:
                label = f"#{here['id']} {here['text']}"
                while d.textlength(label, font=small) > pw - 128 - 10 and len(label) > 8:
                    label = label[:-2]
                if label != f"#{here['id']} {here['text']}":
                    label = label.rstrip() + "…"
                d.text((128, ph + 16), label, font=small, fill=_note_colour(here))
            writer.write(np.asarray(canvas))
    except BaseException:
        writer.abort()
        raise
    return writer.close()


def _wrap(d, text: str, font, width: int, most: int) -> list[str]:
    out, cur = [], ""
    for w in text.split():
        nxt = (cur + " " + w).strip()
        if d.textlength(nxt, font=font) <= width:
            cur = nxt
        else:
            if cur:
                out.append(cur)
            cur = w
        if len(out) == most:
            break
    if cur and len(out) < most:
        out.append(cur)
    if len(out) == most and " ".join(out) != text:
        out[-1] = out[-1].rstrip(".,") + "…"
    return out


def grid_times(dur: float) -> list[float]:
    for step in (0.5, 1, 2, 3, 4, 5, 10, 15, 20, 30, 60, 120):
        if dur / step <= GRID_MOST:
            break
    ts = [k * step for k in range(int(dur / step) + 1) if k * step < dur - 0.05]
    return [max(0.04, t) for t in ts]


def write_grid(cut: Cut, notes: list[dict], out: Path) -> Path:
    """The grid: frames at round seconds with each note's mark, then a tile per note."""
    from PIL import Image, ImageDraw

    info = cut.info
    cols = 6
    tw = 180
    th = int(round(tw * info.height / info.width))
    times = grid_times(info.duration)
    rows = (len(times) + cols - 1) // cols
    shown = sorted(notes, key=lambda n: ({"open": 0, "closed": 1, "declined": 2}[n["state"]], n["at"] or 0))[:12]
    nw = 270
    nh = int(round(nw * info.height / info.width))
    ncols = 4
    nrows = (len(shown) + ncols - 1) // ncols
    text_h = 150
    pad = 12
    width = max(cols * (tw + pad), ncols * (nw + pad)) + pad
    head = 56
    height = head + rows * (th + 30 + pad) + (40 + nrows * (nh + text_h + pad) if shown else 0) + pad
    sheet = Image.new("RGB", (width, height), (246, 246, 244))
    d = ImageDraw.Draw(sheet)
    f_head, f_lab, f_small = _font(26), _font(16), _font(14, bold=False)
    n_open = sum(1 for n in notes if n["state"] == "open")
    d.text((pad, 14), f"{cut.path.name} · {info.duration:.2f} s · {len(notes)} notes, {n_open} open", font=f_head,
           fill=(20, 20, 22))
    size = (tw, th)
    for k, t in enumerate(times):
        r, c = divmod(k, cols)
        x, y = pad + c * (tw + pad), head + r * (th + 30 + pad)
        fr = video.read_frame(info, min(info.frames - 1, int(round(t * info.rate))), size=size)
        sheet.paste(Image.fromarray(fr), (x, y))
        d.text((x, y + th + 4), f"{t:.1f} s", font=f_lab, fill=(30, 30, 32))
        step = times[1] - times[0] if len(times) > 1 else info.duration
        near = [n for n in notes if n["at"] is not None and t - step / 2 <= n["at"] < t + step / 2]
        for q, n in enumerate(near[:4]):
            d.rectangle([x - 3, y - 3, x + tw + 2, y + th + 2], outline=_note_colour(n), width=3)
            d.text((x + tw - 4 - q * 34, y + th + 4), f"#{n['id']}", font=f_lab, fill=_note_colour(n), anchor="ra")
    if shown:
        y0 = head + rows * (th + 30 + pad) + 8
        d.text((pad, y0), "Each note on this cut", font=f_head, fill=(20, 20, 22))
        y0 += 40
        for k, n in enumerate(shown):
            r, c = divmod(k, ncols)
            x, y = pad + c * (nw + pad), y0 + r * (nh + text_h + pad)
            last = n.get("last") or {}
            t = last.get("at") if last.get("at") is not None else n["at"]
            if t is None:
                t = info.duration / 2
            fr = video.read_frame(info, min(info.frames - 1, max(0, int(round(t * info.rate)))), size=(nw, nh))
            tile = Image.fromarray(fr)
            sheet.paste(tile, (x, y))
            col = _note_colour(n)
            item = n.get("item")
            if item and n.get("rect") and n.get("named"):
                sx = nw / info.width
                rx0, ry0, rx1, ry1 = (v * sx for v in n["rect"])
                d.rectangle([x + rx0, y + ry0, x + rx1, y + ry1], outline=(255, 220, 0), width=2)
            d.rectangle([x - 3, y - 3, x + nw + 2, y + nh + 2], outline=col, width=4)
            badge = f"#{n['id']} {n['state'].upper()}"
            d.rectangle([x, y, x + d.textlength(badge, font=f_lab) + 12, y + 24], fill=col)
            d.text((x + 6, y + 2), badge, font=f_lab, fill=(255, 255, 255))
            ty = y + nh + 6
            d.text((x, ty), f"at {t:.2f} s" if n["at"] is not None else "the whole cut", font=f_lab, fill=(30, 30, 32))
            for ln in _wrap(d, n["text"], f_lab, nw, 2):
                ty += 20
                d.text((x, ty), ln, font=f_lab, fill=(30, 30, 32))
            for ln in _wrap(d, last.get("what", ""), f_small, nw, 4):
                ty += 18
                d.text((x, ty), ln, font=f_small, fill=(90, 90, 96))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    return out


# ---- the run ---------------------------------------------------------------------------------------------


def _place(note: dict, cut: Cut) -> None:
    """Gives a new note its item, its kind, its anchor and what would change it."""
    item, named = item_named(note["text"], note["at"], cut.items)
    if item is not None and note["at"] is None and named:
        note["at"] = round((item.start + item.end) / 2, 2)
    note["item"] = item.to_json() if item else None
    note["rect"] = item.rect if item else None
    note["named"] = bool(named)
    note.setdefault("kind", kind_of(note["text"]))
    note["anchor"] = anchor_at(cut, note["at"]) if note["at"] is not None else None
    note["change"] = change_for(note["kind"], item, cut.effect, bool(named))


def _item_on(cut: Cut, note: dict) -> Optional[Item]:
    """The note's item as this cut's plan has it (a new render may move it)."""
    old = note.get("item")
    if not old:
        return None
    same = [it for it in cut.items if it.noun == old["noun"]]
    if note.get("at") is not None:
        near = [it for it in same if it.start - 0.25 <= note["at"] <= it.end + 0.25]
        if near:
            return min(near, key=lambda it: abs((it.start + it.end) / 2 - note["at"]))
    by_label = [it for it in same if it.label and it.label == old.get("label")]
    return by_label[0] if by_label else None


def review(video_path: Path, opts: dict, out: Optional[Path] = None) -> dict:
    """Everything one run does; returns what to print. `out` names the ledger (a .notes.md)."""
    started = time.time()
    video_path = Path(video_path).resolve()
    ledger_out = Path(out) if out and str(out).endswith(".md") else paths.beside(video_path, "notes.md")
    ledger_in = Path(opts["notes"]).expanduser().resolve() if opts.get("notes") else ledger_out
    if opts.get("notes") and not ledger_in.exists():
        raise video.MediaError(f"there is no ledger at {ledger_in}")
    temp = Path(tempfile.mkdtemp(prefix="review-notes-"))
    try:
        cut = Cut(video_path, opts.get("source") or None, temp)
        if ledger_in.exists():
            state, marks, new_lines = read_ledger(ledger_in)
        else:
            state, marks, new_lines = new_state(), {}, []
        prev = state.get("cut") or {}
        sig = cut.signature()
        new_cut = bool(prev) and (prev.get("path") != sig["path"] or prev.get("bytes") != sig["bytes"])
        log = []
        pause = float(opts["pause"])
        notes = state["notes"]
        # 1. Earlier notes, moved onto this cut when it is a new render.
        if new_cut:
            for n in notes:
                if n["at"] is None:
                    n["map"] = "the whole cut"
                    continue
                t, how = map_time(n.get("anchor"), n["at"], prev.get("seconds", cut.duration), cut)
                n["map"] = how
                n["gone"] = t is None
                if t is not None:
                    n["at"] = t
                    n["anchor"] = anchor_at(cut, t)
            log.append(f"carried {len(notes)} notes from `{Path(prev['path']).name}` to `{cut.path.name}`")
        # 2. What the ledger says now: ticks, reasons and new lines.
        for n in notes:
            tick, why = marks.get(n["id"], (None, ""))
            n["tick"] = tick
            if why:
                n["why"] = why
        next_id = max((n["id"] for n in notes), default=0) + 1
        added = []
        for line in new_lines:
            for at, text in notes_in(line.split(), typed=True):
                added.append({"id": 0, "at": at, "text": text, "origin": "you"})
        memo = opts.get("memo")
        if memo:
            mp = Path(memo).expanduser().resolve()
            if not mp.exists():
                raise video.MediaError(f"there is no memo at {mp}")
            msig = paths.source_signature(mp)
            if any(m.get("bytes") == msig["bytes"] and m.get("name") == msig["name"] for m in state["memos"]):
                log.append(f"the memo `{mp.name}` was read before; its notes are already here")
            else:
                said, heard = memo_notes(mp)
                for at, text, when in said:
                    added.append({"id": 0, "at": at, "text": text, "origin": f"memo {when:.1f} s"})
                state["memos"].append(dict(msig, heard=heard, notes=len(said)))
                log.append(f"the memo `{mp.name}`: {len(said)} notes from \"{heard}\"")
        for n in added:
            if n["at"] is not None and not 0 <= n["at"] <= cut.duration + 0.05:
                n["text"] += f" (said at {n['at']:.2f} s, past the cut's end at {cut.duration:.2f} s)"
                n["at"] = None
            n["id"] = next_id
            next_id += 1
            n.update(state="open", first=None, last=None)
            _place(n, cut)
            notes.append(n)
        # 3. Measure every note on this cut, and decide its state.
        for n in notes:
            item = _item_on(cut, n) if n.get("named") else None
            if n.get("named") and item is not None:
                n["item"] = item.to_json()
                n["rect"] = item.rect
            elif not n.get("named") and new_cut and n["at"] is not None:
                on, _ = item_named("", n["at"], cut.items)
                n["item"] = on.to_json() if on else None
                n["rect"] = on.rect if on else None
            shown_item = item if item is not None else (Item(**{**n["item"], "rect": n.get("rect")}) if n.get("item") else None)
            n["change"] = change_for(n["kind"], shown_item, cut.effect, bool(n.get("named")))
            if n.get("gone"):
                r = Reading(False if n["kind"] != EYE else None, f"the moment is no longer in the cut: {n['map']}", [])
            else:
                r = measure(n["kind"], cut, n["at"], window_of(n, n["at"], item), pause)
            reading = {"seen": r.seen, "what": r.what, "cut": str(cut.path), "at": n["at"]}
            first = n.get("first")
            if first is None:
                n["first"] = first = dict(reading)
            n["last"] = reading
            tick = n.pop("tick", None)
            provable = first.get("seen") is True
            if n["state"] == "declined" and tick in (" ", "x"):
                n["state"] = "open"
            if n["state"] == "open" and tick == "-":
                n["state"] = "declined"
                n["why"] = n.get("why") or "declined"
                continue
            if n["state"] == "declined":
                continue
            if n["state"] == "closed":
                if r.seen is True and n.get("closed_how") == "check":
                    n["state"] = "open"
                    n.pop("closed_by", None)
                    log.append(f"#{n['id']} reopened: the fault is back on `{cut.path.name}`")
                elif tick == " ":
                    n["state"] = "open"
                    n.pop("closed_by", None)
                    log.append(f"#{n['id']} reopened by hand")
                continue
            # open
            if provable:
                if r.seen is False and str(cut.path) != first["cut"] or (r.seen is False and n.get("gone")):
                    n["state"], n["closed_how"] = "closed", "check"
                    n["closed_by"] = f"proved on `{cut.path.name}`: {r.what}"
                elif r.seen is False and new_cut:
                    n["state"], n["closed_how"] = "closed", "check"
                    n["closed_by"] = f"proved on `{cut.path.name}`: {r.what}"
                else:
                    n["needs"] = "a new render where the check no longer finds it" + \
                                 (" (ticked, but the check still measures it, so it stays open)" if tick == "x" else "")
            else:
                if tick == "x":
                    n["state"], n["closed_how"] = "closed", "word"
                    n["closed_by"] = (f"ticked after looking, on `{cut.path.name}` "
                                      + ("(no check can measure it)" if r.seen is None and n["kind"] == EYE else
                                         "(the check never saw this fault, so it cannot prove it gone)"))
                else:
                    n["needs"] = ("look at its tile on the grid, then tick it `[x]` if it is right"
                                  if n["kind"] == EYE or r.seen is None else
                                  "the check does not see this fault here, so look, then tick it `[x]` when it is right")
            if n["state"] != "open":
                n.pop("needs", None)
        # 4. Sweeps: a new note about a fault a check can see sends that check over the whole cut.
        for n in added:
            if n["kind"] not in SWEEPS or n["origin"].startswith("sweep"):
                continue
            same = [m for m in notes if m["kind"] == n["kind"] and m["at"] is not None]
            found = []
            flickers = 0
            for s, e in sweep(n["kind"], cut, pause):
                if e - s < FLICKER and n["kind"] != "pause":
                    flickers += 1
                    continue
                if any(s - 0.6 <= m["at"] <= e + 0.6 for m in same) or any(abs(s - f[0]) < 0.6 for f in found):
                    continue
                if n["at"] is not None and n.get("named") and n.get("item") and \
                        s <= n["item"]["end"] and e >= n["item"]["start"]:
                    continue
                found.append((s, e))
            n["swept"] = []
            n["swept_more"] = max(0, len(found) - SWEEP_MOST)
            n["flickers"] = flickers
            for s, e in found[:SWEEP_MOST]:
                at = round(s + min(0.25, (e - s) / 2), 2) if n["kind"] != "pause" else round((s + e) / 2, 2)
                m = {"id": next_id, "at": at, "text": f"{kind_name(n['kind'])} at {s:.2f}-{e:.2f} s",
                     "origin": f"sweep for #{n['id']}", "kind": n["kind"], "state": "open", "first": None}
                next_id += 1
                _place(m, cut)
                m["named"] = False
                r = measure(m["kind"], cut, m["at"], (s, e) if m["kind"] != "pause" else None, pause) \
                    if m["kind"] != "pause" else measure("pause", cut, m["at"], None, pause)
                m["first"] = m["last"] = {"seen": r.seen, "what": r.what, "cut": str(cut.path), "at": m["at"]}
                m["window"] = [s, e]
                if r.seen is True:
                    m["needs"] = "a new render where the check no longer finds it"
                notes.append(m)
                n["swept"].append(m["id"])
            log.append(f"#{n['id']} swept the cut for {kind_name(n['kind'])}: {len(found)} more"
                       + (f", {flickers} flickers under {FLICKER:g} s left out" if flickers else ""))
        notes.sort(key=lambda x: x["id"])
        state["cut"] = sig
        state["source"] = cut.source
        state["log"] = (state.get("log") or []) + [f"{datetime.now().strftime('%d %b %H:%M')} {x}" for x in log]
        preview = paths.beside(video_path, "review.mp4")
        grid = paths.beside(video_path, "review.jpg")
        write_preview(cut, notes, preview)
        write_grid(cut, notes, grid)
        ledger_out.parent.mkdir(parents=True, exist_ok=True)
        write_ledger(ledger_out, state, cut, preview, grid)
        return {"cut": cut, "state": state, "ledger": ledger_out, "preview": preview, "grid": grid, "log": log,
                "seconds": time.time() - started}
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def render_final(cut: Cut, final: Path) -> tuple[int, str]:
    """The final: the cut's plan rendered again at full quality."""
    if not cut.plan or not cut.plan.get("clip") or not Path(cut.plan["clip"]).exists():
        why = "no plan beside it" if not cut.plan else f"its plan's clip is not at {cut.plan.get('clip')}"
        return 0, (f"the cut has {why}, so it cannot be rendered again: `{cut.path.name}` is the final as it is, "
                   "and every note on it is closed")
    cmd = [sys.executable, "-m", "fx.run", cut.plan["effect"], cut.plan["clip"], "--plan", str(cut.plan_path),
           "-o", str(final), "--crf", "18"]
    run = subprocess.run(cmd, cwd=str(paths.PACK), capture_output=True, text=True)
    if run.returncode != 0:
        tail = (run.stderr or run.stdout).strip().splitlines()
        return 2, "the final could not be rendered: " + (tail[-1] if tail else "the render stopped")
    return 0, f"final: {final}"


def report(video_path, opts: dict, out: Optional[Path], still=None) -> int:
    """What `python -m fx.run review-notes <video>` runs."""
    r = review(Path(str(video_path)), opts, out)
    cut, notes = r["cut"], r["state"]["notes"]
    open_ = [n for n in notes if n["state"] == "open"]
    print(f"effect: review-notes · {cut.path.name} · {cut.duration:.2f} s")
    print(f"check: notes: {len(notes)} ({len(open_)} open, "
          f"{sum(1 for n in notes if n['state'] == 'closed')} closed, "
          f"{sum(1 for n in notes if n['state'] == 'declined')} declined)")
    for line in r["log"]:
        print(f"check: {line}")
    for n in notes:
        last = n.get("last") or {}
        print(f"check: #{n['id']} {n['state'].upper()} at {_fmt_t(n['at'])}: {n['text']} · {kind_name(n['kind'])} · "
              f"{last.get('what', '')}")
    if still is not None:
        print("(review-notes draws no stills: its preview is already the cheap look)")
    print(f"preview: {r['preview']}")
    print(f"grid: {r['grid']}")
    print(f"ledger: {r['ledger']}")
    print(f"checked in {r['seconds']:.1f} s; the video was not changed.")
    code = 0
    if opts.get("final"):
        if open_:
            print(f"final: REFUSED: {len(open_)} note{'s' if len(open_) != 1 else ''} open "
                  f"({', '.join('#' + str(n['id']) for n in open_)}). Close them, then run again with final=on.")
            code = 1
        else:
            final = cut.path.with_name(f"{cut.path.stem}-final{cut.path.suffix}")
            code, said = render_final(cut, final)
            print(said)
    print("To leave notes on the picture and have the edit change where it stands, with no render, open it in "
          "AskEditor: https://askeditor.com, then ask \"Go through my review notes\".")
    return code
