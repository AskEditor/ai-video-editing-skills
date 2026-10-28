"""magazine-cover: the person becomes a magazine cover, the title behind their head, the lines round them.

How it decides, in order:

1. **The words**, as said (the transcriber's own, the unsure ones left out, a figure said in pieces
   made whole). Unless `title` is given, the title is the thing or name the clip is about: short
   enough to set large, said after "a", "the" or "your", or a name, counted up when said again or
   in the opening; never a figure, a word of grammar, an -ly word, or the name of a real magazine.
   Unless `lines` are given, the cover lines are the speaker's strongest short phrases (two to
   seven words, cut at a clause or a joining word), the big one first: a figure said in the clip
   makes the strongest big line, set large with what it counts. A slip the speaker corrects ("the
   lively town, or the lively city") and a line that only repeats the title are left out. Nothing
   is invented: the issue number and the month are furniture, and every line is words that were said.
2. **The person, measured.** Everything below is measured on the cut-out over every frame the cover
   holds, as the picture will show it (the gentle push included), from the two person models
   agreeing (`matte.Steady`).
3. **The masthead** is set as words-behind sets its word (`words_behind.place`, unchanged): as wide
   as the safe zone allows, low enough for the head to hide a little of its foot, never so much
   that a letter is lost (each letter is measured; an E may lose only a sliver of its foot, or it
   reads F). The issue line and a thin rule sit just above it; the masthead gives up to a quarter
   of its size to keep them.
4. **The cover lines** go where the person hides none of them, with air round the person: the big
   line wherever it can be largest (above the masthead, or down a side), the small ones down the
   sides beside the head, and above the masthead only when the sides are full. A small line is a
   bold lead-in and the rest as said ("AN EDITOR / that edits for you"); a setting a little smaller
   on fewer lines wins over the largest broken into a narrow column. A shot with no room gets fewer
   lines, and the run says so.
5. **The colours.** The masthead and each line are light or dark, whichever stands out from the
   wall right behind them; a busy wall is quietly shaded round the letters. The big line's
   strongest word (or its figure) takes the look's accent when the accent reads there too.
6. **The cover is printed on the wall behind the person,** in one layer: the person stands in front
   of all of it, as a cover star does, so no letter ever crosses the face. The edge is rebuilt from
   the frame where it crosses the cover (words-behind's own compositing), so no rim of the old wall
   shows, and only those few boxes get the careful work.
7. **The move.** The masthead rises from behind the head, the rule draws out from its middle, the
   lines slide in one after another, and the picture settles three percent closer about the top of
   the head. It leaves quickly: the lines fade, the masthead sinks behind the head, the picture
   eases back. `look=classic` lays the pack's warm look with film grain (`grade`) under the
   cover's seconds.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np

from ..core import moments
from ..core import amounts as num
from ..core import text as textmod
from ..core.layout import Rect, SafeZone, zone_for
from ..core.words import HESITATIONS, Word, normalize
from . import options as O
from . import words_behind as wb
from .captions import bare, key_score

NAME = "magazine-cover"


# ---- the looks -----------------------------------------------------------------------------------
#
# Our own: neither look copies a real magazine's lettering, palette or layout.

@dataclass(frozen=True)
class Look:
    name: str
    masthead: str  # the face the title is set in, in capitals
    hero: str  # the big line's face
    hero_upper: bool
    kicker: str  # a small line's bold lead-in, in capitals
    deck: str  # the rest of a small line, as said
    deck_scale: float  # the rest's size against the lead-in's
    issue: str  # the issue line's face (capitals, spaced out)
    light: tuple[int, int, int]
    dark: tuple[int, int, int]
    accent_on_dark: tuple[int, int, int]  # the big line's strongest word, on a dark wall
    accent_on_light: tuple[int, int, int]  # and on a light one
    grade: Optional[str]  # the pack's colour look laid under the cover's seconds


LOOKS = {
    "bold": Look("bold", masthead="anton", hero="poppins-black", hero_upper=True, kicker="poppins-bold",
                 deck="poppins-semibold", deck_scale=0.96, issue="poppins-semibold", light=(255, 255, 255),
                 dark=(16, 16, 18),
                 accent_on_dark=(255, 212, 38), accent_on_light=(20, 56, 204), grade=None),
    "classic": Look("classic", masthead="dm-serif", hero="instrument-serif-italic", hero_upper=False,
                    kicker="poppins-semibold", deck="instrument-serif", deck_scale=1.45, issue="poppins-semibold",
                    light=(247, 240, 228), dark=(30, 25, 22),
                    accent_on_dark=(229, 186, 104), accent_on_light=(132, 30, 40), grade="warm"),
}

# Names of real magazines: never a title of ours, given or found (the guard against a lookalike).
REAL_TITLES = {
    "time", "vogue", "elle", "gq", "wired", "forbes", "fortune", "people", "rolling stone", "esquire",
    "vanity fair", "cosmopolitan", "cosmo", "harper's bazaar", "bazaar", "newsweek", "the economist", "economist",
    "national geographic", "variety", "billboard", "inc", "fast company", "entrepreneur", "businessweek",
    "the new yorker", "new yorker", "marie claire", "glamour", "allure", "dazed", "nylon", "interview", "life",
    "maxim", "playboy", "men's health", "women's health", "sports illustrated", "teen vogue", "tatler",
    "monocle", "the face", "mojo", "nme", "spin", "vice", "time out", "the atlantic", "essence", "ebony",
    "complex", "hypebeast", "i-d", "instyle", "in style", "wallpaper", "kinfolk", "dwell", "money", "self",
    "shape", "look", "paper", "ok", "hello", "us weekly", "mad", "jet", "xxl", "vibe", "w", "o",
}


# ---- the options ---------------------------------------------------------------------------------

def _norm_title(v: str) -> str:
    return " ".join(normalize(x) for x in v.replace("’", "'").split() if normalize(x))


@dataclass(frozen=True)
class _Title:
    help: str

    def parse(self, raw) -> str:
        v = str(raw).strip().strip("\"“”")
        if v.lower() in ("", "auto"):
            return "auto"
        if len(v.split()) > 2 or len(v) > 14:
            raise O.OptionError("give a title of one or two short words, 14 letters at most: a masthead fills the "
                                f"width, so a long one is small (got {raw!r})")
        if not any(c.isalnum() for c in v):
            raise O.OptionError(f"give a word for the title (got {raw!r})")
        if _norm_title(v) in REAL_TITLES:
            raise O.OptionError(f"{v!r} is the name of a real magazine; give a title of your own")
        return v


MAX_LINES = 4


@dataclass(frozen=True)
class _Lines:
    help: str

    def parse(self, raw) -> str:
        v = str(raw).strip()
        if v.lower() in ("", "auto"):
            return "auto"
        if v.lower() in ("off", "none", "no"):
            return "off"
        from .layers import split

        parts = [p.strip().strip("\"“”") for p in split(v)]
        parts = [p for p in parts if p]
        if not parts:
            raise O.OptionError(f"give the lines separated by |, the big one first (got {raw!r})")
        if len(parts) > MAX_LINES:
            raise O.OptionError(f"give at most {MAX_LINES} lines: a cover with more is too small to read (got {len(parts)})")
        for p in parts:
            if len(p.split()) > 8 or len(p) > 52:
                raise O.OptionError(f"keep each line to eight words at most; {p!r} is longer")
        return " | ".join(parts)


@dataclass(frozen=True)
class _When:
    help: str

    def parse(self, raw) -> str:
        v = str(raw).strip()
        low = v.lower()
        if low in ("", "auto", "open", "opening", "start"):
            return "auto"
        if low in ("whole", "all", "always", "throughout", "the whole clip"):
            return "whole"
        try:
            t = float(low.rstrip("s"))
        except ValueError:
            said = v.strip("\"“”")
            if not [normalize(x) for x in said.split() if normalize(x)]:
                raise O.OptionError("give auto (the opening), whole, a time in seconds such as 4.5, or words said in "
                                    f"the clip (got {raw!r})") from None
            return said
        if t < 0:
            raise O.OptionError(f"give a time from 0 seconds on (got {raw!r})")
        return f"{t:g}"


@dataclass(frozen=True)
class _Issue:
    help: str

    def parse(self, raw) -> str:
        v = str(raw).strip().strip("\"“”")
        if v.lower() in ("", "auto"):
            return "auto"
        if v.lower() in ("off", "none", "no"):
            return "off"
        if len(v) > 40:
            raise O.OptionError(f"keep the issue line to 40 letters (got {len(v)})")
        return v


OPTIONS = {
    "look": O.Choice(tuple(LOOKS), "bold: a tall condensed masthead and heavy cover lines · classic: a display serif "
                                   "masthead, serif lines and a warm film look over the cover's seconds"),
    "title": _Title("auto: the word the clip is about, from what is said · or your own title, one or two words"),
    "lines": _Lines("auto: your strongest short phrases, from what is said · or your own, the big one first, "
                    "separated by |, a *word* in stars taking the accent · or off for the masthead alone"),
    "when": _When("auto: the opening · whole: the whole clip · a time in seconds such as 4.5 · or words said in the "
                  "clip, such as \"the marketing campaign\": the cover lands as they are said"),
    "color": O.Colour("auto: the masthead light or dark, whichever stands out from the wall behind it · a name such "
                      "as yellow or red · or a hex colour such as #ffd400"),
    "issue": _Issue("auto: an issue number and the month the video was filmed · your own line, such as "
                    "\"Issue 12 · Spring\" · or off"),
    "frame": O.Choice(("keep", "9:16"), "keep the clip's own shape, or make it 9:16 following the face"),
    "fix": O.Text("misheard words as wrong=right, separated by ;  e.g. \"Jon=John\""),
}

# The runs `tests/matrix.py magazine-cover` makes: the clips in the master list's order (D, B, A, C),
# then the classic look, lines and a title of your own, a re-hook as words are said, and the whole clip.
MATRIX = [
    ("D", "D", {}),
    ("B", "B", {}),
    ("A", "A", {}),
    ("C", "C", {}),
    ("D-classic", "D", {"look": "classic"}),
    ("B-classic", "B", {"look": "classic"}),
    ("B-own-lines", "B", {"title": "Studio", "lines": "Edit *less* | Your best take, found | Captions in a click"}),
    ("A-rehook", "A", {"when": "the marketing campaign"}),
    ("D-whole", "D", {"when": "whole"}),
]


# ---- the words -----------------------------------------------------------------------------------

_SPLIT_AT = set("and or but so while because then".split())
_LEAD_DROP = _SPLIT_AT | set("well like okay ok yeah also plus cause hey oh".split())
# Words a line never ends on: a cut-off sentence ("…to the marketing campaign and") ends clean.
_TAIL_DROP = set("""and or but so the a an to of in on at for with from by my your our their his her its this that
these those is are was were be been being just like really very also""".split())
_GOOD_START = set("how what why who when where the a an your my our this these those every each its their his her "
                  "one two three four five six seven eight nine ten".split())
_CANNOT_START = set("of to in on at for with from by into onto about over under after before between through "
                    "without down up out off".split())
_YOU = {"you", "your", "you're", "yours", "yourself"}


@dataclass
class Said:
    """One cover line taken from the speech."""
    text: str  # as shown: the words as said, fillers out, its edges tidied
    first: int  # word indexes in the words it was taken from
    last: int
    start: float
    end: float
    score: float
    key: int  # its strongest word (an index), for the accent
    key_text: str = ""  # that word as shown
    figure: str = ""  # the amount, written exactly as said, when the line holds one
    label: str = ""  # what the amount is, in the speaker's words

    def as_plan(self) -> dict:
        return {"text": self.text, "said_at": round(self.start, 2), "score": round(self.score, 1),
                "strongest": self.key_text, **({"figure": self.figure, "label": self.label} if self.figure else {})}


def said_words(clip) -> list[Word]:
    """The words as said, for setting in type: the transcriber's own (`clip.heard`), with what it
    was unsure of left out (`words.sure`), sounds that were only breath left out, no hesitations,
    and a figure said in pieces made whole again ("12" "%" is 12%, "2" ",000" is 2,000). The timed
    words the other effects read are right for landing on the voice, but they drop a lone "%", and
    a cover line is set in type, not timed to the voice. An aside, said about the recording rather
    than to the viewer ("I've got the audio set up"), is left out, and so is every word the speaker
    took back ("the lively town," of "the lively town, or the lively city") and every first attempt
    said again: what the shared picker passes over (`moments.taken_back`) is never the title or in a
    cover line. It is found in the timed words, fillers and all, and each timed word is one of the
    heard words with its edges checked (`words.timed`), so the heard words left out are the same
    ones by position: the edges differ, and a short word ("I") can fall outside an aside's timed
    span."""
    from ..core import words as wordsmod
    from ..core.asides import asides

    quiet = clip.quiet
    timed = clip.words
    # the heard words `words.timed` keeps, in its order: one to one with `timed`
    source = [w for w in clip.heard if w.key and not w.unsure and not wordsmod.unheard(w, quiet)]
    back = moments.taken_back(timed, quiet)  # the asides with them
    if len(source) == len(timed):
        drop = {id(source[i]) for i, w in enumerate(timed) if w in back}
    else:  # never expected; fall back to where each word sits
        spans = [(timed[a.first].start - 0.1, timed[a.last].end + 0.1) for a in asides(timed, quiet)]
        spans += [(w.start, w.end) for w in back.why]
        drop = {id(w) for w in clip.heard if any(a <= (w.start + w.end) / 2 <= b for a, b in spans)}
    heard = [w for w in wordsmod.sure(clip.heard) if not wordsmod.unheard(w, quiet) and id(w) not in drop]
    return [w for w, _, _ in num.glue(heard) if w.key and w.key not in HESITATIONS]


def _scrap(words: list[Word], lines: list[list[int]]) -> set[int]:
    """The lines of a sentence the clip came in on halfway (a lower-case first word at the very
    start): a scrap, never a cover line."""
    if not words or words[0].start > 0.25:
        return set()
    first = words[0].clean.strip("\"'“‘(")
    if not first or not first[0].islower() or words[0].key == "i":
        return set()
    out = set()
    for n, ln in enumerate(lines):
        out.add(n)
        if words[ln[-1]].ends_sentence:
            break
    return out if len(out) < len(lines) else set()


def _pieces(words: list[Word], line: list[int]) -> list[list[int]]:
    """A line cut at the joining words inside it ("…like this and improve your…") once the part
    before has two words of its own. A short part left over is dropped later, never glued on."""
    out, cur = [], []
    for i in line:
        if words[i].key in _SPLIT_AT and len([x for x in cur if words[x].key not in _LEAD_DROP]) >= 2:
            out.append(cur)
            cur = [i]
        else:
            cur.append(i)
    if cur:
        out.append(cur)
    return out


_DROP_AT_END = set("""and or but so the a an my your our their his her its this that these those is are was were be
been being just like really very also""".split())
_DETERMINERS = set("a an the my your our their his her its this these those every each".split())
# Words that come before a thing being named, and only before one ("this" can stand alone: "this saves").
_NAMING = set("a an the my your our their his her its".split())


def _tidy(words: list[Word], span: list[int]) -> list[int]:
    """A span with nothing a line cannot start or end on: no joining word or preposition first, no
    article or joining word last, and no preposition last unless the speaker ended on it ("…what
    draws people in," keeps its "in"; "…down to the" is cut)."""
    while span and (words[span[0]].key in _LEAD_DROP or words[span[0]].key in _CANNOT_START):
        span = span[1:]
    while span:
        w = words[span[-1]]
        if w.key in _DROP_AT_END or (w.key in _TAIL_DROP and not w.ends_clause):
            span = span[:-1]
        else:
            break
    return span


def _shown(words: list[Word], span: list[int]) -> str:
    parts = []
    for i in span:
        t = words[i].clean.strip().replace("'", "’")
        t = t.rstrip(",.;:…").lstrip("\"“‘(").rstrip("\"”’)") if i == span[-1] else t.rstrip(",;:…")
        parts.append(t)
    s = " ".join(p for p in parts if p).strip()
    s = s.strip(",;:. ")
    return s[:1].upper() + s[1:] if s else s


def _best_window(words: list[Word], span: list[int], key: int, most: int = 7) -> list[int]:
    """Within a long span, the stretch of at most `most` words that holds its strongest word and
    another word that says something: ending on that word or on the span's own end (a whole
    thought counts for more), starting where a line can start ("how", "the", "your")."""
    k = span.index(key)
    best, best_cost = None, math.inf
    for e in sorted({k, len(span) - 1}):
        for s in range(max(0, e - most + 1), k + 1):
            cand = span[s:e + 1]
            w0 = words[cand[0]].key
            if w0 in _CANNOT_START or w0 in _LEAD_DROP:
                continue
            if sum(1 for i in cand if key_score(words, i) > 0) < 2:
                continue
            cost = (0.0 if w0 in _GOOD_START else 1.5) + 0.3 * abs(len(cand) - 4.5)
            cost -= 0.6 if e == len(span) - 1 else 0.0
            cost += 2.0 if len(cand) < 3 else 0.0
            if cost < best_cost:
                best, best_cost = cand, cost
    return best or span[max(0, k - most + 1):k + 1]


MOST_WORDS = 7  # a cover line of more words than this is cut round its strongest words
MIN_LINE = 4.5  # a phrase scoring less says too little to be a cover line ("The lively")
MIN_TITLE = 7.0  # a word scoring less is not what the clip is about: the run asks for a title
OPENING_S = 3.5  # the opening, for the title: its first sentence, or this many seconds of it


def cover_lines(words: list[Word], quiet, title: str = "") -> list[Said]:
    """The speaker's strongest short phrases, best first. A phrase holding a figure leads; a phrase
    that holds the title loses a little (the masthead already says it), and one that says nothing
    but the title goes. A phrase holding a word the speaker took back (`moments.taken_back`) is
    never one: "the lively town, or the lively city" gives "The lively city come to life"."""
    back = moments.taken_back(words, quiet)
    lines = wb.lines_of(words, quiet)
    scrap = _scrap(words, lines)
    title_key = normalize(title)
    figures = [a for a in num.spoken(words) if a.kind == "amount" and not a.weak]
    out: list[Said] = []
    for n, ln in enumerate(lines):
        if n in scrap:
            continue
        for piece in _pieces(words, ln):
            span = _tidy(words, piece)
            if len(span) < 2 or any(words[i] in back for i in span):
                continue
            inside = [a for a in figures if span[0] <= a.first and a.last <= span[-1]]
            fig = inside[-1] if inside else None  # "from 2,000 to 197K": the figure it lands on
            scored = sorted(((wb.hero_score(words, i, ln, quiet), i) for i in span), reverse=True)
            if scored[0][0] <= 0 and fig is None:
                continue
            key = scored[0][1] if fig is None else fig.first
            if len(span) > MOST_WORDS:
                span = _tidy(words, _best_window(words, span, key))
                if len(span) < 2:
                    continue
                scored = sorted(((wb.hero_score(words, i, ln, quiet), i) for i in span), reverse=True)
                key = scored[0][1] if fig is None or not (span[0] <= fig.first <= span[-1]) else fig.first
            text = _shown(words, span)
            if not text or len(text) > 52:
                continue
            keys = [words[i].key for i in span]
            content = [words[i].key for i in span if key_score(words, i) > 0]
            if title_key and content and all(k == title_key for k in content):
                continue  # it says nothing the masthead does not
            top3 = [min(10.0, v) for v, _ in scored[:3]] + [0.0, 0.0]
            score = top3[0] + 0.6 * top3[1] + 0.3 * top3[2]
            score += 2.0 if _YOU & set(keys) else 0.0
            score += 1.5 if keys[0] in ("how", "what", "why") else 0.0
            score += 1.0 if 3 <= len(span) <= 6 else (-3.0 if len(span) == 2 else 0.0)
            score -= 3.0 if title_key and title_key in keys else 0.0
            said = Said(text, span[0], span[-1], words[span[0]].start, words[span[-1]].end, score, key,
                        key_text=bare(words[key]))
            if fig is not None and span[0] <= fig.first <= span[-1]:
                said.figure = num.text(num.Unit.of([fig]), fig.value)
                said.label = num.label_for(num.Series(amounts=[fig]), words) or _run_label(fig, figures, words)
                said.key_text = fig.said.strip() or said.key_text
                # a figure leads, and the biggest of its kind leads the rest ("…and today we are at 197,000")
                kin = [a.value for a in figures if a.family == fig.family]
                said.score += 8.0 + (2.0 if kin and fig.value >= max(kin) else 0.0)
            out.append(said)
    out = sorted((s for s in out if s.score >= MIN_LINE), key=lambda s: (-s.score, s.start))
    kept: list[Said] = []
    seen: set[str] = set()
    for s in out:  # the best of any two that share words, and a line said twice once
        norm = " ".join(normalize(x) for x in s.text.split())
        if norm in seen or not all(s.last < k.first or s.first > k.last for k in kept):
            continue
        kept.append(s)
        seen.add(norm)
    return kept


def _run_label(fig, figures: list, words: list[Word]) -> str:
    """What a figure counts when nothing is said after it, from the run it ends: in "from 2,000
    followers to 15,000, and today we are at 197,000", 197,000 is followers too."""
    run = num.series(figures, words, near=fig.start)
    if run is None or not any(a.first == fig.first for a in run.amounts):
        return ""
    return num.label_for(run, words)


def topic_word(words: list[Word], quiet) -> tuple[Optional[str], dict]:
    """The word the clip is about, for the masthead: a thing or a name, short enough to set large
    (a masthead fills the width). A word after "a", "the" or "your" is a thing being named; a name
    said mid-sentence counts for most; a word said again, or in the opening sentence, counts up.
    Never a figure, an -ly word, a word of grammar, the name of a real magazine or a word the
    speaker took back (`moments.taken_back`)."""
    back = moments.taken_back(words, quiet)
    lines = wb.lines_of(words, quiet)
    scrap = _scrap(words, lines)
    said_count: dict[str, int] = {}
    for w in words:
        said_count[w.key] = said_count.get(w.key, 0) + 1
    opening: set[int] = set()  # the opening sentence, or its first few seconds when it runs on
    began = None
    for n, ln in enumerate(lines):
        if n in scrap:
            continue
        began = words[ln[0]].start if began is None else began
        if words[ln[0]].start > began + OPENING_S:
            break
        opening |= set(ln)
        if words[ln[-1]].ends_sentence:
            break
    cands = []
    for n, ln in enumerate(lines):
        if n in scrap:
            continue
        for i in ln:
            w = words[i]
            if w in back:
                continue
            k = w.key
            shown = bare(w)
            letters = sum(c.isalpha() for c in shown)
            if (any(c.isdigit() for c in shown) or letters < 3 or k in REAL_TITLES or k in moments.GENERIC
                    or (k.endswith("ly") and len(k) > 4)):
                continue
            base = key_score(words, i)
            if base <= 0:
                continue
            name = shown[:1].isupper() and i > 0 and not words[i - 1].ends_sentence and k != "i"
            s = min(letters, 6) + (2.0 if 4 <= letters <= 6 else 0.0) - 1.5 * max(0, letters - 8)
            before = [words[i - b].key for b in (1, 2) if i - b >= ln[0]]
            s += 3.0 if any(b in _NAMING for b in before) else (1.0 if before and before[0] in _CANNOT_START
                                                                  else 0.0)
            s -= 4.0 if before and before[0] in moments.BEFORE_VERBS else 0.0
            s += 4.0 if name else 0.0
            s += 2.0 * (said_count.get(k, 1) - 1)
            s += 1.5 if i in opening else 0.0
            s += 1.0 if w.ends_clause else 0.0
            cands.append((s, -w.start, i, shown))
    cands = [c for c in cands if c[0] >= MIN_TITLE]
    if not cands:
        return None, {}
    cands.sort(reverse=True)
    best: dict[str, tuple] = {}
    for c in cands:
        best.setdefault(normalize(c[3]), c)
    ranked = sorted(best.values(), reverse=True)
    s, _, i, shown = ranked[0]
    return shown, {"word": shown, "said_at": round(words[i].start, 2), "score": round(s, 1),
                   "runners_up": [c[3] for c in ranked[1:4]]}


# ---- the issue line --------------------------------------------------------------------------------

def filmed_month(path: str) -> str:
    """The month the video was filmed, from its own record of it, else from the file's date."""
    when = None
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=creation_time:stream_tags=creation_time",
                              "-of", "json", path], capture_output=True, text=True)
        data = json.loads(out.stdout or "{}")
        tags = [data.get("format", {}).get("tags", {})] + [s.get("tags", {}) for s in data.get("streams", [])]
        for t in tags:
            v = t.get("creation_time")
            if v:
                when = dt.datetime.fromisoformat(v.replace("Z", "+00:00"))
                break
    except (ValueError, OSError):
        when = None
    if when is None or when.year < 1990:
        when = dt.datetime.fromtimestamp(os.path.getmtime(path))
    return when.strftime("%B %Y")


def issue_text(opt: str, path: str) -> str:
    if opt == "off":
        return ""
    if opt == "auto":
        return f"Issue 01 · {filmed_month(path)}"
    return opt


# ---- the gentle push -------------------------------------------------------------------------------

def _smoother(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * u * (u * (6 * u - 15) + 10)


@dataclass(frozen=True)
class Push:
    """The picture moved `amount` closer (0.03 is three percent) about (ax, ay), the top of the
    head, so the head keeps its place under the masthead: easing in from `t0` to `t1`, then easing
    back over `back` seconds (none: it stays)."""
    amount: float
    ax: float
    ay: float
    t0: float
    t1: float
    back: float

    def scale(self, t: float) -> float:
        if self.amount <= 0 or t <= self.t0:
            return 1.0
        if t <= self.t1:
            return 1.0 + self.amount * _smoother((t - self.t0) / max(1e-3, self.t1 - self.t0))
        if self.back <= 0:
            return 1.0 + self.amount
        return 1.0 + self.amount * (1.0 - _smoother((t - self.t1) / self.back))

    def matrix(self, t: float, k: float = 1.0) -> np.ndarray:
        """The move at time t, for a picture `k` times the output's size."""
        s = self.scale(t)
        return np.float32([[s, 0, self.ax * k * (1 - s)], [0, s, self.ay * k * (1 - s)]])


def pushed(img: np.ndarray, push: Push, t: float, k: float = 1.0, picture: bool = True) -> np.ndarray:
    import cv2

    if push.scale(t) == 1.0:
        return img
    h, w = img.shape[:2]
    return cv2.warpAffine(img, push.matrix(t, k), (w, h), flags=cv2.INTER_CUBIC if picture else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


# ---- the person, as the cover sees them ------------------------------------------------------------

Q = wb.Q  # measured at a quarter of the output size, as words-behind measures


class Store:
    """The person of chosen output frames at 1/Q of the output size, as the cover sees them: the two
    person models agreeing and steadied (`matte.Steady`), in the output's framing, pushed. The same
    shape as words-behind's mask store, so its placing works on it unchanged."""

    def __init__(self, clip, timeline, framing, out_size: tuple[int, int], frames: list[int], push: Optional[Push]):
        import cv2

        from ..core import matte

        W, H = out_size
        self.w, self.h = max(2, W // Q), max(2, H // Q)
        self.by_frame: dict[int, np.ndarray] = {}
        info = clip.info
        fps, n_src = info.rate, info.frames
        want: dict[int, list[int]] = {}
        for j in frames:
            want.setdefault(timeline.source_frame(j, fps, n_src), []).append(j)
        if not want:
            return
        steady = matte.Steady(person_models(clip), info)
        masks = steady.masks
        sx, sy = masks.width / info.width, masks.height / info.height
        try:
            for k in sorted(want):
                m = steady.get(k)
                x, y, w, h = framing.window(k)
                x0, y0 = int(np.floor(x * sx)), int(np.floor(y * sy))
                x1, y1 = int(np.ceil((x + w) * sx)), int(np.ceil((y + h) * sy))
                crop = m[max(0, y0):min(masks.height, y1), max(0, x0):min(masks.width, x1)]
                small = cv2.resize(crop, (self.w, self.h), interpolation=cv2.INTER_AREA)
                for j in want[k]:
                    self.by_frame[j] = pushed(small, push, j / fps, 1.0 / Q, picture=False) if push else small
        finally:
            steady.close()
        self.fps = fps

    def with_push(self, push: Push) -> "Store":
        """The same frames as the cover sees them once the picture is pushed: each small mask moved
        the same way (no second read of the masks)."""
        out = Store.__new__(Store)
        out.w, out.h, out.fps = self.w, self.h, self.fps
        out.by_frame = {j: pushed(m, push, j / self.fps, 1.0 / Q, picture=False) for j, m in self.by_frame.items()}
        return out

    def stack(self) -> np.ndarray:
        if not self.by_frame:
            return np.zeros((0, self.h, self.w), np.float32)
        return np.stack([self.by_frame[j] for j in sorted(self.by_frame)])

    def crowns(self, level: float = 0.6) -> np.ndarray:
        M = self.stack()
        if not len(M):
            return np.zeros((0, self.w), np.int32)
        hit = M >= level
        return np.where(hit.any(axis=1), hit.argmax(axis=1), M.shape[1]).astype(np.int32)


def person_models(clip) -> list:
    """The person masks the cover cuts out with: both models, each taking what the other misses
    for the person (a vase by the hair, a ceiling sprinkler) away."""
    return [clip.person(), clip.person("person-parts")]


# ---- blocks of text ----------------------------------------------------------------------------------

@lru_cache(maxsize=512)
def text_block(lines: tuple[str, ...], face: str, px: int, align: str, lead: float) -> wb.Block:
    """Lines of text as words-behind's Block, so its ink and its measuring work on them unchanged:
    aligned left, right or centre on their ink, `lead` the gap between lines as a share of the
    capital height, with room above the capitals for taller letters and accents, and for the
    overhang of a slanted letter (its ink is never cut off at the block's edge)."""
    f = textmod.font(face, px)
    cap = textmod.cap_height(face, px)
    boxes = [f.getbbox(ln, anchor="ls") for ln in lines]
    lefts = [float(b[0]) for b in boxes]  # each line's ink, from its pen's start
    rights = [float(b[2]) for b in boxes]
    if align == "left":
        shift = max(0.0, max(-l for l in lefts))
        xs = tuple(shift for _ in lines)
        w = max(shift + r for r in rights)
    elif align == "right":
        w = max(r - min(0.0, l) for l, r in zip(lefts, rights))
        xs = tuple(w - r for r in rights)
    else:
        w = max(r - l for l, r in zip(lefts, rights))
        xs = tuple((w - (r - l)) / 2 - l for l, r in zip(lefts, rights))
    above = max(0.0, max(-float(b[1]) for b in boxes) - cap)
    gap = lead * cap
    tops = tuple(above + k * (cap + gap) for k in range(len(lines)))
    h = above + cap * len(lines) + gap * (len(lines) - 1)
    return wb.Block(tuple(lines), face, int(px), cap, gap, float(w), float(h), tops, xs)


def arrangements(words: list[str], most: int = 4) -> list[tuple[str, ...]]:
    """A phrase set on one to `most` lines, each split as evenly as it can be (words-behind's own)."""
    return wb.arrangements(words, most)


# ---- walls, inks and contrast ----------------------------------------------------------------------

def _lum(rgb) -> float:
    c = np.asarray(rgb, np.float32) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2])


def _ratio(a: float, b: float) -> float:
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


class Walls:
    """A few frames of the cover's seconds, read once as the cover will show them (pushed), to
    measure what is behind any block: the wall's light (median, darker and brighter parts), how busy
    it is, and the person's own light under it (hair under a masthead's foot)."""

    def __init__(self, clip, timeline, framing, store: Store, frames: list[int], push: Optional[Push], before=None):
        from ..core import video

        have = [j for j in frames if j in store.by_frame]
        self.picks = have[:: max(1, len(have) // 6)][:6]
        self.frames = {}
        self.W, self.H = framing.out_w, framing.out_h
        fps, n_src = clip.info.rate, clip.info.frames
        for j in self.picks:
            k = timeline.source_frame(j, fps, n_src)
            f = framing.apply(video.read_frame(clip.info, k), k)
            if before is not None:
                f = before(f, j)
            self.frames[j] = pushed(f, push, j / fps) if push else f
        self.store = store

    def behind(self, rect: Rect) -> dict:
        import cv2

        W, H = self.W, self.H
        pad_x, pad_y = 0.03 * W, 0.25 * rect.h
        x0, y0 = int(max(0, rect.x0 - pad_x)), int(max(0, rect.y0 - pad_y))
        x1, y1 = int(min(W, rect.x1 + pad_x)), int(min(H, rect.y1 + pad_y))
        c0, c1 = int(max(0, rect.y0)), int(min(H, rect.y1 + 0.06 * rect.h))
        lums, busy, crowns = [], [], []
        for j in self.picks:
            f = self.frames[j]
            full = cv2.resize(self.store.by_frame[j], (W, H), interpolation=cv2.INTER_LINEAR)
            region = f[y0:y1, x0:x1]
            if region.size == 0:
                continue
            wall = full[y0:y1, x0:x1] < 0.05
            head = full[c0:c1, int(max(0, rect.x0)):int(min(W, rect.x1))] > 0.9
            if head.sum() >= 20:
                crowns.append(_lums(f[c0:c1, int(max(0, rect.x0)):int(min(W, rect.x1))])[head])
            if wall.sum() < 50:
                continue
            lums.append(_lums(region)[wall])
            g = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            size = (max(8, g.shape[1] // 4), max(8, g.shape[0] // 4))
            g = cv2.resize(g, size, interpolation=cv2.INTER_AREA)
            lap = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))
            ws = cv2.resize(wall.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0.9
            if ws.any():
                busy.append(float(lap[ws].mean()))
        out = {"median": 0.5, "low": 0.3, "high": 0.7, "busy": 0.0}
        if lums:
            every = np.concatenate(lums)
            out = {"median": float(np.median(every)), "low": float(np.percentile(every, 15)),
                   "high": float(np.percentile(every, 85)), "busy": float(np.mean(busy)) if busy else 0.0}
        if crowns:
            out["crown"] = float(np.median(np.concatenate(crowns)))
        return out


def _lums(rgb: np.ndarray) -> np.ndarray:
    c = rgb.astype(np.float32) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]


def choose_ink(wall: dict, inks: dict, covered: float = 0.0) -> tuple[str, tuple[int, int, int], float]:
    """Of the inks given ({name: rgb}), the one that stands out most from the wall (allowing the
    moderate shade a busy wall gets), and from the person's crown as far as it covers the letters."""
    crown = wall.get("crown")
    best = None
    for name, ink in inks.items():
        s = min(wb.shade_needed(wall, ink, "solid"), wb.MODERATE)
        cw = wb.contrast_with(wall, ink, "solid", s)
        cc = _ratio(_lum(ink), crown) if crown is not None else 21.0
        wc = 0.35 * float(np.clip(covered / wb.HIDE_TARGET, 0.0, 1.0))
        score = cw ** (1 - wc) * cc ** wc
        if best is None or score > best[0]:
            best = (score, name, ink, cw)
    _, name, ink, cw = best
    return name, ink, cw


# ---- the layout ----------------------------------------------------------------------------------------

# Readable, measured: the share of a cover line's ink the person may hide (median and 90th percentile
# over the frames), and its most-hidden letter as a share of what that letter may lose.
LINE_HIDE_MED, LINE_HIDE_P90, LINE_LETTER = 0.004, 0.03, 0.25
# Sizes, as the capital height's share of the frame's height.
HERO_CAP = (0.030, 0.064)  # the big line, smallest to largest (and at most 0.42 of the masthead's)
LINE_CAP = (0.0150, 0.021)  # the small ones
FIGURE = 2.1  # a figure in the big line stands this many times the line's own size
ISSUE_CAP = 0.0105
RULE = 0.0016  # the thin rule's thickness
RULE_GAP, ISSUE_GAP = 0.008, 0.006  # the rule above the masthead, and the issue line above the rule
GAP = 0.016  # the room kept between any two parts of the cover
CLEAR = 0.014  # the air kept between a cover line and the person, as a share of the width
FAINT = 2.2  # a colour asked for that stands out less than this from the wall is pointed out
SKY_KEEPS = 0.75  # the masthead may give up to a quarter of its size to keep the issue line and rule above it


@dataclass(frozen=True)
class Sub:
    """One block of text in a part, where it sits in the part, and the ink it takes: the part's own
    (`main`), the accent (`accent`), or the part's own as a bold lead-in (`kicker`)."""
    block: wb.Block
    dx: float
    dy: float
    role: str = "main"


@dataclass
class Part:
    """One part of the cover, placed: what it is, its blocks of text (or a rule), where it sits
    (output pixels), its colours and the shade round it."""
    kind: str  # masthead, issue, rule, hero, line
    x: float
    y: float
    w: float
    h: float
    subs: tuple = ()
    align: str = "left"
    ink: tuple = (255, 255, 255)
    shade: float = 0.0
    shade_rgb: tuple = (0, 0, 0)
    accent: Optional[tuple] = None  # the accent's colour, when the part has one
    accent_word: str = ""  # the word in a main block that takes it
    order: int = 0  # the order it comes in
    where: str = ""
    hidden: tuple = (0.0, 0.0, 0.0)  # median, p90, worst letter
    contrast: float = 0.0
    said: Optional[dict] = None

    @property
    def rect(self) -> Rect:
        return Rect(self.x, self.y, self.x + self.w, self.y + self.h)

    @property
    def text(self) -> str:
        return " / ".join(" ".join(s.block.lines) for s in self.subs)

    @property
    def cap(self) -> float:
        return max((s.block.cap for s in self.subs), default=self.h)


def stack(blocks: list[tuple[wb.Block, str]], align: str, gap: float) -> tuple[tuple[Sub, ...], float, float]:
    """Blocks set one under another, `gap` apart, aligned on the same side: the subs, width, height."""
    w = max(b.w for b, _ in blocks)
    subs, y = [], 0.0
    for b, role in blocks:
        dx = 0.0 if align == "left" else (w - b.w if align == "right" else (w - b.w) / 2)
        subs.append(Sub(b, dx, y, role))
        y += b.h + gap
    return tuple(subs), float(w), float(y - gap)


def split_kicker(words: list[str], key_pos: Optional[int]) -> tuple[list[str], list[str]]:
    """A small cover line as a bold lead-in and the rest: the lead-in runs to its strongest word
    when that comes within the first four words ("AN EDITOR / that edits for you"), else it is the
    first two ("HERE'S HOW / to make a video"). A line of three words or fewer is all lead-in, and
    so is one whose rest would be a lone small word."""
    n = len(words)
    if n <= 3:
        return words, []
    k = key_pos + 1 if key_pos is not None and key_pos + 1 <= 4 and n - key_pos - 1 >= 1 else 2
    small = _TAIL_DROP | _CANNOT_START | _DETERMINERS | _LEAD_DROP
    while k > 1 and normalize(words[k - 1]) in small:  # a lead-in never ends on a small word: CAPTIONS / in a click
        k -= 1
    rest = words[k:]
    if len(rest) == 1 and normalize(rest[0]) in (_TAIL_DROP | _CANNOT_START | _DETERMINERS):
        return words, []
    return words[:k], rest


@dataclass(frozen=True)
class Unit:
    """What a cover line is made of, before it is sized: its words, its strongest word's place,
    and, for the big line, the figure it leads with."""
    kind: str  # hero or line
    words: tuple[str, ...]
    key_pos: Optional[int] = None
    figure: str = ""
    label: str = ""


def settings(u: Unit, look: Look, px: int, align: str) -> list[tuple[tuple[Sub, ...], float, float]]:
    """Every way to set unit `u` at size `px`: its blocks stacked, each block on one to three lines."""
    out = []
    if u.kind == "hero":
        lead = 0.2 if look.hero_upper else 0.12
        words = [w.upper() for w in u.words] if look.hero_upper else list(u.words)
        if u.figure:
            fig = text_block((u.figure,), look.masthead, int(round(px * FIGURE)), align, 0.0)
            rest = [w.upper() for w in u.label.split()] if u.label else words
            for lines in arrangements(rest, 2):
                b = text_block(lines, look.hero, px, align, lead)
                out.append(stack([(fig, "accent"), (b, "main")], align, 0.22 * b.cap))
            return out
        for lines in arrangements(words, 3):
            subs, w, h = stack([(text_block(lines, look.hero, px, align, lead), "main")], align, 0.0)
            out.append((subs, w, h))
        return out
    kick, rest = split_kicker(list(u.words), u.key_pos)
    kick = [w.upper() for w in kick]
    deck_px = max(8, int(round(px * look.deck_scale)))
    for k_lines in arrangements(kick, 2 if len(kick) >= 3 else 1):  # two words stay on one line
        kb = text_block(k_lines, look.kicker, px, align, 0.22)
        if not rest:
            out.append(stack([(kb, "kicker")], align, 0.0))
            continue
        for d_lines in arrangements(rest, 3):  # the rest as heard: a name or an "I've" keeps its capital
            db = text_block(d_lines, look.deck, deck_px, align, 0.3)
            out.append(stack([(kb, "kicker"), (db, "main")], align, 0.42 * kb.cap))
    return out


def _scan_subs(subs: tuple[Sub, ...], tops: np.ndarray, x: float, ys: np.ndarray):
    """The person's share of each block's ink at every height (words-behind's measure), the worst
    of the blocks taken: median and 90th percentile of the ink hidden, and the worst letter."""
    med = np.zeros(len(ys))
    p90 = np.zeros(len(ys))
    lp = np.zeros(len(ys))
    if not len(tops):
        return med, p90, lp
    for s in subs:
        m, p, l = wb._scan(s.block, tops, x + s.dx, ys + s.dy)
        med, p90, lp = np.maximum(med, m), np.maximum(p90, p), np.maximum(lp, l)
    return med, p90, lp


def _rows(subs: tuple[Sub, ...]) -> tuple[int, int]:
    """How many lines a setting runs to, and how many its lead-in takes (one when it has none)."""
    kick = len(subs[0].block.lines) if subs[0].role == "kicker" else 1
    return sum(len(sb.block.lines) for sb in subs), kick


def place_unit(u: Unit, look: Look, size_steps: list[tuple[float, float]], zone: Rect, tops: np.ndarray, W: int,
               H: int, taken: list[Rect], face_box: Optional[Rect], slots: list[tuple[str, Rect]]):
    """The best setting of `u` in one of `slots` (named areas, in order of preference) where the
    person hides none of it, it keeps clear of every part already placed and of the face, and it
    stays in the zone. Best is large, on few lines, with its lead-in on one: a size a little smaller
    on fewer lines wins over the largest broken into a narrow column. `size_steps` are capital
    heights (shares of the frame's height), largest range first. Returns (subs, x, y, w, h, align,
    slot, hidden) or None."""
    face = look.hero if u.kind == "hero" else look.kicker
    cap_per_px = textmod.cap_height(face, 400) / 400.0
    sizes: list[int] = []
    for lo, hi in size_steps:
        px = max(8, int(hi * H / cap_per_px))
        lo_px = max(8, int(lo * H / cap_per_px))
        while px >= lo_px:
            if px not in sizes:
                sizes.append(px)
            px = int(px * 0.95) if px > 40 else px - 2
    if not sizes:
        return None
    top_px = float(sizes[0])
    row_cost = 0.07 if u.kind == "hero" else 0.10
    best = None
    for px in sizes:
        if best is not None and px / top_px <= best[0]:
            break  # nothing smaller can beat it, even on one line
        for rank, (name, area) in enumerate(slots):
            align = "right" if name == "right" else ("centre" if name == "above" else "left")
            for subs, w, h in sorted(settings(u, look, px, align), key=lambda o: _rows(o[0])):
                rows, kick = _rows(subs)
                score = px / top_px - row_cost * (rows - 1) - 0.12 * (kick - 1) - 0.03 * rank
                if best is not None and score <= best[0]:
                    continue
                if w > area.w + 0.5 or h > area.h + 0.5:
                    continue
                x = area.x0 if align == "left" else (area.x1 - w if align == "right" else area.cx - w / 2)
                ys = np.arange(area.y0, area.y1 - h + 0.01, max(2.0, Q / 2.0))
                if name == "above":
                    ys = ys[::-1]  # as close above the masthead as it fits
                if not len(ys):
                    continue
                med, p90, lp = _scan_subs(subs, tops, x, ys)
                # air round the person: the same setting nudged toward them must stay clear too
                near = CLEAR * W
                toward = near if align == "left" else (-near if align == "right" else 0.0)
                m2, p2, l2 = _scan_subs(subs, tops, x + toward, ys + near)
                for i, y in enumerate(ys):
                    if not (max(med[i], m2[i]) <= LINE_HIDE_MED and max(p90[i], p2[i]) <= LINE_HIDE_P90
                            and max(lp[i], l2[i]) <= LINE_LETTER):
                        continue
                    r = Rect(x, float(y), x + w, float(y) + h)
                    if not r.inside(zone) or not _free_of(r, taken, GAP * H):
                        continue
                    if face_box is not None and r.intersects(face_box, margin=0.01 * W):
                        continue
                    best = (score, (subs, x, float(y), w, h, align, name, (float(med[i]), float(p90[i]), float(lp[i]))))
                    break
    return best[1] if best else None


def _free_of(rect: Rect, taken: list[Rect], gap: float) -> bool:
    return not any(rect.intersects(r, margin=gap) for r in taken)


# ---- drawing --------------------------------------------------------------------------------------------

@dataclass
class Sprite2:
    """A part drawn once: premultiplied colour and alpha (the letters, their soft shadow and the
    shade round them), and the letters' own alpha for measuring, all the same size."""
    rgb: np.ndarray
    a: np.ndarray
    letters: np.ndarray
    x: float  # top-left when landed
    y: float


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    import cv2

    return a if sigma <= 0.3 else cv2.GaussianBlur(a, (0, 0), sigma)


def draw_part(p: Part) -> Sprite2:
    """The part as pixels: every block in its ink (the accent where the part has one), a soft
    shadow under light letters, and the shade round them when the wall asks for one."""
    if p.kind == "rule":
        h, w = max(1, int(round(p.h))), max(1, int(round(p.w)))
        a = np.ones((h, w), np.float32)
        return Sprite2(a[..., None] * np.array(p.ink, np.float32), a, a.copy(), p.x, p.y)
    if p.kind == "issue":
        return _draw_spaced(p)
    cap = p.cap
    pad = int(math.ceil(0.5 * cap)) + wb.MARGIN
    hang = int(math.ceil(0.4 * cap))
    Wc, Hc = int(math.ceil(p.w)) + 2 * pad, int(math.ceil(p.h)) + 2 * pad + hang
    letters = np.zeros((Hc, Wc), np.float32)
    rgb = np.zeros((Hc, Wc, 3), np.float32)
    main = np.array(p.ink, np.float32)
    accent = np.array(p.accent if p.accent is not None else p.ink, np.float32)
    for s in p.subs:
        m, _ = wb.block_ink(s.block)
        col = accent if s.role == "accent" else main
        ox, oy = int(round(pad + s.dx - wb.MARGIN)), int(round(pad + s.dy - wb.MARGIN))
        h, w = m.shape
        h, w = min(h, Hc - oy), min(w, Wc - ox)
        m = m[:h, :w]
        tint = np.broadcast_to(col, (h, w, 3)).copy()
        if s.role == "main" and p.accent is not None and p.accent_word:
            acc = _word_mask(s.block, p.accent_word)
            if acc is not None:
                acc = acc[:h, :w]
                tint = tint * (1 - acc[..., None]) + accent * acc[..., None]
        region = letters[oy:oy + h, ox:ox + w]
        rgb[oy:oy + h, ox:ox + w] = rgb[oy:oy + h, ox:ox + w] * (1 - m[..., None]) + m[..., None] * tint
        letters[oy:oy + h, ox:ox + w] = region + m * (1 - region)
    a = letters.copy()
    if _lum(p.ink) > 0.3:  # a soft shadow under light letters lifts them off a busy wall: dark, not colour
        shadow = np.roll(_blur(letters, 0.05 * cap), int(round(0.03 * cap)), axis=0) * 0.38
        a = letters + shadow * (1 - letters)
    if p.shade > 0:
        sh = wb.shade_of(letters, cap, p.shade)
        rgb = (sh * (1 - a))[..., None] * np.array(p.shade_rgb, np.float32) + rgb
        a = sh + a * (1 - sh)
    return Sprite2(rgb.astype(np.float32), a.astype(np.float32), letters, p.x - pad, p.y - pad)


def _word_mask(b: wb.Block, word: str) -> Optional[np.ndarray]:
    """The ink of one word of the block (its first showing), the size of the block's own ink."""
    want = normalize(word)
    for n, (_, wtext, _, _) in enumerate(wb.word_slots(b)):
        if normalize(wtext) == want:
            m, _ = wb.block_ink(b, (n,))
            return m
    return None


SPACING = 0.16  # the air between the issue line's letters, as a share of its size


def spaced_width(text: str, face: str, px: int) -> float:
    f = textmod.font(face, px)
    return sum(f.getlength(ch) for ch in text) + SPACING * px * max(0, len(text) - 1)


def _draw_spaced(p: Part) -> Sprite2:
    """The issue line: small capitals with air between the letters, centred on the part, its
    baseline on the part's foot."""
    from PIL import Image, ImageDraw

    text, face, px = p.said["text"], p.said["face"], int(p.said["px"])
    f = textmod.font(face, px)
    cap = textmod.cap_height(face, px)
    total = spaced_width(text, face, px)
    W, Hh = int(math.ceil(total)) + 8, int(math.ceil(cap * 1.7)) + 8
    img = Image.new("L", (W, Hh), 0)
    d = ImageDraw.Draw(img)
    x, base = 4.0, 4 + cap * 1.3
    for ch in text:
        d.text((x, base), ch, font=f, fill=255, anchor="ls")
        x += f.getlength(ch) + SPACING * px
    m = np.asarray(img, np.float32) / 255.0
    rgb, a, pad = wb.style_letters(m, cap, "solid", p.ink)
    letters = np.pad(m, pad).astype(np.float32)
    return Sprite2(rgb.astype(np.float32), a.astype(np.float32), letters,
                   p.x + p.w / 2 - W / 2 - pad, p.y + p.h - base - pad)


# ---- the motion -----------------------------------------------------------------------------------------

MAST_IN, MAST_LEAD = 0.5, 0.45  # the masthead's rise: how long, and how long before its moment it starts
STAGGER = 0.09  # between one line and the next
PART_IN = 0.34
OUT = 0.32  # every part leaves in this
PUSH = 0.03  # how much closer the picture settles


def _ease_out(u: float, p: float = 3.0) -> float:
    u = min(1.0, max(0.0, u))
    return 1.0 - (1.0 - u) ** p


@dataclass
class Timing:
    t0: float  # the masthead starts to rise
    land: float  # the masthead is in place
    t_out: float  # everything has left (or the clip's end when it holds)
    hold_end: bool

    def start_of(self, p: Part) -> float:
        if p.kind == "masthead":
            return self.t0
        if p.kind == "rule":
            return self.t0 + 0.22
        if p.kind == "issue":
            return self.t0 + 0.32
        return self.land - 0.05 + STAGGER * p.order

    def state(self, p: Part, t: float) -> Optional[tuple[float, float, float, bool, float]]:
        """(opacity, dx, dy, foot cut, width share) of part p at time t, or None when it is not shown."""
        s0 = self.start_of(p)
        if t < s0 or (not self.hold_end and t >= self.t_out):
            return None
        dur = MAST_IN if p.kind == "masthead" else PART_IN
        u = (t - s0) / dur
        op, dx, dy, cut, wshare = 1.0, 0.0, 0.0, False, 1.0
        if u < 1.0:
            if p.kind == "masthead":  # up out of a line behind the head
                op, dy, cut = min(1.0, u / 0.3), (1.0 - _ease_out(u, 3.0)) * 1.05 * p.h, True
            elif p.kind == "rule":
                wshare = _ease_out(u, 3.0)
            elif p.kind == "issue":
                op, dy = _smoother(u), (1.0 - _ease_out(u)) * 0.6 * p.h
            else:  # a line slides in from its own side
                side = 1.0 if p.align == "right" else (-1.0 if p.align == "left" else 0.0)
                op = _smoother(u / 0.8)
                dx = side * (1.0 - _ease_out(u, 3.0)) * 1.4 * p.cap  # a short slide, in step with the text's size
                dy = 0.0 if side else (1.0 - _ease_out(u, 3.0)) * 0.25 * p.h
        if not self.hold_end and t > self.t_out - OUT:
            v = (t - (self.t_out - OUT)) / OUT
            op *= 1.0 - _smoother(v)
            if p.kind == "masthead":
                dy += _ease_out(v, 2.0) * 0.6 * p.h
                cut = True
        return op, dx, dy, cut, wshare


class Painter:
    """Draws the cover on each output frame: the push, then every part in one layer behind the person.

    Only what the cover touches is worked on: the person's cut-out for the rows it spans, and the
    edge and the compositing inside a box round each part (the edge work reaches a few pixels past
    the letters, so a box with that margin gives the same pixels as the whole frame would)."""

    def __init__(self, clip, timeline, framing, parts: list[Part], sprites: list[Sprite2], timing: Timing,
                 push: Optional[Push], grade_layer=None):
        self.clip, self.timeline, self.framing = clip, timeline, framing
        self.parts, self.sprites, self.timing, self.push = parts, sprites, timing, push
        self.grade = grade_layer
        self.W, self.H = framing.out_w, framing.out_h
        self.steady = None
        self.sigma = max(4.0, 0.012 * self.W)
        self.pad = int(math.ceil(4 * self.sigma))  # how far the edge work looks round a pixel
        self.mast = next((k for k, p in enumerate(parts) if p.kind == "masthead"), None)
        self.measured: list[tuple[float, float]] = []  # (t, the masthead's share hidden) once landed
        self._n = 0

    def _small(self, j: int) -> np.ndarray:
        from ..core import matte

        info = self.clip.info
        k = self.timeline.source_frame(j, info.rate, info.frames)
        if self.steady is None:
            self.steady = matte.Steady(person_models(self.clip), info)
        return self.steady.get(k), k

    def _alpha(self, frame: np.ndarray, j: int, y0: int, y1: int) -> tuple[np.ndarray, int]:
        """The person's cut-out for output rows y0..y1 of the picture as read (before the push), at
        least: (the rows worked out, the first of them). On a clip shown whole, only a band of rows
        is worked out, its edges on whole rows of the small mask and with room for the edge filter;
        a reframed clip has the whole frame worked out."""
        m, k = self._small(j)
        masks = self.steady.masks
        W, H = self.W, self.H
        if self.framing.reframed:
            a = masks.full(m, (W, H), guide=frame, crop=self.framing.window(k), lean=wb.EDGE, reach=wb.REACH * W)
            return a, 0
        f = H / masks.height  # output rows per row of the small mask
        margin = self.pad + 8
        s0 = max(0, int(math.floor((y0 - margin) / f)))
        s1 = min(masks.height, int(math.ceil((y1 + margin) / f)))
        r0, r1 = int(round(s0 * f)), min(H, int(round(s1 * f)))
        a = masks.full(m[s0:s1], (W, r1 - r0), guide=frame[r0:r1], lean=wb.EDGE, reach=wb.REACH * W)
        return a, r0

    def active(self, t: float) -> bool:
        return t >= self.timing.t0 and (self.timing.hold_end or t < self.timing.t_out)

    def layer(self, t: float):
        """The parts as they are at time t, in one layer over the rows they span: (first row, last
        row, colour, alpha, the boxes they sit in, the masthead's letters where they are now)."""
        W, H = self.W, self.H
        placed = []
        for k, (p, s) in enumerate(zip(self.parts, self.sprites)):
            st = self.timing.state(p, t)
            if st is None or st[0] <= 0.002:
                continue
            op, dx, dy, cut, wshare = st
            srgb, sa, sl = s.rgb, s.a, s.letters
            x = s.x
            if wshare < 0.999:  # a rule drawing out from its middle
                keep = max(1, int(round(sa.shape[1] * wshare)))
                c0 = (sa.shape[1] - keep) // 2
                srgb, sa, sl = srgb[:, c0:c0 + keep], sa[:, c0:c0 + keep], sl[:, c0:c0 + keep]
                x = s.x + c0
            xi, yi = int(round(x + dx)), int(round(s.y + dy))
            h, w = sa.shape
            dx0, dy0, dx1, dy1 = max(0, xi), max(0, yi), min(W, xi + w), min(H, yi + h)
            if dx1 <= dx0 or dy1 <= dy0:
                continue
            placed.append((k, p, op, cut, srgb, sa, sl, xi, yi, (dx0, dy0, dx1, dy1)))
        if not placed:
            return None
        boxes = _merge([b for *_, b in placed], self.pad, W, H)
        y_lo = min(b[1] for b in boxes)  # the layer spans the boxes, margins and all
        y_hi = max(b[3] for b in boxes)
        rgb = np.zeros((y_hi - y_lo, W, 3), np.float32)
        a = np.zeros((y_hi - y_lo, W), np.float32)
        mast = None
        for k, p, op, cut, srgb, sa, sl, xi, yi, (dx0, dy0, dx1, dy1) in placed:
            src = (slice(dy0 - yi, dy1 - yi), slice(dx0 - xi, dx1 - xi))
            ia, irgb = sa[src] * op, srgb[src] * op
            if cut:  # nothing below where the masthead's foot lands: it rises out of a line
                foot = int(round(p.y + p.h + 0.02 * p.h)) - dy0
                if foot < ia.shape[0]:
                    ia, irgb = ia.copy(), irgb.copy()
                    ia[max(0, foot):], irgb[max(0, foot):] = 0, 0
            dst = (slice(dy0 - y_lo, dy1 - y_lo), slice(dx0, dx1))
            rgb[dst] = rgb[dst] * (1 - ia[..., None]) + irgb
            a[dst] = a[dst] * (1 - ia) + ia
            if k == self.mast and t >= self.timing.land + 0.05 and (self.timing.hold_end or t < self.timing.t_out - OUT):
                mast = ((slice(dy0, dy1), slice(dx0, dx1)), sl[src])
        return y_lo, y_hi, rgb, a, boxes, mast

    def draw(self, frame: np.ndarray, j: int, t: float) -> None:
        import cv2

        if self.grade is not None:
            self.grade.draw(frame, j, t)
        if not self.active(t):
            return
        got = self.layer(t)
        s = self.push.scale(t) if self.push is not None else 1.0
        if got is None:
            if s != 1.0:
                frame[:] = pushed(frame, self.push, t)
            return
        y_lo, y_hi, rgb, a, boxes, mast = got
        W, H = self.W, self.H
        b0, b1 = y_lo, y_hi  # the rows wanted, as shown (pushed)
        if s != 1.0:  # the same rows before the push, a little wider
            ay = self.push.ay
            u0 = int(math.floor(ay + (b0 - ay) / s)) - 2
            u1 = int(math.ceil(ay + (b1 - ay) / s)) + 2
        else:
            u0, u1 = b0, b1
        band, r0 = self._alpha(frame, j, max(0, u0), min(H, u1))
        if s != 1.0:
            frame[:] = pushed(frame, self.push, t)
            M = self.push.matrix(t)
            M[1, 2] += s * r0 - b0  # the band's own rows: from band row y to shown row y
            alpha = cv2.warpAffine(band, M, (W, b1 - b0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
            alpha = band[b0 - r0:b1 - r0]
        # Behind the person everywhere the cover is: a plain blend where the cut-out is sure (all
        # wall, or all person), and the careful edge work (words-behind's) only round the pixels
        # where the person's edge crosses the cover.
        band = frame[b0:b1]
        I = band.astype(np.float32)
        out = I + (1.0 - alpha[..., None]) * (rgb - a[..., None] * I)
        frame[b0:b1] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        for x0, y0, x1, y1 in _edge_boxes(alpha, a, self.pad):
            region = I[y0:y1, x0:x1].astype(np.uint8)  # the picture as it was, before the plain blend
            al = wb.refine_edge(region, alpha[y0:y1, x0:x1], self.sigma * 0.75)
            frame[b0 + y0:b0 + y1, x0:x1] = wb.composite_behind(region, rgb[y0:y1, x0:x1], a[y0:y1, x0:x1], al,
                                                                  self.sigma)
        self._n += 1
        if mast is not None and self._n % 3 == 0:
            (sy, sx), letters = mast
            tot = float(letters.sum())
            if tot > 0:
                hid = alpha[sy.start - b0:sy.stop - b0, sx]
                self.measured.append((t, float((letters * hid).sum()) / tot))

    def close(self) -> None:
        if self.steady is not None:
            self.steady.close()


def _edge_boxes(alpha: np.ndarray, a: np.ndarray, pad: int) -> list[tuple[int, int, int, int]]:
    """Boxes round every pixel where the person's soft edge meets the cover (the person neither
    surely there nor surely not, and the cover drawn there), each grown by `pad`, touching ones
    joined: the only places the edge work changes anything."""
    import cv2

    edge = (alpha > 0.02) & (alpha < 0.98) & (a > 0.004)
    if not edge.any():
        return []
    h, w = edge.shape
    k = 8  # found on a grid eight times coarser, which is plenty for a box
    small = cv2.resize(edge.astype(np.float32), (max(1, w // k), max(1, h // k)), interpolation=cv2.INTER_AREA) > 0
    n, _, stats, _ = cv2.connectedComponentsWithStats(small.astype(np.uint8), connectivity=8)
    boxes = [(int(stats[i, 0] * k), int(stats[i, 1] * k), int((stats[i, 0] + stats[i, 2]) * k + k),
              int((stats[i, 1] + stats[i, 3]) * k + k)) for i in range(1, n)]
    return _merge(boxes, pad, w, h)


def _merge(boxes: list[tuple[int, int, int, int]], pad: int, W: int, H: int) -> list[tuple[int, int, int, int]]:
    """Boxes grown by `pad` (inside the frame), the ones that touch joined into one."""
    out = [(max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)) for x0, y0, x1, y1 in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(out)):
            for k in range(i + 1, len(out)):
                a, b = out[i], out[k]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    out[i] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    out.pop(k)
                    merged = True
                    break
            if merged:
                break
    return out


# ---- deciding what and when ----------------------------------------------------------------------------

def _is_time(v: str) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False


def decide_timing(clip, words: list[Word], quiet, when: str, seconds: float) -> tuple[Timing, dict]:
    """When the cover lands and leaves, and why."""
    info: dict = {}
    if when == "whole":
        land = min(MAST_LEAD, seconds / 4)
        return Timing(max(0.0, land - MAST_LEAD), land, seconds, True), {"when": "the whole clip"}
    if when == "auto":
        land = MAST_LEAD
        line_end = None
        lines = wb.lines_of(words, quiet) if words else []
        scrap = _scrap(words, lines)
        for n, ln in enumerate(lines):
            if n not in scrap and words[ln[-1]].end > 1.0:
                line_end = words[ln[-1]].end
                break
        t_out = min(max((line_end or 0.0) + 0.5, land + 3.2), land + 4.8)
        info["when"] = "the opening" + (f", to the end of its first line ({line_end:.2f} s)" if line_end else "")
    elif _is_time(when):
        land = float(when) + MAST_LEAD
        t_out = land + 3.6
        info["when"] = f"from {float(when):.2f} s"
    else:
        span = wb.find_said(words, when)
        if span is None:
            raise ValueError(f"could not find \"{when}\" in what is said; `python -m fx.run words <video>` lists the "
                             "words, or give a time such as when=4.5")
        land = words[span[0]].start
        end = words[span[1]].end
        t_out = min(max(end + 0.5, land + 2.6), land + 4.4)
        info["when"] = f"as \"{when}\" is said ({land:.2f} s)"
        info["said_at"] = round(land, 3)
    if land >= seconds - 0.8:
        raise ValueError(f"the cover would land at {land:.2f} s, too close to the end of the clip ({seconds:.2f} s)")
    hold_end = t_out >= seconds - 0.5
    return Timing(max(0.0, land - MAST_LEAD), land, seconds if hold_end else t_out, hold_end), info


def _plan_parts(plan: dict, W: int, H: int, PlanError) -> tuple[list[Part], Timing, Optional[Push], dict]:
    """The parts, the timing and the push a plan holds, checked."""
    try:
        parts = []
        for q in plan["parts"]:
            kind = str(q["kind"])
            if kind not in ("masthead", "issue", "rule", "hero", "line"):
                raise PlanError(f"a part's kind must be masthead, issue, rule, hero or line (got {kind!r})")
            p = Part(kind, float(q["x"]), float(q["y"]), float(q["w"]), float(q["h"]), align=str(q.get("align", "left")),
                     ink=tuple(textmod.hex_rgb(q["ink"])), shade=float(q.get("shade", 0.0)),
                     shade_rgb=tuple(textmod.hex_rgb(q.get("shade_ink", "#000000"))),
                     accent=tuple(textmod.hex_rgb(q["accent"])) if q.get("accent") else None,
                     accent_word=str(q.get("accent_word", "")), order=int(q.get("order", 0)), where=str(q.get("where", "")))
            subs = []
            for bq in q.get("blocks", []):
                face, px = str(bq["face"]), int(bq["px"])
                if face not in textmod.FACES:
                    raise PlanError(f"a block's face must be one of the pack's fonts: {', '.join(textmod.FACES)}")
                if not 6 <= px <= 2400:
                    raise PlanError(f"a block's px {px} is not a usable size")
                blk = _block_from(bq, p.align)
                subs.append(Sub(blk, float(bq.get("dx", 0.0)), float(bq.get("dy", 0.0)), str(bq.get("role", "main"))))
            p.subs = tuple(subs)
            if kind in ("masthead", "hero", "line"):
                if not subs:
                    raise PlanError(f"a {kind} part needs its blocks of text")
                p.w = max(sb.dx + sb.block.w for sb in subs)
                p.h = max(sb.dy + sb.block.h for sb in subs)
            elif kind == "issue":
                p.said = {"text": str(q["text"]), "face": str(q["face"]), "px": int(q["px"])}
            parts.append(p)
        tm = plan["timing"]
        timing = Timing(float(tm["t0"]), float(tm["land"]), float(tm["out"]), bool(tm["hold_to_end"]))
        pu = plan.get("push")
        push = Push(float(pu["amount"]), float(pu["ax"]), float(pu["ay"]), float(pu["t0"]), float(pu["t1"]),
                    float(pu["back"])) if pu else None
    except (KeyError, TypeError, ValueError) as e:
        if isinstance(e, PlanError):
            raise
        raise PlanError(f"the plan's parts, timing or push cannot be read ({e}); run once without --plan to get one "
                        "to edit") from None
    if sum(p.kind == "masthead" for p in parts) != 1:
        raise PlanError("the plan needs exactly one masthead part")
    for p in parts:
        if not (0 <= p.x <= W and 0 <= p.y <= H):
            raise PlanError(f"a {p.kind} part sits outside the frame ({p.x:.0f}, {p.y:.0f})")
    if push is not None and not 0.0 <= push.amount <= 0.15:
        raise PlanError("the push is a share from 0 to 0.15")
    return parts, timing, push, plan.get("chosen", {})


# ---- the effect ------------------------------------------------------------------------------------------

def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from ..core import framing as framing_mod
    from ..core.timeline import Timeline
    from . import Job, PlanError
    from . import layers

    if plan is not None:  # the plan is the record: its look comes back with it
        kept = plan.get("options") or {}
        opts = dict(opts, **{k: kept[k] for k in ("look", "frame") if k in kept})
        if opts["look"] not in LOOKS or opts["frame"] not in ("keep", "9:16"):
            raise PlanError("the plan's options name a look or frame this effect does not have")
    look = LOOKS[opts["look"]]
    info = clip.info
    fps = info.rate
    fr = framing_mod.plan(info.width, info.height, fps, info.frames, clip.face, shape=opts["frame"])
    tl = Timeline.whole(info.frames / fps)
    W, H = fr.out_w, fr.out_h
    seconds = tl.duration
    n_out = tl.output_frames(fps)
    zone = SafeZone(W, H).rect
    words = [w for w in clip.words if w.key not in HESITATIONS]
    quiet = clip.quiet
    warnings: list[str] = []

    if plan is not None:
        parts, timing, push, chosen = _plan_parts(plan, W, H, PlanError)
    else:
        timing, when_info = decide_timing(clip, words, quiet, opts["when"], seconds)
        chosen = dict(when_info)

    t_end = seconds if timing.hold_end else timing.t_out
    span = list(range(int(math.floor(timing.t0 * fps)), int(min(n_out - 1, math.ceil(t_end * fps))) + 1))
    held = [j for j in span if timing.land <= j / fps <= max(timing.land, t_end - OUT)]
    sample = held[:: max(1, len(held) // 40)] or span[:: max(1, len(span) // 40)]

    grade_layer = None
    if look.grade:
        grade_layer = _grade_under(clip, layers, look.grade, timing, seconds, fr)

    def graded(frame: np.ndarray, j: int) -> np.ndarray:
        if grade_layer is None:
            return frame
        f = frame.copy()
        grade_layer.draw(f, j, j / fps)
        return f

    if plan is None:
        # the push, about the top of the head as it is while the cover holds
        pre = Store(clip, tl, fr, (W, H), sample, None)
        if not pre.by_frame or max(float(m.max()) for m in pre.by_frame.values()) < 0.5:
            raise ValueError("nobody is in the shot while the cover is on screen, so there is nobody to be its cover star")
        crowns = pre.crowns()
        col = np.median(crowns, axis=0)
        inside = np.flatnonzero(col < pre.h)
        ay = float(np.median(np.min(crowns, axis=1))) * Q
        ax = float(np.median(inside)) * Q if len(inside) else W / 2
        face_x = [r.cx for j in sample if (r := face_at(clip, tl, fr, None, j)) is not None]
        if face_x:
            ax = float(np.median(face_x))
        push = Push(PUSH, ax, ay, timing.t0, timing.t_out - (0 if timing.hold_end else OUT),
                    0.0 if timing.hold_end else OUT)
        store = pre.with_push(push)
    else:
        store = Store(clip, tl, fr, (W, H), sample, push)
    if not store.by_frame:
        raise ValueError("nobody is in the shot while the cover is on screen")

    if plan is None:
        said = said_words(clip)
        title, why_title = _title(opts, said, quiet)
        lines_text, line_info = _lines(opts, said, quiet, title)
        if not lines_text and opts["lines"] == "auto":
            warnings.append("nothing said makes a strong cover line, so the masthead stands alone; give your own: "
                            "lines=\"Big line | small one | another\"")
        issue = issue_text(opts["issue"], clip.path)
        walls = Walls(clip, tl, fr, store, sample, push, before=graded if grade_layer is not None else None)
        parts = layout(look, title, lines_text, line_info, issue, store, walls, clip, tl, fr, sample, opts, warnings,
                       push)
        chosen.update({"title": why_title, "lines": [li.as_plan() if isinstance(li, Said) else li for li in line_info],
                       "issue": issue})

    sprites = [draw_part(p) for p in parts]
    painter = Painter(clip, tl, fr, parts, sprites, timing, push, grade_layer)

    mast = next(p for p in parts if p.kind == "masthead")
    landed = [j for j in range(n_out) if timing.land <= j / fps < (t_end if not timing.hold_end else seconds + 1)]
    over_face = _over_face(clip, tl, fr, store, parts, landed, push)
    outside = [p.kind for p in parts if not p.rect.inside(zone)]
    lines_placed = [p for p in parts if p.kind in ("hero", "line")]
    checks = {
        "title": mast.text + (f" ({chosen.get('title')})" if chosen.get("title") else ""),
        "lines": " · ".join(f"{'BIG ' if p.kind == 'hero' else ''}\"{p.text}\" ({p.where})"
                            for p in sorted(lines_placed, key=lambda p: p.order)) or "none",
        "issue": next((p.said["text"] for p in parts if p.kind == "issue"), "none"),
        "on_screen": f"{timing.t0:.2f}–{t_end:.2f} s" + (f", {chosen['when']}" if chosen.get("when") else ""),
        "frames_over_face": over_face,
        "frames_outside_safe_zone": len(landed) if outside else 0,
        "placed": mast.where,
        "hidden": f"masthead {mast.hidden[0]:.0%} behind the head (at worst {mast.hidden[1]:.0%}; the most-hidden "
                  f"letter at {mast.hidden[2]:.0%} of what it may lose); lines at most "
                  f"{max([p.hidden[1] for p in lines_placed], default=0):.0%}",
        "contrast": " · ".join(f"{p.kind} {p.contrast:.1f}:1" + (f" (wall shaded {p.shade:.0%})" if p.shade else "")
                               for p in parts if p.kind in ("masthead", "hero", "line", "issue")),
        "push": f"the picture settles {push.amount:.0%} closer about the top of the head" if push else "none",
        "warnings": warnings,
    }
    out_plan = {
        "effect": NAME, "clip": clip.path, "options": opts,
        "output": {"width": W, "height": H, "fps": round(fps, 3), "seconds": round(seconds, 3), "framing": fr.note},
        "chosen": chosen,
        "timing": {"t0": round(timing.t0, 6), "land": round(timing.land, 6), "out": round(timing.t_out, 6),
                   "hold_to_end": timing.hold_end},
        "push": {"amount": push.amount, "ax": round(push.ax, 4), "ay": round(push.ay, 4), "t0": round(push.t0, 6),
                 "t1": round(push.t1, 6), "back": round(push.back, 6)} if push else None,
        "parts": [_part_plan(p) for p in parts],
        "grade": look.grade,
        "sheet_times": sheet_times(timing, t_end, fps, n_out),
    }

    def still_times() -> list[float]:
        hold_mid = timing.land + 0.55 + 0.5 * max(0.2, t_end - timing.land - 0.55)
        return [round(v, 3) for v in (timing.t0 + 0.3, timing.land + 0.55, hold_mid, max(timing.land + 0.6, t_end - 0.4))]

    def after(result: dict, out_path) -> dict:
        painter.close()
        if not painter.measured:
            return {}
        h = np.array([v for _, v in painter.measured])
        return {"masthead hidden on the render": f"{np.median(h):.0%} behind the head (at worst {h.max():.0%}) "
                                                 f"over {len(h)} frames"}

    return Job(NAME, clip, tl, fr, painter.draw, out_plan, checks=checks, still_times=still_times, after=after)


GRADE_FINISH = {"grain": "0.3", "vignette": "0.25"}  # the film's grain and corners, on the warm look


def _grade_under(clip, layers, look_name: str, timing: Timing, seconds: float, fr):
    """The pack's colour look over the cover's seconds, laid as the pack lays one effect inside
    another (`layers`), turning off as the cover leaves. A clip reframed to 9:16 has the look's own
    pieces built on this framing instead, since a layer must keep the clip's own shape."""
    end = None if timing.hold_end else max(timing.t0 + 0.5, min(seconds, timing.t_out - OUT))
    if not fr.reframed:
        return layers.prepare(clip, layers.Spec("grade", dict(GRADE_FINISH, look=look_name)), timing.t0, end, fr)
    from . import grade

    duration = clip.info.frames / clip.info.rate
    span = {"start": round(timing.t0, 3), "end": round(end if end is not None else duration, 3), "said": {}}
    recipe = grade.RECIPES[look_name]
    graded = grade.GradedFrames(clip.info, fr, grade.lut_file(recipe.name, 1.0))
    vignette, grain = float(GRADE_FINISH["vignette"]), float(GRADE_FINISH["grain"])
    finish = grade.Finish(fr.out_w, fr.out_h, vignette, grain, grade.subject_centre(clip, fr.out_w, fr.out_h))

    class Reframed:
        def draw(self, frame: np.ndarray, j: int, t: float) -> None:
            w = grade.ramp(t, span, duration)
            if w <= 0:
                return
            g = graded.at(min(clip.info.frames - 1, j))  # the whole clip plays: frame j is the clip's frame j
            frame[:] = g if w >= 0.999 else np.clip(frame + (g.astype(np.float32) - frame) * w + 0.5, 0, 255).astype(np.uint8)
            finish.apply(frame, j, w)

    return Reframed()


def _title(opts: dict, words: list[Word], quiet) -> tuple[str, str]:
    if opts["title"] != "auto":
        return opts["title"].upper(), "as given"
    word, info = topic_word(words, quiet)
    if word is None:
        raise ValueError("no spoken word stands out enough to be the title (or nothing clear is said); give one: "
                         "title=\"YOUR TITLE\"")
    alts = ", ".join(info.get("runners_up", []))
    return word.upper(), f"said at {info['said_at']:.2f} s" + (f"; next best {alts}" if alts else "")


def _lines(opts: dict, words: list[Word], quiet, title: str) -> tuple[list[str], list]:
    if opts["lines"] == "off":
        return [], []
    if opts["lines"] != "auto":
        given, info = [], []
        for raw in opts["lines"].split(" | "):
            marked = [w.strip("*") for w in raw.split() if w.startswith("*") and w.rstrip(".,!?;:").endswith("*")]
            plain = raw.replace("*", "").strip()
            given.append(plain)
            info.append({"text": plain, "as": "given", **({"accent": marked[0]} if marked else {})})
        if "accent" not in info[0]:  # the big line's longest word takes the accent unless one is marked
            ws = [w.strip(".,!?;:\"'") for w in given[0].split()]
            info[0]["accent"] = max(ws, key=lambda w: (sum(c.isalnum() for c in w), -ws.index(w))) if ws else ""
        return given, info
    found = cover_lines(words, quiet, title)
    return [s.text for s in found[:MAX_LINES]], found[:MAX_LINES]


def unit_of(text: str, info, kind: str) -> Unit:
    """A cover line as a unit: its words, its strongest word's place, and its figure."""
    words = tuple(text.split())
    key_pos, figure, label = None, "", ""
    if isinstance(info, Said):
        want = normalize(info.key_text)
        key_pos = next((n for n, w in enumerate(words) if normalize(w) == want), None)
        figure, label = (info.figure, info.label) if kind == "hero" else ("", "")
    elif isinstance(info, dict) and info.get("accent"):
        want = normalize(info["accent"])
        key_pos = next((n for n, w in enumerate(words) if normalize(w) == want), None)
    return Unit(kind, words, key_pos, figure, label)


def layout(look: Look, title: str, lines_text: list[str], line_info: list, issue: str, store: Store, walls: Walls,
           clip, tl, fr, sample: list[int], opts: dict, warnings: list[str], push: Optional[Push]) -> list[Part]:
    """Every part of the cover, placed and coloured."""
    W, H = fr.out_w, fr.out_h
    zone = SafeZone(W, H).rect
    tops = store.crowns()
    face_box = _face_union(clip, tl, fr, sample, push)

    # 1. the masthead, as words-behind sets its word; and room kept above it for the issue line and the
    #    rule, when keeping it costs the masthead little
    issue_px = max(8, int(round(ISSUE_CAP * H / (textmod.cap_height(look.issue, 400) / 400.0))))
    issue_cap = textmod.cap_height(look.issue, issue_px)
    rule_h = max(2.0, RULE * H)
    sky = (issue_cap + ISSUE_GAP * H + rule_h + RULE_GAP * H) if issue else (rule_h + RULE_GAP * H)
    shown = (title.replace("'", "’"),)
    full = wb.place([shown], look.masthead, store, (W, H), None)
    pl = full
    if full.y - zone.y0 < sky:
        with zone_for(W, H, Rect(zone.x0, zone.y0 + sky, zone.x1, zone.y1)):
            try:
                lower = wb.place([shown], look.masthead, store, (W, H), None)
            except ValueError:
                lower = None
        if lower is not None and lower.block.px >= SKY_KEEPS * full.block.px and not lower.warn:
            pl = lower
    if pl.warn:
        warnings.append(pl.warn)
    b = pl.block
    mast = Part("masthead", pl.x, pl.y, b.w, b.h, subs=(Sub(b, 0.0, 0.0, "main"),), align="centre", where=pl.where,
                hidden=(pl.hidden_median, pl.hidden_p90, pl.letter_p90))
    parts = [mast]
    taken = [mast.rect]

    # 2. the issue line and the thin rule, just above the masthead when there is room
    if mast.y - zone.y0 >= sky - 0.5:
        rule_y = mast.y - RULE_GAP * H - rule_h
        rule = Part("rule", mast.x, rule_y, mast.w, rule_h, where="above the masthead")
        parts.append(rule)
        taken.append(rule.rect)
        if issue:
            shown_issue = issue.upper()
            px = issue_px
            while spaced_width(shown_issue, look.issue, px) > mast.w and px > 8:
                px -= 1
            iw = spaced_width(shown_issue, look.issue, px)
            iy = rule_y - ISSUE_GAP * H - textmod.cap_height(look.issue, px)
            ip = Part("issue", mast.rect.cx - iw / 2, iy, iw, textmod.cap_height(look.issue, px),
                      where="above the masthead", said={"text": shown_issue, "face": look.issue, "px": px})
            parts.append(ip)
            taken.append(ip.rect)
    else:
        warnings.append("no room above the masthead for the issue line and the rule: the head sits too high")

    # 3. the cover lines: the big one first and as large as a place allows, then the small ones,
    #    down the sides beside the head where they can, above the masthead where they cannot
    top_used = min(r.y0 for r in taken)
    below = mast.rect.y1 + GAP * H
    sides = Rect(zone.x0, below, zone.x1, zone.y1)
    placed = 0
    big_room = min(HERO_CAP[1], 0.42 * b.cap / H)
    for n, text in enumerate(lines_text):
        hero = n == 0
        above = Rect(zone.x0, zone.y0, zone.x1, top_used - GAP * H)
        top_slot = [("above", above)] if above.h > 0 else []
        side_slots = [("left", sides), ("right", sides)]
        steps = [(min(HERO_CAP[0], big_room), big_room), LINE_CAP] if hero else [LINE_CAP]
        u = unit_of(text, line_info[n], "hero" if hero else "line")
        if hero:  # the big line takes whichever place lets it be largest
            got = place_unit(u, look, steps, zone, tops, W, H, taken, face_box, top_slot + side_slots)
        else:  # a small line goes down the sides, round the person, and above the masthead only when they are full
            got = (place_unit(u, look, steps, zone, tops, W, H, taken, face_box, side_slots)
                   or (place_unit(u, look, steps, zone, tops, W, H, taken, face_box, top_slot) if top_slot else None))
        if got is None:
            continue
        subs, x, y, w, h, align, slot, hid = got
        info = line_info[n]
        said = info.as_plan() if isinstance(info, Said) else dict(info)
        p = Part("hero" if hero else "line", x, y, w, h, subs=subs, align=align, order=placed,
                 where={"above": "above the masthead", "left": "down the left", "right": "down the right"}[slot],
                 hidden=hid, said=said)
        if hero and not u.figure and u.key_pos is not None:
            p.accent_word = u.words[u.key_pos]
        parts.append(p)
        taken.append(p.rect)
        placed += 1
        if slot == "above":
            top_used = min(top_used, p.rect.y0)
    if lines_text and placed < len(lines_text):
        warnings.append(f"room for {placed} of the {len(lines_text)} cover lines: the person fills the rest of the "
                        "frame" if placed else "no room for a cover line round the person: the masthead stands alone")

    # 4. colours: each part against the wall right behind it
    for p in parts:
        if p.kind == "rule":
            continue
        wall = walls.behind(p.rect)
        if p.kind == "masthead" and opts["color"] != "auto":
            p.ink = textmod.hex_rgb(opts["color"])
        else:
            _, p.ink, _ = choose_ink(wall, {"light": look.light, "dark": look.dark},
                                     covered=p.hidden[0] if p.kind == "masthead" else 0.0)
        p.shade = wb.shade_needed(wall, p.ink, "solid")
        p.shade_rgb = (0, 0, 0) if _lum(p.ink) > wall["median"] else (255, 255, 255)
        p.contrast = wb.contrast_with(wall, p.ink, "solid", p.shade)
        if p.kind == "masthead" and opts["color"] != "auto" and p.contrast < FAINT:
            warnings.append(f"the title's colour stands out only {p.contrast:.1f}:1 from the wall behind it; leave "
                            "color at auto for light or dark letters")
        if p.kind == "hero" and (p.accent_word or any(s.role == "accent" for s in p.subs)):
            acc = look.accent_on_dark if _lum(p.ink) > 0.5 else look.accent_on_light
            if wb.contrast_with(wall, acc, "solid", p.shade) >= 3.0:  # an accent that would not read stays in ink
                p.accent = acc
    rule = next((p for p in parts if p.kind == "rule"), None)
    if rule is not None:  # the rule is printed in the issue line's ink, or the masthead's
        rule.ink = next((p.ink for p in parts if p.kind == "issue"), mast.ink)
    return parts


def face_at(clip, tl, fr, push: Optional[Push], j: int) -> Optional[Rect]:
    """The face's own box (brows to chin, cheek to cheek) at output frame j, as the cover shows it."""
    info = clip.info
    k = tl.source_frame(j, info.rate, info.frames)
    b = clip.face.box(k)
    if b is None:
        return None
    r = fr.to_output(b, k)
    if push is None:
        return r
    s = push.scale(j / info.rate)
    return Rect(s * r.x0 + push.ax * (1 - s), s * r.y0 + push.ay * (1 - s),
                s * r.x1 + push.ax * (1 - s), s * r.y1 + push.ay * (1 - s))


def _face_union(clip, tl, fr, sample: list[int], push: Optional[Push] = None) -> Optional[Rect]:
    """The face's own box over the cover's frames, as shown."""
    return Rect.around(r for j in sample if (r := face_at(clip, tl, fr, push, j)) is not None)


def _geometry(b: wb.Block) -> dict:
    """A block exactly as it was set, so a plan renders the same pixels; it is used again only while
    its lines, face and size are the ones it was set for (retyped text is set afresh)."""
    return {"for": [list(b.lines), b.face, b.px], "cap": round(b.cap, 4), "gap": round(b.gap, 4), "w": round(b.w, 4),
            "h": round(b.h, 4), "tops": [round(v, 4) for v in b.line_tops], "xs": [round(v, 4) for v in b.line_x]}


def _block_from(bq: dict, align: str) -> wb.Block:
    lines, face, px = tuple(str(x) for x in bq["lines"]), str(bq["face"]), int(bq["px"])
    g = bq.get("geometry") or {}
    if g.get("for") == [list(lines), face, px] and len(g.get("tops", [])) == len(lines) == len(g.get("xs", [])):
        return wb.Block(lines, face, px, float(g["cap"]), float(g["gap"]), float(g["w"]), float(g["h"]),
                        tuple(float(v) for v in g["tops"]), tuple(float(v) for v in g["xs"]))
    return text_block(lines, face, px, align, float(bq.get("lead", 0.2)))


def _part_plan(p: Part) -> dict:
    out = {"kind": p.kind, "x": round(p.x, 3), "y": round(p.y, 3), "w": round(p.w, 3), "h": round(p.h, 3),
           "align": p.align, "ink": "#%02x%02x%02x" % tuple(int(v) for v in p.ink), "where": p.where, "order": p.order}
    if p.subs:
        out["blocks"] = [{"lines": list(sb.block.lines), "face": sb.block.face, "px": sb.block.px,
                          "lead": round(sb.block.gap / max(1e-6, sb.block.cap), 3), "dx": round(sb.dx, 3),
                          "dy": round(sb.dy, 3), "role": sb.role, "geometry": _geometry(sb.block)} for sb in p.subs]
    if p.kind == "issue" and p.said:
        out.update({"text": p.said["text"], "face": p.said["face"], "px": p.said["px"]})
    if p.shade:
        out.update({"shade": round(p.shade, 6), "shade_ink": "#%02x%02x%02x" % tuple(int(v) for v in p.shade_rgb)})
    if p.accent is not None:
        out.update({"accent": "#%02x%02x%02x" % tuple(int(v) for v in p.accent), "accent_word": p.accent_word})
    if p.kind in ("hero", "line") and p.said:
        out["said"] = p.said
    out["measured"] = {"hidden_median": round(p.hidden[0], 3), "hidden_p90": round(p.hidden[1], 3),
                       "letter_p90": round(p.hidden[2], 3), "contrast": round(p.contrast, 2)}
    return out


def _over_face(clip, tl, fr, store: Store, parts: list[Part], landed: list[int], push: Optional[Push]) -> int:
    """Frames where any letter would show inside the face's own box: only where the cut-out missed
    part of the face. Measured on the sampled frames, counted over every landed frame."""
    import cv2

    face = clip.face
    if not face.any() or not store.by_frame:
        return 0
    W, H = fr.out_w, fr.out_h
    canvas = np.zeros((store.h, store.w), np.float32)
    for p in parts:
        for sb in p.subs:
            ink, _ = wb.block_ink(sb.block)
            sc = 1.0 / Q
            small = cv2.resize(ink, (max(1, int(round(ink.shape[1] * sc))), max(1, int(round(ink.shape[0] * sc)))),
                               interpolation=cv2.INTER_AREA)
            ox, oy = int(round((p.x + sb.dx - wb.MARGIN) * sc)), int(round((p.y + sb.dy - wb.MARGIN) * sc))
            ih, iw = small.shape
            cy0, cx0, cy1, cx1 = max(0, oy), max(0, ox), min(store.h, oy + ih), min(store.w, ox + iw)
            if cy1 > cy0 and cx1 > cx0:
                canvas[cy0:cy1, cx0:cx1] = np.maximum(canvas[cy0:cy1, cx0:cx1],
                                                      small[cy0 - oy:cy1 - oy, cx0 - ox:cx1 - ox])
    landed_set = set(landed)
    checked = [j for j in sorted(store.by_frame) if j in landed_set]
    count = 0
    fps, n_src = clip.info.rate, clip.info.frames
    for j in checked:
        box = face_at(clip, tl, fr, push, j)
        if box is None:
            continue
        r = box.scale(1.0 / Q, 1.0 / Q)
        m = store.by_frame[j]
        x0, y0, x1, y1 = int(max(0, r.x0)), int(max(0, r.y0)), int(min(store.w, r.x1)), int(min(store.h, r.y1))
        if x1 <= x0 or y1 <= y0:
            continue
        shows = float((canvas[y0:y1, x0:x1] * (1.0 - m[y0:y1, x0:x1])).sum()) / ((x1 - x0) * (y1 - y0))
        count += shows > 0.004
    return int(round(count * len(landed) / len(checked))) if checked else 0


def sheet_times(timing: Timing, t_end: float, fps: float, n_out: int) -> list[float]:
    """Six moments: the masthead rising, the lines coming in, three while it holds, and leaving."""
    built = timing.land + 0.05 + STAGGER * 3 + PART_IN
    hold = max(0.3, t_end - built)
    leave = t_end - max(0.25, 2.0 / fps) if timing.hold_end else t_end - 0.5 * OUT  # a frame the video still has
    ts = [timing.t0 + 0.28, timing.land + 0.18, built + 0.15 * hold, built + 0.5 * hold, built + 0.85 * hold, leave]
    return [round(min((n_out - 1) / fps, max(0.0, t)), 3) for t in ts]
