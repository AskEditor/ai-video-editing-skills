"""The moment that matters: the word or the line an effect sets by itself, chosen the same way by all.

What no picker ever sets, wherever it looks:

- **A word the speaker takes back** straight after saying it (`restated`): "the lively town, or
  the lively city" takes "town" back for "city", and a creator would say "I said city".
- **The words of a first attempt** said again (`find_retakes`): the take that clean-cut cuts.
- **An aside** (`asides.asides`): a sentence about the recording rather than to the viewer ("Give
  me a second.", "Testing, testing.", "I think this audio works better."), which clean-cut cuts too.
- Fillers ("um", "uh"), and the words the transcriber was unsure of (`Clip.words` leaves those
  out already, so nothing here reads the words' cache).

`taken_back(words, quiet)` gathers the first three, each with why.

How strong a word or a line is, for every picker:

- `word_strength`: how well one word stands alone, large, for its line: a noun at the end of its
  phrase, a name, a number; never grammar, a filler or a vague word (`NOT_KEY`). The measure
  words-behind was built on, moved here.
- `line_strength`: how well a short stretch of words stands as a line of its own: a whole clause,
  short, its words strong, not cut out of the middle of a phrase. The measure headline was built
  on, moved here, with its words weighed by `word_strength` rather than by their letters.
- In context (`TakenBack`): a word taken back counts for nothing and no line holds it, and the
  word said instead of it counts as what its line says that is new, as the end of a phrase does.

The pickers on top:

- `pick_word`: words-behind's word (the opening line's, unless another is asked for);
- `pick_line`: headline's line (the strongest in the clip, a scrap of a sentence begun before the
  clip only when nothing else stands);
- `strong_starts`: where prompt-box may send a prompt (the start of a spoken line, weighed by the
  line that starts there).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from .asides import asides
from .words import FILLERS, HESITATIONS, Word, pause_after

# ---- the words that are never the point of a line --------------------------------------------
#
# Grammar, filler and words too vague to be the point of a line. Written for this pack. (Captions,
# headline and clean-cut read these through `fx/effects/captions.py` under their old names.)
NOT_KEY = set("""
a an the this that these those my your his her its our their some any no every each all both either neither
few many much more most less least other another such what which whose whatever whichever
i me you he him she it we us they them myself yourself himself herself itself ourselves yourselves themselves
mine yours hers ours theirs who whom someone somebody something anyone anybody anything everyone everybody
everything nobody nothing one ones
i'm i've i'll i'd you're you've you'll you'd he's she's it's we're we've we'll we'd they're they've they'll
they'd that's there's here's what's who's let's isn't aren't wasn't weren't don't doesn't didn't can't couldn't
won't wouldn't shouldn't haven't hasn't hadn't ain't gonna wanna gotta kinda sorta dunno
am is are was were be been being do does did doing done have has had having can could will would shall should
may might must get gets got getting go goes going gone went
about above across after against along among around as at before behind below beneath beside between beyond by
down during except for from in inside into like near of off on onto out outside over past since through
throughout till to toward towards under until up upon with within without via per
and but or nor so yet because although though while whereas if unless than then whether once when whenever
where wherever why how
just really very quite pretty actually basically literally totally simply still even also too only already
almost always never ever often sometimes maybe perhaps probably here there now today again back away well else
instead rather kind sort lot lots bit thing things stuff way ways okay ok yeah yes oh hey hi hello right sure
alright guys
make makes made making take takes took taking put puts say says said saying tell told see saw seen know knew
known think thought want wants wanted need needs let lets come comes came give gives gave look looks mean means
meant use used try tried keep kept feel felt seem seems start started
""".split()) | FILLERS

NUMBER_WORDS = set("""
zero two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen
eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion
first second third half double triple percent
""".split())

# Words that lead into the next word: a phrase that ends on one of these ("how to make a") leaves
# the reader hanging, so the cut avoids it.
LEADS_ON = set("""
a an the my your his her its our their this these those every each some any no
to of in on at for with from by into onto about over under after before between through without
and but or nor so because if than as while when where whether although though that which who
i we they he she i'm i've i'll i'd we're we've we'll we'd they're they've they'll they'd you're you've you'll
you'd he's she's it's there's here's let's
am is are was were be been being have has had do does did will would can could should may might must
gonna wanna gotta very really just quite such
""".split())

# Words that open a new clause: a break just before one is a natural place to breathe, and the
# word before it ("draws people in | and all the way") does not lead on, whatever it is.
OPENS = set("""
and but or so because then which who what when where while although though if unless
i we you they he she it's that's there's here's let's i'm i've we're we've
""".split())

SPECIFIC = 5  # a key word must score at least this (letters, plus extra for numbers and names)
EDGE_PUNCT = "\"'“”‘’.,;:!?()[]{}…-–—"


def shown(word: Word) -> str:
    """The word as a caption shows it: no commas or full stops, but questions and exclamations kept."""
    t = word.clean.strip()
    t = t.rstrip(",.;:…").lstrip("\"“‘(")
    return t.rstrip("\"”’)") or word.text.strip()


def bare(word: Word) -> str:
    """The word with nothing at either edge: how a key word is shown."""
    return word.clean.strip().strip(EDGE_PUNCT) or shown(word)


def key_score(words: list[Word], i: int) -> int:
    """How specific word `i` is: its letters, plus extra for a number, an acronym or a name; 0 for
    grammar and filler."""
    w = words[i]
    k = w.key
    if not k or k in NOT_KEY:
        return 0
    raw = bare(w)
    score = sum(c.isalpha() for c in k)
    if any(c.isdigit() for c in k) or k in NUMBER_WORDS:
        score += 4
    if len(raw) >= 2 and raw.isupper():
        score += 6  # an acronym is nearly always the thing being named (UGC, AI)
    elif raw[:1].isupper() and i > 0 and not words[i - 1].ends_sentence:
        score += 2  # a name in the middle of a sentence
    if w.text.rstrip().endswith("!"):
        score += 1
    return score


# ---- first attempts said again ---------------------------------------------------------------
#
# Moved here from clean-cut, which cuts what this finds, so a picker passes over what a cut takes out.

# Words people repeat on purpose: a double one is kept.
ON_PURPOSE = set("""
very really so no yes yeah ha haha bye go more that had many much again over on far too hey oh wow well blah
knock tick tock boom bang run now quick fast please come hello hi okay ok testing check one two
""".split())
RETAKE_WINDOW = 8.0  # a restart counts when it comes within this many seconds of the first attempt
LONGEST_TAKE = 16  # the most words a first attempt can have


def _broken_off(words: list[Word], a: int, b: int, quiet) -> bool:
    """Does the speech break between words[a] and words[b] (b > a)? A cut-off word, a comma or a
    full stop, a filler between them, or a real pause in the sound."""
    w = words[a]
    if w.text.rstrip().endswith(("-", "—", "–", "…", "...", ",", ".", "?", "!", ";", ":")):
        return True
    if any(x.key in HESITATIONS for x in words[a + 1:b]):
        return True
    return pause_after(words, b - 1, quiet) >= 0.12 or (b - 1 != a and pause_after(words, a, quiet) >= 0.12)


def find_retakes(words: list[Word], quiet) -> list[tuple[int, int, int, str]]:
    """(first, last, restart, reason) for each first attempt to drop: words[first..last] (inclusive)
    are said again from words[restart].

    A phrase begun and then said again within `RETAKE_WINDOW` seconds keeps only its last take.
    The first attempt must be mostly said again, must break off (a cut-off word, a comma, an "um",
    or a real pause), and must not run across a full stop unless every word of it is said again.
    A list ("most ads, most people") repeats one word only and is never taken for a retake. A word
    said twice in a stammer ("the the") is one too, unless people double it on purpose ("very
    very", "no no")."""
    idx = [i for i, w in enumerate(words) if w.key not in HESITATIONS]
    keys = [words[i].key for i in idx]
    found: list[tuple[int, int, int, str]] = []
    floor = 0  # a first attempt never starts before the last restart found
    for j in range(1, len(idx)):
        best = None
        for i in range(j - 1, floor - 1, -1):
            if words[idx[j]].start - words[idx[i]].start > RETAKE_WINDOW or j - i > LONGEST_TAKE:
                break
            span = j - i
            k = 0
            while k < span and j + k < len(idx) and keys[i + k] == keys[j + k]:
                k += 1
            if k == 0:
                continue
            first, restart = words[idx[i]], words[idx[j]]
            last = words[idx[j - 1]]
            inside = [words[idx[x]] for x in range(i, j - 1)]
            whole = k == span and span >= 3  # every word of the first attempt said again, in order
            if any(w.ends_sentence for w in inside) and not whole:
                continue  # a partial match across a full stop is two sentences, not one phrase
            broke = _broken_off(words, idx[j - 1], idx[j], quiet)
            if span == 1:
                # A word said twice: a stammer, unless the word is one people double on purpose or
                # the first ends a clause ("Testing, testing.").
                if (k >= 1 and first.key not in ON_PURPOSE and len(first.key) <= 12
                        and not first.text.rstrip().endswith((",", ".", "?", "!", ";", ":"))
                        and restart.start - first.end < 1.0):
                    cand = (k, i, "stammer")
                else:
                    continue
            else:
                if k < 2:
                    continue
                divergent = span - k
                if divergent > max(2, span // 3):
                    continue  # most of the first attempt is not said again: not a retake
                matched = [words[idx[x]] for x in range(j, j + k)]
                specific = any(w.key not in NOT_KEY for w in matched)
                if not (k >= 4 or (broke and (k >= 3 or specific))):
                    continue
                # A whole phrase said again is a retake; one broken off before its end and begun
                # again (the restart carries on past where it stopped) is a false start.
                reason = "retake" if divergent == 0 and last.ends_clause else "false start"
                cand = (k, i, reason)
            if best is None or cand[0] > best[0]:
                best = cand
        # A word cut off part-way and started again: "pro- product".
        if best is None:
            prev, cur = words[idx[j - 1]], words[idx[j]]
            if (prev.text.rstrip().endswith(("-", "—", "–")) and len(prev.key) >= 2
                    and cur.key.startswith(prev.key) and cur.key != prev.key):
                best = (1, j - 1, "false start")
        if best is not None:
            _, i, reason = best
            found.append((idx[i], idx[j - 1], idx[j], reason))
            floor = j
    return found


# ---- words taken back straight after they are said ---------------------------------------------
#
# A speaker who corrects one word says it again in a few set ways, each with a break before the
# correction: "the lively town, or the lively city", "on Tuesday, I mean Wednesday", "three, sorry,
# four", "red — no, blue". What reads as the same thing but is not one: alternatives ("the red one,
# or the blue one", "a train or a bus", "now or never"), a phrase said again to stress it ("fast, I
# mean really fast"), an answer to a question ("Red? No, blue."), an apology ("I'm late, sorry,
# the traffic"). The rules below keep to the shapes that are corrections; a hit is a strong penalty
# for a picker, so a correction missed costs a little more than one wrongly found, which costs
# little (the picker takes another word).

BREAK = 0.2  # a gap in the voice at least this long, with no comma written, still sets off a correction
LEAD_IN = 4  # the most words of the lead-in a correction says again ("the lively" town, or "the lively" city)
# The words that set a correction off, longest first. "or" alone needs the lead-in said again
# (with a word in it that is not grammar); the others say "this replaces that" by themselves.
_MARKERS = (("or", "rather"), ("or", "i", "mean"), ("i", "mean"), ("i", "meant"), ("sorry",), ("no",), ("or",))
# Before an "or", these make it a choice between two things, not a correction.
_CHOICE = {"either", "whether", "between", "choose", "pick", "prefer", "rather", "than"}
_DASHES = ("-", "—", "–")


def _content(w: Word) -> bool:
    return bool(w.key) and w.key not in NOT_KEY


def _set_off(words: list[Word], a: int, b: int, quiet) -> bool:
    """Is there a break in the voice between words[a] and words[b] (b > a)? A comma, a dash or
    another stop written after words[a], a filler between them, or a pause."""
    if words[a].text.rstrip().endswith((",", ";", ":", "…", "...") + _DASHES):
        return True
    if any(x.key in HESITATIONS for x in words[a + 1:b]):
        return True
    if quiet is not None:
        return pause_after(words, b - 1, quiet) >= BREAK or pause_after(words, a, quiet) >= BREAK
    return words[b].start - words[a].end >= BREAK


def _same_word(a: str, b: str) -> bool:
    """One word said again rather than replaced: the same, or the same with an s."""
    return a == b or (min(len(a), len(b)) >= 3 and a.rstrip("s") == b.rstrip("s"))


def restated(words: list[Word], quiet=None) -> dict[int, int]:
    """The words a speaker replaces straight after saying them: {index of the word taken back:
    index of the word said instead}.

    The word taken back (X) is the word just before the correction, set off from it by a break (a
    comma, a dash, a filler or a pause), never the end of a sentence; the correction is one of:

    - "X, or" with the lead-in before X said again and a new word after it: "the lively town, or
      the lively city" (town -> city). The lead-in must hold a word that is not grammar: "a train
      or a bus" and "the red one, or the blue one" are a choice between two things, as is any
      "or" after "either", "whether" or "between", or in a question.
    - "X, or rather Y", "X, I mean Y", "X, sorry, Y" and "X — no, Y" (after "sorry" or "no", a
      second break): the lead-in may be said again ("on Tuesday, I mean on Wednesday") or not.
      After these, a correction may say its last word again: in "ten dollars, sorry, twelve
      dollars" the word taken back is "ten", and "dollars" is said again with "twelve".

    Either way the new word (Y) must be one that is not grammar, said straight after the marker and
    the lead-in, and not X again: "fast, I mean really fast" stresses "fast" and takes nothing back.
    `quiet` (the clip's quiet stretches) tells a pause from a word ending; without it the gaps
    between the word times are read."""
    return {x: y for x, y, _ in _corrections(words, quiet)}


def _corrections(words: list[Word], quiet) -> list[tuple[int, int, list[int]]]:
    """Each correction `restated` finds: (the word taken back, the word said instead, the lead-in
    before the word taken back that the correction says again: "the lively" of "the lively town,
    or the lively city")."""
    idx = [i for i, w in enumerate(words) if w.key and w.key not in HESITATIONS]
    keys = [words[i].key for i in idx]
    out: list[tuple[int, int, list[int]]] = []
    seen: set[int] = set()
    for p in range(1, len(idx)):
        x = idx[p - 1]
        X = words[x]
        if not _content(X) or X.ends_sentence or x in seen:
            continue
        marker = next((m for m in _MARKERS if tuple(keys[p:p + len(m)]) == m), None)
        if marker is None:
            continue
        m0, m1 = idx[p], idx[p + len(marker) - 1]
        if not _set_off(words, x, m0, quiet):
            continue
        after = p + len(marker)  # the first word after the marker, in idx
        if after >= len(idx):
            continue
        if marker in (("sorry",), ("no",)) and not _set_off(words, m1, idx[after], quiet):
            continue  # "sorry" and "no" set off on both sides: "three, sorry, four"
        # The lead-in said again: the longest run of the words before X that the correction opens with.
        k = next((k for k in range(min(LEAD_IN, p - 1), 0, -1)
                  if keys[p - 1 - k:p - 1] == keys[after:after + k]), 0)
        if after + k >= len(idx):
            continue
        y = idx[after + k]
        Y = words[y]
        if not _content(Y) or _same_word(Y.key, X.key):
            continue
        if any(_same_word(keys[q], X.key) for q in range(after + k + 1, min(len(idx), after + k + 4))):
            tail = _tail_form(words, idx, keys, p, after, marker)
            if tail is not None and tail[0] not in seen:
                out.append(tail)
                seen.add(tail[0])
            continue  # X said again just after: stressed or explained, not replaced
        if marker == ("or",):
            lead = keys[p - 1 - k:p - 1]
            if k == 0 or not any(key not in NOT_KEY for key in lead):
                continue  # "a train or a bus", "now or never": two things, not a correction
            before = keys[max(0, p - 1 - k - 6):p - 1 - k]
            if any(key in _CHOICE for key in before):
                continue
            # A question offering both ("Do you want the red car, or the red bus?").
            end = next((q for q in range(after + k, min(len(idx), after + k + 10)) if words[idx[q]].ends_sentence), None)
            if end is not None and words[idx[end]].text.rstrip().endswith("?"):
                continue
        out.append((x, y, idx[p - 1 - k:p - 1]))
        seen.add(x)
    return out


def _tail_form(words: list[Word], idx: list[int], keys: list[str], p: int, after: int,
               marker: tuple) -> Optional[tuple[int, int, list[int]]]:
    """A correction that says its last word again: "ten dollars, sorry, twelve dollars", "2,000
    followers, no, 3,000 followers", "on the train, I mean on the bus" is not this (the word replaced
    is the last). Here the word before X is replaced (ten -> twelve) and X ("dollars") is said again
    straight after the new word, with the lead-in before them said again too, if any. Only after a
    marker that says "this replaces that" by itself: a bare "or" offers a choice ("the red car, or the
    blue car"). (taken back, said instead, the lead-in said again with X), or None."""
    if marker == ("or",) or p < 2:
        return None
    x2, X = idx[p - 2], words[idx[p - 1]]
    k = next((k for k in range(min(LEAD_IN, p - 2), 0, -1) if keys[p - 2 - k:p - 2] == keys[after:after + k]), 0)
    if after + k + 1 >= len(idx) or not _same_word(keys[after + k + 1], X.key):
        return None
    y = idx[after + k]
    if not _content(words[x2]) or not _content(words[y]) or _same_word(words[y].key, words[x2].key):
        return None
    return x2, y, idx[p - 2 - k:p - 2] + [idx[p - 1]]


# ---- everything a picker passes over ----------------------------------------------------------

@dataclass
class TakenBack:
    """The words a picker never sets, each with why, and the words said instead of one."""
    why: dict = field(default_factory=dict)  # Word -> why it is passed over
    instead: dict = field(default_factory=dict)  # Word taken back -> the Word said instead

    def __contains__(self, w: Word) -> bool:
        return w in self.why

    @property
    def said_instead(self) -> set:
        return set(self.instead.values())

    def holds(self, words: list[Word], first: int, last: int) -> bool:
        """Does `words[first..last]` hold a word passed over: a figure said and taken back ("ten,
        sorry, twelve"), a name in a first attempt, a phrase in an aside. For an effect that keeps its
        own reason for choosing (a figure for a chart, a file's name) and only needs the filter."""
        return any(w in self.why for w in words[max(0, first):last + 1])


def taken_back(words: list[Word], quiet) -> TakenBack:
    """What the speaker took back in `words` (give a clip's whole `words`, fillers and all, so a
    first attempt broken off by an "um" is found): the words replaced straight after they were said
    (`restated`) with the lead-in the correction says again ("the lively" of "the lively town, or
    the lively city"), the words of every first attempt said again (`find_retakes`), and every
    word of an aside, said about the recording rather than to the viewer (`asides`). Words are
    kept by themselves, not by their place, so an effect that leaves fillers out can ask of its own
    list."""
    out = TakenBack()
    for x, y, lead in _corrections(words, quiet):
        out.why[words[x]] = f"taken back for \"{bare(words[y])}\" ({words[y].start:.2f} s)"
        out.instead[words[x]] = words[y]
        for i in lead:  # "the lively" town, or the lively city: said again with the word that replaces it
            out.why.setdefault(words[i], f"said again with \"{bare(words[y])}\" ({words[y].start:.2f} s)")
    for first, last, restart, reason in find_retakes(words, quiet):
        for i in range(first, last + 1):
            out.why.setdefault(words[i], f"a {reason}, said again from {words[restart].start:.2f} s")
    for a in asides(words, quiet):
        for i in range(a.first, a.last + 1):
            out.why.setdefault(words[i], f"an aside ({a.kind}), said about the recording, not to the viewer")
    return out


# ---- how strong a word is ---------------------------------------------------------------------
#
# Specific is not enough: "perspective" is longer than "hooks", and "professionally" longer than
# "videos". What lands is the thing named, which in English sits at the end of its phrase.

_GENERIC = set("""
perspective perspectives issue issues way ways part parts kind kinds type types sort sorts thing things aspect
aspects point points side sides term terms case cases matter fact facts example examples reason reasons amount
lot lots bit bits stuff rest sense regard respect basis manner instance couple while bunch piece pieces number
level levels process situation question questions problem
""".split())
_LY_NOUNS = set("family supply reply ally rally belly jelly italy july lily assembly anomaly butterfly monopoly "
                "holly reality ugly early".split())
_DETERMINERS = set("a an the my your our their his her its this that these those every each some any no another "
                   "one".split())
_PREPOSITIONS = set("of in on at for with from by into onto about over under after before between through "
                    "without to".split())
# A word straight after one of these is usually doing, not naming.
_BEFORE_VERBS = set("i you we they he she to will would can could should must may might while when "
                    "i'll you'll we'll they'll don't didn't can't won't".split())
# Two words that say nothing apart from each other: "of course" names no course.
_IDIOMS = {("of", "course"), ("at", "least"), ("at", "all"), ("in", "fact"), ("you", "know"), ("by", "far"),
           ("for", "sure"), ("no", "doubt"), ("in", "general"), ("at", "first"), ("in", "total")}
PHRASE_END = 3.0  # what the end of a phrase adds: where a sentence puts what is new


# Public names for other effects: words too general to be the point, and words that come before a verb.
GENERIC = _GENERIC
BEFORE_VERBS = _BEFORE_VERBS


def _letters(k: str) -> int:
    return sum(c.isalpha() for c in k) or sum(c.isdigit() for c in k)


def ends_phrase(words: list[Word], i: int, line: list[int], quiet) -> bool:
    """Is word `i` the end of a phrase: a stop written after it, the end of its line, or a pause?"""
    return (words[i].text.rstrip().endswith((",", ".", "!", "?", ";", ":")) or i == line[-1]
            or pause_after(words, i, quiet) >= 0.25)


def _alone(words: list[Word], i: int) -> Optional[float]:
    """What word `i` says by itself, wherever it sits: how specific it is (`key_score`), a long word
    no better than one of eight letters, less for a vague word ("perspective", "issue") or an
    adverb in -ly; None for a word that never counts (grammar, a filler, half of an idiom such as
    "of course", or a word the transcriber gave less than 0.4)."""
    w = words[i]
    k = w.key
    if w.prob < 0.4 or k in HESITATIONS:
        return None
    base = key_score(words, i)  # letters, plus numbers, acronyms and names; 0 for grammar and filler
    if base <= 0:
        return None
    n = _letters(k)
    score = base - n + min(n, 8)  # a longer word is not a better one past eight letters
    if n > 10:
        score -= 0.5 * (n - 10)
    if k in _GENERIC:
        score -= 4
    if k.endswith("ly") and len(k) > 4 and k not in _LY_NOUNS:
        score -= 4
    prev = words[i - 1].key if i > 0 else ""
    if (prev, k) in _IDIOMS:
        return None
    return float(score)


def word_strength(words: list[Word], i: int, line: list[int], quiet) -> float:
    """How well word `i` stands alone, giant, for the line it is in (`line`: its indexes). 0 for a
    word that never should. What it says by itself (`_alone`), and where it sits: the thing named
    is at the end of its phrase, after "the" or "a", not straight after "I" or "to" (doing, not
    naming) nor before "of". The word by itself: `strength` weighs it with what was taken back."""
    score = _alone(words, i)
    if score is None:
        return 0.0
    nxt = words[i + 1].key if i + 1 < len(words) else ""
    prev = words[i - 1].key if i > 0 else ""
    if ends_phrase(words, i, line, quiet):
        score += PHRASE_END
    if nxt == "of":
        score -= 3  # "the perspective of hooks": the noun after "of" is the one named
    if prev in _BEFORE_VERBS:
        score -= 4
    elif prev in _PREPOSITIONS:
        score += 1
    for back in range(1, 4):  # "a video", "your production quality"
        if i - back < line[0]:
            break
        if words[i - back].key in _DETERMINERS:
            score += 2
            break
        if words[i - back].ends_clause:
            break
    return max(0.0, float(score))


def _a_length_of_time(words: list[Word], i: int) -> float:
    """What a number word loses when it is not a figure: "give me a second" names a length of time,
    not a place in a row, so "second" there earns nothing as a number (`key_score` gives 4)."""
    return 4.0 if words[i].key == "second" and i > 0 and words[i - 1].key in ("a", "one") else 0.0


def strength(words: list[Word], i: int, line: list[int], quiet, back: TakenBack) -> float:
    """`word_strength` with what was taken back: nothing for a word taken back, and the end of a
    phrase's weight for the word said instead of one ("the lively *city*"), which is the new thing
    its line says however it is placed."""
    w = words[i]
    if w in back or w.key in HESITATIONS:
        return 0.0
    s = word_strength(words, i, line, quiet)
    if s > 0 and w in back.said_instead and not ends_phrase(words, i, line, quiet):
        s += PHRASE_END
    return max(0.0, s - _a_length_of_time(words, i))


# Adverbs that say how the speaker stands to a line, not what it says: a line of these and grammar
# says nothing ("I eventually went to,"). An adverb of manner says something ("edit your videos
# professionally") and counts like any word.
_HOLLOW = set("""
eventually finally honestly obviously usually recently currently generally definitely certainly seriously
hopefully personally apparently clearly essentially exactly absolutely completely entirely fully nearly mostly
mainly especially particularly specifically ultimately initially originally typically naturally normally
possibly likely surely truly merely barely hardly lately frankly fortunately unfortunately luckily sadly
interestingly importantly additionally consequently previously
""".split())


def line_weight(words: list[Word], i: int, back: TakenBack) -> float:
    """What word `i` adds to a line, wherever it sits, since a line needs its verb as much as its
    noun ("*edit* your videos"): how specific it is (`key_score`: its letters, more for a number,
    an acronym or a name), less for a vague word ("perspective", "issue"); nothing for grammar, a
    filler, half of an idiom ("of course"), an adverb that says nothing (`_HOLLOW`), a word the
    transcriber gave less than 0.4, or a word taken back; and the end of a phrase's weight more for
    the word said instead of one ("the lively *city*")."""
    w = words[i]
    k = w.key
    if w in back or w.prob < 0.4 or k in HESITATIONS or k in _HOLLOW:
        return 0.0
    if i > 0 and (words[i - 1].key, k) in _IDIOMS:
        return 0.0
    base = key_score(words, i)
    if base <= 0:
        return 0.0
    s = base - (4 if k in _GENERIC else 0) - _a_length_of_time(words, i)
    return max(0.0, float(s)) + (PHRASE_END if w in back.said_instead else 0.0)


# ---- lines --------------------------------------------------------------------------------------

def lines(words: list[Word], quiet) -> list[list[int]]:
    """Word indexes split into lines: at the end of a sentence or a clause, where the voice stops,
    and at a filler (which belongs to no line)."""
    out, cur = [], []
    for i, w in enumerate(words):
        if w.key in HESITATIONS:
            if cur:
                out.append(cur)
                cur = []
            continue
        cur.append(i)
        if w.ends_clause or pause_after(words, i, quiet) >= 0.3:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def scrap_lines(words: list[Word], runs: list[list[int]]) -> set[int]:
    """The lines (by number) of a sentence the clip came in on halfway: a scrap, not a line of its
    own. A clip that opens on a word in lower case, within a quarter second of its start, opens
    in the middle of a sentence; its lines up to the first full stop are the scrap."""
    if not words or words[0].start > 0.25:
        return set()
    first = words[0].clean.strip("\"'“‘(")
    if not first or not first[0].islower() or words[0].key == "i":
        return set()
    out = set()
    for n, ln in enumerate(runs):
        out.add(n)
        if words[ln[-1]].ends_sentence:
            break
    return out if len(out) < len(runs) else set()


# Words a line never opens on: it would hang off the line before.
_JOINS = ("and", "but", "or", "so", "because", "then")
MAX_LINE = 6  # the most words a line is weighed over
# A line the speaker opens by hedging is an opinion said in passing, not the point ("I think this
# audio works better"): it loses HEDGE.
_HEDGES = (("i", "think"), ("i", "guess"), ("i", "suppose"), ("i", "feel"), ("maybe",), ("perhaps",), ("probably",))
HEDGE = 3.0


def line_strength(words: list[Word], run: list[int], a: int, b: int, quiet, back: TakenBack) -> Optional[float]:
    """How well words run[a:b] (a stretch of the line `run`) stands as a line of its own, or None
    when it never should: under two words; ending on a word that leads into the next ("the", "to")
    unless a comma closes it, or on a word broken off; opening on "and" or "so"; holding a word
    taken back; or holding a word the transcriber was unsure of.

    Its words count by what each says by itself (`line_weight`: grammar and fillers add nothing, a
    vague word or an adverb little), and one of them must be specific enough to be the point
    (`SPECIFIC`). A clause said whole counts too: a fragment of two long words ("marketing
    campaign") is a label, a whole clause is a line. A line is short, so every letter past eighteen
    costs a little, a line opened by a hedge ("I think…") costs `HEDGE`, and a
    stretch cut out of the middle of a phrase ("your | videos professionally") costs more, unless
    what it leaves behind is the "and" or "or" that joins it to the line before ("or | the lively
    city come to life")."""
    idx = run[a:b]
    n = len(idx)
    if n < 2:
        return None
    first, last = words[idx[0]], words[idx[-1]]
    if (last.key in LEADS_ON and not last.ends_clause) or last.text.rstrip().endswith(("-", "—", "…")):
        return None
    if first.key in _JOINS:
        return None
    if any(words[i] in back for i in idx):
        return None  # never a word the speaker took back
    probs = [words[i].prob for i in idx]
    if min(probs) < 0.5 or sum(probs) / n < 0.75:
        return None  # never words the transcriber was unsure of
    weights = [line_weight(words, i, back) for i in idx]
    content = sum(weights)
    if max(weights) < SPECIFIC:
        return None  # no word in it says enough to be the point of a line ("I eventually went to")
    score = 0.5 * content + 1.2 * content / n
    score += 5.0 if (a == 0 and b == len(run)) else 0.0
    score += 1.5 if a == 0 else 0.0
    score += 2.0 if last.ends_sentence else (1.0 if last.ends_clause else 0.0)
    score -= 4.0 if n < 3 else 0.0
    score -= 1.5 * max(0, n - 5)
    if any(tuple(words[i].key for i in idx[:len(h)]) == h for h in _HEDGES):
        score -= HEDGE
    letters = sum(len(words[i].clean) for i in idx) + n - 1
    score -= 0.25 * max(0, letters - 18)
    # cut out of the middle of a phrase: "your | videos professionally", "edit your videos | professionally"
    before = words[idx[0] - 1] if idx[0] > 0 else None
    after = words[idx[-1] + 1] if idx[-1] + 1 < len(words) else None
    if a > 0 and before is not None and not before.ends_clause:
        if before.key in LEADS_ON and before.key not in _JOINS:
            score -= 3.0
        elif first.key in LEADS_ON:
            score -= 2.0  # "improve | your production quality": the verb left behind
    if b < len(run) and after is not None and not last.ends_clause and after.key not in OPENS:
        score -= 1.5
    return score


SCRAP = 6.0  # what a scrap of a sentence begun before the clip loses, as a line


def pick_line(words: list[Word], quiet, reach: Optional[Callable[[int, int], float]] = None,
              back: Optional[TakenBack] = None, ranked: Optional[list] = None) -> Optional[tuple[int, int]]:
    """The line that makes the point: (first, last) word indexes of the strongest stretch of two to
    six words (`line_strength`), a scrap of a sentence the clip came in on halfway losing `SCRAP`.
    When `reach(first, last)` says how large a line could be set where it would sit (1 at full
    size), a line that would come out small loses to one that would not. `back`: what was taken
    back, from the clip's whole words (found from `words` when not given). `ranked`, when given a
    list, is filled with every line weighed, strongest first: (score, (first, last))."""
    back = taken_back(words, quiet) if back is None else back
    runs = lines(words, quiet)
    scraps = scrap_lines(words, runs)
    scored = []
    for n_run, run in enumerate(runs):
        for a in range(len(run)):
            for b in range(a + 2, min(len(run), a + MAX_LINE) + 1):
                s = line_strength(words, run, a, b, quiet, back)
                if s is not None:
                    scored.append((s - (SCRAP if n_run in scraps else 0.0), (run[a], run[b - 1])))
    if not scored:
        return None
    scored.sort(key=lambda s: -s[0])
    if ranked is not None:
        ranked.extend(scored)
    if reach is None:
        return scored[0][1]
    best, best_score = None, -math.inf
    for score, span in scored[:10]:
        r = max(1e-3, reach(*span))
        total = score + 6.0 * math.log(min(1.0, r))
        if total > best_score:
            best, best_score = span, total
    return best


# ---- the word for a line -----------------------------------------------------------------------

MIN_HERO = 8.0  # a line's best word must score this to be chosen without being asked for


@dataclass
class Candidate:
    index: int
    word: Word
    score: float
    line: list[int]
    scrap: bool = False


def best_in(words: list[Word], line: list[int], quiet, back: Optional[TakenBack] = None) -> Optional[Candidate]:
    """The strongest word of a line (a tie goes to the later word), or None when none stands."""
    back = TakenBack() if back is None else back
    best = None
    for i in line:
        s = strength(words, i, line, quiet, back)
        if s > 0 and (best is None or s >= best.score):
            best = Candidate(i, words[i], s, line)
    return best


def pick_word(words: list[Word], quiet, back: Optional[TakenBack] = None, start_after: float = 0.0,
              within: Optional[tuple[float, float]] = None) -> tuple[Optional[Candidate], list[Candidate]]:
    """The word for the line that lands, and up to three runners-up: the best word of the first
    whole line whose best word scores `MIN_HERO`, else the best that does anywhere, else none.
    `within` limits it to a stretch the person named (the best word there, however plain);
    `start_after` to the lines from a time on. `back`: what was taken back, from the clip's whole
    words (found from `words` when not given)."""
    back = taken_back(words, quiet) if back is None else back
    runs = lines(words, quiet)
    scrap = scrap_lines(words, runs)
    cands = []
    for n, ln in enumerate(runs):
        if within is not None:
            ln = [i for i in ln if within[0] - 0.05 <= words[i].start <= within[1] + 0.05]
            if not ln:
                continue
        elif words[ln[-1]].end < start_after:
            continue
        c = best_in(words, ln, quiet, back)
        if c is not None:
            c.scrap = n in scrap
            cands.append(c)
    ranked = sorted(cands, key=lambda c: -c.score)
    if within is not None:
        return (ranked[0] if ranked else None), ranked[1:4]
    for c in cands:  # the first whole line whose word is strong enough
        if not c.scrap and c.score >= MIN_HERO:
            return c, [r for r in ranked if r is not c][:3]
    good = [c for c in ranked if c.score >= MIN_HERO]
    if good:
        return good[0], [r for r in ranked if r is not good[0]][:3]
    return None, ranked[:3]


def key_word(words: list[Word], quiet, back: Optional[TakenBack] = None, least: float = SPECIFIC) -> Optional[int]:
    """The word of a line given whole (`words` is the line) to set in the accent: its strongest,
    a tie going to the later word, or None when none reaches `least`."""
    c = best_in(words, list(range(len(words))), quiet, back)
    return c.index if c is not None and c.score >= least else None


# ---- where a moment may land ----------------------------------------------------------------------

@dataclass
class Start:
    at: float  # when the word that starts the line begins
    index: int  # that word, in the words given
    strength: Optional[float]  # of the line it opens; None when no line stands there
    line: Optional[tuple[int, int]] = None  # that line: (first, last) word indexes
    sentence: bool = False  # it opens a sentence (the clip's first word, or the first after a full stop)


OPENS_WITHIN = 1.0  # seconds: a moment lands on a line that begins this soon after it, in the same sentence


def strong_starts(words: list[Word], quiet, back: Optional[TakenBack] = None, pause: float = 0.18) -> list[Start]:
    """Where a spoken line begins, each weighed by the line it opens: a word after a pause of
    `pause` seconds, after the end of a clause, or after a filler (the filler itself is never the
    moment). The line it opens is the strongest that begins on it or within `OPENS_WITHIN` seconds
    after it, in the same sentence, so the start of "Here's a clean, bold subtitle for everyday
    posts" opens "bold subtitle for everyday posts" and the "or" of "or the lively city come to
    life" opens "city come to life". A start the speaker took back (a word replaced, or the first
    word of an attempt said again) is left out."""
    back = taken_back(words, quiet) if back is None else back
    runs = lines(words, quiet)
    run_of = {i: (r, pos) for r in runs for pos, i in enumerate(r)}
    best_from: dict[int, tuple[Optional[float], Optional[tuple[int, int]]]] = {}

    def from_word(k: int) -> tuple[Optional[float], Optional[tuple[int, int]]]:
        """The strongest line that begins on word k."""
        if k not in best_from:
            best, line = None, None
            if k in run_of:
                run, pos = run_of[k]
                for b in range(pos + 2, min(len(run), pos + MAX_LINE) + 1):
                    s = line_strength(words, run, pos, b, quiet, back)
                    if s is not None and (best is None or s > best):
                        best, line = s, (run[pos], run[b - 1])
            best_from[k] = (best, line)
        return best_from[k]

    out = []
    for i, w in enumerate(words):
        if w.key in HESITATIONS or w in back or i not in run_of:
            continue
        if i > 0:
            prev = words[i - 1]
            if not (pause_after(words, i - 1, quiet) >= pause or prev.ends_clause or prev.key in HESITATIONS):
                continue
        best, line = None, None
        k = i
        while k < len(words) and words[k].start - w.start <= OPENS_WITHIN:
            s, ln = from_word(k)
            if s is not None and (best is None or s > best):
                best, line = s, ln
            if words[k].ends_sentence:
                break
            k += 1
        before = [x for x in words[:i] if x.key not in HESITATIONS]
        out.append(Start(w.start, i, best, line, sentence=not before or before[-1].ends_sentence))
    return out
