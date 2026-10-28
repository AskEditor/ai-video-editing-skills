"""The parts of a talk: where the speaker moves on to a new idea, found from the words and the sound
alone. No model runs beyond the transcriber, nothing is paid for and nothing leaves the computer.

    found = parts.find(clip.words, clip.quiet, duration)       # the parts, each with a title
    count, given = parts.parse_titles("Intro | 0:31 The train")  # what a person typed as titles=
    found = parts.find(clip.words, clip.quiet, duration, count=count, titles=given)
    lines = parts.chapter_lines(found, duration)               # "00:00 Intro", ... for YouTube
    parts.check_chapters(lines, duration)                      # [] when YouTube would take them

Between every two sentences, and at every real stop in the voice, three kinds of evidence are
weighed:

1. **The voice stops.** A sentence that ends, and how long the quiet after it lasts, read from the
   sound (`clip.quiet`, through `words.pause_after`; the transcriber's own gaps are nearly always
   zero). A long breath between two ideas is the commonest sign of a new one.
2. **A signpost opens the next sentence.** Ordinals and announcements ("first of all", "the second
   thing", "next", "let's talk about", "finally") count fully; "now", "then", "after that" and a
   short question count part way; "so" and "and" barely, since people open every other sentence
   with them.
3. **The words change.** The content words said in the stretch before and the stretch after
   (grammar, filler and vague words left out, plurals folded) are compared: the less they share,
   the likelier a new idea. The measure is a plain cosine between their word counts, each word
   weighted by how rare it is across the talk's sentences (TF-IDF). That is the TextTiling idea
   (Hearst, 1997) at the size of a short talk; on a minute of speech it is the noisiest of the
   three, so it weighs least and mostly decides between two pauses.

The boundaries are then chosen together, by dynamic programming over the candidates: every part
lasts at least `min_part` seconds, and a boundary is taken only when its evidence clears a bar
(`BAR`), so a talk about one thing stays one part. When a number of parts is asked for (as many as
the person gives titles, or a count), exactly that many are made, at the strongest boundaries that
keep every part long enough.

Each part gets a short title of its own key words: the word or pair of words said in it that is
most specific to it and least said elsewhere in the talk (so the part about the hotel is "Hotel",
not "Went"), never a word the speaker took back (`moments.taken_back`). A title the person gives
replaces it, and may carry the time its part begins.

Nothing here draws: the chapter cards, the post copy and anything later read the same parts.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Sequence

from . import moments
from .words import HESITATIONS, Word, normalize, pause_after

# ---- the numbers that decide -------------------------------------------------------------------

BAR = 1.2  # the evidence a boundary must carry to be taken when nobody says how many parts
SPLIT_PAUSE = 0.45  # a stop in the voice this long is a candidate boundary even mid-sentence
PAUSE_FULL = 1.4  # a pause this long counts fully; shorter ones count in proportion
WEIGHTS = {"pause": 1.0, "sentence": 0.6, "signpost": 1.0, "shift": 0.7, "question": 0.35}
SHORTEST = 4.0  # no part is ever shorter than this, whatever is asked
YOUTUBE_SHORTEST = 10.0  # YouTube Help, "Video Chapters", read 28 October 2026: chapters last 10 s or more


def auto_min_part(duration: float) -> float:
    """The shortest a part may be when nobody says: a tenth of the video, kept between 6 and 15
    seconds. A 30-second reel can then be cut where its ideas change, and a talk of a few minutes
    gets parts of a quarter of a minute or more."""
    return float(min(15.0, max(6.0, duration / 10.0)))


def most_parts(duration: float) -> int:
    """The most parts a video is split into when nobody says how many: it grows more slowly than the
    video (4 in half a minute, 6 in two minutes, 9 in five, 18 in half an hour), so a long talk gets
    chapters, not a list of every sentence."""
    return max(2, int(round(0.6 * max(1.0, duration) ** 0.45 + 1.0)))


# ---- the words that carry no subject -------------------------------------------------------------

# Grammar, filler and the vague verbs and nouns people say about anything: two stretches that share
# only these are not on the same subject, and no title is made of them. Written for this pack.
_PLAIN = set("""
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
good great nice cool real big little new old first second third next last
eventually finally slowly obviously anyway course
perspective aspect point case fact idea sense side level part piece matter terms example instance issue
actual whole entire certain different same own main mainly best better worse worst nicest lovely amazing
awesome super dot com people person
""".split())

_ORDINALS = ("one two three four five six seven eight nine ten 1 2 3 4 5 6 7 8 9 10 "
             "first second third fourth fifth sixth seventh eighth ninth tenth").split()


def _stem(key: str) -> str:
    """A word folded for comparing: "trains" and "train", "cities" and "city" count as one."""
    k = key.replace("'s", "").strip("'")
    if len(k) > 4 and k.endswith("ies"):
        return k[:-3] + "y"
    if len(k) > 4 and k.endswith("s") and not k.endswith(("ss", "us", "is")):
        return k[:-1]
    return k


def content(w: Word) -> Optional[str]:
    """The word's folded form when it carries subject matter, else None. A piece the transcriber
    broke off a longer word ("in -built", "askeditor .com") carries none on its own."""
    k = w.key
    if not k or k in _PLAIN or k in HESITATIONS or w.text.strip()[:1] in "-.":
        return None
    if len(k) < 3 and not any(c.isdigit() for c in k) and not w.clean.isupper():
        return None
    return _stem(k)


# ---- signposts ---------------------------------------------------------------------------------------

def _phrases(text: str) -> list[tuple[str, ...]]:
    return [tuple(p.split()) for p in text.split(",") if p.strip()]


# The words a new part is announced with, by how much they say so. Written for this pack.
_FULL = _phrases(
    "first of all, firstly, first, first up, first off, to start, to start with, to begin, to begin with, "
    "let's start, let's begin, let's kick off, second, secondly, third, thirdly, fourth, fifth, next, next up, "
    "moving on, let's move on, on to, onto, now let's, let's talk about, let's look at, let's get into, "
    "let's dive into, let's jump into, let's go over, let's go through, finally, lastly, last but not least, "
    "to finish, to wrap up, to sum up, to recap, in summary, in conclusion, in short, bottom line, "
    "the bottom line, one more thing, another thing, the other thing, the next thing, the last thing, "
    "the final thing, the next step, the last step, the final step, the next one, the last one"
)
_FULL += [("the", o, n) for o in ("first", "second", "third", "fourth", "fifth", "next", "last", "final")
          for n in ("thing", "step", "tip", "part", "one", "reason", "mistake", "rule", "way", "lesson", "idea")]
_FULL += [(lead, n) for lead in ("number", "step", "tip", "part", "rule", "lesson", "reason", "chapter")
          for n in _ORDINALS]
_PART = _phrases(
    "now, so now, okay so, ok so, alright so, all right so, right so, after that, after this, afterwards, "
    "and then, then, eventually, later, later on, meanwhile, once, here's, here is, here are, but here's, "
    "on top of that, what about, how about, now that, at the end, in the end"
)
_SLIGHT = _phrases("so, and so, anyway, anyways, oh and, but, and also, also, plus, and, okay, alright, well")
_LEADS = {"and", "so", "but", "okay", "ok", "alright", "right", "well", "oh", "yeah", "now"} | HESITATIONS
_SUBJECTS = {"i", "we", "you", "they", "he", "she"}
_MOVES_ON = {"then", "eventually", "later", "next", "finally", "also", "afterwards", "now"}
# A new part can begin in the middle of a sentence the transcriber ran on ("...and [pause] after that I
# went..."), but only where a clause can open: never between "I" and "got off", nor before "while".
_OPENERS = set("""
and but so then now after afterwards eventually later meanwhile also plus anyway anyways okay ok alright first
next finally lastly second third another one i we you they he she it it's there here here's this that's let's
what how why my our
""".split()) | HESITATIONS
# Words a part cannot end on, unless the voice stopped after them: "and [pause] I was also..." is the
# speaker holding the floor, and the "and" goes with what came before.
_DANGLING = set("""
a an the my your his her its our their this these those to of in on at for with from by into about over
under after before between through without i we they he she you is are was were be been am have has had do
does did will would can could should may might must very really just that which who whose
""".split())
_CONNECTIVES = {"and", "so", "but", "or", "then", "because"}
# Words a named thing follows: two words after one of these are a thing ("the train station").
_NAMES_THINGS = set("""
the a an my your his her our their its this that these those some any every each no of for to at in on with
from about into onto by one two three four five six seven eight nine ten more most new own whole first last
next another
""".split())


def signpost(keys: Sequence[str]) -> tuple[str, float]:
    """The signpost a sentence opens with, and how strongly it says a new part begins (0 to 1):
    ("first of all", 1.0), ("then", 0.6), ("so", 0.25), or ("", 0)."""
    keys = [k for k in keys if k]

    def match(seq: Sequence[str]) -> tuple[str, float]:
        best = ("", 0.0)
        for table, strength in ((_FULL, 1.0), (_PART, 0.6), (_SLIGHT, 0.25)):
            for p in table:
                longer = strength == best[1] and len(p) > len(best[0].split())
                if tuple(seq[:len(p)]) == p and (strength > best[1] or longer):
                    best = (" ".join(p), strength)
        return best

    found = match(keys)
    # "And finally", "so first", "okay, next": a connective in front never hides the signpost.
    rest = list(keys)
    while rest and rest[0] in _LEADS:
        rest = rest[1:]
        inner = match(rest)
        if inner[1] > found[1]:
            found = (" ".join(keys[:len(keys) - len(rest)] + inner[0].split()), inner[1])
    # A story moves on after its subject: "I then boarded", "we eventually got there". Only words
    # that move a story along count there, and only part way.
    if rest and rest[0] in _SUBJECTS and len(rest) > 1 and rest[1] in _MOVES_ON:
        if 0.6 > found[1]:
            found = (" ".join(keys[:len(keys) - len(rest) + 2]), 0.6)
    return found


# ---- candidates --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Boundary:
    """A place a new part could begin: between word `after` and the word after it."""
    after: int  # the last word before it
    at: float  # when the next part's first word begins (seconds)
    room: tuple[float, float]  # the quiet between the two words: where a card can sit with nobody speaking
    pause: float  # how long the voice stops there, read from the sound
    sentence: bool  # the word before ends a sentence
    signpost: str  # the words the next sentence opens with, when they announce something
    strength: float  # how strongly that signpost announces a new part (0 to 1)
    question: bool  # the next sentence is a short question
    shift: float = 0.0  # how little the words either side share (0: the same subject, 1: nothing in common)
    score: float = 0.0  # all of it weighed

    def why(self) -> str:
        bits = []
        if self.pause >= 0.15:
            bits.append(f"a {self.pause:.1f} s pause")
        if self.signpost:
            bits.append(f"“{self.signpost}”")
        if self.question:
            bits.append("a question")
        bits.append(f"the words change {self.shift:.0%}")
        return ", ".join(bits)


def _quiet_between(a: float, b: float, quiet: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """The longest quiet stretch between two words, clipped to the gap between them."""
    best = (a, a)
    for q0, q1 in quiet:
        lo, hi = max(q0, a - 0.06), min(q1, b + 0.06)
        if hi - lo > best[1] - best[0]:
            best = (max(lo, a), min(hi, b)) if min(hi, b) > max(lo, a) else (lo, hi)
    return best


def candidates(words: Sequence[Word], quiet: Sequence[tuple[float, float]]) -> list[Boundary]:
    """Every place a new part could begin, with its evidence (`score` is filled in by `weigh`)."""
    words = list(words)
    ends = set()
    for i in range(len(words) - 1):
        if words[i].ends_sentence or pause_after(words, i, quiet) >= SPLIT_PAUSE:
            ends.add(i)
    # A stop just before an "um" and one just after it are the same place: keep the one before it,
    # so the part (and anything announcing it) starts with the hesitation, never in the middle of it.
    for i in sorted(ends):
        if i - 1 in ends and words[i].key in HESITATIONS:
            ends.discard(i)
    out = []
    for i in sorted(ends):
        nxt = i + 1
        pause = pause_after(words, i, quiet)
        if not words[i].ends_sentence:
            opens = words[nxt].key in _OPENERS or words[i].ends_clause
            dangles = words[i].key in _DANGLING or (words[i].key in _CONNECTIVES and pause < 0.5)
            if not opens or dangles:
                continue
        # The next sentence's opening words, up to its first stop.
        opening = []
        j = nxt
        while j < len(words) and len(opening) < 7:
            opening.append(words[j].key)
            if words[j].ends_sentence:
                break
            j += 1
        sp, strength = signpost(opening)
        if strength and not words[i].ends_sentence and pause < 1.0:
            strength *= 0.35  # "...which is good, the next step is...": said inside a sentence, not announced
        # A short question opens a part ("Want something more editorial?") more often than not.
        k, n = nxt, 0
        while k < len(words) and n < 9 and not words[k].ends_sentence:
            k, n = k + 1, n + 1
        question = k < len(words) and n < 9 and words[k].text.rstrip().endswith("?")
        room = _quiet_between(words[i].end, words[nxt].start, quiet)
        out.append(Boundary(after=i, at=words[nxt].start, room=room, pause=pause, sentence=words[i].ends_sentence,
                            signpost=sp, strength=round(strength, 3), question=question))
    return out


# ---- the words changing --------------------------------------------------------------------------------

def _sentences(words: Sequence[Word], cuts: Sequence[int]) -> list[list[int]]:
    out, cur = [], []
    marks = set(cuts)
    for i in range(len(words)):
        cur.append(i)
        if i in marks:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def _idf(words: Sequence[Word], units: list[list[int]]) -> dict[str, float]:
    df: Counter = Counter()
    for u in units:
        df.update({c for i in u if (c := content(words[i]))})
    n = max(1, len(units))
    return {w: math.log((1 + n) / (1 + d)) + 1.0 for w, d in df.items()}


def _vector(words: Sequence[Word], idx: Sequence[int], idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(c for i in idx if (c := content(words[i])))
    return {w: n * idf.get(w, 1.0) for w, n in counts.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> Optional[float]:
    if not a or not b:
        return None
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else None


def weigh(words: Sequence[Word], cands: list[Boundary], window: float) -> list[Boundary]:
    """Each candidate with how much the words change across it and its whole score."""
    words = list(words)
    units = _sentences(words, [c.after for c in cands])
    idf = _idf(words, units)
    shifts = []
    for c in cands:
        before = [i for i in range(c.after + 1) if words[i].start >= c.at - window]
        after = [i for i in range(c.after + 1, len(words)) if words[i].start < c.at + window]
        sim = _cosine(_vector(words, before, idf), _vector(words, after, idf))
        shifts.append(None if sim is None else 1.0 - sim)
    known = sorted(s for s in shifts if s is not None)
    usual = known[len(known) // 2] if known else 0.5
    out = []
    for c, shift in zip(cands, shifts):
        # Measured against how much the words usually change in this talk: someone who never repeats a
        # word changes them "completely" at every sentence, which says nothing.
        telling = 0.5 if shift is None else min(1.0, max(0.0, 0.5 + (shift - usual)))
        score = (WEIGHTS["pause"] * min(1.0, c.pause / PAUSE_FULL)
                 + WEIGHTS["sentence"] * (1.0 if c.sentence else 0.0)
                 + WEIGHTS["signpost"] * c.strength
                 + WEIGHTS["shift"] * telling
                 + WEIGHTS["question"] * (1.0 if c.question else 0.0))
        out.append(Boundary(**{**c.__dict__, "shift": round(0.5 if shift is None else shift, 3),
                               "score": round(score, 3)}))
    return out


# ---- choosing the boundaries ------------------------------------------------------------------------------

def opening_min(min_part: float) -> float:
    """The shortest the first part may be. An opening that says what is coming ("Today, three ways to
    edit faster.") is often only a sentence long, so it may be shorter than the parts after it."""
    return min(min_part, max(SHORTEST, 0.6 * min_part))


def choose(cands: Sequence[Boundary], duration: float, min_part: float, count: Optional[int] = None,
           bar: float = BAR, first_min: Optional[float] = None) -> list[Boundary]:
    """The boundaries that make the parts: every part at least `min_part` long (the last to
    `duration`), the first at least `first_min` (by default `opening_min`). With `count` (the
    number of parts), exactly count - 1 boundaries, the strongest set that fits; otherwise the
    boundaries whose score clears `bar`, the set worth most. Fewer than asked come back when the
    video cannot hold that many parts of that length."""
    if count is not None and count <= 1:
        return []
    first_min = opening_min(min_part) if first_min is None else first_min
    cs = sorted((c for c in cands if first_min <= c.at <= duration - min_part), key=lambda c: c.at)
    n = len(cs)
    if not n:
        return []
    # earlier[j]: the candidates far enough before j to sit with it
    earlier = [[i for i in range(j) if cs[j].at - cs[i].at >= min_part] for j in range(n)]
    if count is None:
        # A chain of boundaries, each worth its score less the bar: the best chain ending at each one.
        best = [0.0] * n
        back = [-1] * n
        for j in range(n):
            prev = max(earlier[j], key=lambda i: best[i], default=-1)
            gain = cs[j].score - bar
            if prev >= 0 and best[prev] > 0:
                best[j], back[j] = best[prev] + gain, prev
            else:
                best[j] = gain
        j = max(range(n), key=lambda j: best[j])
        if best[j] <= 0:
            return []
    else:
        want = count - 1
        # best[k][j]: the most a chain of k boundaries ending at candidate j is worth.
        best_k = [[-math.inf] * n for _ in range(want + 1)]
        back_k = [[-1] * n for _ in range(want + 1)]
        for j in range(n):
            best_k[1][j] = cs[j].score
        for k in range(2, want + 1):
            for j in range(n):
                for i in earlier[j]:
                    if best_k[k - 1][i] + cs[j].score > best_k[k][j]:
                        best_k[k][j], back_k[k][j] = best_k[k - 1][i] + cs[j].score, i
        k = next((k for k in range(want, 0, -1) if max(best_k[k]) > -math.inf), 0)  # the most that fit
        if not k:
            return []
        j = max(range(n), key=lambda j: best_k[k][j])
        chain = []
        while k >= 1 and j >= 0:
            chain.append(cs[j])
            j, k = back_k[k][j], k - 1
        return sorted(chain, key=lambda c: c.at)
    chain = []
    while j >= 0:
        chain.append(cs[j])
        j = back[j]
    return sorted(chain, key=lambda c: c.at)


# ---- titles ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Given:
    """A title the person typed, and the time its part begins when they gave one."""
    title: str
    at: Optional[float] = None


_TIME = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{2}(?:\.\d+)?)\s+(.*)$|^\s*(\d+(?:\.\d+)?)s\s+(.*)$")


def parse_time(text: str) -> Optional[float]:
    """"1:05", "01:02:03" or "65s" as seconds, else None."""
    m = re.fullmatch(r"\s*(?:(\d+):)?(\d{1,2}):(\d{2}(?:\.\d+)?)\s*", text)
    if m:
        return int(m.group(1) or 0) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*s\s*", text)
    return float(m.group(1)) if m else None


def parse_titles(spec: str) -> tuple[Optional[int], list[Given]]:
    """What `titles=` says: ("auto" or empty) nothing; a number, how many parts; or titles separated
    by |, each of which may start with the time its part begins ("0:31 The train", "31s The train").
    Returns (count or None, the titles)."""
    s = (spec or "").strip()
    if not s or s.lower() == "auto":
        return None, []
    if re.fullmatch(r"\d{1,2}", s):
        return int(s), []
    out = []
    for piece in s.split("|"):
        piece = piece.strip()
        if not piece:
            continue
        m = _TIME.match(piece)
        if m and m.group(4) is not None and m.group(4).strip():
            at = int(m.group(1) or 0) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
            out.append(Given(m.group(4).strip(), at))
        elif m and m.group(6) is not None and m.group(6).strip():
            out.append(Given(m.group(6).strip(), float(m.group(5))))
        else:
            out.append(Given(piece))
    return (len(out) or None), out


def _display(w: Word, after: Optional[Word] = None) -> str:
    """A word as a title shows it: as heard, without punctuation at its edges, and joined to an
    ending the transcriber split off it ("askeditor" ".com" is "askeditor.com")."""
    t = w.clean.strip().strip("\"'“”‘’.,;:!?()[]{}…-–—")
    if after is not None and after.text.strip().startswith(".") and len(after.key) <= 4 and after.key.isalpha():
        t = f"{t}.{after.key}"
    return t or w.key


def _is_name(words: Sequence[Word], i: int) -> bool:
    t = _display(words[i])
    if len(t) >= 2 and t.isupper():
        return True  # an acronym: DJI, UGC
    if t[:1].isupper() and i > 0 and not words[i - 1].ends_sentence and t.lower() != "i":
        return True  # capitalised in the middle of a sentence: a name or a place
    return any(c.isupper() for c in t[1:])  # GoPro, AskEditor


def _title_case(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


@dataclass
class Part:
    """One part of the talk."""
    index: int
    start: float  # when it begins on the video's timeline (0 for the first)
    end: float  # when the next one begins (the video's end for the last)
    first: int  # its first word (an index into the words it was found from)
    last: int  # its last word
    title: str
    given: bool = False  # the title is the person's own
    boundary: Optional[Boundary] = None  # what showed it begins here (None for the first)
    keys: list[str] = field(default_factory=list)  # its key words, most specific first

    @property
    def seconds(self) -> float:
        return self.end - self.start

    @property
    def room(self) -> Optional[tuple[float, float]]:
        """The quiet just before it begins, where nobody is speaking."""
        return self.boundary.room if self.boundary else None

    def said(self, words: Sequence[Word]) -> str:
        return " ".join(w.text for w in words[self.first:self.last + 1] if w.key not in HESITATIONS)

    def to_json(self) -> dict:
        b = self.boundary
        return {"index": self.index + 1, "start": round(self.start, 3), "end": round(self.end, 3),
                "title": self.title, "given": self.given, "words": [self.first, self.last],
                "keys": self.keys[:5],
                **({"pause_before": [round(b.room[0], 3), round(b.room[1], 3)], "why": b.why(),
                    "score": b.score} if b else {})}


def key_words(words: Sequence[Word], spans: Sequence[tuple[int, int]], quiet=(),
              skip: frozenset = frozenset()) -> list[list[tuple[str, float]]]:
    """For each span of words (a part), its key words and pairs of words, most telling first, each
    with its weight: said in this part more than in the others, specific (long, a name, a figure),
    and said more than once. Pairs are two content words said together in one breath ("train
    station", "flip camera"); a pair outranks either of its words when both are telling. The words
    at the indexes in `skip` (what the speaker took back) count for nothing."""
    words = list(words)
    per_part: list[Counter] = []
    shows: list[dict[str, str]] = []
    for a, b in spans:
        c: Counter = Counter()
        show: dict[str, str] = {}
        for i in range(a, b + 1):
            k = None if i in skip else content(words[i])
            if k:
                c[k] += 1
                d = _display(words[i], words[i + 1] if i + 1 < len(words) else None)
                show.setdefault(k, d if _is_name(words, i) else d.lower())
        per_part.append(c)
        shows.append(show)
    n = max(1, len(spans))
    df: Counter = Counter()
    for c in per_part:
        df.update(set(c))
    out = []
    for p, ((a, b), c) in enumerate(zip(spans, per_part)):
        weight: dict[str, float] = {}
        for k, tf in c.items():
            spec = 1.0 + 0.06 * min(len(k), 10)
            if any(ch.isdigit() for ch in k):
                spec += 0.6
            if k.endswith(("ing", "ed")) and len(k) > 5:
                spec *= 0.7  # most likely a verb ("chilling", "boarded"): a part is named by a thing
            idf = math.log((n + 1) / (df[k] + 0.5)) + 0.4
            weight[k] = (1.0 + 0.6 * math.log(tf)) * idf * spec
        names = {content(words[i]) for i in range(a, b + 1)
                 if i not in skip and content(words[i]) and _is_name(words, i)}
        for k in names:
            weight[k] = weight.get(k, 0.0) * 1.35
        ranked: dict[str, float] = {shows[p][k]: v for k, v in weight.items()}
        # pairs said together, where a thing can be named: after "the", "my", "of", a number, or at
        # the start of a sentence ("the train station", "a chicken patty"), never a subject and its
        # verb ("so captions carry", "and pauses add up")
        for i in range(a, b):
            k1, k2 = content(words[i]), content(words[i + 1])
            if not (k1 and k2) or k1 == k2 or words[i].ends_clause or i in skip or i + 1 in skip:
                continue
            if pause_after(words, i, quiet) >= 0.3:
                continue
            before = words[i - 1] if i > 0 else None
            names = _is_name(words, i) and _is_name(words, i + 1)
            if not (names or before is None or before.ends_sentence or before.key in _NAMES_THINGS
                    or any(ch.isdigit() for ch in before.key)):
                continue
            pair = f"{shows[p][k1]} {shows[p][k2]}"
            v = 0.62 * (weight[k1] + weight[k2])
            if v > ranked.get(pair, 0.0):
                ranked[pair] = v
        order = sorted(ranked.items(), key=lambda kv: (-kv[1], kv[0]))
        # A lone word that is half of a pair said here ("lively" in "lively city") gives way to the
        # pair when the pair is nearly as telling: a title reads as a thing, not an adjective.
        if order and " " not in order[0][0]:
            top, v = order[0]
            for text, w in order[1:]:
                if " " in text and top in text.split() and w >= 0.8 * v:
                    order.remove((text, w))
                    order.insert(0, (text, w))
                    break
        out.append(order)
    return out


def _pick_titles(ranked: list[list[tuple[str, float]]], most_letters: int = 22) -> list[str]:
    """One title a part, never the same as an earlier part's, never longer than `most_letters`."""
    used: set[str] = set()
    out = []
    for options in ranked:
        title = ""
        for text, _ in options:
            if len(text) > most_letters:
                continue
            folded = {_stem(normalize(x)) for x in text.split()}
            if any(folded & {_stem(normalize(x)) for x in u.split()} for u in used):
                continue
            title = text
            break
        if not title and options:
            title = options[0][0][:most_letters].rstrip()
        used.add(title)
        out.append(_title_case(title) if title else "")
    return out


# ---- the whole thing ---------------------------------------------------------------------------------------

def _snap(cands: Sequence[Boundary], words: Sequence[Word], quiet, t: float) -> Boundary:
    """The candidate nearest a time the person gave (within 2.5 s), else a boundary made there, at
    the first word that starts at or after it."""
    near = [c for c in cands if abs(c.at - t) <= 2.5]
    if near:
        return min(near, key=lambda c: (abs(c.at - t), -c.score))
    j = next((i for i, w in enumerate(words) if w.start >= t - 0.05), len(words) - 1)
    i = max(0, j - 1)
    room = _quiet_between(words[i].end, words[j].start, quiet) if j > i else (t, t)
    return Boundary(after=i, at=words[j].start if j > i else t, room=room, pause=pause_after(list(words), i, quiet),
                    sentence=words[i].ends_sentence, signpost="", strength=0.0, question=False, shift=0.0, score=0.0)


def find(words: Sequence[Word], quiet: Sequence[tuple[float, float]], duration: float,
         min_part: Optional[float] = None, count: Optional[int] = None,
         titles: Sequence[Given] = (), first_min: Optional[float] = None) -> list[Part]:
    """The parts of the talk, in order, each with a title.

    `min_part`: the shortest a part may be (None: `auto_min_part`); `first_min` the shortest the
    opening part may be (None: `opening_min`; YouTube chapters pass 10). `count`: how many parts to
    make (None: as many as the evidence holds, at most `most_parts`). `titles`: the person's own,
    which also set the count; one that carries a time pins where its part begins (snapped to the
    nearest real boundary within 2.5 s). A video with no words is one part."""
    words = list(words)
    if not words:
        return [Part(0, 0.0, duration, 0, -1, titles[0].title if titles else "", bool(titles))]
    floor = auto_min_part(duration) if min_part is None else max(SHORTEST, float(min_part))
    cands = weigh(words, candidates(words, quiet), window=max(6.0, min(20.0, floor)))
    if titles:
        count = len(titles)
    pinned = [g for g in titles[1:] if g.at is not None]
    if titles and pinned and len(pinned) == len(titles) - 1:
        chosen = [_snap(cands, words, quiet, g.at) for g in titles[1:]]
    else:
        chosen = choose(cands, duration, floor, count, first_min=first_min)
        if count is None and len(chosen) + 1 > most_parts(duration):
            chosen = choose(cands, duration, floor, most_parts(duration), first_min=first_min)
        if count and len(chosen) < count - 1:
            # The video cannot hold that many parts that long: shorter parts, down to the floor of all.
            for shorter in (floor * 0.75, floor * 0.5, SHORTEST):
                chosen = choose(cands, duration, max(SHORTEST, shorter), count,
                                first_min=None if first_min is None else min(first_min, max(SHORTEST, shorter)))
                if len(chosen) >= count - 1:
                    break
    starts = [0.0] + [c.at for c in chosen]
    ends = starts[1:] + [duration]
    firsts = [0] + [c.after + 1 for c in chosen]
    lasts = [c.after for c in chosen] + [len(words) - 1]
    spans = list(zip(firsts, lasts))
    # A part is never named by what the speaker took back ("the lively town, or the lively city")
    # or by the words of a first attempt said again: the same rule every picker in the core keeps.
    back = moments.taken_back(words, quiet)
    ranked = key_words(words, spans, quiet, skip=frozenset(i for i, w in enumerate(words) if w in back))
    auto = _pick_titles(ranked)
    out = []
    for k, (s, e, (a, b)) in enumerate(zip(starts, ends, spans)):
        given = titles[k].title if k < len(titles) else ""
        out.append(Part(index=k, start=s, end=e, first=a, last=b, title=given or auto[k], given=bool(given),
                        boundary=chosen[k - 1] if k else None, keys=[t for t, _ in ranked[k][:6]]))
    return out


# ---- chapters as YouTube reads them ---------------------------------------------------------------------------

def stamp(seconds: float, hours: bool = False) -> str:
    """A time as a chapter line writes it: "00:00", "01:05", or "1:02:03" in a video of an hour or more."""
    s = int(math.floor(max(0.0, seconds) + 1e-6))
    h, m, sec = s // 3600, (s // 60) % 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if hours else f"{s // 60:02d}:{sec:02d}"


def chapter_lines(found: Sequence[Part], duration: float) -> list[str]:
    """The parts as YouTube chapter lines ("00:00 Title"), each at the second its part begins (the
    first at 00:00)."""
    hours = duration >= 3600
    return [f"{stamp(0.0 if p.index == 0 else p.start, hours)} {p.title}" for p in found]


def check_chapters(lines: Sequence[str], duration: float) -> list[str]:
    """What is wrong with chapter lines by YouTube's own rules (YouTube Help, "Video Chapters", read
    28 October 2026: the first timestamp 00:00, at least three in ascending order, each chapter at
    least 10 seconds). Empty when they pass."""
    problems = []
    times = []
    for ln in lines:
        m = re.match(r"^\s*((?:\d+:)?\d{1,2}:\d{2})\s+(\S.*)$", ln)
        if not m:
            problems.append(f"“{ln.strip()}” is not a time and a title")
            continue
        t = parse_time(m.group(1))
        times.append(0.0 if t is None else t)
    if not times:
        return problems or ["no chapter lines"]
    if times[0] != 0:
        problems.append("the first chapter must start at 00:00")
    if len(times) < 3:
        problems.append(f"YouTube needs at least three chapters; there are {len(times)}")
    for k in range(1, len(times)):
        if times[k] <= times[k - 1]:
            problems.append(f"chapter {k + 1} does not come after chapter {k}")
    for k, t in enumerate(times):
        nxt = times[k + 1] if k + 1 < len(times) else duration
        if nxt - t < YOUTUBE_SHORTEST - 1e-6:
            problems.append(f"chapter {k + 1} lasts {max(0.0, nxt - t):.0f} s; YouTube needs at least 10")
    return problems
