"""Where a caption breaks a sentence into cards: the words shown together, cut where a reader expects.

Every caption the pack draws comes through here (`captions`, clean-cut's captions, split-layout's
panel, the answers prompt-box lays down, and any effect that puts the pack's captions over itself).
A card is the words on screen at once. Where one card ends and the next begins is decided by how
the words hang together first, and only then by how even the cards are:

1. **A sentence ends a card, and so does a long silence** (`HOLD_ACROSS`): a card is never held
   on screen across one. A shorter pause is a good place to cut, never a forced one: unscripted
   speech stops after "the", "and" or "who" while the next word is found, and a card cut there
   leaves the reader hanging on a word that points at nothing.
2. **The best cuts, in order:** after a comma (a dash, a colon) or a real pause in the voice;
   before a word that opens a clause ("and", "but", "so", "because", "which", "who", "once",
   "if", "we"); before a word that opens a phrase ("to", "in", "with", "the", "my", "is"). A cut
   between two words of one phrase ("an inbuilt | video edit") is worse, and one just before a
   word that finishes the phrase in front of it ("set | up", "makes | you", "bigger | than") is
   worse again.
3. **A card does not end on a word that leans forward** ("the", "of", "is", "and", "my", "to",
   "with", "really"), unless something closes it: a comma after it, or a word such as "and" after
   it that shows what it began is over ("the next step is | and what makes you").
4. **What is said as one is never split:** "right now", "kind of", "going to", "thank you", a first
   and last name, a number and what it counts, two pieces of one figure ("500" ",000", "in" "-built").
5. **Cards are even, within the look's number of words.** That number may be passed by one word
   where it keeps a phrase whole; a number the person sets (`max_words`) is never passed. One word
   is alone on a card only when it was said alone: a sentence of its own, or between long silences.

Only which words go together is decided here. When a card shows and for how long stays with the
captions: it starts on its first word and ends after its last.

The word lists are written for this pack, from how people talk on camera: the test clips and
our own takes, read card by card (`tests/phrasing.py` counts what still reads badly).
"""

from __future__ import annotations

import math
import re

from .words import Word, pause_after

BREAK_PAUSE = 0.30  # a real stop in the voice: as good a place to cut as a comma
LIST_PAUSE = 0.12  # a comma said with less than this after it runs straight on ("a clean, bold one")
HOLD_ACROSS = 1.5  # a silence this long between two words always ends the card before it
LONGEST_RUN = 48  # a run of more words than this is halved at its best cut before it is cut into cards


def _set(text: str) -> frozenset[str]:
    return frozenset(text.split())


# ---- the words, by what they do at the edge of a card ------------------------------------------

# Words that open a clause: a cut just before one is where a speaker would breathe.
OPENS_CLAUSE = _set("""
and but or nor so yet then because although though while whereas if unless until once when whenever where
wherever whether which who whom whose what why how that
""")
# Words that show the phrase before them is over: "draws people in | and", "the next step is | and".
COORDINATORS = _set("and but or nor so then")
# The one doing something, at the head of a clause: a cut before is good, a cut after leaves the
# reader waiting for the verb. ("you" and "it" can be either end of a verb, so they are not here.)
SUBJECTS = _set("""
i we they he she i'm i've i'll i'd we're we've we'll we'd they're they've they'll they'd he's he'll he'd
she's she'll she'd you're you've you'll you'd it's it'll that's there's here's what's who's let's
""")
# Words that open a phrase and lean into it.
PREPOSITIONS = _set("""
to of in on at for with from by about into onto through throughout over under after before without within
between among against during toward towards upon via per across around behind beyond below above beside near
like as than since till despite except inside outside along
""")
# Words that usually finish a verb ("set up", "stand out", "come back") rather than start a phrase.
PARTICLES = _set("up down out off back away")
# Words that always begin a noun phrase.
DETERMINERS = _set("a an the my your our their its his every each")
# Words that begin a noun phrase or are one ("this product" / "just like this"): they lean forward
# only into a word that carries their phrase on.
EITHER = _set("""
this that these those some any all both her much many more most less least few another other such no
""")
# Verbs that need what follows them. "have" and "do" can also be the verb itself, with a thing
# after them ("we have | an editor"): then they finish their clause's start and do not lean.
AUXILIARIES = _set("""
am is are was were be been being have has had do does did will would can could should shall may might must
isn't aren't wasn't weren't don't doesn't didn't won't wouldn't can't couldn't shouldn't haven't hasn't hadn't
ain't gonna wanna gotta
""")
HAVE_DO = _set("have has had do does did")
# Words that bend the word after them ("really good", "just ask", "not sure").
BENDS_NEXT = _set("""
very really real so too quite pretty super extremely totally absolutely incredibly just not never even only
also almost still already always
""")
# Words that end the phrase in front of them when they follow its verb: "makes | you" splits one.
OBJECTS = _set("me him us them you it her")
# Words that stand for a noun, so a verb after them is their own: "it | is" splits a clause's start.
PRONOUNS = OBJECTS | _set("""
this that these those there here what who which one everyone everybody someone somebody everything something
anything anyone nothing nobody
""")
NUMBER_WORDS = _set("""
zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen
seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million
billion trillion dozen half
""")
# Words that open a phrase or a clause firmly enough that the word before them is finished.
STARTS = OPENS_CLAUSE | SUBJECTS | PREPOSITIONS | AUXILIARIES
# A full stop after these is the word's own, not the end of a sentence: "Mr. Smith" is one name.
TITLES = _set("mr mrs ms dr st prof")

# Two words said as one: a cut between them reads as two thoughts where the speaker had one. Only
# pairs the rules above would let a cut fall inside are listed ("of course" is safe already: "of"
# leans forward).
SAID_AS_ONE = frozenset(tuple(p.split()) for p in """
right now|right here|right there|right away|right back|right after|right before
kind of|sort of|type of|out of|because of|instead of|lot of|lots of|bunch of|couple of|front of|ahead of
top of|all of|most of|some of|none of|each of|both of|any of|many of|few of|half of|one of|rest of|part of
going to|want to|wants to|wanted to|have to|has to|had to|need to|needs to|needed to|got to|used to|able to
trying to|try to|tried to|supposed to|ought to|like to|love to|hate to|plan to|planning to|seem to|seems to
start to|started to|begin to|began to|happen to|happened to|manage to|managed to|learn to|decided to
thank you|you know|no one|each other|one another|make sure|pretty much|rather than|more than|less than
other than|all right|real quick|real big|real good|up to
out there|up there|down there|over there|back there|out here|up here|down here|over here|back here
""".replace("\n", "|").split("|") if p.strip())

# How good a cut is, by where it falls (lower is better), and what each fault costs on top. Where
# the cuts fall comes first; how even the cards are only settles between cuts nearly as good.
AT_A_STOP = 0.0  # after a comma, a dash, a colon or a real pause
BEFORE_A_CLAUSE = 1.5  # before "and", "because", "which", "we"...
BEFORE_A_PHRASE = 3.0  # before "to", "the", "is"...
INSIDE_A_PHRASE = 4.5  # between two words of one phrase, as far as the words can tell
MAY_HAVE_ENDED = 5.25  # two words into a noun phrase that may have ended: "the lively city | come"
FINISHES_IT = 9.0  # before the word that finishes the phrase in front: "set | up", "makes | you"
LEANS = 15.0  # a card ending on a word that leans forward
SAID_TOGETHER = 15.0  # a cut between two words said as one
GLUED = 100.0  # a cut between two pieces of one word or figure: only if nothing else can be done
EVEN = 0.5  # each word a card is away from the even share of its run: evenness settles ties
OVER = 3.0  # a card past the look's number of words (allowed by one, where it keeps a phrase whole)
EXTRA_CARD = 1.0  # each card more than the look's number of words needs
LONE = 12.0  # one word alone on a card
LONE_POINT = 1.0  # ... when it is the point of its line: "Distribution | is a real big issue"
ONE_TOO_EARLY = 3.0  # a card that opens on the last word of the thought before it
QUICK = 3.0  # each word (up to three) of a card on screen for less than `QUICK_S`: it flashes by
QUICK_S = 0.4
WAITING = 2.0  # a second of silence held inside one card, beyond `WAIT_S`
WAIT_S = 0.5

_EDGE = re.compile(r"^[-–—,.%]")  # a word that is the rest of the one before ("-built", ",000", "%")
_SIGNS = {"$", "£", "€", "#", "@"}  # a word that is the start of the one after


# ---- reading one edge ----------------------------------------------------------------------------

def is_number(w: Word) -> bool:
    return any(c.isdigit() for c in w.key) or w.key in NUMBER_WORDS


def _title(w: Word) -> bool:
    return w.key in TITLES and w.text.rstrip().endswith(".")


def ends_sentence(words: list[Word], i: int) -> bool:
    """Does a sentence end after word i? A full stop after a title ("Dr.") does not end one."""
    return words[i].ends_sentence and not _title(words[i])


def _closed(w: Word) -> bool:
    """A comma, a dash, a colon or a full stop after the word: nothing it began runs on."""
    return w.ends_clause and not _title(w)


def _name(words: list[Word], i: int) -> bool:
    """A word written with a capital that is not there because it starts a sentence."""
    w = words[i]
    t = w.text.strip().lstrip("\"'“‘(")
    if not t[:1].isupper() or w.key in ("i", "i'm", "i've", "i'll", "i'd"):
        return False
    return i > 0 and not ends_sentence(words, i - 1)


_CLOSED_CLASS = (OPENS_CLAUSE | SUBJECTS | PREPOSITIONS | PARTICLES | DETERMINERS | EITHER | AUXILIARIES
                 | BENDS_NEXT | PRONOUNS | COORDINATORS)


def _content(w: Word) -> bool:
    """A word that carries meaning (a thing, an action, a quality), as far as the lists can tell."""
    return bool(w.key) and w.key not in _CLOSED_CLASS and not is_number(w)


def _opens_noun(w: Word) -> bool:
    """A word that begins a noun phrase: "the", "my", "that" (issue), "two" (ways)."""
    return w.key in DETERMINERS or w.key in EITHER or is_number(w)


def _start_of(words: list[Word], i: int) -> int:
    """The first piece of the word that ends at i ("in" for "in" "-built")."""
    while i > 0 and glued(words, i - 1):
        i -= 1
    return i


def into_noun(words: list[Word], i: int, quiet: list[tuple[float, float]]) -> tuple[int, bool]:
    """How far into a noun phrase a cut after word i would fall, and whether the phrase is sure to
    run on: (1, True) for "the lively | city", (2, False) for "an inbuilt video | edit", (2, True)
    for "a real big | issue" and "a clean, bold | subtitle", whose words still wait for their noun.
    (0, False) when no phrase opened by "the", "my", "two" and the like is open, or the next word
    does not carry it on. A comma said without a pause does not close the phrase; one with a pause
    after it does."""
    if i + 1 >= len(words) or not _content(words[i + 1]):
        return 0, False
    if i + 2 < len(words) and words[i + 2].key in PARTICLES:
        return 0, False  # "the audio | set up": a verb and its particle, not more of the thing
    depth, k, waiting = 0, i, False
    while k >= 0 and depth < 3:
        end, k = k, _start_of(words, k)
        w = words[k]
        if _opens_noun(w) and depth and k == end:
            return (depth, waiting or depth == 1) if not _closed(w) else (0, False)
        if not (_content(w) or w.key in BENDS_NEXT or k < end):  # a word in pieces ("in" "-built") is one word
            return 0, False
        w = words[end]
        if end < i and _closed(w):
            if pause_after(words, end, quiet) >= LIST_PAUSE:
                return 0, False
            waiting = True  # "clean, bold": a list of what the thing is like, the thing still to come
        waiting = waiting or w.key in BENDS_NEXT  # "real big": a word that bends the next is not the noun
        depth += 1
        k -= 1
    return 0, False


def stands_alone(w: Word) -> bool:
    """A word that can be the whole of a card, said as the point of its line: a long word that is
    not grammar, an acronym, or a number."""
    raw = w.text.strip().strip("\"'“”‘’.,;:!?()")
    return (_content(w) and (len(w.key) >= 7 or (raw.isupper() and len(raw) >= 2))) or is_number(w)


def glued(words: list[Word], i: int) -> bool:
    """Are word i and the next two pieces of one word or figure ("in" "-built", "500" ",000", "$" "5")?"""
    nxt = words[i + 1].text.strip()
    return bool(_EDGE.match(nxt)) or not any(c.isalnum() for c in nxt) or words[i].text.strip() in _SIGNS


def said_as_one(words: list[Word], i: int) -> bool:
    """Are word i and the next said as one: a fixed pair, a name, a number and what it counts?"""
    a, b = words[i], words[i + 1]
    if (a.key, b.key) in SAID_AS_ONE:
        return True
    if is_number(a) and (is_number(b) or b.key in ("point", "percent", "k", "x")):
        return True  # "five million", "one point five", "ten percent"
    if a.key == "point" and is_number(b) and i > 0 and is_number(words[i - 1]):
        return True
    if is_number(a) and b.key == "and" and i + 2 < len(words) and is_number(words[i + 2]):
        return True  # "two hundred and fifty"
    return _name(words, i) and _name(words, i + 1) and not _closed(a)


def leans(words: list[Word], i: int) -> bool:
    """Does word i lean into the next, so that a card ending on it leaves the reader hanging?"""
    if i + 1 >= len(words):
        return False
    w, nxt = words[i], words[i + 1]
    if _closed(w):
        return False
    k, n = w.key, nxt.key
    if k in DETERMINERS:
        return True  # "the |" is never finished
    if k in HAVE_DO and _opens_noun(nxt):
        return False  # "we have | an inbuilt video edit": the verb itself, then what it has
    if k in OPENS_CLAUSE or k in SUBJECTS or k in PREPOSITIONS or k in AUXILIARIES or k in BENDS_NEXT:
        return n not in COORDINATORS  # "is | and what" and "in | and all": what it began is over
    if k in PARTICLES:
        # "out | of", "down | the hill" go on; "set up | and", "all the way down | to" do not.
        return n == "of" or n in DETERMINERS or is_number(nxt)
    if k in EITHER or is_number(w):
        return n not in STARTS  # "that | issue", "two | ways"; but "like this | and", "in 2024 | we"
    return False


def where_cut(words: list[Word], i: int, quiet: list[tuple[float, float]]) -> float:
    """How good a place the cut between word i and the next is, from `AT_A_STOP` (0, the best) to
    `FINISHES_IT`, before anything the cut breaks is counted (`cut_cost`)."""
    w, nxt = words[i], words[i + 1]
    k, n = w.key, nxt.key
    paused = pause_after(words, i, quiet) >= BREAK_PAUSE
    depth, open_ = into_noun(words, i, quiet)
    if depth and not paused:
        # "the lively | city", "a clean, | bold subtitle": a comma between two words of one thing
        # is no place to stop. Two words in, the phrase may have ended ("the lively city | come"),
        # unless what it has so far still waits for its noun; three words in, it likely has.
        cost = FINISHES_IT if open_ else MAY_HAVE_ENDED if depth == 2 else INSIDE_A_PHRASE
    elif _closed(w) or paused:
        cost = AT_A_STOP
    elif n == "that" and i + 2 < len(words) and _content(words[i + 2]):
        cost = BEFORE_A_PHRASE  # "solve | that issue": "that" begins a thing here, not a clause
    elif n in OPENS_CLAUSE or n in SUBJECTS:
        cost = BEFORE_A_CLAUSE
    elif n in PARTICLES or n == "than":
        cost = FINISHES_IT
    elif n in OBJECTS and k not in STARTS and k not in DETERMINERS and k not in EITHER:
        cost = FINISHES_IT  # "makes | you", "give | me"
    elif n in AUXILIARIES and k in PRONOUNS:
        cost = FINISHES_IT  # "it | is", "you | can"
    elif n == "of":
        cost = INSIDE_A_PHRASE  # "of" belongs to the word before it more than to the word after
    elif n in PREPOSITIONS or n in DETERMINERS or n in EITHER or n in AUXILIARIES:
        cost = BEFORE_A_PHRASE
    else:
        cost = INSIDE_A_PHRASE
    return cost


def cut_cost(words: list[Word], i: int, quiet: list[tuple[float, float]]) -> float:
    """What a cut between word i and the next costs: where it falls, and anything it breaks."""
    cost = where_cut(words, i, quiet)
    if leans(words, i):
        cost += LEANS
    if said_as_one(words, i):
        cost += SAID_TOGETHER
    if glued(words, i):
        cost += GLUED
    return cost


# ---- the cards ---------------------------------------------------------------------------------

def runs(words: list[Word]) -> list[list[int]]:
    """Word indexes split where no card may run on: at the end of each sentence, and at each
    silence of `HOLD_ACROSS` or more."""
    out, cur = [], []
    for i, w in enumerate(words):
        cur.append(i)
        if i == len(words) - 1 or ends_sentence(words, i) or words[i + 1].start - w.end >= HOLD_ACROSS:
            out.append(cur)
            cur = []
    return out


def _card_cost(words: list[Word], run: list[int], j: int, i: int, share: float, most: int,
               place: list[float]) -> float:
    """What the card of run[j:i] costs, apart from the cut at its end. `place[k]` is how good a
    place the cut after run[k] is."""
    size = i - j
    if size == 1 and stands_alone(words[run[j]]):
        cost = LONE_POINT  # a word set apart on purpose is not held to the others' length
    else:
        cost = EVEN * abs(size - share) + (LONE if size == 1 else 0.0)
    if size > most:
        cost += OVER
    if (size >= 2 and j > 0 and place[j - 1] > BEFORE_A_CLAUSE and place[j] <= BEFORE_A_CLAUSE
            and not leans(words, run[j])):
        # "big | issue and most ads": the card opens on the last word of the thought before it,
        # one word short of a good place to start.
        cost += ONE_TOO_EARLY
    first, last = words[run[j]], words[run[i - 1]]
    # How long the card is up: until the next card's first word when one follows straight on.
    shown = (words[run[i]].start if i < len(run) else last.end + QUICK_S) - first.start
    if shown < QUICK_S:
        cost += QUICK * min(size, 3)
    for a, b in zip(run[j:i - 1], run[j + 1:i]):
        cost += WAITING * max(0.0, words[b].start - words[a].end - WAIT_S)
    return cost


def split_run(words: list[Word], run: list[int], quiet: list[tuple[float, float]], most: int,
              over: int = 1) -> list[list[int]]:
    """One run of words as cards of at most `most + over` words, the cuts where they read best."""
    n = len(run)
    if n <= most:
        return [run]
    cap = most + over
    cuts = [cut_cost(words, run[k], quiet) for k in range(n - 1)]  # cuts[k]: after run[k]
    if n > LONGEST_RUN:
        # A long run with no full stop (a list, a take said in one breath) is halved first at its
        # best place to cut near the middle, so the search below only ever weighs a few cards.
        lo, hi = n // 3, n - n // 3
        k = min(range(lo, hi), key=lambda k: (cuts[k], abs(k - n / 2)))
        return (split_run(words, run[:k + 1], quiet, most, over)
                + split_run(words, run[k + 1:], quiet, most, over))
    place = [where_cut(words, run[k], quiet) for k in range(n - 1)]
    fewest, aim = math.ceil(n / cap), math.ceil(n / most)
    best_total, best_cards = math.inf, [run]
    for parts in range(max(fewest, aim - 1), aim + 3):
        if parts > n:
            break
        share = n / parts
        # cost[p][i]: the cheapest way to make the first i words into p cards.
        cost = [[math.inf] * (n + 1) for _ in range(parts + 1)]
        back = [[0] * (n + 1) for _ in range(parts + 1)]
        cost[0][0] = 0.0
        for p in range(1, parts + 1):
            for i in range(p, n + 1):
                for j in range(max(p - 1, i - cap), i):
                    if cost[p - 1][j] == math.inf:
                        continue
                    c = cost[p - 1][j] + _card_cost(words, run, j, i, share, most, place) + (cuts[i - 1] if i < n else 0.0)
                    if c < cost[p][i]:
                        cost[p][i], back[p][i] = c, j
        total = cost[parts][n] + EXTRA_CARD * max(0, parts - aim)
        if total < best_total:
            cards, i = [], n
            for p in range(parts, 0, -1):
                j = back[p][i]
                cards.append(run[j:i])
                i = j
            best_total, best_cards = total, cards[::-1]
    return best_cards


def phrases(words: list[Word], quiet: list[tuple[float, float]], most: int, over: int = 1) -> list[list[int]]:
    """Word indexes grouped into the cards shown together, each of at most `most` words, or
    `most + over` where that keeps a phrase whole (pass `over=0` when the person chose `most`)."""
    if not words:
        return []
    if most <= 1:
        return [[i] for i in range(len(words))]
    out: list[list[int]] = []
    for run in runs(words):
        out.extend(split_run(words, run, quiet, most, over))
    return out

