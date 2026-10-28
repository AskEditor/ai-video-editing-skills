"""Numbers: the amounts said in a clip, and the ones a person types, each as a value with its unit.

    said = amounts.spoken(words)           # every amount heard, in order, with the words that said it
    run = amounts.series(said, words)      # the amounts a chart would show: "from 2K to 197K"
    typed = amounts.parse_list("2K, 15K")  # amounts a person typed
    unit = amounts.Unit.of(run.amounts)    # how the run is written: "$", "K", "%", decimals
    unit.exact(197000) -> "197K"           # a value as it lands; unit.during(57312) while counting

What counts as an amount: figures as the transcriber writes them ("2,000", "197K", "$5", "15%",
"1.5", "10x"), the same with a word after them ("1.5 million", "15 percent", "5 dollars"), and
numbers said in words ("two hundred and fifty thousand", "one point five million"). What does not:
years ("2024", "twenty twenty-four"), lengths of time ("six months", "3 weeks"), and a small number
said on its own ("one of them", "two things"), which is only taken inside a pair ("from two to ten").

Nothing here rounds or converts an amount for the screen: a value is written the way it was said
or typed, shortened ("197,000" to "197K") only when the short form says exactly the same number.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

from .words import Word, normalize

# ---- the words numbers are said with ---------------------------------------------------------------

SMALL = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
SCALE_WORDS = {"thousand": (1e3, "K"), "million": (1e6, "M"), "billion": (1e9, "B"),
               "k": (1e3, "K"), "mil": (1e6, "M"), "bn": (1e9, "B")}
SCALE_LETTERS = {"K": 1e3, "M": 1e6, "B": 1e9}
CURRENCY_WORDS = {"dollars": "$", "dollar": "$", "bucks": "$", "usd": "$", "pounds": "£", "quid": "£",
                  "gbp": "£", "euros": "€", "euro": "€", "eur": "€"}
PERCENT_WORDS = {"percent", "per-cent", "pc"}
TIME_WORDS = {"second", "seconds", "sec", "secs", "minute", "minutes", "min", "mins", "hour", "hours", "hr",
              "hrs", "day", "days", "week", "weeks", "month", "months", "year", "years", "yr", "yrs", "decade",
              "decades", "am", "pm", "o'clock", "oclock", "weekend", "weekends", "night", "nights", "morning",
              "mornings", "quarter", "quarters"}
# Words that may sit between a number and the time word after it ("six whole months", "3 more weeks").
TIME_FILLERS = {"more", "whole", "full", "long", "short", "straight", "or", "so", "and", "a", "half", "extra"}
_TYPED = re.compile(
    r"^(?P<sign>[-+−])?(?P<pre>[$£€])?(?P<sign2>[-+−])?"
    r"(?P<int>\d{1,3}(?:,\d{3})+|\d+)?(?:\.(?P<dec>\d+))?"
    r"(?P<scale>k|m|b|bn|thousand|million|billion)?(?P<suf>%|x|percent)?$", re.I)
_EDGE = re.compile(r"^[\"“'(\[]+|[\"”')\].,!?;:…]+$")


@dataclass(frozen=True)
class Amount:
    """One amount: its value, how it is written, and (when heard) the words that said it."""
    value: float
    prefix: str = ""  # "$", "£", "€" or ""
    suffix: str = ""  # "%", "x" or ""
    scale: str = ""  # "K", "M", "B" when it was said or typed that way ("197K", "1.5 million"), else ""
    decimals: int = 0  # decimal places as said, in its scale ("1.5M" has 1)
    said: str = ""  # the words as heard, or the text as typed
    first: int = -1  # the first and last word that said it (indices into the words given)
    last: int = -1
    start: float = 0.0  # when its first word starts and its last word ends
    end: float = 0.0
    kind: str = "amount"  # "amount", or "year" or "time", which a chart never shows
    weak: bool = False  # a small number with no unit: only taken inside a pair
    label: str = ""  # a typed name for the point ("Jan")

    @property
    def family(self) -> str:
        """What amounts on one chart must share: a percentage, a multiple, a currency, or a count."""
        if self.suffix == "%":
            return "%"
        if self.suffix == "x":
            return "x"
        return self.prefix or "#"

    @property
    def marked(self) -> bool:
        """Said or typed with a unit or a scale, so it is plainly an amount."""
        return bool(self.prefix or self.suffix or self.scale)


# ---- what a person types -----------------------------------------------------------------------------

def parse(text: str) -> Amount:
    """One typed amount: "197000", "197K", "$2.5M", "12%", "3.5x", "1,200", "-4". Raises ValueError
    with a plain message when it is not one."""
    raw = text.strip()
    t = raw.replace(" ", "").replace("−", "-")
    m = _TYPED.fullmatch(t)
    if not m or not (m.group("int") or m.group("dec")):
        raise ValueError(f"{raw!r} is not a number; write it like 2000, 2K, $2.5M or 12%")
    whole = (m.group("int") or "0").replace(",", "")
    dec = m.group("dec") or ""
    value = float(f"{whole}.{dec}" if dec else whole)
    if "-" in ((m.group("sign") or "") + (m.group("sign2") or "")):
        value = -value
    scale = ""
    if m.group("scale"):
        mult, scale = SCALE_WORDS.get(m.group("scale").lower(), (0, ""))
        if not scale:  # "m" or "b" on their own
            scale = m.group("scale").upper()[:1]
            mult = SCALE_LETTERS[scale]
        value *= mult
    suf = (m.group("suf") or "").lower()
    suffix = "%" if suf in ("%", "percent") else ("x" if suf == "x" else "")
    return Amount(value=value, prefix=m.group("pre") or "", suffix=suffix, scale=scale, decimals=len(dec),
                  said=raw)


def _split_values(text: str) -> list[str]:
    """A typed list split into its items. Commas followed by a space, semicolons, bars and spaces
    separate items, so "2,000, 15,000" is two; a bare comma does too ("2000,15000"), except before
    a group of three digits that starts with a nought ("2,000,15,000" is 2,000 and 15,000)."""
    t = text.strip().strip("[]()")
    if re.search(r"[;|]|,\s|\s", t):
        parts = re.split(r"\s*[;|]\s*|,\s+|\s+(?=[$£€\d+\-−]|[A-Za-z][\w ]*:)", t)
    else:
        bits = t.split(",")
        parts = []
        for b in bits:
            if parts and re.fullmatch(r"0\d\d(?:\.\d+)?[kKmMbB%x]?", b):
                parts[-1] += "," + b  # a thousands group, not a new item
            else:
                parts.append(b)
    return [p.strip() for p in parts if p.strip()]


def parse_list(text: str) -> list[Amount]:
    """Typed amounts: "2000,15000,197000", "2K, 15K, 197K", "$2K | $197K", "[12%, 3.5%]", or each
    with a name for its point: "Jan:2K, Feb:15K, Mar:197K"."""
    out = []
    for item in _split_values(text):
        label = ""
        if ":" in item:
            label, item = (s.strip() for s in item.split(":", 1))
        a = parse(item)
        out.append(replace(a, label=label))
    if not out:
        raise ValueError("give at least one number, like 2000,15000,197000")
    return out


# ---- what is said ---------------------------------------------------------------------------------

def _bare(word: Word) -> str:
    """The word without the punctuation around it, keeping a leading currency sign and a trailing %."""
    t = word.text.strip().replace("’", "'")
    return _EDGE.sub("", t)


def _word_value(tokens: list[str]) -> Optional[tuple[float, int, bool, int, str]]:
    """Numbers said in words at the start of `tokens` (each one lower case, hyphenated words split):
    (value, tokens used, a year said in two halves, decimals, the scale letter said), or None."""
    total, current, used, seen, decimals = 0.0, 0.0, 0, False, 0
    letter = ""
    last_kind = ""  # "small", "tens", "hundred", "scale"
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if tok in SMALL or tok in TENS:
            small = SMALL.get(tok, TENS.get(tok))
            # "twenty twenty-four", "nineteen ninety": two numbers in a row with no hundred between
            # them is a year said in halves.
            if seen and last_kind in ("small", "tens") and not (last_kind == "tens" and tok in SMALL and SMALL[tok] < 10):
                first = current
                second, k = small, 1
                if tok in TENS and nxt in SMALL and SMALL[nxt] < 10:
                    second, k = small + SMALL[nxt], 2
                if 10 <= first <= 20 and second < 100:
                    return first * 100 + second, i + k, True, 0, ""
                break
            current += small
            seen, last_kind = True, ("tens" if tok in TENS else "small")
        elif tok == "hundred" and (seen or (i > 0 and tokens[i - 1] == "a")):
            current = (current or 1) * 100
            seen, last_kind = True, "hundred"
        elif tok in ("thousand", "million", "billion") and (seen or (i > 0 and tokens[i - 1] == "a")):
            total += (current or 1) * SCALE_WORDS[tok][0]
            current = 0
            letter = letter or SCALE_WORDS[tok][1]
            seen, last_kind = True, "scale"
        elif tok == "a" and not seen and nxt in ("hundred", "thousand", "million", "billion"):
            pass
        elif tok == "and" and seen and (nxt in SMALL or nxt in TENS):
            pass
        elif tok == "point" and seen and nxt in SMALL and SMALL[nxt] < 10:
            # "one point five million": the digits after the point, then any scale word
            frac, j = "", i + 1
            while j < len(tokens) and tokens[j] in SMALL and SMALL[tokens[j]] < 10:
                frac += str(SMALL[tokens[j]])
                j += 1
            current += float("0." + frac)
            decimals = len(frac)
            i = j
            used = j
            last_kind = "small"
            continue
        else:
            break
        i += 1
        if seen:
            used = i
    if not seen:
        return None
    value = total + current
    if letter:  # decimals as written in its scale: "one million two hundred thousand" is 1.2M
        decimals = _exact_places(value / SCALE_LETTERS[letter], 2) or 0
    return value, used, False, decimals, letter


def _tokens_from(words: list[Word], i: int, limit: int = 12) -> tuple[list[str], list[int]]:
    """The keys of the words from `i`, hyphenated words split, and which word each piece came from."""
    toks, owner = [], []
    for k in range(i, min(len(words), i + limit)):
        for piece in normalize(words[k].text).split("-"):
            if piece:
                toks.append(piece)
                owner.append(k)
    return toks, owner


def _read(words: list[Word], i: int) -> Optional[tuple[Amount, int]]:
    """An amount said from word `i`, and the index of the word after it."""
    bare = _bare(words[i])
    m = _TYPED.fullmatch(bare.replace("−", "-"))
    prefix = suffix = scale = ""
    decimals = 0
    digits = False
    comma = False
    if m and (m.group("int") or m.group("dec")) and not m.group("sign"):
        digits = True
        whole = (m.group("int") or "0")
        comma = "," in whole
        dec = m.group("dec") or ""
        value = float(f"{whole.replace(',', '')}.{dec}" if dec else whole.replace(",", ""))
        decimals = len(dec)
        prefix = m.group("pre") or ""
        if m.group("scale"):
            mult, scale = SCALE_WORDS.get(m.group("scale").lower(), (SCALE_LETTERS.get(m.group("scale").upper()[:1], 0),
                                                                     m.group("scale").upper()[:1]))
            value *= mult
        suf = (m.group("suf") or "").lower()
        suffix = "%" if suf in ("%", "percent") else ("x" if suf == "x" else "")
        j = i + 1
        year_halves = False
    else:
        toks, owner = _tokens_from(words, i)
        got = _word_value(toks)
        if got is None:
            return None
        value, used, year_halves, decimals, scale = got
        j = owner[used - 1] + 1
        # "a" before "hundred" counts as the number's first word only when it is followed by one
        if toks[0] == "a" and used <= 1:
            return None
    # what follows the figure: a hundred or a scale ("2 hundred", "1.5 million"), a unit ("15 percent",
    # "5 dollars"), a lone "K" or "x"
    while j < len(words):
        k = normalize(words[j].text)
        if (k == "and" and j + 2 < len(words) and normalize(words[j + 1].text) == "a"
                and normalize(words[j + 2].text) == "half" and not scale and not suffix and float(value).is_integer()):
            value += 0.5  # "three and a half percent", "two and a half million"
            decimals = max(decimals, 1)
            j += 2
        elif digits and k == "hundred" and not scale:
            value *= 100
        elif k in SCALE_WORDS and not scale and not suffix:
            mult, scale = SCALE_WORDS[k]
            value *= mult
        elif k in PERCENT_WORDS or _bare(words[j]) == "%":
            suffix = "%"
        elif k == "per" and j + 1 < len(words) and normalize(words[j + 1].text) == "cent":
            suffix = "%"
            j += 1
        elif k in CURRENCY_WORDS and not prefix:
            prefix = CURRENCY_WORDS[k]
        elif k == "x" and not suffix:
            suffix = "x"
        else:
            break
        j += 1
        if suffix == "%" or prefix:
            # a unit closes the amount ("15 percent", "five million dollars")
            nk = normalize(words[j].text) if j < len(words) else ""
            if not (nk in SCALE_WORDS and not scale):
                break
    first, last = i, j - 1
    said = " ".join(w.text.strip() for w in words[first:last + 1])
    kind = "amount"
    whole_number = float(value).is_integer()
    if year_halves:
        kind = "year"
    elif digits and not comma and not prefix and not suffix and not scale and whole_number and 1900 <= value <= 2099 \
            and len(bare.strip("$£€%")) == 4:
        kind = "year"
    else:
        # a length of time: "six months", "3 more weeks", "two and a half years"
        k = j
        skipped = 0
        while k < len(words) and skipped <= 2:
            key = normalize(words[k].text)
            if key in TIME_WORDS:
                kind = "time"
                break
            if key in TIME_FILLERS:
                skipped += 1
                k += 1
                continue
            break
        if kind == "amount" and i > 0 and normalize(words[i - 1].text) in ("at", "by", "until", "till") \
                and not (prefix or suffix or scale) and value <= 12:
            kind = "time"  # "at five", "by 3": a time of day
    weak = kind == "amount" and not (prefix or suffix or scale) and abs(value) < 10
    return Amount(value=value, prefix=prefix, suffix=suffix, scale=scale, decimals=decimals, said=said,
                  first=first, last=last, start=words[first].start, end=words[last].end, kind=kind,
                  weak=weak), j


_PIECE = re.compile(r"^(?:[,.]\d|%)")  # ",000", ".5", ".5%,", "%": the rest of a figure split off
_LETTER = re.compile(r"^[kKmMbB][.,!?;:]?$")


def glue(words: Sequence[Word]) -> list[tuple[Word, int, int]]:
    """The words with each figure made whole again: (word, first, last) for each.

    The transcriber writes a figure as several words when it times each piece: "2,000" comes as
    "2" and ",000", "12%" as "12" and "%", "2.5" as "2" and ".5", "$5" sometimes as "$" and "5".
    A piece that follows a figure with no gap is joined back to it (and its times with it)."""
    out: list[tuple[Word, int, int]] = []
    for i, w in enumerate(words):
        t = w.text.strip()
        if out:
            prev, f, _ = out[-1]
            pt = prev.text.strip()
            close = w.start - prev.end <= 0.25
            joined = None
            if close and re.search(r"\d$", pt) and (_PIECE.match(t) or _LETTER.match(t)):
                joined = pt + t
            elif close and pt in ("$", "£", "€") and re.match(r"\d", t):
                joined = pt + t
            if joined is not None:
                out[-1] = (Word(joined, prev.start, w.end, min(prev.prob, w.prob)), f, i)
                continue
        out.append((w, i, i))
    return out


def spoken(words: Sequence[Word]) -> list[Amount]:
    """Every amount heard in `words`, in order (years and lengths of time included, marked by `kind`).
    Each amount's `first` and `last` are indices into `words` as given.

    Read the transcriber's own words (a clip's `heard`), not words that have been tidied for
    captions: a lone "%" has no letters and is dropped from those."""
    pieces = glue(words)
    joined = [w for w, _, _ in pieces]
    out, i = [], 0
    while i < len(joined):
        got = _read(joined, i)
        if got is None:
            i += 1
            continue
        a, j = got
        f, l = pieces[a.first][1], pieces[a.last][2]
        out.append(replace(a, first=f, last=l, start=words[f].start, end=words[l].end))
        i = max(j, i + 1)
    return out


# ---- the run a chart shows -------------------------------------------------------------------------

MAX_GAP = 3.5  # seconds between one amount and the next in the same run
MAX_WORDS_BETWEEN = 10
_JOIN = {"to", "then", "and", "now", "today", "->", "into", "reaching", "hit", "at", "over"}
# Words that may follow a clause's end and still carry on the same run of amounts ("15,000, and
# today we are at 197,000"); anything else after a comma or a full stop starts something new
# ("197K, sales are up 19%").
_LINK = {"and", "then", "now", "to", "today", "tonight", "we", "we're", "i", "i'm", "it", "it's", "they", "are",
         "is", "was", "were", "at", "hit", "reached", "up", "down", "from", "by", "over", "nearly", "almost", "about",
         "around", "just", "finally", "eventually", "that", "this", "last", "next", "year", "month", "week", "on",
         "of", "with", "into", "went", "grew", "rose", "jumped", "fell", "dropped", "climbed", "a", "the", "so",
         "but", "only", "even", "more", "than", "less", "compared", "versus", "vs", "still", "already", "all"}
_CURRENCIES = {"$", "£", "€"}


@dataclass
class Series:
    amounts: list[Amount]  # in the order they are drawn: oldest first
    said_order: list[int] = field(default_factory=list)  # their indices in the order they were said
    reversed: bool = False  # said newest first ("1.5M, up from 200K")
    pattern: str = ""  # "from-to" when said as "from X to Y"

    @property
    def start(self) -> float:
        return min(a.start for a in self.amounts)

    @property
    def end(self) -> float:
        return max(a.end for a in self.amounts)


def _between(words: Sequence[Word], a: Amount, b: Amount) -> list[str]:
    return [normalize(w.text) for w in words[a.last + 1:b.first]]


def _before(words: Sequence[Word], a: Amount, n: int = 2) -> list[str]:
    return [normalize(w.text) for w in words[max(0, a.first - n):a.first]]


def _carries_on(words: Sequence[Word], prev: Amount, a: Amount) -> bool:
    """Whether `a` carries on the run `prev` is in: said soon after it, and with no clause ended
    between them that something other than a linking word follows."""
    if a.start - prev.end > MAX_GAP or a.first - prev.last - 1 > MAX_WORDS_BETWEEN:
        return False
    stop = None
    for k in range(prev.last, a.first):
        if words[k].ends_clause:
            stop = k
    if stop is None:
        return True
    return all(normalize(w.text) in _LINK for w in words[stop + 1:a.first])


def _runs(amounts: list[Amount], words: Sequence[Word]) -> list[list[Amount]]:
    """Amounts close together, of one family and in one train of thought, in the order said."""
    runs: list[list[Amount]] = []
    for a in amounts:
        if a.kind != "amount":
            continue  # a year or a length of time is passed over: "from 2K in 2023 to 197K in 2024"
        cur = runs[-1] if runs else None
        if cur and _carries_on(words, cur[-1], a):
            prev = cur[-1]
            linked = any(k in _LINK for k in _between(words, prev, a))
            fam = {x.family for x in cur} | {a.family}
            # "$3 million, up from 500,000": a currency said once holds for a plain amount linked to it
            mixed = fam <= _CURRENCIES | {"#"} and len(fam & _CURRENCIES) == 1 and linked
            if prev.family == a.family or mixed:
                cur.append(a)
                continue
        runs.append([a])
    return [r for r in runs if r]


def _from_to(words: Sequence[Word], run: list[Amount]) -> bool:
    if len(run) < 2:
        return False
    return "from" in _before(words, run[0], 3) and all(
        any(k in _JOIN for k in _between(words, a, b)[-2:]) for a, b in zip(run, run[1:]))


def series(amounts: list[Amount], words: Sequence[Word], near: Optional[float] = None) -> Optional[Series]:
    """The run of amounts a chart would show, or None when nothing said is worth charting.

    A run is amounts of one family (all per cent, all dollars, all counts) said within a few seconds
    of each other. Two or more beat one; "from X to Y" beats a list; a single amount is taken only
    when it is plainly an amount (a unit, a scale, or 100 and up). Small numbers said alone ("two
    things") are taken only as a pair said "from … to …". With `near` (seconds), the run closest to
    that moment is taken. "X, up from Y" is drawn oldest first, Y then X."""
    words = list(words)
    scored = []
    for run in _runs(amounts, words):
        ft = _from_to(words, run)
        if not ft:  # a small number said on its own is only taken inside "from … to …"
            run = [a for a in run if not a.weak]
        plain = any(a.marked or abs(a.value) >= 1000 for a in run)  # plainly an amount, not a count of things
        if len(run) >= 2:
            score = 10.0 + 2.0 * len(run) + (3.0 if ft else 0.0) + (1.0 if plain else 0.0)
        elif len(run) == 1 and (run[0].marked or abs(run[0].value) >= 100):
            score = 4.0 + (1.0 if plain else 0.0)
        else:
            continue
        scored.append((score, run, ft))
    if not scored:
        return None
    if near is not None:
        def distance(item) -> float:
            # the amounts said at or just after `near` ("when=revenue grew" names the words before them)
            return min(abs(a.start - near) + (1.0 if a.start < near - 0.3 else 0.0) for a in item[1])
        scored.sort(key=lambda it: (distance(it), -it[0]))
    else:
        scored.sort(key=lambda it: (-it[0], it[1][0].start))
    score, run, ft = scored[0]
    order = list(range(len(run)))
    rev = False
    if len(run) == 2 and not ft:
        # "1.5 million, up from 200,000": the second is the older one
        if "from" in _before(words, run[1], 2) and "from" not in _before(words, run[0], 3):
            rev = True
    if rev:
        return Series(amounts=[run[1], run[0]], said_order=[1, 0], reversed=True, pattern="up-from")
    return Series(amounts=list(run), said_order=order, pattern="from-to" if ft else "")


# ---- the words that say what the amounts are -----------------------------------------------------

_STOP = {"the", "a", "an", "and", "or", "but", "to", "from", "in", "on", "at", "of", "for", "with", "by", "up",
         "down", "than", "that", "this", "these", "those", "it", "it's", "is", "was", "were", "are", "be", "been",
         "we", "i", "you", "they", "he", "she", "our", "my", "your", "their", "its", "so", "just", "now", "then",
         "over", "under", "about", "around", "nearly", "almost", "only", "per", "each", "every", "more", "less",
         "like", "as", "have", "has", "had", "do", "did", "get", "got", "went", "go", "goes", "grew", "rose",
         "jumped", "fell", "dropped", "climbed", "hit", "reached", "which", "who", "what", "when", "where", "there",
         "here", "very", "really", "all", "some", "no", "not", "if", "into", "out", "after", "before", "since",
         "uh", "um", "okay", "yeah", "right", "new", "total", "already", "still", "even", "also", "too", "plus"}
_GROW = {"went", "grew", "rose", "jumped", "climbed", "fell", "dropped", "increased", "decreased", "doubled",
         "tripled", "went", "go", "goes", "is", "was", "are", "were", "hit", "reached", "from"}


def _not_a_name(key: str) -> bool:
    """A word that cannot be part of what the amounts are called: a small word, a time, a number."""
    return (not key or key in _STOP or key in TIME_WORDS or key in SMALL or key in TENS or key in SCALE_WORDS
            or key in CURRENCY_WORDS or key in PERCENT_WORDS or any(ch.isdigit() for ch in key))


def _name_after(a: Amount, words: list[Word]) -> str:
    """The word or two said straight after an amount ("197K *followers*", "$15,000 *in revenue*")."""
    after = []
    k = a.last + 1
    if k < len(words) and normalize(words[k].text) in ("in", "of") and k + 1 < len(words):
        nk = normalize(words[k + 1].text)
        if nk and not _not_a_name(nk) and not words[k].ends_clause and not words[a.last].ends_clause:
            k += 1
    while k < len(words) and len(after) < 2 and not words[k - 1].ends_clause:
        key = normalize(words[k].text)
        if _not_a_name(key):
            break
        after.append(words[k].clean.strip(",.;:!?"))
        k += 1
    return " ".join(after)


def label_for(run: Series, words: Sequence[Word]) -> str:
    """What the amounts are, in the speaker's own words, when they say it: the word or two after
    an amount (the last said first), or the thing named before the first ("our *revenue* went
    from…"). Empty when nothing plain is said: a label is never made up."""
    words = list(words)
    for a in sorted(run.amounts, key=lambda a: -a.last):
        name = _name_after(a, words)
        if name:
            return name
    first = min(run.amounts, key=lambda a: a.first)
    k = first.first - 1
    while k >= 0 and normalize(words[k].text) in _GROW | {"to", "at", "by", "up", "down"} and first.first - k <= 3:
        k -= 1
    if k == first.first - 1:
        return ""  # only a name joined to the amounts by a verb: "revenue grew from", "sales are up"
    before = []
    while k >= 0 and len(before) < 2:
        key = normalize(words[k].text)
        if _not_a_name(key) or words[k].ends_clause:
            break
        before.insert(0, words[k].clean.strip(",.;:!?"))
        k -= 1
    return " ".join(before)


# ---- writing values ---------------------------------------------------------------------------------

def _exact_places(x: float, most: int) -> Optional[int]:
    """The fewest decimal places (up to `most`) that write `x` exactly, or None."""
    for d in range(most + 1):
        if abs(round(x, d) - x) < 1e-9 * max(1.0, abs(x)):
            return d
    return None


def _group(n: float, decimals: int) -> str:
    s = f"{abs(n):,.{decimals}f}"
    return ("-" if n < 0 else "") + s


@dataclass(frozen=True)
class Unit:
    """How one chart writes its values: the sign before, the scale letter and sign after, and the
    decimal places, chosen once from its amounts so every value on the card reads the same way."""
    prefix: str = ""
    suffix: str = ""  # "%", "x", or the person's own (" users")
    scale: str = ""  # "K", "M", "B" or ""
    decimals: int = 0  # in that scale

    @staticmethod
    def of(amounts: Sequence[Amount], prefix: Optional[str] = None, suffix: Optional[str] = None) -> "Unit":
        """The unit for `amounts`. `prefix` and `suffix` (when not None) are the person's own; a
        suffix of K, M or B is a scale."""
        last = amounts[-1]
        pre = next((a.prefix for a in amounts if a.prefix), "") if prefix is None else prefix
        suf = next((a.suffix for a in amounts if a.suffix), "") if suffix is None else suffix
        scale = ""
        if suf.strip().upper() in SCALE_LETTERS:
            scale, suf = suf.strip().upper(), ""
        elif suf not in ("%", "x") and not (suffix is not None and suf):
            said = [a.scale for a in amounts if a.scale]
            big = max(abs(a.value) for a in amounts)
            if said:
                scale = said[-1]
            elif big >= 1e9 and all(_exact_places(a.value / 1e9, 2) is not None for a in amounts if abs(a.value) >= 1e9):
                scale = "B"
            elif big >= 1e6 and all(_exact_places(a.value / 1e6, 2) is not None for a in amounts if abs(a.value) >= 1e6):
                scale = "M"
            elif big >= 1e4 and all(_exact_places(a.value / 1e3, 1) is not None for a in amounts if abs(a.value) >= 1e3):
                scale = "K"
        mult = SCALE_LETTERS.get(scale, 1.0)
        places = [p for a in amounts if (p := _exact_places(a.value / mult, 2)) is not None] if scale else \
            [a.decimals if not a.scale else (_exact_places(a.value, 2) or 0) for a in amounts]
        decimals = _exact_places(last.value / mult, 2)
        if decimals is None:
            decimals = max(places) if places else 0
        return Unit(prefix=pre, suffix=suf, scale=scale, decimals=decimals)

    @property
    def mult(self) -> float:
        return SCALE_LETTERS.get(self.scale, 1.0)

    def _letter(self, v: float, most: int) -> tuple[str, Optional[int]]:
        """The largest scale, up to the unit's own, that writes `v` exactly with at most `most`
        decimals (a smaller scale with at most one), and those decimals; ("", None) for none."""
        if not self.scale:
            return "", None
        ladder = [k for k in ("B", "M", "K") if SCALE_LETTERS[k] <= self.mult]
        for k in ladder:
            m = SCALE_LETTERS[k]
            if abs(v) >= m:
                d = _exact_places(v / m, most if k == self.scale else 1)
                if d is not None:
                    return k, d
        return "", None

    def _write(self, v: float, letter: str, d: int) -> tuple[str, str, str, str]:
        n = v / SCALE_LETTERS[letter] if letter else v
        sign = "-" if n < 0 and abs(n) >= 0.5 * 10 ** -d else ""
        return sign + self.prefix, _group(abs(n), d), letter, self.suffix

    def exact(self, v: float) -> tuple[str, str, str, str]:
        """A value where it lands, written exactly: in a scale when the short form is exact ("2K",
        "197.5K", "$500K" beside "$3M"), plain when it is not ("500", "197,450"), with the decimals
        it needs and no more ("3.5%")."""
        letter, d = self._letter(v, 2)
        if letter:
            return self._write(v, letter, d)
        d = _exact_places(v, 2)
        return self._write(v, "", d if d is not None else 2)

    def during(self, v: float, toward: float) -> tuple[str, str, str, str]:
        """A value while counting toward `toward`: in the scale and with the decimals that value is
        written with, so the count settles on it without the figures changing shape; below that
        scale, in the next one down, whole."""
        _, digits, letter, _ = self.exact(toward)
        d = len(digits.split(".")[1]) if "." in digits else 0
        if letter and abs(v) >= SCALE_LETTERS[letter]:
            return self._write(v, letter, d)
        if letter:
            for k in ("M", "K"):
                if SCALE_LETTERS[k] < SCALE_LETTERS[letter] and abs(v) >= SCALE_LETTERS[k]:
                    return self._write(v, k, 0)
            return self._write(v, "", 0)
        return self._write(v, "", d)

    @staticmethod
    def join(parts: tuple[str, str, str, str]) -> str:
        return "".join(parts)


def text(unit: Unit, v: float) -> str:
    return Unit.join(unit.exact(v))
