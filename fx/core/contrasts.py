"""Contrasts said aloud: the two sides of a comparison, and the if/then of a rule, found in the words.

    grids = contrasts.find(clip.words, clip.quiet)     # every comparison heard, the fullest first

Three ways people say them, each read the same way, a clause at a time:

- **This versus that.** "Editing by hand versus editing with a tool", "phone compared to camera": the
  words before the marker, back to the start of their clause, and the words after it, up to the end
  of theirs. Each side lands on its own first word. Said once, it makes the two heads of a grid.
- **If, then.** "If your hook is slow, cut the first second", "When it's quiet, then add music", and
  the question form "Want something more editorial? Then just ask for it.": the condition is the
  clause after "if" (or the question), the result the clause after "then" or after the comma. Rules
  said one after another (no more than `RUN_GAP` apart) are rows of one grid.
- **Before, after.** "Before, I edited for hours. Now, it takes minutes.": a clause opened by
  "before," and the next opened by "now," "after," or "today,". Each pair is a row.

What is never a cell: a side with no word that says anything (`moments.NOT_KEY` alone: "this versus
that"), more than `MOST_WORDS` words, a word the speaker took back (`moments.taken_back`), or a word
the transcriber was unsure of (`Clip.words` leaves those out already). An "if" that does not open a
clause ("see if", "as if", "even if" inside a sentence) is not a rule.

Nothing here draws or decides a layout; `comparison-grid` does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from . import moments
from .words import HESITATIONS, Word, pause_after

MOST_WORDS = 6  # the most words a cell holds as said (a cell is a few words, never a sentence)
REACH = 9  # how far a clause is read for its comma or its "then"
RUN_GAP = 25.0  # seconds between two rules of one grid, at most

_VERSUS = {"versus", "vs"}
_COMPARED = ("compared", ("to", "with"))
_CONDITIONS = {"if", "when", "whenever", "once"}
_OPENS = {"and", "so", "but", "because", "now", "okay", "ok", "well", "then", "also", "plus", "or"}
_NOW = {"now", "after", "afterwards", "today", "these"}
# Words a side stops at, reading away from its marker: they belong to the clause around it.
_STOP_LEFT = {"is", "are", "was", "were", "it's", "that's", "and", "but", "so", "because", "or", "of", "between",
              "about", "like", "i", "we", "you", "they", "think", "say", "said", "comes", "come", "choose", "pick"}
_STOP_RIGHT = {"is", "are", "was", "were", "and", "but", "so", "because", "which", "where", "when", "will", "would",
               "can", "could", "for", "in", "on", "at"}
_LEAD_WORDS = {"just", "like", "um", "uh", "so", "and"}


@dataclass
class Cell:
    text: str  # as a cell shows it: the words said, no punctuation at the edges, a capital first
    at: float  # when it lands: the start of its first word
    words: tuple[int, int]  # its first and last word


@dataclass
class Grid:
    kind: str  # "versus", "if-then" or "before-after"
    heads: tuple[str, str]  # the two column heads: the two sides of a versus, or "If" / "Then"
    head_cells: Optional[tuple[Cell, Cell]] = None  # a versus' heads are said, and land on their words
    rows: list[tuple[Cell, Cell]] = field(default_factory=list)
    phrase: str = ""  # what was said, for a person reading the plan

    @property
    def start(self) -> float:
        cells = list(self.head_cells or ()) + [c for r in self.rows for c in r]
        return min(c.at for c in cells)


def _clean(words: Sequence[Word], a: int, b: int) -> str:
    out = " ".join(words[i].clean.strip() for i in range(a, b + 1) if words[i].key not in HESITATIONS)
    out = out.strip().strip(",.;:!?\"“”'’")
    return out[:1].upper() + out[1:]


def _says_something(words: Sequence[Word], a: int, b: int) -> bool:
    return any(words[i].key and words[i].key not in moments.NOT_KEY for i in range(a, b + 1))


def _trim(words: Sequence[Word], a: int, b: int) -> tuple[int, int]:
    """Lead-in words off the front ("the", "just", "um") and a dangling one off the end."""
    while a < b and words[a].key in _LEAD_WORDS:
        a += 1
    while b > a and (words[b].key in moments.LEADS_ON or words[b].key in HESITATIONS):
        b -= 1
    return a, b


def _ok(words, a, b, back) -> bool:
    return (0 <= a <= b and b - a + 1 <= MOST_WORDS and _says_something(words, a, b)
            and not back.holds(list(words), a, b))


def _opens_clause(words: Sequence[Word], i: int, quiet) -> bool:
    if i == 0:
        return True
    prev = words[i - 1]
    return prev.ends_clause or prev.key in _OPENS or (quiet is not None and pause_after(list(words), i - 1, quiet) >= 0.3)


def _clause_end(words: Sequence[Word], a: int, limit: int = REACH) -> int:
    b = a
    while b < len(words) - 1 and b - a < limit and not words[b].ends_clause and words[b + 1].key != "then":
        b += 1
    return b


def _versus(words, quiet, back) -> list[Grid]:
    out = []
    for i, w in enumerate(words):
        k = w.key
        skip = 1
        if k in _VERSUS:
            pass
        elif k == _COMPARED[0] and i + 1 < len(words) and words[i + 1].key in _COMPARED[1]:
            skip = 2
        else:
            continue
        # the left side: back to the start of its clause
        a = i - 1
        while a > 0 and i - a < MOST_WORDS and not words[a - 1].ends_clause and words[a - 1].key not in _STOP_LEFT:
            a -= 1
        if a < 0 or words[i - 1].ends_sentence:
            continue
        la, lb = _trim(words, a, i - 1)
        # the right side: up to the end of its clause
        c = i + skip
        if c >= len(words):
            continue
        d = c
        while d < len(words) - 1 and d - c + 1 < MOST_WORDS and not words[d].ends_clause \
                and words[d + 1].key not in _STOP_RIGHT:
            d += 1
        ra, rb = _trim(words, c, d)
        if not (_ok(words, la, lb, back) and _ok(words, ra, rb, back)):
            continue
        left = Cell(_clean(words, la, lb), words[la].start, (la, lb))
        right = Cell(_clean(words, ra, rb), words[ra].start, (ra, rb))
        out.append(Grid("versus", (left.text, right.text), (left, right),
                        phrase=" ".join(x.text for x in words[la:rb + 1])))
    return out


def _rules(words, quiet, back) -> list[tuple[Cell, Cell, str]]:
    rules = []
    n = len(words)
    i = 0
    while i < n:
        w = words[i]
        cond = None
        if w.key in _CONDITIONS and _opens_clause(words, i, quiet) and i + 1 < n:
            a = i + 1
            b = _clause_end(words, a)
            if b + 1 < n and (words[b].ends_clause and not words[b].ends_sentence or words[b + 1].key == "then"):
                cond = (a, b)
        elif w.text.rstrip().endswith("?") and i + 1 < n and words[i + 1].key == "then":
            # "Want something more editorial? Then just ask for it."
            a = i
            while a > 0 and i - a < REACH and not words[a - 1].ends_sentence:
                a -= 1
            cond = (a, i)
            b = i
        if cond is None:
            i += 1
            continue
        a, b = cond
        r0 = b + 1
        if r0 < n and words[r0].key == "then":
            r0 += 1
        if r0 >= n:
            break
        r1 = _clause_end(words, r0)
        ca, cb = _trim(words, a, b)
        ra, rb = _trim(words, r0, r1)
        if _ok(words, ca, cb, back) and _ok(words, ra, rb, back):
            rules.append((Cell(_clean(words, ca, cb), words[ca].start, (ca, cb)),
                          Cell(_clean(words, ra, rb), words[ra].start, (ra, rb)),
                          " ".join(x.text for x in words[a:r1 + 1])))
            i = r1 + 1
            continue
        i += 1
    return rules


def _then_and_now(words, quiet, back) -> list[tuple[Cell, Cell, str]]:
    pairs = []
    n = len(words)
    for i, w in enumerate(words):
        if w.key != "before" or not w.ends_clause or w.ends_sentence or not _opens_clause(words, i, quiet):
            continue
        a = i + 1
        b = _clause_end(words, a)
        for j in range(b + 1, min(n, b + 40)):
            if words[j].key in _NOW and words[j].ends_clause and not words[j].ends_sentence \
                    and _opens_clause(words, j, quiet) and words[j].start - words[i].start <= RUN_GAP:
                c = j + 1
                if c >= n:
                    break
                d = _clause_end(words, c)
                ba, bb = _trim(words, a, b)
                na, nb = _trim(words, c, d)
                if _ok(words, ba, bb, back) and _ok(words, na, nb, back):
                    pairs.append((Cell(_clean(words, ba, bb), words[ba].start, (ba, bb)),
                                  Cell(_clean(words, na, nb), words[na].start, (na, nb)),
                                  " ".join(x.text for x in words[i:d + 1])))
                break
    return pairs


def _runs(rows: list[tuple[Cell, Cell, str]]) -> list[list[tuple[Cell, Cell, str]]]:
    out: list = []
    for r in rows:
        if out and r[0].at - out[-1][-1][1].at <= RUN_GAP:
            out[-1].append(r)
        else:
            out.append([r])
    return out


def find(words: Sequence[Word], quiet=None, back: Optional[moments.TakenBack] = None) -> list[Grid]:
    """Every comparison heard in `words` (a clip's whole words), as grids: a versus said with its two
    sides, a run of if/then rules, a run of before/after pairs. The fullest grid first, then the
    earliest."""
    words = list(words)
    back = moments.taken_back(words, quiet or []) if back is None else back
    grids = _versus(words, quiet, back)
    for run in _runs(_rules(words, quiet, back)):
        grids.append(Grid("if-then", ("If", "Then"), None, [(c, r) for c, r, _ in run],
                          phrase=" … ".join(p for _, _, p in run)))
    for run in _runs(_then_and_now(words, quiet, back)):
        grids.append(Grid("before-after", ("Before", "Now"), None, [(c, r) for c, r, _ in run],
                          phrase=" … ".join(p for _, _, p in run)))
    return sorted(grids, key=lambda g: (-(len(g.rows) + (1 if g.head_cells else 0)), g.start))
