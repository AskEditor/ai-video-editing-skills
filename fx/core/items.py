"""Items: the things an effect shows one by one (cards, rows, steps), given or found in the words,
and the moment each one lands.

Two ways to get them, one way to time them:

- **Given** (`items=`): a `.json` file, JSON text, or an effect's own short form. `load()` reads the
  JSON, `entry()` reads one item (a name, `name@4.2`, a picture, or `{"text", "at", "image"}`).
- **Heard**: an effect's grammar finds them in the words (`tier-list`'s rankings, `in_order()`'s
  lists said in order) and returns each as a `Heard`: its words, the word that makes it known, when.
- **Timed** (`time_items`): an item given a time lands on the word that starts nearest it; one given
  none lands where it was heard, else where its name is said (`name_spans`, which finds a name the
  transcriber split or joined), else between the others, and the run says so.

Words are always `clip.words`: corrected for the transcriber's lateness, checked against the sound,
and without the words a second listen could not be sure of. `tokens()` and `card_text()` are the
pieces every grammar shares: the words as keys to match on, and a span of them as a card shows it.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from . import moments
from .words import Word, normalize


class ItemsError(ValueError):
    """Items that cannot be used; the message says what to change."""


# ---- reading the items given -----------------------------------------------------------------------

def load(spec: str, clip_dir: Path) -> tuple[object, Optional[Path]]:
    """The items as data when they are JSON: from a `.json` file (looked for as given, then beside
    the clip) or JSON text. Returns (data, the file's folder or None), or (None, None) when `spec`
    is empty or is an effect's own short form, which the effect reads itself."""
    s = spec.strip()
    if not s:
        return None, None
    if s.lower().endswith(".json") and not s.startswith(("{", "[")):
        path = None
        for candidate in (Path(s).expanduser(), clip_dir / s):
            if candidate.exists():
                path = candidate.resolve()
                break
        if path is None:
            raise ItemsError(f"there is no items file at {s}")
        try:
            return json.loads(path.read_text()), path.parent
        except ValueError as e:
            raise ItemsError(f"{path.name} is not valid JSON: {e}") from None
    if s.startswith(("{", "[")):
        try:
            return json.loads(s), None
        except ValueError as e:
            raise ItemsError(f"the items are not valid JSON: {e}") from None
    return None, None


def entry(raw, base: Optional[Path], clip_dir: Path) -> tuple[str, Optional[float], Optional[str]]:
    """One item: (words, time or None, picture or None). A name may carry its time (`Latte@4.2`);
    a name that is a picture's file name is a picture card."""
    if isinstance(raw, str):
        m = re.fullmatch(r"\s*(.*?)\s*@\s*([0-9]+(?:\.[0-9]+)?)\s*s?\s*", raw)
        name, at = (m.group(1), float(m.group(2))) if m else (raw.strip(), None)
        if re.search(r"\.(png|jpe?g|webp|gif|bmp)$", name, re.I):
            return "", at, picture(name, base, clip_dir)
        return name, at, None
    if isinstance(raw, dict):
        name = str(raw.get("text") or raw.get("name") or "").strip()
        at = raw.get("at", raw.get("time"))
        if at is not None:
            try:
                at = float(at)
            except (TypeError, ValueError):
                raise ItemsError(f"the time for {name!r} is not a number of seconds (got {at!r})") from None
        pic = raw.get("image") or raw.get("picture")
        return name, at, (picture(str(pic), base, clip_dir) if pic else None)
    raise ItemsError(f"an item must be a name or an object with text, image and at (got {raw!r})")


def picture(name: str, base: Optional[Path], clip_dir: Path) -> str:
    """The full path of a picture named in the items: as given, beside the items file, or beside the clip."""
    tried = [Path(name).expanduser()] + ([base / name] if base else []) + [clip_dir / name]
    for p in tried:
        if p.exists():
            from PIL import Image

            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception:
                raise ItemsError(f"{p} is not a picture that can be read") from None
            return str(p.resolve())
    raise ItemsError(f"there is no picture at {name}")


# ---- the words as a grammar reads them --------------------------------------------------------------

@dataclass
class Tok:
    key: str
    w: int  # the word it is part of
    stop: str = ""  # "." or "," when its word ends a sentence or a clause (on the word's last part)
    colon: bool = False  # its word ends with ":"
    cap: bool = False  # its word starts with a capital letter


def tokens(words: list[Word], glued: set[str] = frozenset()) -> list[Tok]:
    """The words as keys to match on: lower case, a hyphenated or slashed word split in its parts,
    "espresso's" as "espresso" and "'s", and a key in `glued` stuck to "tier" ("stier") split off."""
    out = []
    for i, w in enumerate(words):
        raw = w.text.strip()
        k = normalize(raw)
        if not k:
            continue
        tail = raw.rstrip()
        stop = "." if tail.endswith((".", "?", "!")) else ("," if tail.endswith((",", ";", ":", "—", "…")) else "")
        parts = [p for p in re.split(r"[-/]", k) if p]
        expanded: list[str] = []
        for p in parts:
            m = re.fullmatch(r"(.+)'s", p)
            if m:
                expanded += [m.group(1), "'s"]
            elif len(p) > 4 and p.endswith("tier") and p[:-4] in glued:  # "stier"
                expanded += [p[:-4], "tier"]
            else:
                expanded.append(p)
        for n, p in enumerate(expanded):
            last = n == len(expanded) - 1
            out.append(Tok(p, i, stop if last else "", tail.endswith(":") and last, raw[:1].isupper()))
    return out


def card_text(words: list[Word], toks: list[Tok], a: int, b: int) -> str:
    """The words of tokens [a, b) as a card shows them: no punctuation at the edges, no "'s" carried
    over from "espresso's", and a capital letter when the transcriber wrote none."""
    w0, w1 = toks[a].w, toks[b - 1].w
    out = " ".join(words[i].clean.strip() for i in range(w0, w1 + 1)).strip().strip(",.;:!?\"“”")
    if b < len(toks) and toks[b].key == "'s" and toks[b].w == w1:
        out = re.sub(r"['’]s$", "", out)
    out = out.strip().strip(",.;:!?\"“”'’")
    if out and out == out.lower():
        out = " ".join(p[:1].upper() + p[1:] for p in out.split())
    return out


@dataclass
class Heard:
    """An item a grammar found in the words."""
    text: str
    group: int  # the effect's own grouping: a tier-list row, the number said in a list (0 when none)
    at: float  # when it is known: the start of the word that makes it known
    words: tuple[int, int]  # the item's own first and last word
    anchor: int  # the word that makes it known (a ranking's row; a list item's first word)
    phrase: str  # what was said, for a person reading the plan


# ---- where a name is said, and when each item lands -------------------------------------------------

def squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def similar(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if a in (b + "s", b + "es") or b in (a + "s", a + "es"):
        return 0.96
    if min(len(a), len(b)) < 4:
        return 0.0
    return difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio()


def name_spans(words: list[Word], name: str) -> list[tuple[int, int, float]]:
    """Where `name` is said: (first word, last word, how close), closest first, then earliest. A
    name split by the transcriber ("Chat GPT") or joined ("Google Docs" as "GoogleDocs") is found."""
    target = squash(name)
    if not target:
        return []
    out = []
    for i in range(len(words)):
        acc = ""
        for j in range(i, min(len(words), i + 6)):
            acc += squash(words[j].text)
            if not acc:
                continue
            s = similar(acc, target)
            if s >= 0.84:
                out.append((i, j, s))
            if len(acc) >= len(target) + 3:
                break
    return sorted(out, key=lambda c: (-round(c[2], 2), words[c[0]].start))


@dataclass
class Timed:
    """When one item lands, and why then."""
    n: int  # the item's place in the list given
    at: float
    how: str  # "given" (a time was given), "said" (its name is said), "placed" (never said), or the
    # grammar's own word for an item heard (`time_items(heard_how=…)`)
    word: str  # the word it lands on, as heard
    asked: Optional[float] = None  # the time it was given, before it was moved onto a word
    heard: Optional[Heard] = None


def snap(words: list[Word], t: float, within: float = 0.25) -> tuple[float, str]:
    """`t` moved onto the start of the word that starts nearest it, when one starts within `within`
    seconds; else `t` itself. Returns (time, that word or "")."""
    if not words:
        return t, ""
    k = min(range(len(words)), key=lambda i: abs(words[i].start - t))
    if abs(words[k].start - t) <= within:
        return words[k].start, words[k].clean
    return t, ""


def time_items(items: Sequence, words: list[Word], heard: list[Heard], duration: float,
               heard_how: str = "heard", mismatch: Optional[Callable[[object, Heard], Optional[str]]] = None,
               apart: float = 0.2, back: Optional[moments.TakenBack] = None) -> tuple[list[Timed], list[str]]:
    """Each item's moment. `items` are anything with `text`, `at` and `image`. Given a time: the word
    starting nearest it (within a quarter second). Given none: where it was heard (its `Heard`, matched
    by name), else its name where it is said, else between the others in the order given. Returns
    them in time order, two never within `apart` seconds of each other, and notes for a person.
    `mismatch(item, heard)` may return a note when an item is heard somewhere other than it was given.
    A name said where the speaker took it back (`back`, the shared picker's `moments.taken_back` of
    the clip's whole words: a word replaced, a first attempt said again, an aside) never lands
    there: on its saying that was kept, or, said nowhere else, between the others, and a note says
    so. A time given is kept as given."""
    back = moments.TakenBack() if back is None else back

    def taken(a: int, b: int) -> bool:
        return back.holds(words, a, b)

    notes: list[str] = []
    out: list[Optional[Timed]] = [None] * len(items)
    used: set[int] = set()
    said_back: set[int] = set()  # the items whose name is said only where the speaker took it back
    for n, it in enumerate(items):
        if it.at is not None:
            if not 0 <= it.at <= duration:
                raise ItemsError(f"{it.text or 'a picture'} is given {it.at:g} s, outside the clip's {duration:.2f} s")
            at, word = snap(words, it.at)
            out[n] = Timed(n, at, "given", word, asked=it.at)
    # Names said in the clip, the longest first so "Town car" is found before "Town".
    order = sorted((n for n, it in enumerate(items) if it.at is None and it.text), key=lambda n: -len(squash(items[n].text)))
    for n in order:
        it = items[n]
        found = [f for f in heard if similar(squash(f.text), squash(it.text)) >= 0.84
                 and not any(i in used for i in range(f.words[0], f.words[1] + 1))]
        kept = [g for g in found if not taken(g.words[0], g.words[1])]
        if found and not kept:
            said_back.add(n)
        if kept:
            f = kept[-1]
            used.update(range(f.words[0], f.words[1] + 1))
            out[n] = Timed(n, f.at, heard_how, words[f.anchor].clean, heard=f)
            note = mismatch(it, f) if mismatch is not None else None
            if note:
                notes.append(note)
            continue
        for i, j, _ in name_spans(words, it.text):
            if any(k in used for k in range(i, j + 1)):
                continue
            if taken(i, j):
                said_back.add(n)
                continue
            used.update(range(i, j + 1))
            out[n] = Timed(n, words[i].start, "said", " ".join(words[k].clean for k in range(i, j + 1)))
            break
    # Never said, or a picture with no words: between the items before and after it in the order
    # given, on a word, and the run says so.
    for n, it in enumerate(items):
        if out[n] is not None:
            continue
        before = [c.at for c in out[:n] if c is not None]
        after = [c.at for c in out[n + 1:] if c is not None]
        lo = max(before) if before else (words[0].start if words else 0.0)
        hi = min([a for a in after if a > lo], default=(words[-1].start if words else duration))
        target = (lo + hi) / 2 if hi > lo else min(duration - 0.5, lo + 0.8)
        at, word = snap([w for w in words if w not in back], target)
        out[n] = Timed(n, at, "placed", word)
        what = it.text or Path(it.image or "").name
        said = "is said only where you took it back" if n in said_back else "is never said"
        notes.append(f"{what!r} {said}, so it drops in at {at:.2f} s; give it a time with {what}@seconds")
    timed = sorted((c for c in out if c is not None), key=lambda c: c.at)
    for k in range(1, len(timed)):
        if timed[k].at - timed[k - 1].at < apart:
            timed[k].at = timed[k - 1].at + apart
    return timed, notes


# ---- lists said in order ------------------------------------------------------------------------------
#
# A list is said with markers: "number five", "step two", "tip number three", "first", "the second
# thing", "lastly", "and finally", a bare "Five, Canva.", or "then" and "next" once a list is under
# way. "Then" alone starts a list only as a chain ("drop your clip in, then type, then pick"): the
# clause before the first "then" is the first item. A run of items after a colon or with commas and
# a last "and" ("three apps: Notion, Figma and Canva") is a list too. Each item is the clause after
# its marker, without the words that only lead into it ("is going to be", "you want", "we have").
# What is never a list: a "then" after "if" or "when", "the first time", "a second", "a number of",
# a count with nothing after its numbers ("one, two, three, test").

CARDINALS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
             "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
             "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20}
ORDINALS = {"first": 1, "firstly": 1, "1st": 1, "second": 2, "secondly": 2, "2nd": 2, "third": 3, "thirdly": 3, "3rd": 3,
            "fourth": 4, "fourthly": 4, "4th": 4, "fifth": 5, "fifthly": 5, "5th": 5, "sixth": 6, "6th": 6,
            "seventh": 7, "7th": 7, "eighth": 8, "8th": 8, "ninth": 9, "9th": 9, "tenth": 10, "10th": 10}
LAST = {"lastly", "finally"}
# What a numbered item is called: "step two", "tip number three", "the first thing".
LIST_NOUNS = {"step", "tip", "rule", "point", "reason", "thing", "lesson", "mistake", "hack", "tool", "idea", "secret",
              "habit", "way", "part", "pick", "app", "option", "item", "question", "trick", "sign", "spot", "place",
              "one", "stop", "day", "phase", "stage", "move", "fact", "myth", "book", "product", "feature"}
_DETERMINERS = {"the", "my", "our", "your", "their", "his", "her", "a", "an", "this"}
# Words that start a clause without saying anything yet ("and then", "so first", "okay, number one").
_LEADERS = {"and", "so", "okay", "ok", "um", "uh", "erm", "well", "now", "but", "alright", "right", "at", "coming",
            "in", "plus", "also", "oh"}
_SUBJECTS = {"i", "we", "you", "they", "he", "she"}
# Words that only lead into an item: "number one is going to be X", "first, you want X", "we have X".
_FILLERS = {"um", "uh", "erm", "so", "and", "okay", "ok", "well", "now", "like", "just", "simply", "basically",
            "actually", "eventually", "finally", "then", "really", "also", "definitely", "obviously", "of", "course",
            "oh"}
_PRONOUNS = {"i", "we", "you", "they", "it", "he", "she"}
_AUX = {"is", "are", "was", "were", "'s", "would", "will", "be", "going", "to", "gonna", "'ll", "'d", "can", "could",
        "should", "must", "might", "shall"}
_HAVE = {"we've", "i've", "you've", "they've", "have", "has"}
_SAY_THERE = {"there", "here", "that", "this"}  # "there's", "that's": with the "'s" after them
# An item whose first word, once the lead-in is gone, is one of these said nothing ("and then I was like").
_EMPTY_START = {"there", "here", "it", "that", "this", "like", "so", "okay", "ok", "kind", "sort", "yeah", "yes", "no",
                "um", "uh", "something", "anything", "stuff", "things", "one", "two", "three", "four", "five", "six",
                "seven", "eight", "nine", "ten", "again", "too"}
# What ends an item's clause once it has two words: "I went to the station and for a while I waited".
_CLAUSE_ENDS = {"and", "but", "so", "because", "while", "which", "when", "where", "until", "once", "though",
                "although", "whereas", "cause", "'cause", "before", "after", "if"}
# A clause never ends on these: "went to, of course, my hotel" runs on past the aside.
_OPEN_ENDED = {"to", "of", "at", "in", "on", "for", "with", "from", "into", "onto", "the", "a", "an", "my", "your", "our",
               "their", "his", "her", "its", "and", "or", "by", "about", "is", "are", "was"}
_ASIDES = {"of", "course", "um", "uh", "erm", "like", "you", "know", "basically", "actually", "i", "mean"}
_CONDITIONS = {"if", "when", "whenever", "once", "unless"}
_NOT_NEXT = {"week", "time", "year", "month", "day", "video", "episode", "morning", "night", "weekend", "one"}
MAX_ITEM_WORDS = 8
RUN_GAP = 25.0  # seconds between two items of one list, at most
CHAIN_STEP = 8.0  # a chain of "then"s with no list opened before it is said briskly: this far apart at most
CHAIN_LEAD = 6.0  # the clause before a chain's first "then" counts when it started this close before it
_SUBORDINATE = {"but", "because", "so", "which", "when", "while", "if", "although", "though", "since", "as",
                "where", "unless", "until", "whereas", "cause"}


@dataclass
class Step(Heard):
    """One item of a list said in order (a `Heard`: `group` is the number said, 0 when none)."""
    marker: str = ""  # the words that mark it, as said ("Number five", "then"); "" for a chain's first
    marker_at: float = 0.0  # when its marker starts: where the next item starts, for a checklist's tick
    ends: float = 0.0  # when its own words end
    kind: str = ""  # number, step, ordinal, last, count, then, lead or series


@dataclass
class _Mark:
    kind: str
    number: int
    a: int  # its first token
    b: int  # the token after it
    loose: bool = False  # "X and then Y" mid-sentence: an item only inside a list already under way


def _cardinal(key: str) -> Optional[int]:
    if key in CARDINALS:
        return CARDINALS[key]
    if re.fullmatch(r"[0-9]{1,2}", key):
        return int(key)
    return None


def _clause_start(toks: list[Tok], j: int) -> bool:
    """Token j begins a clause: it is the first, or only leaders ("and", "so", "okay") stand between
    it and the end of the last clause."""
    k = j - 1
    while k >= 0:
        if toks[k].stop:
            return True
        if toks[k].key not in _LEADERS:
            return False
        k -= 1
    return True


def _after_subject(toks: list[Tok], j: int) -> bool:
    """Token j follows a subject that begins its clause: "I then boarded", "we finally got"."""
    return j >= 1 and toks[j - 1].key in _SUBJECTS and not toks[j - 1].stop and _clause_start(toks, j - 1)


def _conditional(toks: list[Tok], j: int) -> bool:
    """"then" in "if you want X, then ask": a condition said earlier in the same sentence."""
    k = j - 1
    while k >= 0 and toks[k].stop != ".":
        if toks[k].key in _CONDITIONS:
            return True
        k -= 1
    return False


def _mark_at(toks: list[Tok], j: int) -> Optional[_Mark]:
    t = toks[j]
    nxt = toks[j + 1] if j + 1 < len(toks) else None
    prev = toks[j - 1] if j >= 1 else None
    # "number five", "at number three", "No. 2"
    if t.key in ("number", "no", "num") and nxt is not None and not t.stop:
        n = _cardinal(nxt.key)
        if n is not None and not (prev is not None and prev.key == "a" and not prev.stop):
            after = toks[j + 2] if j + 2 < len(toks) else None
            if not (after is not None and after.key == "of" and not nxt.stop):
                return _Mark("number", n, j, j + 2)
    # "step two", "tip number three", "reason 4"
    if t.key in LIST_NOUNS and t.key != "one" and nxt is not None and not t.stop:
        k = j + 1
        if nxt.key == "number" and not nxt.stop and k + 1 < len(toks):
            k += 1
        n = _cardinal(toks[k].key)
        if n is not None and (prev is None or prev.key not in ("a", "an") or prev.stop):
            return _Mark("step", n, j, k + 1)
    # "first", "firstly", "first of all", "the second thing", "my first pick", "lastly", "and finally"
    if t.key in ORDINALS or t.key in LAST or t.key in ("last", "final", "next"):
        number = ORDINALS.get(t.key, 0)
        kind = "ordinal" if t.key in ORDINALS else ("then" if t.key == "next" else "last")
        a = j
        if prev is not None and prev.key in _DETERMINERS and not prev.stop:
            # "the first thing", "my first pick", "the last step": only with the name of an item after it
            if not (nxt is not None and nxt.key in LIST_NOUNS and not t.stop) or prev.key in ("a", "an"):
                return None
            if not _clause_start(toks, j - 1) and not (j >= 2 and toks[j - 2].key in ("is", "was", "'s")):
                return None
            return _Mark(kind, number, j - 1, j + 2)
        if t.key == "next" and nxt is not None and nxt.key in _NOT_NEXT:
            return None
        if t.key == "last" and not (nxt is not None and nxt.key in ("but", "one", "thing", "step", "up")
                                    or t.stop):
            return None
        if not _clause_start(toks, j) and not (t.key in ("finally", "lastly") and _after_subject(toks, j)):
            return None
        b = j + 1
        if t.key in ("first", "last") and b + 1 < len(toks) and toks[b].key in ("of", "but") and not t.stop:
            b += 2 if toks[b].key == "of" else 3  # "first of all", "last but not least"
        elif b < len(toks) and toks[b].key in ("off", "up", "thing", "things", "step", "one") and not t.stop:
            b += 1  # "first off", "next up", "first thing", "last one"
        return _Mark(kind, number, a, b)
    # "then", "and then", "after that", "afterwards", "from there", "I then boarded", "I eventually went"
    if t.key in ("then", "afterwards", "eventually") or (t.key == "after" and nxt is not None and nxt.key == "that"):
        if t.key == "then" and (_conditional(toks, j) or (nxt is not None and nxt.key == "again")):
            return None
        if _clause_start(toks, j) or _after_subject(toks, j):
            return _Mark("then", 0, j, j + (2 if t.key == "after" else 1))
        if t.key == "then" and prev is not None and prev.key == "and" and not prev.stop:
            return _Mark("then", 0, j - 1, j + 1, loose=True)
        return None
    if t.key == "from" and nxt is not None and nxt.key == "there" and _clause_start(toks, j):
        return _Mark("then", 0, j, j + 2)
    # A bare count at the start of a clause, with a stop after it: "Five, Canva."
    n = _cardinal(t.key)
    if n is not None and t.stop and _clause_start(toks, j) and nxt is not None and _cardinal(nxt.key) is None:
        return _Mark("count", n, j, j + 1)
    return None


def _lead_in(toks: list[Tok], b: int, limit: int) -> int:
    """The first token of the item itself, from token b: fillers, a subject and its helping verbs,
    "you want", "we've got", "there's", "is going to be" and an article are skipped."""
    a = b
    subject = False
    while a < limit:
        t = toks[a]
        nxt = toks[a + 1].key if a + 1 < limit else ""
        if t.key in _FILLERS and not (t.key == "of" and nxt != "course") and not (t.key == "course" and a == b):
            pass
        elif t.key in _PRONOUNS and not subject:
            subject = True
        elif t.key in _AUX:
            pass
        elif t.key in ("want", "need") and subject:
            pass
        elif t.key in _HAVE and (subject or a == b or toks[a - 1].key in _PRONOUNS):
            subject = True
        elif t.key == "got" and a > b and toks[a - 1].key in _HAVE:
            pass
        elif t.key in _SAY_THERE and nxt == "'s":
            pass
        elif t.key in ("comes", "goes", "the", "a", "an"):
            pass
        else:
            break
        if t.stop == "." and a + 1 < limit:
            return limit  # the sentence ended before anything was named
        a += 1
    return a


def _item_span(toks: list[Tok], b: int, limit: int) -> Optional[tuple[int, int]]:
    """The item said from token b (before token `limit`): the lead-in skipped, to the end of its clause."""
    a = _lead_in(toks, b, limit)
    if a >= limit or toks[a].key in _EMPTY_START:
        return None
    e = a
    while e < limit:
        t = toks[e]
        n_words = toks[e].w - toks[a].w + 1
        if e > a and t.key in _CLAUSE_ENDS and n_words >= 3:
            break
        if n_words > MAX_ITEM_WORDS:
            break
        e += 1
        if t.stop:
            if t.key in _OPEN_ENDED and t.stop != "." and e < limit:
                # "went to, of course, my hotel": past the aside to the words it was leading to
                k = e
                while k < limit and toks[k].key in _ASIDES:
                    k += 1
                if k < limit and k > e - 1:
                    e = k
                    continue
            break
    # a clause cut short (by "and", by its length, by the next marker) never ends on a word that
    # leads somewhere; one that ends where the speaker stopped keeps its last word ("care about")
    if not toks[e - 1].stop:
        while e - 1 > a and toks[e - 1].key in _OPEN_ENDED:
            e -= 1
    return (a, e) if e > a else None


def _words_of(words: list[Word], toks: list[Tok], a: int, b: int) -> str:
    """Tokens [a, b) as an item shows them, with any aside the clause ran past left out."""
    keep = []
    k = a
    while k < b:
        if toks[k].key in ("of",) and k + 1 < b and toks[k + 1].key == "course":
            k += 2
            continue
        if toks[k].key in ("um", "uh", "erm"):
            k += 1
            continue
        keep.append(toks[k].w)
        k += 1
    seen, idx = set(), []
    for w in keep:
        if w not in seen:
            seen.add(w)
            idx.append(w)
    out = " ".join(words[i].clean.strip().strip(",.;:!?\"“”") for i in idx).strip()
    out = re.sub(r"\s+", " ", out)
    return out[:1].upper() + out[1:] if out else out


def _step(words, toks, mark: Optional[_Mark], span: tuple[int, int], kind: str, number: int) -> Step:
    a, b = span
    w0, w1 = toks[a].w, toks[b - 1].w
    m_at = words[toks[mark.a].w].start if mark is not None else words[w0].start
    m_txt = " ".join(words[i].clean for i in range(toks[mark.a].w, toks[mark.b - 1].w + 1)) if mark is not None else ""
    lo = toks[mark.a].w if mark is not None else w0
    phrase = " ".join(words[i].text.strip() for i in range(lo, w1 + 1))
    return Step(_words_of(words, toks, a, b), number, words[w0].start, (w0, w1), w0, phrase,
                marker=m_txt.strip(",.;: "), marker_at=m_at, ends=words[w1].end, kind=kind)


def _sentence_before(toks: list[Tok], j: int) -> int:
    """The first token of the sentence that token j sits in."""
    k = j - 1
    while k >= 0 and toks[k].stop != ".":
        k -= 1
    return k + 1


def _series(words: list[Word], toks: list[Tok]) -> list[list[Step]]:
    """Items run together in one sentence: after a colon ("three apps: Notion, Figma and Canva"), or
    three or more parts with commas and a last "and" or "or" ("picks your best line, cuts you out,
    and pops you right out")."""
    out = []
    j = 0
    while j < len(toks):
        s = j
        while j < len(toks) and toks[j].stop != ".":
            j += 1
        e = min(len(toks), j + 1)  # the sentence is tokens [s, e)
        j = e
        colon = next((k for k in range(s, e) if toks[k].colon), None)
        start = colon + 1 if colon is not None else s
        parts, cur = [], []
        for k in range(start, e):
            t = toks[k]
            if t.key in ("and", "or") and cur and k + 1 < e:
                parts.append(cur)
                cur = []
                continue
            cur.append(k)
            if t.stop == ",":
                parts.append(cur)
                cur = []
        if cur:
            parts.append(cur)
        parts = [p for p in parts if p]
        commas = sum(1 for k in range(start, e - 1) if toks[k].stop == ",")
        if colon is None and commas < 1:
            continue
        joined = colon is not None or any(toks[k].key in ("and", "or") and k > start for k in range(start, e)
                                          if toks[k - 1].stop == "," or (k + 1 < e and parts and k < parts[-1][0]))
        spans = []
        for p in parts:
            a, b = p[0], p[-1] + 1
            while a < b and toks[a].key in ("for", "example", "or", "and", "such", "as", "including"):
                a += 1
            if a < b and toks[a].key in _SUBORDINATE:
                spans = []
                break
            a = _lead_in(toks, a, b)
            if a >= b:
                if spans:  # an empty part after the first: not a run of items
                    spans = []
                    break
                continue  # an introduction ("for example,", "oh,")
            while b - 1 > a and toks[b - 1].key in _OPEN_ENDED:
                b -= 1
            if toks[b - 1].w - toks[a].w + 1 > 7 or (b - a == 1 and _cardinal(toks[a].key) is not None):
                spans = []
                break
            spans.append((a, b))
        last_joined = len(parts) >= 2 and any(toks[k].key in ("and", "or") for k in range(parts[-2][-1] + 1, parts[-1][0] + 1))
        if not (colon is not None or last_joined):
            continue
        if len(spans) >= 3 or (colon is not None and len(spans) >= 2):
            out.append([_step(words, toks, None, sp, "series", 0) for sp in spans])
    return out


def lists(words: list[Word]) -> list[list[Step]]:
    """Every list said in the words, in the order said: each two items or more."""
    toks = tokens(words)
    marks = []
    j = 0
    while j < len(toks):
        m = _mark_at(toks, j)
        if m is not None:
            marks.append(m)
            j = m.b
        else:
            j += 1
    starts = [m.a for m in marks]
    runs: list[list[Step]] = []
    cur: list[Step] = []
    chain: list[tuple[_Mark, Step]] = []  # "then"s heard before any list started

    def close() -> None:
        nonlocal cur
        kept = _dedupe(cur)
        if len(kept) >= 2:
            runs.append(kept)
        cur = []

    for i, m in enumerate(marks):
        limit = starts[i + 1] if i + 1 < len(marks) else len(toks)
        span = _item_span(toks, m.b, limit)
        if span is None:
            continue
        step = _step(words, toks, m, span, m.kind, m.number)
        strong = m.kind in ("number", "step", "ordinal", "count", "last")
        if cur and step.marker_at - cur[-1].marker_at > RUN_GAP:
            close()
        if m.loose and not cur:
            continue
        if strong:
            numbers = [s.group for s in cur if s.group]
            restart = False
            if numbers and step.group:
                last = numbers[-1]
                way = (numbers[-1] - numbers[-2]) if len(numbers) >= 2 else 0
                carries = (step.group - last) * way > 0 if way else abs(step.group - last) == 1
                restart = not carries
            if restart or (m.kind == "last" and not cur and not chain):
                close()
                if m.kind == "last":
                    continue
            if not cur and chain:
                chain = []
            cur.append(step)
            continue
        # a "then": part of the list under way, or a chain
        if cur:
            cur.append(step)
            continue
        if chain and step.marker_at - chain[-1][1].marker_at > CHAIN_STEP:
            chain = []
        chain.append((m, step))
        if len(chain) >= 2:
            first_mark = chain[0][0]
            s0 = _sentence_before(toks, first_mark.a)
            lead = _item_span(toks, s0, first_mark.a)
            if lead is not None and chain[0][1].marker_at - words[toks[lead[0]].w].start <= CHAIN_LEAD:
                cur = [_step(words, toks, None, lead, "lead", 0)] + [s for _, s in chain]
            else:
                cur = [s for _, s in chain]
            chain = []
    close()
    marked = runs
    series = _series(words, toks)
    taken = {w for r in marked for s in r for w in range(s.words[0], s.words[1] + 1)}
    for r in series:
        if not any(w in taken for s in r for w in range(s.words[0], s.words[1] + 1)):
            marked.append(r)
    return sorted(marked, key=lambda r: r[0].marker_at)


def _dedupe(run: list[Step]) -> list[Step]:
    """A retake keeps the item said last: "Then just say…", then "Then just say the look you want"."""
    out: list[Step] = []
    for s in run:
        if out:
            a, b = squash(out[-1].text), squash(s.text)
            if a and b and (b.startswith(a) or a.startswith(b)) and s.group == out[-1].group:
                out[-1] = s if len(b) >= len(a) else out[-1]
                continue
        out.append(s)
    return out


def in_order(words: list[Word]) -> list[Step]:
    """The list said in the words: the longest run of items (marked ones before a run of commas), the
    later on a tie, since a retake is usually the better take. [] when none is said."""
    runs = lists(words)
    if not runs:
        return []
    return max(runs, key=lambda r: (r[0].kind != "series", len(r), sum(len(s.text) for s in r), r[0].marker_at))


# Said once a step is done: "Done.", "Check.", "and that's it", "sorted".
DONE = {"done", "check", "checked", "tick", "ticked", "sorted", "finished", "complete", "completed", "boom", "nailed",
        "easy", "ready", "dusted"}
_DONE_PHRASES = (("that", "'s", "it"), ("that", "'s", "done"), ("all", "done"), ("job", "done"), ("we're", "done"),
                 ("it", "'s", "done"), ("there", "we", "go"))


def done_after(words: list[Word], steps: list[Step]) -> list[Optional[tuple[float, int]]]:
    """For each step, the word that says it is done, as (its start, its index), or None: a done word
    standing on its own after the step's words ("Plug in the mic. Done.", "press record, and that's
    it"), before the next step's marker (within six seconds for the last)."""
    toks = tokens(words)
    out: list[Optional[tuple[float, int]]] = []
    for n, s in enumerate(steps):
        until = steps[n + 1].marker_at if n + 1 < len(steps) else s.ends + 6.0
        found = None
        for k, t in enumerate(toks):
            w = words[t.w]
            if t.w <= s.words[1] or w.start >= until:
                continue
            alone = t.stop != "" or k + 1 >= len(toks) or toks[k + 1].stop == "." or (k >= 1 and toks[k - 1].stop != "")
            if t.key in DONE and alone and t.key != "ready":
                found = (w.start, t.w)
                break
            for ph in _DONE_PHRASES:
                if [x.key for x in toks[k:k + len(ph)]] == list(ph):
                    found = (w.start, t.w)
                    break
            if found:
                break
        out.append(found)
    return out


# ---- an item made shorter, never cut inside a word ------------------------------------------------------

_SOFT_WORDS = {"just", "really", "actually", "basically", "simply", "literally", "very", "pretty", "quite", "also",
               "even", "eventually", "finally", "then", "like", "so", "totally", "definitely", "obviously"}
_ARTICLE_WORDS = {"the", "a", "an", "my", "your", "our", "their", "his", "her", "its", "some"}


def shorter(text: str) -> list[str]:
    """`text`, then lighter forms of it: without the softeners ("just", "really", "of course"), then
    without articles ("the", "your"). Every form is whole words; repeats are left out."""
    words = text.split()
    forms = [" ".join(words)]
    cleaned = " ".join(text.replace(", of course,", "").replace("of course", "").split())
    soft = [w for w in cleaned.split() if normalize(w) not in _SOFT_WORDS] or cleaned.split()
    forms.append(" ".join(soft))
    lean = [w for i, w in enumerate(soft) if normalize(w) not in _ARTICLE_WORDS or i == len(soft) - 1] or soft
    forms.append(" ".join(lean))
    out = []
    for f in forms:
        f = f[:1].upper() + f[1:]
        if f and f not in out:
            out.append(f)
    return out


def cut_to(text: str, fits: Callable[[str], bool], mark: str = "…") -> str:
    """The most of `text` that fits, cut only between words and never after a word that leads
    somewhere ("to", "the"), with `mark` where words were left off. The first word alone if nothing
    shorter fits."""
    words = text.split()
    for n in range(len(words), 0, -1):
        head = words[:n]
        while len(head) > 1 and normalize(head[-1]) in _OPEN_ENDED:
            head = head[:-1]
        cand = " ".join(head).rstrip(",;:") + ("" if len(head) == len(words) else mark)
        if fits(cand):
            return cand
    return words[0] if words else text
