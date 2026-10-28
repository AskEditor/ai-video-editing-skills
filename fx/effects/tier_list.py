"""tier-list: a tier board on the shot (S to D by default), each card dropping into its row as it is ranked.

What it does, in order:

1. **What goes where.** Either the items are given (`items`: `S: name, name; A: name@4.2`, the same
   as JSON, or a .json file, with a picture per card if wanted), or, with nothing given, the rankings
   are heard in the speech: "espresso is S tier", "the flat white goes in A", "I'd put iced coffee in
   B", "I'd give decaf a D", "S tier: espresso and cortado", "instant coffee, D tier". The plan records the words each card was
   found on, so a wrong one is easy to see and take out.
2. **When each card lands.** On the word that names it, using the core's word starts (pulled for the
   transcriber's lateness and checked against the sound). An item given a time drops on the word
   that starts nearest it; one given no time drops where its name is said, or on its tier when the
   speaker says both ("espresso is S tier" lands on "S", when the ranking is known). One that is never
   said is placed between the others, and the run says so.
3. **Where the board goes.** Where the head never is while the board is on screen: above the head,
   beside it, or below the chin, always inside the safe zone. The top of the head is read from the
   person mask (hair and hats included) where it agrees with the face, so short hair leaves more room
   than a guess from the face would. Each place is tried at the largest row its room allows; the
   roomiest wins, above the head on a near tie. When even the roomiest leaves the words small, the
   run says so rather than covering the face.
4. **The final layout comes first.** Before the first card lands, every card already has its place:
   cards fill their row left to right in the order they arrive, a row wraps to a second line only
   when it must, and nothing already on the board moves when another card arrives.
5. **The end holds.** The last frame stays on screen `hold` seconds with every card in place, because
   people pause on a finished tier list and screenshot it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..core import items as found_items
from ..core import moments
from ..core import text
from ..core.compose import Sprite, over
from ..core.items import Heard, Tok, card_text, squash, tokens
from ..core.layout import Rect, SafeZone, halo_room
from ..core.words import Word
from . import options as O

NAME = "tier-list"
STYLES = ("solid", "glass", "paper")
DROPS = ("fall", "slide", "pop")
POSITIONS = ("auto", "top", "side", "below")
DEFAULT_TIERS = "S,A,B,C,D"

OPTIONS = {
    "items": O.Text("what goes in each row. Empty: the rankings are found in what you say (\"espresso is S tier\"). "
                    "Or S: name, name; A: name@4.2 (a time in seconds is optional), or a .json file, which can "
                    "also give each card a picture"),
    "tiers": O.Text("the rows from the top, separated by commas (S,A,B,C,D). Any names work (GOAT,Good,Mid,Skip), "
                    "and a row can take its own colour: S=#ff5c7a"),
    "title": O.Text("a heading across the top of the board, such as \"My editing apps\"; empty for none"),
    "position": O.Choice(POSITIONS, "auto: wherever there is most room · top: above your head · side: beside it · "
                                    "below: under your chin. Never over your face, whichever you pick"),
    "style": O.Choice(STYLES, "solid: a dark board with white cards · glass: frosted, the video blurred behind it · "
                              "paper: a light board with index cards"),
    "drop": O.Choice(DROPS, "how a card arrives: fall onto the board · slide along its row · pop in where it goes"),
    "hold": O.Number(0.0, 10.0, "seconds the finished board stays on screen at the end, on the last frame; 0 for none"),
    "sound": O.Switch("a soft tap as each card lands, quieter than your voice: on or off"),
    "frame": O.Choice(("keep", "9:16"), "keep the clip's own shape, or make it 9:16 following the face"),
    "size": O.Size("auto (1080x1920 at most, never enlarged) or a size such as 1080x1920"),
    "fix": O.Text("misheard words as wrong=right, separated by ;  e.g. \"ask editor=AskEditor; cortardo=cortado\""),
}

# Each effect names its own runs on the test clips (tests/matrix.py). The items are sample rankings
# placed on real words of each clip (the clips rank nothing themselves); they live in the lab folder.
MATRIX = [
    ("D", "D", {"items": "{lab}/tier-list/inputs/D-travel.json"}),
    ("D-glass", "D", {"items": "{lab}/tier-list/inputs/D-travel.json", "style": "glass", "drop": "slide"}),
    ("D-paper", "D", {"items": "{lab}/tier-list/inputs/D-travel.json", "style": "paper", "drop": "pop",
                      "title": "Hua Hin, ranked"}),
    ("B", "B", {"items": "{lab}/tier-list/inputs/B-video-kit.json"}),
    ("A", "A", {"items": "{lab}/tier-list/inputs/A-platforms.json"}),
    ("C", "C", {"items": "{lab}/tier-list/inputs/C-apps.json"}),
    ("33", "33", {"items": "{lab}/tier-list/inputs/33-topics.json"}),
]

# ---- the look ------------------------------------------------------------------------------------

# Row colours, top row first: our own, warm at the top to cool at the bottom.
PALETTES = {
    "solid": ["#ff5c7a", "#ff9a3d", "#ffd447", "#4ed69b", "#4fb6ff", "#a98bff", "#c9cdd4", "#ff8ad8"],
    "glass": ["#ff6f88", "#ffa552", "#ffdb5e", "#5fdca7", "#63c0ff", "#b39bff", "#d3d6dc", "#ff9bdd"],
    "paper": ["#e8505b", "#f28f3b", "#efc34a", "#6bbf59", "#4a90d9", "#8e7cc3", "#a8a29e", "#e07bb5"],
}
INK = (20, 21, 25)
PAPER_INK = (33, 29, 24)
FACES = {  # (tier labels, cards, title)
    "solid": ("poppins-black", "poppins-bold", "poppins-black"),
    "glass": ("poppins-black", "poppins-bold", "poppins-black"),
    "paper": ("dm-serif", "dm-serif", "dm-serif"),
}

# Proportions of the board, as shares of `u`, the height of a one-line row.
PAD = 0.12  # the board's own margin
ROW_GAP = 0.07  # between rows
ROW_PAD = 0.07  # above and below the cards in a row
CARD_GAP = 0.09  # between cards
TILE_GAP = 0.09  # between a row's label and its cards
CH = 1.0 - 2 * ROW_PAD  # a card's height
TEXT = 0.54  # a card's words, in pixels per u
LABEL = 0.60  # a one- or two-letter row label
TITLE_H = 0.78
U_MAX = 0.066  # the largest row, as a share of the frame's height (bigger looks shouted)
SHADOW = 0.30  # a solid or paper board's shadow reaches this far past it on every side, as a share of u
CARD_SHADOW = 0.16  # a card's own shadow, the same way
GLASS_MARGIN = 0.12  # a glass board has no shadow, but keeps this much (of u) clear of the safe zone's edges
READABLE = 0.0135  # the smallest card words that read on a phone, as a share of the frame's height
GAP = 0.012  # the room kept between the board and the head, as a share of the frame's height
# The board is sized for where the head is nearly all the time, not for the one moment it rises
# highest: in the few frames (at most this share) where the hair reaches into the board, the person
# is laid back over it from the mask, so the board is behind them. The face itself never is.
BEHIND = 0.04
MIN_WIDE = 0.80  # a board is at least this share of the safe zone's width, so it reads as a board


@dataclass
class Tier:
    label: str
    colour: tuple[int, int, int]


@dataclass
class Item:
    """One card as it was asked for: its words, its row, and when, if a time was given."""
    text: str
    tier: int
    at: Optional[float] = None
    image: Optional[str] = None
    order: int = 0


@dataclass
class Card:
    text: str
    tier: int
    at: float  # when it starts to arrive, in the clip's seconds
    image: Optional[str] = None
    how: str = ""  # "said", "ranked", "given" or "placed"
    word: str = ""  # the word it lands on, as heard
    asked: Optional[float] = None  # the time it was given, before it was moved onto a word
    # Filled in by the layout:
    slot: Optional[Rect] = None  # the card's body, in board pixels
    sprite: Optional[Sprite] = None
    ox: float = 0.0  # where the body's top-left sits inside the sprite
    oy: float = 0.0
    px: int = 0


# Items or tiers that cannot be used; the message says what to change. The core's own error, so a
# file or a time that cannot be read is caught and said the same way as a row that does not exist.
TierError = found_items.ItemsError


# ---- reading the tiers and the items -------------------------------------------------------------

def parse_tiers(spec: str, style: str) -> list[Tier]:
    parts = [p.strip() for p in re.split(r"[,\n]", spec or "") if p.strip()]
    if not parts:
        parts = DEFAULT_TIERS.split(",")
    if len(parts) > 8:
        raise TierError(f"a board holds up to 8 rows; {len(parts)} were given")
    palette = PALETTES[style]
    out = []
    for k, p in enumerate(parts):
        label, _, col = p.partition("=")
        label = label.strip()
        if not label or len(label) > 12:
            raise TierError(f"a row's name must be 1 to 12 letters (got {label!r})")
        try:
            colour = text.colour(col.strip()) if col.strip() else text.hex_rgb(palette[k % len(palette)])
        except ValueError as e:
            raise TierError(f"the colour for row {label!r}: {e}") from None
        out.append(Tier(label, colour))
    keys = [_tier_key(t.label) for t in out]
    if len(set(keys)) != len(keys):
        raise TierError("two rows have the same name")
    return out


def _tier_key(label: str) -> str:
    """A row's name for matching: 'S tier', 's-tier' and 'S' are all 's'."""
    k = re.sub(r"[\s_-]*tiers?$", "", label.strip().lower())
    return re.sub(r"\s+", " ", k).strip()


def _read_items(spec: str, base: Path) -> tuple[object, Optional[Path]]:
    """The items as data: from a JSON file or JSON text (the core reads those), or the short form."""
    data, folder = found_items.load(spec, base)
    s = spec.strip()
    if data is not None or not s:
        return data, folder
    groups: dict[str, list[str]] = {}
    for part in re.split(r"[;\n]", s):
        if not part.strip():
            continue
        if ":" not in part:
            raise TierError(f"write each row as  S: name, name  (got {part.strip()!r})")
        label, _, names = part.partition(":")
        groups.setdefault(label.strip(), []).extend(n.strip() for n in names.split(",") if n.strip())
    return groups, None


_entry = found_items.entry


def parse_items(spec: str, tiers: list[Tier], tiers_given: bool, clip_dir: Path,
                style: str) -> tuple[list[Item], list[Tier]]:
    """The items asked for, and the rows they go in. When the rows were not named, an items list
    whose rows are not the usual S to D names makes the rows itself, in the order written."""
    data, base = _read_items(spec, clip_dir)
    if data is None:
        return [], tiers
    pairs: list[tuple[str, object]] = []
    if isinstance(data, dict):
        rows = data.get("tiers") if isinstance(data.get("tiers"), dict) else data
        for label, entries in rows.items():
            if str(label).lower() in ("about", "note", "notes", "comment") or str(label).startswith(("_", "#")):
                continue  # a note about the file, not a row
            if not isinstance(entries, list):
                entries = [entries]
            pairs.extend((str(label), e) for e in entries)
    elif isinstance(data, list):
        for e in data:
            if not isinstance(e, dict) or not e.get("tier"):
                raise TierError("in a list, each item needs its row, as {\"tier\": \"S\", \"text\": \"…\"}")
            pairs.append((str(e["tier"]), e))
    else:
        raise TierError("the items must be rows of names, such as {\"S\": [\"…\"], \"A\": [\"…\"]}")
    keys = {_tier_key(t.label): k for k, t in enumerate(tiers)}
    labels = list(dict.fromkeys(label for label, _ in pairs))
    if any(_tier_key(lb) not in keys for lb in labels):
        if tiers_given:
            missing = [lb for lb in labels if _tier_key(lb) not in keys]
            raise TierError(f"the items name a row the board does not have ({', '.join(missing)}); "
                            f"the rows are {', '.join(t.label for t in tiers)}")
        tiers = parse_tiers(",".join(labels), style)
        keys = {_tier_key(t.label): k for k, t in enumerate(tiers)}
    items = []
    for n, (label, raw) in enumerate(pairs):
        name, at, pic = _entry(raw, base, clip_dir)
        if not name and not pic:
            raise TierError(f"an item in row {label!r} has no words and no picture")
        if len(name) > 40:
            raise TierError(f"{name[:40]!r}… is too long for a card; keep it under 40 letters")
        items.append(Item(name, keys[_tier_key(label)], at, pic, n))
    if len(items) > 40:
        raise TierError(f"{len(items)} cards is more than a board can show readably; keep it to 40")
    return items, tiers


# ---- finding rankings in the words ---------------------------------------------------------------
#
# A ranking needs two things said close together: a thing, and a row. The patterns, in the words
# people use: "X is S tier", "X goes in A", "X gets a C", "I'd put X in B", "I'd give X a D",
# "S tier: X and Y", "S tier is X", "X, B tier". A single letter on its own counts as a row only
# when it ends a clause ("X goes in D.") or is followed by "tier", so "a" in "this is a tier list" or
# "I made a video" is never taken for row A.

_SPOKEN = {"s": {"ess", "es"}, "a": {"ay", "eh"}, "b": {"be", "bee"}, "c": {"see", "sea", "si"},
           "d": {"dee", "de"}, "e": {"ee"}, "f": {"ef", "eff"}}
_TIER_WORDS = {"tier", "tiers"}
_NOT_A_RANK = {"list", "lists", "maker", "ranking", "rankings", "chart", "video", "videos", "system", "board",
               "template"}
_VERBS = {"is", "'s", "was", "are", "goes", "go", "going", "went", "belongs", "belong", "lands", "land", "sits",
          "sit", "gets", "get", "got", "stays", "stay", "ends", "be", "becomes", "earns", "deserves", "makes"}
_SOFT = {"in", "into", "to", "at", "on", "a", "an", "the", "my", "solid", "strong", "clear", "easy", "easily",
         "definite", "definitely", "firm", "firmly", "low", "high", "top", "bottom", "straight", "right",
         "absolutely", "honestly", "probably", "maybe", "just", "still", "up", "for", "me", "it", "pretty",
         "very", "certified", "true", "real", "upper", "lower", "solidly", "firmly", "clearly", "obviously"}
# Said between a thing and its verb: "the latte would be", "espresso is going to be", "decaf's definitely".
_AUX = {"would", "should", "will", "could", "might", "must", "'d", "'ll", "has", "have", "to", "gotta", "going",
        "be", "been", "is", "'s", "definitely", "probably", "easily", "honestly", "still", "always"}
_PUT = {"put", "putting", "puts", "place", "placing", "throw", "throwing", "drop", "dropping", "move", "moving",
        "stick", "sticking", "rank", "ranking", "slot", "slotting", "toss", "tossing"}
_GIVE = {"give", "giving", "gave", "gives"}
_INTO = {"in", "into", "at", "on", "as", "to"}
_ARTICLES = {"the", "a", "an", "my", "your", "our", "their", "his", "her"}
_THEN = {"is", "are", "'s", "goes", "go", "to", "for", "has", "gets", "includes", "would", "be", "going", "will"}
_NEVER_ITEM = set("""
and but or so then think thinks thought guess feel say said mean honestly personally obviously probably
definitely um uh erm like okay ok now next also i i'd i'm i'll i've we you they he she it that this these those
which what who because if when while well yeah yes no right let's lets to would will should could is are was
goes put up first last finally there here where something anything everything nothing stuff things thing one
best worst good bad great reserved only empty is's of for with from by as at in on into the a an
basically literally really actually just clearly simply totally kind sort pretty very gonna maybe
solid fine decent amazing insane crazy wild elite trash awesome terrible nice cool fire goated top bottom
whatever anyway anyways though yep nope sure alright
""".split())


class Found(Heard):
    """A ranking heard in the words (the core's `Heard`): `group` is its row, and the word that
    makes it known (`anchor`) is the row's name, so it is known once both are said."""

    @property
    def tier(self) -> int:
        return self.group

    @property
    def tier_word(self) -> int:
        return self.anchor


def _tier_at(toks: list[Tok], j: int, tiers: list[Tier], words: list[Word]) -> Optional[tuple[int, int, bool]]:
    """A row named at token j: (row, tokens it takes, "tier" said after it), or None."""
    for k, t in enumerate(tiers):
        parts = _tier_key(t.label).split()
        n = len(parts)
        got = [x.key for x in toks[j:j + n]]
        single = n == 1 and len(parts[0]) == 1
        spoken = single and len(got) == 1 and got[0] in _SPOKEN.get(parts[0], set())
        if got != parts and not spoken:
            continue
        after = toks[j + n] if j + n < len(toks) else None
        has_tier = after is not None and after.key in _TIER_WORDS
        if has_tier:
            beyond = toks[j + n + 1] if j + n + 1 < len(toks) else None
            if beyond is not None and beyond.key in _NOT_A_RANK and not toks[j + n].stop:
                continue  # "a tier list"
            return k, n, True
        if spoken:
            continue  # "see", "be" and "bee" count only before "tier"
        end = toks[j + n - 1]
        ends = bool(end.stop) or j + n >= len(toks)
        if single:
            if not ends:
                continue
            if parts[0] == "a":
                word = words[toks[j].w].text.strip()
                if not word.startswith("A") or (j > 0 and toks[j - 1].stop == "."):
                    continue  # "a" only as a capital that ends a clause: "goes in A."
            return k, n, False
        return k, n, False  # a named row ("GOAT", "Mid"): the pattern around it decides
    return None


def _item_before(toks: list[Tok], end: int, words: list[Word], loose: bool = False) -> Optional[tuple[int, int]]:
    """The thing named just before token `end` (exclusive): up to four tokens, stopping at the end of
    a clause, a word that cannot be part of a name, or another row's "tier". `loose` lets the name
    end on a comma ("Google Docs, C tier"); a name never runs back across a full stop. Returns
    token indexes [a, b)."""
    b = end
    a = end
    while a > 0 and end - a < 4:
        t = toks[a - 1]
        if t.key in _NEVER_ITEM and t.key not in _ARTICLES or t.key in _TIER_WORDS:
            break
        if t.stop and a != end:
            break  # a comma or full stop after this word: the name starts after it
        if a == end and (t.stop == "." or (t.stop and not loose)):
            break
        a -= 1
    while a < b and toks[a].key in _ARTICLES:
        a += 1
    if a >= b:
        return None
    if not any(len(toks[k].key) >= 2 or toks[k].cap for k in range(a, b)):
        return None
    return a, b


def _items_after(toks: list[Tok], start: int, tiers, words) -> list[tuple[int, int]]:
    """'S tier: Claude and Cursor' — the things listed after a row, up to the end of the sentence."""
    out, cur = [], []
    k = start
    while k < len(toks) and len(out) < 6:
        t = toks[k]
        if _tier_at(toks, k, tiers, words) is not None or t.key in _VERBS:
            break
        if t.key in ("and", "&", "plus", "or") or t.key in _NEVER_ITEM and t.key not in _ARTICLES:
            if cur:
                out.append(cur)
            cur = []
            if t.key not in ("and", "&", "plus", "or"):
                break
        else:
            cur.append(k)
            if t.stop:
                out.append(cur)
                cur = []
                if t.stop == ".":
                    break
        k += 1
    if cur:
        out.append(cur)
    spans = []
    for c in out:
        a, b = c[0], c[-1] + 1
        while a < b and toks[a].key in _ARTICLES:
            a += 1
        if a < b and b - a <= 4:
            spans.append((a, b))
    return spans


_text = card_text


def find_rankings(words: list[Word], tiers: list[Tier]) -> list[Found]:
    """Every "X is S tier" in the words, in the order said. A thing ranked twice keeps the last."""
    keys = {_tier_key(t.label) for t in tiers}
    toks = tokens(words, keys)
    found: list[Found] = []
    used: set[int] = set()
    j = 0
    while j < len(toks):
        hit = _tier_at(toks, j, tiers, words)
        if hit is None:
            j += 1
            continue
        tier, n, has_tier = hit
        after = j + n + (1 if has_tier else 0)
        span = None
        # "X is (a solid) S tier", "X goes in A"
        v = j - 1
        skipped = 0
        while v >= 0 and toks[v].key in _SOFT and skipped < 3 and not toks[v].stop:
            v -= 1
            skipped += 1
        if v >= 0 and toks[v].key in _VERBS and not toks[v].stop:
            while v > 0 and toks[v - 1].key in _AUX and not toks[v - 1].stop:
                v -= 1
            span = _item_before(toks, v, words)
        # "I'd put X in B", "I'd give X a C"
        if span is None:
            p = j - 1
            while p >= 0 and toks[p].key in _ARTICLES:
                p -= 1
            if p >= 1 and toks[p].key in _INTO:
                for q in range(p - 1, max(-1, p - 6), -1):
                    if toks[q].key in _PUT:
                        span = (q + 1, p) if q + 1 < p else None
                        break
            if span is None:
                g = j - 1
                while g >= 0 and toks[g].key in ("a", "an", "the"):
                    g -= 1
                for q in range(g, max(-1, g - 5), -1):
                    if toks[q].key in _GIVE:
                        span = (q + 1, g + 1) if q + 1 <= g else None
                        break
            if span is not None:
                a, b = span
                while a < b and toks[a].key in _ARTICLES:
                    a += 1
                span = (a, b) if a < b and all(toks[k].key not in _NEVER_ITEM or toks[k].key in _ARTICLES
                                               for k in range(a, b)) else None
        # "Excel, D tier": a name just before the row, only with "tier" said, and a comma after the
        # name or a capital in it (with no verb, the pause or the capital is what says it is a name)
        if span is None and has_tier:
            e = j
            while e > 0 and toks[e - 1].key in _SOFT - _ARTICLES and toks[e - 1].key not in _INTO:
                e -= 1
            if e > 0 and toks[e - 1].key not in _VERBS | _INTO | _ARTICLES:
                cand = _item_before(toks, e, words, loose=True)
                if (cand and (toks[e - 1].stop == "," or any(toks[k].cap for k in range(*cand)))
                        and not toks[cand[0]].stop == "."):
                    span = cand
        spans = [span] if span is not None else []
        # "S tier: Claude and Cursor", "S tier is Claude"
        if not spans and has_tier:
            k = after
            colon = toks[after - 1].colon
            took = 0
            while k < len(toks) and toks[k].key in _THEN and took < 2 and not toks[k - 1].stop == ".":
                k += 1
                took += 1
            if (colon or took) and not toks[after - 1].stop == ".":
                spans = _items_after(toks, k, tiers, words)
        for a, b in spans:
            if any(toks[x].w in used for x in range(a, b)):
                continue
            w0, w1 = toks[a].w, toks[b - 1].w
            name = _text(words, toks, a, b)
            if not name:
                continue
            tw = toks[j].w
            at = max(words[w0].start, words[tw].start)
            lo, hi = min(w0, tw), max(w1, toks[after - 1].w)
            phrase = " ".join(words[i].text.strip() for i in range(lo, hi + 1))
            found.append(Found(name, tier, at, (w0, w1), tw, phrase))
            used.update(range(w0, w1 + 1))
        j = after if spans else j + 1
    last: dict[str, Found] = {}
    for f in found:
        last[squash(f.text)] = f
    return sorted(last.values(), key=lambda f: f.at)


# ---- timing the items that were given ------------------------------------------------------------


def time_items(items: list[Item], words: list[Word], found: list[Found], duration: float,
               quiet=None) -> tuple[list[Card], list[str]]:
    """Each item's moment (the core's `time_items`). Given a time: the word starting nearest it
    (within a quarter second). Given none: its ranking if one was heard ("X is S tier", at the row),
    else its name where it is said, else between the others in the order given; never where the
    speaker took it back (`moments.taken_back` of `words`, the clip's whole words) unless it is
    said nowhere else."""

    def mismatch(it: Item, f: Found) -> Optional[str]:
        if f.tier != it.tier:
            return f"{it.text}: said as {f.phrase!r} but given row {it.tier + 1}; it stays where it was given"
        return None

    timed, notes = found_items.time_items(items, words, found, duration, heard_how="ranked", mismatch=mismatch,
                                          back=moments.taken_back(words, [] if quiet is None else quiet))
    cards = [Card(items[t.n].text, items[t.n].tier, t.at, items[t.n].image, t.how, t.word, asked=t.asked)
             for t in timed]
    return cards, notes


# ---- motion ---------------------------------------------------------------------------------------

def _clamp01(x: float) -> float:
    return min(1.0, max(0.0, x))


def _out_cubic(x: float) -> float:
    x = _clamp01(x)
    return 1.0 - (1.0 - x) ** 3


def _out_back(x: float, s: float = 1.6) -> float:
    x = _clamp01(x) - 1.0
    return 1.0 + (s + 1.0) * x ** 3 + s * x ** 2


FALL = 0.24  # the fall, then a short settle
SETTLE = 0.14
SLIDE = 0.34
POP = 0.26
LANDS = {"fall": FALL, "slide": 0.75 * SLIDE, "pop": 0.12}  # from the start of the arrival to the moment it lands
SETTLED = {"fall": FALL + SETTLE, "slide": SLIDE, "pop": POP}
BOARD_IN = 0.30  # the board's own entrance
TILE_STEP = 0.045  # the rows' labels come in one after another
PULSE = 0.24


def _motion(drop: str, dt: float, travel: float, lift: float, grow: float = 0.22,
            nudge: float = 0.0) -> tuple[float, float, float, float]:
    """A card `dt` seconds into its arrival: (dx, dy, scale, opacity). `travel` is how far a sliding
    card comes from, and `nudge` how far it runs past its place before settling back (a fixed few
    pixels, smaller than the gap to its left, so it never runs onto the card or label beside it); a
    falling card starts `lift` higher and `grow` larger, as if nearer the eye."""
    if dt >= SETTLED[drop]:
        return 0.0, 0.0, 1.0, 1.0
    if drop == "fall":
        if dt < FALL:
            p = dt / FALL
            fall = p * p  # it speeds up as it falls
            return 0.0, -lift * (1.0 - fall), 1.0 + grow * (1.0 - fall), _clamp01(dt / 0.07)
        q = (dt - FALL) / SETTLE
        return 0.0, 0.0, 1.0 - 0.045 * math.sin(math.pi * q), 1.0
    if drop == "slide":
        p = dt / SLIDE
        past = nudge * math.sin(math.pi * _clamp01((p - 0.55) / 0.45))
        return travel * (1.0 - _out_cubic(p)) - past, 0.0, 1.0, _clamp01(dt / 0.06)
    p = dt / POP
    return 0.0, 0.0, 0.35 + 0.65 * _out_back(p, 2.2), _clamp01(dt / 0.05)


# ---- drawing --------------------------------------------------------------------------------------

def _rgba(w: float, h: float):
    from PIL import Image

    return Image.new("RGBA", (max(1, int(math.ceil(w))), max(1, int(math.ceil(h)))), (0, 0, 0, 0))


def _round_rect(img, box: tuple[float, float, float, float], radius: float, fill, k: int = 4) -> None:
    """A smooth rounded rectangle drawn onto `img` (RGBA, PIL), four times larger and brought down."""
    from PIL import Image, ImageDraw

    x0, y0, x1, y1 = box
    w, h = int(math.ceil(x1 - x0)) + 2, int(math.ceil(y1 - y0)) + 2
    big = Image.new("RGBA", (w * k, h * k), (0, 0, 0, 0))
    fx, fy = x0 - math.floor(x0), y0 - math.floor(y0)
    ImageDraw.Draw(big).rounded_rectangle(
        (fx * k, fy * k, (fx + x1 - x0) * k - 1, (fy + y1 - y0) * k - 1), radius=max(0.0, radius * k), fill=fill)
    small = big.resize((w, h), Image.LANCZOS)
    img.alpha_composite(small, (int(math.floor(x0)), int(math.floor(y0))))


def _shadowed(img, blur: float, dy: float, opacity: float, margin: int):
    """`img` with a soft shadow of its own shape beneath it, in a canvas `margin` larger each side."""
    from PIL import Image, ImageFilter

    W, H = img.width + 2 * margin, img.height + 2 * margin
    a = Image.new("L", (W, H), 0)
    a.paste(img.getchannel("A"), (margin, margin + int(round(dy))))
    if blur > 0:
        a = a.filter(ImageFilter.GaussianBlur(blur))
    a = a.point(lambda v: int(v * opacity))
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out.putalpha(a)
    out.alpha_composite(img, (margin, margin))
    return out


def _fit_picture(path: str, size: int, radius: float):
    """A picture fitted into a square: a logo with a see-through background is fitted whole on the
    card; a photo fills the square."""
    from PIL import Image, ImageDraw, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGBA")
    alpha = np.asarray(im.getchannel("A"))
    if (alpha < 250).mean() > 0.02:  # a cut-out or a logo: whole, with a little room
        inner = max(1, int(round(size * 0.84)))
        im.thumbnail((inner, inner), Image.LANCZOS)
        out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        out.alpha_composite(im, ((size - im.width) // 2, (size - im.height) // 2))
        return out
    im = ImageOps.fit(im, (size, size), Image.LANCZOS)
    mask = Image.new("L", (size * 4, size * 4), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size * 4 - 1, size * 4 - 1), radius=radius * 4, fill=255)
    im.putalpha(mask.resize((size, size), Image.LANCZOS))
    return im


@dataclass
class Row:
    y: float
    h: float
    lines: int


@dataclass
class Geometry:
    u: float
    w: float
    h: float
    tile_w: float
    lane_x: float
    rows: list[Row]
    slots: list[Rect]  # per card, in arrival order, in board pixels
    card_px: list[int]
    label_px: list[int]
    title_h: float
    text_px: int
    cut: bool = False  # some names had to be cut short to fit


def _card_width(card: Card, ch: float, px: float, face: str) -> float:
    padx = 0.30 * ch
    if card.image and card.text:
        return ch + 0.16 * ch + text.advance(card.text, face, int(px)) + padx
    if card.image:
        return ch
    return max(1.25 * ch, text.advance(card.text, face, int(px)) + 2 * padx)


def lay_out(u: float, band_w: float, band_h: float, tiers: list[Tier], cards: list[Card], style: str,
            title: str, zone_w: float, allow_cut: bool = False) -> Optional[Geometry]:
    """The board at row height `u` inside a band of band_w x band_h, or None when it does not fit.
    A name too long for its row at this size counts as not fitting, unless `allow_cut` (the last
    resort, when nothing else fits) lets it be cut short."""
    label_face, card_face, _ = FACES[style]
    ch = CH * u
    pad, gap = PAD * u, TILE_GAP * u
    label_px = []
    tile_w = u
    for t in tiers:
        px = LABEL * u if len(t.label) <= 2 else 0.42 * u
        need = text.advance(t.label, label_face, int(px)) + 0.36 * u
        if need > 3.0 * u:  # a long name: smaller words, never a wider tile than three rows tall
            px = max(0.26 * u, px * 2.64 * u / max(1.0, need - 0.36 * u))
            need = text.advance(t.label, label_face, int(px)) + 0.36 * u
        label_px.append(int(px))
        tile_w = max(tile_w, min(3.0 * u, need))
    lane_max = band_w - 2 * pad - tile_w - gap
    if lane_max < 1.4 * ch:
        return None
    px = TEXT * u
    widths, card_px = [], []
    for c in cards:
        cpx = px
        w = _card_width(c, ch, cpx, card_face)
        if w > lane_max:  # a long name: its own words a little smaller, then cut short
            cpx = max(0.72 * px, px * (lane_max - 0.6 * ch) / max(1.0, w - 0.6 * ch))
            w = _card_width(c, ch, cpx, card_face)
            if w > lane_max + 0.5 and not allow_cut:
                return None
            w = min(lane_max, w)
        widths.append(w)
        card_px.append(int(cpx))
    by_row: list[list[int]] = [[] for _ in tiers]
    for k, c in enumerate(cards):
        by_row[c.tier].append(k)
    cg = CARD_GAP * u
    one_line = max([sum(widths[k] for k in r) + cg * max(0, len(r) - 1) for r in by_row] + [0.0])
    lane_w = min(lane_max, max(one_line + 2 * cg, MIN_WIDE * zone_w - 2 * pad - tile_w - gap, 2.5 * ch))
    board_w = 2 * pad + tile_w + gap + lane_w
    title_h = TITLE_H * u if title else 0.0
    y = pad + (title_h + ROW_GAP * u if title else 0.0)
    rows: list[Row] = []
    slots: list[Optional[Rect]] = [None] * len(cards)
    lane_x = pad + tile_w + gap
    for r, members in enumerate(by_row):
        lines, x, line = 1, cg, 0
        placed = []
        for k in members:
            if x + widths[k] > lane_w - cg + 0.5 and x > cg:
                line += 1
                x = cg
            placed.append((k, x, line))
            x += widths[k] + cg
        lines = line + 1
        h = 2 * ROW_PAD * u + lines * ch + (lines - 1) * cg
        for k, x0, ln in placed:
            yy = y + ROW_PAD * u + ln * (ch + cg)
            slots[k] = Rect(lane_x + x0, yy, lane_x + x0 + widths[k], yy + ch)
        rows.append(Row(y, h, lines))
        y += h + ROW_GAP * u
    board_h = y - ROW_GAP * u + pad
    if board_h > band_h + 0.5 or board_w > band_w + 0.5:
        return None
    return Geometry(u, board_w, board_h, tile_w, lane_x, rows, [s for s in slots if s is not None], card_px,
                    label_px, title_h, int(px))


def halo(style: str, u: float) -> tuple[float, float, float, float]:
    """How far the board's drawing reaches past the board at row height u, in pixels (left, top,
    right, bottom): its shadow's whole canvas. Glass has none (it is the picture blurred inside the
    board), but its bright edge sat 1 px under the zone's top on D and the frosting reads as part of
    the board, so it keeps a small margin too (`GLASS_MARGIN`). The board is placed so that this
    stays inside the safe zone too."""
    if style == "glass":
        m = float(math.ceil(GLASS_MARGIN * u))
        return m, m, m, m
    m = float(math.ceil(SHADOW * u))
    return m, m, m, m


def fit(band: Rect, tiers, cards, style, title, H: float, zone: Rect) -> Optional[Geometry]:
    """The largest board that fits the band, its shadow included: rows from U_MAX of the frame's
    height down, every name whole; only when no size keeps them whole are the longest cut short."""
    floor = 0.010 * H
    for allow_cut in (False, True):
        u = min(U_MAX * H, band.h)
        while u >= floor:
            room = halo_room(band, zone, halo(style, u))
            g = lay_out(u, room.w, room.h, tiers, cards, style, title, zone.w, allow_cut) if room.w > 0 and room.h > 0 else None
            if g is not None:
                g.cut = allow_cut
                return g
            u -= max(0.5, u * 0.01)
    return None


class Board:
    """The tier board of one output: laid out, placed, its pieces drawn once, ready for each frame."""

    def __init__(self, tiers: list[Tier], cards: list[Card], geo: Geometry, x: float, y: float, style: str,
                 drop: str, title: str, board_in: float, size: tuple[int, int], fps: float, n_frames: int,
                 zone: SafeZone, keep: list[Optional[Rect]], front: Optional[PersonFront] = None):
        self.keep, self.front = keep, front
        self.smoke = 0.5  # how dark the frosted glass is: set from what is behind it (`set_smoke`)
        self.tiers, self.cards, self.g = tiers, cards, geo
        self.x, self.y = float(round(x)), float(round(y))
        self.style, self.drop, self.title = style, drop, title
        self.board_in = board_in
        self.W, self.H = size
        self.fps, self.n = fps, n_frames
        self.zone = zone
        self.rect = Rect(self.x, self.y, self.x + geo.w, self.y + geo.h)
        for c, s, px in zip(cards, geo.slots, geo.card_px):
            c.slot, c.px = s, px
        self._build()

    # -- the pieces, drawn once
    def _build(self) -> None:
        from PIL import Image, ImageDraw

        g, u, style = self.g, self.g.u, self.style
        label_face, card_face, title_face = FACES[style]
        W, H = int(math.ceil(g.w)), int(math.ceil(g.h))
        panel = _rgba(W, H)
        radius = 0.20 * u
        if style == "solid":
            _round_rect(panel, (0, 0, g.w, g.h), radius, (16, 17, 20, 236))
        elif style == "glass":
            _round_rect(panel, (0, 0, g.w, g.h), radius, (255, 255, 255, 255))
            self.glass_mask = np.asarray(panel.getchannel("A"), np.float32) / 255.0
            panel = _rgba(W, H)
            edge = max(1.0, 0.022 * u)
            _round_rect(panel, (0, 0, g.w, g.h), radius, (255, 255, 255, 84))
            # Only a bright edge is drawn: the middle is punched out, so the frosted video shows.
            a = np.asarray(panel).copy()
            inner = _rgba(W, H)
            _round_rect(inner, (edge, edge, g.w - edge, g.h - edge), radius - edge, (255, 255, 255, 255))
            a[..., 3] = (a[..., 3].astype(np.float32) * (1 - np.asarray(inner.getchannel("A"), np.float32) / 255)).astype(np.uint8)
            panel = Image.fromarray(a, "RGBA")
            sheen = _rgba(W, H)
            _round_rect(sheen, (0, 0, g.w, g.h * 0.5), radius, (255, 255, 255, 18))
            panel.alpha_composite(sheen)
        else:  # paper: a hairline of ink round the sheet, so it stands off a white wall too
            edge = max(1.0, 0.025 * u)
            _round_rect(panel, (0, 0, g.w, g.h), 0.12 * u, PAPER_INK + (70,))
            inner = _rgba(W, H)
            _round_rect(inner, (edge, edge, g.w - edge, g.h - edge), 0.12 * u - edge, (246, 241, 231, 255))
            a = np.asarray(panel).astype(np.float32)
            b = np.asarray(inner).astype(np.float32)
            ab = b[..., 3:] / 255.0  # the sheet laid over its own outline
            a[..., :3] = b[..., :3] * ab + a[..., :3] * (1 - ab)
            a[..., 3:] = np.maximum(a[..., 3:], b[..., 3:])
            panel = Image.fromarray(a.astype(np.uint8), "RGBA")
            rng = np.random.default_rng(7)
            a = np.asarray(panel).astype(np.float32)
            grain = rng.normal(0.0, 2.4, a.shape[:2])[..., None]
            a[..., :3] = np.clip(a[..., :3] + grain * (a[..., 3:] > 0), 0, 255)
            panel = Image.fromarray(a.astype(np.uint8), "RGBA")
        draw = ImageDraw.Draw(panel)
        lane_r = 0.12 * u
        self.lanes = []
        for r, row in enumerate(g.rows):
            box = (g.lane_x, row.y, g.w - PAD * u, row.y + row.h)
            if style == "solid":
                _round_rect(panel, box, lane_r, (30, 32, 38, 255))
            elif style == "glass":
                _round_rect(panel, box, lane_r, (255, 255, 255, 20))
            elif r < len(g.rows) - 1:  # paper: a hairline between rows
                yy = row.y + row.h + ROW_GAP * u / 2
                draw.line((PAD * u, yy, g.w - PAD * u, yy), fill=PAPER_INK + (34,), width=max(1, int(round(0.025 * u))))
            self.lanes.append(box)
        if self.title:
            face = title_face
            px = int(0.52 * u)
            limit = g.w - 2 * PAD * u
            if text.advance(self.title, face, px) > limit:
                px = text.fit_px(self.title, face, limit)
            ink = (255, 255, 255) if style != "paper" else PAPER_INK
            cap = text.cap_height(face, px)
            draw.text((g.w / 2, PAD * u + (g.title_h + cap) / 2), self.title, font=text.font(face, px),
                      fill=ink + (255,), anchor="ms")
        if style == "glass":
            self.panel = Sprite.from_rgba(np.asarray(panel))
            self.panel_m = 0
        else:
            m = int(math.ceil(SHADOW * u))
            shadowed = _shadowed(panel, 0.20 * u, 0.06 * u, 0.42 if style == "solid" else 0.30, m)
            self.panel = Sprite.from_rgba(np.asarray(shadowed))
            self.panel_m = m
        # The rows' labels, each its own piece: they arrive one after another, and swell as a card lands.
        self.tiles = []
        for r, (t, row) in enumerate(zip(self.tiers, g.rows)):
            tile = _rgba(g.tile_w, row.h)
            colour = t.colour if style != "glass" else t.colour
            _round_rect(tile, (0, 0, g.tile_w, row.h), (0.14 if style != "paper" else 0.10) * u, colour + (255,))
            px = g.label_px[r]
            ink = INK if style != "paper" else PAPER_INK
            cap = text.cap_height(label_face, px)
            ImageDraw.Draw(tile).text((g.tile_w / 2, row.h / 2 + cap / 2), t.label, font=text.font(label_face, px),
                                      fill=ink + (255,), anchor="ms")
            self.tiles.append((Sprite.from_rgba(np.asarray(tile)), PAD * u, row.y))
        # The flash of colour along a row as a card lands in it.
        self.flashes = []
        for t, box in zip(self.tiers, self.lanes):
            fl = _rgba(box[2] - box[0], box[3] - box[1])
            _round_rect(fl, (0, 0, box[2] - box[0], box[3] - box[1]), lane_r, t.colour + (255,))
            self.flashes.append((Sprite.from_rgba(np.asarray(fl)), box[0], box[1]))
        for c in self.cards:
            self._card(c, card_face)

    def _card(self, c: Card, face: str) -> None:
        from PIL import ImageDraw

        u = self.g.u
        s = c.slot
        w, h = s.w, s.h
        body = _rgba(w, h)
        if self.style == "paper":
            _round_rect(body, (0, 0, w, h), 0.07 * u, (31, 27, 22, 44))
            line = max(1.0, 0.025 * u)
            _round_rect(body, (line, line, w - line, h - line), 0.07 * u - line, (255, 254, 250, 255))
            ink = PAPER_INK
        else:
            _round_rect(body, (0, 0, w, h), 0.16 * u, (255, 255, 255, 250))
            ink = INK
        draw = ImageDraw.Draw(body)
        x_text = 0.30 * h
        if c.image:
            inset = 0.07 * h if not c.text else 0.08 * h
            side = int(round(h - 2 * inset))
            pic = _fit_picture(c.image, side, 0.10 * u)
            body.alpha_composite(pic, (int(round(inset)), int(round(inset))))
            x_text = h + 0.16 * h - inset
        if c.text:
            label = c.text
            limit = w - x_text - (0.30 * h if not c.image else 0.24 * h)
            while text.advance(label, face, c.px) > limit + 0.5 and len(label) > 2:
                label = label[:-2].rstrip() + "…"
            cap = text.cap_height(face, c.px)
            if c.image:
                draw.text((x_text, h / 2 + cap / 2), label, font=text.font(face, c.px), fill=ink + (255,), anchor="ls")
            else:
                draw.text((w / 2, h / 2 + cap / 2), label, font=text.font(face, c.px), fill=ink + (255,), anchor="ms")
        m = int(math.ceil(CARD_SHADOW * u))
        img = _shadowed(body, 0.06 * u, 0.04 * u, 0.30 if self.style != "paper" else 0.22, m)
        ox = oy = float(m)
        if self.style == "paper":  # an index card, set down a touch crooked
            from PIL import Image

            angle = ((sum(map(ord, c.text or c.image or "")) % 7) - 3) * 0.45
            big = img.resize((img.width * 2, img.height * 2), Image.LANCZOS).rotate(
                angle, resample=Image.BICUBIC, expand=True)
            rot = big.resize((max(1, big.width // 2), max(1, big.height // 2)), Image.LANCZOS)
            ox += (rot.width - img.width) / 2
            oy += (rot.height - img.height) / 2
            img = rot
        c.sprite, c.ox, c.oy = Sprite.from_rgba(np.asarray(img)), ox, oy

    def set_smoke(self, frames: list[np.ndarray]) -> None:
        """Glass over a bright wall needs more smoke than glass over trees for white cards to stand
        off it: darken until what shows through sits at about a third of full brightness."""
        x0, y0 = int(self.x), int(self.y)
        x1, y1 = int(self.x + self.g.w), int(self.y + self.g.h)
        lum = [float((f[y0:y1, x0:x1].astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)).mean()) / 255
               for f in frames]
        if lum:
            light = float(np.median(lum))
            self.smoke = float(np.clip((light - 0.30) / max(0.05, light - 0.05), 0.35, 0.78))

    # -- where a card is at a moment
    def lands(self, c: Card) -> float:
        return c.at + LANDS[self.drop]

    def _card_state(self, c: Card, t: float) -> Optional[tuple[float, float, float, float]]:
        """(centre x, centre y, scale, opacity) of a card's body in frame pixels, or None before it comes."""
        if t < c.at:
            return None
        s = c.slot
        z = self.zone.rect
        travel = (self.g.w - PAD * self.g.u - s.w) - s.x0
        # A falling card starts above its place and a little larger, never past the safe zone, its
        # shadow counted.
        left, top, right, _ = self._card_reach(c)
        # each side grows by grow/2 of the card and grow of its shadow: (1 + grow) * reach + grow/2 * w
        side = min((self.x + s.x0 - z.x0 - left) / (s.w / 2 + left), (z.x1 - self.x - s.x1 - right) / (s.w / 2 + right))
        above = (self.y + s.y0 - z.y0 - top) / (s.h / 2 + top)
        grow = max(0.0, min(0.22, side, above))
        lift = max(0.0, min(0.55 * self.g.u, self.y + s.y0 - grow / 2 * s.h - (1 + grow) * top - z.y0))
        dx, dy, scale, alpha = _motion(self.drop, t - c.at, max(0.0, travel), lift, grow, 0.6 * CARD_GAP * self.g.u)
        return self.x + s.cx + dx, self.y + s.cy + dy, scale, alpha

    def _card_reach(self, c: Card) -> tuple[float, float, float, float]:
        """How far a card's sprite (its shadow, its tilt) reaches past its slot: (left, top, right, bottom)."""
        sp = getattr(c, "sprite", None)
        if sp is None:  # not drawn yet: the shadow's canvas alone
            m = float(math.ceil(CARD_SHADOW * self.g.u))
            return m, m, m, m
        return c.ox, c.oy, sp.w - c.ox - c.slot.w, sp.h - c.oy - c.slot.h

    def _card_drawn(self, c: Card, t: float) -> Optional[Rect]:
        """The card as drawn at t, its shadow included."""
        st = self._card_state(c, t)
        if st is None:
            return None
        cx, cy, scale, _ = st
        left, top, right, bottom = self._card_reach(c)
        w, h = c.slot.w * scale, c.slot.h * scale
        return Rect(cx - w / 2 - left * scale, cy - h / 2 - top * scale, cx + w / 2 + right * scale,
                    cy + h / 2 + bottom * scale)

    def _card_rect(self, c: Card, t: float) -> Optional[Rect]:
        st = self._card_state(c, t)
        if st is None:
            return None
        cx, cy, scale, _ = st
        w, h = c.slot.w * scale, c.slot.h * scale
        return Rect(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    # -- drawing
    def _glass(self, frame: np.ndarray, opacity: float) -> None:
        import cv2

        x0, y0 = int(self.x), int(self.y)
        m = self.glass_mask
        h, w = m.shape
        x1, y1 = min(frame.shape[1], x0 + w), min(frame.shape[0], y0 + h)
        if x1 <= x0 or y1 <= y0:
            return
        region = frame[y0:y1, x0:x1].astype(np.float32)
        sigma = max(4.0, 0.22 * self.g.u)
        blurred = cv2.GaussianBlur(region, (0, 0), sigma)
        k = self.smoke
        smoke = blurred * (1.0 - k) + np.array([10, 12, 16], np.float32) * k
        a = (m[:y1 - y0, :x1 - x0] * opacity)[..., None]
        frame[y0:y1, x0:x1] = np.clip(region * (1 - a) + smoke * a + 0.5, 0, 255).astype(np.uint8)

    def _behind(self, j: int, t: float) -> Optional[Rect]:
        """The head, when it reaches into the board at this frame and the person can be laid over it."""
        if self.front is None or t < self.board_in or j >= len(self.keep):
            return None
        head = self.keep[j]
        return head if head is not None and head.intersects(self.rect.grow(*(4 * [0.3 * self.g.u]))) else None

    def draw(self, frame: np.ndarray, j: int, t: float) -> None:
        if t < self.board_in:
            return
        head = self._behind(j, t)
        original = frame.copy() if head is not None else None
        self._draw(frame, j, t)
        if head is not None and original is not None:
            self._lay_person_back(frame, original, j, head)

    def _lay_person_back(self, frame: np.ndarray, original: np.ndarray, j: int, head: Rect) -> None:
        """Put the person back over the board where the head reaches into it: only around the head,
        with a soft edge, so a stray patch of the mask elsewhere never cuts a hole in the board."""
        u = self.g.u
        H, W = frame.shape[:2]
        r = self.rect.grow(*(4 * [0.6 * u]))
        x0, y0, x1, y1 = max(0, int(r.x0)), max(0, int(r.y0)), min(W, int(r.x1) + 1), min(H, int(r.y1) + 1)
        if x1 <= x0 or y1 <= y0:
            return
        pm = self.front.alpha(j, original)[y0:y1, x0:x1]
        hb = head.grow(0.12 * head.w, 0.12 * head.h, 0.12 * head.w, 0.04 * head.h)
        feather = max(2.0, 0.25 * u)
        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        wx = np.clip(1.0 - np.maximum(hb.x0 - xs, xs - hb.x1) / feather, 0.0, 1.0)
        wy = np.clip(1.0 - np.maximum(hb.y0 - ys, ys - hb.y1) / feather, 0.0, 1.0)
        a = (pm * wy[:, None] * wx[None, :])[..., None]
        drawn = frame[y0:y1, x0:x1].astype(np.float32)
        frame[y0:y1, x0:x1] = np.clip(drawn * (1.0 - a) + original[y0:y1, x0:x1].astype(np.float32) * a + 0.5,
                                      0, 255).astype(np.uint8)

    def _draw(self, frame: np.ndarray, j: int, t: float) -> None:
        step = 1.0 / self.fps
        k = t - self.board_in + step  # the frame it appears on already shows it
        a = _out_cubic(k / BOARD_IN)
        if self.style == "glass":
            self._glass(frame, a)
        m = self.panel_m
        if a < 0.999:
            s = 0.96 + 0.04 * a
            sp = self.panel.scaled(s)
            cx, cy = self.x + self.g.w / 2, self.y + self.g.h / 2
            over(frame, sp, cx - sp.w / 2, cy - sp.h / 2, a)
        else:
            over(frame, self.panel, self.x - m, self.y - m)
        landed = {}
        for c in self.cards:
            tl = self.lands(c)
            if tl <= t:
                landed[c.tier] = tl
        for r, (sp, tx, ty) in enumerate(self.tiles):
            kk = k - 0.06 - TILE_STEP * r
            if kk <= 0:
                continue
            alpha = _clamp01(kk / 0.12)
            scale = 0.7 + 0.3 * _out_back(kk / 0.22, 2.0)
            since = t - landed[r] if r in landed else None
            if since is not None and 0 <= since < PULSE:
                scale *= 1.0 + 0.08 * math.sin(math.pi * since / PULSE)
                fs, fx, fy = self.flashes[r]
                over(frame, fs, self.x + fx, self.y + fy, 0.30 * (1.0 - since / PULSE))
            if abs(scale - 1.0) > 1e-3:
                ss = sp.scaled(scale)
                over(frame, ss, self.x + tx + sp.w / 2 - ss.w / 2, self.y + ty + sp.h / 2 - ss.h / 2, alpha)
            else:
                over(frame, sp, self.x + tx, self.y + ty, alpha)
        moving = []
        for c in self.cards:
            st = self._card_state(c, t)
            if st is None:
                continue
            if st[2] == 1.0 and st[3] == 1.0 and t - c.at >= SETTLED[self.drop]:
                over(frame, c.sprite, self.x + c.slot.x0 - c.ox, self.y + c.slot.y0 - c.oy)
            else:
                moving.append((c, st))
        for c, (cx, cy, scale, alpha) in moving:
            sp = c.sprite.scaled(scale) if abs(scale - 1.0) > 1e-3 else c.sprite
            bx = (c.ox + c.slot.w / 2) * (sp.w / c.sprite.w)
            by = (c.oy + c.slot.h / 2) * (sp.h / c.sprite.h)
            over(frame, sp, cx - bx, cy - by, alpha)

    # -- the sound of each card landing (Job.sound: the output's sound and its rate in, the same length out)
    def mix(self, sound: np.ndarray, rate: int) -> np.ndarray:
        sound = sound.copy()
        voice = np.abs(sound).max(axis=1) if len(sound) else np.zeros(1)
        level = float(np.percentile(voice, 99.5)) if len(voice) else 0.0
        peak = float(np.clip(0.28 * level, 0.035, 0.16))
        for c in self.cards:
            at = self.lands(c)
            i0 = int(round(at * rate))
            if i0 >= len(sound):
                continue
            tap = _tap(rate, 860.0 * 2 ** (-2 * c.tier / 12)) * peak
            i1 = min(len(sound), i0 + len(tap))
            sound[i0:i1] += tap[:i1 - i0, None]
        return sound

    # -- what a person (or the matrix) checks
    def report(self, keep: list[Optional[Rect]], faces: list[Optional[Rect]]) -> dict:
        """Frame by frame: anything drawn over the face box, or over the head with nobody laid back
        over it (both must be 0); how often the head passes in front of the board; anything outside
        the safe zone (0)."""
        over_face = behind = outside = 0
        zone = self.zone.rect
        for j in range(self.n):
            t = j / self.fps
            if t < self.board_in:
                continue
            rects = [self.rect] + [r for c in self.cards if t - c.at < SETTLED[self.drop]
                                   and (r := self._card_rect(c, t)) is not None]
            head, face = keep[j], faces[j]
            on_face = face is not None and any(r.intersects(face) for r in rects)
            on_head = head is not None and any(r.intersects(head) for r in rects)
            if on_face or (on_head and self.front is None):
                over_face += 1
            elif on_head:
                behind += 1
            drawn = [self.rect.grow(*halo(self.style, self.g.u))] + [
                r for c in self.cards if t - c.at < SETTLED[self.drop] and (r := self._card_drawn(c, t)) is not None]
            if any(not r.inside(zone) for r in drawn):
                outside += 1
        return {"frames_over_face": over_face, "frames_head_in_front": behind, "frames_outside_safe_zone": outside}


def _tap(rate: int, pitch: float) -> np.ndarray:
    """A soft wooden tap, a tenth of a second long, peak 1: a short tone that dies away fast, with a
    tiny click on its front. Made here, so the pack carries no sound files."""
    n = int(0.11 * rate)
    t = np.arange(n, dtype=np.float32) / rate
    body = np.sin(2 * np.pi * pitch * t) * np.exp(-t / 0.020) + 0.4 * np.sin(2 * np.pi * pitch * 2.37 * t) * np.exp(-t / 0.008)
    rng = np.random.default_rng(3)
    click = rng.standard_normal(n).astype(np.float32) * np.exp(-t / 0.0018) * 0.5
    s = (body + click) * np.clip(t / 0.0015, 0, 1)
    return (s / max(1e-6, float(np.abs(s).max()))).astype(np.float32)


# ---- the head, frame by frame ---------------------------------------------------------------------

def head_track(clip, timeline, fr, notes: list[str]) -> tuple[list[Optional[Rect]], list[Optional[Rect]]]:
    """For every output frame: the head to keep clear of (face, ears, chin, and the hair up to the
    crown the person mask finds) and the bare face box, in output pixels. The mask's crown is used
    only where it agrees with the face (between the brows and a tall hat's height above them), so a
    branch or a passer-by merged into the mask never pushes the board down; elsewhere the face's own
    estimate of the hair stands."""
    face = clip.face
    info = clip.info
    n_src = info.frames
    crowns: list[Optional[float]] = [None] * n_src
    if face.any():
        try:
            masks = clip.person()
            for i, m in enumerate(masks.frames()):
                if i >= n_src:
                    break
                h = face.head(i)
                if h is not None:
                    crowns[i] = masks.crown(m, h.x0, h.x1)
        except Exception as e:  # no mask: the face's estimate of the hair alone
            notes.append(f"the person could not be found ({e}); the head's top is estimated from the face")

    def keep(i: int) -> Optional[Rect]:
        h, b = face.head(i), face.box(i)
        if h is None or b is None:
            return None
        c = crowns[i]
        top = h.y0
        if c is not None and b.y0 - 0.95 * b.h <= c <= b.y0 - 0.05 * b.h:
            top = c
        return Rect(h.x0, top, h.x1, h.y1)

    n_out = timeline.output_frames(info.rate)
    out_keep, out_face = [], []
    for j in range(n_out):
        k = timeline.source_frame(j, info.rate, n_src)
        r, b = keep(k), face.box(k)
        out_keep.append(fr.to_output(r, k) if r is not None else None)
        out_face.append(fr.to_output(b, k) if b is not None else None)
    return out_keep, out_face


def bands(zone: Rect, keep: list[Optional[Rect]], faces: list[Optional[Rect]], W: int, H: int,
          behind: bool) -> dict[str, Rect]:
    """The free room on each side of the head, inside the safe zone, over the frames given.

    Two limits: the bare face box, which the board never reaches in any frame, and the head with its
    hair, which it keeps clear of in all but `BEHIND` of the frames when the person can be laid back
    over it (`behind`), else in every frame."""
    gap = GAP * H
    heads = [r for r in keep if r is not None]
    boxes = [r for r in faces if r is not None]
    if not heads or not boxes:  # nothing measured: where a head usually is not
        if H >= W:
            return {"top": Rect(zone.x0, zone.y0, zone.x1, zone.y0 + 0.30 * H)}
        return {"side": Rect(zone.x0, zone.y0, zone.x0 + 0.36 * W, zone.y1)}
    q = 100 * BEHIND if behind else 0.0
    arr = np.array([[r.x0, r.y0, r.x1, r.y1] for r in heads])
    lo = np.percentile(arr, q, axis=0)
    hi = np.percentile(arr, 100 - q, axis=0)
    face = Rect.around(boxes)
    top = min(lo[1] - gap, face.y0 - 2 * gap)
    left = min(lo[0] - gap, face.x0 - 2 * gap)
    right = max(hi[2] + gap, face.x1 + 2 * gap)
    bottom = max(hi[3] + gap, face.y1 + 2 * gap)
    out = {}
    if top - zone.y0 > 0:
        out["top"] = Rect(zone.x0, zone.y0, zone.x1, min(zone.y1, top))
    sides = [Rect(zone.x0, zone.y0, min(zone.x1, left), zone.y1), Rect(max(zone.x0, right), zone.y0, zone.x1, zone.y1)]
    side = max((r for r in sides if r.w > 0), key=lambda r: r.w, default=None)
    if side is not None:
        out["side"] = side
    if zone.y1 - bottom > 0:
        out["below"] = Rect(zone.x0, max(zone.y0, bottom), zone.x1, zone.y1)
    return out


class PersonFront:
    """The person, laid back over the board in the frames where the head rises into it. The mask is
    read in order while the video renders, and by seeking for a still."""

    def __init__(self, clip, timeline, fr, size: tuple[int, int]):
        self.masks = clip.person()
        self.timeline, self.fr, self.size = timeline, fr, size
        self.fps, self.n_src = clip.info.rate, clip.info.frames
        self._it = None
        self._k = -1
        self._m: Optional[np.ndarray] = None

    def _mask(self, k: int) -> np.ndarray:
        if self._it is not None and k < self._k:
            return self.masks.at(k)
        if self._it is None:
            self._it, self._k = iter(self.masks.frames()), -1
        while self._k < k:
            nxt = next(self._it, None)
            if nxt is None:
                break
            self._m, self._k = nxt, self._k + 1
        return self._m if self._m is not None else self.masks.at(k)

    def alpha(self, j: int, guide: np.ndarray) -> np.ndarray:
        """The person at output frame `j`, 0..1 at the output size, its edge following `guide`."""
        k = self.timeline.source_frame(j, self.fps, self.n_src)
        crop = self.fr.window(k) if self.fr.reframed else None
        return self.masks.full(self._mask(k), self.size, guide=guide, crop=crop)


# ---- the effect -----------------------------------------------------------------------------------

def _from_plan(plan: dict, style: str, duration: float) -> tuple[list[Tier], list[Card], str, float]:
    from . import PlanError

    rows = plan.get("tiers")
    cards = plan.get("cards")
    if not isinstance(rows, list) or not rows or not isinstance(cards, list) or not cards:
        raise PlanError("the plan needs its tiers and its cards; run once without --plan to get one to edit")
    spec = ",".join(f"{r['label']}={r['colour']}" if r.get("colour") else str(r["label"]) for r in rows)
    tiers = parse_tiers(spec, style)
    keys = {_tier_key(t.label): k for k, t in enumerate(tiers)}
    out = []
    for c in cards:
        if _tier_key(str(c.get("tier", ""))) not in keys:
            raise PlanError(f"a card is in row {c.get('tier')!r}, which the plan's tiers do not have")
        at = float(c.get("at", -1))
        if not 0 <= at <= duration:
            raise PlanError(f"the card {c.get('text')!r} has a time outside the clip ({at})")
        if not c.get("text") and not c.get("image"):
            raise PlanError("every card needs words or a picture")
        if c.get("image") and not Path(c["image"]).exists():
            raise PlanError(f"there is no picture at {c['image']}")
        out.append(Card(str(c.get("text") or ""), keys[_tier_key(str(c["tier"]))], at, c.get("image") or None,
                        str(c.get("how") or "given"), str(c.get("word") or "")))
    out.sort(key=lambda c: c.at)
    board = plan.get("board") or {}
    return tiers, out, str(plan.get("title") or ""), float(board.get("in", 0.0))


def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from ..core import framing as framing_mod
    from ..core.timeline import Timeline
    from . import Job, PlanError

    info = clip.info
    fr = framing_mod.plan(info.width, info.height, info.rate, info.frames, clip.face, shape=opts["frame"],
                          size=O.size_of(opts["size"]))
    W, H = fr.out_w, fr.out_h
    duration = info.frames / info.rate
    style, drop = opts["style"], opts["drop"]
    notes: list[str] = []
    try:
        if plan is not None:
            tiers, cards, title, board_in = _from_plan(plan, style, duration)
            if opts["title"]:
                title = opts["title"]
            words = clip.words
            for c in cards:  # a time edited in the plan lands on whatever word is said then
                near = min(words, key=lambda w: abs(w.start - c.at), default=None)
                c.word = near.clean if near is not None and abs(near.start - c.at) <= 0.25 else ""
        else:
            tiers = parse_tiers(opts["tiers"], style)
            items, tiers = parse_items(opts["items"], tiers, opts["tiers"].replace(" ", "") != DEFAULT_TIERS,
                                       Path(clip.path).parent, style)
            words = clip.words
            found = find_rankings(words, tiers)
            if items:
                cards, notes = time_items(items, words, found, duration, clip.quiet)
            else:
                if not found:
                    raise PlanError("no ranking was heard in the words (such as \"espresso is S tier\" or \"the latte goes "
                                    "in A\"). Say them that way, or give the items: items=\"S: …, …; A: …\"")
                cards = [Card(f.text, f.tier, f.at, None, "ranked", words[f.tier_word].clean) for f in found]
            title, board_in = opts["title"], 0.0
    except TierError as e:
        raise PlanError(str(e)) from None
    tl = Timeline.whole(duration).held(opts["hold"])
    n_out = tl.output_frames(info.rate)
    keep, faces = head_track(clip, tl, fr, notes)
    zone = SafeZone(W, H)
    j_in = int(max(0, math.floor(board_in * info.rate)))
    head = Rect.around(r for r in keep[j_in:] if r is not None)
    front = None
    if head is not None:
        try:
            front = PersonFront(clip, tl, fr, (W, H))
        except Exception as e:  # no mask: the board keeps clear of the head in every frame instead
            notes.append(f"the person could not be cut out ({e}); the board keeps clear of the head in every frame")
    room = bands(zone.rect, keep[j_in:], faces[j_in:], W, H, behind=front is not None)
    fits = {where: g for where, band in room.items()
            if (g := fit(band, tiers, cards, style, title, H, zone.rect)) is not None}
    want = opts["position"]
    if want != "auto" and want not in fits:
        notes.append(f"there is no room {'beside' if want == 'side' else want} the head for the board, so it goes "
                     f"where there is most room")
        want = "auto"
    if not fits:
        raise PlanError("there is no room anywhere off the face for the board on this clip; a shot with more room "
                        "above the head (or a wider one) makes room for it")
    if want == "auto":
        pool = {w: g for w, g in fits.items() if not g.cut} or fits  # every name whole, where any place allows
        best = max(g.u for g in pool.values())
        for where in ("top", "side", "below"):  # above the head unless somewhere else is clearly roomier
            if where in pool and pool[where].u >= 0.9 * best:
                want = where
                break
    geo, band = fits[want], room[want]
    if geo.cut:
        notes.append("some names are too long for the room off the face and are cut short; shorter names, or "
                     "position=top, keep them whole")
    # Centred on the head across, and in the middle of its room up and down.
    cx = head.cx if (head is not None and want != "side") else band.cx
    band = halo_room(band, zone.rect, halo(style, geo.u))  # where the board itself may go, its shadow inside the zone
    x = min(max(cx - geo.w / 2, band.x0), band.x1 - geo.w)
    y = band.y0 + (band.h - geo.h) / 2
    board = Board(tiers, cards, geo, x, y, style, drop, title, board_in, (W, H), info.rate, n_out, zone, keep, front)
    if style == "glass":
        from ..core import video

        picks = np.linspace(0, info.frames - 1, 6).round().astype(int)
        board.set_smoke([fr.apply(video.read_frame(info, int(k)), int(k)) for k in picks])
    checks = board.report(keep, faces)
    if geo.text_px < READABLE * H:
        notes.append(f"the room off the face is small on this clip, so the cards' words are {geo.text_px} px "
                     f"({geo.text_px / H:.1%} of the height, under the {READABLE:.2%} that reads easily on a phone); "
                     "a shot with more room above the head, fewer rows, or size=1080x1920 makes them clearer")
    by_how = {}
    for c in cards:
        by_how[c.how] = by_how.get(c.how, 0) + 1
    hows = {"ranked": "ranked in the words", "said": "on the words that name them", "given": "at the times given",
            "placed": "placed between the others (never said)"}
    report = {
        "board": f"{want}, {geo.w:.0f}x{geo.h:.0f} px ({geo.h / H:.0%} of the height), rows {geo.u:.0f} px, "
                 f"words {geo.text_px} px",
        "cards": f"{len(cards)}: " + ", ".join(f"{n} {hows[k]}" for k, n in by_how.items()),
        "lands": " | ".join(f"{tiers[c.tier].label}: {c.text or Path(c.image or '').stem} {board.lands(c):.2f} s"
                            + (f" ('{c.word}')" if c.word else "") for c in cards),
        **checks,
        "placed": want,
        "warnings": notes,
    }
    stills = _moments(board, tl, cards)
    out_plan = {
        "effect": NAME, "clip": clip.path, "options": opts,
        "output": {"width": W, "height": H, "fps": round(info.rate, 3), "seconds": round(tl.duration, 3),
                   "framing": fr.note, "hold": tl.hold},
        "title": title,
        "board": {"in": board_in, "position": want, "rect": board.rect.as_list(), "row_px": round(geo.u, 1),
                  "words_px": geo.text_px},
        "tiers": [{"label": t.label, "colour": "#%02x%02x%02x" % t.colour} for t in tiers],
        "cards": [{"tier": tiers[c.tier].label, "text": c.text, "image": c.image, "at": round(c.at, 3),
                   "lands": round(board.lands(c), 3), "how": c.how, "word": c.word}
                  | ({"asked": c.asked} if c.asked is not None and abs(c.asked - c.at) > 1e-3 else {})
                  for c in cards],
        "sheet_times": stills,
    }
    picks = [stills[0], stills[len(stills) // 2], stills[-1]]
    return Job(NAME, clip, tl, fr, board.draw, out_plan, checks=report, still_times=lambda: picks,
               sound=board.mix if opts["sound"] else None)


def _moments(board: Board, tl, cards: list[Card]) -> list[float]:
    """Six moments that show the board best: empty and settled, four just after cards have landed
    (moved past any card still in the air, so each shows the board at rest), and the held finished
    board."""
    end = tl.duration - 0.5 / board.fps
    settle = SETTLED[board.drop]

    def at_rest(t: float) -> float:
        for _ in range(len(cards) + 1):
            flying = [c for c in cards if c.at <= t < c.at + settle]
            if not flying:
                break
            t = max(c.at + settle for c in flying) + 0.05
        return min(end, t)

    first = min([board.board_in + BOARD_IN + 0.25] + [max(board.board_in + 0.4, c.at - 0.05) for c in cards[:1]])
    settled = [at_rest(c.at + settle + 0.12) for c in cards]
    picks = [settled[round(k * (len(settled) - 1) / 3)] for k in range(4)] if len(settled) >= 4 else settled
    hold = tl.kept + tl.hold / 2 if tl.hold > 0 else end
    out = sorted(set([round(first, 3)] + [round(p, 3) for p in picks] + [round(min(end, hold), 3)]))
    for c in cards:  # fewer than six so far: more of the cards at rest
        if len(out) >= 6:
            break
        t = round(at_rest(c.at + settle + 0.12), 3)
        if all(abs(t - o) > 0.2 for o in out):
            out = sorted(out + [t])
    return out[:6]
