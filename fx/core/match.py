"""Matching a person's own files to the words they say: which file goes on which word.

The pack never looks at the pictures to decide. A file lands on a word because the person said so,
or because its name or its tags name what is said:

1. **`map`**: a word said -> a file, given by the person (`lake=lake.mp4; city=rooftops.mp4`).
2. **The file's name and `tags.txt`**: the words in a file's name (`rice-fields.mov` -> rice,
   fields) and the phrases a `tags.txt` beside it lists for it, matched to the words said, a plural
   matching its singular. A file named only with a camera's number (`IMG_5922.MOV`) says nothing,
   so it is "unnamed"; a named file whose words are never said is left out, with that reason.
3. **The things named**: the strongest words that name a thing (nouns, places, things, found by
   rules: never grammar, filler or a doing word, never a word said again). The unnamed files go
   on these, in folder order, once an effect has placed them in time (`folder_order`).

**Never on a word taken back.** Given what the speaker took back (`moments.taken_back`, from the
clip's whole words), a file's name or tags never land on a word the speaker replaced ("the lively
town, or the lively city": a clip tagged "town" is not laid on "town"), a first attempt said again
or an aside about the recording, and the things named are weighed by the shared picker's measure
(`moments.strength`), so the word said instead counts as what its line says new. A word the person
maps lands where it is not taken back, or where it is taken back when it is said nowhere else.

**A logo** names one brand, so it lands only where that brand's whole name is said: `AskEditor.png`
on "AskEditor" or "Ask Editor", never on "ask" or "editor" alone (`brand_words`, `said_whole`). An
item marked `whole` is matched this way, and is never laid on one of the things named.

What an effect does with it:

    pool, named = match.pool(spec, mapping, KIND, load)      # the files, and the words mapped to them
    said = match.said(clip.words)                             # the words a file may land on
    back = moments.taken_back(clip.words, clip.quiet)       # what the speaker took back
    cands, unnamed = match.candidates(said, quiet, pool, named, "keywords", left, back)
    ... the effect places the candidates in time, by its own rules ...
    for moment, idx in match.folder_order(keyword_moments, unnamed): ...

Every item in a pool is the effect's own (a B-roll clip, a picture): anything with `path`, `names`
(the words of its file name), `tags` (its phrases in tags.txt) and `named` (either is there), and
optionally `whole` (True: its names are one brand's name, said whole).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

from . import moments
from .words import HESITATIONS, Word, normalize

# Words in a file name that say nothing about what a file shows: cameras, phones, and editing.
_CAMERA = set("""
img dsc dscf dscn dji mvi mov pxl vid video videos gopr gp clip clips broll roll final edit edited copy export
untitled new raw take shot cam camera iphone android photo pic image hdr sdr lut graded proxy screen recording
footage stock version
screenshot screenshots snapshot capture photos pics images picture pictures scan scanned download downloaded
file unnamed
""".split())
# Apps whose own names a phone or a laptop writes, joined, into the files it saves ("WhatsApp Image…",
# "CleanShot 2026…"): taken out before a name is split into its words, so no half of one is a name.
_SAVED_BY = re.compile(r"(?i)whatsapp|cleanshot|screenshot|snapchat|telegram|messenger")
_STOP = set("""
a an the and or of in on at to for with from by into my your our their his her its this that these those
is are was were be been it i we you they me us them
""".split())
# Doing words: never the thing a file shows.
_DOING = set("""
make makes made making build builds built create creates created show shows grow grows learn learns use uses
start starts work works run runs change changes find finds get gets give gives take takes want wants need needs
try tries love loves like likes watch watches improve improves help helps edit edits went go goes going come
comes came talk talks talked think thinks thought say says said
""".split())
_SUBJECTS = set("that which who it he she they we you i".split())
# Words that describe or count rather than name: never what a file shows.
_NOT_THINGS = set("""
good better best great nice bad worse worst real big bigger biggest small smaller little large huge new newer old
older young hot cold cool warm easy easier hard harder simple fast faster quick slow slower cheap cheaper expensive
high higher low lower long longer short shorter early late ready sure true false right wrong happy sad different
same whole entire full empty free main important actual possible amazing awesome beautiful lovely perfect crazy
second seconds minute minutes hour hours day days week weeks month months year years time times moment moments
""".split())
_HELPERS = set("was were is are be been being had have has got gets".split())
# Said straight before a word that describes what was just named ("the sea was calm", "it looks great").
_BE = set("is are was were am be been being seems seemed looks looked feels felt became becomes stays stayed".split())
# Said just before a person's own name ("it's Sam", "I'm Jo", "my name is…"): the person, not a file.
_INTRO = set("it's i'm im name's".split())
KEY_MIN = 7.0  # a word must score this to carry a file of its own (the pack's word scorer)
RUN_MIN = 4.0  # words scoring this belong to one named thing ("lively city", "production quality")


# ---- the words -----------------------------------------------------------------------------------

def stem(k: str) -> str:
    """A word in its singular, near enough to match: cities -> city, trains -> train, boxes -> box."""
    if len(k) > 4 and k.endswith("ies"):
        return k[:-3] + "y"
    if len(k) > 4 and k.endswith(("ches", "shes", "xes", "sses", "zes")):
        return k[:-2]
    if len(k) > 3 and k.endswith("s") and not k.endswith(("ss", "us", "is")):
        return k[:-1]
    return k


def name_words(path: str | Path) -> tuple[str, ...]:
    """The words a file name says about what the file shows: `rice-fields_mountains.mov` -> rice,
    fields, mountains; `IMG_5922.MOV` -> nothing."""
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", _SAVED_BY.sub(" ", Path(path).stem))
    out = []
    for tok in re.split(r"[^A-Za-z]+", s):
        t = tok.lower()
        if len(t) >= 3 and t not in _CAMERA and t not in _STOP and t not in out:
            out.append(t)
    return tuple(out)


def phrase(text: str) -> tuple[str, ...]:
    """A phrase as it is matched: each word in its singular, the little words left out."""
    return tuple(stem(normalize(x)) for x in str(text).split() if normalize(x) and normalize(x) not in _STOP)


# Words in a logo's file name that describe the file, not the brand: `askeditor-logo-dark.png`.
_LOGO_FILE = set("""
logo logos logotype icon icons mark wordmark lettermark symbol brand brandmark badge emblem glyph favicon
light dark white black mono monochrome colour color colored coloured full horizontal vertical stacked square round
circle transparent primary secondary main alt small large big rgb cmyk svg png print web app the
""".split())


def brand_words(path: str | Path) -> tuple[str, ...]:
    """A logo's brand, from its file name, as the words its name is said in: `AskEditor.png` ->
    ask, editor; `notion-logo-dark.png` -> notion; `3M.png` -> 3m. Short names and figures are
    kept (a brand can be two letters); what describes the file (logo, dark, icon) is not."""
    stem_ = Path(path).stem
    if _SAVED_BY.search(stem_):
        return ()  # a file an app named: its name is a date, not a brand
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", stem_)
    out = []
    for tok in re.split(r"[^A-Za-z0-9]+", s):
        t = tok.lower()
        if not t or t in _LOGO_FILE or t in _CAMERA or t.isdigit():
            continue
        out.append(t)
    if all(t in _STOP for t in out):
        return ()  # only the little words of a name a camera or an app wrote ("Screen Shot … at …")
    return tuple(out)


def _squash(key: str) -> str:
    """A said word with nothing but its letters and figures, and no possessive: AskEditor's -> askeditor."""
    k = key.lower().replace("’", "'")
    if k.endswith("'s"):
        k = k[:-2]
    elif k.endswith("s'"):
        k = k[:-1]
    return re.sub(r"[^a-z0-9]", "", k)


def said_whole(said_words: list[Word], brand: tuple[str, ...]) -> list[int]:
    """Where a brand's whole name is said, by the index of its first word: as one word
    ("AskEditor", "Ask-Editor's") or as its words in a row ("Ask Editor"), heard in at most one
    piece more than it has."""
    target = "".join(brand)
    if not target:
        return []
    keys = [_squash(w.key) for w in said_words]
    out = []
    for i in range(len(keys)):
        acc = ""
        for j in range(i, min(len(keys), i + len(brand) + 1)):
            acc += keys[j]
            if acc == target:
                out.append(i)
                break
            if not target.startswith(acc):
                break
    return out


def said(words: list[Word]) -> list[Word]:
    """The words a file may land on: every word heard for sure, without the ums."""
    return [w for w in words if w.key and w.key not in HESITATIONS and not getattr(w, "unsure", "")]


def occurrences(stems: list[str], wanted: tuple) -> list[int]:
    """Where the run of words `wanted` is said, by the index of its first word in `stems`."""
    n = len(wanted)
    if not n:
        return []
    return [i for i in range(len(stems) - n + 1) if tuple(stems[i:i + n]) == wanted]


# ---- the files -----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Kind:
    """The kind of file an effect takes, and how its messages name it."""
    exts: frozenset  # the endings it reads, lower case with the dot
    plural: str  # "clips or photos": there are no … in a folder
    one: str  # "B-roll": there is no … at a path
    reads: str  # "a clip or a photo this pack reads (…)": a file that is not one is not …
    yours: str  # "your B-roll": map names a file which is not among …
    ask: str  # what to say when nothing was given


def natural(name: str) -> list:
    """A file name's sort key, numbers in number order: IMG_2 before IMG_10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def files(spec: str, kind: Kind) -> list[Path]:
    """The files `spec` names: a folder's files of this kind in name order, or a list of files."""
    spec = str(spec).strip().strip("\"'")
    if not spec:
        return []
    one = Path(os.path.expanduser(spec))
    if one.is_dir():
        found = sorted((p for p in one.iterdir() if p.suffix.lower() in kind.exts and not p.name.startswith(".")),
                       key=lambda p: natural(p.name))
        if not found:
            raise ValueError(f"there are no {kind.plural} in {one}")
        return found
    out = []
    for part in [p.strip().strip("\"'") for p in re.split(r"[,;\n]", spec) if p.strip()]:
        p = Path(os.path.expanduser(part))
        if p.is_dir():
            out += files(str(p), kind)
        elif not p.exists():
            raise ValueError(f"there is no {kind.one} at {p}")
        elif p.suffix.lower() not in kind.exts:
            raise ValueError(f"{p.name} is not {kind.reads}. A HEIC photo needs saving as JPEG first")
        else:
            out.append(p)
    return out


@lru_cache(maxsize=64)
def read_tags(folder: str) -> dict[str, tuple]:
    """`tags.txt` in a folder: one file a line, `lake.mp4: lake, water, sunset` (a name without its
    ending works too). Lines starting with # are notes."""
    f = Path(folder) / "tags.txt"
    if not f.exists():
        return {}
    out: dict[str, tuple] = {}
    for line in f.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(.+?)\s*[:=]\s*(.*)$", line)
        if not m:
            continue
        phrases = tuple(p for p in (phrase(x) for x in re.split(r"[,;]", m.group(2))) if p)
        if phrases:
            out[m.group(1).strip().strip("\"'").lower()] = phrases
    return out


def tags_for(path: Path) -> tuple:
    """The phrases the `tags.txt` beside `path` lists for it."""
    tags = read_tags(str(path.parent))
    return tags.get(path.name.lower()) or tags.get(path.stem.lower()) or ()


def parse_map(text: str, noun: str = "clip",
              examples: tuple = (("lake", "lake.mp4"), ("city", "rooftops.mp4"))) -> list[tuple[str, str]]:
    """`{"lake": "lake.mp4"}` or `lake=lake.mp4; city=rooftops.mp4` -> [("lake", "lake.mp4"), …].
    `noun` and `examples` are how the messages name a file. Raises ValueError, saying what would work."""
    (w1, f1), (w2, f2) = examples
    t = str(text).strip()
    if not t:
        return []
    if t.startswith("{"):
        try:
            data = json.loads(t)
        except ValueError:
            raise ValueError(f'map is not valid JSON; write {{"{w1}": "{f1}"}} or {w1}={f1}; {w2}={f2}') from None
        if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
            raise ValueError(f'map needs words and {noun} names, such as {{"{w1}": "{f1}"}}')
        pairs = list(data.items())
    else:
        pairs = []
        for part in re.split(r"[;\n]|,(?=[^,]*=)", t):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"map pairs go as word={noun} (got {part!r}); for example {w1}={f1}; {w2}={f2}")
            word, name = part.split("=", 1)
            pairs.append((word, name))
    out = []
    for word, name in pairs:
        word, name = word.strip().strip("\"'"), name.strip().strip("\"'")
        if not [normalize(x) for x in word.split() if normalize(x)]:
            raise ValueError(f"map needs a word you say before each {noun} (got {word!r})")
        if not name:
            raise ValueError(f"map needs a {noun} for “{word}”")
        out.append((word, name))
    return out


def pool(spec: str, mapping: list[tuple[str, str]], kind: Kind, load: Callable[[Path], object]):
    """The files in folder order, each made by `load`, and which file each mapped word names
    (word -> index). A file the map gives by its full path joins the pool after the folder's."""
    paths = files(spec, kind)
    if not paths and not mapping:
        raise ValueError(kind.ask)
    items = [load(p) for p in paths]
    where = {str(Path(it.path).resolve()): i for i, it in enumerate(items)}
    by_name: dict[str, list[int]] = {}
    for i, it in enumerate(items):
        by_name.setdefault(Path(it.path).name.lower(), []).append(i)
        by_name.setdefault(Path(it.path).stem.lower(), []).append(i)
    named: dict[str, int] = {}
    for word, name in mapping:
        p = Path(os.path.expanduser(name))
        idx = None
        if p.exists() and p.is_file():
            idx = where.get(str(p.resolve()))
            if idx is None:
                items.append(load(p))
                idx = len(items) - 1
                where[str(p.resolve())] = idx
        else:
            hits = by_name.get(name.lower()) or by_name.get(Path(name).stem.lower())
            if hits:
                idx = hits[0]
        if idx is None:
            raise ValueError(f"map names {name}, which is not among {kind.yours}"
                             + (f" ({spec})" if spec else "") + "; give its full path or put it in the folder")
        named[word] = idx
    return items, named


# ---- the things named ----------------------------------------------------------------------------

@dataclass
class Keyword:
    index: int
    start: float
    word: str
    score: float


def keyword_scores(words: list[Word], quiet, back: Optional[moments.TakenBack] = None) -> list[float]:
    """How well each word names a thing a picture could show: the shared picker's measure
    (`moments.strength`: specific, at the end of its phrase, after "the" or "a"; nothing for a word
    in `back`, what the speaker took back), less for doing words."""
    back = moments.TakenBack() if back is None else back
    scores = [0.0] * len(words)
    for ln in moments.lines(words, quiet):
        for i in ln:
            s = moments.strength(words, i, ln, quiet, back)
            if s <= 0:
                continue
            k = words[i].key
            if k in _DOING or (k.endswith("ing") and len(k) > 5):
                s -= 4
            if i > 0 and words[i - 1].key in _SUBJECTS and k.endswith("s"):
                s -= 3  # "an editor that edits": doing, not naming
            if i > 0 and len(k) > 4 and k.endswith("ed") and words[i - 1].key in _SUBJECTS | _HELPERS:
                s -= 5  # "than I expected", "it was packed": doing or describing
            if i > 0 and words[i - 1].key in _BE and not bare(words[i])[:1].isupper():
                s -= 5  # "my hotel was tiny", "the sea was calm": describing what was named (a name stays)
            if k in _NOT_THINGS:
                s = 0.0
            if (i > 0 and words[i - 1].key in _INTRO) or (i > 1 and words[i - 2].key == "name"
                                                         and words[i - 1].key == "is"):
                s = 0.0  # introducing yourself: your face, not a picture
            scores[i] = max(0.0, s)
    return scores


def bare(word: Word) -> str:
    """A word as a check or a sheet shows it, with nothing at either edge (the captions' own rule)."""
    from ..effects.captions import bare as captions_bare

    return captions_bare(word)


def keywords(words: list[Word], quiet, back: Optional[moments.TakenBack] = None) -> list[Keyword]:
    """The things named, in order: a run of naming words ("the lively city") is one thing, landing
    on its strongest word (a tie goes to the later one, where English puts the thing named). A
    thing said again is not new. `back`: what the speaker took back, never a thing named."""
    scores = keyword_scores(words, quiet, back)
    out, seen = [], set()
    i, n = 0, len(words)
    while i < n:
        if scores[i] < RUN_MIN:
            i += 1
            continue
        j = i
        while j + 1 < n and scores[j + 1] >= RUN_MIN and not words[j].ends_clause:
            j += 1
        best = max(range(i, j + 1), key=lambda m: (scores[m], m))
        k = stem(words[best].key)
        if scores[best] >= KEY_MIN and k not in seen:
            out.append(Keyword(best, float(words[best].start), bare(words[best]), float(scores[best])))
        seen.add(k)
        i = j + 1
    return out


# ---- which file on which word --------------------------------------------------------------------

@dataclass
class Candidate:
    """A moment a file could land on: the word's start, the file (None: one of the things named,
    waiting for an unnamed file), and why."""
    start: float
    item: Optional[int]
    word: str
    why: str  # map, tags, name, keyword
    priority: int  # 3: mapped; 2: its name or tags said; 1: a thing named
    score: float = 0.0


def candidates(said_words: list[Word], quiet, items: list, named: dict[str, int], on: str,
               left: dict[int, str], back: Optional[moments.TakenBack] = None) -> tuple[list[Candidate], list[int]]:
    """Every moment each file could land on, and the unnamed files (in folder order) for the things
    named. `on` is "keywords" (map, names, tags and the things named) or "map" (map, names and tags
    only). `left` holds the files already ruled out (index -> why), and gains the ones that can
    never land, each with its reason; a file in it is never matched. `back`: what the speaker took
    back (`moments.taken_back` of the clip's whole words): a name or a tag never lands there, and a
    word mapped lands there only when it is said nowhere else."""
    back = moments.TakenBack() if back is None else back
    stems = [stem(w.key) for w in said_words]

    def kept(occ: list[int], n: int) -> list[int]:
        return [i for i in occ if not any(w in back for w in said_words[i:i + max(1, n)])]

    cands: list[Candidate] = []
    mapped = set(named.values())
    for word, idx in named.items():
        occ = occurrences(stems, phrase(word))
        if not occ:
            left.setdefault(idx, f"map: “{word}” is never said")
        occ = kept(occ, len(phrase(word))) or occ
        for i in occ:
            cands.append(Candidate(float(said_words[i].start), idx, bare(said_words[i]), "map", 3))
    for idx, it in enumerate(items):
        if idx in mapped or idx in left or not it.named:
            continue
        heard = [(occurrences(stems, ph), len(ph), "tags") for ph in it.tags]
        if getattr(it, "whole", False):
            heard.append((said_whole(said_words, it.names), 1, "name"))
        else:
            heard += [(occurrences(stems, (stem(n),)), 1, "name") for n in it.names]
        hits = [(i, why) for occ, n, why in heard for i in kept(occ, n)]
        if not hits:
            left[idx] = ("its name and tags are said only where the speaker took them back"
                         if any(occ for occ, _, _ in heard) else "its name and tags are never said")
            continue
        for i, why in sorted(set(hits)):
            cands.append(Candidate(float(said_words[i].start), idx, bare(said_words[i]), why, 2))
    if on == "keywords":
        for kw in keywords(said_words, quiet, back):
            cands.append(Candidate(kw.start, None, kw.word, "keyword", 1, kw.score))
    for i, it in enumerate(items):
        if getattr(it, "whole", False) and not it.named and i not in mapped:
            left.setdefault(i, "a logo lands only on its brand's name, and this file's name gives none")
    unnamed = [i for i, it in enumerate(items) if not it.named and i not in mapped and i not in left]
    if on == "map":
        for i in unnamed:
            left[i] = "nothing in its name to match, and on=map"
        unnamed = []
    return cands, unnamed


def folder_order(moments: list, unnamed: list[int]) -> list[tuple[object, int]]:
    """The unnamed files paired with the keyword moments an effect placed: the first file in folder
    order on the earliest moment, and so on (each moment has a `start`)."""
    return list(zip(sorted(moments, key=lambda c: c.start), unnamed))
