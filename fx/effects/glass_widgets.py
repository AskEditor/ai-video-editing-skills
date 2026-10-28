"""glass-widgets: a cluster of frosted glass widgets round the speaker's head, some floating in front
and some tucked behind it, each carrying something of the speaker's own: a figure they say, their
name, the line that makes the point, a ring filling to a share, a small chart, a simple sign.

How it decides, in order:

1. **What the widgets say.** `widgets` lists them (`number: 197K followers | name: Sam | note: Shot
   in one take | progress: 72% edited | stat: 2K,15K,197K views | icon: star`), or, left on auto,
   they come from the clip: each figure said (a run of them is a small chart, a share of a hundred
   is a ring), the name given in `name`, and the line that makes the point, with a sign drawn
   beside it when its words name one (a city, a pin; a video, a play button). Nothing is made up:
   no badge, no brand, no claim the speaker did not make.
2. **When.** Each widget lands on the words that say it, timed from where the voice begins (the
   transcriber marks a word late; the core corrects it); the others land one after another on the
   words after it, never two at once. A figure counts up and a ring fills as it lands. They stay
   `hold` seconds after the last has landed, then leave together, or hold to the end.
3. **Where.** One designed cluster round the head, never over the face and inside the safe zone,
   measured on the person in every frame each widget is up, where its drift takes it (the head
   moves). Every widget is set at one type size, as large as the room allows and never smaller than
   a phone shows clearly unless there is no other way (the run says so); the first figure or line
   leads, a little larger, when that costs nothing. The larger widgets tuck behind the hair and
   shoulders with the person mask, their writing never hidden by the person; the smaller float in
   front. A few arrangements are tried and the most balanced, compact one kept.
4. **Depth.** The front and back layers drift a little as the head moves, by different amounts, and
   the back is a touch softer, so the cluster reads as layers in the room. `depth=0` is flat and
   still, every widget in front.
5. **The glass.** What is behind each widget is blurred on every frame and bent at its rim (behind
   the person, the room is continued in under them first, so their hair never smears into the
   glass), toned so its type reads at about 4.5 : 1 or better, edge-lit, with a soft shadow. One
   band of light crosses the cluster once it has landed, and a soft glass chime sounds well under
   the voice as each widget arrives, each a step higher.
"""

from __future__ import annotations

import json
import math
import re
import zlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core import amounts as num
from ..core import glass as glassmod
from ..core import moments
from ..core import text as textmod
from ..core import words as wordsmod
from ..core.compose import Sprite, behind, over, without_person
from ..core.layout import Rect, SafeZone
from ..core.words import HESITATIONS, Word, normalize, unheard
from . import chart as chartmod
from . import options as O
from .captions import head_in_output
from .headline import _parse_time, _soft_reveal, find_said, keepout_in_output, tokens_from_words
from .layers import split as split_items
from .words_behind import EDGE, REACH, MaskStore, Q, _Hold, _When, refine_edge

NAME = "glass-widgets"
KINDS = ("number", "stat", "name", "note", "progress", "icon")
ICONS = ("check", "star", "play", "heart", "bolt", "spark", "clock", "up", "pin", "chat", "eye", "user", "mic",
         "calendar")
LAYOUTS = ("auto", "around", "above", "bento")
TONES = ("auto", "dark", "light", "tint")


@dataclass(frozen=True)
class _Widgets:
    help: str

    def parse(self, raw) -> str:
        v = str(raw).strip()
        if v.lower() in ("", "auto"):
            return "auto"
        parse_widgets(v, "", strict=False)  # a list that cannot be read is refused now, in plain words
        return v


OPTIONS = {
    "widgets": _Widgets("auto: made from what you say (each figure, the name you give, the line that makes the "
                        "point) · or your own, separated by | : number: 197K followers | name: Sam | note: Shot "
                        "in one take | progress: 72% edited | stat: 2K,15K,197K views | icon: star. Add @ and words "
                        "or seconds to land one there: number: 197K followers @ \"come to life\""),
    "name": O.Text("your name or handle for the name pill, such as Sam or @sam; when you say it, the pill lands "
                   "on it. Empty: no name pill unless the widgets list one"),
    "layout": O.Choice(LAYOUTS, "auto and around: round your head, the larger widgets tucked behind it · above: a "
                                "row above your head, all in front · bento: tiles lined up in rows and columns "
                                "round your head"),
    "depth": O.Number(0, 1, "how much depth: 0 is flat (every widget in front, still); above it the larger "
                            "widgets tuck behind you, and the higher it is the further the layers drift apart as "
                            "your head moves"),
    "tone": O.Choice(TONES, "auto: dark glass over a bright room, light glass over a dark one · dark: smoked glass "
                            "with white type · light: frosted white glass with dark type · tint: glass tinted with "
                            "the accent"),
    "accent": O.Colour("the rings, the chips and the tinted glass: auto, a name such as yellow, or a hex colour"),
    "when": _When("auto: each widget lands on the words that say it, the rest on the words after · words that are "
                  "said: they arrive across them · a time in seconds: they start then"),
    "hold": _Hold("auto: they stay 2.6 seconds after the last has landed, then leave together · seconds such as 4 "
                  "· or end: they hold to the end"),
    "sound": O.Switch("on: a soft glass chime as each widget lands, well under your voice · off: your sound "
                      "exactly as it was"),
    "fix": O.Text("misheard words as wrong=right, separated by ;  e.g. \"Jon=John\""),
}

# The runs `tests/matrix.py glass-widgets` makes. D (a plain wall, room above the head) is the best
# case; B, A and C are the controlled three; F moves. None of the clips says a figure, so the figures
# are given and land across words that are said there; finding them in speech is `chart`'s and
# `amounts`' own (tests/amounts.py).
MATRIX = [
    ("D", "D", {"name": "Eben", "widgets": "note: The lively city | number: 12.4K steps | icon: pin",
                "when": "the lively city come to life"}),
    ("B", "B", {"name": "Eben"}),
    ("A", "A", {"widgets": "number: 197K views | note: Hooks that draw people in", "when": "what draws people in"}),
    ("C", "C", {"name": "Eben"}),
    ("F", "F", {"name": "Eben", "widgets": "number: 48 stops | progress: 72% there | icon: pin", "when": "1.2"}),
    ("D-bento", "D", {"name": "Eben", "layout": "bento", "tone": "light",
                      "widgets": "stat: 2K,15K,197K followers | progress: 72% edited | note: The lively city",
                      "when": "the lively city come to life"}),
    # five widgets on one close shot: they come smaller, and the run says so
    ("D-five", "D", {"name": "Eben", "widgets": "note: The lively city | number: 12.4K steps | progress: 3/5 days | "
                                                "icon: pin", "when": "the lively city come to life"}),
    # the speaker's own name, said on the clip, and the line that makes the point, on violet glass
    ("C-tint", "C", {"name": "Eben", "tone": "tint", "accent": "#6b5cff"}),
]

# ---- the design, in pixels of a frame 1080 wide at size 1 (`s` is that scale on the real frame) ----
PAD = chartmod.PAD
RADIUS = chartmod.RADIUS
LABEL_PX = chartmod.LABEL_PX["counter"]
BODY_PX = 46  # a note
NAME_PX = 44
PILL_H = 100
CHIP_D = 72  # the initials in the name pill
ICON_TILE = 172
ICON_CHIP = 108
ICON_GLYPH = 58
CAPTION_PX = 30
RING_D = 156
RING_W = 17
BAR_H = 20
NOTE_MAX_W = 560
NOTE_MAX_CHARS = 64
LABEL_MAX_CHARS = 26
FACE_LABEL = chartmod.FACE_LABEL
FACE_NUM = chartmod.FACE_NUM
FACE_BODY = "poppins-semibold"
FACE_NAME = "poppins-semibold"

# ---- time -------------------------------------------------------------------------------------------
ENTER = 0.46  # a widget grows in over this long
GROW_FROM = 0.86
CONTENT_LAG = 0.3  # share of its entrance before its writing starts to show
EXIT = 0.34
EXIT_STAGGER = 0.05
GAP = 0.42  # the least time between two arrivals
COUNT = 0.85  # a ring fills, or a bar steps, over this long before its value lands
HOLD = 2.6
MIN_ON = 1.2  # after the last arrival, at least
SWEEP_AFTER = 0.3
SWEEP = 0.95
CHIME_UNDER = 19.0  # dB under the voice's loud syllables, as heard

# ---- place --------------------------------------------------------------------------------------------
K_MAX, K_MIN, K_STEP = 1.0, 0.5, 0.94
K_READ = 0.8  # a label at about 29 pixels of 1080: the smallest a phone shows clearly
K_GIVEN = 0.68  # widgets a person listed may come this small before one is left out
GUTTER = 22  # design pixels between two widgets
FACE_MARGIN = 0.01  # kept clear round the face, a share of the frame's height
HUG_FRONT = 0.016  # the gap a widget in front keeps from the head, a share of the frame's height
CONTENT_HIDDEN = 0.02  # the most of a widget's writing the person may hide, at any moment
TILE_HIDDEN = 0.55  # the most of a widget behind the person may cover, at any moment
TUCK = (0.08, 0.45)  # a widget behind reads as behind when the person covers this share of it, on average

# ---- depth --------------------------------------------------------------------------------------------
FOLLOW = {"front": 0.5, "back": 0.18}  # how much of the head's movement each layer follows, at depth 1
FLOAT = 4.5  # design pixels of slow float, at depth 1
SOFT = 1.4  # design pixels of softness on the back layer, at depth 1
DRIFT_MAX = 0.04  # a share of the frame's width
SHADOW = {"front": 0.42, "back": 0.26}  # nearer the lens, a deeper shadow

ACCENT = {"dark": "#57e3b0", "light": "#0b8f62", "tint": "#6b5cff"}
DARK_INK, DARK_SOFT = (255, 255, 255), (226, 229, 237)
LIGHT_INK, LIGHT_SOFT = (22, 24, 32), (62, 66, 80)

_ease_out = chartmod._ease_out
_ease_in_out = chartmod._ease_in_out
_back_out = chartmod._back_out


# ---- what the widgets are -----------------------------------------------------------------------------

@dataclass
class Spec:
    kind: str
    text: str = ""  # the label, the note, the name, the icon's caption
    value: str = ""  # the figure ("197K"), the values ("2K,15K,197K"), the share ("72%", "3/5"), the sign
    when: str = ""  # words that are said, or seconds: where this one lands
    layer: str = ""  # front or back; empty: decided
    at: Optional[float] = None  # where it is said, when it is (seconds, from the corrected words)
    said: str = ""  # the words it was heard on
    found: str = "given"
    style: str = ""  # a small chart's: line or bars

    def label(self) -> str:
        v = f" {self.value}" if self.value and self.kind != "name" else ""
        t = f" “{self.text}”" if self.text else ""
        return f"{self.kind}{v}{t}"


KIND_WORDS = {"figure": "number", "count": "number", "counter": "number", "big": "number", "chart": "stat",
              "graph": "stat", "bars": "stat", "line": "note", "text": "note", "quote": "note", "handle": "name",
              "pill": "name", "ring": "progress", "bar": "progress", "percent": "progress", "sign": "icon",
              "symbol": "icon", "emoji": "icon"}
ICON_WORDS = {"tick": "check", "done": "check", "location": "pin", "place": "pin", "map": "pin", "time": "clock",
              "lightning": "bolt", "fast": "bolt", "sparkle": "spark", "magic": "spark", "arrow": "up",
              "trend": "up", "growth": "up", "person": "user", "people": "user", "comment": "chat",
              "message": "chat", "video": "play", "love": "heart", "like": "heart", "views": "eye",
              "voice": "mic", "date": "calendar", "day": "calendar"}
# A sign drawn beside a note when its words name one (auto). Plain shapes we draw; never a logo.
SIGN_FOR = [
    ("pin", "city town hotel travel trip street road country beach airport station train flight place"),
    ("play", "video videos film films reel reels clip clips watch movie"),
    ("star", "quality best pro professional professionally great top premium"),
    ("bolt", "fast quick quickly instant instantly speed seconds"),
    ("clock", "time minutes minute hour hours while schedule"),
    ("up", "grow growth growing increase views followers reach"),
    ("chat", "talk comment comments chat message messages ask reply"),
    ("check", "done easy ready finished solve solved started"),
    ("heart", "love loved like favourite favorite"),
    ("eye", "see look looking view watching"),
    ("user", "people person audience someone"),
    ("mic", "voice speak podcast audio sound"),
    ("calendar", "today tomorrow week month date plan"),
    ("spark", "ai automatically edits editor editing edit create new idea"),
]


def _split_when(item: str) -> tuple[str, str]:
    """`number: 197K followers @ "come to life"` -> ("number: 197K followers", "come to life"). An @ that
    starts a handle (`@sam`) is part of the text."""
    found = None
    for m in re.finditer(r"(^|\s)@(\s+|(?=[\"“'\d]))", item):
        found = m
    if found is None:
        return item.strip(), ""
    return item[:found.start()].strip(), item[found.end():].strip().strip("\"“”'").strip()


def _share(value: str) -> Optional[float]:
    """72% -> 0.72, 3/5 -> 0.6, 0.72 -> 0.72; None when it is not a share."""
    v = value.strip()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*%", v)
    if m:
        p = float(m.group(1)) / 100.0
        return p if 0.0 <= p <= 1.0 else None
    m = re.fullmatch(r"(\d+)\s*/\s*(\d+)", v)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return a / b if 0 < b <= 100 and 0 <= a <= b else None
    m = re.fullmatch(r"0?\.\d+|1(?:\.0+)?|0", v)
    return float(v) if m else None


def _complete(kind: str, rest: str, when: str = "", layer: str = "") -> Spec:
    """A widget from its kind and the words after it, each kind read its own way."""
    toks = rest.split()
    spec = Spec(kind, when=when, layer=layer)
    if kind == "number":
        for i, t in enumerate(toks):
            try:
                num.parse(t)
            except ValueError:
                continue
            spec.value, spec.text = t, " ".join(toks[:i] + toks[i + 1:])
            break
    elif kind in ("stat", "progress", "icon"):
        if kind == "stat" and toks and toks[0].lower() in ("line", "bars"):
            spec.style, toks = toks[0].lower(), toks[1:]
        if toks:
            spec.value, spec.text = toks[0], " ".join(toks[1:])
        if kind == "icon":
            v = spec.value.lower()
            spec.value = ICON_WORDS.get(v, v)
    else:
        spec.text = rest.strip()
    return spec


def _check(spec: Spec, where: str, strict: bool = True) -> Spec:
    """The widget, or a plain message saying what is wrong with it. Not `strict`: a name pill may
    still be empty (the `name` option fills it)."""
    k, v, t = spec.kind, spec.value.strip(), spec.text.strip()
    if k == "number":
        try:
            a = num.parse(v)
        except ValueError:
            raise O.OptionError(f"{where}: a number needs its figure, such as number: 197K followers") from None
        if a.kind != "amount":
            raise O.OptionError(f"{where}: {v!r} is not a figure a card can count to")
    elif k == "stat":
        if v.lower() != "auto":
            try:
                num.parse_list(v)
            except ValueError as e:
                raise O.OptionError(f"{where}: {e}; a small chart takes its values oldest first, such as "
                                    f"stat: 2K,15K,197K views") from None
    elif k == "progress":
        if _share(v) is None:
            raise O.OptionError(f"{where}: give how far along, such as progress: 72% edited or progress: 3/5 days")
    elif k == "icon":
        if v not in ICONS:
            raise O.OptionError(f"{where}: the signs drawn are {', '.join(ICONS)} (got {spec.value!r})")
    elif k == "name":
        if not t and strict:
            raise O.OptionError(f"{where}: give the name or handle, such as name: Sam, or set name=")
        if len(t) > 28:
            raise O.OptionError(f"{where}: keep a name to 28 letters at most")
    elif k == "note":
        if not t:
            raise O.OptionError(f"{where}: give the line, such as note: Shot in one take")
        if len(t) > NOTE_MAX_CHARS:
            raise O.OptionError(f"{where}: keep a note to a short line, {NOTE_MAX_CHARS} letters at most "
                                f"(this one has {len(t)})")
    if k in ("number", "stat", "progress", "icon") and len(t) > LABEL_MAX_CHARS:
        raise O.OptionError(f"{where}: keep its label to {LABEL_MAX_CHARS} letters at most")
    if spec.layer and spec.layer not in ("front", "back"):
        raise O.OptionError(f"{where}: layer is front or back")
    return spec


def parse_widgets(raw: str, name: str, strict: bool = True) -> list[Spec]:
    """The widgets a person listed: `kind: what | kind: what`, or the same as JSON
    (`[{"kind": "number", "value": "197K", "text": "followers", "when": "come to life"}]`)."""
    v = raw.strip()
    specs: list[Spec] = []
    if v.startswith("["):
        try:
            items = json.loads(v)
        except ValueError as e:
            raise O.OptionError(f"the list is not JSON that can be read ({e})") from None
        if not isinstance(items, list):
            raise O.OptionError("give a list")
        for n, d in enumerate(items, 1):
            if not isinstance(d, dict) or "kind" not in d:
                raise O.OptionError(f"item {n} needs a kind ({', '.join(KINDS)})")
            kind = KIND_WORDS.get(str(d["kind"]).lower(), str(d["kind"]).lower())
            if kind not in KINDS:
                raise O.OptionError(f"item {n}: the kinds are {', '.join(KINDS)} (got {d['kind']!r})")
            value = str(d.get("value", "")).strip()
            if kind == "icon":
                value = ICON_WORDS.get(value.lower(), value.lower())
            specs.append(Spec(kind, text=str(d.get("text", "")).strip(), value=value,
                              when=str(d.get("when", "")).strip(), layer=str(d.get("layer", "")).strip().lower(),
                              style=str(d.get("style", "")).strip().lower()))
    else:
        for n, item in enumerate(split_items(v), 1):
            if not item.strip():
                continue
            body, when = _split_when(item)
            m = re.match(r"^\s*([A-Za-z]+)\s*:?\s*(.*)$", body, re.S)
            kind = KIND_WORDS.get(m.group(1).lower(), m.group(1).lower()) if m else ""
            if kind not in KINDS:
                raise O.OptionError(f"item {n} ({item.strip()!r}) starts with its kind: "
                                    f"{', '.join(KINDS)}, as in number: 197K followers")
            specs.append(_complete(kind, m.group(2), when))
    for n, s in enumerate(specs, 1):
        if s.kind == "name" and not s.text:
            s.text = name.strip()
        if s.kind in ("number", "stat", "progress", "icon") and s.text and s.text.lower() != "auto":
            s.text = _capital(s.text)
        if s.style and s.style not in ("line", "bars"):
            raise O.OptionError(f"item {n} (stat): a small chart is a line or bars")
        _check(s, f"item {n} ({s.kind})", strict)
    if not specs:
        raise O.OptionError("list at least one, such as name: Sam | note: Shot in one take")
    if len(specs) > 5:
        raise O.OptionError(f"five at most round one head ({len(specs)} were given); a sixth is clutter")
    return specs


# ---- widgets found in the clip --------------------------------------------------------------------------

def first_listen(clip) -> tuple[list[Word], dict[int, float], list[Word]]:
    """The transcriber's own words with the corrections given; for each, where the voice begins on it
    once the core has corrected its timing; and those corrected words. Figures and lines are read
    from the first listen, as `chart` reads them: a second listen can hear a stretched figure again,
    out of its sentence, and bring it back wrong, or hear a word that was never said."""
    heard = wordsmod.apply_fixes(wordsmod.load(clip.path), clip.fix)
    quiet = clip.quiet
    kept = [i for i, w in enumerate(heard) if w.key and not unheard(w, quiet)]
    timed = wordsmod.timed(heard, quiet)
    corrected = {i: w.start for i, w in zip(kept, timed)} if len(kept) == len(timed) else {}
    return heard, corrected, timed


def _capital(t: str) -> str:
    return t[:1].upper() + t[1:]


def _line_text(ws: list[Word]) -> str:
    toks = tokens_from_words(ws)
    return _capital(" ".join(t.text for t in toks)).strip()


def sign_for(text: str) -> Optional[str]:
    keys = {normalize(w) for w in text.split()}
    for sign, words in SIGN_FOR:
        if keys & set(words.split()):
            return sign
    return None


def auto_specs(clip, name: str) -> tuple[list[Spec], list[str]]:
    """The widgets the clip itself gives: the figures said (a run is a small chart, a share of a
    hundred a ring), the name given, and the lines that make the point, with a sign when a line's
    words name one. Returns them in the order they are said, and what was passed over."""
    notes: list[str] = []
    heard, corrected, timed = first_listen(clip)
    quiet = clip.quiet
    specs: list[Spec] = []
    # what the speaker took back is never a widget: a figure ("ten, sorry, twelve"), nor a line
    back_heard, back = moments.taken_back(heard, quiet), moments.taken_back(timed, quiet)

    def start_of(a: num.Amount) -> float:
        for i in range(a.first, a.last + 1):
            if i in corrected:
                return corrected[i]
        return max(0.0, heard[a.first].start - wordsmod.HEARD_LATE)

    said = [a for a in num.spoken(heard) if a.kind == "amount" and not a.weak
            and not back_heard.holds(heard, a.first, a.last)]
    run = num.series(said, heard)
    in_run = set()
    if run is not None and len(run.amounts) >= 2:
        first = min(run.amounts, key=lambda a: a.first)
        specs.append(Spec("stat", text="auto", value="auto", at=start_of(first),
                          said=" … ".join(a.said for a in sorted(run.amounts, key=lambda a: a.first)), found="said"))
        in_run = {(a.first, a.last) for a in run.amounts}
    for a in said:
        if (a.first, a.last) in in_run or len([s for s in specs if s.kind in ("number", "progress", "stat")]) >= 3:
            continue
        label = _capital(num.label_for(num.Series(amounts=[a]), heard))[:LABEL_MAX_CHARS]
        if a.suffix == "%" and 0 < a.value <= 100:
            specs.append(Spec("progress", text=label, value=f"{a.value:g}%", at=start_of(a), said=a.said, found="said"))
        else:
            typed = num.Unit.join(num.Unit.of([a]).exact(a.value))
            specs.append(Spec("number", text=label, value=typed, at=start_of(a), said=a.said, found="said"))
    if name.strip():
        spec = Spec("name", text=name.strip(), found="given")
        words = [w for w in timed if w.key not in HESITATIONS]
        hit = find_said(words, name.strip().lstrip("@"))
        if hit:
            spec.at, spec.said, spec.found = words[hit[0]].start, words[hit[0]].clean, "said"
        specs.append(spec)
    # the line that makes the point, as a note; a second only when nothing else was found, so a
    # weaker scrap of a sentence never makes up the numbers
    spans: list[tuple[int, int]] = []
    pool = [(0, len(timed))]
    want = len(specs) + (1 if specs else 2)
    while len(specs) < want and pool:
        a, b = pool.pop(0)
        if b - a < 2:
            continue
        got = moments.pick_line(timed[a:b], quiet, back=back)
        if got is None:
            continue
        i0, i1 = got[0] + a, got[1] + a
        text = _line_text(timed[i0:i1 + 1])
        if 0 < len(text) <= NOTE_MAX_CHARS and len(text.split()) >= 2:
            specs.append(Spec("note", text=text, at=timed[i0].start, said=text, found="the line that makes the point"))
            spans.append((i0, i1))
        pool = sorted(pool + [(i1 + 1, b), (a, i0)], key=lambda s: -(s[1] - s[0]))
    first_note = next((s for s in specs if s.kind == "note"), None)
    if first_note is not None and len(specs) < 4:
        sign = sign_for(first_note.text)
        if sign:
            specs.append(Spec("icon", value=sign, at=(first_note.at or 0.0) + 0.05, said=first_note.text,
                              found=f"a sign for “{first_note.text}”"))
    if not specs:
        raise O.OptionError("there is nothing of yours to show: no figure is said, no name was given (name=) and no "
                            "line stands out. List the widgets: widgets=\"number: 197K followers | note: Shot in one "
                            "take\"")
    specs.sort(key=lambda s: (s.at if s.at is not None else 1e9))
    return specs[:5], notes


# ---- the look every widget shares ------------------------------------------------------------------------

@dataclass
class Look:
    name: str  # dark, light or tint
    tone: glassmod.Tone
    ink: tuple  # the type
    soft: tuple  # labels
    accent: tuple  # rings, chips, the bar beside a note
    on_accent: tuple  # a sign or initials on the accent
    contrast: float  # the type against the glass where they are closest
    shadow: bool  # type gets a soft shadow (on dark glass)

    def text_shadow(self, s: float):
        return (0.0, 2.0 * s, 4.0 * s, 0.32) if self.shadow else None


def _deep(rgb, target: float = 46.0) -> tuple:
    """A colour taken down to a brightness white type reads on."""
    c = np.asarray(rgb, np.float32)
    b = glassmod.brightness(c)
    return tuple(float(v) for v in c * (target / max(b, 1.0)))


def _mix(a, b, k: float) -> tuple:
    return tuple(int(round(x * (1 - k) + y * k)) for x, y in zip(a, b))


# ---- drawing helpers -----------------------------------------------------------------------------------

def _canvas(w: float, h: float, k: int = 4):
    from PIL import Image

    return Image.new("RGBA", (max(1, int(round(w * k))), max(1, int(round(h * k)))), (0, 0, 0, 0))


def _down(img, w: float, h: float) -> Sprite:
    from PIL import Image

    return Sprite.from_rgba(np.asarray(img.resize((max(1, int(round(w))), max(1, int(round(h)))), Image.LANCZOS)))


def _paint(alpha: np.ndarray, colour) -> Sprite:
    a = np.clip(alpha, 0, 1).astype(np.float32)
    return Sprite(rgb=(np.asarray(colour, np.float32)[None, None, :] * a[..., None]).astype(np.float32), a=a)


def _chip(d: float, colour) -> Sprite:
    """A round chip in `colour`, a little lighter at the top, as if lit from above."""
    from PIL import ImageDraw

    img = _canvas(d, d)
    ImageDraw.Draw(img).ellipse((0, 0, img.width - 1, img.height - 1), fill=(255, 255, 255, 255))
    a = np.asarray(_down(img, d, d).a)
    n = a.shape[0]
    ys = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    lift = 0.16 * np.clip(1.0 - ys / 0.6, 0, 1) ** 1.5
    c = np.asarray(colour, np.float32)
    rgb = (c[None, None, :] + (255.0 - c[None, None, :]) * lift[..., None]) * a[..., None]
    return Sprite(rgb=rgb.astype(np.float32), a=a.astype(np.float32))


def glyph(name: str, d: float) -> np.ndarray:
    """A sign we draw ourselves, as an alpha mask d x d: plain shapes, never anyone's logo."""
    from PIL import ImageDraw

    k = 4
    img = _canvas(d, d, k)
    S = img.width
    dr = ImageDraw.Draw(img)
    W = (255, 255, 255, 255)

    def P(x: float, y: float) -> tuple[float, float]:
        return x * S, y * S

    def line(pts, w: float) -> None:
        q = [P(*p) for p in pts]
        dr.line(q, fill=W, width=max(1, int(round(w * S))), joint="curve")
        r = w * S / 2
        for x, y in (q[0], q[-1]):
            dr.ellipse((x - r, y - r, x + r, y + r), fill=W)

    def disc(cx: float, cy: float, r: float, fill=W) -> None:
        dr.ellipse((*P(cx - r, cy - r), *P(cx + r, cy + r)), fill=fill)

    def ring(cx: float, cy: float, r: float, w: float) -> None:
        dr.ellipse((*P(cx - r, cy - r), *P(cx + r, cy + r)), outline=W, width=max(1, int(round(w * S))))

    clear = (0, 0, 0, 0)
    if name == "check":
        line([(0.2, 0.53), (0.42, 0.74), (0.81, 0.3)], 0.13)
    elif name == "star":
        pts = []
        for i in range(10):
            r = 0.47 if i % 2 == 0 else 0.2
            t = -math.pi / 2 + i * math.pi / 5
            pts.append(P(0.5 + r * math.cos(t), 0.54 + r * math.sin(t)))
        dr.polygon(pts, fill=W)
    elif name == "play":
        tri = [(0.34, 0.2), (0.34, 0.8), (0.84, 0.5)]
        dr.polygon([P(*p) for p in tri], fill=W)
        line(tri + [tri[0]], 0.08)
    elif name == "heart":
        ts = np.linspace(0, 2 * math.pi, 120)
        xs = 16 * np.sin(ts) ** 3
        ys = 13 * np.cos(ts) - 5 * np.cos(2 * ts) - 2 * np.cos(3 * ts) - np.cos(4 * ts)
        dr.polygon([P(0.5 + x / 36, 0.47 - y / 36) for x, y in zip(xs, ys)], fill=W)
    elif name == "bolt":
        dr.polygon([P(*p) for p in [(0.58, 0.06), (0.2, 0.56), (0.46, 0.56), (0.39, 0.94), (0.8, 0.41),
                                    (0.54, 0.41), (0.64, 0.06)]], fill=W)
    elif name == "spark":
        def sparkle(cx, cy, r):
            ts = np.linspace(0, 2 * math.pi, 160)
            dr.polygon([P(cx + r * np.cos(t) ** 3, cy + r * np.sin(t) ** 3) for t in ts], fill=W)
        sparkle(0.44, 0.56, 0.42)
        sparkle(0.8, 0.2, 0.16)
    elif name == "clock":
        ring(0.5, 0.5, 0.4, 0.1)
        line([(0.5, 0.5), (0.5, 0.27)], 0.09)
        line([(0.5, 0.5), (0.67, 0.6)], 0.09)
    elif name == "up":
        line([(0.14, 0.74), (0.4, 0.48), (0.56, 0.62), (0.84, 0.32)], 0.1)
        line([(0.62, 0.3), (0.86, 0.3), (0.86, 0.54)], 0.1)
    elif name == "pin":
        disc(0.5, 0.4, 0.3)
        dr.polygon([P(0.24, 0.5), P(0.76, 0.5), P(0.5, 0.94)], fill=W)
        disc(0.5, 0.4, 0.11, fill=clear)
    elif name == "chat":
        dr.rounded_rectangle((*P(0.12, 0.18), *P(0.88, 0.7)), radius=0.17 * S, fill=W)
        dr.polygon([P(0.28, 0.62), P(0.24, 0.88), P(0.5, 0.66)], fill=W)
    elif name == "eye":
        ts = np.linspace(0, math.pi, 60)
        top = [P(0.08 + 0.84 * (1 - math.cos(t)) / 2, 0.5 - 0.3 * math.sin(t)) for t in ts]
        bot = [P(0.08 + 0.84 * (1 + math.cos(t)) / 2, 0.5 + 0.3 * math.sin(t)) for t in ts]
        dr.polygon(top + bot, fill=W)
        disc(0.5, 0.5, 0.17, fill=clear)
        disc(0.5, 0.5, 0.09)
    elif name == "user":
        disc(0.5, 0.33, 0.19)
        dr.chord((*P(0.16, 0.6), *P(0.84, 1.24)), 180, 360, fill=W)
    elif name == "mic":
        dr.rounded_rectangle((*P(0.37, 0.1), *P(0.63, 0.58)), radius=0.13 * S, fill=W)
        dr.arc((*P(0.24, 0.24), *P(0.76, 0.72)), 0, 180, fill=W, width=int(round(0.08 * S)))
        line([(0.5, 0.72), (0.5, 0.88)], 0.08)
        line([(0.35, 0.88), (0.65, 0.88)], 0.08)
    elif name == "calendar":
        dr.rounded_rectangle((*P(0.14, 0.2), *P(0.86, 0.86)), radius=0.1 * S, outline=W, width=int(round(0.08 * S)))
        dr.rectangle((*P(0.14, 0.26), *P(0.86, 0.4)), fill=W)
        line([(0.34, 0.1), (0.34, 0.27)], 0.08)
        line([(0.66, 0.1), (0.66, 0.27)], 0.08)
        for gx in (0.33, 0.5, 0.67):
            for gy in (0.55, 0.71):
                disc(gx, gy, 0.045)
    return np.asarray(_down(img, d, d).a)


# ---- one widget ------------------------------------------------------------------------------------------

@dataclass
class State:
    opacity: float
    scale: float
    frost: float
    alpha: float  # its writing
    rise: float  # how far below its place the writing still is (pixels)
    p: float  # how far through its entrance (0..1+)


# Extra glass on the side toward the head, a share of the widget's own size, for a widget behind the
# person to tuck into: its writing stays on the far side, the person crosses the rest.
TUCKS = {"down": (0.0, 0.46), "left": (0.4, 0.0), "right": (0.4, 0.0)}
HERO = 1.22  # the first widget of a cluster of three or more is drawn this much larger


def _parts(shape: str) -> tuple[str, str]:
    base, _, tuck = shape.partition("+")
    return base, tuck


class Tile:
    """One widget: its size at a scale, where its writing is, and how it is drawn. Every widget has
    the same entrance (the glass grows in and frosts over, then the writing settles up into it) and
    leaves with the others. Behind the person it may take a tuck: more glass toward the head."""
    base_shapes: tuple[str, ...] = ("one",)
    tucks = True

    def __init__(self, spec: Spec, U: float):
        self.spec = spec
        self.kind = spec.kind
        self.U = U
        self.at = 0.0  # arrives
        self.land = 0.0  # has landed: counted, filled, written
        self.leave: Optional[float] = None
        self.layer = "front"
        self.boost = 1.0  # the hero's
        self.k = 1.0
        self.shape = self.base_shapes[0]
        self.x = self.y = 0.0
        self.w = self.h = 0
        self.bw = self.bh = 0.0  # the widget itself, without its tuck
        self.ox = self.oy = 0.0  # where the widget itself sits inside its glass
        self.look: Optional[Look] = None
        self.index = 0

    @property
    def shapes(self) -> tuple[str, ...]:
        base = self.base_shapes
        if self.layer != "back" or not self.tucks:
            return base
        return tuple(base) + tuple(f"{b}+{t}" for b in base for t in TUCKS)

    # what the layout asks
    def base_size(self, k: float, base: str) -> tuple[float, float]:
        raise NotImplementedError

    def base_content(self, k: float, base: str) -> Rect:
        """Where the writing is inside the widget itself; the person may never hide it."""
        w, h = self.base_size(k, base)
        m = 0.45 * PAD * k * self.U
        return Rect(m, m, w - m, h - m)

    def _slack(self, k: float, shape: str) -> tuple[float, float, float, float, float, float]:
        base, tuck = _parts(shape)
        w, h = self.base_size(k, base)
        fx, fy = TUCKS.get(tuck, (0.0, 0.0))
        sx, sy = fx * w, fy * h
        return w, h, sx, sy, (sx if tuck == "left" else 0.0), 0.0

    def size(self, k: float, shape: str) -> tuple[float, float]:
        w, h, sx, sy, _, _ = self._slack(k, shape)
        return w + sx, h + sy

    def content(self, k: float, shape: str) -> Rect:
        _, _, _, _, ox, oy = self._slack(k, shape)
        return self.base_content(k, _parts(shape)[0]).shift(ox, oy)

    @property
    def rect(self) -> Rect:
        return Rect(self.x, self.y, self.x + self.w, self.y + self.h)

    @property
    def base(self) -> str:
        return _parts(self.shape)[0]

    @property
    def tuck(self) -> str:
        return _parts(self.shape)[1]

    def put(self, x: float, y: float, k: float, shape: str) -> None:
        self.x, self.y, self.k, self.shape = float(x), float(y), float(k), shape
        w, h, sx, sy, ox, oy = self._slack(k, shape)
        self.bw, self.bh, self.ox, self.oy = w, h, ox, oy
        self.w, self.h = int(math.ceil(w + sx)), int(math.ceil(h + sy))

    @property
    def s(self) -> float:
        return self.k * self.U

    def on(self, t: float) -> bool:
        return t >= self.at and (self.leave is None or t < self.leave + EXIT)

    def state(self, t: float) -> Optional[State]:
        if not self.on(t):
            return None
        p = (t - self.at) / ENTER
        op = _ease_out(p / 0.55)
        scale = GROW_FROM + (1.0 - GROW_FROM) * _back_out(min(1.0, p), 1.3)
        frost = min(1.0, 0.35 + 0.65 * _ease_out(p / 0.8))
        ca = _ease_out((p - CONTENT_LAG) / (1.0 - CONTENT_LAG))
        rise = (1.0 - _ease_out(p)) * 10.0 * self.s
        if self.leave is not None and t >= self.leave:
            e = _ease_in_out((t - self.leave) / EXIT)
            op *= 1.0 - e
            scale *= 1.0 - 0.07 * e
            frost *= 1.0 - 0.4 * e
            ca *= (1.0 - e) ** 1.8
        return State(op, scale, frost, ca, rise, p)

    def build(self, look: Look) -> None:
        self.look = look
        s = self.s
        self.pane = glassmod.Pane(self.w, self.h, radius=self.radius(), unit=s, shadow=SHADOW[self.layer])
        self.pane.tone = look.tone

    def radius(self) -> float:
        return RADIUS * self.s

    @property
    def glass(self) -> glassmod.Pane:
        return self.pane

    def draw(self, target: np.ndarray, x: float, y: float, t: float, j: int) -> None:
        st = self.state(t)
        if st is None or st.opacity <= 0.002:
            return
        self.pane.draw(target, x, y, opacity=st.opacity, frost=st.frost, scale=st.scale)
        if st.alpha > 0.003:
            self.draw_content(target, x + self.ox, y + self.oy, t, st)

    def draw_content(self, target, x, y, t, st: State) -> None:
        raise NotImplementedError

    def word(self, text: str, face: str, px: float, colour) -> textmod.Word:
        return textmod.draw_word(text, face, max(6, int(round(px))), fill=colour,
                                 shadow=self.look.text_shadow(self.s) if self.look else None)

    def report(self) -> str:
        return self.spec.label()


class NameTile(Tile):
    """A pill: the initials on a round chip of the accent, then the name."""
    tucks = False

    def _initials(self) -> str:
        t = self.spec.text.strip().lstrip("@")
        parts = [p for p in re.split(r"[\s._-]+", t) if p]
        if not parts:
            return "?"
        if self.spec.text.strip().startswith("@") or len(parts) == 1:
            return parts[0][:1].upper()
        return (parts[0][:1] + parts[-1][:1]).upper()

    def base_size(self, k, base):
        s = k * self.U
        tw = textmod.advance(self.spec.text, FACE_NAME, max(6, int(round(NAME_PX * s))))
        return (14 + CHIP_D + 18) * s + tw + 36 * s, PILL_H * s

    def base_content(self, k, base):
        w, h = self.base_size(k, base)
        return Rect(6 * k * self.U, 6 * k * self.U, w - 6 * k * self.U, h - 6 * k * self.U)

    def radius(self):
        return self.h / 2

    def build(self, look):
        super().build(look)
        s = self.s
        self.chip = _chip(CHIP_D * s, look.accent)
        self.ini_px = max(6, int(round(CHIP_D * 0.42 * s)))
        self.initials = textmod.draw_word(self._initials(), "poppins-bold", self.ini_px, fill=look.on_accent)
        self.name_word = self.word(self.spec.text, FACE_NAME, NAME_PX * s, look.ink)

    def draw_content(self, target, x, y, t, st):
        s = self.s
        q = _ease_out((st.p - 0.15) / 0.55)
        sc = 0.55 + 0.45 * _back_out(min(1.0, max(0.0, (st.p - 0.15) / 0.6)), 1.6)
        cx, cy = x + (14 + CHIP_D / 2) * s, y + self.bh / 2
        chip = self.chip.scaled(sc)
        over(target, chip, cx - chip.w / 2, cy - chip.h / 2, st.alpha * min(1.0, q * 1.5))
        ini = self.initials
        cap = textmod.cap_height("poppins-bold", self.ini_px)
        isp = ini.sprite.scaled(sc)
        bx, by = cx - ini.width * sc / 2, cy + cap * sc / 2
        over(target, isp, bx - ini.ox * sc, by - ini.oy * sc, st.alpha * min(1.0, q * 1.5))
        nw = self.name_word
        tx = x + (14 + CHIP_D + 18) * s
        cap = textmod.cap_height(FACE_NAME, max(6, int(round(NAME_PX * s))))
        ty = y + self.bh / 2 + cap / 2
        sp = _soft_reveal(nw.sprite, nw.ox, nw.ox + nw.width, _ease_out((st.p - 0.35) / 0.65))
        over(target, sp, tx - nw.ox, ty - nw.oy + st.rise * 0.5, st.alpha)


class NoteTile(Tile):
    """A short line of the speaker's own, on one or two lines, with a bar of the accent beside it."""
    base_shapes = ("one", "two")

    def _lines(self, base: str) -> list[str]:
        words = self.spec.text.split()
        if base == "one" or len(words) < 2:
            return [self.spec.text]
        best, best_w = None, math.inf
        for i in range(1, len(words)):
            a, b = " ".join(words[:i]), " ".join(words[i:])
            w = max(textmod.advance(a, FACE_BODY, 100), textmod.advance(b, FACE_BODY, 100))
            if w < best_w:
                best, best_w = [a, b], w
        return best or [self.spec.text]

    def _metrics(self, k, base):
        s = k * self.U
        px = max(6, int(round(BODY_PX * s)))
        lines = self._lines(base)
        widths = [textmod.advance(l, FACE_BODY, px) for l in lines]
        cap = textmod.cap_height(FACE_BODY, px)
        lh = 1.24 * px
        padx, pady = 34 * s, 32 * s
        bar = 7 * s
        text_x = padx + bar + 20 * s
        w = text_x + max(widths) + padx
        h = 2 * pady + cap + (len(lines) - 1) * lh + 0.2 * px
        return s, px, lines, widths, cap, lh, padx, pady, bar, text_x, w, h

    def base_size(self, k, base):
        m = self._metrics(k, base)
        return m[10], m[11]

    def base_content(self, k, base):
        s, px, lines, widths, cap, lh, padx, pady, bar, text_x, w, h = self._metrics(k, base)
        return Rect(padx * 0.6, pady * 0.6, w - padx * 0.6, h - pady * 0.6)

    def build(self, look):
        super().build(look)
        s, px, lines, widths, cap, lh, padx, pady, bar, text_x, w, h = self._metrics(self.k, self.base)
        self.lines = [self.word(l, FACE_BODY, px, look.ink) for l in lines]
        self.m = (s, px, cap, lh, padx, pady, bar, text_x)
        bh = cap + (len(lines) - 1) * lh + 0.06 * px
        self.bar = textmod.rounded_box(bar, bh, bar / 2, fill=tuple(int(c) for c in look.accent))

    def draw_content(self, target, x, y, t, st):
        s, px, cap, lh, padx, pady, bar, text_x = self.m
        grow = _ease_out((st.p - 0.2) / 0.6)
        if grow > 0:
            b = self.bar
            hh = max(1, int(round(b.h * grow)))
            over(target, Sprite(rgb=b.rgb[:hh], a=b.a[:hh]), x + padx, y + pady + 0.02 * px, st.alpha)
        for i, wd in enumerate(self.lines):
            a = st.alpha * _ease_out((st.p - CONTENT_LAG - 0.12 * i) / 0.6)
            by = y + pady + cap + i * lh + st.rise
            over(target, wd.sprite, x + text_x - wd.ox, by - wd.oy, a)


class IconTile(Tile):
    """A square of glass with a sign we draw on a round chip of the accent, and an optional caption."""
    tucks = False

    def base_size(self, k, base):
        s = k * self.U
        w = ICON_TILE * s
        if self.spec.text:
            px = max(6, int(round(CAPTION_PX * s)))
            w = max(w, textmod.advance(self.spec.text, FACE_LABEL, px) + 2 * 26 * s)
            return w, (ICON_TILE + 40) * s
        return w, w

    def radius(self):
        return 44 * self.s

    def build(self, look):
        super().build(look)
        s = self.s
        self.chip = _chip(ICON_CHIP * s, look.accent)
        self.sign = _paint(glyph(self.spec.value, ICON_GLYPH * s), look.on_accent)
        self.cap_px = max(6, int(round(CAPTION_PX * s)))
        self.caption = self.word(self.spec.text, FACE_LABEL, self.cap_px, look.soft) if self.spec.text else None

    def draw_content(self, target, x, y, t, st):
        s = self.s
        cx = x + self.bw / 2
        cy = y + (ICON_TILE / 2 - (6 if self.caption is not None else 0)) * s
        q = (st.p - 0.18) / 0.62
        sc = 0.5 + 0.5 * _back_out(min(1.0, max(0.0, q)), 1.7)
        a = st.alpha * min(1.0, max(0.0, q) * 2.0)
        chip = self.chip.scaled(sc)
        over(target, chip, cx - chip.w / 2, cy - chip.h / 2, a)
        sign = self.sign.scaled(sc)
        over(target, sign, cx - sign.w / 2, cy - sign.h / 2, a)
        if self.caption is not None:
            c = self.caption
            by = y + (ICON_TILE + 8) * s + textmod.cap_height(FACE_LABEL, self.cap_px) * 0.5
            over(target, c.sprite, cx - c.width / 2 - c.ox, by - c.oy + st.rise, st.alpha * 0.95)


class ProgressTile(Tile):
    """How far along: a ring that fills to a share of a hundred, or a bar of steps (3/5) that fill one
    by one, landing as it is said."""
    tucks = False

    def __init__(self, spec, U):
        super().__init__(spec, U)
        v = spec.value.strip()
        self.share = _share(v) or 0.0
        m = re.fullmatch(r"(\d+)\s*/\s*(\d+)", v)
        self.steps = (int(m.group(1)), int(m.group(2))) if m and int(m.group(2)) <= 12 else None
        self.ring = self.steps is None
        pm = re.fullmatch(r"(\d+(?:\.\d+)?)\s*%", v)
        self.percent = float(pm.group(1)) if pm else round(self.share * 100, 1)

    def base_size(self, k, base):
        s = k * self.U
        if self.ring:
            w = (RING_D + 2 * 38) * s
            h = (38 + RING_D) * s + (((22 + LABEL_PX * 0.74 + 30) * s) if self.spec.text else 38 * s)
            if self.spec.text:
                w = max(w, textmod.advance(self.spec.text, FACE_LABEL, max(6, int(round(LABEL_PX * s)))) + 2 * 30 * s)
            return w, h
        lw = textmod.advance(self.spec.text or " ", FACE_LABEL, max(6, int(round(LABEL_PX * s))))
        vw = textmod.advance(self.spec.value, FACE_NUM, max(6, int(round(56 * s))))
        w = max(380 * s, lw + vw + (2 * 36 + 30) * s)
        return w, (36 + 52 + 22 + BAR_H + 36) * s

    def build(self, look):
        super().build(look)
        s = self.s
        self.label_px = max(6, int(round(LABEL_PX * s)))
        self.label = self.word(self.spec.text, FACE_LABEL, self.label_px, look.soft) if self.spec.text else None
        if self.ring:
            d = RING_D * s
            n = max(8, int(math.ceil(d)) + 2)
            yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
            c = (n - 1) / 2
            r = np.hypot(xx - c, yy - c)
            rw = RING_W * s
            mid = d / 2 - rw / 2
            self.band = np.clip(rw / 2 + 0.5 - np.abs(r - mid), 0, 1).astype(np.float32)
            # each pixel's angle from the top, clockwise, as a share of the turn
            self.angle = ((np.arctan2(xx - c, -(yy - c)) / (2 * math.pi)) % 1.0).astype(np.float32)
            self.ring_n, self.ring_c, self.ring_mid = n, c, mid
            track = (255, 255, 255) if look.name != "light" else (20, 24, 36)
            self.track = _paint(self.band * (0.2 if look.name != "light" else 0.12), track)
            self.cap_dot = _chip(rw, look.accent)
            self.figs = _inked(chartmod.Figures(max(6, int(round(50 * s))), s), look)
            self.unit = num.Unit(suffix="%", decimals=1 if self.percent != int(self.percent) else 0)
        else:
            self.figs = _inked(chartmod.Figures(max(6, int(round(56 * s))), s), look)
            a, b = self.steps
            self.seg_w = (self.bw - 2 * 36 * s - (b - 1) * 8 * s) / b
            off = (255, 255, 255) if look.name != "light" else (20, 24, 36)
            self.seg_on = textmod.rounded_box(self.seg_w, BAR_H * s, BAR_H * s / 2,
                                              fill=tuple(int(c) for c in look.accent))
            self.seg_off = textmod.rounded_box(self.seg_w, BAR_H * s, BAR_H * s / 2, fill=off,
                                               opacity=0.2 if look.name != "light" else 0.12)

    def fill(self, t: float) -> float:
        """How far the ring or bar has filled at `t` (0..1 of its share)."""
        a = self.land - COUNT
        return _ease_in_out((t - a) / COUNT) if t < self.land else 1.0

    def draw_content(self, target, x, y, t, st):
        s = self.s
        f = self.fill(t)
        if self.ring:
            n = self.ring_n
            ox, oy = x + self.bw / 2 - n / 2, y + 38 * s + RING_D * s / 2 - n / 2
            over(target, self.track, ox, oy, st.alpha)
            reach = self.share * f
            if reach > 0.002:
                soft = 1.2 / (2 * math.pi * self.ring_mid)
                arc = self.band * np.clip((reach - self.angle) / soft + 0.5, 0, 1)
                over(target, _paint(arc, self.look.accent), ox, oy, st.alpha)
                for frac in (0.0, reach):  # round ends
                    ang = 2 * math.pi * frac
                    px = self.ring_c + self.ring_mid * math.sin(ang)
                    py = self.ring_c - self.ring_mid * math.cos(ang)
                    dot = self.cap_dot
                    over(target, dot, ox + px - dot.w / 2, oy + py - dot.h / 2, st.alpha)
            v = self.percent * f
            parts = self.unit.during(v, self.percent) if f < 1 else self.unit.exact(self.percent)
            fw = self.figs.width(parts)
            cy = y + 38 * s + RING_D * s / 2
            self.figs.draw(target, parts, x + self.bw / 2 - fw / 2, cy + self.figs.cap / 2 + st.rise, st.alpha)
            if self.label is not None:
                lb = self.label
                by = y + (38 + RING_D + 22) * s + textmod.cap_height(FACE_LABEL, self.label_px)
                over(target, lb.sprite, x + self.bw / 2 - lb.width / 2 - lb.ox, by - lb.oy + st.rise, st.alpha * 0.95)
        else:
            a, b = self.steps
            lit = a * f
            top = y + 36 * s
            if self.label is not None:
                lb = self.label
                cap = textmod.cap_height(FACE_LABEL, self.label_px)
                over(target, lb.sprite, x + 36 * s - lb.ox, top + 30 * s + cap / 2 - lb.oy + st.rise, st.alpha * 0.95)
            parts = ("", f"{int(math.floor(lit + 1e-6))}/{b}", "", "")
            fw = self.figs.width(parts)
            self.figs.draw(target, parts, x + self.bw - 36 * s - fw, top + 30 * s + self.figs.cap / 2 + st.rise, st.alpha)
            by = top + (52 + 22) * s
            for i in range(b):
                sx = x + 36 * s + i * (self.seg_w + 8 * s)
                over(target, self.seg_off, sx, by, st.alpha)
                q = min(1.0, max(0.0, lit - i))
                if q > 0:
                    seg = self.seg_on
                    ww = max(1, int(round(seg.w * _ease_out(q))))
                    over(target, Sprite(rgb=seg.rgb[:, :ww], a=seg.a[:, :ww]), sx, by, st.alpha)


def _inked(figs: chartmod.Figures, look: Look) -> chartmod.Figures:
    """`chart`'s figures (drawn a character at a time, so a count never re-sets its type) in this
    look's ink: white with a soft shadow on dark glass, dark and clean on light glass."""
    if look.name != "light":
        return figs
    ink = look.ink

    def glyph_(ch: str, small: bool, _f=figs) -> textmod.Word:
        key = (ch, small)
        if key not in _f._glyphs:
            _f._glyphs[key] = textmod.draw_word(ch, FACE_NUM, _f.small_px if small else _f.px, fill=ink)
        return _f._glyphs[key]

    figs._glyph = glyph_
    return figs


class _Growing:
    """A pane that grows in with its widget: what `chart`'s card draws, at the widget's scale."""

    def __init__(self, pane: glassmod.Pane):
        self.inner = pane
        self.scale = 1.0

    def draw(self, frame, x, y, opacity=1.0, frost=1.0, scale=1.0):
        self.inner.draw(frame, x, y, opacity, frost, scale * self.scale)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class ChartTile(Tile):
    """A figure counting up (`number`), or a small chart (`stat`): `chart`'s own card, put in its
    place in the cluster and grown in with it."""

    def __init__(self, spec, U, make):
        super().__init__(spec, U)
        self.make = make  # (when) -> a chart card that does not place itself
        self.card: Optional[_Card] = None
        self.base_shapes = ("counter",) if spec.kind == "number" else ("tall", "wide")
        self.shape = self.base_shapes[0]

    def prepare(self, when: str) -> None:
        self.card = self.make(when)
        if self.spec.kind == "stat" and self.card.kind == "counter":
            self.base_shapes = ("counter",)
            self.shape = "counter"

    def base_size(self, k, base):
        return self.card._size(base, k)

    def put(self, x, y, k, shape):
        super().put(x, y, k, shape)
        w, h, sx, sy, ox, oy = self._slack(k, shape)
        self.card.put(x, y, k, self.base, slack=(sx, sy, self.tuck))
        self.w, self.h = self.card.cw, self.card.ch
        self.ox = self.oy = 0.0  # the card places its own writing

    @property
    def glass(self):
        return self.card.pane.inner if isinstance(self.card.pane, _Growing) else self.card.pane

    def on(self, t):
        return self.card.start <= t < self.card.end

    def build(self, look):
        self.look = look
        self.card.dress(look, self.layer)

    def draw(self, target, x, y, t, j):
        st = self.state(t)
        if st is None:
            return
        c = self.card
        c.x, c.y = int(round(x)), int(round(y))
        c.pane.scale = st.scale
        chartmod.Chart.draw(c, target, j, t)

    def report(self):
        c = self.card
        texts = " → ".join(num.Unit.join(t) for t in c.texts)
        return f"{self.kind} {texts}" + (f" “{c.label}”" if c.label else "")


class _Card(chartmod.Chart):
    """`chart`'s card, placed by the cluster rather than by itself."""

    def _place(self, forced=None):
        self.where, self.clearance, self.outside = "cluster", "head", False
        self.layout = self._layouts()[0]
        self.k, self.s, self.room = 1.0, self.U, None
        self._measure(self.layout, self.s)
        self.x = self.y = 0

    def _build(self):
        pass  # built once the cluster has put it in place

    def put(self, x: float, y: float, k: float, layout: str, slack: tuple = (0.0, 0.0, "")) -> None:
        """In its place at scale `k`; `slack` is a tuck of empty glass (x, y, side) toward the head,
        the writing kept on the far side of it."""
        self.layout, self.k, self.s = layout, float(k), float(k) * self.U
        self._measure(layout, self.s)
        sx, sy, side = slack
        if side == "left":
            self.m["x_text"] += sx
            if self.m["chart"] is not None:
                x0, y0, x1, y1 = self.m["chart"]
                self.m["chart"] = (x0 + sx, y0, x1 + sx, y1)
        self.cw += int(round(sx))
        self.ch += int(round(sy))
        self.x, self.y = int(round(x)), int(round(y))
        chartmod.Chart._build(self)

    def dress(self, look: Look, layer: str = "front") -> None:
        """The cluster's glass and ink on the card, its shadow as deep as its layer's."""
        pane = self.pane.inner if isinstance(self.pane, _Growing) else self.pane
        pane = glassmod.Pane(pane.w, pane.h, radius=pane.r, unit=pane.u, shadow=SHADOW[layer])
        pane.tone = look.tone
        self.pane = _Growing(pane)
        self.contrast = look.contrast
        if look.name != "light":
            return
        s, m, ink, soft = self.s, self.m, look.ink, look.soft
        self.figs = _inked(chartmod.Figures(m["num_px"], s), look)
        if self.label:
            self.label_sprite = textmod.draw_word(self.label, chartmod.FACE_LABEL, m["label_px"], fill=soft)

        def recolour(sp: Sprite, colour, k: float = 1.0) -> Sprite:
            a = np.clip(sp.a * k, 0, 1)
            return Sprite(rgb=(np.asarray(colour, np.float32)[None, None, :] * a[..., None]).astype(np.float32), a=a)

        if getattr(self, "rules", None) is not None:
            self.rules = recolour(self.rules, ink, 1.4)
        if getattr(self, "bar_off", None) is not None:
            self.bar_off = recolour(self.bar_off, ink, 0.55)
        if getattr(self, "tags", None):
            tag_px = max(8, int(round(chartmod.AXIS_PX * 1.05 * s)))
            self.tags = [textmod.draw_word(num.Unit.join(t), chartmod.FACE_LABEL, tag_px, fill=ink) for t in self.texts]
        if getattr(self, "axis", None):  # one colour and no shadow: recoloured as they are
            self.axis = [(textmod.Word(recolour(wd.sprite, soft), wd.ox, wd.oy, wd.width, wd.ascent), ax, ay)
                         for wd, ax, ay in self.axis]


TILES = {"name": NameTile, "note": NoteTile, "icon": IconTile, "progress": ProgressTile}


# ---- the cluster ---------------------------------------------------------------------------------------

def core_in_output(clip, timeline, fr, shrink: float = 0.08):
    """The middle of the face at each output frame (the detector's box, brows to chin, a little in
    from its edges): what even a widget behind the person never sits behind."""
    face = clip.face
    fps, n_src = clip.info.rate, clip.info.frames

    def at(j: int) -> Optional[Rect]:
        k = timeline.source_frame(j, fps, n_src)
        b = face.box(k)
        if b is None:
            return None
        x, y, w, h, _ = face.trusted[int(np.clip(k, 0, face.n - 1))]
        if not np.isnan(x):
            b = b.union(Rect(x, y, x + w, y + h))
        return fr.to_output(b.grow(-shrink * b.w, -shrink * b.h, -shrink * b.w, -shrink * b.h), k)

    return at


class Cluster:
    def __init__(self, clip, opts: dict, timeline, framing, head_at, face_at, frame_at,
                 plan: Optional[dict] = None, core_at=None):
        self.clip, self.opts = clip, opts
        self.timeline, self.framing = timeline, framing
        self.W, self.H = framing.out_w, framing.out_h
        self.U = self.W / 1080.0
        self.fps = clip.info.rate
        self.n = timeline.output_frames(self.fps)
        self.seconds = self.n / self.fps
        self.zone = SafeZone(self.W, self.H)
        self.head_at, self.face_at, self.frame_at = head_at, face_at, frame_at
        self.core_at = core_at or core_in_output(clip, timeline, framing)
        self.heads = [head_at(j) for j in range(self.n)]
        self.faces = [face_at(j) for j in range(self.n)]
        self.depth = float(opts["depth"])
        self.warnings: list[str] = []
        self.found: list[str] = []
        self.layout = "around" if opts["layout"] == "auto" else opts["layout"]
        self._words_cache: Optional[list[Word]] = None
        # the glass's kind is chosen from the room first: a figure's card draws its line in the accent
        self.tone_name = opts["tone"] if opts["tone"] != "auto" else self._room_tone()
        if plan is not None and (plan.get("cluster") or {}).get("tone") in ("dark", "light", "tint"):
            self.tone_name = plan["cluster"]["tone"]
        self.accent_hex = "#%02x%02x%02x" % self._accent(self.tone_name)
        if plan is not None:
            self._from_plan(plan)
        else:
            self._decide()
        self._drift()
        if plan is None:
            self._place()
        self._dress()
        for i, tile in enumerate(self.tiles):
            tile.index = i
            tile.build(self.look)
        self._prepare_draw()

    # -- what, and when -----------------------------------------------------------------------------------
    def _words(self) -> list[Word]:
        """The words as effects use them: corrected, without hesitations or words the transcriber was
        unsure of (a word that was never said must never time a widget)."""
        if self._words_cache is None:
            _, _, timed = first_listen(self.clip)
            self._words_cache = [w for w in timed if w.key not in HESITATIONS and w.prob >= 0.5]
        return self._words_cache

    def _room_tone(self) -> str:
        """Dark glass stands off a bright room, light glass off a dark one: the room's brightness above
        the shoulders at three moments."""
        z = self.zone.rect
        lum = []
        for q in (0.2, 0.5, 0.8):
            f = self.frame_at(int(min(self.n - 1, round(q * self.n))))
            part = f[int(z.y0):int(0.5 * self.H), int(z.x0):int(z.x1)].astype(np.float32)
            lum.append(float(np.percentile(glassmod.luma(part), 60)))
        return "dark" if float(np.median(lum)) >= 105.0 else "light"

    def _chart_opts(self, spec: Spec, when: str) -> dict:
        kind = "counter" if spec.kind == "number" else (spec.style or "line")
        label = "auto" if spec.text.lower() == "auto" else spec.text
        return {"type": kind, "values": spec.value or "auto", "label": label, "when": when, "position": "auto",
                "accent": self.accent_hex, "prefix": "auto", "suffix": "auto", "style": "glass", "duration": 0.0,
                "sound": self.opts["sound"], "fix": self.opts["fix"]}

    def _make_tile(self, spec: Spec) -> Tile:
        if spec.kind in ("number", "stat"):
            size = (self.W, self.H)

            def make(when: str, spec=spec):
                return _Card(self.clip, self._chart_opts(spec, when), size, self.fps, self.n, self.head_at,
                             self.face_at, self.frame_at)
            return ChartTile(spec, self.U, make)
        return TILES[spec.kind](spec, self.U)

    def _accent(self, tone: str) -> tuple:
        a = self.opts["accent"]
        return textmod.colour(ACCENT[tone] if a == "auto" else a)

    def _decide(self) -> None:
        opts = self.opts
        name = opts["name"].strip()
        if opts["widgets"] == "auto":
            self.specs, _ = auto_specs(self.clip, name)
        else:
            try:
                self.specs = parse_widgets(opts["widgets"], name)
            except O.OptionError as e:
                raise O.OptionError(f"widgets: {e}") from None
            if name and not any(s.kind == "name" for s in self.specs) and len(self.specs) < 5:
                self.specs.append(Spec("name", text=name))
            self._find_said()
        self.tiles = [self._make_tile(s) for s in self.specs]
        self._schedule()
        self._layers()

    def _find_said(self) -> None:
        """Given widgets that are said (a name, a figure) land where they are said."""
        words = self._words()
        for s in self.specs:
            if s.when or s.at is not None:
                continue
            probe = None
            if s.kind == "name":
                probe = s.text.lstrip("@")
            if probe:
                hit = find_said(words, probe)
                if hit:
                    s.at, s.said, s.found = words[hit[0]].start, words[hit[0]].clean, "said"

    def _beats(self) -> list[float]:
        return sorted({round(w.start, 3) for w in self._words()})

    def _anchor(self, when: str, words: list[Word]) -> Optional[float]:
        if not when:
            return None
        t = _parse_time(when)
        if t is not None:
            return t
        hit = find_said(words, when)
        if not hit:
            raise O.OptionError(f"when: the words “{when}” are not said in this video. Read the words with "
                                f"`python -m fx.run words <video>` and use them as heard, or give a time in seconds")
        return words[hit[0]].start

    def _slot(self, want: float, taken: list[float], beats: list[float]) -> float:
        """The first word that starts at or soon after `want` and is not within GAP of another arrival;
        with none near, the first moment after `want` that is."""
        def clear(t: float) -> bool:
            return all(abs(t - u) >= GAP - 1e-6 for u in taken)
        for b in beats:
            if want - 0.05 <= b <= want + 1.2 and clear(b):
                return b
        t = want
        while not clear(t) and t < self.seconds:
            t += 0.05
        return t

    def _schedule(self) -> None:
        """When each widget arrives: on the words that say it, or across the words or the moment `when`
        names, one after another on the words that follow, never two at once."""
        words = self._words()
        beats = self._beats()
        when = self.opts["when"]
        cluster_when = None if when == "auto" else when
        span_beats: Optional[list[float]] = None
        start_at: Optional[float] = None
        if cluster_when is not None:
            t = _parse_time(cluster_when)
            if t is not None:
                if t >= self.seconds - 0.5:
                    raise O.OptionError(f"when: {t:g} s is at the very end of the video ({self.seconds:.2f} s)")
                start_at = t
                self.found.append(f"starting at {t:.2f} s")
            else:
                hit = find_said(words, cluster_when)
                if not hit:
                    raise O.OptionError(f"when: the words “{cluster_when}” are not said in this video. Read the "
                                        f"words with `python -m fx.run words <video>` and use them as heard, or give "
                                        f"a time in seconds")
                span_beats = [w.start for w in words[hit[0]:hit[1] + 1]]
                said = " ".join(w.clean for w in words[hit[0]:hit[1] + 1])
                self.found.append(f"across the words “{said}”")
        n = len(self.tiles)
        fixed: dict[int, float] = {}
        moves: set[int] = set()  # fixed, but free to step aside a little (a pill, a note)
        for i, tile in enumerate(self.tiles):
            s = tile.spec
            if s.when:
                fixed[i] = self._anchor(s.when, words)
                moves.add(i)
                continue
            if cluster_when is not None or s.at is None:
                continue
            if isinstance(tile, ChartTile) and s.found == "said":
                tile.prepare("auto")  # chart lands a said figure on its word; the card arrives before
                if tile.card.said_all or s.value.lower() == "auto":
                    fixed[i] = tile.card.start
                    continue
                tile.card = None
            elif isinstance(tile, ProgressTile) and s.found == "said":
                fixed[i] = max(0.15, s.at - (ENTER * 0.5 + COUNT))  # it fills as the share is said
                moves.add(i)
                continue
            fixed[i] = s.at
            moves.add(i)
        # two said close together: the one that can move steps aside
        seq = sorted(fixed, key=lambda i: fixed[i])
        for a, b in zip(seq, seq[1:]):
            if fixed[b] - fixed[a] < GAP and b in moves:
                fixed[b] = fixed[a] + GAP
        free = [i for i in range(n) if i not in fixed]
        if span_beats:
            m, f = len(span_beats), len(free)
            if f == 1:
                wishes = [span_beats[0]]
            elif m >= f:
                wishes = [span_beats[round(k * (m - 1) / (f - 1))] for k in range(f)]
            else:
                wishes = span_beats + [span_beats[-1] + GAP * (k + 1) for k in range(f - m)]
        else:
            if start_at is not None:
                base = start_at
            elif fixed:
                base = min(fixed.values()) + GAP
            else:
                base = beats[0] if beats else 0.3
                self.found.append("from the first words")
            wishes = None
        arrivals = dict(fixed)
        taken = list(fixed.values())
        prev = None
        for k, i in enumerate(free):
            want = wishes[k] if wishes is not None else (base if prev is None else prev + GAP)
            t = self._slot(max(0.15, want), taken, beats)
            arrivals[i] = t
            taken.append(t)
            prev = t
        for i, tile in enumerate(self.tiles):
            tile.at = arrivals[i]
            if isinstance(tile, ChartTile):
                if tile.card is None:
                    try:
                        tile.prepare(f"{tile.at:.3f}")
                    except O.OptionError as e:
                        raise O.OptionError(f"{tile.spec.kind} {tile.spec.value}: {e}") from None
                tile.at = tile.card.start
                tile.land = tile.card.land
                self.warnings += [f"{tile.spec.kind}: {w}" for w in tile.card.warnings]
            elif isinstance(tile, ProgressTile):
                tile.land = tile.at + ENTER * 0.5 + COUNT
            else:
                tile.land = tile.at + ENTER
        late = [t for t in self.tiles if t.at > self.seconds - 0.8]
        for t in late:
            self.warnings.append(f"{t.report()} would arrive at {t.at:.2f} s, too near the end; it was left out")
        self.tiles = [t for t in self.tiles if t not in late]
        if not self.tiles:
            raise O.OptionError("every widget would arrive in the last moment of the video; give an earlier when=")
        self.tiles.sort(key=lambda t: t.at)
        self.start = min(t.at for t in self.tiles)
        self.last_land = max(t.land for t in self.tiles)
        hold = self.opts["hold"]
        end = self.seconds if hold == "end" else self.last_land + (HOLD if hold == "auto" else float(hold))
        end = max(end, max(t.at for t in self.tiles) + MIN_ON)
        self.end = min(self.seconds, end)
        if self.seconds - self.end < 0.4:
            self.end = self.seconds
        self._leaving()
        self.sweep_at = self.last_land + SWEEP_AFTER

    def _leaving(self) -> None:
        """They leave together: first in, last out, a breath apart; or they hold to the end."""
        self.leaving = self.end < self.seconds - 1e-3
        for i, tile in enumerate(self.tiles):  # in the order they arrived
            tile.leave = (self.end - EXIT - i * EXIT_STAGGER) if self.leaving else None
            if isinstance(tile, ChartTile):
                tile.card.end = (tile.leave + EXIT) if self.leaving else self.seconds

    def _layers(self) -> None:
        """Which widgets tuck behind the person: the larger ones, unless the layout is a row above the
        head or there is no depth; the smaller float in front."""
        flat = self.depth <= 0 or self.layout == "above"
        for tile in self.tiles:
            if tile.spec.layer:
                tile.layer = tile.spec.layer
            elif flat:
                tile.layer = "front"
            else:
                tile.layer = "back" if tile.kind in ("number", "stat", "note") else "front"
        if not flat and len(self.tiles) >= 2 and all(t.layer == "back" for t in self.tiles):
            min(self.tiles, key=lambda t: t.size(0.7, t.base_shapes[0])[0] * t.size(0.7, t.base_shapes[0])[1]).layer = "front"
        # one widget leads: the first figure or chart, else the first line; drawn larger, so the cluster
        # has a hierarchy rather than a row of equals
        leads = [t for t in self.tiles if t.kind in ("stat", "number", "note")]
        self.lead: Optional[Tile] = None
        if len(self.tiles) >= 3 and leads:
            self.lead = min(leads, key=lambda t: (PRIORITY.index(t.kind), t.at))
            self.lead.boost = HERO

    # -- depth: how each widget drifts -------------------------------------------------------------------
    def _drift(self) -> None:
        n = self.n
        cx = np.array([h.cx if h is not None else np.nan for h in self.heads], np.float64)
        cy = np.array([h.cy if h is not None else np.nan for h in self.heads], np.float64)
        ok = ~np.isnan(cx)
        a = int(max(0, math.floor(self.start * self.fps)))
        b = int(min(n - 1, math.ceil(self.end * self.fps)))
        self.span = (a, b)
        frames = np.arange(n)
        if ok.sum() >= 2:
            cx = np.interp(frames, frames[ok], cx[ok])
            cy = np.interp(frames, frames[ok], cy[ok])
            sig = max(1.0, 0.25 * self.fps)
            r = int(3 * sig)
            k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
            k /= k.sum()
            cx = np.convolve(np.pad(cx, r, mode="edge"), k, mode="valid")
            cy = np.convolve(np.pad(cy, r, mode="edge"), k, mode="valid")
            ref = (float(cx[a:b + 1].mean()), float(cy[a:b + 1].mean()))
        else:
            cx = np.zeros(n)
            cy = np.zeros(n)
            ref = (0.0, 0.0)
        cap = DRIFT_MAX * self.W
        t = frames / self.fps
        self.offsets: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for tile in self.tiles:
            f = FOLLOW[tile.layer] * self.depth
            amp = FLOAT * self.U * self.depth
            # each widget floats at its own pace, fixed by what it says, so a plan renders the same
            i = zlib.crc32(f"{tile.kind}:{tile.spec.text}:{tile.spec.value}".encode()) % 997 / 997.0
            px, py = 4.1 + 1.4 * i, 5.3 + 1.1 * i
            fx, fy = 2 * math.pi * i, 2 * math.pi * ((i * 7.0) % 1.0)
            dx = np.clip(f * (cx - ref[0]), -cap, cap) + amp * np.sin(2 * math.pi * t / px + fx)
            dy = np.clip(f * (cy - ref[1]), -cap, cap) + amp * np.sin(2 * math.pi * t / py + fy)
            self.offsets[id(tile)] = (dx.astype(np.float32), dy.astype(np.float32))

    def window(self, tile: Tile) -> tuple[int, int]:
        """The output frames `tile` is on screen."""
        a = int(max(0, math.floor(tile.at * self.fps)))
        end = (tile.leave + EXIT) if tile.leave is not None else self.end
        b = int(min(self.n - 1, math.ceil(end * self.fps)))
        return a, max(a, b)

    def extent(self, tile: Tile, k: Optional[float] = None) -> tuple[float, float, float, float]:
        """How far the widget's drawing reaches past its place over its time on screen, at scale `k`
        (its own, once put): (left, right, top, bottom) as offsets, its drift, the overshoot as it grows
        in and its shadow included, so all of it is kept inside the safe zone."""
        dx, dy = self.offsets[id(tile)]
        a, b = self.window(tile)
        grow = 3.0 * self.U  # the little overshoot as it grows in
        left, top, right, bottom = glassmod.Pane.reach((tile.k if k is None else k) * self.U)
        return (float(dx[a:b + 1].min()) - grow - left, float(dx[a:b + 1].max()) + grow + right,
                float(dy[a:b + 1].min()) - grow - top, float(dy[a:b + 1].max()) + grow + bottom)

    def position(self, tile: Tile, j: int) -> tuple[float, float]:
        dx, dy = self.offsets[id(tile)]
        j = int(min(len(dx) - 1, max(0, j)))
        return tile.x + float(dx[j]), tile.y + float(dy[j])

    # -- where: one cluster round the head ------------------------------------------------------------------
    def _span_frames(self, most: int = 48) -> list[int]:
        a, b = self.span
        step = max(1, int(math.ceil((b - a + 1) / most)))
        return list(range(a, b + 1, step))

    def _measure_person(self) -> None:
        """The person over the frames the cluster is up, sampled: their mask (as integral images, so
        the share of any box they cover is four reads), the face kept clear of by a widget in front,
        and the middle of the face, which even a widget behind never sits behind."""
        a, b = self.span
        self.sample = self._span_frames(40)
        store = MaskStore(self.clip, self.timeline, self.framing, (self.W, self.H), self.sample)
        blank = np.zeros((max(2, self.H // Q), max(2, self.W // Q)), np.float32)
        self.I_frames = [_integral((store.by_frame[j] >= 0.5).astype(np.float32) if j in store.by_frame else blank)
                         for j in self.sample]
        self.keeps = [self.faces[j] for j in self.sample]
        self.cores = [self.core_at(j) for j in self.sample]
        heads = [h for h in self.heads[a:b + 1] if h is not None]
        self.head = (Rect(float(np.mean([h.x0 for h in heads])), float(np.mean([h.y0 for h in heads])),
                          float(np.mean([h.x1 for h in heads])), float(np.mean([h.y1 for h in heads])))
                     if heads else None)
        self.keep = Rect.around(f for f in self.faces[a:b + 1] if f is not None)

    def _place(self) -> None:
        self._measure_person()
        self._fields: dict = {}
        order = sorted(self.tiles, key=lambda t: (-t.boost, PRIORITY.index(t.kind), t.at))
        tiles = list(order)
        given = self.opts["widgets"] != "auto"
        got = None
        while True:
            # readable first: the labels at a size a phone shows clearly; widgets a person listed may
            # come a little smaller before one is left out, ones found in the clip are left out first
            for floor in ((K_READ, K_GIVEN) if given else (K_READ,)):
                got = self._fit(tiles, floor)
                if got is not None:
                    break
            if got is not None or len(tiles) <= 2:
                break
            drop = tiles[-1]
            self.warnings.append(f"there was not room for every widget at a size that reads, clear of the face; "
                                 f"{drop.report()} was left out")
            tiles = tiles[:-1]
        if got is None:
            got = self._fit(tiles, K_MIN)
        if got is None:
            got = self._fit(tiles, K_MIN, margin=True)
            if got is None:
                raise O.OptionError("the face fills the frame, so there is nowhere to put the widgets without covering "
                                    "it; try a shot with more room round the head")
            self.warnings.append("there was no room inside the safe zone clear of the face; the widgets sit in the "
                                 "margin, smaller")
        k, places = got
        self.k = k
        if len(places) < len(self.tiles):  # some were left out: the timing closes up round the rest
            self.tiles = [t for t in self.tiles if t in places]
            self.last_land = max(t.land for t in self.tiles)
            self._leaving()
            self.sweep_at = self.last_land + SWEEP_AFTER
        for tile in self.tiles:  # on whole pixels, so a plan written from it holds them exactly
            x, y, shape = places[tile]
            tile.put(round(x), round(y), round(k * tile.boost, 4), shape)
        if self.k < K_READ - 1e-6:
            self.warnings.append(f"the widgets are small ({self.k:.0%} of their size): there is little room round "
                                 f"the head on this shot")
        self.where = self._describe()

    def _orders(self, tiles: list[Tile]) -> list[list[Tile]]:
        """A few orders to place the widgets in, the lead first: each order gives another cluster, and
        the best of them is kept."""
        lead, rest = tiles[:1], tiles[1:]
        orders = [rest, rest[::-1]]
        for r in range(1, len(rest)):
            orders.append(rest[r:] + rest[:r])
        seen, out = set(), []
        for o in orders:
            key = tuple(id(t) for t in o)
            if key not in seen:
                seen.add(key)
                out.append(lead + o)
        return out[:6]

    def _fit(self, tiles: list[Tile], floor: float, margin: bool = False):
        """The largest scale, down to `floor`, at which every widget finds a place; at that scale, the
        best of a few clusters (each order places the widgets in turn and gives another)."""
        k = K_MAX
        lead = getattr(self, "lead", None)
        lead = lead if lead in tiles else None
        while k >= floor - 1e-6:
            best = None
            for boost in ((HERO, 1.0) if lead is not None else (1.0,)):
                if lead is not None:
                    lead.boost = boost  # a lead drawn larger, unless that costs everyone a readable size
                best = self._best_order(tiles, k, margin)
                if best is not None:
                    return k, best
            k *= K_STEP
        return None

    def _best_order(self, tiles: list[Tile], k: float, margin: bool):
        best = None
        for order in self._orders(tiles):
            places: dict = {}
            placed: list[tuple[Tile, Rect]] = []
            total = 0.0
            for tile in order:
                pick = None
                for shape in tile.shapes:
                    got = self._search(tile, k * tile.boost, shape, placed, margin)
                    if got is not None and (pick is None or got[0] > pick[0]):
                        pick = got
                if pick is None:
                    break
                sc, x, y, shape, rect = pick
                places[tile] = (x, y, shape)
                placed.append((tile, rect))
                total += sc
            if len(places) < len(tiles):
                continue
            total += self._cluster_score([r for _, r in placed])
            if best is None or total > best[0]:
                best = (total, places)
        return best[1] if best is not None else None

    def _cluster_score(self, rects: list[Rect]) -> float:
        """How well the widgets sit together: balanced either side of the head, and compact."""
        hcx = self.head.cx if self.head is not None else self.W / 2
        areas = [r.w * r.h for r in rects]
        tot = sum(areas)
        cx = sum(r.cx * a for r, a in zip(rects, areas)) / tot
        left = sum(a for r, a in zip(rects, areas) if r.cx < hcx)
        right = tot - left
        n_left = sum(1 for r in rects if r.cx < hcx - 0.04 * self.W)
        n_right = sum(1 for r in rects if r.cx > hcx + 0.04 * self.W)
        box = Rect.around(rects)
        return (-3.0 * abs(cx - hcx) / self.W - 1.2 * abs(left - right) / tot
                - 0.9 * abs(n_left - n_right) / len(rects)
                - 1.0 * max(0.0, box.w * box.h / tot - 1.0))

    def _zone(self, margin: bool) -> Rect:
        z = self.zone.rect if not margin else Rect(0.03 * self.W, 0.02 * self.H, 0.97 * self.W, 0.9 * self.H)
        if self.layout == "above" and self.head is not None:
            z = Rect(z.x0, z.y0, z.x1, min(z.y1, self.head.y0 - HUG_FRONT * self.H))
        return z

    def _zone_for(self, margin: bool, k: float) -> Rect:
        """What a widget at scale k with its shadow (`extent`) must stay inside: `_zone`, except that
        on a side drawn by the head rather than the frame's edge the shadow may fall past it."""
        z = self._zone(margin)
        outer = self.zone.rect if not margin else Rect(0.03 * self.W, 0.02 * self.H, 0.97 * self.W, 0.9 * self.H)
        left, top, right, bottom = glassmod.Pane.reach(k * self.U)
        return Rect(z.x0 - (left if z.x0 > outer.x0 + 0.5 else 0.0), z.y0 - (top if z.y0 > outer.y0 + 0.5 else 0.0),
                    z.x1 + (right if z.x1 < outer.x1 - 0.5 else 0.0), z.y1 + (bottom if z.y1 < outer.y1 - 0.5 else 0.0))

    def _person_checks(self, tile: Tile, X, Y, w: float, h: float, k: float, shape: str):
        """Frame by frame, with the widget where its drift takes it: never over the face (a widget
        behind may pass behind the hair and the jaw, never behind the middle of the face), and the
        writing of a widget behind never hidden by the person. Returns what passes, and the share of
        the widget the person covers, on average and at most."""
        dxs, dys = self.offsets[id(tile)]
        c = tile.content(k, shape)
        ok = np.ones(np.shape(X), bool)
        cover = np.zeros(np.shape(X), np.float64)
        most = np.zeros(np.shape(X), np.float64)
        m = FACE_MARGIN * self.H
        back = tile.layer == "back"
        a, b = self.window(tile)
        seen = 0
        for idx, j in enumerate(self.sample):
            if not a <= j <= b:
                continue  # it is not on screen yet, or has gone
            seen += 1
            ox, oy = float(dxs[j]), float(dys[j])
            x0, y0, x1, y1 = X + ox, Y + oy, X + w + ox, Y + h + oy
            box = self.cores[idx] if back else self.keeps[idx]
            if box is not None:
                if not back:
                    box = box.grow(m, m, m, m)
                ok &= ~((x0 < box.x1) & (x1 > box.x0) & (y0 < box.y1) & (y1 > box.y0))
            I = self.I_frames[idx]
            share = _share_in(I, x0, y0, x1, y1)
            cover += share
            most = np.maximum(most, share)
            if back:
                ok &= _share_in(I, X + c.x0 + ox, Y + c.y0 + oy, X + c.x1 + ox, Y + c.y1 + oy) <= CONTENT_HIDDEN
        return ok, cover / max(1, seen), most

    def _field(self, tile: Tile, k: float, shape: str, margin: bool):
        """Everything about a widget's places that does not depend on the others: the grid, what is
        allowed there by the frame and the person, and its own score. Kept, so trying the widgets in
        another order costs little."""
        key = (id(tile), round(k, 4), shape, margin)
        if key in self._fields:
            return self._fields[key]
        w, h = tile.size(k, shape)
        dx0, dx1, dy0, dy1 = self.extent(tile, k)
        z = self._zone_for(margin, k)
        step = max(3.0, 9.0 * self.U)
        xs = np.arange(z.x0 - dx0, z.x1 - w - dx1 + 1e-6, step)
        ys = np.arange(z.y0 - dy0, z.y1 - h - dy1 + 1e-6, step)
        if not len(xs) or not len(ys):
            self._fields[key] = None
            return None
        X, Y = np.meshgrid(xs, ys)
        SX0, SX1, SY0, SY1 = X + dx0, X + w + dx1, Y + dy0, Y + h + dy1
        ok = np.ones(X.shape, bool)
        if self.head is None and not margin:
            ok &= SY1 <= self.zone.rect.y0 + 0.36 * self.H  # no face found: the band near the top
        passes, cover, most = self._person_checks(tile, X, Y, w, h, k, shape)
        ok &= passes
        score = np.zeros(X.shape, np.float64)
        if tile.layer == "back":
            ok &= most <= TILE_HIDDEN
            if _parts(shape)[1]:  # a tuck is there to be crossed by the person; one that is not is just more glass
                ok &= cover >= TUCK[0]
                score += 1.6 * ((cover >= TUCK[0]) & (cover <= TUCK[1])) + 0.6
            else:
                score += 0.8 * ((cover >= 0.03) & (cover <= TUCK[1]))
        else:
            score -= 2.6 * cover  # in front, over the person: allowed, never over the face, not wanted
        if self.head is not None:
            Hh = self.head
            gx = np.maximum(np.maximum(Hh.x0 - SX1, SX0 - Hh.x1), 0)
            gy = np.maximum(np.maximum(Hh.y0 - SY1, SY0 - Hh.y1), 0)
            gap = np.hypot(gx, gy)
            want = HUG_FRONT * self.H if tile.layer == "front" else 0.0
            score -= 2.2 * np.abs(gap - want) / (0.1 * self.W)
            # the upper arc: above the face's middle, never hanging low beside the jaw
            mid = self.keep.cy if self.keep is not None else Hh.cy
            score -= 2.4 * np.clip(Y + h / 2 - mid, 0, None) / self.H
        else:
            score -= 1.0 * (Y + h / 2) / self.H
        got = (X, Y, w, h, (SX0, SX1, SY0, SY1), ok, score)
        self._fields[key] = got
        return got

    def _search(self, tile: Tile, k: float, shape: str, placed: list, margin: bool):
        """The best place for `tile` at scale `k` among the others already placed, or None: every spot
        on a fine grid is tried at once and scored (whole arrays, so it costs milliseconds)."""
        field = self._field(tile, k, shape, margin)
        if field is None:
            return None
        X, Y, w, h, (SX0, SX1, SY0, SY1), ok0, score0 = field
        ok = ok0.copy()
        g = GUTTER * k * self.U
        for _, r in placed:
            ok &= ~((SX0 < r.x1 + g) & (SX1 > r.x0 - g) & (SY0 < r.y1 + g) & (SY1 > r.y0 - g))
        if not ok.any():
            return None
        score = score0.copy()
        cxs = X + w / 2
        hcx = self.head.cx if self.head is not None else self.W / 2
        if tile is getattr(self, "lead", None) and tile.boost > 1.0:
            score -= 2.0 * np.abs(cxs - hcx) / self.W  # the lead stands over the head
        # balance: the cluster's middle near the head's, and each widget on the side the last was not
        xs_all = [r.cx for _, r in placed]
        ws_all = [r.w * r.h for _, r in placed]
        tot = sum(ws_all) + w * h
        centroid = (sum(c * a for c, a in zip(xs_all, ws_all)) + cxs * w * h) / tot
        score -= 1.6 * np.abs(centroid - hcx) / self.W
        if placed:
            prev = placed[-1][1].cx - hcx
            score += 0.5 * ((cxs - hcx) * np.sign(prev) < 0)
            # close to a neighbour, a gutter away: one cluster, not scattered widgets
            near = np.full(X.shape, np.inf)
            for _, r in placed:
                ddx = np.maximum(np.maximum(r.x0 - (X + w), X - r.x1), 0)
                ddy = np.maximum(np.maximum(r.y0 - (Y + h), Y - r.y1), 0)
                near = np.minimum(near, np.hypot(ddx, ddy))
            score -= 1.6 * np.clip(near - 1.6 * g, 0, None) / (0.1 * self.W)
            # lined up with a neighbour's edge
            al = np.zeros(X.shape, np.float64)
            tol = 6.0 * self.U
            for _, r in placed:
                al += ((np.abs(Y - r.y0) < tol) | (np.abs(Y + h - r.y1) < tol)).astype(np.float64)
                al += ((np.abs(X - r.x0) < tol) | (np.abs(X + w - r.x1) < tol)).astype(np.float64)
            score += 0.45 * np.minimum(al, 2.0)
        if self.layout == "bento":
            score += self._bento_bonus(X, Y, w, h, placed, k)
        score = np.where(ok, score, -np.inf)
        i = int(np.argmax(score))
        if not np.isfinite(score.flat[i]):
            return None
        x, y = float(X.flat[i]), float(Y.flat[i])
        x, y = self._snap(x, y, w, h, placed, tile, k, shape, margin)
        return float(score.flat[i]), x, y, shape, Rect(x, y, x + w, y + h)

    def _bento_bonus(self, X, Y, w, h, placed, k):
        """On a bento grid every widget stands against one side of the safe zone, or against a
        neighbour a gutter away, and lines up with its row: tiles that frame the head."""
        z = self.zone.rect
        g = GUTTER * k * self.U
        tol = 5.0 * self.U
        bonus = 1.2 * ((np.abs(X - z.x0) < tol) | (np.abs(X + w - z.x1) < tol))
        for _, r in placed:
            row = (np.abs(Y - r.y0) < tol) | (np.abs(Y + h - r.y1) < tol)
            stack = (np.abs(Y - (r.y1 + g)) < tol) | (np.abs(Y + h + g - r.y0) < tol)
            side = (np.abs(X - (r.x1 + g)) < tol) | (np.abs(X + w + g - r.x0) < tol)
            bonus = bonus + 0.8 * (row | stack) + 0.6 * side
        return bonus

    def _snap(self, x, y, w, h, placed, tile, k, shape, margin):
        """Nudge onto a neighbour's edge a few pixels away, when that stays clear of everything."""
        tol = 10.0 * self.U
        cand = []
        for _, r in placed:
            for yy in (r.y0, r.y1 - h):
                if 0 < abs(y - yy) < tol:
                    cand.append((x, yy))
            for xx in (r.x0, r.x1 - w):
                if 0 < abs(x - xx) < tol:
                    cand.append((xx, y))
        for cx_, cy_ in cand:
            if self._clear(tile, cx_, cy_, w, h, k, shape, placed, margin):
                return cx_, cy_
        return x, y

    def _clear(self, tile, x, y, w, h, k, shape, placed, margin) -> bool:
        dx0, dx1, dy0, dy1 = self.extent(tile, k)
        r = Rect(x + dx0, y + dy0, x + w + dx1, y + h + dy1)
        if not r.inside(self._zone_for(margin, k)):
            return False
        g = GUTTER * k * self.U
        if any(r.intersects(p, margin=g - 0.5) for _, p in placed):
            return False
        ok, cover, most = self._person_checks(tile, np.array([x]), np.array([y]), w, h, k, shape)
        if tile.layer == "back" and float(most[0]) > TILE_HIDDEN:
            return False
        return bool(ok[0])

    def _describe(self) -> str:
        if self.head is None:
            return "in the band near the top (no face was found)"
        above = sum(1 for t in self.tiles if t.rect.y1 <= self.head.y0 + 0.05 * self.H)
        beside = len(self.tiles) - above
        bits = []
        if above:
            bits.append(f"{above} above the head")
        if beside:
            bits.append(f"{beside} beside or below it")
        return ", ".join(bits)

    # -- the look -----------------------------------------------------------------------------------
    def _regions(self) -> list[np.ndarray]:
        out = []
        for t in (self.start + 0.15, (self.start + self.end) / 2, self.end - 0.15):
            j = int(min(self.n - 1, max(0, round(t * self.fps))))
            f = self.frame_at(j)
            for tile in self.tiles:
                r = tile.rect
                x0, y0 = int(max(0, r.x0 - 20)), int(max(0, r.y0 - 20))
                x1, y1 = int(min(self.W, r.x1 + 20)), int(min(self.H, r.y1 + 20))
                if x1 > x0 and y1 > y0:
                    out.append(f[y0:y1, x0:x1])
        return out

    def _dress(self) -> None:
        """The cluster's one glass, toned over what is behind every widget at three moments."""
        regions = self._regions()
        tone = self.tone_name
        accent = self._accent(tone)
        sigma = 15.0 * self.k * self.U
        if tone == "light":
            g, darkest = glassmod.light_tone_for(regions, sigma)
            ink, soft = LIGHT_INK, LIGHT_SOFT
            contrast = glassmod.contrast(darkest, glassmod.brightness(soft))
            on_accent = (255, 255, 255) if glassmod.brightness(accent) < 150 else LIGHT_INK
            look = Look("light", g, ink, soft, accent, on_accent, contrast, shadow=False)
        else:
            tint = _deep(accent) if tone == "tint" else glassmod.SLATE
            g, bright = glassmod.tone_for(regions, sigma, tint=tint)
            if tone == "tint" and g.smoke < 0.55:
                k0 = max(0.0, g.smoke - glassmod.GRADE)
                behind_ = ((bright - 255 * g.lift) / (1 - g.lift) - glassmod.brightness(tint) * k0) / max(1e-3, 1 - k0)
                g = glassmod.Tone(smoke=0.55, lift=g.lift, tint=tint)
                bright = g.brightest(behind_)
            if tone == "tint":
                accent = _mix(accent, (255, 255, 255), 0.5)
            contrast = glassmod.contrast_with_white(bright)
            on_accent = (12, 20, 18) if glassmod.brightness(accent) > 150 else (255, 255, 255)
            look = Look(tone, g, DARK_INK, DARK_SOFT, accent, on_accent, contrast, shadow=True)
        self.look = look
        self.accent_hex = "#%02x%02x%02x" % tuple(int(c) for c in accent)

    # -- drawing ----------------------------------------------------------------------------------------
    def _prepare_draw(self) -> None:
        self.masks = self.clip.person() if any(t.layer == "back" for t in self.tiles) else None
        self._iter = None
        self._k = -1
        self._last: Optional[np.ndarray] = None
        self.sigma = max(4.0, 0.012 * self.W)
        self.soft = SOFT * self.k * self.U * self.depth
        self.box = Rect.around(t.rect for t in self.tiles)

    def _mask(self, k: int) -> np.ndarray:
        if self._iter is None or k < self._k:
            self._iter, self._k = iter(self.masks.frames()), -1
        while self._k < k:
            nxt = next(self._iter, None)
            if nxt is None:
                break
            self._last, self._k = nxt, self._k + 1
        assert self._last is not None
        return self._last

    def person_alpha(self, frame: np.ndarray, j: int, y0: int, y1: int) -> np.ndarray:
        k = self.timeline.source_frame(j, self.clip.info.rate, self.clip.info.frames)
        crop = self.framing.window(k) if self.framing.reframed else None
        a = self.masks.full(self._mask(k), (self.W, self.H), guide=frame, crop=crop, lean=EDGE, reach=REACH * self.W)
        return refine_edge(frame[y0:y1], a[y0:y1], self.sigma * 0.75)

    def _sweep(self, target: np.ndarray, tile: Tile, x: float, y: float, t: float, oy: int = 0) -> None:
        """One band of light across the cluster, once everything has landed. `target` may be a strip
        of the frame starting `oy` rows down."""
        q = (t - self.sweep_at) / SWEEP
        if not 0.0 <= q < 1.0:
            return
        pane = tile.glass
        st = tile.state(t)
        if st is None or st.opacity < 0.5:
            return
        x0, y0 = int(round(x)), int(round(y))
        H, W = target.shape[:2]
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x0 + pane.w), min(H, y0 + pane.h)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        box = self.box
        slope = 0.42
        lo, hi = box.x0 + slope * box.y0, box.x1 + slope * box.y1
        bw = 0.06 * (hi - lo) + 22 * self.U
        centre = lo - 2 * bw + (hi - lo + 4 * bw) * (0.3 * q + 0.7 * _ease_in_out(q))
        ys, xs = np.mgrid[cy0:cy1, cx0:cx1].astype(np.float32)
        u = xs + slope * (ys + oy) - centre
        band = (np.exp(-(u / bw) ** 2) * 0.24 + np.exp(-(u / (2.6 * bw)) ** 2) * 0.06) * st.opacity
        m = pane.mask[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] * band
        reg = target[cy0:cy1, cx0:cx1].astype(np.float32)
        reg += (255.0 - reg) * m[..., None]
        target[cy0:cy1, cx0:cx1] = np.clip(reg + 0.5, 0, 255).astype(target.dtype)

    def _soften(self, plate: np.ndarray, tile: Tile, x: float, y: float) -> None:
        """The back layer a touch softer than the front, inside each widget's own shape."""
        if self.soft < 0.35:
            return
        import cv2

        pane = tile.glass
        x0, y0 = int(round(x)), int(round(y))
        H, W = plate.shape[:2]
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x0 + pane.w), min(H, y0 + pane.h)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        reg = plate[cy0:cy1, cx0:cx1].astype(np.float32)
        soft = cv2.GaussianBlur(reg, (0, 0), self.soft)
        m = pane.mask[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0][..., None]
        plate[cy0:cy1, cx0:cx1] = np.clip(reg * (1 - m) + soft * m + 0.5, 0, 255).astype(plate.dtype)

    def draw(self, frame: np.ndarray, j: int, t: float) -> None:
        if not (self.start <= t < self.end):
            return
        live = [(tile, *self.position(tile, j)) for tile in self.tiles if tile.on(t)]
        if not live:
            return
        back = [(tile, x, y) for tile, x, y in live if tile.layer == "back"]
        front = [(tile, x, y) for tile, x, y in live if tile.layer != "back"]
        if back:
            pad = int(math.ceil(34 * self.k * self.U)) + 22
            y0 = int(max(0, math.floor(min(y for _, _, y in back)) - pad))
            y1 = int(min(self.H, math.ceil(max(y + tile.h for tile, _, y in back)) + pad))
            if y1 > y0:
                region = frame[y0:y1]
                a = self.person_alpha(frame, j, y0, y1)
                clean = np.clip(without_person(region, a) + 0.5, 0, 255).astype(np.uint8)
                plate = clean.copy()
                for tile, x, y in back:
                    tile.draw(plate, x, y - y0, t, j)
                    self._soften(plate, tile, x, y - y0)
                    self._sweep(plate, tile, x, y - y0, t, oy=y0)
                frame[y0:y1] = behind(region, plate, clean, a)
        for tile, x, y in front:
            tile.draw(frame, x, y, t, j)
            self._sweep(frame, tile, x, y, t)

    # -- the sound ------------------------------------------------------------------------------------
    def chimes(self) -> list[tuple[float, int]]:
        return [(tile.at + 0.06, i) for i, tile in enumerate(sorted(self.tiles, key=lambda t: t.at))]

    def mix(self, sound: np.ndarray, rate: int) -> np.ndarray:
        if not self.opts["sound"]:
            return sound
        out = sound
        for tile in self.tiles:  # a figure's own soft ticks as it counts
            if isinstance(tile, ChartTile):
                out = tile.card.mix(out, rate)
        mono = sound.mean(axis=1)
        win = int(0.03 * rate)
        n = len(mono) // win
        rms = np.sqrt((mono[:n * win].reshape(n, win) ** 2).mean(axis=1) + 1e-12) if n else np.array([1e-3])
        loud = rms[rms > 10 ** (-45 / 20)]
        voice = float(np.percentile(loud, 90)) if loud.size else 0.05
        out = out.copy()
        added = np.zeros(len(out), np.float32)
        for t, i in self.chimes():
            c = _chime(rate, CHIMES[min(i, len(CHIMES) - 1)])
            heard = float(np.sqrt((c[:win] ** 2).sum() / max(1, win)))
            gain = voice * 10 ** (-(CHIME_UNDER - 0.6 * i) / 20) / max(1e-9, heard)
            k0 = int(round(t * rate))
            if k0 >= len(out):
                continue
            m = min(len(c), len(out) - k0)
            added[k0:k0 + m] += c[:m] * gain
        out += added[:, None]
        m = len(added) // win
        busy = np.sqrt((added[:m * win].reshape(m, win) ** 2).mean(axis=1) + 1e-12) if m else np.array([0.0])
        busy = busy[busy > 1e-6]
        self.chime_db = 20 * math.log10(max(1e-9, float(busy.max()) if busy.size else 1e-9) / max(1e-9, voice))
        return np.clip(out, -1.0, 1.0)

    # -- checks, plan and the moments to look at -------------------------------------------------------
    def report(self) -> dict:
        """Every frame the cluster is up: a widget in front over the face (hairline to chin, ear to
        ear), or a widget behind over the middle of the face, where the person would be all that
        hides it (both must be 0); anything outside the safe zone (0)."""
        over_face = outside = 0
        z = self.zone.rect
        for j in range(self.n):
            t = j / self.fps
            if not (self.start <= t < self.end):
                continue
            hit = False
            for tile in self.tiles:
                if not tile.on(t):
                    continue
                x, y = self.position(tile, j)
                r = Rect(x, y, x + tile.w, y + tile.h)
                face = self.core_at(j) if tile.layer == "back" else self.faces[j]
                hit |= face is not None and r.intersects(face)
                if not r.grow(*glassmod.Pane.reach(tile.s)).inside(z):  # its shadow counted
                    outside += 1
                    break
            over_face += int(hit)
        hidden = self._hidden_text()
        arrivals = " · ".join(f"{t.report()} at {t.at:.2f} s" + (f" on “{t.spec.said}”" if t.spec.said and t.spec.found == 'said' else "")
                              for t in sorted(self.tiles, key=lambda t: t.at))
        layers = f"{sum(1 for t in self.tiles if t.layer == 'back')} behind, {sum(1 for t in self.tiles if t.layer != 'back')} in front"
        return {"frames_over_face": over_face, "frames_outside_safe_zone": outside,
                "placed": f"{self.layout}: {len(self.tiles)} widgets at {self.k:.0%}, {layers}; {self.where}",
                "arrivals": arrivals, "found": "; ".join(self.found) or "each where it is said, the rest after",
                "on_screen": f"{self.start:.2f}-{self.end:.2f} s" + ("" if self.leaving else " (held to the end)"),
                "writing_hidden_by_the_person": f"{hidden:.1%} at most",
                "glass": f"{self.look.name}; its type at {self.look.contrast:.1f} : 1 where the glass is closest to it",
                "depth": self._depth_note(),
                "chimes": len(self.chimes()) if self.opts["sound"] else 0,
                "warnings": list(self.warnings)}

    def _depth_note(self) -> str:
        """How far each layer drifts while it is up, in pixels: the parallax."""
        if self.depth <= 0:
            return "flat: every widget in front, still"
        out = {}
        for tile in self.tiles:
            dx, dy = self.offsets[id(tile)]
            a, b = self.window(tile)
            reach = float(np.hypot(dx[a:b + 1] - dx[a:b + 1].mean(), dy[a:b + 1] - dy[a:b + 1].mean()).max())
            out[tile.layer] = max(out.get(tile.layer, 0.0), reach)
        bits = [f"{'behind' if k == 'back' else 'in front'} drift up to {v:.0f} px" for k, v in sorted(out.items())]
        soft = f", behind softened {self.soft:.1f} px" if self.soft >= 0.35 else ""
        return "; ".join(bits) + soft

    def _hidden_text(self) -> float:
        """The most of any widget's writing the person hides, over the frames it is up (measured on
        the person mask at a quarter size)."""
        if not any(t.layer == "back" for t in self.tiles):
            return 0.0
        frames = self._span_frames(60)
        store = MaskStore(self.clip, self.timeline, self.framing, (self.W, self.H), frames)
        worst = 0.0
        for j in frames:
            m = store.by_frame.get(j)
            if m is None:
                continue
            t = j / self.fps
            for tile in self.tiles:
                if tile.layer != "back" or not tile.on(t):
                    continue
                x, y = self.position(tile, j)
                c = tile.content(tile.k, tile.shape)
                x0, y0 = int((x + c.x0) / Q), int((y + c.y0) / Q)
                x1, y1 = int(math.ceil((x + c.x1) / Q)), int(math.ceil((y + c.y1) / Q))
                part = m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
                if part.size:
                    worst = max(worst, float((part >= 0.5).mean()))
        return worst

    def plan(self) -> dict:
        return {"layout": self.layout, "tone": self.look.name, "accent": self.accent_hex, "depth": self.depth,
                "size": round(self.k, 3), "start": round(self.start, 3), "end": round(self.end, 3),
                "widgets": [{"kind": t.kind, "text": (t.card.label if isinstance(t, ChartTile) else t.spec.text),
                             "value": (",".join((f"{pt.label}:" if pt.label else "") + num.Unit.join(x)
                                                for pt, x in zip(t.card.points, t.card.texts))
                                       if isinstance(t, ChartTile) else t.spec.value),
                             "style": (t.card.kind if isinstance(t, ChartTile) and t.kind == "stat" else ""),
                             "at": round(t.at, 3), "land": round(t.land, 3), "layer": t.layer, "shape": t.shape,
                             "size": round(t.k, 4),
                             "rect": t.rect.as_list(), "said": t.spec.said, "found": t.spec.found}
                            for t in self.tiles]}

    def _from_plan(self, plan: dict) -> None:
        c = plan.get("cluster") or {}
        items = c.get("widgets")
        if not isinstance(items, list) or not items:
            raise ValueError("the plan has no widgets; run once without --plan to get one to edit")
        self.layout = str(c.get("layout", self.layout))
        self.k = float(c.get("size", 0.7))
        tone = str(c.get("tone", "dark"))
        self.accent_hex = str(c.get("accent", "#%02x%02x%02x" % self._accent(tone if tone in ACCENT else "dark")))
        self.specs, self.tiles = [], []
        for n, d in enumerate(items, 1):
            style = str(d.get("style", ""))
            spec = _check(Spec(str(d["kind"]), text=str(d.get("text", "")), value=str(d.get("value", "")),
                               layer=str(d.get("layer", "front")), said=str(d.get("said", "")),
                               found=str(d.get("found", "the plan")),
                               style=style if style in ("line", "bars") else ""), f"the plan's widget {n}")
            tile = self._make_tile(spec)
            tile.at = float(d["at"])
            tile.layer = spec.layer or "front"
            if isinstance(tile, ChartTile):
                tile.prepare(f"{tile.at:.3f}")
                tile.at, tile.land = tile.card.start, tile.card.land
            else:
                tile.land = float(d.get("land", tile.at + ENTER))
            self.specs.append(spec)
            self.tiles.append(tile)
            r = Rect.of(d["rect"])
            tile._planned = (r.x0, r.y0, str(d.get("shape", tile.base_shapes[0])), float(d.get("size", self.k)))
        self.tiles.sort(key=lambda t: t.at)
        self.start = min(t.at for t in self.tiles)
        self.last_land = max(t.land for t in self.tiles)
        self.end = min(self.seconds, float(c.get("end", self.last_land + HOLD)))
        self._leaving()
        self.sweep_at = self.last_land + SWEEP_AFTER
        self.found.append("from the plan")
        a = int(max(0, math.floor(self.start * self.fps)))
        b = int(min(self.n - 1, math.ceil(self.end * self.fps)))
        self.span = (a, b)
        for tile in self.tiles:
            x, y, shape, k = tile._planned
            tile.put(x, y, k, shape)
        self.keep = Rect.around(f for f in self.faces[a:b + 1] if f is not None)
        self.head = Rect.around(h for h in self.heads[a:b + 1] if h is not None)
        self.where = self._describe()

    def sheet_times(self) -> list[float]:
        """Six moments: the first widget arriving, the second, the last landing, the light crossing,
        the hold, and the leaving (or late in the hold)."""
        ts = sorted(t.at for t in self.tiles)
        a0 = ts[0] + ENTER * 0.55
        a1 = (ts[1] if len(ts) > 1 else ts[0]) + ENTER * 0.7
        land = self.last_land + 0.08
        sweep = self.sweep_at + SWEEP * 0.45
        hold = (sweep + self.end) / 2
        late = self.end - (EXIT * 0.5 if self.leaving else 0.3)
        picks = [a0, a1, land, sweep, hold, late]
        return sorted(min(self.end - 0.5 / self.fps, max(self.start, p)) for p in picks)


PRIORITY = ("stat", "number", "note", "progress", "name", "icon")
CHIMES = (1318.5, 1568.0, 1760.0, 2093.0, 2349.3)  # E6 G6 A6 C7 D7: each widget a step higher


def _integral(occ: np.ndarray) -> np.ndarray:
    I = np.zeros((occ.shape[0] + 1, occ.shape[1] + 1), np.float64)
    I[1:, 1:] = occ.astype(np.float64).cumsum(0).cumsum(1)
    return I


def _share_in(I: np.ndarray, x0, y0, x1, y1) -> np.ndarray:
    """The share of each box (output pixels, arrays of the same shape) the occupancy covers, read from
    its integral image at a quarter size."""
    h, w = I.shape[0] - 1, I.shape[1] - 1
    a0 = np.clip(np.floor(np.asarray(x0) / Q), 0, w).astype(np.int64)
    b0 = np.clip(np.floor(np.asarray(y0) / Q), 0, h).astype(np.int64)
    a1 = np.clip(np.ceil(np.asarray(x1) / Q), 0, w).astype(np.int64)
    b1 = np.clip(np.ceil(np.asarray(y1) / Q), 0, h).astype(np.int64)
    s = I[b1, a1] - I[b0, a1] - I[b1, a0] + I[b0, a0]
    area = np.maximum(1, (a1 - a0) * (b1 - b0))
    return s / area


def _chime(rate: int, pitch: float) -> np.ndarray:
    """A soft glass chime, a quarter of a second, peak 1: a tone and its bell partial dying away, a
    breath of air at its front. Made here, so the pack carries no sound files."""
    n = int(0.26 * rate)
    t = np.arange(n, dtype=np.float32) / rate
    rng = np.random.default_rng(int(pitch))
    s = (np.sin(2 * np.pi * pitch * t) * np.exp(-t / 0.075)
         + 0.32 * np.sin(2 * np.pi * pitch * 2.76 * t) * np.exp(-t / 0.03)
         + 0.12 * np.sin(2 * np.pi * pitch * 0.5 * t) * np.exp(-t / 0.05)
         + rng.standard_normal(n).astype(np.float32) * np.exp(-t / 0.002) * 0.12)
    s *= np.clip(t / 0.0015, 0, 1)
    return (s / max(1e-6, float(np.abs(s).max()))).astype(np.float32)


# ---- the effect ------------------------------------------------------------------------------------------

def prepare(clip, opts: dict, plan: Optional[dict] = None):
    from ..core import framing as framing_mod
    from ..core import video
    from ..core.timeline import Timeline
    from . import Job, PlanError

    info = clip.info
    fr = framing_mod.plan(info.width, info.height, info.rate, info.frames, clip.face, shape="keep")
    tl = Timeline.whole(info.frames / info.rate)

    def frame_at(j: int) -> np.ndarray:
        k = tl.source_frame(j, info.rate, info.frames)
        return fr.apply(video.read_frame(info, k), k)

    try:
        cluster = Cluster(clip, opts, tl, fr, head_in_output(clip, tl, fr), keepout_in_output(clip, tl, fr), frame_at,
                          plan=plan, core_at=core_in_output(clip, tl, fr))
    except ValueError as e:
        if plan is not None and not isinstance(e, O.OptionError):
            raise PlanError(str(e)) from None
        raise
    times = cluster.sheet_times()
    out_plan = {"effect": NAME, "clip": clip.path, "options": opts,
                "output": {"width": fr.out_w, "height": fr.out_h, "fps": round(info.rate, 3),
                           "seconds": round(tl.duration, 3), "framing": fr.note},
                "cluster": cluster.plan(), "sheet_times": [round(t, 3) for t in times]}

    def after(result: dict, out) -> dict:
        if opts["sound"] and getattr(cluster, "chime_db", None) is not None:
            return {"sound": f"{len(cluster.chimes())} chimes as the widgets land, the loudest "
                             f"{-cluster.chime_db:.0f} dB under the voice"}
        return {}

    return Job(NAME, clip, tl, fr, cluster.draw, out_plan, checks=cluster.report(),
               still_times=lambda: [times[0], times[2], times[3], times[4]], after=after, sound=cluster.mix)
