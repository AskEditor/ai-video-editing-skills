"""Asides: where the speaker talks about the recording, not to the viewer.

"Testing, testing", "give me a second", "let me start again", "I've got the audio set up", "are
we rolling?": said while a take is being made, never meant for the film. clean-cut cuts them
(`asides=cut`), and no picker ever sets one: `moments.taken_back` passes over them as it passes
over a word taken back.

A sentence is judged whole, never by one word in it ("testing our product" is no mic check, and
"cut the pauses" no call to a crew), on four things:

1. **What it says.** Every word of it must belong to one of the forms an aside takes (`FORMS`) or
   say nothing ("okay", "all right", "I think", "so"). Each form is of one kind: a mic check,
   asking for a moment, starting again, the equipment, or talk to a crew. A form is *strong* when
   its words are about the recording wherever they are said ("give me a second", "let me start
   again", "I've got the audio set up"), *weak* when the same words could make a point of their
   own ("check", "again", "cut", "one, two", "hold on a second"), and *faint* when they are said to
   the viewer as often as not ("hold on", "wait").
2. **What the take is about.** A remark about a thing the take talks about elsewhere belongs to
   the talk: in a talk about editing audio, "I think this audio works better" is the point. A form
   whose own words ("audio", "testing", "cut") are said in a sentence that is not an aside counts
   two steps fainter.
3. **Where it sits.** Asides gather at the start and the end of a take, sit between pauses, and
   come together. Each of these counts one: in the opening or the closing (the first or last fifth
   of the speech, `EDGE_LEAST` to `EDGE_MOST` seconds), set off by a pause of `SET_OFF` or more on
   both sides, and next to another aside. A strong form needs one of them, a weak one two, a faint
   one all three.
4. **It is short:** at most `MOST_WORDS` words.

A sentence of words that say nothing ("All right.") next to an aside goes with it.

    python -m fx.core.asides <clip> [<clip> …] [--fix wrong=right]   # every sentence, and its verdict
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import Optional

from .words import HESITATIONS, Word, pause_after

STRONG, WEAK, FAINT = 2, 1, 0
NEEDED = 3  # a form's strength plus what the shape of the take says must reach this
MOST_WORDS = 12  # the longest sentence that can be an aside
SET_OFF = 0.30  # seconds of quiet on both sides that set a sentence apart
EDGE_SHARE = 0.20  # the opening and the closing: this share of the speech at each end,
EDGE_LEAST = 8.0  # never less than this many seconds,
EDGE_MOST = 30.0  # nor more
TOPIC_COST = 2  # what a form loses when the take talks about its words elsewhere

# ---- the forms -----------------------------------------------------------------------------------
#
# Each is a regular expression over the sentence's words: lower case, no punctuation, one space
# apart, with the fillers ("um") out (`_tokens`). Written for this pack.

GEAR_WORDS = ("audio sound mic mics microphone microphones camera cameras cam lighting lens tripod battery lav "
              "lavalier headphones earphones airpods recorder recording setup levels volume gain wifi "
              "connection internet signal").split()
_GEAR = "(?:" + "|".join(GEAR_WORDS) + ")"
_DET = r"(?:the|this|my|our|that|some|a)"
_ADJ = r"(?:better|new|proper|good|other|different|external|second|backup|spare|right)"
_NP = rf"(?:{_DET} )?(?:{_ADJ} )*{_GEAR}"
_SOON = r"(?: (?:now|then|first|again|properly|quickly|real quick|a bit|a little))*"
_TIME = r"(?:second|seconds|sec|secs|moment|moments|minute|minutes|min|mins|mo|tick)"
_BRIEF = r"(?:second|sec|moment|mo|tick)"
_MEANS = (r"(?:let me|let's|let us|i'll|i will|i'm going to|i am going to|i'm going to have to|i need to|i have to|"
          r"i've got to|i got to|we'll|we need to|we have to|i should|can i|can we|i want to|i'm just going to|"
          r"let me just|i'm trying to|we should|i'll just)")
_THAT = (r"(?:that|this|it|the whole thing|that bit|that part|that line|that one|this bit|this part|this line|"
         r"the intro|the start|the beginning|the take)")
_AGAIN = r"(?:again|over|once more|one more time|from the top|from the beginning|from the start)"
_COUNT = r"(?:one|two|three|four|five)"
_FIX = (r"(?:fix|adjust|set up|setup|sort out|sort|check|test|turn on|turn up|turn down|turn off|switch on|switch off|"
        r"plug in|unplug|clip on|connect|reposition|charge|get closer to|move closer to|raise|lower)")
_GET = r"(?:get|grab|use|try|change|switch|swap|move|find|bring|put on)"
_STATE = r"(?:on|off|working|recording|plugged in|connected|fixed|sorted|set up|charged)"
_BETTER = r"(?:better|worse|clearer|louder|quieter|crisper|cleaner)"
_FINE = r"(?:fine|okay|good|all right|great)"
_MUCH = r"(?: (?:a lot|much|way|so much|a bit|a little|loads|heaps|so))?"
_SAYS = r"(?:works|is working|sounds|is|seems|looks|should be|should sound|feels)"

# Words that say nothing, said around an aside or between two ("Okay, give me a second.").
FILL = (r"(?:okay|all right|right|so|yeah|yes|yep|oh|ah|well|sorry|now|just|and|cool|great|perfect|nice|good|"
        r"there we go|here we go|i think|i guess|actually|anyway|hey|guys|everyone|everybody|real quick|quickly|"
        r"wait|hmm|okay then|right then|let's see|let me see)")

MIC, WAIT, AGAIN, GEAR, CREW = ("a mic check", "asking for a moment", "starting again", "the equipment",
                                "talk to a crew")
# The words of each kind that a take could be about: when the take says them elsewhere, the form
# belongs to the talk (rule 2 above).
TOPICS = {MIC: {"testing", "test", "tests", "check", "checking", "mic", "microphone", "hear"},
          GEAR: set(GEAR_WORDS), CREW: {"cut", "action", "rolling", "recording", "speed"}, WAIT: set(), AGAIN: set()}


@dataclass(frozen=True)
class Form:
    kind: str
    strength: int
    pattern: str
    rx: re.Pattern = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "rx", re.compile(self.pattern))


FORMS = [
    # A mic check: the words of one and nothing else.
    Form(MIC, STRONG, rf"(?:mic |sound )?(?:testing|test)(?: (?:testing|test|{_COUNT}|mic|check))+"),
    Form(MIC, STRONG, rf"(?:mic|sound) check(?: (?:{_COUNT}|check|mic))*"),
    Form(MIC, STRONG, rf"check(?: (?:check|{_COUNT}|mic))+"),
    Form(MIC, WEAK, r"(?:testing|test)"),
    Form(MIC, FAINT, r"check"),
    Form(MIC, WEAK, r"one two(?: three(?: four)?)?"),
    Form(MIC, STRONG, r"is (?:this|this thing|the mic|my mic|the microphone|my microphone|the camera|this camera|"
                      r"the audio|my audio) (?:still )?(?:on|working|recording|picking (?:me|this) up)"),
    Form(MIC, STRONG, r"(?:can|could) (?:you|everyone|everybody|anyone|you all|you guys|y'all) (?:hear|see) me"
                      r"(?: (?:okay|all right|now|well|clearly|properly))?"),
    Form(MIC, STRONG, rf"(?:does|do) (?:this|the|my) (?:mic|microphone|audio|sound) (?:work|sound (?:{_FINE}|{_BETTER}|clear))"),
    # Asking for a moment, to whoever is in the room.
    # "Give me a second" asks for one. "Give me two minutes", "one second" and "hold on a second"
    # open a pitch as often ("Hold on a second. Did you know…"), so alone they count for little.
    Form(WAIT, STRONG, rf"give (?:me|us) (?:a|one|just a|just one|a quick) {_BRIEF}"),
    Form(WAIT, STRONG, r"(?:one|just one|one quick) (?:sec|secs|moment|tick|mo)|just a (?:sec|moment)"),
    Form(WAIT, STRONG, r"(?:hold on|hang on|wait|just wait) (?:a|one|just a) (?:sec|secs|tick|mo)"),
    Form(WAIT, WEAK, rf"just a second|bear with me (?:a|one) {_TIME}"),
    Form(WAIT, FAINT, rf"give (?:me|us) (?:a|one|just a|a few|a couple of) (?:minute|seconds|moments)|one second|"
                      rf"(?:one|just a) minute|(?:hold on|hang on|wait|just wait) (?:a|one|just a) (?:second|moment|minute)|"
                      rf"bear with me"),
    Form(WAIT, WEAK, rf"(?:i'll|i will|i'm going to) be (?:right back|back in a {_TIME}|one {_TIME})"),
    Form(WAIT, FAINT, r"(?:hold on|hang on|wait|just wait)"),
    # Starting again.
    Form(AGAIN, STRONG, rf"{_MEANS} (?:start|begin|do|try|take|redo|record|film|shoot)(?: {_THAT})? {_AGAIN}"),
    Form(AGAIN, STRONG, rf"{_MEANS} (?:redo|retake|re record|reshoot|re do) {_THAT}"),
    Form(AGAIN, STRONG, rf"(?:let's|let me|we'll|i'll|can we|can i|should i|should we|shall we|shall i) (?:go|try|try it|try that|do it|do that) again"),
    Form(AGAIN, STRONG, r"(?:take|attempt) (?:two|three|four|five|six|seven|eight|nine|ten)"),
    Form(AGAIN, STRONG, r"(?:another|one more|a new|new) take"),
    Form(AGAIN, STRONG, r"(?:i|i've) (?:lost|forgot) my (?:train of thought|place|line|lines)"),
    Form(AGAIN, WEAK, rf"(?:start|starting|begin|beginning) (?:again|over)|from the top|from the beginning|one more time"),
    Form(AGAIN, FAINT, r"again|once more"),
    Form(AGAIN, WEAK, r"i (?:messed|screwed|mucked|stuffed) (?:that|this|it) up|i (?:fluffed|botched|flubbed) (?:that|this|it)"),
    # The equipment, said in the first person or of the one in use now. "The audio was great" is
    # neither: it says how something was, not what is being done to the recording.
    Form(GEAR, STRONG, rf"(?:i've|i have|we've|we have|i now have|now i've)(?: got)? {_NP} "
                       rf"(?:set up|setup|sorted|sorted out|fixed|working|on|ready|plugged in|connected|in place|going|running|back on|charged)"
                       rf"{_SOON}"),
    Form(GEAR, WEAK, rf"(?:i've|i have|we've|we have)(?: got)? (?:{_DET} )?better {_GEAR}{_SOON}"),
    Form(GEAR, STRONG, rf"{_MEANS}(?: just| quickly| quick)? {_FIX}(?: up| on| off| in)? {_NP}"
                       rf"(?: (?:set up|on|up|down|off|in|closer|working|sorted|better|right))?{_SOON}"),
    Form(GEAR, STRONG, rf"(?:{_MEANS}|i'm going to|i'm just going to)(?: just| quickly| quick)? {_GET} "
                       rf"(?:{_DET} )?(?:better|proper|different|another|other|new|second|backup) {_GEAR}"
                       rf"(?: (?:set up|on|in|working))?{_SOON}"),
    Form(GEAR, WEAK, rf"{_MEANS}(?: just| quickly| quick)? {_GET} {_NP}(?: (?:set up|on|in|closer))?{_SOON}"),
    Form(GEAR, STRONG, rf"(?:this|my|our|that|the new|this new) {_GEAR} {_SAYS}{_MUCH} (?:{_BETTER}|{_FINE}|{_STATE}){_SOON}"),
    Form(GEAR, STRONG, rf"the {_GEAR} {_SAYS}{_MUCH} {_BETTER}{_SOON}"),
    Form(GEAR, STRONG, rf"the {_GEAR} (?:is|seems|should be) {_STATE}{_SOON}"),
    Form(GEAR, WEAK, rf"the {_GEAR} (?:is|sounds|seems|looks) {_FINE} now"),
    Form(GEAR, STRONG, rf"(?:is|are) (?:this|we|you|the camera|this camera|the mic|my mic|this mic|the audio|my audio|"
                       rf"the recording) (?:still )?(?:recording|rolling|on|picking (?:me|this) up)"),
    Form(GEAR, WEAK, r"is it (?:still )?(?:recording|rolling|on|working)"),
    Form(GEAR, STRONG, r"(?:how|what) (?:is|does|do) (?:the|this|my) (?:audio|sound|mic|microphone) (?:sound|look|seem)(?: like)?(?: now)?"),
    Form(GEAR, STRONG, rf"(?:is|does) (?:the|this|my) (?:audio|sound|mic|microphone) (?:sound )?(?:{_FINE}|clear|working|"
                       rf"{_BETTER}|too loud|too quiet|loud enough)(?: now)?"),
    # Talk to a crew.
    Form(CREW, STRONG, r"(?:are we|we're|we are|is it|is the camera|are you|you're) (?:still )?(?:rolling|recording)"),
    Form(CREW, STRONG, r"and action"),
    Form(CREW, WEAK, r"(?:and )?(?:cut|action)(?: cut)*"),
    # As often asked of the viewer ("Was that good? Let me know in the comments").
    Form(CREW, FAINT, r"rolling|did you get that|did you get it|did that record|was that okay|was that all right|"
                      r"was that good|how was that|was that better|that was better|that one was better"),
]
_FILL_RX = re.compile(FILL)

# Words written several ways, read one way.
_SAME = {"alright": "all right", "ok": "okay", "okey": "okay", "mike": "mic", "gonna": "going to",
         "wanna": "want to", "gotta": "got to", "lemme": "let me", "gimme": "give me", "1": "one", "2": "two",
         "3": "three", "4": "four", "5": "five", "set-up": "set up", "re-record": "re record", "redo": "redo"}


def _tokens(words: list[Word], idx: list[int]) -> list[str]:
    """The words of a sentence as the forms read them: lower case, no punctuation, fillers out,
    hyphens and the spellings in `_SAME` evened out, and "the audio's" read as "the audio is"."""
    out: list[str] = []
    for i in idx:
        k = words[i].key
        if not k or k in HESITATIONS:
            continue
        k = _SAME.get(k, k)
        for part in k.replace("-", " ").split():
            part = _SAME.get(part, part)
            if part.endswith("'s") and part[:-2] in GEAR_WORDS:
                out += [part[:-2], "is"]
            else:
                out += part.split()
    return out


def _stem(k: str) -> str:
    """A word without its ending, so "testing", "tests" and "test" are one word."""
    for end in ("ing", "ed"):
        if k.endswith(end) and len(k) - len(end) >= 3:
            k = k[:-len(end)]
            break
    if k.endswith("s") and not k.endswith("ss") and len(k) >= 4:
        k = k[:-1]
    return k


@dataclass
class Reading:
    """What a sentence says, read against the forms."""
    forms: list  # the forms its words make, in order (empty when it is not made of them)
    only_fillers: bool  # every word of it says nothing ("All right.")
    tokens: list

    @property
    def strength(self) -> int:
        return max((f.strength for f in self.forms), default=-1)

    @property
    def kind(self) -> Optional[str]:
        best = max(self.forms, key=lambda f: f.strength, default=None)
        return best.kind if best else None

    def topic(self) -> set:
        """The words of the sentence that a take could be about (`TOPICS`), by their stem."""
        kinds = {f.kind for f in self.forms}
        return {_stem(t) for t in self.tokens if any(t in TOPICS[k] for k in kinds)}


def read(tokens: list[str]) -> Reading:
    """Which forms the words make, covering every one of them with forms and fillers (the reading
    with the strongest form, then the most words in forms), or none."""
    n = len(tokens)
    if n == 0:
        return Reading([], True, tokens)
    if n > MOST_WORDS:
        return Reading([], False, tokens)
    best: list = [None] * (n + 1)  # best[j]: (strength, words in forms, forms) covering tokens[:j]
    best[0] = (-1, 0, ())
    for i in range(n):
        if best[i] is None:
            continue
        s0, c0, f0 = best[i]
        for j in range(i + 1, n + 1):
            part = " ".join(tokens[i:j])
            options = [(s0, c0, f0)] if _FILL_RX.fullmatch(part) else []
            options += [(max(s0, f.strength), c0 + j - i, f0 + (f,)) for f in FORMS if f.rx.fullmatch(part)]
            for o in options:
                if best[j] is None or o[:2] > best[j][:2]:
                    best[j] = o
    if best[n] is None:
        return Reading([], False, tokens)
    return Reading(list(best[n][2]), not best[n][2], tokens)


# ---- the take ------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Aside:
    first: int  # its first word, an index into the words given
    last: int  # its last word (inclusive)
    start: float
    end: float
    kind: str  # what it is: "a mic check", "asking for a moment", "starting again", "about the audio", …
    said: str  # the sentence, as heard
    where: str  # what the shape of the take adds: "in the opening, next to another aside"

    @property
    def why(self) -> str:
        return f"{self.kind}, {self.where}" if self.where else self.kind


def sentences(words: list[Word]) -> list[list[int]]:
    """Word indexes, one list per sentence (to each full stop, question or exclamation mark)."""
    out, cur = [], []
    for i, w in enumerate(words):
        cur.append(i)
        if w.ends_sentence:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def _split_long(words: list[Word], idx: list[int], quiet) -> list[list[int]]:
    """A sentence too long to be an aside, cut where the transcriber lost its punctuation: at its
    longest pause of `SET_OFF` or more after a word it marked with nothing (no comma, no full
    stop), until every piece is short enough or no such pause is left. It does not always mark a
    sentence's end: hearing take 33 afresh, it ran "…edit to them I think this audio works better"
    together with no full stop, 0.79 s of quiet between, and the aside at the end went unseen (the
    cached hearing had "them. I think…"). A pause it did mark, with a comma, is a clause it heard
    as one sentence, and a long sentence is never cut there. Each piece is judged as any sentence
    is."""
    said = [i for i in idx if words[i].key and words[i].key not in HESITATIONS]
    if len(said) <= MOST_WORDS or len(idx) < 2:
        return [idx]
    k, longest = None, SET_OFF
    for n in range(len(idx) - 1):
        if words[idx[n]].ends_clause:
            continue
        g = _gap(words, idx[n], quiet)
        if g >= longest:
            k, longest = n, g
    if k is None:
        return [idx]
    return _split_long(words, idx[:k + 1], quiet) + _split_long(words, idx[k + 1:], quiet)


def _gap(words: list[Word], i: int, quiet) -> float:
    """The quiet after word i: from the sound when the quiet stretches are known, else the gap
    between the word times."""
    if i + 1 >= len(words):
        return 0.0
    if quiet is not None:
        return pause_after(words, i, quiet)
    return max(0.0, words[i + 1].start - words[i].end)


@dataclass
class _Sentence:
    idx: list
    reading: Reading
    strength: int = -1  # after the take's topic is taken into account
    opening: bool = False
    closing: bool = False
    set_off: bool = False
    aside: bool = False
    beside: bool = False


def _judge(words: list[Word], quiet) -> list[_Sentence]:
    """Every sentence of the take, read, placed and judged (`asides` returns the ones found)."""
    said = [w for w in words if w.key and w.key not in HESITATIONS]
    out = [_Sentence(piece, read(_tokens(words, piece)))
           for idx in sentences(words) for piece in _split_long(words, idx, quiet)]
    if not said:
        return out
    t0, t1 = said[0].start, said[-1].end
    edge = min(EDGE_MOST, max(EDGE_LEAST, EDGE_SHARE * (t1 - t0)))
    # What the take talks about: the words of every sentence that is not made of asides' forms.
    talk = {_stem(t) for s in out if not s.reading.forms and not s.reading.only_fillers for t in s.reading.tokens}
    for n, s in enumerate(out):
        a, b = s.idx[0], s.idx[-1]
        s.strength = s.reading.strength
        if s.reading.forms and s.reading.topic() & talk:
            s.strength -= TOPIC_COST
        s.opening = words[a].start <= t0 + edge
        s.closing = words[b].end >= t1 - edge
        before = _gap(words, a - 1, quiet) if a > 0 else float("inf")
        after = _gap(words, b, quiet) if b + 1 < len(words) else float("inf")
        s.set_off = before >= SET_OFF and after >= SET_OFF
    # Asides come together: a sentence next to one (with nothing but words that say nothing
    # between) counts one more, so the run is found from its strongest member outwards.
    changed = True
    while changed:
        changed = False
        for n, s in enumerate(out):
            if s.aside or not s.reading.forms or s.strength < FAINT:
                continue
            s.beside = _next_to_aside(out, n)
            if s.strength + (s.opening or s.closing) + s.set_off + s.beside >= NEEDED:
                s.aside = changed = True
    # A sentence of words that say nothing next to an aside goes with it.
    changed = True
    while changed:
        changed = False
        for n, s in enumerate(out):
            if not s.aside and s.reading.only_fillers and _next_to_aside(out, n, through_fillers=False):
                s.aside = changed = True
    for n, s in enumerate(out):
        s.beside = s.aside and _next_to_aside(out, n)
    return out


def _next_to_aside(out: list[_Sentence], n: int, through_fillers: bool = True) -> bool:
    for step in (-1, 1):
        k = n + step
        while 0 <= k < len(out):
            if out[k].aside:
                return True
            if not (through_fillers and out[k].reading.only_fillers):
                break
            k += step
    return False


def _label(out: list[_Sentence], n: int) -> str:
    s = out[n]
    if s.reading.only_fillers:
        both = 0 < n < len(out) - 1 and out[n - 1].aside and out[n + 1].aside
        return "nothing said, between two asides" if both else "nothing said, beside an aside"
    kind = s.reading.kind
    if kind == GEAR:
        gear = next((t for t in s.reading.tokens if t in GEAR_WORDS), None)
        return f"about the {gear}" if gear else "about the equipment"
    return kind or ""


def _where(s: _Sentence) -> str:
    bits = []
    if s.opening and s.closing:
        bits.append("in a take this short, at an end")
    elif s.opening:
        bits.append("in the opening")
    elif s.closing:
        bits.append("in the closing")
    if s.set_off:
        bits.append("set off by pauses")
    if s.beside and not s.reading.only_fillers:
        bits.append("next to another aside")
    return ", ".join(bits)


def asides(words: list[Word], quiet=None) -> list[Aside]:
    """The sentences of a take where the speaker talks about the recording itself (a mic check,
    asking for a moment, starting again, the equipment, talk to a crew), in order: each judged
    whole by what it says, what the take is about and where it sits (see the module's notes).
    Give a clip's whole `words` and its `quiet` stretches; without `quiet` the pauses are read
    from the gaps between the word times."""
    out = []
    judged = _judge(words, quiet)
    for n, s in enumerate(judged):
        if s.aside:
            a, b = s.idx[0], s.idx[-1]
            out.append(Aside(a, b, words[a].start, words[b].end, _label(judged, n),
                             " ".join(words[i].text for i in s.idx), _where(s)))
    return out


def report(words: list[Word], quiet=None) -> list[dict]:
    """Every sentence made of asides' forms, found or not, with what decided it: for reading a
    take's verdicts (`python -m fx.core.asides`)."""
    rows = []
    judged = _judge(words, quiet)
    for n, s in enumerate(judged):
        if not (s.reading.forms or (s.aside and s.reading.only_fillers)):
            continue
        a, b = s.idx[0], s.idx[-1]
        rows.append({"start": round(words[a].start, 2), "end": round(words[b].end, 2),
                     "said": " ".join(words[i].text for i in s.idx), "aside": s.aside, "kind": _label(judged, n),
                     "forms": [f"{f.kind} ({['faint', 'weak', 'strong'][f.strength]})" for f in s.reading.forms],
                     "strength": s.strength, "topic_said_elsewhere": s.strength < s.reading.strength,
                     "opening": s.opening, "closing": s.closing, "set_off": s.set_off, "beside": s.beside})
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    from .clip import Clip

    ap = argparse.ArgumentParser(prog="python -m fx.core.asides", description="The asides in a take, and why.")
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--fix", default=None)
    args = ap.parse_args(argv)
    for path in args.clips:
        clip = Clip(path, fix=args.fix)
        rows = report(clip.words, clip.quiet)
        print(f"{path}: {sum(r['aside'] for r in rows)} aside(s)")
        for r in rows:
            shape = ", ".join(k for k in ("opening", "closing", "set_off", "beside") if r[k])
            print(f"  {'ASIDE' if r['aside'] else 'kept '} {r['start']:6.2f}-{r['end']:6.2f}  \"{r['said']}\"  "
                  f"[{r['kind']}; {', '.join(r['forms']) or 'fillers'}; strength {r['strength']}"
                  f"{'; its words are the talk' if r['topic_said_elsewhere'] else ''}; {shape or 'no shape'}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
