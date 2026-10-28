"""Where the captions break a sentence into cards, read the way a person reading at speed would.

    .venv/bin/python tests/phrasing.py [--pack DIR] [--clip PATH …] [--cards] [--save FILE] [--against FILE]

It takes every clip in the lab's `test-clips/` and `raw-takes/` whose words are already found (the
words file beside it), never 39 or 41 (other people), reads the words as every effect does
(`Clip(...).words`) and has the captions effect make its cards in each look that shows several words
at once. Then it counts the cuts a reader stumbles on:

- **leans**: a card ending on a word that leans forward ("is", "of", "the", "a", "to", "and",
  "or", "my", "our", "your", "with", "for" and their kind) in the middle of a sentence, with no
  comma after it to close it;
- **pair**: a cut between two words people say as one ("right now", "of course", "a lot", "as
  well", "kind of", "real big", "going to", "set up", a first and last name, a number and its unit);
- **lone**: one word alone on a card, unless it was said alone (a sentence of its own, or a long
  silence either side) or is the point (the word the captions would make the key word).

And one more, apart from those three:

- **split**: a phrase cut in its middle: a card ending one word into a noun phrase ("an inbuilt |
  video edit"), or a verb cut from the person it acts on ("what makes | you stand out").

These lists are the reader's, written apart from the ones `fx/core/phrasing.py` cuts by, so the
check is not the cutter marking its own work. First, a few fixed lines (`CASES`) check what the
clips never say: a name, a number and its unit, a figure in pieces, a title, and that a number of
words the person sets is never passed.

`--pack DIR` counts with another copy of the pack (the code before a change), `--save` keeps every
card, `--against` lists each card that differs from a saved run, `--clip` adds a clip from anywhere.
Nothing is rendered and no model runs; the exit status is the number of bad cuts (leans, pair,
lone) and failed fixed lines, so 0 is a pass.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
PACK = TESTS.parent
sys.path.insert(0, str(TESTS))
from matrix import FIXES, LAB  # noqa: E402  (the fix each clip needs, as the matrix passes it)

NEVER = ("39", "41")  # raw takes of other people: never used
LOOKS = ("stacked", "fitted", "serif-chip")  # one-word shows one word a card: nothing to cut
LONG_SILENCE = 1.5  # a word with this much quiet either side was said alone

# ---- what a reader stumbles on ---------------------------------------------------------------------

LEANS_FORWARD = set("""
a an the my your our their its
of to for with from by at in on into onto about than like
and or but nor so because if once when while whether although though unless until who which whose what
is are was were am be been will would can could should must might may shall
i we they he she i'm we're they're you're he's she's it's there's here's that's let's i've we've they've
i'll we'll they'll i'd we'd
very really just real too quite
""".split())
ARTICLES = set("a an the my your our their its".split())
# After these, what a word began is over: "draws people in | and", "the next step is | and".
CLOSES = set("and but or so".split())
# "we have | an editor": the verb itself, then what it has.
HAS = set("have has had do does did".split())
SAID_AS_ONE = {tuple(p.split()) for p in """
right now,right here,right there,right away,of course,a lot,a bit,a few,a little,as well,as much,kind of,sort of
real big,real quick,real good,going to,want to,wants to,have to,has to,need to,needs to,got to,used to,able to
trying to,thank you,you know,each other,no one,at least,at all,in fact,so much,too much,more than,rather than
make sure,all right,out there,up there,down there,over there,set up,sign up,show up,pick up,end up,give up,
wake up,come up,make up,look up,stand out,find out,figure out,check out,work out,point out,turn out,come back,
go back,get back,come in,log in,turn on,turn off,carry on
""".replace("\n", ",").split(",") if p.strip()}
NUMBER_WORDS = set("""
one two three four five six seven eight nine ten eleven twelve twenty thirty forty fifty sixty seventy eighty
ninety hundred thousand million billion dozen
""".split())
UNITS = set("""
percent k m x times seconds minutes hours days weeks months years dollars pounds euros cents people users
followers views subscribers customers hundred thousand million billion grand
""".split()) | NUMBER_WORDS
OPENS_A_NOUN = set("a an the my your our their this that these those".split())
OBJECT_PRONOUNS = set("me him us them you it".split())
# A word followed by one of these is a verb ("set up", "stand out"): the phrase before it has ended.
VERB_PARTICLES = set("up down out off back away".split())
# Words that are not a verb, so a pronoun after them is not what they act on.
NOT_A_VERB = LEANS_FORWARD | OPENS_A_NOUN | CLOSES | set("""
that this these those there here then now well yeah oh okay also only even still never not
""".split())


def _number(w) -> bool:
    return any(c.isdigit() for c in w.key) or w.key in NUMBER_WORDS


def _stop(w) -> bool:
    return w.text.rstrip().endswith((",", ";", ":", ".", "?", "!", "—", "-", "…"))


def _sentence_end(w) -> bool:
    return w.text.rstrip().endswith((".", "?", "!"))


def _capital(words, i) -> bool:
    t = words[i].text.strip().lstrip("\"'“‘(")
    return (t[:1].isupper() and words[i].key not in ("i", "i'm", "i've", "i'll", "i'd")
            and i > 0 and not _sentence_end(words[i - 1]))


def faults(words, cards, key_word) -> list[tuple[int, str, str]]:
    """(card number, kind, why) for each bad cut. `cards` are lists of indexes into `words`."""
    out = []
    for n, card in enumerate(cards):
        last = card[-1]
        nxt = last + 1 if last + 1 < len(words) else None
        w = words[last]
        if nxt is not None and not _stop(w) and w.key in LEANS_FORWARD:
            closed = words[nxt].key in CLOSES and w.key not in ARTICLES
            verb_and_thing = w.key in HAS and words[nxt].key in OPENS_A_NOUN
            if not (closed or verb_and_thing):
                out.append((n, "leans", f"ends on \"{w.text}\""))
        if nxt is not None and not _stop(w):
            a, b = words[last], words[nxt]
            if (a.key, b.key) in SAID_AS_ONE:
                out.append((n, "pair", f"\"{a.text} | {b.text}\""))
            elif _capital(words, last) and _capital(words, nxt):
                out.append((n, "pair", f"a name cut: \"{a.text} | {b.text}\""))
            elif _number(a) and (b.key in UNITS or re.match(r"^[-,.%]", b.text.strip())):
                out.append((n, "pair", f"a number and its unit: \"{a.text} | {b.text}\""))
            elif re.match(r"^[-,.%]", b.text.strip()):
                out.append((n, "pair", f"one word in two pieces: \"{a.text} | {b.text}\""))
            # A phrase cut in its middle: "an inbuilt | video", but not "the audio | set up".
            elif (last >= 1 and words[last - 1].key in OPENS_A_NOUN and w.key not in LEANS_FORWARD
                  and b.key not in LEANS_FORWARD | OPENS_A_NOUN | CLOSES
                  and not (nxt + 1 < len(words) and words[nxt + 1].key in VERB_PARTICLES)):
                out.append((n, "split", f"one word into \"{words[last - 1].text} {a.text} | {b.text}\""))
            elif b.key in OBJECT_PRONOUNS and a.key not in NOT_A_VERB and len(a.key) > 2:
                out.append((n, "split", f"a verb cut from who it acts on: \"{a.text} | {b.text}\""))
        if len(card) == 1:
            i = card[0]
            alone_before = i == 0 or _sentence_end(words[i - 1]) or words[i].start - words[i - 1].end >= LONG_SILENCE
            alone_after = (i == len(words) - 1 or _sentence_end(words[i])
                           or words[i + 1].start - words[i].end >= LONG_SILENCE)
            if not (alone_before and alone_after):
                if key_word([words[i]]) == 0:
                    out.append((n, "point", f"\"{words[i].text}\" alone, as the point of its line"))
                else:
                    out.append((n, "lone", f"\"{words[i].text}\" alone"))
    return out


# ---- lines the clips do not say ---------------------------------------------------------------------
#
# Each line is timed a word every 0.32 s, with no pauses, so only the words decide; then the words
# that must share a card. The first five are the cuts seen by eye on our take (28 October
# 2026); the rest are what the clips never say: a name, a number and its unit, a figure the
# transcriber wrote in pieces, a title's full stop.
CASES = [
    ("What we're doing right now is of it.", ["right now"]),
    ("The marketing platform of UGC is what I'm seeing on builders who are making ads.", ["is what"]),
    ("Most people or most products, once they make the product, which is really good, the next step is.",
     ["once they"]),
    ("And what makes you stand out there to solve that issue.", ["makes you", "stand out"]),
    ("And we have an inbuilt video edit to demo.", ["inbuilt video"]),
    ("Yesterday I met Sarah Jones at the studio and we talked about the launch.", ["Sarah Jones"]),
    ("We grew to 500 people in three months and doubled what we made last year.", ["500 people", "three months"]),
    ("We hit 500 ,000 views on the in -built editor in just one week.", ["500 ,000", "in -built"]),
    ("I spoke to Dr. Smith about the new camera setup that we use.", ["Dr. Smith"]),
    ("It's kind of hard to edit on a phone.", ["kind of"]),
    ("Next week I'm going to film the launch video.", ["going to"]),
]
CASE_MOST = 4  # the stacked look's number


def check_cases(phrases, key_word, Word) -> int:
    """Each fixed line through the captions' cut at the stacked look's own number of words: no bad
    cut, and the words that belong together on one card. Then the same line with that number set by
    the person, which no card may pass (a number that tight can force a poor cut; only the length
    is checked). Returns how many checks failed."""
    failed = 0
    for line, together in CASES:
        words = [Word(t, round(i * 0.32, 3), round(i * 0.32 + 0.27, 3)) for i, t in enumerate(line.split())]
        cards = phrases(words, [], CASE_MOST)
        on = {i: n for n, card in enumerate(cards) for i in card}
        problems = [why for _, kind, why in faults(words, cards, key_word) if kind != "point"]
        for pair in together:
            a, b = pair.split()
            i = next(k for k in range(len(words) - 1) if words[k].text == a and words[k + 1].text == b)
            if on[i] != on[i + 1]:
                problems.append(f"\"{a} | {b}\" cut")
        try:
            by_hand = phrases(words, [], CASE_MOST, 0)
        except TypeError:  # a pack from before a number set by hand was kept to
            by_hand = None
        if by_hand is not None and any(len(c) > CASE_MOST for c in by_hand):
            problems.append(f"set by hand to {CASE_MOST}, a card has more")
        shown = " | ".join(" ".join(words[i].text for i in c) for c in cards)
        print(("ok    " if not problems else "FAIL  ") + shown
              + ("" if not problems else f"\n        {'; '.join(problems)}"))
        failed += bool(problems)
    return failed


# ---- the clips --------------------------------------------------------------------------------

def shown_cards(captions, framing, clip, words, look: str) -> list[list[int]]:
    """The cards the captions effect shows in this look, as indexes into `words`: made by the effect
    itself, at the clip's own size (no face is needed to decide which words share a card)."""
    info = clip.info
    fr = framing.plan(info.width, info.height, info.rate, info.frames, None, shape="keep")
    caps = captions.Captions(words, clip.quiet, (fr.out_w, fr.out_h), info.rate, 1, lambda j: None, look=look)
    at = {id(w): i for i, w in enumerate(words)}
    return [[at[id(w)] for w in c.words] for c in caps.cards]


def lab_clips() -> list[Path]:
    """Every test clip and raw take whose words are found, never 39 or 41."""
    out = []
    for folder in ("test-clips", "raw-takes"):
        for p in sorted((LAB / folder).glob("*.mp4")):
            if p.name.startswith(NEVER):
                continue
            if p.with_name(p.stem + ".words.json").exists():
                out.append(p)
    return out


def short(p: Path) -> str:
    return p.stem.split("-")[0] if (LAB in p.parents) else p.stem


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pack", help="count with this copy of the pack (default: the one this file is in)")
    ap.add_argument("--clip", action="append", default=[], help="a clip from anywhere, as well as the lab's")
    ap.add_argument("--cards", action="store_true", help="list every bad cut, card by card")
    ap.add_argument("--save", help="write every card to this JSON file")
    ap.add_argument("--against", help="list each card that differs from a run saved with --save")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(Path(args.pack).expanduser().resolve() if args.pack else PACK))
    from fx.core import framing
    from fx.core.clip import Clip
    from fx.core.words import HESITATIONS, Word
    from fx.effects import captions

    print("Fixed lines:")
    failed = check_cases(captions.phrases, captions.key_word, Word)
    print()

    clips = lab_clips() + [Path(c).expanduser().resolve() for c in args.clip]
    saved = json.loads(Path(args.against).read_text()) if args.against else None
    record: dict = {}
    totals = {"leans": 0, "pair": 0, "lone": 0, "split": 0, "point": 0}
    print(f"{'clip':<6} {'look':<13} {'cards':>5} {'leans':>6} {'pair':>5} {'lone':>5} {'split':>6} "
          f"{'point':>6} {'over':>5}")
    listing: list[str] = []
    changed: list[str] = []
    for path in clips:
        name = short(path)
        clip = Clip(path, fix=FIXES.get(name))
        words = [w for w in clip.words if w.key not in HESITATIONS and not w.unsure]
        record[name] = {}
        for look in LOOKS:
            most = captions.LOOK[look].most
            cards = shown_cards(captions, framing, clip, words, look)
            texts = [" ".join(words[i].text for i in c) for c in cards]
            record[name][look] = texts
            found = faults(words, cards, captions.key_word)
            count = {k: sum(1 for _, kind, _ in found if kind == k) for k in totals}
            for k in totals:
                totals[k] += count[k]
            over = sum(1 for c in cards if len(c) > most)
            print(f"{name:<6} {look + f' ({most})':<13} {len(cards):>5} {count['leans']:>6} {count['pair']:>5} "
                  f"{count['lone']:>5} {count['split']:>6} {count['point']:>6} {over:>5}")
            for n, kind, why in found:
                listing.append(f"  {name} {look}: [{kind}] {why}   «{texts[n]}»")
            if saved is not None and name in saved and look in saved[name] and saved[name][look] != texts:
                changed.append(_diff(name, look, saved[name][look], texts))
    bad = totals["leans"] + totals["pair"] + totals["lone"]
    print(f"\nbad cuts: {bad} (leans {totals['leans']}, pair {totals['pair']}, lone {totals['lone']}); "
          f"phrases split: {totals['split']}; a word alone as the point: {totals['point']}; "
          f"fixed lines failed: {failed}")
    if args.cards and listing:
        print("\n" + "\n".join(listing))
    if changed:
        print("\nchanged cards (before -> after):\n" + "\n".join(changed))
    if args.save:
        Path(args.save).write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    return min(bad + failed, 125)


def _diff(name: str, look: str, before: list[str], after: list[str]) -> str:
    """The stretches of cards that differ, each as the cards before and the cards after."""
    import difflib

    rows = [f"  {name} {look}:"]
    for op, a0, a1, b0, b1 in difflib.SequenceMatcher(a=before, b=after, autojunk=False).get_opcodes():
        if op != "equal":
            rows.append(f"    {' | '.join(before[a0:a1]) or '(none)'}\n      -> {' | '.join(after[b0:b1]) or '(none)'}")
    return "\n".join(rows)


if __name__ == "__main__":
    sys.exit(main())
