"""review-notes' reading of notes, on fixed text and fixed words: no video, no model.

    .venv/bin/python tests/review_notes.py

Times typed and spoken, what a note is about, the thing it names in a plan, a moment found again in
a new render by the words around it, and the ledger read back as it was written. The run on real
cuts (the preview, the checks, the sweep, the refused final) is in the lab's `review-notes/notes.md`.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from fx.core.words import Word  # noqa: E402
from fx.effects import review_notes as rn  # noqa: E402

failures = 0


def check(what: str, ok: bool, detail: str = "") -> None:
    global failures
    print(("ok   " if ok else "FAIL ") + what + (f"  ({detail})" if detail and not ok else ""))
    failures += 0 if ok else 1


# -- times and notes in a line or a sentence
cases = [
    ("At four point two seconds the caption covers my chin.", False, [(4.2, "The caption covers my chin")]),
    ("Cut the pause at twelve seconds.", False, [(12.0, "Cut the pause")]),
    ("The caption covers my chin at four seconds and it freezes at twenty.", False,
     [(4.0, "The caption covers my chin"), (20.0, "It freezes")]),
    ("at one minute ten the screen goes black", False, [(70.0, "The screen goes black")]),
    ("Twenty three seconds in, the headline is cut off at the edge.", False,
     [(23.0, "The headline is cut off at the edge")]),
    ("at one point the caption is wrong", False, [(None, "At one point the caption is wrong")]),
    ("4.2 s: the caption covers my chin", True, [(4.2, "The caption covers my chin")]),
    ("0:12 cut the pause", True, [(12.0, "Cut the pause")]),
    ("12s - the gap is too long", True, [(12.0, "The gap is too long")]),
    ("the second caption has a typo", True, [(None, "The second caption has a typo")]),
]
for text, typed, want in cases:
    got = rn.notes_in(text.split(), typed=typed)
    check(f"notes in {text!r}", got == want, str(got))
got = [rn.notes_in(c) for c in rn.clauses("At 26 seconds it freezes while I'm talking, and the last caption has a "
                                           "typo.".split())]
got2 = [rn.notes_in(c) for c in rn.clauses("Okay, a couple of notes, at 26 seconds it freezes.".split())]
check("a preamble before a spoken time is not the note", got2 == [[(26.0, "It freezes")]], str(got2))
check("a spoken sentence about two things is two notes",
      got == [[(26.0, "It freezes while I'm talking")], [(None, "The last caption has a typo")]], str(got))

# -- what a note is about
for text, kind in [("the caption covers my chin", "face"), ("cut the pause", "pause"), ("there's dead air", "pause"),
                   ("it freezes", "frozen"), ("the screen goes black", "black"), ("too quiet", "loudness"),
                   ("I can barely hear myself", "loudness"), ("a quiet bit here", "pause"),
                   ("the headline is cut off", "zone"), ("it looks washed out", "colour"),
                   ("the headline should be bigger", rn.EYE), ("the second caption has a typo", rn.EYE)]:
    check(f"{text!r} is about {kind}", rn.kind_of(text) == kind, rn.kind_of(text))

# -- the thing a note names in the plan
plan = {"effect": "clean-cut", "clip": "x.mp4", "options": {}, "removals": [{"start": 1, "end": 2, "words": "um"}],
        "cards": [{"start": 0.0, "end": 1.0, "text": "one", "rect": [0, 0, 10, 10]},
                  {"start": 1.0, "end": 2.5, "text": "two"}, {"start": 2.5, "end": 4.0, "text": "three"}]}
items = rn.plan_items(plan)
check("a plan's captions are its items, and its removals are not", [i.label for i in items] == ["one", "two", "three"])
it, named = rn.item_named("the second caption has a typo", None, items)
check("'the second caption' names caption 2", it is not None and it.n == 2 and named)
it, named = rn.item_named("the last caption", None, items)
check("'the last caption' names the last", it is not None and it.n == 3)
it, named = rn.item_named("the caption covers my chin", 1.2, items)
check("a caption named at a second is the one on screen then", it is not None and it.n == 2 and named)
it, named = rn.item_named("my eyes are covered", 3.0, items)
check("a note naming nothing still says what was on screen, unnamed", it is not None and it.n == 3 and not named)
tier = {"effect": "tier-list", "cards": [{"tier": "A", "text": "Town", "at": 5.26, "lands": 5.5}]}
it = rn.plan_items(tier)[0]
check("a card with only a landing time is held for a moment, as a card", it.noun == "card" and it.end == 5.26 + rn.HELD)
check("a plan's caption words are the cut's words",
      [w.text for w in rn.plan_words({"cards": [{"words": [{"w": "b", "s": 1, "e": 1.2}, {"w": "a", "s": 0, "e": 0.4}]}]})]
      == ["a", "b"])

# -- a moment found again in a new render


def fake_cut(words, duration):
    ws = [Word(t, s, e) for t, s, e in words]
    return SimpleNamespace(words=ws, duration=duration)


old = fake_cut([("so", 0.0, 0.3), ("let's", 0.3, 0.6), ("talk", 0.6, 1.0), ("about", 4.0, 4.3), ("the", 4.3, 4.5),
                ("product", 4.5, 5.0)], 6.0)
a = rn.anchor_at(old, 2.5)  # in the pause between "talk" and "about"
new = fake_cut([("so", 0.0, 0.3), ("let's", 0.3, 0.6), ("talk", 0.6, 1.0), ("about", 1.2, 1.5), ("the", 1.5, 1.7),
                ("product", 1.7, 2.2)], 3.0)
t, how = rn.map_time(a, 2.5, 6.0, new)
check("a moment in a pause lands in the same pause, now shorter", t is not None and 1.0 <= t <= 1.2, f"{t} {how}")
a = rn.anchor_at(old, 4.6)
t, how = rn.map_time(a, 4.6, 6.0, new)
check("a moment inside a word stays in that word", t is not None and 1.7 <= t <= 1.85, f"{t} {how}")
gone = fake_cut([("something", 0.0, 0.5), ("else", 0.5, 1.0)], 2.0)
t, how = rn.map_time(rn.anchor_at(old, 4.6), 4.6, 6.0, gone)
check("a moment whose words were cut is gone", t is None, f"{t} {how}")

# -- the ledger, written and read back
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "talk.notes.md"
    cut = SimpleNamespace(path=Path(tmp) / "talk.mp4", duration=6.0, info=SimpleNamespace(width=720, height=1280),
                          source=None, source_said="", words_from="transcribed", qc_report=None, qc_sheet=None)
    state = rn.new_state()
    state["notes"] = [{"id": 1, "at": 4.2, "text": "The caption covers my chin", "origin": "you", "kind": "face",
                       "state": "open", "item": None, "last": {"seen": True, "what": "seen", "cut": str(cut.path),
                                                                "at": 4.2}},
                      {"id": 2, "at": None, "text": "Too quiet", "origin": "you", "kind": "loudness",
                       "state": "open", "item": None, "last": None}]
    rn.write_ledger(path, state, cut, Path(tmp) / "talk.review.mp4", Path(tmp) / "talk.review.jpg")
    text = path.read_text()
    check("the ledger says the final is refused while notes are open", "refused" in text and "2 open" in text)
    text = text.replace("- [ ] #1 ", "- [x] #1 ").replace("- [ ] #2 ", "- [-] #2 ")
    text = text.replace("  - what: the loudness", "  - what: the loudness\n  - why: it is a quiet film on purpose")
    text = text.replace("\n- \n", "\n- 0:03 cut the pause\n- the headline is too big\n")
    path.write_text(text)
    got_state, marks, new_lines = rn.read_ledger(path)
    check("the state comes back as it was written", [n["id"] for n in got_state["notes"]] == [1, 2])
    check("a tick and a decline are read, with the reason", marks.get(1) == ("x", "") and
          marks.get(2) == ("-", "it is a quiet film on purpose"), str(marks))
    check("new lines are read, and the instructions are not", new_lines == ["0:03 cut the pause",
                                                                           "the headline is too big"], str(new_lines))

print(f"\n{'all passed' if not failures else f'{failures} failed'}")
sys.exit(1 if failures else 0)
